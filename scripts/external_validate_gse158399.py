#!/usr/bin/env python3
"""Frozen external mapping of GSE158399 into the BRCA metastasis atlas."""
from pathlib import Path
import json, gzip, warnings, os
import numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import umap

from scarches.models.scpoli import scPoli
from scvi.external import ContrastiveVI

warnings.filterwarnings('ignore')
SEED=1701
np.random.seed(SEED); torch.manual_seed(SEED)
ROOT=Path(os.environ.get('BRCA_ATLAS_ROOT','/data/Home/liyuhan_codex/PanCanCer_process/fastCNV/BRCA_data_process/scimilarity_research/multiscale_metastasis_atlas_v1'))
EXT=ROOT.parent/'external_validation_GSE158399'
OUT=EXT/'results'; OUT.mkdir(exist_ok=True)
SAMPLES={
 'GSM4798908':('GSM4798908_B2019-1.expression_matrix.txt.gz','Primary'),
 'GSM4798909':('GSM4798909_B2019-2.expression_matrix.txt.gz','Positive_LN'),
 'GSM4798910':('GSM4798910_B2019-3.expression_matrix.txt.gz','Negative_LN')}

def read_matrix(path, sample):
    x=pd.read_csv(path,sep='\t',index_col=0,compression='gzip')
    x=x.groupby(level=0).sum()
    a=ad.AnnData(sp.csr_matrix(x.T.values.astype('float32')),
                 obs=pd.DataFrame(index=[f'{sample}:{c}' for c in x.columns]),
                 var=pd.DataFrame(index=x.index.astype(str)))
    a.obs['sample']=sample
    return a

def sum_genes(a, genes):
    ids=np.flatnonzero(a.var_names.isin(genes))
    if len(ids)==0:return np.zeros(a.n_obs)
    return np.asarray(a.X[:,ids].sum(1)).ravel()

def mean_log_genes(a, genes):
    ids=np.flatnonzero(a.var_names.isin(genes))
    if len(ids)==0:return np.zeros(a.n_obs)
    z=a.X[:,ids].toarray() if sp.issparse(a.X) else a.X[:,ids]
    lib=np.asarray(a.X.sum(1)).ravel()
    return np.log1p(z/np.maximum(lib[:,None],1)*1e4).mean(1)

def prepare_query():
    arr=[]
    for sample,(fn,site) in SAMPLES.items():
        a=read_matrix(EXT/fn,sample); a.obs['true_site']=site; arr.append(a)
    a=ad.concat(arr,join='outer',fill_value=0,index_unique=None)
    total=np.asarray(a.X.sum(1)).ravel(); genes=np.asarray((a.X>0).sum(1)).ravel()
    mt=sum_genes(a,[g for g in a.var_names if g.startswith('MT-')])
    a.obs['total_counts']=total; a.obs['n_genes']=genes; a.obs['pct_mt']=100*mt/np.maximum(total,1)
    epi=['EPCAM','KRT8','KRT18','KRT19','KRT7','KRT14','KRT17','KRT5','MUC1','TACSTD2']
    imm=['PTPRC','CD3D','CD3E','CD79A','MS4A1','LST1','TYROBP','NKG7']
    strm=['COL1A1','COL1A2','DCN','LUM','PECAM1','VWF','RGS5','COL3A1']
    a.obs['epithelial_score']=mean_log_genes(a,epi)
    a.obs['immune_score']=mean_log_genes(a,imm)
    a.obs['stromal_score']=mean_log_genes(a,strm)
    qc=(total>=500)&(genes>=250)&(genes<=7500)&(a.obs.pct_mt.to_numpy()<=25)
    epi_call=(a.obs.epithelial_score.to_numpy()>=0.35)&(a.obs.epithelial_score.to_numpy()>a.obs.immune_score.to_numpy()+.12)&(a.obs.epithelial_score.to_numpy()>a.obs.stromal_score.to_numpy()+.05)
    a.obs['qc_pass']=qc; a.obs['epithelial_candidate']=qc&epi_call
    qc_summary=a.obs.groupby('true_site').agg(n_cells=('sample','size'),qc_pass=('qc_pass','sum'),epithelial_candidates=('epithelial_candidate','sum'),median_genes=('n_genes','median'),median_counts=('total_counts','median'),median_mt=('pct_mt','median'))
    qc_summary.to_csv(OUT/'query_qc_summary.csv')
    q=a[a.obs.epithelial_candidate].copy()
    ref=ad.read_h5ad(ROOT/'balanced_malignant_counts_4k.h5ad',backed='r')
    refgenes=pd.Index(ref.var_names.astype(str)); del ref
    idx=pd.Series(np.arange(a.n_vars),index=a.var_names).reindex(refgenes)
    blocks=[]
    for j in range(0,len(refgenes),500):
        jj=idx.iloc[j:j+500]; block=sp.csr_matrix((q.n_obs,len(jj)),dtype='float32')
        present=np.flatnonzero(jj.notna().to_numpy())
        if len(present):
            src=jj.iloc[present].astype(int).to_numpy(); coo=q.X[:,src].tocoo()
            block=sp.coo_matrix((coo.data,(coo.row,present[coo.col])),shape=(q.n_obs,len(jj))).tocsr()
        blocks.append(block)
    qq=ad.AnnData(sp.hstack(blocks,format='csr'),obs=q.obs.copy(),var=pd.DataFrame(index=refgenes))
    qq.layers['counts']=qq.X.copy(); qq.obs['condition_study']='GSE158399'; qq.obs['malignant_label']='Malignant_epithelial'; qq.obs['GEO']='GSE180286'
    qq.write_h5ad(OUT/'GSE158399_epithelial_query_4k.h5ad',compression='gzip')
    return a,qq,qc_summary

class Net(nn.Module):
    def __init__(self,nin,nclass,nstudy):
        super().__init__(); self.map=nn.Sequential(nn.Linear(nin,128),nn.LayerNorm(128),nn.GELU(),nn.Dropout(.1),nn.Linear(128,40)); self.cls=nn.Linear(40,nclass);self.dom=nn.Linear(40,nstudy)
    def latent(self,x):return F.normalize(self.map(x),dim=1)

def main():
    allq,q,qc=prepare_query(); print(qc,flush=True)
    # Unsupervised scArches/scPoli surgery: fit only the new study embedding, never use site labels.
    if (OUT/'scpoli_query_model').exists():
        query_model=scPoli.load(str(OUT/'scpoli_query_model'),adata=q,map_location='cpu')
    else:
        query_model=scPoli.load_query_data(q,str(ROOT/'scpoli_reference'),labeled_indices=[],freeze=True,freeze_expression=True,remove_dropout=True,map_location='cpu')
        query_model.train(n_epochs=35,pretraining_epochs=35,prototype_training=False,unlabeled_prototype_training=False,eta=0,lr=5e-4)
    zscp=np.asarray(query_model.get_latent(q,mean=True),dtype='float32')
    np.save(OUT/'query_scpoli_latent.npy',zscp)
    query_model.save(str(OUT/'scpoli_query_model'),overwrite=True,save_anndata=False)

    parts=[zscp]; names=['scPoli']
    for n in ['General','Brain','Liver','Bone','Lymph_node']:
        ContrastiveVI.setup_anndata(q,layer='counts',batch_key='GEO')
        m=ContrastiveVI.load(str(ROOT/f'contrastivevi_{n}'),adata=q)
        parts.append(np.asarray(m.get_latent_representation(q,representation_kind='salient'),dtype='float32')); names.append(n)
    screen=ROOT/'hierarchical_gene_screen'
    for n in ['Invasion_DCIS_to_Primary','Metastatic_ability','Distant_spread']:
        ContrastiveVI.setup_anndata(q,layer='counts',batch_key='GEO')
        m=ContrastiveVI.load(str(screen/f'contrastivevi_{n}'),adata=q)
        parts.append(np.asarray(m.get_latent_representation(q,representation_kind='salient'),dtype='float32')); names.append(n)
    X=np.hstack(parts).astype('float32')
    ck=torch.load(ROOT/'stage_organ_map_with_DCIS.pt',map_location='cpu',weights_only=False)
    X=(X-np.asarray(ck['feature_mean']))/np.asarray(ck['feature_sd'])
    net=Net(X.shape[1],len(ck['classes']),len(ck['studies']));net.load_state_dict(ck['state_dict']);net.eval()
    with torch.no_grad():
        xx=torch.as_tensor(X); z=net.latent(xx).numpy(); p=F.softmax(net.cls(net.latent(xx)),1).numpy()
    classes=np.asarray(ck['classes']); refz=np.load(ROOT/'stage_organ_latent_with_DCIS.npy')
    ra=ad.read_h5ad(ROOT/'balanced_malignant_counts_4k.h5ad',backed='r'); ro=ra.obs
    reflabel=np.where(ro.disease_stage.astype(str).eq('DCIS'),'DCIS',np.where(ro.disease_stage.astype(str).eq('Primary'),'Primary',ro.tissue_site.astype(str)))
    cent=np.vstack([refz[reflabel==c].mean(0) for c in classes]);cent/=np.linalg.norm(cent,axis=1,keepdims=True)
    sim=z@cent.T; near=sim.argmax(1)
    # Frozen reference-derived novelty thresholds and 31-neighbour purity.
    refsim=refz@cent.T; thr=np.array([np.quantile(refsim[reflabel==c,i],.05) for i,c in enumerate(classes)])
    nnm=NearestNeighbors(n_neighbors=31,metric='cosine',n_jobs=4).fit(refz)
    _,ind=nnm.kneighbors(z); knn=np.array([[np.mean(reflabel[ii]==c) for c in classes] for ii in ind])
    pred=p.argmax(1); agree=(pred==near)&(pred==knn.argmax(1))
    unknown=(sim[np.arange(len(z)),near]<thr[near]) | ((p.max(1)<.50)&(knn.max(1)<.50)) | (~agree & (p.max(1)<.65))
    out=q.obs.copy();out['predicted_state']=classes[pred];out['prototype_state']=classes[near];out['knn_state']=classes[knn.argmax(1)];out['unknown']=unknown
    out['max_probability']=p.max(1);out['prototype_similarity']=sim[np.arange(len(z)),near];out['knn_purity']=knn.max(1)
    for i,c in enumerate(classes):out[f'P_{c}']=p[:,i];out[f'Sim_{c}']=sim[:,i];out[f'KNN_{c}']=knn[:,i]
    out.to_csv(OUT/'cell_level_predictions.csv')
    np.save(OUT/'query_stage_organ_latent.npy',z)

    accepted=out.loc[~out.unknown]
    tab=pd.crosstab(accepted.true_site,accepted.predicted_state,normalize='index').reindex(columns=classes,fill_value=0)
    counts=pd.crosstab(out.true_site,np.where(out.unknown,'Unknown',out.predicted_state)).reindex(columns=list(classes)+['Unknown'],fill_value=0)
    tab.to_csv(OUT/'accepted_state_fractions.csv');counts.to_csv(OUT/'state_counts_with_unknown.csv')
    # Endpoint test: Primary vs positive LN. Negative LN is a specificity control, not a malignant truth class.
    ev=out[out.true_site.isin(['Primary','Positive_LN'])].copy(); y=(ev.true_site=='Positive_LN').astype(int).to_numpy()
    score=ev['P_Lymph_node'].to_numpy(); predln=(ev.predicted_state=='Lymph_node').astype(int).to_numpy()
    from sklearn.metrics import roc_auc_score,average_precision_score
    metrics={'dataset':'GSE158399','independent_of_training':True,'n_all_cells':int(len(allq)),'n_epithelial_candidates':int(len(q)),
             'n_by_site':out.true_site.value_counts().to_dict(),'unknown_fraction_by_site':out.groupby('true_site').unknown.mean().to_dict(),
             'ln_score_auc_primary_vs_positive_ln':float(roc_auc_score(y,score)) if len(np.unique(y))==2 else None,
             'ln_score_average_precision':float(average_precision_score(y,score)) if len(np.unique(y))==2 else None,
             'accepted_fraction_by_site':(1-out.groupby('true_site').unknown.mean()).to_dict(),
             'warning':'single patient; cell-level AUC is descriptive and not patient-level generalization'}
    (OUT/'external_validation_metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')

    # Reference-fitted UMAP is display only; all scores above use 40-D latent coordinates.
    rng=np.random.default_rng(SEED); rid=[]
    for c in classes:
        ii=np.flatnonzero(reflabel==c);rid.extend(rng.choice(ii,min(1200,len(ii)),replace=False))
    rid=np.asarray(rid); reducer=umap.UMAP(n_neighbors=35,min_dist=.18,metric='cosine',random_state=SEED).fit(refz[rid]); rxy=reducer.embedding_;qxy=reducer.transform(z)
    pal=dict(zip(classes,sns.color_palette('Set2',len(classes))))
    fig,ax=plt.subplots(1,2,figsize=(14,6))
    for c in classes:
        m=reflabel[rid]==c;ax[0].scatter(rxy[m,0],rxy[m,1],s=3,color=pal[c],alpha=.35,label=c,rasterized=True)
    markers={'Primary':'o','Positive_LN':'^','Negative_LN':'s'}
    for s,mk in markers.items():
        m=out.true_site.to_numpy()==s;ax[0].scatter(qxy[m,0],qxy[m,1],s=18,marker=mk,c=np.where(out.unknown.to_numpy()[m],'#777777','#111111'),edgecolors='white',linewidths=.25,label=f'Query {s}',zorder=5)
    ax[0].set(title='Frozen atlas with GSE158399 query',xlabel='UMAP1 (display only)',ylabel='UMAP2');ax[0].legend(fontsize=7,ncol=2,frameon=False)
    plot=counts.div(counts.sum(1),axis=0);plot.plot.bar(stacked=True,ax=ax[1],color={**pal,'Unknown':'#777777'},width=.75)
    ax[1].set(title='External query state assignment',xlabel='',ylabel='Fraction of epithelial candidates',ylim=(0,1));ax[1].legend(fontsize=8,frameon=False,bbox_to_anchor=(1.02,1),loc='upper left');ax[1].tick_params(axis='x',rotation=0)
    fig.tight_layout();fig.savefig(OUT/'GSE158399_external_mapping.png',dpi=240,bbox_inches='tight');plt.close(fig)
    print(json.dumps(metrics,indent=2),flush=True)

if __name__=='__main__': main()


