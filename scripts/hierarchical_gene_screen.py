#!/usr/bin/env python3
"""Hierarchical contrastiveVI + sample-level linear SHAP gene screening."""
from pathlib import Path
import json, os
import numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp, torch
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from scvi.external import ContrastiveVI

ROOT=Path(os.environ.get('BRCA_ATLAS_ROOT','/data/Home/liyuhan_codex/PanCanCer_process/fastCNV/BRCA_data_process/scimilarity_research/multiscale_metastasis_atlas_v1'))
OUT=ROOT/'hierarchical_gene_screen';OUT.mkdir(exist_ok=True)
SEED=1701;rng=np.random.default_rng(SEED);torch.manual_seed(SEED);torch.set_float32_matmul_precision('high')
a=ad.read_h5ad(ROOT/'balanced_malignant_counts_4k.h5ad');o=a.obs

specs={
 'Invasion_DCIS_to_Primary':(o.disease_stage.eq('Primary'),o.disease_stage.eq('DCIS')),
 'Metastatic_ability':(o.disease_stage.eq('Metastasis'),o.disease_stage.eq('Primary')),
 'Distant_spread':(o.tissue_site.isin(['Brain','Liver','Bone']),o.tissue_site.isin(['Breast','Lymph_node']) & ~o.disease_stage.eq('DCIS')),
 'Brain':(o.tissue_site.eq('Brain'),~o.tissue_site.eq('Brain') & ~o.disease_stage.eq('DCIS')),
 'Liver':(o.tissue_site.eq('Liver'),~o.tissue_site.eq('Liver') & ~o.disease_stage.eq('DCIS')),
 'Bone':(o.tissue_site.eq('Bone'),~o.tissue_site.eq('Bone') & ~o.disease_stage.eq('DCIS')),
 'Lymph_node':(o.tissue_site.eq('Lymph_node'),o.tissue_site.isin(['Breast','Brain','Liver','Bone']) & ~o.disease_stage.eq('DCIS')),
}
reuse={'Brain':'Brain','Liver':'Liver','Bone':'Bone','Lymph_node':'Lymph_node'}

def balanced(mask,n=120):
 out=[]
 for _,ids in o.loc[mask].groupby('GSM',observed=True).groups.items():
  ids=np.asarray([a.obs_names.get_loc(x) for x in ids]);out.extend(rng.choice(ids,min(n,len(ids)),replace=False))
 return np.asarray(out,dtype=int)

de_tables={};model_metrics={}
for name,(tm,bm) in specs.items():
 ti=balanced(tm);bi=balanced(bm)
 ContrastiveVI.setup_anndata(a,layer='counts',batch_key='GEO')
 path=ROOT/f'contrastivevi_{reuse[name]}' if name in reuse else OUT/f'contrastivevi_{name}'
 if name in reuse:
  model=ContrastiveVI.load(str(path),adata=a)
 else:
  model=ContrastiveVI(a,n_hidden=128,n_background_latent=16,n_salient_latent=16,n_layers=2,dropout_rate=.15,wasserstein_penalty=.1)
  model.train(background_indices=np.flatnonzero(bm).tolist(),target_indices=np.flatnonzero(tm).tolist(),max_epochs=40,
              accelerator='gpu',devices=1,batch_size=256,early_stopping=True,check_val_every_n_epoch=2,enable_progress_bar=True)
  model.save(str(path),overwrite=True,save_anndata=False)
 # Model-based DE on sample-balanced cells.
 de=model.differential_expression(idx1=ti,idx2=bi,target_idx=ti,n_samples=3,silent=True)
 de.index.name='gene';de.reset_index().to_csv(OUT/f'{name}_contrastiveVI_DE.csv',index=False)
 de_tables[name]=de
 model_metrics[name]={'target_cells_DE':len(ti),'background_cells_DE':len(bi),'target_samples':int(o.loc[tm,'GSM'].nunique()),'background_samples':int(o.loc[bm,'GSM'].nunique())}

# Pseudobulk sample expression.
X=a.layers['counts']; gs=[];rows=[]
for gsm,idx in o.groupby('GSM',observed=True).indices.items():
 v=np.asarray(X[idx].sum(0)).ravel();v=np.log1p(v/max(v.sum(),1)*1e6);gs.append(v)
 r=o.iloc[idx[0]];rows.append({'GSM':gsm,'GEO':r.GEO,'stage':r.disease_stage,'site':r.tissue_site})
E=np.vstack(gs);sm=pd.DataFrame(rows);genes=np.asarray(a.var_names)
all_rank=[]
for name in specs:
 if name=='Invasion_DCIS_to_Primary': use=sm.stage.isin(['DCIS','Primary']); yy=sm.stage.eq('Primary')
 elif name=='Metastatic_ability': use=sm.stage.isin(['Primary','Metastasis']); yy=sm.stage.eq('Metastasis')
 elif name=='Distant_spread': use=~sm.stage.eq('DCIS'); yy=sm.site.isin(['Brain','Liver','Bone'])
 elif name=='Lymph_node': use=~sm.stage.eq('DCIS'); yy=sm.site.eq('Lymph_node')
 else: use=~sm.stage.eq('DCIS'); yy=sm.site.eq(name)
 ids=np.flatnonzero(use);e=E[ids].copy();m=sm.iloc[ids].reset_index(drop=True);y=yy.iloc[ids].to_numpy().astype(int)
 # Remove study means so the classifier cannot use study-level expression offsets.
 for st,q in m.groupby('GEO').groups.items():e[np.asarray(list(q))]-=e[np.asarray(list(q))].mean(0)
 scaler=StandardScaler();ez=scaler.fit_transform(e)
 clf=LogisticRegression(C=.08,class_weight='balanced',max_iter=5000,solver='liblinear').fit(ez,y)
 shap=ez*clf.coef_[0][None,:]  # exact linear SHAP under the independent-feature baseline
 mean_abs=np.abs(shap).mean(0);signed=shap[y==1].mean(0)-shap[y==0].mean(0)
 # Direction consistency only in studies containing both labels.
 effects=[]
 for st,q in m.groupby('GEO').groups.items():
  q=np.asarray(list(q));
  if len(np.unique(y[q]))==2:effects.append(ez[q][y[q]==1].mean(0)-ez[q][y[q]==0].mean(0))
 if effects:
  eff=np.vstack(effects);direction=np.sign(eff).mean(0);ninf=len(effects)
 else:direction=np.zeros(len(genes));ninf=0
 de=de_tables[name].reindex(genes)
 rank=pd.DataFrame({'analysis':name,'gene':genes,'shap_mean_abs':mean_abs,'shap_signed':signed,
                    'cvi_lfc':de.lfc_mean.to_numpy(),'cvi_bayes_factor':de.bayes_factor.to_numpy(),
                    'cvi_proba_de':de.proba_de.to_numpy(),'n_informative_studies':ninf,'direction_consistency':direction})
 for col in ['shap_mean_abs','cvi_bayes_factor']:
  rank[col+'_pct']=rank[col].rank(pct=True)
 rank['evidence_score']=rank.shap_mean_abs_pct+rank.cvi_bayes_factor_pct+np.abs(rank.direction_consistency)
 rank['confidence']=np.where(rank.n_informative_studies>=2,'cross-study',np.where(rank.n_informative_studies==1,'single-study','confounded'))
 rank=rank.sort_values('evidence_score',ascending=False);rank.to_csv(OUT/f'{name}_combined_gene_ranking.csv',index=False);all_rank.append(rank)
 model_metrics[name].update({'training_sample_balanced_accuracy':float(balanced_accuracy_score(y,clf.predict(ez))),
                             'training_sample_auc':float(roc_auc_score(y,clf.predict_proba(ez)[:,1])),
                             'n_informative_studies':ninf})
pd.concat(all_rank).to_csv(OUT/'all_candidate_genes.csv',index=False)
(OUT/'screening_summary.json').write_text(json.dumps(model_metrics,indent=2),encoding='utf-8')
print(json.dumps(model_metrics,indent=2),flush=True)



