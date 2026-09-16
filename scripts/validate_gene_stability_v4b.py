#!/usr/bin/env python3
"""Sample-level stability and confounding audit for full-v4b gene attributions."""
from pathlib import Path
import json,os
import numpy as np, pandas as pd, anndata as ad
from scipy.stats import rankdata

ROOT=Path(os.environ.get('BRCA_ATLAS_ROOT','/data/Home/liyuhan_codex/PanCanCer_process/fastCNV/BRCA_data_process/scimilarity_research/multiscale_metastasis_atlas_v1'))
OUT=ROOT/'full_balanced_split_v4b/gene_validation';OUT.mkdir(parents=True,exist_ok=True)
A=ad.read_h5ad(ROOT/'full_malignant_v4/full_strict_malignant_4k.h5ad');o=A.obs.copy();genes=np.asarray(A.var_names.astype(str));X=A.layers['counts']
rng=np.random.default_rng(20260916)

# Patient/sample pseudobulk; cells are not treated as independent replicates.
expr=[];meta=[]
for gsm,ii in o.groupby(o.GSM.astype(str),observed=True).indices.items():
    v=np.asarray(X[ii].sum(0)).ravel().astype(float);expr.append(np.log1p(v/max(v.sum(),1)*1e6))
    r=o.iloc[ii[0]];meta.append(dict(GSM=gsm,GEO=str(r.GEO),patient_uid=str(r.patient_uid),stage=str(r.disease_stage),site=str(r.tissue_site),n_cells=len(ii),subtype=str(r.subtype)))
E=np.vstack(expr);sm=pd.DataFrame(meta);np.savez_compressed(OUT/'sample_pseudobulk.npz',expression=E,genes=genes);sm.to_csv(OUT/'sample_metadata.csv',index=False)

ig=pd.read_csv(ROOT/'full_balanced_split_v4b/gene_attribution/general_metastasis_gene_ranking.csv').set_index('gene').reindex(genes)
paired=[]
for st,q in sm[sm.stage.isin(['Primary','Metastasis'])].groupby('GEO'):
    y=q.stage.eq('Metastasis').to_numpy();idx=q.index.to_numpy()
    if y.any() and (~y).any():
        # standardized within-study effect avoids between-study expression offsets
        scale=E[idx].std(0);scale[scale<.15]=.15
        paired.append((st,(E[idx[y]].mean(0)-E[idx[~y]].mean(0))/scale,idx,y))
effects=np.vstack([x[1] for x in paired]);meta_effect=effects.mean(0);direction=(effects>0).mean(0);heterogeneity=effects.std(0)

# Stratified sample bootstrap within informative studies.
B=400;boot=np.zeros((B,len(genes)),dtype='float32')
for b in range(B):
    ee=[]
    for _,_,idx,y in paired:
        pos=idx[y];neg=idx[~y];p=rng.choice(pos,len(pos),True);n=rng.choice(neg,len(neg),True);scale=np.vstack([E[p],E[n]]).std(0);scale[scale<.15]=.15;ee.append((E[p].mean(0)-E[n].mean(0))/scale)
    boot[b]=np.mean(ee,0)
boot_prob=(boot>0).mean(0);ci_low=np.quantile(boot,.025,axis=0);ci_high=np.quantile(boot,.975,axis=0)

# Cross-organ direction and metastatic-organ heterogeneity. These are supporting only because most distant sites are study-confounded.
primary=E[sm.stage.eq('Primary')];organ_eff=[]
for site in ['Lymph_node','Brain','Liver','Bone']:
    organ_eff.append(E[sm.site.eq(site)].mean(0)-primary.mean(0))
organ_eff=np.vstack(organ_eff);organ_consistency=(organ_eff>0).mean(0)
metidx=np.flatnonzero(sm.stage.eq('Metastasis'));grand=E[metidx].mean(0);ssb=np.zeros(len(genes));sst=((E[metidx]-grand)**2).sum(0)+1e-8
for site,q in sm.iloc[metidx].groupby('site').groups.items():
    q=np.asarray(list(q));ssb+=len(q)*(E[q].mean(0)-grand)**2
organ_eta=np.clip(ssb/sst,0,1)

def pct(x): return rankdata(np.nan_to_num(x,nan=np.nanmin(x[np.isfinite(x)]) if np.isfinite(x).any() else 0))/len(x)
immune={'PTPRC','CD3D','CD3E','CD3G','CD247','CD79A','CD79B','MS4A1','CD68','CD163','LYZ','FCER1G','TYROBP','NKG7','GNLY','IGHG1','IGHGP','IGHA1','HLA-DRA','CLEC7A'}
ery={'HBA1','HBA2','HBB','HBD','ALAS2'};endo={'PECAM1','VWF','EMCN','KDR','RAMP2','RAMP3','PLVAP','ENG'}
cycle={'MKI67','TOP2A','UBE2C','BIRC5','CENPF','TYMS','PCNA','MCM2','MCM3','MCM4','MCM5','MCM6','MCM7'}
mito={g for g in genes if g.startswith('MT-')};ribo={g for g in genes if g.startswith('RPL') or g.startswith('RPS')}
flag=np.array(['immune' if g in immune else 'erythroid' if g in ery else 'endothelial' if g in endo else 'cell_cycle' if g in cycle else 'mitochondrial' if g in mito else 'ribosomal' if g in ribo else '' for g in genes])
score=.30*pct(ig.score.to_numpy())+.22*pct(meta_effect)+.16*direction+.14*boot_prob+.10*organ_consistency+.08*pct(np.maximum(ig.get('perturbation_delta',pd.Series(0,index=genes)).to_numpy(),0))-.18*pct(organ_eta)-.25*(flag!='')
res=pd.DataFrame(dict(gene=genes,stability_score=score,IG_rank=ig.single_cell_rank.to_numpy(),IG_signed=ig.IG_signed.to_numpy(),IG_abs=ig.IG_abs.to_numpy(),perturbation_delta=ig.get('perturbation_delta',pd.Series(0,index=genes)).to_numpy(),paired_study_effect=meta_effect,paired_direction_consistency=direction,n_informative_studies=len(paired),bootstrap_positive_probability=boot_prob,bootstrap_CI_low=ci_low,bootstrap_CI_high=ci_high,cross_organ_positive_fraction=organ_consistency,metastatic_organ_eta2=organ_eta,contamination_flag=flag))
res['passes_core']=(res.IG_signed>0)&(res.paired_study_effect>0)&(res.paired_direction_consistency>=.8)&(res.bootstrap_positive_probability>=.9)&(res.cross_organ_positive_fraction>=.75)&res.contamination_flag.eq('')
res=res.sort_values(['passes_core','stability_score'],ascending=False).reset_index(drop=True);res['stable_rank']=np.arange(1,len(res)+1);res.to_csv(OUT/'stable_gene_ranking.csv',index=False)

# Permutation empirical P values for the leading 200 genes, shuffling labels only within informative studies.
top=res.head(200).gene.map(pd.Index(genes).get_loc).to_numpy();obs=meta_effect[top];ge=np.zeros(len(top),int);P=1000
for _ in range(P):
    ee=[]
    for _,_,idx,y in paired:
        yp=rng.permutation(y);scale=E[idx].std(0);scale[scale<.15]=.15;ee.append((E[idx[yp]].mean(0)-E[idx[~yp]].mean(0))/scale)
    null=np.mean(ee,0)[top];ge+=(np.abs(null)>=np.abs(obs))
pv=(ge+1)/(P+1);pmap=dict(zip(genes[top],pv));res['permutation_p']=res.gene.map(pmap);res.to_csv(OUT/'stable_gene_ranking.csv',index=False)

summary={'n_cells':int(A.n_obs),'n_samples':len(sm),'n_patients_known':int(sm.loc[~sm.patient_uid.str.contains('SAMPLE::'),'patient_uid'].nunique()),'informative_primary_metastasis_studies':[x[0] for x in paired],'n_core_genes':int(res.passes_core.sum()),'top_core_genes':res.loc[res.passes_core,'gene'].head(30).tolist(),'warning':'Distant organ cohorts are mostly study-confounded; cross-organ consistency is supporting evidence, not causal proof.'}
(OUT/'stability_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))

