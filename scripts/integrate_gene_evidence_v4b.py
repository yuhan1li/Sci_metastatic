#!/usr/bin/env python3
from pathlib import Path
import json,os
import numpy as np,pandas as pd
import matplotlib.pyplot as plt,seaborn as sns

ROOT=Path(os.environ.get('BRCA_ATLAS_ROOT','/data/Home/liyuhan_codex/PanCanCer_process/fastCNV/BRCA_data_process/scimilarity_research/multiscale_metastasis_atlas_v1'))
OUT=ROOT/'full_balanced_split_v4b/gene_validation'
d=pd.read_csv(OUT/'refined_gene_evidence.csv')
t=pd.read_csv(ROOT/'hierarchical_gene_screen/all_candidate_genes_with_TCGA.csv');t=t[t.analysis.eq('Metastatic_ability')][['gene','cvi_lfc','cvi_proba_de','odds_ratio','p_value','fdr']].drop_duplicates('gene')
d=d.merge(t,on='gene',how='left',suffixes=('','_old'))
d['tcga_direction_concordant']=(d.odds_ratio>1)&(d.p_value<.05)&(d.cvi_lfc>0)
d['evidence_class']='D_not_supported'
d.loc[d.evidence_tier.eq('provisional'),'evidence_class']='C_single_cell_provisional'
d.loc[d.evidence_tier.eq('provisional')&d.tcga_direction_concordant,'evidence_class']='B_multimodal_candidate'
d.loc[d.evidence_tier.eq('stable_tumor_intrinsic')&d.tcga_direction_concordant,'evidence_class']='A_replicated_candidate'
order={'A_replicated_candidate':0,'B_multimodal_candidate':1,'C_single_cell_provisional':2,'D_not_supported':3};d['_o']=d.evidence_class.map(order);d=d.sort_values(['_o','final_score']).drop(columns='_o').reset_index(drop=True);d['integrated_rank']=np.arange(1,len(d)+1);d.to_csv(OUT/'integrated_gene_evidence.csv',index=False)

show=pd.concat([d[d.evidence_class.ne('D_not_supported')].head(20),d[d.gene.isin(['COL9A2','UGT2B4','RTN1','S100A7','KRT17'])]]).drop_duplicates('gene').head(25).copy()
mat=pd.DataFrame(index=show.gene)
mat['Model attribution']=1-show.IG_rank.to_numpy()/d.shape[0]
mat['Paired-study effect']=(np.tanh(show.paired_study_effect.to_numpy())+1)/2
mat['Study direction']=show.paired_direction_consistency.to_numpy()
mat['Bootstrap support']=show.bootstrap_positive_probability.to_numpy()
mat['Cross-organ direction']=show.cross_organ_positive_fraction.to_numpy()
mat['Low organ dependence']=1-show.metastatic_organ_eta2.to_numpy()
mat['Low CNV association']=1-np.minimum(np.abs(show.cnv_spearman.to_numpy()),1)
mat['TCGA direction']=np.where(show.odds_ratio.notna(),np.clip((show.odds_ratio.to_numpy()-0.7)/.6,0,1),.5)
fig,ax=plt.subplots(figsize=(12,max(6,.36*len(mat)+2)));sns.heatmap(mat,annot=True,fmt='.2f',cmap='RdYlGn',vmin=0,vmax=1,ax=ax,cbar_kws={'label':'Evidence strength'});ax.set_title('BRCA metastasis-gene evidence audit: independent criteria');ax.set_xlabel('');ax.set_ylabel('');fig.tight_layout();fig.savefig(OUT/'gene_evidence_heatmap.png',dpi=240);plt.close(fig)

failed=d.loc[d.IG_rank<=20].sort_values('IG_rank')[['gene','paired_study_effect','paired_direction_consistency','bootstrap_positive_probability','cross_organ_positive_fraction']]
summary={'n_A_replicated':int(d.evidence_class.eq('A_replicated_candidate').sum()),'A_genes':d.loc[d.evidence_class.eq('A_replicated_candidate'),'gene'].tolist(),'n_B_multimodal':int(d.evidence_class.eq('B_multimodal_candidate').sum()),'B_genes':d.loc[d.evidence_class.eq('B_multimodal_candidate'),'gene'].tolist(),'n_C_single_cell':int(d.evidence_class.eq('C_single_cell_provisional').sum()),'C_genes':d.loc[d.evidence_class.eq('C_single_cell_provisional'),'gene'].head(30).tolist(),'failed_original_top20':failed.to_dict('records')}
(OUT/'integrated_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))

