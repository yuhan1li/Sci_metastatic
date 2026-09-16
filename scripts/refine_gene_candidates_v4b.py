#!/usr/bin/env python3
from pathlib import Path
import json,os
import numpy as np,pandas as pd,anndata as ad
from scipy.stats import spearmanr

ROOT=Path(os.environ.get('BRCA_ATLAS_ROOT','/data/Home/liyuhan_codex/PanCanCer_process/fastCNV/BRCA_data_process/scimilarity_research/multiscale_metastasis_atlas_v1'))
OUT=ROOT/'full_balanced_split_v4b/gene_validation'
d=pd.read_csv(OUT/'stable_gene_ranking.csv');sm=pd.read_csv(OUT/'sample_metadata.csv');z=np.load(OUT/'sample_pseudobulk.npz',allow_pickle=True);E=z['expression'];genes=z['genes'].astype(str)
full=ad.read_h5ad(ROOT/'full_malignant_v4/full_strict_malignant_4k.h5ad',backed='r');fast=ad.read_h5ad(os.environ.get('BRCA_FASTCNV_H5AD','/data/Home/liyuhan_codex/PanCanCer_process/fastCNV/BRCA_data_process/Data_Save/BRCA_all_fastCNV_Epithelial.h5ad'),backed='r')
cnv=fast.obs.cnv_fraction.reindex(full.obs_names);mapped=float(cnv.notna().mean());tmp=pd.DataFrame({'GSM':full.obs.GSM.astype(str).to_numpy(),'cnv_fraction':cnv.to_numpy()});cm=tmp.groupby('GSM').cnv_fraction.mean();sm['mean_cnv_fraction']=sm.GSM.map(cm);sm.to_csv(OUT/'sample_metadata_with_cnv.csv',index=False)

# Correlation with CNV burden at the independent sample level.
rho=[];pv=[]
for j in range(len(genes)):
 q=np.isfinite(sm.mean_cnv_fraction);r=spearmanr(E[q,j],sm.loc[q,'mean_cnv_fraction']);rho.append(r.statistic);pv.append(r.pvalue)
cnvtab=pd.DataFrame({'gene':genes,'cnv_spearman':rho,'cnv_p':pv});d=d.merge(cnvtab,on='gene',how='left')

immune_exact=set('PIK3CG CD37 SELL SP140 ITK CD52 LTB BANK1 IL2RG SASH3 CORO1A CD69 TBC1D10C RFTN2 ARHGDIB PTPRC CD3D CD3E CD3G CD247 MS4A1 CD79A CD79B CD68 CD163 LYZ FCER1G TYROBP NKG7 GNLY TRAC TRBC1 TRBC2 LST1 LAPTM5 CCL5 CXCR4 CXCR3 CCR7 LCK FYN HCK BLK PTPRCAP THEMIS IKZF1 IKZF3 SPI1 B2M'.split())
endothelial=set('PECAM1 VWF EMCN KDR RAMP2 RAMP3 PLVAP ENG ESAM CDH5'.split());ery=set('HBA1 HBA2 HBB HBD ALAS2 AHSP'.split());cycle=set('MKI67 TOP2A UBE2C BIRC5 CENPF TYMS PCNA MCM2 MCM3 MCM4 MCM5 MCM6 MCM7'.split())
def contam(g):
 if g in immune_exact or g.startswith(('IGH','IGK','IGL','HLA-')):return 'immune_or_immunoglobulin'
 if g in endothelial:return 'endothelial'
 if g in ery:return 'erythroid'
 if g in cycle:return 'cell_cycle'
 return ''
d['refined_contamination_flag']=d.gene.map(contam)
d['cnv_associated']=d.cnv_spearman.abs()>=.5
# Confirmed requires concordance of model attribution, five paired studies, bootstrap CI, organ directions, empirical permutation, and no lineage/CNV warning.
d['evidence_tier']='not_supported'
provisional=(d.IG_signed>0)&(d.paired_direction_consistency>=.8)&(d.bootstrap_positive_probability>=.85)&(d.cross_organ_positive_fraction>=.75)&d.refined_contamination_flag.eq('')
d.loc[provisional,'evidence_tier']='provisional'
strong=provisional&(d.bootstrap_CI_low>0)&(d.permutation_p<.1)&(d.IG_rank<=1000)&(d.metastatic_organ_eta2<.25)&(~d.cnv_associated)
d.loc[strong,'evidence_tier']='stable_tumor_intrinsic'
d['final_score']=d.stability_score-.25*d.refined_contamination_flag.ne('')-.15*d.cnv_associated-.10*(d.metastatic_organ_eta2>=.25)
d=d.sort_values(['evidence_tier','final_score'],ascending=[True,False]);order={'stable_tumor_intrinsic':0,'provisional':1,'not_supported':2};d['_o']=d.evidence_tier.map(order);d=d.sort_values(['_o','final_score']).drop(columns='_o').reset_index(drop=True);d['final_rank']=np.arange(1,len(d)+1);d.to_csv(OUT/'refined_gene_evidence.csv',index=False)
summary={'cnv_cell_mapping_fraction':mapped,'n_stable_tumor_intrinsic':int(d.evidence_tier.eq('stable_tumor_intrinsic').sum()),'stable_tumor_intrinsic':d.loc[d.evidence_tier.eq('stable_tumor_intrinsic'),'gene'].head(30).tolist(),'n_provisional':int(d.evidence_tier.eq('provisional').sum()),'top_provisional':d.loc[d.evidence_tier.eq('provisional'),['gene','IG_rank','paired_study_effect','paired_direction_consistency','bootstrap_CI_low','permutation_p','cross_organ_positive_fraction','metastatic_organ_eta2','cnv_spearman']].head(30).to_dict('records'),'n_lineage_flagged':int(d.refined_contamination_flag.ne('').sum()),'interpretation':'A gene is not called stable unless model attribution, patient-level direction, bootstrap CI, permutation, cross-organ direction, lineage filtering and CNV audit agree.'}
(OUT/'refined_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))

