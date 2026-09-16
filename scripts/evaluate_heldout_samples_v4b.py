#!/usr/bin/env python3
from pathlib import Path
import json
import numpy as np
import pandas as pd
import anndata as ad
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
import train_tumor_only_disentangled_v3 as V3

ROOT=V3.ROOT
OUT=ROOT/'full_balanced_split_v4b'
CACHE=ROOT/'full_malignant_v4/full_strict_malignant_4k.h5ad'

a=ad.read_h5ad(CACHE,backed='r')
o=a.obs.copy()
z=np.load(OUT/'validation_latents.npz')
va=z['val']; zg=z['zg_val']; zo=z['zo_val']
ck=torch.load(OUT/'tumor_only_disentangled.pt',map_location='cpu')
m=V3.Net(len(ck['genes']),len(ck['studies']),len(ck['organs']),len(ck['tissues']))
m.load_state_dict(ck['state_dict']);m.eval()
with torch.no_grad():
 gp=F.softmax(m.general(torch.tensor(zg)),1).numpy()[:,1]
 op=F.softmax(m.organ(torch.tensor(zo)),1).numpy()
organs=np.asarray(ck['organs'])
d=o.iloc[va].copy()
d['p_metastasis']=gp
d['pred_stage']=np.where(gp>=.5,'Metastasis','Primary')
d['pred_organ']=organs[op.argmax(1)]
for j,c in enumerate(organs):d['p_'+c]=op[:,j]
d['cell_correct_stage']=d.pred_stage.eq(d.disease_stage.astype(str))
d['cell_correct_organ']=np.where(d.disease_stage.astype(str).eq('Metastasis'),d.pred_organ.eq(d.tissue_site.astype(str)),np.nan)
rows=[]
for gsm,q in d.groupby(d.GSM.astype(str),sort=True):
 true_stage=q.disease_stage.astype(str).mode().iat[0]
 p=float(q.p_metastasis.mean()); pred='Metastasis' if p>=.5 else 'Primary'
 true_org=q.tissue_site.astype(str).mode().iat[0]
 organ_p=q[['p_'+x for x in organs]].mean().to_numpy(); pred_org=organs[organ_p.argmax()]
 rows.append(dict(GSM=gsm,GEO=q.GEO.astype(str).mode().iat[0],patient=q.patient.astype(str).mode().iat[0] if 'patient' in q else '',n_cells=len(q),true_stage=true_stage,p_metastasis_mean=p,p_metastasis_median=float(q.p_metastasis.median()),high_risk_cell_fraction=float((q.p_metastasis>=.5).mean()),pred_stage=pred,stage_correct=pred==true_stage,true_organ=true_org,pred_organ=pred_org,organ_correct=(pred_org==true_org) if true_stage=='Metastasis' else np.nan,organ_confidence=float(organ_p.max())))
s=pd.DataFrame(rows)
s.to_csv(OUT/'heldout_sample_predictions.csv',index=False)
keep=[x for x in ['GSM','GEO','patient','disease_stage','tissue_site','p_metastasis','pred_stage','pred_organ','cell_correct_stage','cell_correct_organ'] if x in d]
d[keep].to_csv(OUT/'heldout_cell_predictions.csv.gz',index=True,compression='gzip')
known=s[s.true_stage.isin(['Primary','Metastasis'])];met=known[known.true_stage.eq('Metastasis')]
summary={'n_heldout_samples':len(s),'n_evaluable_stage_samples':len(known),'sample_stage_accuracy':float(known.stage_correct.mean()),'primary_accuracy':float(known.loc[known.true_stage.eq('Primary'),'stage_correct'].mean()),'metastasis_accuracy':float(met.stage_correct.mean()),'metastatic_sample_organ_accuracy':float(met.organ_correct.mean()),'n_primary_samples':int(known.true_stage.eq('Primary').sum()),'n_metastatic_samples':int(met.shape[0]),'stage_errors':known.loc[~known.stage_correct,['GSM','GEO','n_cells','true_stage','p_metastasis_mean','pred_stage']].to_dict('records'),'organ_errors':met.loc[~met.organ_correct.astype(bool),['GSM','GEO','n_cells','true_organ','pred_organ','organ_confidence']].to_dict('records')}
(OUT/'heldout_sample_summary.json').write_text(json.dumps(summary,indent=2))
# Reproduce the sample-level held-out bar plot used in the report.
plot=s[s.true_stage.isin(['Primary','Metastasis'])].sort_values('p_metastasis_mean',ascending=True)
colors=plot.true_stage.map({'Primary':'#4C78A8','Metastasis':'#E45756'})
fig,ax=plt.subplots(figsize=(11,6));ax.barh(np.arange(len(plot)),plot.p_metastasis_mean,color=colors)
ax.axvline(.5,color='black',ls='--',lw=1.5);ax.set_yticks(np.arange(len(plot)))
ax.set_yticklabels(plot.GSM+' | '+plot.true_stage+' | '+plot.true_organ,fontsize=8)
ax.set(xlabel='Mean predicted metastasis probability',title=f'Entire-sample held-out predictions ({len(plot)} evaluable samples)',xlim=(0,1.05))
fig.tight_layout();fig.savefig(OUT/'heldout_sample_predictions.png',dpi=240);plt.close(fig)
print(json.dumps(summary,indent=2))

