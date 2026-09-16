#!/usr/bin/env python3
"""Create multiple sample-balanced, cross-study prototypes and validation plots."""
from pathlib import Path
import json, os
import numpy as np
import pandas as pd
import anndata as ad
from sklearn.cluster import KMeans
from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns

ROOT=Path(os.environ.get('BRCA_ATLAS_ROOT','/data/Home/liyuhan_codex/PanCanCer_process/fastCNV/BRCA_data_process/scimilarity_research/multiscale_metastasis_atlas_v1'))
adata=ad.read_h5ad(ROOT/'balanced_malignant_counts_4k.h5ad', backed='r')
obs=adata.obs.copy().reset_index(names='cell_id')
parts={'scPoli':np.load(ROOT/'scpoli_latent.npy')}
for n in ['General','Brain','Liver','Bone','Lymph_node']:
    parts[n]=np.load(ROOT/f'contrastivevi_{n}_salient.npy')

# Equal-weight sample centroids in the appropriate latent space.
rows=[]; arrays={k:[] for k in parts}
for gsm, idx in obs.groupby('GSM', sort=True).groups.items():
    ii=np.asarray(list(idx), dtype=int); r=obs.iloc[ii[0]]
    rows.append({'GSM':gsm,'GEO':r.GEO,'patient_uid':r.patient_uid,
                 'tissue_site':r.tissue_site,'disease_stage':r.disease_stage,'n_cells':len(ii)})
    for k,x in parts.items(): arrays[k].append(x[ii].mean(0))
smeta=pd.DataFrame(rows)
for k in arrays: arrays[k]=np.vstack(arrays[k]).astype('float32'); np.save(ROOT/f'sample_centroids_{k}.npy',arrays[k])
smeta.to_csv(ROOT/'sample_centroid_metadata.csv',index=False)

# Each site gets 1..3 states depending on sample count. Retain provenance for audit.
proto=[]; pvec=[]
for site in ['Brain','Liver','Bone','Lymph_node']:
    ids=np.flatnonzero(smeta.tissue_site.eq(site).to_numpy()); x=arrays[site][ids]
    ncl=max(1,min(3,len(ids)//5))
    lab=KMeans(n_clusters=ncl,random_state=1701,n_init=30).fit_predict(StandardScaler().fit_transform(x))
    for c in range(ncl):
        q=ids[lab==c]; v=arrays[site][q].mean(0)
        proto.append({'organ':site,'prototype':f'{site}_P{c+1}','n_samples':len(q),
                      'n_studies':smeta.iloc[q].GEO.nunique(),
                      'studies':'|'.join(sorted(smeta.iloc[q].GEO.unique())),
                      'eligible_cross_study':bool(smeta.iloc[q].GEO.nunique()>=2 and len(q)>=3)})
        pvec.append(v)
pd.DataFrame(proto).to_csv(ROOT/'organ_state_prototypes.csv',index=False)
np.save(ROOT/'organ_state_prototypes.npy',np.vstack(pvec).astype('float32'))

# Diagnostic LOSO classification on sample centroids. This evaluates representations,
# but full fold-refitted generative validation is reported separately in the report.
X=np.hstack([arrays['scPoli'],arrays['General'],arrays['Brain'],arrays['Liver'],arrays['Bone'],arrays['Lymph_node']])
y=smeta.tissue_site.to_numpy(); pred=np.empty(len(y),object); prob=np.zeros((len(y),len(np.unique(y))))
labels=np.array(sorted(np.unique(y)))
folds=[]
for st in sorted(smeta.GEO.unique()):
    te=smeta.GEO.eq(st).to_numpy(); tr=~te
    present=np.unique(y[tr])
    if len(present)<2: continue
    clf=LogisticRegression(max_iter=3000,class_weight='balanced').fit(StandardScaler().fit_transform(X[tr]),y[tr])
    xp=StandardScaler().fit(X[tr]).transform(X[te])
    pp=clf.predict_proba(xp); pred[te]=clf.classes_[pp.argmax(1)]
    for j,c in enumerate(clf.classes_): prob[te,np.where(labels==c)[0][0]]=pp[:,j]
    folds.append({'study':st,'n_samples':int(te.sum()),'accuracy':float(np.mean(pred[te]==y[te]))})
valid=pd.notna(pred)
res={'n_samples':int(valid.sum()),'balanced_accuracy':float(balanced_accuracy_score(y[valid],pred[valid])),
     'macro_f1':float(f1_score(y[valid],pred[valid],average='macro')),
     'labels':labels.tolist(),'confusion':confusion_matrix(y[valid],pred[valid],labels=labels).tolist(),'folds':folds,
     'validation_note':'diagnostic LOSO head; generative encoders were fitted on all samples'}
(ROOT/'diagnostic_loso.json').write_text(json.dumps(res,indent=2),encoding='utf-8')
out=smeta.copy();out['prediction']=pred
for j,c in enumerate(labels):out[f'p_{c}']=prob[:,j]
out.to_csv(ROOT/'diagnostic_loso_predictions.csv',index=False)

fig,ax=plt.subplots(1,2,figsize=(13,5)); cm=np.asarray(res['confusion'])
sns.heatmap(cm,annot=True,fmt='d',xticklabels=labels,yticklabels=labels,cmap='Blues',ax=ax[0]);ax[0].set(xlabel='Predicted',ylabel='Observed',title='Leave-one-study-out diagnostic')
p=pd.DataFrame(proto); sns.barplot(data=p,x='prototype',y='n_samples',hue='eligible_cross_study',ax=ax[1]);ax[1].tick_params(axis='x',rotation=60);ax[1].set_title('Organ-state prototype support')
fig.tight_layout();fig.savefig(ROOT/'atlas_validation_overview.png',dpi=220);plt.close(fig)
print(json.dumps(res,indent=2),flush=True)



