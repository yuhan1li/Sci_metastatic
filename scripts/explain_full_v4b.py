#!/usr/bin/env python3
"""Run integrated-gradient attribution for the 120K model."""
from pathlib import Path
import numpy as np, pandas as pd, anndata as ad
import train_tumor_only_disentangled_v3 as T
import train_full_malignant_v4 as F

MODEL_OUT=T.ROOT/'full_balanced_split_v4b'

def load_full():
    a=ad.read_h5ad(F.OUT/'full_strict_malignant_4k.h5ad')
    x=F.lognorm(a.layers['counts']);o=a.obs.copy();genes=np.asarray(a.var_names.astype(str))
    cstage=np.array(['Primary','Metastasis']);corg=np.array(['Lymph_node','Brain','Liver','Bone'])
    studies=np.array(sorted(o.GEO.astype(str).unique()));tissues=np.array(['Breast','Lymph_node','Brain','Liver','Bone'])
    stage=np.where(o.disease_stage.astype(str).eq('Metastasis'),1,np.where(o.disease_stage.astype(str).eq('Primary'),0,-1))
    organ=pd.Categorical(o.tissue_site.astype(str),categories=corg).codes
    study=pd.Categorical(o.GEO.astype(str),categories=studies).codes
    tissue=pd.Categorical(o.tissue_site.astype(str),categories=tissues).codes
    val_samples=set((MODEL_OUT/'split_samples.txt').read_text().splitlines())
    is_val=o.GSM.astype(str).isin(val_samples).to_numpy();va=np.flatnonzero(is_val);tr=np.flatnonzero(~is_val)
    return x,o,genes,dict(stage=stage,organ=organ,study=study,tissue=tissue),tr,va,cstage,corg,studies,tissues

T.OUT=MODEL_OUT
T.load_data=load_full
import explain_tumor_only_v3 as E

if __name__=='__main__': E.main()

