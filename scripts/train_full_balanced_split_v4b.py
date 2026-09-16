#!/usr/bin/env python3
"""Fair Full-120K comparison using the exact validation samples and model recipe of tumor-only v3."""
from pathlib import Path
import numpy as np,pandas as pd,anndata as ad
import train_tumor_only_disentangled_v3 as V3
import train_full_malignant_v4 as F

OUT=V3.ROOT/'full_balanced_split_v4b';OUT.mkdir(exist_ok=True)
ORIGINAL_V3_LOAD_DATA=V3.load_data

def load_full():
    a=F.prepare();x=F.lognorm(a.layers['counts']);o=a.obs.copy();genes=np.asarray(a.var_names.astype(str))
    cstage=np.array(['Primary','Metastasis']);corg=np.array(['Lymph_node','Brain','Liver','Bone']);studies=np.array(sorted(o.GEO.astype(str).unique()));tissues=np.array(['Breast','Lymph_node','Brain','Liver','Bone'])
    stage=np.where(o.disease_stage.astype(str).eq('Metastasis'),1,np.where(o.disease_stage.astype(str).eq('Primary'),0,-1));organ=pd.Categorical(o.tissue_site.astype(str),categories=corg).codes;study=pd.Categorical(o.GEO.astype(str),categories=studies).codes;tissue=pd.Categorical(o.tissue_site.astype(str),categories=tissues).codes
    # Reuse the exact v3 validation GSMs for a fair 44K-versus-120K comparison.
    _,bo,_,_,_,bva,_,_,_,_=ORIGINAL_V3_LOAD_DATA();val_samples=set(bo.GSM.astype(str).iloc[bva]);is_val=o.GSM.astype(str).isin(val_samples).to_numpy();va=np.flatnonzero(is_val);tr=np.flatnonzero(~is_val)
    (OUT/'split_samples.txt').write_text('\n'.join(sorted(val_samples)))
    return x,o,genes,dict(stage=stage,organ=organ,study=study,tissue=tissue),tr,va,cstage,corg,studies,tissues

V3.OUT=OUT
V3.load_data=load_full
V3.main()

