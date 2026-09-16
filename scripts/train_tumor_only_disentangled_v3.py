#!/usr/bin/env python3
"""Tumor-only disentangled model: nuisance, general metastasis and organ-specific spaces."""
from pathlib import Path
import json,warnings,os
import numpy as np,pandas as pd,anndata as ad,scipy.sparse as sp
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader,WeightedRandomSampler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import balanced_accuracy_score,roc_auc_score,average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt,seaborn as sns
warnings.filterwarnings('ignore');SEED=20260916;rng=np.random.default_rng(SEED);torch.manual_seed(SEED);torch.set_float32_matmul_precision('high')
ROOT=Path(os.environ.get('BRCA_ATLAS_ROOT','/data/Home/liyuhan_codex/PanCanCer_process/fastCNV/BRCA_data_process/scimilarity_research/multiscale_metastasis_atlas_v1'));OUT=ROOT/'tumor_only_v3';OUT.mkdir(exist_ok=True)
EXT=Path(os.environ.get('BRCA_EXTERNAL_H5AD',str(ROOT.parent/'external_validation_GSE158399/results/GSE158399_epithelial_query_4k.h5ad')));DEV=os.environ.get('BRCA_DEVICE','cuda:0')

def lognorm(x):
 x=sp.csr_matrix(x).astype('float32');s=1e4/np.maximum(np.asarray(x.sum(1)).ravel(),1);x=x.multiply(s[:,None]).tocsr();x.data=np.log1p(x.data);return x
def load_data():
 a=ad.read_h5ad(ROOT/'balanced_malignant_counts_4k.h5ad');genes=np.array([g for g in a.var_names.astype(str) if not(g.startswith('MT-') or g.startswith('RPS') or g.startswith('RPL') or g in ['XIST','RPS4Y1','DDX3Y','KDM5D','UTY'])]);ix=pd.Index(a.var_names).get_indexer(genes);x=lognorm(a.layers['counts'][:,ix]);o=a.obs.copy()
 classes_stage=np.array(['Primary','Metastasis']);classes_org=np.array(['Lymph_node','Brain','Liver','Bone']);studies=np.array(sorted(o.GEO.astype(str).unique()));tissues=np.array(['Breast','Lymph_node','Brain','Liver','Bone'])
 stage=np.where(o.disease_stage.astype(str).eq('Metastasis'),1,np.where(o.disease_stage.astype(str).eq('Primary'),0,-1));organ=pd.Categorical(o.tissue_site.astype(str),categories=classes_org).codes;study=pd.Categorical(o.GEO.astype(str),categories=studies).codes;tissue=pd.Categorical(o.tissue_site.astype(str),categories=tissues).codes
 strata=o.disease_stage.astype(str)+'|'+o.tissue_site.astype(str);sg=StratifiedGroupKFold(5,shuffle=True,random_state=SEED);tr,va=next(sg.split(np.zeros(len(o)),strata,groups=o.GSM.astype(str)))
 return x,o,genes,dict(stage=stage,organ=organ,study=study,tissue=tissue),tr,va,classes_stage,classes_org,studies,tissues
class DS(Dataset):
 def __init__(self,x,d,ids):self.x=x;self.d=d;self.ids=np.asarray(ids)
 def __len__(self):return len(self.ids)
 def __getitem__(self,k):
  i=self.ids[k];return (torch.from_numpy(self.x[i].toarray().ravel()),)+tuple(torch.tensor(self.d[n][i],dtype=torch.long) for n in ['stage','organ','study','tissue'])+(torch.tensor(i),)
class GR(torch.autograd.Function):
 @staticmethod
 def forward(ctx,x,a):ctx.a=a;return x.view_as(x)
 @staticmethod
 def backward(ctx,g):return -ctx.a*g,None
class Net(nn.Module):
 def __init__(self,ng,ns,no,nt):
  super().__init__();self.enc=nn.Sequential(nn.Linear(ng,512),nn.LayerNorm(512),nn.GELU(),nn.Dropout(.12),nn.Linear(512,160),nn.LayerNorm(160),nn.GELU());self.zb=nn.Linear(160,24);self.zg=nn.Linear(160,12);self.zo=nn.Linear(160,16);self.dec=nn.Sequential(nn.Linear(52,256),nn.GELU(),nn.Linear(256,ng));self.study=nn.Linear(24,ns);self.tissue=nn.Linear(24,nt);self.general=nn.Linear(12,2);self.organ=nn.Linear(16,no);self.adv_g_study=nn.Linear(12,ns);self.adv_g_tissue=nn.Linear(12,nt);self.adv_o_study=nn.Linear(16,ns)
 def latent(self,x):h=self.enc(x);return self.zb(h),F.normalize(self.zg(h),dim=1),F.normalize(self.zo(h),dim=1)
 def forward(self,x,a=.1):
  zb,zg,zo=self.latent(x);return zb,zg,zo,self.dec(torch.cat([zb,zg,zo],1)),self.study(zb),self.tissue(zb),self.general(zg),self.organ(zo),self.adv_g_study(GR.apply(zg,a)),self.adv_g_tissue(GR.apply(zg,a)),self.adv_o_study(GR.apply(zo,a))
def cov(a,b):a=a-a.mean(0);b=b-b.mean(0);return ((a.T@b)/max(len(a)-1,1)).pow(2).mean()
def encode(m,x,ids):
 out=[[],[],[]];m.eval()
 with torch.no_grad():
  for q in np.array_split(np.asarray(ids),max(1,int(np.ceil(len(ids)/512)))):
   z=m.latent(torch.from_numpy(x[q].toarray()).to(DEV))
   for k,v in enumerate(z):out[k].append(v.cpu().numpy())
 return [np.vstack(v) for v in out]
def probe(a,y,b,z):
 q=y>=0;r=z>=0
 if len(np.unique(y[q]))<2 or len(np.unique(z[r]))<2:return None
 m=make_pipeline(StandardScaler(),LogisticRegression(max_iter=500,class_weight='balanced',C=.5,n_jobs=4)).fit(a[q],y[q]);return float(balanced_accuracy_score(z[r],m.predict(b[r])))
def main():
 x,o,genes,d,tr,va,cstage,corg,studies,tissues=load_data();counts=pd.Series(o.GSM.astype(str).iloc[tr]).value_counts();w=np.array([1/counts[g] for g in o.GSM.astype(str).iloc[tr]]);sampler=WeightedRandomSampler(w,len(tr),replacement=True,generator=torch.Generator().manual_seed(SEED));loader=DataLoader(DS(x,d,tr),batch_size=384,sampler=sampler,num_workers=4,pin_memory=True,drop_last=True)
 m=Net(x.shape[1],len(studies),len(corg),len(tissues)).to(DEV);opt=torch.optim.AdamW(m.parameters(),lr=4e-4,weight_decay=1e-4);hist=[];best=1e9
 for ep in range(65):
  m.train();ls=[];a=min(.20,.20*ep/20)
  for xx,stage,organ,study,tissue,_ in loader:
   xx=xx.to(DEV);stage=stage.to(DEV);organ=organ.to(DEV);study=study.to(DEV);tissue=tissue.to(DEV);zb,zg,zo,rec,ps,pt,pg,po,ags,agt,aos=m(xx,a);loss=.28*F.mse_loss(rec,xx)+.35*(F.cross_entropy(ps,study)+F.cross_entropy(pt,tissue));q=stage>=0;loss+=F.cross_entropy(pg[q],stage[q]);qo=(stage==1)&(organ>=0);loss+=1.1*F.cross_entropy(po[qo],organ[qo]);loss+=.18*(F.cross_entropy(ags,study)+F.cross_entropy(agt,tissue)+F.cross_entropy(aos,study));loss+=.35*(cov(zb,zg)+cov(zb,zo)+cov(zg,zo));opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(m.parameters(),5);opt.step();ls.append(float(loss.detach()))
  hist.append(np.mean(ls));print(ep,hist[-1],flush=True)
  if hist[-1]<best:best=hist[-1];torch.save(m.state_dict(),OUT/'model.pt')
 m.load_state_dict(torch.load(OUT/'model.pt',map_location=DEV));ztr=encode(m,x,tr);zv=encode(m,x,va);names=['batch','general','organ'];probes={}
 for k,n in enumerate(names):probes[n]={'stage':probe(ztr[k],d['stage'][tr],zv[k],d['stage'][va]),'organ':probe(ztr[k][d['stage'][tr]==1],d['organ'][tr][d['stage'][tr]==1],zv[k][d['stage'][va]==1],d['organ'][va][d['stage'][va]==1]),'study':probe(ztr[k],d['study'][tr],zv[k],d['study'][va]),'tissue':probe(ztr[k],d['tissue'][tr],zv[k],d['tissue'][va])}
 with torch.no_grad():gp=F.softmax(m.general(torch.tensor(zv[1],device=DEV)),1).cpu().numpy();op=F.softmax(m.organ(torch.tensor(zv[2],device=DEV)),1).cpu().numpy()
 q=d['stage'][va]>=0;qo=d['stage'][va]==1;metrics={'n_cells':len(o),'n_genes':len(genes),'split':'sample-disjoint stratified 5-fold fold0','cell_general_auc':float(roc_auc_score(d['stage'][va][q],gp[q,1])),'cell_general_ap':float(average_precision_score(d['stage'][va][q],gp[q,1])),'cell_general_balanced_accuracy':float(balanced_accuracy_score(d['stage'][va][q],gp[q].argmax(1))),'cell_organ_balanced_accuracy':float(balanced_accuracy_score(d['organ'][va][qo],op[qo].argmax(1))),'probes':probes}
 # sample-level aggregation
 tmp=pd.DataFrame({'sample':o.GSM.astype(str).iloc[va].to_numpy(),'stage':d['stage'][va],'gp':gp[:,1]});tmp=tmp[tmp.stage>=0].groupby('sample').agg(stage=('stage','first'),gp=('gp','mean'));metrics['sample_general_auc']=float(roc_auc_score(tmp.stage,tmp.gp));metrics['n_validation_samples_general']=len(tmp)
 # frozen external mapping
 if EXT.exists():
  qd=ad.read_h5ad(EXT);ix=pd.Index(qd.var_names).get_indexer(genes);qx=lognorm(qd.layers['counts'][:,ix]);qz=encode(m,qx,np.arange(qd.n_obs));
  with torch.no_grad():qg=F.softmax(m.general(torch.tensor(qz[1],device=DEV)),1).cpu().numpy();qop=F.softmax(m.organ(torch.tensor(qz[2],device=DEV)),1).cpu().numpy()
  true=qd.obs.true_site.astype(str).to_numpy();sel=np.isin(true,['Primary','Positive_LN']);y=(true[sel]=='Positive_LN').astype(int);li=np.flatnonzero(corg=='Lymph_node')[0];metrics['external_GSE158399']={'general_auc':float(roc_auc_score(y,qg[sel,1])),'ln_organ_auc':float(roc_auc_score(y,qop[sel,li])),'mean_general_by_site':pd.Series(qg[:,1]).groupby(true).mean().to_dict(),'mean_ln_by_site':pd.Series(qop[:,li]).groupby(true).mean().to_dict()}
 np.savez_compressed(OUT/'validation_latents.npz',train=tr,val=va,zb_train=ztr[0],zg_train=ztr[1],zo_train=ztr[2],zb_val=zv[0],zg_val=zv[1],zo_val=zv[2]);torch.save({'state_dict':m.cpu().state_dict(),'genes':genes.tolist(),'studies':studies.tolist(),'organs':corg.tolist(),'tissues':tissues.tolist()},OUT/'tumor_only_disentangled.pt');(OUT/'metrics.json').write_text(json.dumps(metrics,indent=2));pd.DataFrame({'epoch':np.arange(len(hist))+1,'loss':hist}).to_csv(OUT/'history.csv',index=False)
 mat=pd.DataFrame(probes).T[['stage','organ','study','tissue']];fig,ax=plt.subplots(1,2,figsize=(12,4.8));sns.heatmap(mat,annot=True,vmin=0,vmax=1,cmap='viridis',ax=ax[0]);ax[0].set_title('Information-probe matrix');ax[1].plot(np.arange(len(hist))+1,hist);ax[1].set(title='Training loss',xlabel='Epoch',ylabel='Loss');fig.tight_layout();fig.savefig(OUT/'validation_overview.png',dpi=240);print(json.dumps(metrics,indent=2),flush=True)
if __name__=='__main__':main()

