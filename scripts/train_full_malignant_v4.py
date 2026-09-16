#!/usr/bin/env python3
"""Full 120k strict-malignant training with all-cell reconstruction and sample-balanced/MIL supervision."""
from pathlib import Path
import sys,json,h5py,warnings,os
import numpy as np,pandas as pd,anndata as ad,scipy.sparse as sp
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import balanced_accuracy_score,roc_auc_score,average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt,seaborn as sns
sys.path.insert(0,str(Path(__file__).parent));import train_tumor_only_disentangled_v3 as V3
warnings.filterwarnings('ignore');SEED=20260917;rng=np.random.default_rng(SEED);torch.manual_seed(SEED);torch.set_float32_matmul_precision('high')
ROOT=V3.ROOT;OUT=ROOT/'full_malignant_v4';OUT.mkdir(exist_ok=True);DEV=os.environ.get('BRCA_DEVICE','cuda:0');SOURCE=Path(os.environ.get('BRCA_ALL_DATA_H5AD','/data/Home/liyuhan_codex/PanCanCer_process/fastCNV/BRCA_data_process/Data_Save/BRCA_all_data.h5ad'));META=Path(os.environ.get('BRCA_CELL_METADATA',str(ROOT.parent/'cancer_reference_atlas_v1/cell_metadata.csv.gz')));EXT=V3.EXT

def prepare():
 cache=OUT/'full_strict_malignant_4k.h5ad'
 if cache.exists():return ad.read_h5ad(cache)
 meta=pd.read_csv(META,low_memory=False).sort_values('source_row').reset_index(drop=True);ref=ad.read_h5ad(ROOT/'balanced_malignant_counts_4k.h5ad',backed='r');genes=np.array([g for g in ref.var_names.astype(str) if not(g.startswith('MT-') or g.startswith('RPS') or g.startswith('RPL') or g in ['XIST','RPS4Y1','DDX3Y','KDM5D','UTY'])]);rows=meta.source_row.to_numpy(np.int64);blocks=[]
 with h5py.File(SOURCE,'r') as f:
  g=f['layers/counts'];ptr=g['indptr'][:];raw=f['var/_index'][:];allgenes=np.asarray([x.decode() if isinstance(x,(bytes,np.bytes_)) else str(x) for x in raw]);gix=pd.Index(allgenes).get_indexer(genes)
  for lo in range(0,len(ptr)-1,8000):
   hi=min(lo+8000,len(ptr)-1);sel=np.flatnonzero((rows>=lo)&(rows<hi));
   if not len(sel):continue
   take=rows[sel]-lo;d0,d1=int(ptr[lo]),int(ptr[hi]);b=sp.csr_matrix((g['data'][d0:d1],g['indices'][d0:d1],ptr[lo:hi+1]-d0),shape=(hi-lo,len(allgenes)));blocks.append(b[take][:,gix])
 x=sp.vstack(blocks,format='csr');a=ad.AnnData(x,obs=meta.set_index('cell_id'),var=pd.DataFrame(index=genes));a.layers['counts']=a.X.copy();a.write_h5ad(cache,compression='gzip');return a
def lognorm(x):
 x=sp.csr_matrix(x).astype('float32');s=1e4/np.maximum(np.asarray(x.sum(1)).ravel(),1);x=x.multiply(s[:,None]).tocsr();x.data=np.log1p(x.data);return x
class DS(Dataset):
 def __init__(self,x,d,ids,w):self.x=x;self.d=d;self.ids=np.asarray(ids);self.w=w
 def __len__(self):return len(self.ids)
 def __getitem__(self,k):
  i=self.ids[k];return (torch.from_numpy(self.x[i].toarray().ravel()),)+tuple(torch.tensor(self.d[n][i],dtype=torch.long) for n in ['stage','organ','study','tissue'])+(torch.tensor(self.w[i],dtype=torch.float32),torch.tensor(i))
def wce(logit,y,w,mask=None):
 if mask is not None:logit=logit[mask];y=y[mask];w=w[mask]
 z=F.cross_entropy(logit,y,reduction='none');return (z*w).sum()/w.sum().clamp_min(1e-6)
def cov(a,b):a=a-a.mean(0);b=b-b.mean(0);return ((a.T@b)/max(len(a)-1,1)).pow(2).mean()
def encode(m,x,ids):
 out=[[],[],[]];m.eval()
 with torch.no_grad():
  for q in np.array_split(np.asarray(ids),max(1,int(np.ceil(len(ids)/512)))):
   z=m.latent(torch.from_numpy(x[q].toarray()).to(DEV))
   for k,v in enumerate(z):out[k].append(v.cpu().numpy())
 return [np.vstack(v) for v in out]
def probe(a,y,b,z):
 q=y>=0;r=z>=0;m=make_pipeline(StandardScaler(),LogisticRegression(max_iter=500,class_weight='balanced',C=.5,n_jobs=4)).fit(a[q],y[q]);return float(balanced_accuracy_score(z[r],m.predict(b[r])))
def main():
 a=prepare();o=a.obs.copy();x=lognorm(a.layers['counts']);orgs=np.array(['Lymph_node','Brain','Liver','Bone']);studies=np.array(sorted(o.GEO.astype(str).unique()));tissues=np.array(['Breast','Lymph_node','Brain','Liver','Bone']);d={'stage':np.where(o.disease_stage.astype(str).eq('Metastasis'),1,np.where(o.disease_stage.astype(str).eq('Primary'),0,-1)),'organ':pd.Categorical(o.tissue_site.astype(str),categories=orgs).codes,'study':pd.Categorical(o.GEO.astype(str),categories=studies).codes,'tissue':pd.Categorical(o.tissue_site.astype(str),categories=tissues).codes}
 strata=o.disease_stage.astype(str)+'|'+o.tissue_site.astype(str);tr,va=next(StratifiedGroupKFold(5,shuffle=True,random_state=SEED).split(np.zeros(len(o)),strata,o.GSM.astype(str)));n=o.GSM.astype(str).value_counts();w=(len(o)/o.GSM.astype(str).nunique()/o.GSM.astype(str).map(n)).to_numpy(float);w=w/w.mean()
 loader=DataLoader(DS(x,d,tr,w),batch_size=512,shuffle=True,num_workers=5,pin_memory=True,drop_last=True);m=V3.Net(x.shape[1],len(studies),len(orgs),len(tissues)).to(DEV);opt=torch.optim.AdamW(m.parameters(),lr=4e-4,weight_decay=1e-4);hist=[];best=1e9
 for ep in range(0 if (OUT/'model_cell.pt').exists() else 45):
  m.train();ls=[];alpha=min(.22,.22*ep/15)
  for xx,st,org,study,tissue,ww,_ in loader:
   xx=xx.to(DEV);st=st.to(DEV);org=org.to(DEV);study=study.to(DEV);tissue=tissue.to(DEV);ww=ww.to(DEV);zb,zg,zo,rec,ps,pt,pg,po,ags,agt,aos=m(xx,alpha);q=st>=0;qo=(st==1)&(org>=0);loss=.30*F.mse_loss(rec,xx)+.35*(wce(ps,study,ww)+wce(pt,tissue,ww))+1.0*wce(pg,st,ww,q)+1.1*wce(po,org,ww,qo)+.22*(wce(ags,study,ww)+wce(agt,tissue,ww)+wce(aos,study,ww))+.35*(cov(zb,zg)+cov(zb,zo)+cov(zg,zo));opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),5);opt.step();ls.append(float(loss.detach()))
  hist.append(np.mean(ls));print('cell',ep,hist[-1],flush=True)
  if hist[-1]<best:best=hist[-1];torch.save(m.state_dict(),OUT/'model_cell.pt')
 # Sample-level multiple-instance fine-tuning: each sample contributes one loss, top 20% cells define bag state.
 m.load_state_dict(torch.load(OUT/'model_cell.pt',map_location=DEV));opt=torch.optim.AdamW(list(m.enc.parameters())+list(m.zg.parameters())+list(m.general.parameters()),lr=8e-5,weight_decay=1e-4);sample_groups={g:np.intersect1d(ii,tr) for g,ii in o.groupby(o.GSM.astype(str)).indices.items()};sample_groups={g:ii for g,ii in sample_groups.items() if len(ii) and np.unique(d['stage'][ii][d['stage'][ii]>=0]).size==1};keys=list(sample_groups);mh=[]
 for ep in range(18):
  rng.shuffle(keys);ls=[]
  for start in range(0,len(keys),8):
   losses=[]
   for g in keys[start:start+8]:
    ii=sample_groups[g];ii=rng.choice(ii,min(256,len(ii)),replace=False);yy=d['stage'][ii[0]]
    if yy<0:continue
    xx=torch.from_numpy(x[ii].toarray()).to(DEV);_,zg,_=m.latent(xx);log=m.general(zg);score=log[:,1]-log[:,0];k=min(len(ii),max(1,int(.2*len(ii))));top=torch.topk(score,k).indices;bag=log[top].mean(0,keepdim=True);losses.append(F.cross_entropy(bag,torch.tensor([yy],device=DEV)))
   if losses:loss=torch.stack(losses).mean();opt.zero_grad();loss.backward();opt.step();ls.append(float(loss.detach()))
  mh.append(np.mean(ls));print('mil',ep,mh[-1],flush=True)
 torch.save(m.state_dict(),OUT/'model.pt');ztr=encode(m,x,tr);zv=encode(m,x,va);names=['batch','general','organ'];probes={}
 for k,nm in enumerate(names):probes[nm]={'stage':probe(ztr[k],d['stage'][tr],zv[k],d['stage'][va]),'organ':probe(ztr[k][d['stage'][tr]==1],d['organ'][tr][d['stage'][tr]==1],zv[k][d['stage'][va]==1],d['organ'][va][d['stage'][va]==1]),'study':probe(ztr[k],d['study'][tr],zv[k],d['study'][va]),'tissue':probe(ztr[k],d['tissue'][tr],zv[k],d['tissue'][va])}
 with torch.no_grad():gp=F.softmax(m.general(torch.tensor(zv[1],device=DEV)),1).cpu().numpy();op=F.softmax(m.organ(torch.tensor(zv[2],device=DEV)),1).cpu().numpy();q=d['stage'][va]>=0;qo=d['stage'][va]==1
 met={'n_cells':len(o),'n_genes':x.shape[1],'n_samples':o.GSM.nunique(),'cell_general_auc':float(roc_auc_score(d['stage'][va][q],gp[q,1])),'cell_general_ap':float(average_precision_score(d['stage'][va][q],gp[q,1])),'cell_general_balanced_accuracy':float(balanced_accuracy_score(d['stage'][va][q],gp[q].argmax(1))),'cell_organ_balanced_accuracy':float(balanced_accuracy_score(d['organ'][va][qo],op[qo].argmax(1))),'probes':probes}
 tmp=pd.DataFrame({'sample':o.GSM.astype(str).iloc[va].to_numpy(),'stage':d['stage'][va],'p':gp[:,1]});tmp=tmp[tmp.stage>=0].groupby('sample').agg(stage=('stage','first'),p=('p',lambda v:np.mean(np.sort(v)[-max(1,int(.2*len(v))):])));met['sample_general_auc']=float(roc_auc_score(tmp.stage,tmp.p));met['n_validation_samples']=len(tmp)
 if EXT.exists():
  e=ad.read_h5ad(EXT);ix=pd.Index(e.var_names).get_indexer(a.var_names);ex=lognorm(e.layers['counts'][:,ix]);ez=encode(m,ex,np.arange(e.n_obs));
  with torch.no_grad():eg=F.softmax(m.general(torch.tensor(ez[1],device=DEV)),1).cpu().numpy();eo=F.softmax(m.organ(torch.tensor(ez[2],device=DEV)),1).cpu().numpy();truth=e.obs.true_site.astype(str).to_numpy();sel=np.isin(truth,['Primary','Positive_LN']);yy=(truth[sel]=='Positive_LN').astype(int);li=np.flatnonzero(orgs=='Lymph_node')[0];met['external_GSE158399']={'general_auc':float(roc_auc_score(yy,eg[sel,1])),'ln_auc':float(roc_auc_score(yy,eo[sel,li])),'mean_general_by_site':pd.Series(eg[:,1]).groupby(truth).mean().to_dict(),'mean_ln_by_site':pd.Series(eo[:,li]).groupby(truth).mean().to_dict()}
 torch.save({'state_dict':m.cpu().state_dict(),'genes':a.var_names.tolist(),'studies':studies.tolist(),'organs':orgs.tolist(),'tissues':tissues.tolist()},OUT/'full_malignant_model.pt');(OUT/'metrics.json').write_text(json.dumps(met,indent=2));pd.DataFrame({'epoch':range(1,len(hist)+1),'cell_loss':hist}).to_csv(OUT/'cell_history.csv',index=False);pd.DataFrame({'epoch':range(1,len(mh)+1),'mil_loss':mh}).to_csv(OUT/'mil_history.csv',index=False)
 mat=pd.DataFrame(probes).T[['stage','organ','study','tissue']];fig,ax=plt.subplots(1,2,figsize=(12,4.8));sns.heatmap(mat,annot=True,vmin=0,vmax=1,cmap='viridis',ax=ax[0]);ax[0].set_title('Full-120K information probes');ax[1].plot(hist,label='all-cell');ax[1].plot(np.arange(len(mh))+len(hist),mh,label='MIL');ax[1].legend();ax[1].set(title='Training loss',xlabel='Epoch');fig.tight_layout();fig.savefig(OUT/'validation_overview.png',dpi=240);print(json.dumps(met,indent=2),flush=True)
if __name__=='__main__':main()

