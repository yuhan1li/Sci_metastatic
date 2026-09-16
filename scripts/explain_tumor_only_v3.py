#!/usr/bin/env python3
"""Integrated gradients, perturbation and cross-sample stability for tumor-only v3."""
from pathlib import Path
import sys,json
import numpy as np,pandas as pd,torch
import torch.nn.functional as F
sys.path.insert(0,str(Path(__file__).parent));import train_tumor_only_disentangled_v3 as T
ROOT=T.ROOT;OUT=T.OUT/'gene_attribution';OUT.mkdir(exist_ok=True);rng=np.random.default_rng(T.SEED)

def balanced_ids(obs,mask,cap_sample=35,maxn=1400):
 ids=[]
 pos=np.flatnonzero(np.asarray(mask)); gsm=obs.GSM.astype(str).to_numpy()
 for g in np.unique(gsm[pos]):
  ii=pos[gsm[pos]==g];ids.extend(rng.choice(ii,min(cap_sample,len(ii)),replace=False))
 ids=np.asarray(ids,dtype=int)
 return rng.choice(ids,maxn,replace=False) if len(ids)>maxn else ids
def ig(model,x,ids,baseline,space,head,targets,steps=20,bs=96):
 attrs=[]
 for ch in np.array_split(ids,max(1,int(np.ceil(len(ids)/bs)))):
  raw=torch.from_numpy(x[ch].toarray()).to(T.DEV);base=torch.tensor(baseline,device=T.DEV).expand_as(raw);grad=0
  for a in torch.linspace(.05,1,steps,device=T.DEV):
   xx=(base+a*(raw-base)).detach().requires_grad_(True);zb,zg,zo=model.latent(xx);z={'batch':zb,'general':zg,'organ':zo}[space];logit=head(z);tar=torch.tensor(targets[ch],device=T.DEV,dtype=torch.long);v=logit.gather(1,tar[:,None]).sum();grad+=torch.autograd.grad(v,xx)[0]
  attrs.append(((raw-base)*grad/steps).detach().cpu().numpy())
 return np.vstack(attrs)
def perturb(model,x,ids,baseline,head,space,target,genes):
 xx=torch.from_numpy(x[ids].toarray()).to(T.DEV)
 with torch.no_grad():
  z={'general':model.latent(xx)[1],'organ':model.latent(xx)[2]}[space];orig=F.softmax(head(z),1)[:,target]
 out=[]
 for g in genes:
  q=xx.clone();q[:,g]=float(baseline[g])
  with torch.no_grad():z={'general':model.latent(q)[1],'organ':model.latent(q)[2]}[space];p=F.softmax(head(z),1)[:,target]
  out.append(float((orig-p).mean().cpu()))
 return np.asarray(out)
def rank_table(genes,attr,groups,bpen,pert=None):
 signed=attr.mean(0);ab=np.abs(attr).mean(0);df=pd.DataFrame({'gene':genes,'IG_signed':signed,'IG_abs':ab,'batch_penalty':bpen})
 sm=pd.DataFrame(attr).assign(group=np.asarray(groups)).groupby('group').mean();df['direction_consistency']=np.maximum((sm>0).mean(0).to_numpy(),(sm<0).mean(0).to_numpy())
 if pert is not None:
  df['perturbation_delta']=0.;df.loc[pert[0],'perturbation_delta']=pert[1]
 def pct(v):return pd.Series(v).rank(pct=True).to_numpy()
 df['score']=.38*pct(df.IG_abs)+.22*pct(np.abs(df.IG_signed))+.20*pct(df.direction_consistency)+(.20*pct(np.maximum(df.get('perturbation_delta',0),0)) if pert is not None else .20*pct(df.IG_abs))-.28*pct(df.batch_penalty)
 df=df.sort_values('score',ascending=False).reset_index(drop=True);df['single_cell_rank']=np.arange(1,len(df)+1);return df
def main():
 x,o,genes,d,tr,va,cstage,corg,studies,tissues=T.load_data();m=T.Net(x.shape[1],len(studies),len(corg),len(tissues)).to(T.DEV);m.load_state_dict(torch.load(T.OUT/'model.pt',map_location=T.DEV));m.eval();primary=np.flatnonzero(d['stage']==0);baseline=np.asarray(x[rng.choice(primary,min(2000,len(primary)),replace=False)].mean(0)).ravel().astype('float32')
 # Nuisance attribution penalty: correct study + correct tissue logits.
 bid=balanced_ids(o,np.ones(len(o),dtype=bool),20,1000);a1=ig(m,x,bid,baseline,'batch',m.study,d['study'],12);a2=ig(m,x,bid,baseline,'batch',m.tissue,d['tissue'],12);bpen=(np.abs(a1).mean(0)+np.abs(a2).mean(0))/2
 # General metastasis attribution.
 gid=balanced_ids(o,d['stage']==1,35,1400);tar=np.ones(len(o),dtype=int);ga=ig(m,x,gid,baseline,'general',m.general,tar,20);top=np.argsort(-np.abs(ga).mean(0))[:300];pdlt=perturb(m,x,gid[:700],baseline,m.general,'general',1,top);gdf=rank_table(genes,ga,o.GSM.astype(str).iloc[gid].to_numpy(),bpen,(top,pdlt));gdf.to_csv(OUT/'general_metastasis_gene_ranking.csv',index=False)
 summaries={'general_top20':gdf.head(20).gene.tolist()}
 # Organ-specific attributions use other-metastasis mean as baseline and subtract same nuisance penalty.
 for k,name in enumerate(corg):
  ids=balanced_ids(o,(d['stage']==1)&(d['organ']==k),40,900);other=np.flatnonzero((d['stage']==1)&(d['organ']!=k));base=np.asarray(x[rng.choice(other,min(2000,len(other)),replace=False)].mean(0)).ravel().astype('float32');targets=np.full(len(o),k,dtype=int);aa=ig(m,x,ids,base,'organ',m.organ,targets,20);tp=np.argsort(-np.abs(aa).mean(0))[:150];dd=perturb(m,x,ids[:500],base,m.organ,'organ',k,tp);df=rank_table(genes,aa,o.GSM.astype(str).iloc[ids].to_numpy(),bpen,(tp,dd));df.to_csv(OUT/f'{name}_gene_ranking.csv',index=False);summaries[f'{name}_top20']=df.head(20).gene.tolist()
 (OUT/'summary.json').write_text(json.dumps(summaries,indent=2));print(json.dumps(summaries,indent=2),flush=True)
if __name__=='__main__':main()

