#!/usr/bin/env python3
"""Six-class descriptive cell atlas: DCIS, primary and four metastatic sites."""
from pathlib import Path
import json, os
import numpy as np, pandas as pd, anndata as ad, torch
from torch import nn
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, silhouette_score, confusion_matrix
from scvi.external import ContrastiveVI
import umap
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

ROOT=Path(os.environ.get('BRCA_ATLAS_ROOT','/data/Home/liyuhan_codex/PanCanCer_process/fastCNV/BRCA_data_process/scimilarity_research/multiscale_metastasis_atlas_v1'))
SCREEN=ROOT/'hierarchical_gene_screen';SEED=1701;rng=np.random.default_rng(SEED);torch.manual_seed(SEED);torch.set_float32_matmul_precision('high')
a=ad.read_h5ad(ROOT/'balanced_malignant_counts_4k.h5ad');o=a.obs.reset_index(names='cell_id')

# Add the three stage-specific salient representations to the previous reference features.
parts=[np.load(ROOT/'scpoli_latent.npy')]
part_names=['scPoli']
for n in ['General','Brain','Liver','Bone','Lymph_node']:
 parts.append(np.load(ROOT/f'contrastivevi_{n}_salient.npy'));part_names.append(n)
ContrastiveVI.setup_anndata(a,layer='counts',batch_key='GEO')
for n in ['Invasion_DCIS_to_Primary','Metastatic_ability','Distant_spread']:
 f=SCREEN/f'{n}_salient.npy'
 if f.exists():z=np.load(f)
 else:
  m=ContrastiveVI.load(str(SCREEN/f'contrastivevi_{n}'),adata=a)
  z=np.asarray(m.get_latent_representation(a,representation_kind='salient'),dtype='float32');np.save(f,z)
 parts.append(z);part_names.append(n)
X=np.hstack(parts).astype('float32');mu=X.mean(0);sd=np.maximum(X.std(0),1e-5);X=(X-mu)/sd

label=np.where(o.disease_stage.eq('DCIS'),'DCIS',np.where(o.disease_stage.eq('Primary'),'Primary',o.tissue_site)).astype(str)
classes=np.array(['DCIS','Primary','Lymph_node','Brain','Liver','Bone']);studies=np.array(sorted(o.GEO.unique()))
y=pd.Categorical(label,categories=classes).codes.astype('int64');d=pd.Categorical(o.GEO,categories=studies).codes.astype('int64');samples=o.GSM.astype(str).to_numpy();dev='cuda'

class GR(torch.autograd.Function):
 @staticmethod
 def forward(ctx,x,a):ctx.a=a;return x.view_as(x)
 @staticmethod
 def backward(ctx,g):return -ctx.a*g,None
class Net(nn.Module):
 def __init__(self):
  super().__init__();self.map=nn.Sequential(nn.Linear(X.shape[1],128),nn.LayerNorm(128),nn.GELU(),nn.Dropout(.1),nn.Linear(128,40));self.cls=nn.Linear(40,len(classes));self.dom=nn.Linear(40,len(studies))
 def latent(self,x):return F.normalize(self.map(x),dim=1)
def supcon(z,yy,temp=.12):
 sim=z@z.T/temp;sim-=sim.max(1,keepdim=True).values.detach();eye=torch.eye(len(z),device=z.device,dtype=torch.bool);pos=(yy[:,None]==yy[None,:])&~eye;den=torch.logsumexp(sim.masked_fill(eye,-1e9),1);return -((sim-den[:,None])*pos).sum()/pos.sum().clamp_min(1)

pool={c:{g:np.flatnonzero((y==c)&(samples==g)) for g in np.unique(samples[y==c])} for c in range(len(classes))}
net=Net().to(dev);opt=torch.optim.AdamW(net.parameters(),lr=5e-4,weight_decay=1e-3)
for step in range(2200):
 ids=[]
 for c in range(len(classes)):
  gs=list(pool[c]);chosen=rng.choice(gs,56,replace=True);ids += [int(rng.choice(pool[c][g])) for g in chosen]
 ids=np.asarray(ids);xx=torch.as_tensor(X[ids],device=dev);yy=torch.as_tensor(y[ids],device=dev);dd=torch.as_tensor(d[ids],device=dev)
 z=net.latent(xx);ce=F.cross_entropy(net.cls(z),yy);con=supcon(z,yy);adv=F.cross_entropy(net.dom(GR.apply(z,.08)),dd);loss=ce+.18*con+.08*adv
 opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(net.parameters(),5);opt.step()
 if step%400==0:print(step,float(loss),float(ce),flush=True)

net.eval();Z=[];P=[]
with torch.no_grad():
 for q in np.array_split(np.arange(len(X)),int(np.ceil(len(X)/2048))):
  z=net.latent(torch.as_tensor(X[q],device=dev));Z.append(z.cpu().numpy());P.append(F.softmax(net.cls(z),1).cpu().numpy())
Z=np.vstack(Z).astype('float32');P=np.vstack(P).astype('float32');np.save(ROOT/'stage_organ_latent_with_DCIS.npy',Z);np.save(ROOT/'stage_organ_probabilities_with_DCIS.npy',P)
torch.save({'state_dict':net.cpu().state_dict(),'classes':classes.tolist(),'studies':studies.tolist(),'parts':part_names,'feature_mean':mu,'feature_sd':sd},ROOT/'stage_organ_map_with_DCIS.pt')

show=[]
for _,ii in o.groupby('GSM',sort=True).groups.items():
 ii=np.asarray(list(ii));show.extend(rng.choice(ii,min(250,len(ii)),replace=False))
show=np.asarray(show);xy=umap.UMAP(n_neighbors=35,min_dist=.18,metric='cosine',random_state=SEED).fit_transform(Z[show]);np.save(ROOT/'stage_organ_display_umap_with_DCIS.npy',xy);np.save(ROOT/'stage_organ_display_indices_with_DCIS.npy',show)
pal={'DCIS':'#E8A5C6','Primary':'#D95F5F','Lymph_node':'#8C65C6','Brain':'#4C78A8','Liver':'#59A14F','Bone':'#E39C37'}
disp={'DCIS':'DCIS (non-invasive)','Primary':'Primary invasive breast','Lymph_node':'Lymph-node metastasis','Brain':'Brain metastasis','Liver':'Liver metastasis','Bone':'Bone metastasis'}
fig,ax=plt.subplots(figsize=(11,8.5))
for c in classes:
 q=label[show]==c;ax.scatter(xy[q,0],xy[q,1],s=4,c=pal[c],label=disp[c],alpha=.58,rasterized=True);cx,cy=np.median(xy[q],axis=0);ax.text(cx,cy,disp[c],ha='center',va='center',fontsize=11,fontweight='bold',color='white',path_effects=[pe.withStroke(linewidth=3.2,foreground='#333')])
ax.legend(markerscale=4,frameon=False,loc='best');ax.set(xticks=[],yticks=[],xlabel='UMAP1',ylabel='UMAP2',title='BRCA malignant-cell progression and organ reference map')
fig.tight_layout();fig.savefig(ROOT/'stage_organ_single_cell_map_with_DCIS.png',dpi=240);plt.close(fig)

pred=P.argmax(1);ids=rng.choice(len(Z),min(15000,len(Z)),replace=False)
m={'n_cells':len(Z),'n_display_cells':len(show),'classes':classes.tolist(),'class_cell_counts':pd.Series(label).value_counts().to_dict(),
   'training_balanced_accuracy':float(balanced_accuracy_score(y,pred)),'training_class_silhouette':float(silhouette_score(Z[ids],y[ids])),
   'training_study_silhouette':float(silhouette_score(Z[ids],d[ids])),'confusion':confusion_matrix(y,pred,labels=range(len(classes))).tolist(),
   'warning':'label-supervised descriptive map; DCIS comes from one study and is not cross-study validated'}
(ROOT/'stage_organ_map_with_DCIS_metrics.json').write_text(json.dumps(m,indent=2),encoding='utf-8');print(json.dumps(m,indent=2),flush=True)



