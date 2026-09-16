#!/usr/bin/env python3
"""Train scPoli and five contrastiveVI reference components."""
from pathlib import Path
import json, traceback, os
import numpy as np
import pandas as pd
import anndata as ad
import torch
from sklearn.metrics import silhouette_score
from scarches.models.scpoli import scPoli
from scvi.external import ContrastiveVI

ROOT = Path(os.environ.get('BRCA_ATLAS_ROOT','/data/Home/liyuhan_codex/PanCanCer_process/fastCNV/BRCA_data_process/scimilarity_research/multiscale_metastasis_atlas_v1'))
DATA = ROOT / 'balanced_malignant_counts_4k.h5ad'
SEED = 1701
np.random.seed(SEED); torch.manual_seed(SEED)
adata = ad.read_h5ad(DATA)
adata.obs['condition_study'] = adata.obs['condition_study'].astype(str)
adata.obs['malignant_label'] = adata.obs['malignant_label'].astype(str)

# Pass 1 scPoli: study is modeled explicitly; organ is never supplied as nuisance.
if (ROOT/'scpoli_latent.npy').exists() and (ROOT/'scpoli_reference').exists():
    z=np.load(ROOT/'scpoli_latent.npy')
    print('Reusing completed scPoli reference',flush=True)
else:
    scp = scPoli(adata, condition_keys=['condition_study'],
                cell_type_keys=['malignant_label'], hidden_layer_sizes=[256,128],
                latent_dim=32, embedding_dims=8, recon_loss='nb', use_ln=True)
    scp.train(n_epochs=80, pretraining_epochs=60, prototype_training=False,
              unlabeled_prototype_training=False, early_stopping_kwargs={'early_stopping_metric':'val_loss'})
    scp.save(str(ROOT / 'scpoli_reference'), overwrite=True, save_anndata=False)
    z = np.asarray(scp.get_latent(adata, mean=True), dtype=np.float32)
    np.save(ROOT / 'scpoli_latent.npy', z)

def safe_silhouette(labels, x, max_n=12000):
    labels = np.asarray(labels).astype(str)
    ok = pd.Series(labels).value_counts()
    valid = np.isin(labels, ok[ok >= 2].index)
    ids = np.flatnonzero(valid)
    if len(ids) > max_n: ids = np.random.default_rng(SEED).choice(ids, max_n, replace=False)
    return float(silhouette_score(x[ids], labels[ids], metric='euclidean'))

metrics = {'scpoli': {
    'latent_dim': int(z.shape[1]),
    'site_silhouette_cell': safe_silhouette(adata.obs.tissue_site, z),
    'study_silhouette_cell': safe_silhouette(adata.obs.GEO, z),
}}

specs = {
 'General': ('all metastases', lambda o: o.disease_stage.eq('Metastasis'),
             lambda o: ~o.disease_stage.eq('Metastasis')),
 'Brain': ('brain vs primary and non-brain', lambda o: o.tissue_site.eq('Brain'),
           lambda o: ~o.tissue_site.eq('Brain')),
 'Liver': ('liver vs primary and non-liver', lambda o: o.tissue_site.eq('Liver'),
           lambda o: ~o.tissue_site.eq('Liver')),
 'Bone': ('bone vs primary and non-bone', lambda o: o.tissue_site.eq('Bone'),
          lambda o: ~o.tissue_site.eq('Bone')),
 'Lymph_node': ('LN vs primary and distant sites', lambda o: o.tissue_site.eq('Lymph_node'),
                lambda o: ~o.tissue_site.eq('Lymph_node')),
}

latent_parts = {'scpoli': z}
for name, (description, target_fn, background_fn) in specs.items():
    if (ROOT/f'contrastivevi_{name}_salient.npy').exists() and (ROOT/f'contrastivevi_{name}_background.npy').exists():
        print(f'Reusing completed contrastiveVI {name}', flush=True)
        s=np.load(ROOT/f'contrastivevi_{name}_salient.npy')
        metrics[f'contrastivevi_{name}']={'description':description,'reused':True}
        continue
    print(f'Training contrastiveVI {name}', flush=True)
    target = np.flatnonzero(target_fn(adata.obs).to_numpy())
    background = np.flatnonzero(background_fn(adata.obs).to_numpy())
    ContrastiveVI.setup_anndata(adata, layer='counts', batch_key='GEO')
    model = ContrastiveVI(adata, n_hidden=128, n_background_latent=16,
                          n_salient_latent=16, n_layers=2, dropout_rate=.15,
                          wasserstein_penalty=.1)
    model.train(background_indices=background.tolist(), target_indices=target.tolist(),
                max_epochs=40, accelerator='gpu', devices=1, batch_size=256,
                early_stopping=True, check_val_every_n_epoch=2,
                enable_progress_bar=True)
    model.save(str(ROOT / f'contrastivevi_{name}'), overwrite=True, save_anndata=False)
    s = np.asarray(model.get_latent_representation(adata, representation_kind='salient'), dtype=np.float32)
    b = np.asarray(model.get_latent_representation(adata, representation_kind='background'), dtype=np.float32)
    np.save(ROOT / f'contrastivevi_{name}_salient.npy', s)
    np.save(ROOT / f'contrastivevi_{name}_background.npy', b)
    latent_parts[f'{name}_salient'] = s
    metrics[f'contrastivevi_{name}'] = {
        'description': description, 'n_target': int(len(target)), 'n_background': int(len(background)),
        'target_silhouette_cell': safe_silhouette(target_fn(adata.obs).astype(str), s),
        'study_silhouette_cell': safe_silhouette(adata.obs.GEO, s),
    }
    (ROOT/'training_metrics.partial.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')

(ROOT/'training_metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
print(json.dumps(metrics, indent=2), flush=True)


