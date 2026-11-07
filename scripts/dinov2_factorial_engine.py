"""Pure train/query regression and frozen vision primitives for the 2x2 contrast."""
from __future__ import annotations
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

REPS = ('small266', 'small518', 'large266', 'large518')
ALPHAS = (100., 1000., 10000.)
CS = (1., 10., 100.)
GAMMAS = ('scale', 0.1/64)
PCA_DIM = 64
# Exact ties prefer Ridge, more regularization, then smaller/native representation.
CANDIDATES = tuple([{'family': 'ridge', 'alpha': a} for a in reversed(ALPHAS)] +
                   [{'family': 'svr', 'C': c, 'gamma': g, 'epsilon': .1} for c in CS for g in GAMMAS])


def finite(x, shape=None):
    a = np.asarray(x, dtype=np.float64)
    if (shape is not None and a.shape != shape) or not np.isfinite(a).all():
        raise ValueError('Invalid numeric shape or nonfinite values')
    return a


def inner_splits(y, seed):
    y = finite(y)
    if y.ndim != 1 or len(y) < 90: raise ValueError('At least90 context labels required')
    strata = np.argsort(np.argsort(y, kind='stable'), kind='stable')*10//len(y)
    return list(StratifiedKFold(3, shuffle=True, random_state=seed).split(y, strata))


def fit_candidates(train_x, train_y, query_x, indices=None):
    """Query labels are never accepted. Fit each scaler/PCA on this context only.

    Share one PCA across six SVR choices; this is mathematically the same as
    refitting the identical deterministic PCA pipeline for each choice.
    """
    x = finite(train_x); q = finite(query_x); y = finite(train_y, (len(x),))
    if x.ndim != 2 or q.ndim != 2 or q.shape[1] != x.shape[1]: raise ValueError('Feature width mismatch')
    indices = list(range(len(CANDIDATES))) if indices is None else list(indices)
    if not indices or len(set(indices)) != len(indices) or any(i not in range(len(CANDIDATES)) for i in indices):
        raise ValueError('Invalid candidate indices')
    sx = StandardScaler().fit(x); sy = StandardScaler().fit(y[:, None])
    z, zq = sx.transform(x), sx.transform(q)
    zy = sy.transform(y[:, None]).ravel()
    result = {}; projection = None
    if any(CANDIDATES[i]['family']=='svr' for i in indices):
        if min(z.shape) < PCA_DIM: raise ValueError('PCA64 is fixed; no implicit dimension reduction')
        projection = PCA(PCA_DIM, whiten=False, svd_solver='full').fit(z)
        pc, pq = projection.transform(z), projection.transform(zq)
    for i in indices:
        candidate = CANDIDATES[i]
        if candidate['family']=='ridge':
            model = Ridge(alpha=candidate['alpha']).fit(z, zy); pred = model.predict(zq)
        else:
            model = SVR(C=candidate['C'], gamma=candidate['gamma'], epsilon=candidate['epsilon'],
                        kernel='rbf', cache_size=256).fit(pc, zy)
            pred = model.predict(pq)
        result[str(i)] = finite(sy.inverse_transform(pred[:, None]).ravel(), (len(q),)).tolist()
    detail = {'train_rows': len(x), 'query_rows': len(q), 'feature_width': x.shape[1],
              'candidate_indices': indices, 'pca_components': PCA_DIM if projection is not None else None,
              'pca_explained_variance': float(projection.explained_variance_ratio_.sum()) if projection is not None else None,
              'x_mean': sx.mean_.tolist(), 'y_mean': float(sy.mean_[0]), 'y_scale': float(sy.scale_[0]),
              'query_labels_accepted': False}
    return {'predictions': result, 'audit': detail}


def select(inner_predictions, y):
    """Fixed representation/family controls and joint choice using pooled inner MSE."""
    y = finite(y)
    if tuple(inner_predictions) != REPS: raise ValueError('All four representations required in fixed order')
    scores = {(rep, i): float(np.mean((finite(inner_predictions[rep][str(i)], y.shape)-y)**2))
              for rep in REPS for i in range(len(CANDIDATES))}
    def best(keys):
        rep, i = min(keys, key=lambda k: (scores[k], k[1], REPS.index(k[0])))
        return {'representation': rep, 'candidate_index': i, 'candidate': CANDIDATES[i],
                'inner_rmse': float(np.sqrt(scores[rep, i]))}
    choices = {}
    for rep in REPS:
        for family in ('ridge', 'svr'):
            choices[rep+'_'+family] = best([(rep, i) for i,c in enumerate(CANDIDATES) if c['family']==family])
    for family in ('ridge', 'svr'):
        choices[family+'_joint'] = best([k for k in scores if CANDIDATES[k[1]]['family']==family])
    choices['joint_nested'] = best(list(scores))
    return choices, [{'representation': r, 'candidate_index': i, **CANDIDATES[i], 'inner_mse': s}
                     for (r,i),s in scores.items()]


def transform_image(raw, size):
    """Float32 grayscale [0,1]; same original pixels at266, bicubic full image at518."""
    import torch
    if raw.ndim!=4 or tuple(raw.shape[1:])!=(1,256,256) or raw.dtype!=torch.float32:
        raise ValueError('Expected float32 N1x256x256')
    if size==266:
        rgb = raw.repeat(1,3,1,1)
    elif size==518:
        # Explicit interpolation of continuous intensities; no clipping or contrast normalization.
        rgb = torch.nn.functional.interpolate(raw, size=(518,518), mode='bicubic',
                                              align_corners=False, antialias=True).repeat(1,3,1,1)
    else: raise ValueError('Only266/518 supported')
    mean = rgb.new_tensor([.485,.456,.406])[None,:,None,None]
    std = rgb.new_tensor([.229,.224,.225])[None,:,None,None]
    rgb = (rgb-mean)/std
    return torch.nn.functional.pad(rgb,(5,5,5,5),mode='reflect') if size==266 else rgb


def pool_tokens(tokens, size, dim):
    import torch
    if tokens.ndim!=3 or tuple(tokens.shape[1:])!=(1+(size//14)**2,dim) or tokens.dtype!=torch.float32:
        raise ValueError('Unexpected token schema/precision; register variants are excluded')
    patches = tokens[:,1:]
    x = torch.cat((tokens[:,0],patches.mean(1),patches.std(1,correction=0)),1)
    if not torch.isfinite(x).all(): raise ValueError('Nonfinite pooled features')
    return x


def architecture(name, meta=False):
    import timm
    if __package__: from .setup_dinov2_factorial import MODELS
    else: from setup_dinov2_factorial import MODELS
    model = timm.create_model(MODELS[name]['architecture'],pretrained=False,num_classes=0,
        img_size=518,dynamic_img_size=True,dynamic_img_pad=False,weight_init='skip',
        **({'device':'meta'} if meta else {}))
    spec = MODELS[name]
    if (model.embed_dim != spec['dim'] or len(model.blocks)!=spec['depth'] or model.num_prefix_tokens!=1
        or model.num_reg_tokens!=0 or tuple(model.patch_embed.patch_size)!=(14,14)
        or tuple(model.pos_embed.shape)!=(1,1370,spec['dim'])):
        raise ValueError('Unexpected DINOv2 architecture')
    return model


def load_model(name):
    from safetensors.torch import load_file
    if __package__: from . import setup_dinov2_factorial as assets
    else: import setup_dinov2_factorial as assets
    assets.validate_model_files(name)
    model = architecture(name,meta=True)
    model.load_state_dict(load_file(str(assets.ASSETS/name/'model.safetensors')),strict=True,assign=True)
    if any(p.is_meta for p in model.parameters()): raise ValueError('Unmaterialized model')
    model.requires_grad_(False); model.eval()
    return model
