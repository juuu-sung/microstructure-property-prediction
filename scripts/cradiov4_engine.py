"""Pinned native RADIO inference and context-only blend-aware Ridge selection."""
from __future__ import annotations
import importlib
import json
import sys
from types import SimpleNamespace
import warnings
import numpy as np

if __package__:
    from . import setup_cradiov4 as assets
    from . import dinov2_factorial_engine as ridge
else:
    import setup_cradiov4 as assets
    import dinov2_factorial_engine as ridge

RIDGE_INDICES = (0, 1, 2)  # Exact tie: alpha10000,1000,100.
PRIMARY_SIZE = 592
FALLBACK_SIZE = 512
SUMMARY_DIM = 2304
SPATIAL_DIM = 1152
WIDTH = SUMMARY_DIM + 2*SPATIAL_DIM


def architecture(meta=False):
    """Same construction as pinned hf_model.RADIOModel, without its HF wrapper.

    Uses untouched official native modules; no transformers install/remote code
    loader. The only key mapping is removing the wrapper's 'radio_model.' prefix.
    All keys/shapes are loaded strictly, including conditioner and summary indices.
    """
    import torch
    from contextlib import nullcontext
    assets.validate(require_weights=False)
    native, conditioner = native_modules()
    cfg = json.loads((assets.ASSETS/'config.json').read_text(encoding='utf-8'))
    with torch.device('meta') if meta else nullcontext():
        backbone = native.create_model_from_args(SimpleNamespace(**cfg['args']))
        norm = conditioner.get_default_conditioner()
        norm.dtype = torch.float32
        summary = torch.tensor([i for i,t in enumerate(cfg['args']['teachers']) if t.get('use_summary', True)], dtype=torch.int64)
        model = native.RADIOModel(backbone, norm, patch_size=cfg['patch_size'], max_resolution=cfg['max_resolution'],
            preferred_resolution=native.Resolution(*cfg['preferred_resolution']), summary_idxs=summary,
            window_size=None, adaptors={}, feature_normalizer=None, inter_feature_normalizer=None)
    if (model.summary_dim != SUMMARY_DIM or model.embed_dim != SPATIAL_DIM or model.patch_size != 16
            or model.num_summary_tokens != 10 or sum(p.numel() for p in model.parameters()) != 431237232):
        raise ValueError('Unexpected official native architecture')
    return model


def native_modules():
    """Load pinned modules without leaking setuptools' vendored search path.

    The vendor's distutils import initializes setuptools, which appends its
    _vendor directory to sys.path. That makes importlib.metadata enumerate a
    second packaging/platformdirs/etc. distribution and invalidates unrelated
    archived protocols although no installed package changed. Restore the exact
    caller path after import (also on failure); do not weaken protocol checks.
    """
    original_path = list(sys.path)
    try:
        if str(assets.ROOT) not in sys.path:
            sys.path.insert(0, str(assets.ROOT))
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='Overwriting .* in registry.*')
            native = importlib.import_module('models.cradiov4_so400m.radio_model')
            conditioner = importlib.import_module('models.cradiov4_so400m.input_conditioner')
        return native, conditioner
    finally:
        sys.path[:] = original_path


def load_model():
    from safetensors.torch import load_file
    assets.validate()
    model = architecture(meta=True)
    state = load_file(str(assets.ASSETS/'model.safetensors'))
    prefix = 'radio_model.'
    if not all(k.startswith(prefix) for k in state):
        raise ValueError('Unexpected checkpoint wrapper')
    state = {k[len(prefix):]:v for k,v in state.items()}
    model.load_state_dict(state, strict=True, assign=True)
    if any(p.is_meta for p in model.parameters()) or model.summary_idxs.tolist() != [0,1]:
        raise ValueError('Incomplete materialization/summary indices')
    model.requires_grad_(False)
    return model.eval()


def transform(raw, size):
    import torch
    if (raw.dtype != torch.float32 or raw.ndim != 4 or tuple(raw.shape[1:]) != (1,256,256)
            or size not in (PRIMARY_SIZE,FALLBACK_SIZE) or not torch.isfinite(raw).all()
            or raw.min() < 0 or raw.max() > 1):
        raise ValueError('Expected native grayscale float32 [0,1] and locked size')
    # Official RADIO consumes [0,1] and applies its own CLIP conditioner.
    # Bicubic overshoot is clamped, explicitly; no ImageNet normalization/crop.
    return torch.nn.functional.interpolate(raw, size=(size,size), mode='bicubic',
        align_corners=False, antialias=True).clamp(0,1).repeat(1,3,1,1)


def pool(output, size):
    import torch
    summary, spatial = output
    if (summary.ndim != 2 or summary.shape[1] != SUMMARY_DIM or spatial.ndim != 3
            or tuple(spatial.shape) != (len(summary),(size//16)**2,SPATIAL_DIM)):
        raise ValueError('Unexpected summary/spatial schema; registers must not enter pooling')
    summary, spatial = summary.float(), spatial.float()
    features = torch.cat((summary, spatial.mean(1), spatial.std(1, correction=0)), dim=1)
    if not torch.isfinite(features).all():
        raise ValueError('Nonfinite C-RADIO features')
    return features


def choose(inner_predictions, y, scattering_inner):
    """Alpha selection only from within-outer out-of-fold predictions.

    scattering_inner must be generated with the SAME inner query held out from
    every base learner and its alpha selection; never pass global OOF here.
    """
    y = ridge.finite(y)
    s = ridge.finite(scattering_inner, y.shape)
    if set(inner_predictions) != {str(i) for i in RIDGE_INDICES}:
        raise ValueError('Unexpected Ridge candidate set')
    scores = []
    for i in RIDGE_INDICES:
        p = ridge.finite(inner_predictions[str(i)], y.shape)
        scores.append({'index':i,'alpha':ridge.CANDIDATES[i]['alpha'],
            'standalone_mse':float(np.mean((p-y)**2)), 'blend_mse':float(np.mean((.5*s+.5*p-y)**2))})
    best = min(scores, key=lambda r:(r['blend_mse'],r['index']))
    standalone = min(scores, key=lambda r:(r['standalone_mse'],r['index']))
    return {'blend':best,'standalone':standalone,'scores':scores}


def compose(scattering, old_large, radio, retuned_large):
    s = ridge.finite(scattering)
    l,c,lc = (ridge.finite(v,s.shape) for v in (old_large,radio,retuned_large))
    return {'current':.5*s+.5*l, 'radio_ridge':c, 'large_retuned_ridge':lc,
        'large_retuned_blend':.5*s+.5*lc, 'radio_replaced_blend':.5*s+.5*c,
        'large_retuned_complementary':.5*s+.25*l+.25*lc,
        'radio_complementary_blend':.5*s+.25*l+.25*c}
