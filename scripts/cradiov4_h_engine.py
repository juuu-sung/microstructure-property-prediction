"""Frozen H native encoder and matched D4 image-level pooling, no downloads."""
import json
from types import SimpleNamespace
import numpy as np
if __package__:
    from . import setup_cradiov4_h as assets, large_radio_d4_engine as d4
else:
    import setup_cradiov4_h as assets
    import large_radio_d4_engine as d4

SUMMARY, SPATIAL, WIDTH, SIZE = 2560, 1280, 5120, 592


def architecture(meta=False):
    import torch
    from contextlib import nullcontext
    assets.validate(require_weights=False)
    native, conditioner = d4.radio.native_modules()
    cfg = json.loads((assets.ASSETS/'config.json').read_text(encoding='utf-8'))
    with torch.device('meta') if meta else nullcontext():
        backbone = native.create_model_from_args(SimpleNamespace(**cfg['args']))
        norm = conditioner.get_default_conditioner(); norm.dtype = torch.float32
        summary = torch.tensor([i for i,t in enumerate(cfg['args']['teachers']) if t.get('use_summary', True)], dtype=torch.int64)
        model = native.RADIOModel(backbone, norm, patch_size=16, max_resolution=cfg['max_resolution'],
            preferred_resolution=native.Resolution(*cfg['preferred_resolution']), summary_idxs=summary,
            window_size=None, adaptors={}, feature_normalizer=None, inter_feature_normalizer=None)
    if (model.summary_dim != SUMMARY or model.embed_dim != SPATIAL or model.num_summary_tokens != 10
            or model.patch_size != 16 or sum(p.numel() for p in model.parameters()) != assets.spec()['parameters']):
        raise ValueError('Unexpected H architecture')
    return model


def load_model():
    import torch
    from safetensors.torch import load_file
    assets.validate()
    model = architecture(meta=True)
    state = load_file(str(assets.ASSETS/'model.safetensors'))
    prefix = 'radio_model.'
    if not all(k.startswith(prefix) for k in state): raise ValueError('Unexpected H checkpoint wrapper')
    model.load_state_dict({k[len(prefix):]:v for k,v in state.items()}, strict=True, assign=True)
    if any(p.is_meta or p.dtype != torch.float32 for p in model.parameters()) or model.summary_idxs.tolist() != [0, 1]:
        raise ValueError('Incomplete H materialization/summary indices')
    model.requires_grad_(False)
    return model.eval()


def pool(output):
    import torch
    summary, spatial = output
    if (summary.ndim != 2 or summary.shape[1] != SUMMARY or spatial.shape != (len(summary), (SIZE//16)**2, SPATIAL)):
        raise ValueError('Unexpected H tokens; prefix tokens must not enter spatial pooling')
    summary, spatial = summary.float(), spatial.float()
    x = torch.cat((summary, spatial.mean(1), spatial.std(1, correction=0)), dim=1)
    if not torch.isfinite(x).all(): raise ValueError('Nonfinite H features')
    return x


def forward(name, model, raw):
    import torch
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        output = model(d4.radio.transform(raw, SIZE))
    return pool(output)


def extract_views(model, raw):
    return d4.extract_views('radio', model, raw, forward_fn=forward)


def compose(scattering, large_d4, old_radio_d4, huge_d4):
    s = d4.ridge.finite(scattering)
    l, c, h = [d4.ridge.finite(v, s.shape) for v in (large_d4, old_radio_d4, huge_d4)]
    return {'current': .5*s+.25*l+.25*c, 'so400m_d4_ridge': c, 'h_d4_ridge': h,
            'h_d4_replaced_blend': .5*s+.25*l+.25*h}
