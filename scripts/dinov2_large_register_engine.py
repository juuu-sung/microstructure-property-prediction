"""Matched FP32 Large-reg4 D4, excluding registers from spatial statistics."""
import numpy as np
if __package__:
    from . import setup_dinov2_large_register as assets, large_radio_d4_engine as d4
else:
    import setup_dinov2_large_register as assets
    import large_radio_d4_engine as d4

WIDTH, SIZE, DIM, PREFIX = 3072, 518, 1024, 5
PRIMARY = 'large_register_equal_blend'
SECONDARY = 'large_register_replaced_blend'
BLENDS = (PRIMARY, SECONDARY)


def architecture(meta=False):
    import torch
    import timm
    from contextlib import nullcontext
    assets.validate(require_weights=False)
    with torch.device('meta') if meta else nullcontext():
        model = timm.create_model(assets.spec()['architecture'], pretrained=False, num_classes=0,
            img_size=SIZE, dynamic_img_size=True, dynamic_img_pad=False, weight_init='skip')
    if (model.embed_dim != DIM or len(model.blocks) != 24 or model.num_prefix_tokens != PREFIX
            or model.num_reg_tokens != 4 or not model.no_embed_class or tuple(model.patch_embed.patch_size) != (14,14)
            or tuple(model.pos_embed.shape) != (1,1369,DIM) or tuple(model.reg_token.shape) != (1,4,DIM)
            or sum(p.numel() for p in model.parameters()) != assets.spec()['parameters']):
        raise ValueError('Unexpected register-Large architecture')
    return model


def load_model():
    import torch
    from safetensors.torch import load_file
    assets.validate()
    model = architecture(meta=True)
    model.load_state_dict(load_file(str(assets.ASSETS/'model.safetensors')), strict=True, assign=True)
    if any(p.is_meta or p.dtype != torch.float32 for p in model.parameters()): raise ValueError('Incomplete/incorrect precision weights')
    model.requires_grad_(False)
    return model.eval()


def pool(tokens):
    import torch
    if tokens.ndim != 3 or tuple(tokens.shape[1:]) != (PREFIX+1369, DIM) or tokens.dtype != torch.float32:
        raise ValueError('Expected CLS+4register+1369patch FP32 tokens')
    # timm order is CLS, four registers, then the spatial patches.
    patches = tokens[:, PREFIX:]
    x = torch.cat((tokens[:,0], patches.mean(1), patches.std(1, correction=0)), 1)
    if not torch.isfinite(x).all(): raise ValueError('Nonfinite register-Large features')
    return x


def forward(name, model, raw):
    import torch
    with torch.inference_mode(), torch.autocast(raw.device.type, enabled=False):
        return pool(model.forward_features(d4.ridge.transform_image(raw, SIZE)))


def extract_views(model, raw): return d4.extract_views('large', model, raw, forward_fn=forward)


def choose(inner, y):
    y = d4.ridge.finite(y)
    if set(inner) != {str(i) for i in d4.INDICES}: raise ValueError('Unexpected alpha candidates')
    rows = [{'index':i, 'alpha':d4.ridge.CANDIDATES[i]['alpha'],
             'selection_mse':float(np.mean((d4.ridge.finite(inner[str(i)], y.shape)-y)**2))} for i in d4.INDICES]
    return {'selected':min(rows, key=lambda r:(r['selection_mse'], r['index'])), 'scores':rows, 'objective':'standalone'}


def compose(scattering, large_d4, radio_d4, register_d4):
    s = d4.ridge.finite(scattering)
    l, c, r = [d4.ridge.finite(x, s.shape) for x in (large_d4, radio_d4, register_d4)]
    return {'current':.5*s+.25*l+.25*c, 'large_d4_ridge':l, 'large_register_d4_ridge':r,
            PRIMARY:.5*s+.25*c+.125*l+.125*r, SECONDARY:.5*s+.25*c+.25*r}
