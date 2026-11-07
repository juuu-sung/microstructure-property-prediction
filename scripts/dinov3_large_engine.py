"""DINOv3-L/16 frozen features and fixed 50:25:25 branch replacement."""
from __future__ import annotations
import numpy as np

if __package__:
    from . import setup_dinov3_large as assets
    from . import cradiov4_engine as previous
else:
    import setup_dinov3_large as assets
    import cradiov4_engine as previous

ridge = previous.ridge
INDICES = previous.RIDGE_INDICES
SIZE, DIM, WIDTH, PREFIX = 592, 1024, 3072, 5


def architecture(device='cpu'):
    import timm
    model = timm.create_model(assets.spec()['model'], pretrained=False, num_classes=0,
                             img_size=SIZE, device=device)
    if (model.num_prefix_tokens != PREFIX or model.num_features != DIM
            or model.patch_embed.patch_size != (16,16)
            or sum(p.numel() for p in model.parameters()) != 303079424):
        raise ValueError('Unexpected DINOv3 architecture')
    return model


def load_model():
    from safetensors.torch import load_file
    assets.validate()
    # Construct on CPU: RoPE has a nonpersistent periods buffer, which must NOT
    # remain uninitialized after a meta/to_empty load. Strict loading of all keys.
    model = architecture()
    state = load_file(str(assets.ASSETS/'model.safetensors'))
    model.load_state_dict(state, strict=True, assign=True)
    if any(t.is_meta for t in list(model.parameters())+list(model.buffers())):
        raise ValueError('Unmaterialized model tensors')
    return model.requires_grad_(False).eval()


def transform(raw):
    import torch
    if (raw.dtype != torch.float32 or raw.ndim != 4 or tuple(raw.shape[1:]) != (1,256,256)
            or not torch.isfinite(raw).all() or raw.min() < 0 or raw.max() > 1):
        raise ValueError('Expected float32 native grayscale [0,1]')
    # Identical policy to archived DINOv2: continuous bicubic, no clamping,
    # ImageNet mean/std, entire image; only output size differs (patch16 vs14).
    rgb = torch.nn.functional.interpolate(raw, size=(SIZE,SIZE), mode='bicubic',
        align_corners=False, antialias=True).repeat(1,3,1,1)
    return (rgb-rgb.new_tensor([.485,.456,.406])[None,:,None,None])/rgb.new_tensor([.229,.224,.225])[None,:,None,None]


def pool(tokens):
    import torch
    if tokens.ndim != 3 or tuple(tokens.shape[1:]) != (PREFIX+(SIZE//16)**2,DIM):
        raise ValueError('Expected CLS+4registers+37x37 spatial tokens')
    t = tokens.float()
    patch = t[:,PREFIX:]
    x = torch.cat((t[:,0],patch.mean(1),patch.std(1,correction=0)),dim=1)
    if not torch.isfinite(x).all():
        raise ValueError('Nonfinite pooled features')
    return x


def choose(inner_predictions, y, inner_scattering):
    # Deliberately retain the published C-RADIO experiment's objective for BOTH
    # DINO versions. This does not claim to optimize the final 50:25:25 blend.
    return previous.choose(inner_predictions,y,inner_scattering)


def compose(scattering, radio, old_large, retuned_large, dino3):
    s = ridge.finite(scattering)
    c,l,lc,d = [ridge.finite(v,s.shape) for v in (radio,old_large,retuned_large,dino3)]
    return {'current':.5*s+.25*c+.25*l, 'dinov2_retuned_ridge':lc,
        'dinov3_ridge':d, 'matched_control':.5*s+.25*c+.25*lc,
        'dinov3_replaced_blend':.5*s+.25*c+.25*d}
