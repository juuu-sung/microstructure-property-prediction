"""Eight exact pixel views, archived encoders and matched Ridge selection."""
from __future__ import annotations
import numpy as np
if __package__:
    from . import cradiov4_engine as radio
else:
    import cradiov4_engine as radio

ridge = radio.ridge
NAMES = ('large', 'radio')
WIDTHS = {'large': 3072, 'radio': 4608}
INDICES = (0, 1, 2)  # 10000,1000,100; exact ties prefer stronger regularization.
PRIMARY = 'both_d4_blend'
BLENDS = ('large_d4_blend', 'radio_d4_blend', PRIMARY)


def views(raw):
    import torch
    if raw.ndim != 4 or raw.shape[-1] != raw.shape[-2]:
        raise ValueError('D4 requires square NCHW images')
    for flip in (False, True):
        x = torch.flip(raw, dims=(-1,)) if flip else raw
        for k in range(4):
            yield torch.rot90(x, k, dims=(-2, -1)).contiguous()


def forward(name, model, raw):
    import torch
    with torch.inference_mode():
        if name == 'large':
            with torch.autocast(raw.device.type, enabled=False):
                return ridge.pool_tokens(model.forward_features(ridge.transform_image(raw, 518)), 518, 1024)
        if name == 'radio':
            with torch.autocast('cuda', dtype=torch.bfloat16):
                output = model(radio.transform(raw, 592))
            return radio.pool(output, 592)
    raise ValueError('Unknown encoder')


def extract_views(name, model, raw, forward_fn=None):
    """Pool each view independently, FP64 mean then FP32, one row per image."""
    run = forward_fn or forward
    identity = None
    total = None
    for k, view in enumerate(views(raw)):
        x = run(name, model, view).detach().float().cpu().numpy()
        if x.ndim != 2 or not np.isfinite(x).all():
            raise ValueError('Invalid view features')
        if k == 0:
            identity = x.copy()
            total = x.astype(np.float64)
        else:
            if x.shape != identity.shape:
                raise ValueError('View width changed')
            total += x
    return identity, (total / 8).astype(np.float32)


def anchor(name, actual, expected):
    if actual.shape != expected.shape or not np.isfinite(actual).all():
        raise ValueError('Identity anchor shape/nonfinite mismatch')
    tol = 1e-4 if name == 'large' else 1e-5
    np.testing.assert_allclose(actual, expected, rtol=tol, atol=tol,
                               err_msg=f'{name} original-view anchor failed; cache not accepted')
    return float(np.max(np.abs(actual.astype(float) - expected.astype(float))))


def choose(name, inner, y, scattering):
    y = ridge.finite(y)
    if name not in NAMES or set(inner) != {str(i) for i in INDICES}:
        raise ValueError('Invalid representation/candidate set')
    s = ridge.finite(scattering, y.shape)
    rows = []
    for i in INDICES:
        p = ridge.finite(inner[str(i)], y.shape)
        standalone = float(np.mean((p-y)**2))
        blend = float(np.mean((.5*s+.5*p-y)**2))
        rows.append({'index': i, 'alpha': ridge.CANDIDATES[i]['alpha'],
                     'standalone_mse': standalone, 'scattering_equal_mse': blend,
                     'selection_mse': standalone if name == 'large' else blend})
    best = min(rows, key=lambda v: (v['selection_mse'], v['index']))
    return {'selected': best, 'scores': rows,
            'objective': 'standalone' if name == 'large' else '.5S_inner+.5C_inner'}


def compose(scattering, large, radio_raw, large_d4, radio_d4):
    s = ridge.finite(scattering)
    l, c, ld, cd = [ridge.finite(x, s.shape) for x in (large, radio_raw, large_d4, radio_d4)]
    return {'current': .5*s+.25*l+.25*c, 'large_raw': l, 'radio_raw': c,
            'large_d4': ld, 'radio_d4': cd,
            'large_d4_blend': .5*s+.25*ld+.25*c,
            'radio_d4_blend': .5*s+.25*l+.25*cd,
            PRIMARY: .5*s+.25*ld+.25*cd}
