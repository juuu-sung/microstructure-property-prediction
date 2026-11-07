"""Pinned DRUNet-gray inference and target-blind pilot diagnostics.

Official DPIR UNetRes, MIT (scripts/vendor/drunet/LICENSE). No learning here.
Pixel values/conditioning are [0,1]; diagnostic noise proxies are in 0..255 units.
"""
from pathlib import Path
import hashlib
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter, sobel

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT/'models/drunet_gray/drunet_gray.pth'
WEIGHT_SHA = 'e27fb29456732c604c8ee3ac3e92ecadd2a7a4e36a8be675e92ebbf93a240de4'
URL = 'https://github.com/cszn/KAIR/releases/download/v1.0/drunet_gray.pth'
RECIPE = {
    'model': 'Official DRUNet-gray UNetRes,2 input channels,1 output,nb4,nc64/128/256/512',
    'weights_sha256': WEIGHT_SHA,
    'noise_proxy': 'Per-image Haar HH MAD/0.67448975; fine texture may contaminate it.',
    'condition': 'min(0.5 * per-image noise_proxy_in_255_units, 10.0) / 255.0',
    'strength_multiplier': .5, 'condition_cap_255': 10.,
    'input': 'Original native256 gray float32/255 + spatially constant condition channel.',
    'output': 'Direct UNetRes output (not predicted noise); clamp[0,1], keep float32; no uint8 requantization.',
    'inference': 'Frozen eval,FP32,no AMP/TF32,batch1; no tiling,x8,rotation,contrast correction or strength search.',
    'pilot': '16 IDs: sorted original noise proxy,4 equal rank bins; positions10/35/65/90% within each bin. No y or residuals.',
    'limits': 'Change diagnostics and visual review are not segmentation ground truth or proof of denoising accuracy.'}


def validate_assets():
    if not WEIGHTS.exists():
        raise FileNotFoundError(f'Official weights missing: {WEIGHTS}; source {URL}')
    if hashlib.sha256(WEIGHTS.read_bytes()).hexdigest() != WEIGHT_SHA:
        raise ValueError('DRUNet weights checksum differs')


def load_model(device):
    import torch
    if __package__:
        from .vendor.drunet.network_unet import UNetRes
    else:
        from vendor.drunet.network_unet import UNetRes
    validate_assets()
    model = UNetRes(in_nc=2, out_nc=1, nc=[64,128,256,512], nb=4,
                   act_mode='R', downsample_mode='strideconv', upsample_mode='convtranspose')
    model.load_state_dict(torch.load(WEIGHTS, map_location='cpu', weights_only=True), strict=True)
    return model.requires_grad_(False).eval().to(device)


def read_gray(ident):
    # Train only. Callers must supply a frozen training ID, never an arbitrary path.
    if not str(ident).startswith('TRAIN_') or not str(ident)[6:].isdigit():
        raise ValueError('Expected a training image ID')
    with Image.open(ROOT/'open/train'/f'{ident}.png') as im:
        if im.mode != 'L' or im.size != (256,256):
            raise ValueError('Expected native256 grayscale input')
        return np.array(im, dtype=np.uint8)


def noise_proxy(raw255):
    x = np.asarray(raw255, dtype=np.float64)
    if x.ndim != 2 or any(s % 2 for s in x.shape) or not np.isfinite(x).all():
        raise ValueError('Expected finite even-sized grayscale image')
    hh = (x[::2,::2]-x[::2,1::2]-x[1::2,::2]+x[1::2,1::2])/2
    return float(np.median(np.abs(hh-np.median(hh)))/.67448975)


def condition_sigma(raw):
    return min(RECIPE['strength_multiplier']*noise_proxy(raw), RECIPE['condition_cap_255'])/255.


def select_pilot(ids, noise):
    ids, noise = np.asarray(ids), np.asarray(noise, dtype=np.float64)
    if len(ids) != len(noise) or len(ids) < 64 or len(set(ids)) != len(ids) or not np.isfinite(noise).all():
        raise ValueError('Invalid pilot population')
    order = np.lexsort((ids, noise))
    return np.array([group[int(np.floor(q*(len(group)-1)))]
                     for group in np.array_split(order,4) for q in (.1,.35,.65,.9)], dtype=int)


def restore(model, raw, device):
    import torch
    if raw.dtype != np.uint8 or raw.shape != (256,256):
        raise ValueError('Expected uint8 native256 image')
    x = torch.from_numpy(raw.astype(np.float32)/np.float32(255))[None,None].to(device)
    sigma = condition_sigma(raw)
    model_input = torch.cat((x, torch.full_like(x, sigma)), dim=1)
    with torch.inference_mode():
        output = model(model_input)
    if output.shape != x.shape or output.dtype != torch.float32 or not torch.isfinite(output).all():
        raise ValueError('DRUNet output schema changed')
    clipped = float(((output < 0) | (output > 1)).float().mean().item())
    return output.clamp(0,1)[0,0].cpu().numpy(), {'condition_sigma_255': sigma*255, 'clipped_fraction': clipped}


def diagnostics(raw, restored):
    x, y = raw.astype(np.float64), restored.astype(np.float64)*255
    gx, gy = [], []
    for image, target in ((x,gx),(y,gy)):
        smooth = gaussian_filter(image,1.)
        target.extend([sobel(smooth,axis=k)/8 for k in (0,1)])
    mag0 = np.hypot(*gx); mag1 = np.hypot(*gy)
    weak = (mag0 >= np.quantile(mag0,.5)) & (mag0 <= np.quantile(mag0,.8)) & (mag0 > 1e-6)
    # Contrast around local dark structures, without declaring them pores/grains.
    dark0 = gaussian_filter(x,3.)-gaussian_filter(x,1.)
    dark1 = gaussian_filter(y,3.)-gaussian_filter(y,1.)
    dark = (dark0 >= np.quantile(dark0,.95)) & (dark0 > 1e-6)
    return {'noise_proxy_before': noise_proxy(x), 'noise_proxy_after': noise_proxy(y),
        'change_rmse_255': float(np.sqrt(np.mean((x-y)**2))),
        'mean_shift_255': float((y-x).mean()), 'max_abs_change_255': float(np.max(np.abs(x-y))),
        'weak_gradient_median_ratio': float(np.median(mag1[weak]/mag0[weak])) if weak.any() else None,
        'dark_contrast_median_ratio': float(np.median(dark1[dark]/dark0[dark])) if dark.any() else None}


def preview(raw, restored, title, path):
    """Scientific pixel comparison: fixed scale; no contrast stretching."""
    path.parent.mkdir(parents=True, exist_ok=True)
    y = restored*255
    images = [raw, np.clip(np.rint(y),0,255).astype(np.uint8),
              np.clip(np.rint(128+8*(y-raw)),0,255).astype(np.uint8)]
    canvas = Image.new('RGB',(1560,578),'white'); draw = ImageDraw.Draw(canvas)
    draw.text((8,8), title, fill='black')
    for k,(a,label) in enumerate(zip(images,('Original','DRUNet weak','Signed change x8 (gray=0)'))):
        draw.text((8+k*520,32),label,fill='black')
        canvas.paste(Image.fromarray(a).resize((512,512),Image.Resampling.NEAREST),(8+k*520,54))
    canvas.save(path)


def overview(raws, restored, ids, sigmas, path):
    canvas = Image.new('RGB',(786,len(ids)*286),'white'); draw=ImageDraw.Draw(canvas)
    for i,(x,y,ident,sigma) in enumerate(zip(raws,restored,ids,sigmas)):
        draw.text((4,i*286+4), f'{ident} | sigma={sigma:.2f} | original / restored / change x8',fill='black')
        arrays=(x,np.clip(np.rint(y*255),0,255).astype(np.uint8),
                np.clip(np.rint(128+8*(y*255-x)),0,255).astype(np.uint8))
        for k,a in enumerate(arrays):canvas.paste(Image.fromarray(a),(4+k*262,i*286+24))
    canvas.save(path)
