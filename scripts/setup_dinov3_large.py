"""Download one pinned public timm DINOv3 checkpoint; no inference or fitting."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil

if __package__:
    from .setup_cradiov4 import sha256, verify
else:
    from setup_cradiov4 import sha256, verify

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT/'models/dinov3_large'
LOCK = Path(__file__).with_name('dinov3_large_assets.lock.json')


def spec():
    return json.loads(LOCK.read_text(encoding='utf-8'))


def validate(require_weights=True):
    lock = spec()
    for name, expected in lock['files'].items():
        if name == 'model.safetensors' and not require_weights:
            continue
        verify(ASSETS/name, expected)
    cfg = json.loads((ASSETS/'config.json').read_text(encoding='utf-8'))
    if (cfg['architecture'] != 'vit_large_patch16_dinov3' or cfg['num_classes'] != 0
            or cfg['num_features'] != 1024 or cfg['pretrained_cfg']['mean'] != [.485,.456,.406]
            or cfg['pretrained_cfg']['std'] != [.229,.224,.225]):
        raise ValueError('Unexpected checkpoint/preprocessing configuration')
    return lock


def setup():
    from huggingface_hub import hf_hub_download
    lock = spec()
    ASSETS.mkdir(parents=True, exist_ok=True)
    for name, expected in lock['files'].items():
        target = ASSETS/name
        if target.exists():
            verify(target, expected)
            continue
        print('Downloading pinned '+name, flush=True)
        source = hf_hub_download(lock['repo'], name, revision=lock['revision'])
        temp = target.with_name(target.name+'.partial')
        shutil.copyfile(source, temp)
        verify(temp, expected)
        os.replace(temp, target)
    validate()
    print('DINOv3 assets verified. No competition forwards or training.')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--validate-only', action='store_true')
    a = p.parse_args()
    if a.validate_only:
        validate()
        print('Assets verified offline.')
    else:
        setup()
