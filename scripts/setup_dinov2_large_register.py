"""Pinned register-Large assets. No installation, training or model forward."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT/'models/dinov2_large_register'
LOCK = Path(__file__).with_name('dinov2_large_register_assets.lock.json')


def spec(): return json.loads(LOCK.read_text(encoding='utf-8'))


def verify(path, expected):
    if not path.is_file() or path.stat().st_size != expected['bytes']: raise ValueError('Pinned asset missing/size mismatch: '+str(path))
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024), b''): h.update(chunk)
    if h.hexdigest() != expected['sha256']: raise ValueError('Pinned asset hash mismatch: '+str(path))


def validate(require_weights=True):
    m = spec()
    for name, expected in m['files'].items():
        if name == 'model.safetensors' and not require_weights: continue
        verify(ASSETS/name, expected)
    cfg = json.loads((ASSETS/'config.json').read_text(encoding='utf-8'))
    pc = cfg['pretrained_cfg']
    if (cfg['architecture'] != m['architecture'] or cfg['num_classes'] != 0 or cfg['num_features'] != 1024
            or pc['input_size'] != [3, 518, 518] or pc['license'] != 'apache-2.0'
            or pc['mean'] != [.485, .456, .406] or pc['std'] != [.229, .224, .225]):
        raise ValueError('Unexpected register checkpoint configuration')
    if 'license: apache-2.0' not in (ASSETS/'README.md').read_text(encoding='utf-8'): raise ValueError('Unexpected model license')
    return m


def setup():
    from huggingface_hub import hf_hub_download
    m = spec(); ASSETS.mkdir(parents=True, exist_ok=True)
    for name, expected in m['files'].items():
        dest = ASSETS/name
        if dest.exists(): verify(dest, expected); continue
        print(f'Preparing register-Large/{name} @ {m["revision"]}', flush=True)
        source = hf_hub_download(m['repo'], name, revision=m['revision'])
        tmp = dest.with_name(dest.name+'.partial'); shutil.copyfile(source, tmp)
        verify(tmp, expected); os.replace(tmp, dest)
    validate(); print('Register-Large assets verified. No model forward or training.')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--validate-only', action='store_true'); a = p.parse_args()
    if a.validate_only: validate(); print('Register-Large weights verified offline.')
    else: setup()
