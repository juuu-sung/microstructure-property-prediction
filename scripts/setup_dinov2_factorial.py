"""User-run pinned Small/Large download. No competition data or model forward."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / 'models/dinov2_factorial'
MODELS = {
    'small': {'architecture': 'vit_small_patch14_dinov2.lvd142m',
              'repo': 'timm/vit_small_patch14_dinov2.lvd142m',
              'revision': '4610ca143709d58a633b6397a74412c2c3842454',
              'sha256': '04d27f3400d059fc0cfd7d17dd1909a75bf3ea8fb3eeb48b97cb99e57ee20081',
              'bytes': 88240510, 'dim': 384, 'depth': 12, 'revision_date': '2025-01-21'},
    'large': {'architecture': 'vit_large_patch14_dinov2.lvd142m',
              'repo': 'timm/vit_large_patch14_dinov2.lvd142m',
              'revision': '4741e1cafbf45415e77074bb0cb42dba76c8684a',
              'sha256': '0424a5d1b515278cba3c6640ccbeaacc41de59d3a93df0dd5e494285eea2b355',
              'bytes': 1217502758, 'dim': 1024, 'depth': 24, 'revision_date': '2025-01-20'},
}
FILES = ('model.safetensors', 'config.json', 'README.md')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()


def validate_model_files(name):
    spec = MODELS[name]; directory = ASSETS/name
    path = directory/'model.safetensors'
    if not path.is_file(): raise FileNotFoundError('Run setup_dinov2_factorial.py first: '+str(path))
    if path.stat().st_size != spec['bytes'] or sha256(path) != spec['sha256']:
        raise ValueError('Checkpoint SHA256/size mismatch: '+name)
    config = json.loads((directory/'config.json').read_text(encoding='utf-8'))
    if config['architecture'] != spec['architecture'].split('.')[0] or config['num_classes'] != 0:
        raise ValueError('Unexpected checkpoint architecture/classifier')
    card = (directory/'README.md').read_text(encoding='utf-8')
    if 'license: apache-2.0' not in card:
        raise ValueError('Pinned model card must declare Apache-2.0')
    return {filename: sha256(directory/filename) for filename in FILES}


def validate():
    path = ASSETS/'assets.json'
    if not path.exists(): raise FileNotFoundError('Run scripts/setup_dinov2_factorial.py first.')
    manifest = json.loads(path.read_text(encoding='utf-8'))
    if manifest['models'] != MODELS: raise ValueError('Asset specification changed')
    for name in MODELS:
        if validate_model_files(name) != manifest['files'][name]: raise ValueError('Asset files changed: '+name)
    return manifest


def setup():
    if (ASSETS/'assets.json').exists():
        validate(); print('Pinned DINOv2 Small/Large assets verified; no downloads.'); return
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import LocalEntryNotFoundError
    for name, spec in MODELS.items():
        for filename in FILES:
            target = ASSETS/name/filename
            if target.exists(): continue
            print(f'Preparing {name}/{filename} @ {spec["revision"]}', flush=True)
            try:
                source = hf_hub_download(spec['repo'], filename, revision=spec['revision'], local_files_only=True)
            except LocalEntryNotFoundError:
                source = hf_hub_download(spec['repo'], filename, revision=spec['revision'])
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_name(target.name+'.partial')
            shutil.copyfile(source, tmp)
            if filename == 'model.safetensors' and sha256(tmp) != spec['sha256']:
                raise ValueError('Downloaded checkpoint failed SHA256; not activated')
            os.replace(tmp, target)
        validate_model_files(name)
    result = {'models': MODELS, 'files': {n: validate_model_files(n) for n in MODELS},
              'license': 'Apache-2.0 per pinned model cards. config.json may retain legacy cc-by-nc-4.0.',
              'source': 'https://github.com/facebookresearch/dinov2',
              'scope': 'Public pretrained assets only. No external dataset or competition data.'}
    tmp = ASSETS/'assets.json.tmp'
    tmp.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    os.replace(tmp, ASSETS/'assets.json')
    validate(); print('Small/Large assets ready. No model forward or training executed.')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--validate-only', action='store_true')
    a = p.parse_args()
    if a.validate_only: validate(); print('Assets verified offline.')
    else: setup()
