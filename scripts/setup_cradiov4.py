"""User-run download of one pinned public model; no training or inference."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT/'models/cradiov4_so400m'
LOCK = Path(__file__).with_name('cradiov4_assets.lock.json')


def spec():
    return json.loads(LOCK.read_text(encoding='utf-8'))


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def verify(path, expected):
    if not path.is_file() or path.stat().st_size != expected['bytes'] or sha256(path) != expected['sha256']:
        raise ValueError('Pinned asset missing/changed: '+str(path))


def validate(require_weights=True):
    manifest = spec()
    for name, expected in manifest['files'].items():
        if name == 'model.safetensors' and not require_weights:
            continue
        verify(ASSETS/name, expected)
    cfg = json.loads((ASSETS/'config.json').read_text(encoding='utf-8'))
    if (cfg['version'] != 'c-radio_v4-so400m' or cfg['patch_size'] != 16
            or cfg['vitdet_window_size'] is not None or cfg['adaptor_names'] is not None
            or cfg['feature_normalizer_config'] is not None or cfg['inter_feature_normalizer_config'] is not None
            or cfg['args']['pretrained'] or cfg['args']['initial_checkpoint']):
        raise ValueError('Unsupported configuration; no remote initialization permitted')
    processor = json.loads((ASSETS/'preprocessor_config.json').read_text(encoding='utf-8'))
    if processor['do_normalize'] or processor['do_center_crop']:
        raise ValueError('Unexpected preprocessing')
    return manifest


def setup():
    from huggingface_hub import hf_hub_download
    import requests
    manifest = spec()
    ASSETS.mkdir(parents=True, exist_ok=True)
    for name, expected in manifest['files'].items():
        path = ASSETS/name
        if path.exists():
            verify(path, expected)
            continue
        tmp = path.with_name(path.name+'.partial')
        print('Downloading pinned asset '+name, flush=True)
        if name == 'LICENSE.pdf':
            r = requests.get(manifest['license_url'], timeout=60)
            r.raise_for_status()
            tmp.write_bytes(r.content)
        else:
            source = hf_hub_download(manifest['repo'], name, revision=manifest['revision'])
            shutil.copyfile(source, tmp)
        verify(tmp, expected)
        os.replace(tmp, path)
    (ASSETS/'NOTICE.txt').write_text('Licensed by NVIDIA Corporation under the NVIDIA Open Model License\n'
        'Source: https://huggingface.co/'+manifest['repo']+'\nRevision: '+manifest['revision']+'\n', encoding='utf-8')
    validate()
    print('Pinned C-RADIOv4 assets verified. No model forward or competition fitting.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    if args.validate_only:
        validate()
        print('Assets verified offline.')
    else:
        setup()
