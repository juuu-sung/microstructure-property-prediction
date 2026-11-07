"""Pinned H weights; reuse byte-identical archived official RADIO native code."""
import argparse
import json
import os
from pathlib import Path
import shutil
if __package__:
    from . import setup_cradiov4 as shared
else:
    import setup_cradiov4 as shared

ROOT = shared.ROOT
ASSETS = ROOT/'models/cradiov4_h'
LOCK = Path(__file__).with_name('cradiov4_h_assets.lock.json')


def spec(): return json.loads(LOCK.read_text(encoding='utf-8'))


def validate(require_weights=True):
    manifest = spec()
    for name, expected in manifest['files'].items():
        if name == 'model.safetensors' and not require_weights: continue
        shared.verify(ASSETS/name, expected)
    for name, expected in manifest['shared_native'].items(): shared.verify(shared.ASSETS/name, expected)
    cfg = json.loads((ASSETS/'config.json').read_text(encoding='utf-8'))
    if (cfg['version'] != 'c-radio_v4-h' or cfg['args']['model'] != 'vit_huge_patch16_224'
            or cfg['patch_size'] != 16 or cfg['vitdet_window_size'] is not None or cfg['adaptor_names'] is not None
            or cfg['feature_normalizer_config'] is not None or cfg['inter_feature_normalizer_config'] is not None
            or cfg['args']['pretrained'] or cfg['args']['initial_checkpoint']):
        raise ValueError('Unexpected H architecture/pretrained configuration')
    return manifest


def setup():
    from huggingface_hub import hf_hub_download
    m = spec(); ASSETS.mkdir(parents=True, exist_ok=True)
    for name, expected in m['files'].items():
        dest = ASSETS/name
        if dest.exists(): shared.verify(dest, expected); continue
        print(f'Preparing C-RADIOv4-H/{name} @ {m["revision"]}', flush=True)
        if name == 'LICENSE.pdf':
            source = shared.ASSETS/name; shared.verify(source, expected)
        else: source = Path(hf_hub_download(m['repo'], name, revision=m['revision']))
        tmp = dest.with_name(dest.name+'.partial'); shutil.copyfile(source, tmp)
        shared.verify(tmp, expected); os.replace(tmp, dest)
    validate()
    (ASSETS/'NOTICE.txt').write_text('Licensed by NVIDIA Corporation under the NVIDIA Open Model License\n'
        'Source: https://huggingface.co/'+m['repo']+'\nRevision: '+m['revision']+'\n', encoding='utf-8')
    print('Pinned H assets verified. No model forward or training.')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--validate-only', action='store_true'); a = p.parse_args()
    if a.validate_only: validate(); print('H weights verified offline.')
    else: setup()
