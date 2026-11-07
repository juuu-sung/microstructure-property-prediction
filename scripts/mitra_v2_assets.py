"""Pinned public Mitra assets; no competition data access."""
from pathlib import Path
import hashlib
import json
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'models/mitra_v2'
LOCK = Path(__file__).with_name('mitra_v2_assets.lock.json')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(2**20), b''):
            h.update(part)
    return h.hexdigest()


def spec():
    return json.loads(LOCK.read_text(encoding='utf-8'))


def validate(download=False):
    s = spec()
    for prefix, base in [('code', BASE/'vendor'), ('weight', BASE)]:
        for name, expected in s[prefix+'_files'].items():
            p = base/name
            if not p.exists() and download:
                p.parent.mkdir(parents=True, exist_ok=True)
                url = f'https://huggingface.co/{s[prefix+"_repo"]}/resolve/{s[prefix+"_revision"]}/{name}'
                temp = p.with_name(p.name+'.part')
                urllib.request.urlretrieve(url, temp)
                if temp.stat().st_size != expected['bytes'] or sha(temp) != expected['sha256']:
                    raise ValueError('Downloaded asset hash mismatch: '+name)
                temp.replace(p)
            if not p.is_file() or p.stat().st_size != expected['bytes'] or sha(p) != expected['sha256']:
                raise ValueError('Pinned asset missing/changed: '+str(p))
    return s


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--download', action='store_true')
    a = p.parse_args()
    validate(a.download)
    print('Pinned Mitra-v2 code/config/weights verified.')
