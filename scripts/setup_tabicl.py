"""Pin local TabICLv2 assets without installing into the archived Python environment.

Downloads official files only. No competition data, training, or inference.
The package and psutil live under .deps-tabicl, without discoverable dist-info.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import platform
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / 'models/tabicl_v2'
DEPS = ROOT / '.deps-tabicl'
REVISION = 'd700cf089377a49ad3cf4d74e3e151c84dfe28b8'
WEIGHT = 'tabicl-regressor-v2-20260212.ckpt'
WEIGHT_SHA = '0db9cb538f114e79026bf08f45f41ad8dd7ad2de2aaca9a5ca8cd3bd9748ae7a'
PACKAGES = {
    'tabicl': ('2.2.0', 'https://files.pythonhosted.org/packages/77/22/bbac2df399b8ed5a699bde38496ebb129a045c5d160fe52786727de932f3/tabicl-2.2.0-py3-none-any.whl',
               'c80b8ec2719be26bc0bf1dd758a2361f58019d8f7877038b785529a425789ce4'),
    'psutil': ('7.2.2', 'https://files.pythonhosted.org/packages/b4/90/e2159492b5426be0c1fef7acba807a03511f97c5f86b3caeda6ad92351a7/psutil-7.2.2-cp37-abi3-win_amd64.whl',
               'eb7e81434c8d223ec4a219b5fc1c47d0417b12be7ea866e24fb5ad6e84b3d988'),
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def inventory():
    return {d.metadata['Name'].lower(): d.version for d in importlib.metadata.distributions()}


def download(url, path, expected=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if expected and sha(path) != expected:
            raise ValueError(f'Existing asset hash mismatch: {path}')
        return
    tmp = path.with_suffix(path.suffix+'.partial')
    print(f'Downloading {path.name}', flush=True)
    request = urllib.request.Request(url, headers={'User-Agent': 'microstructure-tabicl-setup/1.0'})
    with urllib.request.urlopen(request, timeout=120) as source, tmp.open('wb') as dest:
        for block in iter(lambda: source.read(1024*1024), b''):
            dest.write(block)
    if expected and sha(tmp) != expected:
        raise ValueError(f'Download hash mismatch: {path}')
    os.replace(tmp, path)


def wheel_files(archive, package):
    """Extract only the selected import package and preserve notices separately."""
    files = {}
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            p = PurePosixPath(name)
            if p.is_absolute() or '..' in p.parts or '\\' in name:
                raise ValueError('Unsafe wheel member')
            if name.endswith('/'):
                continue
            if p.parts[0] == package:
                dest = DEPS.joinpath(*p.parts)
            elif p.parts[0].endswith('.dist-info') and ('licenses' in p.parts or p.name in ('LICENSE', 'METADATA')):
                dest = ASSETS / 'notices' / package / Path(*p.parts[1:])
            else:
                continue
            contents = z.read(name)
            if dest.exists() and dest.read_bytes() != contents:
                raise ValueError(f'Existing source differs: {dest}')
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                dest.write_bytes(contents)
            files[str(dest.relative_to(ROOT)).replace('\\', '/')] = sha(dest)
    if not (DEPS/package/'__init__.py').exists():
        raise ValueError('Missing isolated package')
    return files


def validate():
    path = ASSETS/'assets.json'
    if not path.exists():
        raise ValueError('Run scripts/setup_tabicl.py first.')
    manifest = json.loads(path.read_text(encoding='utf8'))
    if manifest['revision'] != REVISION or manifest['checkpoint_sha256'] != WEIGHT_SHA:
        raise ValueError('TabICL asset identity changed')
    for relative, expected in manifest['files'].items():
        if sha(ROOT/relative) != expected:
            raise ValueError(f'TabICL asset changed: {relative}')
    sources = {str(p.relative_to(ROOT)).replace('\\','/') for package in PACKAGES
               for p in (DEPS/package).rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    expected = {n for n in manifest['files'] if n.startswith('.deps-tabicl/')}
    if sources != expected or any(DEPS.glob('*.dist-info')):
        raise ValueError('Unexpected isolated sources or discoverable distribution metadata')
    if manifest['release_evidence']['date'] != '2026-02-12T07:34:31.000Z':
        raise ValueError('Checkpoint release evidence changed')
    return manifest


def setup():
    if (ASSETS/'assets.json').exists():
        validate()
        print('TabICLv2 assets verified offline; environment unchanged.')
        return
    if platform.system() != 'Windows' or platform.machine().lower() not in ('amd64','x86_64'):
        raise ValueError('This isolated psutil wheel is pinned for Windows AMD64.')
    before = inventory()
    files = {}
    for package, (_, url, expected) in PACKAGES.items():
        wheel = ASSETS/'downloads'/url.rsplit('/',1)[-1]
        download(url, wheel, expected)
        files.update(wheel_files(wheel, package))
        files[str(wheel.relative_to(ROOT)).replace('\\','/')] = sha(wheel)
    url = f'https://huggingface.co/jingang/TabICL/resolve/{REVISION}/{WEIGHT}'
    download(url, ASSETS/WEIGHT, WEIGHT_SHA)
    evidence_url = 'https://huggingface.co/api/models/jingang/TabICL/commits/main'
    download(evidence_url, ASSETS/'release_commits.json')
    commits = json.loads((ASSETS/'release_commits.json').read_text(encoding='utf8'))
    release = next(c for c in commits if c['id'] == REVISION)
    if release['date'] != '2026-02-12T07:34:31.000Z':
        raise ValueError('Unexpected checkpoint publication timestamp')
    card_revision = '4dcd344ece2c00be9e831fdd35bed57b5ad83e19'
    download(f'https://huggingface.co/jingang/TabICL/raw/{card_revision}/README.md', ASSETS/'MODEL_CARD.txt')
    if 'license: bsd-3-clause' not in (ASSETS/'MODEL_CARD.txt').read_text(encoding='utf8'):
        raise ValueError('Expected model license missing')
    for name in (WEIGHT,'release_commits.json','MODEL_CARD.txt'):
        files[str((ASSETS/name).relative_to(ROOT)).replace('\\','/')] = sha(ASSETS/name)
    if inventory() != before:
        raise ValueError('Archived Python environment changed')
    manifest = {'revision': REVISION, 'checkpoint_sha256': WEIGHT_SHA,
        'release_evidence': {'url': evidence_url, 'id': release['id'], 'date': release['date'], 'title': release['title']},
        'license': 'BSD-3-Clause; upstream notices retained', 'model_card_revision': card_revision,
        'packages': {n: {'version': v, 'url': u, 'sha256': h} for n,(v,u,h) in PACKAGES.items()},
        'files': files, 'environment_changed': False, 'competition_data_used': False}
    tmp = ASSETS/'assets.json.tmp'
    tmp.write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf8')
    os.replace(tmp, ASSETS/'assets.json')
    validate()
    print('Ready: TabICLv2 2.2.0, pinned February12 checkpoint; no model inference.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    if args.validate_only:
        validate()
        print('Local assets verified.')
    else:
        setup()
