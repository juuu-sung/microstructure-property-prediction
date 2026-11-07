"""Download two pinned public checkpoints and a hash-verified isolated source package.

No pip install, environment modification, competition input, forward pass or fit.
Only this setup command uses the network. Subsequent experiment commands are local.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / 'models/micronet_efficientnet_b0'
DEPS = ROOT / '.deps-micronet'
RELEASE = {
    'url': 'https://files.pythonhosted.org/packages/2e/a0/dd40b50aebf0028054b6b35062948da01123d7be38d08b6b1e5435df6363/efficientnet_pytorch-0.7.1.tar.gz',
    'sha256': '00b9b261effce59d2d47aae2ad238c29a2a65175470f41ada7ecac439b7c1ee1',
}
WEIGHTS = {
    'imagenet': {
        'repo': 'smp-hub/efficientnet-b0.imagenet',
        'revision': '4cb4507c8a1f8d271fab0003f19f07ba366ed318',
        'filename': 'model.safetensors',
        'sha256': '4b276ec2b6949a7bf7748885da6af1f79df812abf2442a75639d9b584d8c69be',
    },
    'micronet': {
        'repo': 'jstuckner/microscopy-efficientnet-b0-imagenet-micronet',
        'revision': '6facf658e6c151786eccba594b06f1e253d06b99',
        'filename': 'efficientnet-b0_imagenet-micronet_weights.pth',
        'sha256': '81b3ae8cb2035ab96aa2da398ff0429cfb50973f112349ca7eda287a68d5bc8b',
    },
}
PACKAGE_FILES = ('__init__.py', 'model.py', 'utils.py')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def download(url, path, expected=None):
    """Never replace an existing final asset with a different download."""
    if path.exists():
        if expected and sha256(path) != expected:
            raise ValueError(f'Existing file has the wrong SHA256; preserve and inspect: {path}')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.partial')
    request = urllib.request.Request(url, headers={'User-Agent': 'microstructure-experiment-setup/1'})
    with urllib.request.urlopen(request, timeout=60) as response, temporary.open('wb') as out:
        for chunk in iter(lambda: response.read(1024 * 1024), b''):
            out.write(chunk)
    if expected and sha256(temporary) != expected:
        raise ValueError(f'Download SHA256 mismatch; not activated: {temporary}')
    os.replace(temporary, path)


def install_package(archive):
    """Copy a whitelist of source files; never execute setup.py or extract paths."""
    if sha256(archive) != RELEASE['sha256']:
        raise ValueError('Source archive changed')
    prefix = 'efficientnet_pytorch-0.7.1/'
    members = {f'efficientnet_pytorch/{n}': DEPS / 'efficientnet_pytorch' / n for n in PACKAGE_FILES}
    with tarfile.open(fileobj=io.BytesIO(archive.read_bytes()), mode='r:gz') as tar:
        for name, target in members.items():
            member = tar.getmember(prefix + name)
            if not member.isfile():
                raise ValueError('Expected a regular package file')
            payload = tar.extractfile(member).read()
            if target.exists() and target.read_bytes() != payload:
                raise ValueError(f'Isolated package already differs: {target}')
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
    return members.values()


def asset_path(name):
    return ASSETS / name / WEIGHTS[name]['filename']


def validate():
    path = ASSETS / 'assets.json'
    if not path.is_file():
        raise ValueError('Run scripts/setup_micronet.py first; model assets are not ready.')
    manifest = json.loads(path.read_text(encoding='utf8'))
    if manifest['weights'] != WEIGHTS or manifest['package_release'] != RELEASE:
        raise ValueError('Pinned asset configuration differs')
    for name in WEIGHTS:
        if sha256(asset_path(name)) != WEIGHTS[name]['sha256']:
            raise ValueError('Checkpoint hash mismatch: ' + name)
    required = {str((DEPS / 'efficientnet_pytorch' / n).relative_to(ROOT)).replace('\\', '/') for n in PACKAGE_FILES}
    required.add('.deps-micronet/LICENSE-EfficientNet-PyTorch.txt')
    if set(manifest['package_files']) != required:
        raise ValueError('Isolated package file set differs')
    for group in ('package_files', 'documentation'):
        for name, expected in manifest[group].items():
            target = (ROOT / name).resolve()
            if not target.is_relative_to(ROOT) or sha256(target) != expected:
                raise ValueError('Asset changed: ' + name)
    return manifest


def setup():
    if (ASSETS / 'assets.json').exists():
        validate()
        print('Pinned checkpoints and isolated package verified; no downloads or fitting.')
        return
    ASSETS.mkdir(parents=True, exist_ok=True)
    archive = ASSETS / 'efficientnet_pytorch-0.7.1.tar.gz'
    download(RELEASE['url'], archive, RELEASE['sha256'])
    installed = list(install_package(archive))
    package_license = DEPS / 'LICENSE-EfficientNet-PyTorch.txt'
    download('https://raw.githubusercontent.com/lukemelas/EfficientNet-PyTorch/master/LICENSE', package_license)
    installed.append(package_license)
    docs = []
    for name, spec in WEIGHTS.items():
        prefix = f"https://huggingface.co/{spec['repo']}/resolve/{spec['revision']}/"
        download(prefix + spec['filename'], asset_path(name), spec['sha256'])
        print(f'Verified {name}: {spec["sha256"]}', flush=True)
        for filename in ('README.md', 'encoder.py' if name == 'micronet' else 'config.json'):
            target = ASSETS / name / filename
            download(prefix + filename, target)
            docs.append(target)
    path = ASSETS / 'LICENSE-NASA-MicroNet.txt'
    download('https://raw.githubusercontent.com/nasa/pretrained-microscopy-models/main/LICENSE.txt', path)
    docs.append(path)
    manifest = {
        'weights': WEIGHTS, 'package_release': RELEASE,
        'package_files': {p.relative_to(ROOT).as_posix(): sha256(p) for p in installed},
        'documentation': {p.relative_to(ROOT).as_posix(): sha256(p) for p in docs},
        'scope': 'Public pretrained weights only; no external training dataset, competition images, forward or fit.',
    }
    tmp = ASSETS / 'assets.json.tmp'
    tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    os.replace(tmp, ASSETS / 'assets.json')
    validate()
    print(f'Ready: {ASSETS}; isolated Python source: {DEPS}; existing environment unchanged.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    if args.validate_only:
        validate()
        print('Assets verified offline.')
    else:
        setup()
