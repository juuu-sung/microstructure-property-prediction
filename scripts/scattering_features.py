"""Fixed, label-free 201-dimensional wavelet scattering descriptor.

Kymatio lives in an isolated project directory so archived environment inventories
remain unchanged. All transforms run on CPU, one native256 image at a time.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import importlib.metadata
from pathlib import Path
import sys
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DEPS = ROOT / '.deps-scattering'
VERSIONS = {'kymatio': '0.3.0', 'appdirs': '1.4.4', 'configparser': '7.2.0'}
J, L, EPS = 5, 8, 1e-6
PAIRS = tuple((j1, j2) for j1 in range(J) for j2 in range(j1 + 1, J))
NAMES = ('s0',) + tuple(f's1_j{j}_theta{k}' for j in range(J) for k in range(L))
NAMES += tuple(f's2_j{j1}_j{j2}_delta{d}' for j1, j2 in PAIRS for d in range(L))
NAMES += tuple(f's2_normalized_j{j1}_j{j2}_delta{d}' for j1, j2 in PAIRS for d in range(L))
assert len(NAMES) == 201


def dependency_binding():
    found = {d.metadata['Name'].lower(): d.version
             for d in importlib.metadata.distributions(path=[str(DEPS)])}
    if found != VERSIONS:
        raise RuntimeError('Missing/different isolated dependencies. Install requirements-scattering.txt '
                           'with --target .deps-scattering --no-deps; do not modify the existing venv.')
    files = {p.relative_to(DEPS).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(DEPS.rglob('*')) if p.is_file() and
             (p.suffix == '.py' or p.name in ('METADATA', 'WHEEL', 'LICENSE', 'LICENSE.txt', 'LICENSE.rst'))}
    return {'versions': found, 'files': files, 'isolated_directory': str(DEPS.resolve())}


@contextmanager
def isolated_imports():
    old_path = sys.path.copy()
    try:
        sys.path.insert(0, str(DEPS))
        yield
    finally:
        sys.path[:] = old_path


def read_gray(path):
    with Image.open(path) as image:
        if image.mode != 'L' or image.size != (256, 256):
            raise ValueError('Expected native256x256 8-bit grayscale image; no conversion or resizing')
        return np.asarray(image, dtype=np.uint8).copy()


def summarize(paths):
    """Map by explicit scale/orientation metadata, never by returned list position."""
    coefficients = {}
    for p in paths:
        key = (tuple(p['j']), tuple(p['theta']))
        value = np.asarray(p['coef'])
        if key in coefficients or value.shape != (8, 8) or not np.isfinite(value).all():
            raise ValueError('Duplicate/malformed scattering path or coefficient map')
        coefficients[key] = float(value.mean(dtype=np.float64))
    expected = {((), ())}
    expected.update(((j,), (theta,)) for j in range(J) for theta in range(L))
    expected.update(((a, b), (u, v)) for a, b in PAIRS for u in range(L) for v in range(L))
    if set(coefficients) != expected or len(coefficients) != 681:
        raise ValueError('Expected all681 unique scattering paths')
    s0 = coefficients[((), ())]
    s1 = np.array([[coefficients[((j,), (theta,))] for theta in range(L)] for j in range(J)])
    s2 = np.array([[np.mean([coefficients[((a, b), (theta, (theta + delta) % L))]
                            for theta in range(L)]) for delta in range(L)] for a, b in PAIRS])
    denominator = s1.mean(axis=1)
    if min(s1.min(), s2.min()) < -1e-10:
        raise ValueError('Unexpected negative modulus averages')
    normalized = s2 / np.array([denominator[a] + EPS for a, _ in PAIRS])[:, None]
    vector = np.concatenate(([s0], s1.ravel(), s2.ravel(), normalized.ravel()))
    validate_matrix(vector[None, :])
    return vector, {'path_count': len(coefficients), 'map_shape': [8, 8],
                    's1_scale_means': denominator.tolist(),
                    'low_energy_scale_count': int((denominator <= EPS).sum()),
                    'max_abs_normalized_s2': float(np.abs(normalized).max())}


def validate_matrix(x):
    if x.ndim != 2 or x.shape[1] != 201 or x.dtype != np.float64 or not np.isfinite(x).all():
        raise ValueError('Expected finite float64 scattering201 matrix')


class Extractor:
    def __init__(self):
        self.dependencies = dependency_binding()
        with isolated_imports():
            import kymatio
            # kymatio.numpy eagerly imports3D code that uses removed scipy.sph_harm.
            # Load the identical2D frontend directly; do not patch scipy or vendor files.
            from kymatio.scattering2d.frontend.numpy_frontend import ScatteringNumPy2D as Scattering2D
            if not Path(kymatio.__file__).resolve().is_relative_to(DEPS.resolve()):
                raise RuntimeError('Kymatio imported from unexpected location')
            self.transform = Scattering2D(J=J, shape=(256, 256), L=L, max_order=2,
                                          pre_pad=False, backend='numpy', out_type='list')
        if (self.transform._M_padded, self.transform._N_padded) != (320, 320):
            raise ValueError('Unexpected library reflection padding')

    def __call__(self, gray):
        if gray.shape != (256, 256) or gray.dtype != np.uint8:
            raise ValueError('Expected uint8 native image')
        # Fixed physical pixel scale, no per-image normalization or denoising.
        x = np.ascontiguousarray(gray, dtype=np.float64) / 255.
        return summarize(self.transform(x))


def synthetic_check():
    """Numerical/perturbation diagnostics, not a claim of semantic measurement quality."""
    extractor = Extractor()
    grid = np.full((256, 256), 180, np.uint8)
    grid[16::32, :] = 40
    grid[:, 16::32] = 40
    baseline, detail = extractor(grid)
    repeat, _ = extractor(grid)
    np.testing.assert_array_equal(baseline, repeat)
    blank, _ = extractor(np.full_like(grid, 128))
    # Kymatio's discretized lowpass has DC gain slightly different from1.
    # Compare against its actual frequency-zero gain, not an assumed pixel mean.
    dc_gain = float(extractor.transform.phi['levels'][0][0, 0])
    np.testing.assert_allclose(blank[0], (128 / 255.) * dc_gain, rtol=0, atol=1e-12)
    if np.abs(blank[1:121]).max() > 1e-6:
        raise ValueError('Constant image has excessive raw scattering energy')
    rng = np.random.default_rng(271828)
    variants = {'brightness_plus10': grid + np.uint8(10),
                'translate2_reflect': np.pad(grid, 2, mode='reflect')[:256, :256],
                'noise_sigma2': np.clip(np.rint(grid.astype(float) + rng.normal(0, 2, grid.shape)), 0, 255).astype(np.uint8)}
    changes = {}
    for name, image in variants.items():
        row, _ = extractor(image)
        changes[name] = {block: float(np.linalg.norm(row[sl] - baseline[sl]) /
                                      max(np.linalg.norm(baseline[sl]), 1e-12))
                         for block, sl in {'s0': slice(0, 1), 's1': slice(1, 41),
                                          's2': slice(41, 121), 's2_normalized': slice(121, 201)}.items()}
    return {'competition_images_used': 0, 'labels_used': False, 'features': 201,
            'path_detail': detail, 'exact_repeat': True, 'blank_s0': float(blank[0]),
            'lowpass_dc_gain': dc_gain,
            'blank_raw_max_abs': float(np.abs(blank[1:121]).max()),
            'perturbations_relative_l2': changes,
            'interpretation': 'Descriptive sensitivity only; no label-based acceptance/tuning or grain/phase/pore accuracy claim.'}
