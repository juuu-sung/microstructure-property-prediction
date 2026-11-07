"""User-run ImageNet vs ImageNet->MicroNet B0 frozen-feature comparison.

prepare/validate: archived-result audits only. check: synthetic forward only.
extract: native train500, no fit. evaluate: nested Ridge on cached features.
No test access, neural updates, final500 supervised fit, or submission changes.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import os
from pathlib import Path
import platform
import sys
import time
from types import SimpleNamespace

import numpy as np

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
if __package__:
    from . import setup_micronet as assets
    from . import evaluate_scattering_tabpfn as previous
else:
    import setup_micronet as assets
    import evaluate_scattering_tabpfn as previous

t = previous.t
base, core = t.common, t.core
ROOT = base.ROOT
OUT = ROOT / 'reports/efficientnet_micronet'
SEEDS = (42, 2026)
REPS = ('imagenet', 'micronet')
ALPHAS = (10., 100., 1000., 3000.)
BLOCKS = (10, 15)
CHANNELS = (112, 320)
WIDTH = 2 * sum(CHANNELS)
CONTROL = 'current_imagenet_equal'
PRIMARY = 'current_micronet_equal'
PROTOCOL = {
    'version': 1, 'seeds': list(SEEDS), 'representations': list(REPS),
    'architecture': 'efficientnet_pytorch0.7.1 EfficientNet-B0, image_size256 static SAME padding; '
                    'remove only classifier _fc, keep all other checkpoint keys strictly loaded.',
    'weights': assets.WEIGHTS,
    'input': {'size': [256, 256], 'RGB': 'repeat original grayscale three times', 'range': [0, 1],
              'mean': base.EXPECTED_MEAN, 'std': base.EXPECTED_STD,
              'resize': False, 'crop': False, 'denoise': False, 'D4': False, 'AMP': False},
    'features': {'block_indices_zero_based': list(BLOCKS), 'channels': list(CHANNELS),
                 'map_sizes': [[16, 16], [8, 8]], 'width': WIDTH,
                 'pool': 'block10 mean112,std112,block15 mean320,std320; spatial population std; no head map',
                 'backbone': 'eval, requires_grad=False, frozen BatchNorm; no label-dependent feature operation'},
    'regression': {'alphas': list(ALPHAS), 'outer_folds': 5, 'inner_folds': 3,
                   'inner_seed': 'seed+outer_fold', 'inner_strata': 'target rank deciles inside outer train400',
                   'selection': 'Pooled inner MSE per fixed representation; exact ties higher alpha. '
                                'X/y StandardScaler fitted separately inside each training subset.',
                   'representation_selection': False, 'fits_per_seed': 130, 'threads': 2},
    'current': 'Archived same-fold Scattering486 blend, never refitted or retuned.',
    'blend': 'current_imagenet_equal=.5*current+.5*imagenet; '
             'current_micronet_equal=.5*current+.5*micronet. Fixed before extraction, no clipping/weight search.',
    'primary': PRIMARY, 'control': CONTROL, 'priority_mean_gain': .30,
    'decision': 'Both seeds must improve current and ImageNet blend control; expansion additionally '
                'mean gain vs current>=0.30. Review standalone, common16, remaining484, middle80, tails, '
                'noise, folds and new worst20. No automatic adoption or significance claim.',
    'scope': 'Train500 only, frozen features then130 Ridge fits/seed. No neural fine-tuning, test, '
             'final500 supervised fit or submission. Same repeatedly studied500, no independent holdout.',
    'attribution': 'NASA MicroNet; EfficientNet-PyTorch MIT; Built with PriorLabs-TabPFN (archived current component).',
}


def environment():
    return {'python': platform.python_version(), **{n: importlib.metadata.version(n) for n in
            ('torch', 'numpy', 'scipy', 'scikit-learn', 'threadpoolctl', 'Pillow', 'safetensors')}}


def source_refs():
    return {Path(p).name: t.file_ref(p) for p in
            (__file__, assets.__file__, previous.__file__, t.__file__, base.__file__, core.__file__)}


def unavailable(*args, **kwargs):
    raise ValueError('A required archived result is incomplete; implicit fitting is forbidden.')


def load_state():
    """Verify the actual current Scattering486 OOF, not the older TabPFN blend."""
    asset_manifest = assets.validate()
    data, hand, historical, old_plan = previous.validate(previous.OUT)
    scattering, _ = previous.load_features(previous.OUT, data, old_plan)
    x = np.c_[hand, scattering]
    refs = {}
    archived = {}
    for seed in SEEDS:
        folder = previous.OUT / f'seed{seed}'
        binding = previous.evaluation_binding(previous.OUT, old_plan, historical[seed], seed)
        if not previous.completed(folder, binding):
            raise ValueError(f'Incomplete current Scattering486 seed{seed}')
        pred = np.full(500, np.nan)
        for fold in range(5):
            vi, p = previous.evaluate_fold(folder, binding, x, data.y, historical[seed]['folds'], seed, fold, unavailable)
            pred[vi] = p
        metrics, expected, _, _ = previous.result_data(data, historical[seed], pred, old_plan['frozen']['common_failure_ids'])
        if t.read_json(folder / 'summary.json')['metrics'] != metrics:
            raise ValueError('Current metrics do not match archived predictions')
        rows = core.align_rows(core.read_rows(folder / 'oof.csv', set(expected[0])), data.ids, folder)
        for key in expected[0]:
            if key != 'ID':
                np.testing.assert_array_equal([float(r[key]) for r in rows], [float(r[key]) for r in expected])
        refs[seed] = {'current': np.array([r[previous.PRIMARY] for r in expected]),
                      'folds': historical[seed]['folds'].copy()}
        archived[str(seed)] = {n: t.file_ref(folder / n) for n in ('completed_evaluation.json', 'summary.json', 'oof.csv')}
    frozen = {
        'protocol': PROTOCOL, 'sources': source_refs(), 'environment': environment(), 'assets': asset_manifest,
        'asset_manifest': t.file_ref(assets.ASSETS / 'assets.json'),
        'scattering_plan': t.file_ref(previous.OUT / 'protocol.json'), 'scattering_results': archived,
        'train_csv': t.file_ref(ROOT / 'open/train.csv'),
        'image_inventory': old_plan['frozen']['image_inventory'],
        'ids_sha256': base.sha256_array(data.ids), 'target_sha256': base.sha256_array(data.y),
        'noise_sha256': base.sha256_array(data.noise),
        'common_failure_ids': old_plan['frozen']['common_failure_ids'],
        'references': {str(s): {n: base.sha256_array(v) for n, v in refs[s].items()} for s in SEEDS},
        'submissions': [t.file_ref(ROOT / p) for p in (
            'reports/final_ensemble/submission/submission.csv',
            'reports/final_tabpfn_blend/submission/submission.csv',
            'reports/final_scattering_blend/submission/submission.csv')],
    }
    return data, refs, frozen


def prepare(folder):
    data, refs, frozen = load_state()
    path = folder / 'protocol.json'
    if path.exists():
        if t.read_json(path)['frozen'] != frozen:
            raise ValueError('Existing protocol differs; preserve old outputs and use another folder.')
        print('Prepared protocol and current references verified; no fitting or extraction.')
        return
    if folder.exists() and any(folder.iterdir()):
        raise ValueError('Nonempty unprepared output folder')
    t.write_json(path, {'frozen': frozen, 'created_unix': time.time(), 'competition_fits': 0, 'competition_forwards': 0})
    print(f'Prepared ImageNet/MicroNet {WIDTH} features;130 Ridge fits/seed,260 total; {path}')
    for seed in SEEDS:
        print(f'Current Scattering486 seed{seed}: {core.metrics(data.y, refs[seed]["current"])["rmse"]:.6f}')


def validate(folder):
    if not (folder / 'protocol.json').is_file():
        raise ValueError('Run prepare before this command.')
    data, refs, frozen = load_state()
    plan = t.read_json(folder / 'protocol.json')
    if plan['frozen'] != frozen or plan['competition_fits'] != 0 or plan['competition_forwards'] != 0:
        raise ValueError('Frozen code, assets, inputs, environment or reference predictions changed.')
    return data, refs, plan


def import_encoder():
    assets.validate()
    dependency = str(assets.DEPS)
    if dependency not in sys.path:
        sys.path.insert(0, dependency)
    import efficientnet_pytorch
    if Path(efficientnet_pytorch.__file__).resolve().parent != assets.DEPS / 'efficientnet_pytorch':
        raise ValueError('Unexpected EfficientNet import location')
    if efficientnet_pytorch.__version__ != '0.7.1':
        raise ValueError('Unexpected EfficientNet source version')
    return efficientnet_pytorch.EfficientNet


def load_model(name):
    import torch
    from safetensors.torch import load_file
    if name not in REPS:
        raise ValueError('Unknown representation')
    encoder = import_encoder()
    model = encoder.from_name('efficientnet-b0', image_size=256)
    del model._fc
    path = assets.asset_path(name)
    state = load_file(str(path)) if path.suffix == '.safetensors' else torch.load(path, map_location='cpu', weights_only=True)
    # Published files contain the classification head. It is the only allowed omission.
    if set(k for k in state if k.startswith('_fc.')) != {'_fc.weight', '_fc.bias'}:
        raise ValueError('Unexpected classifier schema')
    if tuple(state['_fc.weight'].shape) != (1000, 1280) or tuple(state['_fc.bias'].shape) != (1000,):
        raise ValueError('Checkpoint classifier shape differs')
    state = {k: v for k, v in state.items() if not k.startswith('_fc.')}
    model.load_state_dict(state, strict=True)
    if len(model._blocks) != 16 or [model._blocks[i]._block_args.output_filters for i in BLOCKS] != list(CHANNELS):
        raise ValueError('Unexpected B0 block layout')
    model.requires_grad_(False)
    model.eval()
    return model


def forward_maps(model, x):
    if tuple(x.shape[1:]) != (3, 256, 256):
        raise ValueError('Expected native RGB256 input')
    x = model._swish(model._bn0(model._conv_stem(x)))
    maps = []
    for index, block in enumerate(model._blocks):
        x = block(x, drop_connect_rate=0.)
        if index in BLOCKS:
            maps.append(x)
    if [tuple(m.shape[1:]) for m in maps] != [(112, 16, 16), (320, 8, 8)]:
        raise ValueError('Native feature map dimensions differ')
    return maps


def pool_maps(maps):
    import torch
    if [tuple(m.shape[1:]) for m in maps] != [(112, 16, 16), (320, 8, 8)]:
        raise ValueError('Wrong feature map layout')
    features = torch.cat([v for m in maps for v in (m.mean((2, 3)), m.std((2, 3), correction=0))], dim=1)
    if features.shape[1] != WIDTH or not torch.isfinite(features).all():
        raise ValueError('Invalid pooled features')
    return features


def configure(device_name):
    import torch
    device, detail = base.configure_runtime(torch, np, SimpleNamespace(device=device_name, seed=42, threads=2))
    return torch, device, {'versions': environment(), 'device': detail, 'amp': False, 'tf32': False}


def check(device_name, batch_size=8):
    if batch_size < 1:
        raise ValueError('batch-size must be positive')
    torch, device, runtime = configure(device_name)
    generator = torch.Generator().manual_seed(1729)
    gray = torch.rand(batch_size, 1, 256, 256, generator=generator)
    x = gray.repeat(1, 3, 1, 1)
    x = (x - torch.tensor(base.EXPECTED_MEAN)[None, :, None, None]) / torch.tensor(base.EXPECTED_STD)[None, :, None, None]
    results = {}
    for name in REPS:
        model = load_model(name).to(device=device, dtype=torch.float32)
        before = base.feature_state_hash(model)
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        with torch.inference_mode():
            maps = forward_maps(model, x.to(device))
            features = pool_maps(maps)
            # Check our tap locations against the upstream independent endpoint implementation.
            endpoints = model.extract_endpoints(x.to(device))
            for actual, key in zip(maps, ('reduction_4', 'reduction_5')):
                torch.testing.assert_close(actual, endpoints[key], rtol=0, atol=0)
        if before != base.feature_state_hash(model):
            raise ValueError('Frozen state changed on synthetic forward')
        results[name] = {'feature_shape': list(features.shape), 'map_shapes': [list(m.shape) for m in maps],
                         'state_sha256': before, 'strict_checkpoint_load': True, 'upstream_endpoint_match': True,
                         'frozen_state_unchanged': True,
                         'peak_gpu_allocated_bytes': torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None}
        del model, maps, features, endpoints
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    if results['imagenet']['state_sha256'] == results['micronet']['state_sha256']:
        raise ValueError('Candidate and control weights are identical')
    result = {'synthetic_only': True, 'competition_forwards': 0, 'batch_size': batch_size, 'runtime': runtime, 'models': results}
    print(base.canonical_json(result))
    return result


def validate_feature_bundle(path, data, plan):
    if not path.is_file():
        raise ValueError('Features missing. Complete extract before evaluate; no regression started.')
    metadata_path = path.with_name('metadata.json')
    if not metadata_path.is_file():
        raise ValueError('Feature metadata missing. Run extract to finish/rebuild the incomplete cache.')
    meta = t.read_json(metadata_path)
    if meta['plan'] != t.digest(plan['frozen']) or meta['file_sha256'] != base.sha256_file(path):
        raise ValueError('Feature binding or file hash changed')
    if meta['test_accessed'] is not False or meta['labels_used_for_features'] is not False:
        raise ValueError('Feature scope differs')
    features = {}
    with np.load(path, allow_pickle=False) as z:
        if set(z.files) != {'ids', *REPS}:
            raise ValueError('Unexpected feature keys')
        np.testing.assert_array_equal(z['ids'], data.ids)
        for name in REPS:
            features[name] = z[name].copy()
            if features[name].shape != (len(data.ids), WIDTH) or features[name].dtype != np.float32 or not np.isfinite(features[name]).all():
                raise ValueError('Invalid feature array: ' + name)
        for name in z.files:
            if meta['arrays'][name] != base.sha256_array(z[name]):
                raise ValueError('Feature array hash changed')
    return features, meta


def load_features(folder, data, plan):
    return validate_feature_bundle(folder / 'features/features.npz', data, plan)


def extract(folder, device_name, batch_size):
    if batch_size < 1:
        raise ValueError('batch-size must be positive')
    data, _, plan = validate(folder)
    path = folder / 'features/features.npz'
    if path.with_name('metadata.json').exists():
        load_features(folder, data, plan)
        print('Completed feature cache verified; no re-extraction.')
        return
    torch, device, runtime = configure(device_name)
    dataset = base.TrainImageDataset(ROOT / 'open', data.ids, base.EXPECTED_MEAN, base.EXPECTED_STD)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0,
                                        pin_memory=device.type == 'cuda', drop_last=False)
    arrays = {'ids': data.ids}
    states = {}
    started = time.perf_counter()
    for name in REPS:
        model = load_model(name).to(device=device, dtype=torch.float32)
        state = base.feature_state_hash(model)
        parts, count = [], 0
        with torch.inference_mode():
            for step, x in enumerate(loader):
                parts.append(pool_maps(forward_maps(model, x.to(device=device, dtype=torch.float32))).cpu().numpy())
                count += len(x)
                if step % 5 == 0 or count == len(data.ids):
                    print(f'Extracted {name} {count}/{len(data.ids)}; features={WIDTH}; elapsed={time.perf_counter()-started:.1f}s', flush=True)
        if state != base.feature_state_hash(model) or any(m.training for m in model.modules()) or any(p.requires_grad for p in model.parameters()):
            raise ValueError('Backbone/BatchNorm was not frozen')
        arrays[name] = np.concatenate(parts)
        states[name] = state
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    if states['imagenet'] == states['micronet'] or np.array_equal(arrays['imagenet'], arrays['micronet']):
        raise ValueError('Control and candidate unexpectedly identical')
    validate(folder)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('wb') as out:
        np.savez_compressed(out, **arrays)
    os.replace(tmp, path)
    t.write_json(path.with_name('metadata.json'), {
        'plan': t.digest(plan['frozen']), 'file_sha256': base.sha256_file(path),
        'arrays': {n: base.sha256_array(x) for n, x in arrays.items()}, 'states': states,
        'runtime': runtime, 'batch_size': batch_size, 'seconds': time.perf_counter()-started,
        'test_accessed': False, 'labels_used_for_features': False, 'neural_updates': 0,
    })
    load_features(folder, data, plan)
    print(f'Saved {path}; ImageNet and MicroNet=(500,{WIDTH}); no neural updates.')


def fit_fold(features, y, folds, seed, fold, *, factory=None):
    """Only outer-training labels affect splits, scalers, alpha selection and fits."""
    from sklearn.model_selection import StratifiedKFold
    factory = factory or (lambda alpha, name: core.make_estimator(alpha))
    ti, vi = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
    if (len(ti), len(vi)) != (400, 100):
        raise ValueError('Expected outer train400 / held-out100')
    yt = y[ti]
    splits = list(StratifiedKFold(3, shuffle=True, random_state=seed+fold).split(yt, core.rank_groups(yt, 10)))
    result = {'validation_indices': vi.tolist(), 'predictions': {}, 'selection': {}, 'candidates': {}}
    for name in REPS:
        x = features[name].astype(np.float64)
        candidates = []
        for alpha in ALPHAS:
            inner = np.full(len(ti), np.nan)
            for it, iv in splits:
                model = factory(alpha, name)
                model.fit(x[ti[it]], yt[it])
                inner[iv] = model.predict(x[ti[iv]])
            if not np.isfinite(inner).all():
                raise ValueError('Nonfinite inner predictions')
            candidates.append({'alpha': alpha, 'inner_mse': float(np.mean((inner-yt)**2))})
        chosen = min(candidates, key=lambda row: (row['inner_mse'], -row['alpha']))
        model = factory(chosen['alpha'], name)
        model.fit(x[ti], yt)
        pred = np.asarray(model.predict(x[vi]), dtype=float)
        if pred.shape != (100,) or not np.isfinite(pred).all():
            raise ValueError('Nonfinite outer predictions')
        result['predictions'][name] = pred.tolist()
        result['selection'][name] = chosen
        result['candidates'][name] = candidates
    return result


def fold_cell(folder, binding, features, y, folds, seed, fold, *, runner=None):
    bound = {'evaluation': binding, 'fold': fold, 'train_indices': np.flatnonzero(folds != fold).tolist(),
             'valid_indices': np.flatnonzero(folds == fold).tolist()}
    path = folder / f'fold{fold}.json'
    if path.exists():
        saved = t.read_json(path)
        if saved['binding'] != bound or saved['result_sha256'] != t.digest(saved['result']):
            raise ValueError('Fold cache binding/result changed')
        result = saved['result']
    else:
        result = (runner or fit_fold)(features, y, folds, seed, fold)
        t.write_json(path, {'binding': bound, 'result': result, 'result_sha256': t.digest(result)})
    if result['validation_indices'] != bound['valid_indices'] or set(result['predictions']) != set(REPS):
        raise ValueError('Invalid cached prediction layout')
    for name in REPS:
        pred = np.asarray(result['predictions'][name])
        candidates = result['candidates'][name]
        if pred.shape != (100,) or not np.isfinite(pred).all():
            raise ValueError('Invalid cached predictions')
        if [c['alpha'] for c in candidates] != list(ALPHAS) or not all(np.isfinite(c['inner_mse']) and c['inner_mse'] >= 0 for c in candidates):
            raise ValueError('Invalid cached candidate scores')
        if min(candidates, key=lambda c: (c['inner_mse'], -c['alpha'])) != result['selection'][name]:
            raise ValueError('Cached inner choice differs')
    return result


def compose(ref, predictions):
    return {'current': ref['current'], **predictions,
            CONTROL: .5*ref['current'] + .5*predictions['imagenet'],
            PRIMARY: .5*ref['current'] + .5*predictions['micronet']}


def result_data(data, ref, predictions, common_ids):
    output = compose(ref, predictions)
    if any(p.shape != data.y.shape or not np.isfinite(p).all() for p in output.values()):
        raise ValueError('Incomplete predictions')
    metrics = {n: core.metrics(data.y, p) for n, p in output.items()}
    for values in metrics.values():
        values['gain_vs_current'] = metrics['current']['rmse'] - values['rmse']
        values['gain_vs_control'] = metrics[CONTROL]['rmse'] - values['rmse']
    target, noise = core.rank_groups(data.y, 10), core.rank_groups(data.noise, 4)
    common = np.isin(data.ids, common_ids)
    masks = {'all': np.ones(len(data.y), bool), 'low10': target == 0, 'high10': target == 9,
             'tails20': (target == 0) | (target == 9), 'middle80': (target > 0) & (target < 9),
             'common16': common, 'remaining484': ~common}
    masks.update({f'noiseQ{q+1}': noise == q for q in range(4)})
    masks.update({f'fold{f}': ref['folds'] == f for f in range(5)})
    worst = {n: set(np.argsort((p-data.y)**2, kind='stable')[-20:].tolist()) for n, p in output.items()}
    for name in ('current', CONTROL, PRIMARY):
        masks[name + '_worst20'] = np.isin(np.arange(len(data.y)), list(worst[name]))
    diag = [{'model': n, 'group': g, **core.metrics(data.y[m], p[m])}
            for n, p in output.items() for g, m in masks.items() if m.any()]
    oof = [{'ID': str(i), 'hardness': float(data.y[j]), 'fold': int(ref['folds'][j]),
            **{n: float(p[j]) for n, p in output.items()}} for j, i in enumerate(data.ids)]
    errors = []
    for j in np.argsort(-(output[PRIMARY]-data.y)**2, kind='stable'):
        record = {'ID': str(data.ids[j]), 'hardness': float(data.y[j]), 'noise': float(data.noise[j]),
                  'fold': int(ref['folds'][j]), 'common16': bool(common[j])}
        for name, p in output.items():
            record[name + '_error'] = float(p[j]-data.y[j])
            record[name + '_worst20'] = j in worst[name]
        for name in ('current', CONTROL):
            record['sse_gain_vs_' + name] = float((output[name][j]-data.y[j])**2-(output[PRIMARY][j]-data.y[j])**2)
        errors.append(record)
    err, old_err = output[PRIMARY]-data.y, output['current']-data.y
    detail = {
        'standalone_gain_vs_imagenet': metrics['imagenet']['rmse']-metrics['micronet']['rmse'],
        'common16_improved_vs_current': int(((err**2 < old_err**2) & common).sum()),
        'common16_in_candidate_worst20': int(sum(common[list(worst[PRIMARY])])),
        'common16_same_error_sign': int((np.sign(err[common]) == np.sign(old_err[common])).sum()),
        'common16_sse_fraction': float(np.sum(err[common]**2)/np.sum(err**2)) if np.any(err) else 0.,
        'common16_current_rmse': core.metrics(data.y[common], output['current'][common])['rmse'],
        'common16_control_rmse': core.metrics(data.y[common], output[CONTROL][common])['rmse'],
        'common16_candidate_rmse': core.metrics(data.y[common], output[PRIMARY][common])['rmse'],
        'new_worst20_ids': [str(data.ids[j]) for j in sorted(worst[PRIMARY]-worst['current'])],
        'left_worst20_ids': [str(data.ids[j]) for j in sorted(worst['current']-worst[PRIMARY])],
        'residual_correlation_vs_current': core.correlation(err, old_err),
        'folds_improved_vs_current': sum(core.metrics(data.y[ref['folds']==f], output[PRIMARY][ref['folds']==f])['rmse'] <
                                        core.metrics(data.y[ref['folds']==f], output['current'][ref['folds']==f])['rmse'] for f in range(5)),
    }
    return metrics, oof, diag, errors, detail


def expected_outputs():
    return {'oof.csv', 'summary.json', 'diagnostics.csv', 'per_image_errors.csv', 'common16.csv', 'report.md',
            *{f'fold{f}.json' for f in range(5)}}


def completed(folder, binding):
    if not (folder / 'completed.json').exists():
        return False
    saved = t.read_json(folder / 'completed.json')
    if saved['binding'] != binding or set(saved['files']) != expected_outputs():
        raise ValueError('Completion manifest/binding changed')
    for name, sha in saved['files'].items():
        if base.sha256_file(folder / name) != sha:
            raise ValueError('Completed artifact changed: ' + name)
    return True


def evaluation_binding(folder, seed, plan):
    return {'seed': seed, 'plan': t.digest(plan['frozen']), 'features': t.file_ref(folder / 'features/features.npz'),
            'feature_metadata': t.file_ref(folder / 'features/metadata.json'), 'environment': environment()}


def save_results(folder, data, ref, predictions, selections, binding, plan):
    metrics, oof, diag, errors, detail = result_data(data, ref, predictions, plan['frozen']['common_failure_ids'])
    for filename, rows in (('oof.csv', oof), ('diagnostics.csv', diag), ('per_image_errors.csv', errors),
                            ('common16.csv', [row for row in errors if row['common16']])):
        t.write_csv(folder / filename, rows)
    t.write_json(folder / 'summary.json', {'binding': binding, 'metrics': metrics, 'selection': selections,
                 'diagnosis': detail, 'primary': PRIMARY, 'ridge_fits': 130, 'neural_updates': 0,
                 'test_accessed': False, 'independent_holdout': False})
    lines = [f'# EfficientNet ImageNet/MicroNet · seed{binding["seed"]}', '',
             '| Model | OOF RMSE | Gain vs current486 | Gain vs ImageNet blend |', '|---|---:|---:|---:|']
    lines += [f'| {n} | {m["rmse"]:.6f} | {m["gain_vs_current"]:+.6f} | {m["gain_vs_control"]:+.6f} |' for n, m in metrics.items()]
    lines += ['', 'Fixed ImageNet/MicroNet representations; inner alpha selection only. Primary is50:50 current/MicroNet.',
              'Same repeatedly studied500; no independent holdout or significance. No neural updates or test access.',
              'Built with PriorLabs-TabPFN (unchanged archived current component).', '', '## Common failures',
              f'Current/control/candidate common16 RMSE: {detail["common16_current_rmse"]:.6f} / '
              f'{detail["common16_control_rmse"]:.6f} / {detail["common16_candidate_rmse"]:.6f}.',
              f'Candidate worst20 retains {detail["common16_in_candidate_worst20"]}/16; '
              f'{detail["common16_improved_vs_current"]}/16 errors improved; '
              f'{detail["common16_same_error_sign"]}/16 retain error sign.',
              'New worst20: ' + ', '.join(detail['new_worst20_ids']),
              'See diagnostics.csv, common16.csv and per_image_errors.csv; numeric gates alone do not authorize adoption.']
    (folder / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf8')
    t.write_json(folder / 'completed.json', {'binding': binding,
                 'files': {n: base.sha256_file(folder / n) for n in sorted(expected_outputs())}})
    for name in (*REPS, CONTROL, PRIMARY):
        m = metrics[name]
        print(f'{name}: RMSE={m["rmse"]:.6f}; gain vs current={m["gain_vs_current"]:+.6f}; gain vs control={m["gain_vs_control"]:+.6f}')
    print(f'Report: {folder / "report.md"}')


def aggregate(records):
    if [r['seed'] for r in records] != list(SEEDS):
        raise ValueError('Both fixed seeds required')
    gain = float(np.mean([r['gain_vs_current'] for r in records]))
    positive = all(r['gain_vs_current'] > 0 and r['gain_vs_control'] > 0 for r in records)
    return {'primary': PRIMARY, 'per_seed': records, 'mean_gain_vs_current': gain,
            'mean_gain_vs_control': float(np.mean([r['gain_vs_control'] for r in records])),
            'all_seeds_improve_both': positive, 'passes_expansion_priority': positive and gain >= .30,
            'additional_review_required': 'Standalone, common16, remaining484, middle80, tails, noise and new worst20; no automatic adoption.',
            'aggregation': 'Mean per-seed RMSE differences; same500; no prediction averaging or independent inference.'}


def compare(folder, state=None):
    data, refs, plan = state or validate(folder)
    features, _ = load_features(folder, data, plan)
    records = []
    for seed in SEEDS:
        sub = folder / f'seed{seed}'
        binding = evaluation_binding(folder, seed, plan)
        if not completed(sub, binding):
            raise ValueError(f'Complete evaluate --seed {seed} before compare.')
        predictions = {name: np.full(len(data.ids), np.nan) for name in REPS}
        for fold in range(5):
            cell = fold_cell(sub, binding, features, data.y, refs[seed]['folds'], seed, fold, runner=unavailable)
            for name in REPS:
                predictions[name][cell['validation_indices']] = cell['predictions'][name]
        metrics, _, _, _, detail = result_data(data, refs[seed], predictions, plan['frozen']['common_failure_ids'])
        summary = t.read_json(sub / 'summary.json')
        if summary['metrics'] != metrics or summary['diagnosis'] != detail:
            raise ValueError('Saved summary differs from recomputed predictions')
        records.append({'seed': seed, 'current_rmse': metrics['current']['rmse'], 'control_rmse': metrics[CONTROL]['rmse'],
                        'candidate_rmse': metrics[PRIMARY]['rmse'], 'gain_vs_current': metrics[PRIMARY]['gain_vs_current'],
                        'gain_vs_control': metrics[PRIMARY]['gain_vs_control'], 'diagnosis': detail})
    result = {**aggregate(records), 'protocol_sha256': t.digest(plan['frozen'])}
    t.write_json(folder / 'comparison.json', result)
    print(base.canonical_json(result))
    return result


def evaluate(folder, seed):
    from threadpoolctl import threadpool_limits
    data, refs, plan = validate(folder)
    features, _ = load_features(folder, data, plan)
    sub = folder / f'seed{seed}'
    binding = evaluation_binding(folder, seed, plan)
    if completed(sub, binding):
        print(f'Completed seed{seed} verified; no refitting.')
    else:
        predictions = {n: np.full(len(data.ids), np.nan) for n in REPS}
        selections = []
        with threadpool_limits(limits=2):
            for fold in range(5):
                result = fold_cell(sub, binding, features, data.y, refs[seed]['folds'], seed, fold)
                for name in REPS:
                    predictions[name][result['validation_indices']] = result['predictions'][name]
                selections.append({'fold': fold, 'choices': result['selection']})
                print(f'Outer{fold}: ImageNet alpha={result["selection"]["imagenet"]["alpha"]:g}; '
                      f'MicroNet alpha={result["selection"]["micronet"]["alpha"]:g}; outer predictions saved', flush=True)
        validate(folder)
        load_features(folder, data, plan)
        save_results(sub, data, refs[seed], predictions, selections, binding, plan)
    if all((folder / f'seed{s}/completed.json').exists() for s in SEEDS):
        compare(folder, (data, refs, plan))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'validate', 'check', 'extract', 'evaluate', 'compare'))
    parser.add_argument('--output-dir', type=Path, default=OUT)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--seed', type=int, choices=SEEDS, default=42)
    args = parser.parse_args()
    folder = base.project_path(args.output_dir)
    if args.command == 'prepare':
        prepare(folder)
    elif args.command == 'validate':
        validate(folder)
        print('Current486, folds, train inventory, assets and frozen protocol verified; no fits or forwards.')
    elif args.command == 'check':
        check(args.device, args.batch_size)
    elif args.command == 'extract':
        extract(folder, args.device, args.batch_size)
    elif args.command == 'evaluate':
        evaluate(folder, args.seed)
    else:
        compare(folder)


if __name__ == '__main__':
    main()
