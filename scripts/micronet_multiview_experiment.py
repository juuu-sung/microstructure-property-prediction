"""User-run native-scale MicroNet global versus global+patch regression.

prepare/validate audit existing artifacts; check uses synthetic inputs only.
extract performs frozen train-image forwards; train fits small cached-feature heads.
The pretrained encoder, human annotations, test set and submissions are not modified.
"""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import time

import numpy as np

if __package__:
    from . import efficientnet_micronet_experiment as old
else:
    import efficientnet_micronet_experiment as old

t, base, core = old.t, old.base, old.core
ROOT = old.ROOT
OUT = ROOT / 'reports/micronet_multiview'
SEEDS = (42, 2026)
REPS = ('global_only', 'global_patches')
PRIMARY, CONTROL = 'current_multiview_equal', 'current_global_equal'
RECTS = ((0, 0, 128, 128), (128, 0, 128, 128), (0, 128, 128, 128), (128, 128, 128, 128))
# center x/y in [-1,1], then full-image indicator; fixed top-left, top-right, bottom-left, bottom-right.
POSITIONS = ((0., 0., 1.), (-.5, -.5, 0.), (.5, -.5, 0.), (-.5, .5, 0.), (.5, .5, 0.))
EPOCHS = (0, 5, 10, 15, 20, 25, 30, 35, 40)
HIDDEN, BATCH = 32, 32
LR, WEIGHT_DECAY = .001, .01
PROTOCOL = {
    'version': 1, 'seeds': list(SEEDS), 'representations': list(REPS),
    'encoder': 'Pinned ImageNet->MicroNet EfficientNet-B0; eval and requires_grad=False including BN; no fine-tuning.',
    'views': {'global': [0, 0, 256, 256], 'patches_xywh': [list(r) for r in RECTS],
              'order': ['full', 'top_left', 'top_right', 'bottom_left', 'bottom_right'],
              'positions': [list(p) for p in POSITIONS], 'resize': False, 'overlap': False,
              'denoise': False, 'augmentation': False, 'normalization': 'Unchanged pinned ImageNet mean/std'},
    'features': {'width_per_view': old.WIDTH, 'taps': list(old.BLOCKS), 'pool': 'Each tap spatial mean and population std',
                 'global': 'Reuse exact archived864 after newly computed global allclose(rtol=1e-4,atol=3e-5).',
                 'patch': 'Same weights/static padding; verify native128 outputs against independently constructed128 encoder.'},
    'head': {'hidden': HIDDEN, 'projection': 'Linear(864+3,32), GELU, LayerNorm32',
             'aggregation': 'Gated attention: softmax(w(tanh(Vh)*sigmoid(Uh))) over eligible views',
             'readout': 'Dropout0.1 then Linear32->1, zero initialized; standardized target',
             'control': 'Identical nominal architecture and paired initialization; one eligible token vs five. '
                        'Single-token attention has no effective selection capacity.',
             'scaler': 'Feature channel mean/std on eligible tokens of fitting images only; '
                       'target mean/std on fitting image labels only; no clipping/PCA.',
             'units': 'One loss per original image; patches are never independent target-labelled samples.'},
    'optimization': {'optimizer': 'AdamW', 'lr': LR, 'weight_decay': WEIGHT_DECAY, 'batch_size_images': BATCH,
                     'max_epochs': 40, 'checkpoints': list(EPOCHS), 'gradient_clip_norm': 1.,
                     'lr_schedule': 'epoch e starts at lr*(0.1+0.9*(1+cos(pi*(e-1)/40))/2); refit same prefix',
                     'loss': 'Unweighted image-level standardized MSE', 'AMP': False, 'TF32': False,
                     'random_seed': 'seed + 1009*outer_fold + inner_fold; refit uses +9; paired across representations'},
    'validation': {'outer_folds': 5, 'inner_folds': 3, 'inner_split': 'Rank deciles of outer training targets; seed+outer_fold',
                   'selection': 'Separate epoch per fixed representation; pooled inner SSE/count in original target units; ties earlier epoch',
                   'outer': 'Refit from scratch on400 for selected epoch; do not score or select using held-out100',
                   'max_trajectories_per_seed': 40, 'scope': 'Same repeatedly studied500; no independent holdout'},
    'blend': 'Fixed current Scattering48650% + each neural model50%; no weight fitting or clipping',
    'primary': PRIMARY, 'control': CONTROL,
    'expansion_gate': 'Both seeds improve current and global control; mean gain vs current>=0.30; '
                      'review common16/remaining484, tails, noise and new worst20. Not significance or automatic adoption.',
    'scope': 'No annotation use, test access, whole500 supervised fit, automatic unfreezing or submission generation.',
    'attribution': 'NASA MicroNet; EfficientNet-PyTorch; attention-inspired image-level regression; '
                   'Built with PriorLabs-TabPFN (unchanged archived baseline).',
}


def source_refs():
    return {Path(p).name: t.file_ref(p) for p in
            (__file__, ROOT / 'scripts/run_micronet_multiview.ps1')}


def load_state():
    data, refs, previous = old.validate(old.OUT)
    _, meta = old.load_features(old.OUT, data, previous)
    for seed in SEEDS:
        sub = old.OUT / f'seed{seed}'
        if not old.completed(sub, old.evaluation_binding(old.OUT, seed, previous)):
            raise ValueError(f'Archived MicroNet seed{seed} is incomplete')
        rows = core.align_rows(core.read_rows(sub / 'oof.csv', {'ID', 'hardness', 'fold', 'micronet'}), data.ids, sub)
        np.testing.assert_array_equal([float(r['hardness']) for r in rows], data.y)
        np.testing.assert_array_equal([int(r['fold']) for r in rows], refs[seed]['folds'])
        refs[seed]['frozen_micronet_ridge'] = np.array([float(r['micronet']) for r in rows])
    frozen = {'protocol': PROTOCOL, 'sources': source_refs(), 'environment': old.environment(),
              'upstream_plan': t.file_ref(old.OUT / 'protocol.json'),
              'upstream_cache': t.file_ref(old.OUT / 'features/features.npz'),
              'upstream_metadata': t.file_ref(old.OUT / 'features/metadata.json'),
              'upstream_state_hash': meta['states']['micronet'],
              'upstream_results': {str(s): t.file_ref(old.OUT / f'seed{s}/completed.json') for s in SEEDS},
              'ids_sha256': base.sha256_array(data.ids), 'target_sha256': base.sha256_array(data.y),
              'references': {str(s): {k: base.sha256_array(v) for k, v in refs[s].items()} for s in SEEDS},
              'common_failure_ids': previous['frozen']['common_failure_ids']}
    return data, refs, frozen


def prepare(folder):
    data, refs, frozen = load_state()
    path = folder / 'protocol.json'
    if path.exists():
        if t.read_json(path)['frozen'] != frozen:
            raise ValueError('Prepared inputs/code changed; preserve results and use a new output folder')
        print('Prepared protocol verified; no extraction or competition fitting.')
        return
    if folder.exists() and any(folder.iterdir()):
        raise ValueError('Nonempty unprepared output folder')
    t.write_json(path, {'frozen': frozen, 'created_unix': time.time(), 'competition_fits': 0, 'competition_forwards': 0})
    print(f'Prepared global-only vs global+4patch; {len(data.ids)} train images; {path}')
    for seed in SEEDS:
        print(f'Current seed{seed}: {core.metrics(data.y, refs[seed]["current"])["rmse"]:.6f}')


def validate(folder):
    if not (folder / 'protocol.json').is_file():
        raise ValueError('Run prepare first')
    data, refs, frozen = load_state()
    plan = t.read_json(folder / 'protocol.json')
    if plan['frozen'] != frozen:
        raise ValueError('Frozen protocol/code/environment/inputs/references changed')
    return data, refs, plan


def native_maps(model, x):
    if x.ndim != 4 or tuple(x.shape[1:]) not in ((3, 256, 256), (3, 128, 128)):
        raise ValueError('Expected native RGB256 or RGB128')
    size = x.shape[-1]
    z = model._swish(model._bn0(model._conv_stem(x)))
    maps = []
    for index, block in enumerate(model._blocks):
        z = block(z, drop_connect_rate=0.)
        if index in old.BLOCKS:
            maps.append(z)
    expected = [(112, size // 16, size // 16), (320, size // 32, size // 32)]
    if [tuple(m.shape[1:]) for m in maps] != expected:
        raise ValueError('Unexpected native map geometry')
    return maps


def native_features(model, x):
    import torch
    maps = native_maps(model, x)
    features = torch.cat([v for m in maps for v in (m.mean((2, 3)), m.std((2, 3), correction=0))], 1)
    if features.shape[1] != old.WIDTH or not torch.isfinite(features).all():
        raise ValueError('Invalid native features')
    return features


def image_views(x):
    if tuple(x.shape[1:]) != (3, 256, 256):
        raise ValueError('Expected full native256 tensor')
    return [x] + [x[:, :, y:y+h, xx:xx+w] for xx, y, w, h in RECTS]


def assert_global_anchor(fresh, archived):
    if fresh.shape != archived.shape or not np.isfinite(fresh).all():
        raise ValueError('Invalid global anchor shape/values')
    delta = np.abs(fresh.astype(float) - archived.astype(float))
    if not np.allclose(fresh, archived, rtol=1e-4, atol=3e-5):
        raise ValueError(f'Global864 anchor mismatch: max_abs={delta.max():.8g}; no new cache saved')
    return {'max_abs': float(delta.max()), 'mean_abs': float(delta.mean()), 'rtol': 1e-4, 'atol': 3e-5}


def load_features(folder, data, plan):
    path = folder / 'features/features.npz'
    if not path.is_file() or not path.with_name('metadata.json').is_file():
        raise ValueError('Features incomplete: run extract successfully before train; no fitting started')
    meta = t.read_json(path.with_name('metadata.json'))
    if meta['plan'] != t.digest(plan['frozen']) or meta['file_sha256'] != base.sha256_file(path):
        raise ValueError('Feature cache binding/hash changed')
    if meta['labels_used'] is not False or meta['test_accessed'] is not False or meta['encoder_updates'] != 0:
        raise ValueError('Feature scope changed')
    if meta['encoder_state'] != plan['frozen']['upstream_state_hash']:
        raise ValueError('Pretrained state differs')
    with np.load(path, allow_pickle=False) as z:
        if set(z.files) != {'ids', 'views'}:
            raise ValueError('Unexpected feature keys')
        np.testing.assert_array_equal(z['ids'], data.ids)
        x = z['views'].copy()
        if x.shape != (len(data.ids), 5, old.WIDTH) or x.dtype != np.float32 or not np.isfinite(x).all():
            raise ValueError('Expected finite float32 [500,5,864] features')
        if any(meta['arrays'][n] != base.sha256_array(z[n]) for n in z.files):
            raise ValueError('Feature array hash differs')
    with np.load(old.OUT / 'features/features.npz', allow_pickle=False) as z:
        np.testing.assert_array_equal(x[:, 0], z['micronet'])
    return x, meta


def extract(folder, device_name, batch_size):
    if batch_size < 1:
        raise ValueError('batch-size must be positive')
    data, _, plan = validate(folder)
    path = folder / 'features/features.npz'
    if path.with_name('metadata.json').exists():
        load_features(folder, data, plan)
        print('Completed five-view features verified; no re-extraction.')
        return
    torch, device, runtime = old.configure(device_name)
    model = old.load_model('micronet').to(device)
    state = base.feature_state_hash(model)
    if state != plan['frozen']['upstream_state_hash']:
        raise ValueError('Model state differs from archived MicroNet')
    dataset = base.TrainImageDataset(ROOT / 'open', data.ids, base.EXPECTED_MEAN, base.EXPECTED_STD)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    parts, count = [], 0
    started = time.perf_counter()
    with torch.inference_mode():
        for step, x in enumerate(loader):
            x = x.to(device)
            # Sequential views keep encoder memory proportional to one image batch, not five.
            parts.append(np.stack([native_features(model, v).cpu().numpy() for v in image_views(x)], axis=1))
            count += len(x)
            if step % 5 == 0 or count == len(data.ids):
                print(f'Extracted {count}/{len(data.ids)} images, 5 native views; elapsed={time.perf_counter()-started:.1f}s', flush=True)
    if state != base.feature_state_hash(model) or any(m.training for m in model.modules()):
        raise ValueError('Encoder/BatchNorm changed')
    views = np.concatenate(parts)
    with np.load(old.OUT / 'features/features.npz', allow_pickle=False) as z:
        anchor = assert_global_anchor(views[:, 0], z['micronet'])
        views[:, 0] = z['micronet']  # Matched control reuses bit-identical archived full-image input.
    validate(folder)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    arrays = {'ids': data.ids, 'views': views}
    with tmp.open('wb') as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(tmp, path)
    t.write_json(path.with_name('metadata.json'), {
        'plan': t.digest(plan['frozen']), 'file_sha256': base.sha256_file(path),
        'arrays': {k: base.sha256_array(v) for k, v in arrays.items()}, 'encoder_state': state,
        'global_anchor': anchor, 'runtime': runtime, 'batch_size': batch_size,
        'seconds': time.perf_counter()-started, 'labels_used': False, 'test_accessed': False, 'encoder_updates': 0})
    load_features(folder, data, plan)
    print(f'Saved {path}; shape={views.shape}; global max difference={anchor["max_abs"]:.8g}')


def make_head(torch, width=old.WIDTH):
    class AttentionRegressor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.project = torch.nn.Sequential(torch.nn.Linear(width+3, HIDDEN), torch.nn.GELU(), torch.nn.LayerNorm(HIDDEN))
            self.v = torch.nn.Linear(HIDDEN, 16)
            self.u = torch.nn.Linear(HIDDEN, 16)
            self.attention = torch.nn.Linear(16, 1, bias=False)
            self.drop = torch.nn.Dropout(.1)
            self.readout = torch.nn.Linear(HIDDEN, 1)
            torch.nn.init.zeros_(self.readout.weight)
            torch.nn.init.zeros_(self.readout.bias)
            self.register_buffer('positions', torch.tensor(POSITIONS, dtype=torch.float32))

        def forward(self, x):
            if x.ndim != 3 or x.shape[1] not in (1, 5):
                raise ValueError('Expected image bags with one or five views')
            pos = self.positions[:x.shape[1]].expand(len(x), -1, -1)
            h = self.project(torch.cat((x, pos), -1))
            score = self.attention(torch.tanh(self.v(h))*torch.sigmoid(self.u(h)))
            weights = torch.softmax(score, dim=1)
            return self.readout(self.drop((weights*h).sum(1))).squeeze(-1)
    return AttentionRegressor()


def view_subset(features, name):
    if name not in REPS:
        raise ValueError('Unknown representation')
    return features[:, :1] if name == 'global_only' else features


def fit_scaling(x, y):
    if len(x) != len(y) or not len(y) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Invalid fitting arrays')
    flat = np.asarray(x, np.float64).reshape(-1, x.shape[-1])
    mean, scale = flat.mean(0), flat.std(0)
    scale[scale < 1e-8] = 1.
    return {'mean': mean, 'scale': scale, 'ymean': float(np.mean(y)), 'yscale': max(float(np.std(y)), 1e-8)}


def lr_factor(epoch):
    return .1 + .9*(1 + math.cos(math.pi*(epoch-1)/40))/2


def trajectory(xtrain, ytrain, xquery, *, seed, device, epochs, query_y=None, checkpoint=None, log=''):
    """Only fitting arrays set preprocessing. query_y scores fixed checkpoints, never affects updates."""
    import torch
    epochs = tuple(epochs)
    if not epochs or min(epochs) < 0 or list(epochs) != sorted(set(epochs)):
        raise ValueError('Invalid checkpoint epochs')
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)
    scaling = fit_scaling(xtrain, ytrain)
    def transform(x):
        return torch.as_tensor(((x-scaling['mean'])/scaling['scale']).astype(np.float32), device=device)
    train, query = transform(xtrain), transform(xquery)
    target = torch.as_tensor((np.asarray(ytrain)-scaling['ymean'])/scaling['yscale'], dtype=torch.float32, device=device)
    model = make_head(torch, xtrain.shape[-1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    records = []
    def score(epoch):
        model.eval()
        with torch.inference_mode():
            pred = model(query).cpu().numpy().astype(float)*scaling['yscale'] + scaling['ymean']
            fitted = model(train).cpu().numpy().astype(float)*scaling['yscale'] + scaling['ymean']
        if not np.isfinite(pred).all() or not np.isfinite(fitted).all():
            raise ValueError('Nonfinite predictions')
        row = {'epoch': epoch, 'predictions': pred.tolist(), 'train_sse': float(np.sum((fitted-ytrain)**2)),
               'train_n': len(ytrain)}
        if query_y is not None:
            row.update({'valid_sse': float(np.sum((pred-query_y)**2)), 'valid_n': len(query_y)})
        records.append(row)
    if 0 in epochs:
        score(0)
    for epoch in range(1, max(epochs)+1):
        model.train()
        for group in optimizer.param_groups:
            group['lr'] = LR*lr_factor(epoch)
        order = rng.permutation(len(ytrain))
        for start in range(0, len(order), BATCH):
            idx = torch.as_tensor(order[start:start+BATCH], device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = (model(train[idx])-target[idx]).square().mean()
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
        if epoch in epochs:
            score(epoch)
            if log:
                print(f'{log}: epoch{epoch}/{max(epochs)} completed', flush=True)
    artifact = None
    if checkpoint is not None:
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        tmp = checkpoint.with_name(checkpoint.name+'.tmp')
        torch.save({'state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    'scaling': {k: torch.as_tensor(v) for k, v in scaling.items()},
                    'seed': seed, 'epoch': max(epochs), 'input_width': xtrain.shape[-1], 'views': xtrain.shape[1]}, tmp)
        os.replace(tmp, checkpoint)
        artifact = t.file_ref(checkpoint)
    return {'records': records, 'scaling': {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in scaling.items()},
            'checkpoint': artifact}


def select_epoch(inner_runs, epochs=EPOCHS):
    curves = []
    for epoch in epochs:
        records = [next(r for r in run['records'] if r['epoch'] == epoch) for run in inner_runs]
        n = sum(r['valid_n'] for r in records)
        mse = sum(r['valid_sse'] for r in records)/n
        if not np.isfinite(mse) or mse < 0:
            raise ValueError('Invalid inner score')
        curves.append({'epoch': epoch, 'inner_mse': mse, 'inner_rmse': math.sqrt(mse), 'n': n,
                       'train_rmse': math.sqrt(sum(r['train_sse'] for r in records)/sum(r['train_n'] for r in records))})
    return min(curves, key=lambda r: (r['inner_mse'], r['epoch'])), curves


def cached_cell(path, binding, runner):
    if path.exists():
        saved = t.read_json(path)
        if saved['binding'] != binding or saved['result_sha256'] != t.digest(saved['result']):
            raise ValueError('Cached training cell binding/result changed: ' + str(path))
        result = saved['result']
        if result.get('checkpoint') and t.file_ref(result['checkpoint']['path']) != result['checkpoint']:
            raise ValueError('Checkpoint changed')
        return result
    result = runner()
    t.write_json(path, {'binding': binding, 'result': result, 'result_sha256': t.digest(result)})
    return result


def fit_fold(features, y, folds, seed, fold, *, device, folder=None, binding=None, run=trajectory):
    from sklearn.model_selection import StratifiedKFold
    ti, vi = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
    if (len(ti), len(vi)) != (400, 100):
        raise ValueError('Expected outer400/100')
    yt = y[ti]
    splits = list(StratifiedKFold(3, shuffle=True, random_state=seed+fold).split(yt, core.rank_groups(yt, 10)))
    result = {'validation_indices': vi.tolist(), 'predictions': {}, 'selection': {}, 'curves': {}}
    for name in REPS:
        x = view_subset(features, name)
        inner = []
        def execute(role, train_idx, valid_idx, epochs, model_seed, labels):
            cell_binding = {'evaluation': binding, 'outer': fold, 'name': name, 'role': role,
                            'train_indices': train_idx.tolist(), 'valid_indices': valid_idx.tolist(),
                            'epochs': list(epochs), 'model_seed': model_seed}
            path = folder / 'cells' / f'fold{fold}' / name / f'{role}.json' if folder else None
            checkpoint = path.with_suffix('.pt') if path and role == 'refit' else None
            def action():
                return run(x[train_idx], y[train_idx], x[valid_idx], seed=model_seed, device=device,
                           epochs=epochs, query_y=labels, checkpoint=checkpoint,
                           log=f'Outer{fold} {name} {role}' if folder else '')
            return cached_cell(path, cell_binding, action) if path else action()
        for inner_fold, (it, iv) in enumerate(splits):
            inner.append(execute(f'inner{inner_fold}', ti[it], ti[iv], EPOCHS,
                                 seed+1009*fold+inner_fold, yt[iv]))
        chosen, curves = select_epoch(inner)
        # Outer targets are never passed to the fit/refit runner or checkpoint selector.
        refit = execute('refit', ti, vi, (chosen['epoch'],), seed+1009*fold+9, None)
        pred = np.asarray(refit['records'][0]['predictions'], dtype=float)
        if pred.shape != (100,) or not np.isfinite(pred).all():
            raise ValueError('Invalid outer predictions')
        result['predictions'][name] = pred.tolist()
        result['selection'][name] = chosen
        result['curves'][name] = curves
        if folder and run is not old.unavailable:
            print(f'Outer{fold}: {name} selected epoch={chosen["epoch"]}; pooled inner RMSE={chosen["inner_rmse"]:.6f}; outer predictions stored', flush=True)
    return result


def fold_cell(folder, binding, features, y, folds, seed, fold, device, *, runner=None):
    bound = {'evaluation': binding, 'fold': fold}
    existed = (folder / f'fold{fold}.json').exists()
    result = cached_cell(folder / f'fold{fold}.json', bound,
                         lambda: (runner or fit_fold)(features, y, folds, seed, fold, device=device,
                                                       folder=folder, binding=binding))
    if result['validation_indices'] != np.flatnonzero(folds == fold).tolist() or set(result['predictions']) != set(REPS):
        raise ValueError('Cached fold prediction layout differs')
    for name in REPS:
        if np.asarray(result['predictions'][name]).shape != (100,) or not np.isfinite(result['predictions'][name]).all():
            raise ValueError('Invalid fold predictions')
        curve = result['curves'][name]
        if [r['epoch'] for r in curve] != list(EPOCHS) or any(r['n'] != 400 or not np.isfinite(r['inner_mse']) for r in curve):
            raise ValueError('Invalid saved inner curve')
        if result['selection'][name] != min(curve, key=lambda r: (r['inner_mse'], r['epoch'])):
            raise ValueError('Cached epoch selection differs')
    if existed:
        # A fold summary must never hide missing/tampered trajectory cells or checkpoints.
        checked = fit_fold(features, y, folds, seed, fold, device=device, folder=folder,
                           binding=binding, run=old.unavailable)
        if checked != result:
            raise ValueError('Saved fold differs from verified training cells')
    return result


def result_data(data, ref, predictions, common_ids):
    output = {'current': ref['current'], 'frozen_micronet_ridge': ref['frozen_micronet_ridge'], **predictions,
              'current_frozen_micronet_equal': .5*ref['current']+.5*ref['frozen_micronet_ridge'],
              CONTROL: .5*ref['current']+.5*predictions['global_only'],
              PRIMARY: .5*ref['current']+.5*predictions['global_patches']}
    if any(p.shape != data.y.shape or not np.isfinite(p).all() for p in output.values()):
        raise ValueError('Invalid output predictions')
    metrics = {n: core.metrics(data.y, p) for n, p in output.items()}
    for m in metrics.values():
        m.update(gain_vs_current=metrics['current']['rmse']-m['rmse'], gain_vs_control=metrics[CONTROL]['rmse']-m['rmse'])
    common = np.isin(data.ids, common_ids)
    decile, quartile = core.rank_groups(data.y, 10), core.rank_groups(data.noise, 4)
    masks = {'all': np.ones(len(data.y), bool), 'common16': common, 'remaining484': ~common,
             'low10': decile == 0, 'high10': decile == 9, 'middle80': (decile > 0)&(decile < 9),
             'tails20': (decile == 0)|(decile == 9)}
    masks.update({f'noiseQ{q+1}': quartile == q for q in range(4)})
    masks.update({f'fold{f}': ref['folds'] == f for f in range(5)})
    worst = {n: set(np.argsort((p-data.y)**2, kind='stable')[-20:].tolist()) for n, p in output.items()}
    for n in ('current', CONTROL, PRIMARY):
        masks[n+'_worst20'] = np.isin(np.arange(len(data.y)), list(worst[n]))
    diag = [{'model': n, 'group': g, **core.metrics(data.y[m], p[m])} for n, p in output.items() for g, m in masks.items() if m.any()]
    rows = [{'ID': str(i), 'hardness': float(data.y[j]), 'fold': int(ref['folds'][j]),
             **{n: float(p[j]) for n, p in output.items()}} for j, i in enumerate(data.ids)]
    errors = []
    for j in np.argsort(-(output[PRIMARY]-data.y)**2, kind='stable'):
        record = {'ID': str(data.ids[j]), 'hardness': float(data.y[j]), 'fold': int(ref['folds'][j]),
                  'noise': float(data.noise[j]), 'common16': bool(common[j])}
        for n, p in output.items():
            record.update({n+'_error': float(p[j]-data.y[j]), n+'_worst20': j in worst[n]})
        for n in ('current', CONTROL):
            record['sse_gain_vs_'+n] = float((output[n][j]-data.y[j])**2-(output[PRIMARY][j]-data.y[j])**2)
        errors.append(record)
    err, prior = output[PRIMARY]-data.y, output['current']-data.y
    detail = {
        'standalone_gain_vs_global': metrics['global_only']['rmse']-metrics['global_patches']['rmse'],
        'common16_improved_vs_current': int(((err**2 < prior**2)&common).sum()),
        'common16_in_candidate_worst20': int(sum(common[list(worst[PRIMARY])])),
        'common16_same_error_sign': int((np.sign(err[common]) == np.sign(prior[common])).sum()),
        'common16_rmse': {n: core.metrics(data.y[common], p[common])['rmse'] for n, p in output.items()},
        'remaining484_rmse': {n: core.metrics(data.y[~common], p[~common])['rmse'] for n, p in output.items()},
        'common16_sse_fraction': float(np.sum(err[common]**2)/np.sum(err**2)) if np.any(err) else 0.,
        'new_worst20_ids': [str(data.ids[j]) for j in sorted(worst[PRIMARY]-worst['current'])],
        'left_worst20_ids': [str(data.ids[j]) for j in sorted(worst['current']-worst[PRIMARY])],
        'residual_correlation_vs_current': core.correlation(err, prior),
    }
    return metrics, rows, diag, errors, detail


def evaluation_binding(folder, seed, plan, device):
    import torch
    return {'seed': seed, 'plan': t.digest(plan['frozen']), 'features': t.file_ref(folder / 'features/features.npz'),
            'metadata': t.file_ref(folder / 'features/metadata.json'), 'environment': old.environment(),
            'device': str(device), 'device_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu',
            'torch_cuda_build': torch.version.cuda, 'head_batch_size': BATCH}


def expected_outputs():
    return {'oof.csv', 'summary.json', 'diagnostics.csv', 'per_image_errors.csv', 'common16.csv', 'report.txt',
            *{f'fold{f}.json' for f in range(5)}}


def completed(folder, binding):
    path = folder / 'completed.json'
    if not path.exists():
        return False
    done = t.read_json(path)
    if done['binding'] != binding or not expected_outputs().issubset(done['files']):
        raise ValueError('Completion binding/schema differs')
    actual = {str(p.relative_to(folder)).replace('\\', '/') for p in (folder / 'cells').rglob('*') if p.is_file()}
    if set(done['files']) != expected_outputs() | actual:
        raise ValueError('Training cell inventory differs')
    for name, sha in done['files'].items():
        if base.sha256_file(folder / name) != sha:
            raise ValueError('Completed result changed: '+name)
    return True


def save_results(folder, data, ref, predictions, selections, binding, plan):
    metrics, rows, diag, errors, detail = result_data(data, ref, predictions, plan['frozen']['common_failure_ids'])
    for n, r in (('oof.csv', rows), ('diagnostics.csv', diag), ('per_image_errors.csv', errors),
                 ('common16.csv', [r for r in errors if r['common16']])):
        t.write_csv(folder / n, r)
    t.write_json(folder / 'summary.json', {'binding': binding, 'metrics': metrics, 'selection': selections,
                                         'diagnosis': detail, 'primary': PRIMARY, 'test_accessed': False,
                                         'encoder_updates': 0, 'independent_holdout': False})
    lines = [f'MicroNet global vs native patches - seed{binding["seed"]}', '',
             'Model | OOF RMSE | Gain vs Scattering486 | Gain vs global blend']
    lines += [f'{n}: {m["rmse"]:.6f} | {m["gain_vs_current"]:+.6f} | {m["gain_vs_control"]:+.6f}' for n, m in metrics.items()]
    lines += ['', 'Fixed two representations and50:50 blends; epoch selected using pooled3 inner folds.',
              'Feature and target scalers fitted only on fitting images; no per-patch hardness targets.',
              'Frozen encoder. Same repeatedly studied500; no independent holdout or automatic adoption.',
              f'Common16 improved: {detail["common16_improved_vs_current"]}; in candidate worst20: {detail["common16_in_candidate_worst20"]}.',
              'See diagnostics.csv, common16.csv, per_image_errors.csv and fold*.json training curves.',
              'Built with PriorLabs-TabPFN (archived baseline component).']
    (folder / 'report.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    names = expected_outputs() | {str(p.relative_to(folder)).replace('\\', '/') for p in (folder / 'cells').rglob('*') if p.is_file()}
    t.write_json(folder / 'completed.json', {'binding': binding, 'files': {n: base.sha256_file(folder / n) for n in sorted(names)}})
    for n in (*REPS, CONTROL, PRIMARY):
        m = metrics[n]
        print(f'{n}: RMSE={m["rmse"]:.6f}; gain vs current={m["gain_vs_current"]:+.6f}; gain vs global control={m["gain_vs_control"]:+.6f}')
    print(f'Report: {folder / "report.txt"}')


def collect(folder, binding, data, ref, features, seed, device, runner=None):
    predictions = {n: np.full(len(data.y), np.nan) for n in REPS}
    selections, seen = [], np.zeros(len(data.y), int)
    for fold in range(5):
        result = fold_cell(folder, binding, features, data.y, ref['folds'], seed, fold, device, runner=runner)
        vi = result['validation_indices']
        seen[vi] += 1
        for name in REPS:
            predictions[name][vi] = result['predictions'][name]
        selections.append({'fold': fold, 'choices': result['selection']})
    np.testing.assert_array_equal(seen, np.ones(len(data.y), int))
    return predictions, selections


def compare(folder):
    data, refs, plan = validate(folder)
    features, _ = load_features(folder, data, plan)
    records = []
    for seed in SEEDS:
        sub = folder / f'seed{seed}'
        if not (sub / 'completed.json').exists():
            raise ValueError(f'Complete train --seed {seed} first')
        binding = t.read_json(sub / 'completed.json')['binding']
        if binding['seed'] != seed or binding['plan'] != t.digest(plan['frozen']) or binding['features'] != t.file_ref(folder / 'features/features.npz') or binding['metadata'] != t.file_ref(folder / 'features/metadata.json'):
            raise ValueError('Completed result no longer matches inputs')
        completed(sub, binding)
        predictions, selections = collect(sub, binding, data, refs[seed], features, seed, None, runner=old.unavailable)
        metrics, _, _, _, detail = result_data(data, refs[seed], predictions, plan['frozen']['common_failure_ids'])
        summary = t.read_json(sub / 'summary.json')
        if summary['metrics'] != metrics or summary['selection'] != selections or summary['diagnosis'] != detail:
            raise ValueError('Summary does not match archived folds')
        records.append({'seed': seed, 'current_rmse': metrics['current']['rmse'], 'control_rmse': metrics[CONTROL]['rmse'],
                        'candidate_rmse': metrics[PRIMARY]['rmse'], 'gain_vs_current': metrics[PRIMARY]['gain_vs_current'],
                        'gain_vs_control': metrics[PRIMARY]['gain_vs_control'], 'diagnosis': detail})
    positive = all(r['gain_vs_current'] > 0 and r['gain_vs_control'] > 0 for r in records)
    gain = float(np.mean([r['gain_vs_current'] for r in records]))
    result = {'primary': PRIMARY, 'per_seed': records, 'all_seeds_improve_both': positive,
              'mean_gain_vs_current': gain, 'mean_gain_vs_control': float(np.mean([r['gain_vs_control'] for r in records])),
              'passes_expansion_priority': positive and gain >= .30,
              'additional_review_required': 'Common16 and remaining484, standalone, tails, noise, new worst20; no automatic adoption.',
              'aggregation': 'Mean per-seed differences; same500, no prediction averaging or independent inference.',
              'protocol_sha256': t.digest(plan['frozen'])}
    t.write_json(folder / 'comparison.json', result)
    print(base.canonical_json(result))
    return result


def train(folder, seed, device_name):
    torch, device, _ = old.configure(device_name)
    data, refs, plan = validate(folder)
    features, _ = load_features(folder, data, plan)
    sub = folder / f'seed{seed}'
    binding = evaluation_binding(folder, seed, plan, device)
    if completed(sub, binding):
        print(f'Completed seed{seed} verified; no refitting.')
        return
    predictions, selections = collect(sub, binding, data, refs[seed], features, seed, device)
    validate(folder)
    load_features(folder, data, plan)
    save_results(sub, data, refs[seed], predictions, selections, binding, plan)


def check(device_name, batch_size):
    torch, device, runtime = old.configure(device_name)
    if batch_size < 1:
        raise ValueError('batch-size must be positive')
    generator = torch.Generator().manual_seed(1731)
    x = torch.rand(batch_size, 1, 256, 256, generator=generator).repeat(1, 3, 1, 1).to(device)
    x = (x-torch.tensor(base.EXPECTED_MEAN, device=device)[None, :, None, None])/torch.tensor(base.EXPECTED_STD, device=device)[None, :, None, None]
    model = old.load_model('micronet').to(device)
    state = base.feature_state_hash(model)
    other = old.import_encoder().from_name('efficientnet-b0', image_size=128)
    del other._fc
    other.load_state_dict(model.cpu().state_dict(), strict=True)
    model.to(device)
    other = other.to(device).eval().requires_grad_(False)
    with torch.inference_mode():
        features = torch.stack([native_features(model, v) for v in image_views(x)], 1)
        torch.testing.assert_close(features[:, 0], old.pool_maps(old.forward_maps(model, x)), rtol=0, atol=0)
        endpoints = other.extract_endpoints(image_views(x)[1])
        for actual, key in zip(native_maps(model, image_views(x)[1]), ('reduction_4', 'reduction_5')):
            torch.testing.assert_close(actual, endpoints[key], rtol=0, atol=0)
    if state != base.feature_state_hash(model):
        raise ValueError('Encoder state changed')
    head = make_head(torch).to(device)
    counts = sum(p.numel() for p in head.parameters())
    # Synthetic differentiability check only; no real images or targets are used.
    head.train()
    optimizer = torch.optim.AdamW(head.parameters(), lr=LR)
    synthetic = features.detach().clone()
    optimizer.zero_grad(set_to_none=True)
    loss = (head(synthetic)-torch.linspace(-1, 1, batch_size, device=device)).square().mean()
    loss.backward()
    if not all(p.grad is None or torch.isfinite(p.grad).all() for p in head.parameters()):
        raise ValueError('Synthetic head gradient is invalid')
    optimizer.step()
    print(base.canonical_json({'runtime': runtime, 'synthetic_feature_shape': list(features.shape),
                               'head_parameters': counts, 'native128_padding_check': 'matched',
                               'global_forward_check': 'matched', 'encoder_unchanged': True,
                               'competition_forwards': 0, 'competition_fits': 0}))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'validate', 'check', 'extract', 'train', 'compare'))
    parser.add_argument('--output-dir', type=Path, default=OUT)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch-size', type=int, default=8, help='Encoder extraction/check batch; head batch fixed32')
    parser.add_argument('--seed', type=int, choices=SEEDS, default=42)
    args = parser.parse_args(argv)
    folder = args.output_dir.resolve()
    if args.command == 'prepare':
        prepare(folder)
    elif args.command == 'validate':
        validate(folder)
        print('Protocol, archived references and source files verified.')
    elif args.command == 'check':
        check(args.device, args.batch_size)
    elif args.command == 'extract':
        extract(folder, args.device, args.batch_size)
    elif args.command == 'train':
        train(folder, args.seed, args.device)
    else:
        compare(folder)


if __name__ == '__main__':
    main()
