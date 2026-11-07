"""User-run fixed TabPFN-v2 ablation: hand185+scattering201 [+ MicroNet PCA64].

Built with PriorLabs-TabPFN. prepare/validate only verify existing train archives.
check uses synthetic data. evaluate fits fold-local PCA and TabPFN contexts;
compare only verifies saved cells. No new image extraction, test access or submission.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import os
from pathlib import Path
import time

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

if __package__:
    from . import efficientnet_micronet_experiment as micro
else:
    import efficientnet_micronet_experiment as micro

scat = micro.previous
t = scat.t
core = t.core
ROOT = t.ROOT
OUT = ROOT / 'reports/micronet_tabpfn'
SEEDS = (42, 2026)
CONTROL = 'control386_blend'
PRIMARY = 'micronet450_blend'
REPS = ('tabpfn386', 'tabpfn450')
PROTOCOL = {
    'version': 1, 'attribution': 'Built with PriorLabs-TabPFN', 'seeds': list(SEEDS),
    'representations': {'tabpfn486': 'hand185 + morphology100 + scattering201; archived control',
        'tabpfn386': 'hand185 + scattering201',
        'tabpfn450': 'hand185 + scattering201 + MicroNet-global864 StandardScaler/PCA64'},
    'pca': {'n_components': 64, 'whiten': False, 'svd_solver': 'full',
        'scaler': 'StandardScaler(with_mean=True,with_std=True)', 'dtype': 'float64', 'threads': 2,
        'fit_scope': 'Outer training400 only. Transform held-out100 with saved training parameters. '
                     'No label inputs, full500 fit, PCA dimension search, or explained-variance cutoff.'},
    'regressor': t.PROTOCOL,
    'fixed_options': 'Pinned TabPFN-v2;8 estimators;FP32;same archived runtime;seed+1009*fold+9. '
                     'Fixed representations; no new inner selection or hyperparameter search.',
    'blend': 'Base=.25*hand185+.25*originalResNet+.5*DINO_D4. Each blend=.5*Base+.5*TabPFN_variant. '
             'Replace ONLY the TabPFN50% component; do not mix new predictions with current486 blend.',
    'reference': 'Verify/reuse all10 completed Scattering486 outer predictions; no implicit baseline refit.',
    'scope': 'Train only;20 new TabPFN context fits and10 PCA fits total across two seeds. '
             'Reuse pretrained MicroNet-global864 features; no backbone training, patch features, test or submission.',
    'primary': PRIMARY, 'priority_mean_gain': .30,
    'decision': 'Both seeds improve current486 and control386 blends; expansion priority additionally '
                'mean gain vs current486>=0.30. Overall performance first; common16 is diagnostic, '
                'not a mandatory adoption gate. No automatic adoption or follow-on dimension/weight search.',
    'interpretation': 'Same repeatedly studied500, not independent holdout. Added PCA utility does not '
                      'by itself establish nonlinear interaction or guarantee competition improvement.',
}


def forbidden(*args, **kwargs):
    raise ValueError('Required completed cell missing; implicit fitting is forbidden.')


def load_state():
    data, base285, historical, old_plan = scat.validate(scat.OUT)
    scatter, _ = scat.load_features(scat.OUT, data, old_plan)
    md, micro_refs, micro_plan = micro.validate(micro.OUT)
    mf, _ = micro.load_features(micro.OUT, md, micro_plan)
    np.testing.assert_array_equal(data.ids, md.ids)
    np.testing.assert_array_equal(data.y, md.y)
    hand = data.features[t.previous.confirmation.AUGMENTED]
    np.testing.assert_array_equal(hand, base285[:, :185])
    raw = {'fixed386': np.c_[hand, scatter], 'micronet864': mf['micronet'].astype(np.float64)}
    if hand.shape != (500, 185) or raw['fixed386'].shape != (500, 386) or raw['micronet864'].shape != (500, 864):
        raise ValueError('Unexpected hand/scattering/MicroNet dimensions')
    if not all(np.isfinite(x).all() for x in raw.values()):
        raise ValueError('Nonfinite input features')
    refs, archives = {}, {}
    for seed in SEEDS:
        previous = historical[seed]
        folder = scat.OUT / f'seed{seed}'
        binding = scat.evaluation_binding(scat.OUT, old_plan, previous, seed)
        if not scat.completed(folder, binding):
            raise ValueError('Incomplete Scattering486 baseline')
        pred = np.full(500, np.nan)
        with contextlib.redirect_stdout(io.StringIO()):
            for fold in range(5):
                vi, p = scat.evaluate_fold(folder, binding, np.c_[base285, scatter], data.y,
                                          previous['folds'], seed, fold, forbidden)
                pred[vi] = p
        current = .5 * (previous['base'] + pred)
        np.testing.assert_array_equal(current, micro_refs[seed]['current'])
        np.testing.assert_array_equal(previous['folds'], micro_refs[seed]['folds'])
        refs[seed] = {'folds': previous['folds'], 'base': previous['base'], 'current486': current,
                      'tabpfn486': pred, 'runtime': previous['runtime']}
        archives[str(seed)] = t.file_ref(folder / 'completed_evaluation.json')
    frozen = {'protocol': PROTOCOL,
        'sources': {p.name: t.file_ref(p) for p in (Path(__file__), ROOT/'scripts/run_micronet_tabpfn.ps1')},
        'environment': t.environment(), 'scattering_plan': t.file_ref(scat.OUT/'protocol.json'),
        'micronet_plan': t.file_ref(micro.OUT/'protocol.json'),
        'feature_files': [t.file_ref(p) for p in (scat.OUT/'features/features.npz',
            scat.OUT/'features/metadata.json', micro.OUT/'features/features.npz', micro.OUT/'features/metadata.json')],
        'raw_arrays': {n: t.common.sha256_array(x) for n, x in raw.items()},
        'ids_sha256': t.common.sha256_array(data.ids), 'target_sha256': t.common.sha256_array(data.y),
        'noise_sha256': t.common.sha256_array(data.noise), 'baseline_archives': archives,
        'common_failure_ids': old_plan['frozen']['common_failure_ids'],
        'references': {str(s): {'runtime': refs[s]['runtime'], 'arrays': {
            n: t.common.sha256_array(refs[s][n]) for n in ('folds', 'base', 'current486', 'tabpfn486')}} for s in SEEDS}}
    return data, raw, refs, frozen


def prepare(folder):
    data, _, refs, frozen = load_state()
    path = folder/'protocol.json'
    if path.exists():
        if t.read_json(path)['frozen'] != frozen:
            raise ValueError('Frozen experiment changed; preserve existing results.')
        print('Prepared protocol verified; no fitting or feature extraction.')
        return
    if folder.exists() and any(folder.iterdir()):
        raise ValueError('Nonempty unprepared output folder')
    t.write_json(path, {'frozen': frozen, 'created_unix': time.time(),
                        'competition_fits_during_preparation': 0, 'competition_forwards_during_preparation': 0})
    print(f'Prepared386/450 vs archived486;20 new TabPFN contexts,10 fold-local PCA fits: {path}')
    for seed in SEEDS:
        print(f'Baseline486 seed{seed}: RMSE={core.metrics(data.y, refs[seed]["current486"])["rmse"]:.6f}')


def validate(folder):
    if not (folder/'protocol.json').is_file():
        raise ValueError('Run prepare first.')
    data, raw, refs, frozen = load_state()
    plan = t.read_json(folder/'protocol.json')
    if (plan['frozen'] != frozen or plan['competition_fits_during_preparation'] != 0
            or plan['competition_forwards_during_preparation'] != 0):
        raise ValueError('Frozen sources, features, runtime or baseline changed.')
    return data, raw, refs, plan


def fit_pca(train):
    """This API accepts neither query samples nor targets."""
    train = np.asarray(train, dtype=np.float64)
    if train.ndim != 2 or train.shape[1] != 864 or train.shape[0] < 65 or not np.isfinite(train).all():
        raise ValueError('PCA needs >=65 finite training rows and864 MicroNet columns')
    with threadpool_limits(limits=2):
        scaler = StandardScaler().fit(train)
        model = PCA(n_components=64, whiten=False, svd_solver='full').fit(scaler.transform(train))
    return {'scaler_mean': scaler.mean_, 'scaler_scale': scaler.scale_, 'scaler_var': scaler.var_,
        'pca_mean': model.mean_, 'components': model.components_,
        'explained_variance': model.explained_variance_, 'explained_variance_ratio': model.explained_variance_ratio_,
        'singular_values': model.singular_values_, 'n_samples': np.array([train.shape[0]], dtype=np.int64)}


def validate_pca(params, rows):
    shapes = {'scaler_mean': (864,), 'scaler_scale': (864,), 'scaler_var': (864,), 'pca_mean': (864,),
              'components': (64, 864), 'explained_variance': (64,), 'explained_variance_ratio': (64,),
              'singular_values': (64,), 'n_samples': (1,)}
    if set(params) != set(shapes) or any(params[k].shape != s or not np.isfinite(params[k]).all() for k, s in shapes.items()):
        raise ValueError('Invalid PCA parameter schema')
    if int(params['n_samples'][0]) != rows or np.any(params['scaler_scale'] <= 0):
        raise ValueError('Invalid training scope or scaler')


def project(x, params):
    with threadpool_limits(limits=2):
        value = ((np.asarray(x, dtype=np.float64) - params['scaler_mean']) / params['scaler_scale']
                 - params['pca_mean']) @ params['components'].T
    if value.shape != (len(x), 64) or not np.isfinite(value).all():
        raise ValueError('Invalid PCA projection')
    return value


def pca_cell(folder, binding, raw_micro, ti, vi, allow_fit):
    if np.intersect1d(ti, vi).size or len(ti) != 400 or len(vi) != 100 or raw_micro.shape != (500, 864):
        raise ValueError('Expected disjoint outer400/100 PCA inputs')
    expected = {'experiment': binding, 'train_idx': ti.tolist(), 'query_idx': vi.tolist(),
        'train_x_sha256': t.common.sha256_array(raw_micro[ti]), 'query_x_sha256': t.common.sha256_array(raw_micro[vi]),
        'settings': PROTOCOL['pca']}
    path, meta_path = folder/'parameters.npz', folder/'completed.json'
    if meta_path.exists():
        meta = t.read_json(meta_path)
        if meta['binding'] != expected or meta['parameters_sha256'] != t.common.sha256_file(path):
            raise ValueError('PCA cache binding or parameters changed')
        with np.load(path, allow_pickle=False) as z:
            params = {n: z[n].copy() for n in z.files}
    else:
        if not allow_fit:
            forbidden()
        params = fit_pca(raw_micro[ti])
        validate_pca(params, 400)
        folder.mkdir(parents=True, exist_ok=True)
        with path.with_suffix('.npz.tmp').open('wb') as h:
            np.savez_compressed(h, **params)
        os.replace(path.with_suffix('.npz.tmp'), path)
        t.write_json(meta_path, {'binding': expected, 'parameters_sha256': t.common.sha256_file(path),
            'labels_used': False, 'fit_rows': 400, 'query_rows': 100,
            'explained_variance_ratio_sum': float(params['explained_variance_ratio'].sum())})
    validate_pca(params, 400)
    return project(raw_micro, params), float(params['explained_variance_ratio'].sum())


def evaluate_fold(folder, binding, raw, y, folds, seed, fold, predictor, allow_fit):
    ti, vi = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
    if (len(ti), len(vi)) != (400, 100) or raw['fixed386'].shape != (500, 386):
        raise ValueError('Expected500 fixed386 rows and outer400/100')
    if not np.isfinite(raw['fixed386']).all():
        raise ValueError('Nonfinite fixed386 features')
    projected, variance = pca_cell(folder/f'fold{fold}_pca', binding, raw['micronet864'], ti, vi, allow_fit)
    matrices = {'tabpfn386': raw['fixed386'], 'tabpfn450': np.c_[raw['fixed386'], projected]}
    predictions = {n: t.run_cell(folder/f'fold{fold}_{n}', binding, n, x, y[ti], ti, vi,
                                seed+1009*fold+9, predictor if allow_fit else forbidden) for n, x in matrices.items()}
    return vi, predictions, variance


def result_data(data, ref, predictions, common_ids):
    output = {'tabpfn486': ref['tabpfn486'], **predictions, 'current486': ref['current486'],
              CONTROL: .5*(ref['base']+predictions['tabpfn386']),
              PRIMARY: .5*(ref['base']+predictions['tabpfn450'])}
    if set(predictions) != set(REPS) or any(p.shape != data.y.shape or not np.isfinite(p).all() for p in output.values()):
        raise ValueError('Invalid OOF predictions')
    np.testing.assert_array_equal(output['current486'], .5*(ref['base']+output['tabpfn486']))
    baseline = core.metrics(data.y, output['current486'])['rmse']
    control = core.metrics(data.y, output[CONTROL])['rmse']
    metrics = {n: {**core.metrics(data.y, p), 'gain_vs_current486': baseline-core.metrics(data.y, p)['rmse'],
                  'gain_vs_control386': control-core.metrics(data.y, p)['rmse']} for n, p in output.items()}
    target, noise = core.rank_groups(data.y, 10), core.rank_groups(data.noise, 4)
    common = np.isin(data.ids, common_ids)
    if common.sum() != 16:
        raise ValueError('Expected historical common16')
    masks = {'low10': target==0, 'high10': target==9, 'middle80': (target>0)&(target<9),
             'tails20': (target==0)|(target==9), 'common16': common, 'remaining484': ~common}
    masks.update({f'noiseQ{k+1}': noise==k for k in range(4)})
    masks.update({f'fold{f}': ref['folds']==f for f in range(5)})
    worst = {n: set(np.argsort((p-data.y)**2, kind='stable')[-20:].tolist()) for n, p in output.items()}
    for n in ('current486', CONTROL, PRIMARY):
        masks[n+'_worst20'] = np.isin(np.arange(len(data.ids)), list(worst[n]))
    diagnostic = [{'model': n, 'group': g, **core.metrics(data.y[m], p[m])}
                  for n, p in output.items() for g, m in masks.items() if m.any()]
    rows = [{'ID': str(i), 'hardness': float(data.y[j]), 'fold': int(ref['folds'][j]),
             **{n: float(p[j]) for n, p in output.items()}} for j, i in enumerate(data.ids)]
    errors = [{'ID': str(data.ids[j]), 'hardness': float(data.y[j]), 'fold': int(ref['folds'][j]),
        'common16': bool(common[j]), 'noise': float(data.noise[j]),
        **{n+'_error': float(p[j]-data.y[j]) for n, p in output.items()},
        'gain_squared_error_vs_current486': float((output['current486'][j]-data.y[j])**2-(output[PRIMARY][j]-data.y[j])**2),
        'gain_squared_error_vs_control386': float((output[CONTROL][j]-data.y[j])**2-(output[PRIMARY][j]-data.y[j])**2)}
        for j in np.argsort(-(output[PRIMARY]-data.y)**2, kind='stable')]
    err, old = output[PRIMARY]-data.y, output['current486']-data.y
    detail = {'common16_rmse': {n: core.metrics(data.y[common], p[common])['rmse'] for n, p in output.items()},
        'remaining484_rmse': {n: core.metrics(data.y[~common], p[~common])['rmse'] for n, p in output.items()},
        'common16_improved_vs_current486': int(((err**2<old**2)&common).sum()),
        'common16_same_error_sign': int((np.sign(err[common])==np.sign(old[common])).sum()),
        'common16_in_candidate_worst20': int(common[list(worst[PRIMARY])].sum()),
        'common16_sse_fraction': float(np.sum(err[common]**2)/np.sum(err**2)) if np.any(err) else 0.,
        'new_worst20_ids': [str(data.ids[i]) for i in sorted(worst[PRIMARY]-worst['current486'])],
        'left_worst20_ids': [str(data.ids[i]) for i in sorted(worst['current486']-worst[PRIMARY])],
        'residual_correlation_vs_current486': core.correlation(err, old),
        'standalone_gain_vs486': metrics['tabpfn486']['rmse']-metrics['tabpfn450']['rmse'],
        'standalone_gain_vs386': metrics['tabpfn386']['rmse']-metrics['tabpfn450']['rmse'],
        'folds_improved_vs_current486': sum(core.metrics(data.y[masks[f'fold{f}']], output[PRIMARY][masks[f'fold{f}']])['rmse']
            < core.metrics(data.y[masks[f'fold{f}']], output['current486'][masks[f'fold{f}']])['rmse'] for f in range(5))}
    return metrics, rows, diagnostic, errors, detail


def evaluation_binding(plan, ref, seed):
    return {'plan': t.digest(plan['frozen']), 'seed': seed, 'runtime': ref['runtime']}


def expected_outputs():
    return {'summary.json', 'oof.csv', 'diagnostics.csv', 'per_image_errors.csv', 'common16.csv', 'report.txt'} | {
        f'fold{f}_{n}/completed.json' for f in range(5) for n in REPS} | {
        f'fold{f}_pca/{n}' for f in range(5) for n in ('parameters.npz', 'completed.json')}


def completed(folder, binding):
    if not (folder/'completed_evaluation.json').exists():
        return False
    if set(t.read_json(folder/'completed_evaluation.json')['files']) != expected_outputs():
        raise ValueError('Incomplete result manifest')
    return t.verify_completed(folder, binding)


def collect(folder, data, raw, ref, plan, seed, predictor, allow_fit):
    pred = {n: np.full(500, np.nan) for n in REPS}
    variance = []
    binding = evaluation_binding(plan, ref, seed)
    for fold in range(5):
        vi, p, v = evaluate_fold(folder, binding, raw, data.y, ref['folds'], seed, fold, predictor, allow_fit)
        for n in REPS:
            pred[n][vi] = p[n]
        variance.append(v)
        if allow_fit:
            print(f'Outer{fold}:386/450 predictions saved; PCA64 training variance retained={v:.4f}; no outer score used', flush=True)
    return pred, variance


def save_results(folder, data, ref, pred, variance, plan, seed):
    binding = evaluation_binding(plan, ref, seed)
    metrics, rows, diag, errors, detail = result_data(data, ref, pred, plan['frozen']['common_failure_ids'])
    for name, records in (('oof.csv', rows), ('diagnostics.csv', diag), ('per_image_errors.csv', errors),
                          ('common16.csv', [r for r in errors if r['common16']])):
        t.write_csv(folder/name, records)
    t.write_json(folder/'summary.json', {'seed': seed, 'binding': binding, 'metrics': metrics, 'diagnosis': detail,
        'pca64_variance_retained_by_fold': variance, 'primary': PRIMARY, 'new_tabpfn_context_fits': 10,
        'new_fold_local_pca_fits': 5, 'archived486_context_fits_reused': 5, 'inner_search': False,
        'independent_holdout': False, 'test_accessed': False, 'attribution': 'Built with PriorLabs-TabPFN'})
    lines = [f'MicroNet PCA64 + TabPFN-v2: seed{seed}', 'Built with PriorLabs-TabPFN.', '',
             'Model / RMSE / gain vs current486 blend / gain vs control386 blend']
    lines += [f'{n}: {m["rmse"]:.6f} / {m["gain_vs_current486"]:+.6f} / {m["gain_vs_control386"]:+.6f}' for n, m in metrics.items()]
    lines += ['', 'Standalone gains (450 vs486 / vs386): '
              f'{detail["standalone_gain_vs486"]:+.6f} / {detail["standalone_gain_vs386"]:+.6f}',
              'Blends=.5*(hand185/ResNet/DINO base)+.5*TabPFN variant. Only TabPFN50% changes.',
              '386=hand185+scattering201;450 adds MicroNet-global864 scaled/PCA64;486 retains morphology100.',
              'PCA fitted on training400 only, independently for each fold; fixed64, no whitening/search.',
              'Existing486 contexts verified/reused. Same8 estimators, FP32, fit seed and runtime.',
              'Review common16.csv, diagnostics.csv and per_image_errors.csv; common16 is diagnostic, not an adoption gate.',
              'Mean gain>=0.30 is expansion priority only. Same500, no independent inference or automatic adoption.',
              'No full500 supervised fit, test access, or submission creation.']
    (folder/'report.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    t.write_json(folder/'completed_evaluation.json', {'binding': binding,
        'files': {n: t.common.sha256_file(folder/n) for n in sorted(expected_outputs())}})
    for n in ('tabpfn386', 'tabpfn450', 'current486', CONTROL, PRIMARY):
        m = metrics[n]
        print(f'{n}: RMSE={m["rmse"]:.6f}; gain vs current486={m["gain_vs_current486"]:+.6f}; '
              f'gain vs control386={m["gain_vs_control386"]:+.6f}', flush=True)
    print(f'Report: {folder/"report.txt"}')


def verify_results(folder, data, raw, ref, plan, seed):
    if not completed(folder, evaluation_binding(plan, ref, seed)):
        raise ValueError(f'Incomplete seed{seed}; run evaluate first')
    with contextlib.redirect_stdout(io.StringIO()):
        pred, variance = collect(folder, data, raw, ref, plan, seed, forbidden, False)
    metrics, rows, _, _, detail = result_data(data, ref, pred, plan['frozen']['common_failure_ids'])
    summary = t.read_json(folder/'summary.json')
    if summary['metrics'] != metrics or summary['diagnosis'] != detail or summary['pca64_variance_retained_by_fold'] != variance:
        raise ValueError('Recomputed metrics/PCA diagnostics differ')
    saved = core.align_rows(core.read_rows(folder/'oof.csv', set(rows[0])), data.ids, folder)
    for n in rows[0]:
        if n != 'ID':
            np.testing.assert_array_equal([float(r[n]) for r in saved], [float(r[n]) for r in rows])
    return metrics, detail


def evaluate(folder, device, seed):
    data, raw, refs, plan = validate(folder)
    ref = refs[seed]
    sub = folder/f'seed{seed}'
    sub.mkdir(parents=True, exist_ok=True)
    if completed(sub, evaluation_binding(plan, ref, seed)):
        verify_results(sub, data, raw, ref, plan, seed)
        print(f'Completed seed{seed} verified; no PCA refitting or model inference.')
        return
    predictor = t.Predictor(t.verify_weights(t.OUT), device)
    if predictor.runtime != ref['runtime']:
        raise ValueError(f'Use exact archived runtime; no fallback: {ref["runtime"]}')
    pred, variance = collect(sub, data, raw, ref, plan, seed, predictor, True)
    validate(folder)
    save_results(sub, data, ref, pred, variance, plan, seed)


def aggregate(records):
    if [r['seed'] for r in records] != list(SEEDS):
        raise ValueError('Both fixed seeds required in order')
    mean = float(np.mean([r['gain_vs_current486'] for r in records]))
    positive = all(r['gain_vs_current486'] > 0 and r['gain_vs_control386'] > 0 for r in records)
    return {'primary': PRIMARY, 'per_seed': records, 'mean_gain_vs_current486': mean,
        'mean_gain_vs_control386': float(np.mean([r['gain_vs_control386'] for r in records])),
        'all_seeds_improve_both': positive, 'passes_expansion_priority': positive and mean >= .30,
        'additional_review_required': 'Overall performance, standalone, common16/remaining484, tails, noise and new worst20. '
                                      'Common16 is diagnostic, not a mandatory adoption gate; no automatic adoption.',
        'aggregation': 'Mean per-seed RMSE differences; same500, no prediction averaging or independent inference.'}


def compare(folder):
    data, raw, refs, plan = validate(folder)
    records = []
    for seed in SEEDS:
        metrics, detail = verify_results(folder/f'seed{seed}', data, raw, refs[seed], plan, seed)
        m = metrics[PRIMARY]
        records.append({'seed': seed, 'current486_rmse': metrics['current486']['rmse'],
            'control386_rmse': metrics[CONTROL]['rmse'], 'candidate450_rmse': m['rmse'],
            'gain_vs_current486': m['gain_vs_current486'], 'gain_vs_control386': m['gain_vs_control386'], 'diagnosis': detail})
    result = {**aggregate(records), 'protocol_sha256': t.digest(plan['frozen'])}
    t.write_json(folder/'comparison.json', result)
    print(t.common.canonical_json(result), flush=True)
    return result


def check(folder, device):
    """Synthetic only; real TabPFN smoke test is user-run."""
    predictor = t.Predictor(t.verify_weights(t.OUT), device)
    binding = {'protocol': PROTOCOL, 'runtime': predictor.runtime, 'seed': 314159,
               'weights': t.file_ref(t.verify_weights(t.OUT))}
    path = folder/'synthetic_check.json'
    if path.exists():
        saved = t.read_json(path)
        if saved['binding'] != binding or saved['result_sha256'] != t.digest(saved['result']):
            raise ValueError('Synthetic check binding/result changed')
        print('Synthetic check verified; no repeated model inference.')
        return
    rng = np.random.default_rng(314159)
    fixed, raw_micro = rng.normal(size=(500, 386)), rng.normal(size=(500, 864))
    fixed[:, 0] = 0.
    params = fit_pca(raw_micro[:400])
    validate_pca(params, 400)
    projected = project(raw_micro, params)
    y = 100 + 3*fixed[:, 2] - 2*fixed[:, 3] + rng.normal(size=500)
    result = {'competition_data_used': False, 'gradient_updates': 0, 'models': {}}
    for name, x in (('tabpfn386', fixed), ('tabpfn450', np.c_[fixed, projected])):
        pred, meta = predictor(x[:400], y[:400], x[400:], 314159)
        result['models'][name] = {'shape': list(pred.shape), 'finite': bool(np.isfinite(pred).all()), 'metadata': meta}
        print(f'Synthetic400/100 {name} passed; {meta["seconds"]:.1f}s', flush=True)
    t.write_json(path, {'binding': binding, 'result': result, 'result_sha256': t.digest(result)})


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('prepare', 'validate', 'check', 'evaluate', 'compare'))
    p.add_argument('--output-dir', type=t.common.project_path, default=OUT)
    p.add_argument('--seed', type=int, choices=SEEDS, default=42)
    p.add_argument('--device', default='cuda')
    a = p.parse_args(argv)
    if a.command == 'prepare': prepare(a.output_dir)
    elif a.command == 'validate': validate(a.output_dir); print('Inputs/baseline/protocol verified; no fitting.')
    elif a.command == 'check': check(a.output_dir, a.device)
    elif a.command == 'evaluate': evaluate(a.output_dir, a.device, a.seed)
    else: compare(a.output_dir)


if __name__ == '__main__':
    main()
