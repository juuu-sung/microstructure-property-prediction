"""Audit saved scattering cells and OOF predictions; never extract or fit."""
import contextlib
import csv
import io
from pathlib import Path
import numpy as np
if __package__:
    from . import evaluate_scattering_tabpfn as e
else:
    import evaluate_scattering_tabpfn as e


def same_rows(path, expected):
    with path.open(encoding='utf-8', newline='') as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != len(expected):
        raise ValueError(f'Row count differs: {path}')
    for actual, wanted in zip(rows, expected, strict=True):
        if set(actual) != set(wanted):
            raise ValueError(f'Columns differ: {path}')
        for key, value in wanted.items():
            if isinstance(value, (str, bool)):
                if actual[key] != str(value):
                    raise ValueError(f'String/bool differs: {path} {key}')
            else:
                np.testing.assert_allclose(float(actual[key]), value, rtol=0, atol=1e-9)


def main():
    data, base, refs, plan = e.validate(e.OUT)
    added, meta = e.load_features(e.OUT, data, plan)
    x = np.c_[base, added]
    protected = {}
    def protect(path):
        protected[str(path)] = e.t.common.sha256_file(path)
    for name in ('protocol.json', 'comparison.json', 'features/metadata.json', 'features/features.npz'):
        protect(e.OUT/name)
    for item in plan['frozen']['sources'].values():
        protect(Path(item['path']))
    for item in plan['frozen']['submissions']:
        if e.t.file_ref(item['path']) != item:
            raise ValueError('Archived submission changed')
        protect(Path(item['path']))
    s1 = added[:, 1:41].reshape(500, 5, 8).mean(axis=2)
    denominators = np.stack([s1[:, a] + e.sf.EPS for a, _ in e.sf.PAIRS], axis=1)
    normalized = added[:, 41:121].reshape(500, 10, 8) / denominators[:, :, None]
    np.testing.assert_array_equal(normalized.reshape(500, 80), added[:, 121:])
    for j, image_id in enumerate(data.ids):
        path = e.OUT/'features/cells'/f'{image_id}.json'
        protect(path)
        detail = e.t.read_json(path)['result']['detail']
        if detail['path_count'] != 681 or detail['map_shape'] != [8, 8]:
            raise ValueError('Saved scattering path metadata differs')
        np.testing.assert_array_equal(detail['s1_scale_means'], s1[j])
        if detail['low_energy_scale_count'] != int((s1[j] <= e.sf.EPS).sum()):
            raise ValueError('Low-energy count differs')
        if detail['max_abs_normalized_s2'] != float(np.abs(added[j, 121:]).max()):
            raise ValueError('Normalized range differs')
    def forbidden(*args):
        raise ValueError('Missing completed cell; review never refits')
    records, details, timing = [], [], []
    common = np.isin(data.ids, plan['frozen']['common_failure_ids'])
    for seed in e.SEEDS:
        folder = e.OUT/f'seed{seed}'
        binding = e.evaluation_binding(e.OUT, plan, refs[seed], seed)
        if not e.completed(folder, binding):
            raise ValueError('Incomplete or changed evaluation')
        protect(folder/'completed_evaluation.json')
        for name in e.expected_outputs():
            protect(folder/name)
        pred = np.full(500, np.nan)
        with contextlib.redirect_stdout(io.StringIO()):
            for fold in range(5):
                vi, p = e.evaluate_fold(folder, binding, x, data.y, refs[seed]['folds'], seed, fold, forbidden)
                pred[vi] = p
                ti = np.flatnonzero(refs[seed]['folds'] != fold)
                cell = e.t.read_json(folder/f'fold{fold}_candidate/completed.json')['result']['metadata']
                old = e.t.read_json(e.t.OUT/f'seed{seed}/fold{fold}_morphology285_outer/completed.json')['result']['metadata']
                if cell['runtime'] != refs[seed]['runtime'] or cell['resolved_config'] != old['resolved_config']:
                    raise ValueError('Runtime or regressor configuration differs from control')
                if (cell['context_rows'], cell['query_rows'], cell['columns']) != (400, 100, 486):
                    raise ValueError('Context/query dimensions differ')
                np.testing.assert_allclose(cell['context_mean'], data.y[ti].mean(), rtol=0, atol=1e-9)
                np.testing.assert_allclose(cell['context_std'], data.y[ti].std(), rtol=0, atol=1e-9)
                timing.append({'seed': seed, 'fold': fold, 'seconds': cell['seconds'],
                               'peak_gpu_allocated_bytes': cell['peak_gpu_allocated_bytes']})
        metrics, rows, diag, errors = e.result_data(data, refs[seed], pred, plan['frozen']['common_failure_ids'])
        if metrics != e.t.read_json(folder/'summary.json')['metrics']:
            raise ValueError('Summary differs')
        for name, expected in (('oof.csv', rows), ('diagnostics.csv', diag), ('per_image_errors.csv', errors)):
            same_rows(folder/name, expected)
        m = metrics[e.PRIMARY]
        records.append({'seed': seed, 'current_rmse': metrics['current']['rmse'],
            'fixed285_control_rmse': metrics['fixed285_control_blend']['rmse'], 'candidate_rmse': m['rmse'],
            'gain_vs_current': m['gain_vs_current'], 'gain_vs_control': m['gain_vs_control']})
        lookup = {(r['model'], r['group']): r for r in diag}
        groups = []
        for group in sorted({r['group'] for r in diag}):
            a, b, c = [lookup[(n, group)] for n in ('current', 'fixed285_control_blend', e.PRIMARY)]
            groups.append({'group': group, 'n': c['n'], 'current_rmse': a['rmse'], 'control_rmse': b['rmse'],
                'candidate_rmse': c['rmse'], 'gain_vs_current': a['rmse']-c['rmse'],
                'gain_vs_control': b['rmse']-c['rmse'], 'sse_gain_vs_current': a['sse']-c['sse'],
                'sse_gain_vs_control': b['sse']-c['sse']})
        candidate = .5*(refs[seed]['base']+pred)
        err = candidate-data.y
        current_err = refs[seed]['current']-data.y
        control_err = refs[seed]['control']-data.y
        worst = lambda arr: set(np.argsort(arr**2, kind='stable')[-20:])
        cw, nw = worst(current_err), worst(err)
        detail = {'seed': seed, 'metrics': metrics,
            'standalone_gain_vs_285': metrics['tabpfn285_control']['rmse']-metrics['tabpfn486']['rmse'],
            'prediction_rms_change_vs_control': float(np.sqrt(np.mean((candidate-refs[seed]['control'])**2))),
            'residual_correlation_with_control': float(np.corrcoef(err, control_err)[0, 1]),
            'folds_improved_vs_current': sum(g['gain_vs_current'] > 0 for g in groups if g['group'].startswith('fold')),
            'folds_improved_vs_control': sum(g['gain_vs_control'] > 0 for g in groups if g['group'].startswith('fold')),
            'images_improved_vs_current': sum(r['gain_squared_error_vs_current'] > 0 for r in errors),
            'images_improved_vs_control': sum(r['gain_squared_error_vs_control'] > 0 for r in errors),
            'common16_improved_vs_current': sum(r['common16'] and r['gain_squared_error_vs_current'] > 0 for r in errors),
            'common16_improved_vs_control': sum(r['common16'] and r['gain_squared_error_vs_control'] > 0 for r in errors),
            'common16_same_error_sign_as_current': int((np.sign(err[common]) == np.sign(current_err[common])).sum()),
            'common16_in_candidate_worst20': int(sum(common[list(nw)])),
            'common16_sse_fraction': float(np.sum(err[common]**2)/np.sum(err**2)),
            'worst20_overlap_with_current': len(cw & nw),
            'new_worst20_ids': [str(data.ids[j]) for j in sorted(nw-cw)],
            'left_worst20_ids': [str(data.ids[j]) for j in sorted(cw-nw)],
            'most_hurt_vs_control': sorted(errors, key=lambda r:r['gain_squared_error_vs_control'])[:5],
            'most_helped_vs_control': sorted(errors, key=lambda r:-r['gain_squared_error_vs_control'])[:5],
            'groups': groups}
        details.append(detail)
    comparison = {**e.aggregate(records), 'protocol_sha256': e.t.digest(plan['frozen'])}
    if comparison != e.t.read_json(e.OUT/'comparison.json'):
        raise ValueError('Comparison differs')
    for path, sha in protected.items():
        if e.t.common.sha256_file(path) != sha:
            raise ValueError('Protected input/output changed during review')
    result = {'scope': 'CPU saved cells/arrays only; no extraction, fits, inference, test images or submission generation',
        'decision': 'preserve_modest_candidate; no_expansion; common16_not_resolved; keep_current_submission',
        'comparison': comparison, 'details': details, 'timing': timing,
        'feature_checks': {'shape': list(added.shape), 'combined_shape': list(x.shape),
            'zero_variance_columns': int((np.ptp(added, axis=0) == 0).sum()),
            'low_energy_scale_count': int((s1 <= e.sf.EPS).sum()),
            'normalized_s2_range': [float(added[:,121:].min()), float(added[:,121:].max())],
            'feature_extraction_seconds': meta['seconds']},
        'checks': '500 cell bindings/vectors/path metadata/normalization;10 outer bindings/train-only contexts/exact archived runtime/config;1000 OOF rows/all group-error arithmetic;50:50;unchanged source/protocol/cache/submissions',
        'sources': {str(Path(__file__).resolve()): e.t.common.sha256_file(__file__)},
        'protected_files': protected}
    e.t.write_json(e.OUT/'review_results.json', result)
    print(e.t.common.canonical_json({k: v for k, v in result.items() if k not in ('protected_files','details','sources','timing')}))
    for d in details:
        print(e.t.common.canonical_json({k: v for k, v in d.items() if k not in ('metrics','groups','most_hurt_vs_control','most_helped_vs_control')}))


if __name__ == '__main__':
    main()
