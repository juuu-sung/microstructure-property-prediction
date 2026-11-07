"""Recompute archived MicroNet metrics without fitting, extracting or changing results."""
from __future__ import annotations

import csv
from pathlib import Path
import numpy as np

if __package__:
    from . import efficientnet_micronet_experiment as e
else:
    import efficientnet_micronet_experiment as e


def same_rows(path, expected):
    with path.open(encoding='utf8', newline='') as handle:
        actual = list(csv.DictReader(handle))
    if len(actual) != len(expected):
        raise ValueError(f'Row count differs: {path}')
    for got, wanted in zip(actual, expected, strict=True):
        if set(got) != set(wanted):
            raise ValueError(f'Columns differ: {path}')
        for key, value in wanted.items():
            if isinstance(value, (str, bool)):
                if got[key] != str(value):
                    raise ValueError(f'Value differs: {path} {key}')
            else:
                np.testing.assert_allclose(float(got[key]), value, rtol=0, atol=1e-9)


def main():
    data, refs, plan = e.validate(e.OUT)
    features, meta = e.load_features(e.OUT, data, plan)
    protected = {}

    def protect(path):
        path = Path(path)
        protected[str(path)] = e.base.sha256_file(path)

    for name in ('protocol.json', 'comparison.json', 'features/features.npz', 'features/metadata.json'):
        protect(e.OUT / name)
    for entry in plan['frozen']['sources'].values():
        protect(entry['path'])
    for entry in plan['frozen']['submissions']:
        if e.t.file_ref(entry['path']) != entry:
            raise ValueError('Archived submission differs')
        protect(entry['path'])

    common_ids = plan['frozen']['common_failure_ids']
    common = np.isin(data.ids, common_ids)
    if len(data.ids) != 500 or common.sum() != 16:
        raise ValueError('Unexpected image inventory or common16')
    records, details, all_groups, common_rows, error_by_seed = [], [], [], [], {}
    for seed in e.SEEDS:
        folder = e.OUT / f'seed{seed}'
        binding = e.evaluation_binding(e.OUT, seed, plan)
        if not e.completed(folder, binding):
            raise ValueError(f'Missing completed seed{seed}; review cannot refit')
        for name in e.expected_outputs() | {'completed.json'}:
            protect(folder / name)
        predictions = {name: np.full(500, np.nan) for name in e.REPS}
        selection, seen = [], np.zeros(500, dtype=int)
        for fold in range(5):
            cell = e.fold_cell(folder, binding, features, data.y, refs[seed]['folds'], seed, fold,
                               runner=e.unavailable)
            vi = cell['validation_indices']
            seen[vi] += 1
            for name in e.REPS:
                predictions[name][vi] = cell['predictions'][name]
            selection.append({'fold': fold, 'choices': cell['selection']})
        np.testing.assert_array_equal(seen, np.ones(500, dtype=int))
        metrics, oof, diag, errors, diagnosis = e.result_data(data, refs[seed], predictions, common_ids)
        summary = e.t.read_json(folder / 'summary.json')
        expected = {'binding': binding, 'metrics': metrics, 'selection': selection,
                    'diagnosis': diagnosis, 'primary': e.PRIMARY, 'ridge_fits': 130,
                    'neural_updates': 0, 'test_accessed': False, 'independent_holdout': False}
        if summary != expected:
            raise ValueError('Summary differs from archived fold arithmetic')
        for name, rows in (('oof.csv', oof), ('diagnostics.csv', diag), ('per_image_errors.csv', errors),
                           ('common16.csv', [r for r in errors if r['common16']])):
            same_rows(folder / name, rows)
        records.append({'seed': seed, 'current_rmse': metrics['current']['rmse'],
                        'control_rmse': metrics[e.CONTROL]['rmse'], 'candidate_rmse': metrics[e.PRIMARY]['rmse'],
                        'gain_vs_current': metrics[e.PRIMARY]['gain_vs_current'],
                        'gain_vs_control': metrics[e.PRIMARY]['gain_vs_control'], 'diagnosis': diagnosis})
        lookup = {(r['model'], r['group']): r for r in diag}
        groups = {}
        for group in sorted({r['group'] for r in diag}):
            a, b, c = [lookup[(n, group)] for n in ('current', e.CONTROL, e.PRIMARY)]
            row = {'seed': seed, 'group': group, 'n': c['n'], 'current_rmse': a['rmse'],
                   'imagenet_blend_rmse': b['rmse'], 'candidate_rmse': c['rmse'],
                   'gain_vs_current': a['rmse'] - c['rmse'], 'gain_vs_control': b['rmse'] - c['rmse'],
                   'sse_gain_vs_current': a['sse'] - c['sse'], 'sse_gain_vs_control': b['sse'] - c['sse']}
            groups[group] = row
            all_groups.append(row)
        total, benefit, rest = [groups[g]['sse_gain_vs_current'] for g in ('all', 'common16', 'remaining484')]
        np.testing.assert_allclose(total, benefit + rest, rtol=0, atol=1e-8)
        by_id = {r['ID']: r for r in errors}
        error_by_seed[seed] = by_id
        for image_id in sorted(common_ids):
            row = by_id[image_id]
            common_rows.append({'seed': seed, 'ID': image_id, 'hardness': row['hardness'],
                                'current_error': row['current_error'],
                                'micronet_error': row['micronet_error'],
                                'candidate_error': row[e.PRIMARY + '_error'],
                                'candidate_worst20': row[e.PRIMARY + '_worst20'],
                                'sse_gain_vs_current': row['sse_gain_vs_current']})
        details.append({'seed': seed, 'metrics': metrics, 'diagnosis': diagnosis, 'selection': selection,
                        'sse_budget': {'total_gain': total, 'common16_gain': benefit, 'remaining484_gain': rest,
                                       'common16_benefit_offset_by_rest_fraction': -rest / benefit},
                        'images_improved': sum(r['sse_gain_vs_current'] > 0 for r in errors),
                        'standalone_residual_correlation_vs_current': e.core.correlation(
                            predictions['micronet'] - data.y, refs[seed]['current'] - data.y),
                        'common16_current_sse_fraction': lookup['current', 'common16']['sse'] / metrics['current']['sse'],
                        'common16_left_worst20': [i for i in diagnosis['left_worst20_ids'] if i in common_ids],
                        'new_worst20': [by_id[i] for i in diagnosis['new_worst20_ids']],
                        'most_helped': sorted(errors, key=lambda r: -r['sse_gain_vs_current'])[:5],
                        'most_hurt': sorted(errors, key=lambda r: r['sse_gain_vs_current'])[:5]})

    comparison = {**e.aggregate(records), 'protocol_sha256': e.t.digest(plan['frozen'])}
    if comparison != e.t.read_json(e.OUT / 'comparison.json'):
        raise ValueError('Comparison differs')
    both_better, both_worse, mixed = [], [], []
    for image_id in sorted(common_ids):
        gains = [error_by_seed[s][image_id]['sse_gain_vs_current'] for s in e.SEEDS]
        (both_better if all(g > 0 for g in gains) else both_worse if all(g < 0 for g in gains) else mixed).append(image_id)
    for path, sha in protected.items():
        if e.base.sha256_file(path) != sha:
            raise ValueError('Protected artifact changed during audit')
    result = {'scope': 'Read saved arrays/cells only; no extraction, fitting, neural inference, test images or submission writes',
              'verified': True, 'comparison': comparison, 'details': details,
              'common16_across_seeds': {'improved_both_ids': both_better, 'worsened_both_ids': both_worse,
                                        'mixed_ids': mixed},
              'feature_checks': {'shape': {n: list(x.shape) for n, x in features.items()},
                                 'constant_columns': {n: int((np.ptp(x, axis=0) == 0).sum()) for n, x in features.items()},
                                 'seconds': meta['seconds'], 'batch_size': meta['batch_size'],
                                 'runtime': meta['runtime'], 'neural_updates': meta['neural_updates']},
              'checks': ['Protocol/sources/environment/assets/train inventory/current references unchanged',
                         'Both 500x864 caches and completion manifests verified',
                         '10 fold cells: disjoint train400/validation100, 1 OOF prediction/image/seed',
                         '20 internal alpha choices match stored pooled inner minima',
                         'OOF/summary/all subgroup and per-image tables recomputed',
                         'Both fixed 50:50 blends and common16+remaining484 SSE accounting verified',
                         'Original experiment results and all three submissions unchanged'],
              'decision': 'Keep current Scattering486 submission; archive MicroNet as domain-transfer evidence; '
                          'no expansion, automatic fine-tuning, ratio search or further seed confirmation from this result',
              'limits': 'Same repeatedly studied500; descriptive subgroup analysis, no independent holdout or significance claim; '
                        'frozen mean/std+Ridge does not exhaust all MicroNet adaptation options',
              'protected_files': protected, 'audit_script_sha256': e.base.sha256_file(__file__)}
    out = e.OUT / 'review'
    e.t.write_json(out / 'review_results.json', result)
    e.t.write_csv(out / 'group_comparison.csv', all_groups)
    e.t.write_csv(out / 'common16_changes.csv', common_rows)
    print(e.base.canonical_json({'verified': True, 'decision': result['decision'],
                               'mean_gain': comparison['mean_gain_vs_current'],
                               'common16_across_seeds': result['common16_across_seeds'],
                               'details': [{k: d[k] for k in ('seed', 'sse_budget', 'images_improved',
                                           'standalone_residual_correlation_vs_current', 'common16_left_worst20')}
                                           for d in details], 'output': str(out)}))


if __name__ == '__main__':
    main()
