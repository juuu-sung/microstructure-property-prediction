"""User-run, CPU-only confirmation of frozen Large518 + nested Ridge.

Reuses existing features and same-fold Scattering486 OOF. No image forwards,
TabPFN refits, test access, mixture search, or automatic submission adoption.
"""
from __future__ import annotations

import argparse
import contextlib
import io
from pathlib import Path
import time
import numpy as np
from threadpoolctl import threadpool_limits

if __package__:
    from . import dinov2_factorial_experiment as original
else:
    import dinov2_factorial_experiment as original

e = original
eng = e.eng
core = e.core
ROOT = e.ROOT
OUT = ROOT / 'reports/dinov2_large518_confirmation'
ARCHIVE = ROOT / 'reports/scattering_tabpfn_confirmation'
SEEDS = (31415, 27182, 16180)
RIDGE_INDICES = (0, 1, 2)  # Preserve original exact-tie preference: 10000,1000,100.
PRIMARY = 'current_large518_ridge_equal'
FOCUS = ('common16', 'remaining484', 'noiseQ4')
PROTOCOL = {
    'version': 1, 'seeds': list(SEEDS), 'primary': PRIMARY,
    'representation': 'Archived frozen DINOv2 Large518 CLS+patch mean+population std,3072; unchanged FP32 features.',
    'alpha_candidates': [100., 1000., 10000.],
    'alpha_selection': 'Original pooled inner MSE; exact ties prefer larger alpha. Representation is fixed.',
    'outer': 'Exact archived 5 outer folds per seed,400 train/100 query.',
    'inner': '3 target-rank-stratified folds within outer400; seed+outer_fold; identical original engine.',
    'preprocessing': 'X/y StandardScaler fit on inner train only; refit selected Ridge on outer400. No PCA/SVR.',
    'reference': 'Matching-seed scattering_replaced_blend from Scattering486 confirmation, not its current column.',
    'blend': '0.5 matching-seed Scattering486 OOF +0.5 Large518 Ridge OOF; no clipping or weight search.',
    'scope': 'User runs CPU Ridge fitting only. No encoder/TabPFN fitting, image extraction, test or submission.',
    'planned_ridge_fits': 150, 'cpu_threads': 2,
    'diagnostics': 'Standalone/blend, common16, remaining484, low10/high10/middle80, all noise quartiles, folds, new worst20.',
    'decision': 'Report overall and subgroup direction separately; any remaining484/noiseQ4 worsening stays visible. No automatic adoption.',
    'expansion_priority_mean_gain': .5,
    'aggregation': 'Mean of per-seed OOF RMSE differences; only these3 seeds; no prediction averaging.',
    'limits': 'Large518 was prioritized after42/2026 diagnostics. Repeatedly studied same500 and reused splits, not independent holdout.',
    'stop': 'Finish the fixed3 seeds once; no extra seeds, feature/alpha/weight search from outer results.',
    'attribution': 'DINOv2 (Meta), timm; preserved Scattering486 ensemble includes PriorLabs-TabPFN.'}


def forbidden(*args, **kwargs):
    raise RuntimeError('Reference/compare must never fit. Run evaluate to complete missing cells.')


def archived_reference(data, archive_plan, seed, archive=ARCHIVE):
    folder = archive / f'seed{seed}'
    done = e.read_json(folder / 'completed_evaluation.json')
    if done['binding']['plan'] != e.digest(archive_plan['frozen']) or done['binding']['seed'] != seed:
        raise ValueError('Archived Scattering486 completion scope differs')
    required = {'oof.csv', 'summary.json', 'diagnostics.csv', 'per_image_errors.csv', 'report.md'}
    if not required.issubset(done['files']):
        raise ValueError('Incomplete archived Scattering486 outputs')
    for name, sha in done['files'].items():
        path = (folder / name).resolve()
        if not path.is_relative_to(folder.resolve()) or e.base.sha256_file(path) != sha:
            raise ValueError('Archived Scattering486 output changed: ' + name)
    rows = core.align_rows(core.read_rows(folder / 'oof.csv', {'ID', 'hardness', 'fold', 'scattering_replaced_blend'}), data.ids, folder)
    np.testing.assert_array_equal(core.finite_column(rows, 'hardness', folder), data.y)
    folds = core.integer_column(rows, 'fold', folder)
    fold_path = archive / f'folds_seed{seed}.csv'
    fold_rows = core.align_rows(core.read_rows(fold_path, {'ID', 'fold'}), data.ids, fold_path)
    np.testing.assert_array_equal(folds, core.integer_column(fold_rows, 'fold', fold_path))
    if len(folds) != 500 or set(folds) != set(range(5)) or any(sum(folds == k) != 100 for k in range(5)):
        raise ValueError('Expected archived five400/100 splits')
    if e.base.sha256_array(folds) != archive_plan['frozen']['references'][str(seed)]['arrays']['folds']:
        raise ValueError('Archived fold fingerprint differs from its frozen protocol')
    current = core.finite_column(rows, 'scattering_replaced_blend', folder)
    summary = e.read_json(folder / 'summary.json')
    if summary['binding'] != done['binding'] or summary['primary'] != 'scattering_replaced_blend':
        raise ValueError('Archived summary scope differs')
    if abs(core.metrics(data.y, current)['rmse'] - summary['metrics']['scattering_replaced_blend']['rmse']) > 1e-10:
        raise ValueError('Archived current score differs')
    files = {n: e.file_ref(folder / n) for n in ('oof.csv', 'summary.json', 'completed_evaluation.json')}
    files['folds'] = e.file_ref(fold_path)
    return {'folds': folds, 'current': current}, files


def load_state():
    data, _, source_plan = e.validate(e.OUT)
    features, feature_refs = e.load_features(e.OUT, data, source_plan)
    archive_plan = e.read_json(ARCHIVE / 'protocol.json')
    common = source_plan['frozen']['common_failure_ids']
    if archive_plan['frozen']['common_failure_ids'] != common:
        raise ValueError('Historical common16 changed')
    if tuple(eng.CANDIDATES[i] for i in RIDGE_INDICES) != tuple({'family': 'ridge', 'alpha': a} for a in (10000., 1000., 100.)):
        raise ValueError('Original Ridge candidate order changed')
    refs, files = {}, {}
    for seed in SEEDS:
        refs[seed], files[str(seed)] = archived_reference(data, archive_plan, seed)
    frozen = {'protocol': PROTOCOL, 'common_failure_ids': common,
        'sources': {p.name: e.file_ref(p) for p in (Path(__file__), ROOT / 'scripts/run_dinov2_large518_confirmation.ps1')},
        'source_plan': e.file_ref(e.OUT / 'protocol.json'), 'feature': feature_refs['large518'],
        'archive_plan': e.file_ref(ARCHIVE / 'protocol.json'), 'references': files,
        'environment': e.environment(), 'ids_sha256': e.base.sha256_array(data.ids),
        'y_sha256': e.base.sha256_array(data.y), 'noise_sha256': e.base.sha256_array(data.noise),
        'preserved_submission': source_plan['frozen']['preserved_submission']}
    return data, features['large518'], refs, frozen


def prepare(folder=OUT):
    data, x, refs, frozen = load_state()
    path = folder / 'protocol.json'
    if path.exists():
        if e.read_json(path)['frozen'] != frozen:
            raise ValueError('Frozen confirmation changed; preserve existing results')
    else:
        if folder.exists() and any(folder.iterdir()):
            raise ValueError('Nonempty unprepared output folder')
        e.write_json(path, {'frozen': frozen, 'created_unix': time.time(), 'competition_fits': 0})
    print(f'Prepared/verified Large518 {x.shape}, alpha100/1000/10000,50:50; no fitting or extraction.')
    for seed in SEEDS:
        print(f'Scattering486 seed{seed}: RMSE={core.metrics(data.y, refs[seed]["current"])["rmse"]:.6f}')


def validate(folder=OUT):
    data, x, refs, frozen = load_state()
    plan = e.read_json(folder / 'protocol.json')
    if plan['frozen'] != frozen or plan['competition_fits'] != 0:
        raise ValueError('Frozen inputs/sources/environment changed; do not overwrite old results')
    return data, x, refs, plan


def fit_fold(folder, binding, train_x, train_y, query_x, seed, allow_fit=True, runner=None):
    """Outer-query labels/current predictions are intentionally not accepted."""
    runner = runner or eng.fit_candidates
    y = eng.finite(train_y)
    inner = {str(i): np.full(len(y), np.nan) for i in RIDGE_INDICES}
    for k, (ti, vi) in enumerate(eng.inner_splits(y, seed)):
        scope = {'parent': binding, 'inner': k, 'train': ti.tolist(), 'query': vi.tolist(),
                 'x': e.base.sha256_array(train_x[ti]), 'query_x': e.base.sha256_array(train_x[vi]),
                 'y': e.base.sha256_array(y[ti]), 'candidates': list(RIDGE_INDICES)}
        result = e.cell(folder / f'inner{k}.json', scope,
                        lambda: runner(train_x[ti], y[ti], train_x[vi], RIDGE_INDICES), allow_fit)
        if set(result['predictions']) != set(inner):
            raise ValueError('Ridge-only inner predictions differ')
        for key, p in result['predictions'].items():
            inner[key][vi] = eng.finite(p, (len(vi),))
    scores = {i: float(np.mean((eng.finite(inner[str(i)], y.shape) - y)**2)) for i in RIDGE_INDICES}
    best = min(RIDGE_INDICES, key=lambda i: (scores[i], i))
    scope = {'parent': binding, 'candidates': [best], 'x': e.base.sha256_array(train_x),
             'query_x': e.base.sha256_array(query_x), 'y': e.base.sha256_array(y)}
    result = e.cell(folder / 'refit.json', scope,
                    lambda: runner(train_x, y, query_x, [best]), allow_fit)
    if set(result['predictions']) != {str(best)}:
        raise ValueError('Ridge refit selection differs')
    choice = {'representation': 'large518', 'alpha': eng.CANDIDATES[best]['alpha'],
              'candidate_index': best, 'inner_rmse': float(np.sqrt(scores[best])),
              'inner_scores': [{'alpha': eng.CANDIDATES[i]['alpha'], 'mse': scores[i]} for i in RIDGE_INDICES]}
    return eng.finite(result['predictions'][str(best)], (len(query_x),)), choice


def binding_for(plan, seed):
    return {'plan': e.digest(plan['frozen']), 'seed': seed}


def collect(folder, data, x, ref, binding, seed, allow_fit=True, runner=None):
    ridge = np.full(len(data.y), np.nan)
    choices = []
    with threadpool_limits(limits=2):
        for fold in range(5):
            ti = np.flatnonzero(ref['folds'] != fold)
            vi = np.flatnonzero(ref['folds'] == fold)
            scope = {'evaluation': binding, 'outer_fold': fold, 'train_indices': ti.tolist(), 'query_indices': vi.tolist()}
            ridge[vi], choice = fit_fold(folder / f'fold{fold}', scope, x[ti], data.y[ti], x[vi], seed + fold, allow_fit, runner)
            choices.append({'fold': fold, **choice})
            print(f'seed{seed} outer{fold}: Large518 alpha={choice["alpha"]:g}; inner={choice["inner_rmse"]:.6f}; outer labels unused', flush=True)
    eng.finite(ridge, data.y.shape)
    return {'current': ref['current'], 'large518_ridge': ridge, PRIMARY: .5*ref['current'] + .5*ridge}, choices


def result_data(data, ref, predictions, selections, plan, seed):
    common = np.isin(data.ids, plan['frozen']['common_failure_ids'])
    target, noise = core.rank_groups(data.y, 10), core.rank_groups(data.noise, 4)
    masks = {'common16': common, 'remaining484': ~common, 'low10': target == 0,
             'high10': target == 9, 'middle80': (target > 0) & (target < 9)}
    masks.update({f'noiseQ{k+1}': noise == k for k in range(4)})
    masks.update({f'fold{k}': ref['folds'] == k for k in range(5)})
    metrics = {n: core.metrics(data.y, p) for n, p in predictions.items()}
    for m in metrics.values():
        m['gain_vs_current'] = metrics['current']['rmse'] - m['rmse']
    groups = []
    details = {}
    old = ref['current'] - data.y
    worst = {n: set(np.argsort((p-data.y)**2, kind='stable')[-20:]) for n, p in predictions.items()}
    for name, p in predictions.items():
        err = p - data.y
        group_metrics = {g: core.metrics(data.y[m], p[m]) for g, m in masks.items()}
        for g, m in masks.items():
            group_metrics[g]['gain_vs_current'] = core.metrics(data.y[m], ref['current'][m])['rmse'] - group_metrics[g]['rmse']
            groups.append({'model': name, 'group': g, **group_metrics[g]})
        details[name] = {'groups': group_metrics,
            'common16_improved': int(((err**2 < old**2) & common).sum()),
            'remaining484_improved': int(((err**2 < old**2) & ~common).sum()),
            'common16_same_error_sign': int((np.sign(err[common]) == np.sign(old[common])).sum()),
            'common16_in_worst20': int(sum(common[list(worst[name])])),
            'folds_improved': sum(group_metrics[f'fold{k}']['gain_vs_current'] > 0 for k in range(5)),
            'residual_correlation_vs_current': core.correlation(err, old),
            'common16_sse_gain': float((old**2-err**2)[common].sum()),
            'remaining484_sse_gain': float((old**2-err**2)[~common].sum()),
            'new_worst20_ids': [str(data.ids[i]) for i in sorted(worst[name]-worst['current'])],
            'left_worst20_ids': [str(data.ids[i]) for i in sorted(worst['current']-worst[name])]}
    summary = {'seed': seed, 'binding': binding_for(plan, seed), 'primary': PRIMARY,
               'metrics': metrics, 'diagnosis': details, 'encoder_updates': 0,
               'test_accessed': False, 'independent_holdout': False}
    rows = [{'ID': str(ident), 'hardness': float(data.y[i]), 'fold': int(ref['folds'][i]),
             **{n: float(p[i]) for n, p in predictions.items()}} for i, ident in enumerate(data.ids)]
    errors = [{'ID': str(data.ids[i]), 'hardness': float(data.y[i]), 'fold': int(ref['folds'][i]),
        'noise': float(data.noise[i]), 'noise_quartile': int(noise[i])+1, 'common16': bool(common[i]),
        'new_worst20': i in worst[PRIMARY]-worst['current'], 'left_worst20': i in worst['current']-worst[PRIMARY],
        **{n+'_error': float(p[i]-data.y[i]) for n, p in predictions.items()},
        'candidate_sse_gain': float(old[i]**2-(predictions[PRIMARY][i]-data.y[i])**2)}
        for i in np.argsort(-(predictions[PRIMARY]-data.y)**2, kind='stable')]
    return summary, {'oof.csv': rows, 'diagnostics.csv': groups, 'per_image_errors.csv': errors,
                     'common16.csv': [r for r in errors if r['common16']]}


def report_text(summary):
    lines = [f'Fixed Large518 Ridge confirmation seed{summary["seed"]}',
             'Primary=current_large518_ridge_equal; gain>0 means improvement.']
    lines += [f'{n}: RMSE={m["rmse"]:.6f}; gain vs current={m["gain_vs_current"]:+.6f}' for n, m in summary['metrics'].items()]
    for g in ('common16', 'remaining484', 'noiseQ4', 'middle80', 'low10', 'high10'):
        old = summary['diagnosis']['current']['groups'][g]
        new = summary['diagnosis'][PRIMARY]['groups'][g]
        lines.append(f'{g}: {old["rmse"]:.6f} -> {new["rmse"]:.6f}; gain={new["gain_vs_current"]:+.6f}')
    d = summary['diagnosis'][PRIMARY]
    lines += [f'Common16 improved={d["common16_improved"]}/16; in worst20={d["common16_in_worst20"]}/16; same error sign={d["common16_same_error_sign"]}/16.',
        f'New worst20: {d["new_worst20_ids"]}', 'No automatic adoption; inspect standalone and other484/noise as well.',
        'Same repeatedly studied500/reused splits, not independent holdout. Historical42/2026 excluded from aggregate.',
        'No encoder/TabPFN refits, test access, mixture tuning or submission changes.']
    return '\n'.join(lines)+'\n'


def save_results(folder, data, ref, predictions, selections, plan, seed):
    summary, outputs = result_data(data, ref, predictions, selections, plan, seed)
    for name, rows in outputs.items():
        e.write_csv(folder/name, rows)
    e.write_json(folder/'summary.json', summary)
    e.write_json(folder/'selection.json', selections)
    (folder/'report.txt').write_text(report_text(summary), encoding='utf-8')
    e.write_json(folder/'completed.json', {'binding': binding_for(plan, seed),
        'files': {n: e.base.sha256_file(folder/n) for n in e.OUTPUTS}})
    print(report_text(summary), end='')
    print('Report: '+str(folder/'report.txt'))
    return summary


def read_completed(folder, data, x, ref, plan, seed):
    binding = binding_for(plan, seed)
    if not e.completed(folder, binding):
        raise FileNotFoundError(f'Run evaluate --seed {seed} first; no implicit fitting')
    with contextlib.redirect_stdout(io.StringIO()):
        p, choices = collect(folder, data, x, ref, binding, seed, allow_fit=False, runner=forbidden)
    summary, outputs = result_data(data, ref, p, choices, plan, seed)
    if e.read_json(folder/'summary.json') != summary or e.read_json(folder/'selection.json') != choices:
        raise ValueError('Saved results do not match cached predictions')
    if (folder/'report.txt').read_text(encoding='utf-8') != report_text(summary):
        raise ValueError('Saved report differs')
    for name, rows in outputs.items():
        # Compare exact CSV serialization semantics, including error flags and group diagnostics.
        import csv
        expected = list(csv.DictReader(io.StringIO(core.csv_text(rows))))
        with (folder/name).open(encoding='utf-8', newline='') as handle:
            if list(csv.DictReader(handle)) != expected:
                raise ValueError('Saved CSV differs from cache: '+name)
    return summary


def evaluate(folder=OUT, seed=None):
    data, x, refs, plan = validate(folder)
    for s in SEEDS if seed is None else (seed,):
        if s not in SEEDS:
            raise ValueError('Only the fixed three confirmation seeds are allowed')
        sub = folder/f'seed{s}'
        if (sub/'completed.json').exists():
            read_completed(sub, data, x, refs[s], plan, s)
            print(f'Completed seed{s} verified; no refitting')
            continue
        predictions, choices = collect(sub, data, x, refs[s], binding_for(plan, s), s)
        validate(folder)
        save_results(sub, data, refs[s], predictions, choices, plan, s)


def aggregate(summaries):
    if [s['seed'] for s in summaries] != list(SEEDS) or any(s['primary'] != PRIMARY for s in summaries):
        raise ValueError('Require all three fixed seeds in order; no historical seeds')
    rows = []
    for s in summaries:
        m, d = s['metrics'], s['diagnosis'][PRIMARY]
        row = {'seed': s['seed'], 'current_rmse': m['current']['rmse'],
               'candidate_rmse': m[PRIMARY]['rmse'], 'standalone_rmse': m['large518_ridge']['rmse'],
               'gain_vs_current': m[PRIMARY]['gain_vs_current'], 'folds_improved': d['folds_improved']}
        for g in FOCUS:
            row[g] = {'current_rmse': s['diagnosis']['current']['groups'][g]['rmse'],
                      'candidate_rmse': d['groups'][g]['rmse'], 'gain': d['groups'][g]['gain_vs_current']}
        row.update({k: d[k] for k in ('common16_improved', 'common16_in_worst20', 'common16_same_error_sign',
            'remaining484_improved', 'common16_sse_gain', 'remaining484_sse_gain', 'new_worst20_ids', 'left_worst20_ids')})
        rows.append(row)
    overall = all(r['gain_vs_current'] > 0 for r in rows)
    focus_ok = {g: all(r[g]['gain'] >= 0 for r in rows) for g in FOCUS}
    mean = float(np.mean([r['gain_vs_current'] for r in rows]))
    conclusion = ('overall_and_focus_direction_supported' if overall and all(focus_ok.values()) else
                  'overall_improves_but_subgroup_tradeoff' if overall else 'mixed_or_not_supported')
    return {'seeds': list(SEEDS), 'primary': PRIMARY, 'per_seed': rows, 'mean_gain_vs_current': mean,
        'mean_gain_by_group': {g: float(np.mean([r[g]['gain'] for r in rows])) for g in FOCUS},
        'all_seeds_improve_overall': overall, 'all_seeds_nonworsening_by_group': focus_ok,
        'passes_numeric_expansion_priority': overall and mean >= PROTOCOL['expansion_priority_mean_gain'],
        'conclusion': conclusion, 'automatic_adoption': False, 'additional_review_required': PROTOCOL['diagnostics'],
        'aggregation': PROTOCOL['aggregation'], 'historical_seeds_in_aggregate': False, 'independent_holdout': False}


def compare(folder=OUT):
    data, x, refs, plan = validate(folder)
    result = aggregate([read_completed(folder/f'seed{s}', data, x, refs[s], plan, s) for s in SEEDS])
    result['protocol_sha256'] = e.digest(plan['frozen'])
    e.write_json(folder/'confirmation_summary.json', result)
    flat = []
    for r in result['per_seed']:
        row = {k: r[k] for k in ('seed', 'current_rmse', 'candidate_rmse', 'standalone_rmse', 'gain_vs_current', 'folds_improved')}
        row.update({g+'_'+k: v for g in FOCUS for k, v in r[g].items()})
        flat.append(row)
    e.write_csv(folder/'comparison.csv', flat)
    lines = ['Fixed Large518 + Ridge, 3-split confirmation; gain>0 means improvement.']
    for r in result['per_seed']:
        lines.append(f'seed{r["seed"]}: {r["current_rmse"]:.6f} -> {r["candidate_rmse"]:.6f}; gain={r["gain_vs_current"]:+.6f}')
        lines.extend(f'  {g}: {r[g]["current_rmse"]:.6f} -> {r[g]["candidate_rmse"]:.6f}; gain={r[g]["gain"]:+.6f}' for g in FOCUS)
    lines += [f'Mean overall gain={result["mean_gain_vs_current"]:+.6f}', 'Conclusion: '+result['conclusion'],
              'Subgroup warnings remain separate from overall improvement; no automatic adoption.',
              PROTOCOL['aggregation'], PROTOCOL['limits']]
    (folder/'report.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(e.base.canonical_json(result))
    print('Report: '+str(folder/'report.txt'))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'evaluate', 'compare'))
    parser.add_argument('--seed', type=int, choices=SEEDS)
    parser.add_argument('--output-dir', type=Path, default=OUT)
    args = parser.parse_args()
    if args.command == 'prepare': prepare(args.output_dir)
    elif args.command == 'evaluate': evaluate(args.output_dir, args.seed)
    else: compare(args.output_dir)


if __name__ == '__main__':
    main()
