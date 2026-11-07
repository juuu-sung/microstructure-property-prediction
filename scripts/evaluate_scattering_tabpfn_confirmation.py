"""User-run fixed scattering486 confirmation; Built with PriorLabs-TabPFN.

Reuse native-image features and same-fold archived285/current controls.
Only evaluate creates new TabPFN contexts: 3 seeds x 5 folds, no inner search.
"""
from __future__ import annotations
import argparse
import contextlib
import csv
import io
from pathlib import Path
import time
import numpy as np
if __package__:
    from . import evaluate_scattering_tabpfn as experiment
    from . import evaluate_tabpfn_v2_confirmation as archive
else:
    import evaluate_scattering_tabpfn as experiment
    import evaluate_tabpfn_v2_confirmation as archive

t = experiment.t
OUT = t.ROOT/'reports/scattering_tabpfn_confirmation'
SEEDS = (31415, 27182, 16180)
PRIMARY = experiment.PRIMARY
PROTOCOL = {
    'version': 1, 'seeds': list(SEEDS), 'primary': PRIMARY,
    'features': 'Existing285 plus exactly the archived scattering201 =486; no re-extraction or feature changes.',
    'regressor': t.PROTOCOL,
    'selection_override': 'Fixed486; no new inner185/285 choice, hyperparameter search, weight search or clipping.',
    'folds': 'Exact same outer5 folds from completed TabPFN confirmation for each listed seed;400 context/100 query.',
    'model_seed': 'seed+1009*fold+9; identical to archived285 outer cell',
    'current': 'Matching-seed archived base50%+nested185/285 TabPFN50%; not seed42/2026 predictions.',
    'control': 'Matching-seed base50%+fixed285 TabPFN50%; cached contexts verified, never silently refit.',
    'candidate': 'Matching-seed base50%+fixed486 TabPFN50%; base=hand185*.25+originalResNet*.25+DINO_D4*.5.',
    'new_context_fits': 15, 'new_gradient_training': False, 'test_access': False,
    'diagnostics': 'Standalone, low10/high10/middle80, noise quartiles, historical common16, remaining484, new worst20 and folds.',
    'decision': 'Direction supported only when all3 seeds improve current AND fixed285. Expansion priority additionally mean gain vs current>=0.30; diagnostics required, no automatic adoption.',
    'priority_mean_gain': .30,
    'stop': 'Finish these3 seeds once; do not add seeds, change J/L/features/weights or tune from results.',
    'aggregation': 'Mean of three per-seed OOF RMSE differences; excludes historical42/2026, no prediction averaging.',
    'interpretation': 'Same repeatedly studied500, previously used splits; not independent holdout,1500 independent samples or statistical significance.',
}


def forbidden(*args):
    raise ValueError('Missing completed cell; no implicit fitting in reference/validation/compare')


def archived_reference(data, features, base_ref, old_plan, seed):
    summary = archive.read_completed(archive.OUT, data, base_ref, old_plan, seed)
    folder = archive.OUT/f'seed{seed}'
    rows = t.core.align_rows(t.core.read_rows(folder/'oof.csv', {'ID','hardness','fold'}), data.ids, folder)
    col = lambda n: t.core.finite_column(rows, n, folder)
    np.testing.assert_array_equal(col('hardness'), data.y)
    np.testing.assert_array_equal(t.core.integer_column(rows, 'fold', folder), base_ref['folds'])
    np.testing.assert_array_equal(col('current'), base_ref['current'])
    np.testing.assert_array_equal(col(archive.PRIMARY), .5*(base_ref['current']+col(t.PRIMARY)))
    # The old120 cells are read only, including all inner choices and outer285 predictions.
    with contextlib.redirect_stdout(io.StringIO()):
        for fold in range(5):
            vi, outputs, choice = t.evaluate_fold(features, data.y, base_ref['folds'], seed, fold,
                                                  folder, summary['binding'], forbidden)
            if choice != summary['selection'][fold]:
                raise ValueError('Archived inner selection differs')
            for name, p in outputs.items():
                np.testing.assert_array_equal(p, col(name if name == t.PRIMARY else 'tabpfn_'+name)[vi])
    return {'folds': base_ref['folds'], 'base': base_ref['current'], 'current': col(archive.PRIMARY),
            'tabpfn285': col('tabpfn_morphology285'),
            'control': .5*(base_ref['current']+col('tabpfn_morphology285')),
            'runtime': summary['binding']['runtime'], 'archive': t.file_ref(folder/'completed_evaluation.json')}


def load_inputs():
    data, base285, historical_refs, source_plan = experiment.validate(experiment.OUT)
    added, _ = experiment.load_features(experiment.OUT, data, source_plan)
    old_data, features, base_refs, old_plan = archive.validate(archive.OUT)
    for a, b in ((data.ids, old_data.ids), (data.y, old_data.y), (data.noise, old_data.noise),
                 (base285, features['morphology285'])):
        np.testing.assert_array_equal(a, b)
    refs = {s: archived_reference(data, features, base_refs[s], old_plan, s) for s in SEEDS}
    if any(r['runtime'] != historical_refs[42]['runtime'] for r in refs.values()):
        raise ValueError('Confirmation runtime differs from original experiment')
    x = np.c_[base285, added]
    frozen = {'protocol': PROTOCOL, 'source': t.file_ref(__file__),
        'archive_source': t.file_ref(archive.__file__),
        'scattering_plan': t.file_ref(experiment.OUT/'protocol.json'),
        'scattering_payload': t.digest(source_plan['frozen']),
        'feature_metadata': t.file_ref(experiment.OUT/'features/metadata.json'),
        'feature_array_sha256': t.common.sha256_array(x),
        'confirmation_plan': t.file_ref(archive.OUT/'protocol.json'),
        'confirmation_payload': t.digest(old_plan['frozen']),
        'common_failure_ids': source_plan['frozen']['common_failure_ids'],
        'references': {str(s): {'archive': r['archive'], 'runtime': r['runtime'],
            'arrays': {n: t.common.sha256_array(r[n]) for n in ('folds','base','current','tabpfn285','control')}}
            for s, r in refs.items()}, 'submissions': source_plan['frozen']['submissions']}
    return data, x, refs, frozen


def fold_rows(data, ref):
    return [{'ID': str(i), 'fold': int(f)} for i, f in zip(data.ids, ref['folds'], strict=True)]


def same_rows(path, expected):
    with path.open(encoding='utf-8', newline='') as handle:
        actual = list(csv.DictReader(handle))
    wanted = list(csv.DictReader(io.StringIO(t.core.csv_text(expected))))
    if actual != wanted:
        raise ValueError('Saved rows differ: '+str(path))


def prepare(folder=OUT):
    data, _, refs, frozen = load_inputs()
    if (folder/'protocol.json').exists():
        validate(folder)
        print('Existing confirmation preparation matches; unchanged'); return
    if folder.exists() and any(folder.iterdir()):
        raise ValueError('Nonempty unprepared destination; refusing overwrite')
    folder.mkdir(parents=True, exist_ok=True)
    for seed, ref in refs.items():
        t.write_csv(folder/f'folds_seed{seed}.csv', fold_rows(data, ref))
    t.write_json(folder/'protocol.json', {'frozen': frozen, 'created_unix': time.time(),
                                         'new_context_fits_during_preparation': 0})
    print(f'Prepared fixed486, seeds={SEEDS},15 planned context fits; no extraction or fitting: {folder}')


def validate(folder=OUT):
    data, x, refs, frozen = load_inputs()
    plan = t.read_json(folder/'protocol.json')
    if plan['frozen'] != frozen or plan['new_context_fits_during_preparation'] != 0:
        raise ValueError('Frozen confirmation sources/inputs/settings changed')
    for seed, ref in refs.items():
        same_rows(folder/f'folds_seed{seed}.csv', fold_rows(data, ref))
    return data, x, refs, plan


def binding_for(plan, ref, seed):
    return {'plan': t.digest(plan['frozen']), 'seed': seed, 'runtime': ref['runtime']}


def build_results(data, ref, pred, plan, seed):
    metrics, rows, diag, errors = experiment.result_data(data, ref, pred, plan['frozen']['common_failure_ids'])
    lookup = {(r['model'], r['group']): r for r in diag}
    summary = {'seed': seed, 'binding': binding_for(plan, ref, seed), 'primary': PRIMARY,
        'metrics': metrics, 'new_context_fits': 5, 'archived_control_fits_reused': 5,
        'inner_search': False, 'test_accessed': False, 'independent_holdout': False,
        'folds_improved_vs_current': sum(lookup[('current', f'fold{f}')]['rmse'] > lookup[(PRIMARY, f'fold{f}')]['rmse'] for f in range(5)),
        'folds_improved_vs_control': sum(lookup[('fixed285_control_blend', f'fold{f}')]['rmse'] > lookup[(PRIMARY, f'fold{f}')]['rmse'] for f in range(5)),
        'attribution': 'Built with PriorLabs-TabPFN'}
    return summary, rows, diag, errors


def report_text(summary):
    lines = [f"# Scattering486 confirmation - seed{summary['seed']}", '', 'Built with PriorLabs-TabPFN.', '',
        '| Model | OOF RMSE | Gain vs current | Gain vs fixed285 blend |', '|---|---:|---:|---:|']
    lines += [f"| {n} | {m['rmse']:.6f} | {m['gain_vs_current']:+.6f} | {m['gain_vs_control']:+.6f} |" for n, m in summary['metrics'].items()]
    lines += ['', 'Fixed486; Base50%+TabPFN48650%; both controls use these exact outer folds.',
        'No new inner search, feature extraction, gradient training or test access.',
        'Finish all3 seeds before deciding; mean gain>=0.30 is expansion priority only, not automatic adoption.',
        'Review standalone, common16, remaining484, middle80, tails, noise and new worst20.',
        'Same500; not independent holdout. Historical42/2026 excluded from aggregate.', '']
    return '\n'.join(lines)


def read_completed(folder, data, x, ref, plan, seed):
    target = folder/f'seed{seed}'; binding = binding_for(plan, ref, seed)
    if not experiment.completed(target, binding):
        raise ValueError('Completed confirmation required; no implicit fitting')
    pred = np.full(len(data.ids), np.nan)
    with contextlib.redirect_stdout(io.StringIO()):
        for fold in range(5):
            vi, p = experiment.evaluate_fold(target, binding, x, data.y, ref['folds'], seed, fold, forbidden)
            pred[vi] = p
    summary, rows, diag, errors = build_results(data, ref, pred, plan, seed)
    if summary != t.read_json(target/'summary.json'):
        raise ValueError('Summary does not match saved cells')
    for name, records in (('oof.csv', rows), ('diagnostics.csv', diag), ('per_image_errors.csv', errors)):
        same_rows(target/name, records)
    if (target/'report.md').read_text(encoding='utf-8') != report_text(summary):
        raise ValueError('Saved report differs')
    return summary


def evaluate_seed(folder, device, seed):
    if seed not in SEEDS:
        raise ValueError('Only the fixed three confirmation seeds are allowed')
    data, x, refs, plan = validate(folder)
    ref = refs[seed]; target = folder/f'seed{seed}'; binding = binding_for(plan, ref, seed)
    if (target/'completed_evaluation.json').exists():
        read_completed(folder, data, x, ref, plan, seed)
        print(f'Completed seed{seed} verified; no refitting'); return
    predictor = t.Predictor(t.verify_weights(t.OUT), device)
    if predictor.runtime != ref['runtime']:
        raise ValueError('Use exact archived runtime; no CPU/precision fallback: '+str(ref['runtime']))
    target.mkdir(parents=True, exist_ok=True)
    pred = np.full(len(data.ids), np.nan)
    for fold in range(5):
        vi, p = experiment.evaluate_fold(target, binding, x, data.y, ref['folds'], seed, fold, predictor)
        pred[vi] = p
        print(f'seed={seed} outer={fold}: fixed486 context400/query100 completed', flush=True)
    validate(folder)
    summary, rows, diag, errors = build_results(data, ref, pred, plan, seed)
    for name, records in (('oof.csv', rows), ('diagnostics.csv', diag), ('per_image_errors.csv', errors)):
        t.write_csv(target/name, records)
    t.write_json(target/'summary.json', summary)
    (target/'report.md').write_text(report_text(summary), encoding='utf-8')
    t.write_json(target/'completed_evaluation.json', {'binding': binding,
        'files': {n: t.common.sha256_file(target/n) for n in sorted(experiment.expected_outputs())}})
    m = summary['metrics'][PRIMARY]
    print(f"PRIMARY seed={seed}: RMSE={m['rmse']:.6f}; gain vs current={m['gain_vs_current']:+.6f}; gain vs control={m['gain_vs_control']:+.6f}", flush=True)


def aggregate(summaries):
    if [s['seed'] for s in summaries] != list(SEEDS) or any(s['primary'] != PRIMARY for s in summaries):
        raise ValueError('All3 fixed seeds and fixed primary required; no historical seeds')
    rows = [{'seed': s['seed'], 'current_rmse': s['metrics']['current']['rmse'],
             'fixed285_control_rmse': s['metrics']['fixed285_control_blend']['rmse'],
             'candidate_rmse': s['metrics'][PRIMARY]['rmse'],
             'gain_vs_current': s['metrics'][PRIMARY]['gain_vs_current'],
             'gain_vs_control': s['metrics'][PRIMARY]['gain_vs_control'],
             'folds_improved_vs_current': s['folds_improved_vs_current'],
             'folds_improved_vs_control': s['folds_improved_vs_control']} for s in summaries]
    numeric = [[r[k] for k in ('current_rmse','fixed285_control_rmse','candidate_rmse','gain_vs_current','gain_vs_control')] for r in rows]
    if not np.isfinite(numeric).all():
        raise ValueError('Nonfinite aggregate')
    gain = float(np.mean([r['gain_vs_current'] for r in rows]))
    supported = all(r['gain_vs_current'] > 0 and r['gain_vs_control'] > 0 for r in rows)
    return {'seeds': list(SEEDS), 'primary': PRIMARY, 'per_seed': rows,
        'mean_gain_vs_current': gain, 'mean_gain_vs_control': float(np.mean([r['gain_vs_control'] for r in rows])),
        'all_seeds_improve_both': supported,
        'passes_expansion_priority': supported and gain >= PROTOCOL['priority_mean_gain'],
        'conclusion': 'direction_supported' if supported else 'mixed_or_not_supported',
        'additional_review_required': PROTOCOL['diagnostics']+'; no automatic adoption.',
        'aggregation': PROTOCOL['aggregation'], 'historical_seeds_in_aggregate': False, 'independent_holdout': False}


def compare(folder=OUT):
    data, x, refs, plan = validate(folder)
    result = aggregate([read_completed(folder, data, x, refs[s], plan, s) for s in SEEDS])
    result['protocol_sha256'] = t.digest(plan['frozen'])
    t.write_json(folder/'confirmation_summary.json', result)
    t.write_csv(folder/'comparison.csv', result['per_seed'])
    lines = ['# Fixed scattering486 - three split confirmation', '', 'Built with PriorLabs-TabPFN.', '',
        '| Seed | Current | Fixed285 blend | Candidate | Gain vs current | Gain vs control |', '|---|---:|---:|---:|---:|---:|']
    lines += [f"| {r['seed']} | {r['current_rmse']:.6f} | {r['fixed285_control_rmse']:.6f} | {r['candidate_rmse']:.6f} | {r['gain_vs_current']:+.6f} | {r['gain_vs_control']:+.6f} |" for r in result['per_seed']]
    lines += ['', f"Direction: {result['conclusion']}; expansion priority>=0.30: {result['passes_expansion_priority']}.",
        'Review per-seed diagnostics/common16/new worst20 before deciding; original submission preserved.',
        'Same500; no independent inference, prediction averaging or historical42/2026 in this aggregate.', '']
    (folder/'report.md').write_text('\n'.join(lines), encoding='utf-8')
    print(t.common.canonical_json(result), flush=True)
    print('Report:', folder/'report.md')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('prepare','validate','evaluate','compare'))
    p.add_argument('--device', default='cuda')
    p.add_argument('--seed', type=int, choices=SEEDS)
    a = p.parse_args(argv)
    if a.seed is not None and a.command != 'evaluate':
        p.error('--seed only applies to evaluate')
    if a.command == 'prepare': prepare()
    elif a.command == 'validate':
        validate(); print('Fixed486, archived120 cells and same-fold controls validated; no fitting')
    elif a.command == 'compare': compare()
    else:
        for seed in SEEDS if a.seed is None else (a.seed,):
            evaluate_seed(OUT, a.device, seed)
        if all((OUT/f'seed{s}/completed_evaluation.json').exists() for s in SEEDS): compare()
        else: print('Remaining fixed seeds required before aggregate conclusion')


if __name__ == '__main__': main()
