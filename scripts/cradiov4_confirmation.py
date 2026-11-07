"""Fixed S:L:C=50:25:25 confirmation on three previously used splits.

Only user-run evaluate fits competition contexts. prepare/compare are read-only
with respect to models. No encoder forwards, test access or submission writes.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import io
from pathlib import Path
import time
import numpy as np
from threadpoolctl import threadpool_limits

if __package__:
    from . import cradiov4_experiment as original
else:
    import cradiov4_experiment as original

e, eng, core = original.e, original.eng, original.core
large_confirmation = original.diagnostics
residual = original.residual
t = residual.t
ROOT = original.ROOT
SOURCE = ROOT/'reports/cradiov4_importfix'
OUT = ROOT/'reports/cradiov4_confirmation'
SEEDS = (31415, 27182, 16180)
PRIMARY = 'radio_complementary_blend'
CONTROL = 'large_retuned_complementary'
GROUPS = ('common16', 'remaining484', 'middle80', 'low10', 'high10',
          'noiseQ1', 'noiseQ2', 'noiseQ3', 'noiseQ4')
PROTOCOL = {
    'version': 1, 'seeds': list(SEEDS), 'primary': PRIMARY, 'control': CONTROL,
    'current': '.5S+.5L; S=entire archived Scattering486 ensemble, L=archived standalone-selected raw Large518 Ridge.',
    'candidate': '.5S+.25L+.25C; C=archived frozen C-RADIO4608 features with new fold-local Ridge.',
    'matched_control': '.5S+.25L+.25L_retuned; same alpha grid, partitions and selection objective as C.',
    'alphas': [100., 1000., 10000.],
    'selection': 'PRESERVE original objective: minimize pooled INNER MSE of .5S_inner+.5encoder. Exact ties prefer larger alpha. Do NOT retune for .5S+.25L+.25C.',
    'inner_scattering': 'Fresh3-fold context266/267 -> query133/134 within EACH outer400. Every base Ridge alpha is selected by a further3-fold context-only CV. Never substitute global OOF or in-sample predictions.',
    'base_alpha_grids': {k: list(v) for k, v in residual.eng.GRIDS.items()},
    'base_seed_rule': 'Inner splits seed+outer; base selection/TabPFN seed+1009*outer+inner; identical historical protocol.',
    'tabpfn': 'Pinned local V2,8 estimators,FP32, same archived CUDA runtime. Context preprocessing only; no neural weight updates.',
    'preprocessing': 'Encoder X/y StandardScaler fit only inner train; selected Ridge refit outer400. No PCA, feature changes or clipping.',
    'outer': 'Exact archived confirmation 5outer400/100 per seed; original Large518 standalone predictions and alphas must reproduce.',
    'features': 'Reuse checked Large5183072 and C-RADIO4608 caches; no encoder loading/extraction.',
    'planned_tabpfn_contexts': 45, 'planned_base_ridge_fits': 1755,
    'planned_encoder_ridge_fits_range': [300, 315],
    'diagnostics': 'Current AND matched control; standalone, common16, remaining484, middle80/tails, all noise quartiles, folds and new worst20.',
    'decision': 'Separate overall gains from group tradeoffs. Mean gain>=0.30 AND all3 seeds improve both controls is expansion priority only, not automatic adoption.',
    'priority_mean_gain': .30,
    'stop': 'Exactly these3 seeds; no alpha/weight/feature search from outer results. No special common16 weighting.',
    'scope': 'User executes45 fresh inner TabPFN contexts and CPU Ridge fitting. No encoder gradient updates, outer Scattering refits, test access, final500 fitting or submission changes.',
    'limits': 'Candidate promoted from prespecified historical secondary after42/2026. Same repeatedly studied500 and reused splits, not independent holdout.',
    'aggregation': 'Mean of3 per-seed OOF RMSE differences; exclude42/2026, no averaging predictions or independent-sample inference.',
    'attribution': 'NVIDIA C-RADIO, Meta DINOv2; preserved Scattering ensemble includes PriorLabs-TabPFN.'}


def forbidden(*args, **kwargs):
    raise RuntimeError('Missing completed artifact; no implicit fitting. Run evaluate.')


def load_state():
    with contextlib.redirect_stdout(io.StringIO()):
        data, large, _, radio_plan = original.validate(SOURCE)
        radio, radio_ref = original.load_features(SOURCE, data, radio_plan)
        ld, lx, scat_refs, lp = large_confirmation.validate(large_confirmation.OUT)
        rd, raw, _, rp = residual.validate()
    for other in (ld, rd):
        for name in ('ids', 'y', 'noise'):
            np.testing.assert_array_equal(getattr(data, name), getattr(other, name))
    np.testing.assert_array_equal(large, lx)
    archive = large_confirmation.ARCHIVE
    sp = e.read_json(archive/'protocol.json')
    common = radio_plan['frozen']['common_failure_ids']
    if common != lp['frozen']['common_failure_ids'] or common != sp['frozen']['common_failure_ids']:
        raise ValueError('Historical common16 changed')
    refs, files = {}, {}
    for seed in SEEDS:
        folder = large_confirmation.OUT/f'seed{seed}'
        with contextlib.redirect_stdout(io.StringIO()):
            large_confirmation.read_completed(folder, data, large, scat_refs[seed], lp, seed)
        rows = core.align_rows(core.read_rows(folder/'oof.csv', {'ID', 'hardness', 'fold', 'large518_ridge', large_confirmation.PRIMARY}), data.ids, folder)
        np.testing.assert_array_equal(core.finite_column(rows, 'hardness', folder), data.y)
        folds = core.integer_column(rows, 'fold', folder)
        np.testing.assert_array_equal(folds, scat_refs[seed]['folds'])
        old_large = core.finite_column(rows, 'large518_ridge', folder)
        current = core.finite_column(rows, large_confirmation.PRIMARY, folder)
        np.testing.assert_allclose(current, .5*scat_refs[seed]['current']+.5*old_large, rtol=0, atol=1e-12)
        choices = e.read_json(folder/'selection.json')
        if [c['fold'] for c in choices] != list(range(5)):
            raise ValueError('Archived Large518 fold choices differ')
        # Prove base and encoder inner queries are identical, before fitting.
        for fold in range(5):
            y = data.y[folds != fold]
            a, b = residual.eng.splits(y, seed+fold), eng.ridge.inner_splits(y, seed+fold)
            if len(a) != 3 or len(b) != 3:
                raise ValueError('Expected exactly3 inner splits')
            for left, right in zip(a, b, strict=True):
                for u, v in zip(left, right, strict=True):
                    np.testing.assert_array_equal(u, v)
        refs[seed] = {'folds': folds, 'scattering': scat_refs[seed]['current'], 'large': old_large,
                      'current': current, 'large_alphas': [c['alpha'] for c in choices],
                      'runtime': sp['frozen']['references'][str(seed)]['runtime']}
        files[str(seed)] = {n: e.file_ref(folder/n) for n in ('completed.json', 'oof.csv', 'selection.json')}
        files[str(seed)]['arrays'] = {n: e.base.sha256_array(refs[seed][n]) for n in ('folds', 'scattering', 'large', 'current')}
    if any(refs[s]['runtime'] != refs[SEEDS[0]]['runtime'] for s in SEEDS):
        raise ValueError('Archived runtimes differ')
    sources = (Path(__file__), ROOT/'scripts/run_cradiov4_confirmation.ps1')
    frozen = {'protocol': PROTOCOL, 'sources': {p.name: e.file_ref(p) for p in sources},
              'source_radio_plan': e.file_ref(SOURCE/'protocol.json'), 'source_radio_payload': e.digest(radio_plan['frozen']),
              'radio_features': radio_ref, 'large_plan': e.file_ref(large_confirmation.OUT/'protocol.json'),
              'large_payload': e.digest(lp['frozen']), 'residual_plan': e.file_ref(residual.OUT/'protocol.json'),
              'residual_payload': e.digest(rp['frozen']), 'scattering_confirmation_plan': e.file_ref(archive/'protocol.json'),
              'references': files, 'common_failure_ids': common, 'environment': e.environment(),
              'raw_arrays': {n: e.base.sha256_array(x) for n, x in raw.items()},
              'ids_sha256': e.base.sha256_array(data.ids), 'y_sha256': e.base.sha256_array(data.y),
              'noise_sha256': e.base.sha256_array(data.noise),
              'preserved_submissions': radio_plan['frozen']['preserved_submissions']}
    return data, large, radio, raw, refs, frozen


def prepare(folder=OUT):
    *_, frozen = load_state()
    path = folder/'protocol.json'
    if path.exists():
        if e.read_json(path)['frozen'] != frozen:
            raise ValueError('Prepared confirmation changed; preserve old results')
    else:
        if folder.exists() and any(folder.iterdir()):
            raise ValueError('Nonempty unprepared output folder')
        e.write_json(path, {'frozen': frozen, 'created_unix': time.time(), 'competition_fits_during_preparation': 0})
    print('Prepared: seeds31415/27182/16180, S:L:C=50:25:25;45 new INNER TabPFN contexts planned. No fitting/extraction.')


def validate(folder=OUT):
    *state, frozen = load_state()
    plan = e.read_json(folder/'protocol.json')
    if plan['frozen'] != frozen or plan['competition_fits_during_preparation'] != 0:
        raise ValueError('Frozen confirmation inputs/source/environment changed')
    return (*state, plan)


def binding_for(plan, seed, ref):
    return {'plan': e.digest(plan['frozen']), 'seed': seed, 'runtime': ref['runtime']}


def check(device='cuda', folder=OUT):
    *_, refs, plan = validate(folder)
    predictor = t.Predictor(t.verify_weights(t.OUT), device)
    if predictor.runtime != refs[SEEDS[0]]['runtime']:
        raise ValueError('Use exact archived CUDA runtime; no fallback')
    path = folder/'synthetic_check.json'
    if path.exists():
        read_check(folder, plan, predictor.runtime)
        print('Synthetic check verified; no additional model calls.'); return
    rng = np.random.default_rng(314159)
    x = rng.normal(size=(400, 486)); y = 190+4*x[:, 0]+rng.normal(size=400)
    p, metadata = predictor(x[:267], y[:267], x[267:], 314159)
    eng.ridge.finite(p, (133,))
    result = {'runtime': predictor.runtime, 'context': 267, 'query': 133,
              'finite': True, 'competition_fits': 0, 'metadata': metadata}
    e.write_json(path, {'plan': e.digest(plan['frozen']), 'result': result, 'sha256': e.digest(result)})
    print('Synthetic TabPFN267/133 passed; no competition fitting or encoder inference.')


def read_check(folder, plan, runtime):
    saved = e.read_json(folder/'synthetic_check.json'); result = saved['result']
    if (saved['plan'] != e.digest(plan['frozen']) or saved['sha256'] != e.digest(result)
            or result['runtime'] != runtime or result['context'] != 267 or result['query'] != 133
            or result['competition_fits'] != 0 or result['finite'] is not True):
        raise ValueError('Synthetic check/runtime changed')


def collect(folder, binding, data, large, radio, raw, ref, seed, allow_fit, predictor, runner=None):
    vectors = {n: np.full(len(data.y), np.nan) for n in ('large', 'radio')}
    choices = []
    for fold in range(5):
        ti, vi = np.flatnonzero(ref['folds'] != fold), np.flatnonzero(ref['folds'] == fold)
        if (len(ti), len(vi)) != (400, 100):
            raise ValueError('Expected five400/100 folds')
        target = folder/f'fold{fold}'
        # Only labels in this outer TRAIN are passed to base/encoder fitting.
        cross = residual.crossfit_components(target/'baseline', binding, raw, ti, data.y[ti], seed, fold, predictor, allow_fit)
        inner_s = residual.eng.ensemble(cross)
        scope = {'evaluation': binding, 'fold': fold, 'train': ti.tolist(), 'query': vi.tolist(),
                 'inner_scattering': e.base.sha256_array(inner_s)}
        with threadpool_limits(limits=2):
            p, c = original.fit_fold(target/'encoders', scope,
                {'large': large[ti], 'radio': radio[ti]}, data.y[ti],
                {'large': large[vi], 'radio': radio[vi]}, inner_s, seed+fold, allow_fit, runner)
        np.testing.assert_allclose(p['large_anchor'], ref['large'][vi], rtol=0, atol=1e-8)
        if c['large']['standalone']['alpha'] != ref['large_alphas'][fold]:
            raise ValueError('Original Large518 alpha anchor differs')
        for name in vectors:
            vectors[name][vi] = p[name]
        choices.append({'fold': fold, 'choices': c})
        if allow_fit:
            print(f'seed{seed} outer{fold}: C alpha={c["radio"]["blend"]["alpha"]:g}; matched L alpha={c["large"]["blend"]["alpha"]:g}; original anchor matched; outer labels unused', flush=True)
    predictions = eng.compose(ref['scattering'], ref['large'], vectors['radio'], vectors['large'])
    np.testing.assert_allclose(predictions['current'], ref['current'], rtol=0, atol=1e-12)
    return predictions, choices


def result_data(data, ref, predictions, choices, plan, seed, binding):
    # Map the promoted complementary candidate into the shared diagnostics API,
    # so worst20 flags and per-image SSE gains refer to the actual primary.
    mapped = {large_confirmation.PRIMARY if n == PRIMARY else n: v for n, v in predictions.items()}
    summary, files = large_confirmation.result_data(data, ref, mapped, choices, plan, seed)
    for key in ('metrics', 'diagnosis'):
        summary[key][PRIMARY] = summary[key].pop(large_confirmation.PRIMARY)
    for rows in files.values():
        for row in rows:
            for key in list(row):
                new = key.replace(large_confirmation.PRIMARY, PRIMARY)
                if key != new:
                    row[new] = row.pop(key)
            if row.get('model') == large_confirmation.PRIMARY:
                row['model'] = PRIMARY
    for name, metrics in summary['metrics'].items():
        control = 'large_retuned_ridge' if name in ('radio_ridge', 'large_retuned_ridge') else (
            'large_retuned_blend' if name in ('radio_replaced_blend', 'large_retuned_blend') else CONTROL)
        metrics.update(matched_control=control, gain_vs_control=summary['metrics'][control]['rmse']-metrics['rmse'])
    for name, diagnostic in summary['diagnosis'].items():
        control = summary['metrics'][name]['matched_control']
        for group, metrics in diagnostic['groups'].items():
            metrics['gain_vs_control'] = summary['diagnosis'][control]['groups'][group]['rmse']-metrics['rmse']
    for row in files['diagnostics.csv']:
        row['gain_vs_control'] = summary['diagnosis'][row['model']]['groups'][row['group']]['gain_vs_control']
    summary.update(binding=binding, primary=PRIMARY, selection=choices, anchor_matched=True,
        planned_inner_tabpfn_contexts=15, outer_scattering_refits=0, encoder_forwards=0,
        inner_scattering_uses_global_oof=False)
    return summary, files


def report_text(summary):
    lines = [f'Fixed C-RADIO confirmation seed{summary["seed"]}', PROTOCOL['candidate'],
             'Gain>0 means improvement; comparison is against CURRENT S50/L50.']
    for name, m in summary['metrics'].items():
        lines.append(f'{name}: RMSE={m["rmse"]:.6f}; gain vs current={m["gain_vs_current"]:+.6f}; gain vs matched control={m["gain_vs_control"]:+.6f}')
    d = summary['diagnosis'][PRIMARY]
    for group in GROUPS:
        m = d['groups'][group]
        lines.append(f'{group}: RMSE={m["rmse"]:.6f}; gain vs current={m["gain_vs_current"]:+.6f}; gain vs control={m["gain_vs_control"]:+.6f}')
    lines += [f'Common16 improved={d["common16_improved"]}/16; in worst20={d["common16_in_worst20"]}/16; same error sign={d["common16_same_error_sign"]}/16.',
        f'Remaining484 improved={d["remaining484_improved"]}/484; folds improved={d["folds_improved"]}/5.',
        f'New worst20: {d["new_worst20_ids"]}', PROTOCOL['selection'], PROTOCOL['decision'], PROTOCOL['limits']]
    return '\n'.join(lines)+'\n'


def expected_outputs():
    return {'oof.csv', 'diagnostics.csv', 'per_image_errors.csv', 'common16.csv', 'summary.json', 'selection.json', 'report.txt'} | {
        f'fold{f}/baseline/inner{k}/completed.json' for f in range(5) for k in range(3)} | {
        f'fold{f}/encoders/{n}/{cell}.json' for f in range(5) for n in ('large', 'radio') for cell in ('inner0', 'inner1', 'inner2', 'refit')}


def completed(folder, binding):
    path = folder/'completed_evaluation.json'
    if not path.exists():
        return False
    saved = e.read_json(path)
    if saved['binding'] != binding or set(saved['files']) != expected_outputs():
        raise ValueError('Completion scope differs')
    for name, sha in saved['files'].items():
        if e.base.sha256_file(folder/name) != sha:
            raise ValueError('Completed artifact changed: '+name)
    return True


def evaluate(folder=OUT, seed=31415, device='cuda', allow_fit=True):
    if seed not in SEEDS:
        raise ValueError('Only the fixed3 confirmation seeds are allowed')
    data, large, radio, raw, refs, plan = validate(folder)
    ref = refs[seed]; sub = folder/f'seed{seed}'; binding = binding_for(plan, seed, ref)
    done = completed(sub, binding)
    if not done and not allow_fit:
        raise FileNotFoundError('Incomplete evaluation; run evaluate, no implicit fitting')
    predictor, runner = forbidden, forbidden
    if not done:
        predictor = t.Predictor(t.verify_weights(t.OUT), device)
        if predictor.runtime != ref['runtime']:
            raise ValueError('Use archived CUDA runtime; no CPU/precision fallback')
        read_check(folder, plan, predictor.runtime)
        runner = eng.ridge.fit_candidates
    with contextlib.redirect_stdout(io.StringIO()) if done else contextlib.nullcontext():
        predictions, choices = collect(sub, binding, data, large, radio, raw, ref, seed, not done, predictor, runner)
    summary, files = result_data(data, ref, predictions, choices, plan, seed, binding)
    if done:
        if e.read_json(sub/'summary.json') != summary or e.read_json(sub/'selection.json') != choices:
            raise ValueError('Saved result differs from cached recomputation')
        if (sub/'report.txt').read_text(encoding='utf-8') != report_text(summary):
            raise ValueError('Saved report changed')
        for name, rows in files.items():
            with (sub/name).open(encoding='utf-8', newline='') as handle:
                if list(csv.DictReader(handle)) != list(csv.DictReader(io.StringIO(core.csv_text(rows)))):
                    raise ValueError('CSV differs: '+name)
        print(f'Completed seed{seed} verified; no refitting.'); return summary
    # Revalidate static inputs before committing results, as in historical runs.
    validate(folder)
    for name, rows in files.items():
        e.write_csv(sub/name, rows)
    e.write_json(sub/'summary.json', summary); e.write_json(sub/'selection.json', choices)
    (sub/'report.txt').write_text(report_text(summary), encoding='utf-8')
    e.write_json(sub/'completed_evaluation.json', {'binding': binding,
        'files': {name: e.base.sha256_file(sub/name) for name in sorted(expected_outputs())}})
    print(report_text(summary)); print('Report: '+str(sub/'report.txt'))
    return summary


def aggregate(summaries):
    if [s['seed'] for s in summaries] != list(SEEDS) or any(s['primary'] != PRIMARY for s in summaries):
        raise ValueError('Exactly3 fixed seeds in order required; exclude42/2026')
    rows = []
    for s in summaries:
        m, d = s['metrics'], s['diagnosis'][PRIMARY]
        row = {'seed': s['seed'], 'current_rmse': m['current']['rmse'], 'control_rmse': m[CONTROL]['rmse'],
               'candidate_rmse': m[PRIMARY]['rmse'], 'standalone_rmse': m['radio_ridge']['rmse'],
               'gain_vs_current': m[PRIMARY]['gain_vs_current'], 'gain_vs_control': m[PRIMARY]['gain_vs_control']}
        row['groups'] = {g: {'current_rmse': s['diagnosis']['current']['groups'][g]['rmse'],
            'control_rmse': s['diagnosis'][CONTROL]['groups'][g]['rmse'], 'candidate_rmse': d['groups'][g]['rmse'],
            'gain_vs_current': d['groups'][g]['gain_vs_current'], 'gain_vs_control': d['groups'][g]['gain_vs_control']} for g in GROUPS}
        row.update({k: v for k, v in d.items() if k != 'groups'})
        rows.append(row)
    both = all(r['gain_vs_current'] > 0 and r['gain_vs_control'] > 0 for r in rows)
    focus = {g: all(r['groups'][g]['gain_vs_current'] >= 0 and r['groups'][g]['gain_vs_control'] >= 0 for r in rows) for g in GROUPS}
    gain = float(np.mean([r['gain_vs_current'] for r in rows]))
    return {'seeds': list(SEEDS), 'primary': PRIMARY, 'per_seed': rows,
        'mean_gain_vs_current': gain, 'mean_gain_vs_control': float(np.mean([r['gain_vs_control'] for r in rows])),
        'mean_group_gains': {g: {k: float(np.mean([r['groups'][g][k] for r in rows])) for k in ('gain_vs_current', 'gain_vs_control')} for g in GROUPS},
        'all_seeds_improve_both': both, 'all_seeds_nonworsening_vs_both_by_group': focus,
        'passes_numeric_expansion_priority': both and gain >= PROTOCOL['priority_mean_gain'],
        'conclusion': ('overall_and_focus_direction_supported' if both and all(focus[g] for g in ('common16', 'remaining484', 'noiseQ4')) else 'overall_improves_but_subgroup_tradeoff' if both else 'mixed_or_not_supported'),
        'automatic_adoption': False, 'independent_holdout': False, 'historical_seeds_in_aggregate': False,
        'aggregation': PROTOCOL['aggregation'], 'additional_review_required': PROTOCOL['diagnostics']}


def compare(folder=OUT):
    summaries = [evaluate(folder, seed, allow_fit=False) for seed in SEEDS]
    result = aggregate(summaries); result['protocol_sha256'] = summaries[0]['binding']['plan']
    e.write_json(folder/'confirmation_summary.json', result)
    lines = ['C-RADIO fixed S:L:C=50:25:25 confirmation; positive gain is improvement.']
    rows = []
    for record in result['per_seed']:
        lines.append(f'seed{record["seed"]}: {record["current_rmse"]:.6f} -> {record["candidate_rmse"]:.6f}; gain vs current={record["gain_vs_current"]:+.6f}, vs control={record["gain_vs_control"]:+.6f}')
        for group in GROUPS:
            g = record['groups'][group]
            rows.append({'seed': record['seed'], 'group': group, **g})
            lines.append(f'  {group}: gain vs current={g["gain_vs_current"]:+.6f}, vs control={g["gain_vs_control"]:+.6f}')
    lines += [f'Mean overall gain={result["mean_gain_vs_current"]:+.6f}', 'Conclusion: '+result['conclusion'],
              PROTOCOL['decision'], PROTOCOL['aggregation'], PROTOCOL['limits']]
    e.write_csv(folder/'comparison.csv', rows)
    (folder/'report.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(e.base.canonical_json(result)); print('Report: '+str(folder/'report.txt'))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'check', 'evaluate', 'compare'))
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, choices=SEEDS, default=SEEDS[0])
    parser.add_argument('--output-dir', type=Path, default=OUT)
    args = parser.parse_args()
    folder = args.output_dir if args.output_dir.is_absolute() else ROOT/args.output_dir
    if args.command == 'prepare': prepare(folder)
    elif args.command == 'check': check(args.device, folder)
    elif args.command == 'evaluate': evaluate(folder, args.seed, args.device)
    else: compare(folder)


if __name__ == '__main__':
    main()
