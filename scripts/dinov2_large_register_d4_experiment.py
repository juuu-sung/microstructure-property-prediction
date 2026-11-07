"""User-run register-Large D4: fixed12.5% mix and25% replacement."""
from __future__ import annotations
import argparse
import contextlib
import csv
import gc
import io
from pathlib import Path
import time
import numpy as np
from threadpoolctl import threadpool_limits
if __package__:
    from . import large_radio_d4_experiment as prior, dinov2_large_register_engine as eng
else:
    import large_radio_d4_experiment as prior
    import dinov2_large_register_engine as eng

e, core, diagnostics = prior.e, prior.core, prior.diagnostics
ROOT, OUT, SEEDS = e.ROOT, e.ROOT/'reports/dinov2_large_register_d4', (42, 2026)
PRIMARY = eng.PRIMARY
FOCUS = ('common16', 'remaining484', 'noiseQ4')
PROTOCOL = {
    'version':1, 'seeds':list(SEEDS), 'primary':PRIMARY, 'secondary':eng.SECONDARY,
    'encoder':'Pinned timm DINOv2-Large reg4; frozen/eval FP32, batch1, no AMP/TF32. No encoder updates.',
    'input':'Native256 gray CPU float32/255; exact sequential8D4 pixel permutations; same bicubic518/no clamp/ImageNet normalization as archived Large.',
    'features':'CLS1024 +1369spatial patch mean1024/population std1024. Explicitly exclude4register tokens.8view FP64 mean->FP32;3072 per original image.',
    'regression':'Context-only X/y StandardScaler+Ridge alpha100/1000/10000; exact ties larger alpha. No PCA.',
    'selection':'Preserve archived Large pooled inner STANDALONE MSE; seed+outer rank-decile3fold. No outer labels/globalOOF for alpha choice.',
    'control':'Verified completed Large D4 and SO400M D4 OOF reused; no base encoder/control/TabPFN refits.',
    'contrast':'Current=.5S+.25L+.25C. PRIMARY=.5S+.125L+.125R+.25C. SECONDARY=.5S+.25R+.25C. S=entire Scattering486 ensemble, R=registerLargeD4. Fixed weights, no selection among ratios.',
    'scope':'Train500 only. User executes syntheticGPU check, feature extraction and CPU Ridge. No test/final500 fits/submission writes.',
    'decision':'Inspect both fixed contrasts, overall/standalone/common16/remaining484/noiseQ1..4/tails/folds/newworst20; no automatic adoption.0.30 is expansion priority, not submission eligibility.',
    'limits':'Repeatedly studied500/reused splits, not independent holdout. Register model has different pretrained weights; this does not isolate registers alone.'}


def load_state():
    eng.assets.validate(require_weights=False)
    with contextlib.redirect_stdout(io.StringIO()):
        state = prior.validate()
        data, raw, old, upstream = state
        for seed in SEEDS:
            prior.evaluate(seed, allow_fit=False, state=state)
    refs, files = {}, {}
    for seed in SEEDS:
        sub = prior.OUT/f'seed{seed}'
        rows = core.read_rows(sub/'oof.csv', {'ID', 'hardness', 'fold', 'large_d4', 'radio_d4', 'both_d4_blend'})
        if [r['ID'] for r in rows] != data.ids.tolist(): raise ValueError('Reference ID order differs')
        np.testing.assert_array_equal([float(r['hardness']) for r in rows], data.y)
        np.testing.assert_array_equal([int(r['fold']) for r in rows], old[seed]['folds'])
        refs[seed] = {**old[seed], **{n: np.array([float(r[n]) for r in rows]) for n in ('large_d4', 'radio_d4')},
                      'current': np.array([float(r['both_d4_blend']) for r in rows])}
        ref = refs[seed]
        np.testing.assert_allclose(ref['current'], .5*ref['scattering']+.25*ref['large_d4']+.25*ref['radio_d4'], rtol=0, atol=1e-10)
        files[str(seed)] = {n: e.file_ref(sub/n) for n in ('completed.json', 'oof.csv', 'summary.json')}
    paths = [Path(__file__), Path(eng.__file__), Path(eng.assets.__file__), eng.assets.LOCK,
             ROOT/'scripts/run_dinov2_large_register_d4.ps1']
    frozen = {'protocol': PROTOCOL, 'sources': {p.name: e.file_ref(p) for p in paths},
        'model_assets': eng.assets.spec(), 'upstream_protocol': e.file_ref(prior.OUT/'protocol.json'),
        'upstream_runtime': prior.read_check(upstream)['runtime'], 'references': files,
        'environment': e.environment(), 'training_inventory': upstream['frozen']['training_inventory'],
        'common_failure_ids': upstream['frozen']['common_failure_ids'],
        'inner_reference_hashes': {str(s): {str(f): e.base.sha256_array(refs[s]['inner_scattering'][f]) for f in range(5)} for s in SEEDS},
        'submissions': {**upstream['frozen']['submissions'],
                        'final_large_radio_d4_blend': e.file_ref(ROOT/'reports/final_large_radio_d4_blend/submission/submission.csv')}}
    return data, raw, refs, frozen


def prepare():
    *_, frozen = load_state()
    path = OUT/'protocol.json'
    if path.exists():
        if e.read_json(path)['frozen'] != frozen: raise ValueError('Prepared sources/inputs changed; preserve existing results')
    else:
        if OUT.exists() and any(OUT.iterdir()): raise ValueError('Nonempty unprepared output directory')
        e.write_json(path, {'frozen': frozen, 'created_unix': time.time()})
    print('Register-Large D4 prepared/verified. Current D4 OOF reconstructed; no competition fits/forwards.')


def validate():
    data, raw, refs, frozen = load_state()
    plan = e.read_json(OUT/'protocol.json')
    if plan['frozen'] != frozen: raise ValueError('Sources, inputs, environment or protocol changed')
    return data, raw, refs, plan


def bind(plan, **kw): return {'plan': e.digest(plan['frozen']), **kw}


def read_check(plan):
    obj = e.read_json(OUT/'synthetic_check.json'); r = obj['result']
    if (obj['binding'] != bind(plan) or obj['sha256'] != e.digest(r) or not r['passed']
            or r['competition_forwards'] != 0 or r['runtime'] != plan['frozen']['upstream_runtime']
            or r['shape'] != [1, eng.WIDTH]): raise ValueError('Register-Large synthetic check missing/stale')
    return r


def check(device='cuda'):
    *_, plan = validate()
    eng.assets.validate()
    # Exactly one runtime configuration per process; interop threads cannot be reset.
    torch, dev, rt = prior.runtime(device)
    if rt != plan['frozen']['upstream_runtime']: raise ValueError('Runtime must match archived D4 precision/device')
    if (OUT/'synthetic_check.json').exists():
        read_check(plan); print('Register-Large synthetic check verified; no repeat.'); return
    yy, xx = torch.meshgrid(torch.arange(256, device=dev), torch.arange(256, device=dev), indexing='ij')
    raw = ((xx*3+yy*7+(xx//13)*(yy//17)) % 256).float()[None, None]/255.
    free_before, total = torch.cuda.mem_get_info(dev)
    model = eng.load_model().to(dev)
    if any(p.requires_grad for p in model.parameters()) or model.training: raise ValueError('Register-Large must be frozen/eval')
    torch.cuda.reset_peak_memory_stats(dev)
    eng.forward('large', model, raw); torch.cuda.synchronize(dev)
    start = time.perf_counter()
    identity, mean = eng.extract_views(model, raw)
    torch.cuda.synchronize(dev); seconds = time.perf_counter()-start
    repeat = eng.forward('large', model, raw).cpu().numpy()
    maxdiff = prior.eng.anchor('large', identity, repeat)
    _, rotated = eng.extract_views(model, torch.rot90(raw, 1, (-2, -1)))
    prior.eng.anchor('large', mean, rotated)
    if mean.shape != (1, eng.WIDTH) or mean.dtype != np.float32: raise ValueError('Unexpected register-Large pooled features')
    r = {'passed': True, 'runtime': rt, 'competition_forwards': 0, 'competition_fits': 0,
         'encoder_updates': 0, 'shape': list(mean.shape), 'eight_view_seconds': seconds,
         'identity_repeat_maxdiff': maxdiff, 'd4_rotation_maxdiff': float(np.max(np.abs(mean-rotated))),
         'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(dev),
         'peak_cuda_reserved_bytes': torch.cuda.max_memory_reserved(dev),
         'free_cuda_bytes_before_model': free_before, 'total_cuda_bytes': total,
         'estimated_train500_forward_minutes': seconds*500/60, 'estimated_test1000_forward_minutes': seconds*1000/60,
         'timing_note': 'Synthetic extrapolation only, excludes download, disk, hashes, CPU Ridge, cold starts and contention.'}
    del model; gc.collect(); torch.cuda.empty_cache()
    validate()  # Native import must not alter dependency provenance.
    e.write_json(OUT/'synthetic_check.json', {'binding': bind(plan), 'result': r, 'sha256': e.digest(r)})
    print(e.base.canonical_json(r), flush=True)


def feature_scope(plan, ck): return bind(plan, stage='register_large_d4_features', check=e.digest(ck))


def read_bundle(path, ids, plan, ck):
    x, meta = e.npz_read(path, ids, eng.WIDTH*2, feature_scope(plan, ck))
    if meta['extra']['views_per_image'] != 8 or meta['extra']['encoder_updates'] != 0: raise ValueError('Register-Large feature metadata differs')
    return x, meta


def load_features(data, raw, plan):
    path = OUT/'features/features.npz'
    x, _ = read_bundle(path, data.ids, plan, read_check(plan))
    return x[:, eng.WIDTH:].copy(), e.file_ref(path)


def extract(device='cuda'):
    data, raw_anchor, _, plan = validate()
    eng.assets.validate()
    ck = read_check(plan)
    torch, dev, rt = prior.runtime(device)
    if rt != ck['runtime']: raise ValueError('Extraction runtime changed')
    folder = OUT/'features'; final = folder/'features.npz'
    if final.exists():
        load_features(data, raw_anchor, plan); print('Register-Large500 feature rows verified; no new forwards.'); return
    from PIL import Image
    model = None; parts = []; elapsed_saved = 0.; start_all = time.perf_counter()
    try:
        for start in range(0, len(data.ids), 16):
            ids = data.ids[start:start+16]; path = folder/f'chunks/{start:04d}.npz'
            if path.exists(): x, meta = read_bundle(path, ids, plan, ck)
            else:
                if model is None: model = eng.load_model().to(dev)
                rows = []; start_chunk = time.perf_counter()
                for ident in ids:
                    with Image.open(ROOT/'open/train'/f'{ident}.png') as im:
                        if im.mode != 'L' or im.size != (256, 256): raise ValueError('Unexpected native train image')
                        a = np.array(im, dtype=np.uint8, copy=True)
                    native = torch.from_numpy(a.astype(np.float32)/np.float32(255))[None, None].to(dev)
                    identity, mean = eng.extract_views(model, native)
                    if mean.shape != (1, eng.WIDTH): raise ValueError('Register-Large feature width mismatch')
                    rows.append(np.concatenate((identity, mean), axis=1)[0])
                x = np.stack(rows).astype(np.float32)
                meta = e.npz_save(path, ids, x, feature_scope(plan, ck),
                    {'views_per_image': 8, 'encoder_updates': 0, 'seconds': time.perf_counter()-start_chunk})
            parts.append(x); elapsed_saved += meta['extra']['seconds']
            print(f'Register-Large: {start+len(ids)}/500 extracted/verified;8views; elapsed={time.perf_counter()-start_all:.1f}s', flush=True)
        validate()
        e.npz_save(final, data.ids, np.concatenate(parts), feature_scope(plan, ck),
                   {'views_per_image': 8, 'encoder_updates': 0, 'seconds': elapsed_saved, 'runtime': rt})
        load_features(data, raw_anchor, plan)
    finally:
        if model is not None: del model
        gc.collect(); torch.cuda.empty_cache()
    print('Register-Large D4 features complete; train500x3072, no encoder updates/test forwards.')


def fit_fold(folder, scope, train_x, train_y, query_x, seed, allow_fit=True, runner=None):
    """No outer-query labels/global OOF; matched3inner alpha objective."""
    ridge = eng.d4.ridge; run = runner or ridge.fit_candidates
    x, y, q = ridge.finite(train_x), ridge.finite(train_y), ridge.finite(query_x)
    if x.ndim != 2 or q.ndim != 2 or len(x) != len(y) or x.shape[1] != q.shape[1]: raise ValueError('Feature shapes differ')
    inner = {str(i): np.full(len(y), np.nan) for i in eng.d4.INDICES}
    for k, (ti, vi) in enumerate(ridge.inner_splits(y, seed)):
        binding = {'parent': scope, 'inner': k, 'train': ti.tolist(), 'query': vi.tolist(),
                   'x': e.base.sha256_array(x[ti]), 'q': e.base.sha256_array(x[vi]),
                   'y': e.base.sha256_array(y[ti]), 'candidates': list(eng.d4.INDICES)}
        result = e.cell(folder/f'inner{k}.json', binding, lambda: run(x[ti], y[ti], x[vi], eng.d4.INDICES), allow_fit)
        if set(result['predictions']) != set(inner): raise ValueError('Unexpected candidate set')
        for i, p in result['predictions'].items(): inner[i][vi] = ridge.finite(p, (len(vi),))
    choice = eng.choose(inner, y); best = choice['selected']['index']
    binding = {'parent': scope, 'x': e.base.sha256_array(x), 'q': e.base.sha256_array(q),
               'y': e.base.sha256_array(y), 'candidates': [best]}
    result = e.cell(folder/'refit.json', binding, lambda: run(x, y, q, [best]), allow_fit)
    if set(result['predictions']) != {str(best)}: raise ValueError('Refit selection differs')
    return ridge.finite(result['predictions'][str(best)], (len(q),)), choice


def report(s):
    lines = [f'Register-Large518 D4 seed{s["seed"]}; gain>0 means improvement.',
             'PRIMARY=Large12.5%+register12.5%; SECONDARY=register25%; S50% and Radio25% fixed.']
    for n,m in s['metrics'].items(): lines.append(f'{n}: RMSE={m["rmse"]:.6f}; gain vs current={m["gain_vs_current"]:+.6f}')
    lines.append(f'Register standalone gain vs archived Large: {s["standalone_gain_vs_large"]:+.6f}')
    for name in eng.BLENDS:
        d = s['diagnosis'][name]; lines.append(name+' groups:')
        for g in (*FOCUS,'noiseQ1','noiseQ2','noiseQ3','middle80','low10','high10'):
            old,new=s['diagnosis']['current']['groups'][g],d['groups'][g]
            lines.append(f'  {g}: {old["rmse"]:.6f} -> {new["rmse"]:.6f}; gain={new["gain_vs_current"]:+.6f}')
        lines += [f'  Common16 improved={d["common16_improved"]}/16; worst20={d["common16_in_worst20"]}/16; same sign={d["common16_same_error_sign"]}/16.',
                  f'  New worst20: {d["new_worst20_ids"]}']
    return '\n'.join(lines+[PROTOCOL['selection'],PROTOCOL['decision'],PROTOCOL['limits']])+'\n'


def expected_files():
    return {'oof.csv', 'diagnostics.csv', 'common16.csv', 'per_image_errors.csv', 'summary.json', 'report.txt'} | {
        f'fold{f}/{c}.json' for f in range(5) for c in ('inner0', 'inner1', 'inner2', 'refit')}


def evaluate(seed, allow_fit=True, state=None):
    data, raw, refs, plan = state or validate()
    ref = refs[seed]; features, fr = load_features(data, raw, plan)
    scope = bind(plan, seed=seed, features=fr); folder = OUT/f'seed{seed}'; path = folder/'completed.json'
    done = path.exists()
    if done:
        complete = e.read_json(path)
        if complete['binding'] != scope or set(complete['files']) != expected_files(): raise ValueError('Completion scope changed')
        for n, h in complete['files'].items():
            if e.base.sha256_file(folder/n) != h: raise ValueError('Completed artifact changed: '+n)
    elif not allow_fit: raise FileNotFoundError('Incomplete register-Large result; no implicit fitting')
    pred = np.full(len(data.ids), np.nan); choices = []
    with threadpool_limits(limits=2):
        for f in range(5):
            ti, qi = np.flatnonzero(ref['folds'] != f), np.flatnonzero(ref['folds'] == f)
            if (len(ti), len(qi)) != (400, 100): raise ValueError('Require five400/100 folds')
            pred[qi], choice = fit_fold(folder/f'fold{f}', {**scope, 'fold': f, 'train': ti.tolist(), 'query': qi.tolist()},
                features[ti], data.y[ti], features[qi], seed+f, allow_fit and not done)
            choices.append({'fold': f, **choice})
            if not done: print(f'Outer{f}: Register-Large D4 alpha={choice["selected"]["alpha"]:g}; outer labels unused', flush=True)
    vectors = eng.compose(ref['scattering'], ref['large_d4'], ref['radio_d4'], pred)
    np.testing.assert_allclose(vectors['current'], ref['current'], atol=1e-10, rtol=0)
    mapped = {diagnostics.PRIMARY if n == PRIMARY else n: v for n, v in vectors.items()}
    s, files = diagnostics.result_data(data, ref, mapped, choices, plan, seed)
    for k in ('metrics', 'diagnosis'): s[k][PRIMARY] = s[k].pop(diagnostics.PRIMARY)
    for rows in files.values():
        for row in rows:
            for k in list(row):
                if diagnostics.PRIMARY in k: row[k.replace(diagnostics.PRIMARY, PRIMARY)] = row.pop(k)
            if row.get('model') == diagnostics.PRIMARY: row['model'] = PRIMARY
    s.update(binding=scope, primary=PRIMARY, secondary=eng.SECONDARY, selection=choices, current_d4_oof_matched=True,
             inner_uses_global_oof=False, new_tabpfn_fits=0,
             standalone_gain_vs_large=s['metrics']['large_d4_ridge']['rmse']-s['metrics']['large_register_d4_ridge']['rmse'])
    if done:
        if e.read_json(folder/'summary.json') != s or (folder/'report.txt').read_text(encoding='utf-8') != report(s): raise ValueError('Report reconstruction differs')
        for n, rows in files.items():
            with (folder/n).open(encoding='utf-8', newline='') as handle:
                if list(csv.DictReader(handle)) != list(csv.DictReader(io.StringIO(core.csv_text(rows)))): raise ValueError('CSV reconstruction differs: '+n)
        print(f'Completed seed{seed} verified; no refitting.'); return s
    for n, rows in files.items(): e.write_csv(folder/n, rows)
    e.write_json(folder/'summary.json', s); (folder/'report.txt').write_text(report(s), encoding='utf-8')
    e.write_json(path, {'binding': scope, 'files': {n: e.base.sha256_file(folder/n) for n in sorted(expected_files())}})
    print(report(s)); print('Report: '+str(folder/'report.txt')); return s


def compare(state=None, summaries=None):
    state = state or validate()
    summaries = summaries or [evaluate(seed, False, state) for seed in SEEDS]
    if [s['seed'] for s in summaries] != list(SEEDS): raise ValueError('Both fixed seeds required')
    result = {'primary': PRIMARY, 'secondary': eng.SECONDARY, 'automatic_adoption': False, 'independent_holdout': False,
        'protocol_sha256': e.digest(state[-1]['frozen']), 'comparisons': {},
        'per_seed': [{'seed':s['seed'],'metrics':s['metrics'],'diagnosis':s['diagnosis'],'standalone_gain_vs_large':s['standalone_gain_vs_large']} for s in summaries],
        'aggregation':'Mean per-seed differences, same500/reused splits; no prediction averaging or independent inference.',
        'additional_review_required':PROTOCOL['decision']}
    for name in eng.BLENDS:
        gains=[s['metrics'][name]['gain_vs_current'] for s in summaries]
        result['comparisons'][name]={'mean_gain_vs_current':float(np.mean(gains)), 'all_seeds_improve':all(v>0 for v in gains),
            'passes_expansion_priority':all(v>0 for v in gains) and float(np.mean(gains))>=.3,
            'mean_group_gains':{g:float(np.mean([s['diagnosis'][name]['groups'][g]['gain_vs_current'] for s in summaries])) for g in FOCUS},
            'all_seeds_nonworsening':{g:all(s['diagnosis'][name]['groups'][g]['gain_vs_current']>=0 for s in summaries) for g in FOCUS}}
    e.write_json(OUT/'comparison.json',result)
    print(e.base.canonical_json({k:v for k,v in result.items() if k!='per_seed'}))
    print('Comparison: '+str(OUT/'comparison.json')); return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('prepare', 'check', 'extract', 'evaluate', 'compare', 'verify'))
    p.add_argument('--device', default='cuda'); p.add_argument('--seed', type=int, choices=SEEDS)
    a = p.parse_args()
    if a.command == 'prepare': prepare()
    elif a.command == 'check': check(a.device)
    elif a.command == 'extract': extract(a.device)
    elif a.command in ('evaluate', 'verify'):
        state = validate()
        results = [evaluate(s, a.command == 'evaluate', state) for s in (SEEDS if a.seed is None else (a.seed,))]
        if a.seed is None: compare(state, results)
    else: compare()


if __name__ == '__main__': main()
