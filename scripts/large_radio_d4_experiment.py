"""User-run D4 contrast, fixed S:L:C=50:25:25; no test or submission writes."""
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
    from . import cradiov4_experiment as prior, large_radio_d4_engine as eng
else:
    import cradiov4_experiment as prior
    import large_radio_d4_engine as eng

e, core, diagnostics = prior.e, prior.core, prior.diagnostics
ROOT = e.ROOT
SOURCE = ROOT/'reports/cradiov4_importfix'
OUT = ROOT/'reports/large_radio_d4'
SEEDS = (42, 2026)
PRIMARY = eng.PRIMARY
PROTOCOL = {
    'version': 1, 'seeds': list(SEEDS), 'primary': PRIMARY,
    'views': 'Native256 gray: horizontal flip if applicable, then rot90 k=0,1,2,3. Eight pixel permutations; no random augmentation/crop/denoise.',
    'large': 'Pinned DINOv2-L518, FP32/no AMP/noTF32, batch1; existing bicubic518 without clamp, ImageNet normalization, CLS+patch mean+population std3072.',
    'radio': 'Pinned C-RADIOv4-SO400M592, FP32 weights/BF16 forward/noTF32, batch1; existing bicubic592/clamp/internal conditioner; summary+spatial mean+population std4608.',
    'pooling': 'Per-view pooling then equal8-view FP64 sum/mean ->FP32. One feature row/target per original image; original-view anchor on ALL500.',
    'anchor': 'Large rtol/atol1e-4; Radio1e-5. Original features/OOF preserved. Encoder models loaded sequentially for8GB.',
    'regression': 'Same context-only StandardScalerX/y+Ridge alpha100/1000/10000, exact ties larger alpha; no PCA.',
    'selection': 'Preserve archived objectives: Large alpha uses pooled inner standalone MSE; Radio alpha uses .5Scattering_inner+.5Radio_inner MSE. Neither is reoptimized for final50:25:25. Inner3rank-decile splits seed+outer; outer5x400/100.',
    'inner_scattering': 'Verified existing context266/267->query133/134 caches only. Never substitute global OOF. No new TabPFN fits.',
    'contrast': 'Current .5S+.25Lraw+.25Craw; replace L only,C only,both. Both-D4 is prespecified primary. No choice among these by outer score is claimed unbiased.',
    'decision': 'Overall and common16/remaining484/noiseQ1..4/tails/folds/newworst20. Mean .30 is expansion priority, NOT submission eligibility. No automatic adoption.',
    'scope': 'Train500 only, frozen encoders, no test reads, final500 regression or submission changes. User executes extract/evaluate.',
    'limits': 'Repeatedly studied500/reused splits, not independent holdout. D4 can remove useful absolute orientation.'}


def load_state():
    with contextlib.redirect_stdout(io.StringIO()):
        data, large, old, cp = prior.validate(SOURCE)
        radio, fr = prior.load_features(SOURCE, data, cp)
        for seed in SEEDS:
            prior.evaluate(SOURCE, seed, allow_fit=False)
    if prior.read_check(SOURCE, cp)['size'] != 592:
        raise ValueError('This contrast is fixed to the archived592 encoder')
    refs, files = {}, {}
    for seed in SEEDS:
        sub = SOURCE/f'seed{seed}'
        rows = core.read_rows(sub/'oof.csv', {'ID', 'hardness', 'fold', 'radio_ridge', 'radio_complementary_blend'})
        if [r['ID'] for r in rows] != data.ids.tolist():
            raise ValueError('Reference ID order differs')
        np.testing.assert_array_equal([float(r['hardness']) for r in rows], data.y)
        np.testing.assert_array_equal([int(r['fold']) for r in rows], old[seed]['folds'])
        refs[seed] = {**old[seed], 'radio': np.array([float(r['radio_ridge']) for r in rows]),
                     'current': np.array([float(r['radio_complementary_blend']) for r in rows])}
        np.testing.assert_allclose(refs[seed]['current'], .5*old[seed]['scattering']+.25*old[seed]['large']+.25*refs[seed]['radio'], atol=1e-10, rtol=0)
        files[str(seed)] = {n: e.file_ref(sub/n) for n in ('completed_evaluation.json', 'oof.csv', 'summary.json')}
    paths = [Path(__file__), Path(eng.__file__), ROOT/'scripts/run_large_radio_d4.ps1']
    frozen = {'protocol': PROTOCOL, 'sources': {p.name: e.file_ref(p) for p in paths},
              'upstream_protocol': e.file_ref(SOURCE/'protocol.json'), 'radio_features': fr,
              'large_features': cp['frozen']['large_features'], 'references': files,
              'environment': e.environment(), 'training_inventory': cp['frozen']['training_inventory'],
              'common_failure_ids': cp['frozen']['common_failure_ids'],
              'inner_reference_hashes': {str(s): {str(f): e.base.sha256_array(refs[s]['inner_scattering'][f]) for f in range(5)} for s in SEEDS},
              'submissions': {n: e.file_ref(ROOT/f'reports/{n}/submission/submission.csv') for n in ('final_scattering_blend', 'final_large518_blend', 'final_cradiov4_blend')}}
    return data, {'large': large, 'radio': radio}, refs, frozen


def prepare():
    *_, frozen = load_state()
    path = OUT/'protocol.json'
    if path.exists():
        if e.read_json(path)['frozen'] != frozen:
            raise ValueError('Prepared inputs/code changed; preserve prior results')
    else:
        if OUT.exists() and any(OUT.iterdir()):
            raise ValueError('Nonempty unprepared experiment directory')
        e.write_json(path, {'frozen': frozen, 'created_unix': time.time()})
    print('Prepared/verified D4; current OOF reconstructed, no fits/forwards.')


def validate():
    data, raw, refs, frozen = load_state()
    plan = e.read_json(OUT/'protocol.json')
    if plan['frozen'] != frozen:
        raise ValueError('Sources, inputs, environment or protocol changed')
    return data, raw, refs, plan


def bind(plan, **kw):
    return {'plan': e.digest(plan['frozen']), **kw}


def runtime(device):
    torch, dev, info = e.configure(device)
    if dev.type != 'cuda' or not torch.cuda.is_bf16_supported():
        raise ValueError('CUDA with BF16 support required for archived Radio precision')
    return torch, dev, {'device': str(dev), 'gpu': info.get('gpu_name'), 'batch_size': 1,
                       'large': 'FP32', 'radio': 'BF16-forward/FP32-weights', 'tf32': False}


def model_for(name):
    return eng.ridge.load_model('large') if name == 'large' else eng.radio.load_model()


def read_check(plan):
    obj = e.read_json(OUT/'synthetic_check.json')
    r = obj['result']
    if obj['binding'] != bind(plan) or obj['sha256'] != e.digest(r) or not r['passed'] or r['competition_forwards'] != 0:
        raise ValueError('Synthetic check missing/stale')
    if set(r['encoders']) != set(eng.NAMES):
        raise ValueError('Both encoders must pass synthetic check')
    return r


def check(device='cuda'):
    *_, plan = validate()
    torch, dev, rt = runtime(device)
    if (OUT/'synthetic_check.json').exists():
        r = read_check(plan)
        if r['runtime'] != rt:
            raise ValueError('Checked runtime differs')
        print('Synthetic check verified; no repeat.'); return
    encoders = {}
    # Synthetic asymmetric pixels only. No competition image is read here.
    yy, xx = torch.meshgrid(torch.arange(256, device=dev), torch.arange(256, device=dev), indexing='ij')
    raw = ((xx*3+yy*7+(xx//13)*(yy//17)) % 256).float()[None, None]/255.
    for name in eng.NAMES:
        model = model_for(name).to(dev)
        if any(p.requires_grad for p in model.parameters()) or model.training:
            raise ValueError('Encoder must be frozen/eval')
        torch.cuda.reset_peak_memory_stats(dev)
        eng.forward(name, model, raw); torch.cuda.synchronize(dev)
        started = time.perf_counter()
        identity, averaged = eng.extract_views(name, model, raw)
        torch.cuda.synchronize(dev)
        seconds = time.perf_counter()-started
        repeat = eng.forward(name, model, raw).cpu().numpy()
        eng.anchor(name, identity, repeat)
        if averaged.shape != (1, eng.WIDTHS[name]) or not np.isfinite(averaged).all():
            raise ValueError('Synthetic D4 output mismatch')
        encoders[name] = {'shape': list(averaged.shape), 'eight_view_seconds': seconds,
                          'encoder_frozen': True, 'peak_cuda_bytes': torch.cuda.max_memory_allocated(dev),
                          'estimated_500_image_forward_minutes': seconds*500/60}
        print(name+': '+e.base.canonical_json(encoders[name]), flush=True)
        del model
        gc.collect(); torch.cuda.empty_cache()
    r = {'passed': True, 'runtime': rt, 'competition_forwards': 0, 'competition_fits': 0, 'encoders': encoders,
         'timing_note': 'Synthetic forward extrapolation excludes disk IO, hashes, CPU Ridge, cold starts and GPU contention.'}
    e.write_json(OUT/'synthetic_check.json', {'binding': bind(plan), 'result': r, 'sha256': e.digest(r)})


def feature_scope(plan, ck, name):
    return bind(plan, stage='d4_features', encoder=name, check=e.digest(ck))


def unpack_bundle(path, ids, raw, plan, ck, name):
    width = eng.WIDTHS[name]
    x, meta = e.npz_read(path, ids, 2*width, feature_scope(plan, ck, name))
    diff = eng.anchor(name, x[:, :width], raw)
    if meta['extra']['views_per_image'] != 8 or meta['extra']['encoder_updates'] != 0:
        raise ValueError('Invalid D4 metadata')
    return x[:, width:].copy(), diff, meta


def load_features(data, raw, plan):
    ck = read_check(plan)
    values, files = {}, {}
    for name in eng.NAMES:
        p = OUT/f'features/{name}/features.npz'
        values[name], _, _ = unpack_bundle(p, data.ids, raw[name], plan, ck, name)
        files[name] = e.file_ref(p)
    return values, files


def extract(device='cuda'):
    data, raw_anchor, _, plan = validate()
    ck = read_check(plan); torch, dev, rt = runtime(device)
    if rt != ck['runtime']:
        raise ValueError('Extraction runtime changed')
    from PIL import Image
    for name in eng.NAMES:
        folder = OUT/f'features/{name}';final = folder/'features.npz'
        if final.exists():
            unpack_bundle(final, data.ids, raw_anchor[name], plan, ck, name)
            print(name+': all500 verified; no new forwards.', flush=True); continue
        model = None; parts = []; chunk_seconds = 0.; started = time.perf_counter(); maxdiff = 0.
        try:
            for start in range(0, len(data.ids), 16):
                ids = data.ids[start:start+16]; path = folder/f'chunks/{start:04d}.npz'
                scope = feature_scope(plan, ck, name)
                if path.exists():
                    x, meta = e.npz_read(path, ids, 2*eng.WIDTHS[name], scope)
                    if meta['extra']['views_per_image'] != 8 or meta['extra']['encoder_updates'] != 0:
                        raise ValueError('Partial chunk protocol mismatch')
                else:
                    if model is None:
                        model = model_for(name).to(dev)
                    rows = []; t0 = time.perf_counter()
                    for offset, ident in enumerate(ids):
                        with Image.open(ROOT/'open/train'/f'{ident}.png') as im:
                            if im.mode != 'L' or im.size != (256, 256):
                                raise ValueError('Unexpected native image')
                            a = np.array(im, dtype=np.uint8, copy=True)
                        # Match the two archived arithmetic paths exactly, including
                        # where uint8/255 conversion happens (CPU vs CUDA).
                        if name == 'large':
                            native = torch.from_numpy(a.astype(np.float32)/np.float32(255))[None, None].to(dev)
                        else:
                            native = torch.from_numpy(a).to(device=dev, dtype=torch.float32)[None, None]/255.
                        identity, mean = eng.extract_views(name, model, native)
                        eng.anchor(name, identity, raw_anchor[name][start+offset:start+offset+1])
                        rows.append(np.concatenate((identity, mean), axis=1)[0])
                    x = np.stack(rows).astype(np.float32)
                    meta = e.npz_save(path, ids, x, scope, {'views_per_image': 8, 'encoder_updates': 0,
                                                        'seconds': time.perf_counter()-t0})
                diff = eng.anchor(name, x[:, :eng.WIDTHS[name]], raw_anchor[name][start:start+len(ids)])
                maxdiff = max(maxdiff, diff); parts.append(x);chunk_seconds += meta['extra']['seconds']
                elapsed = time.perf_counter()-started
                print(f'{name}: {start+len(ids)}/500 extracted/verified; 8 views; elapsed={elapsed:.1f}s; identity maxdiff={maxdiff:.3g}', flush=True)
            # Detect input mutations during long-running extraction.
            validate()
            e.npz_save(final, data.ids, np.concatenate(parts), feature_scope(plan, ck, name),
                       {'views_per_image': 8, 'encoder_updates': 0, 'seconds': chunk_seconds,
                        'max_identity_absdiff': maxdiff, 'runtime': rt})
            unpack_bundle(final, data.ids, raw_anchor[name], plan, ck, name)
        finally:
            if model is not None:
                del model
            gc.collect(); torch.cuda.empty_cache()
    load_features(data, raw_anchor, plan)
    print('Both D4 feature bundles complete. No encoder updates/test access.')


def fit_fold(folder, scope, train_x, train_y, query_x, inner_scattering, seed, allow_fit=True, runner=None):
    """No outer-query labels, whole-dataset targets or global OOF accepted."""
    run = runner or eng.ridge.fit_candidates
    y = eng.ridge.finite(train_y); s = eng.ridge.finite(inner_scattering, y.shape)
    if set(train_x) != set(eng.NAMES) or set(query_x) != set(eng.NAMES):
        raise ValueError('Both D4 representations required')
    splits = eng.ridge.inner_splits(y, seed)
    predictions, choices = {}, {}
    for name in eng.NAMES:
        x, q = eng.ridge.finite(train_x[name]), eng.ridge.finite(query_x[name])
        if len(x) != len(y) or x.ndim != 2 or q.ndim != 2 or x.shape[1] != q.shape[1]:
            raise ValueError('Feature shape differs')
        inner = {str(i): np.full(len(y), np.nan) for i in eng.INDICES}
        for k, (ti, vi) in enumerate(splits):
            binding = {'parent': scope, 'name': name, 'inner': k, 'train': ti.tolist(), 'query': vi.tolist(),
                       'x': e.base.sha256_array(x[ti]), 'q': e.base.sha256_array(x[vi]),
                       'y': e.base.sha256_array(y[ti]), 'candidates': list(eng.INDICES)}
            result = e.cell(folder/name/f'inner{k}.json', binding,
                            lambda: run(x[ti], y[ti], x[vi], eng.INDICES), allow_fit)
            if set(result['predictions']) != set(inner):
                raise ValueError('Candidate prediction schema changed')
            for i, p in result['predictions'].items():
                inner[i][vi] = eng.ridge.finite(p, (len(vi),))
        choice = eng.choose(name, inner, y, s); best = choice['selected']['index']
        binding = {'parent': scope, 'name': name, 'x': e.base.sha256_array(x), 'q': e.base.sha256_array(q),
                   'y': e.base.sha256_array(y), 'inner_scattering': e.base.sha256_array(s), 'candidates': [best]}
        result = e.cell(folder/name/'refit.json', binding, lambda: run(x, y, q, [best]), allow_fit)
        if set(result['predictions']) != {str(best)}:
            raise ValueError('Refit selection differs')
        predictions[name] = eng.ridge.finite(result['predictions'][str(best)], (len(q),))
        choices[name] = choice
    return predictions, choices


def report(s):
    lines = [f'Large518/C-RADIO592 D4 contrast seed{s["seed"]}', 'Primary=both_d4_blend; gain>0 means improvement.']
    for name, m in s['metrics'].items():
        lines.append(f'{name}: RMSE={m["rmse"]:.6f}; gain vs current={m["gain_vs_current"]:+.6f}')
    for name in eng.BLENDS:
        d = s['diagnosis'][name];lines.append(name+' groups:')
        for g in ('common16', 'remaining484', 'noiseQ4', 'middle80', 'low10', 'high10'):
            m = d['groups'][g];lines.append(f'  {g}: RMSE={m["rmse"]:.6f}; gain={m["gain_vs_current"]:+.6f}')
        lines.append(f'  common16 improved={d["common16_improved"]}/16; in worst20={d["common16_in_worst20"]}/16; same error sign={d["common16_same_error_sign"]}/16.')
    return '\n'.join(lines+[PROTOCOL['selection'], PROTOCOL['decision'], PROTOCOL['limits']])+'\n'


def expected_files():
    return {'oof.csv', 'diagnostics.csv', 'common16.csv', 'per_image_errors.csv', 'summary.json', 'report.txt'} | {
        f'fold{f}/{n}/{c}.json' for f in range(5) for n in eng.NAMES for c in ('inner0', 'inner1', 'inner2', 'refit')}


def evaluate(seed, allow_fit=True, state=None):
    data, raw, refs, plan = state or validate()
    ref = refs[seed];features, fr = load_features(data, raw, plan)
    scope = bind(plan, seed=seed, features=fr);folder = OUT/f'seed{seed}';path = folder/'completed.json'
    done = path.exists()
    if done:
        complete = e.read_json(path)
        if complete['binding'] != scope or set(complete['files']) != expected_files():
            raise ValueError('Completion scope changed')
        for n, h in complete['files'].items():
            if e.base.sha256_file(folder/n) != h:
                raise ValueError('Completed artifact changed: '+n)
    elif not allow_fit:
        raise FileNotFoundError('Incomplete result; no implicit fitting')
    p = {n: np.full(len(data.ids), np.nan) for n in eng.NAMES};choices = []
    with threadpool_limits(limits=2):
        for fold in range(5):
            ti = np.flatnonzero(ref['folds'] != fold);qi = np.flatnonzero(ref['folds'] == fold)
            if (len(ti), len(qi)) != (400, 100):
                raise ValueError('Expected five400/100 folds')
            pp, cc = fit_fold(folder/f'fold{fold}', {**scope, 'fold': fold, 'train': ti.tolist(), 'query': qi.tolist()},
                              {n: features[n][ti] for n in eng.NAMES}, data.y[ti],
                              {n: features[n][qi] for n in eng.NAMES}, ref['inner_scattering'][fold],
                              seed+fold, allow_fit and not done)
            for n in p:
                p[n][qi] = pp[n]
            choices.append({'fold': fold, 'choices': cc})
            print(f'Outer{fold}: Large D4 alpha={cc["large"]["selected"]["alpha"]:g}; Radio D4 alpha={cc["radio"]["selected"]["alpha"]:g}; outer labels unused', flush=True)
    vectors = eng.compose(ref['scattering'], ref['large'], ref['radio'], p['large'], p['radio'])
    np.testing.assert_allclose(vectors['current'], ref['current'], atol=1e-10, rtol=0)
    mapped = {diagnostics.PRIMARY if n == PRIMARY else n: v for n, v in vectors.items()}
    s, files = diagnostics.result_data(data, ref, mapped, choices, plan, seed)
    for key in ('metrics', 'diagnosis'):
        s[key][PRIMARY] = s[key].pop(diagnostics.PRIMARY)
    for rows in files.values():
        for row in rows:
            for k in list(row):
                if diagnostics.PRIMARY in k:
                    row[k.replace(diagnostics.PRIMARY, PRIMARY)] = row.pop(k)
            if row.get('model') == diagnostics.PRIMARY:
                row['model'] = PRIMARY
    s.update(binding=scope, primary=PRIMARY, selection=choices, original_oof_anchor_matched=True,
             inner_uses_global_oof=False, new_tabpfn_fits=0)
    if done:
        if e.read_json(folder/'summary.json') != s or (folder/'report.txt').read_text(encoding='utf-8') != report(s):
            raise ValueError('Saved report reconstruction differs')
        for n, rows in files.items():
            with (folder/n).open(encoding='utf-8', newline='') as f:
                if list(csv.DictReader(f)) != list(csv.DictReader(io.StringIO(core.csv_text(rows)))):
                    raise ValueError('Saved CSV reconstruction differs: '+n)
        print(f'Completed seed{seed} verified; no refitting.');return s
    for n, rows in files.items():
        e.write_csv(folder/n, rows)
    e.write_json(folder/'summary.json', s);(folder/'report.txt').write_text(report(s), encoding='utf-8')
    e.write_json(path, {'binding': scope, 'files': {n: e.base.sha256_file(folder/n) for n in sorted(expected_files())}})
    print(report(s));print('Report: '+str(folder/'report.txt'));return s


def compare():
    state = validate();summaries = [evaluate(seed, False, state) for seed in SEEDS]
    result = {'primary': PRIMARY, 'automatic_adoption': False, 'independent_holdout': False,
              'protocol_sha256': e.digest(state[-1]['frozen']), 'comparisons': {},
              'per_seed': [{'seed': s['seed'], 'metrics': s['metrics'], 'diagnosis': s['diagnosis']} for s in summaries],
              'aggregation': 'Mean of per-seed differences, same500/reused splits; no prediction averaging or independent inference.',
              'additional_review_required': PROTOCOL['decision']}
    for n in eng.BLENDS:
        gains = [s['metrics'][n]['gain_vs_current'] for s in summaries]
        result['comparisons'][n] = {'mean_gain_vs_current': float(np.mean(gains)), 'all_seeds_improve': all(v > 0 for v in gains),
            'mean_group_gains': {g: float(np.mean([s['diagnosis'][n]['groups'][g]['gain_vs_current'] for s in summaries])) for g in ('common16', 'remaining484', 'noiseQ4')},
            'all_seeds_nonworsening': {g: all(s['diagnosis'][n]['groups'][g]['gain_vs_current'] >= 0 for s in summaries) for g in ('common16', 'remaining484', 'noiseQ4')}}
    e.write_json(OUT/'comparison.json', result)
    print(e.base.canonical_json({k: v for k, v in result.items() if k != 'per_seed'}))
    print('Comparison: '+str(OUT/'comparison.json'))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('prepare', 'check', 'extract', 'evaluate', 'compare', 'verify'))
    p.add_argument('--device', default='cuda');p.add_argument('--seed', type=int, choices=SEEDS, default=42)
    a = p.parse_args()
    if a.command == 'prepare':prepare()
    elif a.command == 'check':check(a.device)
    elif a.command == 'extract':extract(a.device)
    elif a.command == 'evaluate':evaluate(a.seed)
    elif a.command == 'verify':evaluate(a.seed, allow_fit=False)
    else:compare()


if __name__ == '__main__':main()
