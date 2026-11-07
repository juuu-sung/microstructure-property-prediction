"""User-run final S50 + Large518-D4 Ridge25 + C-RADIO-D4 Ridge25.

Preparation only verifies inputs. Reuse verified train500 D4 features and old
full500 cross-fitted Scattering contexts. Fit two Ridge heads, extract test D4
features sequentially, and write a separate submission. Never upload.
"""
from __future__ import annotations
import argparse
import contextlib
import gc
import io
import os
from pathlib import Path
import time
import numpy as np
from threadpoolctl import threadpool_limits
if __package__:
    from . import large_radio_d4_experiment as exp
    from . import final_cradiov4_submission as old
else:
    import large_radio_d4_experiment as exp
    import final_cradiov4_submission as old

e, eng, core, ROOT = exp.e, exp.eng, exp.core, exp.ROOT
numeric, shared = old.previous, old.shared
OUT = ROOT/'reports/final_large_radio_d4_blend'
SEED, ANCHORS = 31415, 16
PROTOCOL = {
    'version': 1, 'selection_seed': SEED, 'alpha_candidates': [100., 1000., 10000.],
    'selection': 'Same3 rank-stratified folds within all500. Large: pooled standalone MSE. Radio: pooled .5S_inner+.5C_inner MSE. Exact ties prefer larger alpha. No hardcoded winner or final-blend reoptimization.',
    'inner_scattering': 'Reuse verified full500 seed31415 context333/334->query166/167 caches from final_cradiov4_blend. Context-only nested base fitting. Never global OOF or full500 in-sample S. Missing/stale cache fails; no implicit TabPFN refits.',
    'train': 'Reuse500 D4 feature rows. Each encoder:9 CV Ridge fits+1 all500 selected Ridge; train-only X/y scalers. No encoder updates.',
    'planned_fits': {'ridge_cv': 18, 'ridge_full500': 2, 'encoder': 0, 'tabpfn': 0},
    'large': exp.PROTOCOL['large'], 'radio': exp.PROTOCOL['radio'],
    'views': exp.PROTOCOL['views'], 'pooling': exp.PROTOCOL['pooling'],
    'anchors': 'First16 train IDs, both identity and D4 mean checked before each encoder test extraction; Large rtol/atol1e-4, Radio1e-5.',
    'blend': {'scattering486': .5, 'large518_d4_ridge': .25, 'cradiov4_d4_ridge': .25},
    'effective_weights': {'hand185': .0625, 'originalResNet': .0625, 'DINO_small_D4': .125, 'TabPFN486': .25, 'Large518_D4_Ridge': .25, 'CRADIO_D4_Ridge': .25},
    'score_scope': 'Final alpha-selection CV is a tuning diagnostic, not independent performance.',
    'test': 'User-run1000 images; eight views each; batch1; sequential encoders; no test fitting or model selection.',
    'preservation': 'Reuse the entire verified Scattering486 ensemble prediction. Preserve all earlier submissions and models. Separate output tree, fixed50:25:25, no clipping/calibration/uploads/downloads.',
    'resume': 'Verify file/array/source/ID hashes; reuse completed heads, CV cells and per-image feature chunks. Fail on changed bindings.'}


def protect_destination(path):
    root, dest = OUT.resolve(), Path(path).resolve()
    if not root.is_relative_to((ROOT/'reports').resolve()) or not (dest == root or dest.is_relative_to(root)):
        raise ValueError('D4 final outputs must stay in their separate reports tree')


def binding_for(plan, stage, name=None):
    b = {'plan': e.digest(plan['frozen']), 'stage': stage}
    if name is not None: b['encoder'] = name
    return b


def reuse_inner_scattering(data, raw, old_plan, old_selection):
    """Read-only reconstruction; cached context scope includes every train/query ID."""
    if old.SEED != SEED:
        raise ValueError('Archived Scattering split seed differs')
    splits = eng.ridge.inner_splits(data.y, SEED)
    for a, b in zip(splits, old.residual.eng.splits(data.y, SEED), strict=True):
        for u, v in zip(a, b, strict=True): np.testing.assert_array_equal(u, v)
    binding = old.binding_for(old_plan, 'full500_radio_ridge')
    with contextlib.redirect_stdout(io.StringIO()):
        cross = old.residual.crossfit_components(old.OUT/'model/cv/baseline', binding, raw,
            np.arange(len(data.y)), data.y, SEED, 0, old.c.forbidden, False)
    s = old.residual.eng.ensemble(cross)
    if e.base.sha256_array(s) != old_selection['inner_scattering_sha256']:
        raise ValueError('Archived context-only Scattering predictions changed')
    return eng.ridge.finite(s, data.y.shape)


def read_current(ids, scattering, large, old_plan):
    folder = old.OUT/'submission';done = e.read_json(folder/'completed.json');binding = done['binding']
    expected = {**old.binding_for(old_plan, 'submission'),
        'model': e.file_ref(old.OUT/'model/completed.json'),
        'test_features': e.file_ref(old.OUT/'test_features/completed.json'),
        'alpha': e.read_json(old.OUT/'model/selection.json')['alpha'],
        'scattering_submission_sha256': numeric.SCATTERING_SHA, 'large_submission_sha256': old.LARGE_SUBMISSION_SHA}
    if binding != expected or not numeric.verify_manifest(folder, binding, ('submission.csv', 'component_predictions.csv', 'validation.json')):
        raise ValueError('Preserved S:L:C submission provenance differs')
    if not numeric.verify_manifest(old.OUT/'test_features', old.binding_for(old_plan, 'test_radio'), ('features.npz', 'anchor.npz')):
        raise ValueError('Preserved Radio test features incomplete')
    rows = core.read_rows(folder/'component_predictions.csv', {'ID', 'scattering486', 'large518_ridge', 'cradiov4_ridge', 'hardness'})
    if [r['ID'] for r in rows] != list(ids): raise ValueError('Preserved sample order differs')
    col = lambda n: core.finite_column(rows, n, folder)
    np.testing.assert_array_equal(scattering, col('scattering486'))
    np.testing.assert_array_equal(large, col('large518_ridge'))
    p = .5*scattering+.25*large+.25*col('cradiov4_ridge')
    np.testing.assert_array_equal(p, col('hardness')); shared.verify_submission(folder/'submission.csv', ids, p)
    audit = e.read_json(folder/'validation.json')
    if audit['weights'] != old.PROTOCOL['blend'] or audit['submission_sha256'] != e.base.sha256_file(folder/'submission.csv'):
        raise ValueError('Preserved S:L:C audit changed')
    return p, {n: e.file_ref(folder/n) for n in ('submission.csv', 'component_predictions.csv', 'validation.json', 'completed.json')}


def load_state():
    with contextlib.redirect_stdout(io.StringIO()):
        state = exp.validate(); data, raw, _, dp = state
        features, files = exp.load_features(data, raw, dp)
        summaries = [exp.evaluate(seed, allow_fit=False, state=state) for seed in exp.SEEDS]
        od, ox, base, scattering, large, op = old.validate()
        _, previous_selection = old.load_head(old.OUT, op, od, ox, base)
    np.testing.assert_array_equal(data.ids, od.ids); np.testing.assert_array_equal(data.y, od.y)
    np.testing.assert_array_equal(raw['radio'], ox)
    if len(data.ids) != 500 or any(features[n].shape != (500, eng.WIDTHS[n]) for n in eng.NAMES):
        raise ValueError('Expected all500 D4 training rows')
    # These are training-only confirmation files; no fitting/forwards in prepare.
    comparison = e.read_json(exp.OUT/'comparison.json')
    expected_seeds = [{'seed': s['seed'], 'metrics': s['metrics'], 'diagnosis': s['diagnosis']} for s in summaries]
    if comparison['protocol_sha256'] != e.digest(dp['frozen']) or comparison['per_seed'] != expected_seeds:
        raise ValueError('Reviewed D4 result changed')
    inner_s = reuse_inner_scattering(data, base, op, previous_selection)
    current, previous_files = read_current(op['frozen']['test']['ids'], scattering, large, op)
    ck = exp.read_check(dp)
    frozen = {'protocol': PROTOCOL,
        'sources': {p.name: e.file_ref(p) for p in (Path(__file__), ROOT/'scripts/run_final_large_radio_d4.ps1', Path(old.__file__), Path(numeric.__file__), Path(shared.__file__))},
        'd4_protocol': e.file_ref(exp.OUT/'protocol.json'), 'd4_protocol_payload': e.digest(dp['frozen']),
        'd4_check': e.file_ref(exp.OUT/'synthetic_check.json'), 'train_runtime': ck['runtime'],
        'd4_comparison': e.file_ref(exp.OUT/'comparison.json'),
        'd4_completed': {str(s): e.file_ref(exp.OUT/f'seed{s}/completed.json') for s in exp.SEEDS},
        'train_features': files, 'feature_sha256': {n: e.base.sha256_array(features[n]) for n in eng.NAMES},
        'ids_sha256': e.base.sha256_array(data.ids), 'target_sha256': e.base.sha256_array(data.y),
        'old_final_plan': e.file_ref(old.OUT/'plan.json'), 'old_final_model': e.file_ref(old.OUT/'model/completed.json'),
        'inner_scattering': {'sha256': e.base.sha256_array(inner_s), 'seed': SEED,
            'contexts': {str(k): e.file_ref(old.OUT/f'model/cv/baseline/inner{k}/completed.json') for k in range(3)}},
        'test': op['frozen']['test'], 'preserved_previous': previous_files,
        'other_preserved': op['frozen']['previous'], 'earlier_submissions': op['frozen']['earlier_submissions'],
        'anchor_ids': data.ids[:ANCHORS].tolist(),
        'anchor_sha256': {n: e.base.sha256_array(np.concatenate((raw[n][:ANCHORS], features[n][:ANCHORS]), axis=1)) for n in eng.NAMES}}
    return data, features, raw, inner_s, scattering, current, frozen


def prepare(folder=OUT):
    protect_destination(folder); *_, frozen = load_state(); path = folder/'plan.json'
    if path.exists():
        if e.read_json(path)['frozen'] != frozen: raise ValueError('Prepared D4 final plan changed; preserve artifacts')
    else:
        if folder.exists() and any(folder.iterdir()): raise ValueError('Nonempty unprepared output directory')
        e.write_json(path, {'frozen': frozen, 'created_unix': time.time(), 'fits_during_preparation': 0, 'forwards_during_preparation': 0})
    print('Prepared S50/LargeD4-25/RadioD4-25; cached train features and full500 Scattering inner contexts verified. No fitting/forwards.')


def validate(folder=OUT):
    protect_destination(folder); *state, frozen = load_state(); plan = e.read_json(folder/'plan.json')
    if plan['frozen'] != frozen or plan['fits_during_preparation'] != 0 or plan['forwards_during_preparation'] != 0:
        raise ValueError('Final D4 sources, inputs, runtime or preserved files changed')
    return (*state, plan)


def select_alpha(folder, binding, name, x, y, inner_s, allow_fit=True, runner=None):
    """Training arrays only; no test predictions, labels or mixture search."""
    run = runner or eng.ridge.fit_candidates
    y = eng.ridge.finite(y, (len(x),)); s = eng.ridge.finite(inner_s, y.shape)
    splits = eng.ridge.inner_splits(y, SEED)
    inner = {str(i): np.full(len(y), np.nan) for i in eng.INDICES}
    for k, (ti, vi) in enumerate(splits):
        scope = {'parent': binding, 'name': name, 'inner': k, 'train': ti.tolist(), 'query': vi.tolist(),
                 'x': e.base.sha256_array(x[ti]), 'q': e.base.sha256_array(x[vi]), 'y': e.base.sha256_array(y[ti]), 'candidates': list(eng.INDICES)}
        result = e.cell(folder/f'inner{k}.json', scope, lambda: run(x[ti], y[ti], x[vi], eng.INDICES), allow_fit)
        if set(result['predictions']) != set(inner): raise ValueError('Expected exact3 Ridge candidates')
        for key, p in result['predictions'].items(): inner[key][vi] = eng.ridge.finite(p, (len(vi),))
    choices = eng.choose(name, inner, y, s)
    return {'alpha': choices['selected']['alpha'], 'choices': choices, 'selection_seed': SEED,
            'selection_rmse': float(np.sqrt(choices['selected']['selection_mse'])),
            'n_train': len(y), 'cv_validation_sizes': [len(v) for _, v in splits],
            'inner_scattering_sha256': e.base.sha256_array(s), 'test_used_for_fit': False, 'score_scope': PROTOCOL['score_scope']}


def model_names():
    return ('heads.npz', 'metadata.json', 'selection.json', 'cv/inner0.json', 'cv/inner1.json', 'cv/inner2.json')


def load_head(folder, plan, name, x, y, inner_s):
    out = folder/'models'/name; binding = binding_for(plan, 'full500_ridge', name)
    if not numeric.verify_manifest(out, binding, model_names()): raise FileNotFoundError('Final head incomplete; run train')
    head, meta = shared.read_archive(out, 'heads.npz')
    selection = select_alpha(out/'cv', binding, name, x, y, inner_s, False, old.c.forbidden)
    if selection != e.read_json(out/'selection.json') or meta['selection'] != selection or meta['binding'] != binding or meta['test_used_for_fit'] is not False:
        raise ValueError('Head selection/provenance changed')
    if set(head) != numeric.HEAD_KEYS or selection['alpha'] not in PROTOCOL['alpha_candidates']:
        raise ValueError('Invalid Ridge head schema')
    for key, value in head.items():
        shape = (x.shape[1],) if key in ('x_mean', 'x_scale', 'coef') else ()
        if value.shape != shape or value.dtype != np.float64 or not np.isfinite(value).all(): raise ValueError('Invalid numeric head')
    if np.any(head['x_scale'] <= 0) or head['y_scale'] <= 0: raise ValueError('Invalid scale')
    return head, selection


def train(folder=OUT):
    data, features, _, s, _, _, plan = validate(folder)
    for name in eng.NAMES:
        x = features[name]; out = folder/'models'/name; binding = binding_for(plan, 'full500_ridge', name)
        if (out/'completed.json').exists():
            _, sel = load_head(folder, plan, name, x, data.y, s)
            print(f'{name}: final alpha={sel["alpha"]:g} verified; no refitting.', flush=True); continue
        with threadpool_limits(limits=2):
            sel = select_alpha(out/'cv', binding, name, x, data.y, s)
            head = numeric.fit_numeric_head(x, data.y, sel['alpha'])
        validate(folder)
        shared.save_archive(out, head, {'binding': binding, 'selection': sel, 'test_used_for_fit': False,
            'array_sha256': {k: e.base.sha256_array(v) for k, v in head.items()}}, 'heads.npz')
        e.write_json(out/'selection.json', sel); numeric.finish(out, binding, model_names())
        load_head(folder, plan, name, x, data.y, s)
        print(f'{name}: all500 fitted; alpha={sel["alpha"]:g}; selectionRMSE={sel["selection_rmse"]:.6f} (tuning diagnostic).', flush=True)


class D4Extractor:
    def __init__(self, name, device, expected_runtime):
        torch, dev, runtime = exp.runtime(device)
        if runtime != expected_runtime or name not in eng.NAMES: raise ValueError('Use locked train runtime; no fallback')
        self.name, self.torch, self.device, self.runtime = name, torch, dev, runtime
        self.model = exp.model_for(name).to(dev)
        if any(m.training for m in self.model.modules()) or any(p.requires_grad for p in self.model.parameters()):
            raise ValueError('Encoder must be frozen/eval')

    def __call__(self, split, ids):
        from PIL import Image
        if split not in ('train', 'test'): raise ValueError('Invalid split')
        rows = []; torch = self.torch
        for ident in ids:
            with Image.open(ROOT/'open'/split/(str(ident)+'.png')) as im:
                if im.mode != 'L' or im.size != (256, 256): raise ValueError('Expected256 grayscale')
                a = np.array(im, dtype=np.uint8, copy=True)
            # Preserve archived CPU vs CUDA uint8/255 arithmetic exactly.
            if self.name == 'large':
                native = torch.from_numpy(a.astype(np.float32)/np.float32(255))[None, None].to(self.device)
            else:
                native = torch.from_numpy(a).to(device=self.device, dtype=torch.float32)[None, None]/255.
            identity, mean = eng.extract_views(self.name, self.model, native)
            rows.append(np.concatenate((identity, mean), axis=1))
        return np.concatenate(rows).astype(np.float32)

    def close(self):
        del self.model; gc.collect(); self.torch.cuda.empty_cache()


def anchor_record(folder, plan, name, expected, extractor=None):
    ids = np.asarray(plan['frozen']['anchor_ids']); width = eng.WIDTHS[name]
    binding = {'parent': binding_for(plan, 'test_d4', name), 'expected_sha256': e.base.sha256_array(expected)}
    if binding['expected_sha256'] != plan['frozen']['anchor_sha256'][name]: raise ValueError('Expected anchor changed')
    path = folder/'test_features'/name/'anchor.npz'; exists = path.exists()
    if exists: actual, _ = e.npz_read(path, ids, 2*width, binding)
    else:
        if extractor is None: raise FileNotFoundError('Train anchor missing; run extract')
        actual = extractor('train', ids)
    # Both untransformed feature and D4 mean must reproduce train extraction.
    diff = eng.anchor(name, actual, expected)
    if actual.dtype != np.float32: raise ValueError('Anchor precision differs')
    if not exists: e.npz_save(path, ids, actual, binding, {'passed': True, 'max_abs_difference': diff})
    return e.file_ref(path)


def test_chunk(folder, plan, name, index, extractor=None):
    item = plan['frozen']['test']['images'][index];source = ROOT/'open'/item['relative_path']
    if e.base.sha256_file(source) != item['sha256']: raise ValueError('Test source changed: '+item['id'])
    ids = np.asarray([item['id']]); width = eng.WIDTHS[name]
    binding = {'parent': binding_for(plan, 'test_d4', name), 'image': item, 'offset': index}
    path = folder/'test_features'/name/'chunks'/f'{index:04d}.npz'
    if path.exists(): return e.npz_read(path, ids, width, binding)[0]
    if extractor is None: raise FileNotFoundError('Test chunk missing; run extract')
    bundle = extractor('test', ids)
    if bundle.shape != (1, 2*width) or bundle.dtype != np.float32 or not np.isfinite(bundle).all(): raise ValueError('Invalid D4 bundle')
    if e.base.sha256_file(source) != item['sha256']: raise ValueError('Test source changed during extraction')
    x = bundle[:, width:].copy(); e.npz_save(path, ids, x, binding); return x


def load_test_cache(folder, plan, name, anchor):
    out = folder/'test_features'/name; binding = binding_for(plan, 'test_d4', name)
    if not numeric.verify_manifest(out, binding, ('features.npz', 'anchor.npz')): raise FileNotFoundError('Test cache incomplete; run extract')
    x, meta = e.npz_read(out/'features.npz', np.asarray(plan['frozen']['test']['ids']), eng.WIDTHS[name], binding)
    extra = meta['extra']
    if (extra['runtime'] != plan['frozen']['train_runtime'] or extra['views'] != 8 or extra['test_used_for_fit'] is not False
            or extra['anchor'] != anchor_record(folder, plan, name, anchor)):
        raise ValueError('D4 test provenance changed')
    if set(extra['chunks']) != {f'{i:04d}.npz' for i in range(len(x))}: raise ValueError('Incomplete chunk manifest')
    for i in range(len(x)):
        if e.base.sha256_file(out/'chunks'/f'{i:04d}.npz') != extra['chunks'][f'{i:04d}.npz']: raise ValueError('Chunk changed')
        np.testing.assert_array_equal(x[i:i+1], test_chunk(folder, plan, name, i))
    return x


def extract(folder=OUT, device='cuda'):
    data, features, raw, s, _, _, plan = validate(folder)
    for name in eng.NAMES: load_head(folder, plan, name, features[name], data.y, s)
    for name in eng.NAMES:
        anchor = np.concatenate((raw[name][:ANCHORS], features[name][:ANCHORS]), axis=1)
        out = folder/'test_features'/name
        if (out/'completed.json').exists():
            load_test_cache(folder, plan, name, anchor); print(name+': test cache verified; no forwards.', flush=True); continue
        extractor = D4Extractor(name, device, plan['frozen']['train_runtime']); started = time.perf_counter()
        try:
            anchor_ref = anchor_record(folder, plan, name, anchor, extractor)
            print(name+': first16 identity+D4 anchors matched; starting/resuming test.', flush=True)
            rows = []
            for i in range(len(plan['frozen']['test']['ids'])):
                rows.append(test_chunk(folder, plan, name, i, extractor))
                if i == 0 or (i+1)%50 == 0: print(f'{name}: test{i+1}/{len(plan["frozen"]["test"]["ids"])}; seconds={time.perf_counter()-started:.1f}', flush=True)
            validate(folder)
            e.npz_save(out/'features.npz', np.asarray(plan['frozen']['test']['ids']), np.concatenate(rows), binding_for(plan, 'test_d4', name),
                {'anchor': anchor_ref, 'runtime': extractor.runtime, 'views': 8, 'test_used_for_fit': False,
                 'seconds': time.perf_counter()-started,
                 'chunks': {f'{i:04d}.npz': e.base.sha256_file(out/'chunks'/f'{i:04d}.npz') for i in range(len(rows))}})
            numeric.finish(out, binding_for(plan, 'test_d4', name), ('features.npz', 'anchor.npz'))
        finally: extractor.close()
        load_test_cache(folder, plan, name, anchor)
    print('Both D4 test caches ready. No test fitting.')


def write_submission(out, ids, scattering, large, radio, current, binding):
    protect_destination(out)
    s, l, c, old_pred = [eng.ridge.finite(v, (len(ids),)) for v in (scattering, large, radio, current)]
    if len(set(ids)) != len(ids): raise ValueError('Duplicate sample IDs')
    p = .5*s+.25*l+.25*c
    rows = [{'ID': str(ident), 'scattering486': float(s[i]), 'large518_d4_ridge': float(l[i]),
             'cradiov4_d4_ridge': float(c[i]), 'hardness': float(p[i])} for i, ident in enumerate(ids)]
    names = ('submission.csv', 'component_predictions.csv', 'validation.json')
    if numeric.verify_manifest(out, binding, names):
        shared.verify_submission(out/'submission.csv', ids, p); numeric.same_csv(out/'component_predictions.csv', rows)
        print('Separate D4 submission verified; unchanged.'); return
    out.mkdir(parents=True, exist_ok=True);temp = out/'submission.csv.tmp'
    temp.write_text(shared.submission_text(ids, p), encoding='utf-8', newline='')
    shared.verify_submission(temp, ids, p); os.replace(temp, out/'submission.csv')
    e.write_csv(out/'component_predictions.csv', rows)
    e.write_json(out/'validation.json', {'binding': binding, 'rows': len(ids), 'columns': ['ID', 'hardness'],
        'weights': PROTOCOL['blend'], 'effective_weights': PROTOCOL['effective_weights'],
        'sample_order_matches': True, 'all_finite': True, 'csv_round_trip_exact': True,
        'uploaded': False, 'clipping_or_calibration': False, 'test_used_for_fit': False,
        'prediction_min': float(p.min()), 'prediction_max': float(p.max()), 'prediction_mean': float(p.mean()),
        'prediction_std': float(p.std()), 'mean_absolute_change_vs_current': float(np.abs(p-old_pred).mean()),
        'max_absolute_change_vs_current': float(np.abs(p-old_pred).max()),
        'submission_sha256': e.base.sha256_file(out/'submission.csv')})
    numeric.finish(out, binding, names)
    print(f'Submission ready: {out/"submission.csv"}; rows={len(ids)}; S50/LargeD4-25/RadioD4-25; ID order/finite/CSV round-trip verified. Prior submissions preserved.')


def predict(folder=OUT):
    data, features, raw, s, scattering, current, plan = validate(folder)
    p, alphas = {}, {}
    for name in eng.NAMES:
        head, sel = load_head(folder, plan, name, features[name], data.y, s)
        anchor = np.concatenate((raw[name][:ANCHORS], features[name][:ANCHORS]), axis=1)
        test_x = load_test_cache(folder, plan, name, anchor)
        with threadpool_limits(limits=2): p[name] = shared.predict_head(head, test_x)
        alphas[name] = sel['alpha']
    validate(folder)
    binding = {**binding_for(plan, 'submission'), 'alphas': alphas,
        'models': {n: e.file_ref(folder/'models'/n/'completed.json') for n in eng.NAMES},
        'test_features': {n: e.file_ref(folder/'test_features'/n/'completed.json') for n in eng.NAMES},
        'scattering_submission_sha256': numeric.SCATTERING_SHA}
    write_submission(folder/'submission', plan['frozen']['test']['ids'], scattering, p['large'], p['radio'], current, binding)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=('prepare', 'validate', 'train', 'extract', 'predict'))
    p.add_argument('--device', default='cuda'); a = p.parse_args()
    if a.stage == 'prepare': prepare()
    elif a.stage == 'validate': validate(); print('Final D4 inputs verified; no fitting/forwards.')
    elif a.stage == 'train': train()
    elif a.stage == 'extract': extract(device=a.device)
    else: predict()


if __name__ == '__main__': main()
