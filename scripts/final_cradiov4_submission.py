"""User-run full500 C-RADIO Ridge and separate S:L:C=50:25:25 submission.

prepare audits only. train fits3 inner Scattering contexts/9 C-RADIO Ridge
candidates and one full500 head. extract locks train anchors before test.
predict does no fitting. Preserved S/L predictions are never retrained.
"""
from __future__ import annotations
import argparse
import contextlib
import io
import os
from pathlib import Path
import time
import numpy as np
from threadpoolctl import threadpool_limits

if __package__:
    from . import cradiov4_confirmation as confirmation
    from . import final_large518_submission as previous
else:
    import cradiov4_confirmation as confirmation
    import final_large518_submission as previous

c = confirmation
e, eng, core, ROOT = c.e, c.eng, c.core, c.ROOT
shared, residual, t = previous.shared, c.residual, c.t
OUT = ROOT/'reports/final_cradiov4_blend'
SEED, ANCHORS, WIDTH = 31415, 16, 4608
LARGE_SUBMISSION_SHA = '2f78444be2e47b52939af9819546cedc960503c0e3761289041e5e4e8560b080'
PROTOCOL = {
    'version': 1, 'seed': SEED, 'alpha_candidates': [100., 1000., 10000.],
    'representation': 'Reuse frozen C-RADIO summary2304+spatial mean1152+population std1152;4608 features.',
    'selection': 'Three train-only rank-stratified folds across500, seed31415. Minimize pooled MSE of .5S_inner+.5C_inner. Exact ties prefer larger alpha. Preserve original objective, NOT standalone or final50:25:25 objective.',
    'inner_S': 'Fresh333/334 context ->166/167 query on SAME3 partitions. Each base Ridge alpha selected by further3fold CV inside that context. TabPFN V2/8 estimators/FP32 context only. No global OOF substitution.',
    'base_seed_rule': 'Same residual crossfit rule with outer=0: base seeds31415+inner, splits31415.',
    'base_alpha_grids': {k: list(v) for k, v in residual.eng.GRIDS.items()},
    'final_fit': 'Selected C-RADIO Ridge fit all500 with train-only X/y StandardScalers. No encoder gradient updates.',
    'planned_fits': {'inner_tabpfn_contexts': 3, 'base_ridge': 117, 'radio_cv_ridge': 9, 'radio_full500_ridge': 1},
    'score_scope': 'Final alpha-selection CV is a tuning diagnostic, not independent performance.',
    'test': 'Same locked native256->bicubic592 (or previously locked512), internal conditioner, FP32 weights/BF16 autocast/FP32 pooling, batch1, frozen eval. No new resolution/precision search.',
    'anchor': {'count': ANCHORS, 'choice': 'first16 train IDs; no label/error selection', 'rtol': 1e-4, 'atol': 1e-4},
    'blend': {'scattering486': .5, 'large518_ridge': .25, 'cradiov4_ridge': .25},
    'effective_weights': {'hand185': .0625, 'originalResNet': .0625, 'DINO_D4': .125, 'TabPFN486': .25, 'Large518_Ridge': .25, 'CRADIO_Ridge': .25},
    'preserved_large_submission_sha256': LARGE_SUBMISSION_SHA,
    'preserved_scattering_submission_sha256': previous.SCATTERING_SHA,
    'preservation': 'S and standalone L come from verified archived component CSVs, not the old mixed submission treated as L. No S/L final refits.',
    'scope': 'No test-based fitting, calibration, clipping, mixture search, model downloads or uploads. Separate sample-ordered1000-row CSV.',
    'resume': 'Verify stage manifests and array/ID/source hashes. Resume inner contexts, Ridge cells and individual test images; changed bindings fail.',
    'attribution': 'NVIDIA C-RADIO; preserved ensemble includes Meta DINOv2 and PriorLabs-TabPFN.'}


def protect_destination(path):
    root = OUT.resolve(); dest = Path(path).resolve()
    if not (dest == root or dest.is_relative_to(root)):
        raise ValueError('Final C-RADIO outputs must stay in their separate output tree')
    if not root.is_relative_to((ROOT/'reports').resolve()):
        raise ValueError('Output root must be inside workspace reports')


def binding_for(plan, stage):
    return {'plan': e.digest(plan['frozen']), 'stage': stage}


def read_previous(ids):
    scattering, scatter_refs = previous.read_scattering(ids)
    folder = previous.OUT/'submission'
    plan = e.read_json(previous.OUT/'plan.json')
    done = e.read_json(folder/'completed.json')
    binding = done['binding']
    if (binding['plan'] != e.digest(plan['frozen']) or binding['stage'] != 'submission'
            or binding['scattering_submission_sha256'] != previous.SCATTERING_SHA):
        raise ValueError('Preserved Large518 submission scope differs')
    if not previous.verify_manifest(folder, binding, ('submission.csv', 'component_predictions.csv', 'validation.json')):
        raise ValueError('Preserved submission incomplete')
    if e.base.sha256_file(folder/'submission.csv') != LARGE_SUBMISSION_SHA:
        raise ValueError('Reviewed Large518 submission changed')
    for key, path in (('model', previous.OUT/'model/completed.json'), ('test_features', previous.OUT/'test_features/completed.json')):
        if binding[key] != e.file_ref(path):
            raise ValueError('Preserved Large518 component provenance changed')
    if not previous.verify_manifest(previous.OUT/'model', previous.stage_binding(plan, 'full500_ridge'), ('heads.npz', 'metadata.json', 'selection.json')):
        raise ValueError('Preserved Large518 model incomplete')
    if not previous.verify_manifest(previous.OUT/'test_features', previous.stage_binding(plan, 'test_large518'), ('features.npz', 'anchor.npz')):
        raise ValueError('Preserved Large518 test cache incomplete')
    parts = core.read_rows(folder/'component_predictions.csv', {'ID', 'scattering486', 'large518_ridge', 'hardness'})
    if [r['ID'] for r in parts] != list(ids):
        raise ValueError('Preserved component order differs from sample')
    col = lambda name: core.finite_column(parts, name, folder)
    np.testing.assert_array_equal(scattering, col('scattering486'))
    large = col('large518_ridge')
    np.testing.assert_array_equal(col('hardness'), .5*scattering+.5*large)
    shared.verify_submission(folder/'submission.csv', ids, col('hardness'))
    audit = e.read_json(folder/'validation.json')
    if audit['binding'] != binding or audit['weights'] != previous.PROTOCOL['blend'] or audit['submission_sha256'] != LARGE_SUBMISSION_SHA:
        raise ValueError('Preserved Large518 blend audit differs')
    refs = {'scattering': scatter_refs, 'large_plan': e.file_ref(previous.OUT/'plan.json'),
            'large': {n: e.file_ref(folder/n) for n in ('completed.json', 'submission.csv', 'component_predictions.csv', 'validation.json')}}
    return scattering, large, refs


def load_state():
    with contextlib.redirect_stdout(io.StringIO()):
        data, _, x, raw, refs, cp = c.validate(c.OUT)
    summaries, completion_refs = [], {}
    for seed in c.SEEDS:
        sub = c.OUT/f'seed{seed}'; binding = c.binding_for(cp, seed, refs[seed])
        if not c.completed(sub, binding):
            raise ValueError('Require all3 completed confirmation runs')
        summary = e.read_json(sub/'summary.json')
        if summary['binding'] != binding or not summary['anchor_matched'] or summary['test_accessed']:
            raise ValueError('Confirmation scope changed')
        summaries.append(summary)
        completion_refs[str(seed)] = e.file_ref(sub/'completed_evaluation.json')
    result = c.aggregate(summaries); result['protocol_sha256'] = e.digest(cp['frozen'])
    if result != e.read_json(c.OUT/'confirmation_summary.json') or not result['all_seeds_improve_both']:
        raise ValueError('Reviewed confirmation result changed')
    radio_plan = e.read_json(c.SOURCE/'protocol.json')
    checked = c.original.read_check(c.SOURCE, radio_plan)
    inventory = shared.test_inventory(ROOT/'open')
    if x.shape != (500, WIDTH) or set(data.ids)&set(inventory['ids']):
        raise ValueError('Unexpected train shape or train/test overlap')
    scattering, large, old = read_previous(inventory['ids'])
    assets = eng.assets.validate()  # Offline hash validation, not model loading.
    sources = (Path(__file__), ROOT/'scripts/run_final_cradiov4.ps1', Path(previous.__file__), Path(shared.__file__))
    frozen = {'protocol': PROTOCOL, 'sources': {p.name: e.file_ref(p) for p in sources},
        'confirmation_plan': e.file_ref(c.OUT/'protocol.json'), 'confirmation_payload': e.digest(cp['frozen']),
        'confirmation_summary': e.file_ref(c.OUT/'confirmation_summary.json'), 'completed_confirmations': completion_refs,
        'radio_features': cp['frozen']['radio_features'], 'assets': assets,
        'train_runtime': checked['runtime'], 'size': checked['size'],
        'tabpfn_runtime': refs[c.SEEDS[0]]['runtime'],
        'ids_sha256': e.base.sha256_array(data.ids), 'target_sha256': e.base.sha256_array(data.y),
        'feature_sha256': e.base.sha256_array(x), 'raw_sha256': {n: e.base.sha256_array(v) for n, v in raw.items()},
        'anchor_ids': data.ids[:ANCHORS].tolist(), 'anchor_values_sha256': e.base.sha256_array(x[:ANCHORS]),
        'test': inventory, 'previous': old,
        'earlier_submissions': [e.file_ref(ROOT/f'reports/{name}/submission/submission.csv') for name in ('final_ensemble', 'final_tabpfn_blend')]}
    return data, x, raw, scattering, large, frozen


def prepare(folder=OUT):
    protect_destination(folder)
    *_, frozen = load_state(); path = folder/'plan.json'
    if path.exists():
        if e.read_json(path)['frozen'] != frozen:
            raise ValueError('Prepared final plan changed; preserve existing artifacts')
    else:
        if folder.exists() and any(folder.iterdir()):
            raise ValueError('Nonempty unprepared final directory')
        e.write_json(path, {'frozen': frozen, 'created_unix': time.time(), 'fits_during_preparation': 0, 'forwards_during_preparation': 0})
    print('Prepared full500 C-RADIO Ridge and separate S50/L25/C25 submission; no fitting/inference.')


def validate(folder=OUT):
    protect_destination(folder)
    *state, frozen = load_state(); plan = e.read_json(folder/'plan.json')
    if plan['frozen'] != frozen or plan['fits_during_preparation'] != 0 or plan['forwards_during_preparation'] != 0:
        raise ValueError('Frozen final sources/inputs/environment changed')
    return (*state, plan)


def select_alpha(folder, binding, x, y, raw, predictor, allow_fit=True, runner=None):
    """No test data/predictions, public scores or archived global OOF accepted."""
    runner = runner or eng.ridge.fit_candidates
    y = eng.ridge.finite(y, (len(x),))
    splits = eng.ridge.inner_splits(y, SEED)
    for a, b in zip(splits, residual.eng.splits(y, SEED), strict=True):
        for u, v in zip(a, b, strict=True):
            np.testing.assert_array_equal(u, v)
    cross = residual.crossfit_components(folder/'baseline', binding, raw, np.arange(len(y)), y, SEED, 0, predictor, allow_fit)
    s = residual.eng.ensemble(cross)
    inner = {str(i): np.full(len(y), np.nan) for i in eng.RIDGE_INDICES}
    with threadpool_limits(limits=2):
        for k, (ti, vi) in enumerate(splits):
            scope = {'parent': binding, 'inner': k, 'train': ti.tolist(), 'query': vi.tolist(),
                'x': e.base.sha256_array(x[ti]), 'q': e.base.sha256_array(x[vi]), 'y': e.base.sha256_array(y[ti]), 'candidates': list(eng.RIDGE_INDICES)}
            saved = e.cell(folder/f'radio/inner{k}.json', scope, lambda: runner(x[ti], y[ti], x[vi], eng.RIDGE_INDICES), allow_fit)
            if set(saved['predictions']) != set(inner):
                raise ValueError('Expected exact3 Ridge candidates')
            for key, p in saved['predictions'].items():
                inner[key][vi] = eng.ridge.finite(p, (len(vi),))
    choices = eng.choose(inner, y, s)
    return {'alpha': choices['blend']['alpha'], 'choices': choices,
        'selection_rmse': float(np.sqrt(choices['blend']['blend_mse'])), 'selection_seed': SEED,
        'cv_validation_sizes': [len(v) for _, v in splits], 'n_train': len(y),
        'inner_scattering_sha256': e.base.sha256_array(s), 'test_used_for_fit': False,
        'score_scope': PROTOCOL['score_scope']}


def model_names():
    return {'heads.npz', 'metadata.json', 'selection.json'} | {
        f'cv/baseline/inner{k}/completed.json' for k in range(3)} | {f'cv/radio/inner{k}.json' for k in range(3)}


def load_head(folder, plan, data, x, raw):
    out = folder/'model'; binding = binding_for(plan, 'full500_radio_ridge')
    if not previous.verify_manifest(out, binding, model_names()):
        raise FileNotFoundError('Final model incomplete; run train')
    head, meta = shared.read_archive(out, 'heads.npz')
    with contextlib.redirect_stdout(io.StringIO()):
        selection = select_alpha(out/'cv', binding, x, data.y, raw, c.forbidden, False, c.forbidden)
    if (selection != e.read_json(out/'selection.json') or meta['selection'] != selection
            or meta['binding'] != binding or meta['test_used_for_fit'] is not False):
        raise ValueError('Final head provenance changed')
    if set(head) != previous.HEAD_KEYS or selection['alpha'] not in PROTOCOL['alpha_candidates'] or selection['n_train'] != len(data.y):
        raise ValueError('Final numeric head schema differs')
    for key, value in head.items():
        shape = (x.shape[1],) if key in ('x_mean', 'x_scale', 'coef') else ()
        if value.shape != shape or value.dtype != np.float64 or not np.isfinite(value).all():
            raise ValueError('Invalid numeric head')
    if np.any(head['x_scale'] <= 0) or head['y_scale'] <= 0:
        raise ValueError('Invalid fitted scales')
    return head, selection


def train(folder=OUT, device='cuda'):
    data, x, raw, _, _, plan = validate(folder); out = folder/'model'
    if (out/'completed.json').exists():
        _, selected = load_head(folder, plan, data, x, raw)
        print(f'Final model verified; alpha={selected["alpha"]:g}; no refitting.'); return
    predictor = t.Predictor(t.verify_weights(t.OUT), device)
    if predictor.runtime != plan['frozen']['tabpfn_runtime']:
        raise ValueError('Use archived TabPFN runtime; no fallback')
    binding = binding_for(plan, 'full500_radio_ridge')
    selected = select_alpha(out/'cv', binding, x, data.y, raw, predictor)
    with threadpool_limits(limits=2):
        head = previous.fit_numeric_head(x, data.y, selected['alpha'])
    validate(folder)
    meta = {'binding': binding, 'selection': selected, 'test_used_for_fit': False,
            'array_sha256': {k: e.base.sha256_array(v) for k, v in head.items()}}
    shared.save_archive(out, head, meta, 'heads.npz'); e.write_json(out/'selection.json', selected)
    previous.finish(out, binding, sorted(model_names()))
    load_head(folder, plan, data, x, raw)
    print(f'Final C-RADIO Ridge alpha={selected["alpha"]:g}; all{len(data.y)} rows fitted. Selection RMSE={selected["selection_rmse"]:.6f} is not independent validation.')


class RadioExtractor:
    def __init__(self, device, expected_runtime, size):
        torch, resolved, runtime = c.original.runtime(device)
        if runtime != expected_runtime or size not in (592, 512):
            raise ValueError('C-RADIO runtime/resolution differs from train cache; no fallback')
        self.torch, self.device, self.runtime, self.size = torch, resolved, runtime, size
        self.model = eng.load_model().to(resolved)
        if any(m.training for m in self.model.modules()) or any(p.requires_grad for p in self.model.parameters()):
            raise ValueError('Encoder must remain frozen/eval')

    def __call__(self, split, ids):
        from PIL import Image
        if split not in ('train', 'test'):
            raise ValueError('Invalid split')
        rows = []
        for ident in ids:
            with Image.open(ROOT/'open'/split/(str(ident)+'.png')) as im:
                if im.mode != 'L' or im.size != (256, 256):
                    raise ValueError('Expected native256 grayscale')
                a = np.array(im, dtype=np.uint8)
            raw = self.torch.from_numpy(a.copy()).to(device=self.device, dtype=self.torch.float32)[None, None]/255.
            rows.append(c.original.forward(self.model, raw, self.size, self.torch).cpu().numpy())
        return np.concatenate(rows).astype(np.float32)

    def close(self):
        del self.model
        self.torch.cuda.empty_cache()


def anchor_record(folder, plan, expected, extractor=None):
    ids = np.asarray(plan['frozen']['anchor_ids'])
    binding = {'parent': binding_for(plan, 'test_radio'), 'expected_sha256': e.base.sha256_array(expected)}
    path = folder/'test_features/anchor.npz'; exists = path.exists()
    if exists:
        actual, _ = e.npz_read(path, ids, WIDTH, binding)
    else:
        if extractor is None: raise FileNotFoundError('Train anchor missing; run extract')
        actual = extractor('train', ids)
    tol = PROTOCOL['anchor']
    if (actual.shape != expected.shape or actual.dtype != np.float32 or not np.isfinite(actual).all()
            or not np.allclose(actual, expected, rtol=tol['rtol'], atol=tol['atol'])):
        raise ValueError('C-RADIO train anchor mismatch; no test forwards allowed')
    if not exists:
        e.npz_save(path, ids, actual, binding, {'passed': True, 'max_absolute_difference': float(np.max(np.abs(actual-expected)))})
    return e.file_ref(path)


def test_chunk(folder, plan, index, extractor=None):
    item = plan['frozen']['test']['images'][index]; source = ROOT/'open'/item['relative_path']
    if e.base.sha256_file(source) != item['sha256']:
        raise ValueError('Test image changed: '+item['id'])
    ids = np.asarray([item['id']])
    binding = {'parent': binding_for(plan, 'test_radio'), 'image': item, 'offset': index}
    path = folder/'test_features/chunks'/f'{index:04d}.npz'
    if path.exists(): return e.npz_read(path, ids, WIDTH, binding)[0]
    if extractor is None: raise FileNotFoundError('Test chunk missing; run extract')
    x = extractor('test', ids)
    if x.shape != (1, WIDTH) or x.dtype != np.float32 or not np.isfinite(x).all():
        raise ValueError('Invalid test feature')
    if e.base.sha256_file(source) != item['sha256']:
        raise ValueError('Image changed during extraction')
    e.npz_save(path, ids, x, binding); return x


def load_test_cache(folder, plan, train_x):
    out = folder/'test_features'; binding = binding_for(plan, 'test_radio')
    if not previous.verify_manifest(out, binding, ('features.npz', 'anchor.npz')):
        raise FileNotFoundError('Test cache incomplete; run extract')
    x, meta = e.npz_read(out/'features.npz', np.asarray(plan['frozen']['test']['ids']), WIDTH, binding)
    extra = meta['extra']
    if (extra['runtime'] != plan['frozen']['train_runtime'] or extra['size'] != plan['frozen']['size']
            or extra['test_used_for_fit'] is not False or extra['anchor'] != anchor_record(folder, plan, train_x[:ANCHORS])):
        raise ValueError('Test cache provenance differs')
    if set(extra['chunks']) != {f'{i:04d}.npz' for i in range(len(x))}:
        raise ValueError('Incomplete chunk manifest')
    for i in range(len(x)):
        if e.base.sha256_file(out/'chunks'/f'{i:04d}.npz') != extra['chunks'][f'{i:04d}.npz']:
            raise ValueError('Test chunk changed')
        np.testing.assert_array_equal(x[i:i+1], test_chunk(folder, plan, i))
    return x


def extract(folder=OUT, device='cuda'):
    data, x, raw, _, _, plan = validate(folder)
    load_head(folder, plan, data, x, raw)  # Lock training before any test forwards.
    out = folder/'test_features'
    if (out/'completed.json').exists():
        load_test_cache(folder, plan, x); print('Test features verified; no new forwards.'); return
    extractor = RadioExtractor(device, plan['frozen']['train_runtime'], plan['frozen']['size'])
    started = time.perf_counter()
    try:
        anchor = anchor_record(folder, plan, x[:ANCHORS], extractor)
        print('Train16 C-RADIO anchor matched; starting/resuming test extraction.', flush=True)
        rows = []
        for i in range(len(plan['frozen']['test']['ids'])):
            rows.append(test_chunk(folder, plan, i, extractor))
            if i == 0 or (i+1)%50 == 0:
                print(f'C-RADIO test {i+1}/{len(plan["frozen"]["test"]["ids"])}; seconds={time.perf_counter()-started:.1f}', flush=True)
        validate(folder)
        e.npz_save(out/'features.npz', np.asarray(plan['frozen']['test']['ids']), np.concatenate(rows), binding_for(plan, 'test_radio'),
            {'anchor': anchor, 'runtime': extractor.runtime, 'size': extractor.size, 'test_used_for_fit': False,
             'chunks': {f'{i:04d}.npz': e.base.sha256_file(out/'chunks'/f'{i:04d}.npz') for i in range(len(rows))}})
        previous.finish(out, binding_for(plan, 'test_radio'), ('features.npz', 'anchor.npz'))
    finally:
        extractor.close()
    load_test_cache(folder, plan, x)
    print('Test C-RADIO features ready; no test fitting.')


def write_submission(out, ids, scattering, large, radio, binding):
    protect_destination(out)
    s, l, r = (eng.ridge.finite(v, (len(ids),)) for v in (scattering, large, radio))
    if len(set(ids)) != len(ids): raise ValueError('Duplicate sample IDs')
    p = .5*s+.25*l+.25*r
    rows = [{'ID': str(ident), 'scattering486': float(s[i]), 'large518_ridge': float(l[i]),
             'cradiov4_ridge': float(r[i]), 'hardness': float(p[i])} for i, ident in enumerate(ids)]
    names = ('submission.csv', 'component_predictions.csv', 'validation.json')
    if previous.verify_manifest(out, binding, names):
        shared.verify_submission(out/'submission.csv', ids, p)
        previous.same_csv(out/'component_predictions.csv', rows)
        print('Separate C-RADIO submission verified; unchanged.'); return
    out.mkdir(parents=True, exist_ok=True)
    temp = out/'submission.csv.tmp'
    temp.write_text(shared.submission_text(ids, p), encoding='utf-8', newline='')
    shared.verify_submission(temp, ids, p); os.replace(temp, out/'submission.csv')
    e.write_csv(out/'component_predictions.csv', rows)
    old = .5*s+.5*l
    audit = {'binding': binding, 'rows': len(ids), 'columns': ['ID', 'hardness'],
        'sample_order_matches': True, 'all_finite': True, 'csv_round_trip_exact': True,
        'weights': PROTOCOL['blend'], 'effective_weights': PROTOCOL['effective_weights'],
        'uploaded': False, 'clipping_or_calibration': False, 'test_used_for_fit': False,
        'prediction_min': float(p.min()), 'prediction_max': float(p.max()), 'prediction_mean': float(p.mean()),
        'prediction_std': float(p.std()), 'mean_absolute_change_vs_current': float(np.abs(p-old).mean()),
        'max_absolute_change_vs_current': float(np.abs(p-old).max()),
        'submission_sha256': e.base.sha256_file(out/'submission.csv')}
    e.write_json(out/'validation.json', audit); previous.finish(out, binding, names)
    print(f'Submission ready: {out/"submission.csv"}; rows={len(ids)}; S50/L25/C25; IDs/finite/CSV verified; prior submissions preserved.')


def predict(folder=OUT):
    data, x, raw, scattering, large, plan = validate(folder)
    head, selected = load_head(folder, plan, data, x, raw)
    features = load_test_cache(folder, plan, x)
    with threadpool_limits(limits=2): radio = shared.predict_head(head, features)
    validate(folder)
    binding = {**binding_for(plan, 'submission'), 'model': e.file_ref(folder/'model/completed.json'),
        'test_features': e.file_ref(folder/'test_features/completed.json'), 'alpha': selected['alpha'],
        'scattering_submission_sha256': previous.SCATTERING_SHA, 'large_submission_sha256': LARGE_SUBMISSION_SHA}
    write_submission(folder/'submission', plan['frozen']['test']['ids'], scattering, large, radio, binding)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=('prepare', 'validate', 'train', 'extract', 'predict'))
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    if args.stage == 'prepare': prepare()
    elif args.stage == 'validate': validate(); print('Frozen final inputs verified; no fitting/inference.')
    elif args.stage == 'train': train(device=args.device)
    elif args.stage == 'extract': extract(device=args.device)
    else: predict()


if __name__ == '__main__': main()
