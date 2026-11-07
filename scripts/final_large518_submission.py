"""User-run final Large518 Ridge + preserved Scattering486 50:50 submission.

prepare only freezes/verifies files. train selects alpha using training500 only.
extract checks train anchors then infers test features; predict applies numeric
heads without fitting. Offline, resumable, no overwrite of prior submissions.
"""
from __future__ import annotations
import argparse
import contextlib
import csv
import io
import os
from pathlib import Path
import time
import numpy as np
from threadpoolctl import threadpool_limits

if __package__:
    from . import evaluate_dinov2_large518_confirmation as confirmation
    from . import final_submission as shared
else:
    import evaluate_dinov2_large518_confirmation as confirmation
    import final_submission as shared

c = confirmation
e, eng, core, ROOT = c.e, c.eng, c.core, c.ROOT
OUT = ROOT/'reports/final_large518_blend'
SCATTERING = ROOT/'reports/final_scattering_blend'
SCATTERING_SHA = 'c3622568f205e511a1d3dd981722e51f9feebb805facdae1118cb041cc2cc739'
WIDTH, ANCHORS, SEED = 3072, 16, 31415
HEAD_KEYS = {'x_mean', 'x_scale', 'coef', 'intercept', 'y_mean', 'y_scale'}
PROTOCOL = {
    'version': 1, 'representation': 'Frozen DINOv2 Large518 CLS/patch mean/population std3072, original cached train features.',
    'alpha_candidates': [100., 1000., 10000.], 'selection_seed': SEED,
    'selection': '3-fold rank-stratified CV within all500 training rows; pooled MSE; exact ties prefer larger alpha. Same Ridge/scalers as confirmation.',
    'final_fit': 'Selected Ridge refit on all500; X/y scalers fit only train.9 CV fits+1 full fit. No PCA or SVR.',
    'selection_score': 'Training-only alpha-selection diagnostic, not independent performance estimate.',
    'test_extraction': 'Same local pinned Large weights, full256->bicubic518, ImageNet normalization, eval, FP32, no AMP/TF32, batch1.',
    'anchor': {'count': ANCHORS, 'choice': 'first16 train.csv IDs, no target selection', 'rtol': 1e-4, 'atol': 1e-4},
    'blend': {'preserved_scattering486': .5, 'large518_ridge': .5},
    'effective_weights': {'hand185': .0625, 'originalResNet': .0625, 'DINO_D4': .125, 'TabPFN486': .25, 'Large518_Ridge': .5},
    'preserved_scattering_sha256': SCATTERING_SHA,
    'scope': 'No existing model refits, clipping, calibration, mixture search, downloads or upload. Separate sample-ordered1000-row CSV.',
    'resume': 'Verify source/array/file/ID hashes and per-chunk cache; reuse complete stages; fail on changed bindings.',
    'attribution': 'DINOv2 (Meta), timm. Preserved ensemble includes PriorLabs-TabPFN.'}


def protect_destination(folder):
    resolved = Path(folder).resolve()
    protected = [ROOT/'open', ROOT/'models', ROOT/'scripts']
    protected += [p for p in (ROOT/'reports').iterdir() if p.is_dir() and p.resolve() != OUT.resolve()]
    for path in protected:
        path = path.resolve()
        if resolved == path or resolved.is_relative_to(path) or path.is_relative_to(resolved):
            raise ValueError('Output overlaps protected inputs/results: '+str(folder))


def verify_manifest(folder, binding, names):
    path = folder/'completed.json'
    if not path.exists():
        return False
    saved = e.read_json(path)
    if saved['binding'] != binding or set(saved['files']) != set(names):
        raise ValueError('Completed stage scope differs: '+str(folder))
    for name, sha in saved['files'].items():
        if e.base.sha256_file(folder/name) != sha:
            raise ValueError('Completed stage file changed: '+str(folder/name))
    return True


def finish(folder, binding, names):
    e.write_json(folder/'completed.json', {'binding': binding, 'files': {n: e.base.sha256_file(folder/n) for n in names}})


def stage_binding(plan, stage):
    return {'plan': e.digest(plan['frozen']), 'stage': stage}


def read_scattering(ids):
    folder = SCATTERING/'submission'
    path = folder/'submission.csv'
    plan = e.read_json(SCATTERING/'plan.json')
    done = e.read_json(folder/'completed.json')
    if done['binding']['plan'] != e.digest(plan['frozen']) or done['binding']['stage'] != 'fixed486_full500_test1000':
        raise ValueError('Wrong preserved Scattering486 provenance')
    verify_manifest(folder, done['binding'], ('submission.csv', 'component_predictions.csv', 'validation.json'))
    if e.base.sha256_file(path) != SCATTERING_SHA:
        raise ValueError('Preserved Scattering486 submission differs from reviewed submission')
    if shared.submission_ids(path, len(ids)) != list(ids):
        raise ValueError('Preserved Scattering486 IDs/order differ from sample')
    rows = core.read_rows(path, {'ID', 'hardness'})
    values = core.finite_column(rows, 'hardness', path)
    component_path = folder/'component_predictions.csv'
    parts = core.read_rows(component_path, {'ID', 'base', 'tabpfn486', 'hardness'})
    if [r['ID'] for r in parts] != list(ids):
        raise ValueError('Preserved component ID order differs')
    col = lambda name: core.finite_column(parts, name, component_path)
    np.testing.assert_array_equal(values, .5*(col('base')+col('tabpfn486')))
    np.testing.assert_array_equal(values, col('hardness'))
    audit = e.read_json(folder/'validation.json')
    if audit['submission_sha256'] != SCATTERING_SHA or audit['weights'] != {'base_hand185_resnet_dino': .5, 'tabpfn486': .5}:
        raise ValueError('Preserved blend specification differs')
    return values, {n: e.file_ref(folder/n) for n in ('submission.csv', 'component_predictions.csv', 'validation.json', 'completed.json')}


def load_state():
    data, x, refs, cp = c.validate()
    summaries = [c.read_completed(c.OUT/f'seed{s}', data, x, refs[s], cp, s) for s in c.SEEDS]
    result = c.aggregate(summaries)
    result['protocol_sha256'] = e.digest(cp['frozen'])
    review = e.read_json(c.OUT/'review/result_review.json')
    if result != e.read_json(c.OUT/'confirmation_summary.json') or review['confirmation'] != result or not result['all_seeds_improve_overall']:
        raise ValueError('Reviewed confirmation evidence changed')
    inventory = shared.test_inventory(ROOT/'open')
    if set(data.ids) & set(inventory['ids']) or x.shape != (500, WIDTH):
        raise ValueError('Unexpected train/test shape or overlap')
    previous, previous_files = read_scattering(inventory['ids'])
    model_refs = e.assets.validate_model_files('large')
    runtime_path = e.OUT/'features/large518/runtime.json'
    runtime = e.read_json(runtime_path)['runtime']
    _, feature_meta = e.npz_read(e.OUT/'features/large518/features.npz', data.ids, WIDTH)
    if any(feature_meta['extra'].get(k) != v for k, v in runtime.items()) or runtime['batch_size'] != 1 or runtime['precision'] != 'float32' or runtime['amp'] is not False:
        raise ValueError('Original Large518 runtime changed')
    prior_csvs = [ROOT/'reports/final_ensemble/submission/submission.csv', ROOT/'reports/final_tabpfn_blend/submission/submission.csv']
    frozen = {'protocol': PROTOCOL, 'sources': {p.name: e.file_ref(p) for p in
        (Path(__file__), Path(shared.__file__), ROOT/'scripts/run_final_large518.ps1')},
        'confirmation_plan': e.file_ref(c.OUT/'protocol.json'), 'confirmation_summary': e.file_ref(c.OUT/'confirmation_summary.json'),
        'confirmation_review': e.file_ref(c.OUT/'review/result_review.json'), 'feature': cp['frozen']['feature'],
        'ids_sha256': e.base.sha256_array(data.ids), 'y_sha256': e.base.sha256_array(data.y),
        'train_runtime_file': e.file_ref(runtime_path), 'train_runtime': runtime,
        'weights': model_refs, 'environment': e.environment(), 'test': inventory,
        'anchor_ids': data.ids[:ANCHORS].tolist(), 'anchor_values_sha256': e.base.sha256_array(x[:ANCHORS]),
        'scattering_submission': previous_files, 'other_prior_submissions': [e.file_ref(p) for p in prior_csvs]}
    return data, x, previous, frozen


def prepare(folder=OUT):
    protect_destination(folder)
    *_, frozen = load_state()
    path = folder/'plan.json'
    if path.exists():
        if e.read_json(path)['frozen'] != frozen:
            raise ValueError('Existing final plan differs; preserve it')
    else:
        if folder.exists() and any(folder.iterdir()):
            raise ValueError('Nonempty unprepared output directory')
        e.write_json(path, {'frozen': frozen, 'created_unix': time.time(), 'fits_during_preparation': 0, 'forwards_during_preparation': 0})
    print('Prepared/verified full500 Ridge + Scattering486 50:50; no fitting or feature extraction.')


def validate(folder=OUT):
    protect_destination(folder)
    data, x, previous, frozen = load_state()
    plan = e.read_json(folder/'plan.json')
    if plan['frozen'] != frozen or plan['fits_during_preparation'] != 0 or plan['forwards_during_preparation'] != 0:
        raise ValueError('Frozen final sources/inputs/environment changed')
    return data, x, previous, plan


def select_alpha(folder, binding, x, y, allow_fit=True, runner=None):
    """Training arrays only. No test features, base predictions, or public scores."""
    runner = runner or eng.fit_candidates
    y = eng.finite(y, (len(x),))
    predictions = {i: np.full(len(y), np.nan) for i in c.RIDGE_INDICES}
    splits = eng.inner_splits(y, SEED)
    for k, (ti, vi) in enumerate(splits):
        scope = {'parent': binding, 'inner': k, 'train': ti.tolist(), 'valid': vi.tolist(),
                 'x': e.base.sha256_array(x[ti]), 'valid_x': e.base.sha256_array(x[vi]), 'y': e.base.sha256_array(y[ti])}
        value = e.cell(folder/f'inner{k}.json', scope, lambda: runner(x[ti], y[ti], x[vi], c.RIDGE_INDICES), allow_fit)
        if set(value['predictions']) != {str(i) for i in predictions}:
            raise ValueError('Expected only the three fixed Ridge candidates')
        for i in predictions:
            predictions[i][vi] = eng.finite(value['predictions'][str(i)], (len(vi),))
    scores = [{'alpha': eng.CANDIDATES[i]['alpha'], 'pooled_mse': float(np.mean((eng.finite(p)-y)**2))} for i, p in predictions.items()]
    winner = min(scores, key=lambda r: (r['pooled_mse'], -r['alpha']))
    return {'alpha': winner['alpha'], 'inner_rmse': float(np.sqrt(winner['pooled_mse'])), 'candidates': scores,
            'selection_seed': SEED, 'cv_validation_sizes': [len(v) for _, v in splits],
            'n_train': len(y), 'score_scope': PROTOCOL['selection_score'], 'test_used_for_fit': False}


def fit_numeric_head(x, y, alpha):
    model = core.make_estimator(alpha).fit(eng.finite(x), eng.finite(y))
    head = shared.pack_head(model)
    np.testing.assert_allclose(shared.predict_head(head, x), model.predict(eng.finite(x)), rtol=0, atol=1e-10)
    return head


def load_head(folder, plan, data, x):
    output = folder/'model'
    binding = stage_binding(plan, 'full500_ridge')
    if not verify_manifest(output, binding, ('heads.npz', 'metadata.json', 'selection.json')):
        raise FileNotFoundError('Final Ridge model incomplete. Run train first.')
    head, meta = shared.read_archive(output, 'heads.npz')
    selection = select_alpha(output/'cv', binding, x, data.y, allow_fit=False, runner=c.forbidden)
    if selection != e.read_json(output/'selection.json') or meta['selection'] != selection or meta['binding'] != binding or meta['test_used_for_fit'] is not False:
        raise ValueError('Final Ridge provenance differs')
    if set(head) != HEAD_KEYS or selection['alpha'] not in PROTOCOL['alpha_candidates'] or selection['n_train'] != len(data.y):
        raise ValueError('Unexpected final Ridge schema')
    for key, value in head.items():
        shape = (x.shape[1],) if key in ('x_mean', 'x_scale', 'coef') else ()
        if value.dtype != np.float64 or value.shape != shape or not np.isfinite(value).all():
            raise ValueError('Invalid numeric Ridge head')
    if np.any(head['x_scale'] <= 0) or head['y_scale'] <= 0:
        raise ValueError('Invalid fitted scaler')
    return head, selection


def train(folder=OUT):
    data, x, _, plan = validate(folder)
    output = folder/'model'
    if (output/'completed.json').exists():
        _, selection = load_head(folder, plan, data, x)
        print(f'Final Ridge verified; alpha={selection["alpha"]:g}; no refitting.'); return
    binding = stage_binding(plan, 'full500_ridge')
    with threadpool_limits(limits=2):
        selection = select_alpha(output/'cv', binding, x, data.y)
        head = fit_numeric_head(x, data.y, selection['alpha'])
    validate(folder)
    meta = {'binding': binding, 'selection': selection, 'test_used_for_fit': False,
            'array_sha256': {k: e.base.sha256_array(v) for k, v in head.items()}}
    shared.save_archive(output, head, meta, 'heads.npz')
    e.write_json(output/'selection.json', selection)
    finish(output, binding, ('heads.npz', 'metadata.json', 'selection.json'))
    load_head(folder, plan, data, x)
    print(f'Final Large518 Ridge: alpha={selection["alpha"]:g}; refitted all{len(data.y)} training rows. Selection RMSE={selection["inner_rmse"]:.6f} is not independent validation.')


class LargeExtractor:
    def __init__(self, device_name, expected_runtime):
        torch, device, info = e.configure(device_name)
        self.runtime = {'device': str(device), 'gpu': info.get('gpu_name'), 'batch_size': 1, 'precision': 'float32', 'amp': False}
        if self.runtime != expected_runtime:
            raise ValueError('Use the original Large518 device/runtime (FP32,batch1); no fallback: '+str(self.runtime))
        self.torch, self.device = torch, device
        self.model = eng.load_model('large').to(device)
        if any(m.training for m in self.model.modules()) or any(p.requires_grad for p in self.model.parameters()):
            raise ValueError('Large backbone must stay frozen/eval')

    def __call__(self, split, ids):
        if split not in ('train', 'test'):
            raise ValueError('Invalid split')
        parts = []
        for ident in ids:
            raw = shared.read_gray(ROOT/'open'/split/(str(ident)+'.png')).astype(np.float32)/np.float32(255)
            tensor = self.torch.from_numpy(raw[None, None]).to(self.device)
            with self.torch.inference_mode():
                out = self.model.forward_features(eng.transform_image(tensor, 518))
                parts.append(eng.pool_tokens(out, 518, 1024).cpu().numpy())
        return np.concatenate(parts)

    def close(self):
        del self.model
        if self.device.type == 'cuda': self.torch.cuda.empty_cache()


def anchor_record(folder, plan, expected, extractor=None):
    ids = np.asarray(plan['frozen']['anchor_ids'])
    binding = {'parent': stage_binding(plan, 'test_large518'), 'expected_sha256': e.base.sha256_array(expected)}
    path = folder/'test_features/anchor.npz'
    exists = path.exists()
    if exists:
        actual, _ = e.npz_read(path, ids, WIDTH, binding)
    else:
        if extractor is None: raise FileNotFoundError('Train anchor incomplete. Run extract first.')
        actual = extractor('train', ids)
    if actual.shape != expected.shape or actual.dtype != np.float32 or not np.isfinite(actual).all() or not np.allclose(actual, expected, rtol=1e-4, atol=1e-4):
        raise ValueError('Large518 train anchor mismatch; no test feature extraction permitted')
    if not exists:
        e.npz_save(path, ids, actual, binding, {'max_absolute_difference': float(np.max(np.abs(actual-expected))), 'passed': True})
    return e.file_ref(path)


def test_chunk(folder, plan, index, extractor=None):
    item = plan['frozen']['test']['images'][index]
    source = ROOT/'open'/item['relative_path']
    if e.base.sha256_file(source) != item['sha256']:
        raise ValueError('Test image changed: '+item['id'])
    ids = np.asarray([item['id']])
    binding = {'parent': stage_binding(plan, 'test_large518'), 'image': item, 'offset': index}
    path = folder/'test_features/chunks'/f'{index:04d}.npz'
    if path.exists():
        return e.npz_read(path, ids, WIDTH, binding)[0]
    if extractor is None: raise FileNotFoundError('Test chunk missing. Run extract first.')
    features = extractor('test', ids)
    if features.shape != (1, WIDTH) or features.dtype != np.float32 or not np.isfinite(features).all():
        raise ValueError('Invalid Large518 test feature')
    if e.base.sha256_file(source) != item['sha256']:
        raise ValueError('Test image changed during extraction')
    e.npz_save(path, ids, features, binding)
    return features


def load_test_cache(folder, plan, train_x):
    output = folder/'test_features'
    binding = stage_binding(plan, 'test_large518')
    if not verify_manifest(output, binding, ('features.npz', 'anchor.npz')):
        raise FileNotFoundError('Test Large518 cache incomplete. Run extract first.')
    features, meta = e.npz_read(output/'features.npz', np.asarray(plan['frozen']['test']['ids']), WIDTH, binding)
    extra = meta['extra']
    if extra['runtime'] != plan['frozen']['train_runtime'] or extra['test_used_for_fit'] is not False or extra['anchor'] != anchor_record(folder, plan, train_x[:ANCHORS]):
        raise ValueError('Test cache runtime/anchor provenance differs')
    expected_names = {f'{j:04d}.npz' for j in range(len(features))}
    if set(extra['chunks']) != expected_names:
        raise ValueError('Incomplete chunk manifest')
    for j in range(len(features)):
        if e.base.sha256_file(output/'chunks'/f'{j:04d}.npz') != extra['chunks'][f'{j:04d}.npz']:
            raise ValueError('Test chunk hash differs')
        np.testing.assert_array_equal(features[j:j+1], test_chunk(folder, plan, j))
    return features


def extract(folder=OUT, device='cuda'):
    data, x, _, plan = validate(folder)
    load_head(folder, plan, data, x)  # Lock train-only selection before new test inference.
    output = folder/'test_features'
    if (output/'completed.json').exists():
        load_test_cache(folder, plan, x); print('Test Large518 cache verified; no model forwards.'); return
    extractor = LargeExtractor(device, plan['frozen']['train_runtime'])
    start = time.perf_counter()
    try:
        anchor = anchor_record(folder, plan, x[:ANCHORS], extractor)
        print('Train16 Large518 anchor matched; starting/resuming test extraction.', flush=True)
        rows = []
        for j in range(len(plan['frozen']['test']['ids'])):
            rows.append(test_chunk(folder, plan, j, extractor))
            if j == 0 or (j+1)%50 == 0:
                print(f'Large518 test {j+1}/{len(plan["frozen"]["test"]["ids"])}; seconds={time.perf_counter()-start:.1f}', flush=True)
        validate(folder)
        e.npz_save(output/'features.npz', np.asarray(plan['frozen']['test']['ids']), np.concatenate(rows), stage_binding(plan, 'test_large518'),
            {'anchor': anchor, 'runtime': extractor.runtime, 'test_used_for_fit': False,
             'chunks': {f'{j:04d}.npz': e.base.sha256_file(output/'chunks'/f'{j:04d}.npz') for j in range(len(rows))}})
        finish(output, stage_binding(plan, 'test_large518'), ('features.npz', 'anchor.npz'))
    finally:
        extractor.close()
    load_test_cache(folder, plan, x)
    print('Test Large518 features complete; no test fitting.')


def same_csv(path, rows):
    expected = list(csv.DictReader(io.StringIO(core.csv_text(rows))))
    with path.open(encoding='utf-8', newline='') as f:
        if list(csv.DictReader(f)) != expected:
            raise ValueError('Saved component CSV differs')


def write_submission(output, ids, previous, ridge, binding):
    protect_destination(output)
    previous, ridge = eng.finite(previous, (len(ids),)), eng.finite(ridge, (len(ids),))
    if len(set(ids)) != len(ids): raise ValueError('Duplicate test IDs')
    values = .5*previous + .5*ridge
    rows = [{'ID': str(i), 'scattering486': float(previous[j]), 'large518_ridge': float(ridge[j]), 'hardness': float(values[j])} for j, i in enumerate(ids)]
    names = ('submission.csv', 'component_predictions.csv', 'validation.json')
    if verify_manifest(output, binding, names):
        shared.verify_submission(output/'submission.csv', ids, values)
        same_csv(output/'component_predictions.csv', rows)
        print('Separate Large518 submission verified; unchanged.'); return
    # Incomplete files may only be rebuilt within this separate, bound output tree.
    output.mkdir(parents=True, exist_ok=True)
    temp = output/'submission.csv.tmp'
    temp.write_text(shared.submission_text(ids, values), encoding='utf-8', newline='')
    shared.verify_submission(temp, ids, values)
    os.replace(temp, output/'submission.csv')
    e.write_csv(output/'component_predictions.csv', rows)
    audit = {'binding': binding, 'rows': len(ids), 'columns': ['ID', 'hardness'],
        'sample_order_matches': True, 'all_finite': True, 'csv_round_trip_exact': True,
        'weights': PROTOCOL['blend'], 'effective_weights': PROTOCOL['effective_weights'],
        'uploaded': False, 'clipping_or_calibration': False, 'test_used_for_fit': False,
        'prediction_min': float(values.min()), 'prediction_max': float(values.max()),
        'prediction_mean': float(values.mean()), 'prediction_std': float(values.std()),
        'mean_absolute_change_vs_scattering486': float(np.abs(values-previous).mean()),
        'max_absolute_change_vs_scattering486': float(np.abs(values-previous).max()),
        'submission_sha256': e.base.sha256_file(output/'submission.csv')}
    e.write_json(output/'validation.json', audit)
    finish(output, binding, names)
    print(f'Submission ready: {output/"submission.csv"}; rows={len(ids)}; Scattering48650%+Large51850%; ID/finite/CSV verified; prior submissions preserved.')


def predict(folder=OUT):
    data, x, previous, plan = validate(folder)
    head, selection = load_head(folder, plan, data, x)
    features = load_test_cache(folder, plan, x)
    with threadpool_limits(limits=2):
        ridge = shared.predict_head(head, features)
    validate(folder)
    binding = {**stage_binding(plan, 'submission'), 'model': e.file_ref(folder/'model/completed.json'),
        'test_features': e.file_ref(folder/'test_features/completed.json'), 'alpha': selection['alpha'],
        'scattering_submission_sha256': SCATTERING_SHA}
    write_submission(folder/'submission', plan['frozen']['test']['ids'], previous, ridge, binding)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'validate', 'train', 'extract', 'predict'))
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output-dir', type=Path, default=OUT)
    args = parser.parse_args()
    folder = args.output_dir if args.output_dir.is_absolute() else ROOT/args.output_dir
    if args.stage == 'prepare': prepare(folder)
    elif args.stage == 'validate': validate(folder); print('Frozen inputs verified; no fitting or forwards.')
    elif args.stage == 'train': train(folder)
    elif args.stage == 'extract': extract(folder, args.device)
    else: predict(folder)


if __name__ == '__main__': main()
