"""User-run cross-fitted correction of the preserved Scattering486 ensemble.

prepare/validate/compare never fit competition models. check uses synthetic data.
evaluate generates fresh within-outer cross-fitted residuals before correction.
Built with PriorLabs-TabPFN. No image training, test access or submission changes.
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
    from . import evaluate_scattering_tabpfn as scat
    from . import residual_correction_engine as eng
else:
    import evaluate_scattering_tabpfn as scat
    import residual_correction_engine as eng

t, core, ROOT = scat.t, scat.t.core, scat.ROOT
OUT = ROOT/'reports/residual_correction'
SEEDS = (42, 2026)
PRIMARY = 'residual_corrected'
PROTOCOL = {
    'version': 1, 'seeds': list(SEEDS), 'primary': PRIMARY,
    'question': 'Can a small model predict systematic residuals of the current Scattering486 ensemble?',
    'base': 'current=.125*hand185+.125*originalResNet1792+.25*DINO_D4_1152+.5*TabPFN486. No new backbone or base weight change.',
    'outer': 'Reuse exact original5outer splits and verified archived context400/query100 component predictions.',
    'residual_targets': 'Within EACH outer400 generate3-fold context266/267 -> query133/134 component predictions. residual=y-current_crossfit. Never use global OOF residuals as fitting labels.',
    'base_ridge_selection': 'For EACH context266/267 select alpha inside a FURTHER3-fold CV, context-local rank deciles, X/yStandardScalers. Never reuse outer400-selected alpha for these crossfits.',
    'base_alpha_grids': {k: list(v) for k, v in eng.GRIDS.items()},
    'tabpfn': 'Pinned local V2/8 estimators/FP32, identical original runtime. No neural weight updates.',
    'seeds_rule': 'Crossfit3 uses seed+outer. Each base context/its alpha CV uses seed+1009*outer+inner. Correction uses seed+1009*outer+9.',
    'meta_inputs': '26: component predictions4, weighted ensemble1, component std1, three component-minus-TabPFN differences3, raw Haar noise proxy1, StandardScaler/PCA16 of raw486 features.',
    'projection': 'Fixed16, full SVD, no whitening. Fit only outer400; no outer100 images/features in fitted preprocessing. No PCA dimension search.',
    'candidate': {'model': 'HistGradientBoostingRegressor', **eng.TREE, 'shrinkage': eng.SHRINKAGE},
    'controls': 'Zero correction=current; StandardScaler/Ridge(alpha100) residual calibration on current prediction alone, same targets and fixed0.5 shrinkage.',
    'prediction': 'current +0.5*predicted_residual. Positive residual raises prediction. Not another50:50 blend with standalone hardness.',
    'selection': 'All correction parameters are prespecified. No correction tuning/early stopping, OOF selection of correction strength, clipping or common16 upweighting.',
    'work': 'Two seeds:30 fresh TabPFN contexts +1170 small Ridge fits,10 PCA fits,10 shallow residual trees,10 calibration fits. No GPU backbone training. Resume at completed base cell or completed outer correction.',
    'decision': 'Compare both seeds against current and calibration. Expansion priority: both improve both controls AND mean gain vs current>=0.30. Review common16 AND remaining484/middle80/tails/noise/folds/newworst20. No automatic adoption.',
    'limits': 'Same repeatedly studied500, not independent holdout. Residual training predictors use266/267 rows; deployment query predictors use400: possible training-size shift. Existing features may not contain missing information.',
    'scope': 'Train only; no new image features, test data, final500 fit or submission. Common16/target bins are diagnostics only.',
    'sources': ['https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingRegressor.html',
                'https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html'],
    'attribution': 'Built with PriorLabs-TabPFN'}


def unavailable(*args, **kwargs):
    raise ValueError('Completed artifact missing; implicit fitting forbidden')


def load_state():
    with contextlib.redirect_stdout(io.StringIO()):
        data, hand285, historical, upstream = scat.validate(scat.OUT)
        scattering, _ = scat.load_features(scat.OUT, data, upstream)
    cf = t.previous.confirmation
    raw = {'hand185': data.features[cf.AUGMENTED].astype(np.float64),
           'resnet1792': data.features[cf.RESNET].astype(np.float64),
           'dino1152': data.features[cf.DINO].astype(np.float64),
           'tabpfn486': np.c_[hand285, scattering]}
    for name, dim in zip(eng.COMPONENTS, (185,1792,1152,486)):
        eng.finite(raw[name], (500, dim))
    refs, files = {}, {}
    for seed in SEEDS:
        old = t.previous.load_references(seed, data)
        folder = scat.OUT/f'seed{seed}'
        binding = scat.evaluation_binding(scat.OUT, upstream, historical[seed], seed)
        if not scat.completed(folder, binding):
            raise ValueError('Incomplete current baseline')
        tab = np.full(500, np.nan)
        with contextlib.redirect_stdout(io.StringIO()):
            for fold in range(5):
                vi, p = scat.evaluate_fold(folder, binding, raw['tabpfn486'], data.y,
                    historical[seed]['folds'], seed, fold, unavailable)
                tab[vi] = p
        comp = np.column_stack([old['hand'], old['resnet'], old['dino'], tab])
        current = eng.ensemble(comp)
        path = folder/'oof.csv'
        rows = core.align_rows(core.read_rows(path, {'ID','hardness','fold',scat.PRIMARY}), data.ids, path)
        expected = core.finite_column(rows, scat.PRIMARY, path)
        np.testing.assert_allclose(current, expected, rtol=0, atol=1e-10)
        np.testing.assert_array_equal(core.finite_column(rows, 'hardness', path), data.y)
        np.testing.assert_array_equal(core.integer_column(rows, 'fold', path), old['folds'])
        np.testing.assert_array_equal(old['folds'], historical[seed]['folds'])
        refs[seed] = {'folds': old['folds'], 'components': comp, 'current': expected,
                      'runtime': historical[seed]['runtime']}
        files[str(seed)] = {n: t.file_ref(folder/n) for n in ('completed_evaluation.json','oof.csv','summary.json')}
    frozen = {'protocol': PROTOCOL, 'environment': t.environment(),
        'sources': {p.name: t.file_ref(p) for p in (Path(__file__), Path(eng.__file__), ROOT/'scripts/run_residual_correction.ps1')},
        'upstream_plan': t.file_ref(scat.OUT/'protocol.json'), 'upstream_payload': t.digest(upstream['frozen']),
        'arrays': {n: t.common.sha256_array(x) for n,x in raw.items()},
        'target_sha256': t.common.sha256_array(data.y), 'ids_sha256': t.common.sha256_array(data.ids),
        'noise_sha256': t.common.sha256_array(data.noise), 'reference_files': files,
        'references': {str(s): {n: t.common.sha256_array(refs[s][n]) for n in ('folds','components','current')} for s in SEEDS},
        'common_failure_ids': upstream['frozen']['common_failure_ids'],
        'preserved_submission': t.file_ref(ROOT/'reports/final_scattering_blend/submission/submission.csv')}
    return data, raw, refs, frozen


def prepare(folder=OUT):
    _, _, _, frozen = load_state()
    path = folder/'protocol.json'
    if path.exists():
        if t.read_json(path)['frozen'] != frozen:
            raise ValueError('Existing protocol changed; preserve old experiment')
    else:
        if folder.exists() and any(folder.iterdir()):
            raise ValueError('Nonempty unprepared output folder')
        t.write_json(path, {'frozen': frozen, 'created_unix': time.time(), 'competition_fits': 0})
    print('Prepared/verified:30 new inner TabPFN contexts; no competition fitting during preparation.')


def validate(folder=OUT):
    data, raw, refs, frozen = load_state()
    plan = t.read_json(folder/'protocol.json')
    if plan['frozen'] != frozen or plan['competition_fits'] != 0:
        raise ValueError('Protocol/source/input/environment changed')
    return data, raw, refs, plan


def checked_cell(path, binding, fit, allow_fit):
    if path.exists():
        saved = t.read_json(path)
        if saved['binding'] != binding or saved['sha256'] != t.digest(saved['result']):
            raise ValueError('Cell content or scope changed: '+str(path))
        return saved['result']
    if not allow_fit:
        unavailable()
    result = fit()
    t.write_json(path, {'binding': binding, 'result': result, 'sha256': t.digest(result)})
    return result


def crossfit_components(folder, binding, raw, outer_indices, outer_y, seed, fold, predictor, allow_fit):
    """Only outer-training labels are accepted; stored global OOF is unavailable here."""
    outer_indices = np.asarray(outer_indices, dtype=int)
    y = eng.finite(outer_y, (len(outer_indices),))
    if len(np.unique(outer_indices)) != len(outer_indices):
        raise ValueError('Duplicate outer fitting index')
    p = np.full((len(y),4), np.nan)
    for inner, (ti, vi) in enumerate(eng.splits(y, seed+fold)):
        context, query = outer_indices[ti], outer_indices[vi]
        if np.intersect1d(context, query).size:
            raise ValueError('Context/query overlap')
        model_seed = seed+1009*fold+inner
        scope = {'evaluation': binding, 'context': context.tolist(), 'query': query.tolist(),
            'context_y': t.common.sha256_array(y[ti]), 'model_seed': model_seed,
            'context_x': {n: t.common.sha256_array(x[context]) for n,x in raw.items()},
            'query_x': {n: t.common.sha256_array(x[query]) for n,x in raw.items()}}
        def run():
            values, detail = eng.fit_base({n:x[context] for n,x in raw.items()}, y[ti],
                {n:x[query] for n,x in raw.items()}, model_seed, predictor)
            eng.finite(values, (len(query),4))
            return {'components': values.tolist(), 'detail': detail}
        result = checked_cell(folder/f'inner{inner}/completed.json', scope, run, allow_fit)
        p[vi] = eng.finite(result['components'], (len(query),4))
        print(f'Outer{fold} inner{inner}: verified context={len(context)} query={len(query)}; residual source ready', flush=True)
    return eng.finite(p, (len(y),4))


def fold_cell(folder, binding, raw, train_indices, train_y, train_noise,
              query_indices, query_noise, query_components, seed, fold, predictor, allow_fit):
    ti, vi = np.asarray(train_indices), np.asarray(query_indices)
    if np.intersect1d(ti, vi).size:
        raise ValueError('Outer train/query overlap')
    cross = crossfit_components(folder, binding, raw, ti, train_y, seed, fold, predictor, allow_fit)
    scope = {'evaluation': binding, 'train': ti.tolist(), 'query': vi.tolist(),
        'train_y': t.common.sha256_array(train_y), 'crossfit_components': t.common.sha256_array(cross),
        'query_components': t.common.sha256_array(query_components),
        'train_noise': t.common.sha256_array(train_noise), 'query_noise': t.common.sha256_array(query_noise),
        'train_x': t.common.sha256_array(raw['tabpfn486'][ti]), 'query_x': t.common.sha256_array(raw['tabpfn486'][vi]),
        'correction_seed': seed+1009*fold+9}
    def run():
        delta, detail = eng.fit_corrections(raw['tabpfn486'][ti], train_noise, cross, train_y,
            raw['tabpfn486'][vi], query_noise, query_components, scope['correction_seed'])
        return {'deltas': {n: a.tolist() for n,a in delta.items()}, 'detail': detail,
                'training_residuals': (train_y-eng.ensemble(cross)).tolist()}
    saved = checked_cell(folder/'correction.json', scope, run, allow_fit)
    if set(saved['deltas']) != {'tree_delta', 'calibration_delta'}:
        raise ValueError('Invalid correction fields')
    delta = {n: eng.finite(a, (len(vi),)) for n,a in saved['deltas'].items()}
    np.testing.assert_array_equal(saved['training_residuals'], train_y-eng.ensemble(cross))
    detail = saved['detail']
    if (detail['context_rows'] != len(ti) or detail['query_rows'] != len(vi)
        or detail['tree_iterations'] != eng.TREE['max_iter'] or detail['shrinkage'] != eng.SHRINKAGE
        or detail['query_labels_accepted'] or detail['early_stopping']):
        raise ValueError('Correction audit mismatch')
    return delta, cross


def binding_for(plan, seed, ref):
    return {'plan': t.digest(plan['frozen']), 'seed': seed, 'runtime': ref['runtime']}


def collect(folder, data, raw, ref, plan, seed, predictor, allow_fit):
    deltas = {n: np.full(len(data.ids), np.nan) for n in ('tree_delta','calibration_delta')}
    rows = []
    binding = binding_for(plan, seed, ref)
    for fold in range(5):
        ti, vi = np.flatnonzero(ref['folds'] != fold), np.flatnonzero(ref['folds'] == fold)
        if (len(ti), len(vi)) != (400,100):
            raise ValueError('Expected original400/100 split')
        delta, cross = fold_cell(folder/f'fold{fold}', binding, raw, ti, data.y[ti], data.noise[ti],
            vi, data.noise[vi], ref['components'][vi], seed, fold, predictor, allow_fit)
        for n in deltas:
            deltas[n][vi] = delta[n]
        rows += [{'outer_fold': fold, 'ID': str(data.ids[j]), 'hardness': float(data.y[j]),
                  **{name: float(cross[k,l]) for l,name in enumerate(eng.COMPONENTS)},
                  'crossfit_current': float(eng.ensemble(cross[k:k+1])[0]),
                  'training_residual': float(data.y[j]-eng.ensemble(cross[k:k+1])[0])}
                 for k,j in enumerate(ti)]
        print(f'Outer{fold}: residual/calibration predictions stored; no outer score used', flush=True)
    return eng.compose(ref['current'], deltas), deltas, rows


def report_data(data, ref, predictions, common_ids):
    scores = {n: core.metrics(data.y,p) for n,p in predictions.items()}
    for s in scores.values():
        s['gain_vs_current'] = scores['current']['rmse']-s['rmse']
        s['gain_vs_calibration'] = scores['calibration_control']['rmse']-s['rmse']
    common = np.isin(data.ids, common_ids)
    targets, noise = core.rank_groups(data.y,10), core.rank_groups(data.noise,4)
    masks = {'common16': common, 'remaining484': ~common, 'low10': targets==0,
             'high10': targets==9, 'middle80': (targets>0)&(targets<9)}
    masks.update({f'noiseQ{k+1}': noise==k for k in range(4)})
    masks.update({f'fold{k}': ref['folds']==k for k in range(5)})
    diag = [{'model':n, 'group':g, **core.metrics(data.y[m],p[m]),
             'sse':float(np.sum((p[m]-data.y[m])**2))}
            for n,p in predictions.items() for g,m in masks.items() if m.any()]
    worst = {n:set(np.argsort((p-data.y)**2,kind='stable')[-20:]) for n,p in predictions.items()}
    detail = {'common16_improved_vs_current': int(np.sum(common & (abs(predictions[PRIMARY]-data.y)<abs(predictions['current']-data.y)))),
        'common16_same_error_sign': int(np.sum(common & (np.sign(predictions[PRIMARY]-data.y)==np.sign(predictions['current']-data.y)))),
        'common16_in_candidate_worst20': int(np.sum(common[list(worst[PRIMARY])])),
        'new_worst20_ids': sorted(str(data.ids[j]) for j in worst[PRIMARY]-worst['current']),
        'left_worst20_ids': sorted(str(data.ids[j]) for j in worst['current']-worst[PRIMARY]),
        'group_rmse': {g:{n:core.metrics(data.y[m],p[m])['rmse'] for n,p in predictions.items()} for g,m in masks.items()},
        'common16_is_diagnostic_only': True}
    return scores, diag, detail


def expected_files():
    return {'oof.csv','summary.json','diagnostics.csv','per_image_errors.csv','crossfit_training.csv','report.txt'} | {
        f'fold{f}/inner{k}/completed.json' for f in range(5) for k in range(3)} | {
        f'fold{f}/correction.json' for f in range(5)}


def completed(folder, binding):
    path = folder/'completed.json'
    if not path.exists():
        return False
    saved = t.read_json(path)
    if saved['binding'] != binding or set(saved['files']) != expected_files():
        raise ValueError('Completed seed binding/files changed')
    for name, sha in saved['files'].items():
        if t.common.sha256_file(folder/name) != sha:
            raise ValueError('Completed artifact changed: '+name)
    return True


def save_results(folder, data, ref, predictions, deltas, crossrows, plan, seed):
    scores, diag, detail = report_data(data, ref, predictions, plan['frozen']['common_failure_ids'])
    binding = binding_for(plan, seed, ref)
    rows = [{'ID':str(i),'hardness':float(data.y[j]),'fold':int(ref['folds'][j]),
             **{n:float(p[j]) for n,p in predictions.items()}, **{n:float(p[j]) for n,p in deltas.items()}}
            for j,i in enumerate(data.ids)]
    common = set(plan['frozen']['common_failure_ids'])
    errors = [{'ID':r['ID'],'common16':r['ID'] in common,'hardness':r['hardness'],
               **{n+'_error':r[n]-r['hardness'] for n in predictions},
               'sse_gain_vs_current':(r['current']-r['hardness'])**2-(r[PRIMARY]-r['hardness'])**2}
              for r in sorted(rows,key=lambda r:-(r[PRIMARY]-r['hardness'])**2)]
    for name, values in [('oof.csv',rows),('diagnostics.csv',diag),('per_image_errors.csv',errors),('crossfit_training.csv',crossrows)]:
        t.write_csv(folder/name, values)
    t.write_json(folder/'summary.json', {'seed':seed,'binding':binding,'metrics':scores,'diagnosis':detail,
        'primary':PRIMARY,'independent_holdout':False,'test_accessed':False,
        'new_tabpfn_contexts':15,'residual_training_uses_archived_global_oof':False})
    lines = ['Scattering486 residual correction; Built with PriorLabs-TabPFN.', '']
    lines += [f'{n}: RMSE={m["rmse"]:.6f}; gain vs current={m["gain_vs_current"]:+.6f}; gain vs calibration={m["gain_vs_calibration"]:+.6f}' for n,m in scores.items()]
    lines += ['', 'Current +0.5*predicted residual; no change to original component weights.',
        'Fresh3-fold base predictions inside each outer400; base Ridge tuning restricted further to each context266/267.',
        'Correction fixed in advance; no correction grid search/early stopping. PCA/scalers fit outer400 only.',
        'Crossfit training base size266/267 differs from query base size400; possible size shift.',
        'Common16, remaining484 and group diagnostics are descriptive; no target/error-based fitting weights.',
        'Same repeatedly studied500; no independent holdout. Original submissions preserved.']
    (folder/'report.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    t.write_json(folder/'completed.json', {'binding':binding,
        'files':{name:t.common.sha256_file(folder/name) for name in sorted(expected_files())}})
    for line in lines[2:5]: print(line)
    print(f'Report: {folder/"report.txt"}')


def evaluate(device, seed, folder=OUT):
    data, raw, refs, plan = validate(folder)
    target = folder/f'seed{seed}'; target.mkdir(parents=True,exist_ok=True)
    binding = binding_for(plan, seed, refs[seed])
    if completed(target, binding):
        print(f'Completed seed{seed} verified; no refitting'); return
    predictor = t.Predictor(t.verify_weights(t.OUT), device)
    if predictor.runtime != refs[seed]['runtime']:
        raise ValueError('Use exact archived CUDA runtime; no fallback')
    if not (folder/'synthetic_check.json').exists():
        raise ValueError('Run check before evaluate')
    check_result = t.read_json(folder/'synthetic_check.json')
    if check_result['plan'] != t.digest(plan['frozen']) or check_result['runtime'] != predictor.runtime:
        raise ValueError('Synthetic check/runtime changed')
    predictions, deltas, crossrows = collect(target,data,raw,refs[seed],plan,seed,predictor,True)
    validate(folder)
    save_results(target,data,refs[seed],predictions,deltas,crossrows,plan,seed)


def compare(folder=OUT):
    data, raw, refs, plan = validate(folder)
    records = []
    for seed in SEEDS:
        target = folder/f'seed{seed}'; binding = binding_for(plan,seed,refs[seed])
        if not completed(target,binding):
            raise ValueError('Both completed seeds required')
        with contextlib.redirect_stdout(io.StringIO()):
            predictions, deltas, crossrows = collect(target,data,raw,refs[seed],plan,seed,unavailable,False)
        scores, diag, detail = report_data(data,refs[seed],predictions,plan['frozen']['common_failure_ids'])
        saved = t.read_json(target/'summary.json')
        if saved['metrics'] != scores or saved['diagnosis'] != detail:
            raise ValueError('Result summary inconsistent with saved cells')
        rows = core.align_rows(core.read_rows(target/'oof.csv', {'ID',*predictions,*deltas}),data.ids,target)
        for n,p in {**predictions,**deltas}.items():
            np.testing.assert_array_equal(core.finite_column(rows,n,target),p)
        records.append({'seed':seed,'current_rmse':scores['current']['rmse'],
            'control_rmse':scores['calibration_control']['rmse'],'candidate_rmse':scores[PRIMARY]['rmse'],
            'gain_vs_current':scores[PRIMARY]['gain_vs_current'],
            'gain_vs_control':scores[PRIMARY]['gain_vs_calibration'],'diagnosis':detail})
    both = all(r['gain_vs_current']>0 and r['gain_vs_control']>0 for r in records)
    mean = float(np.mean([r['gain_vs_current'] for r in records]))
    result = {'primary':PRIMARY,'per_seed':records,'mean_gain_vs_current':mean,
        'mean_gain_vs_control':float(np.mean([r['gain_vs_control'] for r in records])),
        'all_seeds_improve_both':both,'passes_expansion_priority':both and mean>=.30,
        'additional_review_required':'Common16 AND remaining484, middle80/tails/noise/newworst20; no automatic adoption.',
        'aggregation':'Mean per-seed differences; same500, not independent inference or prediction averaging.',
        'protocol_sha256':t.digest(plan['frozen'])}
    t.write_json(folder/'comparison.json',result)
    print(t.common.canonical_json(result))


def check(device, folder=OUT):
    _, _, refs, plan = validate(folder)
    predictor = t.Predictor(t.verify_weights(t.OUT),device)
    if predictor.runtime != refs[42]['runtime']:
        raise ValueError('Use same archived runtime')
    rng = np.random.default_rng(1729)
    x = rng.normal(size=(400,486)); y = 200+3*x[:,0]+rng.normal(size=400)
    pred, meta = predictor(x[:267],y[:267],x[267:],1729)
    components = np.column_stack([y+rng.normal(size=400)*4 for _ in range(4)])
    with threadpool_limits(limits=2):
        delta, detail = eng.fit_corrections(x[:300],x[:300,0],components[:300],y[:300],
            x[300:],x[300:,0],components[300:],1729)
    for a in delta.values(): eng.finite(a,(100,))
    t.write_json(folder/'synthetic_check.json',{'plan':t.digest(plan['frozen']),
        'runtime':predictor.runtime,'competition_fits':0,'synthetic_context':267,'synthetic_query':133,
        'tabpfn_finite':bool(np.isfinite(pred).all()),'tree_iterations':detail['tree_iterations'],'metadata':meta})
    print('Synthetic TabPFN267/133 + fixed residual/calibration passed; competition fits=0.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('prepare','validate','check','evaluate','compare'))
    p.add_argument('--device',default='cuda');p.add_argument('--seed',type=int,choices=SEEDS,default=42)
    a=p.parse_args()
    if a.command=='prepare':prepare()
    elif a.command=='validate':validate();print('Frozen sources/inputs/current anchor verified; no fitting.')
    elif a.command=='check':check(a.device)
    elif a.command=='evaluate':evaluate(a.device,a.seed)
    else:compare()


if __name__=='__main__':main()
