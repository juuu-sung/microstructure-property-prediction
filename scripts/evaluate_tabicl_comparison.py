"""User-run fixed TabICLv2/TabPFNv2 x raw486/blockPCA96 comparison.

Built with PriorLabs-TabPFN and soda-inria/TabICL. prepare/validate/compare do
not fit models. check uses synthetic data only. evaluate is the user-run step.
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import io
import os
from pathlib import Path
import sys
import time

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

if __package__:
    from . import evaluate_scattering_tabpfn as scat
    from . import setup_tabicl as assets
else:
    import evaluate_scattering_tabpfn as scat
    import setup_tabicl as assets

t, core, ROOT = scat.t, scat.t.core, scat.ROOT
OUT = ROOT/'reports/tabicl_comparison'
SEEDS = (42, 2026)
BLOCKS = (('hand185', 0, 185), ('morphology100', 185, 285), ('scattering201', 285, 486))
REPS = ('tabpfn96', 'tabicl486', 'tabicl96')
BLENDS = {n: n+'_blend' for n in REPS}
PRIMARY, SECONDARY = BLENDS['tabicl486'], BLENDS['tabicl96']
CONTROLS = {'tabpfn96': 'current486', 'tabicl486': 'current486', 'tabicl96': BLENDS['tabpfn96']}
ICL_CONFIG = dict(n_estimators=8, norm_methods=['none', 'power'], feat_shuffle_method='latin',
    outlier_threshold=4., batch_size=1, kv_cache=False, allow_auto_download=False,
    checkpoint_version=assets.WEIGHT, use_amp=False, use_fa3=False, offload_mode='cpu', n_jobs=4, verbose=False)
PROTOCOL = {
    'version': 1, 'attribution': 'Built with PriorLabs-TabPFN and soda-inria/TabICL',
    'seeds': list(SEEDS), 'primary': PRIMARY, 'secondary': SECONDARY,
    'representations': {'raw486': 'hand185+morphology100+scattering201; unchanged',
        'block_pca96': 'StandardScaler/PCA32 independently per each of the three retained blocks'},
    'pca': {'components_per_block': 32, 'blocks': [list(b) for b in BLOCKS],
        'whiten': False, 'svd_solver': 'full', 'dtype': 'float64', 'threads': 2,
        'fit_scope': 'Only outer training400; no query/targets in fit API; no dimension or variance cutoff search'},
    'tabicl': ICL_CONFIG, 'tabpfn': t.PROTOCOL,
    'fit_scope': 'Fixed5 outer400/100 per seed; no inner search, representation selection or gradient updates. '
                 '30 new context fits,10 blockPCA fits (three PCA32 each). Fit seed=seed+1009*fold+9.',
    'baseline': 'Verify/reuse archived TabPFNv2 raw486 predictions; no implicit baseline refits.',
    'blend': 'Base=.25*hand185+.25*originalResNet+.5*DINO_D4. '
             'Each blend=.5*Base+.5*variant; replace ONLY existing TabPFN50%.',
    'runtime': 'FP32, deterministic, no TF32; TabICL batch1, CPU embedding offload, no KV cache/FA3. '
               'Same GPU as archived TabPFN. User-run synthetic400/100 checks before any real fit; no precision fallback.',
    'selection': 'Raw486 TabICL primary; PCA96 TabICL fixed secondary; PCA96 TabPFN compression control. '
                 'Report all cells; no best-of-two posthoc headline or per-fold winner.',
    'priority_mean_gain': .30,
    'decision': 'Both seeds improve current486 and matched-representation TabPFN control; mean gain>=.30 '
                'is expansion priority, not significance or automatic adoption. Common16 is diagnostic, not a training filter/veto.',
    'scope': 'Train-only; no image extraction, test access, full500 fit, submission changes or weight search. '
             'Repeatedly studied same500, not independent holdout.',
}


def forbidden(*args, **kwargs):
    raise ValueError('Required completed cell missing; implicit fitting is forbidden.')


def load_state():
    data, base285, historical, old_plan = scat.validate(scat.OUT)
    scatter, _ = scat.load_features(scat.OUT, data, old_plan)
    np.testing.assert_array_equal(data.features[t.previous.confirmation.AUGMENTED], base285[:, :185])
    raw = np.c_[base285, scatter].astype(np.float64)
    if raw.shape != (500, 486) or not np.isfinite(raw).all():
        raise ValueError('Expected aligned finite raw486 features')
    refs, archives = {}, {}
    for seed in SEEDS:
        previous = historical[seed]
        folder = scat.OUT/f'seed{seed}'
        binding = scat.evaluation_binding(scat.OUT, old_plan, previous, seed)
        if not scat.completed(folder, binding):
            raise ValueError('Incomplete Scattering486 baseline')
        pred = np.full(500, np.nan)
        with contextlib.redirect_stdout(io.StringIO()):
            for fold in range(5):
                vi, p = scat.evaluate_fold(folder, binding, raw, data.y, previous['folds'], seed, fold, forbidden)
                pred[vi] = p
        metrics, rows, _, _ = scat.result_data(data, previous, pred, old_plan['frozen']['common_failure_ids'])
        if t.read_json(folder/'summary.json')['metrics'] != metrics:
            raise ValueError('Archived486 recomputed metrics differ')
        saved = core.align_rows(core.read_rows(folder/'oof.csv', set(rows[0])), data.ids, folder)
        for name in rows[0]:
            if name != 'ID':
                np.testing.assert_array_equal([float(r[name]) for r in saved], [float(r[name]) for r in rows])
        refs[seed] = {'folds': previous['folds'], 'base': previous['base'],
            'current486': .5*(previous['base']+pred), 'tabpfn486': pred, 'runtime': previous['runtime']}
        archives[str(seed)] = t.file_ref(folder/'completed_evaluation.json')
    manifest = assets.validate()
    frozen = {'protocol': PROTOCOL,
        'sources': {p.name: t.file_ref(p) for p in (Path(__file__), Path(assets.__file__), ROOT/'scripts/run_tabicl_comparison.ps1')},
        'environment': t.environment(), 'scattering_plan': t.file_ref(scat.OUT/'protocol.json'),
        'assets_manifest': t.file_ref(assets.ASSETS/'assets.json'), 'assets_digest': t.digest(manifest),
        'features': [t.file_ref(scat.OUT/'features'/n) for n in ('features.npz', 'metadata.json')],
        'raw486_sha256': t.common.sha256_array(raw), 'ids_sha256': t.common.sha256_array(data.ids),
        'target_sha256': t.common.sha256_array(data.y), 'noise_sha256': t.common.sha256_array(data.noise),
        'baseline_archives': archives, 'common_failure_ids': old_plan['frozen']['common_failure_ids'],
        'references': {str(s): {'runtime': refs[s]['runtime'], 'arrays': {
            n: t.common.sha256_array(refs[s][n]) for n in ('folds','base','current486','tabpfn486')}} for s in SEEDS}}
    return data, raw, refs, frozen


def prepare(folder):
    data, _, refs, frozen = load_state()
    path = folder/'protocol.json'
    if path.exists():
        if t.read_json(path)['frozen'] != frozen:
            raise ValueError('Frozen experiment changed; preserve existing results.')
        print('Prepared protocol verified; no fitting.')
        return
    if folder.exists() and any(folder.iterdir()):
        raise ValueError('Nonempty unprepared output folder')
    t.write_json(path, {'frozen': frozen, 'created_unix': time.time(), 'competition_fits': 0, 'competition_forwards': 0})
    print(f'Prepared fixed2x2 comparison;30 new contexts; {path}')
    for seed in SEEDS:
        print(f'Current486 seed{seed}: RMSE={core.metrics(data.y, refs[seed]["current486"])["rmse"]:.6f}')


def validate(folder):
    if not (folder/'protocol.json').is_file():
        raise ValueError('Run prepare first.')
    data, raw, refs, frozen = load_state()
    plan = t.read_json(folder/'protocol.json')
    if plan['frozen'] != frozen or plan['competition_fits'] or plan['competition_forwards']:
        raise ValueError('Frozen sources, assets, features, runtime or baseline changed.')
    return data, raw, refs, plan


def fit_pca(train):
    """No query rows or targets accepted. Each block retains fixed32 PCs."""
    train = np.asarray(train, dtype=np.float64)
    if train.ndim != 2 or train.shape[1] != 486 or len(train) < 33 or not np.isfinite(train).all():
        raise ValueError('PCA requires >=33 finite training rows and486 columns')
    params = {'n_samples': np.array([len(train)], dtype=np.int64)}
    with threadpool_limits(limits=2):
        for name, start, stop in BLOCKS:
            scaler = StandardScaler().fit(train[:, start:stop])
            model = PCA(n_components=32, whiten=False, svd_solver='full').fit(scaler.transform(train[:, start:stop]))
            for key, value in dict(scaler_mean=scaler.mean_, scaler_scale=scaler.scale_, pca_mean=model.mean_,
                components=model.components_, variance_ratio=model.explained_variance_ratio_).items():
                params[name+'_'+key] = value
    return params


def validate_pca(params, rows):
    shapes = {'n_samples': (1,)}
    for name, start, stop in BLOCKS:
        shapes.update({name+'_'+k: v for k,v in {'scaler_mean': (stop-start,), 'scaler_scale': (stop-start,),
            'pca_mean': (stop-start,), 'components': (32, stop-start), 'variance_ratio': (32,)}.items()})
    if set(params) != set(shapes) or any(params[k].shape != v or not np.isfinite(params[k]).all() for k,v in shapes.items()):
        raise ValueError('Invalid blockPCA parameter schema')
    if int(params['n_samples'][0]) != rows or any(np.any(params[n+'_scaler_scale'] <= 0) for n,_,_ in BLOCKS):
        raise ValueError('Invalid PCA training scope or scaler')


def project(x, params):
    with threadpool_limits(limits=2):
        value = np.concatenate([((x[:, start:stop]-params[name+'_scaler_mean']) / params[name+'_scaler_scale']
            - params[name+'_pca_mean']) @ params[name+'_components'].T for name,start,stop in BLOCKS], axis=1)
    if value.shape != (len(x), 96) or not np.isfinite(value).all():
        raise ValueError('Invalid blockPCA projection')
    return value


def pca_cell(folder, binding, raw, ti, vi, allow_fit):
    if (raw.shape != (500,486) or len(ti) != 400 or len(vi) != 100
            or not np.array_equal(np.sort(np.r_[ti,vi]), np.arange(500))):
        raise ValueError('Expected disjoint outer400/100 partition')
    expected = {'experiment': binding, 'train_idx': ti.tolist(), 'query_idx': vi.tolist(),
        'train_x_sha256': t.common.sha256_array(raw[ti]), 'query_x_sha256': t.common.sha256_array(raw[vi]),
        'settings': PROTOCOL['pca']}
    path, meta_path = folder/'parameters.npz', folder/'completed.json'
    if meta_path.exists():
        meta = t.read_json(meta_path)
        if meta['binding'] != expected or meta['parameters_sha256'] != t.common.sha256_file(path):
            raise ValueError('PCA cache binding or parameters changed')
        with np.load(path, allow_pickle=False) as z:
            params = {n: z[n].copy() for n in z.files}
    else:
        if not allow_fit:
            forbidden()
        params = fit_pca(raw[ti])
        validate_pca(params, 400)
        folder.mkdir(parents=True, exist_ok=True)
        with path.with_suffix('.npz.tmp').open('wb') as f:
            np.savez_compressed(f, **params)
        os.replace(path.with_suffix('.npz.tmp'), path)
        t.write_json(meta_path, {'binding': expected, 'parameters_sha256': t.common.sha256_file(path),
            'labels_used': False, 'fit_rows': 400, 'query_rows': 100})
    validate_pca(params, 400)
    variance = {n: float(params[n+'_variance_ratio'].sum()) for n,_,_ in BLOCKS}
    return project(raw, params), variance


def import_tabicl():
    assets.validate()
    before = t.environment()
    if str(assets.DEPS) not in sys.path:
        sys.path.insert(0, str(assets.DEPS))
    import tabicl
    import psutil
    if (Path(tabicl.__file__).resolve().parent != assets.DEPS/'tabicl'
            or Path(psutil.__file__).resolve().parent != assets.DEPS/'psutil'):
        raise ValueError('Expected pinned isolated packages')
    if t.environment() != before:
        raise ValueError('Isolated imports changed archived distribution inventory')
    return tabicl.TabICLRegressor


class ICLPredictor:
    def __init__(self, device_name):
        self.model_class = import_tabicl()
        self.torch, self.device = t.configure_device(device_name, 0)
        self.runtime = {'device': str(self.device), 'gpu': self.torch.cuda.get_device_name(self.device) if self.device.type=='cuda' else None,
            'precision': 'float32', 'threads': 4, 'telemetry': False, 'hf_offline': True}

    def __call__(self, train_x, train_y, query_x, seed):
        started = time.perf_counter()
        t.configure_device(str(self.device), seed)
        if self.device.type == 'cuda':
            self.torch.cuda.reset_peak_memory_stats(self.device)
        model = self.model_class(**ICL_CONFIG, model_path=assets.ASSETS/assets.WEIGHT,
            device=str(self.device), random_state=seed)
        try:
            model.fit(train_x.copy(), train_y.copy())
            pred = np.asarray(model.predict(query_x.copy()), dtype=np.float64).reshape(-1)
            if pred.shape != (len(query_x),) or not np.isfinite(pred).all():
                raise ValueError('Invalid TabICL prediction')
            if self.device.type == 'cuda':
                self.torch.cuda.synchronize(self.device)
            return pred, {'seconds': time.perf_counter()-started, 'runtime': self.runtime,
                'context_rows': len(train_x), 'query_rows': len(query_x), 'columns': train_x.shape[1],
                'config': ICL_CONFIG, 'checkpoint_sha256': assets.WEIGHT_SHA,
                'peak_gpu_allocated_bytes': int(self.torch.cuda.max_memory_allocated(self.device)) if self.device.type=='cuda' else None}
        finally:
            del model
            gc.collect()
            if self.device.type == 'cuda':
                self.torch.cuda.empty_cache()


def make_predictors(device, ref):
    pfn = t.Predictor(t.verify_weights(t.OUT), device)
    if pfn.runtime != ref['runtime']:
        raise ValueError(f'Use exact archived runtime; no fallback: {ref["runtime"]}')
    icl = ICLPredictor(device)
    if icl.runtime != ref['runtime']:
        raise ValueError('TabICL runtime differs from archived FP32 GPU runtime')
    return {'tabpfn96': pfn, 'tabicl486': icl, 'tabicl96': icl}


def evaluate_fold(folder, binding, raw, y, folds, seed, fold, predictors, allow_fit):
    ti, vi = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
    projected, variance = pca_cell(folder/f'fold{fold}_pca', binding, raw, ti, vi, allow_fit)
    matrices = {'tabpfn96': projected, 'tabicl486': raw, 'tabicl96': projected}
    predictions = {n: t.run_cell(folder/f'fold{fold}_{n}', binding, n, x, y[ti], ti, vi,
        seed+1009*fold+9, predictors[n] if allow_fit else forbidden) for n,x in matrices.items()}
    return vi, predictions, variance


def result_data(data, ref, predictions, common_ids):
    output = {'tabpfn486': ref['tabpfn486'], **predictions, 'current486': ref['current486'],
              **{BLENDS[n]: .5*(ref['base']+p) for n,p in predictions.items()}}
    if set(predictions) != set(REPS) or any(p.shape != data.y.shape or not np.isfinite(p).all() for p in output.values()):
        raise ValueError('Invalid OOF predictions')
    np.testing.assert_array_equal(output['current486'], .5*(ref['base']+output['tabpfn486']))
    metrics = {n: core.metrics(data.y, p) for n,p in output.items()}
    for n in REPS:
        for name in (n, BLENDS[n]):
            matched = ('tabpfn96' if n == 'tabicl96' else 'tabpfn486') if name==n else CONTROLS[n]
            metrics[name]['gain_vs_current486'] = metrics['current486']['rmse']-metrics[name]['rmse']
            metrics[name]['matched_control'] = matched
            metrics[name]['gain_vs_matched_control'] = metrics[matched]['rmse']-metrics[name]['rmse']
    target, noise = core.rank_groups(data.y,10), core.rank_groups(data.noise,4)
    common = np.isin(data.ids, common_ids)
    if common.sum() != 16 or len(set(common_ids)) != 16:
        raise ValueError('Expected historical common16')
    masks = {'low10': target==0, 'high10': target==9, 'middle80': (target>0)&(target<9),
        'tails20': (target==0)|(target==9), 'common16': common, 'remaining484': ~common}
    masks.update({f'noiseQ{k+1}': noise==k for k in range(4)})
    masks.update({f'fold{f}': ref['folds']==f for f in range(5)})
    worst = {n: set(np.argsort((p-data.y)**2,kind='stable')[-20:].tolist()) for n,p in output.items()}
    for n in ('current486', *BLENDS.values()):
        masks[n+'_worst20'] = np.isin(np.arange(len(data.ids)), list(worst[n]))
    diag = [{'model': n, 'group': g, **core.metrics(data.y[m],p[m])} for n,p in output.items() for g,m in masks.items() if m.any()]
    rows = [{'ID': str(i), 'hardness': float(data.y[j]), 'fold': int(ref['folds'][j]),
             **{n: float(p[j]) for n,p in output.items()}} for j,i in enumerate(data.ids)]
    errors = [{'ID': str(data.ids[j]), 'hardness': float(data.y[j]), 'fold': int(ref['folds'][j]),
        'common16': bool(common[j]), 'noise': float(data.noise[j]), **{n+'_error': float(p[j]-data.y[j]) for n,p in output.items()},
        **{n+'_squared_error_gain_vs_current486': float((ref['current486'][j]-data.y[j])**2-(p[j]-data.y[j])**2)
           for n,p in output.items() if n in BLENDS.values()}}
        for j in np.argsort(-(output[PRIMARY]-data.y)**2,kind='stable')]
    detail = {}
    old = ref['current486']-data.y
    for rep in REPS:
        n = BLENDS[rep]
        err = output[n]-data.y
        group_rmse = {g: {key: core.metrics(data.y[masks[g]],p[masks[g]])['rmse'] for key,p in output.items()}
                      for g in ('common16','remaining484')}
        detail[n] = {**{g+'_rmse': m for g,m in group_rmse.items()},
            'common16_improved_vs_current486': int(((err**2<old**2)&common).sum()),
            'common16_same_error_sign': int((np.sign(err[common])==np.sign(old[common])).sum()),
            'common16_in_candidate_worst20': int(common[list(worst[n])].sum()),
            'common16_sse_fraction': float(np.sum(err[common]**2)/np.sum(err**2)) if np.any(err) else 0.,
            'both_groups_improve_vs_current486': all(m[n]<m['current486'] for m in group_rmse.values()),
            'new_worst20_ids': [str(data.ids[i]) for i in sorted(worst[n]-worst['current486'])],
            'left_worst20_ids': [str(data.ids[i]) for i in sorted(worst['current486']-worst[n])],
            'residual_correlation_vs_current486': core.correlation(err,old),
            'standalone_gain_vs_matched_control': metrics[rep]['gain_vs_matched_control'],
            'folds_improved_vs_current486': sum(core.metrics(data.y[masks[f'fold{f}']], output[n][masks[f'fold{f}']])['rmse']
                < core.metrics(data.y[masks[f'fold{f}']], ref['current486'][masks[f'fold{f}']])['rmse'] for f in range(5))}
    return metrics, rows, diag, errors, detail


def evaluation_binding(plan, ref, seed):
    return {'plan': t.digest(plan['frozen']), 'seed': seed, 'runtime': ref['runtime']}


def expected_outputs():
    return {'summary.json','oof.csv','diagnostics.csv','per_image_errors.csv','common16.csv','report.txt'} | {
        f'fold{f}_{n}/completed.json' for f in range(5) for n in REPS} | {
        f'fold{f}_pca/{n}' for f in range(5) for n in ('parameters.npz','completed.json')}


def completed(folder, binding):
    if not (folder/'completed_evaluation.json').exists():
        return False
    if set(t.read_json(folder/'completed_evaluation.json')['files']) != expected_outputs():
        raise ValueError('Incomplete result manifest')
    return t.verify_completed(folder,binding)


def collect(folder,data,raw,ref,plan,seed,predictors,allow_fit):
    pred, variance = {n: np.full(500,np.nan) for n in REPS}, []
    for fold in range(5):
        vi,p,v = evaluate_fold(folder,evaluation_binding(plan,ref,seed),raw,data.y,ref['folds'],seed,fold,predictors,allow_fit)
        for n in REPS:
            pred[n][vi] = p[n]
        variance.append(v)
        if allow_fit:
            print(f'Outer{fold}: three fixed cells stored; no outer score used; PCA variance={v}',flush=True)
    return pred,variance


def save_results(folder,data,ref,pred,variance,plan,seed):
    binding = evaluation_binding(plan,ref,seed)
    metrics,rows,diag,errors,detail = result_data(data,ref,pred,plan['frozen']['common_failure_ids'])
    for name,records in (('oof.csv',rows),('diagnostics.csv',diag),('per_image_errors.csv',errors),
                          ('common16.csv',[r for r in errors if r['common16']])):
        t.write_csv(folder/name,records)
    t.write_json(folder/'summary.json',{'seed':seed,'binding':binding,'metrics':metrics,'diagnosis':detail,
        'pca_variance_retained_by_fold':variance,'primary':PRIMARY,'secondary':SECONDARY,
        'new_context_fits':15,'new_fold_local_block_pca_fits':5,'archived486_contexts_reused':5,
        'inner_search':False,'independent_holdout':False,'test_accessed':False,'attribution':PROTOCOL['attribution']})
    lines = [f'TabICLv2 x raw486/blockPCA96: seed{seed}',PROTOCOL['attribution'],'',
        f'Primary={PRIMARY}; secondary={SECONDARY}; no best-of-two selection.',PROTOCOL['blend'],'']
    for n,m in metrics.items():
        line = f'{n}: RMSE={m["rmse"]:.6f}'
        if 'matched_control' in m:
            line += f'; gain vs current486={m["gain_vs_current486"]:+.6f}; gain vs {m["matched_control"]}={m["gain_vs_matched_control"]:+.6f}'
        lines.append(line)
    lines += ['', 'PCA variance by fold/block: '+t.common.canonical_json(variance),
        'Check common16.csv, diagnostics.csv, per_image_errors.csv and summary.json.',PROTOCOL['decision'],PROTOCOL['scope']]
    (folder/'report.txt').write_text('\n'.join(lines)+'\n',encoding='utf8')
    t.write_json(folder/'completed_evaluation.json',{'binding':binding,
        'files':{n:t.common.sha256_file(folder/n) for n in sorted(expected_outputs())}})
    print('\n'.join(lines[5:5+len(metrics)]),flush=True)
    print(f'Report: {folder/"report.txt"}')


def verify_results(folder,data,raw,ref,plan,seed):
    if not completed(folder,evaluation_binding(plan,ref,seed)):
        raise ValueError(f'Incomplete seed{seed}; run evaluate first')
    with contextlib.redirect_stdout(io.StringIO()):
        pred,variance = collect(folder,data,raw,ref,plan,seed,{},False)
    metrics,rows,_,_,detail = result_data(data,ref,pred,plan['frozen']['common_failure_ids'])
    summary = t.read_json(folder/'summary.json')
    if summary['metrics'] != metrics or summary['diagnosis'] != detail or summary['pca_variance_retained_by_fold'] != variance:
        raise ValueError('Recomputed metrics/PCA diagnostics differ')
    saved = core.align_rows(core.read_rows(folder/'oof.csv',set(rows[0])),data.ids,folder)
    for n in rows[0]:
        if n != 'ID':
            np.testing.assert_array_equal([float(r[n]) for r in saved],[float(r[n]) for r in rows])
    return metrics,detail


def check_binding(plan,ref):
    return {'plan':t.digest(plan['frozen']),'runtime':ref['runtime'],'synthetic_seed':314159,
        'repeat_rtol':1e-6,'repeat_atol':1e-5,'dimensions':[96,486]}


def verify_check(folder,plan,ref):
    path = folder/'synthetic_check.json'
    if not path.exists():
        raise ValueError('Run check successfully before evaluate; GPU capacity not yet verified.')
    saved = t.read_json(path)
    if saved['binding'] != check_binding(plan,ref) or saved['result_sha256'] != t.digest(saved['result']):
        raise ValueError('Synthetic check binding/result changed')
    if set(saved['result']['models']) != set(REPS) or not all(m['finite'] and m['repeatable'] for m in saved['result']['models'].values()):
        raise ValueError('Synthetic check did not pass all fixed cells')


def check(folder,device):
    _,_,refs,plan = validate(folder)
    if (folder/'synthetic_check.json').exists():
        verify_check(folder,plan,refs[42])
        print('Synthetic checks verified; no repeated inference.')
        return
    predictors = make_predictors(device,refs[42])
    rng = np.random.default_rng(314159)
    raw = rng.normal(size=(500,486)); raw[:,0]=0
    params = fit_pca(raw[:400]); validate_pca(params,400)
    pca = project(raw,params)
    y = 100+3*raw[:,2]-2*raw[:,3]+rng.normal(size=500)
    result = {'competition_data_used_for_models':False,'gradient_updates':0,'models':{}}
    for name,x in (('tabpfn96',pca),('tabicl486',raw),('tabicl96',pca)):
        print(f'Synthetic400/100 {name}: checking finite predictions, memory and repeatability...',flush=True)
        pred,meta = predictors[name](x[:400],y[:400],x[400:],314159)
        repeat,_ = predictors[name](x[:400],y[:400],x[400:],314159)
        np.testing.assert_allclose(pred,repeat,rtol=1e-6,atol=1e-5)
        result['models'][name] = {'finite':bool(np.isfinite(pred).all()),'repeatable':True,
            'shape':list(pred.shape),'max_repeat_difference':float(np.max(np.abs(pred-repeat))),'metadata':meta}
        print(f'Synthetic {name} passed; first pass={meta["seconds"]:.1f}s',flush=True)
    t.write_json(folder/'synthetic_check.json',{'binding':check_binding(plan,refs[42]),'result':result,'result_sha256':t.digest(result)})


def evaluate(folder,device,seed):
    data,raw,refs,plan = validate(folder)
    ref,sub = refs[seed],folder/f'seed{seed}'
    sub.mkdir(parents=True,exist_ok=True)
    if completed(sub,evaluation_binding(plan,ref,seed)):
        verify_results(sub,data,raw,ref,plan,seed)
        print(f'Completed seed{seed} verified; no refitting or inference.')
        return
    verify_check(folder,plan,ref)
    predictors = make_predictors(device,ref)
    pred,variance = collect(sub,data,raw,ref,plan,seed,predictors,True)
    validate(folder)
    save_results(sub,data,ref,pred,variance,plan,seed)


def aggregate(records):
    if [r['seed'] for r in records] != list(SEEDS):
        raise ValueError('Both fixed seeds required in order')
    variants = {}
    for rep in REPS:
        name = BLENDS[rep]
        mean = float(np.mean([r['variants'][name]['gain_vs_current486'] for r in records]))
        positive = all(r['variants'][name]['gain_vs_current486']>0 and
                       r['variants'][name]['gain_vs_matched_control']>0 for r in records)
        variants[name] = {'mean_gain_vs_current486':mean,
            'mean_gain_vs_matched_control':float(np.mean([r['variants'][name]['gain_vs_matched_control'] for r in records])),
            'all_seeds_improve_both':positive,'passes_expansion_priority':positive and mean>=.30,
            'both_groups_improve_in_both_seeds':all(r['diagnosis'][name]['both_groups_improve_vs_current486'] for r in records)}
    return {'primary':PRIMARY,'secondary':SECONDARY,'per_seed':records,'variants':variants,
        'selection_performed':False,'additional_review_required':'Standalone, common16/remaining484, tails, noise and new worst20; no automatic adoption.',
        'aggregation':'Mean per-seed RMSE differences; same500; no prediction averaging or independent inference.'}


def compare(folder):
    data,raw,refs,plan = validate(folder)
    records = []
    for seed in SEEDS:
        metrics,detail = verify_results(folder/f'seed{seed}',data,raw,refs[seed],plan,seed)
        records.append({'seed':seed,'current486_rmse':metrics['current486']['rmse'],
            'variants':{n:metrics[n] for n in BLENDS.values()},'diagnosis':detail})
    result = {**aggregate(records),'protocol_sha256':t.digest(plan['frozen'])}
    t.write_json(folder/'comparison.json',result)
    print(t.common.canonical_json(result),flush=True)
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('prepare','validate','check','evaluate','compare'))
    p.add_argument('--output-dir',type=t.common.project_path,default=OUT)
    p.add_argument('--seed',type=int,choices=SEEDS,default=42)
    p.add_argument('--device',default='cuda')
    a = p.parse_args(argv)
    if a.command=='prepare': prepare(a.output_dir)
    elif a.command=='validate': validate(a.output_dir); print('Inputs/assets/baseline/protocol verified; no fitting.')
    elif a.command=='check': check(a.output_dir,a.device)
    elif a.command=='evaluate': evaluate(a.output_dir,a.device,a.seed)
    else: compare(a.output_dir)


if __name__=='__main__':
    main()
