"""User-run scattering201 + existing285 controlled local TabPFN-v2 experiment.

Built with PriorLabs-TabPFN. prepare/validate: no image forward or regression.
check-features: synthetic CPU only. check: synthetic486 TabPFN GPU smoke test.
extract: native train500 CPU features. evaluate: five outer context fits per seed.
No test access, gradient updates, full500 supervised fit, or submission changes.
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
    from . import evaluate_boundary_tabpfn as archive
    from . import scattering_features as sf
else:
    import evaluate_boundary_tabpfn as archive
    import scattering_features as sf

t = archive.t
ROOT = t.ROOT
OUT = ROOT / 'reports/scattering_tabpfn'
SEEDS = (42, 2026)
REP = 'morphology285_scattering201'
PRIMARY = 'scattering_replaced_blend'
PROTOCOL = {
    'version': 1, 'attribution': 'Built with PriorLabs-TabPFN', 'seeds': list(SEEDS),
    'dimensions': {'existing': 285, 'scattering': 201, 'combined': 486},
    'input': 'Native256x256 uint8 grayscale /255 float64. No resize, crop, D4, denoise or per-image contrast normalization.',
    'scattering': {'library': 'kymatio0.3.0', 'backend': 'numpy CPU', 'J': 5, 'L': 8, 'max_order': 2,
        'frontend': 'Direct ScatteringNumPy2D avoids unused3D sph_harm import; unmodified vendor code.',
        'batch': 1, 'threads': 2, 'padding': 'Library reflect32 each side to320x320; unpad final10x10 to8x8.',
        'path_validation': 'All681 unique (j,theta) keys and8x8 maps. Never rely on path ordering.',
        'summary': 'Spatial mean S0(1), S1 in j/theta order(40), S2 in j1<j2/delta order(80), '
                   'S2 normalized by mean_theta(S1 at j1)+1e-6(80). '
                   'S2[j1,j2,delta]=mean_theta1 coefficient[j1,j2,theta1,(theta1+delta)%8]. '
                   'Retain S1 absolute orientation. No clipping, PCA, learned filter or feature selection.',
        'feature_names': list(sf.NAMES), 'denominator_epsilon': 1e-6},
    'regressor': t.PROTOCOL,
    'evaluation_override': 'Fixed285 vs fixed486. No new inner representation/hyperparameter search. '
        'Same5 outer folds, context400/query100, seed+1009*fold+9, n_estimators8, FP32, exact archived runtime.',
    'control': 'Verify and reuse existing fixed285 cells. Current historical OOF retains nested185/285 choice; '
               'report both current and fixed285 control. No silent control refits.',
    'blend': 'Base=hand185*.25+originalResNet*.25+DINO_D4*.5. '
             'Candidate=.5*Base+.5*TabPFN486. Control=.5*Base+.5*TabPFN285. No weight search or clipping.',
    'primary': PRIMARY, 'priority_mean_gain': .30,
    'decision': 'Both seeds improve current AND fixed285 control; expansion priority additionally '
                'mean RMSE gain vs current>=0.30. Review standalone, middle80%, tails, noise, '
                'common16, remaining484 and new worst20. No automatic adoption; no same-grid micro-tuning.',
    'scope': 'Train only;10 new context fits total. Extraction does not accept targets. '
             'No test, neural training, final500 regression or submission.',
    'interpretation': 'Same repeatedly studied500; no independent holdout, significance or physical segmentation claim.',
}


def load_state():
    data, features, old, source_plan = t.validate(t.OUT)
    refs = {s: archive.archived_reference(data, features, old[s], source_plan, s) for s in SEEDS}
    for s in SEEDS:
        np.testing.assert_allclose(refs[s]['base'], .25*old[s]['hand']+.25*old[s]['resnet']+.5*old[s]['dino'], rtol=0, atol=1e-10)
    audit_path = ROOT/'reports/model_portfolio_audit_20261011/numeric_audit.json'
    common = t.read_json(audit_path)['cross_seed']['common_worst20_ids']
    if len(set(common)) != 16 or not set(common).issubset(set(data.ids)):
        raise ValueError('Historical common16 IDs differ')
    inventory = [{'ID': str(i), 'sha256': t.common.sha256_file(ROOT/'open/train'/f'{i}.png')} for i in data.ids]
    frozen = {'protocol': PROTOCOL, 'sources': {Path(p).name: t.file_ref(p) for p in
                  (__file__, sf.__file__, archive.__file__, t.__file__, ROOT/'requirements-scattering.txt')},
        'dependencies': sf.dependency_binding(), 'environment': t.environment(),
        'source_plan': t.file_ref(t.OUT/'protocol.json'), 'source_payload': t.digest(source_plan['frozen']),
        'image_inventory': inventory, 'target_sha256': t.common.sha256_array(data.y),
        'noise_sha256': t.common.sha256_array(data.noise),
        'original285_sha256': t.common.sha256_array(features['morphology285']),
        'audit': t.file_ref(audit_path), 'common_failure_ids': common,
        'references': {str(s): {'archive': refs[s]['archive'], 'runtime': refs[s]['runtime'],
            'arrays': {n: t.common.sha256_array(refs[s][n]) for n in ('folds','base','current','tabpfn285','control')}} for s in SEEDS},
        'submissions': [t.file_ref(ROOT/p) for p in ('reports/final_ensemble/submission/submission.csv',
                                                   'reports/final_tabpfn_blend/submission/submission.csv')]}
    return data, features['morphology285'], refs, frozen


def prepare(folder):
    *_, frozen = load_state()
    path = folder/'protocol.json'
    if path.exists():
        if t.read_json(path)['frozen'] != frozen:
            raise ValueError('Frozen protocol changed; preserve existing experiment')
        print('Existing preparation matches; no extraction or fitting'); return
    if folder.exists() and any(folder.iterdir()):
        raise ValueError('Nonempty unprepared output directory')
    t.write_json(path, {'frozen': frozen, 'created_unix': time.time(), 'competition_fits': 0, 'competition_forwards': 0})
    print(f'Prepared fixed285 vs486,201 new features,10 new outer context fits planned: {path}')


def validate(folder):
    data, base, refs, frozen = load_state()
    plan = t.read_json(folder/'protocol.json')
    if plan['frozen'] != frozen or plan['competition_fits'] != 0 or plan['competition_forwards'] != 0:
        raise ValueError('Frozen sources, dependencies, inputs or settings changed')
    return data, base, refs, plan


def extraction_cell(folder, plan, item, extractor):
    source = ROOT/'open/train'/f"{item['ID']}.png"
    if t.common.sha256_file(source) != item['sha256']:
        raise ValueError('Image changed before extraction/cache reuse')
    binding = {'plan': t.digest(plan['frozen']), 'image': item}
    path = folder/'features/cells'/f"{item['ID']}.json"
    if path.exists():
        saved = t.read_json(path)
        if saved['binding'] != binding or saved['result_sha256'] != t.digest(saved['result']):
            raise ValueError('Extraction cell changed')
        row = np.array(saved['result']['features'], dtype=np.float64)
    else:
        row, detail = extractor(sf.read_gray(source))
        sf.validate_matrix(row[None, :])
        if t.common.sha256_file(source) != item['sha256']:
            raise ValueError('Image changed during extraction')
        result = {'features': row.tolist(), 'detail': detail}
        t.write_json(path, {'binding': binding, 'result': result, 'result_sha256': t.digest(result)})
    sf.validate_matrix(row[None, :])
    return row


def load_features(folder, data, plan):
    target = folder/'features'
    if not (target/'metadata.json').exists():
        raise FileNotFoundError('Scattering cache incomplete. Run extract successfully before evaluate.')
    meta = t.read_json(target/'metadata.json')
    if meta['plan'] != t.digest(plan['frozen']) or meta['npz_sha256'] != t.common.sha256_file(target/'features.npz'):
        raise ValueError('Feature metadata/binding changed')
    if set(meta['cells']) != {f'cells/{i}.json' for i in data.ids}:
        raise ValueError('Incomplete feature cell set')
    with np.load(target/'features.npz', allow_pickle=False) as z:
        if set(z.files) != {'ids','names','scattering'}:
            raise ValueError('Unexpected cache schema')
        np.testing.assert_array_equal(z['ids'], data.ids)
        np.testing.assert_array_equal(z['names'], np.array(sf.NAMES))
        x = z['scattering'].copy()
    sf.validate_matrix(x)
    if x.shape != (500,201) or t.common.sha256_array(x) != meta['array_sha256']:
        raise ValueError('Invalid feature array')
    for j, item in enumerate(plan['frozen']['image_inventory']):
        name = f"cells/{item['ID']}.json"
        if t.common.sha256_file(target/name) != meta['cells'][name]:
            raise ValueError('Feature cell digest changed')
        cell = t.read_json(target/name)
        if cell['binding'] != {'plan': t.digest(plan['frozen']), 'image': item} or cell['result_sha256'] != t.digest(cell['result']):
            raise ValueError('Invalid cell binding/payload')
        np.testing.assert_array_equal(x[j], cell['result']['features'])
    return x, meta


def extract(folder):
    data, _, _, plan = validate(folder)
    if (folder/'features/metadata.json').exists():
        load_features(folder, data, plan); print('Completed features verified; no re-extraction'); return
    started = time.perf_counter()
    extractor = sf.Extractor()
    rows = []
    for j, item in enumerate(plan['frozen']['image_inventory']):
        rows.append(extraction_cell(folder, plan, item, extractor))
        if (j+1) % 25 == 0 or j == 0:
            print(f'Extracted/reused {j+1}/500 train images; scattering201; elapsed={time.perf_counter()-started:.1f}s', flush=True)
    validate(folder)
    x = np.stack(rows); sf.validate_matrix(x)
    target = folder/'features'; target.mkdir(exist_ok=True, parents=True)
    with (target/'features.npz.tmp').open('wb') as h:
        np.savez_compressed(h, ids=data.ids, names=np.array(sf.NAMES), scattering=x)
    os.replace(target/'features.npz.tmp', target/'features.npz')
    t.write_json(target/'metadata.json', {'plan': t.digest(plan['frozen']),
        'npz_sha256': t.common.sha256_file(target/'features.npz'), 'array_sha256': t.common.sha256_array(x),
        'cells': {f'cells/{i}.json': t.common.sha256_file(target/'cells'/f'{i}.json') for i in data.ids},
        'dimensions': [500,201], 'combined_dimensions': [500,486], 'original285_changed': False,
        'labels_used_for_features': False, 'test_accessed': False, 'seconds': time.perf_counter()-started,
        'column_min': x.min(0).tolist(), 'column_max': x.max(0).tolist(),
        'note': 'Ranges are descriptive only; no adaptive clipping, feature deletion or selection.'})
    load_features(folder, data, plan)
    print(f'Saved {target / "features.npz"}; scattering=(500,201); candidate=(500,486); original285 unchanged')


def evaluate_fold(folder, binding, x, y, folds, seed, fold, predictor):
    ti, vi = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
    if (len(ti), len(vi)) != (400,100) or x.shape != (500,486) or not np.isfinite(x).all():
        raise ValueError('Expected finite486 features, outer400/100')
    pred = t.run_cell(folder/f'fold{fold}_candidate', binding, REP, x, y[ti], ti, vi, seed+1009*fold+9, predictor)
    return vi, pred


def result_data(data, ref, pred, common_ids):
    allp = {'current': ref['current'], 'tabpfn285_control': ref['tabpfn285'], 'tabpfn486': pred,
            'fixed285_control_blend': ref['control'], PRIMARY: .5*(ref['base']+pred)}
    if any(p.shape != data.y.shape or not np.isfinite(p).all() for p in allp.values()):
        raise ValueError('Invalid OOF predictions')
    current_rmse = t.core.metrics(data.y, ref['current'])['rmse']
    control_rmse = t.core.metrics(data.y, ref['control'])['rmse']
    metrics = {n: {**t.core.metrics(data.y,p), 'gain_vs_current': current_rmse-t.core.metrics(data.y,p)['rmse'],
                  'gain_vs_control': control_rmse-t.core.metrics(data.y,p)['rmse']} for n,p in allp.items()}
    target, noise = t.core.rank_groups(data.y,10), t.core.rank_groups(data.noise,4)
    common = np.isin(data.ids,common_ids)
    masks = {'low10': target==0, 'high10': target==9, 'tails20': (target==0)|(target==9),
             'middle80': (target>0)&(target<9), 'common16': common, 'remaining484': ~common}
    masks.update({f'noiseQ{k+1}': noise==k for k in range(4)})
    masks.update({f'fold{f}': ref['folds']==f for f in range(5)})
    for name in ('current', PRIMARY):
        ix = np.argsort((allp[name]-data.y)**2,kind='stable')[-20:]
        masks[name+'_worst20'] = np.isin(np.arange(len(data.ids)),ix)
    diag = [{'model': n, 'group': g, **t.core.metrics(data.y[m],p[m]),
             'sse': float(np.sum((data.y[m]-p[m])**2))} for n,p in allp.items() for g,m in masks.items() if m.any()]
    rows = [{'ID': str(i), 'hardness': float(data.y[j]), 'fold': int(ref['folds'][j]),
             **{n: float(p[j]) for n,p in allp.items()}} for j,i in enumerate(data.ids)]
    errors = [{'ID': str(data.ids[j]), 'hardness': float(data.y[j]), 'fold': int(ref['folds'][j]),
               'common16': bool(common[j]), 'current_error': float(ref['current'][j]-data.y[j]),
               'control_error': float(ref['control'][j]-data.y[j]),
               'candidate_error': float(allp[PRIMARY][j]-data.y[j]),
               'gain_squared_error_vs_current': float((ref['current'][j]-data.y[j])**2-(allp[PRIMARY][j]-data.y[j])**2),
               'gain_squared_error_vs_control': float((ref['control'][j]-data.y[j])**2-(allp[PRIMARY][j]-data.y[j])**2)}
              for j in np.argsort(-(allp[PRIMARY]-data.y)**2,kind='stable')]
    return metrics,rows,diag,errors


def expected_outputs():
    return {'oof.csv','summary.json','diagnostics.csv','per_image_errors.csv','report.md'} | {f'fold{f}_candidate/completed.json' for f in range(5)}


def completed(folder, binding):
    if not (folder/'completed_evaluation.json').exists(): return False
    if set(t.read_json(folder/'completed_evaluation.json')['files']) != expected_outputs():
        raise ValueError('Incomplete result set')
    return t.verify_completed(folder,binding)


def save_results(folder,data,ref,pred,plan,binding,seed):
    metrics,rows,diag,errors = result_data(data,ref,pred,plan['frozen']['common_failure_ids'])
    for name,records in (('oof.csv',rows),('diagnostics.csv',diag),('per_image_errors.csv',errors)):
        t.write_csv(folder/name,records)
    t.write_json(folder/'summary.json', {'seed': seed, 'binding': binding, 'metrics': metrics,
        'primary': PRIMARY, 'new_context_fits': 5, 'archived_control_fits_reused': 5,
        'inner_search': False, 'independent_holdout': False, 'test_accessed': False,
        'attribution': 'Built with PriorLabs-TabPFN'})
    lines = [f'# Scattering201 + TabPFN-v2 · seed{seed}', '', 'Built with PriorLabs-TabPFN.', '',
             '| Model | OOF RMSE | Gain vs current | Gain vs fixed285 blend |', '|---|---:|---:|---:|']
    lines += [f"| {n} | {m['rmse']:.6f} | {m['gain_vs_current']:+.6f} | {m['gain_vs_control']:+.6f} |" for n,m in metrics.items()]
    lines += ['', f'Primary: {PRIMARY}; base50% + fixed486 TabPFN50%.',
              'Fixed285 control isolates added features from historical185/285 inner selection.',
              'No new inner search; preprocessing fitted within each context400; original8-estimator FP32 runtime.',
              'Review diagnostics/per_image_errors: standalone, middle80%, tails, noise, common16, remaining484 and new worst20.',
              'Two-seed mean gain>=0.30 is expansion priority only, not significance or automatic adoption.',
              'Same500 repeatedly studied; no independent holdout or test score. No new submission.']
    (folder/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    t.write_json(folder/'completed_evaluation.json', {'binding': binding,
        'files': {n: t.common.sha256_file(folder/n) for n in sorted(expected_outputs())}})
    for name in ('tabpfn486','fixed285_control_blend',PRIMARY):
        m=metrics[name]
        print(f"{name}: RMSE={m['rmse']:.6f}; gain vs current={m['gain_vs_current']:+.6f}; gain vs control={m['gain_vs_control']:+.6f}",flush=True)
    print(f"Report: {folder/'report.md'}")


def evaluation_binding(folder,plan,ref,seed):
    return {'plan': t.digest(plan['frozen']), 'feature_metadata': t.file_ref(folder/'features/metadata.json'),
            'seed': seed, 'runtime': ref['runtime']}


def evaluate(folder,device,seed):
    data,base,refs,plan = validate(folder); added,_ = load_features(folder,data,plan)
    target=folder/f'seed{seed}'; target.mkdir(parents=True,exist_ok=True)
    binding=evaluation_binding(folder,plan,refs[seed],seed)
    if completed(target,binding):
        print(f'Completed seed{seed} verified; no refitting')
    else:
        predictor=t.Predictor(t.verify_weights(t.OUT),device)
        if predictor.runtime != refs[seed]['runtime']:
            raise ValueError('Use exact archived device/runtime; no CPU/precision fallback: '+str(refs[seed]['runtime']))
        x=np.concatenate((base,added),axis=1); pred=np.full(len(data.ids),np.nan)
        for fold in range(5):
            vi,p=evaluate_fold(target,binding,x,data.y,refs[seed]['folds'],seed,fold,predictor); pred[vi]=p
        validate(folder); load_features(folder,data,plan)
        save_results(target,data,refs[seed],pred,plan,binding,seed)
    if all((folder/f'seed{s}/completed_evaluation.json').exists() for s in SEEDS): compare(folder)


def aggregate(records):
    if [r['seed'] for r in records] != list(SEEDS): raise ValueError('Both fixed seeds required in order')
    mean=float(np.mean([r['gain_vs_current'] for r in records]))
    positive=all(r['gain_vs_current']>0 and r['gain_vs_control']>0 for r in records)
    return {'primary': PRIMARY, 'per_seed': records, 'mean_gain_vs_current': mean,
        'mean_gain_vs_control': float(np.mean([r['gain_vs_control'] for r in records])),
        'all_seeds_improve_both': positive, 'passes_expansion_priority': positive and mean>=.30,
        'additional_review_required': 'Standalone, middle80%, tails, noise, common16, remaining484, new worst20; no automatic adoption.',
        'aggregation': 'Mean per-seed OOF RMSE differences; same500, no prediction averaging or independent inference.'}


def compare(folder):
    data,base,refs,plan=validate(folder);added,_=load_features(folder,data,plan)
    x=np.concatenate((base,added),axis=1);records=[]
    def forbidden(*args): raise ValueError('Completed cell missing; no implicit fitting')
    for seed in SEEDS:
        target=folder/f'seed{seed}';binding=evaluation_binding(folder,plan,refs[seed],seed)
        if not completed(target,binding): raise ValueError('Expected completed seed results')
        pred=np.full(len(data.ids),np.nan)
        with contextlib.redirect_stdout(io.StringIO()):
            for fold in range(5):
                vi,p=evaluate_fold(target,binding,x,data.y,refs[seed]['folds'],seed,fold,forbidden);pred[vi]=p
        metrics,rows,_,_=result_data(data,refs[seed],pred,plan['frozen']['common_failure_ids'])
        if t.read_json(target/'summary.json')['metrics'] != metrics: raise ValueError('Metric recomputation differs')
        saved=t.core.align_rows(t.core.read_rows(target/'oof.csv',set(rows[0])),data.ids,target)
        for key in rows[0]:
            if key!='ID': np.testing.assert_array_equal([float(r[key]) for r in saved],[float(r[key]) for r in rows])
        m=metrics[PRIMARY]
        records.append({'seed': seed, 'current_rmse': metrics['current']['rmse'],
            'fixed285_control_rmse': metrics['fixed285_control_blend']['rmse'],
            'candidate_rmse': m['rmse'], 'gain_vs_current': m['gain_vs_current'], 'gain_vs_control': m['gain_vs_control']})
    result={**aggregate(records),'protocol_sha256': t.digest(plan['frozen'])}
    t.write_json(folder/'comparison.json',result);print(t.common.canonical_json(result),flush=True)


def check_features(folder):
    # Dependencies only, never calls load_state or reads competition images.
    environment=t.environment(); result=sf.synthetic_check()
    if t.environment()!=environment: raise ValueError('Isolated import contaminated environment inventory')
    t.write_json(folder/'synthetic_features_check.json',result)
    print(t.common.canonical_json(result))


def check(folder,device):
    # Only synthetic arrays; no competition labels/images/predictions loaded.
    sf.dependency_binding()
    predictor=t.Predictor(t.verify_weights(t.OUT),device)
    rng=np.random.default_rng(314159);x=rng.normal(size=(500,486));x[:,0]=0.
    y=100+3*x[:,2]-2*x[:,3]+rng.normal(size=len(x))
    pred,meta=predictor(x[:400],y[:400],x[400:],314159)
    t.write_json(folder/f'synthetic_tabpfn_check_{predictor.device.type}.json',
        {'competition_data_used': False, 'gradient_updates': 0, 'columns': 486,
         'finite': bool(np.isfinite(pred).all()), 'shape': list(pred.shape), 'metadata': meta})
    print(f'Synthetic400/100 with486 columns passed on {predictor.device}; {meta["seconds"]:.1f}s')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('prepare','validate','check-features','check','extract','evaluate','compare'))
    p.add_argument('--output-dir',type=t.common.project_path,default=OUT)
    p.add_argument('--seed',type=int,choices=SEEDS,default=42);p.add_argument('--device',default='cuda')
    a=p.parse_args(argv)
    if a.command in ('check-features','extract'):
        with threadpool_limits(limits=2):
            if a.command=='check-features': check_features(a.output_dir)
            else: extract(a.output_dir)
    elif a.command=='prepare': prepare(a.output_dir)
    elif a.command=='validate': validate(a.output_dir);print('Frozen protocol/reference controls match; no extraction/fitting')
    elif a.command=='check': check(a.output_dir,a.device)
    elif a.command=='evaluate': evaluate(a.output_dir,a.device,a.seed)
    else: compare(a.output_dir)


if __name__=='__main__': main()
