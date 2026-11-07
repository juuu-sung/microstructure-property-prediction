"""User-run fixed486 versus legacy500 versus refined500 TabPFN experiment.

Built with PriorLabs-TabPFN. No segmentation training or pseudo-label adoption.
prepare/validate only audit; extract runs CPU train500; evaluate fits train400
contexts on the same outer folds. No test, final500 fit, or submission writes.
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
    from . import evaluate_scattering_tabpfn as previous
    from . import refined_structure_features as features
else:
    import evaluate_scattering_tabpfn as previous
    import refined_structure_features as features

t = previous.t
ROOT = t.ROOT
OUT = ROOT/'reports/refined_structure_tabpfn'
SEEDS = (42, 2026)
REPS = features.REPS
PRIMARY = 'refined_structure_blend'
CONTROL = 'legacy_structure_control_blend'
PROTOCOL = {
    'version': 1, 'attribution': 'Built with PriorLabs-TabPFN', 'seeds': list(SEEDS),
    'representations': {'current': 486, 'legacy14': 500, 'refined14': 500},
    'features': features.SPEC, 'detector_profile': features.detector.PROFILE,
    'regressor': t.PROTOCOL,
    'override': 'No inner search or choice between representations; two fixed candidates. '
                'Same archived CUDA FP32 local v2 checkpoint and8 estimators. No feature-limit override:500 exactly. '
                'Context400/query100, seed+1009*fold+9 for both branches;20 context fits total.',
    'control': 'Reuse verified fixed486 OOF without refitting. Additional legacy500 control '
               'uses the same14-column schema with original v2 boundary and compact objects. '
               'Refined500 uses new boundary/object proposals; neither is segmentation truth.',
    'missing': 'Only14-summary nullable columns are median-imputed on outer context400. '
               'All-missing context =>0 plus missing flag. No query-derived preprocessing.',
    'blend': 'B=hand185*.25+originalResNet*.25+DINO_D4*.5. Current=.5*B+.5*TabPFN486. '
             'Each experiment replaces only TabPFN50%. No seed averaging, weight search or clipping.',
    'primary': PRIMARY, 'priority_mean_gain': .30,
    'decision': 'Both seeds must improve current486 AND legacy500 control. Mean gain>=.30 vs current '
                'is investment priority, not statistical significance or automatic adoption. '
                'Review common16, remaining484, tails, middle80, noise, folds and new worst20.',
    'scope': 'User-run train500 extraction + OOF only. No supervised pixel masks, U-Net, test or submission. '
             'Detector/summary thresholds frozen before OOF. All500 have been studied: not independent holdout.',
    'authorization': 'User requests one experiment with the improved automatic detector. '
                     'Old training_approved=False continues to prohibit calling proposals human/semantic truth; '
                     'this new experiment uses summaries as unverified covariates only.',
}


def forbidden(*args):
    raise ValueError('Required archived cell absent; no implicit refitting')


def load_state():
    data, base, old_refs, source_plan = previous.validate(previous.OUT)
    scattering, _ = previous.load_features(previous.OUT, data, source_plan)
    x486 = np.c_[base, scattering]
    detector_plan = features.detector.validate()
    refs = {}
    for seed in SEEDS:
        folder = previous.OUT/f'seed{seed}'
        binding = previous.evaluation_binding(previous.OUT, source_plan, old_refs[seed], seed)
        if not previous.completed(folder, binding):
            raise ValueError('Complete both archived scattering OOF runs first')
        pred = np.full(500, np.nan)
        with contextlib.redirect_stdout(io.StringIO()):
            for fold in range(5):
                vi, p = previous.evaluate_fold(folder, binding, x486, data.y, old_refs[seed]['folds'], seed, fold, forbidden)
                pred[vi] = p
        metrics, rows, _, _ = previous.result_data(data, old_refs[seed], pred, source_plan['frozen']['common_failure_ids'])
        if t.read_json(folder/'summary.json')['metrics'] != metrics:
            raise ValueError('Archived486 metrics differ')
        stored = t.core.align_rows(t.core.read_rows(folder/'oof.csv', set(rows[0])), data.ids, folder)
        for key in rows[0]:
            if key != 'ID':
                np.testing.assert_array_equal([float(r[key]) for r in stored], [r[key] for r in rows])
        refs[seed] = {'folds': old_refs[seed]['folds'], 'base': old_refs[seed]['base'],
                      'current': .5*(old_refs[seed]['base']+pred), 'tabpfn486': pred,
                      'runtime': old_refs[seed]['runtime'], 'archive': t.file_ref(folder/'completed_evaluation.json')}
    anchors = {}
    for stage in ('development', 'check'):
        with np.load(features.detector.OUT/stage/'maps.npz', allow_pickle=False) as z:
            for i, rec in enumerate(detector_plan['records']):
                case = f'C{i+1:02d}'
                if case+'_before' in z.files:
                    anchors[rec['ID']] = {k: t.common.sha256_array(z[case+'_'+k]) for k in features.MAP_KEYS}
    if len(anchors) != 12:
        raise ValueError('All12 reviewed detector anchors required')
    for name, sha in detector_plan['protected_before'].items():
        if t.common.sha256_file(name) != sha:
            raise ValueError('Protected prior artifact changed: '+name)
    frozen = {'protocol': PROTOCOL,
              'sources': {Path(p).name: t.file_ref(p) for p in (__file__, features.__file__)},
              'source_plan': t.file_ref(previous.OUT/'protocol.json'),
              'detector_plan': t.file_ref(features.detector.OUT/'protocol.json'),
              'detector_stages': [t.file_ref(features.detector.OUT/s/'completion.json') for s in ('development','check')],
              'source_features': t.file_ref(previous.OUT/'features/metadata.json'),
              'feature486_sha256': t.common.sha256_array(x486), 'target_sha256': t.common.sha256_array(data.y),
              'noise_sha256': t.common.sha256_array(data.noise),
              'common_failure_ids': source_plan['frozen']['common_failure_ids'],
              'image_inventory': source_plan['frozen']['image_inventory'], 'anchors': anchors,
              'environment': t.environment(), 'measurement_versions': features.detector.old.v1.old.versions(),
              'references': {str(s): {'archive': refs[s]['archive'], 'runtime': refs[s]['runtime'],
                    'arrays': {n:t.common.sha256_array(refs[s][n]) for n in ('folds','base','current','tabpfn486')}} for s in SEEDS},
              'best_submission': t.file_ref(ROOT/'reports/final_scattering_blend/submission/submission.csv')}
    return data, x486, refs, frozen


def prepare(folder):
    *_, frozen = load_state()
    path = folder/'protocol.json'
    if path.exists():
        if t.read_json(path)['frozen'] != frozen:
            raise ValueError('Frozen plan changed; preserve previous experiment')
        print('Existing preparation matches; no extraction/fitting'); return
    if folder.exists() and any(folder.iterdir()):
        raise ValueError('Nonempty unprepared output folder')
    t.write_json(path, {'frozen': frozen, 'created_unix': time.time(), 'fits_during_preparation': 0})
    print(f'Prepared486 vs legacy500 vs refined500;20 new outer context fits planned: {path}')


def validate(folder):
    data, x, refs, frozen = load_state()
    plan = t.read_json(folder/'protocol.json')
    if plan['frozen'] != frozen or plan['fits_during_preparation'] != 0:
        raise ValueError('Frozen sources, inputs, detector, runtime or controls changed')
    return data, x, refs, plan


def serialize(row):
    return [None if np.isnan(v) else float(v) for v in row]


def extraction_cell(folder, plan, item, extractor=features.extract):
    source = ROOT/'open/train'/f"{item['ID']}.png"
    if t.common.sha256_file(source) != item['sha256']:
        raise ValueError('Image changed before extraction/cache reuse')
    path = folder/'features/cells'/f"{item['ID']}.json"
    binding = {'plan': t.digest(plan['frozen']), 'image': item}
    if path.exists():
        cell = t.read_json(path)
        if cell['binding'] != binding or cell['result_sha256'] != t.digest(cell['result']):
            raise ValueError('Extraction cell changed')
        result = cell['result']
    else:
        gray = features.detector.old.v1.old.read_image(source)
        rows, maps = extractor(gray)
        for row in rows.values(): features.validate_matrix(row[None,:])
        result = {'features': {rep:serialize(rows[rep]) for rep in REPS},
                  'map_hashes': {k:t.common.sha256_array(maps[k]) for k in features.MAP_KEYS}}
        if t.common.sha256_file(source) != item['sha256']:
            raise ValueError('Image changed during extraction')
    anchor = plan['frozen']['anchors'].get(item['ID'])
    if anchor is not None and anchor != result['map_hashes']:
        raise ValueError('Reviewed detector anchor differs')
    rows = {rep:np.asarray(result['features'][rep],dtype=np.float64) for rep in REPS}
    for row in rows.values(): features.validate_matrix(row[None,:])
    if not path.exists():
        t.write_json(path, {'binding': binding, 'result': result, 'result_sha256': t.digest(result)})
    return rows


def load_features(folder, data, plan):
    target = folder/'features'
    if not (target/'metadata.json').exists():
        raise FileNotFoundError('Feature extraction incomplete. Run extract successfully before evaluate.')
    meta = t.read_json(target/'metadata.json')
    if meta['plan'] != t.digest(plan['frozen']) or meta['npz_sha256'] != t.common.sha256_file(target/'features.npz'):
        raise ValueError('Feature binding/cache changed')
    if set(meta['cells']) != {f'cells/{i}.json' for i in data.ids}:
        raise ValueError('Incomplete cell set')
    with np.load(target/'features.npz', allow_pickle=False) as z:
        if set(z.files) != {'ids','names',*REPS}:
            raise ValueError('Unexpected feature schema')
        np.testing.assert_array_equal(z['ids'], data.ids)
        np.testing.assert_array_equal(z['names'], features.NAMES)
        arrays = {r:z[r].copy() for r in REPS}
    for rep,x in arrays.items():
        features.validate_matrix(x)
        if x.shape != (500,14): raise ValueError('Expected500 images')
    for i,item in enumerate(plan['frozen']['image_inventory']):
        name = f"cells/{item['ID']}.json"
        if t.common.sha256_file(target/name) != meta['cells'][name]:
            raise ValueError('Feature cell digest changed')
        rows = extraction_cell(folder,plan,item,forbidden)
        for rep in REPS: np.testing.assert_array_equal(arrays[rep][i],rows[rep])
    return arrays,meta


def extract(folder):
    data, _, _, plan = validate(folder)
    if (folder/'features/metadata.json').exists():
        load_features(folder,data,plan); print('Completed features verified; no re-extraction'); return
    start=time.perf_counter();rows=[]
    for i,item in enumerate(plan['frozen']['image_inventory']):
        rows.append(extraction_cell(folder,plan,item))
        if i==0 or (i+1)%10==0:
            print(f'Extracted/reused {i+1}/500; legacy14/refined14; elapsed={time.perf_counter()-start:.1f}s',flush=True)
    validate(folder)
    arrays={r:np.stack([v[r] for v in rows]) for r in REPS}
    target=folder/'features';target.mkdir(parents=True,exist_ok=True)
    with (target/'features.npz.tmp').open('wb') as f:
        np.savez_compressed(f,ids=data.ids,names=np.asarray(features.NAMES),**arrays)
    os.replace(target/'features.npz.tmp',target/'features.npz')
    t.write_json(target/'metadata.json',{'plan':t.digest(plan['frozen']),
        'npz_sha256':t.common.sha256_file(target/'features.npz'),
        'cells':{f'cells/{i}.json':t.common.sha256_file(target/'cells'/f'{i}.json') for i in data.ids},
        'seconds':time.perf_counter()-start,'labels_used_for_features':False,'test_accessed':False,
        'reviewed_anchors':12,'training_masks_created':False,'dimensions':{r:[500,14] for r in REPS}})
    load_features(folder,data,plan)
    print(f'Saved {target/"features.npz"}; both model inputs500 columns;12 detector anchors matched')


class CandidatePredictor:
    def __init__(self, base): self.base=base;self.runtime=base.runtime
    def __call__(self,x,y,query,seed):
        a,b,meta=features.impute_context(x,query)
        p,runtime=self.base(a,y,b,seed)
        return p,{**runtime,'outer_context_imputation':meta}


def evaluate_fold(folder,binding,matrices,y,folds,seed,fold,predictor):
    ti,vi=np.flatnonzero(folds!=fold),np.flatnonzero(folds==fold)
    if (len(ti),len(vi))!=(400,100): raise ValueError('Expected outer400/100')
    output={}
    for rep in REPS:
        if matrices[rep].shape!=(500,500):raise ValueError('Expected500 columns within v2 limit')
        output[rep]=t.run_cell(folder/f'fold{fold}_{rep}',binding,rep,matrices[rep],y[ti],ti,vi,
                               seed+1009*fold+9,predictor)
    return vi,output


def result_data(data,ref,pred,common_ids):
    allp={'current486_blend':ref['current'],'tabpfn486':ref['tabpfn486'],
          'tabpfn_legacy500':pred['legacy14'],'tabpfn_refined500':pred['refined14'],
          CONTROL:.5*(ref['base']+pred['legacy14']),PRIMARY:.5*(ref['base']+pred['refined14'])}
    if any(p.shape!=data.y.shape or not np.isfinite(p).all() for p in allp.values()):
        raise ValueError('Invalid predictions')
    rmse=lambda p:t.core.metrics(data.y,p)['rmse']
    metrics={n:{**t.core.metrics(data.y,p),'gain_vs_current':rmse(ref['current'])-rmse(p),
               'gain_vs_control':rmse(allp[CONTROL])-rmse(p)} for n,p in allp.items()}
    target,noise=t.core.rank_groups(data.y,10),t.core.rank_groups(data.noise,4)
    common=np.isin(data.ids,common_ids)
    masks={'low10':target==0,'high10':target==9,'tails20':(target==0)|(target==9),
           'middle80':(target>0)&(target<9),'common16':common,'remaining484':~common}
    masks.update({f'noiseQ{k+1}':noise==k for k in range(4)})
    masks.update({f'fold{k}':ref['folds']==k for k in range(5)})
    worst={n:np.argsort((allp[n]-data.y)**2,kind='stable')[-20:] for n in ('current486_blend',CONTROL,PRIMARY)}
    masks.update({n+'_worst20':np.isin(np.arange(len(data.y)),ix) for n,ix in worst.items()})
    new=np.setdiff1d(worst[PRIMARY],worst['current486_blend'])
    masks['new_worst20']=np.isin(np.arange(len(data.y)),new)
    diag=[{'model':n,'group':g,**t.core.metrics(data.y[m],p[m]),'sse':float(np.sum((p[m]-data.y[m])**2))}
          for n,p in allp.items() for g,m in masks.items() if m.any()]
    rows=[{'ID':str(i),'hardness':float(data.y[j]),'fold':int(ref['folds'][j]),
           **{n:float(p[j]) for n,p in allp.items()}} for j,i in enumerate(data.ids)]
    errors=[{'ID':str(data.ids[j]),'hardness':float(data.y[j]),'fold':int(ref['folds'][j]),
             'common16':bool(common[j]),'new_worst20':bool(j in new),
             'current_error':float(ref['current'][j]-data.y[j]),
             'control_error':float(allp[CONTROL][j]-data.y[j]),'candidate_error':float(allp[PRIMARY][j]-data.y[j]),
             'gain_squared_error_vs_current':float((ref['current'][j]-data.y[j])**2-(allp[PRIMARY][j]-data.y[j])**2),
             'gain_squared_error_vs_control':float((allp[CONTROL][j]-data.y[j])**2-(allp[PRIMARY][j]-data.y[j])**2)}
            for j in np.argsort(-(allp[PRIMARY]-data.y)**2,kind='stable')]
    return metrics,rows,diag,errors


def expected_outputs():
    return {'oof.csv','summary.json','diagnostics.csv','per_image_errors.csv','report.md'} | {
        f'fold{f}_{r}/completed.json' for f in range(5) for r in REPS}


def completed(folder,binding):
    if not (folder/'completed_evaluation.json').exists():return False
    if set(t.read_json(folder/'completed_evaluation.json')['files'])!=expected_outputs():
        raise ValueError('Incomplete result set')
    return t.verify_completed(folder,binding)


def evaluation_binding(folder,plan,ref,seed):
    return {'plan':t.digest(plan['frozen']),'feature_metadata':t.file_ref(folder/'features/metadata.json'),
            'seed':seed,'runtime':ref['runtime']}


def save_results(folder,data,ref,pred,plan,binding,seed):
    metrics,rows,diag,errors=result_data(data,ref,pred,plan['frozen']['common_failure_ids'])
    for name,records in (('oof.csv',rows),('diagnostics.csv',diag),('per_image_errors.csv',errors)):
        t.write_csv(folder/name,records)
    t.write_json(folder/'summary.json',{'seed':seed,'binding':binding,'metrics':metrics,'primary':PRIMARY,
        'new_context_fits':10,'inner_search':False,'test_accessed':False,'independent_holdout':False,
        'attribution':'Built with PriorLabs-TabPFN'})
    lines=[f'# Refined automatic structure + TabPFN · seed{seed}','','Built with PriorLabs-TabPFN.','',
           '| Model | RMSE | Gain vs current486 | Gain vs legacy500 |','|---|---:|---:|---:|']
    lines += [f"| {n} | {m['rmse']:.6f} | {m['gain_vs_current']:+.6f} | {m['gain_vs_control']:+.6f} |" for n,m in metrics.items()]
    lines += ['','Fixed486 +14 proposal summaries. Same outer5,400/100, local v2/8 estimators/FP32.',
              'Only TabPFN50% replaced. Both summary branches use the same columns and context-only imputation.',
              'No pixel supervision, semantic accuracy claim, independent holdout, test prediction or submission.',
              'Inspect diagnostics and per_image_errors: common16, remaining484, tails, middle80, noise and new worst20.',
              'Both seeds must beat current and legacy control; mean gain0.30 is expansion priority, not significance.']
    (folder/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    t.write_json(folder/'completed_evaluation.json',{'binding':binding,
        'files':{n:t.common.sha256_file(folder/n) for n in sorted(expected_outputs())}})
    for n in ('current486_blend',CONTROL,PRIMARY):
        m=metrics[n];print(f"{n}: RMSE={m['rmse']:.6f}; gain vs current={m['gain_vs_current']:+.6f}; gain vs control={m['gain_vs_control']:+.6f}",flush=True)
    print(f'Report: {folder/"report.md"}')


def evaluate(folder,device,seed):
    data,base,refs,plan=validate(folder);added,_=load_features(folder,data,plan)
    target=folder/f'seed{seed}';target.mkdir(parents=True,exist_ok=True)
    binding=evaluation_binding(folder,plan,refs[seed],seed)
    if completed(target,binding):print(f'Completed seed{seed} verified; no refitting')
    else:
        predictor=CandidatePredictor(t.Predictor(t.verify_weights(t.OUT),device))
        if predictor.runtime!=refs[seed]['runtime']:raise ValueError('Archived device/runtime required; no fallback')
        matrices={r:np.c_[base,added[r]] for r in REPS};pred={r:np.full(500,np.nan) for r in REPS}
        for fold in range(5):
            vi,outputs=evaluate_fold(target,binding,matrices,data.y,refs[seed]['folds'],seed,fold,predictor)
            for r in REPS:pred[r][vi]=outputs[r]
            print(f'Outer{fold}: legacy500/refined500 predictions saved; no outer-driven selection',flush=True)
        validate(folder);load_features(folder,data,plan)
        save_results(target,data,refs[seed],pred,plan,binding,seed)
    if all((folder/f'seed{s}/completed_evaluation.json').exists() for s in SEEDS):compare(folder)


def aggregate(records):
    if [r['seed'] for r in records]!=list(SEEDS):raise ValueError('Both fixed seeds required')
    mean=float(np.mean([r['gain_vs_current'] for r in records]))
    positive=all(r['gain_vs_current']>0 and r['gain_vs_control']>0 for r in records)
    return {'primary':PRIMARY,'per_seed':records,'mean_gain_vs_current':mean,
        'mean_gain_vs_control':float(np.mean([r['gain_vs_control'] for r in records])),
        'all_seeds_improve_both':positive,'passes_expansion_priority':positive and mean>=.30,
        'additional_review_required':'Common16/remaining484, tails/middle80, noise, new worst20; no automatic adoption.',
        'aggregation':'Mean of two per-seed OOF RMSE differences; same500; no independent inference or prediction averaging.'}


def compare(folder):
    data,base,refs,plan=validate(folder);added,_=load_features(folder,data,plan)
    matrices={r:np.c_[base,added[r]] for r in REPS};records=[]
    for seed in SEEDS:
        target=folder/f'seed{seed}';binding=evaluation_binding(folder,plan,refs[seed],seed)
        if not completed(target,binding):raise ValueError('Complete both seeds before compare')
        pred={r:np.full(500,np.nan) for r in REPS}
        with contextlib.redirect_stdout(io.StringIO()):
            for fold in range(5):
                vi,outputs=evaluate_fold(target,binding,matrices,data.y,refs[seed]['folds'],seed,fold,forbidden)
                for r in REPS:pred[r][vi]=outputs[r]
        metrics,rows,_,_=result_data(data,refs[seed],pred,plan['frozen']['common_failure_ids'])
        if t.read_json(target/'summary.json')['metrics']!=metrics:raise ValueError('Metric recomputation differs')
        saved=t.core.align_rows(t.core.read_rows(target/'oof.csv',set(rows[0])),data.ids,target)
        for k in rows[0]:
            if k!='ID':np.testing.assert_array_equal([float(r[k]) for r in saved],[r[k] for r in rows])
        m=metrics[PRIMARY]
        records.append({'seed':seed,'current_rmse':metrics['current486_blend']['rmse'],
            'legacy_control_rmse':metrics[CONTROL]['rmse'],'candidate_rmse':m['rmse'],
            'gain_vs_current':m['gain_vs_current'],'gain_vs_control':m['gain_vs_control']})
    result={**aggregate(records),'protocol_sha256':t.digest(plan['frozen'])}
    t.write_json(folder/'comparison.json',result);print(t.common.canonical_json(result),flush=True)


def check(folder,device):
    # Synthetic only; validate does not execute this method during preparation.
    predictor=CandidatePredictor(t.Predictor(t.verify_weights(t.OUT),device))
    rng=np.random.default_rng(314159);x=rng.normal(size=(500,486))
    extra=np.zeros((500,14),dtype=np.float64)
    extra[:,features.NULLABLE]=np.nan;extra[:,features.FLAGS]=1
    p,meta=predictor(np.c_[x[:400],extra[:400]],100+3*x[:400,2],np.c_[x[400:],extra[400:]],314159)
    t.write_json(folder/'synthetic_tabpfn_check.json',{'competition_data_used':False,'shape':list(p.shape),
        'finite':bool(np.isfinite(p).all()),'metadata':meta})
    print(f'Synthetic500-column context400/query100 passed on {predictor.runtime["device"]}')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('prepare','validate','extract','check','evaluate','compare'))
    p.add_argument('--seed',type=int,choices=SEEDS,default=42);p.add_argument('--device',default='cuda')
    a=p.parse_args()
    if a.command=='prepare':prepare(OUT)
    elif a.command=='validate':validate(OUT);print('Frozen controls/runtime/12 anchors/protected files verified; no fitting')
    elif a.command=='extract':
        with threadpool_limits(limits=2):extract(OUT)
    elif a.command=='check':check(OUT,a.device)
    elif a.command=='evaluate':evaluate(OUT,a.device,a.seed)
    else:compare(OUT)


if __name__=='__main__':main()
