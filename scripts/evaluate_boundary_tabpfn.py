"""User-run fixed boundary/quality features + local TabPFN-v2 ablation.

Built with PriorLabs-TabPFN. prepare/validate perform no extraction or fitting.
extract: train500 CPU only. evaluate: one fixed candidate, five outer fits/seed.
No feature/epoch/weight selection, test inference, submission, or old-file mutation.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import os
from pathlib import Path
import time
import numpy as np
if __package__:
    from . import evaluate_tabpfn_v2 as t
    from . import audit_boundary_quality_v3 as quality
else:
    import evaluate_tabpfn_v2 as t
    import audit_boundary_quality_v3 as quality

ROOT = t.ROOT
OUT = ROOT/'reports/boundary_tabpfn'
SEEDS = (42, 2026)
PRIMARY = 'boundary_replaced_blend'
REP = 'morphology285_boundary37'
STATES = ('repeatable_proxy','borderline_proxy','unstable_proxy','unavailable','null_like')
NULLABLE = ('observed_mean_spacing','observed_median_spacing','observed_h_spacing','observed_v_spacing',
            'repeatable_spacing','spacing_sensitivity_relative_width','spacing_worst_relative_change')
VALUES = NULLABLE + ('boundary_fraction','valley_fraction','step_fraction','log_intervals','log_h_intervals',
    'log_v_intervals','transect_coverage','h_coverage','v_coverage','occupied_tile_fraction','scale_f1',
    'support_min','noise_proxy','log_structure_noise_ratio','log_spatial_ratio','observation_available',
    'all_probes_observable','observable_probe_fraction') + tuple('status_'+s for s in STATES)
NAMES = VALUES + tuple(n+'_missing' for n in NULLABLE)
assert len(VALUES)==30 and len(NAMES)==37
PROTOCOL = {
    'version':1,'attribution':'Built with PriorLabs-TabPFN', 'seeds':list(SEEDS),
    'dimensions':{'control':285,'new':37,'candidate':322},'feature_names':list(NAMES),
    'feature_definition':'Fixed v2 detector + v3.assess with its six internal probes; no external probes or changed thresholds. '
        'Available mixed-boundary mean/median/H/V spacings, repeatable spacing, sensitivity relative width and worst change; '
        'boundary/valley/step fractions, log1p interval counts, H/V/combined coverage, occupied_tiles/16, scale F1, '
        'support_min, noise, log1p structure/noise and spatial ratios, observation flags, observable probe fraction, five status flags. '
        'Seven nullable columns each get a fixed missing indicator. Not grain size, pore fraction or semantic labels.',
    'missing':'Raw cache uses NaN (JSON null), never interpreted as physical zero. Only seven nullable added columns are '
        'median-imputed using the outer training400. All-missing context column uses neutral0 with its missing flag retained. '
        'Query values never set medians. Original285 are unchanged. No sample removal.',
    'predictor':t.PROTOCOL,
    'evaluation_override':'Fixed285 and fixed322; no185/285 inner choice for new candidate, no inner search. '
        'Original current OOF retains its historical nested185/285 choice. Outer5 partitions, model seed=seed+1009*fold+9; '
        'settings/weights/runtime exactly match archived285 controls. Five NEW context fits per seed, ten total.',
    'control_reuse':'Verify archived285 outer cells, all historical inner selections, OOF IDs/labels/folds/50:50 arithmetic and runtime; no control refits.',
    'blend':'B=hand185 Ridge25%+single-view ResNet25%+DINO D4 50%. Current=0.5*B+0.5*archived nested TabPFN. '
        'Fixed285 control=0.5*B+0.5*archived285. Candidate=0.5*B+0.5*new322. No extra mixing or clipping.',
    'primary':PRIMARY,'separate_control':'fixed285_control_blend',
    'decision':'Both seed gains must be positive against current AND fixed285 control. Priority additionally mean gain vs current>=0.10. '
        'Review tails, noise, quality groups and common failures before any submission preparation. No automatic adoption.',
    'priority_mean_gain':.10,
    'scope':'train-only feature extraction and OOF evaluation; no neural weight updates, test use, final500 fit or submission.',
    'interpretation':'Same repeatedly studied500; not independent holdout or significance. Features frozen before new OOF. '
        'Thresholds, feature count, seeds and blend weights will not be tuned on these two outputs.',
    'authorization':'User explicitly requested this controlled predictive experiment after v3 diagnostic; this does not change '
        'the old no-automatic-adoption finding or validate physical segmentation.',
}


def vector(record):
    m=record['v2_metrics']; available=record['observed_spacing'] is not None
    width=None if m['raw_spacing'] is None else (record['sensitivity_max']-record['sensitivity_min'])/m['raw_spacing']
    vals=[record['observed_spacing'],m['median_spacing'] if available else None,
          m['h_spacing'] if available else None,m['v_spacing'] if available else None,
          record['recommended_proxy_spacing'],width,record['spacing_change_max'],
          m['boundary_area'],m['valley_area'],m['step_area'],
          *[np.log1p(m[n]) for n in ('intervals','h_intervals','v_intervals')],m['usable_transect_fraction'],
          m['h_usable_transects']/256,m['v_usable_transects']/256,m['occupied_tiles']/16,
          m['scale_f1'],record['support_min'],m['noise_proxy'],np.log1p(m['structure_noise_ratio']),
          np.log1p(record['spatial_evidence']['ratio']),float(record['observation']['available']),
          float(record['all_probes_observable']),np.mean([p['observation']['available'] for p in record['probes']]),
          *[float(record['status']==s) for s in STATES]]
    result=np.array([np.nan if x is None else x for x in vals],dtype=np.float64)
    result=np.r_[result,np.isnan(result[:7]).astype(float)]
    validate_matrix(result[None,:])
    return result


def validate_matrix(x):
    if x.ndim!=2 or x.shape[1]!=37 or x.dtype!=np.float64:raise ValueError('Expected float64 boundary37')
    if np.isinf(x).any() or not np.isfinite(x[:,7:]).all():raise ValueError('Invalid nonnullable features')
    if not np.array_equal(x[:,30:],np.isnan(x[:,:7]).astype(float)):raise ValueError('Missing flags differ')
    if not np.isin(x[:,25:30],(0.,1.)).all() or not np.all(x[:,25:30].sum(axis=1)==1):raise ValueError('Invalid quality status')


def impute_context(train_x,query_x):
    if train_x.shape[1]!=322 or query_x.shape[1]!=322:raise ValueError('Expected322 columns')
    a,b=train_x.copy(),query_x.copy();validate_matrix(a[:,285:]);validate_matrix(b[:,285:])
    if not np.isfinite(a[:,:285]).all() or not np.isfinite(b[:,:285]).all():raise ValueError('Original285 must be finite')
    medians=[];empty=[]
    for j in range(285,292):
        observed=a[np.isfinite(a[:,j]),j]
        median=float(np.median(observed)) if len(observed) else 0.
        if not len(observed):empty.append(j)
        a[np.isnan(a[:,j]),j]=median;b[np.isnan(b[:,j]),j]=median;medians.append(median)
    if not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('Nonfinite imputed model input')
    return a,b,{'columns':list(range(285,292)),'context_medians':medians,'all_missing_context_columns':empty,
                'query_used_for_fit':False,'missing_indicators_retained':True}


class CandidatePredictor:
    def __init__(self,base):self.base=base;self.runtime=base.runtime
    def __call__(self,x,y,query,seed):
        a,b,imputation=impute_context(x,query)
        pred,metadata=self.base(a,y,b,seed)
        return pred,{**metadata,'outer_context_imputation':imputation}


def verify_manifest(folder):
    m=t.read_json(folder/'completion_manifest.json')
    for name,sha in m['artifacts'].items():
        if t.common.sha256_file(folder/name)!=sha:raise ValueError('Audit artifact changed: '+name)
    for name,sha in m.get('scripts',{}).items():
        if t.common.sha256_file(ROOT/name)!=sha:raise ValueError('Audit source changed: '+name)


def archived_reference(data,features,old,plan,seed):
    folder=t.OUT/f'seed{seed}';summary=t.read_json(folder/'summary.json');binding=summary['binding']
    if binding['plan']!=t.digest(plan['frozen']) or binding['seed']!=seed or not t.verify_completed(folder,binding):
        raise ValueError('Historical TabPFN plan/results differ')
    expected={f'fold{f}_{r}_{stage}/completed.json' for f in range(5) for r in t.REPS for stage in ('inner0','inner1','inner2','outer')}
    expected.update(('oof.csv','summary.json','diagnostics.csv','per_image_errors.csv','report.md'))
    if set(t.read_json(folder/'completed_evaluation.json')['files'])!=expected:raise ValueError('Incomplete archived cells')
    rows=t.core.align_rows(t.core.read_rows(folder/'oof.csv',{'ID','hardness','fold',t.PRIMARY,t.SECONDARY}),data.ids,folder)
    col=lambda n:t.core.finite_column(rows,n,folder)
    np.testing.assert_array_equal(col('hardness'),data.y)
    np.testing.assert_array_equal(t.core.integer_column(rows,'fold',folder),old['folds'])
    np.testing.assert_array_equal(col('current'),old['current'])
    np.testing.assert_array_equal(col(t.SECONDARY),.5*(old['current']+col(t.PRIMARY)))
    # Reuse the original fold engine strictly through its verified cache; any missing cell fails before a fit.
    def forbidden(*args):raise ValueError('Archived control cell missing; no implicit refitting')
    with contextlib.redirect_stdout(io.StringIO()):
        for f in range(5):
            vi,outputs,choice=t.evaluate_fold(features,data.y,old['folds'],seed,f,folder,binding,forbidden)
            if choice!=summary['selection'][f]:raise ValueError('Archived inner selection differs')
            for n,p in outputs.items():
                np.testing.assert_array_equal(p,col(n if n==t.PRIMARY else 'tabpfn_'+n)[vi])
    return {'folds':old['folds'],'base':old['current'],'current':col(t.SECONDARY),
            'tabpfn285':col('tabpfn_morphology285'),'control':.5*(old['current']+col('tabpfn_morphology285')),
            'runtime':binding['runtime'],'archive':t.file_ref(folder/'completed_evaluation.json')}


def load_state():
    data,features,old,source_plan=t.validate(t.OUT)
    qplan=quality.validate()
    for folder in (quality.detector.v1.OUT,quality.detector.OUT,quality.OUT):verify_manifest(folder)
    refs={seed:archived_reference(data,features,old[seed],source_plan,seed) for seed in SEEDS}
    images=[{'ID':str(i),'sha256':t.common.sha256_file(ROOT/'open/train'/f'{i}.png')} for i in data.ids]
    if len(images)!=500 or len(set(data.ids))!=500:raise ValueError('Expected unique train500')
    worst=[set(data.ids[np.argsort((refs[s]['current']-data.y)**2,kind='stable')[-20:]]) for s in SEEDS]
    anchors={}
    for subset in ('historical','fresh'):
        for r in t.read_json(quality.OUT/subset/'quality_details.json'):
            anchors[r['ID']]=t.digest({k:v for k,v in r.items() if k not in ('ID','noise_quintile','subset')})
    frozen={'protocol':PROTOCOL,'source':t.file_ref(__file__),'source_plan':t.file_ref(t.OUT/'protocol.json'),
            'source_plan_payload':t.digest(source_plan['frozen']),'quality_plan':t.file_ref(quality.OUT/'protocol.json'),
            'quality_manifest':t.file_ref(quality.OUT/'completion_manifest.json'),
            'image_inventory':images,'quality_anchor_hashes':anchors,
            'target_sha256':t.common.sha256_array(data.y),'original285_sha256':t.common.sha256_array(features['morphology285']),
            'common_failure_ids':sorted(worst[0]&worst[1]),
            'reference_files':{str(s):refs[s]['archive'] for s in SEEDS},
            'reference_arrays':{str(s):{n:t.common.sha256_array(refs[s][n]) for n in ('folds','base','current','tabpfn285','control')} for s in SEEDS}}
    return data,features['morphology285'],refs,frozen


def prepare(folder):
    *_,frozen=load_state();path=folder/'protocol.json'
    if path.exists():
        if t.read_json(path)['frozen']!=frozen:raise ValueError('Frozen preparation changed; preserve prior outputs')
        print('Existing preparation matches');return
    if folder.exists() and any(folder.iterdir()):raise ValueError('Nonempty unprepared folder')
    t.write_json(path,{'frozen':frozen,'created_unix':time.time(),'fits_during_preparation':0,'images_extracted_during_preparation':0})
    print(f'Prepared fixed285 vs322, seeds42/2026; no extraction or fitting: {path}')


def validate(folder):
    data,x,refs,frozen=load_state();plan=t.read_json(folder/'protocol.json')
    if plan['frozen']!=frozen or plan['fits_during_preparation']!=0:raise ValueError('Frozen sources/inputs/runtime/settings differ')
    return data,x,refs,plan


def extraction_cell(folder,plan,image,*,assessor=None,reader=None):
    assessor=assessor or quality.assess;reader=reader or quality.detector.v1.old.read_image
    path=folder/'features/cells'/f"{image['ID']}.json"
    binding={'plan':t.digest(plan['frozen']),'image':image}
    if path.exists():
        saved=t.read_json(path)
        if saved['binding']!=binding or saved['result_sha256']!=t.digest(saved['result']):raise ValueError('Changed extraction cell')
        result=saved['result'];row=vector(result['quality'])
        if result['features']!=[None if np.isnan(v) else float(v) for v in row]:raise ValueError('Cached features differ')
    else:
        source=ROOT/'open/train'/f"{image['ID']}.png"
        if t.common.sha256_file(source)!=image['sha256']:raise ValueError('Image changed before extraction')
        record,_=assessor(reader(source));row=vector(record)
        result={'quality':record,'features':[None if np.isnan(v) else float(v) for v in row]}
        anchor=plan['frozen']['quality_anchor_hashes'].get(image['ID'])
        if anchor is not None and t.digest(record)!=anchor:raise ValueError('Quality audit anchor changed')
        if t.common.sha256_file(source)!=image['sha256']:raise ValueError('Image changed during extraction')
        t.write_json(path,{'binding':binding,'result':result,'result_sha256':t.digest(result)})
    return row,result['quality']['status']


def load_features(folder,data,plan):
    target=folder/'features';meta=t.read_json(target/'metadata.json')
    if meta['plan']!=t.digest(plan['frozen']) or meta['npz_sha256']!=t.common.sha256_file(target/'features.npz'):raise ValueError('Feature binding/cache changed')
    expected={f"cells/{i}.json" for i in data.ids}
    if set(meta['cells'])!=expected:raise ValueError('Incomplete extraction cells')
    for name,sha in meta['cells'].items():
        if t.common.sha256_file(target/name)!=sha:raise ValueError('Extraction cell changed')
    with np.load(target/'features.npz',allow_pickle=False) as z:
        if set(z.files)!={'ids','boundary','names'}:raise ValueError('Unexpected cache arrays')
        np.testing.assert_array_equal(z['ids'],data.ids)
        if z['names'].tolist()!=list(NAMES):raise ValueError('Changed feature names')
        x=z['boundary'].copy()
    validate_matrix(x)
    if x.shape!=(500,37) or meta['array_sha256']!=t.common.sha256_array(x):raise ValueError('Invalid feature matrix')
    return x,meta


def extract(folder):
    data,_,_,plan=validate(folder)
    if (folder/'features/metadata.json').exists():
        load_features(folder,data,plan);print('Completed features verified; no re-extraction');return
    started=time.perf_counter();rows=[];states=[]
    for j,image in enumerate(plan['frozen']['image_inventory']):
        row,state=extraction_cell(folder,plan,image);rows.append(row);states.append(state)
        if (j+1)%25==0:print(f'Extracted/reused {j+1}/500; boundary37; {time.perf_counter()-started:.1f}s',flush=True)
    validate(folder)
    x=np.stack(rows);validate_matrix(x);target=folder/'features';target.mkdir(exist_ok=True)
    with (target/'features.npz.tmp').open('wb') as h:np.savez_compressed(h,ids=data.ids,boundary=x,names=np.array(NAMES))
    os.replace(target/'features.npz.tmp',target/'features.npz')
    t.write_json(target/'metadata.json',{'plan':t.digest(plan['frozen']),'npz_sha256':t.common.sha256_file(target/'features.npz'),
        'array_sha256':t.common.sha256_array(x),'cells':{f'cells/{i}.json':t.common.sha256_file(target/'cells'/f'{i}.json') for i in data.ids},
        'new_dimensions':37,'combined_dimensions':322,'status_counts':{s:states.count(s) for s in STATES},
        'nullable_counts':{n:int(np.isnan(x[:,j]).sum()) for j,n in enumerate(NULLABLE)},
        'labels_used_for_features':False,'test_accessed':False,'seconds':time.perf_counter()-started})
    load_features(folder,data,plan);print(f'Saved {target}/features.npz; boundary=(500,37), candidate=(500,322); anchors27 matched')


def evaluate_fold(folder,binding,x,y,folds,seed,fold,predictor):
    ti,vi=np.flatnonzero(folds!=fold),np.flatnonzero(folds==fold)
    if (len(ti),len(vi))!=(400,100):raise ValueError('Expected400/100 outer split')
    p=t.run_cell(folder/f'fold{fold}_candidate',binding,REP,x,y[ti],ti,vi,seed+1009*fold+9,predictor)
    return vi,p


def result_data(data,ref,boundary,pred,common_ids):
    allp={'current':ref['current'],'tabpfn285_control':ref['tabpfn285'],'tabpfn322':pred,
          'fixed285_control_blend':ref['control'],PRIMARY:.5*(ref['base']+pred)}
    if any(v.shape!=data.y.shape or not np.isfinite(v).all() for v in allp.values()):raise ValueError('Invalid OOF')
    metrics={n:{**t.core.metrics(data.y,p),
                'gain_vs_current':t.core.metrics(data.y,ref['current'])['rmse']-t.core.metrics(data.y,p)['rmse'],
                'gain_vs_control':t.core.metrics(data.y,ref['control'])['rmse']-t.core.metrics(data.y,p)['rmse']} for n,p in allp.items()}
    target=t.core.rank_groups(data.y,10);noise=t.core.rank_groups(data.noise,4)
    worst=np.argsort((ref['current']-data.y)**2,kind='stable')[-20:]
    masks={'low10':target==0,'high10':target==9,'tails20':(target==0)|(target==9),
           'middle80':(target>0)&(target<9),'current_worst20':np.isin(np.arange(len(data.y)),worst),
           'historical_common_worst20':np.isin(data.ids,common_ids)}
    masks.update({f'fold{f}':ref['folds']==f for f in range(5)})
    masks.update({f'noiseQ{k+1}':noise==k for k in range(4)})
    masks.update({f'quality_{s}':boundary[:,25+j]==1 for j,s in enumerate(STATES)})
    diag=[{'model':n,'group':g,**t.core.metrics(data.y[mask],p[mask])} for n,p in allp.items() for g,mask in masks.items() if mask.any()]
    rows=[{'ID':str(i),'hardness':float(data.y[j]),'fold':int(ref['folds'][j]),**{n:float(p[j]) for n,p in allp.items()}} for j,i in enumerate(data.ids)]
    errors=[{'ID':str(data.ids[j]),'hardness':float(data.y[j]),'fold':int(ref['folds'][j]),'quality_status':STATES[int(np.argmax(boundary[j,25:30]))],
        'current_error':float(ref['current'][j]-data.y[j]),'control_error':float(ref['control'][j]-data.y[j]),
        'candidate_error':float(allp[PRIMARY][j]-data.y[j]),
        'gain_squared_error_vs_current':float((ref['current'][j]-data.y[j])**2-(allp[PRIMARY][j]-data.y[j])**2),
        'gain_squared_error_vs_control':float((ref['control'][j]-data.y[j])**2-(allp[PRIMARY][j]-data.y[j])**2)} for j in np.argsort(-(ref['current']-data.y)**2,kind='stable')]
    return metrics,rows,diag,errors


def expected_outputs():
    return {'oof.csv','summary.json','diagnostics.csv','per_image_errors.csv','report.md'}|{f'fold{f}_candidate/completed.json' for f in range(5)}


def completed(folder,binding):
    if not (folder/'completed_evaluation.json').exists():return False
    done=t.read_json(folder/'completed_evaluation.json')
    if set(done['files'])!=expected_outputs():raise ValueError('Incomplete evaluation output set')
    return t.verify_completed(folder,binding)


def save_results(target,data,ref,boundary,pred,plan,binding,seed):
    metrics,rows,diag,errors=result_data(data,ref,boundary,pred,plan['frozen']['common_failure_ids'])
    t.write_csv(target/'oof.csv',rows);t.write_csv(target/'diagnostics.csv',diag);t.write_csv(target/'per_image_errors.csv',errors)
    summary={'seed':seed,'binding':binding,'metrics':metrics,'primary':PRIMARY,'new_context_fits':5,
             'control_fits_reused':5,'inner_search':False,'independent_holdout':False,'test_accessed':False,
             'physical_semantics':'unverified mixed-boundary proxies','attribution':'Built with PriorLabs-TabPFN'}
    t.write_json(target/'summary.json',summary)
    lines=[f'# Boundary/quality TabPFN-v2 ablation · seed{seed}','','Built with PriorLabs-TabPFN.','',
           '| Model | OOF RMSE | Gain vs current | Gain vs fixed285 blend |','|---|---:|---:|---:|']
    lines += [f"| {n} | {m['rmse']:.6f} | {m['gain_vs_current']:+.6f} | {m['gain_vs_control']:+.6f} |" for n,m in metrics.items()]
    lines += ['',f'Primary: {PRIMARY}. Original base50% + fixed322 TabPFN50%.',
              'Current retains historical nested185/285 selection. Fixed285 blend isolates the new feature effect.',
              'Five new outer400/100 context fits; unchanged local model/seed/FP32 runtime; no new hyperparameter or inner selection.',
              'Missing feature medians use only each context400; missing flags retained. No sample exclusion.',
              'Review per_image_errors.csv and diagnostics.csv (tails/noise/quality/common failures) after both seeds.',
              'Same500, not an independent holdout or a submission score; no physical grain/pore labels.']
    (target/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    t.write_json(target/'completed_evaluation.json',{'binding':binding,'files':{n:t.common.sha256_file(target/n) for n in sorted(expected_outputs())}})
    for name in ('fixed285_control_blend',PRIMARY):
        m=metrics[name];print(f"{name}: RMSE={m['rmse']:.6f}; gain vs current={m['gain_vs_current']:+.6f}; gain vs control={m['gain_vs_control']:+.6f}",flush=True)
    print(f"Report: {target/'report.md'}")


def evaluate(folder,device,seed):
    data,base,refs,plan=validate(folder);boundary,meta=load_features(folder,data,plan)
    predictor=CandidatePredictor(t.Predictor(t.verify_weights(t.OUT),device))
    if predictor.runtime!=refs[seed]['runtime']:raise ValueError('Use same device/runtime as archived285 control: '+str(refs[seed]['runtime']))
    binding={'plan':t.digest(plan['frozen']),'feature_metadata':t.file_ref(folder/'features/metadata.json'),'seed':seed,'runtime':predictor.runtime}
    target=folder/f'seed{seed}';target.mkdir(parents=True,exist_ok=True)
    if completed(target,binding):print(f'Completed seed{seed} verified; no refitting')
    else:
        x=np.concatenate([base,boundary],axis=1);pred=np.full(len(data.ids),np.nan)
        for fold in range(5):
            vi,p=evaluate_fold(target,binding,x,data.y,refs[seed]['folds'],seed,fold,predictor);pred[vi]=p
        validate(folder);load_features(folder,data,plan)
        save_results(target,data,refs[seed],boundary,pred,plan,binding,seed)
    if all((folder/f'seed{s}/completed_evaluation.json').exists() for s in SEEDS):compare(folder)


def aggregate(records):
    if [r['seed'] for r in records]!=list(SEEDS):raise ValueError('Both fixed seeds required in order')
    gain=float(np.mean([r['gain_vs_current'] for r in records]));control=float(np.mean([r['gain_vs_control'] for r in records]))
    positive=all(r['gain_vs_current']>0 and r['gain_vs_control']>0 for r in records)
    return {'per_seed':records,'mean_gain_vs_current':gain,'mean_gain_vs_control':control,'all_seeds_improve_both':positive,
            'meets_priority_rule':positive and gain>=PROTOCOL['priority_mean_gain'],
            'additional_review_required':'tails/noise/quality/common failures; no automatic adoption',
            'aggregation':'mean per-seed OOF RMSE differences; no averaging predictions or independent-sample inference'}


def compare(folder):
    data,base,refs,plan=validate(folder);boundary,_=load_features(folder,data,plan);records=[]
    x=np.concatenate([base,boundary],axis=1)
    def forbidden(*args):raise ValueError('Evaluation cell missing')
    for seed in SEEDS:
        target=folder/f'seed{seed}';summary=t.read_json(target/'summary.json');binding=summary['binding']
        expected={'plan':t.digest(plan['frozen']),'feature_metadata':t.file_ref(folder/'features/metadata.json'),'seed':seed,'runtime':refs[seed]['runtime']}
        if binding!=expected or not completed(target,binding):raise ValueError('Completed binding differs')
        pred=np.full(len(data.ids),np.nan)
        with contextlib.redirect_stdout(io.StringIO()):
            for fold in range(5):
                vi,p=evaluate_fold(target,binding,x,data.y,refs[seed]['folds'],seed,fold,forbidden);pred[vi]=p
        metrics,rows,diag,errors=result_data(data,refs[seed],boundary,pred,plan['frozen']['common_failure_ids'])
        if metrics!=summary['metrics']:raise ValueError('Metric recomputation differs')
        saved=t.core.align_rows(t.core.read_rows(target/'oof.csv',{'ID','hardness','fold'}),data.ids,target)
        for key in rows[0]:
            if key!='ID':np.testing.assert_array_equal([float(r[key]) for r in saved],[float(r[key]) for r in rows])
        m=metrics[PRIMARY];records.append({'seed':seed,'current_rmse':metrics['current']['rmse'],
            'fixed285_control_rmse':metrics['fixed285_control_blend']['rmse'],'candidate_rmse':m['rmse'],
            'gain_vs_current':m['gain_vs_current'],'gain_vs_control':m['gain_vs_control']})
    result=aggregate(records);t.write_json(folder/'comparison.json',result);print(t.common.canonical_json(result),flush=True)


def check(folder,device):
    # Only synthetic arrays. Users may run this before feature extraction.
    base=t.Predictor(t.verify_weights(t.OUT),device);predictor=CandidatePredictor(base)
    rng=np.random.default_rng(314159);x=rng.normal(size=(500,322));x[:,285:]=0.;x[:,285:292]=rng.normal(size=(500,7))
    x[:,285]=np.nan;x[::5,286]=np.nan;x[400:,285]=.5
    x[:,315:322]=np.isnan(x[:,285:292]);x[:,310]=1.
    y=100+3*x[:,2]-2*x[:,3]+rng.normal(size=500)
    pred,metadata=predictor(x[:400],y[:400],x[400:],314159)
    t.write_json(folder/f'synthetic_check_{base.device.type}.json',{'competition_data_used':False,'features':322,
        'finite':bool(np.isfinite(pred).all()),'prediction_shape':list(pred.shape),'metadata':metadata})
    print(f'Synthetic400/100 with322 columns and missing features passed on {base.device}; {metadata["seconds"]:.1f}s')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('prepare','validate','check','extract','evaluate','compare'))
    p.add_argument('--output-dir',type=t.common.project_path,default=OUT)
    p.add_argument('--seed',type=int,choices=SEEDS,default=42);p.add_argument('--device',default='cuda')
    a=p.parse_args(argv)
    if a.command=='prepare':prepare(a.output_dir)
    elif a.command=='validate':validate(a.output_dir);print('Frozen preparation matches; no extraction/fitting')
    elif a.command=='check':check(a.output_dir,a.device)
    elif a.command=='extract':extract(a.output_dir)
    elif a.command=='evaluate':evaluate(a.output_dir,a.device,a.seed)
    else:compare(a.output_dir)


if __name__=='__main__':main()
