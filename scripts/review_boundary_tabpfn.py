"""Audit completed boundary TabPFN experiment; cached predictions only, no fits."""
import contextlib
import csv
import io
from pathlib import Path
import numpy as np
if __package__:
    from . import evaluate_boundary_tabpfn as b
else:
    import evaluate_boundary_tabpfn as b


def same_rows(path,expected,keys=()):
    with path.open(encoding='utf-8',newline='') as f:
        reader=csv.DictReader(f)
        if not set(keys).issubset(reader.fieldnames):raise ValueError('Missing columns: '+str(path))
        rows=list(reader)
    if len(rows)!=len(expected):raise ValueError('Row count differs: '+str(path))
    for actual,want in zip(rows,expected,strict=True):
        if set(actual)!=set(want):raise ValueError('Columns differ: '+str(path))
        for key,value in want.items():
            if isinstance(value,str):
                if actual[key]!=value:raise ValueError('String differs: '+str(path)+' '+key)
            else:np.testing.assert_allclose(float(actual[key]),value,rtol=0,atol=1e-9)


def main():
    data,base,refs,plan=b.validate(b.OUT)
    boundary,meta=b.load_features(b.OUT,data,plan)
    x=np.c_[base,boundary];states=[];protected={}
    protect=lambda p:protected.update({str(p):b.t.common.sha256_file(p)})
    for p in (b.OUT/'protocol.json',b.OUT/'comparison.json',b.OUT/'features/metadata.json',b.OUT/'features/features.npz'):
        protect(p)
    for image,expected in zip(plan['frozen']['image_inventory'],boundary,strict=True):
        path=b.OUT/'features/cells'/f"{image['ID']}.json";protect(path)
        saved=b.t.read_json(path)
        if saved['binding']!={'plan':b.t.digest(plan['frozen']),'image':image}:raise ValueError('Feature cell binding differs')
        if saved['result_sha256']!=b.t.digest(saved['result']):raise ValueError('Feature result differs')
        row=b.vector(saved['result']['quality']);np.testing.assert_array_equal(row,expected)
        if saved['result']['features']!=[None if np.isnan(v) else float(v) for v in row]:raise ValueError('Feature values differ')
        anchor=plan['frozen']['quality_anchor_hashes'].get(image['ID'])
        if anchor and b.t.digest(saved['result']['quality'])!=anchor:raise ValueError('Stored diagnostic anchor differs')
        if saved['result']['quality']['physical_grain_size'] is not None:raise ValueError('Unexpected physical semantics')
        states.append(saved['result']['quality']['status'])
    if meta['status_counts']!={s:states.count(s) for s in b.STATES}:raise ValueError('Status counts differ')
    if meta['nullable_counts']!={n:int(np.isnan(boundary[:,j]).sum()) for j,n in enumerate(b.NULLABLE)}:raise ValueError('Missing counts differ')
    def forbidden(*args):raise ValueError('Missing completed cell; review never refits')
    records=[];details=[];timing=[]
    for seed in b.SEEDS:
        folder=b.OUT/f'seed{seed}';s=b.t.read_json(folder/'summary.json');binding=s['binding']
        expected={'plan':b.t.digest(plan['frozen']),'feature_metadata':b.t.file_ref(b.OUT/'features/metadata.json'),
                  'seed':seed,'runtime':refs[seed]['runtime']}
        if binding!=expected or not b.completed(folder,binding):raise ValueError('Invalid completed experiment')
        protect(folder/'completed_evaluation.json')
        for name in b.expected_outputs():protect(folder/name)
        prediction=np.full(500,np.nan)
        with contextlib.redirect_stdout(io.StringIO()):
            for fold in range(5):
                vi,p=b.evaluate_fold(folder,binding,x,data.y,refs[seed]['folds'],seed,fold,forbidden);prediction[vi]=p
                ti=np.flatnonzero(refs[seed]['folds']!=fold)
                cell=b.t.read_json(folder/f'fold{fold}_candidate/completed.json')['result']['metadata']
                _,_,imputation=b.impute_context(x[ti],x[vi])
                if cell['outer_context_imputation']!=imputation:raise ValueError('Context-only medians differ')
                if cell['runtime']!=refs[seed]['runtime'] or (cell['context_rows'],cell['query_rows'],cell['columns'])!=(400,100,322):
                    raise ValueError('Prediction runtime or dimensions differ')
                np.testing.assert_allclose(cell['context_mean'],data.y[ti].mean(),rtol=0,atol=1e-9)
                np.testing.assert_allclose(cell['context_std'],data.y[ti].std(),rtol=0,atol=1e-9)
                if cell['resolved_config']['MAX_NUMBER_OF_FEATURES']!=500 or cell['resolved_config']['MIN_UNIQUE_FOR_NUMERICAL_FEATURES']!=0:
                    raise ValueError('Model configuration differs')
                timing.append({'seed':seed,'fold':fold,'seconds':cell['seconds'],
                               'peak_gpu_allocated_bytes':cell['peak_gpu_allocated_bytes'],
                               'all_missing_context_columns':imputation['all_missing_context_columns']})
        metrics,rows,diag,errors=b.result_data(data,refs[seed],boundary,prediction,plan['frozen']['common_failure_ids'])
        if metrics!=s['metrics']:raise ValueError('Summary metrics differ')
        same_rows(folder/'oof.csv',rows,('ID','hardness','fold'))
        same_rows(folder/'diagnostics.csv',diag,('model','group','rmse'))
        same_rows(folder/'per_image_errors.csv',errors,('ID','candidate_error'))
        m=metrics[b.PRIMARY];records.append({'seed':seed,'current_rmse':metrics['current']['rmse'],
            'fixed285_control_rmse':metrics['fixed285_control_blend']['rmse'],'candidate_rmse':m['rmse'],
            'gain_vs_current':m['gain_vs_current'],'gain_vs_control':m['gain_vs_control']})
        groups={r['group'] for r in diag};lookup={(r['model'],r['group']):r for r in diag}
        gd=[{'group':g,'n':lookup[(b.PRIMARY,g)]['n'],
             'current_rmse':lookup[('current',g)]['rmse'],'control_rmse':lookup[('fixed285_control_blend',g)]['rmse'],
             'candidate_rmse':lookup[(b.PRIMARY,g)]['rmse'],
             'gain_vs_control':lookup[('fixed285_control_blend',g)]['rmse']-lookup[(b.PRIMARY,g)]['rmse']} for g in sorted(groups)]
        candidate=.5*(refs[seed]['base']+prediction)
        details.append({'seed':seed,'standalone285_rmse':metrics['tabpfn285_control']['rmse'],
            'standalone322_rmse':metrics['tabpfn322']['rmse'],
            'prediction_correlation_with_control':float(np.corrcoef(candidate,refs[seed]['control'])[0,1]),
            'prediction_rms_change_vs_control':float(np.sqrt(np.mean((candidate-refs[seed]['control'])**2))),
            'folds_improved_vs_control':sum(r['gain_vs_control']>0 for r in gd if r['group'].startswith('fold')),
            'images_improved_vs_current':sum(r['gain_squared_error_vs_current']>0 for r in errors),
            'images_improved_vs_control':sum(r['gain_squared_error_vs_control']>0 for r in errors),
            'most_hurt_vs_control':sorted(errors,key=lambda r:r['gain_squared_error_vs_control'])[:5],
            'most_helped_vs_control':sorted(errors,key=lambda r:-r['gain_squared_error_vs_control'])[:5],
            'groups':gd})
    comparison=b.aggregate(records)
    if comparison!=b.t.read_json(b.OUT/'comparison.json'):raise ValueError('Comparison differs')
    prep=b.t.read_json(b.OUT/'preparation_audit.json')
    for r in prep['preserved_submissions']:
        if b.t.file_ref(r['path'])!=r:raise ValueError('Original submission changed')
        protect(Path(r['path']))
    for path,sha in protected.items():
        if b.t.common.sha256_file(path)!=sha:raise ValueError('Protected output changed during review')
    value={'scope':'saved arrays/cells only; no fitting, extraction, inference or test images',
           'decision':'do_not_adopt_boundary37; preserve current submission; stop this fixed-feature family',
           'comparison':comparison,'status_counts':meta['status_counts'],'nullable_counts':meta['nullable_counts'],
           'common_failure_ids':plan['frozen']['common_failure_ids'],'details':details,'timing':timing,
           'checks':'500 feature cell bindings/vectors,27 anchors,10 outer cells/context medians/runtime,all OOF/group/error arithmetic,50:50,old audits/submissions',
           'sources':{str(Path(__file__).resolve()):b.t.common.sha256_file(__file__)},'protected_files':protected}
    b.t.write_json(b.OUT/'review_results.json',value)
    print(b.t.common.canonical_json({'decision':value['decision'],'comparison':comparison,'status_counts':meta['status_counts'],
                                   'diagnostics':[{k:v for k,v in d.items() if k not in ('groups','most_hurt_vs_control','most_helped_vs_control')} for d in details]}))


if __name__=='__main__':main()
