"""Read-only audit of available completed refined-structure OOF runs.

Never fits, extracts or finishes missing runs. A one-seed review is explicitly
provisional. Writes only separate review artifacts, not experiment results.
"""
from pathlib import Path
import contextlib
import io
from unittest.mock import patch
import numpy as np
if __package__:
    from . import evaluate_refined_structure as e
    from .review_scattering_tabpfn import same_rows
else:
    import evaluate_refined_structure as e
    from review_scattering_tabpfn import same_rows


def main():
    t=e.t
    with patch.object(t,'Predictor',side_effect=AssertionError('Review cannot run inference')), \
         patch.object(e.features,'extract',side_effect=AssertionError('Review cannot extract images')):
        data,base,refs,plan=e.validate(e.OUT)
        added,meta=e.load_features(e.OUT,data,plan)
        matrices={r:np.c_[base,added[r]] for r in e.REPS}
        common=np.isin(data.ids,plan['frozen']['common_failure_ids'])
        if common.sum()!=16:raise ValueError('Expected frozen historical common16')
        protected={}
        def protect(path):protected[str(path)]=t.common.sha256_file(path)
        for path in (e.OUT/'protocol.json',e.OUT/'features/metadata.json',e.OUT/'features/features.npz'):
            protect(path)
        for path in plan['frozen']['sources'].values():protect(Path(path['path']))
        protect(Path(plan['frozen']['best_submission']['path']))
        for name in meta['cells']:protect(e.OUT/'features'/name)
        details=[];records=[];common_rows=[];all_groups=[];timing=[];missing=[]
        for seed in e.SEEDS:
            folder=e.OUT/f'seed{seed}'
            if not (folder/'completed_evaluation.json').exists():
                missing.append({'seed':seed,'directory_exists':folder.exists(),
                    'completed_cells':len(list(folder.glob('fold*/completed.json'))),
                    'state':'no_complete_result; no inference about whether running elsewhere'})
                continue
            ref=refs[seed];binding=e.evaluation_binding(e.OUT,plan,ref,seed)
            if not e.completed(folder,binding):raise ValueError('Invalid completed evaluation')
            protect(folder/'completed_evaluation.json')
            for name in e.expected_outputs():protect(folder/name)
            pred={r:np.full(500,np.nan) for r in e.REPS}
            with contextlib.redirect_stdout(io.StringIO()):
                for fold in range(5):
                    vi,outputs=e.evaluate_fold(folder,binding,matrices,data.y,ref['folds'],seed,fold,e.forbidden)
                    ti=np.flatnonzero(ref['folds']!=fold)
                    original=t.read_json(e.previous.OUT/f'seed{seed}/fold{fold}_candidate/completed.json')['result']['metadata']
                    for r in e.REPS:
                        pred[r][vi]=outputs[r]
                        cell=t.read_json(folder/f'fold{fold}_{r}/completed.json')['result']['metadata']
                        if cell['runtime']!=ref['runtime'] or cell['resolved_config']!=original['resolved_config']:
                            raise ValueError('Runtime/config differs from fixed486')
                        if (cell['context_rows'],cell['query_rows'],cell['columns'])!=(400,100,500):
                            raise ValueError('Unexpected context/query dimensions')
                        np.testing.assert_allclose(cell['context_mean'],data.y[ti].mean(),rtol=0,atol=1e-9)
                        np.testing.assert_allclose(cell['context_std'],data.y[ti].std(),rtol=0,atol=1e-9)
                        _,_,imputation=e.features.impute_context(matrices[r][ti],matrices[r][vi])
                        if cell['outer_context_imputation']!=imputation:raise ValueError('Context-only imputation differs')
                        timing.append({'seed':seed,'fold':fold,'representation':r,'seconds':cell['seconds'],
                                       'peak_gpu_allocated_bytes':cell['peak_gpu_allocated_bytes']})
            metrics,rows,diag,errors=e.result_data(data,ref,pred,plan['frozen']['common_failure_ids'])
            if metrics!=t.read_json(folder/'summary.json')['metrics']:raise ValueError('Recomputed metrics differ')
            for name,expected in (('oof.csv',rows),('diagnostics.csv',diag),('per_image_errors.csv',errors)):
                same_rows(folder/name,expected)
            candidate=.5*(ref['base']+pred['refined14']);control=.5*(ref['base']+pred['legacy14'])
            err,ce,be=candidate-data.y,ref['current']-data.y,control-data.y
            worst=lambda p:set(np.argsort(p**2,kind='stable')[-20:])
            cw,bw,nw=worst(ce),worst(be),worst(err)
            lookup={(r['model'],r['group']):r for r in diag};groups=[]
            for group in sorted({r['group'] for r in diag}):
                a,b,c=[lookup[(name,group)] for name in ('current486_blend',e.CONTROL,e.PRIMARY)]
                g={'seed':seed,'group':group,'n':c['n'],'current_rmse':a['rmse'],'control_rmse':b['rmse'],
                   'candidate_rmse':c['rmse'],'gain_vs_current':a['rmse']-c['sse']**.5/c['n']**.5,
                   'gain_vs_control':b['rmse']-c['rmse'],'sse_gain_vs_current':a['sse']-c['sse'],
                   'sse_gain_vs_control':b['sse']-c['sse']}
                # Independent SSE/RMSE consistency check.
                np.testing.assert_allclose(g['gain_vs_current'],a['rmse']-c['rmse'],rtol=0,atol=1e-12)
                groups.append(g)
            all_groups.extend(groups)
            d={'seed':seed,'metrics':metrics,'groups':groups,
               'standalone_gain_vs_486':metrics['tabpfn486']['rmse']-metrics['tabpfn_refined500']['rmse'],
               'standalone_gain_vs_legacy500':metrics['tabpfn_legacy500']['rmse']-metrics['tabpfn_refined500']['rmse'],
               'folds_improved_vs_current':sum(r['gain_vs_current']>0 for r in groups if r['group'].startswith('fold')),
               'folds_improved_vs_control':sum(r['gain_vs_control']>0 for r in groups if r['group'].startswith('fold')),
               'images_improved_vs_current':int((ce**2>err**2).sum()),
               'images_improved_vs_control':int((be**2>err**2).sum()),
               'prediction_rms_change_vs_current':float(np.sqrt(np.mean((candidate-ref['current'])**2))),
               'prediction_rms_change_vs_control':float(np.sqrt(np.mean((candidate-control)**2))),
               'common16_improved_vs_current':int((ce[common]**2>err[common]**2).sum()),
               'common16_improved_vs_control':int((be[common]**2>err[common]**2).sum()),
               'common16_same_error_sign_as_current':int((np.sign(err[common])==np.sign(ce[common])).sum()),
               'common16_worst20_counts':{'current':sum(common[list(cw)]).item(),
                   'control':sum(common[list(bw)]).item(),'candidate':sum(common[list(nw)]).item()},
               'common16_absolute_error_range':[float(abs(err[common]).min()),float(abs(err[common]).max())],
               'common16_sse_fraction':float(np.sum(err[common]**2)/np.sum(err**2)),
               'worst20_overlap_current':len(cw&nw),'worst20_overlap_control':len(bw&nw),
               'new_worst20_ids':[str(data.ids[j]) for j in sorted(nw-cw)],
               'left_worst20_ids':[str(data.ids[j]) for j in sorted(cw-nw)],
               'most_helped_vs_current':sorted(errors,key=lambda r:-r['gain_squared_error_vs_current'])[:5],
               'most_hurt_vs_current':sorted(errors,key=lambda r:r['gain_squared_error_vs_current'])[:5]}
            details.append(d)
            m=metrics[e.PRIMARY]
            records.append({'seed':seed,'current_rmse':metrics['current486_blend']['rmse'],
                'legacy_control_rmse':metrics[e.CONTROL]['rmse'],'candidate_rmse':m['rmse'],
                'gain_vs_current':m['gain_vs_current'],'gain_vs_control':m['gain_vs_control']})
            for j in np.flatnonzero(common):
                common_rows.append({'seed':seed,'ID':str(data.ids[j]),'hardness':float(data.y[j]),
                    'current_error':float(ce[j]),'control_error':float(be[j]),'candidate_error':float(err[j]),
                    'absolute_error_gain_vs_current':float(abs(ce[j])-abs(err[j])),
                    'absolute_error_gain_vs_control':float(abs(be[j])-abs(err[j])),
                    'squared_error_gain_vs_current':float(ce[j]**2-err[j]**2),
                    'candidate_worst20':j in nw})
        if not details:raise ValueError('No completed seeds available to review')
        comparison=None
        if not missing:
            comparison={**e.aggregate(records),'protocol_sha256':t.digest(plan['frozen'])}
            if (e.OUT/'comparison.json').exists() and comparison!=t.read_json(e.OUT/'comparison.json'):
                raise ValueError('Saved comparison differs')
        for path,sha in protected.items():
            if t.common.sha256_file(path)!=sha:raise ValueError('Protected file changed during review')
        feature_stats={rep:{'rows':len(x),'columns':x.shape[1],
            'missing_counts':{name:int(np.isnan(x[:,j]).sum()) for j,name in enumerate(e.features.NAMES)},
            'range':{name:([float(np.nanmin(x[:,j])),float(np.nanmax(x[:,j]))] if np.isfinite(x[:,j]).any() else None)
                     for j,name in enumerate(e.features.NAMES)}} for rep,x in added.items()}
        result={'scope':'Saved arrays/cells only; no feature extraction, inference, fitting, test or submission writes.',
            'complete_seeds':[d['seed'] for d in details],'missing_complete_results':missing,
            'provisional':bool(missing),'comparison':comparison,'per_seed':records,'details':details,
            'feature_stats':feature_stats,'timing':timing,'extraction_seconds':meta['seconds'],
            'protected_files':protected,'review_source':t.file_ref(__file__),
            'checks':'Frozen inputs/500 feature cells/context-only imputation/exact archived runtime and config; '
                     'OOF arithmetic/all diagnostics/per-image errors/50:50; original source/cache/submission preserved.',
            'independent_holdout':False,'prediction_averaging':False}
        review=e.OUT/'review';review.mkdir(exist_ok=True)
        t.write_json(review/'review_results.json',result)
        t.write_csv(review/'common16_changes.csv',common_rows)
        t.write_csv(review/'group_comparison.csv',all_groups)
        print(t.common.canonical_json({k:result[k] for k in ('complete_seeds','missing_complete_results','provisional','per_seed','comparison')}))
        for d in details:
            print(t.common.canonical_json({k:v for k,v in d.items() if k not in ('metrics','groups','most_helped_vs_current','most_hurt_vs_current')}))
            print(t.common.canonical_json([r for r in d['groups'] if r['group'] in ('common16','remaining484','middle80','low10','high10','noiseQ1','noiseQ2','noiseQ3','noiseQ4')]))


if __name__=='__main__':main()
