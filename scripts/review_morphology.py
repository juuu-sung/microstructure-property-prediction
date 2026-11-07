"""Audit saved morphology OOF results and diagnostics without fitting or image extraction."""
import csv
import json
from pathlib import Path
import sys
from unittest import mock

if __package__:
    from . import evaluate_morphology as e
else:
    import evaluate_morphology as e

np=e.np


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def close(a,b):
    np.testing.assert_allclose(a,b,rtol=1e-10,atol=1e-9)


def main():
    root=e.OUT
    protected={str(p):e.core.sha256(p) for p in root.rglob('*')
        if p.is_file() and p.name!='review_results.json'}
    prep=read(root/'preparation.json')
    protected.update(prep['protected_original_files'])
    protected.update(prep['protected_final_files'])
    assert all(e.core.sha256(Path(p))==h for p,h in protected.items())
    prior=read(e.ROOT/'reports/local_structure_confirmation/review_results.json')
    common17=set(prior['repeated_failures']['new_three_seeds_common_candidate_worst20'])
    assert len(common17)==17
    with mock.patch.object(e,'make_model',side_effect=AssertionError('No fit permitted')), \
         mock.patch.object(e,'nested',side_effect=AssertionError('No fit permitted')), \
         mock.patch.object(e.extract,'morphology_features',side_effect=AssertionError('No extraction permitted')):
        data=e.load_base();plan=e.validate_plan(root,data)
        features,meta=e.load_morphology(root/'features/features.npz',data,plan,root/'protocol.json')
        details=[];summaries=[];per_image=[]
        for seed in e.SEEDS:
            folder=root/f'seed{seed}';s=read(folder/'summary.json');summaries.append(s)
            assert all(e.core.sha256(folder/p)==h for p,h in read(folder/'completed.json').items())
            assert s['protocol_sha256']==e.core.sha256(root/'protocol.json')
            assert s['features_sha256']==e.core.sha256(root/'features/features.npz')
            assert s['versions']==e.versions() and s['fit_count']==230 and s['primary']==e.PRIMARY
            refs=e.load_references(seed,data)
            rows=e.core.align_rows(e.core.read_rows(folder/'oof.csv',{'ID','hardness','fold',*e.NAMES}),data.ids,folder/'oof.csv')
            np.testing.assert_array_equal(e.core.finite_column(rows,'hardness',folder),data.y)
            np.testing.assert_array_equal([int(r['fold']) for r in rows],refs['folds'])
            pred={n:e.core.finite_column(rows,n,folder) for n in e.NAMES}
            allpred,anchor=e.add_blends(pred,refs);close(anchor,s['anchor_max_abs_difference'])
            for n,p in allpred.items():close(e.core.finite_column(rows,n,folder),p)
            metrics,groups,errors=e.diagnose(data,allpred,refs['folds'])
            for actual,saved in zip(metrics,s['metrics']):
                assert actual['model']==saved['model']
                for k,v in actual.items():
                    if k!='model':close(v,saved[k])
            for name,computed,keys in [('metrics',metrics,['model']),('group_metrics',groups,['model','group']),('per_image_errors',errors,['ID'])]:
                with (folder/(name+'.csv')).open(encoding='utf-8-sig',newline='') as stream:
                    stored=list(csv.DictReader(stream))
                assert len(stored)==len(computed)
                for actual,saved in zip(computed,stored):
                    for k,v in actual.items():
                        if isinstance(v,(bool,str)):assert str(v)==saved[k]
                        else:close(v,float(saved[k]))
            search=read(folder/'inner_search.json');assert len(search)==70
            for selection in s['selections']:
                fold=selection['outer_fold']
                assert selection['inner_seed']==seed+fold
                assert selection['outer_train_n']==400 and selection['outer_valid_n']==100
                assert sorted(selection['inner_validation_sizes'])==[133,133,134]
                for name in e.NAMES:
                    family,_=e.model_spec(name)
                    candidates=[r for r in search if r['outer_fold']==fold and r['model']==name]
                    configs=e.RIDGE_CONFIGS if family=='ridge' else e.CAT_CONFIGS
                    assert [r['config'] for r in candidates]==configs
                    for r in candidates:close(r['inner_mse'],np.average(r['inner_fold_mse'],weights=selection['inner_validation_sizes']))
                    assert selection['selected'][name]==min(candidates,key=lambda r:e.choice_key(family,r))
            group_lookup={(r['model'],r['group']):r for r in groups}
            delta=(allpred[e.PRIMARY]-data.y)**2-(allpred['current_blend']-data.y)**2
            worst=lambda p:set(data.ids[np.argsort(-(p-data.y)**2,kind='stable')[:20]])
            current_worst=worst(allpred['current_blend']);candidate_worst=worst(allpred[e.PRIMARY])
            entry={'seed':seed,'elapsed_seconds':s['elapsed_seconds'],'anchor_max_abs_difference':anchor,
                'rmse':{r['model']:r['rmse'] for r in metrics},
                'selected_configs':[{ 'fold':r['outer_fold'], **{n:c['config'] for n,c in r['selected'].items()}} for r in s['selections']],
                'groups':{g:{n:group_lookup[n,g]['rmse'] for n in ['current_blend',e.PRIMARY,'catboost_existing_blend','ridge_morphology_blend']} for g in ['tails20','middle80','noise_q1','noise_q2','noise_q3','noise_q4']},
                'fold_deltas_vs_current':[group_lookup[e.PRIMARY,f'fold{j}']['rmse']-group_lookup['current_blend',f'fold{j}']['rmse'] for j in range(5)],
                'net_sse_change':float(delta.sum()),
                'group_sse_change':{g:group_lookup[e.PRIMARY,g]['sse']-group_lookup['current_blend',g]['sse'] for g in ['tails20','middle80','noise_q1','noise_q2','noise_q3','noise_q4']},
                'improved_images':int(np.sum(delta<0)),
                'top20_overlap':len(current_worst&candidate_worst),
                'prior_common17_remaining_worst20':sorted(common17&candidate_worst),
                'prior_common17_improved':sorted(str(i) for i,d in zip(data.ids,delta) if i in common17 and d<0),
                'residual_correlation_vs_current':float(np.corrcoef(allpred[e.PRIMARY]-data.y,allpred['current_blend']-data.y)[0,1])}
            details.append(entry)
            per_image.append([{'ID':str(i),'delta_sse':float(d),'current':float(c),'candidate':float(p),'hardness':float(y)}
                for i,d,c,p,y in zip(data.ids,delta,allpred['current_blend'],allpred[e.PRIMARY],data.y)])
        aggregate=e.aggregate(summaries)
        assert aggregate==read(root/'comparison.json')
        assert all(e.core.sha256(Path(p))==h for p,h in protected.items())
        assert 'torch' not in sys.modules
        result={'date_kst':'2026-10-09','scope':'Saved predictions only; no model fit, feature extraction, image reads, GPU inference or test prediction',
            'review_script_sha256':e.core.sha256(Path(__file__)),'protected_files':protected,
            'aggregate':aggregate,'details':details,'per_image_by_seed':dict(zip(map(str,e.SEEDS),per_image)),
            'prior_common17_remaining_both':sorted(set(details[0]['prior_common17_remaining_worst20'])&set(details[1]['prior_common17_remaining_worst20'])),
            'prior_common17_improved_both':sorted(set(details[0]['prior_common17_improved'])&set(details[1]['prior_common17_improved'])),
            'features_shape':list(features[e.AUGMENTED].shape),'constant_new_columns':int(np.sum(np.ptp(features[e.AUGMENTED][:,185:],axis=0)==0)),
            'feature_extraction_seconds':meta['elapsed_seconds'],'fit_count_checked':460,'fits_during_review':0}
        e.extract.write_json(root/'review_results.json',result)
        print(json.dumps({k:v for k,v in result.items() if k not in ['protected_files','per_image_by_seed']},indent=2))


if __name__=='__main__':main()
