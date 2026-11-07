"""Saved-array audit and failure analysis. No fitting, inference or extraction."""
from pathlib import Path
from unittest.mock import patch
import numpy as np
if __package__:
    from . import evaluate_scattering_tabpfn_confirmation as c
else:
    import evaluate_scattering_tabpfn_confirmation as c


def main():
    t = c.t
    with patch.object(t, 'Predictor', side_effect=AssertionError('No inference in review')), \
         patch.object(t.core, 'make_estimator', side_effect=AssertionError('No regression fitting')), \
         patch.object(c.experiment.sf, 'Extractor', side_effect=AssertionError('No image extraction')):
        data, x, refs, plan = c.validate()
        protected = t.read_json(c.OUT/'preparation_audit.json')['protected_files'].copy()
        for path in (c.OUT/'protocol.json', c.OUT/'confirmation_summary.json', c.OUT/'comparison.csv', c.OUT/'report.md'):
            protected[str(path)] = t.common.sha256_file(path)
        summaries, details, common_rows, all_errors, timing = [], [], [], [], []
        common = np.isin(data.ids, plan['frozen']['common_failure_ids'])
        if common.sum() != 16: raise ValueError('Expected frozen common16')
        for seed in c.SEEDS:
            target = c.OUT/f'seed{seed}'; ref = refs[seed]
            summary = c.read_completed(c.OUT, data, x, ref, plan, seed)
            summaries.append(summary)
            protected[str(target/'completed_evaluation.json')] = t.common.sha256_file(target/'completed_evaluation.json')
            for name in c.experiment.expected_outputs():
                protected[str(target/name)] = t.common.sha256_file(target/name)
            rows = t.core.align_rows(t.core.read_rows(target/'oof.csv', {'ID','hardness','fold',c.PRIMARY}), data.ids, target)
            col = lambda n: t.core.finite_column(rows, n, target)
            p = col(c.PRIMARY); current = ref['current']; control = ref['control']
            err, ce, be = p-data.y, current-data.y, control-data.y
            for name in summary['metrics']:
                np.testing.assert_allclose(np.sqrt(np.mean((col(name)-data.y)**2)), summary['metrics'][name]['rmse'], rtol=0, atol=1e-12)
            for fold in range(5):
                cell = t.read_json(target/f'fold{fold}_candidate/completed.json')['result']['metadata']
                old = t.read_json(c.archive.OUT/f'seed{seed}/fold{fold}_morphology285_outer/completed.json')['result']['metadata']
                if cell['runtime'] != ref['runtime'] or cell['resolved_config'] != old['resolved_config']:
                    raise ValueError('Runtime/config differs from archived285')
                if (cell['context_rows'],cell['query_rows'],cell['columns']) != (400,100,486):
                    raise ValueError('Context/query dimensions differ')
                ti = np.flatnonzero(ref['folds'] != fold)
                np.testing.assert_allclose(cell['context_mean'], data.y[ti].mean(), rtol=0, atol=1e-9)
                np.testing.assert_allclose(cell['context_std'], data.y[ti].std(), rtol=0, atol=1e-9)
                timing.append({'seed':seed, 'fold':fold, 'seconds':cell['seconds'], 'peak_gpu_allocated_bytes':cell['peak_gpu_allocated_bytes']})
            _, _, diag, errors = c.experiment.result_data(data, ref, col('tabpfn486'), plan['frozen']['common_failure_ids'])
            lookup = {(r['model'],r['group']):r for r in diag}
            groups = []
            for group in sorted({r['group'] for r in diag}):
                a,b,d = [lookup[(n,group)] for n in ('current','fixed285_control_blend',c.PRIMARY)]
                groups.append({'group':group, 'n':d['n'], 'current_rmse':a['rmse'], 'control_rmse':b['rmse'],
                    'candidate_rmse':d['rmse'], 'gain_vs_current':a['rmse']-d['rmse'], 'gain_vs_control':b['rmse']-d['rmse'],
                    'sse_gain_vs_current':a['sse']-d['sse'], 'sse_gain_vs_control':b['sse']-d['sse']})
            worst = lambda error:set(np.argsort(error**2,kind='stable')[-20:])
            cw,bw,nw = worst(ce),worst(be),worst(err)
            gain = be**2-err**2
            record = {'seed':seed, 'metrics':summary['metrics'], 'groups':groups,
                'standalone_gain_vs_285':summary['metrics']['tabpfn285_control']['rmse']-summary['metrics']['tabpfn486']['rmse'],
                'prediction_rms_change_vs_control':float(np.sqrt(np.mean((p-control)**2))),
                'residual_correlation_with_control':float(np.corrcoef(err,be)[0,1]),
                'images_improved_vs_control':int((gain>0).sum()), 'images_improved_vs_current':int((ce**2>err**2).sum()),
                'common16_improved_vs_control':int((gain[common]>0).sum()),
                'common16_improved_vs_current':int((ce[common]**2>err[common]**2).sum()),
                'common16_same_error_sign_as_control':int((np.sign(err[common])==np.sign(be[common])).sum()),
                'common16_same_error_sign_as_current':int((np.sign(err[common])==np.sign(ce[common])).sum()),
                'common16_in_candidate_worst20':int(sum(common[list(nw)])),
                'common16_sse_share':float(np.sum(err[common]**2)/np.sum(err**2)),
                'worst20_overlap_current':len(cw&nw), 'worst20_overlap_control':len(bw&nw),
                'new_worst20_vs_current':[str(data.ids[j]) for j in sorted(nw-cw)],
                'new_worst20_vs_control':[str(data.ids[j]) for j in sorted(nw-bw)],
                'left_worst20_vs_current':[str(data.ids[j]) for j in sorted(cw-nw)],
                'left_worst20_vs_control':[str(data.ids[j]) for j in sorted(bw-nw)],
                'most_helped_vs_control':sorted(errors,key=lambda r:-r['gain_squared_error_vs_control'])[:5],
                'most_hurt_vs_control':sorted(errors,key=lambda r:r['gain_squared_error_vs_control'])[:5]}
            details.append(record)
            for j, image_id in enumerate(data.ids):
                row = {'seed':seed,'ID':str(image_id),'hardness':float(data.y[j]),'fold':int(ref['folds'][j]),
                    'common16':bool(common[j]),'current_error':float(ce[j]),'control_error':float(be[j]),
                    'candidate_error':float(err[j]),'candidate_worst20':j in nw,
                    'absolute_error_gain_vs_control':float(abs(be[j])-abs(err[j])),
                    'squared_error_gain_vs_control':float(gain[j])}
                all_errors.append(row)
                if common[j]:common_rows.append(row)
        aggregate = {**c.aggregate(summaries),'protocol_sha256':t.digest(plan['frozen'])}
        if aggregate != t.read_json(c.OUT/'confirmation_summary.json'):
            raise ValueError('Saved confirmation aggregate differs')
        c.same_rows(c.OUT/'comparison.csv', aggregate['per_seed'])
        selection_path = t.ROOT/'reports/final_tabpfn_blend/selection/selection.json'
        selection = t.read_json(selection_path)
        common_summary = []
        for image_id in sorted(plan['frozen']['common_failure_ids']):
            rr = [r for r in common_rows if r['ID'] == image_id]
            common_summary.append({'ID':image_id,'improved_seeds_vs_control':sum(r['absolute_error_gain_vs_control']>0 for r in rr),
                'mean_abs_current_error':float(np.mean([abs(r['current_error']) for r in rr])),
                'mean_abs_control_error':float(np.mean([abs(r['control_error']) for r in rr])),
                'mean_abs_candidate_error':float(np.mean([abs(r['candidate_error']) for r in rr])),
                'mean_absolute_error_gain_vs_control':float(np.mean([r['absolute_error_gain_vs_control'] for r in rr]))})
        for path, sha in protected.items():
            if t.common.sha256_file(path) != sha: raise ValueError('Protected file changed: '+path)
        selected_groups = ('common16','remaining484','low10','high10','middle80','tails20','noiseQ1','noiseQ2','noiseQ3','noiseQ4')
        group_means = {g:{k:float(np.mean([next(r for r in d['groups'] if r['group']==g)[k] for d in details]))
                          for k in ('gain_vs_current','gain_vs_control','sse_gain_vs_current','sse_gain_vs_control')}
                       for g in selected_groups}
        result = {'date_kst':'2026-10-12','scope':'CPU saved arrays/cells only; no fitting, extraction, GPU/test inference or submission generation',
            'comparison':aggregate, 'details':details, 'common16':common_summary, 'mean_group_gains':group_means,
            'common16_summary_is_mean_absolute_error_not_mean_prediction':True,
            'checks':'frozen inputs/500 feature cells/120 archived contexts/15 new cells;400/100 and exact runtime;1500 OOF rows/all diagnostics/errors/50:50/aggregate;source/cache/submission preservation',
            'final_submission_selected_representation':selection['representation'], 'final_selection_source':t.file_ref(selection_path),
            'timing':timing, 'protected_files':protected, 'review_source':t.file_ref(__file__)}
        t.write_json(c.OUT/'review_results.json',result)
        t.write_csv(c.OUT/'common16_changes.csv',common_rows)
        t.write_csv(c.OUT/'common16_summary.csv',common_summary)
        t.write_csv(c.OUT/'review_per_image.csv',all_errors)
    print(t.common.canonical_json({'audit':'passed','comparison':aggregate,'mean_group_gains':group_means}))
    for d in details:
        print(t.common.canonical_json({k:v for k,v in d.items() if k not in ('metrics','groups','most_helped_vs_control','most_hurt_vs_control')}))
    print('Common16 mean absolute errors across3 seeds:',t.common.canonical_json(common_summary))


if __name__=='__main__':main()
