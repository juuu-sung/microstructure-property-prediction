"""Read-only audit of saved ResNet texture predictions; no fitting or forward."""
import csv
from pathlib import Path
import numpy as np
if __package__:
    from . import resnet_texture_experiment as e
else:
    import resnet_texture_experiment as e


def csv_rows(path):
    with path.open(encoding='utf8',newline='') as f:return list(csv.DictReader(f))


def main():
    data,raw,refs,plan,_=e.validate(e.OUT)
    features,meta=e.load_features(e.OUT,data,raw,plan)
    protected={}
    def protect(path):protected[str(path)]=e.base.sha256_file(path)
    for p in (e.OUT/'protocol.json',e.OUT/'comparison.json',e.OUT/'features/features.npz',e.OUT/'features/metadata.json'):
        protect(p)
    for p in plan['frozen']['submissions']:protect(Path(p['path']))
    for n in ('early_moments2048','early_cov3104'):
        np.testing.assert_array_equal(features[n][:,:1792],raw)
    for lo,hi in ((1856,1920),(1984,2048)):
        if (features['early_moments2048'][:,lo:hi]<0).any():raise ValueError('Negative early standard deviation')
    norms=[]
    for lo in (2048,2576):
        norm=np.linalg.norm(features['early_cov3104'][:,lo:lo+528].astype(float),axis=1)
        if not np.all(np.isclose(norm,1,rtol=0,atol=1e-6)|(norm==0)):raise ValueError('Invalid covariance normalization')
        norms.append({'minimum':float(norm.min()),'maximum':float(norm.max()),'zero_vectors':int((norm==0).sum())})
    def forbidden(*args):raise ValueError('Review cannot refit a missing fold')
    reviews=[];comparison_rows=[]
    for seed in e.SEEDS:
        folder=e.OUT/f'seed{seed}';ref=refs[seed];binding=e.evaluation_binding(e.OUT,seed,plan)
        if not e.completed(folder,binding):raise ValueError('Experiment incomplete')
        protect(folder/'completed.json')
        for n in e.t.read_json(folder/'completed.json')['files']:protect(folder/n)
        summary=e.t.read_json(folder/'summary.json')
        if summary['binding']!=binding or summary['primary']!=e.PRIMARY:raise ValueError('Summary binding differs')
        prediction={n:np.full(len(data.y),np.nan) for n in e.REPS};choices=[]
        for fold in range(5):
            cell=e.fold_cell(folder,binding,features,data.y,ref['folds'],seed,fold,runner=forbidden)
            for n in e.REPS:
                if any(not np.isfinite(c['inner_mse']) or c['inner_mse']<0 for c in cell['candidates'][n]):
                    raise ValueError('Invalid inner score')
                prediction[n][cell['validation_indices']]=cell['predictions'][n]
            choices.append(cell['selection'])
        if choices!=summary['selection']:raise ValueError('Selections differ')
        output=e.compose_predictions(ref,prediction)
        rows=e.core.align_rows(e.core.read_rows(folder/'oof.csv',{'ID','hardness','fold',*output}),data.ids,folder)
        np.testing.assert_array_equal(e.core.finite_column(rows,'hardness',folder),data.y)
        np.testing.assert_array_equal(e.core.integer_column(rows,'fold',folder),ref['folds'])
        metrics={}
        for n,p in output.items():
            np.testing.assert_array_equal(e.core.finite_column(rows,n,folder),p)
            metrics[n]=e.core.metrics(data.y,p)
        if metrics!=summary['metrics']:raise ValueError('Recalculated summary differs')
        ranks=e.core.rank_groups(data.y,10);noise=e.core.rank_groups(data.noise,4)
        masks={'all':np.ones(len(data.y),bool),'low10':ranks==0,'middle80':(ranks>0)&(ranks<9),
               'high10':ranks==9,'common16':np.isin(data.ids,plan['frozen']['common_failure_ids'])}
        masks.update({f'fold{f}':ref['folds']==f for f in range(5)})
        masks.update({f'noiseQ{q+1}':noise==q for q in range(4)})
        saved=csv_rows(folder/'diagnostics.csv');lookup={(r['model'],r['group']):r for r in saved}
        if len(saved)!=len(output)*len(masks) or len(lookup)!=len(saved):raise ValueError('Diagnostic groups differ')
        group_metrics={}
        for n,p in output.items():
            for g,mask in masks.items():
                m=e.core.metrics(data.y[mask],p[mask]);group_metrics[n,g]=m
                for k,v in m.items():np.testing.assert_allclose(float(lookup[n,g][k]),v,rtol=0,atol=1e-9)
        errors=e.core.read_rows(folder/'per_image_errors.csv',{'ID','hardness','common16'})
        expected_order=data.ids[np.argsort(-(ref['current']-data.y)**2,kind='stable')].tolist()
        if [r['ID'] for r in errors]!=expected_order:raise ValueError('Error sort differs')
        aligned=e.core.align_rows(errors,data.ids,folder)
        for i,r in enumerate(aligned):
            for n in ('current',*e.BLENDS):
                np.testing.assert_allclose(float(r[n+'_error']),output[n][i]-data.y[i],rtol=0,atol=1e-10)
            for n in ('current',*e.BLENDS[:2]):
                gain=(output[n][i]-data.y[i])**2-(output[e.PRIMARY][i]-data.y[i])**2
                np.testing.assert_allclose(float(r['gain_squared_vs_'+n]),gain,rtol=0,atol=1e-9)
            if r['common16']!=str(bool(masks['common16'][i])):raise ValueError('Common failure flag differs')
            np.testing.assert_allclose([float(r['hardness']),float(r['noise'])],[data.y[i],data.noise[i]],rtol=0,atol=1e-10)
        groups=[]
        for g,mask in masks.items():
            candidate=group_metrics[e.PRIMARY,g]
            groups.append({'group':g,'n':int(mask.sum()),'candidate_rmse':candidate['rmse'],
                **{n+'_rmse':group_metrics[n,g]['rmse'] for n in ('current',*e.BLENDS[:2])},
                **{'gain_vs_'+n:group_metrics[n,g]['rmse']-candidate['rmse'] for n in ('current',*e.BLENDS[:2])}})
        best={n:set(np.argsort((p-data.y)**2,kind='stable')[-20:]) for n,p in output.items()}
        worst_common=np.flatnonzero(masks['common16'])
        diag={'seed':seed,'metrics':metrics,'groups':groups,
            'alphas':[{n:c[n]['alpha'] for n in e.REPS} for c in choices],
            'folds_improved_vs_current':sum(r['gain_vs_current']>0 for r in groups if r['group'].startswith('fold')),
            'images_improved_vs_current':sum(float(r['gain_squared_vs_current'])>0 for r in errors),
            'prediction_rms_change':float(np.sqrt(np.mean((output[e.PRIMARY]-output['current'])**2))),
            'residual_correlation':float(np.corrcoef(output[e.PRIMARY]-data.y,output['current']-data.y)[0,1]),
            'worst20_overlap_with_current':len(best['current']&best[e.PRIMARY]),
            'common16_in_candidate_worst20':len(set(worst_common)&best[e.PRIMARY]),
            'most_hurt':sorted(errors,key=lambda r:float(r['gain_squared_vs_current']))[:5],
            'most_helped':sorted(errors,key=lambda r:-float(r['gain_squared_vs_current']))[:5]}
        reviews.append(diag)
        comparison_rows.append({'seed':seed,'current_rmse':metrics['current']['rmse'],
            'candidate_rmse':metrics[e.PRIMARY]['rmse'],
            **{'gain_vs_'+n:metrics[n]['rmse']-metrics[e.PRIMARY]['rmse'] for n in ('current',*e.BLENDS[:2])}})
    comparison=e.t.read_json(e.OUT/'comparison.json')
    if comparison_rows!=comparison['per_seed'] or comparison['protocol_sha256']!=e.t.digest(plan['frozen']):
        raise ValueError('Comparison differs')
    gain=float(np.mean([r['gain_vs_current'] for r in comparison_rows]))
    positive=all(r['gain_vs_'+n]>0 for r in comparison_rows for n in ('current',*e.BLENDS[:2]))
    if (gain!=comparison['mean_gain_vs_current'] or positive!=comparison['all_seeds_improve_all_controls'] or
        comparison['passes_expansion_priority']!=(positive and gain>=e.PROTOCOL['priority_mean_gain'])):
        raise ValueError('Decision gates differ')
    for p,sha in protected.items():
        if e.base.sha256_file(p)!=sha:raise ValueError('Protected output changed during review')
    result={'scope':'Saved caches/predictions only; no extraction, fitting, GPU inference, test access or submission',
        'decision':'Preserve as small-positive candidate; keep current submission; no expansion or weight tuning justified',
        'comparison':comparison,'mean_gain_from_covariance_vs_early_control':float(np.mean([
            r['gain_vs_early_moments_control_blend'] for r in comparison_rows])),
        'feature_identity_max_difference':meta['identity_max_difference'],'extraction_seconds':meta['seconds'],
        'covariance_norms':norms,'reviewed_outer_cells':10,'reviewed_prediction_rows':1000,
        'limitations':'Same repeatedly studied500. Inner score records/choice checked without refitting; '
            'GPU identity result taken from validated saved metadata, no repeated image forward. '
            'Noise quartiles use this experiment data.noise, not physical ground-truth noise.',
        'seeds':reviews,'protected_files':protected,'source':e.t.file_ref(__file__)}
    e.t.write_json(e.OUT/'review_results.json',result)
    print(e.base.canonical_json({'decision':result['decision'],'mean_gain':gain,
        'mean_covariance_gain':result['mean_gain_from_covariance_vs_early_control'],
        'identity_max_difference':meta['identity_max_difference'],
        'diagnostics':[{k:r[k] for k in ('seed','folds_improved_vs_current','images_improved_vs_current',
            'prediction_rms_change','residual_correlation','worst20_overlap_with_current','common16_in_candidate_worst20')} for r in reviews]}))


if __name__=='__main__':main()
