"""Retrospective train-only failure diagnosis; cached features/OOF, no model fits.

Neighbors use only each outer training set for feature standardization and
reference points. Their labels are displayed diagnostically, never used for
feature selection or a proposed prediction. Common16 is an error-selected group.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
if __package__:
    from . import resnet_texture_experiment as e
else:
    import resnet_texture_experiment as e

ROOT=e.ROOT
OUT=ROOT/'reports/common_failure_diagnosis_20261011'
SOURCES={}
EXPERTS={
 'local_structure':('hand_plus_structure_nested',),
 'frozen_resnet18':('final_gap_nested','multiscale_mean_std_nested'),
 'frozen_dinov2_small':('joint_nested',),
 'dinov2_d4':('d4_mean_cls_patch_mean_std_nested',),
 'dinov2_spatial':('global_plus_2x2_nested',),
 'dinov2_kernel_lower':('joint_nested',),
 'morphology':('ridge_morphology','catboost_existing','catboost_morphology'),
 'tabpfn_v2':('tabpfn_existing185','tabpfn_morphology285'),
 'boundary_tabpfn':('tabpfn322',),
 'convnext_full':('convnext_frozen_ridge','convnext_nested'),
 'resnet_texture':('early_moments2048','early_cov3104'),
 'dinov2_partial':('head_only_nested','partial_nested'),
 'dinov2_ridge_initialized_160':('ridge_partial_nested',),
}


def record(path):
    path=Path(path);SOURCES[str(path)]=e.base.sha256_file(path);return path


def table(path,data,folds=None):
    df=pd.read_csv(record(path)).set_index('ID')
    if len(df)!=500 or not df.index.is_unique or set(df.index)!=set(data.ids):raise ValueError('Invalid prediction IDs')
    df=df.loc[data.ids]
    if 'hardness' in df:np.testing.assert_allclose(df.hardness,data.y,rtol=0,atol=1e-10)
    if folds is not None:np.testing.assert_array_equal(df.fold,folds)
    return df


def features_npz(path,data,keys):
    with np.load(record(path),allow_pickle=False) as z:
        np.testing.assert_array_equal(z['ids'],data.ids)
        values=[z[k].astype(float) for k in keys]
    if not all(np.isfinite(v).all() for v in values):raise ValueError('Invalid features')
    return np.concatenate(values,axis=1)


def distances(a,b):
    d=(a*a).sum(1)[:,None]+(b*b).sum(1)[None,:]-2*a@b.T
    return np.sqrt(np.maximum(d,0)/a.shape[1])


def neighbor_diagnosis(x,data,folds,seed,name):
    rows=[]
    for fold in range(5):
        ti=np.flatnonzero(folds!=fold);vi=np.flatnonzero(folds==fold)
        mean=x[ti].mean(0);std=x[ti].std(0);std=np.where(std<1e-12,1,std)
        train=(x[ti]-mean)/std;query=(x[vi]-mean)/std
        dt=distances(train,train);np.fill_diagonal(dt,np.inf)
        train_k5=np.partition(dt,4,axis=1)[:,:5].mean(1)
        dq=distances(query,train);nn=np.argsort(dq,axis=1,kind='stable')[:,:5]
        for j,i in enumerate(vi):
            idx=ti[nn[j]];distance=float(dq[j,nn[j]].mean());yy=data.y[idx]
            rows.append({'ID':str(data.ids[i]),'seed':seed,'representation':name,'hardness':float(data.y[i]),
                'distance5':distance,'support_percentile':float(np.mean(train_k5<=distance)*100),
                'neighbor_y_mean':float(yy.mean()),'neighbor_y_std':float(yy.std()),
                'neighbor_y_min':float(yy.min()),'neighbor_y_max':float(yy.max()),
                'target_outside_neighbor_range':bool(data.y[i]<yy.min() or data.y[i]>yy.max()),
                'delta_y_vs_neighbor_mean':float(data.y[i]-yy.mean()),
                **{f'neighbor{k+1}':str(data.ids[n]) for k,n in enumerate(idx)}})
    return rows


def run():
    data,raw,refs,plan,_=e.validate(e.OUT)
    tex,_=e.load_features(e.OUT,data,raw,plan)
    common_ids=plan['frozen']['common_failure_ids'];common=np.isin(data.ids,common_ids)
    if len(set(common_ids)) != 16 or int(common.sum()) != 16:
        raise ValueError('Expected exactly16 frozen common failures')
    record(e.OUT/'protocol.json');record(e.OUT/'features/features.npz')
    audit=table(ROOT/'reports/audit_train_features.csv',data)
    feature_sets={
      'hand285':np.c_[features_npz(ROOT/'reports/local_structure/features/features.npz',data,['hand_original','local_structure']),
                       features_npz(ROOT/'reports/morphology/features/features.npz',data,['morphology'])],
      'dino_d4_1152':features_npz(ROOT/'reports/dinov2_d4/features/features.npz',data,['d4_mean_cls_patch_mean_std']),
      'resnet_texture3104':tex['early_cov3104'].astype(float)}
    y=data.y;deciles=e.core.rank_groups(y,10);tails=(deciles==0)|(deciles==9)
    noise=audit.haar_hh_noise_proxy.to_numpy();noise_rank=e.core.rank_groups(noise,4)
    quality=[]
    for ident in data.ids:
        p=ROOT/'reports/boundary_tabpfn/features/cells'/f'{ident}.json'
        q=e.t.read_json(record(p))['result']['quality'];quality.append(q['status'])
    quality=np.asarray(quality)
    population={}
    for name,mask in {'all500':np.ones(500,bool),'common16':common,'other484':~common,'tail_noncommon':tails&~common}.items():
        population[name]={'n':int(mask.sum()),'low10':int((mask&(deciles==0)).sum()),'high10':int((mask&(deciles==9)).sum()),
            'noise_median':float(np.median(noise[mask])), 'noise_lower_two_rank_quartiles':int((mask&(noise_rank<2)).sum()),
            'noise_quartile_counts':[int((mask&(noise_rank==k)).sum()) for k in range(4)],
            'quality_counts':{str(k):int((quality[mask]==k).sum()) for k in np.unique(quality)}}
    model_results=[];image_records=[];oracle_results=[];neighbors=[];baseline_results=[]
    for seed in e.SEEDS:
        folds=refs[seed]['folds'];current=refs[seed]['current']
        expert={}
        for folder,columns in EXPERTS.items():
            df=table(ROOT/f'reports/{folder}/seed{seed}/oof.csv',data,folds)
            for column in columns:
                label=folder+':'+column;pred=df[column].to_numpy();expert[label]=pred
                model_results.append({'seed':seed,'expert':label,'all_rmse':e.core.metrics(y,pred)['rmse'],
                    'common16_rmse':e.core.metrics(y[common],pred[common])['rmse'],
                    'common16_within10':int((np.abs(pred[common]-y[common])<10).sum()),
                    'common16_within20':int((np.abs(pred[common]-y[common])<20).sum())})
        names=list(expert);p=np.stack(list(expert.values()),axis=1)
        if p.shape != (500,20) or not np.isfinite(p).all():
            raise ValueError('Expected20 finite saved model predictions per image')
        error=p-y[:,None];same_side=(error.min(1)>0)|(error.max(1)<0)
        best=np.argmin(np.abs(error),axis=1)
        for i in np.flatnonzero(common):
            image_records.append({'ID':str(data.ids[i]),'seed':seed,'hardness':float(y[i]),
                'current_prediction':float(current[i]),'current_error':float(current[i]-y[i]),
                'expert_min':float(p[i].min()),'expert_max':float(p[i].max()),
                'expert_sd':float(p[i].std()),'same_error_side_all_experts':bool(same_side[i]),
                'best_expert_oracle_not_validation':names[best[i]],'best_abs_error_oracle':float(abs(error[i,best[i]])),
                'noise_proxy':float(noise[i]),'quality_status':str(quality[i])})
        impossible_oracle=np.clip(y,p.min(1),p.max(1))
        oracle_results.append({'seed':seed,'experts':len(names),'same_side_common16':int(same_side[common].sum()),
            'common16_oracle_convex_hull_rmse':e.core.metrics(y[common],impossible_oracle[common])['rmse'],
            'all500_oracle_convex_hull_rmse':e.core.metrics(y,impossible_oracle)['rmse'],
            'meaning':'Target-aware per-image projection into min/max of20 saved expert predictions. '
                'Unattainable diagnostic lower bound for convex mixing these saved predictions, NOT a model or validation score.'})
        se=(current-y)**2
        baseline_results.append({'seed':seed,'current_rmse':float(np.sqrt(se.mean())),
            'common16_rmse':float(np.sqrt(se[common].mean())),
            'common16_sse_share':float(se[common].sum()/se.sum()),
            'counterfactual_common16_perfect_rmse_over_all500':float(np.sqrt(se[~common].sum()/500)),
            'other484_rmse':float(np.sqrt(se[~common].mean())),
            'caveat':'Counterfactual replaces16 known labels perfectly and is not implementable; no changes to predictions.'})
        for name,x in feature_sets.items():neighbors.extend(neighbor_diagnosis(x,data,folds,seed,name))
    neighbor_frame=pd.DataFrame(neighbors);neighbor_frame['common16']=neighbor_frame.ID.isin(common_ids)
    neighbor_frame['tail_noncommon']=neighbor_frame.ID.isin(data.ids[tails&~common])
    neighbor_summary=[]
    for (seed,rep),df in neighbor_frame.groupby(['seed','representation'],sort=False):
        for label,mask in [('common16',df.common16),('other484',~df.common16),('tail_noncommon',df.tail_noncommon)]:
            d=df[mask]
            neighbor_summary.append({'seed':int(seed),'representation':rep,'group':label,'n':len(d),
                'median_support_percentile':float(d.support_percentile.median()),
                'support_percentile_above95_count':int((d.support_percentile>95).sum()),
                'target_outside_neighbor_range':int(d.target_outside_neighbor_range.sum()),
                'median_abs_target_minus_neighbors':float(d.delta_y_vs_neighbor_mean.abs().median()),
                'median_neighbor_target_std':float(d.neighbor_y_std.median())})
    OUT.mkdir(exist_ok=True,parents=True)
    pd.DataFrame(image_records).to_csv(OUT/'common16_experts.csv',index=False)
    pd.DataFrame(model_results).to_csv(OUT/'expert_scores.csv',index=False)
    neighbor_frame.to_csv(OUT/'neighbors_all500.csv',index=False)
    neighbor_frame[neighbor_frame.common16].to_csv(OUT/'neighbors_common16.csv',index=False)
    report={'scope':'Train-only retrospective diagnosis; no supervised fitting, GPU inference or test access',
        'population':population,'baseline':baseline_results,'oracle':oracle_results,'neighbor_summary':neighbor_summary,
        'neighbor_method':'k=5 Euclidean RMS in per-outer-training standardized feature space; '
            'query outside its training fold; support percentile vs training leave-one-out5-neighbor mean distance. '
            'Descriptive distance, not calibrated OOD probability or a new predictor. Fixed3 representations; no label-based neighbor selection.',
        'limitations':['Common16 selected from repeated OOF errors; associations are posthoc and not causal.',
            'Same500 reused; feature similarity is not identical physical structure or evidence of wrong labels.',
            'Quality status measures repeatable boundary proxy, not segmentation accuracy.',
            'Inner/outer correlations and oracle bounds do not validate a learned gate, mixture or calibration.'],
        'common_ids':common_ids,'sources':SOURCES,'script':e.t.file_ref(__file__)}
    for path,sha in SOURCES.items():
        if e.base.sha256_file(path)!=sha:raise ValueError('Input changed during diagnosis')
    e.t.write_json(OUT/'diagnosis.json',report)
    print(e.base.canonical_json({k:report[k] for k in ('population','baseline','oracle','neighbor_summary')}))


if __name__=='__main__':
    with threadpool_limits(limits=2):run()
