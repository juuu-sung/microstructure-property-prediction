"""Read-only model-result audit; no model fitting, feature extraction or inference.

Groups, correlations and calibration slopes are retrospective diagnostics only.
Repeated seeds are not independent samples. Existing experiment files are read-only.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/model_portfolio_audit_20261011'
SOURCES = {}


def record(path):
    path = ROOT / path
    SOURCES[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return path


def read_csv(path):
    frame = pd.read_csv(record(path))
    assert frame.ID.is_unique and len(frame) == 500
    return frame.set_index('ID').sort_index()


def read_json(path):
    return json.loads(record(path).read_text(encoding='utf-8'))


def metric(y, p):
    e = p-y
    return {'n': len(y), 'rmse': float(np.sqrt(np.mean(e**2))),
            'mae': float(np.mean(np.abs(e))), 'bias_pred_minus_y': float(e.mean()),
            'sse': float(e@e), 'y_std': float(y.std()), 'prediction_std': float(p.std()),
            'min_prediction': float(p.min()), 'max_prediction': float(p.max())}


def run():
    audit = read_csv('reports/audit_train_features.csv')
    y = audit.hardness.to_numpy()
    ids = audit.index
    qlow, qhigh = np.quantile(y, [.1, .9])
    masks = {'low10': y <= qlow, 'middle80': (y > qlow) & (y < qhigh), 'high10': y >= qhigh}
    for name in ('haar_hh_noise_proxy', 'variance_minus_noise_proxy'):
        bins = pd.qcut(audit[name], 4, labels=False).to_numpy()
        for i in range(4):
            masks[f'{name}_q{i+1}'] = bins == i
    current_predictions, current_worst = [], []
    seed_results = []
    for seed in [42, 2026, 31415, 27182, 16180]:
        base = 'tabpfn_v2' if seed in (42, 2026) else 'tabpfn_v2_confirmation'
        df = read_csv(f'reports/{base}/seed{seed}/oof.csv')
        assert df.index.equals(ids) and np.allclose(df.hardness, y, atol=1e-12, rtol=0)
        p = df.current_tabpfn_equal.to_numpy()
        assert np.allclose(p, .5*(df.current+df.tabpfn_joint_nested), atol=1e-12, rtol=0)
        e = p-y
        rank = np.argsort(e**2)[::-1]
        current_predictions.append(p)
        current_worst.append(set(rank[:20]))
        cov = float(np.mean((p-p.mean())*(y-y.mean())))
        # Analytic descriptive slope; not fitted/applied calibration predictions.
        item = {'seed': seed, 'current_tabpfn_equal': metric(y, p),
                'previous_ensemble': metric(y, df.current.to_numpy()),
                'groups': {name: metric(y[m], p[m]) for name,m in masks.items()},
                'worst20_sse_share': float(np.sum(e[rank[:20]]**2)/np.sum(e**2)),
                'worst50_sse_share': float(np.sum(e[rank[:50]]**2)/np.sum(e**2)),
                'descriptive_y_on_prediction_slope': cov/float(p.var()),
                'descriptive_best_affine_rmse_not_validation': float(np.sqrt(y.var()-cov**2/p.var())),
                'prediction_residual_correlation': float(np.corrcoef(p,e)[0,1]),
                'tail20_sse_share': float(np.sum(e[masks['low10']|masks['high10']]**2)/np.sum(e**2))}
        seed_results.append(item)
    matrix = np.stack(current_predictions, axis=1)
    errors = matrix-y[:,None]
    common = set.intersection(*current_worst)
    ordered = np.argsort(np.mean(errors**2, axis=1))[::-1]
    details = pd.DataFrame({'ID': ids, 'hardness': y,
                           'mean_absolute_error_across_seeds': np.mean(abs(errors),axis=1),
                           'mean_squared_error_across_seeds': np.mean(errors**2,axis=1),
                           'prediction_sd_across_seeds': matrix.std(axis=1),
                           'worst20_seed_count': [sum(i in s for s in current_worst) for i in range(500)],
                           'noise': audit.haar_hh_noise_proxy.to_numpy(),
                           'structural_contrast_proxy': audit.variance_minus_noise_proxy.to_numpy()})
    for j, seed in enumerate([42, 2026, 31415, 27182, 16180]):
        details[f'prediction_seed{seed}'] = matrix[:,j]
    residual_families, model_scores = {}, []
    mappings = [('frozen_dinov2_small','joint_nested'),('dinov2_d4','d4_mean_cls_patch_mean_std_nested'),
                ('frozen_resnet18','joint_nested'),('feature_fusion','joint_nested'),
                ('dinov2_kernel_lower','joint_nested'),('dinov2_spatial','global_plus_2x2_nested'),
                ('convnext_full','convnext_nested'),('local_structure','hand_plus_structure_nested'),
                ('tabpfn_v2','tabpfn_joint_nested'),('tabpfn_v2','current_tabpfn_equal')]
    for seed in [42, 2026]:
        residuals = {}
        for folder,col in mappings:
            f = read_csv(f'reports/{folder}/seed{seed}/oof.csv')
            assert f.index.equals(ids) and np.allclose(f.hardness,y,atol=1e-12,rtol=0)
            p = f[col].to_numpy()
            label = f'{folder}:{col}'
            model_scores.append({'seed':seed,'model':label, **metric(y,p)})
            residuals[label] = p-y
        residual_families[str(seed)] = pd.DataFrame(residuals).corr().to_dict()
    conv = read_json('reports/convnext_full/review_results.json')
    curves = []
    for seed in (42,2026):
        summ = read_json(f'reports/convnext_full/seed{seed}/summary.json')
        for fold in range(5):
            for profile in ('none','d4','outer'):
                folder=f'fold{fold}_outer' if profile=='outer' else f'fold{fold}_inner_{profile}'
                curve=read_json(f'reports/convnext_full/seed{seed}/{folder}/curve.json')
                best=min(curve,key=lambda r:r['valid_rmse']) if profile!='outer' else curve[-1]
                curves.append({'seed':seed,'fold':fold,'profile':profile,'epochs':len(curve),
                               'best_epoch':best['epoch'],'train_at_best':best['train_rmse'],
                               'best_valid':best['valid_rmse'],'last_train':curve[-1]['train_rmse'],
                               'last_valid':curve[-1]['valid_rmse'],
                               'steps':sum(r['optimizer_steps'] for r in curve),
                               'amp_skips':sum(r['amp_skipped_steps'] for r in curve),
                               'skips_after_epoch1':sum(r['amp_skipped_steps'] for r in curve[1:])})
    inner=[r for r in curves if r['profile']!='outer']
    report={'scope':'Saved train-only OOF and training curves. No model fitting, new extraction, GPU inference or test predictions.',
            'limitations':['Post-hoc analysis of extensively reused500 images; no independent inference.',
                           'Calibration slope is descriptive, not a validated transformation.',
                           'Tail groups are defined by targets and cannot route test predictions.',
                           'Common-failure selection uses errors and is diagnostic only.'],
            'per_seed':seed_results,'cross_seed':{
                'common_worst20_count':len(common),'common_worst20_ids':ids[sorted(common)].tolist(),
                'mean_prediction_sd':float(matrix.std(axis=1).mean()),
                'median_prediction_sd':float(np.median(matrix.std(axis=1))),
                'common_worst20_mean_sse_share':float(np.sum(errors[sorted(common)]**2)/np.sum(errors**2)),
                'error_correlation':np.corrcoef(errors.T).tolist(),
                'top12':details.iloc[ordered[:12]].to_dict('records')},
            'model_scores':model_scores,'model_error_correlations':residual_families,
            'convnext':{'curves':curves,'sanity':conv['sanity'],
                'inner_last_train_range':[min(r['last_train'] for r in inner),max(r['last_train'] for r in inner)],
                'best_epoch_range':[min(r['best_epoch'] for r in inner),max(r['best_epoch'] for r in inner)],
                'steps_total':sum(r['steps'] for r in curves),
                'amp_skips_after_epoch1':sum(r['skips_after_epoch1'] for r in curves)},
            'sources_sha256':SOURCES}
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'numeric_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    details.iloc[ordered].to_csv(OUT/'failure_audit.csv',index=False)
    pd.DataFrame(model_scores).to_csv(OUT/'model_scores.csv',index=False)
    pd.DataFrame(curves).to_csv(OUT/'convnext_curves.csv',index=False)
    # Scientific diagnostic figure from existing predictions/curves only.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,2,figsize=(12,4.9),layout='constrained')
    p=matrix[:,0]
    points=axes[0].scatter(y,p,c=audit.haar_hh_noise_proxy,cmap='viridis',s=19,alpha=.7)
    fig.colorbar(points,ax=axes[0],label='Noise proxy (not label noise)',fraction=.045,pad=.03)
    axes[0].plot([145,270],[145,270],color='#bd4949',ls='--',lw=1,label='Perfect prediction')
    for ident in ['TRAIN_000406','TRAIN_000466','TRAIN_000467']:
        i=ids.get_loc(ident)
        axes[0].annotate(ident[-6:],(y[i],p[i]),xytext=(5,7),textcoords='offset points',fontsize=8)
    axes[0].set(xlim=(145,270),ylim=(145,270),xlabel='Observed hardness',ylabel='Out-of-fold prediction',
                title='Current 50:50 blend | seed42 | RMSE 13.610')
    axes[0].legend(loc='upper left',frameon=False,fontsize=9)
    curve=read_json('reports/convnext_full/seed42/fold0_inner_none/curve.json')
    epochs=[r['epoch'] for r in curve]
    axes[1].plot(epochs,[r['train_rmse'] for r in curve],label='Train (320 images)',color='#238783')
    axes[1].plot(epochs,[r['valid_rmse'] for r in curve],label='Inner validation (80 images)',color='#cb5f47')
    axes[1].axvline(11,color='#777777',ls=':',lw=1,label='Selected epoch 11')
    axes[1].set(xlabel='Epoch',ylabel='RMSE',title='ConvNeXt | seed42 fold0 | no augmentation',ylim=(0,20))
    axes[1].legend(frameon=False,fontsize=9)
    for ax in axes:
        ax.spines[['top','right']].set_visible(False)
        ax.grid(alpha=.15)
    fig.suptitle('Retrospective diagnostics; same 500 images, no new training',fontsize=12)
    fig.savefig(OUT/'diagnostics.png',dpi=160)
    fig.savefig(OUT/'diagnostics.pdf')
    plt.close(fig)
    print(json.dumps({'current':[{'seed':r['seed'],**r['current_tabpfn_equal'],
            'tail_share':r['tail20_sse_share'],'top20_share':r['worst20_sse_share'],
            'calibration_slope':r['descriptive_y_on_prediction_slope']} for r in seed_results],
            'cross_seed':{k:v for k,v in report['cross_seed'].items() if k not in ('top12','error_correlation')},
            'convnext':{k:v for k,v in report['convnext'].items() if k!='curves'}},indent=2))


if __name__=='__main__':
    run()
