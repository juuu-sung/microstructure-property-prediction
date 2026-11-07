"""User-run frozen ResNet18 early moments/covariance ablation; local weights only.

prepare/validate: read-only audits and protocol, no competition fit/forward.
check: pretrained model on synthetic input only. extract: train500, no labels.
evaluate: fixed three representations, nested Ridge, no neural weight updates.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import os
from pathlib import Path
import platform
import time
from types import SimpleNamespace
import numpy as np

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
if __package__:
    from . import extract_resnet_d4 as original
    from . import evaluate_boundary_tabpfn as archive
else:
    import extract_resnet_d4 as original
    import evaluate_boundary_tabpfn as archive

t, base, core = archive.t, original.base, original.core
ROOT = base.ROOT
OUT = ROOT/'reports/resnet_texture'
OLD = ROOT/'reports/frozen_resnet18/features.npz'
SEEDS = (42, 2026)
REPS = ('original1792', 'early_moments2048', 'early_cov3104')
WIDTHS = dict(zip(REPS, (1792, 2048, 3104)))
ALPHAS = (100., 1000., 10000.)
BLENDS = ('retuned_control_blend', 'early_moments_control_blend', 'texture_replaced_blend')
PRIMARY = BLENDS[-1]
PROJECTION_SEEDS = (271828, 271829)
EPS = 1e-12
PROTOCOL = {
    'version': 2, 'seeds': list(SEEDS), 'representations': WIDTHS,
    'backbone': {'repo': base.HF_REPO, 'revision': base.DEFAULT_REVISION,
                 'indices': [0,1,2,3,4], 'channels': [64,64,128,256,512],
                 'spatial_sizes': [128,64,32,16,8], 'requires_grad': False, 'batchnorm': 'eval'},
    'input': 'Original256x256 grayscale repeated RGB, original ImageNet normalization; '
             'FP32, no AMP, crop, resize, denoise, D4 or augmentation.',
    'original1792': 'Use unchanged cached layer2/3/4 mean/std after all500 identity re-extraction check.',
    'identity_batch_policy': 'Match the original cache extraction batch size and ordered batch boundaries, '
        'including its unpadded final partial batch. Other extraction batch sizes are rejected before model load. '
        'Keep original identity atol=1e-5 and rtol=1e-6; do not relax tolerances for batch-dependent FP32 differences.',
    'early256': 'Concat act1 mean/std then layer1 mean/std; population std;64 channels each.',
    'covariance1056': 'For act1 and layer1 separately: fixed64x32 orthonormal projection; '
        'center projected channels per image over spatial locations; C=Z@Z.T/(H*W). '
        'Elementwise sign(C)*sqrt(abs(C)); upper triangle row-major including diagonal; '
        'offdiagonal multiply sqrt(2); each528-vector divided by max(L2norm,1e-12). '
        'Concatenate act1 then layer1. Centering and normalization are per image, not fitted to data.',
    'projection': {'method': 'numpy default_rng standard_normal(64,32), reduced QR; '
                   'Q column signs chosen to make diag(R)>=0; FP32 storage, fixed before extraction',
                   'seeds': list(PROJECTION_SEEDS), 'data_or_targets_used': False},
    'regression': {'alphas': list(ALPHAS), 'inner_folds': 3, 'outer_folds': 5,
        'inner_seed': 'seed+outer_fold', 'stratification': 'outer training target rank deciles',
        'selection': 'Pooled inner MSE separately per fixed representation; exact ties favor higher alpha. '
            'No joint representation choice. X/y StandardScaler fitted inside every training fold.',
        'fits_per_seed': 150, 'threads': 2},
    'blend': 'Current=hand185*0.125+originalResNet*0.125+DINO_D4*0.25+archived_nested_TabPFN*0.5. '
        'Each new blend=current+0.125*(newResNet-originalResNet). No other component changes, tuning or clipping.',
    'primary': PRIMARY, 'controls': list(BLENDS[:2]),
    'decision': 'Both seeds must improve vs current AND both controls. Expansion priority additionally '
        'mean gain vs current>=0.30. Review standalone contribution, middle80%, tails, noise and fixed common16 failures. '
        'Not automatic adoption; no posthoc feature, projection, alpha-grid or blend-weight search.',
    'priority_mean_gain': .30,
    'scope': 'Train500 only. No test, final500 supervised fit, submission or neural weight updates. '
        'Same repeatedly studied500, not independent holdout or significance. Current submission preserved.',
    'research': 'Inspired by channel interactions in Bilinear CNN (https://arxiv.org/abs/1504.07889); '
        'this is a bounded centered-covariance variant, not a reproduction or a validated grain measurement.',
}


def projections():
    result=[]
    for seed in PROJECTION_SEEDS:
        q,r=np.linalg.qr(np.random.default_rng(seed).standard_normal((64,32)), mode='reduced')
        q*=np.where(np.diag(r)<0,-1.,1.)[None,:]
        result.append(q.astype(np.float32))
    return np.stack(result)


def sources():
    return {Path(p).name:t.file_ref(p) for p in (__file__, original.__file__, base.__file__,
            core.__file__, archive.__file__, t.__file__)}


def environment():
    return {'python':platform.python_version(), **{n:importlib.metadata.version(n) for n in
        ('numpy','scipy','scikit-learn','threadpoolctl','torch','torchvision','timm','Pillow','safetensors','huggingface-hub')}}


def load_state():
    data,tab_features,old,old_plan=t.validate(t.OUT)
    ids,raw,metadata=original.read_original(np,OLD,ROOT/'open/train.csv')
    np.testing.assert_array_equal(ids,data.ids)
    inventory=base.training_inventory(ROOT/'open')
    for name in ('train_csv_sha256','image_manifest_sha256'):
        if metadata[name]!=inventory[name]:raise ValueError('Original images/labels changed')
    checkpoints=base.cached_checkpoint_files(base.DEFAULT_REVISION)
    names={c['filename'] for c in checkpoints}
    if 'config.json' not in names or not names.intersection({'model.safetensors','pytorch_model.bin'}):
        raise ValueError('Pinned local ResNet weights missing; no downloads')
    refs={s:archive.archived_reference(data,tab_features,old[s],old_plan,s) for s in SEEDS}
    for seed in SEEDS:
        refs[seed]['resnet']=old[seed]['resnet']
        np.testing.assert_allclose(refs[seed]['base'],.25*old[seed]['hand']+.25*old[seed]['resnet']+
                                   .5*old[seed]['dino'],rtol=0,atol=1e-10)
    audit_path=ROOT/'reports/model_portfolio_audit_20261011/numeric_audit.json'
    audit=t.read_json(audit_path)
    # The fixed failure group is diagnostic only, never a model input or selection rule.
    common=audit['cross_seed']['common_worst20_ids']
    if len(set(common))!=16 or not set(common).issubset(set(ids)):raise ValueError('Common16 audit differs')
    for p,sha in audit['sources_sha256'].items():
        if base.sha256_file(ROOT/p)!=sha:raise ValueError('Portfolio audit source changed: '+p)
    frozen={'protocol':PROTOCOL,'sources':sources(),'environment':environment(),
        'old_plan':t.file_ref(t.OUT/'protocol.json'),'old_resnet':original.reference(OLD),
        'reference_batch_size':metadata['extraction_config']['batch_size'],
        'original_state_sha256':metadata['feature_state_sha256'], 'checkpoints':checkpoints,
        'train_csv_sha256':inventory['train_csv_sha256'],'image_manifest_sha256':inventory['image_manifest_sha256'],
        'ids_sha256':base.sha256_array(ids),'target_sha256':base.sha256_array(data.y),
        'noise_sha256':base.sha256_array(data.noise),'projection_sha256':base.sha256_array(projections()),
        'audit':t.file_ref(audit_path),'common_failure_ids':common,
        'references':{str(s):{'archive':refs[s]['archive'],
             'arrays':{k:base.sha256_array(refs[s][k]) for k in ('folds','current','base','resnet')}} for s in SEEDS},
        'submissions':[t.file_ref(ROOT/p) for p in ('reports/final_ensemble/submission/submission.csv',
                                                'reports/final_tabpfn_blend/submission/submission.csv')]}
    return data,raw,refs,frozen,inventory


def prepare(folder):
    *_,frozen,inventory=load_state()
    path=folder/'protocol.json'
    if path.exists():
        if t.read_json(path)['frozen']!=frozen:raise ValueError('Frozen protocol differs; preserve previous experiment')
        print('Existing preparation matches; no fitting or extraction');return
    if folder.exists() and any(folder.iterdir()):raise ValueError('Nonempty unprepared output directory')
    t.write_json(path,{'frozen':frozen,'created_unix':time.time(),'competition_fits':0,'competition_forwards':0})
    print(f'Prepared original1792 / early2048 / covariance3104;300 Ridge fits total planned: {path}')


def validate(folder):
    data,raw,refs,frozen,inventory=load_state();plan=t.read_json(folder/'protocol.json')
    if plan['frozen']!=frozen or plan['competition_fits']!=0 or plan['competition_forwards']!=0:
        raise ValueError('Frozen sources, inputs, runtime or settings differ')
    return data,raw,refs,plan,inventory


def covariance_vector(torch,activation,projection):
    if activation.ndim!=4 or activation.shape[1]!=64 or projection.shape!=(64,32):
        raise ValueError('Expected64-channel map and fixed64x32 projection')
    z=projection.T@activation.flatten(2)
    # Algebraically equivalent centering; constant maps become exactly zero in FP32.
    z=z-z[:,:,:1]
    z=z-z.mean(dim=2,keepdim=True)
    c=(z@z.transpose(1,2))/z.shape[2]
    c=torch.sign(c)*torch.sqrt(torch.abs(c))
    ij=torch.triu_indices(32,32,device=c.device)
    v=c[:,ij[0],ij[1]]
    v=v*torch.where(ij[0]==ij[1],1.,2.**.5)
    return v/torch.linalg.vector_norm(v,dim=1,keepdim=True).clamp_min(EPS)


def pool_maps(torch,maps,projection):
    if len(maps)!=5 or [m.shape[1] for m in maps]!=[64,64,128,256,512]:
        raise ValueError('Expected all five ResNet maps')
    _,identity=base.pool_feature_maps(torch,maps[2:])
    early=torch.cat([v for m in maps[:2] for v in (m.mean((2,3)),m.std((2,3),correction=0))],dim=1)
    cov=torch.cat([covariance_vector(torch,m,p) for m,p in zip(maps[:2],projection)],dim=1)
    if early.shape[1]!=256 or cov.shape[1]!=1056 or not torch.isfinite(cov).all():
        raise ValueError('Invalid early/covariance features')
    return identity,early,cov


def build_model(torch,timm,resolve_config):
    model=timm.create_model(f'hf-hub:{base.HF_REPO}@{base.DEFAULT_REVISION}',
                           pretrained=True,features_only=True,out_indices=(0,1,2,3,4))
    config=resolve_config(model)
    if (model.feature_info.channels()!=[64,64,128,256,512] or
        list(config['mean'])!=base.EXPECTED_MEAN or list(config['std'])!=base.EXPECTED_STD):
        raise ValueError('Architecture or normalization differs')
    model.requires_grad_(False);model.eval()
    return model


def model_runtime(device,batch_size):
    _,torch,timm,resolve_config,versions=base.import_runtime()
    args=SimpleNamespace(seed=42,threads=2,device=device)
    resolved,info=base.configure_runtime(torch,np,args)
    model=build_model(torch,timm,resolve_config).to(device=resolved,dtype=torch.float32)
    if any(m.training for m in model.modules()) or any(p.requires_grad for p in model.parameters()):
        raise ValueError('Backbone and BatchNorm must be frozen')
    p=torch.from_numpy(projections()).to(resolved)
    return torch,resolved,model,p,{'versions':versions,'device':info,'batch_size':batch_size,'amp':False}


def check(folder,device):
    *_,plan,inventory=validate(folder)
    batch_size=plan['frozen']['reference_batch_size']
    torch,resolved,model,p,runtime=model_runtime(device,batch_size)
    expected=plan['frozen']['original_state_sha256']
    if base.feature_state_hash(model)!=expected:raise ValueError('Pretrained state differs from archived extraction')
    with torch.inference_mode():
        parts=pool_maps(torch,model(torch.randn(batch_size,3,256,256,device=resolved)),p)
    if base.feature_state_hash(model)!=expected:raise ValueError('Frozen model state changed')
    print(t.common.canonical_json({'synthetic_forward_shapes':[list(x.shape) for x in parts],
          'pretrained_state_matched':True,'competition_forwards':0,'runtime':runtime}))


def load_features(folder,data,raw,plan):
    path=folder/'features/features.npz'
    if not path.is_file():
        raise ValueError('Feature extraction has not completed: '+str(path)+'. '
            'Run extract --device cuda --batch-size '+str(plan['frozen']['reference_batch_size'])+
            ' successfully before evaluate; no regression was started.')
    with np.load(path,allow_pickle=False) as a:
        if set(a.files)!={'ids','early256','covariance1056','projection','metadata_json'}:raise ValueError('Cache schema differs')
        np.testing.assert_array_equal(a['ids'],data.ids)
        np.testing.assert_array_equal(a['projection'],projections())
        meta=t.read_json(path.with_name('metadata.json'))
        if a['metadata_json'].shape!=() or a['metadata_json'].item()!=base.canonical_json(meta):
            raise ValueError('Cache metadata mismatch')
        if meta['plan']!=t.digest(plan['frozen']) or meta['state_sha256']!=plan['frozen']['original_state_sha256']:
            raise ValueError('Cache plan/model mismatch')
        if meta['test_accessed'] is not False or meta['labels_saved'] is not False or meta['identity_matched'] is not True:
            raise ValueError('Cache scope/identity check failed')
        if (meta['runtime']['batch_size']!=plan['frozen']['reference_batch_size'] or
                meta['identity_atol']!=original.MATCH_ATOL or meta['identity_rtol']!=original.MATCH_RTOL):
            raise ValueError('Cache identity batch size or tolerance differs from original extraction')
        for n,w in (('early256',256),('covariance1056',1056)):
            if a[n].dtype!=np.float32 or a[n].shape!=(len(data.ids),w) or not np.isfinite(a[n]).all():
                raise ValueError('Invalid features: '+n)
        for n in ('ids','early256','covariance1056','projection'):
            if base.sha256_array(a[n])!=meta['arrays'][n]:raise ValueError('Cache integrity failure: '+n)
        early=np.c_[raw,a['early256']];cov=np.c_[early,a['covariance1056']]
    return dict(zip(REPS,(raw,early,cov))),meta


def require_reference_batch(batch_size,plan):
    expected=plan['frozen']['reference_batch_size']
    if type(expected) is not int or expected<1:raise ValueError('Invalid reference extraction batch size')
    if batch_size!=expected:
        raise ValueError(f'Original cache used batch_size={expected}; requested {batch_size}. '
            f'Use --batch-size {expected} to preserve its FP32 anchor. '
            'Batch-size changes can change numerical results; tolerances will not be relaxed.')


def identity_check(actual,previous,ids):
    if actual.shape!=previous.shape or actual.ndim!=2 or len(ids)!=len(actual):
        raise ValueError('Original1792 anchor shape/ID count differs')
    if not np.isfinite(actual).all() or not np.isfinite(previous).all():
        raise ValueError('Original1792 anchor contains nonfinite values')
    delta=np.abs(actual-previous)
    bad=~np.isclose(actual,previous,atol=original.MATCH_ATOL,rtol=original.MATCH_RTOL)
    if bad.any():
        row,col=np.argwhere(bad)[0]
        raise ValueError(f'Original1792 anchor failed at {ids[row]}, column={col}: '
            f'actual={actual[row,col]:.10g}, reference={previous[row,col]:.10g}, '
            f'abs_difference={delta[row,col]:.10g}, mismatched_values={int(bad.sum())}; '
            f'atol={original.MATCH_ATOL}, rtol={original.MATCH_RTOL}; no new cache saved. '
            'Check input, device/runtime and original batch size; do not bypass this check.')
    return float(delta.max())


def extract(folder,device,batch_size):
    data,raw,_,plan,inventory=validate(folder)
    require_reference_batch(batch_size,plan)
    path=folder/'features/features.npz'
    if path.exists():
        load_features(folder,data,raw,plan);print('Completed feature cache verified; no re-extraction');return
    torch,resolved,model,p,runtime=model_runtime(device,batch_size)
    state=base.feature_state_hash(model)
    if state!=plan['frozen']['original_state_sha256']:raise ValueError('Checkpoint state mismatch')
    ds=base.TrainImageDataset(ROOT/'open',data.ids.tolist(),base.EXPECTED_MEAN,base.EXPECTED_STD)
    loader=torch.utils.data.DataLoader(ds,batch_size=batch_size,shuffle=False,num_workers=0,
                                      pin_memory=resolved.type=='cuda',drop_last=False)
    early=[];cov=[];count=0;max_difference=0.;start=time.perf_counter()
    with torch.inference_mode():
        for step,x in enumerate(loader):
            identity,e,c=pool_maps(torch,model(x.to(resolved,dtype=torch.float32)),p)
            identity=identity.cpu().numpy();previous=raw[count:count+len(x)]
            difference=identity_check(identity,previous,data.ids[count:count+len(x)])
            max_difference=max(max_difference,difference)
            early.append(e.cpu().numpy());cov.append(c.cpu().numpy());count+=len(x)
            if step%5==0 or count==len(data.ids):
                print(f'Extracted {count}/{len(data.ids)} early moments/covariance; elapsed={time.perf_counter()-start:.1f}s',flush=True)
    base.ensure_inputs_unchanged(ROOT/'open',inventory)
    if base.feature_state_hash(model)!=state or sources()!=plan['frozen']['sources'] or environment()!=plan['frozen']['environment']:
        raise ValueError('Model, sources or runtime changed during extraction')
    if original.reference(OLD)!=plan['frozen']['old_resnet']:raise ValueError('Original cache changed')
    arrays={'ids':data.ids,'early256':np.concatenate(early),'covariance1056':np.concatenate(cov),'projection':projections()}
    meta={'plan':t.digest(plan['frozen']),'state_sha256':state,'runtime':runtime,'identity_matched':True,
        'identity_max_difference':max_difference,'identity_atol':original.MATCH_ATOL,'identity_rtol':original.MATCH_RTOL,
        'arrays':{k:base.sha256_array(v) for k,v in arrays.items()},'test_accessed':False,'labels_saved':False,
        'seconds':time.perf_counter()-start}
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.npz.tmp')
    with tmp.open('wb') as f:np.savez_compressed(f,**arrays,metadata_json=np.asarray(base.canonical_json(meta)))
    t.write_json(path.with_name('metadata.json'),meta);os.replace(tmp,path)
    load_features(folder,data,raw,plan)
    print(f'Saved {path}; early=256; covariance=1056; total=3104; max original difference={max_difference:.3g}')


def fit_fold(features,y,folds,seed,fold,*,factory=None):
    """Only outer-training labels enter splits, scalers, selection and estimators."""
    from sklearn.model_selection import StratifiedKFold
    factory=factory or (lambda a,rep:core.make_estimator(a))
    ti=np.flatnonzero(folds!=fold);vi=np.flatnonzero(folds==fold);yt=y[ti]
    if len(ti)<30 or not len(vi):raise ValueError('Insufficient fold sizes')
    splits=list(StratifiedKFold(3,shuffle=True,random_state=seed+fold).split(yt,core.rank_groups(yt,10)))
    result={'validation_indices':vi.tolist(),'predictions':{},'selection':{},'candidates':{}}
    for rep in REPS:
        x=features[rep].astype(np.float64);candidates=[]
        for alpha in ALPHAS:
            inner=np.full(len(ti),np.nan)
            for it,iv in splits:
                model=factory(alpha,rep);model.fit(x[ti[it]],yt[it]);inner[iv]=model.predict(x[ti[iv]])
            if not np.isfinite(inner).all():raise ValueError('Nonfinite inner prediction')
            candidates.append({'alpha':alpha,'inner_mse':float(np.mean((inner-yt)**2))})
        chosen=min(candidates,key=lambda v:(v['inner_mse'],-v['alpha']))
        model=factory(chosen['alpha'],rep);model.fit(x[ti],yt);pred=np.asarray(model.predict(x[vi]),dtype=float)
        if pred.shape!=(len(vi),) or not np.isfinite(pred).all():raise ValueError('Invalid outer predictions')
        result['predictions'][rep]=pred.tolist();result['selection'][rep]=chosen;result['candidates'][rep]=candidates
    return result


def fold_cell(folder,binding,features,y,folds,seed,fold,*,runner=fit_fold):
    path=folder/f'fold{fold}.json';bound={**binding,'fold':fold}
    if path.exists():
        record=t.read_json(path)
        if record['binding']!=bound or record['result_sha256']!=t.digest(record['result']):raise ValueError('Changed fold cache')
        result=record['result']
    else:
        result=runner(features,y,folds,seed,fold)
        t.write_json(path,{'binding':bound,'result':result,'result_sha256':t.digest(result)})
    vi=np.flatnonzero(folds==fold)
    if result['validation_indices']!=vi.tolist() or set(result['predictions'])!=set(REPS):raise ValueError('Fold cache schema differs')
    for rep,p in result['predictions'].items():
        if np.asarray(p).shape!=(len(vi),) or not np.isfinite(p).all():raise ValueError('Bad cached predictions')
        choice=min(result['candidates'][rep],key=lambda v:(v['inner_mse'],-v['alpha']))
        if choice!=result['selection'][rep] or [c['alpha'] for c in result['candidates'][rep]]!=list(ALPHAS):
            raise ValueError('Cached inner selection differs')
    return result


def compose_predictions(ref,predictions):
    output={'current':ref['current'],'original_resnet':ref['resnet'],**predictions}
    output.update({blend:ref['current']+.125*(predictions[rep]-ref['resnet']) for rep,blend in zip(REPS,BLENDS)})
    return output


def save_results(folder,data,ref,predictions,choices,binding,plan):
    output=compose_predictions(ref,predictions)
    metrics={k:core.metrics(data.y,p) for k,p in output.items()}
    rows=[{'ID':str(i),'hardness':float(y),'fold':int(ref['folds'][j]),
           **{k:float(p[j]) for k,p in output.items()}} for j,(i,y) in enumerate(zip(data.ids,data.y))]
    t.write_csv(folder/'oof.csv',rows)
    ranks=core.rank_groups(data.y,10);noise=core.rank_groups(data.noise,4)
    masks={'all':np.ones(len(data.y),bool),'low10':ranks==0,'middle80':(ranks>0)&(ranks<9),'high10':ranks==9,
           'common16':np.isin(data.ids,plan['frozen']['common_failure_ids'])}
    masks.update({f'fold{f}':ref['folds']==f for f in range(5)})
    masks.update({f'noiseQ{q+1}':noise==q for q in range(4)})
    diag=[]
    for name,p in output.items():
        for group,mask in masks.items():
            if mask.any():diag.append({'model':name,'group':group,**core.metrics(data.y[mask],p[mask])})
    t.write_csv(folder/'diagnostics.csv',diag)
    errors=[]
    for i in np.argsort(-(ref['current']-data.y)**2,kind='stable'):
        error={'ID':str(data.ids[i]),'hardness':float(data.y[i]),'noise':float(data.noise[i]),
               'common16':bool(masks['common16'][i])}
        for name in ('current',*BLENDS):error[name+'_error']=float(output[name][i]-data.y[i])
        for name in ('current',*BLENDS[:2]):
            error['gain_squared_vs_'+name]=float((output[name][i]-data.y[i])**2-(output[PRIMARY][i]-data.y[i])**2)
        errors.append(error)
    t.write_csv(folder/'per_image_errors.csv',errors)
    summary={'binding':binding,'metrics':metrics,'selection':choices,'primary':PRIMARY,
        'competition_ridge_fits':150,'neural_weight_updates':0,'test_accessed':False,'independent_holdout':False}
    t.write_json(folder/'summary.json',summary)
    lines=[f"# ResNet early texture · seed{binding['seed']}",'',
        '| Model | OOF RMSE | Gain vs current |','|---|---:|---:|']
    for n,m in metrics.items():lines.append(f"| {n} | {m['rmse']:.6f} | {metrics['current']['rmse']-m['rmse']:+.6f} |")
    lines+=['','Primary is fixed covariance3104 replacement; early2048 and retuned1792 are separate controls.',
        'Only ResNet12.5% replaced. No representation choice, blend tuning, test inference or independent holdout.',
        'Review diagnostics.csv and per_image_errors.csv before adoption. Both seeds must improve vs all three references.',
        'Expansion priority: mean gain vs current>=0.30; this is a practical rule, not statistical significance.','',
        'Built with PriorLabs-TabPFN (unchanged archived component).']
    (folder/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    names=['oof.csv','summary.json','diagnostics.csv','per_image_errors.csv','report.md',*[f'fold{f}.json' for f in range(5)]]
    t.write_json(folder/'completed.json',{'binding':binding,'files':{n:base.sha256_file(folder/n) for n in names}})
    for name in BLENDS:
        print(f"{name}: RMSE={metrics[name]['rmse']:.6f}; gain vs current={metrics['current']['rmse']-metrics[name]['rmse']:+.6f}")
    print(f'Report: {folder/"report.md"}')


def completed(folder,binding):
    path=folder/'completed.json'
    if not path.exists():return False
    m=t.read_json(path)
    names={'oof.csv','summary.json','diagnostics.csv','per_image_errors.csv','report.md',*[f'fold{f}.json' for f in range(5)]}
    if m['binding']!=binding or set(m['files'])!=names:raise ValueError('Completion binding/artifact set differs')
    for n,sha in m['files'].items():
        if base.sha256_file(folder/n)!=sha:raise ValueError('Completed artifact changed: '+n)
    return True


def evaluation_binding(folder,seed,plan):
    return {'seed':seed,'plan':t.digest(plan['frozen']),'features':t.file_ref(folder/'features/features.npz'),
            'metadata':t.file_ref(folder/'features/metadata.json'),'environment':environment(),'threads':2}


def compare(folder,plan):
    rows=[]
    for s in SEEDS:
        sub=folder/f'seed{s}';binding=evaluation_binding(folder,s,plan)
        if not completed(sub,binding):return
        m=t.read_json(sub/'summary.json')['metrics']
        rows.append({'seed':s,'current_rmse':m['current']['rmse'],'candidate_rmse':m[PRIMARY]['rmse'],
                     **{'gain_vs_'+name:m[name]['rmse']-m[PRIMARY]['rmse'] for name in ('current',*BLENDS[:2])}})
    mean=float(np.mean([r['gain_vs_current'] for r in rows]))
    positive=all(r['gain_vs_'+n]>0 for r in rows for n in ('current',*BLENDS[:2]))
    result={'primary':PRIMARY,'per_seed':rows,'mean_gain_vs_current':mean,'all_seeds_improve_all_controls':positive,
        'passes_expansion_priority':positive and mean>=PROTOCOL['priority_mean_gain'],
        'additional_review_required':'Middle80%, tails, noise, common16 and standalone representation contribution; no automatic adoption.',
        'aggregation':'Mean per-seed RMSE differences; same500, no prediction averaging or independent inference.',
        'protocol_sha256':t.digest(plan['frozen'])}
    t.write_json(folder/'comparison.json',result);print(base.canonical_json(result))


def evaluate(folder,seed):
    from threadpoolctl import threadpool_limits
    data,raw,refs,plan,_=validate(folder)
    features,_=load_features(folder,data,raw,plan);sub=folder/f'seed{seed}'
    binding=evaluation_binding(folder,seed,plan)
    if completed(sub,binding):
        print(f'Completed seed{seed} verified; no refitting');compare(folder,plan);return
    ref=refs[seed];predictions={n:np.full(len(data.y),np.nan) for n in REPS};choices=[]
    with threadpool_limits(limits=2):
        for f in range(5):
            cell=fold_cell(sub,binding,features,data.y,ref['folds'],seed,f)
            if cell['selection'][REPS[0]]['alpha']==1000.:
                np.testing.assert_allclose(cell['predictions'][REPS[0]],ref['resnet'][cell['validation_indices']],
                                           rtol=0,atol=1e-8,err_msg='Original alpha1000 ResNet OOF anchor differs')
            for n in REPS:predictions[n][cell['validation_indices']]=cell['predictions'][n]
            choices.append(cell['selection'])
            print(f"Outer{f}: "+'; '.join(f"{n} alpha={cell['selection'][n]['alpha']:g}" for n in REPS)+
                  '; fixed representations, outer predictions stored',flush=True)
    if any(not np.isfinite(p).all() for p in predictions.values()):raise ValueError('OOF incomplete')
    # No changed inputs/source/cache can silently finalize an interrupted or concurrently edited run.
    validate(folder)
    if evaluation_binding(folder,seed,plan)!=binding:raise ValueError('Feature/runtime binding changed during evaluation')
    save_results(sub,data,ref,predictions,choices,binding,plan);compare(folder,plan)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('prepare','validate','check','extract','evaluate','compare'))
    p.add_argument('--output-dir',type=Path,default=OUT)
    p.add_argument('--device',default='cuda');p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--seed',type=int,choices=SEEDS,default=42)
    a=p.parse_args(argv);folder=a.output_dir.resolve()
    if a.batch_size<1:p.error('Batch size must be positive')
    if folder==ROOT.resolve() or folder.is_relative_to((ROOT/'open').resolve()):p.error('Output overlaps protected input directory')
    if a.command=='prepare':prepare(folder)
    elif a.command=='validate':
        data,raw,_,plan,_=validate(folder)
        if (folder/'features/features.npz').exists():load_features(folder,data,raw,plan)
        print('Protocol, inputs, original caches, archived predictions and submissions verified; no fitting/forward')
    elif a.command=='check':check(folder,a.device)
    elif a.command=='extract':extract(folder,a.device,a.batch_size)
    elif a.command=='evaluate':evaluate(folder,a.seed)
    else:
        data,raw,_,plan,_=validate(folder);load_features(folder,data,raw,plan);compare(folder,plan)


if __name__=='__main__':main()
