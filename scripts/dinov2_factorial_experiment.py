"""User-run Small/Large x266/518 frozen DINOv2 + nested Ridge/PCA64-SVR.

prepare/validate: local audits only; check: synthetic model forwards only.
extract: train500 images only; evaluate: cached-feature nested regression.
No test reads, encoder updates, all500 supervised fit or submission changes.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import time
from types import SimpleNamespace
import numpy as np
from threadpoolctl import threadpool_limits

if __package__:
    from . import dinov2_factorial_engine as eng
    from . import setup_dinov2_factorial as assets
    from . import extract_frozen_resnet as base
    from . import evaluate_frozen_dinov2 as old_dino
else:
    import dinov2_factorial_engine as eng
    import setup_dinov2_factorial as assets
    import extract_frozen_resnet as base
    import evaluate_frozen_dinov2 as old_dino

core = old_dino.core
ROOT = base.ROOT
OUT = ROOT/'reports/dinov2_factorial'
SEEDS = (42,2026)
PRIMARY = 'current_joint_equal'
CONTROL = 'current_small266_ridge_equal'
PROTOCOL = {
    'version':1,'seeds':list(SEEDS),'representations':list(eng.REPS),'weights':assets.MODELS,
    'input':{'raw':'native256 grayscale float32/255 repeatedRGB',
             '266':'normalize with fixed ImageNet mean/std, reflect5 each side; exact legacy transform',
             '518':'torch bicubic resize entire raw float image to518, align_corners=False, antialias=True, no clipping, then fixed ImageNet normalization',
             'augmentation':None,'crop':False,'denoise':False,'D4':False},
    'encoder':'Frozen,eval,FP32,no AMP/TF32; no register tokens. timm dynamic positional interpolation.',
    'pooling':'final normalized CLS + patch mean + patch population std, widths1152/3072',
    'anchor':{'small266':'all500 vs archived FP32 Small266, no replacing freshly extracted values',
              'rtol':1e-4,'atol':1e-4,'weights_state_hash':'must exactly match archived feature_state_sha256'},
    'regression':{'candidates':list(eng.CANDIDATES),'outer':5,'inner':3,
        'pca':{'components':64,'solver':'full','whiten':False},
        'scalers':'X/y StandardScaler fit only context; PCA fit only inner train / outer train during refit',
        'inner_seed':'seed+outer_fold','strata':'outer-train target rank deciles only',
        'selection':'pooled inner MSE; exact tie candidate-order then representation-order',
        'controls':'Each fixed representation/family gets its own inner-selected hyperparameters.',
        'joint':'All36 model/resolution/regressor candidates selected only by inner MSE; outer labels never accepted by fit_fold.',
        'pca_reuse':'One deterministic PCA per representation/inner split; six SVR candidates share it.',
        'cpu_threads':2},
    'blend':'current=.125hand185+.125ResNet+.25DINO_D4+.5TabPFN486; read verified archived OOF only. '
            'Every new blend=.5current+.5candidate, no clipping/weight search.',
    'primary':PRIMARY,'control':CONTROL,
    'decision':'Both seeds improve current/control; mean gain>=.5 is expansion priority only. '
               'Review standalone, common16 AND remaining484, middle80, tails, noise, folds, newworst20. No automatic adoption.',
    'limits':'Same repeatedly studied500, not independent holdout. Representation-level best outer score is exploratory, not joint selection.',
    'scope':'No encoder update, test access, final500 supervised fit, or submission. User executes extraction and fitting.',
    'sources':['https://arxiv.org/html/2501.18637v2','https://github.com/facebookresearch/dinov2'],
    'attribution':'DINOv2 (Meta), timm. Preserved current ensemble includes PriorLabs-TabPFN.'}


def digest(value): return hashlib.sha256(base.canonical_json(value).encode('utf-8')).hexdigest()
def read_json(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def file_ref(path): return {'path':str(Path(path).resolve()),'sha256':base.sha256_file(path)}


def write_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    os.replace(tmp,path)


def write_csv(path, rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with tmp.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    os.replace(tmp,path)


def environment():
    return {'python':platform.python_version(),**{n:importlib.metadata.version(n) for n in
        ('torch','torchvision','timm','safetensors','numpy','scipy','scikit-learn','Pillow','threadpoolctl')}}


def load_state():
    data=core.load_inputs(ROOT/'reports/frozen_dinov2_small/features.npz',ROOT/'open/train.csv',
        ROOT/'reports/folds.csv',ROOT/'reports/benchmark_oof_seed42.csv',ROOT/'reports/audit_train_features.csv',
        42,feature_dimensions=old_dino.FEATURE_DIMENSIONS)
    old_dino.validate_feature_metadata(data,ROOT/'reports/frozen_dinov2_small/features.npz',
                                      ROOT/'reports/frozen_dinov2_small/metadata.json')
    inventory=base.training_inventory(ROOT/'open')
    if inventory['image_manifest_sha256']!=data.metadata['image_manifest_sha256']:
        raise ValueError('Images no longer match original Small266 features')
    refs={};files={}
    upstream=read_json(ROOT/'reports/scattering_tabpfn/protocol.json')
    common_ids=upstream['frozen']['common_failure_ids']
    if len(set(common_ids))!=16 or not set(common_ids).issubset(set(data.ids)):
        raise ValueError('Historical common16 differs')
    for seed in SEEDS:
        folder=ROOT/f'reports/scattering_tabpfn/seed{seed}'
        done=read_json(folder/'completed_evaluation.json')
        if done['binding']['plan']!=digest(upstream['frozen']) or done['binding']['seed']!=seed:
            raise ValueError('Scattering completion scope mismatch')
        for name,sha in done['files'].items():
            path=(folder/name).resolve()
            if not path.is_relative_to(folder.resolve()) or base.sha256_file(path)!=sha:
                raise ValueError('Scattering archived file changed: '+name)
        rows=core.align_rows(core.read_rows(folder/'oof.csv',{'ID','hardness','fold','scattering_replaced_blend'}),data.ids,folder)
        np.testing.assert_array_equal(core.finite_column(rows,'hardness',folder),data.y)
        fold_path=ROOT/('reports/folds.csv' if seed==42 else 'reports/benchmark_folds_seed2026.csv')
        fold_rows=core.align_rows(core.read_rows(fold_path,{'ID','fold'}),data.ids,fold_path)
        folds=core.integer_column(fold_rows,'fold',fold_path)
        np.testing.assert_array_equal(core.integer_column(rows,'fold',folder),folds)
        if set(folds)!=set(range(5)) or any(sum(folds==k)!=100 for k in range(5)):
            raise ValueError('Expected exact five400/100 splits')
        current=core.finite_column(rows,'scattering_replaced_blend',folder)
        summary=read_json(folder/'summary.json')
        if abs(core.metrics(data.y,current)['rmse']-summary['metrics']['scattering_replaced_blend']['rmse'])>1e-10:
            raise ValueError('Archived current score mismatch')
        refs[seed]={'folds':folds,'current':current}
        files[str(seed)]={n:file_ref(folder/n) for n in ('oof.csv','summary.json','completed_evaluation.json')}
        files[str(seed)]['folds']=file_ref(fold_path)
    paths=[Path(__file__),Path(eng.__file__),Path(assets.__file__),Path(base.__file__),
           Path(core.__file__),Path(old_dino.__file__),ROOT/'scripts/run_dinov2_factorial.ps1']
    frozen={'protocol':PROTOCOL,'environment':environment(),'sources':{p.name:file_ref(p) for p in paths},
        'inputs':data.provenance,'inventory':inventory,'ids_sha256':base.sha256_array(data.ids),
        'y_sha256':base.sha256_array(data.y),'noise_sha256':base.sha256_array(data.noise),
        'upstream_plan':file_ref(ROOT/'reports/scattering_tabpfn/protocol.json'),
        'references':files,'common_failure_ids':common_ids,
        'preserved_submission':file_ref(ROOT/'reports/final_scattering_blend/submission/submission.csv')}
    return data,refs,frozen


def prepare(folder=OUT):
    data,refs,frozen=load_state();path=folder/'protocol.json'
    if path.exists():
        if read_json(path)['frozen']!=frozen: raise ValueError('Existing protocol changed; use a new output directory')
    else:
        if folder.exists() and any(folder.iterdir()): raise ValueError('Nonempty unprepared directory')
        write_json(path,{'frozen':frozen,'created_unix':time.time(),'competition_fits':0,'competition_forwards':0})
    print('Prepared/verified Small/Large x266/518; no feature extraction or fitting.')
    for seed in SEEDS: print(f'Current seed{seed}: {core.metrics(data.y,refs[seed]["current"])["rmse"]:.6f}')


def validate(folder=OUT):
    if not (folder/'protocol.json').exists(): raise FileNotFoundError('Run prepare first.')
    data,refs,frozen=load_state();plan=read_json(folder/'protocol.json')
    if plan['frozen']!=frozen: raise ValueError('Frozen sources/inputs/environment changed; preserve old folder')
    return data,refs,plan


def configure(device_name):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    import torch
    args=SimpleNamespace(device=device_name,seed=123,threads=2)
    device,info=base.configure_runtime(torch,np,args)
    return torch,device,info


def check(device_name):
    assets.validate();torch,device,info=configure(device_name)
    results=[]
    for name in assets.MODELS:
        model=eng.load_model(name).to(device)
        for size in (266,518):
            if device.type=='cuda': torch.cuda.reset_peak_memory_stats(device)
            raw=torch.linspace(0,1,256*256,device=device).reshape(1,1,256,256)
            start=time.perf_counter()
            with torch.inference_mode():
                x=eng.transform_image(raw,size)
                output=eng.pool_tokens(model.forward_features(x),size,assets.MODELS[name]['dim'])
            if device.type=='cuda':torch.cuda.synchronize(device)
            record={'representation':name+str(size),'shape':list(output.shape),'seconds':time.perf_counter()-start,
                    'peak_cuda_bytes':torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None,
                    'encoder_frozen':not any(p.requires_grad for p in model.parameters())}
            results.append(record);print(base.canonical_json(record),flush=True)
            del raw,x,output
        del model
        if device.type=='cuda':torch.cuda.empty_cache()
    print(base.canonical_json({'runtime':info,'synthetic_only':True,'competition_forwards':0,'results':results}))


def npz_save(path, ids, x, binding, extra=None):
    path.parent.mkdir(parents=True,exist_ok=True)
    metadata={'binding':binding,'ids_sha256':base.sha256_array(ids),'x_sha256':base.sha256_array(x),
              'extra':extra or {}}
    tmp=path.with_name(path.name+'.tmp')
    with tmp.open('wb') as f:
        np.savez_compressed(f,ids=ids,x=x,metadata_json=np.array(base.canonical_json(metadata)))
    os.replace(tmp,path)
    return metadata


def npz_read(path, ids, width, binding=None):
    if not path.exists(): raise FileNotFoundError('Feature cache incomplete. Run extract first: '+str(path))
    with np.load(path,allow_pickle=False) as a:
        if set(a.files)!={'ids','x','metadata_json'}:raise ValueError('Invalid feature schema')
        actual_ids,x=a['ids'],a['x'];m=json.loads(a['metadata_json'].item())
    if not np.array_equal(ids,actual_ids) or x.dtype!=np.float32 or x.shape!=(len(ids),width) or not np.isfinite(x).all():
        raise ValueError('Feature IDs/shape/precision/values differ')
    if binding is not None and m['binding']!=binding: raise ValueError('Feature scope changed')
    if m['ids_sha256']!=base.sha256_array(actual_ids) or m['x_sha256']!=base.sha256_array(x):
        raise ValueError('Feature payload hash mismatch')
    return x,m


def feature_scope(plan,rep):
    name='small' if rep.startswith('small') else 'large'
    return {'plan':digest(plan['frozen']),'representation':rep,'weights':assets.MODELS[name]}


def validate_anchor(actual, expected):
    if actual.shape!=expected.shape: raise ValueError('Small266 anchor shape changed')
    diff=np.abs(actual.astype(np.float64)-expected.astype(np.float64))
    detail={'max_abs_difference':float(diff.max()),'mean_abs_difference':float(diff.mean()),
            'rtol':PROTOCOL['anchor']['rtol'],'atol':PROTOCOL['anchor']['atol']}
    if not np.allclose(actual,expected,rtol=detail['rtol'],atol=detail['atol']):
        raise ValueError('Small266 anchor failed; no complete feature bundle activated: '+base.canonical_json(detail))
    return detail


def load_features(folder,data,plan):
    values={};refs={}
    for rep in eng.REPS:
        width=1152 if rep.startswith('small') else 3072
        path=folder/'features'/rep/'features.npz'
        x,m=npz_read(path,data.ids,width,feature_scope(plan,rep))
        if m['extra'].get('precision')!='float32' or m['extra'].get('amp') is not False:
            raise ValueError('Features must use fixed FP32/noAMP protocol')
        if rep=='small266': validate_anchor(x,data.features['cls_patch_mean_std'])
        values[rep]=x;refs[rep]=file_ref(path)
    return values,refs


def extract(folder,device_name,batch_size):
    if batch_size<1:raise ValueError('batch-size must be positive')
    data,_,plan=validate(folder);assets.validate()
    torch,device,info=configure(device_name)
    from PIL import Image
    runtime={'device':str(device),'gpu':info.get('gpu_name'),'batch_size':batch_size,'precision':'float32','amp':False}
    for name,spec in assets.MODELS.items():
        model=None
        try:
            for size in (266,518):
                rep=name+str(size);directory=folder/'features'/rep;scope=feature_scope(plan,rep)
                path=directory/'features.npz'
                if path.exists():
                    x,_=npz_read(path,data.ids,3*spec['dim'],scope)
                    if rep=='small266':validate_anchor(x,data.features['cls_patch_mean_std'])
                    print(rep+': complete cache verified; no new forwards',flush=True);continue
                if model is None:
                    model=eng.load_model(name)
                    if name=='small' and base.feature_state_hash(model)!=data.metadata['feature_state_sha256']:
                        raise ValueError('Small checkpoint state does not match original extractor')
                    model.to(device)
                binding={'feature_scope':scope,'runtime':runtime}
                state_path=directory/'runtime.json'
                if state_path.exists() and read_json(state_path)!=binding:
                    raise ValueError('Extraction runtime/batch changed; use the original command or a fresh output folder')
                write_json(state_path,binding)
                rows=[];start=time.perf_counter()
                for offset in range(0,len(data.ids),batch_size):
                    ids=data.ids[offset:offset+batch_size]
                    chunk=directory/'chunks'/f'{offset:04d}.npz'
                    chunk_binding={**binding,'offset':offset}
                    if chunk.exists():
                        values,_=npz_read(chunk,ids,3*spec['dim'],chunk_binding)
                    else:
                        raw=[]
                        for ident in ids:
                            with Image.open(ROOT/'open/train'/f'{ident}.png') as im:
                                if im.mode!='L' or im.size!=(256,256):raise ValueError('Input image schema changed')
                                raw.append(np.array(im,dtype=np.float32)/np.float32(255))
                        pixels=torch.from_numpy(np.stack(raw)[:,None]).to(device)
                        with torch.inference_mode():
                            values=eng.pool_tokens(model.forward_features(eng.transform_image(pixels,size)),size,spec['dim']).cpu().numpy()
                        if rep=='small266':validate_anchor(values,data.features['cls_patch_mean_std'][offset:offset+len(ids)])
                        npz_save(chunk,ids,values,chunk_binding)
                    rows.append(values)
                    done=offset+len(ids)
                    if offset==0 or done%50==0 or done==len(data.ids):
                        print(f'{rep}: {done}/{len(data.ids)} verified/extracted; elapsed={time.perf_counter()-start:.1f}s',flush=True)
                matrix=np.concatenate(rows)
                anchor=validate_anchor(matrix,data.features['cls_patch_mean_std']) if rep=='small266' else None
                base.ensure_inputs_unchanged(ROOT/'open',plan['frozen']['inventory'])
                npz_save(path,data.ids,matrix,scope,{**runtime,'anchor':anchor,'seconds':time.perf_counter()-start})
                print(f'Saved {path}; shape={matrix.shape}',flush=True)
        finally:
            if model is not None:del model
            if device.type=='cuda':torch.cuda.empty_cache()
    load_features(folder,data,plan)
    print('All four feature bundles verified. Encoder updates=0.')


def cell(path,binding,producer,allow_fit):
    if path.exists():
        saved=read_json(path)
        if saved['binding']!=binding or saved['sha256']!=digest(saved['result']):
            raise ValueError('Cell scope/payload changed: '+str(path))
        return saved['result']
    if not allow_fit:raise FileNotFoundError('Incomplete evaluation; run evaluate first: '+str(path))
    result=producer();write_json(path,{'binding':binding,'result':result,'sha256':digest(result)})
    return result


def fit_fold(folder,binding,train_features,train_y,query_features,seed,allow_fit=True,runner=None):
    """No outer-query labels or full target array are available to this function."""
    runner=runner or eng.fit_candidates
    y=eng.finite(train_y);splits=eng.inner_splits(y,seed)
    if tuple(train_features)!=eng.REPS or tuple(query_features)!=eng.REPS: raise ValueError('Four representations required')
    inner={r:{str(i):np.full(len(y),np.nan) for i in range(len(eng.CANDIDATES))} for r in eng.REPS}
    for rep,x in train_features.items():
        for k,(ti,vi) in enumerate(splits):
            scope={'parent':binding,'rep':rep,'inner':k,'train':ti.tolist(),'query':vi.tolist(),
                   'x':base.sha256_array(x[ti]),'query_x':base.sha256_array(x[vi]),'y':base.sha256_array(y[ti])}
            result=cell(folder/rep/f'inner{k}.json',scope,lambda:runner(x[ti],y[ti],x[vi]),allow_fit)
            if set(result['predictions'])!=set(inner[rep]):raise ValueError('Candidate prediction set differs')
            for key,p in result['predictions'].items():inner[rep][key][vi]=eng.finite(p,(len(vi),))
    choices,scores=eng.select(inner,y)
    selected={r:sorted({c['candidate_index'] for c in choices.values() if c['representation']==r}) for r in eng.REPS}
    refits={}
    for rep,indices in selected.items():
        scope={'parent':binding,'rep':rep,'candidates':indices,'x':base.sha256_array(train_features[rep]),
               'query_x':base.sha256_array(query_features[rep]),'y':base.sha256_array(y)}
        result=cell(folder/rep/'refit.json',scope,
            lambda:runner(train_features[rep],y,query_features[rep],indices),allow_fit)
        if set(result['predictions'])!={str(i) for i in indices}:raise ValueError('Refit selection differs')
        refits[rep]=result['predictions']
    predictions={n:eng.finite(refits[c['representation']][str(c['candidate_index'])],
                            (len(query_features[c['representation']]),)) for n,c in choices.items()}
    return predictions,choices,scores


def evaluation_binding(plan,seed,feature_refs):
    return {'plan':digest(plan['frozen']),'seed':seed,'features':feature_refs}


def collect(folder,data,ref,features,binding,seed,allow_fit=True,runner=None):
    predictions={};selections=[]
    with threadpool_limits(limits=2):
        for fold in range(5):
            ti=np.flatnonzero(ref['folds']!=fold);vi=np.flatnonzero(ref['folds']==fold)
            local={'evaluation':binding,'outer_fold':fold,'train_indices':ti.tolist(),'query_indices':vi.tolist()}
            p,c,s=fit_fold(folder/f'fold{fold}',local,{r:x[ti] for r,x in features.items()},data.y[ti],
                          {r:x[vi] for r,x in features.items()},seed+fold,allow_fit,runner)
            for name,values in p.items():
                if name not in predictions:predictions[name]=np.full(len(data.y),np.nan)
                predictions[name][vi]=values
            selections.append({'fold':fold,'choices':c,'inner_scores':s})
            choice=c['joint_nested']
            print(f'Outer{fold}: {choice["representation"]} {choice["candidate"]}; inner={choice["inner_rmse"]:.6f}; outer labels unused',flush=True)
    output={'current':ref['current'],**predictions}
    output.update({'current_'+n.removesuffix('_nested')+'_equal':.5*ref['current']+.5*p for n,p in predictions.items()})
    return output,selections


def report_data(data,ref,predictions,common_ids):
    scores={n:core.metrics(data.y,p) for n,p in predictions.items()}
    for m in scores.values():
        m['gain_vs_current']=scores['current']['rmse']-m['rmse']
        m['gain_vs_control']=scores[CONTROL]['rmse']-m['rmse']
    target=core.rank_groups(data.y,10);noise=core.rank_groups(data.noise,4);common=np.isin(data.ids,common_ids)
    masks={'common16':common,'remaining484':~common,'low10':target==0,'high10':target==9,'middle80':(target>0)&(target<9)}
    masks.update({f'noiseQ{k+1}':noise==k for k in range(4)})
    masks.update({f'fold{k}':ref['folds']==k for k in range(5)})
    groups=[{'model':n,'group':g,**core.metrics(data.y[m],p[m])} for n,p in predictions.items() for g,m in masks.items()]
    worst={n:set(np.argsort((p-data.y)**2,kind='stable')[-20:]) for n,p in predictions.items()}
    details={}
    for n,p in predictions.items():
        err=p-data.y;old=ref['current']-data.y
        details[n]={'group_rmse':{g:core.metrics(data.y[m],p[m])['rmse'] for g,m in masks.items()},
            'common16_improved':int(((err**2<old**2)&common).sum()),
            'common16_in_worst20':int(sum(common[list(worst[n])])),
            'common16_same_error_sign':int((np.sign(err[common])==np.sign(old[common])).sum()),
            'residual_correlation_vs_current':core.correlation(err,old),
            'new_worst20_ids':[str(data.ids[i]) for i in sorted(worst[n]-worst['current'])],
            'left_worst20_ids':[str(data.ids[i]) for i in sorted(worst['current']-worst[n])]}
    oof=[{'ID':str(ident),'hardness':float(data.y[i]),'fold':int(ref['folds'][i]),
          **{n:float(p[i]) for n,p in predictions.items()}} for i,ident in enumerate(data.ids)]
    errors=[{'ID':str(data.ids[i]),'hardness':float(data.y[i]),'fold':int(ref['folds'][i]),'noise':float(data.noise[i]),
             'common16':bool(common[i]),**{n+'_error':float(p[i]-data.y[i]) for n,p in predictions.items()}}
            for i in np.argsort(-(predictions[PRIMARY]-data.y)**2,kind='stable')]
    return scores,details,groups,oof,errors


OUTPUTS=('summary.json','oof.csv','diagnostics.csv','per_image_errors.csv','common16.csv','selection.json','report.txt')


def completed(folder,binding):
    path=folder/'completed.json'
    if not path.exists():return False
    done=read_json(path)
    if done['binding']!=binding or set(done['files'])!=set(OUTPUTS):raise ValueError('Completion scope changed')
    for name,sha in done['files'].items():
        if base.sha256_file(folder/name)!=sha:raise ValueError('Completed result changed: '+name)
    return True


def save_results(folder,data,ref,predictions,selections,plan,binding):
    scores,details,groups,oof,errors=report_data(data,ref,predictions,plan['frozen']['common_failure_ids'])
    write_json(folder/'summary.json',{'binding':binding,'metrics':scores,'diagnosis':details,
        'primary':PRIMARY,'control':CONTROL,'encoder_updates':0,'test_accessed':False,'independent_holdout':False})
    write_json(folder/'selection.json',selections)
    for name,rows in [('oof.csv',oof),('diagnostics.csv',groups),('per_image_errors.csv',errors),
                     ('common16.csv',[r for r in errors if r['common16']])]:write_csv(folder/name,rows)
    lines=['DINOv2 Small/Large x266/518; seed'+str(binding['seed']),
           'PRIMARY='+PRIMARY+'; CONTROL='+CONTROL,
           'Model | OOF RMSE | gain vs current | gain vs Small266 Ridge blend']
    lines += [f'{n}: {m["rmse"]:.6f} | {m["gain_vs_current"]:+.6f} | {m["gain_vs_control"]:+.6f}' for n,m in scores.items()]
    lines += ['', 'Selection is inner-only; fixed-representation scores are diagnostics, not automatic adoption.',
              'Same repeatedly studied500, not independent holdout. No encoder updates or test predictions.',
              'See common16.csv, diagnostics.csv, per_image_errors.csv and selection.json.',
              'Current ensemble includes PriorLabs-TabPFN. New regressors: Ridge and PCA64/RBF-SVR.']
    (folder/'report.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    write_json(folder/'completed.json',{'binding':binding,'files':{n:base.sha256_file(folder/n) for n in OUTPUTS}})
    for n in ('small266_ridge','small518_ridge','large266_ridge','large518_ridge','joint_nested',CONTROL,PRIMARY):
        print(f'{n}: RMSE={scores[n]["rmse"]:.6f}; gain vs current={scores[n]["gain_vs_current"]:+.6f}',flush=True)
    print('Report: '+str(folder/'report.txt'))


def evaluate(folder,seed):
    data,refs,plan=validate(folder);features,fr=load_features(folder,data,plan)
    binding=evaluation_binding(plan,seed,fr);sub=folder/f'seed{seed}'
    if completed(sub,binding):print(f'Completed seed{seed} verified; no refitting.');return
    p,choices=collect(sub,data,refs[seed],features,binding,seed)
    validate(folder);load_features(folder,data,plan)
    save_results(sub,data,refs[seed],p,choices,plan,binding)


def compare(folder):
    data,refs,plan=validate(folder);features,fr=load_features(folder,data,plan)
    summaries=[];records=[]
    for seed in SEEDS:
        sub=folder/f'seed{seed}';binding=evaluation_binding(plan,seed,fr)
        if not completed(sub,binding):raise FileNotFoundError(f'Run evaluate --seed {seed} first')
        p,choices=collect(sub,data,refs[seed],features,binding,seed,allow_fit=False)
        scores,details,*_=report_data(data,refs[seed],p,plan['frozen']['common_failure_ids'])
        summary=read_json(sub/'summary.json')
        if summary['metrics']!=scores or summary['diagnosis']!=details or read_json(sub/'selection.json')!=choices:
            raise ValueError('Saved results do not match cached predictions')
        summaries.append(scores)
        records.append({'seed':seed,'current_rmse':scores['current']['rmse'],'candidate_rmse':scores[PRIMARY]['rmse'],
            'control_rmse':scores[CONTROL]['rmse'],'gain_vs_current':scores[PRIMARY]['gain_vs_current'],
            'gain_vs_control':scores[PRIMARY]['gain_vs_control'],
            'diagnosis':{n:details[n] for n in ('current',CONTROL,PRIMARY,'joint_nested')}})
    mean_gain=float(np.mean([r['gain_vs_current'] for r in records]))
    positive=all(r['gain_vs_current']>0 and r['gain_vs_control']>0 for r in records)
    result={'primary':PRIMARY,'per_seed':records,'mean_gain_vs_current':mean_gain,
        'all_seeds_improve_both':positive,'passes_expansion_priority':positive and mean_gain>=.5,
        'all_models_mean_gain_vs_current':{n:float(np.mean([s[n]['gain_vs_current'] for s in summaries])) for n in summaries[0]},
        'protocol_sha256':digest(plan['frozen']),
        'additional_review_required':'Standalone, common16/remaining484, middle80, tails, noise, newworst20. No automatic adoption.',
        'aggregation':'Mean per-seed differences. Same500, no averaging predictions or independent-sample inference.'}
    write_json(folder/'comparison.json',result)
    print(base.canonical_json(result));return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('prepare','validate','check','extract','evaluate','compare'))
    p.add_argument('--output-dir',type=Path,default=OUT)
    p.add_argument('--device',default='cuda');p.add_argument('--batch-size',type=int,default=1)
    p.add_argument('--seed',type=int,choices=SEEDS,default=42)
    a=p.parse_args(argv);folder=a.output_dir if a.output_dir.is_absolute() else ROOT/a.output_dir
    if a.command=='prepare':prepare(folder)
    elif a.command=='validate':validate(folder);print('Protocol, images and preserved references verified; no fit/forward.')
    elif a.command=='check':check(a.device)
    elif a.command=='extract':extract(folder,a.device,a.batch_size)
    elif a.command=='evaluate':evaluate(folder,a.seed)
    else:compare(folder)


if __name__=='__main__':main()
