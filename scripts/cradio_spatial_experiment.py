"""User-run frozen C-RADIO dense readout contrast; replace only current C25%."""
from __future__ import annotations
import argparse
import contextlib
import csv
import gc
import io
import math
import os
from pathlib import Path
import shutil
import time
import numpy as np
from threadpoolctl import threadpool_limits
if __package__:
    from . import cradiov4_experiment as prior, cradio_spatial_engine as eng
else:
    import cradiov4_experiment as prior
    import cradio_spatial_engine as eng

e,core,diagnostics=prior.e,prior.core,prior.diagnostics
ROOT=e.ROOT;SOURCE=ROOT/'reports/cradiov4_importfix'
OUT=ROOT/'reports/cradio_spatial';CHECK=ROOT/'reports/cradio_spatial_runtime/check.json'
PRIMARY='spatial_replaced_blend';CONTROL='global_pool_blend';SEEDS=(42,2026)
PROTOCOL={
    'version':1,'seeds':list(SEEDS),'primary':PRIMARY,'control':CONTROL,
    'encoder':'Existing pinned frozen C-RADIOv4-SO400M, native256 fullimage bicubic592, internal conditioner, BF16 inference, FP32 storage; no new weights/downloads/augmentation.',
    'features':'summary2304 plus spatial37x37x1152. Excludes register/CLS tokens. All500 old pooled4608 anchors must match rtol1e-5,atol1e-5. No token subsampling/PCA.',
    'heads':{'global_pool':'Shared1152->32/GELU; uniform mean/std; summary2304->32/GELU; Linear96->32/GELU/Dropout.1/Linear32->1.',
             'spatial_attention':'Same shared modules; residual depthwise3x3+pointwise32 initialized as identity; gated attention16 initialized uniform; weighted mean/std. No per-image LayerNorm.',
             'initialization':'Shared seed; same shared initial parameters; readout last layer zero. Reset dropout RNG after module construction. One image-level prediction/loss.'},
    'optimization':{'epochs':list(eng.EPOCHS),'batch_images':eng.BATCH,'lr':eng.LR,'weight_decay':eng.WD,
                    'optimizer':'AdamW','schedule':'cosine40 with .1 floor, same prefix on refit','loss':'unweighted standardized-target MSE','gradient_clip':1.,'head_precision':'FP32,noTF32,noAMP'},
    'scaling':'Train-only per-token-channel and summary-channel mean/std, y mean/std. Per-image float64 token moments aggregate only fitting images. No query-based stats.',
    'validation':'5 outer400/100 x3 inner per seed. Every image and all its tokens stay together. Paired head seeds=seed+1009*outer+inner; refit+9.',
    'selection':'Choose epoch by pooled INNER MSE of .5S_inner+.25L_inner+.25head; exact tie earlier epoch. S_inner verified context-only cache. L_inner reselected standalone alpha100/1000/10000 by additional3fold CV within inner context; larger alpha on tie. Never use global OOF for inner baseline.',
    'blend':'.5Scattering486+.25Large518+.25C_head. Preserve existing S/L predictions, existing C-Ridge control and submitted S:L:C=50:25:25.',
    'diagnostics':'Standalone, current/Ridge, learned-global control, common16,remaining484,tails,middle80,noiseQ1..4,folds,newworst20,training curves and actual gradient/weight updates.',
    'scope':'User runs train-image extraction and80 small-head trajectories plus context-only Large Ridge fits. No encoder/TabPFN updates,test access,final500 fitting,mixture search or submission changes.',
    'decision':'Bothseeds improve current and learned-global control. Inspect common16 AND remaining484/noise. Mean gain>=.30 prioritizes expansion only; no automatic adoption.',
    'limits':'Same repeatedly studied500/reused splits, not independent holdout. Spatial convolution+attention tested together. Attention is not verified physical segmentation.'}


def load_state():
    with contextlib.redirect_stdout(io.StringIO()):
        data,large,old,cp=prior.validate(SOURCE)
        pooled,fr=prior.load_features(SOURCE,data,cp)
        for seed in SEEDS:prior.evaluate(SOURCE,seed,False)
    if prior.read_check(SOURCE,cp)['size']!=592:raise ValueError('This experiment is fixed to592')
    refs={};files={}
    for seed in SEEDS:
        sub=SOURCE/f'seed{seed}'
        rows=core.read_rows(sub/'oof.csv',{'ID','hardness','fold','radio_ridge','radio_complementary_blend'})
        if [r['ID'] for r in rows]!=data.ids.tolist():raise ValueError('Reference ID order differs')
        np.testing.assert_array_equal([float(r['hardness']) for r in rows],data.y)
        np.testing.assert_array_equal([int(r['fold']) for r in rows],old[seed]['folds'])
        refs[seed]={**old[seed],'radio':np.array([float(r['radio_ridge']) for r in rows]),
                    'current':np.array([float(r['radio_complementary_blend']) for r in rows])}
        np.testing.assert_allclose(refs[seed]['current'],.5*old[seed]['scattering']+.25*old[seed]['large']+.25*refs[seed]['radio'],atol=1e-10,rtol=0)
        files[str(seed)]={n:e.file_ref(sub/n) for n in ('completed_evaluation.json','oof.csv','summary.json')}
    paths=[Path(__file__),Path(eng.__file__),ROOT/'scripts/check_cradio_spatial.py',ROOT/'scripts/run_cradio_spatial.ps1']
    frozen={'protocol':PROTOCOL,'source_files':{p.name:e.file_ref(p) for p in paths},
        'upstream_protocol':e.file_ref(SOURCE/'protocol.json'),'pooled_features':fr,'references':files,
        'environment':e.environment(),'assets':prior.eng.assets.spec(),'common_failure_ids':cp['frozen']['common_failure_ids'],
        'training_inventory':cp['frozen']['training_inventory'],'large_features':cp['frozen']['large_features'],
        'inner_reference_hashes':{str(s):{str(f):e.base.sha256_array(refs[s]['inner_scattering'][f]) for f in range(5)} for s in SEEDS},
        'submissions':{n:e.file_ref(ROOT/f'reports/{n}/submission/submission.csv') for n in ('final_scattering_blend','final_large518_blend','final_cradiov4_blend')}}
    return data,large,pooled,refs,frozen


def checked(frozen):
    r=e.read_json(CHECK)
    if r['frozen_sha256']!=e.digest(frozen) or not r['passed'] or r['competition_fits']!=0 or r['competition_forwards']!=0:
        raise ValueError('Run check first: synthetic check missing/stale')
    return r


def check(device='cuda'):
    if __package__:from . import check_cradio_spatial as bench
    else:import check_cradio_spatial as bench
    *_,frozen=load_state()
    if CHECK.exists():
        r=checked(frozen);_,_,runtime=prior.runtime(device)
        if runtime!=r['runtime']:raise ValueError('Checked runtime changed')
        print('Synthetic check verified; no repeat.');print(e.base.canonical_json(r['timing_estimate']));return
    r=bench.run(device);r['frozen_sha256']=e.digest(frozen);e.write_json(CHECK,r)
    print(e.base.canonical_json(r));print('Check: '+str(CHECK))


def prepare():
    *_,frozen=load_state();checked(frozen);frozen['capacity_check']=e.file_ref(CHECK)
    path=OUT/'protocol.json'
    if path.exists():
        if e.read_json(path)['frozen']!=frozen:raise ValueError('Prepared protocol changed; preserve old results')
    else:
        if OUT.exists() and any(OUT.iterdir()):raise ValueError('Nonempty unprepared experiment')
        e.write_json(path,{'frozen':frozen,'created_unix':time.time(),'competition_fits':0,'competition_forwards':0})
    print('Prepared: frozen592 dense tokens; global/spatial heads; only C25% replaced. No training.')


def validate():
    data,large,pooled,refs,frozen=load_state();checked(frozen);frozen['capacity_check']=e.file_ref(CHECK)
    plan=e.read_json(OUT/'protocol.json')
    if plan['frozen']!=frozen:raise ValueError('Sources, inputs, protocol or environment changed')
    return data,large,pooled,refs,plan


def binding(plan,**extra):return {'plan':e.digest(plan['frozen']),**extra}


def load_tokens(plan):
    folder=OUT/'features';r=e.read_json(folder/'complete.json')
    if r['binding']!=binding(plan) or r['rows']!=500 or r['encoder_updates']!=0:raise ValueError('Feature binding differs')
    for name,ref in r['files'].items():
        if e.file_ref(folder/name)!=ref:raise ValueError('Feature changed: '+name)
    tokens=np.load(folder/'tokens.npy',mmap_mode='r',allow_pickle=False)
    if tokens.shape!=(500,1369,1152) or tokens.dtype!=np.float32:raise ValueError('Unexpected token array')
    with np.load(folder/'auxiliary.npz',allow_pickle=False) as a:
        aux={k:a[k].copy() for k in a.files}
    return tokens,aux,e.file_ref(folder/'complete.json')


def extract(device='cuda'):
    data,_,pooled,_,plan=validate();torch,dev,runtime=prior.runtime(device)
    if runtime!=checked_runtime(plan):raise ValueError('Encoder runtime changed')
    folder=OUT/'features'
    if (folder/'complete.json').exists():load_tokens(plan);print('All500 tokens verified; no forwards.');return
    folder.mkdir(parents=True,exist_ok=True);path=folder/'tokens.npy'
    if not path.exists():
        if shutil.disk_usage(folder).free<4*2**30:raise ValueError('Need at least4GiB disk for token cache')
        a=np.lib.format.open_memmap(path,mode='w+',dtype=np.float32,shape=(500,1369,1152));a.flush();del a
    tokenmap=np.lib.format.open_memmap(path,mode='r+')
    if tokenmap.shape!=(500,1369,1152) or tokenmap.dtype!=np.float32:raise ValueError('Partial token schema differs')
    model=None;allaux=[];started=time.perf_counter();maxdiff=0.
    from PIL import Image
    for start in range(0,500,16):
        end=min(start+16,500);ids=data.ids[start:end];part=folder/f'chunks/{start:04d}.npz';receipt=part.with_suffix('.json')
        scope=binding(plan,start=start,ids=ids.tolist(),runtime=runtime)
        if receipt.exists():
            r=e.read_json(receipt)
            if r['binding']!=scope or r['auxiliary']!=e.file_ref(part) or r['tokens_sha256']!=e.base.sha256_array(tokenmap[start:end]):
                raise ValueError('Partial feature chunk changed')
            with np.load(part,allow_pickle=False) as z:aux={k:z[k].copy() for k in z.files}
        else:
            if model is None:model=prior.eng.load_model().to(dev)
            ss=[];mm=[];vv=[];pp=[]
            for i,ident in enumerate(ids,start):
                with Image.open(ROOT/'open/train'/f'{ident}.png') as im:
                    if im.mode!='L' or im.size!=(256,256):raise ValueError('Unexpected image schema')
                    raw=torch.from_numpy(np.array(im,dtype=np.uint8,copy=True)).to(device=dev,dtype=torch.float32)[None,None]/255.
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):output=model(prior.eng.transform(raw,592))
                features=prior.eng.pool(output,592).cpu().numpy()[0]
                np.testing.assert_allclose(features,pooled[i],rtol=1e-5,atol=1e-5,err_msg='Original4608 anchor failed')
                summary,spatial=output;s=summary.float().cpu().numpy()[0];t=spatial.float().cpu().numpy()[0]
                if t.shape!=(1369,1152) or not np.isfinite(t).all():raise ValueError('Invalid spatial features')
                tokenmap[i]=t;ss.append(s);mm.append(t.mean(0,dtype=np.float64));vv.append(np.mean(np.square(t.astype(np.float64)),axis=0));pp.append(features)
            aux={'ids':ids,'summary':np.stack(ss),'means':np.stack(mm),'second':np.stack(vv),'pooled':np.stack(pp)}
            part.parent.mkdir(parents=True,exist_ok=True);temp=part.with_name(part.stem+'.pending.npz')
            np.savez(temp,**aux);os.replace(temp,part);tokenmap.flush()
            e.write_json(receipt,{'binding':scope,'auxiliary':e.file_ref(part),'tokens_sha256':e.base.sha256_array(tokenmap[start:end])})
        np.testing.assert_array_equal(aux['ids'],ids)
        np.testing.assert_allclose(aux['pooled'],pooled[start:end],rtol=1e-5,atol=1e-5)
        maxdiff=max(maxdiff,float(np.max(np.abs(aux['pooled']-pooled[start:end]))));allaux.append(aux)
        print(f'Extracted/verified {end}/500; all1369 tokens retained; elapsed={time.perf_counter()-started:.1f}s',flush=True)
    tokenmap.flush();del tokenmap,model;gc.collect();torch.cuda.empty_cache()
    aux={k:np.concatenate([a[k] for a in allaux]) for k in allaux[0]}
    np.savez(folder/'auxiliary.npz',**aux)
    validate()
    e.write_json(folder/'complete.json',{'binding':binding(plan),'rows':500,'encoder_updates':0,'max_anchor_absdiff':maxdiff,
        'seconds':time.perf_counter()-started,'files':{n:e.file_ref(folder/n) for n in ('tokens.npy','auxiliary.npz')}})
    load_tokens(plan);print('Dense features ready: '+str(path))


def checked_runtime(plan):
    r=e.read_json(CHECK)
    if e.file_ref(CHECK)!=plan['frozen']['capacity_check']:raise ValueError('Capacity check changed')
    return r['runtime']


def inner_large(x,y,q,seed):
    """Extra CV within the inner context; query labels are not accepted."""
    ridge=prior.eng.ridge;pred={str(i):np.full(len(y),np.nan) for i in prior.eng.RIDGE_INDICES}
    for ti,vi in ridge.inner_splits(y,seed):
        r=ridge.fit_candidates(x[ti],y[ti],x[vi],prior.eng.RIDGE_INDICES)
        for k,v in r['predictions'].items():pred[k][vi]=v
    scores={k:float(np.mean((v-y)**2)) for k,v in pred.items()}
    best=min(scores,key=lambda k:(scores[k],int(k)))
    r=ridge.fit_candidates(x,y,q,[int(best)])
    return {'predictions':r['predictions'][best],'alpha':ridge.CANDIDATES[int(best)]['alpha'],
            'inner_mse':scores,'train_rows':len(y),'query_rows':len(q),'query_labels_accepted':False}


def validate_run(r,ti,qi,epochs):
    a=r['audit']
    if a['train_indices']!=ti.tolist() or a['query_indices']!=qi.tolist() or a['epochs']!=list(epochs):raise ValueError('Head scope changed')
    if a['optimizer_steps']!=math.ceil(len(ti)/eng.BATCH)*max(epochs) or a['encoder_updates']!=0 or a['query_labels_accepted']:
        raise ValueError('Incomplete/unexpected head training')
    if [v['epoch'] for v in r['records']]!=list(epochs):raise ValueError('Checkpoint coverage changed')
    for row in r['records']:prior.eng.ridge.finite(row['predictions'],(len(qi),))
    if max(epochs)>0 and (a['parameter_delta_l2']<=0 or a['max_gradient_norm']<=0):raise ValueError('No head update')
    if len(epochs)==1 and not r.get('checkpoint_ref'):raise ValueError('Missing refit checkpoint')
    if r.get('checkpoint_ref') and e.file_ref(r['checkpoint_path'])!=r['checkpoint_ref']:raise ValueError('Head checkpoint changed')


def fit_fold(folder,scope,store,large,ti,qi,y,scattering_inner,seed,fold,allow_fit,run=eng.trajectory):
    # y contains only outer training labels; no outer query labels enter this function.
    splits=prior.eng.ridge.inner_splits(y,seed+fold)
    other=.5*np.asarray(scattering_inner,float).copy();choice=[];predictions={}
    for k,(a,b) in enumerate(splits):
        ctx,query=ti[a],ti[b]
        bound={**scope,'role':'inner_large','inner':k,'train':ctx.tolist(),'query':query.tolist(),
            'y':e.base.sha256_array(y[a]),'x':e.base.sha256_array(large[ctx]),'q':e.base.sha256_array(large[query]),'seed':seed+1009*fold+k}
        r=e.cell(folder/f'inner_large{k}.json',bound,lambda:inner_large(large[ctx],y[a],large[query],seed+1009*fold+k),allow_fit)
        other[b]+=.25*prior.eng.ridge.finite(r['predictions'],(len(b),))
    for kind in eng.REPS:
        runs=[]
        def execute(label,ctx,target,query,epochs,model_seed):
            path=folder/kind/f'{label}.json';checkpoint=path.with_suffix('.pt') if label=='refit' else None
            bound={**scope,'kind':kind,'role':label,'train':ctx.tolist(),'query':query.tolist(),
                   'y':e.base.sha256_array(target),'epochs':list(epochs),'model_seed':model_seed}
            def producer():
                result=run(store,ctx,target,query,kind=kind,seed=model_seed,epochs=epochs,checkpoint=checkpoint,
                           log=f'seed{seed} outer{fold} {kind} {label}')
                result['checkpoint_ref']=e.file_ref(result['checkpoint_path']) if result.get('checkpoint_path') else None
                return result
            result=e.cell(path,bound,producer,allow_fit);validate_run(result,ctx,query,epochs);return result
        for k,(a,b) in enumerate(splits):runs.append(execute(f'inner{k}',ti[a],y[a],ti[b],eng.EPOCHS,seed+1009*fold+k))
        best,curves=eng.select_epoch(runs,splits,y,other);epoch=best['epoch']
        result=execute('refit',ti,y,qi,(epoch,),seed+1009*fold+9)
        predictions[kind]=np.array(result['records'][0]['predictions'])
        choice.append({'kind':kind,'selected':best,'curves':curves,'refit_audit':result['audit'],
                       'inner_audits':[r['audit'] for r in runs]})
        print(f'Outer{fold}: {kind} selected epoch={epoch}, inner blend RMSE={math.sqrt(best["blend_mse"]):.6f}; outer labels unused',flush=True)
    return predictions,choice


def report(s):
    lines=[f'C-RADIO frozen spatial readout seed{s["seed"]}; only current C25% replaced; gain>0 improvement.']
    for n,m in s['metrics'].items():lines.append(f'{n}: RMSE={m["rmse"]:.6f}; gain vs current={m["gain_vs_current"]:+.6f}; gain vs matched control={m["gain_vs_control"]:+.6f}')
    d=s['diagnosis'][PRIMARY]
    for g in ('common16','remaining484','noiseQ1','noiseQ2','noiseQ3','noiseQ4','low10','high10','middle80'):
        lines.append(f'{g}: {s["diagnosis"]["current"]["groups"][g]["rmse"]:.6f} -> {d["groups"][g]["rmse"]:.6f}')
    lines += [f'Common16 improved={d["common16_improved"]}/16; worst20={d["common16_in_worst20"]}/16; same sign={d["common16_same_error_sign"]}/16.',f'New worst20={d["new_worst20_ids"]}',PROTOCOL['decision'],PROTOCOL['limits']]
    return '\n'.join(lines)+'\n'


def evaluate(seed,device='cuda',allow_fit=True,state=None):
    data,large,_,refs,plan=state or validate();ref=refs[seed];sub=OUT/f'seed{seed}'
    tokens,aux,fr=load_tokens(plan);np.testing.assert_array_equal(aux['ids'],data.ids)
    scope=binding(plan,seed=seed,features=fr);complete=sub/'completed.json';done=complete.exists()
    if done:
        receipt=e.read_json(complete)
        if receipt['binding']!=scope:raise ValueError('Completion scope changed')
        for n,h in receipt['files'].items():
            if e.base.sha256_file(sub/n)!=h:raise ValueError('Completed artifact changed: '+n)
    if not done and not allow_fit:raise FileNotFoundError('Incomplete; no implicit training')
    store=None
    if not done and allow_fit:
        _,dev,rt=prior.runtime(device)
        if rt!=checked_runtime(plan):raise ValueError('Training runtime changed')
        store=eng.TokenStore(tokens,aux['summary'],aux['means'],aux['second'],dev)
    preds={k:np.full(500,np.nan) for k in eng.REPS};choices=[]
    with threadpool_limits(limits=2):
        for fold in range(5):
            ti=np.flatnonzero(ref['folds']!=fold);qi=np.flatnonzero(ref['folds']==fold)
            if (len(ti),len(qi))!=(400,100):raise ValueError('Outer folds changed')
            p,c=fit_fold(sub/f'fold{fold}',{**scope,'fold':fold},store,large,ti,qi,data.y[ti],ref['inner_scattering'][fold],seed,fold,allow_fit and not done)
            for n in preds:preds[n][qi]=p[n]
            choices.append({'fold':fold,'heads':c})
    p=eng.compose(ref['scattering'],ref['large'],ref['radio'],preds['global_pool'],preds['spatial_attention'])
    np.testing.assert_allclose(p['current'],ref['current'],atol=1e-10,rtol=0)
    mapped={diagnostics.PRIMARY if n==PRIMARY else n:v for n,v in p.items()}
    s,files=diagnostics.result_data(data,ref,mapped,choices,plan,seed)
    for key in ('metrics','diagnosis'):s[key][PRIMARY]=s[key].pop(diagnostics.PRIMARY)
    for rows in files.values():
        for row in rows:
            for k in list(row):
                if diagnostics.PRIMARY in k:row[k.replace(diagnostics.PRIMARY,PRIMARY)]=row.pop(k)
            if row.get('model')==diagnostics.PRIMARY:row['model']=PRIMARY
    files['training_curves.csv']=[{'seed':seed,'fold':f['fold'],'kind':h['kind'],**r,
        'inner_blend_rmse':math.sqrt(r['blend_mse']),'selected':r['epoch']==h['selected']['epoch']}
        for f in choices for h in f['heads'] for r in h['curves']]
    s.update(binding=scope,primary=PRIMARY,control=CONTROL,selection=choices,ridge_anchor_matched=True,
             inner_uses_global_oof=False,new_tabpfn_fits=0)
    for n,m in s['metrics'].items():
        ctrl='global_pool' if n in ('radio_ridge','global_pool','spatial_attention') else CONTROL
        m.update(matched_control=ctrl,gain_vs_control=s['metrics'][ctrl]['rmse']-m['rmse'])
    if done:
        if e.read_json(sub/'summary.json')!=s or (sub/'report.txt').read_text(encoding='utf-8')!=report(s):raise ValueError('Report reconstruction mismatch')
        for n,rows in files.items():
            with (sub/n).open(encoding='utf-8',newline='') as f:
                if list(csv.DictReader(f))!=list(csv.DictReader(io.StringIO(core.csv_text(rows)))):raise ValueError('CSV reconstruction mismatch')
        print(f'Completed seed{seed} verified; no refitting.');return s
    for n,rows in files.items():e.write_csv(sub/n,rows)
    e.write_json(sub/'summary.json',s);(sub/'report.txt').write_text(report(s),encoding='utf-8')
    names=list(files)+['summary.json','report.txt']
    names += [str(p.relative_to(sub)) for p in sub.glob('fold*/*') if p.is_file()]
    names += [str(p.relative_to(sub)) for p in sub.glob('fold*/*/*') if p.is_file() and p.suffix in ('.json','.pt')]
    e.write_json(complete,{'binding':scope,'files':{n:e.base.sha256_file(sub/n) for n in sorted(names)}})
    print(report(s));print('Report: '+str(sub/'report.txt'));return s


def compare():
    state=validate();summaries=[evaluate(s,allow_fit=False,state=state) for s in SEEDS]
    rows=[{'seed':s['seed'],'current_rmse':s['metrics']['current']['rmse'],'control_rmse':s['metrics'][CONTROL]['rmse'],
        'candidate_rmse':s['metrics'][PRIMARY]['rmse'],'gain_vs_current':s['metrics'][PRIMARY]['gain_vs_current'],
        'gain_vs_control':s['metrics'][PRIMARY]['gain_vs_control'],'diagnosis':s['diagnosis'][PRIMARY]} for s in summaries]
    good=all(r['gain_vs_current']>0 and r['gain_vs_control']>0 for r in rows);mean=float(np.mean([r['gain_vs_current'] for r in rows]))
    result={'primary':PRIMARY,'per_seed':rows,'mean_gain_vs_current':mean,'all_seeds_improve_both':good,
        'all_seeds_nonworsening_by_group':{g:all(r['diagnosis']['groups'][g]['gain_vs_current']>=0 for r in rows) for g in ('common16','remaining484','noiseQ4')},
        'passes_expansion_priority':good and mean>=.3,'automatic_adoption':False,'independent_holdout':False,
        'protocol_sha256':e.digest(state[-1]['frozen']),'aggregation':'Mean per-seed OOF RMSE differences; same500, no prediction averaging or independent inference.'}
    e.write_json(OUT/'comparison.json',result);print(e.base.canonical_json(result));return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('check','prepare','extract','evaluate','compare'))
    p.add_argument('--device',default='cuda');p.add_argument('--seed',type=int,choices=SEEDS,default=42);a=p.parse_args()
    if a.command=='check':check(a.device)
    elif a.command=='prepare':prepare()
    elif a.command=='extract':extract(a.device)
    elif a.command=='evaluate':evaluate(a.seed,a.device)
    else:compare()


if __name__=='__main__':main()
