"""User-run split-local MicroNet masked reconstruction + nested Ridge comparison."""
from __future__ import annotations
import argparse
import contextlib
import io
from pathlib import Path
import time
import numpy as np
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
if __package__:
    from . import micronet_mim_engine as eng
else:
    import micronet_mim_engine as eng
old,t,base=eng.old,eng.t,eng.base
core,ROOT=old.core,old.ROOT
OUT=ROOT/'reports/micronet_mim'
SEEDS=(42,2026)
ALPHAS=old.ALPHAS
MODES=('frozen_ridge','mim_nested')
PRIMARY='mim_nested'
CONTROL_BLEND,CANDIDATE_BLEND='current_frozen_equal','current_mim_equal'
PROTOCOL={
 'version':1,'seeds':list(SEEDS),'primary':PRIMARY,'secondary':CANDIDATE_BLEND,
 'question':'Does split-local unlabeled masked reconstruction improve MicroNet representations for hardness prediction?',
 'input':'Native256 grayscale repeated3, fixed ImageNet normalization. No resize, crop, rotation, brightness jitter or denoising.',
 'model':'Pinned local MicroNet-B0. All used encoder convolutions/BN affine parameters train during SSL; BN running buffers and unused classifier-head layers frozen. No adapters.',
 'ssl':{'objective':'Reconstruct original grayscale raw0..1 pixels, L1 on masked pixels only. No hardness values or query images in SSL optimizer.',
    'mask':'39 of64 nonoverlapping32x32 tiles, independently sampled per image per presentation; mask ratio39/64. Fixed raw0.5 fill before normalization.',
    'decoder':'Conv1x1(320,1024) + PixelShuffle32, initialized weight Normal(0,.001), bias .5. Discard for regression.',
    'epochs':list(eng.EPOCHS),'max_epochs':80,'optimizer':'AdamW','encoder_lr':1e-4,'decoder_lr':1e-3,
    'weight_decay':.01,'no_decay':'bias and1D weights','clip_norm':1.,
    'schedule':'5epoch linear warmup; cosine multiplier1 at6 to0.1 at80. Same fixed prefix for outer refits.',
    'batch':8,'accumulation':2,'precision':'FP32, deterministic, TF32/AMP off, no drop-connect'},
 'representation':'After SSL freeze encoder; CLEAN original images -> block10/15 spatial mean/population std864, exactly the original frozen MicroNet representation.',
 'regression':{'model':'Ridge','alphas':list(ALPHAS),'preprocessing':'Independent X/y StandardScalers fitted only on each Ridge fitting subset.',
    'selection':'Control selects alpha at epoch0. Candidate jointly selects epoch0/20/40/80 and alpha by pooled3-inner MSE; ties earlier epoch then larger alpha. Epoch0 is explicit no-SSL fallback.'},
 'split':'Existing5outer x3inner. Each inner SSL uses only its266/267 fitting images; each outer SSL uses only400 fitting images. Never pretrain one encoder on all500 before CV.',
 'ssl_seed':'seed+1009*outer_fold+inner_fold; outer refit+9. Each trajectory starts from pinned MicroNet, not a preceding split.',
 'work':'30 inner80epoch SSL trajectories +10 outer selected-epoch refits, counting epoch0 fallback. Control uses archived frozen features; recompute Ridge and verify original OOF anchor.',
 'audit':'Hardness labels absent from SSL Runner signature. Query images only in eval-only clean feature extraction with state/RNG guard. Early/late encoder and decoder gradients/updates required. BN/unused weights frozen. Resume every5 epochs.',
 'blend':'Predeclared .5*current Scattering486 + .5*standalone, equally for control and candidate. Entire current ensemble is blended; no weight search.',
 'decision':'Evaluate standalone and fixed blends against current and matched control; common16 AND remaining484, middle80/tails/noise/folds/new worst20. Both seeds consistent, mean0.30 RMSE gain for expansion priority, not significance; no automatic adoption.',
 'scope':'Only current fitting train images used for SSL; same repeatedly studied500, no independent holdout. No test images, external datasets, new weights, full500 final fit or submission changes.',
 'limitations':'SimMIM-inspired CNN adaptation, not original ViT/Swin reproduction. Reconstruction can learn brightness/noise; lower reconstruction loss does not establish better hardness prediction or denoising.',
 'sources':['https://arxiv.org/abs/2111.09886','https://github.com/microsoft/SimMIM'],
 'attribution':'NASA MicroNet; EfficientNet-PyTorch; existing baseline includes PriorLabs-TabPFN.'}


def unavailable(*args,**kwargs):raise ValueError('Missing completed SSL cell; implicit fitting forbidden')


def load_state():
    with contextlib.redirect_stdout(io.StringIO()):
        data,refs,old_plan=old.validate(old.OUT)
        features,meta=old.load_features(old.OUT,data,old_plan)
    archived={}
    for seed in SEEDS:
        sub=old.OUT/f'seed{seed}'
        if not old.completed(sub,old.evaluation_binding(old.OUT,seed,old_plan)):raise ValueError('Incomplete frozen MicroNet anchor')
        rows=core.align_rows(core.read_rows(sub/'oof.csv',{'ID','hardness','fold','micronet'}),data.ids,sub)
        refs[seed]['frozen_anchor']=np.array([float(r['micronet']) for r in rows])
        np.testing.assert_array_equal([float(r['hardness']) for r in rows],data.y)
        np.testing.assert_array_equal([int(r['fold']) for r in rows],refs[seed]['folds'])
        archived[str(seed)]={n:t.file_ref(sub/n) for n in ('completed.json','oof.csv','summary.json')}
    frozen={'protocol':PROTOCOL,'upstream':old_plan['frozen'],'environment':t.environment(),
        'common_failure_ids':old_plan['frozen']['common_failure_ids'],
        'features':t.file_ref(old.OUT/'features/features.npz'),'metadata':meta,'control_results':archived,
        'sources':{p.name:t.file_ref(p) for p in (Path(__file__),Path(eng.__file__),Path(eng.util.__file__),ROOT/'scripts/run_micronet_mim.ps1')}}
    return data,refs,features['micronet'],frozen


def prepare(folder):
    data,refs,x,frozen=load_state();path=folder/'protocol.json'
    if path.exists():
        if t.read_json(path)['frozen']!=frozen:raise ValueError('Frozen experiment changed')
        print('Preparation verified; no fitting.');return
    if folder.exists() and any(folder.iterdir()):raise ValueError('Nonempty unprepared folder')
    t.write_json(path,{'frozen':frozen,'created_unix':time.time(),'competition_fits':0,'competition_forwards':0})
    print('Prepared split-local masked reconstruction; no competition forwards or fitting.')
    for s in SEEDS:print(f'Scattering486 seed{s}: {core.metrics(data.y,refs[s]["current"])["rmse"]:.6f}')


def validate(folder):
    data,refs,x,frozen=load_state();plan=t.read_json(folder/'protocol.json')
    if plan['frozen']!=frozen or plan['competition_fits'] or plan['competition_forwards']:raise ValueError('Inputs, sources, environment or protocol changed')
    return data,refs,x,plan


def runtime(torch,device):
    return {'device':str(device),'gpu':torch.cuda.get_device_name(device) if device.type=='cuda' else None,
        'torch_cuda':torch.version.cuda,'precision':'float32','threads':torch.get_num_threads(),'batch':eng.BATCH,'accumulation':eng.ACCUM}


def verify_check(folder,plan):
    path=folder/'synthetic_check.json'
    if not path.exists():raise ValueError('Run check successfully before train')
    saved=t.read_json(path);r=saved['result']
    if saved['plan']!=t.digest(plan['frozen']) or saved['sha256']!=t.digest(r):raise ValueError('Synthetic check changed')
    if not all(r[k] for k in ('masked_values_hidden','clean_features_match','encoder_updated','bn_frozen','no_competition_images')):
        raise ValueError('Synthetic check incomplete')
    return r['runtime']


def check(folder,device_name):
    _,_,_,plan=validate(folder);torch,device=t.configure_device(device_name,1729);rt=runtime(torch,device)
    if (folder/'synthetic_check.json').exists():
        if verify_check(folder,plan)!=rt:raise ValueError('Checked runtime changed')
        print('Synthetic check verified; no competition training.');return
    rng=np.random.default_rng(1729);pixels=rng.integers(0,256,(16,256,256),dtype=np.uint8)
    model=eng.make_model(torch).to(device);initial=eng.snapshot(model)
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
    raw=eng.raw_batch(torch,pixels,np.arange(2),device);mask=torch.as_tensor(eng.masks(rng,2),device=device)
    a=eng.masked_input(torch,raw,mask);b=eng.masked_input(torch,raw*(1-mask)+(1-raw)*mask,mask)
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    model.eval()
    with torch.inference_mode():
        clean=eng.normalized(torch,raw)
        torch.testing.assert_close(model.features(clean),old.pool_maps(old.forward_maps(model.encoder,clean)),rtol=0,atol=0)
        if tuple(model(a).shape)!=(2,1,256,256):raise ValueError('Invalid reconstruction shape')
    opt=eng.optimizer_for(torch,model);grad={g:0. for g in eng.GROUPS}
    for _ in range(2):loss,steps=eng.optimize(torch,model,pixels,device,opt,rng,grad)
    audit=eng.audit(model,initial,2,grad)
    result={'runtime':rt,'masked_values_hidden':True,'clean_features_match':True,'encoder_updated':audit['encoder_updated'],
        'bn_frozen':audit['bn_buffers_unchanged'],'no_competition_images':True,'audit':audit,'masked_l1':loss,
        'peak_gpu_bytes':int(torch.cuda.max_memory_allocated(device)) if device.type=='cuda' else None}
    t.write_json(folder/'synthetic_check.json',{'plan':t.digest(plan['frozen']),'result':result,'sha256':t.digest(result)})
    print(base.canonical_json(result))


def validate_cell(result,folder,expected):
    epochs=expected['epochs'];available=[ep for ep in eng.EPOCHS if ep<=epochs]
    if (result['fitting_indices']!=expected['train_idx'] or result['query_indices']!=expected['query_idx']
        or result['hardness_labels_used'] or result['query_images_used_for_ssl']):raise ValueError('SSL data scope mismatch')
    required={f'features_epoch{ep}.npz' for ep in available}
    if expected['role']=='outer':required.add('model.pt')
    if set(result['artifacts'])!=required:raise ValueError('SSL artifacts incomplete')
    for n,h in result['artifacts'].items():
        if base.sha256_file(folder/n)!=h:raise ValueError('SSL artifact changed: '+n)
    if [r['epoch'] for r in result['records']]!=[0,*range(5,epochs+1,5)]:raise ValueError('SSL curve incomplete')
    arrays={}
    for row in result['records']:
        if row['epoch'] not in available:continue
        a=row['update_audit'];active=row['epoch']>0
        if (a['encoder_updated']!=active or not a['bn_buffers_unchanged'] or not a['unused_parameters_unchanged']
            or (a['initial_encoder_sha256']!=a['final_encoder_sha256'])!=active):raise ValueError('SSL update audit failed')
        for g in eng.GROUPS:
            if active and not (a['max_gradient_l2'][g]>0 and a['max_abs_parameter_change'][g]>0):raise ValueError('Missing SSL gradient/update')
            if not active and (a['max_gradient_l2'][g]!=0 or a['max_abs_parameter_change'][g]!=0):raise ValueError('Epoch0 not frozen')
        name=f'features_epoch{row["epoch"]}.npz'
        if row['feature_file']!=name:raise ValueError('Wrong feature filename')
        with np.load(folder/name,allow_pickle=False) as z:
            if set(z.files)!={'train','query'}:raise ValueError('Invalid feature schema')
            arrays[row['epoch']]={key:z[key].copy() for key in z.files}
        for key,indices in (('train',expected['train_idx']),('query',expected['query_idx'])):
            x=arrays[row['epoch']][key]
            if x.shape!=(len(indices),864) or x.dtype!=np.float32 or not np.isfinite(x).all():raise ValueError('Invalid features')
    if result['audit']!=result['records'][-1]['update_audit']:raise ValueError('Final audit mismatch')
    return arrays


def cell(folder,binding,ti,vi,epochs,seed,role,runner):
    if role not in ('inner','outer') or epochs not in eng.EPOCHS or np.intersect1d(ti,vi).size:raise ValueError('Invalid SSL cell scope')
    expected={'evaluation':binding,'train_idx':ti.tolist(),'query_idx':vi.tolist(),'epochs':epochs,'seed':seed,'role':role}
    folder.mkdir(parents=True,exist_ok=True);marker=folder/'binding.json';path=folder/'completed.json'
    if marker.exists() and t.read_json(marker)!=expected:raise ValueError('SSL cell binding changed')
    if path.exists():
        saved=t.read_json(path)
        if saved['binding']!=expected or saved['sha256']!=t.digest(saved['result']):raise ValueError('SSL completion changed')
        result=saved['result']
    else:
        if runner is unavailable:unavailable()
        t.write_json(marker,expected);result=runner(ti,vi,epochs,seed,folder,expected)
        validate_cell(result,folder,expected)
        t.write_json(path,{'binding':expected,'result':result,'sha256':t.digest(result)})
    arrays=validate_cell(result,folder,expected)
    if not folder.resolve().is_relative_to(ROOT):raise ValueError('Cell outside workspace')
    for name in ('progress.json','resume_0.pt','resume_1.pt'):
        disposable=folder/name
        if disposable.exists():disposable.unlink()
    return arrays,result


def ridge_predict(tx,yt,qx,alpha):
    model=core.make_estimator(alpha)
    model.fit(tx.astype(np.float64),yt)
    p=np.asarray(model.predict(qx.astype(np.float64)),dtype=np.float64)
    if p.shape!=(len(qx),) or not np.isfinite(p).all():raise ValueError('Invalid Ridge predictions')
    return p


def fit_fold(folder,binding,x0,y,folds,seed,fold,runner):
    ti,vi=np.flatnonzero(folds!=fold),np.flatnonzero(folds==fold)
    if (len(ti),len(vi))!=(400,100):raise ValueError('Expected400/100 outer')
    yt=y[ti];splits=list(StratifiedKFold(3,shuffle=True,random_state=seed+fold).split(yt,core.rank_groups(yt,10)))
    path=folder/f'fold{fold}.json'
    fold_binding={'evaluation':binding,'seed':seed,'fold':fold,'train_idx':ti.tolist(),'query_idx':vi.tolist(),'train_y':base.sha256_array(yt)}
    if path.exists():
        saved=t.read_json(path);result=saved['result']
        if saved['binding']!=fold_binding or saved['sha256']!=t.digest(result):raise ValueError('Fold result/binding changed')
        candidates=result['candidates'];key=lambda c:(c['inner_rmse'],c['epoch'],-c['alpha'])
        if ([(c['epoch'],c['alpha']) for c in candidates]!=[(ep,a) for ep in eng.EPOCHS for a in ALPHAS]
            or any(not np.isfinite(c['inner_rmse']) or c['inner_rmse']<0 for c in candidates)):raise ValueError('Invalid search scores')
        if result['selection']!={'frozen_ridge':min([c for c in candidates if c['epoch']==0],key=key),'mim_nested':min(candidates,key=key)}:
            raise ValueError('Selection does not match inner scores')
        for k,(it,iv) in enumerate(splits):
            arrays,run=cell(folder/'cells'/f'fold{fold}'/f'inner{k}',binding,ti[it],ti[iv],80,seed+1009*fold+k,'inner',unavailable)
            if run['audit']!=result['inner_audits'][k]:raise ValueError('Inner audit mismatch')
            np.testing.assert_allclose(arrays[0]['train'],x0[ti[it]],rtol=2e-5,atol=3e-4)
            np.testing.assert_allclose(arrays[0]['query'],x0[ti[iv]],rtol=2e-5,atol=3e-4)
        selected=result['selection']['mim_nested']
        _,run=cell(folder/'cells'/f'fold{fold}'/'refit',binding,ti,vi,selected['epoch'],seed+1009*fold+9,'outer',unavailable)
        if run['audit']!=result['outer_audit'] or result['validation_indices']!=vi.tolist():raise ValueError('Outer audit/scope mismatch')
        if set(result['predictions'])!=set(MODES):raise ValueError('Wrong prediction arms')
        for values in result['predictions'].values():
            if np.asarray(values).shape!=(100,) or not np.isfinite(values).all():raise ValueError('Invalid outer predictions')
        if selected['epoch']==0 and result['predictions']['mim_nested']!=result['predictions']['frozen_ridge']:raise ValueError('Epoch0 fallback differs from control')
        return result
    scores={(ep,alpha):0. for ep in eng.EPOCHS for alpha in ALPHAS};inner_audits=[]
    for k,(it,iv) in enumerate(splits):
        arrays,run=cell(folder/'cells'/f'fold{fold}'/f'inner{k}',binding,ti[it],ti[iv],80,seed+1009*fold+k,'inner',runner)
        # Immutable epoch0 uses the archived feature values for an exact control;
        # fresh epoch0 extraction checks architecture/input identity within FP32 tolerance.
        np.testing.assert_allclose(arrays[0]['train'],x0[ti[it]],rtol=2e-5,atol=3e-4)
        np.testing.assert_allclose(arrays[0]['query'],x0[ti[iv]],rtol=2e-5,atol=3e-4)
        for ep in eng.EPOCHS:
            tx,qx=(x0[ti[it]],x0[ti[iv]]) if ep==0 else (arrays[ep]['train'],arrays[ep]['query'])
            for alpha in ALPHAS:
                pred=ridge_predict(tx,yt[it],qx,alpha)
                scores[ep,alpha]+=float(np.sum((pred-yt[iv])**2))
        inner_audits.append(run['audit'])
    candidates=[{'epoch':ep,'alpha':alpha,'inner_rmse':float(np.sqrt(sse/400))} for (ep,alpha),sse in scores.items()]
    key=lambda c:(c['inner_rmse'],c['epoch'],-c['alpha'])
    control=min([c for c in candidates if c['epoch']==0],key=key);chosen=min(candidates,key=key)
    arrays,run=cell(folder/'cells'/f'fold{fold}'/'refit',binding,ti,vi,chosen['epoch'],seed+1009*fold+9,'outer',runner)
    np.testing.assert_allclose(arrays[0]['train'],x0[ti],rtol=2e-5,atol=3e-4)
    np.testing.assert_allclose(arrays[0]['query'],x0[vi],rtol=2e-5,atol=3e-4)
    control_pred=ridge_predict(x0[ti],yt,x0[vi],control['alpha'])
    tx,qx=(x0[ti],x0[vi]) if chosen['epoch']==0 else (arrays[chosen['epoch']]['train'],arrays[chosen['epoch']]['query'])
    pred=ridge_predict(tx,yt,qx,chosen['alpha'])
    result={'validation_indices':vi.tolist(),'predictions':{'frozen_ridge':control_pred.tolist(),'mim_nested':pred.tolist()},
        'selection':{'frozen_ridge':control,'mim_nested':chosen},'candidates':candidates,'inner_audits':inner_audits,'outer_audit':run['audit']}
    if runner is unavailable:unavailable()
    t.write_json(path,{'binding':fold_binding,'result':result,'sha256':t.digest(result)})
    if runner is not unavailable:print(f'Outer{fold}: SSL epoch={chosen["epoch"]}, alpha={chosen["alpha"]:g}; inner={chosen["inner_rmse"]:.6f}; outer labels unused',flush=True)
    return result


def collect(folder,binding,data,ref,x0,seed,runner):
    pred={n:np.full(500,np.nan) for n in MODES};selections=[]
    with threadpool_limits(limits=2):
        for f in range(5):
            result=fit_fold(folder,binding,x0,data.y,ref['folds'],seed,f,runner)
            for n in MODES:pred[n][result['validation_indices']]=result['predictions'][n]
            selections.append({'fold':f,'choices':result['selection'],'audit':result['outer_audit']})
    np.testing.assert_allclose(pred['frozen_ridge'],ref['frozen_anchor'],rtol=0,atol=1e-9)
    return pred,selections


def binding_for(plan,rt,seed):return {'plan':t.digest(plan['frozen']),'runtime':rt,'seed':seed}

def result_data(data,ref,predictions,common_ids):
    output={'current':ref['current'],**predictions,CONTROL_BLEND:.5*(ref['current']+predictions['frozen_ridge']),
            CANDIDATE_BLEND:.5*(ref['current']+predictions['mim_nested'])}
    if set(predictions)!=set(MODES) or any(p.shape!=(500,) or not np.isfinite(p).all() for p in output.values()):raise ValueError('Invalid OOF')
    metrics={n:core.metrics(data.y,p) for n,p in output.items()}
    for n,m in metrics.items():
        control=CONTROL_BLEND if n.endswith('_equal') else 'frozen_ridge'
        m.update(gain_vs_current=metrics['current']['rmse']-m['rmse'],matched_control=control,gain_vs_control=metrics[control]['rmse']-m['rmse'])
    common=np.isin(data.ids,common_ids);d=core.rank_groups(data.y,10);q=core.rank_groups(data.noise,4)
    if common.sum()!=16:raise ValueError('Historical common16 changed')
    masks={'all':np.ones(500,bool),'common16':common,'remaining484':~common,'middle80':(d>0)&(d<9),'low10':d==0,'high10':d==9}
    masks.update({f'noiseQ{k+1}':q==k for k in range(4)});masks.update({f'fold{k}':ref['folds']==k for k in range(5)})
    worst={n:set(np.argsort((p-data.y)**2,kind='stable')[-20:]) for n,p in output.items()}
    for n in output:masks[n+'_worst20']=np.isin(np.arange(500),list(worst[n]))
    diag=[{'model':n,'group':g,**core.metrics(data.y[m],p[m])} for n,p in output.items() for g,m in masks.items()]
    rows=[{'ID':str(i),'hardness':float(data.y[j]),'fold':int(ref['folds'][j]),**{n:float(p[j]) for n,p in output.items()}} for j,i in enumerate(data.ids)]
    errors=[{'ID':str(data.ids[j]),'hardness':float(data.y[j]),'fold':int(ref['folds'][j]),'common16':bool(common[j]),
        **{n+'_error':float(p[j]-data.y[j]) for n,p in output.items()}} for j in np.argsort(-(predictions['mim_nested']-data.y)**2,kind='stable')]
    detail={}
    for n in ('frozen_ridge','mim_nested',CONTROL_BLEND,CANDIDATE_BLEND):
        err=output[n]-data.y;old_err=ref['current']-data.y
        detail[n]={'common16_rmse':{k:core.metrics(data.y[common],p[common])['rmse'] for k,p in output.items()},
          'remaining484_rmse':{k:core.metrics(data.y[~common],p[~common])['rmse'] for k,p in output.items()},
          'common16_improved':int(((err**2<old_err**2)&common).sum()),'common16_same_error_sign':int((np.sign(err[common])==np.sign(old_err[common])).sum()),
          'common16_in_worst20':int(common[list(worst[n])].sum()),'common16_sse_fraction':float((err[common]**2).sum()/(err**2).sum()) if np.any(err) else 0.,
          'new_worst20_ids':[str(data.ids[i]) for i in sorted(worst[n]-worst['current'])],
          'left_worst20_ids':[str(data.ids[i]) for i in sorted(worst['current']-worst[n])],
          'residual_correlation_vs_current':core.correlation(err,old_err)}
    return metrics,rows,diag,errors,detail


def expected_outputs(folder):
    names={'summary.json','oof.csv','diagnostics.csv','per_image_errors.csv','common16.csv','report.txt'}|{f'fold{f}.json' for f in range(5)}
    for f in range(5):
        for role in ('inner0','inner1','inner2','refit'):
            prefix=f'cells/fold{f}/{role}'
            names.update({prefix+'/binding.json',prefix+'/completed.json'})
            saved=t.read_json(folder/prefix/'completed.json')
            for name in saved['result']['artifacts']:
                if Path(name).name!=name:raise ValueError('Unsafe artifact name')
                names.add(prefix+'/'+name)
    return names


def completed(folder,binding):
    path=folder/'completed_evaluation.json'
    if not path.exists():return False
    done=t.read_json(path)
    if done['binding']!=binding or set(done['files'])!=expected_outputs(folder):raise ValueError('Evaluation manifest changed')
    for name,h in done['files'].items():
        if base.sha256_file(folder/name)!=h:raise ValueError('Completed artifact changed: '+name)
    return True


def save_results(folder,binding,data,ref,pred,selections,plan):
    m,rows,diag,errors,detail=result_data(data,ref,pred,plan['frozen']['common_failure_ids'])
    for name,values in (('oof.csv',rows),('diagnostics.csv',diag),('per_image_errors.csv',errors),('common16.csv',[r for r in errors if r['common16']])):t.write_csv(folder/name,values)
    t.write_json(folder/'summary.json',{'binding':binding,'metrics':m,'diagnosis':detail,'selection':selections,
        'primary':PRIMARY,'secondary':CANDIDATE_BLEND,'frozen_anchor_matched':True,'test_accessed':False,'independent_holdout':False})
    lines=[f'Split-local MicroNet masked reconstruction: seed{binding["seed"]}',PROTOCOL['blend'],'']
    lines += [f'{n}: RMSE={v["rmse"]:.6f}; gain vs current={v["gain_vs_current"]:+.6f}; gain vs {v["matched_control"]}={v["gain_vs_control"]:+.6f}' for n,v in m.items()]
    lines += ['',f'Outer refits selecting SSL epoch>0: {sum(s["audit"]["encoder_updated"] for s in selections)}/5.',
        'Original frozen MicroNet Ridge OOF anchor matched.',PROTOCOL['decision'],PROTOCOL['scope'],PROTOCOL['limitations']]
    (folder/'report.txt').write_text('\n'.join(lines)+'\n',encoding='utf8')
    t.write_json(folder/'completed_evaluation.json',{'binding':binding,'files':{n:base.sha256_file(folder/n) for n in sorted(expected_outputs(folder))}})
    print('\n'.join(lines[3:8]),flush=True);print(f'Report: {folder/"report.txt"}')


def verify_results(folder,binding,data,ref,x0,plan,seed):
    if not completed(folder,binding):raise ValueError('Incomplete evaluation')
    pred,selection=collect(folder,binding,data,ref,x0,seed,unavailable)
    metrics,rows,_,_,detail=result_data(data,ref,pred,plan['frozen']['common_failure_ids'])
    saved=t.read_json(folder/'summary.json')
    if saved['metrics']!=metrics or saved['selection']!=selection or saved['diagnosis']!=detail:raise ValueError('Recomputed results differ')
    actual=core.align_rows(core.read_rows(folder/'oof.csv',set(rows[0])),data.ids,folder)
    for n in rows[0]:
        if n!='ID':np.testing.assert_array_equal([float(r[n]) for r in actual],[float(r[n]) for r in rows])
    return metrics,detail,selection


def train(folder,seed,device_name):
    data,refs,x0,plan=validate(folder);rt=verify_check(folder,plan)
    sub=folder/f'seed{seed}';binding=binding_for(plan,rt,seed)
    if completed(sub,binding):
        verify_results(sub,binding,data,refs[seed],x0,plan,seed);print('Completed seed verified; no refitting.');return
    torch,device=t.configure_device(device_name,seed)
    if runtime(torch,device)!=rt:raise ValueError('Use the checked device/runtime; no automatic fallback')
    from PIL import Image
    pixels=[]
    for ident in data.ids:
        with Image.open(ROOT/'open/train'/f'{ident}.png') as im:
            if im.mode!='L' or im.size!=(256,256):raise ValueError('Unexpected image format')
            pixels.append(np.array(im,dtype=np.uint8))
    runner=eng.Runner(torch,device,np.stack(pixels),x0)
    pred,selections=collect(sub,binding,data,refs[seed],x0,seed,runner)
    validate(folder);save_results(sub,binding,data,refs[seed],pred,selections,plan)


def compare(folder):
    data,refs,x0,plan=validate(folder);rt=verify_check(folder,plan);records=[]
    for seed in SEEDS:
        m,d,s=verify_results(folder/f'seed{seed}',binding_for(plan,rt,seed),data,refs[seed],x0,plan,seed)
        records.append({'seed':seed,'current_rmse':m['current']['rmse'],'standalone':m[PRIMARY],'frozen_control':m['frozen_ridge'],
            'mim_blend':m[CANDIDATE_BLEND],'control_blend':m[CONTROL_BLEND],'diagnosis':d,
            'ssl_selected_folds':sum(x['audit']['encoder_updated'] for x in s),'selected_epochs':[x['choices'][PRIMARY]['epoch'] for x in s]})
    tests={}
    for key in ('standalone','mim_blend'):
        gain=float(np.mean([r[key]['gain_vs_current'] for r in records]));paired=all(r[key]['gain_vs_control']>0 for r in records)
        positive=paired and all(r[key]['gain_vs_current']>0 for r in records)
        tests[key]={'mean_gain_vs_current':gain,'mean_gain_vs_control':float(np.mean([r[key]['gain_vs_control'] for r in records])),
            'all_seeds_improve_paired_control':paired,'all_seeds_improve_both':positive,'passes_expansion_priority':positive and gain>=.30}
    result={'primary':PRIMARY,'secondary':CANDIDATE_BLEND,'per_seed':records,'comparisons':tests,
        'additional_review_required':'All500, common16 AND remaining484, tails/noise/new worst20, reconstruction loss vs regression performance. No automatic adoption.',
        'aggregation':'Mean per-seed differences; same500, no prediction averaging or independent inference.','protocol_sha256':t.digest(plan['frozen'])}
    t.write_json(folder/'comparison.json',result);print(base.canonical_json(result));return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('prepare','validate','check','train','compare'))
    p.add_argument('--seed',type=int,choices=SEEDS,default=42);p.add_argument('--device',default='cuda');p.add_argument('--output-dir',type=base.project_path,default=OUT)
    a=p.parse_args(argv)
    if a.command=='prepare':prepare(a.output_dir)
    elif a.command=='validate':validate(a.output_dir);print('Inputs/protocol/anchors verified; no fitting.')
    elif a.command=='check':check(a.output_dir,a.device)
    elif a.command=='train':train(a.output_dir,a.seed,a.device)
    else:compare(a.output_dir)


if __name__=='__main__':main()

