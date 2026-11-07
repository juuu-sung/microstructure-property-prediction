"""User-run MicroNet-B0 last-two-stage fine-tuning versus frozen encoder control.

prepare/validate/compare only verify archives. check trains on synthetic images.
train performs paired5outer x3inner evaluation; never trains on test images.
"""
from __future__ import annotations
import argparse
import contextlib
import io
import math
from pathlib import Path
import time
import numpy as np
from sklearn.model_selection import StratifiedKFold

if __package__:
    from . import micronet_finetune_engine as eng
else:
    import micronet_finetune_engine as eng
old,t,base=eng.old,eng.t,eng.base
core,ROOT=old.core,old.ROOT
OUT=ROOT/'reports/micronet_finetune'
SEEDS=(42,2026)
MODES=eng.MODES
PRIMARY='partial'
CONTROL_BLEND,PARTIAL_BLEND='current_head_equal','current_partial_equal'
PROTOCOL={
 'version':1,'seeds':list(SEEDS),'primary':PRIMARY,'secondary':PARTIAL_BLEND,
 'input':'Native256 whole image, grayscale repeated3, fixed ImageNet mean/std. No crop, patches, resize, denoise, augmentation or TTA.',
 'encoder':'Pinned NASA MicroNet EfficientNet-B0; EfficientNet-PyTorch0.7.1. Semantic stage6=blocks11..14(192ch); stage7=block15(320ch).',
 'pooling':'Unchanged block10 and block15 spatial mean/population std,864 features. No PCA/scaler/feature search.',
 'head':'LayerNorm864 -> Dropout0.1 -> Linear864,1; same seeded default initialization in both modes.',
 'modes':{'head_only':'All encoder parameters frozen.', 'partial':'After5 head-only warmup epochs, unfreeze only blocks11..15, including their BN affine parameters; BN running statistics remain fixed.'},
 'training':{'max_epochs':40,'checkpoints':list(eng.EPOCHS),'warmup_epochs':5,
    'head_lr_warmup':.001,'head_lr_after':.0003,'stage6_stage7_lr':1e-5,'optimizer':'AdamW reset in BOTH arms at epoch6',
    'weight_decay':.01,'no_decay':'Bias and1D norm weights','schedule':'Fixed40 horizon; after warmup cosine from1 at epoch6 to0.1 at epoch40; outer refit uses identical prefix.',
    'loss':'Unweighted image MSE; target population mean/std fitted only on fitting labels.',
    'batch':8,'accumulation':2,'effective_batch':16,'last_accumulation':'Normalize by actual number of samples in each optimizer step',
    'clip_norm':1.,'precision':'FP32; no AMP/TF32; deterministic; no backbone drop-connect; BN eval'},
 'validation':'Existing5 outer400/100 x3 inner stratified rank-deciles within outer train; inner seed=seed+outer_fold. '
              'Choose each arm epoch by pooled3-inner MSE; ties earlier epoch. Outer100 labels never passed to training/selection.',
 'run_seed':'seed+1009*outer_fold+inner_fold; outer refit uses+9; paired between arms.',
 'work':'60 inner40epoch trajectories +20 outer selected-epoch refits across both seeds. Epoch0 allowed; selected<=5 explicitly means no encoder update.',
 'audit':'Per-stage finite nonzero gradients AND parameter changes; frozen weights and ALL BN buffers unchanged. Record at every saved epoch.',
 'blend':'Predeclared secondary=.5*current Scattering486+.5*new standalone; same formula for frozen-head control. No weight search.',
 'decision':'Report standalone partial vs frozen-head and current Scattering486 first; then both fixed50:50 blends. '
            'Both seeds must support gains before promotion;0.30 mean RMSE gain is expansion priority, not significance. '
            'Review common16/remaining484, all500, tails/noise/new worst20; no automatic adoption.',
 'scope':'Same repeatedly studied500; no independent holdout, test access, full500 supervised fit, new pretrained weights or submission edits.',
 'attribution':'NASA MicroNet, EfficientNet-PyTorch; Built with PriorLabs-TabPFN (archived baseline).'}


def unavailable(*args,**kwargs):raise ValueError('Missing completed training cell; implicit fitting forbidden.')


def load_state():
    with contextlib.redirect_stdout(io.StringIO()):data,refs,upstream=old.load_state()
    frozen={'protocol':PROTOCOL,'upstream':upstream,'upstream_sha256':t.digest(upstream),'environment':t.environment(),
      'sources':{p.name:t.file_ref(p) for p in (Path(__file__),Path(eng.__file__),ROOT/'scripts/run_micronet_finetune.ps1')}}
    return data,refs,frozen


def prepare(folder):
    data,refs,frozen=load_state();path=folder/'protocol.json'
    if path.exists():
        if t.read_json(path)['frozen']!=frozen:raise ValueError('Frozen experiment changed; preserve existing artifacts.')
        print('Preparation verified; no model fitting or forwards.');return
    if folder.exists() and any(folder.iterdir()):raise ValueError('Nonempty unprepared folder')
    t.write_json(path,{'frozen':frozen,'created_unix':time.time(),'competition_fits':0,'competition_forwards':0})
    print('Prepared native256 MicroNet partial vs head-only;80 trajectories across2 seeds.')
    for s in SEEDS:print(f'Current Scattering486 seed{s}: {core.metrics(data.y,refs[s]["current"])["rmse"]:.6f}')


def validate(folder):
    data,refs,frozen=load_state();plan=t.read_json(folder/'protocol.json')
    if plan['frozen']!=frozen or plan['competition_fits'] or plan['competition_forwards']:raise ValueError('Protocol, assets, inputs or source changed')
    return data,refs,plan


def runtime(torch,device):
    return {'device':str(device),'gpu':torch.cuda.get_device_name(device) if device.type=='cuda' else None,
            'torch_cuda':torch.version.cuda,'precision':'float32','threads':torch.get_num_threads(),'batch':eng.BATCH,'accumulation':eng.ACCUM}


def verify_check(folder,plan):
    path=folder/'synthetic_check.json'
    if not path.exists():raise ValueError('Run check successfully before train.')
    saved=t.read_json(path)
    if saved['plan']!=t.digest(plan['frozen']) or saved['result_sha256']!=t.digest(saved['result']):raise ValueError('Synthetic check binding changed')
    r=saved['result']
    if set(r['audits'])!=set(MODES) or not r['audits']['partial']['encoder_updated'] or r['audits']['head_only']['encoder_updated']:
        raise ValueError('Synthetic gradient/update check incomplete')
    return r['runtime']


def check(folder,device_name):
    _,_,plan=validate(folder)
    torch,device=t.configure_device(device_name,1729)
    rt=runtime(torch,device)
    if (folder/'synthetic_check.json').exists():
        if verify_check(folder,plan)!=rt:raise ValueError('Check runtime changed; do not mix devices.')
        print('Synthetic GPU/update check already verified.');return
    rng=np.random.default_rng(1729)
    pixels=rng.integers(0,256,size=(16,256,256),dtype=np.uint8)
    y=rng.normal(size=16).astype(np.float64)
    audits={};initial_head=[];peak={}
    for mode in MODES:
        torch.manual_seed(1729)
        if device.type=='cuda':torch.cuda.manual_seed_all(1729);torch.cuda.reset_peak_memory_stats(device)
        model=eng.make_model(torch).to(device);initial=eng.snapshot(model)
        initial_head.append(eng.state_hash({n:v for n,v in initial.items() if n.startswith('head.')}))
        grad={g:0. for g in ('head','stage6','stage7')}
        for epoch in (1,6):
            opt=eng.optimizer_for(torch,model,mode,epoch)
            loss,steps=eng.optimize(torch,model,pixels,np.arange(16),y,device,opt,np.random.default_rng(1729+epoch),grad)
        audits[mode]=eng.audit(model,initial,mode,6,grad)
        peak[mode]=int(torch.cuda.max_memory_allocated(device)) if device.type=='cuda' else None
        pred=eng.predict(torch,model,pixels,np.arange(16),device,0.,1.)
        if pred.shape!=(16,):raise ValueError('Synthetic output shape differs')
        print(f'Synthetic {mode}: gradients/updates/frozen weights/BN verified; peak bytes={peak[mode]}',flush=True)
        del model,opt,initial
        if device.type=='cuda':torch.cuda.empty_cache()
    if len(set(initial_head))!=1:raise ValueError('Paired head initialization differs')
    result={'runtime':rt,'audits':audits,'paired_head_initialization':True,'peak_gpu_bytes':peak,
      'competition_images_used':False,'note':'Two synthetic optimizer steps at phase1/6, not performance training.'}
    t.write_json(folder/'synthetic_check.json',{'plan':t.digest(plan['frozen']),'result':result,'result_sha256':t.digest(result)})


def cell(folder,binding,ti,yt,vi,yv,mode,epochs,seed,runner):
    expected={'evaluation':binding,'train_idx':ti.tolist(),'query_idx':vi.tolist(),
      'train_y_sha256':base.sha256_array(yt),'query_y_sha256':None if yv is None else base.sha256_array(yv),
      'mode':mode,'epochs':epochs,'seed':seed}
    folder.mkdir(parents=True,exist_ok=True)
    marker=folder/'binding.json'
    if marker.exists() and t.read_json(marker)!=expected:raise ValueError('Training cell binding changed')
    path=folder/'completed.json'
    if path.exists():
        saved=t.read_json(path)
        if saved['binding']!=expected or saved['result_sha256']!=t.digest(saved['result']):raise ValueError('Training cell result changed')
        result=saved['result']
    else:
        if runner is unavailable:unavailable()
        t.write_json(marker,expected)
        result=runner(ti,yt,vi,yv,mode,epochs,seed,folder,expected)
        validate_run(result,folder,yt,yv,len(vi),mode,epochs)
        t.write_json(path,{'binding':expected,'result':result,'result_sha256':t.digest(result)})
    validate_run(result,folder,yt,yv,len(vi),mode,epochs)
    # Completion preserves predictions/audits (and outer model). Only this cell's
    # disposable optimizer checkpoints can now be removed; interrupted cells keep them.
    if not folder.resolve().is_relative_to(ROOT):raise ValueError('Cell outside workspace')
    for name in ('progress.json','resume_0.pt','resume_1.pt'):
        temporary=folder/name
        if temporary.exists():temporary.unlink()
    return result


def validate_run(result,folder,yt,yv,nq,mode,epochs):
    if result['target_mean']!=float(np.mean(yt)) or result['target_std']!=max(float(np.std(yt)),1e-8):raise ValueError('Target scaling not training-only')
    records=result['records']
    if [r['epoch'] for r in records]!=[ep for ep in eng.EPOCHS if ep<=epochs]:raise ValueError('Checkpoint curve incomplete')
    for r in records:
        p=np.asarray(r['predictions']);a=r['update_audit']
        if p.shape!=(nq,) or not np.isfinite(p).all() or r['train_n']!=len(yt) or not np.isfinite(r['train_sse']) or r['train_sse']<0:
            raise ValueError('Invalid curve predictions/metrics')
        if yv is None:
            if 'valid_sse' in r or 'valid_n' in r:raise ValueError('Outer query labels used')
        elif r['valid_n']!=len(yv) or not np.isclose(r['valid_sse'],float(np.sum((p-yv)**2)),rtol=1e-12,atol=1e-10):
            raise ValueError('Internal validation metric differs')
        active=mode=='partial' and r['epoch']>eng.WARMUP
        if a['encoder_updated']!=active or not a['bn_and_other_buffers_unchanged'] or not a['frozen_parameters_unchanged']:
            raise ValueError('Encoder/BN audit failed')
        for g in ('stage6','stage7'):
            if active and not (a['max_abs_parameter_change'][g]>0 and a['max_gradient_l2'][g]>0):raise ValueError('Missing stage update evidence')
            if not active and (a['max_abs_parameter_change'][g]!=0 or a['max_gradient_l2'][g]!=0):raise ValueError('Unexpected encoder update')
    if result['audit']!=records[-1]['update_audit']:raise ValueError('Final update audit mismatch')
    checkpoint=result['checkpoint']
    if (yv is None)!=(checkpoint is not None):raise ValueError('Missing outer checkpoint')
    if checkpoint and (checkpoint['name']!='model.pt' or base.sha256_file(folder/'model.pt')!=checkpoint['sha256']):raise ValueError('Outer checkpoint changed')


def select_epoch(runs):
    curves=[]
    for ep in eng.EPOCHS:
        rs=[next(r for r in run['records'] if r['epoch']==ep) for run in runs]
        n=sum(r['valid_n'] for r in rs)
        if n!=400:raise ValueError('Expected pooled inner400 predictions')
        curves.append({'epoch':ep,'inner_rmse':math.sqrt(sum(r['valid_sse'] for r in rs)/n),
            'train_rmse':math.sqrt(sum(r['train_sse'] for r in rs)/sum(r['train_n'] for r in rs))})
    return min(curves,key=lambda r:(r['inner_rmse'],r['epoch'])),curves


def fit_fold(folder,binding,y,folds,seed,fold,runner):
    ti,vi=np.flatnonzero(folds!=fold),np.flatnonzero(folds==fold)
    if (len(ti),len(vi))!=(400,100):raise ValueError('Expected outer400/100')
    yt=y[ti]
    splits=list(StratifiedKFold(3,shuffle=True,random_state=seed+fold).split(yt,core.rank_groups(yt,10)))
    result={'validation_indices':vi.tolist(),'predictions':{},'selection':{},'curves':{},'refit_audits':{}}
    for mode in MODES:
        inner=[]
        for k,(it,iv) in enumerate(splits):
            inner.append(cell(folder/'cells'/f'fold{fold}'/mode/f'inner{k}',binding,ti[it],yt[it],ti[iv],yt[iv],
                              mode,40,seed+1009*fold+k,runner))
        chosen,curves=select_epoch(inner)
        # Outer labels are deliberately not an argument to the runner.
        refit=cell(folder/'cells'/f'fold{fold}'/mode/'refit',binding,ti,yt,vi,None,mode,chosen['epoch'],seed+1009*fold+9,runner)
        result['predictions'][mode]=refit['records'][-1]['predictions']
        result['selection'][mode]=chosen;result['curves'][mode]=curves;result['refit_audits'][mode]=refit['audit']
        if runner is not unavailable:
            state='ENCODER UPDATED' if refit['audit']['encoder_updated'] else 'NO ENCODER UPDATE'
            print(f'Outer{fold}: {mode} selected epoch={chosen["epoch"]}; pooled inner={chosen["inner_rmse"]:.6f} [{state}]',flush=True)
    path=folder/f'fold{fold}.json'
    if path.exists():
        if t.read_json(path)!=result:raise ValueError('Fold summary differs from verified training cells')
    elif runner is unavailable:unavailable()
    else:t.write_json(path,result)
    return result


def collect(folder,binding,data,ref,seed,runner):
    predictions={n:np.full(500,np.nan) for n in MODES};selections=[]
    for fold in range(5):
        result=fit_fold(folder,binding,data.y,ref['folds'],seed,fold,runner)
        for mode in MODES:predictions[mode][result['validation_indices']]=result['predictions'][mode]
        selections.append({'fold':fold,'choices':result['selection'],'audits':result['refit_audits']})
    return predictions,selections


def result_data(data,ref,predictions,common_ids):
    output={'current':ref['current'],**predictions,CONTROL_BLEND:.5*(ref['current']+predictions['head_only']),
            PARTIAL_BLEND:.5*(ref['current']+predictions['partial'])}
    if set(predictions)!=set(MODES) or any(p.shape!=(500,) or not np.isfinite(p).all() for p in output.values()):raise ValueError('Invalid OOF')
    metrics={n:core.metrics(data.y,p) for n,p in output.items()}
    for n,m in metrics.items():
        control=CONTROL_BLEND if n.endswith('_equal') else 'head_only'
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
        **{n+'_error':float(p[j]-data.y[j]) for n,p in output.items()}} for j in np.argsort(-(predictions['partial']-data.y)**2,kind='stable')]
    detail={}
    for n in ('head_only','partial',CONTROL_BLEND,PARTIAL_BLEND):
        err=output[n]-data.y;old_err=ref['current']-data.y
        detail[n]={'common16_rmse':{k:core.metrics(data.y[common],p[common])['rmse'] for k,p in output.items()},
          'remaining484_rmse':{k:core.metrics(data.y[~common],p[~common])['rmse'] for k,p in output.items()},
          'common16_improved':int(((err**2<old_err**2)&common).sum()),'common16_same_error_sign':int((np.sign(err[common])==np.sign(old_err[common])).sum()),
          'common16_in_worst20':int(common[list(worst[n])].sum()),'common16_sse_fraction':float((err[common]**2).sum()/(err**2).sum()) if np.any(err) else 0.,
          'new_worst20_ids':[str(data.ids[i]) for i in sorted(worst[n]-worst['current'])],
          'left_worst20_ids':[str(data.ids[i]) for i in sorted(worst['current']-worst[n])],
          'residual_correlation_vs_current':core.correlation(err,old_err)}
    return metrics,rows,diag,errors,detail


def binding_for(plan,rt,seed):return {'plan':t.digest(plan['frozen']),'runtime':rt,'seed':seed}


def expected_outputs():
    names={'summary.json','oof.csv','diagnostics.csv','per_image_errors.csv','common16.csv','report.txt'}|{f'fold{f}.json' for f in range(5)}
    for f in range(5):
        for mode in MODES:
            for role in ('inner0','inner1','inner2','refit'):
                prefix=f'cells/fold{f}/{mode}/{role}'
                names.update({prefix+'/binding.json',prefix+'/completed.json'})
                if role=='refit':names.add(prefix+'/model.pt')
    return names


def completed(folder,binding):
    path=folder/'completed_evaluation.json'
    if not path.exists():return False
    done=t.read_json(path)
    if done['binding']!=binding or set(done['files'])!=expected_outputs():raise ValueError('Result manifest/binding changed')
    for name,h in done['files'].items():
        if base.sha256_file(folder/name)!=h:raise ValueError('Completed artifact changed: '+name)
    return True


def save_results(folder,binding,data,ref,pred,selections,plan):
    m,rows,diag,errors,detail=result_data(data,ref,pred,plan['frozen']['upstream']['common_failure_ids'])
    for name,values in (('oof.csv',rows),('diagnostics.csv',diag),('per_image_errors.csv',errors),('common16.csv',[r for r in errors if r['common16']])):t.write_csv(folder/name,values)
    t.write_json(folder/'summary.json',{'binding':binding,'metrics':m,'diagnosis':detail,'selection':selections,'primary':PRIMARY,'secondary':PARTIAL_BLEND,'test_accessed':False,'independent_holdout':False})
    lines=[f'MicroNet last2 stages vs head-only: seed{binding["seed"]}','Primary=standalone partial; secondary=current_partial_equal.',PROTOCOL['blend'],'']
    lines += [f'{n}: RMSE={v["rmse"]:.6f}; gain vs current={v["gain_vs_current"]:+.6f}; gain vs {v["matched_control"]}={v["gain_vs_control"]:+.6f}' for n,v in m.items()]
    lines += ['',f'Outer partial refits with encoder updates: {sum(s["audits"]["partial"]["encoder_updated"] for s in selections)}/5.',
              PROTOCOL['audit'],PROTOCOL['decision'],PROTOCOL['scope'],'Inspect common16.csv / diagnostics.csv / per_image_errors.csv and fold*.json.']
    (folder/'report.txt').write_text('\n'.join(lines)+'\n',encoding='utf8')
    t.write_json(folder/'completed_evaluation.json',{'binding':binding,'files':{n:base.sha256_file(folder/n) for n in sorted(expected_outputs())}})
    print('\n'.join(lines[4:9]),flush=True);print(f'Report: {folder/"report.txt"}')


def verify_results(folder,binding,data,ref,plan,seed):
    if not completed(folder,binding):raise ValueError('Incomplete evaluation')
    pred,selection=collect(folder,binding,data,ref,seed,unavailable)
    metrics,rows,_,_,detail=result_data(data,ref,pred,plan['frozen']['upstream']['common_failure_ids'])
    saved=t.read_json(folder/'summary.json')
    if saved['metrics']!=metrics or saved['selection']!=selection or saved['diagnosis']!=detail:raise ValueError('Recomputed results differ')
    actual=core.align_rows(core.read_rows(folder/'oof.csv',set(rows[0])),data.ids,folder)
    for n in rows[0]:
        if n!='ID':np.testing.assert_array_equal([float(r[n]) for r in actual],[float(r[n]) for r in rows])
    return metrics,detail,selection


def train(folder,seed,device_name):
    data,refs,plan=validate(folder);rt=verify_check(folder,plan)
    sub=folder/f'seed{seed}';binding=binding_for(plan,rt,seed)
    if completed(sub,binding):verify_results(sub,binding,data,refs[seed],plan,seed);print('Completed seed verified; no training.');return
    torch,device=t.configure_device(device_name,seed)
    if runtime(torch,device)!=rt:raise ValueError('Use the checked device/runtime; no automatic fallback.')
    from PIL import Image
    pixels=[]
    for ident in data.ids:
        with Image.open(ROOT/'open/train'/f'{ident}.png') as im:
            if im.mode!='L' or im.size!=(256,256):raise ValueError('Unexpected image format')
            pixels.append(np.array(im,dtype=np.uint8))
    runner=eng.Runner(torch,device,np.stack(pixels))
    pred,selections=collect(sub,binding,data,refs[seed],seed,runner)
    validate(folder);save_results(sub,binding,data,refs[seed],pred,selections,plan)


def compare(folder):
    data,refs,plan=validate(folder);rt=verify_check(folder,plan);records=[]
    for seed in SEEDS:
        m,d,s=verify_results(folder/f'seed{seed}',binding_for(plan,rt,seed),data,refs[seed],plan,seed)
        records.append({'seed':seed,'current_rmse':m['current']['rmse'],'standalone':m['partial'],'head_control':m['head_only'],
          'partial_blend':m[PARTIAL_BLEND],'control_blend':m[CONTROL_BLEND],'diagnosis':d,
          'outer_encoder_updated_count':sum(x['audits']['partial']['encoder_updated'] for x in s)})
    tests={}
    for key in ('standalone','partial_blend'):
        gain=float(np.mean([r[key]['gain_vs_current'] for r in records]));paired=all(r[key]['gain_vs_control']>0 for r in records)
        positive=paired and all(r[key]['gain_vs_current']>0 for r in records)
        tests[key]={'mean_gain_vs_current':gain,'mean_gain_vs_control':float(np.mean([r[key]['gain_vs_control'] for r in records])),
          'all_seeds_improve_paired_control':paired,'all_seeds_improve_both':positive,'passes_expansion_priority':positive and gain>=.30}
    result={'primary':PRIMARY,'secondary':PARTIAL_BLEND,'per_seed':records,'comparisons':tests,
      'additional_review_required':'All500, common16 AND remaining484, tails/noise/new worst20. No automatic adoption.',
      'aggregation':'Mean per-seed differences; same500, no prediction averaging or independent inference.','protocol_sha256':t.digest(plan['frozen'])}
    t.write_json(folder/'comparison.json',result);print(base.canonical_json(result));return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('prepare','validate','check','train','compare'))
    p.add_argument('--seed',type=int,choices=SEEDS,default=42);p.add_argument('--device',default='cuda');p.add_argument('--output-dir',type=base.project_path,default=OUT)
    a=p.parse_args(argv)
    if a.command=='prepare':prepare(a.output_dir)
    elif a.command=='validate':validate(a.output_dir);print('Inputs/protocol/baseline verified; no training.')
    elif a.command=='check':check(a.output_dir,a.device)
    elif a.command=='train':train(a.output_dir,a.seed,a.device)
    else:compare(a.output_dir)


if __name__=='__main__':main()
