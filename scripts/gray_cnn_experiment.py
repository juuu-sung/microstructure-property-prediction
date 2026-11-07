"""User-run scratch gray CNN with matched early-stride control and nested epochs."""
from __future__ import annotations
import argparse
import contextlib
import csv
import io
import math
from pathlib import Path
import time
import numpy as np

if __package__:
    from . import gray_cnn_engine as eng
    from . import evaluate_dinov2_large518_confirmation as diagnostics
else:
    import gray_cnn_engine as eng
    import evaluate_dinov2_large518_confirmation as diagnostics
e=eng.shared;core=e.core;ROOT=e.ROOT;OUT=ROOT/'reports/gray_cnn'
SEEDS=(42,2026);MODES=eng.MODES
PRIMARY='native_stride1';CONTROL='early_stride2'
SECONDARY='current_native_equal';CONTROL_BLEND='current_early_equal'
PROTOCOL={
    'version':1,'seeds':list(SEEDS),'primary':PRIMARY,'secondary':SECONDARY,
    'input':'Native256 gray, uint8/255. No resize,crop,denoise,rotation,brightness jitter or TTA.',
    'architecture':'Scratch CNN widths16/32/64/96,3x3 convolutions,4-group GroupNorm,SiLU,residual blocks. '
        'Candidate stem stride1 and first residual block preserve256; later avgpool2 gives128/64/32. '
        'Control changes ONLY first conv stride2, giving128/64/32/16. Same parameter count and seeded initial tensors.',
    'summary':'4-stage spatial mean/population std (416) + raw global mean/std and 2x2 mean/std(10) ->426.',
    'head':'Linear426->64,SiLU,Dropout0.1,Linear64->1. All parameters trainable from epoch1.',
    'training':{'max_epochs':80,'checkpoints':list(eng.EPOCHS),'batch':8,'accumulation':2,'effective_batch':16,
        'optimizer':'AdamW,lr3e-4,decay0.01 on multidimensional weights; norm/bias decay0.',
        'schedule':'linear warmup epochs1..5 then cosine to lr3e-5 at80. Fixed80 horizon for inner AND shorter outer refit prefixes.',
        'loss':'Unweighted image-level MSE. y population mean/std fitted only on training context.',
        'clip_gradient':1.,'precision':'FP32,no AMP/TF32,deterministic; GroupNorm has no batch running statistics.',
        'selection':'3-inner pooled SSE/400; each mode selects epoch independently, ties earlier. No outer-label checkpoint selection.',
        'no_search':'No architecture/LR/weight/augmentation search. Epoch selection only.'},
    'validation':'Archived5 outer400/100 x3 inner train-only rank deciles; seed+fold. Same two repeatedly used seeds.',
    'initialization_seed':'seed+1009*outer_fold+inner_index; outer refit inner_index9; paired between arms.',
    'reference':'Current=.5Scattering486+.5RAW Large518 Ridge, matching-seed archived OOF (Public12.312226 submission family).',
    'blend':'Secondary=.5current whole ensemble+.5new CNN. Both arms use same predeclared formula. No mixture tuning.',
    'work':'60 inner80-epoch trajectories +20 outer chosen-epoch refits across2 seeds/2 arms; no pretraining/download.',
    'decision':'Compare standalone and fixed blend against current AND matched control; common16,remaining484,tails,noise,newworst20. '
        'Mean gain0.30 is expansion priority only, not statistical significance or automatic adoption.',
    'scope':'User executes competition training. No test image access, final500 fit or submission changes.',
    'limits':'Same repeatedly studied500, not independent holdout. Stride contrast also changes feature-map resolution/receptive fields; '
        'positive result would not prove fine boundaries were learned. Scratch learning on500 may overfit.',
    'sources':['https://docs.pytorch.org/docs/2.10/generated/torch.nn.GroupNorm.html',
               'https://docs.pytorch.org/docs/2.10/generated/torch.optim.AdamW.html']}


def forbidden(*args,**kwargs):raise RuntimeError('Missing training cell; implicit training forbidden')


def load_state():
    data,scattering,oldplan=e.validate(e.OUT);features,fr=e.load_features(e.OUT,data,oldplan)
    refs={};files={}
    for seed in SEEDS:
        folder=e.OUT/f'seed{seed}';binding=e.evaluation_binding(oldplan,seed,fr)
        if not e.completed(folder,binding):raise ValueError('Original reference incomplete')
        with contextlib.redirect_stdout(io.StringIO()):
            p,_=e.collect(folder,data,scattering[seed],features,binding,seed,False,forbidden)
        rows=core.align_rows(core.read_rows(folder/'oof.csv',{'ID','current_large518_ridge_equal'}),data.ids,folder)
        current=core.finite_column(rows,'current_large518_ridge_equal',folder)
        np.testing.assert_array_equal(current,p['current_large518_ridge_equal'])
        refs[seed]={'current':current,'folds':scattering[seed]['folds']}
        files[str(seed)]={n:e.file_ref(folder/n) for n in ('oof.csv','completed.json')}
    frozen={'protocol':PROTOCOL,'sources':{p.name:e.file_ref(p) for p in (Path(__file__),Path(eng.__file__),Path(diagnostics.__file__),ROOT/'scripts/run_gray_cnn.ps1')},
        'source_protocol':e.file_ref(e.OUT/'protocol.json'),'references':files,'environment':e.environment(),
        'common_failure_ids':oldplan['frozen']['common_failure_ids'],'training_inventory':oldplan['frozen']['inventory'],
        'ids_sha256':e.base.sha256_array(data.ids),'y_sha256':e.base.sha256_array(data.y),'noise_sha256':e.base.sha256_array(data.noise),
        'preserved_submissions':{n:e.file_ref(ROOT/f'reports/{n}/submission/submission.csv') for n in ('final_scattering_blend','final_large518_blend')}}
    return data,refs,frozen


def prepare(folder=OUT):
    data,refs,frozen=load_state();path=folder/'protocol.json'
    if path.exists():
        if e.read_json(path)['frozen']!=frozen:raise ValueError('Frozen experiment changed; preserve old directory')
    else:
        if folder.exists() and any(folder.iterdir()):raise ValueError('Nonempty unprepared folder')
        e.write_json(path,{'frozen':frozen,'created_unix':time.time()})
    print('Prepared/verified scratch gray CNN stride1/stride2. No competition fits/forwards.')
    for seed in SEEDS:print(f'Current seed{seed}: RMSE={core.metrics(data.y,refs[seed]["current"])["rmse"]:.6f}')


def validate(folder=OUT):
    data,refs,frozen=load_state();plan=e.read_json(folder/'protocol.json')
    if plan['frozen']!=frozen:raise ValueError('Frozen sources, environment or input hashes changed')
    return data,refs,plan


def runtime(torch,device):
    return {'device':str(device),'gpu':torch.cuda.get_device_name(device) if device.type=='cuda' else None,
            'precision':'float32','amp':False,'threads':torch.get_num_threads(),'batch':eng.BATCH,'accumulation':eng.ACCUM}


def verify_check(folder,plan):
    saved=e.read_json(folder/'synthetic_check.json')
    if saved['plan']!=e.digest(plan['frozen']) or saved['sha256']!=e.digest(saved['result']):raise ValueError('Synthetic check changed')
    r=saved['result']
    if set(r['modes'])!=set(MODES) or not r['paired_initial_state']:raise ValueError('Missing paired check')
    if any(not a['audit']['all_stages_updated'] for a in r['modes'].values()):raise ValueError('Unverified gradient path')
    return r['runtime']


def check(folder=OUT,device_name='cuda'):
    _,_,plan=validate(folder);torch,device,_=e.configure(device_name);rt=runtime(torch,device)
    if (folder/'synthetic_check.json').exists():
        if verify_check(folder,plan)!=rt:raise ValueError('Checked runtime differs')
        print('Synthetic architecture/update check verified; no rerun.');return
    rng=np.random.default_rng(1701);pixels=rng.integers(0,256,size=(16,256,256),dtype=np.uint8)
    target=np.linspace(-1,1,16);records={};initial_hashes=[]
    for mode in MODES:
        torch.manual_seed(1701)
        if device.type=='cuda':torch.cuda.manual_seed_all(1701);torch.cuda.reset_peak_memory_stats(device)
        model=eng.make_model(torch,mode).to(device);initial=eng.snapshot(model);initial_hashes.append(eng.state_hash(initial))
        x=eng.batch_images(torch,pixels,np.arange(2),device)
        with torch.inference_mode():shapes=[list(a.shape) for a in model.maps(x)]
        expected=[256,128,64,32] if mode==PRIMARY else [128,64,32,16]
        if [s[-1] for s in shapes]!=expected:raise ValueError('Wrong feature resolutions')
        opt=eng.optimizer_for(torch,model);grad={g:0. for g in eng.GROUPS};t0=time.perf_counter()
        for epoch in range(1,4):
            eng.optimize(torch,model,pixels,np.arange(16),target,device,opt,np.random.default_rng(1701+epoch),grad)
        if device.type=='cuda':torch.cuda.synchronize(device)
        result={'map_shapes':shapes,'audit':eng.audit(model,initial,3,grad),'three_steps_seconds':time.perf_counter()-t0,
                'peak_gpu_allocated_bytes':int(torch.cuda.max_memory_allocated(device)) if device.type=='cuda' else None}
        pred=eng.predict(torch,model,pixels,np.arange(16),device,0.,1.)
        if pred.shape!=(16,):raise ValueError('Wrong regression shape')
        records[mode]=result;print(e.base.canonical_json({'mode':mode,**result}),flush=True)
        del model,opt,x,initial
        if device.type=='cuda':torch.cuda.empty_cache()
    if len(set(initial_hashes))!=1:raise ValueError('Matched arms initial weights differ')
    result={'runtime':rt,'modes':records,'paired_initial_state':True,'competition_images_used':False,
            'note':'Three synthetic optimizer steps per arm; this validates execution, not predictive performance.'}
    e.write_json(folder/'synthetic_check.json',{'plan':e.digest(plan['frozen']),'result':result,'sha256':e.digest(result)})


def validate_run(result,folder,yt,yv,nq,epochs):
    if result['target_mean']!=float(np.mean(yt)) or result['target_std']!=max(float(np.std(yt)),1e-8):raise ValueError('Target scale not fitted on training context')
    if [r['epoch'] for r in result['records']]!=[ep for ep in eng.EPOCHS if ep<=epochs]:raise ValueError('Missing learning curve')
    for r in result['records']:
        pred=np.asarray(r['predictions']);a=r['update_audit']
        if pred.shape!=(nq,) or not np.isfinite(pred).all() or r['train_n']!=len(yt) or not np.isfinite(r['train_sse']) or r['train_sse']<0:raise ValueError('Invalid saved trajectory')
        if yv is None:
            if 'valid_sse' in r or 'valid_n' in r:raise ValueError('Outer labels leaked')
        elif r['valid_n']!=len(yv) or not np.isclose(r['valid_sse'],np.sum((pred-yv)**2),rtol=1e-12,atol=1e-10):raise ValueError('Invalid inner validation metric')
        active=r['epoch']>0
        if a['all_stages_updated']!=active:raise ValueError('Update status mismatch')
        for g in eng.GROUPS:
            if active and not(a['max_gradient_l2'][g]>0 and a['max_abs_parameter_change'][g]>0):raise ValueError('Missing encoder update evidence')
            if not active and (a['max_gradient_l2'][g]!=0 or a['max_abs_parameter_change'][g]!=0):raise ValueError('Epoch0 changed')
    if result['audit']!=result['records'][-1]['update_audit']:raise ValueError('Final audit mismatch')
    checkpoint=result['checkpoint']
    if (yv is None)!=(checkpoint is not None):raise ValueError('Outer checkpoint missing')
    if checkpoint and (checkpoint['name']!='model.pt' or checkpoint['sha256']!=e.base.sha256_file(folder/'model.pt')):raise ValueError('Checkpoint changed')


def cell(folder,binding,ti,yt,vi,yv,mode,epochs,seed,runner):
    scope={'evaluation':binding,'train_idx':ti.tolist(),'query_idx':vi.tolist(),'train_y':e.base.sha256_array(yt),
           'query_y':None if yv is None else e.base.sha256_array(yv),'mode':mode,'epochs':epochs,'seed':seed}
    marker=folder/'binding.json';complete=folder/'completed.json'
    if marker.exists() and e.read_json(marker)!=scope:raise ValueError('Cell scope changed')
    if complete.exists():
        saved=e.read_json(complete)
        if saved['binding']!=scope or saved['sha256']!=e.digest(saved['result']):raise ValueError('Cell checksum changed')
        result=saved['result']
    else:
        if runner is forbidden:forbidden()
        e.write_json(marker,scope);result=runner(ti,yt,vi,yv,mode,epochs,seed,folder,scope)
        validate_run(result,folder,yt,yv,len(vi),epochs)
        e.write_json(complete,{'binding':scope,'result':result,'sha256':e.digest(result)})
    validate_run(result,folder,yt,yv,len(vi),epochs)
    return result


def select_epoch(runs):
    curves=[]
    for epoch in eng.EPOCHS:
        rows=[next(r for r in run['records'] if r['epoch']==epoch) for run in runs]
        n=sum(r['valid_n'] for r in rows)
        if n!=400:raise ValueError('Expected pooled inner400')
        curves.append({'epoch':epoch,'inner_rmse':math.sqrt(sum(r['valid_sse'] for r in rows)/n),
                       'train_rmse':math.sqrt(sum(r['train_sse'] for r in rows)/sum(r['train_n'] for r in rows))})
    return min(curves,key=lambda r:(r['inner_rmse'],r['epoch'])),curves


def fit_fold(folder,binding,y,folds,seed,fold,runner):
    ti,vi=np.flatnonzero(folds!=fold),np.flatnonzero(folds==fold)
    if (len(ti),len(vi))!=(400,100):raise ValueError('Expected outer400/100')
    yt=y[ti];splits=e.eng.inner_splits(yt,seed+fold)
    result={'validation_indices':vi.tolist(),'predictions':{},'selection':{},'curves':{},'refit_audits':{}}
    for mode in MODES:
        inner=[cell(folder/'cells'/f'fold{fold}'/mode/f'inner{k}',binding,ti[a],yt[a],ti[b],yt[b],mode,80,seed+1009*fold+k,runner)
               for k,(a,b) in enumerate(splits)]
        chosen,curves=select_epoch(inner)
        refit=cell(folder/'cells'/f'fold{fold}'/mode/'refit',binding,ti,yt,vi,None,mode,chosen['epoch'],seed+1009*fold+9,runner)
        result['predictions'][mode]=refit['records'][-1]['predictions'];result['selection'][mode]=chosen
        result['curves'][mode]=curves;result['refit_audits'][mode]=refit['audit']
        if runner is not forbidden:print(f'Outer{fold}: {mode} epoch={chosen["epoch"]}; inner={chosen["inner_rmse"]:.6f}; all_stages_updated={refit["audit"]["all_stages_updated"]}',flush=True)
    path=folder/f'fold{fold}.json'
    if path.exists():
        if e.read_json(path)!=result:raise ValueError('Fold result differs from cells')
    elif runner is forbidden:forbidden()
    else:e.write_json(path,result)
    return result


def collect(folder,binding,data,ref,seed,runner):
    p={m:np.full(500,np.nan) for m in MODES};choices=[]
    for fold in range(5):
        r=fit_fold(folder,binding,data.y,ref['folds'],seed,fold,runner)
        for m in MODES:p[m][r['validation_indices']]=r['predictions'][m]
        choices.append({'fold':fold,'choices':r['selection'],'audits':r['refit_audits']})
    return p,choices


def binding_for(plan,rt,seed):return {'plan':e.digest(plan['frozen']),'runtime':rt,'seed':seed}


def result_data(data,ref,p,choices,plan,seed,binding):
    output={'current':ref['current'],**p,CONTROL_BLEND:.5*(ref['current']+p[CONTROL]),SECONDARY:.5*(ref['current']+p[PRIMARY])}
    if set(p)!=set(MODES) or any(v.shape!=(500,) or not np.isfinite(v).all() for v in output.values()):raise ValueError('Invalid OOF schema')
    mapped={diagnostics.PRIMARY if n==PRIMARY else n:v for n,v in output.items()}
    s,files=diagnostics.result_data(data,ref,mapped,choices,plan,seed)
    for section in ('metrics','diagnosis'):s[section][PRIMARY]=s[section].pop(diagnostics.PRIMARY)
    for rows in files.values():
        for row in rows:
            for k in list(row):
                new=k.replace(diagnostics.PRIMARY,PRIMARY)
                if k!=new:row[new]=row.pop(k)
            if row.get('model')==diagnostics.PRIMARY:row['model']=PRIMARY
    s.update(binding=binding,primary=PRIMARY,secondary=SECONDARY,selection=choices)
    s.pop('encoder_updates',None)
    s['trained_outer_models']={m:sum(c['audits'][m]['all_stages_updated'] for c in choices) for m in MODES}
    for n,m in s['metrics'].items():
        ctrl=CONTROL_BLEND if n in (CONTROL_BLEND,SECONDARY) else CONTROL
        m.update(matched_control=ctrl,gain_vs_control=s['metrics'][ctrl]['rmse']-m['rmse'])
    return s,files


def expected_outputs():
    names=set(e.OUTPUTS)|{f'fold{f}.json' for f in range(5)}
    for f in range(5):
        for m in MODES:
            for role in ('inner0','inner1','inner2','refit'):
                prefix=f'cells/fold{f}/{m}/{role}'
                names.update((prefix+'/binding.json',prefix+'/completed.json'))
                if role=='refit':names.add(prefix+'/model.pt')
    return names


def completed(folder,binding):
    path=folder/'completed_evaluation.json'
    if not path.exists():return False
    done=e.read_json(path)
    if done['binding']!=binding or set(done['files'])!=expected_outputs():raise ValueError('Completed scope changed')
    for name,sha in done['files'].items():
        if e.base.sha256_file(folder/name)!=sha:raise ValueError('Completed artifact changed: '+name)
    return True


def report_text(s):
    lines=[f'Scratch grayscale CNN, stride1 vs stride2, seed{s["seed"]}',PROTOCOL['reference'],'Gain>0 is improvement.']
    for n,m in s['metrics'].items():lines.append(f'{n}: RMSE={m["rmse"]:.6f}; gain vs current={m["gain_vs_current"]:+.6f}; gain vs {m["matched_control"]}={m["gain_vs_control"]:+.6f}')
    for n in (PRIMARY,SECONDARY):
        lines.append(n+' subgroup gains:')
        for g in ('common16','remaining484','noiseQ1','noiseQ2','noiseQ3','noiseQ4','middle80','low10','high10'):
            v=s['diagnosis'][n]['groups'][g];lines.append(f'  {g}: RMSE={v["rmse"]:.6f}; gain={v["gain_vs_current"]:+.6f}')
        a=s['diagnosis'][n];lines.append(f'  Common16 improved={a["common16_improved"]}, in worst20={a["common16_in_worst20"]}, same error sign={a["common16_same_error_sign"]}; remaining484 improved={a["remaining484_improved"]}')
    lines.extend((PROTOCOL['decision'],PROTOCOL['limits'],'See fold*.json for full inner learning curves.'))
    return '\n'.join(lines)+'\n'


def save_results(folder,binding,data,ref,p,choices,plan,seed):
    s,files=result_data(data,ref,p,choices,plan,seed,binding)
    for name,rows in files.items():e.write_csv(folder/name,rows)
    e.write_json(folder/'summary.json',s);e.write_json(folder/'selection.json',choices)
    (folder/'report.txt').write_text(report_text(s),encoding='utf-8')
    e.write_json(folder/'completed_evaluation.json',{'binding':binding,'files':{n:e.base.sha256_file(folder/n) for n in sorted(expected_outputs())}})
    print(report_text(s));print('Report: '+str(folder/'report.txt'));return s


def read_completed(folder,binding,data,ref,plan,seed):
    if not completed(folder,binding):raise FileNotFoundError('Complete train first; no implicit fitting')
    p,c=collect(folder,binding,data,ref,seed,forbidden);s,files=result_data(data,ref,p,c,plan,seed,binding)
    if e.read_json(folder/'summary.json')!=s or e.read_json(folder/'selection.json')!=c:raise ValueError('Summary differs from caches')
    if (folder/'report.txt').read_text(encoding='utf-8')!=report_text(s):raise ValueError('Report differs')
    for n,rows in files.items():
        with (folder/n).open(encoding='utf-8',newline='') as f:
            if list(csv.DictReader(f))!=list(csv.DictReader(io.StringIO(core.csv_text(rows)))):raise ValueError('CSV differs: '+n)
    return s


def train(folder=OUT,seed=42,device_name='cuda'):
    data,refs,plan=validate(folder);rt=verify_check(folder,plan);sub=folder/f'seed{seed}';binding=binding_for(plan,rt,seed)
    if completed(sub,binding):read_completed(sub,binding,data,refs[seed],plan,seed);print('Completed seed verified; no retraining.');return
    torch,device,_=e.configure(device_name)
    if runtime(torch,device)!=rt:raise ValueError('Use checked runtime/device; no implicit fallback')
    from PIL import Image
    pixels=[]
    for ident in data.ids:
        with Image.open(ROOT/'open/train'/f'{ident}.png') as im:
            if im.mode!='L' or im.size!=(256,256):raise ValueError('Unexpected native gray image')
            pixels.append(np.array(im,dtype=np.uint8))
    p,c=collect(sub,binding,data,refs[seed],seed,eng.Runner(torch,device,np.stack(pixels)))
    validate(folder);save_results(sub,binding,data,refs[seed],p,c,plan,seed)


def aggregate(summaries):
    if [s['seed'] for s in summaries]!=list(SEEDS):raise ValueError('Both fixed seeds required')
    comparisons={}
    for name in (PRIMARY,SECONDARY):
        gains=[s['metrics'][name]['gain_vs_current'] for s in summaries]
        ctrls=[s['metrics'][name]['gain_vs_control'] for s in summaries]
        focus={g:all(s['diagnosis'][name]['groups'][g]['gain_vs_current']>=0 for s in summaries) for g in ('common16','remaining484','noiseQ4')}
        positive=all(g>0 for g in gains+ctrls)
        comparisons[name]={'mean_gain_vs_current':float(np.mean(gains)),'mean_gain_vs_control':float(np.mean(ctrls)),
            'all_seeds_improve_both':positive,'all_seeds_nonworsening_by_group':focus,
            'passes_expansion_priority':positive and all(focus.values()) and float(np.mean(gains))>=.3}
    return {'primary':PRIMARY,'secondary':SECONDARY,'comparisons':comparisons,
        'per_seed':[{'seed':s['seed'],'metrics':s['metrics'],'diagnosis':s['diagnosis'],'trained_outer_models':s['trained_outer_models']} for s in summaries],
        'automatic_adoption':False,'independent_holdout':False,'aggregation':'Mean per-seed differences; same500, no prediction averaging or independent inference.'}


def compare(folder=OUT):
    data,refs,plan=validate(folder);rt=verify_check(folder,plan)
    result=aggregate([read_completed(folder/f'seed{s}',binding_for(plan,rt,s),data,refs[s],plan,s) for s in SEEDS])
    result['protocol_sha256']=e.digest(plan['frozen']);e.write_json(folder/'comparison.json',result)
    print(e.base.canonical_json(result));return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('prepare','check','train','compare'))
    p.add_argument('--output-dir',type=Path,default=OUT);p.add_argument('--device',default='cuda');p.add_argument('--seed',type=int,choices=SEEDS,default=42)
    a=p.parse_args();folder=a.output_dir if a.output_dir.is_absolute() else ROOT/a.output_dir
    if a.command=='prepare':prepare(folder)
    elif a.command=='check':check(folder,a.device)
    elif a.command=='train':train(folder,a.seed,a.device)
    else:compare(folder)


if __name__=='__main__':main()
