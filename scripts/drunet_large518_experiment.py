"""Original vs weak DRUNet input, frozen Large518 + nested Ridge.

Pilot first, then user-run extraction/regression. Existing Scattering486 contributes
exactly50% in both arms. No test access, new encoder training or submissions.
"""
from __future__ import annotations
import argparse
import contextlib
import csv
import io
from pathlib import Path
import time
import numpy as np
from threadpoolctl import threadpool_limits

if __package__:
    from . import drunet_restoration as denoise
    from . import evaluate_dinov2_large518_confirmation as ridge
else:
    import drunet_restoration as denoise
    import evaluate_dinov2_large518_confirmation as ridge
e, eng, core = ridge.e, ridge.eng, ridge.core
ROOT = e.ROOT
OUT = ROOT/'reports/drunet_large518'
SEEDS = (42,2026)
PRIMARY = 'restored_replaced_blend'
PROTOCOL = {
    'version':1, 'seeds':list(SEEDS), 'primary':PRIMARY, 'restoration':denoise.RECIPE,
    'representation':'Frozen DINOv2 Large518, CLS+patch mean+population std,3072; identical original transform after restoration.',
    'ridge':'alpha100/1000/10000;5 archived outer x3 inner; X/y scalers fit on context only; ties prefer larger alpha.',
    'baseline':'current=.5Scattering486+.5original Large518 Ridge; verified same-fold OOF, not Public score.',
    'candidate':'.5same Scattering486+.5restored Large518 Ridge; only Large input changes.',
    'raw_control':'Re-fit raw cached3072 with identical Ridge function; match archived raw OOF/alpha before accepting results.',
    'scope':'Train500 only. User runs extraction and Ridge fitting; pretrained DRUNet/Large encoder frozen. No test/submission.',
    'pilot_review':'Inspect16 label-blind pairs at fixed contrast; change proxies do not prove structural correctness.',
    'stop':'One fixed restoration recipe. No strength/weight/alpha expansion from outer scores.',
    'decision':'Overall,common16,remaining484,noiseQ1..4,tails,newworst20; no automatic adoption.',
    'expansion_priority_mean_gain':.3,
    'limits':'Same repeatedly studied500/reused folds; pilot selection target-blind, not an independent holdout.',
    'attribution':'DRUNet/DPIR (Kai Zhang et al.,MIT); DINOv2 (Meta). Scattering486 ensemble includes PriorLabs-TabPFN.'}


def load_state():
    data, scat, source_plan = e.validate(e.OUT)
    features, feature_refs = e.load_features(e.OUT,data,source_plan)
    denoise.validate_assets()
    references, files = {}, {}
    for seed in SEEDS:
        folder=e.OUT/f'seed{seed}'
        binding=e.evaluation_binding(source_plan,seed,feature_refs)
        if not e.completed(folder,binding):raise FileNotFoundError('Original evaluation incomplete')
        with contextlib.redirect_stdout(io.StringIO()):
            p,choices=e.collect(folder,data,scat[seed],features,binding,seed,False,ridge.forbidden)
        saved=core.align_rows(core.read_rows(folder/'oof.csv',{'ID','large518_ridge','current_large518_ridge_equal'}),data.ids,folder)
        for key in ('large518_ridge','current_large518_ridge_equal'):
            np.testing.assert_array_equal(p[key],core.finite_column(saved,key,folder))
        references[seed]={'folds':scat[seed]['folds'],'scattering':scat[seed]['current'],
            'raw_ridge':p['large518_ridge'],'current':p['current_large518_ridge_equal'],
            'alphas':[v['choices']['large518_ridge']['candidate']['alpha'] for v in choices]}
        files[str(seed)]={n:e.file_ref(folder/n) for n in ('completed.json','oof.csv','selection.json')}
    pilot=denoise.select_pilot(data.ids,data.noise)
    paths=[Path(__file__),Path(denoise.__file__),Path(ridge.__file__),
           ROOT/'scripts/run_drunet_large518.ps1',ROOT/'scripts/vendor/drunet/network_unet.py',
           ROOT/'scripts/vendor/drunet/basicblock.py',ROOT/'scripts/vendor/drunet/LICENSE']
    frozen={'protocol':PROTOCOL,'sources':{p.name:e.file_ref(p) for p in paths},
        'source_protocol':e.file_ref(e.OUT/'protocol.json'),'raw_features':feature_refs['large518'],
        'weight':e.file_ref(denoise.WEIGHTS),'references':files,'environment':e.environment(),
        'common_failure_ids':source_plan['frozen']['common_failure_ids'],
        'pilot_ids':data.ids[pilot].tolist(),'ids_sha256':e.base.sha256_array(data.ids),
        'y_sha256':e.base.sha256_array(data.y),'noise_sha256':e.base.sha256_array(data.noise),
        'preserved_submissions':{n:e.file_ref(ROOT/f'reports/{n}/submission/submission.csv')
            for n in ('final_scattering_blend','final_large518_blend')}}
    return data,features['large518'],references,frozen


def prepare(folder=OUT):
    _,_,_,frozen=load_state(); path=folder/'protocol.json'
    if path.exists():
        if e.read_json(path)['frozen']!=frozen:raise ValueError('Frozen experiment changed; preserve existing directory')
    else:
        if folder.exists() and any(folder.iterdir()):raise ValueError('Nonempty unprepared directory')
        e.write_json(path,{'frozen':frozen,'created_unix':time.time()})
    print('Prepared/verified fixed weak-DRUNet comparison. No fitting or image forwards.')


def validate(folder=OUT):
    data,x,refs,frozen=load_state();plan=e.read_json(folder/'protocol.json')
    if plan['frozen']!=frozen:raise ValueError('Frozen inputs/code/environment changed; preserve old results')
    return data,x,refs,plan


def bind(plan, **extra):return {'plan':e.digest(plan['frozen']),**extra}


def runtime(device_name):
    torch,device,info=e.configure(device_name)
    return torch,device,{'device':str(device),'gpu':info.get('gpu_name'),'batch_size':1,'precision':'float32','amp':False}


def pilot(folder=OUT,device_name='cuda'):
    data,_,_,plan=validate(folder)
    ids=np.array(plan['frozen']['pilot_ids'],dtype=data.ids.dtype); path=folder/'pilot'
    if (path/'manifest.json').exists():
        verify_pilot(folder,plan);print('Pilot cache verified; no new forwards.');return
    torch,device,rt=runtime(device_name); model=denoise.load_model(device)
    raws=[];outputs=[];rows=[];t0=time.perf_counter()
    for ident in ids:
        raw=denoise.read_gray(ident);restored,info=denoise.restore(model,raw,device)
        row={'ID':str(ident),**info,**denoise.diagnostics(raw,restored)}
        denoise.preview(raw,restored,f'{ident} | sigma={info["condition_sigma_255"]:.3f}',path/f'{ident}.png')
        raws.append(raw);outputs.append(restored);rows.append(row)
        print(f'Pilot {len(rows)}/16 {ident}: sigma={info["condition_sigma_255"]:.3f}; change RMS={row["change_rmse_255"]:.3f}',flush=True)
    del model
    if device.type=='cuda':torch.cuda.empty_cache()
    values=np.stack(outputs).reshape(len(ids),-1)
    e.npz_save(path/'restored.npz',ids,values,bind(plan,stage='pilot'),{'runtime':rt})
    e.write_csv(path/'change_diagnostics.csv',rows)
    for q in range(4):
        section=slice(q*4,(q+1)*4)
        denoise.overview(raws[section],outputs[section],ids[section],[r['condition_sigma_255'] for r in rows[section]],path/f'noiseQ{q+1}.png')
    names=['restored.npz','change_diagnostics.csv']+[f'{i}.png' for i in ids]+[f'noiseQ{q+1}.png' for q in range(4)]
    e.write_json(path/'manifest.json',{'binding':bind(plan,stage='pilot'),'runtime':rt,'ids':ids.tolist(),
        'files':{n:e.base.sha256_file(path/n) for n in names},'competition_fits':0,'seconds':time.perf_counter()-t0})
    print('Pilot ready: '+str(path)+'; inspect images before full extraction.')


def verify_pilot(folder,plan,require_review=False):
    path=folder/'pilot';m=e.read_json(path/'manifest.json')
    if m['binding']!=bind(plan,stage='pilot') or m['ids']!=plan['frozen']['pilot_ids']:
        raise ValueError('Pilot scope changed')
    expected={'restored.npz','change_diagnostics.csv',*[f'{i}.png' for i in m['ids']],*[f'noiseQ{q+1}.png' for q in range(4)]}
    if set(m['files'])!=expected:raise ValueError('Pilot outputs incomplete')
    for n,h in m['files'].items():
        if e.base.sha256_file(path/n)!=h:raise ValueError('Pilot output changed: '+n)
    if require_review:
        review=e.read_json(path/'visual_review.json')
        if review.get('pilot_manifest')!=e.file_ref(path/'manifest.json') or review.get('decision')!='allow_fixed_experiment':
            raise ValueError('Pilot visual review does not support full extraction')
    return m


def extraction_scope(folder,plan):
    verify_pilot(folder,plan,True)
    return bind(plan,stage='features',review=e.file_ref(folder/'pilot/visual_review.json'))


def extract(folder=OUT,device_name='cuda'):
    data,raw_features,_,plan=validate(folder);scope=extraction_scope(folder,plan)
    destination=folder/'features/features.npz'
    if destination.exists():
        e.npz_read(destination,data.ids,3072,scope);print('Complete restored features verified; no new forwards.');return
    torch,device,rt=runtime(device_name)
    runtime_path=folder/'features/runtime.json'
    if runtime_path.exists() and e.read_json(runtime_path)!=rt:raise ValueError('Extraction runtime changed; resume original device')
    e.write_json(runtime_path,rt)
    # Restore first, release DRUNet, then load Large. This keeps 8GB VRAM use bounded.
    denoiser=None;restored_images=[];t0=time.perf_counter()
    pilot_meta=verify_pilot(folder,plan,True)
    pilot_ids=np.array(plan['frozen']['pilot_ids'],dtype=data.ids.dtype)
    pilot_pixels,_=e.npz_read(folder/'pilot/restored.npz',pilot_ids,256*256,bind(plan,stage='pilot'))
    for index,ident in enumerate(data.ids):
        path=folder/'restored_chunks'/f'{index:04d}.npz';local={**scope,'runtime':rt,'index':index}
        if path.exists(): pixels,_=e.npz_read(path,data.ids[index:index+1],256*256,local)
        else:
            if denoiser is None:denoiser=denoise.load_model(device)
            image,_=denoise.restore(denoiser,denoise.read_gray(ident),device)
            pixels=image.reshape(1,-1)
            if ident in pilot_ids:
                expected=pilot_pixels[np.flatnonzero(pilot_ids==ident)[0]]
                if not np.allclose(pixels[0],expected,rtol=1e-5,atol=1e-5):raise ValueError('Pilot restoration differs from extraction')
            e.npz_save(path,data.ids[index:index+1],pixels,local)
        restored_images.append(pixels.reshape(256,256))
        if index==0 or (index+1)%50==0:print(f'Restored/verified {index+1}/500; seconds={time.perf_counter()-t0:.1f}',flush=True)
    del denoiser
    if device.type=='cuda':torch.cuda.empty_cache()
    model=eng.load_model('large').to(device)
    # Raw controls must still reproduce the exact historical preprocessing.
    anchors=[]
    for ident in pilot_ids:
        index=int(np.flatnonzero(data.ids==ident)[0])
        raw=torch.from_numpy(denoise.read_gray(ident).astype(np.float32)/np.float32(255))[None,None].to(device)
        with torch.inference_mode():
            value=eng.pool_tokens(model.forward_features(eng.transform_image(raw,518)),518,1024).cpu().numpy()[0]
        if not np.allclose(value,raw_features[index],rtol=1e-4,atol=1e-4):raise ValueError('Original Large518 anchor failed')
        anchors.append({'ID':str(ident),'max_abs_difference':float(np.max(np.abs(value-raw_features[index])))})
    rows=[]
    for index,ident in enumerate(data.ids):
        path=folder/'features/chunks'/f'{index:04d}.npz';local={**scope,'runtime':rt,'index':index}
        if path.exists():values,_=e.npz_read(path,data.ids[index:index+1],3072,local)
        else:
            pixels=torch.from_numpy(restored_images[index])[None,None].to(device)
            with torch.inference_mode():
                values=eng.pool_tokens(model.forward_features(eng.transform_image(pixels,518)),518,1024).cpu().numpy()
            e.npz_save(path,data.ids[index:index+1],values,local)
        rows.append(values)
        if index==0 or (index+1)%50==0:print(f'Large518 restored {index+1}/500; seconds={time.perf_counter()-t0:.1f}',flush=True)
    validate(folder)
    e.npz_save(destination,data.ids,np.concatenate(rows),scope,{'runtime':rt,'raw_anchors':anchors,'seconds':time.perf_counter()-t0})
    print('Complete: '+str(destination)+'; encoder updates=0; test access=0.')


def load_features(folder,data,plan):
    return e.npz_read(folder/'features/features.npz',data.ids,3072,extraction_scope(folder,plan))[0]


def collect(folder,data,features,ref,binding,seed,allow_fit=True,runner=None):
    predictions={name:np.full(len(data.y),np.nan) for name in ('raw_large518','restored_large518')};choices=[]
    with threadpool_limits(limits=2):
        for fold in range(5):
            ti=np.flatnonzero(ref['folds']!=fold);vi=np.flatnonzero(ref['folds']==fold)
            row={'fold':fold}
            for name,x in features.items():
                scope={'evaluation':binding,'outer_fold':fold,'representation':name,
                       'train_indices':ti.tolist(),'query_indices':vi.tolist()}
                predictions[name][vi],choice=ridge.fit_fold(folder/name/f'fold{fold}',scope,x[ti],data.y[ti],x[vi],seed+fold,allow_fit,runner)
                row[name]={**choice,'representation':name}
            if not np.allclose(predictions['raw_large518'][vi],ref['raw_ridge'][vi],rtol=0,atol=1e-9):
                raise ValueError('Original Ridge OOF anchor failed')
            if row['raw_large518']['alpha']!=ref['alphas'][fold]:raise ValueError('Original alpha selection differs')
            choices.append(row)
            print(f'Outer{fold}: raw alpha={row["raw_large518"]["alpha"]:g}; restored alpha={row["restored_large518"]["alpha"]:g}; raw anchor matched',flush=True)
    return {'current':ref['current'],'scattering486':ref['scattering'],**predictions,
            PRIMARY:.5*ref['scattering']+.5*predictions['restored_large518']},choices


def result_data(data,ref,predictions,choices,plan,seed):
    # Reuse thoroughly tested group/error diagnostics without changing its globals.
    mapped={ridge.PRIMARY if n==PRIMARY else n:p for n,p in predictions.items()}
    summary,outputs=ridge.result_data(data,ref,mapped,choices,plan,seed)
    summary['primary']=PRIMARY
    summary['metrics'][PRIMARY]=summary['metrics'].pop(ridge.PRIMARY)
    summary['diagnosis'][PRIMARY]=summary['diagnosis'].pop(ridge.PRIMARY)
    for rows in outputs.values():
        for row in rows:
            for key in list(row):
                new=key.replace(ridge.PRIMARY,PRIMARY)
                if new!=key:row[new]=row.pop(key)
            if row.get('model')==ridge.PRIMARY:row['model']=PRIMARY
    summary['raw_anchor_matched']=True
    return summary,outputs


def report_text(s):
    lines=[f'Weak DRUNet input vs raw Large518, seed{s["seed"]}',
        'Current=.5Scattering486+.5RAW Large518; candidate changes only Large input. Gain>0 is improvement.']
    for n,m in s['metrics'].items():lines.append(f'{n}: RMSE={m["rmse"]:.6f}; gain vs current={m["gain_vs_current"]:+.6f}')
    for g in ('common16','remaining484','noiseQ1','noiseQ2','noiseQ3','noiseQ4','low10','high10','middle80'):
        old=s['diagnosis']['current']['groups'][g];new=s['diagnosis'][PRIMARY]['groups'][g]
        lines.append(f'{g}: {old["rmse"]:.6f} -> {new["rmse"]:.6f}; gain={new["gain_vs_current"]:+.6f}')
    d=s['diagnosis'][PRIMARY]
    lines.extend([f'Common16 improved={d["common16_improved"]}/16; remaining in worst20={d["common16_in_worst20"]}/16; same error sign={d["common16_same_error_sign"]}/16.',
        f'Remaining484 improved={d["remaining484_improved"]}/484; new worst20={d["new_worst20_ids"]}',PROTOCOL['limits'],
        'No automatic adoption, strength search or submission changes.'])
    return '\n'.join(lines)+'\n'


def save_results(folder,data,ref,predictions,choices,plan,seed):
    s,outputs=result_data(data,ref,predictions,choices,plan,seed)
    for name,rows in outputs.items():e.write_csv(folder/name,rows)
    e.write_json(folder/'summary.json',s);e.write_json(folder/'selection.json',choices)
    (folder/'report.txt').write_text(report_text(s),encoding='utf-8')
    e.write_json(folder/'completed.json',{'binding':ridge.binding_for(plan,seed),'files':{n:e.base.sha256_file(folder/n) for n in e.OUTPUTS}})
    print(report_text(s));print('Report: '+str(folder/'report.txt'))
    return s


def read_completed(folder,data,features,ref,plan,seed):
    binding=ridge.binding_for(plan,seed)
    if not e.completed(folder,binding):raise FileNotFoundError('Run evaluate first')
    with contextlib.redirect_stdout(io.StringIO()):
        p,c=collect(folder,data,features,ref,binding,seed,False,ridge.forbidden)
    s,outputs=result_data(data,ref,p,c,plan,seed)
    if e.read_json(folder/'summary.json')!=s or e.read_json(folder/'selection.json')!=c:raise ValueError('Saved result differs from caches')
    if (folder/'report.txt').read_text(encoding='utf-8')!=report_text(s):raise ValueError('Report changed')
    for name,rows in outputs.items():
        with (folder/name).open(encoding='utf-8',newline='') as f:
            if list(csv.DictReader(f))!=list(csv.DictReader(io.StringIO(core.csv_text(rows)))):raise ValueError('Saved CSV differs: '+name)
    return s


def evaluate(folder=OUT,seed=42):
    data,raw,refs,plan=validate(folder)
    features={'raw_large518':raw,'restored_large518':load_features(folder,data,plan)}
    sub=folder/f'seed{seed}'
    if (sub/'completed.json').exists():
        read_completed(sub,data,features,refs[seed],plan,seed);print('Completed result verified; no refitting.');return
    p,c=collect(sub,data,features,refs[seed],ridge.binding_for(plan,seed),seed)
    validate(folder);load_features(folder,data,plan)
    save_results(sub,data,refs[seed],p,c,plan,seed)


def aggregate(summaries):
    if [s['seed'] for s in summaries]!=list(SEEDS):raise ValueError('Both fixed seeds required in order')
    records=[]
    for s in summaries:
        m=s['metrics'];d=s['diagnosis'][PRIMARY]
        records.append({'seed':s['seed'],'current_rmse':m['current']['rmse'],'candidate_rmse':m[PRIMARY]['rmse'],
            'gain_vs_current':m[PRIMARY]['gain_vs_current'],
            'standalone_gain':m['raw_large518']['rmse']-m['restored_large518']['rmse'],
            'group_gains':{g:v['gain_vs_current'] for g,v in d['groups'].items()},'diagnosis':d})
    gains=[r['gain_vs_current'] for r in records]
    focus={g:all(r['group_gains'][g]>=0 for r in records) for g in ('common16','remaining484','noiseQ1','noiseQ4')}
    return {'primary':PRIMARY,'per_seed':records,'mean_gain_vs_current':float(np.mean(gains)),
        'all_seeds_improve_overall':all(g>0 for g in gains),'all_seeds_nonworsening_by_group':focus,
        'mean_gain_by_group':{g:float(np.mean([r['group_gains'][g] for r in records])) for g in records[0]['group_gains']},
        'passes_expansion_priority':all(g>0 for g in gains) and all(focus.values()) and float(np.mean(gains))>=.3,
        'automatic_adoption':False,'independent_holdout':False,
        'aggregation':'Mean of per-seed OOF differences; no prediction averaging or independent-sample inference.'}


def compare(folder=OUT):
    data,raw,refs,plan=validate(folder);features={'raw_large518':raw,'restored_large518':load_features(folder,data,plan)}
    result=aggregate([read_completed(folder/f'seed{s}',data,features,refs[s],plan,s) for s in SEEDS])
    result['protocol_sha256']=e.digest(plan['frozen'])
    e.write_json(folder/'comparison.json',result);print(e.base.canonical_json(result));return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('prepare','pilot','extract','evaluate','compare'))
    p.add_argument('--output-dir',type=Path,default=OUT);p.add_argument('--device',default='cuda')
    p.add_argument('--seed',type=int,choices=SEEDS,default=42);a=p.parse_args()
    folder=a.output_dir if a.output_dir.is_absolute() else ROOT/a.output_dir
    if a.command=='prepare':prepare(folder)
    elif a.command=='pilot':pilot(folder,a.device)
    elif a.command=='extract':extract(folder,a.device)
    elif a.command=='evaluate':evaluate(folder,a.seed)
    else:compare(folder)


if __name__=='__main__':main()
