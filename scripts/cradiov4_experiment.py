"""Frozen C-RADIOv4 + blend-aware nested Ridge; user-run competition work."""
from __future__ import annotations
import argparse
import contextlib
import csv
import gc
import io
from pathlib import Path
import time
import numpy as np
from threadpoolctl import threadpool_limits

if __package__:
    from . import cradiov4_engine as eng
    from . import evaluate_dinov2_large518_confirmation as diagnostics
    from . import residual_correction_experiment as residual
else:
    import cradiov4_engine as eng
    import evaluate_dinov2_large518_confirmation as diagnostics
    import residual_correction_experiment as residual

e=diagnostics.e
core=e.core
ROOT=e.ROOT
OUT=ROOT/'reports/cradiov4'
SEEDS=(42,2026)
PRIMARY='radio_replaced_blend'
SECONDARY='radio_complementary_blend'
CONTROL='large_retuned_blend'
PROTOCOL={
    'version':1,'seeds':list(SEEDS),'primary':PRIMARY,'secondary':SECONDARY,
    'encoder':'C-RADIOv4-SO400M,431237232 parameters, frozen/eval, pinned native official modules, strict safetensors loading.',
    'input':'Native256 gray uint8/255 -> full-image bicubic592, align_corners=False, antialias=True, clamp[0,1], repeatRGB. Internal official conditioner only. No crop/denoise/D4.',
    'resolution':'592 primary (37x37 patch16 tokens). ONLY CUDA OOM in synthetic batch1 check permits512 fallback, before competition forwards/scores. Size then locked.',
    'precision':'FP32 parameters/input, CUDA BF16 autocast, float32 feature pooling, no TF32. Batch1. No precision search/fallback.',
    'feature':'Official summary2304 + spatial mean1152 + population std1152 =4608. Official forward excludes3 CLS+7register tokens from spatial output.',
    'regression':'X/y StandardScaler + Ridge, alpha100/1000/10000; exact ties prefer larger alpha; no PCA.',
    'splits':'Archived seed42/2026 five outer400/100; three inner train-only rank deciles seed+fold.',
    'selection':'Each encoder alpha minimizes pooled INNER RMSE of .5Scattering+.5encoder. No outer labels or global OOF used for selection.',
    'inner_scattering':'Reuse residual_correction verified context266/267 -> query133/134 caches, same inner partitions. Its3 base Ridge alphas selected in a further3-fold context-only CV; TabPFN context only. No implicit refits.',
    'matched_control':'Retune archived raw Large518 using SAME blend objective/grid/splits; also reproduce original standalone-selected Large518 OOF anchor.',
    'current':'.5S+.5L, S=archived entire Scattering486 ensemble, L=archived standalone-selected raw Large518 Ridge.',
    'primary_formula':'.5S+.5C. C uses primary blend-selected alpha.',
    'secondary_formula':'.5S+.25L+.25C, same C and alpha; fixed diagnostic, no separate alpha/weight search.',
    'secondary_control':'.5S+.25L+.25L_retuned; detects changes from alpha objective alone.',
    'standalone':'radio_ridge is the SAME primary blend-selected C, not separately optimized for standalone RMSE.',
    'decision':'Compare primary vs current AND matched control on both seeds. Mean gain>=0.30 is investment priority only. Inspect common16/remaining484/noiseQ1..4/tails/folds/newworst20. No automatic adoption.',
    'scope':'Train500 only. User executes feature extraction/Ridge fitting. No new encoder or TabPFN training, test use, final500 fit, clipping or submission.',
    'limits':'Same repeatedly studied500/reused folds, not independent holdout. Global pooling still loses spatial relationships. Token-grid matching does not equate architectures/pretraining/precision.',
    'sources':['https://huggingface.co/nvidia/C-RADIOv4-SO400M','https://github.com/NVlabs/RADIO','https://arxiv.org/abs/2601.17237'],
    'attribution':'Licensed by NVIDIA Corporation under the NVIDIA Open Model License. Preserved Scattering ensemble includes PriorLabs-TabPFN.'}


def forbidden(*args,**kwargs):
    raise RuntimeError('Missing prerequisite/completed cell. Implicit fitting is prohibited.')


def load_state():
    eng.assets.validate(require_weights=False)
    with contextlib.redirect_stdout(io.StringIO()):
        data,scat,oldplan=e.validate(e.OUT)
        xs,feature_refs=e.load_features(e.OUT,data,oldplan)
        rd,raw,rr,rp=residual.validate()
    np.testing.assert_array_equal(data.ids,rd.ids)
    np.testing.assert_array_equal(data.y,rd.y)
    refs,files={},{}
    for seed in SEEDS:
        folder=e.OUT/f'seed{seed}'
        binding=e.evaluation_binding(oldplan,seed,feature_refs)
        if not e.completed(folder,binding):raise ValueError('Large518 reference incomplete')
        with contextlib.redirect_stdout(io.StringIO()):
            p,choices=e.collect(folder,data,scat[seed],xs,binding,seed,False,forbidden)
        np.testing.assert_allclose(rr[seed]['current'],scat[seed]['current'],rtol=0,atol=1e-10)
        np.testing.assert_array_equal(rr[seed]['folds'],scat[seed]['folds'])
        rb=residual.binding_for(rp,seed,rr[seed])
        rf=residual.OUT/f'seed{seed}'
        if not residual.completed(rf,rb):raise ValueError('Leak-free inner baseline caches required; no global OOF substitution')
        inner={}
        for fold in range(5):
            ti=np.flatnonzero(scat[seed]['folds']!=fold)
            a=residual.eng.splits(data.y[ti],seed+fold)
            b=eng.ridge.inner_splits(data.y[ti],seed+fold)
            for left,right in zip(a,b):
                for u,v in zip(left,right):np.testing.assert_array_equal(u,v)
            with contextlib.redirect_stdout(io.StringIO()):
                cross=residual.crossfit_components(rf/f'fold{fold}',rb,raw,ti,data.y[ti],seed,fold,forbidden,False)
            inner[fold]=residual.eng.ensemble(cross)
        refs[seed]={'folds':scat[seed]['folds'],'scattering':scat[seed]['current'],
            'large':p['large518_ridge'],'current':p['current_large518_ridge_equal'],'inner_scattering':inner,
            'large_alphas':[v['choices']['large518_ridge']['candidate']['alpha'] for v in choices]}
        files[str(seed)]={'large_completed':e.file_ref(folder/'completed.json'),
            'large_oof':e.file_ref(folder/'oof.csv'),'inner_completed':e.file_ref(rf/'completed.json'),
            'inner_arrays':{str(k):e.base.sha256_array(v) for k,v in inner.items()}}
    paths=[Path(__file__),Path(eng.__file__),Path(eng.assets.__file__),eng.assets.LOCK,
           Path(diagnostics.__file__),ROOT/'scripts/run_cradiov4.ps1']
    frozen={'protocol':PROTOCOL,'assets':eng.assets.spec(),'sources':{p.name:e.file_ref(p) for p in paths},
        'environment':e.environment(),'source_plan':e.file_ref(e.OUT/'protocol.json'),
        'residual_plan':e.file_ref(residual.OUT/'protocol.json'),'large_features':feature_refs['large518'],
        'references':files,'training_inventory':oldplan['frozen']['inventory'],
        'common_failure_ids':oldplan['frozen']['common_failure_ids'],'ids_sha256':e.base.sha256_array(data.ids),
        'y_sha256':e.base.sha256_array(data.y),'noise_sha256':e.base.sha256_array(data.noise),
        'preserved_submissions':{n:e.file_ref(ROOT/f'reports/{n}/submission/submission.csv') for n in ('final_scattering_blend','final_large518_blend')}}
    return data,xs['large518'],refs,frozen


def prepare(folder=OUT):
    data,_,refs,frozen=load_state()
    path=folder/'protocol.json'
    if path.exists():
        if e.read_json(path)['frozen']!=frozen:raise ValueError('Prepared experiment changed; preserve old directory')
    else:
        if folder.exists() and any(folder.iterdir()):raise ValueError('Nonempty unprepared directory')
        e.write_json(path,{'frozen':frozen,'created_unix':time.time(),'competition_fits':0,'competition_forwards':0})
    print('Prepared:4608 features, alpha100/1000/10000, blend-aware nested selection. Existing inner Scattering caches verified; no fits/forwards.')
    for seed in SEEDS:print(f'Current seed{seed}: {core.metrics(data.y,refs[seed]["current"])["rmse"]:.6f}')


def validate(folder=OUT):
    data,large,refs,frozen=load_state()
    plan=e.read_json(folder/'protocol.json')
    if plan['frozen']!=frozen:raise ValueError('Frozen experiment inputs/code/environment changed')
    return data,large,refs,plan


def bind(plan,**extra):return {'plan':e.digest(plan['frozen']),**extra}


def runtime(device_name):
    torch,device,info=e.configure(device_name)
    if device.type!='cuda' or not torch.cuda.is_bf16_supported():raise ValueError('Checked CUDA BF16 device required')
    return torch,device,{'device':str(device),'gpu':info.get('gpu_name'),'batch_size':1,'amp':'bfloat16','parameters':'float32','pooling':'float32','tf32':False}


def forward(model,raw,size,torch):
    with torch.inference_mode(),torch.autocast(device_type='cuda',dtype=torch.bfloat16):
        output=model(eng.transform(raw,size))
    return eng.pool(output,size)


def read_check(folder,plan):
    saved=e.read_json(folder/'synthetic_check.json')
    r=saved['result']
    if saved['binding']!=bind(plan) or saved['sha256']!=e.digest(r):raise ValueError('Synthetic check changed')
    if r['size'] not in (592,512) or r['shape']!=[1,eng.WIDTH] or r['competition_forwards']!=0 or not r['encoder_frozen']:
        raise ValueError('Invalid synthetic check')
    if (r['size']==512)!=bool(r['oom592']):raise ValueError('Fallback requires recorded592 OOM')
    return r


def check(folder=OUT,device_name='cuda'):
    _,_,_,plan=validate(folder)
    torch,device,rt=runtime(device_name)
    if (folder/'synthetic_check.json').exists():
        r=read_check(folder,plan)
        if r['runtime']!=rt:raise ValueError('Checked runtime changed')
        eng.assets.validate()
        print('Synthetic check already verified; locked size='+str(r['size']));return
    model=eng.load_model().to(device)
    raw=torch.linspace(0,1,256*256,device=device).reshape(1,1,256,256)
    oom=None
    for size in (eng.PRIMARY_SIZE,eng.FALLBACK_SIZE):
        try:
            torch.cuda.reset_peak_memory_stats(device)
            t0=time.perf_counter()
            a=forward(model,raw,size,torch)
            b=forward(model,raw,size,torch)
            torch.cuda.synchronize(device)
            np.testing.assert_allclose(a.cpu().numpy(),b.cpu().numpy(),rtol=1e-5,atol=1e-5)
            result={'size':size,'runtime':rt,'shape':list(a.shape),'seconds':time.perf_counter()-t0,
                'peak_cuda_bytes':torch.cuda.max_memory_allocated(device),'oom592':oom,
                'encoder_frozen':not any(p.requires_grad for p in model.parameters()),'competition_forwards':0,
                'max_repeat_difference':float((a-b).abs().max()),'summary_width':eng.SUMMARY_DIM,'spatial_width':eng.SPATIAL_DIM}
            e.write_json(folder/'synthetic_check.json',{'binding':bind(plan),'result':result,'sha256':e.digest(result)})
            print(e.base.canonical_json(result));return
        except torch.cuda.OutOfMemoryError:
            if size!=eng.PRIMARY_SIZE:raise
            oom='CUDA OOM at592 batch1 synthetic check; fallback fixed before competition inference.'
            # Avoid retaining prior forward tensors in the failed-check frame.
            if 'a' in locals():del a
            if 'b' in locals():del b
            gc.collect();torch.cuda.empty_cache()
            print(oom,flush=True)


def feature_binding(plan,check_result):return bind(plan,stage='features',check=e.digest(check_result))


def load_features(folder,data,plan):
    ck=read_check(folder,plan)
    path=folder/'features/features.npz'
    x,m=e.npz_read(path,data.ids,eng.WIDTH,feature_binding(plan,ck))
    if m['extra']!={'runtime':ck['runtime'],'size':ck['size'],'width':eng.WIDTH}:raise ValueError('Feature schema changed')
    return x,e.file_ref(path)


def extract(folder=OUT,device_name='cuda'):
    data,_,_,plan=validate(folder);ck=read_check(folder,plan)
    torch,device,rt=runtime(device_name)
    if rt!=ck['runtime']:raise ValueError('Runtime differs from locked check')
    path=folder/'features/features.npz'
    if path.exists():load_features(folder,data,plan);print('All500 features verified; no new forwards.');return
    from PIL import Image
    model=eng.load_model().to(device);size=ck['size'];scope=feature_binding(plan,ck)
    values=[];t0=time.perf_counter()
    for start in range(0,len(data.ids),16):
        ids=data.ids[start:start+16];part=folder/f'features/chunks/{start:04d}.npz'
        if part.exists():
            x,_=e.npz_read(part,ids,eng.WIDTH,scope)
        else:
            rows=[]
            for ident in ids:
                with Image.open(ROOT/'open/train'/f'{ident}.png') as im:
                    if im.mode!='L' or im.size!=(256,256):raise ValueError('Unexpected native image schema')
                    a=np.array(im,dtype=np.uint8)
                raw=torch.from_numpy(a.copy()).to(device=device,dtype=torch.float32)[None,None]/255.
                rows.append(forward(model,raw,size,torch).cpu().numpy()[0])
            x=np.stack(rows).astype(np.float32)
            e.npz_save(part,ids,x,scope)
        values.append(x)
        print(f'C-RADIO extracted/verified {start+len(ids)}/500; size={size}; elapsed={time.perf_counter()-t0:.1f}s',flush=True)
    del model
    torch.cuda.empty_cache()
    validate(folder)
    e.npz_save(path,data.ids,np.concatenate(values),scope,{'runtime':rt,'size':size,'width':eng.WIDTH})
    load_features(folder,data,plan)
    print('Features ready: '+str(path))


def fit_fold(folder,binding,train_x,train_y,query_x,inner_scattering,seed,allow_fit=True,runner=None):
    """No outer-query labels or global OOF inputs are accepted."""
    runner=runner or eng.ridge.fit_candidates
    y=eng.ridge.finite(train_y)
    s=eng.ridge.finite(inner_scattering,y.shape)
    if set(train_x)!={'large','radio'} or set(query_x)!=set(train_x):raise ValueError('Both encoders required')
    splits=eng.ridge.inner_splits(y,seed)
    predicted,choices={},{}
    for name in ('large','radio'):
        x=eng.ridge.finite(train_x[name]);q=eng.ridge.finite(query_x[name])
        if len(x)!=len(y) or x.ndim!=2 or q.ndim!=2 or x.shape[1]!=q.shape[1]:raise ValueError('Feature shape mismatch')
        inner={str(i):np.full(len(y),np.nan) for i in eng.RIDGE_INDICES}
        for k,(ti,vi) in enumerate(splits):
            scope={'parent':binding,'representation':name,'inner':k,'train':ti.tolist(),'query':vi.tolist(),
                'x':e.base.sha256_array(x[ti]),'q':e.base.sha256_array(x[vi]),'y':e.base.sha256_array(y[ti]),'candidates':list(eng.RIDGE_INDICES)}
            r=e.cell(folder/name/f'inner{k}.json',scope,
                lambda:runner(x[ti],y[ti],x[vi],eng.RIDGE_INDICES),allow_fit)
            if set(r['predictions'])!=set(inner):raise ValueError('Inner candidate schema changed')
            for i,p in r['predictions'].items():inner[i][vi]=eng.ridge.finite(p,(len(vi),))
        choice=eng.choose(inner,y,s);best=choice['blend']['index']
        indices=sorted({best,choice['standalone']['index']}) if name=='large' else [best]
        scope={'parent':binding,'representation':name,'x':e.base.sha256_array(x),'q':e.base.sha256_array(q),
            'y':e.base.sha256_array(y),'candidates':indices,'inner_scattering':e.base.sha256_array(s)}
        r=e.cell(folder/name/'refit.json',scope,lambda:runner(x,y,q,indices),allow_fit)
        if set(r['predictions'])!={str(i) for i in indices}:raise ValueError('Refit choice changed')
        predicted[name]=eng.ridge.finite(r['predictions'][str(best)],(len(q),))
        if name=='large':predicted['large_anchor']=eng.ridge.finite(r['predictions'][str(choice['standalone']['index'])],(len(q),))
        choices[name]=choice
    return predicted,choices


def collect(folder,binding,data,large,radio,ref,seed,allow_fit=True,runner=None):
    vectors={n:np.full(len(data.y),np.nan) for n in ('large','radio','large_anchor')};choices=[]
    with threadpool_limits(limits=2):
        for fold in range(5):
            ti=np.flatnonzero(ref['folds']!=fold);vi=np.flatnonzero(ref['folds']==fold)
            if (len(ti),len(vi))!=(400,100):raise ValueError('Expected five400/100 folds')
            scope={'evaluation':binding,'fold':fold,'train':ti.tolist(),'query':vi.tolist()}
            p,c=fit_fold(folder/f'fold{fold}',scope,{'large':large[ti],'radio':radio[ti]},data.y[ti],
                {'large':large[vi],'radio':radio[vi]},ref['inner_scattering'][fold],seed+fold,allow_fit,runner)
            np.testing.assert_allclose(p['large_anchor'],ref['large'][vi],rtol=0,atol=1e-8)
            if c['large']['standalone']['alpha']!=ref['large_alphas'][fold]:raise ValueError('Original alpha anchor differs')
            for n in vectors:vectors[n][vi]=p[n]
            choices.append({'fold':fold,'choices':c})
            if allow_fit:print(f'Outer{fold}: radio alpha={c["radio"]["blend"]["alpha"]:g}; matched Large alpha={c["large"]["blend"]["alpha"]:g}; original anchor matched; outer labels unused',flush=True)
    output=eng.compose(ref['scattering'],ref['large'],vectors['radio'],vectors['large'])
    np.testing.assert_allclose(output['current'],ref['current'],rtol=0,atol=1e-12)
    return output,choices


def result_data(data,ref,p,choices,plan,seed,binding):
    mapped={diagnostics.PRIMARY if n==PRIMARY else n:v for n,v in p.items()}
    s,files=diagnostics.result_data(data,ref,mapped,choices,plan,seed)
    for key in ('metrics','diagnosis'):s[key][PRIMARY]=s[key].pop(diagnostics.PRIMARY)
    for rows in files.values():
        for row in rows:
            for k in list(row):
                new=k.replace(diagnostics.PRIMARY,PRIMARY)
                if k!=new:row[new]=row.pop(k)
            if row.get('model')==diagnostics.PRIMARY:row['model']=PRIMARY
    s.update(binding=binding,primary=PRIMARY,secondary=SECONDARY,selection=choices,anchor_matched=True,
             inner_scattering_uses_global_oof=False,new_tabpfn_fits=0)
    for n,m in s['metrics'].items():
        ctrl='large_retuned_ridge' if n in ('radio_ridge','large_retuned_ridge') else ('large_retuned_complementary' if n in (SECONDARY,'large_retuned_complementary') else CONTROL)
        m.update(matched_control=ctrl,gain_vs_control=s['metrics'][ctrl]['rmse']-m['rmse'])
    return s,files


def report_text(s):
    lines=[f'Frozen C-RADIOv4 blend-aware Ridge seed{s["seed"]}',PROTOCOL['current'],'Gain>0 means improvement.']
    for n,m in s['metrics'].items():lines.append(f'{n}: RMSE={m["rmse"]:.6f}; gain vs current={m["gain_vs_current"]:+.6f}; gain vs matched control={m["gain_vs_control"]:+.6f}')
    for n in (PRIMARY,SECONDARY):
        lines.append(n+' diagnostics:')
        for g in ('common16','remaining484','noiseQ1','noiseQ2','noiseQ3','noiseQ4','low10','high10','middle80'):
            m=s['diagnosis'][n]['groups'][g];lines.append(f'  {g}: RMSE={m["rmse"]:.6f}; gain={m["gain_vs_current"]:+.6f}')
        a=s['diagnosis'][n];lines.append(f'  Common16 improved={a["common16_improved"]}/16, worst20={a["common16_in_worst20"]}/16, same error sign={a["common16_same_error_sign"]}/16; remaining484 improved={a["remaining484_improved"]}/484.')
    return '\n'.join(lines+[PROTOCOL['selection'],PROTOCOL['decision'],PROTOCOL['limits']])+'\n'


def expected_outputs():
    return {'oof.csv','diagnostics.csv','per_image_errors.csv','common16.csv','summary.json','selection.json','report.txt'} | {
        f'fold{f}/{n}/{cell}.json' for f in range(5) for n in ('large','radio') for cell in ('inner0','inner1','inner2','refit')}


def completed(folder,binding):
    path=folder/'completed_evaluation.json'
    if not path.exists():return False
    d=e.read_json(path)
    if d['binding']!=binding or set(d['files'])!=expected_outputs():raise ValueError('Completion scope differs')
    for name,h in d['files'].items():
        if e.base.sha256_file(folder/name)!=h:raise ValueError('Completed output changed: '+name)
    return True


def evaluate(folder=OUT,seed=42,allow_fit=True):
    data,large,refs,plan=validate(folder);radio,fr=load_features(folder,data,plan)
    sub=folder/f'seed{seed}';binding=bind(plan,seed=seed,features=fr)
    done=completed(sub,binding)
    p,c=collect(sub,binding,data,large,radio,refs[seed],seed,allow_fit and not done,
                None if allow_fit and not done else forbidden)
    s,files=result_data(data,refs[seed],p,c,plan,seed,binding)
    if done:
        if e.read_json(sub/'summary.json')!=s or e.read_json(sub/'selection.json')!=c:raise ValueError('Saved summary differs from recomputation')
        if (sub/'report.txt').read_text(encoding='utf-8')!=report_text(s):raise ValueError('Saved report differs')
        for n,rows in files.items():
            with (sub/n).open(encoding='utf-8',newline='') as f:
                if list(csv.DictReader(f))!=list(csv.DictReader(io.StringIO(core.csv_text(rows)))):raise ValueError('CSV mismatch: '+n)
        print(f'Completed seed{seed} verified; no fitting.');return s
    if not allow_fit:raise FileNotFoundError('Incomplete evaluation')
    for name,rows in files.items():e.write_csv(sub/name,rows)
    e.write_json(sub/'summary.json',s);e.write_json(sub/'selection.json',c)
    (sub/'report.txt').write_text(report_text(s),encoding='utf-8')
    e.write_json(sub/'completed_evaluation.json',{'binding':binding,'files':{n:e.base.sha256_file(sub/n) for n in sorted(expected_outputs())}})
    print(report_text(s));print('Report: '+str(sub/'report.txt'));return s


def aggregate(summaries):
    if [s['seed'] for s in summaries]!=list(SEEDS):raise ValueError('Both prespecified seeds required')
    comparisons={}
    for name in (PRIMARY,SECONDARY):
        gains=[s['metrics'][name]['gain_vs_current'] for s in summaries]
        controls=[s['metrics'][name]['gain_vs_control'] for s in summaries]
        all_improve=all(v>0 for v in gains+controls)
        comparisons[name]={'mean_gain_vs_current':float(np.mean(gains)),'mean_gain_vs_control':float(np.mean(controls)),
            'all_seeds_improve_both':all_improve,'passes_numeric_expansion_priority':all_improve and np.mean(gains)>=.3}
    # Convert NumPy comparison to JSON-safe bool.
    for v in comparisons.values():v['passes_numeric_expansion_priority']=bool(v['passes_numeric_expansion_priority'])
    return {'primary':PRIMARY,'secondary':SECONDARY,'comparisons':comparisons,
        'per_seed':[{'seed':s['seed'],'metrics':s['metrics'],'diagnosis':s['diagnosis']} for s in summaries],
        'automatic_adoption':False,'independent_holdout':False,'additional_review_required':PROTOCOL['decision'],
        'aggregation':'Mean per-seed OOF RMSE differences; same500; no prediction averaging or independent-sample inference.'}


def compare(folder=OUT):
    summaries=[evaluate(folder,seed,False) for seed in SEEDS]
    result=aggregate(summaries);result['protocol_sha256']=summaries[0]['binding']['plan']
    e.write_json(folder/'comparison.json',result)
    print(e.base.canonical_json(result))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','check','extract','evaluate','compare'))
    parser.add_argument('--device',default='cuda');parser.add_argument('--seed',type=int,choices=SEEDS,default=42)
    parser.add_argument('--output-dir',type=Path,default=OUT)
    a=parser.parse_args();folder=a.output_dir if a.output_dir.is_absolute() else ROOT/a.output_dir
    if a.command=='prepare':prepare(folder)
    elif a.command=='check':check(folder,a.device)
    elif a.command=='extract':extract(folder,a.device)
    elif a.command=='evaluate':evaluate(folder,a.seed)
    else:compare(folder)


if __name__=='__main__':main()
