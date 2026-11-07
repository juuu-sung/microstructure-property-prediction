"""User-run DINOv3-L592 versus DINOv2-L518; retain Scattering and C-RADIO."""
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
    from . import dinov3_large_engine as eng
    from . import cradiov4_experiment as prior
else:
    import dinov3_large_engine as eng
    import cradiov4_experiment as prior

e, core, diagnostics = prior.e, prior.core, prior.diagnostics
ROOT = e.ROOT
SOURCE = ROOT/'reports/cradiov4_importfix'
OUT = ROOT/'reports/dinov3_large'
SEEDS = (42,2026)
PRIMARY = 'dinov3_replaced_blend'
PROTOCOL = {
    'version':1, 'primary':PRIMARY, 'seeds':list(SEEDS),
    'encoder':'Pinned timm DINOv3 ViT-L/16 LVD1689M,303079424 parameters,frozen/eval. No encoder updates.',
    'input':'Native256 gray -> continuous bicubic592 full image,align_corners=False,antialias=True,no clamp; repeatRGB; official ImageNet normalization. No crops/denoise/D4.',
    'pooling':'Post-normalization CLS1024 + spatial mean1024 + population std1024 =3072; exclude4 registers.',
    'comparison':'DINOv3-L592 patch16 and archived DINOv2-L518 patch14 both37x37 tokens/3072features. Architecture/pretraining differ; not a pure objective ablation.',
    'precision':'FP32 input/weights/forward/pooling,batch1,no AMP/TF32; synthetic CUDA check first; no automatic size or precision fallback.',
    'ridge':'X/y StandardScaler context-only, Ridge alpha100/1000/10000; exact ties prefer largeralpha; no PCA.',
    'selection':'Same prior C-RADIO rule for BOTH DINO versions: pooled INNER MSE of .5S_inner+.5encoder, three rank-decile inner folds seed+fold. This surrogate is NOT the final50:25:25 objective.',
    'inner_scattering':'Reuse verified context-only Scattering caches. No global OOF substitution; every query excluded from fitting/scaler/inner alpha selection of base models.',
    'reference':'.5S+.25L_original+.25C = latest submitted formula; S is entire Scattering486 ensemble. C and all existing model predictions stay fixed.',
    'control':'.5S+.25L_retuned+.25C; same features and selection objective/grid as candidate. Reproduce original standalone-selected L and previous blend-selected L anchors.',
    'candidate':'.5S+.25DINOv3+.25C; replace ONLY Large51825%. No weight search.',
    'standalone':'dinov3_ridge uses the SAME blend-selected alpha, not a separately optimized score.',
    'diagnostics':'Common16/remaining484,low10/high10/middle80,noiseQ1..4,folds,newworst20,error signs,residual correlation; controls and standalone included.',
    'priority':'Both seeds improve current AND matched control; mean RMSEgain>=0.30 prioritizes expansion only. Group tradeoffs reported separately; no automatic adoption.',
    'scope':'Train500 only; no test-image access, final fit, submission, encoder or TabPFN fitting. User runs feature extraction and Ridge training.',
    'limits':'Repeatedly studied same500/reused folds, not independent holdout. C-RADIO contains DINOv3 teacher information; incremental signal may be small. No guaranteed Public improvement.',
    'sources':['https://huggingface.co/timm/vit_large_patch16_dinov3.lvd1689m','https://github.com/facebookresearch/dinov3'],
    'attribution':'DINOv3 by Meta, DINOv3 License; timm port by Ross Wightman. License and model card preserved.'}


def forbidden(*a,**k):
    raise RuntimeError('Missing completed prerequisite; implicit fitting/forward prohibited')


def load_state():
    eng.assets.validate(require_weights=False)
    with contextlib.redirect_stdout(io.StringIO()):
        data,large,old,oldplan = prior.validate(SOURCE)
    refs,files = {},{}
    for seed in SEEDS:
        # Reconstruct old results from hashed cells, without refitting, before
        # reading the OOF columns used as the new reference and matched anchor.
        with contextlib.redirect_stdout(io.StringIO()):
            prior.evaluate(SOURCE,seed,False)
        sub = SOURCE/f'seed{seed}'
        with (sub/'oof.csv').open(encoding='utf-8',newline='') as f:
            rows = list(csv.DictReader(f))
        if [r['ID'] for r in rows] != data.ids.tolist():
            raise ValueError('C-RADIO OOF ID order changed')
        np.testing.assert_array_equal([float(r['hardness']) for r in rows],data.y)
        np.testing.assert_array_equal([int(r['fold']) for r in rows],old[seed]['folds'])
        values = {n:eng.ridge.finite([float(r[n]) for r in rows],data.y.shape)
                  for n in ('radio_ridge','radio_complementary_blend','large_retuned_ridge')}
        refs[seed] = {**old[seed], 'radio':values['radio_ridge'],
            'current':values['radio_complementary_blend'], 'retuned_large':values['large_retuned_ridge']}
        check = eng.compose(old[seed]['scattering'],refs[seed]['radio'],old[seed]['large'],
                            refs[seed]['retuned_large'],refs[seed]['retuned_large'])
        np.testing.assert_allclose(check['current'],refs[seed]['current'],rtol=0,atol=1e-12)
        files[str(seed)] = {n:e.file_ref(sub/n) for n in ('completed_evaluation.json','oof.csv','summary.json')}
    paths = [Path(__file__),Path(eng.__file__),Path(eng.assets.__file__),eng.assets.LOCK,
             ROOT/'scripts/run_dinov3_large.ps1',Path(prior.__file__),Path(eng.previous.__file__),
             Path(eng.ridge.__file__),Path(diagnostics.__file__)]
    frozen = {'protocol':PROTOCOL, 'assets':eng.assets.spec(),
        'sources':{p.name:e.file_ref(p) for p in paths}, 'environment':e.environment(),
        'source_protocol':e.file_ref(SOURCE/'protocol.json'), 'references':files,
        'large_features':oldplan['frozen']['large_features'],
        'training_inventory':oldplan['frozen']['training_inventory'],
        'common_failure_ids':oldplan['frozen']['common_failure_ids'],
        'ids_sha256':e.base.sha256_array(data.ids), 'y_sha256':e.base.sha256_array(data.y),
        'preserved_submissions':{name:e.file_ref(ROOT/f'reports/{name}/submission/submission.csv')
            for name in ('final_scattering_blend','final_large518_blend','final_cradiov4_blend')}}
    return data,large,refs,frozen


def prepare(folder=OUT):
    data,_,refs,frozen = load_state()
    path = folder/'protocol.json'
    if path.exists():
        if e.read_json(path)['frozen'] != frozen:
            raise ValueError('Prepared inputs/code changed; preserve directory')
    else:
        if folder.exists() and any(folder.iterdir()):
            raise ValueError('Nonempty unprepared directory')
        e.write_json(path,{'frozen':frozen,'created_unix':time.time(),
            'competition_fits':0,'competition_forwards':0})
    print('Prepared DINOv3-L592,3072features; replace Large25%, keep S50%/C25%. No fits/forwards.')
    for seed in SEEDS:
        print(f'Current seed{seed}: {core.metrics(data.y,refs[seed]["current"])["rmse"]:.6f}')


def validate(folder=OUT):
    data,large,refs,frozen = load_state()
    plan = e.read_json(folder/'protocol.json')
    if plan['frozen'] != frozen:
        raise ValueError('Frozen sources/inputs/environment changed')
    return data,large,refs,plan


def bind(plan,**extra):
    return {'plan':e.digest(plan['frozen']),**extra}


def runtime(device_name):
    torch,device,info = e.configure(device_name)
    if device.type != 'cuda':
        raise ValueError('This experiment is locked to CUDA FP32')
    if torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:
        raise ValueError('TF32 must be disabled')
    return torch,device,{'device':str(device),'gpu':info.get('gpu_name'),
        'batch_size':1,'dtype':'float32','amp':False,'tf32':False,'size':eng.SIZE}


def forward(model,raw,torch):
    with torch.inference_mode(),torch.autocast(device_type='cuda',enabled=False):
        return eng.pool(model.forward_features(eng.transform(raw)))


def read_check(folder,plan):
    saved = e.read_json(folder/'synthetic_check.json')
    r = saved['result']
    if (saved['binding'] != bind(plan) or saved['sha256'] != e.digest(r)
            or r['shape'] != [1,eng.WIDTH] or r['competition_forwards'] != 0
            or not r['encoder_frozen'] or r['size'] != eng.SIZE):
        raise ValueError('Synthetic check scope/schema changed')
    return r


def check(folder=OUT,device_name='cuda'):
    _,_,_,plan = validate(folder)
    torch,device,rt = runtime(device_name)
    eng.assets.validate()
    if (folder/'synthetic_check.json').exists():
        if read_check(folder,plan)['runtime'] != rt:
            raise ValueError('Checked runtime changed')
        print('Synthetic check verified; no competition inference.');return
    model = eng.load_model().to(device)
    raw = torch.linspace(0,1,256*256,device=device).reshape(1,1,256,256)
    torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()
    a,b = forward(model,raw,torch),forward(model,raw,torch)
    torch.cuda.synchronize(device)
    np.testing.assert_allclose(a.cpu().numpy(),b.cpu().numpy(),rtol=1e-5,atol=1e-5)
    r = {'runtime':rt,'size':eng.SIZE,'shape':list(a.shape),'seconds':time.perf_counter()-t0,
        'peak_cuda_bytes':torch.cuda.max_memory_allocated(device),
        'encoder_frozen':not any(p.requires_grad for p in model.parameters()),
        'competition_forwards':0,'max_repeat_difference':float((a-b).abs().max())}
    e.write_json(folder/'synthetic_check.json',{'binding':bind(plan),'result':r,'sha256':e.digest(r)})
    print(e.base.canonical_json(r))


def feature_binding(plan,ck):
    return bind(plan,stage='features',check=e.digest(ck))


def load_features(folder,data,plan):
    ck = read_check(folder,plan)
    path = folder/'features/features.npz'
    x,m = e.npz_read(path,data.ids,eng.WIDTH,feature_binding(plan,ck))
    if m['extra'] != {'runtime':ck['runtime'],'size':eng.SIZE,'width':eng.WIDTH}:
        raise ValueError('Feature metadata differs')
    return x,e.file_ref(path)


def extract(folder=OUT,device_name='cuda'):
    data,_,_,plan = validate(folder)
    ck = read_check(folder,plan)
    torch,device,rt = runtime(device_name)
    if ck['runtime'] != rt:
        raise ValueError('Runtime changed')
    path = folder/'features/features.npz'
    if path.exists():
        load_features(folder,data,plan);print('All500 cached features verified.');return
    from PIL import Image
    model = eng.load_model().to(device)
    scope = feature_binding(plan,ck)
    values=[];t0=time.perf_counter()
    for start in range(0,len(data.ids),16):
        ids=data.ids[start:start+16]
        part=folder/f'features/chunks/{start:04d}.npz'
        if part.exists():
            x,_=e.npz_read(part,ids,eng.WIDTH,scope)
        else:
            rows=[]
            for ident in ids:
                with Image.open(ROOT/'open/train'/f'{ident}.png') as im:
                    if im.mode!='L' or im.size!=(256,256):
                        raise ValueError('Expected native256 grayscale')
                    a=np.array(im,dtype=np.uint8)
                raw=torch.from_numpy(a.copy()).to(device=device,dtype=torch.float32)[None,None]/255.
                rows.append(forward(model,raw,torch).cpu().numpy()[0])
            x=np.stack(rows).astype(np.float32)
            e.npz_save(part,ids,x,scope)
        values.append(x)
        print(f'DINOv3 extracted/verified {start+len(ids)}/500; elapsed={time.perf_counter()-t0:.1f}s',flush=True)
    del model
    torch.cuda.empty_cache()
    validate(folder)
    e.npz_save(path,data.ids,np.concatenate(values),scope,{'runtime':rt,'size':eng.SIZE,'width':eng.WIDTH})
    load_features(folder,data,plan)
    print('Features ready: '+str(path))


def fit_fold(folder,binding,train_x,y,query_x,inner_scattering,seed,allow_fit=True,runner=None):
    """No outer-query labels/global OOF accepted. Refit matched2 and candidate3."""
    runner = runner or eng.ridge.fit_candidates
    y=eng.ridge.finite(y);s=eng.ridge.finite(inner_scattering,y.shape)
    if set(train_x)!={'dinov2','dinov3'} or set(query_x)!=set(train_x):
        raise ValueError('Both DINO versions required')
    predictions,choices={},{}
    for name in ('dinov2','dinov3'):
        x,q=eng.ridge.finite(train_x[name]),eng.ridge.finite(query_x[name])
        if x.ndim!=2 or q.ndim!=2 or len(x)!=len(y) or x.shape[1]!=q.shape[1]:
            raise ValueError('Invalid train/query feature schema')
        inner={str(i):np.full(len(y),np.nan) for i in eng.INDICES}
        for k,(ti,vi) in enumerate(eng.ridge.inner_splits(y,seed)):
            scope={'parent':binding,'representation':name,'inner':k,'train':ti.tolist(),'query':vi.tolist(),
                'x':e.base.sha256_array(x[ti]),'q':e.base.sha256_array(x[vi]),'y':e.base.sha256_array(y[ti]),'indices':list(eng.INDICES)}
            r=e.cell(folder/name/f'inner{k}.json',scope,lambda:runner(x[ti],y[ti],x[vi],eng.INDICES),allow_fit)
            if set(r['predictions'])!=set(inner):raise ValueError('Inner candidate schema changed')
            for i,p in r['predictions'].items():inner[i][vi]=eng.ridge.finite(p,(len(vi),))
        choice=eng.choose(inner,y,s);best=choice['blend']['index']
        indices=sorted({best,choice['standalone']['index']}) if name=='dinov2' else [best]
        scope={'parent':binding,'representation':name,'x':e.base.sha256_array(x),'q':e.base.sha256_array(q),
            'y':e.base.sha256_array(y),'indices':indices,'inner_scattering':e.base.sha256_array(s)}
        r=e.cell(folder/name/'refit.json',scope,lambda:runner(x,y,q,indices),allow_fit)
        if set(r['predictions'])!={str(i) for i in indices}:raise ValueError('Refit candidate schema changed')
        predictions[name]=eng.ridge.finite(r['predictions'][str(best)],(len(q),))
        if name=='dinov2':predictions['anchor']=eng.ridge.finite(r['predictions'][str(choice['standalone']['index'])],(len(q),))
        choices[name]=choice
    return predictions,choices


def collect(folder,binding,data,large,dino3,ref,seed,allow_fit=True,runner=None):
    vectors={n:np.full(len(data.y),np.nan) for n in ('dinov2','dinov3')};choices=[]
    with threadpool_limits(limits=2):
        for fold in range(5):
            ti=np.flatnonzero(ref['folds']!=fold);vi=np.flatnonzero(ref['folds']==fold)
            if (len(ti),len(vi))!=(400,100):raise ValueError('Expected400/100 partitions')
            scope={'evaluation':binding,'fold':fold,'train':ti.tolist(),'query':vi.tolist()}
            p,c=fit_fold(folder/f'fold{fold}',scope,{'dinov2':large[ti],'dinov3':dino3[ti]},data.y[ti],
                {'dinov2':large[vi],'dinov3':dino3[vi]},ref['inner_scattering'][fold],seed+fold,allow_fit,runner)
            np.testing.assert_allclose(p['anchor'],ref['large'][vi],rtol=0,atol=1e-8)
            np.testing.assert_allclose(p['dinov2'],ref['retuned_large'][vi],rtol=0,atol=1e-8)
            if c['dinov2']['standalone']['alpha']!=ref['large_alphas'][fold]:raise ValueError('DINOv2 alpha anchor changed')
            for n in vectors:vectors[n][vi]=p[n]
            choices.append({'fold':fold,'choices':c})
            if allow_fit:print(f'Outer{fold}: DINOv3 alpha={c["dinov3"]["blend"]["alpha"]:g}; DINOv2 anchors matched; outer labels unused',flush=True)
    p=eng.compose(ref['scattering'],ref['radio'],ref['large'],vectors['dinov2'],vectors['dinov3'])
    np.testing.assert_allclose(p['current'],ref['current'],rtol=0,atol=1e-12)
    return p,choices


def result_data(data,ref,p,choices,plan,seed,binding):
    mapped={diagnostics.PRIMARY if n==PRIMARY else n:v for n,v in p.items()}
    s,files=diagnostics.result_data(data,ref,mapped,choices,plan,seed)
    for k in ('metrics','diagnosis'):s[k][PRIMARY]=s[k].pop(diagnostics.PRIMARY)
    for rows in files.values():
        for row in rows:
            for k in list(row):
                new=k.replace(diagnostics.PRIMARY,PRIMARY)
                if k!=new:row[new]=row.pop(k)
            if row.get('model')==diagnostics.PRIMARY:row['model']=PRIMARY
    s.update(binding=binding,primary=PRIMARY,selection=choices,dinov2_anchors_matched=True,
        inner_scattering_uses_global_oof=False,new_tabpfn_fits=0)
    for n,m in s['metrics'].items():
        ctrl='dinov2_retuned_ridge' if n in ('dinov2_retuned_ridge','dinov3_ridge') else 'matched_control'
        m.update(matched_control=ctrl,gain_vs_control=s['metrics'][ctrl]['rmse']-m['rmse'])
    return s,files


def report_text(s):
    lines=[f'DINOv3-L592 branch replacement seed{s["seed"]}',PROTOCOL['reference'],'Gain>0 means improvement.']
    for n,m in s['metrics'].items():
        lines.append(f'{n}: RMSE={m["rmse"]:.6f}; gain vs current={m["gain_vs_current"]:+.6f}; gain vs control={m["gain_vs_control"]:+.6f}')
    for g in ('common16','remaining484','noiseQ1','noiseQ2','noiseQ3','noiseQ4','low10','high10','middle80'):
        a=s['diagnosis']['current']['groups'][g];b=s['diagnosis'][PRIMARY]['groups'][g]
        lines.append(f'{g}: {a["rmse"]:.6f} -> {b["rmse"]:.6f}; gain={b["gain_vs_current"]:+.6f}')
    d=s['diagnosis'][PRIMARY]
    lines += [f'Common16 improved={d["common16_improved"]}/16; in worst20={d["common16_in_worst20"]}/16; same sign={d["common16_same_error_sign"]}/16.',
        f'New worst20: {d["new_worst20_ids"]}',PROTOCOL['selection'],PROTOCOL['priority'],PROTOCOL['limits']]
    return '\n'.join(lines)+'\n'


def expected_outputs():
    return {'oof.csv','diagnostics.csv','per_image_errors.csv','common16.csv','summary.json','selection.json','report.txt'} | {
        f'fold{f}/{n}/{cell}.json' for f in range(5) for n in ('dinov2','dinov3') for cell in ('inner0','inner1','inner2','refit')}


def completed(folder,binding):
    path=folder/'completed_evaluation.json'
    if not path.exists():return False
    d=e.read_json(path)
    if d['binding']!=binding or set(d['files'])!=expected_outputs():raise ValueError('Completion scope differs')
    for n,h in d['files'].items():
        if e.base.sha256_file(folder/n)!=h:raise ValueError('Completed output changed: '+n)
    return True


def evaluate(folder=OUT,seed=42,allow_fit=True):
    data,large,refs,plan=validate(folder);dino3,fr=load_features(folder,data,plan)
    sub=folder/f'seed{seed}';binding=bind(plan,seed=seed,features=fr)
    done=completed(sub,binding)
    p,c=collect(sub,binding,data,large,dino3,refs[seed],seed,allow_fit and not done,
        None if allow_fit and not done else forbidden)
    s,files=result_data(data,refs[seed],p,c,plan,seed,binding)
    if done:
        if e.read_json(sub/'summary.json')!=s or e.read_json(sub/'selection.json')!=c:raise ValueError('Saved summary differs')
        if (sub/'report.txt').read_text(encoding='utf-8')!=report_text(s):raise ValueError('Saved report differs')
        for name,rows in files.items():
            with (sub/name).open(encoding='utf-8',newline='') as f:
                if list(csv.DictReader(f))!=list(csv.DictReader(io.StringIO(core.csv_text(rows)))):raise ValueError('Saved CSV differs')
        print(f'Completed seed{seed} verified; no fitting.');return s
    if not allow_fit:raise FileNotFoundError('Incomplete evaluation; no implicit fitting')
    for n,rows in files.items():e.write_csv(sub/n,rows)
    e.write_json(sub/'summary.json',s);e.write_json(sub/'selection.json',c)
    (sub/'report.txt').write_text(report_text(s),encoding='utf-8')
    e.write_json(sub/'completed_evaluation.json',{'binding':binding,
        'files':{n:e.base.sha256_file(sub/n) for n in sorted(expected_outputs())}})
    print(report_text(s));print('Report: '+str(sub/'report.txt'));return s


def aggregate(summaries):
    if [s['seed'] for s in summaries]!=list(SEEDS):raise ValueError('Both prespecified seeds required')
    rows=[]
    focus=('common16','remaining484','noiseQ4')
    for s in summaries:
        m=s['metrics'][PRIMARY];d=s['diagnosis'][PRIMARY]
        rows.append({'seed':s['seed'],'current_rmse':s['metrics']['current']['rmse'],
            'candidate_rmse':m['rmse'],'control_rmse':s['metrics']['matched_control']['rmse'],
            'standalone_rmse':s['metrics']['dinov3_ridge']['rmse'],
            'gain_vs_current':m['gain_vs_current'],'gain_vs_control':m['gain_vs_control'],
            'diagnosis':d})
    gain=float(np.mean([r['gain_vs_current'] for r in rows]))
    good=all(r['gain_vs_current']>0 and r['gain_vs_control']>0 for r in rows)
    return {'primary':PRIMARY,'per_seed':rows,'mean_gain_vs_current':gain,
        'mean_gain_vs_control':float(np.mean([r['gain_vs_control'] for r in rows])),
        'all_seeds_improve_both':good,'passes_numeric_expansion_priority':good and gain>=.3,
        'all_seeds_nonworsening_by_group':{g:all(r['diagnosis']['groups'][g]['gain_vs_current']>=0 for r in rows) for g in focus},
        'mean_gain_by_group':{g:float(np.mean([r['diagnosis']['groups'][g]['gain_vs_current'] for r in rows])) for g in focus},
        'automatic_adoption':False,'independent_holdout':False,'additional_review_required':PROTOCOL['diagnostics'],
        'aggregation':'Mean per-seed RMSE differences; same500, no prediction averaging or independent inference.'}


def compare(folder=OUT):
    summaries=[evaluate(folder,seed,False) for seed in SEEDS]
    result=aggregate(summaries);result['protocol_sha256']=summaries[0]['binding']['plan']
    e.write_json(folder/'comparison.json',result);print(e.base.canonical_json(result))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('prepare','check','extract','evaluate','compare'))
    p.add_argument('--device',default='cuda');p.add_argument('--seed',type=int,choices=SEEDS,default=42)
    p.add_argument('--output-dir',type=Path,default=OUT)
    a=p.parse_args();folder=a.output_dir if a.output_dir.is_absolute() else ROOT/a.output_dir
    if a.command=='prepare':prepare(folder)
    elif a.command=='check':check(folder,a.device)
    elif a.command=='extract':extract(folder,a.device)
    elif a.command=='evaluate':evaluate(folder,a.seed)
    else:compare(folder)


if __name__=='__main__':main()
