"""User-run C-RADIO D4 Fisher contrast. Preserve S50% + LargeD4 25%."""
from __future__ import annotations
import argparse
import contextlib
import csv
import gc
import hashlib
import io
import os
from pathlib import Path
import shutil
import time
from types import SimpleNamespace
from dataclasses import asdict
import numpy as np
from threadpoolctl import threadpool_limits
if __package__:
    from . import large_radio_d4_experiment as prior, cradio_fisher_engine as eng
else:
    import large_radio_d4_experiment as prior
    import cradio_fisher_engine as eng

e, core, diagnostics = prior.e, prior.core, prior.diagnostics
ROOT, OUT, SEEDS = e.ROOT, e.ROOT/'reports/cradio_fisher_d4', (42,2026)
PRIMARY, CONTROL = eng.PRIMARY, eng.CONTROL
FOCUS = ('common16','remaining484','noiseQ4')
TOKEN_SHAPE = (500,8,1369,1152)
SAMPLE_SHAPE = (500,128,1152)
CHUNK = 8
PROTOCOL = {
    'version':1, 'seeds':list(SEEDS), 'primary':PRIMARY, 'control':CONTROL,
    'encoder':'Existing pinned SO400M592; frozen/eval FP32 weights, BF16 forward, noTF32, batch1 sequential8D4. Same CUDA uint8/255, bicubic/clamp/internal conditioner as archived Radio.',
    'features':'All1369 spatial tokens x1152 excluding summaries/registers. Store FP32 disk memmap500x8x1369x1152. Existing identity andD4 global4608 anchors on ALL500; no global feature replacement.',
    'sample':'16 deterministic random distinct positions perview,128/image, balanced across images/views. Seed from SHA256 of imageID; labels never used.',
    'dictionary':{**asdict(eng.SPEC), 'pca':'whiten=True,randomized,iterated_power4,context-only',
        'gmm':'diagonal,n_init1,kmeans initialization,context-only; nonconvergence stops, no silent retuning'},
    'encoding':'Per-view first/second-order Fisher512: signed square root thenL2; append raw soft occupancy8; FP64 equal8view average->FP32. Original global4608 concatenated with520 =5128.',
    'regression':'Train-context X/y StandardScaler+Ridge100/1000/10000; larger alpha wins exact ties. Both representations fixed, no outer-score feature choice.',
    'selection':'B(global) andC(global+FV) select alpha by pooledINNER MSE of .5S_inner+.25LargeD4_inner+.25C_inner. A preserves archived .5S+.5C surrogate selector.',
    'inner_baselines':'S_inner existing verified same-context cache. LargeD4 alpha selected by additional3fold CV wholly inside each inner context, then refit there. Never use globalOOF/outer400-selected alpha as inner baseline.',
    'validation':'5outer400/100 x3inner for eachseed. Every image/allviews/tokens stay together. Fit PCA/GMM separately for every inner context and outerrefit:40 dictionaries. Each inner/query transform uses its context dictionary only.',
    'blend':'A=.5S+.25LargeD4+.25archivedRadioD4; B replace C25% with retunedglobal; C replace C25% with global+FV. No mixture tuning; S=entireScattering486 ensemble.',
    'scope':'User runs syntheticCUDA check, train500 token extraction and CPU PCA/GMM/Ridge. No encoder/TabPFN updates,test reads,final500 supervised fit or submission changes.',
    'decision':'Inspect bothseeds overall,common16,remaining484,noiseQ1..4,tails,folds,newworst20 against current AND retunedglobal. .30 is expansion priority only; no automatic adoption.',
    'limits':'Same repeatedly studied500/reused splits; no independent holdout. Cluster IDs are not verified physical phases/grains. No predicted Public improvement.'}


def load_state():
    with contextlib.redirect_stdout(io.StringIO()):
        state = prior.validate()
        data, raw, old, upstream = state
        features, fr = prior.load_features(data,raw,upstream)
        for seed in SEEDS: prior.evaluate(seed,allow_fit=False,state=state)
    refs, files = {}, {}
    for seed in SEEDS:
        sub = prior.OUT/f'seed{seed}'
        rows = core.read_rows(sub/'oof.csv',{'ID','hardness','fold','large_d4','radio_d4','both_d4_blend'})
        if [r['ID'] for r in rows] != data.ids.tolist(): raise ValueError('Reference IDs/order changed')
        np.testing.assert_array_equal([float(r['hardness']) for r in rows],data.y)
        np.testing.assert_array_equal([int(r['fold']) for r in rows],old[seed]['folds'])
        refs[seed] = {**old[seed], **{n:np.array([float(r[n]) for r in rows]) for n in ('large_d4','radio_d4')},
            'current':np.array([float(r['both_d4_blend']) for r in rows])}
        r = refs[seed]
        np.testing.assert_allclose(r['current'],.5*r['scattering']+.25*r['large_d4']+.25*r['radio_d4'],rtol=0,atol=1e-10)
        files[str(seed)] = {n:e.file_ref(sub/n) for n in ('completed.json','oof.csv','summary.json')}
    paths = [Path(__file__),Path(eng.__file__),ROOT/'scripts/run_cradio_fisher.ps1']
    frozen = {'protocol':PROTOCOL, 'sources':{p.name:e.file_ref(p) for p in paths},
        'upstream_protocol':e.file_ref(prior.OUT/'protocol.json'), 'upstream_runtime':prior.read_check(upstream)['runtime'],
        'features':fr, 'references':files, 'environment':e.environment(),
        'training_inventory':upstream['frozen']['training_inventory'], 'common_failure_ids':upstream['frozen']['common_failure_ids'],
        'inner_reference_hashes':{str(s):{str(f):e.base.sha256_array(refs[s]['inner_scattering'][f]) for f in range(5)} for s in SEEDS},
        'submissions':{**upstream['frozen']['submissions'],
            'final_large_radio_d4_blend':e.file_ref(ROOT/'reports/final_large_radio_d4_blend/submission/submission.csv')}}
    return data, {'raw_radio':raw['radio'],**features}, refs, frozen


def prepare():
    *_, frozen = load_state(); path = OUT/'protocol.json'
    if path.exists():
        if e.read_json(path)['frozen'] != frozen: raise ValueError('Prepared sources/inputs changed; preserve existing results')
    else:
        if OUT.exists() and any(OUT.iterdir()): raise ValueError('Nonempty unprepared output directory')
        e.write_json(path,{'frozen':frozen,'created_unix':time.time()})
    print('Prepared/verified A=current, B=global retuned, C=global+Fisher; no fits/forwards.')


def validate():
    data, features, refs, frozen = load_state(); plan = e.read_json(OUT/'protocol.json')
    if plan['frozen'] != frozen: raise ValueError('Sources, inputs, environment or protocol changed')
    return data, features, refs, plan


def bind(plan, **kw): return {'plan':e.digest(plan['frozen']),**kw}


def array_hash(x):
    """Stream image slices, avoiding a multi-gigabyte tobytes allocation."""
    h = hashlib.sha256(str((x.shape,x.dtype.str)).encode())
    for part in x: h.update(np.ascontiguousarray(part).tobytes())
    return h.hexdigest()


def atomic_npz(path, arrays):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp = path.with_name(path.stem+'.pending.npz')
    np.savez(tmp,**arrays); os.replace(tmp,path)


def radio_tokens(model, raw):
    import torch
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        output = model(eng.d4.radio.transform(raw,592))
    # Pool outside autocast, exactly as the archived D4 extractor.
    pooled = eng.d4.radio.pool(output,592).detach().cpu().numpy()[0]
    tokens = output[1].detach().float().cpu().numpy()[0]
    if tokens.shape != (1369,1152) or not np.isfinite(tokens).all(): raise ValueError('Invalid spatial tokens')
    return pooled,tokens


def read_check(plan):
    obj = e.read_json(OUT/'synthetic_check.json'); r = obj['result']
    if (obj['binding'] != bind(plan) or obj['sha256'] != e.digest(r) or not r['passed']
            or r['competition_forwards'] != 0 or r['runtime'] != plan['frozen']['upstream_runtime']):
        raise ValueError('Synthetic capacity/anchor check missing or stale')
    return r


def check(device='cuda'):
    *_, plan = validate(); torch,dev,rt = prior.runtime(device)
    if rt != plan['frozen']['upstream_runtime']: raise ValueError('Runtime differs from archived D4')
    if (OUT/'synthetic_check.json').exists(): read_check(plan); print('Synthetic check verified; no repeat.'); return
    model = eng.d4.radio.load_model().to(dev)
    if model.training or any(p.requires_grad for p in model.parameters()): raise ValueError('Encoder must be frozen/eval')
    yy,xx = torch.meshgrid(torch.arange(256,device=dev),torch.arange(256,device=dev),indexing='ij')
    raw = ((xx*3+yy*7+(xx//13)*(yy//17))%256).float()[None,None]/255.
    radio_tokens(model,raw); torch.cuda.synchronize(dev); torch.cuda.reset_peak_memory_stats(dev)
    start = time.perf_counter(); pooled = []; shapes = []
    for view in eng.d4.views(raw):
        p,t = radio_tokens(model,view); pooled.append(p); shapes.append(list(t.shape))
    torch.cuda.synchronize(dev); seconds = time.perf_counter()-start
    identity,mean = eng.d4.extract_views('radio',model,raw)
    eng.d4.anchor('radio',np.array(pooled[:1]),identity)
    eng.d4.anchor('radio',np.mean(np.array(pooled,dtype=np.float64),axis=0)[None].astype(np.float32),mean)
    r = {'passed':True,'runtime':rt,'competition_forwards':0,'competition_fits':0,'encoder_updates':0,
        'eight_view_seconds':seconds,'estimated_train500_encoder_minutes':seconds*500/60,
        'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(dev),'spatial_shapes':shapes,
        'token_disk_GiB':int(np.prod(TOKEN_SHAPE))*4/2**30,
        'timing_note':'Encoder only; CPU dictionary fits/disk/hash times excluded. No actual competition images used.'}
    del model; gc.collect(); torch.cuda.empty_cache(); validate()
    e.write_json(OUT/'synthetic_check.json',{'binding':bind(plan),'result':r,'sha256':e.digest(r)})
    print(e.base.canonical_json(r),flush=True)


def token_scope(plan, ck): return bind(plan,stage='D4_spatial_tokens',check=e.digest(ck))


def load_tokens(data, features, plan):
    folder = OUT/'features'; receipt = e.read_json(folder/'complete.json')
    scope = token_scope(plan,read_check(plan))
    if receipt['binding'] != scope or receipt['encoder_updates'] != 0 or receipt['rows'] != 500:
        raise ValueError('Token cache scope differs')
    for name in ('tokens.npy','samples.npy','global.npz'):
        if e.file_ref(folder/name) != receipt['files'][name]: raise ValueError('Token cache changed: '+name)
    tokens = np.load(folder/'tokens.npy',mmap_mode='r',allow_pickle=False)
    samples = np.load(folder/'samples.npy',mmap_mode='r',allow_pickle=False)
    if tokens.shape != TOKEN_SHAPE or samples.shape != SAMPLE_SHAPE or tokens.dtype != np.float32 or samples.dtype != np.float32:
        raise ValueError('Token cache geometry/precision differs')
    pooled,_ = e.npz_read(folder/'global.npz',data.ids,9216,scope)
    eng.d4.anchor('radio',pooled[:,:4608],features['raw_radio'])
    eng.d4.anchor('radio',pooled[:,4608:],features['radio'])
    return SimpleNamespace(tokens=tokens,samples=samples,ids=data.ids),e.file_ref(folder/'complete.json')


def open_cache(path, shape):
    if not path.exists():
        a = np.lib.format.open_memmap(path,mode='w+',dtype=np.float32,shape=shape); a.flush(); del a
    a = np.lib.format.open_memmap(path,mode='r+')
    if a.shape != shape or a.dtype != np.float32: raise ValueError('Partial cache shape/precision changed')
    return a


def extract(device='cuda'):
    data,features,_,plan = validate(); ck = read_check(plan); scope = token_scope(plan,ck)
    if (OUT/'features/complete.json').exists(): load_tokens(data,features,plan); print('All500 D4 token rows verified; no new forwards.'); return
    torch,dev,rt = prior.runtime(device)
    if rt != ck['runtime']: raise ValueError('Extraction runtime changed')
    folder = OUT/'features'; folder.mkdir(parents=True,exist_ok=True)
    missing = sum(int(np.prod(shape))*4 for name,shape in (('tokens.npy',TOKEN_SHAPE),('samples.npy',SAMPLE_SHAPE)) if not (folder/name).exists())
    if shutil.disk_usage(folder).free < missing+2*2**30: raise ValueError('Need approximately26GiB free disk for FP32 token cache; no files removed')
    tokenmap = open_cache(folder/'tokens.npy',TOKEN_SHAPE); samplemap = open_cache(folder/'samples.npy',SAMPLE_SHAPE)
    model = None; rows = []; seconds = 0.; start_run = time.perf_counter()
    from PIL import Image
    try:
        for start in range(0,500,CHUNK):
            end = min(start+CHUNK,500); ids = data.ids[start:end]
            path = folder/f'chunks/{start:04d}.npz'; receipt = path.with_suffix('.json')
            b = {**scope,'start':start,'ids':ids.tolist()}
            if receipt.exists():
                obj = e.read_json(receipt)
                if (obj['binding'] != b or obj['pooled'] != e.file_ref(path)
                        or obj['tokens_hash'] != array_hash(tokenmap[start:end])
                        or obj['samples_hash'] != array_hash(samplemap[start:end])): raise ValueError('Partial token chunk changed')
                x,_ = e.npz_read(path,ids,9216,b)
            else:
                if model is None: model = eng.d4.radio.load_model().to(dev)
                t0 = time.perf_counter(); values = []
                for i in range(start,end):
                    with Image.open(ROOT/'open/train'/f'{data.ids[i]}.png') as im:
                        if im.mode != 'L' or im.size != (256,256): raise ValueError('Unexpected native image')
                        raw = torch.from_numpy(np.array(im,dtype=np.uint8,copy=True)).to(device=dev,dtype=torch.float32)[None,None]/255.
                    pp = []; pos = eng.sample_positions(data.ids[i])
                    for v,view in enumerate(eng.d4.views(raw)):
                        p,t = radio_tokens(model,view); tokenmap[i,v] = t; pp.append(p)
                        samplemap[i,v*16:(v+1)*16] = t[pos[v]]
                    identity = pp[0]; avg = np.mean(np.array(pp,dtype=np.float64),axis=0).astype(np.float32)
                    eng.d4.anchor('radio',identity[None],features['raw_radio'][i:i+1])
                    eng.d4.anchor('radio',avg[None],features['radio'][i:i+1])
                    values.append(np.concatenate((identity,avg)))
                x = np.stack(values).astype(np.float32)
                tokenmap.flush(); samplemap.flush(); e.npz_save(path,ids,x,b)
                obj = {'binding':b,'pooled':e.file_ref(path),'tokens_hash':array_hash(tokenmap[start:end]),
                    'samples_hash':array_hash(samplemap[start:end]),'seconds':time.perf_counter()-t0}
                e.write_json(receipt,obj)
            eng.d4.anchor('radio',x[:,:4608],features['raw_radio'][start:end])
            eng.d4.anchor('radio',x[:,4608:],features['radio'][start:end])
            rows.append(x); seconds += obj['seconds']
            print(f'Radio D4 dense tokens {end}/500 extracted/verified; elapsed={time.perf_counter()-start_run:.1f}s',flush=True)
        validate(); e.npz_save(folder/'global.npz',data.ids,np.concatenate(rows),scope)
        tokenmap.flush(); samplemap.flush()
        e.write_json(folder/'complete.json',{'binding':scope,'rows':500,'encoder_updates':0,'seconds':seconds,
            'files':{n:e.file_ref(folder/n) for n in ('tokens.npy','samples.npy','global.npz')}})
    finally:
        del tokenmap,samplemap,model; gc.collect(); torch.cuda.empty_cache()
    load_tokens(data,features,plan); print('Train500 D4 tokens ready; no encoder updates/test access.')


def distribution_features(folder, scope, store, ctx, query, seed, allow_fit=True):
    ctx,query = np.asarray(ctx),np.asarray(query)
    if len(set(ctx)) != len(ctx) or len(set(query)) != len(query) or set(ctx)&set(query): raise ValueError('Context/query images overlap')
    binding = {**scope,'dictionary_context':ctx.tolist(),'seed':int(seed),'spec':asdict(eng.SPEC)}
    path,receipt = folder/'dictionary.npz',folder/'dictionary.json'
    if receipt.exists():
        obj = e.read_json(receipt)
        if obj['binding'] != binding or obj['file'] != e.file_ref(path): raise ValueError('Dictionary cache changed')
        with np.load(path,allow_pickle=False) as z: params = {k:z[k].copy() for k in z.files}
    else:
        if not allow_fit: raise FileNotFoundError('Missing context dictionary; no implicit fitting')
        t0 = time.perf_counter()
        params,audit = eng.fit_dictionary(store.samples[ctx],seed,spec=eng.SPEC)
        atomic_npz(path,params)
        obj = {'binding':binding,'file':e.file_ref(path),'audit':audit,'fit_seconds':time.perf_counter()-t0}
        e.write_json(receipt,obj)
    eng.validate_dictionary(params,store.samples.shape[-1],spec=eng.SPEC)
    all_indices = np.concatenate((ctx,query)); ids = store.ids[all_indices]
    fb = {**scope,'dictionary':e.file_ref(receipt),'image_indices':all_indices.tolist()}
    fp = folder/'features.npz'
    if fp.exists():
        matrix,meta = e.npz_read(fp,ids,eng.SPEC.width,fb)
    else:
        if not allow_fit: raise FileNotFoundError('Missing Fisher features; no implicit transform')
        t0 = time.perf_counter(); out = []
        for k,i in enumerate(all_indices):
            out.append(eng.encode_image(store.tokens[i],params,spec=eng.SPEC))
            if (k+1)%100 == 0: print(f'  {folder.name}: Fisher encoded {k+1}/{len(all_indices)}; seconds={time.perf_counter()-t0:.1f}',flush=True)
        matrix = np.stack(out)
        meta = e.npz_save(fp,ids,matrix,fb,{'encode_seconds':time.perf_counter()-t0})
    return matrix[:len(ctx)],matrix[len(ctx):],{'dictionary':e.file_ref(receipt),'features':e.file_ref(fp),
        'audit':obj['audit'],'fit_seconds':obj['fit_seconds'],'encode_seconds':meta['extra']['encode_seconds']}


def fit_fold(folder, scope, store, features, ti, qi, y, scattering_inner, seed, fold, allow_fit=True):
    """Only outer-training y is accepted. No query labels or global OOF baseline."""
    y = eng.ridge.finite(y,(len(ti),)); other = .5*eng.ridge.finite(scattering_inner,y.shape).copy()
    splits = eng.ridge.inner_splits(y,seed+fold)
    inner = {rep:{str(i):np.full(len(y),np.nan) for i in eng.d4.INDICES} for rep in eng.REPS}
    audits = []
    for k,(a,b) in enumerate(splits):
        ctx,query = ti[a],ti[b]; model_seed = seed+1009*fold+k
        lb = {**scope,'role':'inner_large','inner':k,'context':ctx.tolist(),'query':query.tolist(),
            'x':e.base.sha256_array(features['large'][ctx]),'q':e.base.sha256_array(features['large'][query]),
            'y':e.base.sha256_array(y[a]),'seed':model_seed}
        r = e.cell(folder/f'inner_large{k}.json',lb,
            lambda:eng.inner_large(features['large'][ctx],y[a],features['large'][query],model_seed),allow_fit)
        other[b] += .25*eng.ridge.finite(r['predictions'],(len(b),))
        fv,fq,audit = distribution_features(folder/f'inner{k}',scope,store,ctx,query,model_seed,allow_fit)
        audits.append({'role':f'inner{k}',**audit,'large_alpha':r['alpha']})
        for rep in eng.REPS:
            x,q = features['radio'][ctx],features['radio'][query]
            if rep == 'global_plus_fisher': x,q = np.column_stack((x,fv)),np.column_stack((q,fq))
            binding = {**scope,'role':f'inner{k}','representation':rep,'context':ctx.tolist(),'query':query.tolist(),
                'x':e.base.sha256_array(x),'q':e.base.sha256_array(q),'y':e.base.sha256_array(y[a]),'candidates':list(eng.d4.INDICES)}
            rr = e.cell(folder/f'inner{k}/{rep}.json',binding,lambda:eng.ridge.fit_candidates(x,y[a],q,eng.d4.INDICES),allow_fit)
            if set(rr['predictions']) != set(inner[rep]): raise ValueError('Candidate prediction coverage changed')
            for i,p in rr['predictions'].items(): inner[rep][i][b] = eng.ridge.finite(p,(len(b),))
    chosen = {rep:eng.choose(inner[rep],y,other) for rep in eng.REPS}
    fv,fq,audit = distribution_features(folder/'refit',scope,store,ti,qi,seed+1009*fold+9,allow_fit)
    audits.append({'role':'refit',**audit}); predictions = {}
    for rep in eng.REPS:
        x,q = features['radio'][ti],features['radio'][qi]
        if rep == 'global_plus_fisher': x,q = np.column_stack((x,fv)),np.column_stack((q,fq))
        best = chosen[rep]['selected']['index']
        binding = {**scope,'role':'refit','representation':rep,'context':ti.tolist(),'query':qi.tolist(),
            'x':e.base.sha256_array(x),'q':e.base.sha256_array(q),'y':e.base.sha256_array(y),'candidate':best,
            'inner_other_sha256':e.base.sha256_array(other)}
        rr = e.cell(folder/f'refit/{rep}.json',binding,lambda:eng.ridge.fit_candidates(x,y,q,[best]),allow_fit)
        if set(rr['predictions']) != {str(best)}: raise ValueError('Refit selection differs')
        predictions[rep] = eng.ridge.finite(rr['predictions'][str(best)],(len(qi),))
    return predictions,{'choices':chosen,'dictionaries':audits,'inner_other_sha256':e.base.sha256_array(other)}


def report(s):
    lines = [f'C-RADIO D4 Fisher seed{s["seed"]}; only C25% changes; gain>0 means improvement.']
    for n,m in s['metrics'].items():
        lines.append(f'{n}: RMSE={m["rmse"]:.6f}; gain vs current={m["gain_vs_current"]:+.6f}; gain vs matched control={m["gain_vs_control"]:+.6f}')
    for n in (CONTROL,PRIMARY):
        d = s['diagnosis'][n]; lines.append(n+' groups:')
        for g in (*FOCUS,'noiseQ1','noiseQ2','noiseQ3','middle80','low10','high10'):
            m = d['groups'][g]
            lines.append(f'  {g}: RMSE={m["rmse"]:.6f}; gain vs current={m["gain_vs_current"]:+.6f}; gain vs control={m["gain_vs_control"]:+.6f}')
        lines += [f'  common16 improved={d["common16_improved"]}/16; worst20={d["common16_in_worst20"]}/16; same sign={d["common16_same_error_sign"]}/16.',
            f'  new worst20={d["new_worst20_ids"]}']
    return '\n'.join(lines+[PROTOCOL['selection'],PROTOCOL['decision'],PROTOCOL['limits']])+'\n'


def result_files(folder):
    return {str(p.relative_to(folder)) for p in folder.rglob('*') if p.is_file() and p.name != 'completed.json'}


def evaluate(seed, allow_fit=True, state=None, token_state=None):
    data,features,refs,plan = state or validate(); ref = refs[seed]
    store,fr = token_state or load_tokens(data,features,plan)
    scope = bind(plan,seed=seed,features=fr); folder = OUT/f'seed{seed}'; complete = folder/'completed.json'; done = complete.exists()
    if done:
        receipt = e.read_json(complete)
        if receipt['binding'] != scope or set(receipt['files']) != result_files(folder): raise ValueError('Completed scope/files changed')
        for n,h in receipt['files'].items():
            if e.base.sha256_file(folder/n) != h: raise ValueError('Completed artifact changed: '+n)
    elif not allow_fit: raise FileNotFoundError('Incomplete evaluation; no implicit fitting')
    pred = {rep:np.full(len(data.ids),np.nan) for rep in eng.REPS}; choices = []; start = time.perf_counter()
    with threadpool_limits(limits=2):
        for f in range(5):
            ti,qi = np.flatnonzero(ref['folds'] != f),np.flatnonzero(ref['folds'] == f)
            if (len(ti),len(qi)) != (400,100): raise ValueError('Outer split changed')
            p,c = fit_fold(folder/f'fold{f}',{**scope,'fold':f},store,features,ti,qi,data.y[ti],
                ref['inner_scattering'][f],seed,f,allow_fit and not done)
            for rep in pred: pred[rep][qi] = p[rep]
            choices.append({'fold':f,**c})
            if not done:
                print(f'Outer{f}: global alpha={c["choices"]["global_retuned"]["selected"]["alpha"]:g}; Fisher alpha={c["choices"]["global_plus_fisher"]["selected"]["alpha"]:g}; elapsed={time.perf_counter()-start:.1f}s; outer labels unused',flush=True)
    vectors = eng.compose(ref['scattering'],ref['large_d4'],ref['radio_d4'],pred['global_retuned'],pred['global_plus_fisher'])
    np.testing.assert_allclose(vectors['current'],ref['current'],rtol=0,atol=1e-10)
    mapped = {diagnostics.PRIMARY if n == PRIMARY else n:v for n,v in vectors.items()}
    s,files = diagnostics.result_data(data,ref,mapped,choices,plan,seed)
    for key in ('metrics','diagnosis'): s[key][PRIMARY] = s[key].pop(diagnostics.PRIMARY)
    for rows in files.values():
        for row in rows:
            for key in list(row):
                if diagnostics.PRIMARY in key: row[key.replace(diagnostics.PRIMARY,PRIMARY)] = row.pop(key)
            if row.get('model') == diagnostics.PRIMARY: row['model'] = PRIMARY
    for n,m in s['metrics'].items():
        ctrl = 'global_retuned_ridge' if n in ('radio_d4_ridge','global_retuned_ridge','global_plus_fisher_ridge') else CONTROL
        m['gain_vs_control'] = s['metrics'][ctrl]['rmse']-m['rmse']
        for g,gm in s['diagnosis'][n]['groups'].items(): gm['gain_vs_control'] = s['diagnosis'][ctrl]['groups'][g]['rmse']-gm['rmse']
    files['selection.csv'] = [{'fold':c['fold'],'representation':rep,**row,
        'selected':row['index']==detail['selected']['index']} for c in choices for rep,detail in c['choices'].items() for row in detail['scores']]
    files['dictionary_audit.csv'] = [{'fold':c['fold'],'role':d['role'],**d['audit'],
        'fit_seconds':d['fit_seconds'],'encode_seconds':d['encode_seconds']} for c in choices for d in c['dictionaries']]
    s.update(binding=scope,primary=PRIMARY,control=CONTROL,selection=choices,current_d4_anchor_matched=True,
        inner_uses_global_oof=False,new_tabpfn_fits=0,dictionary_contexts=20)
    if done:
        if e.read_json(folder/'summary.json') != s or (folder/'report.txt').read_text(encoding='utf-8') != report(s): raise ValueError('Report reconstruction differs')
        for n,rows in files.items():
            with (folder/n).open(encoding='utf-8',newline='') as handle:
                if list(csv.DictReader(handle)) != list(csv.DictReader(io.StringIO(core.csv_text(rows)))): raise ValueError('CSV reconstruction differs: '+n)
        print(f'Completed seed{seed} verified; no refitting.'); return s
    for n,rows in files.items(): e.write_csv(folder/n,rows)
    e.write_json(folder/'summary.json',s); (folder/'report.txt').write_text(report(s),encoding='utf-8')
    e.write_json(complete,{'binding':scope,'files':{n:e.base.sha256_file(folder/n) for n in sorted(result_files(folder))}})
    print(report(s)); print('Report: '+str(folder/'report.txt')); return s


def compare(state=None, summaries=None, token_state=None):
    state = state or validate()
    token_state = token_state or load_tokens(state[0],state[1],state[-1])
    summaries = summaries or [evaluate(seed,False,state,token_state) for seed in SEEDS]
    if [s['seed'] for s in summaries] != list(SEEDS): raise ValueError('Both fixed seeds required')
    result = {'primary':PRIMARY,'control':CONTROL,'automatic_adoption':False,'independent_holdout':False,
        'protocol_sha256':e.digest(state[-1]['frozen']),'comparisons':{},
        'per_seed':[{'seed':s['seed'],'metrics':s['metrics'],'diagnosis':s['diagnosis']} for s in summaries],
        'aggregation':'Mean per-seed differences; same500/reused splits, no prediction averaging or independent inference.',
        'additional_review_required':PROTOCOL['decision']}
    for name in (CONTROL,PRIMARY):
        good = all(s['metrics'][name]['gain_vs_current']>0 for s in summaries)
        gains = [s['metrics'][name]['gain_vs_current'] for s in summaries]
        both = good and all(s['metrics'][name]['gain_vs_control']>0 for s in summaries)
        result['comparisons'][name] = {'mean_gain_vs_current':float(np.mean(gains)),
            'mean_gain_vs_control':float(np.mean([s['metrics'][name]['gain_vs_control'] for s in summaries])),
            'all_seeds_improve_current':good,'all_seeds_improve_both':both,
            'mean_group_gains':{g:float(np.mean([s['diagnosis'][name]['groups'][g]['gain_vs_current'] for s in summaries])) for g in FOCUS},
            'all_seeds_nonworsening_vs_current':{g:all(s['diagnosis'][name]['groups'][g]['gain_vs_current']>=0 for s in summaries) for g in FOCUS},
            'all_seeds_nonworsening_vs_control':{g:all(s['diagnosis'][name]['groups'][g]['gain_vs_control']>=0 for s in summaries) for g in FOCUS},
            'passes_expansion_priority':both and float(np.mean(gains))>=.3}
    e.write_json(OUT/'comparison.json',result)
    print(e.base.canonical_json({k:v for k,v in result.items() if k!='per_seed'})); print('Comparison: '+str(OUT/'comparison.json')); return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('prepare','check','extract','evaluate','verify','compare'))
    p.add_argument('--device',default='cuda'); p.add_argument('--seed',type=int,choices=SEEDS); a = p.parse_args()
    if a.command == 'prepare': prepare()
    elif a.command == 'check': check(a.device)
    elif a.command == 'extract': extract(a.device)
    else:
        state = validate(); token_state = load_tokens(state[0],state[1],state[-1])
        if a.command == 'compare': compare(state,token_state=token_state)
        else:
            results = [evaluate(seed,a.command=='evaluate',state,token_state) for seed in (SEEDS if a.seed is None else (a.seed,))]
            if a.seed is None: compare(state,results,token_state)


if __name__ == '__main__': main()
