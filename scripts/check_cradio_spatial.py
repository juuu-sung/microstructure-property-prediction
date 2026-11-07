"""Synthetic-only, full geometry memory and speed measurement; no competition fits."""
from __future__ import annotations
import gc
import math
import time
import numpy as np
import torch
if __package__:
    from . import cradiov4_experiment as prior, cradio_spatial_engine as eng
else:
    import cradiov4_experiment as prior
    import cradio_spatial_engine as eng


def run(device_name):
    torch,device,runtime=prior.runtime(device_name)
    model=prior.eng.load_model().to(device)
    if any(p.requires_grad for p in model.parameters()):raise ValueError('Encoder is not frozen')
    raw=torch.linspace(0,1,256*256,device=device).reshape(1,1,256,256)
    elapsed=[];anchor=None
    for k in range(7):
        torch.cuda.synchronize(device);start=time.perf_counter()
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):out=model(prior.eng.transform(raw,592))
        pooled=prior.eng.pool(out,592)
        if anchor is None:anchor=pooled.detach().clone()
        else:torch.testing.assert_close(pooled,anchor,rtol=1e-5,atol=1e-5)
        if out[1].shape!=(1,1369,1152) or out[0].shape!=(1,2304):raise ValueError('Token shape changed')
        torch.cuda.synchronize(device)
        if k>=2:elapsed.append(time.perf_counter()-start)
    del out,pooled,model,raw,anchor;gc.collect();torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    # All500 feature rows resident, matching real training storage. Synthetic only.
    torch.manual_seed(314159)
    tokens=torch.randn(500,1369,1152,device=device,dtype=torch.float32)
    rng=np.random.default_rng(314159);summary=rng.normal(size=(500,2304)).astype(np.float32)
    store=eng.TokenStore(tokens,summary,np.zeros((500,1152)),np.ones((500,1152)),device)
    scale=store.tensors(store.scaling(np.arange(400),rng.normal(size=400)))
    results={}
    for kind in eng.REPS:
        torch.manual_seed(1234);head=eng.Readout(kind).to(device)
        opt=torch.optim.AdamW(head.parameters(),lr=eng.LR,weight_decay=eng.WD)
        target=torch.linspace(-1,1,eng.BATCH,device=device)
        times=[];first={k:v.detach().clone() for k,v in head.named_parameters()};gradmax={k:0. for k in first}
        for step in range(28):
            indices=(np.arange(eng.BATCH)+eng.BATCH*step)%500
            torch.cuda.synchronize(device);start=time.perf_counter();head.train();opt.zero_grad(set_to_none=True)
            pred=head(*store.batch(indices,scale));loss=(pred-target).square().mean()
            if not torch.isfinite(loss):raise ValueError('Nonfinite synthetic loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(),1.,error_if_nonfinite=True)
            for n,p in head.named_parameters():
                if p.grad is not None:gradmax[n]=max(gradmax[n],float(p.grad.abs().max()))
            opt.step();torch.cuda.synchronize(device)
            if step>=4:times.append(time.perf_counter()-start)
        updates={k:float((p.detach()-first[k]).abs().max()) for k,p in head.named_parameters()}
        if not all(v>0 for v in updates.values()) or not all(v>0 for v in gradmax.values()):
            raise ValueError('Some head parameters did not receive gradients/update')
        head.eval();inference=[]
        for k in range(12):
            torch.cuda.synchronize(device);start=time.perf_counter()
            with torch.inference_mode():pred=head(*store.batch(np.arange(eng.BATCH),scale))
            if not torch.isfinite(pred).all():raise ValueError('Bad synthetic prediction')
            torch.cuda.synchronize(device)
            if k>=2:inference.append(time.perf_counter()-start)
        results[kind]={'parameters':sum(p.numel() for p in head.parameters()),
            'median_train_step_seconds':float(np.median(times)),'median_inference_batch_seconds':float(np.median(inference)),
            'all_parameter_gradients_nonzero':True,'all_parameter_updates_nonzero':True,'steps_tested':28,
            'peak_allocated_bytes':torch.cuda.max_memory_allocated(device),'peak_reserved_bytes':torch.cuda.max_memory_reserved(device)}
        print(kind+': '+str(results[kind]),flush=True)
        del pred,head,opt,first;gc.collect();torch.cuda.empty_cache()
    # 30 inner trajectories and10 refits per kind, across the two seeds.
    def estimate(refit_epochs):
        return sum((30*17*40+10*25*refit_epochs)*r['median_train_step_seconds']+
                   (30*26*9+10*32)*r['median_inference_batch_seconds'] for r in results.values())
    base_low,base_high=estimate(10),estimate(40)
    extraction=500*float(np.median(elapsed))
    timing={'encoder500_forward_seconds':extraction,'head_compute_minutes_refit10':base_low/60,
        'head_compute_minutes_all_refit40':base_high/60,
        'planning_total_minutes_range':[math.ceil((base_low+extraction)*1.25/60+5),math.ceil((base_high+extraction)*1.8/60+15)],
        'assumptions':'60 inner trajectories run40epochs;20 refits selected0..40. Lower scenario refit10, upper40. Planning range adds IO/hashes/Ridge/scaling overhead; actual times vary with GPU contention. No measured competition performance.'}
    return {'passed':True,'runtime':runtime,'competition_fits':0,'competition_forwards':0,
        'synthetic_encoder_forward_median_seconds':float(np.median(elapsed)),'heads':results,
        'synthetic_store_shape':[500,1369,1152],'token_cache_GiB':500*1369*1152*4/2**30,
        'timing_estimate':timing,'torch_version':torch.__version__}
