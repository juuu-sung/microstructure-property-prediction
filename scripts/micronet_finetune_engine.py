"""Trainable MicroNet stages6/7; paired head-only control, local checkpoints."""
from __future__ import annotations
import gc
import hashlib
import math
import os
from pathlib import Path
import time
import numpy as np

if __package__:
    from . import efficientnet_micronet_experiment as old
else:
    import efficientnet_micronet_experiment as old
t, base = old.t, old.base
MODES = ('head_only', 'partial')
EPOCHS = (0, 5, 10, 15, 20, 25, 30, 35, 40)
WARMUP, BATCH, ACCUM = 5, 8, 2


def group(name):
    if name.startswith('head.'):
        return 'head'
    if name.startswith('encoder._blocks.'):
        i = int(name.split('.')[2])
        if 11 <= i <= 14: return 'stage6'
        if i == 15: return 'stage7'
    return 'frozen'


def make_model(torch):
    class Regressor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = old.load_model('micronet')
            layout = [b._block_args.output_filters for b in self.encoder._blocks]
            if layout != [16,24,24,40,40,80,80,80,112,112,112,192,192,192,192,320]:
                raise ValueError('Unexpected B0 semantic stage layout')
            self.head = torch.nn.Sequential(torch.nn.LayerNorm(864),torch.nn.Dropout(.1),torch.nn.Linear(864,1))

        def forward(self,x):
            return self.head(old.pool_maps(old.forward_maps(self.encoder,x))).squeeze(-1)
    return Regressor()


def phase(model,mode,epoch):
    if mode not in MODES: raise ValueError('Unknown training mode')
    model.train()
    model.encoder.eval()  # BN running statistics and stochastic backbone layers stay frozen.
    for name,p in model.named_parameters():
        p.requires_grad_(group(name)=='head' or (mode=='partial' and epoch>WARMUP and group(name) in ('stage6','stage7')))


def optimizer_for(torch,model,mode,epoch):
    phase(model,mode,epoch)
    groups = {}
    for name,p in model.named_parameters():
        if not p.requires_grad: continue
        lr = (.001 if epoch<=WARMUP else .0003) if group(name)=='head' else 1e-5
        decay = .01 if p.ndim>1 and not name.endswith('.bias') else 0.
        groups.setdefault((lr,decay),{'params':[],'lr':lr,'base_lr':lr,'weight_decay':decay})['params'].append(p)
    return torch.optim.AdamW(list(groups.values()))


def lr_factor(epoch):
    if not 1<=epoch<=40: raise ValueError('Epoch outside fixed horizon')
    return 1. if epoch<=WARMUP else .1+.9*(1+math.cos(math.pi*(epoch-6)/34))/2


def snapshot(model):
    return {n:v.detach().cpu().clone() for n,v in model.state_dict().items()}


def state_hash(state):
    h = hashlib.sha256()
    for n,v in sorted(state.items()):
        a = v.detach().cpu().contiguous().numpy()
        h.update(n.encode());h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()


def audit(model,initial,mode,epochs,grad_max):
    params = dict(model.named_parameters())
    current = snapshot(model)
    deltas = {g:0. for g in ('head','stage6','stage7','frozen')}
    for n,p in params.items():
        delta = float((p.detach().cpu()-initial[n]).abs().max())
        deltas[group(n)] = max(deltas[group(n)],delta)
    buffers_unchanged = all(model_state.equal(initial[n]) for n,model_state in current.items() if n not in params)
    if not buffers_unchanged or deltas['frozen']!=0:
        raise ValueError('Frozen prefix/unused weights or BN buffers changed')
    active = mode=='partial' and epochs>WARMUP
    for g in ('stage6','stage7'):
        if active and not (grad_max[g]>0 and deltas[g]>0):
            raise ValueError(f'{g}: expected actual gradients AND weight updates')
        if not active and (grad_max[g]!=0 or deltas[g]!=0):
            raise ValueError('Encoder updated in frozen control/warmup')
    if epochs>0 and not (grad_max['head']>0 and deltas['head']>0):
        raise ValueError('Head did not receive gradients and updates')
    return {'max_abs_parameter_change':deltas,'max_gradient_l2':dict(grad_max),
        'bn_and_other_buffers_unchanged':buffers_unchanged,'frozen_parameters_unchanged':True,
        'encoder_updated':active,'initial_encoder_sha256':state_hash({n:v for n,v in initial.items() if n.startswith('encoder.')}),
        'final_encoder_sha256':state_hash({n:v for n,v in current.items() if n.startswith('encoder.')}),
        'trainable_parameter_count':sum(p.numel() for p in model.parameters() if p.requires_grad)}


def batch_images(torch,pixels,indices,device):
    x = torch.as_tensor(pixels[np.asarray(indices)].copy(),device=device,dtype=torch.float32).unsqueeze(1)/255.
    x = x.expand(-1,3,-1,-1)
    return (x-torch.tensor(base.EXPECTED_MEAN,device=device)[None,:,None,None])/torch.tensor(base.EXPECTED_STD,device=device)[None,:,None,None]


def predict(torch,model,pixels,indices,device,mean,scale,batch=BATCH):
    model.eval()
    output = []
    with torch.inference_mode():
        for start in range(0,len(indices),batch):
            p = model(batch_images(torch,pixels,indices[start:start+batch],device))
            output.append(p.cpu().numpy().astype(np.float64)*scale+mean)
    value = np.concatenate(output)
    if not np.isfinite(value).all(): raise ValueError('Nonfinite predictions')
    return value


def optimize(torch,model,pixels,indices,target,device,optimizer,rng,grad_max,batch=BATCH,accum=ACCUM):
    order = rng.permutation(len(indices));steps=0;total=0.
    for start in range(0,len(order),batch*accum):
        chunk = order[start:start+batch*accum]
        optimizer.zero_grad(set_to_none=True)
        for offset in range(0,len(chunk),batch):
            local = chunk[offset:offset+batch]
            pred = model(batch_images(torch,pixels,indices[local],device))
            y = torch.as_tensor(target[local],device=device,dtype=torch.float32)
            loss = (pred-y).square().sum()/len(chunk)
            if not torch.isfinite(loss): raise ValueError('Nonfinite loss')
            loss.backward();total += float(loss.detach())*len(chunk)
        squared = {g:0. for g in grad_max}
        for name,p in model.named_parameters():
            if p.grad is None: continue
            g=group(name)
            if g=='frozen' or not p.requires_grad: raise ValueError('Gradient in frozen parameter')
            if not torch.isfinite(p.grad).all(): raise ValueError('Nonfinite gradient')
            squared[g] += float(p.grad.detach().square().sum())
        for g,norm2 in squared.items():grad_max[g]=max(grad_max[g],math.sqrt(norm2))
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        optimizer.step();steps+=1
    return total/len(indices),steps


def atomic_save(torch,path,payload):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp');torch.save(payload,tmp);os.replace(tmp,path)


class Runner:
    """Only inner queries may have labels; outer queries never reach optimizer/selection."""
    def __init__(self,torch,device,pixels):
        self.torch,self.device,self.pixels=torch,device,pixels

    def __call__(self,train_idx,train_y,query_idx,query_y,mode,epochs,seed,folder,binding):
        torch,device=self.torch,self.device
        if mode not in MODES or epochs not in EPOCHS:raise ValueError('Invalid fixed mode/epoch')
        if np.intersect1d(train_idx,query_idx).size:raise ValueError('Train/query overlap')
        if len(train_y)!=len(train_idx) or not np.isfinite(train_y).all():raise ValueError('Invalid fitting targets')
        if query_y is not None and (len(query_y)!=len(query_idx) or not np.isfinite(query_y).all()):raise ValueError('Invalid internal labels')
        torch.manual_seed(seed)
        if device.type=='cuda':torch.cuda.manual_seed_all(seed);torch.cuda.reset_peak_memory_stats(device)
        rng=np.random.default_rng(seed)
        model=make_model(torch).to(device)
        initial=snapshot(model)
        mean,scale=float(np.mean(train_y)),max(float(np.std(train_y)),1e-8)
        target=(train_y-mean)/scale
        records=[];start_epoch=1;optimizer=None
        grad_max={g:0. for g in ('head','stage6','stage7')}
        progress=folder/'progress.json'
        try:
            if progress.exists():
                saved=t.read_json(progress)
                if saved['binding']!=binding or saved['slot'] not in ('resume_0.pt','resume_1.pt'):
                    raise ValueError('Resume binding changed')
                path=folder/saved['slot']
                if base.sha256_file(path)!=saved['sha256']:raise ValueError('Resume checkpoint changed')
                state=torch.load(path,map_location='cpu',weights_only=True)
                if state['binding']!=binding:raise ValueError('Resume payload binding changed')
                model.load_state_dict(state['model'],strict=True)
                optimizer=optimizer_for(torch,model,mode,max(1,state['epoch']))
                optimizer.load_state_dict(state['optimizer'])
                rng.bit_generator.state=state['numpy_rng']
                torch.set_rng_state(state['torch_rng'])
                if device.type=='cuda':torch.cuda.set_rng_state(state['cuda_rng'],device)
                records,grad_max,start_epoch=state['records'],state['grad_max'],state['epoch']+1
                print(f'Resumed {folder.name}: epoch{start_epoch-1}',flush=True)
                del state

            def score(epoch):
                fitted=predict(torch,model,self.pixels,train_idx,device,mean,scale)
                pred=predict(torch,model,self.pixels,query_idx,device,mean,scale)
                row={'epoch':epoch,'predictions':pred.tolist(),'train_sse':float(np.sum((fitted-train_y)**2)),'train_n':len(train_y)}
                if query_y is not None:row.update(valid_sse=float(np.sum((pred-query_y)**2)),valid_n=len(query_y))
                row['update_audit']=audit(model,initial,mode,epoch,grad_max)
                records.append(row)
                valid=f'; inner RMSE={math.sqrt(row["valid_sse"]/row["valid_n"]):.5f}' if query_y is not None else '; outer labels not used'
                print(f'{folder.name} {mode} epoch={epoch}: train RMSE={math.sqrt(row["train_sse"]/len(train_y)):.5f}{valid}; encoder_updated={row["update_audit"]["encoder_updated"]}',flush=True)

            if start_epoch==1:phase(model,mode,0);score(0)
            for epoch in range(start_epoch,epochs+1):
                phase(model,mode,epoch)
                if optimizer is None or epoch==WARMUP+1:optimizer=optimizer_for(torch,model,mode,epoch)
                for g in optimizer.param_groups:g['lr']=g['base_lr']*lr_factor(epoch)
                started=time.perf_counter()
                loss,steps=optimize(torch,model,self.pixels,train_idx,target,device,optimizer,rng,grad_max)
                if epoch in EPOCHS:
                    score(epoch);records[-1].update(last_epoch_loss=loss,last_epoch_optimizer_steps=steps,last_epoch_seconds=time.perf_counter()-started)
                    slot=f'resume_{(epoch//5)%2}.pt';path=folder/slot
                    atomic_save(torch,path,{'binding':binding,'epoch':epoch,'model':model.state_dict(),
                        'optimizer':optimizer.state_dict(),'numpy_rng':rng.bit_generator.state,'torch_rng':torch.get_rng_state(),
                        'cuda_rng':torch.cuda.get_rng_state(device) if device.type=='cuda' else None,'records':records,'grad_max':grad_max})
                    t.write_json(progress,{'binding':binding,'slot':slot,'sha256':base.sha256_file(path)})
            if [r['epoch'] for r in records]!=[ep for ep in EPOCHS if ep<=epochs]:raise ValueError('Incomplete checkpoint curve')
            final=audit(model,initial,mode,epochs,grad_max)
            checkpoint=None
            if query_y is None:
                path=folder/'model.pt'
                atomic_save(torch,path,{'binding':binding,'model':model.state_dict(),'target_mean':mean,'target_std':scale,'audit':final})
                checkpoint={'name':'model.pt','sha256':base.sha256_file(path)}
            return {'records':records,'target_mean':mean,'target_std':scale,'audit':final,'checkpoint':checkpoint,
                'peak_gpu_allocated_bytes':int(torch.cuda.max_memory_allocated(device)) if device.type=='cuda' else None}
        finally:
            del model
            gc.collect()
            if device.type=='cuda':torch.cuda.empty_cache()
