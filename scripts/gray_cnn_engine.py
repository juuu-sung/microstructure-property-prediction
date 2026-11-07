"""Small scratch grayscale CNN: stride1 vs matched stride2 control."""
from __future__ import annotations
import gc
import hashlib
import math
import os
import time
import numpy as np

if __package__:
    from . import dinov2_factorial_experiment as shared
else:
    import dinov2_factorial_experiment as shared

MODES=('early_stride2','native_stride1')
EPOCHS=tuple(range(0,81,5))
BATCH,ACCUM=8,2
GROUPS=('stem','stage1','stage2','stage3','stage4','head')
WIDTHS=(16,32,64,96)


def make_model(torch,mode):
    if mode not in MODES:raise ValueError('Unknown fixed architecture')
    nn=torch.nn
    class Block(nn.Module):
        def __init__(self,c):
            super().__init__()
            self.body=nn.Sequential(nn.Conv2d(c,c,3,padding=1,bias=False),nn.GroupNorm(4,c),nn.SiLU(),
                                    nn.Conv2d(c,c,3,padding=1,bias=False),nn.GroupNorm(4,c))
        def forward(self,x):return torch.nn.functional.silu(x+self.body(x))
    def down(cin,cout):
        return nn.Sequential(nn.AvgPool2d(2),nn.Conv2d(cin,cout,3,padding=1,bias=False),
                             nn.GroupNorm(4,cout),nn.SiLU(),Block(cout))
    class GrayCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem=nn.Sequential(nn.Conv2d(1,16,3,stride=1 if mode=='native_stride1' else 2,padding=1,bias=False),
                                    nn.GroupNorm(4,16),nn.SiLU())
            self.stage1=Block(16)
            self.stage2=down(16,32);self.stage3=down(32,64);self.stage4=down(64,96)
            self.head=nn.Sequential(nn.Linear(426,64),nn.SiLU(),nn.Dropout(.1),nn.Linear(64,1))
        def maps(self,x):
            # Only the first stride differs. All kernels/weights/channel counts match.
            a=self.stage1(self.stem((x-.5)/.25));b=self.stage2(a);c=self.stage3(b);d=self.stage4(c)
            return (a,b,c,d)
        def forward(self,x):
            pieces=[]
            for value in self.maps(x):
                pieces.extend((value.mean((2,3)),value.std((2,3),correction=0)))
            # Preserve unnormalized absolute brightness/contrast alongside normalized feature maps.
            pieces.extend((x.mean((2,3)),x.std((2,3),correction=0)))
            cells=x.reshape(x.shape[0],1,2,128,2,128).permute(0,1,2,4,3,5).reshape(x.shape[0],4,-1)
            pieces.extend((cells.mean(2),cells.std(2,correction=0)))
            return self.head(torch.cat(pieces,1)).squeeze(1)
    return GrayCNN()


def group(name):
    value=name.split('.')[0]
    if value not in GROUPS:raise ValueError('Unexpected parameter group '+name)
    return value


def snapshot(model):return {n:v.detach().cpu().clone() for n,v in model.state_dict().items()}


def state_hash(state):
    h=hashlib.sha256()
    for n,v in sorted(state.items()):
        a=v.detach().cpu().contiguous().numpy()
        h.update(n.encode());h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()


def audit(model,initial,epoch,grad_max):
    deltas={g:0. for g in GROUPS}
    for name,p in model.named_parameters():
        change=float((p.detach().cpu()-initial[name]).abs().max())
        deltas[group(name)]=max(deltas[group(name)],change)
    if epoch>0 and any(deltas[g]<=0 or grad_max[g]<=0 for g in GROUPS):raise ValueError('Missing gradient or update in a trainable stage')
    if epoch==0 and (any(deltas.values()) or any(grad_max.values())):raise ValueError('Unexpected epoch0 update')
    return {'max_abs_parameter_change':deltas,'max_gradient_l2':dict(grad_max),
            'all_stages_updated':epoch>0,'trainable_parameters':sum(p.numel() for p in model.parameters()),
            'initial_state_sha256':state_hash(initial),'final_state_sha256':state_hash(model.state_dict())}


def optimizer_for(torch,model):
    groups=[{'params':[p for n,p in model.named_parameters() if p.ndim>1], 'weight_decay':.01},
            {'params':[p for n,p in model.named_parameters() if p.ndim<=1], 'weight_decay':0.}]
    return torch.optim.AdamW(groups,lr=3e-4)


def lr_at(epoch):
    if not 1<=epoch<=80:raise ValueError('Epoch outside frozen horizon')
    if epoch<=5:return 3e-4*epoch/5
    return 3e-4*(.1+.9*(1+math.cos(math.pi*(epoch-5)/75))/2)


def batch_images(torch,pixels,indices,device):
    return torch.as_tensor(pixels[np.asarray(indices)].copy(),device=device,dtype=torch.float32).unsqueeze(1)/255.


def predict(torch,model,pixels,indices,device,mean,scale,batch=BATCH):
    model.eval();output=[]
    with torch.inference_mode():
        for i in range(0,len(indices),batch):
            pred=model(batch_images(torch,pixels,indices[i:i+batch],device))
            output.append(pred.cpu().numpy().astype(np.float64)*scale+mean)
    result=np.concatenate(output)
    if not np.isfinite(result).all():raise ValueError('Nonfinite prediction')
    return result


def optimize(torch,model,pixels,indices,target,device,optimizer,rng,grad_max,batch=BATCH,accum=ACCUM):
    model.train();order=rng.permutation(len(indices));total=0.;steps=0
    for start in range(0,len(order),batch*accum):
        chunk=order[start:start+batch*accum];optimizer.zero_grad(set_to_none=True)
        for offset in range(0,len(chunk),batch):
            local=chunk[offset:offset+batch]
            pred=model(batch_images(torch,pixels,indices[local],device))
            y=torch.as_tensor(target[local],device=device,dtype=torch.float32)
            loss=(pred-y).square().sum()/len(chunk)
            if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
            loss.backward();total+=float(loss.detach())*len(chunk)
        squared={g:0. for g in GROUPS}
        for name,p in model.named_parameters():
            if p.grad is None or not torch.isfinite(p.grad).all():raise ValueError('Missing/nonfinite gradient: '+name)
            squared[group(name)]+=float(p.grad.square().sum())
        for g,norm2 in squared.items():grad_max[g]=max(grad_max[g],math.sqrt(norm2))
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        optimizer.step();steps+=1
    return total/len(indices),steps


def atomic_save(torch,path,payload):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.tmp')
    torch.save(payload,tmp);os.replace(tmp,path)


class Runner:
    def __init__(self,torch,device,pixels):self.torch,self.device,self.pixels=torch,device,pixels

    def __call__(self,ti,yt,vi,yv,mode,epochs,seed,folder,binding):
        torch,device=self.torch,self.device
        if mode not in MODES or epochs not in EPOCHS or np.intersect1d(ti,vi).size:raise ValueError('Invalid model/epoch/train-query overlap')
        if len(yt)!=len(ti) or not np.isfinite(yt).all():raise ValueError('Invalid training labels')
        if yv is not None and (len(yv)!=len(vi) or not np.isfinite(yv).all()):raise ValueError('Invalid inner labels')
        folder.mkdir(parents=True,exist_ok=True)
        torch.manual_seed(seed)
        if device.type=='cuda':torch.cuda.manual_seed_all(seed);torch.cuda.reset_peak_memory_stats(device)
        rng=np.random.default_rng(seed);model=make_model(torch,mode).to(device);initial=snapshot(model)
        mean,scale=float(np.mean(yt)),max(float(np.std(yt)),1e-8);target=(yt-mean)/scale
        optimizer=optimizer_for(torch,model);records=[];grad_max={g:0. for g in GROUPS};start=1
        progress=folder/'progress.json';t0=time.perf_counter()
        try:
            if progress.exists():
                marker=shared.read_json(progress)
                if marker['binding']!=binding or marker['slot'] not in ('resume_0.pt','resume_1.pt'):raise ValueError('Resume scope changed')
                path=folder/marker['slot']
                if shared.base.sha256_file(path)!=marker['sha256']:raise ValueError('Resume checksum differs')
                state=torch.load(path,map_location='cpu',weights_only=True)
                if state['binding']!=binding or state['epoch'] not in EPOCHS or state['epoch']>epochs:raise ValueError('Resume payload differs')
                model.load_state_dict(state['model'],strict=True);optimizer.load_state_dict(state['optimizer'])
                rng.bit_generator.state=state['numpy_rng'];torch.set_rng_state(state['torch_rng'])
                if device.type=='cuda':torch.cuda.set_rng_state(state['cuda_rng'],device)
                records,grad_max,start=state['records'],state['grad_max'],state['epoch']+1
                print(f'Resumed {mode}/{folder.name} epoch{start-1}',flush=True);del state

            def score(epoch):
                fitted=predict(torch,model,self.pixels,ti,device,mean,scale)
                pred=predict(torch,model,self.pixels,vi,device,mean,scale)
                row={'epoch':epoch,'predictions':pred.tolist(),'train_sse':float(np.sum((fitted-yt)**2)),
                     'train_n':len(yt),'update_audit':audit(model,initial,epoch,grad_max)}
                if yv is not None:row.update(valid_sse=float(np.sum((pred-yv)**2)),valid_n=len(yv))
                records.append(row)
                suffix=f'; inner RMSE={math.sqrt(row["valid_sse"]/len(yv)):.5f}' if yv is not None else '; outer labels unused'
                print(f'{mode}/{folder.name} epoch{epoch}: train RMSE={math.sqrt(row["train_sse"]/len(yt)):.5f}{suffix}',flush=True)

            if start==1:score(0)
            for epoch in range(start,epochs+1):
                for g in optimizer.param_groups:g['lr']=lr_at(epoch)
                tick=time.perf_counter()
                loss,steps=optimize(torch,model,self.pixels,ti,target,device,optimizer,rng,grad_max)
                if epoch in EPOCHS:
                    score(epoch);records[-1].update(last_epoch_loss=loss,last_epoch_steps=steps,last_epoch_seconds=time.perf_counter()-tick)
                    slot=f'resume_{(epoch//5)%2}.pt';path=folder/slot
                    atomic_save(torch,path,{'binding':binding,'epoch':epoch,'model':model.state_dict(),'optimizer':optimizer.state_dict(),
                        'numpy_rng':rng.bit_generator.state,'torch_rng':torch.get_rng_state(),
                        'cuda_rng':torch.cuda.get_rng_state(device) if device.type=='cuda' else None,'records':records,'grad_max':grad_max})
                    shared.write_json(progress,{'binding':binding,'slot':slot,'sha256':shared.base.sha256_file(path)})
            if [r['epoch'] for r in records]!=[ep for ep in EPOCHS if ep<=epochs]:raise ValueError('Incomplete trajectory')
            final=audit(model,initial,epochs,grad_max);checkpoint=None
            if yv is None:
                path=folder/'model.pt'
                atomic_save(torch,path,{'binding':binding,'model':model.state_dict(),'mode':mode,
                    'target_mean':mean,'target_std':scale,'audit':final})
                checkpoint={'name':'model.pt','sha256':shared.base.sha256_file(path)}
            return {'records':records,'target_mean':mean,'target_std':scale,'audit':final,'checkpoint':checkpoint,
                'seconds':time.perf_counter()-t0,'peak_gpu_allocated_bytes':int(torch.cuda.max_memory_allocated(device)) if device.type=='cuda' else None}
        finally:
            del model,optimizer;gc.collect()
            if device.type=='cuda':torch.cuda.empty_cache()
