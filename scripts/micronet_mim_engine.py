"""Split-local masked image reconstruction. No hardness labels are accepted here.

SimMIM-inspired objective, independently implemented for the pinned MicroNet CNN.
This is not a reproduction of the published ViT/Swin recipe.
"""
from __future__ import annotations
import gc
import math
import os
import time
from pathlib import Path
import numpy as np

if __package__:
    from . import micronet_finetune_engine as util
else:
    import micronet_finetune_engine as util
old,t,base=util.old,util.t,util.base
EPOCHS=(0,20,40,80)
BATCH,ACCUM=8,2
MASK_SIDE,MASK_COUNT=32,39  # ceil(0.6*64), fixed before evaluation.
GROUPS=('early_encoder','late_encoder','decoder')
snapshot,state_hash,atomic_save=util.snapshot,util.state_hash,util.atomic_save


def group(name):
    if name.startswith('decoder.'):return 'decoder'
    if name.startswith(('encoder._conv_stem.','encoder._bn0.')):return 'early_encoder'
    if name.startswith('encoder._blocks.'):
        return 'early_encoder' if int(name.split('.')[2])<=10 else 'late_encoder'
    return 'frozen'


def make_model(torch):
    class MaskedMicroNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder=old.load_model('micronet')
            self.decoder=torch.nn.Sequential(torch.nn.Conv2d(320,32*32,1),torch.nn.PixelShuffle(32))
            torch.nn.init.normal_(self.decoder[0].weight,std=.001)
            torch.nn.init.constant_(self.decoder[0].bias,.5)

        def forward(self,x):return self.decoder(old.forward_maps(self.encoder,x)[-1])
        def features(self,x):return old.pool_maps(old.forward_maps(self.encoder,x))
    return MaskedMicroNet()


def phase(model):
    model.train();model.encoder.eval()  # Fixed BN buffers, no drop-connect.
    for name,p in model.named_parameters():p.requires_grad_(group(name)!='frozen')


def optimizer_for(torch,model):
    phase(model);groups={}
    for n,p in model.named_parameters():
        if not p.requires_grad:continue
        lr=.001 if group(n)=='decoder' else .0001
        wd=.01 if p.ndim>1 and not n.endswith('.bias') else 0.
        groups.setdefault((lr,wd),{'params':[],'lr':lr,'base_lr':lr,'weight_decay':wd})['params'].append(p)
    return torch.optim.AdamW(list(groups.values()))


def lr_factor(epoch):
    if not 1<=epoch<=80:raise ValueError('Epoch outside fixed80 schedule')
    if epoch<=5:return epoch/5
    return .1+.9*(1+math.cos(math.pi*(epoch-6)/74))/2


def masks(rng,n):
    coarse=np.zeros((n,64),dtype=np.float32)
    for row in coarse:row[rng.permutation(64)[:MASK_COUNT]]=1.
    return np.repeat(np.repeat(coarse.reshape(n,1,8,8),MASK_SIDE,axis=2),MASK_SIDE,axis=3)


def raw_batch(torch,pixels,indices,device):
    return torch.as_tensor(pixels[np.asarray(indices)].copy(),dtype=torch.float32,device=device).unsqueeze(1)/255.


def normalized(torch,raw):
    mean=torch.tensor(base.EXPECTED_MEAN,device=raw.device)[None,:,None,None]
    std=torch.tensor(base.EXPECTED_STD,device=raw.device)[None,:,None,None]
    return (raw.expand(-1,3,-1,-1)-mean)/std


def masked_input(torch,raw,mask):
    # Original masked pixels cannot reach the encoder. No target-dependent fill.
    return normalized(torch,raw*(1-mask)+.5*mask)


def masked_l1(torch,pred,raw,mask):
    if pred.shape!=raw.shape or mask.shape!=raw.shape:raise ValueError('Reconstruction shape mismatch')
    count=mask.sum((1,2,3))
    if not bool((count>0).all()):raise ValueError('Empty reconstruction mask')
    return ((pred-raw).abs()*mask).sum((1,2,3))/count


def optimize(torch,model,train_pixels,device,optimizer,rng,grad_max,batch=BATCH,accum=ACCUM):
    """Only fitting images are passed; no query pixels, targets or baseline OOF."""
    phase(model);order=rng.permutation(len(train_pixels));total=0.;steps=0
    for start in range(0,len(order),batch*accum):
        chunk=order[start:start+batch*accum];optimizer.zero_grad(set_to_none=True)
        # Draw per-image masks before microbatch splitting: accumulation invariant.
        mask_array=masks(rng,len(chunk))
        for offset in range(0,len(chunk),batch):
            local=chunk[offset:offset+batch]
            raw=raw_batch(torch,train_pixels,local,device)
            mask=torch.as_tensor(mask_array[offset:offset+batch],device=device)
            losses=masked_l1(torch,model(masked_input(torch,raw,mask)),raw,mask)
            loss=losses.sum()/len(chunk)
            if not torch.isfinite(loss):raise ValueError('Nonfinite masked reconstruction loss')
            loss.backward();total+=float(losses.detach().sum())
        sums={g:0. for g in GROUPS}
        for name,p in model.named_parameters():
            if p.grad is None:continue
            g=group(name)
            if g=='frozen' or not p.requires_grad:raise ValueError('Gradient in frozen parameter')
            if not torch.isfinite(p.grad).all():raise ValueError('Nonfinite gradient')
            sums[g]+=float(p.grad.detach().square().sum())
        for g,v in sums.items():grad_max[g]=max(grad_max[g],math.sqrt(v))
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        optimizer.step();steps+=1
    return total/len(train_pixels),steps


def audit(model,initial,epoch,grad_max):
    params=dict(model.named_parameters());current=snapshot(model)
    delta={g:0. for g in (*GROUPS,'frozen')}
    for n,p in params.items():
        if not p.isfinite().all():raise ValueError('Nonfinite model parameter')
        delta[group(n)]=max(delta[group(n)],float((p.detach().cpu()-initial[n]).abs().max()))
    buffers=all(v.equal(initial[n]) for n,v in current.items() if n not in params)
    if not buffers or delta['frozen']!=0:raise ValueError('BN buffer or unused pretrained weights changed')
    for g in GROUPS:
        if epoch>0 and not (delta[g]>0 and grad_max[g]>0):raise ValueError('Missing update evidence: '+g)
        if epoch==0 and (delta[g]!=0 or grad_max[g]!=0):raise ValueError('Epoch0 changed')
    before=state_hash({n:v for n,v in initial.items() if n.startswith('encoder.')})
    after=state_hash({n:v for n,v in current.items() if n.startswith('encoder.')})
    if (before!=after)!=(epoch>0):raise ValueError('Encoder update hash mismatch')
    return {'encoder_updated':epoch>0,'bn_buffers_unchanged':buffers,'unused_parameters_unchanged':True,
        'max_abs_parameter_change':delta,'max_gradient_l2':dict(grad_max),
        'initial_encoder_sha256':before,'final_encoder_sha256':after,
        'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad)}


def features(torch,model,pixels,device,batch=BATCH):
    model.eval();parts=[]
    with torch.inference_mode():
        for start in range(0,len(pixels),batch):
            x=normalized(torch,raw_batch(torch,pixels,np.arange(start,min(start+batch,len(pixels))),device))
            parts.append(model.features(x).cpu().numpy())
    output=np.concatenate(parts)
    if output.shape!=(len(pixels),864) or output.dtype!=np.float32 or not np.isfinite(output).all():
        raise ValueError('Invalid clean pooled864 features')
    return output


def save_npz(path,**arrays):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.tmp')
    with tmp.open('wb') as f:np.savez_compressed(f,**arrays)
    os.replace(tmp,path)


class Runner:
    """Independent SSL model for every inner/outer fitting subset; labels absent."""
    def __init__(self,torch,device,pixels,frozen_features=None):
        self.torch,self.device,self.pixels=torch,device,pixels
        self.frozen_features=frozen_features

    def __call__(self,train_idx,query_idx,epochs,seed,folder,binding):
        torch,device=self.torch,self.device
        if epochs not in EPOCHS or np.intersect1d(train_idx,query_idx).size:raise ValueError('Invalid SSL partition/epoch')
        if not len(train_idx) or not len(query_idx):raise ValueError('Empty partition')
        train_pixels=self.pixels[train_idx].copy();query_pixels=self.pixels[query_idx].copy()
        torch.manual_seed(seed)
        if device.type=='cuda':torch.cuda.manual_seed_all(seed);torch.cuda.reset_peak_memory_stats(device)
        rng=np.random.default_rng(seed);model=make_model(torch).to(device);phase(model)
        initial=snapshot(model);optimizer=optimizer_for(torch,model)
        records=[];artifacts={};grad_max={g:0. for g in GROUPS};start_epoch=1
        folder.mkdir(parents=True,exist_ok=True);progress=folder/'progress.json'
        try:
            if progress.exists():
                saved=t.read_json(progress)
                if saved['binding']!=binding or saved['slot'] not in ('resume_0.pt','resume_1.pt'):raise ValueError('Resume binding changed')
                path=folder/saved['slot']
                if base.sha256_file(path)!=saved['sha256']:raise ValueError('Resume checkpoint changed')
                state=torch.load(path,map_location='cpu',weights_only=True)
                if state['binding']!=binding:raise ValueError('Resume payload binding changed')
                for n,h in state['artifacts'].items():
                    if base.sha256_file(folder/n)!=h:raise ValueError('Resumed feature artifact changed')
                model.load_state_dict(state['model'],strict=True);optimizer.load_state_dict(state['optimizer'])
                rng.bit_generator.state=state['numpy_rng'];torch.set_rng_state(state['torch_rng'])
                if device.type=='cuda':torch.cuda.set_rng_state(state['cuda_rng'],device)
                records,artifacts,grad_max=state['records'],state['artifacts'],state['grad_max']
                start_epoch=state['epoch']+1;del state
                print(f'Resumed SSL {folder.name}: epoch{start_epoch-1}',flush=True)

            def extract(epoch):
                # Query inference is eval-only and restores RNG; it cannot alter
                # model/optimizer or the subsequent SSL trajectory.
                before=state_hash(snapshot(model))
                with torch.random.fork_rng(devices=[device.index] if device.type=='cuda' else []):
                    tx=features(torch,model,train_pixels,device)
                    qx=features(torch,model,query_pixels,device)
                if before!=state_hash(snapshot(model)):raise ValueError('Feature extraction changed model state')
                if epoch==0 and self.frozen_features is not None:
                    # Fail before any SSL update if input/model identity differs.
                    np.testing.assert_allclose(tx,self.frozen_features[train_idx],rtol=2e-5,atol=3e-4)
                    np.testing.assert_allclose(qx,self.frozen_features[query_idx],rtol=2e-5,atol=3e-4)
                path=folder/f'features_epoch{epoch}.npz'
                save_npz(path,train=tx,query=qx)
                artifacts[path.name]=base.sha256_file(path)
                return {'epoch':epoch,'feature_file':path.name,'update_audit':audit(model,initial,epoch,grad_max),
                    'feature_std_mean':float(tx.std(axis=0).mean()),'feature_std_min':float(tx.std(axis=0).min())}

            if start_epoch==1:records.append(extract(0))
            for epoch in range(start_epoch,epochs+1):
                for group_ in optimizer.param_groups:group_['lr']=group_['base_lr']*lr_factor(epoch)
                started=time.perf_counter()
                loss,steps=optimize(torch,model,train_pixels,device,optimizer,rng,grad_max)
                if epoch%5==0:
                    row=extract(epoch) if epoch in EPOCHS else {'epoch':epoch}
                    row.update(train_masked_l1=loss,optimizer_steps=steps,seconds=time.perf_counter()-started)
                    records.append(row)
                    print(f'{folder.name} SSL epoch={epoch}: train masked L1={loss:.6f}; fitting images={len(train_idx)}; no hardness labels',flush=True)
                    slot=f'resume_{(epoch//5)%2}.pt';path=folder/slot
                    atomic_save(torch,path,{'binding':binding,'epoch':epoch,'model':model.state_dict(),'optimizer':optimizer.state_dict(),
                        'numpy_rng':rng.bit_generator.state,'torch_rng':torch.get_rng_state(),
                        'cuda_rng':torch.cuda.get_rng_state(device) if device.type=='cuda' else None,
                        'records':records,'artifacts':artifacts,'grad_max':grad_max})
                    t.write_json(progress,{'binding':binding,'slot':slot,'sha256':base.sha256_file(path)})
            final=audit(model,initial,epochs,grad_max)
            if binding['role']=='outer':
                path=folder/'model.pt'
                atomic_save(torch,path,{'binding':binding,'model':model.state_dict(),'audit':final})
                artifacts[path.name]=base.sha256_file(path)
            return {'records':records,'artifacts':artifacts,'audit':final,'fitting_indices':train_idx.tolist(),
                'query_indices':query_idx.tolist(),'hardness_labels_used':False,'query_images_used_for_ssl':False,
                'peak_gpu_bytes':int(torch.cuda.max_memory_allocated(device)) if device.type=='cuda' else None}
        finally:
            del model,optimizer;gc.collect()
            if device.type=='cuda':torch.cuda.empty_cache()
