"""Frozen dense-feature readouts. No query labels enter fitting or scaling."""
from __future__ import annotations
import math
import os
import time
from pathlib import Path
import numpy as np
import torch

REPS = ('global_pool', 'spatial_attention')
EPOCHS = tuple(range(0, 41, 5))
HIDDEN, BATCH, LR, WD = 32, 16, 3e-4, .01


class Readout(torch.nn.Module):
    def __init__(self, kind, channels=1152, summary_dim=2304, hidden=HIDDEN):
        super().__init__()
        if kind not in REPS: raise ValueError('Unknown readout')
        self.kind = kind
        # Construct shared modules first: identical seeds yield identical shared weights.
        self.project = torch.nn.Linear(channels, hidden)
        self.summary_project = torch.nn.Linear(summary_dim, hidden)
        self.head = torch.nn.Sequential(torch.nn.Linear(3*hidden, hidden), torch.nn.GELU(),
                                        torch.nn.Dropout(.1), torch.nn.Linear(hidden, 1))
        torch.nn.init.zeros_(self.head[-1].weight)
        torch.nn.init.zeros_(self.head[-1].bias)
        if kind == 'spatial_attention':
            self.depthwise = torch.nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden)
            self.pointwise = torch.nn.Conv2d(hidden, hidden, 1)
            torch.nn.init.zeros_(self.pointwise.weight)
            torch.nn.init.zeros_(self.pointwise.bias)
            self.v = torch.nn.Linear(hidden, 16)
            self.u = torch.nn.Linear(hidden, 16)
            self.attention = torch.nn.Linear(16, 1, bias=False)
            torch.nn.init.zeros_(self.attention.weight)

    def forward(self, tokens, summary):
        n = math.isqrt(tokens.shape[1])
        if n*n != tokens.shape[1]: raise ValueError('Square token grid required')
        h = torch.nn.functional.gelu(self.project(tokens))
        if self.kind == 'spatial_attention':
            grid = h.transpose(1, 2).reshape(len(h), h.shape[-1], n, n)
            local = self.pointwise(torch.nn.functional.gelu(self.depthwise(grid)))
            h = h + local.flatten(2).transpose(1, 2)
            score = self.attention(torch.tanh(self.v(h))*torch.sigmoid(self.u(h)))
            w = torch.softmax(score, dim=1)
            mean = (w*h).sum(1)
            var = (w*(h-mean[:, None]).square()).sum(1)
        else:
            mean = h.mean(1)
            var = (h-mean[:, None]).square().mean(1)
        std = torch.sqrt(var.clamp_min(0)+1e-6)
        s = torch.nn.functional.gelu(self.summary_project(summary))
        return self.head(torch.cat((mean, std, s), dim=1)).squeeze(-1)


class TokenStore:
    """Only frozen input features, no targets. Copy in chunks to bound host RAM."""
    def __init__(self, tokens, summary, means, second, device):
        if tokens.ndim != 3 or len(summary) != len(tokens): raise ValueError('Bad feature schema')
        self.device = torch.device(device)
        self.summary_np = np.asarray(summary, dtype=np.float64)
        self.means, self.second = np.asarray(means, np.float64), np.asarray(second, np.float64)
        if self.means.shape != self.second.shape or self.means.shape != (len(tokens),tokens.shape[-1]):
            raise ValueError('Bad per-image moments')
        if not all(np.isfinite(a).all() for a in (self.summary_np,self.means,self.second)):
            raise ValueError('Nonfinite auxiliary features')
        if isinstance(tokens, torch.Tensor):
            self.tokens = tokens.to(device=self.device, dtype=torch.float32)
        else:
            self.tokens = torch.empty(tokens.shape, dtype=torch.float32, device=self.device)
            for start in range(0,len(tokens),BATCH):
                a = np.array(tokens[start:start+BATCH], dtype=np.float32, copy=True)
                if not np.isfinite(a).all(): raise ValueError('Nonfinite tokens')
                self.tokens[start:start+len(a)].copy_(torch.from_numpy(a).to(self.device))
        self.summary = torch.as_tensor(self.summary_np.astype(np.float32), device=self.device)

    def scaling(self, indices, y):
        idx = np.asarray(indices, dtype=np.int64); y = np.asarray(y, dtype=np.float64)
        if len(idx) != len(y) or len(set(idx.tolist())) != len(idx) or not np.isfinite(y).all():
            raise ValueError('Invalid training scope')
        mean = self.means[idx].mean(0)
        var = np.maximum(0,self.second[idx].mean(0)-mean**2)
        sm = self.summary_np[idx].mean(0); ss = self.summary_np[idx].std(0)
        scale = np.sqrt(var); scale[scale<1e-8]=1.; ss[ss<1e-8]=1.
        return {'token_mean':mean,'token_scale':scale,'summary_mean':sm,'summary_scale':ss,
                'y_mean':float(y.mean()),'y_scale':max(float(y.std()),1e-8)}

    def tensors(self, scaling):
        return {k:torch.as_tensor(v,dtype=torch.float32,device=self.device) for k,v in scaling.items()}

    def batch(self, indices, scaling):
        idx=torch.as_tensor(indices,dtype=torch.long,device=self.device)
        x=(self.tokens[idx]-scaling['token_mean'])/scaling['token_scale']
        s=(self.summary[idx]-scaling['summary_mean'])/scaling['summary_scale']
        return x,s


def predict(model, store, indices, scaling):
    model.eval(); parts=[]
    with torch.inference_mode():
        for start in range(0,len(indices),BATCH):
            p=model(*store.batch(indices[start:start+BATCH],scaling))
            parts.append((p*scaling['y_scale']+scaling['y_mean']).cpu().numpy().astype(np.float64))
    out=np.concatenate(parts)
    if not np.isfinite(out).all(): raise ValueError('Nonfinite prediction')
    return out


def trajectory(store, train_indices, train_y, query_indices, *, kind, seed, epochs,
               checkpoint=None, log=''):
    """No query labels or global target vector accepted. Refit uses same LR prefix."""
    ti,qi=np.asarray(train_indices,dtype=np.int64),np.asarray(query_indices,dtype=np.int64)
    if np.intersect1d(ti,qi).size: raise ValueError('Train/query overlap')
    epochs=tuple(epochs)
    if not epochs or tuple(sorted(set(epochs)))!=epochs or any(v not in EPOCHS for v in epochs):
        raise ValueError('Unexpected checkpoints')
    y=np.asarray(train_y,np.float64); scale=store.scaling(ti,y);sc=store.tensors(scale)
    torch.manual_seed(seed)
    if store.device.type=='cuda': torch.cuda.manual_seed_all(seed)
    model=Readout(kind,store.tokens.shape[-1],store.summary.shape[-1]).to(store.device)
    initial={k:v.detach().clone() for k,v in model.named_parameters()}
    # Reset training RNG after kind-specific module creation to pair dropout/shuffles.
    torch.manual_seed(seed+123456)
    if store.device.type=='cuda': torch.cuda.manual_seed_all(seed+123456)
    rng=np.random.default_rng(seed)
    opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WD)
    target=torch.as_tensor((y-scale['y_mean'])/scale['y_scale'],dtype=torch.float32,device=store.device)
    records=[];steps=0;max_grad=0.;started=time.perf_counter()
    if store.device.type=='cuda':torch.cuda.reset_peak_memory_stats(store.device)
    def score(epoch):
        p=predict(model,store,qi,sc);fitted=predict(model,store,ti,sc)
        sse=float(np.sum((fitted-y)**2))
        records.append({'epoch':epoch,'predictions':p.tolist(),'train_sse':sse,'train_n':len(ti)})
        if log:print(f'{log}: epoch={epoch}, train RMSE={math.sqrt(sse/len(ti)):.5f}; query predictions stored',flush=True)
    if 0 in epochs:score(0)
    for epoch in range(1,max(epochs)+1):
        model.train()
        factor=.1+.9*(1+math.cos(math.pi*(epoch-1)/40))/2
        for group in opt.param_groups:group['lr']=LR*factor
        order=rng.permutation(len(ti))
        for start in range(0,len(ti),BATCH):
            pos=order[start:start+BATCH]
            opt.zero_grad(set_to_none=True)
            pred=model(*store.batch(ti[pos],sc))
            loss=(pred-target[torch.as_tensor(pos,device=store.device)]).square().mean()
            if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
            loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            max_grad=max(max_grad,float(norm));opt.step();steps+=1
        if epoch in epochs:score(epoch)
    delta=float(sum((p.detach()-initial[n]).square().sum().item() for n,p in model.named_parameters()))**.5
    if max(epochs)>0 and (delta<=0 or max_grad<=0):raise ValueError('Head did not update')
    checkpoint_path=None
    if checkpoint is not None:
        checkpoint=Path(checkpoint);checkpoint.parent.mkdir(parents=True,exist_ok=True)
        temp=checkpoint.with_name(checkpoint.name+'.tmp')
        torch.save({'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},
                    'scaling':{k:torch.as_tensor(v) for k,v in scale.items()},'kind':kind,'seed':seed,
                    'epoch':max(epochs),'encoder_updates':0},temp)
        os.replace(temp,checkpoint);checkpoint_path=str(checkpoint.resolve())
    result={'records':records,'audit':{'train_indices':ti.tolist(),'query_indices':qi.tolist(),
        'train_n':len(ti),'query_n':len(qi),'epochs':list(epochs),'optimizer_steps':steps,
        'parameter_delta_l2':delta,'max_gradient_norm':max_grad,'parameter_count':sum(p.numel() for p in model.parameters()),
        'encoder_updates':0,'query_labels_accepted':False,'seconds':time.perf_counter()-started,
        'peak_allocated_bytes':torch.cuda.max_memory_allocated(store.device) if store.device.type=='cuda' else None},
        'scaling':{k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in scale.items()},
        'checkpoint_path':checkpoint_path}
    return result


def select_epoch(runs, splits, y, inner_other):
    """inner_other=.5S_inner+.25L_inner, each fit without the same inner query."""
    y=np.asarray(y,np.float64);other=np.asarray(inner_other,np.float64)
    if len(runs)!=3 or other.shape!=y.shape or not np.isfinite(other).all():raise ValueError('Bad inner baseline')
    curves=[]
    for epoch in EPOCHS:
        p=np.full(len(y),np.nan);covered=np.zeros(len(y),int);train_sse=0.;train_n=0
        for run,(_,vi) in zip(runs,splits):
            row=next(r for r in run['records'] if r['epoch']==epoch)
            v=np.asarray(row['predictions'],float)
            if v.shape!=(len(vi),) or not np.isfinite(v).all():raise ValueError('Bad inner predictions')
            p[vi]=v;covered[vi]+=1;train_sse+=row['train_sse'];train_n+=row['train_n']
        if not np.all(covered==1):raise ValueError('Inner coverage differs')
        curves.append({'epoch':epoch,'blend_mse':float(np.mean((other+.25*p-y)**2)),
                       'standalone_rmse':float(np.sqrt(np.mean((p-y)**2))),
                       'train_rmse':math.sqrt(train_sse/train_n)})
    best=min(curves,key=lambda r:(r['blend_mse'],r['epoch']))
    return best,curves


def compose(scattering, large, ridge, global_pool, spatial):
    arrays=[np.asarray(a,dtype=np.float64) for a in (scattering,large,ridge,global_pool,spatial)]
    if any(a.shape!=arrays[0].shape or a.ndim!=1 or not np.isfinite(a).all() for a in arrays):
        raise ValueError('Bad predictions')
    s,l,r,g,c=arrays;other=.5*s+.25*l
    return {'current':other+.25*r,'radio_ridge':r,'global_pool':g,'spatial_attention':c,
            'global_pool_blend':other+.25*g,'spatial_replaced_blend':other+.25*c}
