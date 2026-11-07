"""Small synthetic GPU parity check; no competition data or weights used."""
import contextlib
import copy
import json
import sys
from pathlib import Path
import numpy as np
import torch
import mitra_v2_assets as assets
import mitra_v2_raw486_runtime as memory


def main():
    sys.path.insert(0,str(assets.BASE/'vendor/src'))
    from autogluon.tabular.models.mitra._internal.models.tab2d import Tab2D
    from mitra_finetune import speed
    torch.set_num_threads(4);torch.manual_seed(7381);torch.cuda.manual_seed_all(7381)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=Tab2D(dim=32,dim_output=17,n_layers=2,n_heads=4,task='REGRESSION',
                use_pretrained_weights=False,path_to_weights='',device='cuda').cuda()
    initial=copy.deepcopy(model.state_dict())
    args=(torch.randn(2,35,40,device='cuda'),torch.randint(0,17,(2,35),device='cuda'),
          torch.randn(2,33,40,device='cuda'),torch.zeros(2,40,dtype=torch.bool,device='cuda'),
          torch.zeros(2,35,dtype=torch.bool,device='cuda'),torch.zeros(2,33,dtype=torch.bool,device='cuda'))

    def run(bf16,grad):
        model.load_state_dict(initial);model.zero_grad(set_to_none=True)
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-5,weight_decay=.3)
        context=torch.autocast('cuda',dtype=torch.bfloat16) if bf16 else contextlib.nullcontext()
        with torch.set_grad_enabled(grad),context:
            pred=model(*(a.clone() for a in args))
            loss=pred.float().square().mean()
        if grad:loss.backward()
        grads={k:v.grad.detach().float().cpu().numpy().copy() for k,v in model.named_parameters() if v.grad is not None}
        if grad:optimizer.step()
        return pred.detach().float().cpu().numpy(),grads,{k:v.detach().float().cpu().numpy().copy() for k,v in model.named_parameters()}

    cases=[(False,True),(True,True),(True,False)]
    baseline={case:run(*case) for case in cases}
    memory.install(); results=[]
    for case in cases:
        p,g,w=run(*case);bp,bg,bw=baseline[case]
        atol,rtol=(2e-2,2e-2) if case[0] else (2e-6,2e-5)
        np.testing.assert_allclose(p,bp,rtol=rtol,atol=atol)
        assert set(g)==set(bg)
        for k in g:np.testing.assert_allclose(g[k],bg[k],rtol=rtol,atol=atol)
        pred_error=float(np.linalg.norm(p-bp)/max(np.linalg.norm(bp),1e-12))
        numerator=sum(float(np.sum((g[k]-bg[k])**2)) for k in g)
        denominator=sum(float(np.sum(bg[k]**2)) for k in g)
        grad_error=(numerator/max(denominator,1e-30))**.5
        assert pred_error < (0.01 if case[0] else 1e-5)
        assert grad_error < (0.02 if case[0] else 1e-5)
        max_update_diff=max(float(np.max(np.abs(w[k]-bw[k]))) for k in w)
        assert max_update_diff<=2.1e-5
        results.append({'bf16':case[0],'gradient':case[1],'output_relative_l2':pred_error,
                        'gradient_relative_l2':grad_error,'adamw_parameter_max_difference':max_update_diff})
    checkpoint=speed.DeviceCheckpoint();checkpoint.reset(model)
    checkpoint(model,1.0)
    assert all(v.device.type=='cpu' for v in checkpoint.best_model.values())
    best={k:v.clone() for k,v in model.state_dict().items()}
    with torch.no_grad():
        for v in model.parameters():v.add_(.1)
    checkpoint.set_to_best(model)
    for k,v in model.state_dict().items():torch.testing.assert_close(v,best[k],rtol=0,atol=0)
    result={'passed':True,'competition_fits':0,'runtime_sha256':assets.sha(memory.__file__),
            'test_sha256':assets.sha(__file__),'policy':memory.POLICY,'cases':results,
            'scope':'Small synthetic Tab2D, full sequence attention retained; FP32/BF16 outputs, all gradients and one AdamW update; CPU checkpoint exact restore.'}
    path=assets.ROOT/'reports/mitra_v2_raw486_runtime/math_check.json';path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result))


if __name__=='__main__':main()
