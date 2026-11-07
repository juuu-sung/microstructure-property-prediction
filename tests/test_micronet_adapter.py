"""Synthetic-only update, validation leakage, resume and artifact tests."""
from contextlib import contextmanager,redirect_stdout
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
from scripts import micronet_adapter_experiment as e
g=e.eng


@contextmanager
def temporary():
    parent=Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory(dir=parent,prefix='.micronet_adapter_test_') as name:
        if Path(name).resolve().parent!=parent:raise ValueError('Invalid cleanup path')
        yield Path(name)


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__();self.encoder=torch.nn.Module()
        self.encoder._blocks=torch.nn.ModuleList([torch.nn.Linear(4,4) for _ in range(16)])
        self.encoder.bn=torch.nn.BatchNorm1d(4)
        self.head=torch.nn.Sequential(torch.nn.Dropout(.1),torch.nn.Linear(4,1))
        self.adapters=torch.nn.ModuleDict({str(i):torch.nn.Sequential(torch.nn.Linear(4,2),torch.nn.GELU(),torch.nn.Linear(2,4)) for i in g.BLOCKS})
        for a in self.adapters.values():torch.nn.init.zeros_(a[2].weight);torch.nn.init.zeros_(a[2].bias)
        self.adapters_enabled=False
    def forward(self,x):
        x=self.encoder.bn(x)
        for i,b in enumerate(self.encoder._blocks):
            x=torch.tanh(x+.05*b(x))
            if self.adapters_enabled and i in g.BLOCKS:x=x+self.adapters[str(i)](x)
        return self.head(x).squeeze(-1)


def batch(torch,pixels,indices,device):return torch.as_tensor(pixels[indices].copy(),device=device,dtype=torch.float32)


def fixture():
    rng=np.random.default_rng(19)
    data=SimpleNamespace(ids=np.array([f'TRAIN_{i:06d}' for i in range(500)]),y=rng.normal(100,10,500),noise=rng.random(500))
    ref={'current':data.y+3,'folds':np.repeat(np.arange(5),100)}
    plan={'frozen':{'upstream':{'common_failure_ids':list(data.ids[:16])}}}
    return data,ref,plan


class FakeRunner:
    calls=[]
    def __call__(self,ti,yt,vi,yv,mode,epochs,seed,folder,binding):
        self.calls.append({'train':ti.copy(),'y':yt.copy(),'query':vi.copy(),'query_y':yv,'mode':mode,'seed':seed,'epochs':epochs})
        records=[]
        for ep in g.EPOCHS:
            if ep>epochs:continue
            pred=np.full(len(vi),float(yt.mean())+ep*.001)
            active=mode=='adapter' and ep>5
            audit={'encoder_updated':False,'adapters_updated':active,'bn_and_other_buffers_unchanged':True,'frozen_parameters_unchanged':True,
             'initial_encoder_sha256':'synthetic','final_encoder_sha256':'synthetic','trainable_parameter_count':21129 if active else 2593,
             'max_abs_parameter_change':{'head':float(ep>0),'frozen':0.,**{k:float(active) for k in g.ADAPTER_GROUPS}},
             'max_gradient_l2':{'head':float(ep>0),**{k:float(active) for k in g.ADAPTER_GROUPS}}}
            row={'epoch':ep,'predictions':pred.tolist(),'train_sse':float(len(yt)),'train_n':len(yt),'update_audit':audit}
            if yv is not None:row.update(valid_sse=float(np.sum((pred-yv)**2)),valid_n=len(yv))
            records.append(row)
        checkpoint=None
        if yv is None:
            path=folder/'model.pt';path.write_bytes(b'synthetic checkpoint')
            checkpoint={'name':'model.pt','sha256':e.base.sha256_file(path)}
        return {'records':records,'audit':records[-1]['update_audit'],'target_mean':float(yt.mean()),'target_std':float(yt.std()),'checkpoint':checkpoint}


class AdapterTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2);FakeRunner.calls=[]

    def test_actual_micronet_identity_gradients_and_frozen_bn(self):
        # Real native256 encoder, synthetic pixels only, two CPU steps per arm.
        from scripts import micronet_finetune_engine as previous
        rng=np.random.default_rng(42);pixels=rng.integers(0,256,size=(2,256,256),dtype=np.uint8)
        initial_heads=[]
        torch.manual_seed(12);original=previous.make_model(torch).eval();original_rng=torch.get_rng_state()
        x=g.batch_images(torch,pixels,np.arange(2),torch.device('cpu'))
        with torch.inference_mode():expected=original(x)
        for mode in g.MODES:
            torch.manual_seed(12);model=g.make_model(torch);initial=g.snapshot(model)
            self.assertTrue(torch.get_rng_state().equal(original_rng))
            for name,value in original.state_dict().items():
                torch.testing.assert_close(value,initial[name],rtol=0,atol=0)
            self.assertEqual(sum(p.numel() for p in model.adapters.parameters()),18536)
            model.eval();model.adapters_enabled=True
            with torch.inference_mode():torch.testing.assert_close(model(x),expected,rtol=0,atol=0)
            initial_heads.append(g.state_hash({n:v for n,v in initial.items() if n.startswith('head.')}))
            opt=g.optimizer_for(torch,model,mode,6);grad={k:0. for k in g.GRAD_GROUPS}
            self.assertEqual(sum(p.numel() for p in model.parameters() if p.requires_grad),21129 if mode=='adapter' else 2593)
            for step in range(2):
                _,steps=g.optimize(torch,model,pixels,np.arange(2),np.array([-1.,1.]),torch.device('cpu'),opt,np.random.default_rng(5+step),grad,batch=2,accum=1)
            a=g.audit(model,initial,mode,6,grad)
            self.assertEqual(steps,1);self.assertFalse(a['encoder_updated']);self.assertEqual(a['adapters_updated'],mode=='adapter')
            self.assertTrue(a['bn_and_other_buffers_unchanged'])
            self.assertEqual(a['initial_encoder_sha256'],a['final_encoder_sha256'])
            self.assertFalse(any(m.training for m in model.encoder.modules() if isinstance(m,torch.nn.modules.batchnorm._BatchNorm)))
            if mode=='adapter':
                for name,p in model.adapters.named_parameters():
                    if name.endswith('weight'):
                        self.assertGreater(p.grad.norm().item(),0,name)
                        self.assertGreater((p-initial['adapters.'+name]).abs().max().item(),0,name)
        self.assertEqual(initial_heads[0],initial_heads[1])

    def test_schedule_matches_previous80_and_synthetic_check(self):
        from scripts import micronet_finetune_80_engine as previous
        for epoch in range(1,81):self.assertEqual(g.lr_factor(epoch),previous.lr_factor(epoch))
        data,ref,plan=fixture()
        # Exercise the user-facing check on synthetic CPU images, in a disposable
        # test folder so it cannot mark the user's GPU runtime as already checked.
        with temporary() as folder,redirect_stdout(io.StringIO()),patch.object(e,'validate',return_value=(data,{42:ref},plan)):
            e.check(folder,'cpu')
            saved=e.t.read_json(folder/'synthetic_check.json')['result']
            self.assertEqual(saved['parameter_counts'],{'adapters':18536,'head_only':2593,'adapter':21129})
            self.assertEqual(e.verify_check(folder,plan)['device'],'cpu')
            self.assertTrue(saved['identity_start']);self.assertTrue(saved['paired_warmup'])

    def test_audit_rejects_encoder_mutation_and_missing_adapter_update(self):
        m=Tiny();initial=g.snapshot(m);grad={k:0. for k in g.GRAD_GROUPS}
        with self.assertRaisesRegex(ValueError,'expected actual'):g.audit(m,initial,'adapter',10,grad)
        with torch.no_grad():m.encoder._blocks[0].weight.add_(1)
        with self.assertRaisesRegex(ValueError,'Frozen encoder'):g.audit(m,initial,'head_only',0,grad)

    def test_warmup_is_identical_and_encoder_frozen(self):
        x=np.random.default_rng(4).normal(size=(16,4));target=x[:,0]
        states=[]
        with patch.object(g,'batch_images',side_effect=batch):
            for mode in g.MODES:
                torch.manual_seed(10);m=Tiny();initial=g.snapshot(m);opt=g.optimizer_for(torch,m,mode,1)
                grad={k:0. for k in g.GRAD_GROUPS}
                g.optimize(torch,m,x,np.arange(16),target,torch.device('cpu'),opt,np.random.default_rng(1),grad)
                self.assertFalse(g.audit(m,initial,mode,1,grad)['encoder_updated']);states.append(g.snapshot(m))
        for n in states[0]:torch.testing.assert_close(states[0][n],states[1][n],rtol=0,atol=0)

    def test_accumulation_weights_short_final_chunk_correctly(self):
        x=np.random.default_rng(4).normal(size=(19,4));target=x[:,0]
        states=[]
        with patch.object(g,'batch_images',side_effect=batch):
            for batchsize,accum in ((8,2),(16,1)):
                torch.manual_seed(10);m=Tiny();m.head=torch.nn.Linear(4,1)
                opt=g.optimizer_for(torch,m,'adapter',6);grad={k:0. for k in g.GRAD_GROUPS}
                _,steps=g.optimize(torch,m,x,np.arange(19),target,torch.device('cpu'),opt,np.random.default_rng(1),grad,batch=batchsize,accum=accum)
                self.assertEqual(steps,2);states.append(g.snapshot(m))
        for n in states[0]:torch.testing.assert_close(states[0][n],states[1][n],rtol=1e-5,atol=2e-7)

    def test_resume_exact_predictions_scaling_and_query_label_independence(self):
        x=np.random.default_rng(9).normal(size=(20,4));ti,vi=np.arange(16),np.arange(16,20)
        yt=x[ti,0]*5+100;yv=x[vi,0]*5+100;optimize=g.optimize
        with temporary() as root,redirect_stdout(io.StringIO()),patch.object(g,'make_model',side_effect=lambda torch:Tiny()),patch.object(g,'batch_images',side_effect=batch):
            runner=g.Runner(torch,torch.device('cpu'),x)
            baseline=runner(ti,yt,vi,yv,'adapter',10,12,root/'baseline',{})
            count=0
            def interrupted(*args,**kwargs):
                nonlocal count
                count+=1
                if count==6:raise RuntimeError('intentional interruption after epoch5 checkpoint')
                return optimize(*args,**kwargs)
            with patch.object(g,'optimize',side_effect=interrupted):
                with self.assertRaisesRegex(RuntimeError,'intentional'):runner(ti,yt,vi,yv,'adapter',10,12,root/'resume',{})
            resumed=runner(ti,yt,vi,yv,'adapter',10,12,root/'resume',{})
            poisoned=runner(ti,yt,vi,yv+1e5,'adapter',10,12,root/'poisoned',{})
            for a,b,c in zip(baseline['records'],resumed['records'],poisoned['records']):
                np.testing.assert_array_equal(a['predictions'],b['predictions'])
                np.testing.assert_array_equal(a['predictions'],c['predictions'])
                self.assertEqual(a['update_audit'],b['update_audit'])
            self.assertEqual(baseline['target_mean'],float(yt.mean()))
            with (root/'resume'/e.t.read_json(root/'resume/progress.json')['slot']).open('ab') as f:f.write(b'corruption')
            with self.assertRaisesRegex(ValueError,'checkpoint changed'):runner(ti,yt,vi,yv,'adapter',10,12,root/'resume',{})

    def test_outer_labels_never_reach_runner_and_cache_is_read_only(self):
        data,ref,plan=fixture()
        with temporary() as folder,redirect_stdout(io.StringIO()):
            a=e.fit_fold(folder,{},data.y,ref['folds'],42,0,FakeRunner())
            self.assertEqual(len(FakeRunner.calls),8)
            self.assertEqual([c['epochs'] for c in FakeRunner.calls if c['query_y'] is not None],[80]*6)
            poisoned=data.y.copy();poisoned[:100]+=1e6
            b=e.fit_fold(folder,{},poisoned,ref['folds'],42,0,e.unavailable)
            self.assertEqual(a,b)
            for call in FakeRunner.calls:
                self.assertFalse(np.intersect1d(call['train'],call['query']).size)
                self.assertTrue(set(call['train']).issubset(set(range(100,500))))
                if len(call['train'])==400:self.assertIsNone(call['query_y'])
                else:self.assertTrue(set(call['query']).issubset(set(range(100,500))))
                np.testing.assert_array_equal(call['y'],data.y[call['train']])
            path=folder/'cells/fold0/adapter/refit/model.pt';path.write_bytes(b'corruption')
            with self.assertRaisesRegex(ValueError,'checkpoint changed'):e.fit_fold(folder,{},data.y,ref['folds'],42,0,e.unavailable)

    def test_complete_two_seed_pipeline_eighty_trajectories_no_refits(self):
        data,ref,plan=fixture();refs={s:ref for s in e.SEEDS};rt={'synthetic':True}
        with temporary() as folder,redirect_stdout(io.StringIO()),patch.object(e,'validate',return_value=(data,refs,plan)),patch.object(e,'verify_check',return_value=rt):
            for seed in e.SEEDS:
                sub=folder/f'seed{seed}';binding=e.binding_for(plan,rt,seed)
                p,s=e.collect(sub,binding,data,ref,seed,FakeRunner());e.save_results(sub,binding,data,ref,p,s,plan)
            self.assertEqual(len(FakeRunner.calls),80)
            with patch.object(g,'Runner',side_effect=AssertionError('No model')):
                for seed in e.SEEDS:e.train(folder,seed,'cpu')
                result=e.compare(folder)
            self.assertEqual(result['primary'],'adapter');self.assertEqual(result['secondary'],e.ADAPTER_BLEND)
            self.assertEqual(len(FakeRunner.calls),80)
            rows=e.core.read_rows(folder/'seed42/oof.csv',{'ID',e.ADAPTER_BLEND,'current','adapter'})
            for r in rows:self.assertAlmostEqual(float(r[e.ADAPTER_BLEND]),.5*(float(r['current'])+float(r['adapter'])))
            summary=e.t.read_json(folder/'seed42/summary.json')
            self.assertFalse(summary['test_accessed'])
            with (folder/'seed42/oof.csv').open('a') as f:f.write('bad')
            with self.assertRaisesRegex(ValueError,'artifact changed'):e.compare(folder)

    def test_missing_check_stops_before_model(self):
        data,ref,plan=fixture()
        with temporary() as folder,patch.object(e,'validate',return_value=(data,{42:ref},plan)),patch.object(e.t,'configure_device',side_effect=AssertionError('No model')):
            with self.assertRaisesRegex(ValueError,'Run check successfully'):e.train(folder,42,'cuda')


if __name__=='__main__':unittest.main()
