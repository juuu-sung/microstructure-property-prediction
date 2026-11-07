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
from scripts import micronet_finetune_experiment as e
g=e.eng


@contextmanager
def temporary():
    parent=Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory(dir=parent,prefix='.micronet_ft_test_') as name:
        if Path(name).resolve().parent!=parent:raise ValueError('Invalid cleanup path')
        yield Path(name)


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__();self.encoder=torch.nn.Module()
        self.encoder._blocks=torch.nn.ModuleList([torch.nn.Linear(4,4) for _ in range(16)])
        self.encoder.bn=torch.nn.BatchNorm1d(4)
        self.head=torch.nn.Sequential(torch.nn.Dropout(.1),torch.nn.Linear(4,1))
    def forward(self,x):
        x=self.encoder.bn(x)
        for b in self.encoder._blocks:x=torch.tanh(x+.05*b(x))
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
            active=mode=='partial' and ep>5
            audit={'encoder_updated':active,'bn_and_other_buffers_unchanged':True,'frozen_parameters_unchanged':True,
             'max_abs_parameter_change':{'head':float(ep>0),'frozen':0.,'stage6':float(active),'stage7':float(active)},
             'max_gradient_l2':{'head':float(ep>0),'stage6':float(active),'stage7':float(active)}}
            row={'epoch':ep,'predictions':pred.tolist(),'train_sse':float(len(yt)),'train_n':len(yt),'update_audit':audit}
            if yv is not None:row.update(valid_sse=float(np.sum((pred-yv)**2)),valid_n=len(yv))
            records.append(row)
        checkpoint=None
        if yv is None:
            path=folder/'model.pt';path.write_bytes(b'synthetic checkpoint')
            checkpoint={'name':'model.pt','sha256':e.base.sha256_file(path)}
        return {'records':records,'audit':records[-1]['update_audit'],'target_mean':float(yt.mean()),'target_std':float(yt.std()),'checkpoint':checkpoint}


class FineTuneTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2);FakeRunner.calls=[]

    def test_actual_micronet_stage_gradients_and_frozen_bn(self):
        # One actual native256 synthetic optimizer step per arm; CPU, not competition images.
        rng=np.random.default_rng(42);pixels=rng.integers(0,256,size=(2,256,256),dtype=np.uint8)
        initial_heads=[]
        for mode in g.MODES:
            torch.manual_seed(12);model=g.make_model(torch);initial=g.snapshot(model)
            initial_heads.append(g.state_hash({n:v for n,v in initial.items() if n.startswith('head.')}))
            opt=g.optimizer_for(torch,model,mode,6);grad={k:0. for k in ('head','stage6','stage7')}
            _,steps=g.optimize(torch,model,pixels,np.arange(2),np.array([-1.,1.]),torch.device('cpu'),opt,np.random.default_rng(5),grad,batch=2,accum=1)
            a=g.audit(model,initial,mode,6,grad)
            self.assertEqual(steps,1);self.assertEqual(a['encoder_updated'],mode=='partial')
            self.assertTrue(a['bn_and_other_buffers_unchanged'])
            self.assertFalse(any(m.training for m in model.encoder.modules() if isinstance(m,torch.nn.modules.batchnorm._BatchNorm)))
        self.assertEqual(initial_heads[0],initial_heads[1])

    def test_warmup_is_identical_and_encoder_frozen(self):
        x=np.random.default_rng(4).normal(size=(16,4));target=x[:,0]
        states=[]
        with patch.object(g,'batch_images',side_effect=batch):
            for mode in g.MODES:
                torch.manual_seed(10);m=Tiny();initial=g.snapshot(m);opt=g.optimizer_for(torch,m,mode,1)
                grad={k:0. for k in ('head','stage6','stage7')}
                g.optimize(torch,m,x,np.arange(16),target,torch.device('cpu'),opt,np.random.default_rng(1),grad)
                self.assertFalse(g.audit(m,initial,mode,1,grad)['encoder_updated']);states.append(g.snapshot(m))
        for n in states[0]:torch.testing.assert_close(states[0][n],states[1][n],rtol=0,atol=0)

    def test_accumulation_weights_short_final_chunk_correctly(self):
        x=np.random.default_rng(4).normal(size=(19,4));target=x[:,0]
        states=[]
        with patch.object(g,'batch_images',side_effect=batch):
            for batchsize,accum in ((8,2),(16,1)):
                torch.manual_seed(10);m=Tiny();m.head=torch.nn.Linear(4,1)
                opt=g.optimizer_for(torch,m,'partial',6);grad={k:0. for k in ('head','stage6','stage7')}
                _,steps=g.optimize(torch,m,x,np.arange(19),target,torch.device('cpu'),opt,np.random.default_rng(1),grad,batch=batchsize,accum=accum)
                self.assertEqual(steps,2);states.append(g.snapshot(m))
        for n in states[0]:torch.testing.assert_close(states[0][n],states[1][n],rtol=1e-5,atol=2e-7)

    def test_resume_exact_predictions_scaling_and_query_label_independence(self):
        x=np.random.default_rng(9).normal(size=(20,4));ti,vi=np.arange(16),np.arange(16,20)
        yt=x[ti,0]*5+100;yv=x[vi,0]*5+100;optimize=g.optimize
        with temporary() as root,redirect_stdout(io.StringIO()),patch.object(g,'make_model',side_effect=lambda torch:Tiny()),patch.object(g,'batch_images',side_effect=batch):
            runner=g.Runner(torch,torch.device('cpu'),x)
            baseline=runner(ti,yt,vi,yv,'partial',10,12,root/'baseline',{})
            count=0
            def interrupted(*args,**kwargs):
                nonlocal count
                count+=1
                if count==6:raise RuntimeError('intentional interruption after epoch5 checkpoint')
                return optimize(*args,**kwargs)
            with patch.object(g,'optimize',side_effect=interrupted):
                with self.assertRaisesRegex(RuntimeError,'intentional'):runner(ti,yt,vi,yv,'partial',10,12,root/'resume',{})
            resumed=runner(ti,yt,vi,yv,'partial',10,12,root/'resume',{})
            poisoned=runner(ti,yt,vi,yv+1e5,'partial',10,12,root/'poisoned',{})
            for a,b,c in zip(baseline['records'],resumed['records'],poisoned['records']):
                np.testing.assert_array_equal(a['predictions'],b['predictions'])
                np.testing.assert_array_equal(a['predictions'],c['predictions'])
                self.assertEqual(a['update_audit'],b['update_audit'])
            self.assertEqual(baseline['target_mean'],float(yt.mean()))
            with (root/'resume'/e.t.read_json(root/'resume/progress.json')['slot']).open('ab') as f:f.write(b'corruption')
            with self.assertRaisesRegex(ValueError,'checkpoint changed'):runner(ti,yt,vi,yv,'partial',10,12,root/'resume',{})

    def test_outer_labels_never_reach_runner_and_cache_is_read_only(self):
        data,ref,plan=fixture()
        with temporary() as folder,redirect_stdout(io.StringIO()):
            a=e.fit_fold(folder,{},data.y,ref['folds'],42,0,FakeRunner())
            self.assertEqual(len(FakeRunner.calls),8)
            poisoned=data.y.copy();poisoned[:100]+=1e6
            b=e.fit_fold(folder,{},poisoned,ref['folds'],42,0,e.unavailable)
            self.assertEqual(a,b)
            for call in FakeRunner.calls:
                self.assertFalse(np.intersect1d(call['train'],call['query']).size)
                self.assertTrue(set(call['train']).issubset(set(range(100,500))))
                if len(call['train'])==400:self.assertIsNone(call['query_y'])
                else:self.assertTrue(set(call['query']).issubset(set(range(100,500))))
                np.testing.assert_array_equal(call['y'],data.y[call['train']])
            path=folder/'cells/fold0/partial/refit/model.pt';path.write_bytes(b'corruption')
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
            self.assertEqual(result['primary'],'partial');self.assertEqual(result['secondary'],e.PARTIAL_BLEND)
            self.assertEqual(len(FakeRunner.calls),80)
            rows=e.core.read_rows(folder/'seed42/oof.csv',{'ID',e.PARTIAL_BLEND,'current','partial'})
            for r in rows:self.assertAlmostEqual(float(r[e.PARTIAL_BLEND]),.5*(float(r['current'])+float(r['partial'])))
            summary=e.t.read_json(folder/'seed42/summary.json')
            self.assertFalse(summary['test_accessed'])
            with (folder/'seed42/oof.csv').open('a') as f:f.write('bad')
            with self.assertRaisesRegex(ValueError,'artifact changed'):e.compare(folder)

    def test_missing_check_stops_before_model(self):
        data,ref,plan=fixture()
        with temporary() as folder,patch.object(e,'validate',return_value=(data,{42:ref},plan)),patch.object(e.t,'configure_device',side_effect=AssertionError('No model')):
            with self.assertRaisesRegex(ValueError,'Run check successfully'):e.train(folder,42,'cuda')


if __name__=='__main__':unittest.main()
