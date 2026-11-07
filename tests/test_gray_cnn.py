"""Synthetic-only architecture, optimization, leakage, resumability and report tests."""
import copy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
from scripts import gray_cnn_experiment as e
g=e.eng


def temporary():return tempfile.TemporaryDirectory(dir=Path(__file__).parent,prefix='.gray_cnn_')


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        for name in g.GROUPS[:-1]:setattr(self,name,torch.nn.Linear(4,4))
        self.head=torch.nn.Sequential(torch.nn.Dropout(.1),torch.nn.Linear(4,1))
    def forward(self,x):
        for name in g.GROUPS[:-1]:x=torch.tanh(x+.1*getattr(self,name)(x))
        return self.head(x).squeeze(1)


def batch(torch,pixels,indices,device):return torch.as_tensor(pixels[indices].copy(),dtype=torch.float32,device=device)


def fixture():
    rng=np.random.default_rng(7);y=rng.normal(200,15,500)
    data=SimpleNamespace(ids=np.array([f'TRAIN_{i:06d}' for i in range(500)]),y=y,noise=rng.uniform(size=500))
    ref={'folds':np.repeat(np.arange(5),100),'current':y+5}
    plan={'frozen':{'common_failure_ids':data.ids[:16].tolist(),'synthetic':True}}
    return data,ref,plan


class FakeRunner:
    calls=[]
    def __call__(self,ti,yt,vi,yv,mode,epochs,seed,folder,binding):
        self.calls.append({'train':ti.copy(),'query':vi.copy(),'yt':yt.copy(),'yv':yv,'seed':seed,'mode':mode})
        records=[]
        for ep in g.EPOCHS:
            if ep>epochs:continue
            p=np.full(len(vi),float(np.mean(yt))+ep*.001)
            audit={'all_stages_updated':ep>0,'max_gradient_l2':{name:float(ep>0) for name in g.GROUPS},
                   'max_abs_parameter_change':{name:float(ep>0) for name in g.GROUPS}}
            row={'epoch':ep,'predictions':p.tolist(),'train_sse':float(len(yt)),'train_n':len(yt),'update_audit':audit}
            if yv is not None:row.update(valid_n=len(yv),valid_sse=float(np.sum((p-yv)**2)))
            records.append(row)
        checkpoint=None
        if yv is None:
            (folder/'model.pt').write_bytes(b'synthetic model')
            checkpoint={'name':'model.pt','sha256':e.e.base.sha256_file(folder/'model.pt')}
        return {'records':records,'audit':records[-1]['update_audit'],'checkpoint':checkpoint,
                'target_mean':float(np.mean(yt)),'target_std':max(float(np.std(yt)),1e-8)}


class GrayCNNTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2);FakeRunner.calls=[]

    def test_native_resolutions_matched_weights_count_and_actual_updates(self):
        pixels=np.random.default_rng(3).integers(0,256,(2,256,256),dtype=np.uint8)
        states=[];counts=[]
        for mode in g.MODES:
            torch.manual_seed(51);m=g.make_model(torch,mode);initial=g.snapshot(m);states.append(g.state_hash(initial))
            counts.append(sum(p.numel() for p in m.parameters()))
            with torch.no_grad():maps=m.maps(g.batch_images(torch,pixels,np.arange(2),'cpu'))
            self.assertEqual([a.shape[-1] for a in maps],[256,128,64,32] if mode==e.PRIMARY else [128,64,32,16])
            self.assertFalse(any(isinstance(a,torch.nn.modules.batchnorm._BatchNorm) for a in m.modules()))
            grad={n:0. for n in g.GROUPS};opt=g.optimizer_for(torch,m)
            _,steps=g.optimize(torch,m,pixels,np.arange(2),np.array([-1.,1.]),torch.device('cpu'),opt,np.random.default_rng(3),grad,batch=2,accum=1)
            self.assertEqual(steps,1);self.assertTrue(g.audit(m,initial,1,grad)['all_stages_updated'])
            self.assertLess(counts[-1],500000)
        self.assertEqual(states[0],states[1]);self.assertEqual(counts[0],counts[1])

    def test_brightness_preserved_and_query_batch_independence(self):
        torch.manual_seed(3);m=g.make_model(torch,e.PRIMARY).eval();seen=[]
        handle=m.head[0].register_forward_pre_hook(lambda module,args:seen.append(args[0].detach()))
        x=torch.full((2,1,256,256),.3)
        with torch.no_grad():p=m(x)
        self.assertEqual(seen[-1].shape,(2,426))
        torch.testing.assert_close(seen[-1][:,-10],torch.full((2,),.3))
        x[1]=.8
        with torch.no_grad():p2=m(x)
        torch.testing.assert_close(p[:1],p2[:1],rtol=0,atol=0)
        handle.remove()

    def test_accumulation_normalizes_last_partial_chunk(self):
        pixels=np.random.default_rng(5).normal(size=(13,4));y=pixels[:,0];states=[]
        with patch.object(g,'batch_images',side_effect=batch):
            for batch_size,accum in ((4,4),(16,1)):
                torch.manual_seed(3);m=Tiny();m.head[0].p=0;opt=g.optimizer_for(torch,m)
                grad={n:0. for n in g.GROUPS}
                _,steps=g.optimize(torch,m,pixels,np.arange(13),y,'cpu',opt,np.random.default_rng(6),grad,batch=batch_size,accum=accum)
                self.assertEqual(steps,1);states.append(g.snapshot(m))
        for n in states[0]:torch.testing.assert_close(states[0][n],states[1][n],rtol=1e-5,atol=2e-7)

    def test_resume_exact_predictions_and_query_labels_do_not_affect_fit(self):
        pixels=np.random.default_rng(8).normal(size=(20,4));ti,vi=np.arange(16),np.arange(16,20)
        yt=200+pixels[ti,0]*5;yv=200+pixels[vi,0]*5;original=g.optimize
        with temporary() as temp,redirect_stdout(io.StringIO()),patch.object(g,'make_model',side_effect=lambda *a:Tiny()),patch.object(g,'batch_images',side_effect=batch):
            root=Path(temp);runner=g.Runner(torch,torch.device('cpu'),pixels)
            baseline=runner(ti,yt,vi,yv,e.PRIMARY,10,30,root/'baseline',{})
            count=0
            def interrupted(*args,**kwargs):
                nonlocal count
                count+=1
                if count==6:raise RuntimeError('interrupt after saved epoch5')
                return original(*args,**kwargs)
            with patch.object(g,'optimize',side_effect=interrupted):
                with self.assertRaisesRegex(RuntimeError,'interrupt'):runner(ti,yt,vi,yv,e.PRIMARY,10,30,root/'resume',{})
            resumed=runner(ti,yt,vi,yv,e.PRIMARY,10,30,root/'resume',{})
            poisoned=runner(ti,yt,vi,yv+1e6,e.PRIMARY,10,30,root/'poisoned',{})
            for a,b,c in zip(baseline['records'],resumed['records'],poisoned['records']):
                np.testing.assert_array_equal(a['predictions'],b['predictions']);np.testing.assert_array_equal(a['predictions'],c['predictions'])
                self.assertEqual(a['update_audit'],b['update_audit'])
            marker=e.e.read_json(root/'resume/progress.json');(root/'resume'/marker['slot']).write_bytes(b'bad')
            with self.assertRaisesRegex(ValueError,'checksum'):runner(ti,yt,vi,yv,e.PRIMARY,10,30,root/'resume',{})

    def test_outer_label_poisoning_nested_membership_and_cached_read(self):
        data,ref,plan=fixture()
        with temporary() as temp,redirect_stdout(io.StringIO()):
            folder=Path(temp);a=e.fit_fold(folder,{},data.y,ref['folds'],42,0,FakeRunner())
            self.assertEqual(len(FakeRunner.calls),8)
            y=data.y.copy();y[:100]+=1e6
            b=e.fit_fold(folder,{},y,ref['folds'],42,0,e.forbidden)
            self.assertEqual(a,b)
            for call in FakeRunner.calls:
                self.assertFalse(set(call['train'])&set(call['query']))
                self.assertTrue(set(call['train']).issubset(set(range(100,500))))
                np.testing.assert_array_equal(call['yt'],data.y[call['train']])
                if len(call['train'])==400:self.assertIsNone(call['yv'])
                else:self.assertTrue(set(call['query']).issubset(set(range(100,500))))
            (folder/'cells/fold0/native_stride1/refit/model.pt').write_bytes(b'bad')
            with self.assertRaisesRegex(ValueError,'Checkpoint'):e.fit_fold(folder,{},y,ref['folds'],42,0,e.forbidden)

    def test_complete_two_seed_synthetic_pipeline_and_readonly_comparison(self):
        data,ref,plan=fixture();rt={'synthetic':True};summaries=[]
        with temporary() as temp,redirect_stdout(io.StringIO()),patch.object(e,'validate',return_value=(data,{s:ref for s in e.SEEDS},plan)),patch.object(e,'verify_check',return_value=rt):
            folder=Path(temp)
            for seed in e.SEEDS:
                binding=e.binding_for(plan,rt,seed);sub=folder/f'seed{seed}'
                p,c=e.collect(sub,binding,data,ref,seed,FakeRunner())
                summaries.append(e.save_results(sub,binding,data,ref,p,c,plan,seed))
                rows=e.core.read_rows(sub/'oof.csv',{'ID',e.PRIMARY,e.SECONDARY,'current'})
                for row in rows:self.assertAlmostEqual(float(row[e.SECONDARY]),.5*(float(row['current'])+float(row[e.PRIMARY])))
            self.assertEqual(len(FakeRunner.calls),80)
            with patch.object(g,'Runner',side_effect=AssertionError('No fitting')),patch.object(e.e,'configure',side_effect=AssertionError('No model')):
                result=e.compare(folder)
                for seed in e.SEEDS:e.train(folder,seed,'cpu')
            self.assertFalse(result['automatic_adoption']);json.dumps(result,allow_nan=False)
            for s in summaries:
                for name in (e.PRIMARY,e.SECONDARY):
                    self.assertIn('common16',s['diagnosis'][name]['groups']);self.assertIn('remaining484',s['diagnosis'][name]['groups'])
                self.assertNotIn('encoder_updates',s)
            (folder/'seed42/common16.csv').write_text('tamper')
            with self.assertRaises(ValueError):e.compare(folder)

    def test_fixed_schedule_prefix_epoch_choice_and_driver(self):
        self.assertEqual(g.lr_at(5),3e-4);self.assertAlmostEqual(g.lr_at(80),3e-5)
        runs=[{'records':[{'epoch':ep,'valid_n':n,'valid_sse':float(n),'train_n':400-n,'train_sse':float(400-n)} for ep in g.EPOCHS]} for n in (134,133,133)]
        selected,_=e.select_epoch(runs);self.assertEqual(selected['epoch'],0)
        text=(e.ROOT/'scripts/run_gray_cnn.ps1').read_text()
        self.assertIn('$LASTEXITCODE -ne 0',text);self.assertNotIn('download',text);self.assertNotIn('submission',text)


if __name__=='__main__':unittest.main()
