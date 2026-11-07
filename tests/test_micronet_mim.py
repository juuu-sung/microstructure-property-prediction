"""Synthetic-only masking, split isolation, actual gradient and cache tests."""
from contextlib import contextmanager,redirect_stdout
import inspect
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
from scripts import micronet_mim_experiment as e
g=e.eng


@contextmanager
def temporary():
    parent=Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory(dir=parent,prefix='.micronet_mim_test_') as name:
        if Path(name).resolve().parent!=parent:raise ValueError('Invalid cleanup path')
        yield Path(name)


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__();self.encoder=torch.nn.Module()
        self.encoder._conv_stem=torch.nn.Conv2d(3,4,1)
        self.encoder._bn0=torch.nn.BatchNorm2d(4)
        self.encoder._blocks=torch.nn.ModuleDict({'15':torch.nn.Conv2d(4,4,1)})
        self.encoder.unused=torch.nn.Linear(2,2)
        self.decoder=torch.nn.Conv2d(4,1,1)
    def encode(self,x):
        return torch.tanh(self.encoder._blocks['15'](self.encoder._bn0(self.encoder._conv_stem(x))))
    def forward(self,x):return self.decoder(self.encode(x))
    def features(self,x):return self.encode(x).mean((2,3)).repeat(1,216)


def tiny_masks(rng,n):
    mask=np.zeros((n,16),dtype=np.float32)
    for row in mask:row[rng.permutation(16)[:8]]=1
    return mask.reshape(n,1,4,4)


def fixture():
    ids=np.array([f'TRAIN_{i:06d}' for i in range(500)])
    y=100+5*np.sin(np.arange(500)/3)
    data=SimpleNamespace(ids=ids,y=y,noise=np.arange(500,dtype=float))
    x0=np.zeros((500,864),np.float32);folds=np.repeat(np.arange(5),100)
    anchor=np.array([y[folds!=f].mean() for f in folds])
    ref={'current':y+3,'folds':folds,'frozen_anchor':anchor}
    plan={'frozen':{'common_failure_ids':list(ids[:16])}}
    return data,ref,x0,plan


def fake_ridge(tx,yt,qx,alpha):return yt.mean()+qx[:,0].astype(float)


class FakeRunner:
    def __init__(self,improve=True):self.calls=[];self.improve=improve
    def __call__(self,ti,vi,epochs,seed,folder,binding):
        self.calls.append({'train':ti.copy(),'query':vi.copy(),'epochs':epochs,'seed':seed,'role':binding['role']})
        records=[];artifacts={}
        for ep in [0,*range(5,epochs+1,5)]:
            row={'epoch':ep}
            if ep in g.EPOCHS:
                tx=np.zeros((len(ti),864),np.float32);qx=np.zeros((len(vi),864),np.float32)
                if ep>0 and self.improve:tx[:,0]=5*np.sin(ti/3);qx[:,0]=5*np.sin(vi/3)
                path=folder/f'features_epoch{ep}.npz';g.save_npz(path,train=tx,query=qx)
                artifacts[path.name]=e.base.sha256_file(path)
                a={'encoder_updated':ep>0,'bn_buffers_unchanged':True,'unused_parameters_unchanged':True,
                    'initial_encoder_sha256':'initial','final_encoder_sha256':'changed' if ep>0 else 'initial',
                    'max_abs_parameter_change':{**{k:float(ep>0) for k in g.GROUPS},'frozen':0.},
                    'max_gradient_l2':{k:float(ep>0) for k in g.GROUPS}}
                row.update(feature_file=path.name,update_audit=a)
            records.append(row)
        if binding['role']=='outer':
            path=folder/'model.pt';path.write_bytes(b'synthetic model');artifacts[path.name]=e.base.sha256_file(path)
        return {'records':records,'artifacts':artifacts,'audit':records[-1]['update_audit'],
            'fitting_indices':ti.tolist(),'query_indices':vi.tolist(),'hardness_labels_used':False,'query_images_used_for_ssl':False}


class MIMTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2)

    def test_masks_hide_all_target_values_and_loss_only_counts_mask(self):
        rng=np.random.default_rng(4);mask=torch.from_numpy(g.masks(rng,3))
        self.assertEqual(tuple(mask.shape),(3,1,256,256))
        np.testing.assert_array_equal(mask.sum((1,2,3)),[39*32*32]*3)
        raw=torch.rand_like(mask);changed=raw*(1-mask)+(1-raw)*mask
        torch.testing.assert_close(g.masked_input(torch,raw,mask),g.masked_input(torch,changed,mask),rtol=0,atol=0)
        prediction=raw.clone().requires_grad_();loss=g.masked_l1(torch,prediction+1,raw,mask).mean();loss.backward()
        self.assertEqual(float(loss.detach()),1.);self.assertEqual(float(prediction.grad[mask==0].abs().sum()),0.)
        self.assertGreater(float(prediction.grad[mask==1].abs().sum()),0.)

    def test_real_encoder_clean_control_and_ssl_updates(self):
        rng=np.random.default_rng(4);pixels=rng.integers(0,256,(2,256,256),dtype=np.uint8)
        torch.manual_seed(7);m=g.make_model(torch);initial=g.snapshot(m);m.eval()
        raw=g.raw_batch(torch,pixels,np.arange(2),torch.device('cpu'));x=g.normalized(torch,raw)
        with torch.inference_mode():
            torch.testing.assert_close(m.features(x),g.old.pool_maps(g.old.forward_maps(m.encoder,x)),rtol=0,atol=0)
            self.assertEqual(tuple(m(x).shape),(2,1,256,256))
        opt=g.optimizer_for(torch,m);grad={k:0. for k in g.GROUPS}
        loss,steps=g.optimize(torch,m,pixels,torch.device('cpu'),opt,rng,grad,batch=2,accum=1)
        self.assertTrue(np.isfinite(loss));self.assertEqual(steps,1)
        a=g.audit(m,initial,1,grad);self.assertTrue(a['encoder_updated']);self.assertTrue(a['bn_buffers_unchanged'])
        self.assertTrue(all(grad[k]>0 for k in g.GROUPS))
        self.assertFalse(any(b.training for b in m.encoder.modules() if isinstance(b,torch.nn.modules.batchnorm._BatchNorm)))

    def test_query_pixels_cannot_change_ssl_training_and_resume_is_exact(self):
        rng=np.random.default_rng(19);pixels=rng.integers(0,256,(20,4,4),dtype=np.uint8)
        ti,vi=np.arange(16),np.arange(16,20);optimize=g.optimize
        self.assertFalse(any('label' in n or n.endswith('_y') for n in inspect.signature(g.Runner.__call__).parameters))
        with temporary() as root,redirect_stdout(io.StringIO()),patch.object(g,'make_model',side_effect=lambda torch:Tiny()),patch.object(g,'masks',side_effect=tiny_masks):
            runner=g.Runner(torch,torch.device('cpu'),pixels)
            original=runner(ti,vi,20,17,root/'original',{'role':'inner'})
            count=0
            def interrupted(*a,**kw):
                nonlocal count
                count+=1
                if count==11:raise RuntimeError('planned interruption')
                return optimize(*a,**kw)
            with patch.object(g,'optimize',side_effect=interrupted):
                with self.assertRaisesRegex(RuntimeError,'planned'):runner(ti,vi,20,17,root/'resumed',{'role':'inner'})
            resumed=runner(ti,vi,20,17,root/'resumed',{'role':'inner'})
            poison=pixels.copy();poison[vi]=255-poison[vi]
            other=g.Runner(torch,torch.device('cpu'),poison)(ti,vi,20,17,root/'poison',{'role':'inner'})
            self.assertEqual(original['audit'],resumed['audit']);self.assertEqual(original['audit'],other['audit'])
            for ep in (0,20):
                with np.load(root/f'original/features_epoch{ep}.npz') as a,np.load(root/f'resumed/features_epoch{ep}.npz') as b,np.load(root/f'poison/features_epoch{ep}.npz') as c:
                    np.testing.assert_array_equal(a['train'],b['train']);np.testing.assert_array_equal(a['query'],b['query'])
                    np.testing.assert_array_equal(a['train'],c['train'])
                    self.assertFalse(np.array_equal(a['query'],c['query']))
            marker=e.t.read_json(root/'resumed/progress.json')
            with (root/'resumed'/marker['slot']).open('ab') as f:f.write(b'corrupt')
            with self.assertRaisesRegex(ValueError,'Resume checkpoint changed'):runner(ti,vi,20,17,root/'resumed',{'role':'inner'})

    def test_gradient_accumulation_including_short_chunk(self):
        pixels=np.random.default_rng(4).integers(0,256,(19,4,4),dtype=np.uint8);states=[]
        with patch.object(g,'masks',side_effect=tiny_masks):
            for batch,accum in ((8,2),(16,1)):
                torch.manual_seed(18);m=Tiny();opt=g.optimizer_for(torch,m);grad={k:0. for k in g.GROUPS}
                _,steps=g.optimize(torch,m,pixels,torch.device('cpu'),opt,np.random.default_rng(4),grad,batch=batch,accum=accum)
                self.assertEqual(steps,2);states.append(g.snapshot(m))
        for k in states[0]:torch.testing.assert_close(states[0][k],states[1][k],rtol=1e-5,atol=2e-7)

    def test_nested_scope_outer_target_poison_and_corruption(self):
        data,ref,x0,plan=fixture();runner=FakeRunner()
        with temporary() as root,redirect_stdout(io.StringIO()),patch.object(e,'ridge_predict',side_effect=fake_ridge):
            result=e.fit_fold(root,{},x0,data.y,ref['folds'],42,0,runner)
            self.assertEqual(result['selection']['mim_nested']['epoch'],20)
            self.assertEqual(len(runner.calls),4)
            for call in runner.calls:
                self.assertFalse(np.intersect1d(call['train'],call['query']).size)
                self.assertTrue(set(call['train']).issubset(set(range(100,500))))
                if call['role']=='inner':self.assertTrue(set(call['query']).issubset(set(range(100,500))))
                else:self.assertEqual(call['query'].tolist(),list(range(100)))
            poisoned=data.y.copy();poisoned[:100]+=1e6
            with patch.object(e,'ridge_predict',side_effect=AssertionError('No implicit fitting')):
                self.assertEqual(result,e.fit_fold(root,{},x0,poisoned,ref['folds'],42,0,e.unavailable))
            with (root/'cells/fold0/inner0/features_epoch20.npz').open('ab') as f:f.write(b'corruption')
            with self.assertRaisesRegex(ValueError,'artifact changed'):e.fit_fold(root,{},x0,data.y,ref['folds'],42,0,e.unavailable)

    def test_epoch0_fallback_and_full_pipeline_both_seeds(self):
        data,ref,x0,plan=fixture();refs={s:ref for s in e.SEEDS};rt={'synthetic':True};runner=FakeRunner(improve=False)
        with temporary() as root,redirect_stdout(io.StringIO()),patch.object(e,'ridge_predict',side_effect=fake_ridge),patch.object(e,'validate',return_value=(data,refs,x0,plan)),patch.object(e,'verify_check',return_value=rt):
            for seed in e.SEEDS:
                sub=root/f'seed{seed}';binding=e.binding_for(plan,rt,seed)
                pred,sel=e.collect(sub,binding,data,ref,x0,seed,runner)
                self.assertTrue(all(x['choices'][e.PRIMARY]['epoch']==0 for x in sel))
                np.testing.assert_array_equal(pred['frozen_ridge'],pred[e.PRIMARY])
                e.save_results(sub,binding,data,ref,pred,sel,plan)
            self.assertEqual(len(runner.calls),40)
            with patch.object(e,'ridge_predict',side_effect=AssertionError('No fitting')),patch.object(g,'Runner',side_effect=AssertionError('No model')):
                e.train(root,42,'cpu');result=e.compare(root)
            self.assertTrue(all(s['ssl_selected_folds']==0 for s in result['per_seed']))
            rows=e.core.read_rows(root/'seed42/oof.csv',{'current','mim_nested',e.CANDIDATE_BLEND})
            for row in rows:self.assertAlmostEqual(float(row[e.CANDIDATE_BLEND]),.5*(float(row['current'])+float(row['mim_nested'])))
            with (root/'seed42/oof.csv').open('a') as f:f.write('bad')
            with self.assertRaisesRegex(ValueError,'artifact changed'):e.compare(root)

    def test_real_ridge_scaling_only_sees_training_values(self):
        rng=np.random.default_rng(45);tx=rng.normal(size=(40,8));yt=100+5*tx[:,0];qx=rng.normal(size=(6,8))
        a=e.ridge_predict(tx,yt,qx,100)
        b=e.ridge_predict(tx,yt,np.r_[qx,np.full((1,8),1e6)],100)
        np.testing.assert_allclose(a,b[:6],rtol=0,atol=1e-12)
        model=e.core.make_estimator(100);model.fit(tx,yt)
        np.testing.assert_allclose(model.regressor_.named_steps['standardscaler'].mean_,tx.mean(0))
        self.assertAlmostEqual(model.transformer_.mean_[0],yt.mean())

    def test_missing_check_stops_before_model(self):
        data,ref,x0,plan=fixture()
        with temporary() as root,patch.object(e,'validate',return_value=(data,{42:ref},x0,plan)),patch.object(e.t,'configure_device',side_effect=AssertionError('No model')):
            with self.assertRaisesRegex(ValueError,'Run check successfully'):e.train(root,42,'cuda')

    def test_initial_anchor_failure_stops_before_ssl_update(self):
        pixels=np.random.default_rng(5).integers(0,256,(20,4,4),dtype=np.uint8)
        wrong=np.full((20,864),1e6,np.float32)
        with temporary() as root,patch.object(g,'make_model',side_effect=lambda torch:Tiny()),patch.object(g,'optimize',side_effect=AssertionError('Must not train')) as optimize:
            with self.assertRaisesRegex(AssertionError,'Not equal to tolerance'):
                g.Runner(torch,torch.device('cpu'),pixels,wrong)(np.arange(16),np.arange(16,20),20,4,root,{'role':'inner'})
            optimize.assert_not_called()

    def test_user_check_with_real_encoder_synthetic_cpu(self):
        data,ref,x0,plan=fixture()
        with temporary() as root,redirect_stdout(io.StringIO()),patch.object(e,'validate',return_value=(data,{42:ref},x0,plan)):
            e.check(root,'cpu');self.assertEqual(e.verify_check(root,plan)['device'],'cpu')


if __name__=='__main__':unittest.main()
