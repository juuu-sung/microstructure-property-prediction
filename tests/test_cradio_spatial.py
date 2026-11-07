"""Synthetic-only tests: leakage boundaries, matched heads, caches and artifacts."""
import contextlib
import copy
import inspect
import io
import math
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from scripts import cradio_spatial_engine as eng
from scripts import cradio_spatial_experiment as exp


class SpatialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def store(self):
        rng=np.random.default_rng(12)
        x=rng.normal(size=(80,9,4)).astype(np.float32);s=rng.normal(size=(80,6)).astype(np.float32)
        return eng.TokenStore(x,s,x.mean(1,dtype=np.float64),np.mean(x.astype(np.float64)**2,axis=1),'cpu')

    def test_train_only_scaling_and_query_label_free_api(self):
        store=self.store();ti=np.arange(60);y=np.sin(ti)
        scale=store.scaling(ti,y)
        expected=store.tokens[ti].numpy().reshape(-1,4).astype(float)
        np.testing.assert_allclose(scale['token_mean'],expected.mean(0),atol=1e-12)
        np.testing.assert_allclose(scale['token_scale'],expected.std(0),atol=1e-12)
        store.means[60:]+=1000;store.second[60:]+=1e8;store.summary_np[60:]-=1000
        changed=store.scaling(ti,y)
        for k in scale:np.testing.assert_array_equal(scale[k],changed[k])
        self.assertNotIn('query_y',inspect.signature(eng.trajectory).parameters)
        self.assertNotIn('query_y',inspect.signature(exp.inner_large).parameters)

    def test_matched_initialization_and_actual_spatial_dependency(self):
        torch.manual_seed(77);g=eng.Readout('global_pool',4,6)
        torch.manual_seed(77);c=eng.Readout('spatial_attention',4,6)
        for k,v in g.state_dict().items():torch.testing.assert_close(v,c.state_dict()[k],rtol=0,atol=0)
        with torch.no_grad():
            g.head[-1].weight.normal_(0,.2);c.head[-1].weight.copy_(g.head[-1].weight)
        g.eval();c.eval();x=torch.randn(3,9,4);s=torch.randn(3,6);perm=torch.tensor([8,1,5,2,7,0,3,6,4])
        torch.testing.assert_close(g(x,s),c(x,s),atol=1e-6,rtol=1e-6)
        torch.testing.assert_close(g(x,s),g(x[:,perm],s),atol=1e-6,rtol=1e-6)
        with torch.no_grad():c.pointwise.weight.normal_(0,.3)
        self.assertGreater(float((c(x,s)-c(x[:,perm],s)).detach().abs().max()),1e-5)

    def test_training_checkpoint_reproduces_predictions_and_prefix(self):
        store=self.store();ti=np.arange(60);qi=np.arange(60,80);y=100+10*store.means[ti,0]
        with tempfile.TemporaryDirectory(dir=exp.ROOT/'.task-tmp',prefix='spatial-test-') as tmp:
            ck=Path(tmp)/'head.pt'
            run=eng.trajectory(store,ti,y,qi,kind='spatial_attention',seed=1,epochs=(0,5),checkpoint=ck)
            self.assertEqual(run['audit']['optimizer_steps'],20)
            self.assertGreater(run['audit']['parameter_delta_l2'],0)
            saved=torch.load(ck,weights_only=True);head=eng.Readout('spatial_attention',4,6)
            head.load_state_dict(saved['state_dict'])
            sc=store.tensors(saved['scaling'])
            np.testing.assert_allclose(eng.predict(head,store,qi,sc),run['records'][-1]['predictions'],atol=0,rtol=0)
            prefix=eng.trajectory(store,ti,y,qi,kind='spatial_attention',seed=1,epochs=(5,))
            np.testing.assert_array_equal(prefix['records'][0]['predictions'],run['records'][-1]['predictions'])
            with self.assertRaises(ValueError):eng.trajectory(store,ti,y,ti[:3],kind='global_pool',seed=1,epochs=(0,))

    def test_epoch_selection_uses_fixed_blend_not_standalone(self):
        y=np.array([0.,1.,2.,3.,4.,5.]);splits=[];runs=[]
        # Other components already predict75% of y plus1; correct new branch is y-4.
        other=.75*y+1
        for vi in np.array_split(np.arange(6),3):
            splits.append((np.setdiff1d(np.arange(6),vi),vi))
            runs.append({'records':[{'epoch':ep,'predictions':(y[vi]-(4 if ep==5 else 0)).tolist(),
                'train_sse':1.,'train_n':4} for ep in eng.EPOCHS]})
        best,curves=eng.select_epoch(runs,splits,y,other)
        self.assertEqual(best['epoch'],5);self.assertEqual(best['blend_mse'],0.)
        self.assertGreater(best['standalone_rmse'],curves[0]['standalone_rmse'])
        for r in runs:
            for row in r['records']:row['predictions']=y[np.array_split(np.arange(6),3)[runs.index(r)]].tolist()
        self.assertEqual(eng.select_epoch(runs,splits,y,other)[0]['epoch'],0)

    def test_fold_boundaries_resume_and_tamper_rejection(self):
        ti=np.arange(400);qi=np.arange(400,500);y=ti.astype(float)+1000;large=np.arange(500,dtype=float)[:,None]
        calls=[]
        def large_run(x,target,q,seed):
            np.testing.assert_array_equal(target,x[:,0]+1000)
            self.assertFalse(set(x[:,0])&set(q[:,0]));self.assertLessEqual(len(x),267)
            return {'predictions':np.full(len(q),target.mean()).tolist()}
        def head_run(store,ctx,target,query,**kwargs):
            np.testing.assert_array_equal(target,ctx+1000);self.assertFalse(set(ctx)&set(query))
            calls.append((ctx.copy(),query.copy()))
            ep=kwargs['epochs'];ck=kwargs['checkpoint']
            if ck is not None:ck.parent.mkdir(parents=True,exist_ok=True);ck.write_bytes(b'synthetic checkpoint')
            return {'records':[{'epoch':n,'predictions':np.full(len(query),target.mean()).tolist(),
                'train_sse':1.,'train_n':len(ctx)} for n in ep],
                'audit':{'train_indices':ctx.tolist(),'query_indices':query.tolist(),'epochs':list(ep),
                    'optimizer_steps':math.ceil(len(ctx)/eng.BATCH)*max(ep),'encoder_updates':0,'query_labels_accepted':False,
                    'parameter_delta_l2':float(max(ep)>0),'max_gradient_norm':float(max(ep)>0)},
                'checkpoint_path':str(ck) if ck else None}
        with tempfile.TemporaryDirectory(dir=exp.ROOT/'.task-tmp',prefix='spatial-fold-') as tmp:
            path=Path(tmp)
            with patch.object(exp,'inner_large',side_effect=large_run),contextlib.redirect_stdout(io.StringIO()):
                p,c=exp.fit_fold(path,{'test':1},None,large,ti,qi,y,np.zeros(400),42,0,True,head_run)
                self.assertEqual(len(calls),8)
                with patch.object(exp,'inner_large',side_effect=AssertionError('unexpected refit')):
                    p2,c2=exp.fit_fold(path,{'test':1},None,large,ti,qi,y,np.zeros(400),42,0,False,
                                     run=lambda *a,**k:(_ for _ in ()).throw(AssertionError('unexpected refit')))
            self.assertEqual(c,c2)
            for k in p:np.testing.assert_array_equal(p[k],p2[k])
            cell=path/'global_pool/inner0.json';obj=exp.e.read_json(cell);obj['result']['records'][0]['predictions'][0]+=1
            exp.e.write_json(cell,obj)
            with self.assertRaises(ValueError):exp.fit_fold(path,{'test':1},None,large,ti,qi,y,np.zeros(400),42,0,False,head_run)

    def test_fixed_quarter_composition_and_missing_cache_fail(self):
        x=np.arange(5,dtype=float)
        p=eng.compose(x,2*x,3*x,4*x,5*x)
        np.testing.assert_array_equal(p[exp.PRIMARY]-p['current'],.25*(5*x-3*x))
        with tempfile.TemporaryDirectory(dir=exp.ROOT/'.task-tmp',prefix='spatial-cache-') as tmp:
            with self.assertRaises(FileNotFoundError):exp.e.cell(Path(tmp)/'missing.json',{},lambda:self.fail('fit'),False)

    def test_full_evaluation_exports_and_reconstructs_without_fitting(self):
        rng=np.random.default_rng(6);ids=np.array([f'SYNTH_{i:06d}' for i in range(500)])
        y=190+rng.normal(size=500)*15
        data=SimpleNamespace(ids=ids,y=y,noise=rng.uniform(size=500))
        folds=np.arange(500)%5;large=rng.normal(size=(500,3))
        s=np.full(500,190.);l=np.full(500,191.);r=np.full(500,192.)
        ref={'folds':folds,'scattering':s,'large':l,'radio':r,'current':.5*s+.25*l+.25*r,
             'inner_scattering':{f:np.full(400,190.) for f in range(5)}}
        plan={'frozen':{'common_failure_ids':ids[:16].tolist()}}
        tokens=rng.normal(size=(500,9,4)).astype(np.float32)
        aux={'ids':ids,'summary':rng.normal(size=(500,6)).astype(np.float32),
             'means':tokens.mean(1,dtype=float),'second':np.mean(tokens.astype(float)**2,1)}
        actual_fold=exp.fit_fold;calls=[]
        def head_run(store,ctx,target,query,**kw):
            calls.append(1);ck=kw['checkpoint'];epochs=kw['epochs']
            if ck:ck.parent.mkdir(parents=True,exist_ok=True);ck.write_bytes(b'fake synthetic head')
            return {'records':[{'epoch':ep,'predictions':np.full(len(query),target.mean()).tolist(),
                'train_sse':float(((target-target.mean())**2).sum()),'train_n':len(ctx)} for ep in epochs],
                'audit':{'train_indices':ctx.tolist(),'query_indices':query.tolist(),'epochs':list(epochs),
                    'optimizer_steps':math.ceil(len(ctx)/eng.BATCH)*max(epochs),'encoder_updates':0,
                    'query_labels_accepted':False,'parameter_delta_l2':float(max(epochs)>0),'max_gradient_norm':float(max(epochs)>0)},
                'checkpoint_path':str(ck) if ck else None}
        def wrapped(*a,**kw):return actual_fold(*a,**kw,run=head_run)
        with tempfile.TemporaryDirectory(dir=exp.ROOT/'.task-tmp',prefix='spatial-results-') as tmp:
            state=(data,large,None,{42:ref},plan)
            with patch.object(exp,'OUT',Path(tmp)),patch.object(exp,'validate',return_value=state),\
                 patch.object(exp,'load_tokens',return_value=(tokens,aux,{'synthetic':True})),\
                 patch.object(exp.prior,'runtime',return_value=(torch,torch.device('cpu'),{})),\
                 patch.object(exp,'checked_runtime',return_value={}),patch.object(exp,'fit_fold',side_effect=wrapped),\
                 patch.object(exp,'inner_large',side_effect=lambda x,y,q,seed:{'predictions':np.full(len(q),y.mean()).tolist()}),\
                 contextlib.redirect_stdout(io.StringIO()):
                first=exp.evaluate(42)
                self.assertEqual(len(calls),40)
                second=exp.evaluate(42,allow_fit=False)
                self.assertEqual(len(calls),40);self.assertEqual(first,second)
                self.assertTrue(first['ridge_anchor_matched']);self.assertIn(exp.PRIMARY,first['metrics'])
                self.assertEqual(first['diagnosis'][exp.PRIMARY]['groups']['common16']['n'],16)
                self.assertEqual(first['diagnosis'][exp.PRIMARY]['groups']['remaining484']['n'],484)
                self.assertTrue((Path(tmp)/'seed42/training_curves.csv').exists())
                p=Path(tmp)/'seed42/oof.csv';p.write_text(p.read_text()+'tampered',encoding='utf-8')
                with self.assertRaises(ValueError):exp.evaluate(42,allow_fit=False)


if __name__=='__main__':unittest.main()
