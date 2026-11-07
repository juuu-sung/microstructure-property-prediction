"""Synthetic-only checks: pooling, protocol, context isolation and safe resume."""
import inspect
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch

from scripts import dinov3_large_engine as eng
from scripts import dinov3_large_experiment as exp


class Dino3Tests(unittest.TestCase):
    def test_meta_architecture_matches_public_header(self):
        # No weights or competition images loaded; only tensor shapes.
        m=eng.architecture('meta')
        path=eng.assets.ASSETS/'safetensors_header.json'
        self.assertEqual(eng.assets.sha256(path),eng.assets.spec()['header_sha256'])
        h=json.loads(path.read_text());h.pop('__metadata__',None)
        self.assertEqual({k:list(v.shape) for k,v in m.state_dict().items()},
                         {k:v['shape'] for k,v in h.items()})

    def test_register_exclusion_and_population_std(self):
        x=torch.zeros(1,1374,1024)
        x[:,0]=2;x[:,1:5]=10000
        x[:,5:]=torch.arange(1369)[:,None]
        actual=eng.pool(x)
        self.assertEqual(list(actual.shape),[1,3072])
        torch.testing.assert_close(actual[:,:1024],torch.full((1,1024),2.))
        torch.testing.assert_close(actual[:,1024:2048],torch.full((1,1024),684.))
        torch.testing.assert_close(actual[:,2048:],x[:,5:].std(1,correction=0))
        with self.assertRaises(ValueError):eng.pool(x[:,:1370])

    def test_transform_preserves_full_constant_image(self):
        x=eng.transform(torch.ones(1,1,256,256)*.5)
        self.assertEqual(tuple(x.shape),(1,3,592,592))
        expected=(torch.tensor([.5,.5,.5])-torch.tensor([.485,.456,.406]))/torch.tensor([.229,.224,.225])
        torch.testing.assert_close(x[0,:,250,250],expected,rtol=1e-5,atol=1e-6)
        with self.assertRaises(ValueError):eng.transform(torch.ones(1,1,256,256)*2)

    def test_fixed_formula_and_exact_tie(self):
        s,c,l,lc,d=[np.full(4,n) for n in (10,20,30,40,50)]
        p=eng.compose(s,c,l,lc,d)
        np.testing.assert_array_equal(p['current'],17.5)
        np.testing.assert_array_equal(p['matched_control'],20.)
        np.testing.assert_array_equal(p[exp.PRIMARY],22.5)
        out=eng.choose({str(i):[0.,0.] for i in eng.INDICES},np.zeros(2),np.zeros(2))
        self.assertEqual(out['blend']['alpha'],10000)

    def test_context_only_fit_resume_and_tamper(self):
        rng=np.random.default_rng(123)
        y=rng.normal(size=100);x=rng.normal(size=(100,6));q=rng.normal(size=(11,6))
        seen=[]
        def spy(tx,ty,qx,indices):
            seen.append((len(tx),len(qx)))
            self.assertEqual(len(tx),len(ty))
            return eng.ridge.fit_candidates(tx,ty,qx,indices)
        self.assertNotIn('query_y',inspect.signature(exp.fit_fold).parameters)
        with tempfile.TemporaryDirectory(dir=exp.ROOT/'reports') as td:
            folder=Path(td)
            args=(folder,{'synthetic':True},{'dinov2':x,'dinov3':x},y,
                {'dinov2':q,'dinov3':q},np.zeros(100),42)
            with exp.threadpool_limits(limits=2):
                p,c=exp.fit_fold(*args,runner=spy)
            self.assertEqual(len(seen),8)
            self.assertEqual(sum(a==100 and b==11 for a,b in seen),2)
            self.assertTrue(all(a+b==100 for a,b in seen if b!=11))
            p2,c2=exp.fit_fold(*args,allow_fit=False,runner=exp.forbidden)
            self.assertEqual(c,c2)
            for k in p:np.testing.assert_array_equal(p[k],p2[k])
            # Changing the context invalidates cache rather than silently reusing it.
            altered=x.copy();altered[0,0]+=1
            with self.assertRaises(ValueError):
                exp.fit_fold(folder,{'synthetic':True},{'dinov2':altered,'dinov3':x},y,
                    {'dinov2':q,'dinov3':q},np.zeros(100),42,False,exp.forbidden)

    def test_query_statistics_are_not_fit(self):
        rng=np.random.default_rng(1);x=rng.normal(size=(100,6));y=rng.normal(size=100)
        q=rng.normal(size=(9,6))
        with exp.threadpool_limits(limits=2):
            a=eng.ridge.fit_candidates(x,y,q,[1])
            b=eng.ridge.fit_candidates(x,y,np.vstack([q,np.full((1,6),1e8)]),[1])
        np.testing.assert_allclose(a['predictions']['1'],b['predictions']['1'][:9],rtol=0,atol=1e-10)
        np.testing.assert_allclose(a['audit']['x_mean'],x.mean(0))

    def test_reporting_and_aggregate_refer_to_new_current(self):
        n=500;ids=np.array([f'TRAIN_{i:06d}' for i in range(n)])
        y=np.linspace(140,250,n);ref={'current':y+3,'folds':np.arange(n)%5}
        data=SimpleNamespace(ids=ids,y=y,noise=np.arange(n,dtype=float))
        plan={'frozen':{'common_failure_ids':ids[:16].tolist()}}
        preds={'current':y+3,'matched_control':y+2,'dinov2_retuned_ridge':y+4,
               'dinov3_ridge':y+1.5,exp.PRIMARY:y+1}
        summaries=[]
        for seed in exp.SEEDS:
            s,files=exp.result_data(data,ref,preds,[],plan,seed,{'test':True})
            self.assertEqual(s['primary'],exp.PRIMARY)
            self.assertAlmostEqual(s['metrics'][exp.PRIMARY]['gain_vs_current'],2)
            self.assertAlmostEqual(s['metrics'][exp.PRIMARY]['gain_vs_control'],1)
            self.assertEqual(s['diagnosis'][exp.PRIMARY]['common16_improved'],16)
            self.assertEqual(len(files['common16.csv']),16)
            self.assertIn(exp.PRIMARY,files['oof.csv'][0])
            self.assertNotIn(exp.diagnostics.PRIMARY,files['oof.csv'][0])
            summaries.append(s)
        r=exp.aggregate(summaries)
        self.assertTrue(r['all_seeds_improve_both'])
        self.assertTrue(r['passes_numeric_expansion_priority'])
        self.assertFalse(r['automatic_adoption'])
        self.assertIn('common16: ',exp.report_text(summaries[0]))

    def test_feature_cache_rejects_id_permutation(self):
        with tempfile.TemporaryDirectory(dir=exp.ROOT/'reports') as td:
            path=Path(td)/'features.npz';ids=np.array(['a','b'])
            exp.e.npz_save(path,ids,np.zeros((2,3072),dtype=np.float32),{'test':1})
            with self.assertRaises(ValueError):exp.e.npz_read(path,ids[::-1],3072,{'test':1})


if __name__=='__main__':unittest.main()
