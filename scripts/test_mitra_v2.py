"""Synthetic unit checks only. No competition fitting or GPU work."""
import tempfile
from pathlib import Path
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import mitra_v2_engine as eng
import mitra_v2_worker as worker


class MitraProtocolTests(unittest.TestCase):
    def test_exact_quarter_replacement(self):
        rng=np.random.default_rng(71)
        b,l,c,t,u,m=rng.normal(size=(6,500))
        p=eng.compose(b,l,c,t,u,m)
        old=.5*(.5*b+.5*t)+.25*l+.25*c
        np.testing.assert_allclose(p['current'],old,atol=1e-14)
        np.testing.assert_allclose(p['mitra_replaced_blend']-p['current'],.25*(m-t),atol=1e-14)
        np.testing.assert_allclose(p['tabpfn_svd256_control_blend']-p['current'],.25*(u-t),atol=1e-14)

    def test_projection_ignores_query_distribution(self):
        rng=np.random.default_rng(9)
        x=rng.normal(size=(400,486));q=rng.normal(size=(100,486))
        z,v,m=eng.project(x,q)
        z2,v2,m2=eng.project(x,q*10+90)
        np.testing.assert_array_equal(z,z2)
        self.assertEqual(m,m2)
        self.assertFalse(np.allclose(v,v2))
        np.testing.assert_allclose(np.mean(z,axis=0),0,atol=1e-12)

    def test_complete_bags_and_steps_required(self):
        good={'completed':True,'optimizer_steps':50,'epochs_observed':50,
            'train_rows':350,'val_rows':50,'width':256,
            'max_samples_support_before':20480,'max_samples_support_after':20480,
            'precision':'bfloat16','lr':1e-5}
        rows=[good.copy() for _ in range(8)]
        worker.validate_audits(rows,50,8)
        for bad in ({'optimizer_steps':49},{'completed':False},{'width':128},
                    {'max_samples_support_after':10240},{'lr':1e-4}):
            broken=[good.copy() for _ in range(8)];broken[-1].update(bad)
            with self.assertRaises(ValueError):worker.validate_audits(broken,50,8)
        with self.assertRaises(ValueError):worker.validate_audits(rows[:7],50,7)

    def test_invalid_shape_and_nonfinite_rejected(self):
        with self.assertRaises(ValueError):eng.finite([1,float('nan')])
        with self.assertRaises(ValueError):eng.project(np.zeros((400,486)),np.zeros((101,486)))

    def test_fold_payload_and_resume_without_fitting(self):
        import mitra_v2_experiment as exp
        rng=np.random.default_rng(81)
        x=rng.normal(size=(500,486));y=rng.normal(size=500)
        data=SimpleNamespace(y=y)
        b,l,c,t=rng.normal(size=(4,500))
        folds=np.arange(500)%5
        ref={'base':b,'large':l,'radio':c,'tab486':t,'folds':folds,
             'current':eng.compose(b,l,c,t,t,t)['current']}
        calls=[]
        def fake_tab(inp,out,seed):
            with np.load(inp,allow_pickle=False) as a:
                self.assertEqual(set(a.files),{'train_x','train_y','query_x'})
                pred=np.full(100,a['train_y'].mean())
            calls.append(('tab',seed))
            return {'predictions':pred.tolist(),'runtime':{'synthetic_unit_stub':True}}
        def fake_worker(inp,out,seed):
            with np.load(inp,allow_pickle=False) as a:
                self.assertEqual(set(a.files),{'train_x','train_y','query_x'})
                fold=(seed-42-9)//1009
                np.testing.assert_array_equal(a['train_y'],y[folds!=fold])
                self.assertEqual(a['train_x'].shape,(400,256))
                pred=np.full(100,a['train_y'].mean()+.1)
            calls.append(('mitra',seed))
            audit={'completed':True,'optimizer_steps':50,'epochs_observed':50,
                'train_rows':350,'val_rows':50,'width':256,
                'max_samples_support_before':20480,'max_samples_support_after':20480,
                'precision':'bfloat16','lr':1e-5}
            r={'predictions':pred.tolist(),'audit':[audit.copy() for _ in range(8)],'bag_members':8,
               'synthetic':False,'steps_tested':50,'recipe':worker.RECIPE,'input_sha256':worker.assets.sha(inp),
               'environment':{},'worker_sha256':worker.assets.sha(worker.__file__),
               'assets':worker.assets.spec(),'seed':seed,'seconds':0,'peak_allocated_bytes':0}
            exp.e.write_json(out,r)
            return r
        with tempfile.TemporaryDirectory(dir=worker.ROOT/'.task-tmp/mitra') as tmp:
            folder=Path(tmp)
            with (patch.object(exp,'run_tabpfn',fake_tab),
                  patch.object(exp,'run_worker',fake_worker)):
                first,a=exp.collect(folder,data,x,ref,{'test':'synthetic'},42,True,{})
                self.assertEqual(len(calls),10)
                second,a2=exp.collect(folder,data,x,ref,{'test':'synthetic'},42,False,{})
                self.assertEqual(len(calls),10)
                for name in first:np.testing.assert_array_equal(first[name],second[name])
                self.assertEqual(a,a2)
                with self.assertRaises(ValueError):
                    exp.collect(folder,data,x,ref,{'test':'changed'},42,False,{})


if __name__=='__main__':unittest.main()
