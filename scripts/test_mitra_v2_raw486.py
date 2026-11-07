"""Synthetic unit coverage: payload boundaries, fixed mixture and cache integrity."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import mitra_v2_raw486_experiment as exp

TMP_ROOT=(exp.ROOT/'.task-tmp/raw486-unit-tests').resolve()
TMP_ROOT.mkdir(parents=True,exist_ok=True)


def workspace_temp():
    temp=tempfile.TemporaryDirectory(dir=TMP_ROOT)
    assert Path(temp.name).resolve().is_relative_to(TMP_ROOT)
    return temp


def audit():
    return {'completed':True,'optimizer_steps':50,'epochs_observed':50,
            'configured_epochs':50,'warmup_steps':10,'weight_decay':.3,
            'train_rows':350,'val_rows':50,'width':486,
            'max_samples_support_before':20480,'max_samples_support_after':20480,
            'precision':'bfloat16','lr':1e-5,'preprocessor_dim_embedding':None,
            'preprocessor_singular_columns':3,'transformed_width':483}


class Raw486Tests(unittest.TestCase):
    def test_audit_rejects_compression_partial_bags_and_context_reduction(self):
        good=[audit() for _ in range(8)]
        exp.worker.validate_audits(good,50,8)
        for change in ({'width':256},{'optimizer_steps':49},{'max_samples_support_after':10240},
                       {'preprocessor_dim_embedding':256},{'transformed_width':256}):
            bad=copy.deepcopy(good);bad[-1].update(change)
            with self.assertRaises(ValueError):exp.worker.validate_audits(bad,50,8)
        with self.assertRaises(ValueError):exp.worker.validate_audits(good[:7],50,7)

    def test_payload_is_exact_and_refuses_changed_query(self):
        rng=np.random.default_rng(332);x=rng.normal(size=(400,486));q=rng.normal(size=(100,486));y=rng.normal(size=400)
        with workspace_temp() as tmp:
            folder=Path(tmp);scope={'fixture':'raw'}
            exp.raw_cell(folder,scope,x,y,q,True)
            with np.load(folder/'raw486.npz',allow_pickle=False) as a:
                self.assertEqual(set(a.files),{'train_x','train_y','query_x'})
                np.testing.assert_array_equal(a['train_x'],x)
                np.testing.assert_array_equal(a['query_x'],q)
            exp.raw_cell(folder,scope,x,y,q,False)
            with self.assertRaises(ValueError):exp.raw_cell(folder,scope,x,y,q+1,False)
            with self.assertRaises(ValueError):exp.raw_cell(folder,scope,x[:,:256],y,q,True)
            with self.assertRaises(FileNotFoundError):exp.raw_cell(folder/'missing',scope,x,y,q,False)

    def test_full_evaluation_and_resume_reconstruct_without_fit(self):
        rng=np.random.default_rng(82);x=rng.normal(size=(500,486));y=rng.normal(size=500)
        data=SimpleNamespace(y=y,ids=np.array([f'SYN_{i:06}' for i in range(500)]),noise=rng.uniform(size=500))
        b,l,c,t=rng.normal(size=(4,500));folds=np.arange(500)%5
        ref={'base':b,'large':l,'radio':c,'tab486':t,'folds':folds,'current':.25*(b+l+c+t)}
        plan={'frozen':{'common_failure_ids':data.ids[:16].tolist(),'upstream':{'mitra_environment':{}}}}
        state=(data,x,{42:ref},plan);calls=[]
        def fake_worker(inp,out,seed):
            fold=(seed-42-9)//1009;mask=folds!=fold
            with np.load(inp,allow_pickle=False) as a:
                self.assertEqual(set(a.files),{'train_x','train_y','query_x'})
                np.testing.assert_array_equal(a['train_x'],x[mask]);np.testing.assert_array_equal(a['train_y'],y[mask])
                np.testing.assert_array_equal(a['query_x'],x[~mask]);pred=np.full(100,y[mask].mean())
            calls.append(fold)
            r={'predictions':pred.tolist(),'synthetic':False,'steps_tested':50,'recipe':exp.worker.RECIPE,
               'audit':[audit() for _ in range(8)],'bag_members':8,'raw486_retained':True,
               'worker_sha256':exp.assets.sha(exp.worker.__file__),'assets':exp.assets.spec(),
               'memory_runtime':{'policy':exp.memory.POLICY,'sha256':exp.assets.sha(exp.memory.__file__)},
               'memory_guard':{'policy':exp.guard.POLICY,'events':[]},'math_check_sha256':exp.assets.sha(exp.MATH),
               'peak_allocated_bytes':0,'total_vram':2**33,'environment':{},'seed':seed,
               'input_sha256':exp.assets.sha(inp),'competition_fits':8,'competition_queries':100,'seconds':0}
            exp.e.write_json(out,r);return r
        with workspace_temp() as tmp, patch.object(exp,'run_worker',fake_worker):
            folder=Path(tmp);s=exp.evaluate(folder,42,True,state)
            self.assertEqual(calls,[0,1,2,3,4]);s2=exp.evaluate(folder,42,False,state)
            self.assertEqual(s,s2);self.assertEqual(calls,[0,1,2,3,4])
            p,_=exp.collect(folder/'seed42',data,x,ref,{'plan':exp.e.digest(plan['frozen']),'seed':42},42,False,{})
            np.testing.assert_allclose(p[exp.PRIMARY]-ref['current'],.25*(p['mitra_raw486']-t),rtol=0,atol=1e-14)
            with self.assertRaises(FileNotFoundError):exp.evaluate(folder/'missing',42,False,state)
            # One corrupt completed artifact must fail, never silently retrain.
            with (folder/'seed42/oof.csv').open('a') as f:f.write('\ncorrupt')
            with self.assertRaises(ValueError):exp.evaluate(folder,42,True,state)
            self.assertEqual(calls,[0,1,2,3,4])


if __name__=='__main__':unittest.main()
