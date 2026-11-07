"""Synthetic arrays and mocked predictor only; no competition fits or GPU use."""
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from scripts import evaluate_tabpfn_v2 as t


class FakePredictor:
    def __init__(self): self.calls=[]
    def __call__(self,x,y,q,seed):
        self.calls.append((x.copy(),y.copy(),q.copy(),seed))
        return q[:,1]+(0 if x.shape[1]==185 else .5), {"seconds":0.,"context_mean":float(y.mean())}


def data_fixture():
    rng=np.random.default_rng(15)
    x=rng.normal(size=(500,285))
    x[:,0]=np.arange(500)
    y=x[:,1].copy()
    return {"existing185":x[:,:185].copy(),"morphology285":x.copy()},y,np.repeat(np.arange(5),100)


class TabPFNTests(unittest.TestCase):
    def test_fold_isolates_outer_labels_and_selects_only_inner(self):
        features,y,folds=data_fixture()
        a,b=FakePredictor(),FakePredictor()
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            i,p,c=t.evaluate_fold(features,y,folds,42,0,root/'a',{"fixed":True},a)
            y2=y.copy();y2[:100]+=99999
            j,q,d=t.evaluate_fold(features,y2,folds,42,0,root/'b',{"fixed":True},b)
            self.assertEqual(len(a.calls),8)
            self.assertEqual(c,d)
            self.assertEqual(c['representation'],'existing185')
            for name in p: np.testing.assert_array_equal(p[name],q[name])
            for old,new in zip(a.calls,b.calls):
                for k in range(3): np.testing.assert_array_equal(old[k],new[k])
                np.testing.assert_array_equal(old[1],y[old[0][:,0].astype(int)])
                self.assertFalse(set(old[0][:,0])&set(old[2][:,0]))
                self.assertTrue(set(old[0][:,0]).isdisjoint(range(100)))
                self.assertEqual(old[3],new[3])
            self.assertEqual([len(x[0]) for x in a.calls[:3]],[266,267,267])
            self.assertEqual(len(a.calls[-1][0]),400)
            self.assertEqual([x[3] for x in a.calls[:3]],[x[3] for x in a.calls[3:6]])
            # Completed context cells must not refit on rerun.
            t.evaluate_fold(features,y,folds,42,0,root/'a',{"fixed":True},a)
            self.assertEqual(len(a.calls),8)

    def test_pooled_selection_ties_prefer185(self):
        y=np.arange(11.)
        self.assertEqual(t.select_representation({n:y.copy() for n in t.REPS},y)[0],'existing185')
        self.assertEqual(t.select_representation({'existing185':y+1,'morphology285':y},y)[0],'morphology285')
        with self.assertRaises(ValueError):
            t.select_representation({'existing185':y*float('nan'),'morphology285':y},y)

    def test_cell_rejects_changed_inputs_settings_or_corruption(self):
        features,y,_=data_fixture();x=features['existing185']
        ti,vi=np.arange(40),np.arange(40,50)
        fake=FakePredictor()
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)
            a=t.run_cell(path,{'device':'cpu'},'existing185',x,y[ti],ti,vi,7,fake)
            b=t.run_cell(path,{'device':'cpu'},'existing185',x,y[ti],ti,vi,7,fake)
            np.testing.assert_array_equal(a,b);self.assertEqual(len(fake.calls),1)
            with self.assertRaises(ValueError):t.run_cell(path,{'device':'cuda'},'existing185',x,y[ti],ti,vi,7,fake)
            with self.assertRaises(ValueError):t.run_cell(path,{'device':'cpu'},'existing185',x,y[ti]+1,ti,vi,7,fake)
            changed=x.copy();changed[45,2]+=1
            with self.assertRaises(ValueError):t.run_cell(path,{'device':'cpu'},'existing185',changed,y[ti],ti,vi,7,fake)
            record=t.read_json(path/'completed.json');record['result']['prediction'][0]+=1;t.write_json(path/'completed.json',record)
            with self.assertRaises(ValueError):t.run_cell(path,{'device':'cpu'},'existing185',x,y[ti],ti,vi,7,fake)

    def test_overlap_and_nonfinite_do_not_create_completed_output(self):
        features,y,_=data_fixture();x=features['existing185']
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)
            with self.assertRaises(ValueError):t.run_cell(path,{},'existing185',x,y[:10],np.arange(10),np.arange(8,12),0,FakePredictor())
            self.assertFalse((path/'completed.json').exists())
            with self.assertRaises(ValueError):
                t.run_cell(path,{},'existing185',x,y[:10],np.arange(10),np.arange(10,20),0,lambda *a:(np.full(10,np.nan),{'seconds':0.}))
            self.assertFalse((path/'completed.json').exists())

    def test_report_and_comparison_use_fixed_half_blend(self):
        features,y,folds=data_fixture()
        data=SimpleNamespace(ids=np.array([f'TRAIN_{i:06d}' for i in range(500)]),y=y,noise=np.arange(500.))
        refs={'current':y+2,'folds':folds,'controls':{n:y+2 for n in ('ridge_existing','ridge_morphology','catboost_existing','catboost_morphology')}}
        predictions={n:y+1 for n in (*t.REPS,t.PRIMARY)}
        plan={'frozen':{'synthetic':True}};binding={'plan':t.digest(plan['frozen'])}
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            for seed in t.SEEDS:
                path=root/f'seed{seed}';path.mkdir()
                t.summarize(path,data,refs,predictions,[],binding,seed)
            with patch.object(t,'validate',return_value=(data,features,{s:refs for s in t.SEEDS},plan)):
                t.compare(root)
                result=t.read_json(root/'comparison.json')
                self.assertTrue(result['passes_numeric_priority_gate'])
                for row in result['per_seed']:
                    self.assertAlmostEqual(row['primary_gain'],1.)
                    self.assertAlmostEqual(row['secondary_gain'],.5)
                with (root/'seed42/oof.csv').open('a') as f:f.write('corrupt')
                with self.assertRaises(ValueError):t.compare(root)

    def test_local_v2_settings_and_telemetry_disabled(self):
        import torch
        from tabpfn_common_utils.telemetry.core.service import ProductTelemetry
        self.assertEqual(os.environ['HF_HUB_OFFLINE'],'1')
        self.assertFalse(ProductTelemetry().telemetry_enabled())
        model=t.make_model(Path('/synthetic/tabpfn-v2-regressor.ckpt'),torch.device('cpu'),42)
        self.assertEqual(model.n_estimators,8)
        self.assertEqual(model.inference_precision,torch.float32)
        self.assertEqual(model.model_path,Path('/synthetic/tabpfn-v2-regressor.ckpt'))
        self.assertEqual(model.random_state,42)
        self.assertEqual(model.inference_config,{'MIN_UNIQUE_FOR_NUMERICAL_FEATURES':0})
        self.assertFalse(model.differentiable_input)

    def test_cuda_device_resolution(self):
        import torch
        with patch.object(torch.cuda,'is_available',return_value=True),patch.object(torch.cuda,'current_device',return_value=0), \
             patch.object(torch.cuda,'device_count',return_value=1),patch.object(torch.cuda,'set_device') as set_device, \
             patch.object(torch,'manual_seed'),patch.object(torch.cuda,'manual_seed_all'):
            _,d=t.configure_device('cuda',0)
            self.assertEqual(d,torch.device('cuda:0'))
            set_device.assert_called_once_with(d)


if __name__=='__main__':unittest.main()
