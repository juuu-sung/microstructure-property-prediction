"""Synthetic-only residual scope, nested selection, cache and report tests."""
from contextlib import contextmanager, redirect_stdout
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from threadpoolctl import threadpool_limits
from scripts import residual_correction_experiment as e
from scripts import residual_correction_engine as eng


@contextmanager
def temporary():
    parent = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory(dir=parent, prefix='.residual_test_') as name:
        yield Path(name)


def fixture(n=500):
    rng=np.random.default_rng(333)
    raw={name:rng.normal(size=(n,24)) for name in eng.COMPONENTS}
    for x in raw.values():x[:,0]=np.arange(n)
    y=190+5*raw['tabpfn486'][:,1]+rng.normal(size=n)*5
    p=np.column_stack([y+3+rng.normal(size=n)*4 for _ in range(4)])
    data=SimpleNamespace(ids=np.array([f'SYNTH_{j}' for j in range(n)]),y=y,
                         noise=np.abs(raw['tabpfn486'][:,2]))
    ref={'folds':np.arange(n)%5,'components':p,'current':eng.ensemble(p),'runtime':{'synthetic':True}}
    plan={'frozen':{'common_failure_ids':list(data.ids[:16]),'synthetic':True}}
    return data,raw,ref,plan


def fake_base(tx,y,qx,seed,predictor):
    p=185+.02*np.mean(y)+qx['tabpfn486'][:,1]
    return np.column_stack([p+k*.2 for k in range(4)]),{'synthetic':True}


class ResidualTests(unittest.TestCase):
    def setUp(self):
        self.limits=threadpool_limits(limits=2);self.limits.__enter__()

    def tearDown(self):self.limits.__exit__(None,None,None)

    def test_residual_sign_and_zero_correction(self):
        p=np.array([100.,200.])
        z=eng.compose(p,{'tree_delta':np.array([6.,-8.]),'calibration_delta':np.zeros(2)})
        np.testing.assert_array_equal(z['residual_corrected'],[103,196])
        np.testing.assert_array_equal(z['calibration_control'],p)
        np.testing.assert_array_equal(eng.ensemble(np.tile([100,200,300,400],(2,1))),[312.5,312.5])

    def test_base_cv_and_tabpfn_fit_never_accept_query_labels(self):
        data,raw,_,_=fixture(120)
        calls=[]; original=e.core.make_estimator
        class Spy:
            def __init__(self,alpha):self.model=original(alpha)
            def fit(self,x,y):
                calls.append(x[:,0].copy())
                self.model.fit(x,y);return self
            def predict(self,x):return self.model.predict(x)
        contexts=[]
        def pred(x,y,q,seed):
            contexts.append((x[:,0].copy(),q[:,0].copy(),y.copy()))
            return np.full(len(q),y.mean()),{'synthetic':True}
        with patch.object(e.core,'make_estimator',side_effect=Spy):
            values,detail=eng.fit_base({n:x[:90] for n,x in raw.items()},data.y[:90],
                                       {n:x[90:] for n,x in raw.items()},99,pred)
        self.assertEqual(values.shape,(30,4))
        self.assertEqual(len(calls),39)
        self.assertEqual(sum(len(c)==90 for c in calls),3)
        self.assertTrue(all(set(c).issubset(range(90)) for c in calls))
        self.assertTrue(all(len(c)==60 for c in calls if len(c)!=90))
        np.testing.assert_array_equal(contexts[0][2],data.y[:90])
        self.assertFalse(set(contexts[0][0])&set(contexts[0][1]))
        self.assertEqual(set(detail['ridge_selections']),set(eng.GRIDS))

    def test_crossfit_owns_each_row_once_and_excludes_outer_query(self):
        data,raw,ref,plan=fixture();ti=np.flatnonzero(ref['folds']!=0)
        observed=[]
        def spy(tx,y,qx,seed,predictor):
            a=tx['hand185'][:,0].astype(int);b=qx['hand185'][:,0].astype(int)
            self.assertFalse(set(a)&set(b));self.assertEqual(set(a)|set(b),set(ti))
            np.testing.assert_array_equal(y,data.y[a]);observed.extend(b.tolist())
            return fake_base(tx,y,qx,seed,predictor)
        with temporary() as folder,patch.object(eng,'fit_base',side_effect=spy),redirect_stdout(io.StringIO()):
            out=e.crossfit_components(folder,{'synthetic':True},raw,ti,data.y[ti],42,0,None,True)
        self.assertEqual(out.shape,(400,4));self.assertEqual(sorted(observed),ti.tolist())

    def test_cpu_limits_restore_tabpfn_torch_threads(self):
        import torch
        # Exit this test's surrounding CPU limiter before emulating TabPFN4.
        prior=torch.get_num_threads();torch.set_num_threads(4)
        try:
            data,raw,_,_=fixture(120)
            def predictor(x,y,q,seed):
                self.assertEqual(torch.get_num_threads(),4)
                return np.full(len(q),y.mean()),{'synthetic':True}
            eng.fit_base({n:x[:90] for n,x in raw.items()},data.y[:90],
                          {n:x[90:] for n,x in raw.items()},9,predictor)
            self.assertEqual(torch.get_num_threads(),4)
        finally:torch.set_num_threads(prior)

    def test_outer_label_poisoning_cannot_change_corrections(self):
        data,raw,ref,plan=fixture();ti=np.flatnonzero(ref['folds']!=0);vi=np.flatnonzero(ref['folds']==0)
        changed=data.y.copy();changed[vi]+=1e8
        with temporary() as folder,patch.object(eng,'fit_base',side_effect=fake_base),redirect_stdout(io.StringIO()):
            a,_=e.fold_cell(folder/'a',{},raw,ti,data.y[ti],data.noise[ti],vi,data.noise[vi],ref['components'][vi],42,0,None,True)
            b,_=e.fold_cell(folder/'b',{},raw,ti,changed[ti],data.noise[ti],vi,data.noise[vi],ref['components'][vi],42,0,None,True)
        for n in a:np.testing.assert_array_equal(a[n],b[n])

    def test_query_features_do_not_fit_projection_or_tree(self):
        data,raw,ref,_=fixture();x=raw['tabpfn486'];p=ref['components']
        a,info=eng.fit_corrections(x[:400],data.noise[:400],p[:400],data.y[:400],x[400:],data.noise[400:],p[400:],7)
        q=x[400:].copy();q[1:]+=1e8
        b,other=eng.fit_corrections(x[:400],data.noise[:400],p[:400],data.y[:400],q,data.noise[400:],p[400:],7)
        self.assertEqual(info,other)
        for n in a:self.assertEqual(a[n][0],b[n][0])
        params=eng.fit_projection(x[:400]);np.testing.assert_allclose(params['mean'],x[:400].mean(0))
        self.assertEqual(eng.meta_features(p[:400],data.noise[:400],eng.project(x[:400],params)).shape,(400,26))

    def test_cache_reuse_rejects_scope_or_payload_mutation(self):
        with temporary() as folder:
            path=folder/'completed.json'
            value=e.checked_cell(path,{'scope':1},lambda:{'x':[1,2]},True)
            self.assertEqual(e.checked_cell(path,{'scope':1},e.unavailable,False),value)
            with self.assertRaises(ValueError):e.checked_cell(path,{'scope':2},e.unavailable,False)
            saved=e.t.read_json(path);saved['result']['x'][0]=999;e.t.write_json(path,saved)
            with self.assertRaises(ValueError):e.checked_cell(path,{'scope':1},e.unavailable,False)

    def test_changed_crossfit_inputs_fail_without_refitting(self):
        data,raw,ref,_=fixture();ti=np.flatnonzero(ref['folds']!=0)
        with temporary() as folder,patch.object(eng,'fit_base',side_effect=fake_base),redirect_stdout(io.StringIO()):
            e.crossfit_components(folder,{},raw,ti,data.y[ti],42,0,None,True)
            y=data.y[ti].copy();y+=10
            with self.assertRaises(ValueError):e.crossfit_components(folder,{},raw,ti,y,42,0,None,False)
            raw['hand185'][ti,1]+=1
            with self.assertRaises(ValueError):e.crossfit_components(folder,{},raw,ti,data.y[ti],42,0,None,False)

    def test_missing_cache_and_overlap_fail_closed(self):
        data,raw,ref,_=fixture();ti=np.arange(100);vi=np.arange(80,120)
        with temporary() as folder:
            with self.assertRaises(ValueError):e.checked_cell(folder/'none.json',{},e.unavailable,False)
            with self.assertRaises(ValueError):e.fold_cell(folder,{},raw,ti,data.y[ti],data.noise[ti],vi,data.noise[vi],ref['components'][vi],42,0,None,True)

    def test_saved_pipeline_compare_uses_no_fitting(self):
        data,raw,ref,plan=fixture();refs={s:ref for s in e.SEEDS}
        with temporary() as folder,redirect_stdout(io.StringIO()):
            for seed in e.SEEDS:
                sub=folder/f'seed{seed}';sub.mkdir()
                with patch.object(eng,'fit_base',side_effect=fake_base):
                    p,d,rows=e.collect(sub,data,raw,ref,plan,seed,None,True)
                e.save_results(sub,data,ref,p,d,rows,plan,seed)
                self.assertTrue(e.completed(sub,e.binding_for(plan,seed,ref)))
            with patch.object(e,'validate',return_value=(data,raw,refs,plan)), \
                 patch.object(eng,'fit_base',side_effect=e.unavailable), \
                 patch.object(eng,'fit_corrections',side_effect=e.unavailable):
                e.compare(folder)
            result=e.t.read_json(folder/'comparison.json')
            self.assertEqual(len(result['per_seed']),2)
            self.assertEqual(set(result['per_seed'][0]['diagnosis']['group_rmse']),
                {'common16','remaining484','low10','high10','middle80',*[f'noiseQ{i}' for i in range(1,5)],*[f'fold{i}' for i in range(5)]})
            (folder/'seed42/diagnostics.csv').write_text('tampered',encoding='utf-8')
            with self.assertRaises(ValueError):e.completed(folder/'seed42',e.binding_for(plan,42,ref))

    def test_driver_stops_on_failure_and_does_not_touch_submission(self):
        text=(e.ROOT/'scripts/run_residual_correction.ps1').read_text(encoding='utf-8')
        self.assertIn('$LASTEXITCODE -ne 0',text)
        self.assertEqual(text.count("'evaluate'"),2)
        self.assertNotIn('final_submission',text)


if __name__=='__main__':unittest.main()
