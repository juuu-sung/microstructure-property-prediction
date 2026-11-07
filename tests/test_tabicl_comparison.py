"""Synthetic CPU-only checks. No competition fitting or neural forwards."""
from contextlib import contextmanager, redirect_stdout
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from scripts import evaluate_tabicl_comparison as e


@contextmanager
def temporary():
    parent = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory(dir=parent,prefix='.tabicl_test_') as name:
        if Path(name).resolve().parent != parent:
            raise ValueError('Cleanup outside tests')
        yield Path(name)


def fixture():
    rng = np.random.default_rng(27)
    raw = rng.normal(size=(500,486)); raw[:,0]=np.arange(500); raw[:,185]=0
    data = SimpleNamespace(ids=np.array([f'TRAIN_{i:06d}' for i in range(500)]),
        y=100+3*raw[:,1],noise=np.arange(500.))
    base = data.y+4+raw[:,2]
    ref = {'folds':np.repeat(np.arange(5),100),'base':base,'tabpfn486':data.y+2,
        'current486':.5*(base+data.y+2),'runtime':{'synthetic':True}}
    plan = {'frozen':{'common_failure_ids':list(data.ids[:16]),'synthetic':True}}
    return data,raw,ref,plan


class Fake:
    calls = []
    def __init__(self,name): self.name=name
    def __call__(self,x,y,q,seed):
        self.calls.append({'name':self.name,'train':x.copy(),'query':q.copy(),'y':y.copy(),'seed':seed})
        return y.mean()+.03*q[:,1], {'seconds':0.}


def predictors(*args): return {n:Fake(n) for n in e.REPS}


class TabICLTests(unittest.TestCase):
    def setUp(self): Fake.calls=[]

    def test_block_pca_matches_sklearn_and_query_independent(self):
        _,raw,_,_=fixture()
        params=e.fit_pca(raw[100:]);e.validate_pca(params,400)
        expected=[]
        with threadpool_limits(limits=2):
            for name,start,stop in e.BLOCKS:
                scaler=StandardScaler().fit(raw[100:,start:stop])
                model=PCA(n_components=32,svd_solver='full',whiten=False).fit(scaler.transform(raw[100:,start:stop]))
                expected.append(model.transform(scaler.transform(raw[:100,start:stop])))
                np.testing.assert_allclose(params[name+'_scaler_mean'],raw[100:,start:stop].mean(0),atol=1e-14)
        np.testing.assert_allclose(e.project(raw[:100],params),np.c_[*expected],atol=1e-12,rtol=1e-12)
        self.assertEqual(params['morphology100_scaler_scale'][0],1.)
        changed=raw.copy();changed[:100]+=1e6
        again=e.fit_pca(changed[100:])
        for key in params: np.testing.assert_array_equal(params[key],again[key])

    def test_outer_scope_label_poisoning_and_shared_pca(self):
        data,raw,ref,_=fixture()
        with temporary() as folder,redirect_stdout(io.StringIO()):
            vi,pred,_=e.evaluate_fold(folder,{},raw,data.y,ref['folds'],42,0,predictors(),True)
            poisoned=data.y.copy();poisoned[vi]+=1e6
            with patch.object(e,'fit_pca',side_effect=AssertionError('No refit')):
                _,again,_=e.evaluate_fold(folder,{},raw,poisoned,ref['folds'],42,0,{},False)
            for n in pred: np.testing.assert_array_equal(pred[n],again[n])
        self.assertEqual([c['train'].shape for c in Fake.calls],[(400,96),(400,486),(400,96)])
        self.assertEqual([c['query'].shape for c in Fake.calls],[(100,96),(100,486),(100,96)])
        for c in Fake.calls:
            np.testing.assert_array_equal(c['y'],data.y[100:]);self.assertEqual(c['seed'],51)
        np.testing.assert_array_equal(Fake.calls[0]['train'],Fake.calls[2]['train'])
        np.testing.assert_array_equal(Fake.calls[0]['query'],Fake.calls[2]['query'])
        np.testing.assert_array_equal(Fake.calls[1]['train'],raw[100:])

    def test_cache_integrity_inputs_and_missing_cache(self):
        _,raw,_,_=fixture();ti,vi=np.arange(100,500),np.arange(100)
        with temporary() as folder:
            e.pca_cell(folder,{},raw,ti,vi,True)
            changed=raw.copy();changed[0,5]+=.1
            with self.assertRaisesRegex(ValueError,'cache binding'):
                e.pca_cell(folder,{},changed,ti,vi,False)
            with (folder/'parameters.npz').open('ab') as f:f.write(b'corrupt')
            with self.assertRaisesRegex(ValueError,'parameters changed'):
                e.pca_cell(folder,{},raw,ti,vi,False)
        with temporary() as folder,patch.object(e,'fit_pca',side_effect=AssertionError('No fit')):
            with self.assertRaisesRegex(ValueError,'implicit fitting'):
                e.pca_cell(folder,{},raw,ti,vi,False)

    def test_partial_resume_no_repeated_context_or_pca(self):
        data,raw,ref,_=fixture()
        def interrupted(*args): raise RuntimeError('Interrupted')
        p=predictors();p['tabicl486']=interrupted
        with temporary() as folder,redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError,'Interrupted'):
                e.evaluate_fold(folder,{},raw,data.y,ref['folds'],42,0,p,True)
            with patch.object(e,'fit_pca',side_effect=AssertionError('No PCA refit')):
                e.evaluate_fold(folder,{},raw,data.y,ref['folds'],42,0,predictors(),True)
            self.assertEqual([c['name'] for c in Fake.calls],list(e.REPS))
            changed=data.y.copy();changed[100]+=1
            with self.assertRaisesRegex(ValueError,'Completed cell changed'):
                e.evaluate_fold(folder,{},raw,changed,ref['folds'],42,0,{},False)

    def test_blend_formula_controls_and_sse_partition(self):
        data,_,ref,plan=fixture()
        predictions={'tabpfn96':data.y+1,'tabicl486':data.y+.5,'tabicl96':data.y+.25}
        metrics,rows,diag,errors,detail=e.result_data(data,ref,predictions,plan['frozen']['common_failure_ids'])
        for n in e.REPS:
            name=e.BLENDS[n];expected=.5*(ref['base']+predictions[n])
            np.testing.assert_array_equal([r[name] for r in rows],expected)
            self.assertFalse(np.allclose(expected,.5*(ref['current486']+predictions[n])))
            groups={r['group']:r for r in diag if r['model']==name}
            self.assertAlmostEqual(groups['common16']['sse']+groups['remaining484']['sse'],metrics[name]['sse'])
            self.assertAlmostEqual(groups['middle80']['sse']+groups['tails20']['sse'],metrics[name]['sse'])
        self.assertEqual(metrics[e.SECONDARY]['matched_control'],'tabpfn96_blend')
        self.assertEqual(metrics[e.PRIMARY]['matched_control'],'current486')
        self.assertAlmostEqual(metrics['tabicl486']['gain_vs_matched_control'],1.5)
        self.assertAlmostEqual(metrics['tabicl96']['gain_vs_matched_control'],.75)
        self.assertEqual(sum(r['common16'] for r in errors),16)
        self.assertEqual(set(detail),set(e.BLENDS.values()))

    def test_thirty_context_end_to_end_and_read_only_resume(self):
        data,raw,ref,plan=fixture();refs={s:ref for s in e.SEEDS}
        with temporary() as folder,redirect_stdout(io.StringIO()), \
                patch.object(e,'validate',return_value=(data,raw,refs,plan)), \
                patch.object(e,'verify_check'),patch.object(e,'make_predictors',side_effect=predictors):
            for seed in e.SEEDS:e.evaluate(folder,'synthetic',seed)
            self.assertEqual(len(Fake.calls),30)
            self.assertEqual([c['name'] for c in Fake.calls],list(e.REPS)*10)
            for i,c in enumerate(Fake.calls):
                self.assertEqual(c['seed'],e.SEEDS[i//15]+1009*((i%15)//3)+9)
            with patch.object(e,'fit_pca',side_effect=AssertionError('No fit')), \
                    patch.object(e,'make_predictors',side_effect=AssertionError('No model')):
                for seed in e.SEEDS:e.evaluate(folder,'synthetic',seed)
                result=e.compare(folder)
            self.assertEqual(result['primary'],e.PRIMARY)
            self.assertFalse(result['selection_performed'])
            self.assertEqual(len(result['per_seed']),2)
            self.assertEqual(len(Fake.calls),30)
            summary=e.t.read_json(folder/'seed42/summary.json')
            self.assertFalse(summary['inner_search'])
            self.assertEqual(summary['archived486_contexts_reused'],5)
            with (folder/'seed42/oof.csv').open('a') as f:f.write('invalid')
            with self.assertRaises(ValueError):e.compare(folder)

    def test_missing_synthetic_check_stops_before_models_or_pca(self):
        data,raw,ref,plan=fixture()
        with temporary() as folder,patch.object(e,'validate',return_value=(data,raw,{42:ref},plan)), \
                patch.object(e,'make_predictors',side_effect=AssertionError('No models')), \
                patch.object(e,'fit_pca',side_effect=AssertionError('No PCA')):
            with self.assertRaisesRegex(ValueError,'Run check successfully'):e.evaluate(folder,'cuda',42)

    def test_exact_runtime_required(self):
        class FakePFN:
            runtime={'wrong':True}
            def __init__(self,*args):pass
        _,_,ref,_=fixture()
        with patch.object(e.t,'Predictor',FakePFN),patch.object(e.t,'verify_weights'), \
                patch.object(e,'ICLPredictor',side_effect=AssertionError('Stop early')):
            with self.assertRaisesRegex(ValueError,'exact archived runtime'):e.make_predictors('cuda',ref)

    def test_numeric_gate_no_common16_veto_or_best_of_two(self):
        records=[{'seed':seed,'variants':{n:{'gain_vs_current486':.31,'gain_vs_matched_control':.1} for n in e.BLENDS.values()},
                  'diagnosis':{n:{'both_groups_improve_vs_current486':False} for n in e.BLENDS.values()}} for seed in e.SEEDS]
        r=e.aggregate(records)
        self.assertTrue(r['variants'][e.PRIMARY]['passes_expansion_priority'])
        self.assertFalse(r['variants'][e.PRIMARY]['both_groups_improve_in_both_seeds'])
        records[1]['variants'][e.PRIMARY]['gain_vs_current486']=-.1
        r=e.aggregate(records)
        self.assertFalse(r['variants'][e.PRIMARY]['all_seeds_improve_both'])
        self.assertTrue(r['variants'][e.SECONDARY]['passes_expansion_priority'])
        self.assertEqual(r['primary'],e.PRIMARY)
        self.assertFalse(r['selection_performed'])

    def test_invalid_shapes_and_overlapping_partition_rejected(self):
        _,raw,_,_=fixture()
        with self.assertRaises(ValueError):e.fit_pca(raw[:32])
        with temporary() as folder:
            with self.assertRaises(ValueError):e.pca_cell(folder,{},raw,np.arange(400),np.arange(100),True)


if __name__=='__main__':unittest.main()
