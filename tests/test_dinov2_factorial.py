"""Synthetic-only model/transform/split leakage/cache/end-to-end tests."""
from contextlib import contextmanager,redirect_stdout
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from threadpoolctl import threadpool_limits
from scripts import dinov2_factorial_engine as eng
from scripts import dinov2_factorial_experiment as e
from scripts import extract_frozen_dinov2 as old


@contextmanager
def temporary():
    with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent,prefix='.dino_factorial_') as p:
        yield Path(p)


def fixture(n=150):
    rng=np.random.default_rng(48)
    x={r:rng.normal(size=(n,70)) for r in eng.REPS}
    for a in x.values():a[:,0]=np.arange(n)
    y=190+5*x['large518'][:,1]+rng.normal(size=n)
    data=SimpleNamespace(ids=np.array([f'SYNTH{i:06d}' for i in range(n)]),y=y,noise=rng.uniform(size=n))
    ref={'current':y+rng.normal(size=n)*8,'folds':np.arange(n)%5}
    plan={'frozen':{'common_failure_ids':data.ids[:16].tolist(),'synthetic':True}}
    return data,ref,x,plan


def fake_fit(tx,y,qx,indices=None):
    indices=range(9) if indices is None else indices
    return {'predictions':{str(i):(np.full(len(qx),np.mean(y))+.1*i+qx[:,1]).tolist() for i in indices},
            'audit':{'synthetic':True}}


class FactorialTests(unittest.TestCase):
    def setUp(self):
        self.limit=threadpool_limits(limits=2);self.limit.__enter__()

    def tearDown(self):self.limit.__exit__(None,None,None)

    def test_four_architectures_token_shapes_without_weights_or_download(self):
        import torch
        torch.set_num_threads(2)
        for name,dim in [('small',384),('large',1024)]:
            model=eng.architecture(name,meta=True).eval()
            for size in (266,518):
                with torch.inference_mode():out=model.forward_features(torch.empty(1,3,size,size,device='meta'))
                self.assertEqual(tuple(out.shape),(1,1+(size//14)**2,dim))

    def test_native_transform_matches_original_and_resize_is_full_image(self):
        import torch
        a=np.random.default_rng(8).integers(0,256,(256,256),dtype=np.uint8).astype(np.float32)/np.float32(255)
        raw=torch.from_numpy(a)[None,None]
        out=eng.transform_image(raw,266)
        expected=np.repeat(a[None],3,axis=0)
        expected-=np.array([.485,.456,.406],np.float32)[:,None,None]
        expected/=np.array([.229,.224,.225],np.float32)[:,None,None]
        torch.testing.assert_close(out[0],old.reflect_pad_input(torch,torch.from_numpy(expected)),rtol=0,atol=0)
        enlarged=eng.transform_image(raw,518)
        direct=torch.nn.functional.interpolate(raw,size=(518,518),mode='bicubic',align_corners=False,antialias=True)
        torch.testing.assert_close(enlarged[:,0:1]*.229+.485,direct)

    def test_pooling_excludes_cls_and_uses_population_std(self):
        import torch
        a=torch.zeros(2,362,384);a[:,0]=1000
        a[:,1:]=torch.arange(361)[None,:,None]
        x=eng.pool_tokens(a,266,384)
        self.assertEqual(x.shape,(2,1152))
        self.assertEqual(x[0,384].item(),180.)
        self.assertAlmostEqual(x[0,768].item(),np.arange(361).std(),places=4)
        for bad in (a[:,:361],a.double()):
            with self.assertRaises(ValueError):eng.pool_tokens(bad,266,384)

    def test_all_scalers_and_pca_fit_context_only_query_rows_cannot_change_fit(self):
        data,_,x,_=fixture();a=x['small266'];q=a[120:].copy()
        result=eng.fit_candidates(a[:120],data.y[:120],q)
        q[1:]+=1e8
        changed=eng.fit_candidates(a[:120],data.y[:120],q)
        self.assertEqual(result['audit'],changed['audit'])
        np.testing.assert_allclose(result['audit']['x_mean'],a[:120].mean(0))
        for key in result['predictions']:
            self.assertAlmostEqual(result['predictions'][key][0],changed['predictions'][key][0],places=7)
        self.assertEqual(len(result['predictions']),9)
        self.assertEqual(result['audit']['pca_components'],64)

    def test_shared_preprocessing_matches_independent_sklearn_pipelines(self):
        from sklearn.compose import TransformedTargetRegressor
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        from sklearn.svm import SVR
        from sklearn.linear_model import Ridge
        data,_,x,_=fixture();a=x['small266']
        actual=eng.fit_candidates(a[:120],data.y[:120],a[120:])
        for i in (1,3,8):
            c=eng.CANDIDATES[i]
            regression=(make_pipeline(StandardScaler(),Ridge(alpha=c['alpha'])) if c['family']=='ridge'
                else make_pipeline(StandardScaler(),PCA(64,svd_solver='full'),SVR(C=c['C'],gamma=c['gamma'],epsilon=.1)))
            model=TransformedTargetRegressor(regressor=regression,transformer=StandardScaler()).fit(a[:120],data.y[:120])
            np.testing.assert_allclose(actual['predictions'][str(i)],model.predict(a[120:]),rtol=1e-10,atol=1e-9)

    def test_nested_contexts_exclude_outer_query_and_own_inner_query_once(self):
        data,ref,x,_=fixture();ti=np.flatnonzero(ref['folds']!=0);vi=np.flatnonzero(ref['folds']==0)
        calls=[]
        def spy(tx,y,qx,indices=None):
            context=tx[:,0].astype(int);query=qx[:,0].astype(int)
            self.assertFalse(set(context)&set(query));self.assertTrue(set(context).issubset(set(ti)))
            np.testing.assert_array_equal(y,data.y[context]);calls.append((context,query,indices))
            return fake_fit(tx,y,qx,indices)
        with temporary() as p:
            result,choices,scores=e.fit_fold(p,{}, {r:a[ti] for r,a in x.items()},data.y[ti],
                                           {r:a[vi] for r,a in x.items()},42,runner=spy)
        self.assertEqual(len(calls),16)
        self.assertEqual(len(scores),36);self.assertEqual(len(choices),11)
        for k in range(4):
            owned=np.concatenate([calls[3*k+j][1] for j in range(3)])
            np.testing.assert_array_equal(np.sort(owned),ti)
        self.assertTrue(all(p.shape==(30,) for p in result.values()))

    def test_inner_selection_ties_and_joint_rule(self):
        y=np.arange(90.)
        p={r:{str(i):y+1 for i in range(9)} for r in eng.REPS}
        c,_=eng.select(p,y)
        self.assertEqual(c['joint_nested']['representation'],'small266')
        self.assertEqual(c['joint_nested']['candidate'],{'family':'ridge','alpha':10000.})
        p['large518']['8']=y.copy()
        c,_=eng.select(p,y)
        self.assertEqual(c['joint_nested']['representation'],'large518')
        self.assertEqual(c['joint_nested']['candidate_index'],8)
        self.assertEqual(c['ridge_joint']['representation'],'small266')

    def test_outer_label_poisoning_does_not_change_fitting_or_selection(self):
        data,ref,x,_=fixture();ti=np.flatnonzero(ref['folds']!=0);vi=np.flatnonzero(ref['folds']==0)
        y=data.y.copy();y[vi]+=1e8
        with temporary() as p:
            a=e.fit_fold(p/'a',{}, {r:v[ti] for r,v in x.items()},data.y[ti],{r:v[vi] for r,v in x.items()},42,runner=fake_fit)
            b=e.fit_fold(p/'b',{}, {r:v[ti] for r,v in x.items()},y[ti],{r:v[vi] for r,v in x.items()},42,runner=fake_fit)
        self.assertEqual(a[1:],b[1:])
        for key in a[0]:np.testing.assert_array_equal(a[0][key],b[0][key])

    def test_resume_without_fitting_and_scope_tamper_rejected(self):
        data,ref,x,_=fixture();ti=np.flatnonzero(ref['folds']!=0);vi=np.flatnonzero(ref['folds']==0)
        args=({}, {r:v[ti] for r,v in x.items()},data.y[ti],{r:v[vi] for r,v in x.items()},42)
        def fail(*a):raise AssertionError('No refitting allowed')
        with temporary() as p:
            a=e.fit_fold(p,*args,runner=fake_fit)
            b=e.fit_fold(p,*args,allow_fit=False,runner=fail)
            for key in a[0]:np.testing.assert_array_equal(a[0][key],b[0][key])
            with self.assertRaises(ValueError):e.fit_fold(p,{'changed':1},*args[1:],allow_fit=False,runner=fail)
            payload=e.read_json(p/'small266/inner0.json');payload['result']['predictions']['0'][0]+=1
            e.write_json(p/'small266/inner0.json',payload)
            with self.assertRaises(ValueError):e.fit_fold(p,*args,allow_fit=False,runner=fail)

    def test_feature_bundle_hash_id_and_anchor_guards(self):
        ids=np.array(['SYNTH0','SYNTH1']);x=np.ones((2,1152),np.float32)
        with temporary() as p:
            f=p/'x.npz';e.npz_save(f,ids,x,{'scope':1})
            loaded,_=e.npz_read(f,ids,1152,{'scope':1});np.testing.assert_array_equal(loaded,x)
            for bad_ids,width,binding in [(ids[::-1],1152,{'scope':1}),(ids,3072,{'scope':1}),(ids,1152,{'scope':2})]:
                with self.assertRaises(ValueError):e.npz_read(f,bad_ids,width,binding)
        self.assertEqual(e.validate_anchor(x,x)['max_abs_difference'],0)
        with self.assertRaises(ValueError):e.validate_anchor(x+.1,x)

    def test_incomplete_cache_fails_without_implicit_training(self):
        with temporary() as p:
            with self.assertRaises(FileNotFoundError):e.cell(p/'missing.json',{},lambda:None,False)
            with self.assertRaises(FileNotFoundError):e.npz_read(p/'missing.npz',np.array(['x']),1152)

    def test_saved_results_compare_without_fitting_and_fixed_blend(self):
        data,ref,x,plan=fixture();refs={s:ref for s in e.SEEDS};fr={'synthetic':True}
        with temporary() as p,redirect_stdout(io.StringIO()):
            for seed in e.SEEDS:
                binding=e.evaluation_binding(plan,seed,fr);sub=p/f'seed{seed}'
                pred,choices=e.collect(sub,data,ref,x,binding,seed,runner=fake_fit)
                np.testing.assert_array_equal(pred[e.PRIMARY],.5*ref['current']+.5*pred['joint_nested'])
                e.save_results(sub,data,ref,pred,choices,plan,binding)
            with patch.object(e,'validate',return_value=(data,refs,plan)),patch.object(e,'load_features',return_value=(x,fr)), \
                 patch.object(eng,'fit_candidates',side_effect=AssertionError('No fitting allowed')):
                result=e.compare(p)
            self.assertEqual(len(result['per_seed']),2)
            self.assertEqual(result['primary'],e.PRIMARY)
            self.assertIn('remaining484',result['per_seed'][0]['diagnosis'][e.PRIMARY]['group_rmse'])
            (p/'seed42/diagnostics.csv').write_text('tamper',encoding='utf-8')
            with self.assertRaises(ValueError):e.completed(p/'seed42',e.evaluation_binding(plan,42,fr))

    def test_driver_fails_fast_and_does_not_train_backbone(self):
        text=(e.ROOT/'scripts/run_dinov2_factorial.ps1').read_text()
        self.assertIn('$LASTEXITCODE -ne 0',text)
        self.assertEqual(text.count("'evaluate'"),2)
        self.assertIn("'--batch-size', '1'",text)


if __name__=='__main__':unittest.main()
