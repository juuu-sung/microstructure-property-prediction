"""Synthetic CPU tests only; no competition features, fitting, or GPU use."""
from contextlib import contextmanager, redirect_stdout
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
from scripts import resnet_texture_experiment as e


@contextmanager
def temporary():
    parent=Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory(dir=parent,prefix='.texture_test_') as td:
        if Path(td).resolve().parent!=parent:raise ValueError('Cleanup target outside tests')
        yield Path(td)


def fixture():
    rng=np.random.default_rng(22);x=rng.normal(size=(500,8))
    y=200+3*x[:,0]-2*x[:,1]+rng.normal(size=500)
    data=SimpleNamespace(ids=np.array([f'TRAIN_{i:06d}' for i in range(500)]),y=y,noise=x[:,3])
    folds=np.repeat(np.arange(5),100)
    features={n:x.copy() for n in e.REPS}
    ref={'current':y+3,'resnet':y+8,'folds':folds}
    plan={'frozen':{'common_failure_ids':data.ids[:16].tolist(),'reference_batch_size':16}}
    return data,features,ref,plan


class TextureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_projections_fixed_orthonormal_no_global_random_state(self):
        a=e.projections();np.random.seed(130);np.random.randn(99)
        np.testing.assert_array_equal(a,e.projections())
        self.assertEqual(a.shape,(2,64,32))
        for p in a:np.testing.assert_allclose(p.T@p,np.eye(32),rtol=0,atol=3e-7)
        self.assertFalse(np.array_equal(a[0],a[1]))

    def test_covariance_captures_relations_beyond_marginal_moments(self):
        # Same per-channel mean/std, opposite channel co-occurrence.
        a=torch.zeros(1,64,2,2);a[:,0]=torch.tensor([[0.,1.],[0.,1.]])
        a[:,1]=a[:,0];b=a.clone();b[:,1]=1-b[:,1]
        np.testing.assert_array_equal(a.mean((2,3)),b.mean((2,3)))
        np.testing.assert_array_equal(a.std((2,3)),b.std((2,3)))
        p=torch.eye(64)[:,:32]
        va=e.covariance_vector(torch,a,p);vb=e.covariance_vector(torch,b,p)
        self.assertEqual(va.shape,(1,528));self.assertFalse(torch.allclose(va,vb))
        torch.testing.assert_close(torch.linalg.vector_norm(va,dim=1),torch.ones(1))

    def test_covariance_spatial_permutation_translation_and_constant(self):
        torch.manual_seed(14);a=torch.randn(2,64,8,8);p=torch.from_numpy(e.projections()[0])
        v=e.covariance_vector(torch,a,p)
        perm=a.flatten(2)[:,:,torch.randperm(64)].reshape_as(a)
        torch.testing.assert_close(v,e.covariance_vector(torch,perm,p),rtol=2e-4,atol=3e-6)
        torch.testing.assert_close(v,e.covariance_vector(torch,a+3,p),rtol=5e-4,atol=1e-5)
        zero=e.covariance_vector(torch,torch.full_like(a,.1234),p)
        self.assertTrue(torch.isfinite(zero).all());self.assertEqual(torch.count_nonzero(zero).item(),0)

    def test_pooling_layout_and_frozen_bn_synthetic_model(self):
        import timm
        model=timm.create_model(e.base.MODEL_NAME,pretrained=False,features_only=True,out_indices=(0,1,2,3,4))
        model.requires_grad_(False);model.eval();before=e.base.feature_state_hash(model)
        with torch.inference_mode():
            maps=model(torch.randn(1,3,256,256));raw,early,cov=e.pool_maps(torch,maps,torch.from_numpy(e.projections()))
        self.assertEqual([tuple(x.shape) for x in (raw,early,cov)],[(1,1792),(1,256),(1,1056)])
        torch.testing.assert_close(raw,e.base.pool_feature_maps(torch,maps[2:])[1],rtol=0,atol=0)
        torch.testing.assert_close(early[:,:64],maps[0].mean((2,3)),rtol=0,atol=0)
        self.assertEqual(before,e.base.feature_state_hash(model))

    def test_outer_label_isolation_real_synthetic_ridge(self):
        from threadpoolctl import threadpool_limits
        data,x,ref,_=fixture();y2=data.y.copy();y2[:100]+=10000
        with threadpool_limits(limits=2):
            a=e.fit_fold(x,data.y,ref['folds'],42,0)
            b=e.fit_fold(x,y2,ref['folds'],42,0)
        self.assertEqual(a,b)
        # Query data and targets cannot fit either scaler.
        model=e.core.make_estimator(100).fit(x[e.REPS[0]][100:],data.y[100:])
        np.testing.assert_allclose(model.regressor_.steps[0][1].mean_,x[e.REPS[0]][100:].mean(0))
        np.testing.assert_allclose(model.transformer_.mean_,[data.y[100:].mean()])

    def test_pooled_mse_ties_and_fit_count(self):
        data,x,ref,_=fixture();calls=[]
        class Constant:
            def fit(self,a,y):calls.append((len(a),len(y)));return self
            def predict(self,a):return np.full(len(a),201.)
        result=e.fit_fold(x,data.y,ref['folds'],42,0,factory=lambda a,r:Constant())
        self.assertEqual(len(calls),30)
        self.assertEqual([z[0] for z in calls].count(400),3)
        expected=float(np.mean((data.y[100:]-201)**2))
        for rep in e.REPS:
            self.assertEqual(result['selection'][rep]['alpha'],10000.)
            for c in result['candidates'][rep]:self.assertAlmostEqual(c['inner_mse'],expected)

    def test_resume_no_refit_and_tamper_rejected(self):
        data,x,ref,_=fixture()
        with temporary() as folder:
            a=e.fold_cell(folder,{'seed':42},x,data.y,ref['folds'],42,0)
            def forbidden(*args):raise AssertionError('Unexpected refit')
            self.assertEqual(a,e.fold_cell(folder,{'seed':42},x,data.y,ref['folds'],42,0,runner=forbidden))
            with self.assertRaises(ValueError):e.fold_cell(folder,{'seed':2026},x,data.y,ref['folds'],42,0,runner=forbidden)
            saved=e.t.read_json(folder/'fold0.json');saved['result']['predictions'][e.REPS[0]][0]+=1
            e.t.write_json(folder/'fold0.json',saved)
            with self.assertRaises(ValueError):e.fold_cell(folder,{'seed':42},x,data.y,ref['folds'],42,0,runner=forbidden)

    def test_blend_only_resnet_one_eighth_and_report_artifacts(self):
        data,_,ref,plan=fixture();pred={n:ref['resnet']+8*(j+1) for j,n in enumerate(e.REPS)}
        out=e.compose_predictions(ref,pred)
        for j,n in enumerate(e.BLENDS):np.testing.assert_allclose(out[n],ref['current']+j+1)
        for n in pred:pred[n]=ref['resnet'].copy()
        out=e.compose_predictions(ref,pred)
        for n in e.BLENDS:np.testing.assert_array_equal(out[n],ref['current'])
        with temporary() as folder:
            for f in range(5):e.t.write_json(folder/f'fold{f}.json',{'mock':True})
            binding={'seed':42}
            with redirect_stdout(io.StringIO()):e.save_results(folder,data,ref,pred,[],binding,plan)
            self.assertTrue(e.completed(folder,binding))
            summary=e.t.read_json(folder/'summary.json')
            self.assertEqual(summary['metrics'][e.PRIMARY]['rmse'],3)
            (folder/'oof.csv').write_text('changed',encoding='utf8')
            with self.assertRaises(ValueError):e.completed(folder,binding)

    def test_feature_cache_array_id_and_hash_checks(self):
        data,_,_,plan=fixture();raw=np.zeros((500,1792),np.float32)
        arrays={'ids':data.ids,'early256':np.zeros((500,256),np.float32),
                'covariance1056':np.zeros((500,1056),np.float32),'projection':e.projections()}
        plan['frozen']['original_state_sha256']='mock'
        meta={'plan':e.t.digest(plan['frozen']),'state_sha256':'mock','test_accessed':False,
              'labels_saved':False,'identity_matched':True,
              'runtime':{'batch_size':16},'identity_atol':e.original.MATCH_ATOL,'identity_rtol':e.original.MATCH_RTOL,
              'arrays':{k:e.base.sha256_array(v) for k,v in arrays.items()}}
        with temporary() as folder:
            path=folder/'features/features.npz';path.parent.mkdir()
            def save():
                np.savez(path,**arrays,metadata_json=np.asarray(e.base.canonical_json(meta)))
                e.t.write_json(path.with_name('metadata.json'),meta)
            save();features,_=e.load_features(folder,data,raw,plan)
            self.assertEqual({k:v.shape[1] for k,v in features.items()},e.WIDTHS)
            arrays['early256'][0,0]=1;save()
            with self.assertRaises(ValueError):e.load_features(folder,data,raw,plan)
            arrays['early256'][0,0]=0;arrays['ids']=data.ids[::-1];save()
            with self.assertRaises(AssertionError):e.load_features(folder,data,raw,plan)

    def test_reference_batch_rejected_before_gpu_load(self):
        data,_,ref,plan=fixture()
        with temporary() as folder,patch.object(e,'validate',return_value=(data,None,ref,plan,None)), \
             patch.object(e,'model_runtime',side_effect=AssertionError('GPU must not be loaded')):
            with self.assertRaisesRegex(ValueError,'Use --batch-size 16'):
                e.extract(folder,'cuda',8)
        e.require_reference_batch(16,plan)

    def test_actual_anchor_failure_not_hidden_by_tolerance_change(self):
        actual=np.array([[.37562325596809387]],dtype=np.float32)
        reference=np.array([[.3756338655948639]],dtype=np.float32)
        with self.assertRaisesRegex(ValueError,'TRAIN_000033, column=0'):
            e.identity_check(actual,reference,['TRAIN_000033'])
        self.assertEqual(e.identity_check(reference,reference,['TRAIN_000033']),0.)
        with self.assertRaisesRegex(ValueError,'nonfinite'):
            e.identity_check(np.array([[np.nan]]),reference,['TRAIN_000033'])

    def test_missing_feature_message_precedes_regression(self):
        data,_,ref,plan=fixture()
        with temporary() as folder,patch.object(e,'validate',return_value=(data,None,ref,plan,None)), \
             patch.object(e,'fit_fold',side_effect=AssertionError('No regression permitted')):
            with self.assertRaisesRegex(ValueError,'Run extract --device cuda --batch-size 16'):
                e.evaluate(folder,42)

    def test_synthetic_check_uses_reference_batch(self):
        import timm
        model=timm.create_model(e.base.MODEL_NAME,pretrained=False,features_only=True,out_indices=(0,1,2,3,4))
        model.requires_grad_(False);model.eval()
        plan={'frozen':{'reference_batch_size':16,'original_state_sha256':e.base.feature_state_hash(model)}}
        def runtime(device,batch):
            self.assertEqual(batch,16)
            return torch,torch.device('cpu'),model,torch.from_numpy(e.projections()),{'batch_size':batch}
        with patch.object(e,'validate',return_value=(None,None,None,plan,None)), \
             patch.object(e,'model_runtime',side_effect=runtime),redirect_stdout(io.StringIO()) as out:
            e.check(Path('.'),'cpu')
        self.assertIn('[16,1792]',out.getvalue())

    def test_two_seed_pipeline_300_fits_completion_and_resume(self):
        data,x,ref,plan=fixture();calls=[]
        class Constant:
            def fit(self,a,y):calls.append(len(y));return self
            def predict(self,a):return np.full(len(a),201.)
        refs={s:ref for s in e.SEEDS}
        binding=lambda folder,seed,plan:{'seed':seed,'mock':True}
        with temporary() as folder, patch.object(e,'validate',return_value=(data,None,refs,plan,None)), \
             patch.object(e,'load_features',return_value=(x,{})),patch.object(e,'evaluation_binding',side_effect=binding), \
             patch.object(e.core,'make_estimator',side_effect=lambda alpha:Constant()),redirect_stdout(io.StringIO()):
            e.evaluate(folder,42);self.assertFalse((folder/'comparison.json').exists())
            e.evaluate(folder,2026);self.assertEqual(len(calls),300)
            comparison=e.t.read_json(folder/'comparison.json')
            self.assertEqual(len(comparison['per_seed']),2)
            self.assertFalse(comparison['all_seeds_improve_all_controls'])
            for seed in e.SEEDS:
                rows=e.core.read_rows(folder/f'seed{seed}/oof.csv',{'ID','hardness','fold',e.PRIMARY})
                self.assertEqual(len(rows),500)
                e.evaluate(folder,seed)
            self.assertEqual(len(calls),300)


if __name__=='__main__':unittest.main()
