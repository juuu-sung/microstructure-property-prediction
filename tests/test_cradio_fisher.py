"""Synthetic tests only; no competition fits, checkpoint loads or GPU forwards."""
import contextlib
from dataclasses import asdict
import inspect
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits
from scripts import cradio_fisher_engine as eng
from scripts import cradio_fisher_experiment as exp


class FisherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = threadpool_limits(limits=2)
        (exp.ROOT/'.task-tmp').mkdir(exist_ok=True)

    @classmethod
    def tearDownClass(cls): cls.threads.restore_original_limits()

    def test_fisher_matches_direct_diagonal_gmm_definition(self):
        rng = np.random.default_rng(2); z = rng.normal(size=(29,3))
        p = {'weights':np.array([.3,.7]),'means':np.array([[1.,0.,-1.],[-.5,1.,.5]]),
            'covariances':np.array([[.8,1.5,.9],[1.3,.7,1.1]])}
        residual = z[:,None,:]-p['means'][None]
        logp = np.log(p['weights'])[None]-.5*np.sum(residual**2/p['covariances'][None]+np.log(2*np.pi*p['covariances'])[None],axis=2)
        prob = np.exp(logp-logsumexp(logp,axis=1,keepdims=True))
        u = np.mean(prob[:,:,None]*residual/np.sqrt(p['covariances'])[None],axis=0)/np.sqrt(p['weights'])[:,None]
        v = np.mean(prob[:,:,None]*(residual**2/p['covariances'][None]-1),axis=0)/np.sqrt(2*p['weights'])[:,None]
        expected = np.r_[u.ravel(),v.ravel()]; expected = np.sign(expected)*np.sqrt(np.abs(expected)); expected /= np.linalg.norm(expected)
        actual = eng.fisher_projected(z,p)
        np.testing.assert_allclose(actual,np.r_[expected,prob.mean(0)],atol=1e-12)
        self.assertAlmostEqual(actual[-2:].sum(),1.)

    def test_balanced_samples_and_image_level_view_average(self):
        spec = eng.FisherSpec(components=3,clusters=2,samples_per_view=2)
        rng = np.random.default_rng(3); tokens = rng.normal(size=(8,12,5)).astype(np.float32)
        pos = eng.sample_positions('SYNTH_ID',12,spec)
        np.testing.assert_array_equal(pos,eng.sample_positions('SYNTH_ID',12,spec))
        self.assertTrue(all(len(set(p))==2 for p in pos))
        samples = eng.sample_views(tokens,'SYNTH_ID',spec)
        np.testing.assert_array_equal(samples,np.concatenate([tokens[v,pos[v]] for v in range(8)]))
        context = np.stack([samples+i*.15 for i in range(6)])
        p,a = eng.fit_dictionary(context,12,spec)
        np.testing.assert_allclose(p['pca_mean'],context.reshape(-1,5).mean(0),atol=2e-6)
        self.assertEqual(a['context_images'],6); self.assertFalse(a['query_samples_used']); self.assertFalse(a['labels_used'])
        actual = eng.encode_image(tokens,p,spec)
        expected = np.mean([eng.fisher_projected(eng.project(v,p),p) for v in tokens],axis=0).astype(np.float32)
        np.testing.assert_array_equal(actual,expected)
        np.testing.assert_allclose(actual,eng.encode_image(tokens[::-1],p,spec),atol=1e-7)
        self.assertEqual(actual.shape,(14,)); self.assertEqual(actual.dtype,np.float32)

    def test_final_blend_selector_and_stronger_alpha_tie(self):
        y = np.full(100,10.)
        inner = {'0':np.full(100,8.),'1':np.full(100,12.),'2':np.full(100,15.)}
        # S=10,L=6 gives other6.5. Desired candidate is14, not the old50:50 objective's10.
        choice = eng.choose(inner,y,np.full(100,6.5))
        self.assertEqual(choice['selected']['index'],2)
        old = eng.d4.choose('radio',inner,y,np.full(100,10.))
        self.assertEqual(old['selected']['index'],0)
        tie = eng.choose({str(i):y for i in eng.d4.INDICES},y,.75*y)
        self.assertEqual(tie['selected']['alpha'],10000)
        self.assertNotIn('query_y',inspect.signature(exp.fit_fold).parameters)
        self.assertNotIn('query_y',inspect.signature(eng.inner_large).parameters)

    def test_only_radio_quarter_changes(self):
        x = np.arange(20,dtype=float); a,b,c = x+2,x-3,x+7
        p = eng.compose(x,x+1,a,b,c)
        np.testing.assert_allclose(p[eng.CONTROL]-p['current'],.25*(b-a))
        np.testing.assert_allclose(p[eng.PRIMARY]-p['current'],.25*(c-a))

    def test_dictionary_cache_context_only_resume_and_tamper(self):
        spec = eng.FisherSpec(components=3,clusters=2,samples_per_view=2)
        rng = np.random.default_rng(20); ids = np.array([f'SYNTH_{i}' for i in range(12)])
        tokens = rng.normal(size=(12,8,12,5)).astype(np.float32)
        samples = np.stack([eng.sample_views(t,i,spec) for t,i in zip(tokens,ids)])
        store = SimpleNamespace(ids=ids,tokens=tokens,samples=samples); ctx=np.arange(8); query=np.arange(8,12)
        real_fit = eng.fit_dictionary; calls=[]
        def fit(x,seed,**kw):
            np.testing.assert_array_equal(x,samples[ctx]); calls.append(1); return real_fit(x,seed,**kw)
        with tempfile.TemporaryDirectory(dir=exp.ROOT/'.task-tmp',prefix='fisher-dict-') as tmp, patch.object(eng,'SPEC',spec):
            folder = Path(tmp)
            with patch.object(eng,'fit_dictionary',side_effect=fit):
                a,b,audit = exp.distribution_features(folder,{'synthetic':True},store,ctx,query,42)
            self.assertEqual(len(calls),1);self.assertEqual(a.shape,(8,14));self.assertEqual(b.shape,(4,14))
            with patch.object(eng,'fit_dictionary',side_effect=AssertionError('unexpected fit')),patch.object(eng,'encode_image',side_effect=AssertionError('unexpected encoding')):
                aa,bb,again = exp.distribution_features(folder,{'synthetic':True},store,ctx,query,42,False)
            np.testing.assert_array_equal(a,aa);np.testing.assert_array_equal(b,bb);self.assertEqual(audit,again)
            receipt=exp.e.read_json(folder/'dictionary.json');self.assertEqual(receipt['binding']['dictionary_context'],ctx.tolist())
            with (folder/'dictionary.npz').open('ab') as f:f.write(b'corrupt')
            with self.assertRaises(ValueError): exp.distribution_features(folder,{'synthetic':True},store,ctx,query,42,False)

    def test_stream_hash_and_memmap_resume(self):
        with tempfile.TemporaryDirectory(dir=exp.ROOT/'.task-tmp',prefix='fisher-map-') as tmp:
            path=Path(tmp)/'small.npy';x=exp.open_cache(path,(3,2,4,5));x[:]=np.arange(x.size).reshape(x.shape);x.flush()
            digest=exp.array_hash(x);del x
            x=exp.open_cache(path,(3,2,4,5));self.assertEqual(digest,exp.array_hash(x));x[1,0,0,0]+=1
            self.assertNotEqual(digest,exp.array_hash(x));del x
            with self.assertRaises(ValueError):exp.open_cache(path,(4,2,4,5))

    def test_full_nested_contexts_reports_replay_and_tamper(self):
        ids=np.array([f'SYNTH_{i:06d}' for i in range(500)]); index=np.arange(500,dtype=float)
        data=SimpleNamespace(ids=ids,y=1000+index,noise=np.random.default_rng(5).uniform(size=500))
        features={'large':np.column_stack((index,index*.01)), 'radio':np.column_stack((index,index*.02))}
        folds=np.arange(500)%5; s=np.full(500,1250.);l=s+1;c=s+2
        refs={seed:{'folds':folds,'scattering':s,'large_d4':l,'radio_d4':c,'current':.5*s+.25*l+.25*c,
            'inner_scattering':{f:np.full(400,1250.) for f in range(5)}} for seed in exp.SEEDS}
        plan={'frozen':{'common_failure_ids':ids[:16].tolist()}}; calls=[];contexts=[]
        def runner(x,y,q,indices):
            np.testing.assert_array_equal(y,1000+x[:,0]); self.assertFalse(set(x[:,0])&set(q[:,0]));self.assertLessEqual(len(y),400)
            calls.append((len(y),len(q)))
            return {'predictions':{str(i):np.full(len(q),float(y.mean())+i*.1).tolist() for i in indices}}
        def distribution(folder,scope,store,ctx,query,seed,allow_fit):
            self.assertFalse(set(ctx)&set(query));self.assertLessEqual(len(ctx),400)
            contexts.append((ctx.copy(),query.copy(),allow_fit))
            audit={'context_images':len(ctx),'labels_used':False,'query_samples_used':False}
            return np.zeros((len(ctx),2)),np.zeros((len(query),2)),{'audit':audit,'fit_seconds':0.,'encode_seconds':0.}
        with tempfile.TemporaryDirectory(dir=exp.ROOT/'.task-tmp',prefix='fisher-eval-') as tmp:
            state=(data,features,refs,plan);token_state=(SimpleNamespace(ids=ids),{'synthetic':True})
            with patch.object(exp,'OUT',Path(tmp)),patch.object(exp,'validate',return_value=state), \
                    patch.object(exp,'load_tokens',return_value=token_state),patch.object(exp,'distribution_features',side_effect=distribution), \
                    patch.object(eng.ridge,'fit_candidates',side_effect=runner),contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(FileNotFoundError):exp.evaluate(42,False)
                self.assertEqual(calls,[])
                first=[exp.evaluate(seed) for seed in exp.SEEDS];self.assertEqual(len(calls),200)
                self.assertEqual(len(contexts),40);self.assertEqual(sum(len(a)==400 for a,b,c in contexts),10)
                second=[exp.evaluate(seed,False) for seed in exp.SEEDS]
                self.assertEqual(first,second);self.assertEqual(len(calls),200)
                result=exp.compare(state,second,token_state);self.assertFalse(result['automatic_adoption'])
                self.assertEqual(set(result['comparisons']),{eng.CONTROL,eng.PRIMARY})
                for model in (eng.CONTROL,eng.PRIMARY):
                    self.assertEqual(first[0]['diagnosis'][model]['groups']['common16']['n'],16)
                    self.assertEqual(first[0]['diagnosis'][model]['groups']['remaining484']['n'],484)
                    self.assertIn('gain_vs_control',first[0]['diagnosis'][model]['groups']['noiseQ4'])
                with (Path(tmp)/'seed42/oof.csv').open() as f:
                    header=f.readline();self.assertIn(eng.CONTROL,header);self.assertIn(eng.PRIMARY,header)
                p=Path(tmp)/'seed42/report.txt';p.write_text('tampered')
                with self.assertRaises(ValueError):exp.evaluate(42,False)


if __name__=='__main__':unittest.main()
