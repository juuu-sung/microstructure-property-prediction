"""Synthetic/meta-only checks. No checkpoint download or competition fit/forward."""
import contextlib
import inspect
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
from scripts import dinov2_large_register_engine as eng
from scripts import dinov2_large_register_d4_experiment as exp


class RegisterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def test_pinned_meta_architecture_and_environment(self):
        before, path = exp.e.environment(), list(sys.path)
        with patch('torch.hub.load', side_effect=AssertionError('No download')):
            model = eng.architecture(meta=True)
        self.assertTrue(all(p.is_meta for p in model.parameters()))
        self.assertEqual(sum(p.numel() for p in model.parameters()), 304370688)
        self.assertEqual((model.embed_dim, len(model.blocks), model.num_prefix_tokens, model.num_reg_tokens), (1024,24,5,4))
        self.assertEqual(list(model.pos_embed.shape), [1,1369,1024])
        self.assertEqual(list(sys.path), path); self.assertEqual(exp.e.environment(), before)

    def test_registers_excluded_not_treated_as_spatial_patches(self):
        tokens = torch.empty(2,1374,1024)
        tokens[:,0] = 3.; tokens[:,1:5] = 1e8
        spatial = torch.arange(1369, dtype=torch.float32)[None,:,None].expand(2,-1,1024)
        tokens[:,5:] = spatial
        x = eng.pool(tokens)
        self.assertEqual(x.shape,(2,3072))
        torch.testing.assert_close(x[:,:1024], tokens[:,0])
        torch.testing.assert_close(x[:,1024:2048], spatial.mean(1))
        torch.testing.assert_close(x[:,2048:], spatial.std(1,correction=0))
        tokens[:,1:5] = -1e8
        torch.testing.assert_close(x,eng.pool(tokens))
        with self.assertRaises(ValueError): eng.pool(tokens[:,:1370])
        with self.assertRaises(ValueError): eng.pool(tokens.to(torch.bfloat16))

    def test_d4_eight_views_and_fp32_image_level_mean(self):
        raw=torch.arange(9,dtype=torch.float32).reshape(1,1,3,3);calls=[]
        def fake(name,model,x):
            calls.append(x.numpy().tobytes())
            return x[:,0,0,0,None].expand(-1,eng.WIDTH)
        with patch.object(eng,'forward',side_effect=fake):
            first,mean=eng.extract_views(None,raw)
        self.assertEqual(len(set(calls)),8)
        np.testing.assert_array_equal(first,np.zeros((1,3072)))
        np.testing.assert_array_equal(mean,np.full((1,3072),4.));self.assertEqual(mean.dtype,np.float32)

    def test_two_fixed_compositions_preserve_other_three_quarters(self):
        s=np.arange(10,dtype=float);l=s+1;c=s+2;r=s+9
        p=eng.compose(s,l,c,r)
        np.testing.assert_allclose(p['current'],.5*s+.25*l+.25*c)
        np.testing.assert_allclose(p[eng.PRIMARY]-p['current'],.125*(r-l))
        np.testing.assert_allclose(p[eng.SECONDARY]-p['current'],.25*(r-l))
        np.testing.assert_allclose(p[eng.PRIMARY],.5*p['current']+.5*p[eng.SECONDARY])

    def test_selection_matches_archived_large_and_stronger_alpha_tie(self):
        y=np.arange(10,dtype=float);inner={'0':y,'1':y-2,'2':y-1}
        picked=eng.choose(inner,y)
        old=eng.d4.choose('large',inner,y,y+2)
        self.assertEqual(picked['selected']['index'],old['selected']['index'])
        self.assertEqual(picked['objective'],'standalone')
        self.assertEqual(eng.choose({str(i):y for i in eng.d4.INDICES},y)['selected']['alpha'],10000)
        self.assertNotIn('query_y',inspect.signature(exp.fit_fold).parameters)
        self.assertNotIn('inner_scattering',inspect.signature(exp.fit_fold).parameters)

    def test_nested_context_only_resume_and_tamper_detection(self):
        x=np.arange(500,dtype=float)[:,None];y=x[:400,0]+1000;calls=[]
        def runner(ctx,target,query,indices):
            np.testing.assert_array_equal(target,ctx[:,0]+1000)
            self.assertFalse(set(ctx[:,0]) & set(query[:,0]));self.assertLessEqual(len(ctx),400)
            calls.append((len(ctx),len(query)))
            return {'predictions':{str(i):np.full(len(query),target.mean()+i).tolist() for i in indices}}
        with tempfile.TemporaryDirectory(dir=exp.ROOT/'.task-tmp',prefix='register-fold-') as tmp:
            args=(Path(tmp),{'synthetic':True},x[:400],y,x[400:],42)
            result,choice=exp.fit_fold(*args,runner=runner)
            self.assertEqual(len(calls),4);self.assertEqual(sum(n==400 for n,_ in calls),1)
            with patch.object(eng.d4.ridge,'fit_candidates',side_effect=AssertionError('Unexpected fitting')):
                replay,same=exp.fit_fold(*args,allow_fit=False)
            np.testing.assert_array_equal(result,replay);self.assertEqual(choice,same)
            p=Path(tmp)/'inner0.json';obj=exp.e.read_json(p);obj['result']['predictions']['0'][0]+=1;exp.e.write_json(p,obj)
            with self.assertRaises(ValueError):exp.fit_fold(*args,allow_fit=False)

    def test_both_contrasts_export_and_replay_without_refitting(self):
        rng=np.random.default_rng(6);ids=np.array([f'SYNTH_{i:06d}' for i in range(500)])
        data=SimpleNamespace(ids=ids,y=190+rng.normal(size=500)*15,noise=rng.uniform(size=500))
        folds=np.arange(500)%5;s=np.full(500,190.);l=s+1;c=s+2
        ref={'folds':folds,'scattering':s,'large_d4':l,'radio_d4':c,'current':.5*s+.25*l+.25*c}
        features=rng.normal(size=(500,3)).astype(np.float32);plan={'frozen':{'common_failure_ids':ids[:16].tolist()}};calls=[]
        def runner(ctx,target,query,indices):
            calls.append(1)
            return {'predictions':{str(i):np.full(len(query),target.mean()+i*.1).tolist() for i in indices}}
        with tempfile.TemporaryDirectory(dir=exp.ROOT/'.task-tmp',prefix='register-eval-') as tmp:
            state=(data,None,{s:ref for s in exp.SEEDS},plan)
            with patch.object(exp,'OUT',Path(tmp)),patch.object(exp,'validate',return_value=state), \
                 patch.object(exp,'load_features',return_value=(features,{'synthetic':True})), \
                 patch.object(eng.d4.ridge,'fit_candidates',side_effect=runner),contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(FileNotFoundError):exp.evaluate(42,allow_fit=False)
                self.assertEqual(len(calls),0)
                first=[exp.evaluate(s) for s in exp.SEEDS];self.assertEqual(len(calls),40)
                second=[exp.evaluate(s,allow_fit=False) for s in exp.SEEDS]
                self.assertEqual(first,second);self.assertEqual(len(calls),40)
                for name in eng.BLENDS:
                    d=first[0]['diagnosis'][name]
                    self.assertEqual(d['groups']['common16']['n'],16)
                    self.assertEqual(d['groups']['remaining484']['n'],484)
                    self.assertIn('new_worst20_ids',d)
                result=exp.compare(state,second)
                self.assertEqual(set(result['comparisons']),set(eng.BLENDS));self.assertFalse(result['automatic_adoption'])
                exp.compare();self.assertEqual(len(calls),40)
                with (Path(tmp)/'seed42/oof.csv').open() as f:
                    header=f.readline();self.assertIn(eng.PRIMARY,header);self.assertIn(eng.SECONDARY,header)
                p=Path(tmp)/'seed42/report.txt';p.write_text('tampered')
                with self.assertRaises(ValueError):exp.evaluate(42,allow_fit=False)


if __name__ == '__main__':unittest.main()
