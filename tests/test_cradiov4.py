"""Synthetic tests only: no competition fits/forwards or model downloads."""
import inspect
import json
import copy
import contextlib
import io
import subprocess
import sys
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from scripts import cradiov4_engine as eng
from scripts import cradiov4_experiment as exp
from scripts import recover_cradiov4_import as recovery


def temporary():
    return tempfile.TemporaryDirectory(dir=Path(__file__).parent, prefix='.cradiov4_')


class CRadioTests(unittest.TestCase):
    def test_recovery_rejects_any_unapproved_protocol_or_source_change(self):
        old=exp.e.read_json(exp.OUT/'protocol.json')
        current=copy.deepcopy(old['frozen'])
        for name,(_,after) in recovery.PATCHED_SOURCES.items():current['sources'][name]['sha256']=after
        recovery.verify_patch(old,current)
        changed=copy.deepcopy(current);changed['protocol']['regression']='different alpha grid'
        with self.assertRaises(ValueError):recovery.verify_patch(old,changed)
        changed=copy.deepcopy(current);changed['y_sha256']='different labels'
        with self.assertRaises(ValueError):recovery.verify_patch(old,changed)
        changed=copy.deepcopy(current);changed['sources']['cradiov4_engine.py']['sha256']='unreviewed revision'
        with self.assertRaises(ValueError):recovery.verify_patch(old,changed)

    def test_recovery_preserves_feature_values_and_original_artifacts(self):
        old=exp.e.read_json(exp.OUT/'protocol.json')
        current=copy.deepcopy(old['frozen'])
        for name,(_,after) in recovery.PATCHED_SOURCES.items():current['sources'][name]['sha256']=after
        check=exp.e.read_json(exp.OUT/'synthetic_check.json')
        ids=np.array([f'SYNTH_{i:06d}' for i in range(500)])
        data=SimpleNamespace(ids=ids)
        values=np.broadcast_to(np.arange(500,dtype=np.float32)[:,None],(500,4608)).copy()
        with tempfile.TemporaryDirectory(dir=exp.ROOT/'reports',prefix='.cradiov4_recovery_') as tmp:
            source=Path(tmp)/'source';target=Path(tmp)/'recovered'
            exp.e.write_json(source/'protocol.json',old);exp.e.write_json(source/'synthetic_check.json',check)
            scope=exp.feature_binding(old,check['result'])
            for start in range(0,500,16):exp.e.npz_save(source/f'features/chunks/{start:04d}.npz',ids[start:start+16],values[start:start+16],scope)
            before={str(p):exp.e.base.sha256_file(p) for p in source.rglob('*') if p.is_file()}
            with patch.object(exp,'load_state',return_value=(data,None,None,current)),patch.object(eng,'load_model',side_effect=AssertionError('no inference')),patch.object(eng.ridge,'fit_candidates',side_effect=AssertionError('no fitting')),contextlib.redirect_stdout(io.StringIO()):
                recovery.recover(source,target)
                recovery.recover(source,target)  # Safe idempotent resume.
            loaded,_=exp.load_features(target,data,exp.e.read_json(target/'protocol.json'))
            np.testing.assert_array_equal(loaded,values)
            self.assertEqual(before,{str(p):exp.e.base.sha256_file(p) for p in source.rglob('*') if p.is_file()})

    def test_real_native_import_keeps_dependency_snapshot_in_fresh_process(self):
        code = '''
import sys
from scripts import cradiov4_engine as eng
from scripts import evaluate_tabpfn_v2 as t
before = t.environment()
paths = list(sys.path)
eng.architecture(meta=True)
assert paths == sys.path, 'Native import leaked search paths'
assert before == t.environment(), 'Native import changed distribution enumeration'
'''
        result = subprocess.run([sys.executable, '-c', code], cwd=exp.ROOT,
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)

    def test_native_import_restores_search_path_on_failure(self):
        before = list(sys.path)
        def failing_import(name):
            sys.path.append('synthetic_setuptools_vendor')
            raise ImportError('synthetic failure')
        with patch.object(eng.importlib, 'import_module', side_effect=failing_import):
            with self.assertRaises(ImportError):eng.native_modules()
        self.assertEqual(before, sys.path)

    def test_pinned_native_schema_matches_official_checkpoint_header(self):
        model=eng.architecture(meta=True).eval()
        header_path=eng.assets.ASSETS/'safetensors_header.json'
        header=json.loads(header_path.read_text())
        actual=model.state_dict()
        expected={k.removeprefix('radio_model.'):v for k,v in header.items() if k!='__metadata__'}
        self.assertEqual(set(actual),set(expected))
        for k,v in actual.items():self.assertEqual(list(v.shape),expected[k]['shape'])
        # Meta forward executes official topology without allocating weights.
        with torch.inference_mode():
            for size in (592,512):
                summary,spatial=model(torch.empty(1,3,size,size,device='meta'))
                self.assertEqual(tuple(summary.shape),(1,2304))
                self.assertEqual(tuple(spatial.shape),(1,(size//16)**2,1152))

    def test_preprocessing_pooling_and_no_duplicate_normalization(self):
        raw=torch.linspace(0,1,256*256).reshape(1,1,256,256)
        rgb=eng.transform(raw,592)
        self.assertEqual(tuple(rgb.shape),(1,3,592,592))
        self.assertGreaterEqual(float(rgb.min()),0)
        self.assertLessEqual(float(rgb.max()),1)
        torch.testing.assert_close(rgb[:,0],rgb[:,1])
        const=eng.transform(torch.full_like(raw,.5),512)
        torch.testing.assert_close(const,torch.full_like(const,.5))
        summary=torch.full((1,2304),2,dtype=torch.bfloat16)
        tokens=torch.arange(1024,dtype=torch.float32)[None,:,None].expand(1,1024,1152).to(torch.bfloat16)
        p=eng.pool((summary,tokens),512)
        self.assertEqual(p.dtype,torch.float32)
        self.assertEqual(tuple(p.shape),(1,4608))
        torch.testing.assert_close(p[:,:2304],summary.float())
        torch.testing.assert_close(p[:,2304:3456],tokens.float().mean(1))
        torch.testing.assert_close(p[:,3456:],tokens.float().std(1,correction=0))
        with self.assertRaises(ValueError):eng.pool((summary,tokens[:,:-1]),512)

    def test_blend_objective_differs_from_standalone_and_ties(self):
        y=np.arange(100,dtype=float)
        p={'0':np.zeros(100),'1':y.copy(),'2':y*3}
        result=eng.choose(p,y,2*y)
        self.assertEqual(result['blend']['index'],0)
        self.assertEqual(result['standalone']['index'],1)
        self.assertEqual(eng.choose({str(i):y for i in range(3)},y,y)['blend']['index'],0)
        out=eng.compose(y,2*y,3*y,4*y)
        np.testing.assert_array_equal(out[exp.PRIMARY],2*y)
        np.testing.assert_array_equal(out[exp.SECONDARY],1.75*y)

    def test_context_only_scaling_cache_resume_and_tampering(self):
        rng=np.random.default_rng(55);x=rng.normal(size=(120,5));y=x[:,0]*3+rng.normal(size=120)
        q=rng.normal(size=(8,5));scat=y+rng.normal(size=120)
        seen=[]
        def runner(tx,ty,qx,indices):
            seen.append((tx.copy(),ty.copy(),qx.copy()))
            r=eng.ridge.fit_candidates(tx,ty,qx,indices)
            np.testing.assert_allclose(r['audit']['x_mean'],tx.mean(0))
            self.assertAlmostEqual(r['audit']['y_mean'],ty.mean())
            self.assertFalse(r['audit']['query_labels_accepted'])
            return r
        with temporary() as tmp,threadpool_limits(limits=2):
            path=Path(tmp)
            p,c=exp.fit_fold(path,{'synthetic':1},{'large':x,'radio':x*2},y,{'large':q,'radio':q*2},scat,42,runner=runner)
            self.assertEqual(len(seen),8)
            again,choice=exp.fit_fold(path,{'synthetic':1},{'large':x,'radio':x*2},y,{'large':q,'radio':q*2},scat,42,False,exp.forbidden)
            self.assertEqual(choice,c)
            for name in p:np.testing.assert_array_equal(p[name],again[name])
            # A change to inner baseline must invalidate the selected refit.
            with self.assertRaises(ValueError):exp.fit_fold(path,{'synthetic':1},{'large':x,'radio':x*2},y,{'large':q,'radio':q*2},scat+10,42,False,exp.forbidden)
            cell=path/'radio/inner0.json';r=exp.e.read_json(cell)
            r['result']['predictions']['0'][0]+=1;exp.e.write_json(cell,r)
            with self.assertRaises(ValueError):exp.fit_fold(path,{'synthetic':1},{'large':x,'radio':x*2},y,{'large':q,'radio':q*2},scat,42,False,exp.forbidden)
        self.assertNotIn('query_y',inspect.signature(exp.fit_fold).parameters)

    def test_outer_labels_do_not_change_fold_predictions(self):
        rng=np.random.default_rng(123);x=rng.normal(size=(500,5));z=rng.normal(size=(500,6));y=190+8*x[:,0]+rng.normal(size=500)
        folds=np.arange(500)%5;ti=np.flatnonzero(folds!=0);vi=np.flatnonzero(folds==0)
        s=np.full(400,190.)
        with temporary() as tmp,threadpool_limits(limits=2):
            p,c=exp.fit_fold(Path(tmp),{}, {'large':x[ti],'radio':z[ti]},y[ti],{'large':x[vi],'radio':z[vi]},s,42)
            y[vi]=1e9
            p2,c2=exp.fit_fold(Path(tmp),{}, {'large':x[ti],'radio':z[ti]},y[ti],{'large':x[vi],'radio':z[vi]},s,42,False,exp.forbidden)
            self.assertEqual(c,c2)
            for k in p:np.testing.assert_array_equal(p[k],p2[k])

    def test_synthetic_full_pipeline_readonly_comparison_and_csv_tamper(self):
        rng=np.random.default_rng(97);n=500
        large=rng.normal(size=(n,5));radio=rng.normal(size=(n,6));y=190+8*large[:,0]+rng.normal(size=n)*7
        data=SimpleNamespace(ids=np.array([f'SYNTH_{i:06d}' for i in range(n)]),y=y,noise=rng.uniform(size=n))
        plan={'frozen':{'common_failure_ids':data.ids[:16].tolist()}}
        refs={};folds=np.arange(n)%5
        # Build synthetic archived standalone Large anchor with the same contexts.
        with threadpool_limits(limits=2):
            for seed in exp.SEEDS:
                anchor=np.full(n,np.nan);alphas=[];inner_s={}
                for fold in range(5):
                    ti=np.flatnonzero(folds!=fold);vi=np.flatnonzero(folds==fold)
                    preds={str(i):np.full(400,np.nan) for i in range(3)}
                    inner_s[fold]=np.full(400,190.)
                    for it,iv in eng.ridge.inner_splits(y[ti],seed+fold):
                        r=eng.ridge.fit_candidates(large[ti][it],y[ti][it],large[ti][iv],eng.RIDGE_INDICES)
                        for i,p in r['predictions'].items():preds[i][iv]=p
                    choice=eng.choose(preds,y[ti],inner_s[fold])['standalone']
                    anchor[vi]=eng.ridge.fit_candidates(large[ti],y[ti],large[vi],[choice['index']])['predictions'][str(choice['index'])]
                    alphas.append(choice['alpha'])
                scat=np.full(n,190.)
                refs[seed]={'folds':folds,'large':anchor,'scattering':scat,'current':.5*scat+.5*anchor,'inner_scattering':inner_s,'large_alphas':alphas}
            with temporary() as tmp,patch.object(exp,'validate',return_value=(data,large,refs,plan)),patch.object(exp,'load_features',return_value=(radio,{'synthetic':'features'})):
                folder=Path(tmp)
                for seed in exp.SEEDS:exp.evaluate(folder,seed)
                with patch.object(eng.ridge,'fit_candidates',side_effect=AssertionError('unexpected refit')):
                    r=exp.compare(folder)
                    self.assertEqual(r['primary'],exp.PRIMARY)
                    self.assertFalse(r['automatic_adoption'])
                csvpath=folder/'seed42/oof.csv';csvpath.write_text(csvpath.read_text()+'bad\n')
                with self.assertRaises(ValueError):exp.evaluate(folder,42,False)

    def test_runner_stops_after_error(self):
        text=(exp.ROOT/'scripts/run_cradiov4.ps1').read_text()
        self.assertIn('$LASTEXITCODE -ne 0',text)
        self.assertIn('throw',text)
        self.assertLess(text.index("'check'"),text.index("'extract'"))
        self.assertLess(text.index("'extract'"),text.index("'evaluate'"))


if __name__=='__main__':unittest.main()
