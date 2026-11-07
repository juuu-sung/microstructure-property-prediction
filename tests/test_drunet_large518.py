"""Synthetic-only controls for image restoration and the matched nested experiment."""
import copy
import io
import inspect
import json
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from scripts import drunet_large518_experiment as d


def fixture():
    rng=np.random.default_rng(17);n=150
    x=rng.normal(size=(n,8));x[:,0]=np.arange(n)
    y=200+5*x[:,1]+rng.normal(size=n)
    data=SimpleNamespace(ids=np.array([f'SYNTH{i:06d}' for i in range(n)]),y=y,noise=rng.uniform(size=n))
    folds=np.arange(n)%5
    ref={'folds':folds,'scattering':y+rng.normal(size=n)*10,'raw_ridge':np.zeros(n),'alphas':[10000.]*5}
    # Identical predictions per candidate: deterministic preference for alpha10000.
    for k in range(5):ref['raw_ridge'][folds==k]=np.mean(y[folds!=k])+x[folds==k,1]
    ref['current']=.5*ref['scattering']+.5*ref['raw_ridge']
    plan={'frozen':{'synthetic':True,'common_failure_ids':data.ids[:16].tolist()}}
    return data,ref,{'raw_large518':x,'restored_large518':x.copy()},plan


def fake_fit(tx,ty,qx,indices):
    assert not set(tx[:,0].astype(int)) & set(qx[:,0].astype(int))
    assert set(indices).issubset({0,1,2})
    return {'predictions':{str(i):(np.mean(ty)+qx[:,1]).tolist() for i in indices}}


class RestorationTests(unittest.TestCase):
    def temporary(self):return tempfile.TemporaryDirectory(dir=Path(__file__).parent,prefix='.drunet_')

    def test_pilot_is_target_blind_balanced_and_order_stable(self):
        ids=np.array([f'TRAIN_{i:06d}' for i in range(500)]);noise=np.arange(500)%13
        selected=d.denoise.select_pilot(ids,noise)
        self.assertEqual(len(set(selected)),16)
        order=np.lexsort((ids,noise));bins=np.empty(500,int);bins[order]=np.arange(500)//125
        self.assertEqual(np.bincount(bins[selected]).tolist(),[4,4,4,4])
        permutation=np.random.default_rng(11).permutation(500)
        selected2=d.denoise.select_pilot(ids[permutation],noise[permutation])
        np.testing.assert_array_equal(ids[selected],ids[permutation][selected2])
        self.assertEqual(list(inspect.signature(d.denoise.select_pilot).parameters),['ids','noise'])

    def test_condition_units_cap_and_no_batch_statistics(self):
        quiet=np.full((256,256),128,dtype=np.uint8)
        noisy=np.random.default_rng(1).integers(0,256,(256,256),dtype=np.uint8)
        self.assertEqual(d.denoise.condition_sigma(quiet),0.)
        self.assertAlmostEqual(d.denoise.condition_sigma(noisy),10/255)
        rng=np.random.default_rng(3)
        moderate=np.clip(np.rint(128+rng.normal(0,5,(256,256))),0,255).astype(np.uint8)
        self.assertAlmostEqual(d.denoise.condition_sigma(moderate),.5*d.denoise.noise_proxy(moderate)/255)

    def test_restore_direct_output_not_noise_and_float_preservation(self):
        import torch
        class Probe(torch.nn.Module):
            def forward(self,x):
                self.seen=x.clone()
                return x[:,:1]+.001
        model=Probe();raw=np.full((256,256),100,dtype=np.uint8)
        restored,info=d.denoise.restore(model,raw,torch.device('cpu'))
        self.assertEqual(model.seen.shape,(1,2,256,256))
        self.assertEqual(restored.dtype,np.float32)
        np.testing.assert_allclose(restored,100/255+.001,rtol=1e-6)
        self.assertEqual(float(model.seen[:,1].max()),0.)
        self.assertEqual(info['clipped_fraction'],0.)

    def test_same_input_control_and_exact_half_scattering(self):
        data,ref,features,plan=fixture()
        with self.temporary() as tmp,redirect_stdout(io.StringIO()):
            p,c=d.collect(Path(tmp),data,features,ref,{},42,runner=fake_fit)
        np.testing.assert_array_equal(p['raw_large518'],p['restored_large518'])
        np.testing.assert_array_equal(p[d.PRIMARY],ref['current'])
        self.assertEqual(len(c),5)

    def test_raw_anchor_catches_reference_or_selection_mismatch(self):
        data,ref,features,plan=fixture();ref['raw_ridge'][0]+=1
        with self.temporary() as tmp,redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError,'anchor failed'):
                d.collect(Path(tmp),data,features,ref,{},42,runner=fake_fit)

    def test_resume_readonly_and_precise_group_accounting(self):
        data,ref,features,plan=fixture();summaries=[]
        with self.temporary() as tmp,redirect_stdout(io.StringIO()):
            folder=Path(tmp)
            for seed in d.SEEDS:
                sub=folder/f'seed{seed}';binding=d.ridge.binding_for(plan,seed)
                p,c=d.collect(sub,data,features,ref,binding,seed,runner=fake_fit)
                summaries.append(d.save_results(sub,data,ref,p,c,plan,seed))
                with patch.object(d.eng,'fit_candidates',side_effect=AssertionError('No fits')):
                    s=d.read_completed(sub,data,features,ref,plan,seed)
                self.assertEqual(s,summaries[-1]);self.assertEqual(s['primary'],d.PRIMARY)
                detail=s['diagnosis'][d.PRIMARY]
                self.assertEqual(detail['common16_sse_gain']+detail['remaining484_sse_gain'],0.)
                self.assertEqual(s['metrics'][d.PRIMARY]['gain_vs_current'],0.)
                self.assertIn('noiseQ4',detail['groups']);self.assertIn('remaining484',detail['groups'])
            result=d.aggregate(summaries)
            json.dumps(result,allow_nan=False)
            self.assertFalse(result['automatic_adoption']);self.assertFalse(result['passes_expansion_priority'])
            (folder/'seed42/common16.csv').write_text('tampered',encoding='utf-8')
            with self.assertRaises(ValueError):d.read_completed(folder/'seed42',data,features,ref,plan,42)

    def test_subgroup_worsening_not_hidden_by_positive_overall(self):
        data,ref,features,plan=fixture()
        p={'current':ref['current'],'scattering486':ref['scattering'],'raw_large518':ref['raw_ridge'],
           'restored_large518':data.y,d.PRIMARY:.5*ref['scattering']+.5*data.y}
        summaries=[d.result_data(data,ref,p,[],plan,s)[0] for s in d.SEEDS]
        summaries[0]['diagnosis'][d.PRIMARY]['groups']['remaining484']['gain_vs_current']=-.1
        result=d.aggregate(summaries)
        self.assertFalse(result['all_seeds_nonworsening_by_group']['remaining484'])
        self.assertFalse(result['passes_expansion_priority'])
        with self.assertRaises(ValueError):d.aggregate(summaries[:1])

    def test_runner_fail_fast_and_does_not_fit_or_select_denoiser_strength(self):
        text=(d.ROOT/'scripts/run_drunet_large518.ps1').read_text()
        self.assertIn('$LASTEXITCODE -ne 0',text)
        self.assertNotIn('submission',text)
        self.assertNotIn('--sigma',text)
        self.assertEqual(d.PROTOCOL['restoration']['condition_cap_255'],10.)


if __name__=='__main__':unittest.main()
