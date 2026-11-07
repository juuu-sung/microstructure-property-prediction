"""Synthetic extension tests; no competition training or GPU inference."""
from contextlib import redirect_stdout
import copy
import io
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
import torch

from scripts import micronet_finetune_80_experiment as e
from tests.test_micronet_finetune import Tiny,batch,temporary,fixture,FakeRunner

prior=e.previous
g=e.eng
oldg=prior.eng


class FakeExtended:
    def __init__(self,late_wins=False):self.calls=[];self.late_wins=late_wins
    def __call__(self,ti,yt,vi,yv,mode,epochs,seed,folder,binding):
        self.calls.append({'ti':ti.copy(),'vi':vi.copy(),'yv':yv,'epochs':epochs})
        source=e.read_reference(binding)
        rows=[]
        for ep in g.EPOCHS:
            if ep>epochs:break
            shared=[r for r in source['records'] if r['epoch']==ep]
            if shared:row=copy.deepcopy(shared[0])
            else:
                row=copy.deepcopy(source['records'][-1]);row['epoch']=ep
                pred=np.full(len(vi),float(yt.mean())+1000)
                # Fake inner predictions exercise late-checkpoint selection only;
                # the real runner's target independence is tested separately.
                if self.late_wins and yv is not None:pred=yv+.1/ep
                row['predictions']=pred.tolist()
                if yv is not None:row['valid_sse']=float(np.sum((pred-yv)**2))
                active=mode=='partial' and ep>5
                row['update_audit']['encoder_updated']=active
                for stage in ('stage6','stage7'):
                    row['update_audit']['max_abs_parameter_change'][stage]=float(active)
                    row['update_audit']['max_gradient_l2'][stage]=float(active)
            rows.append(row)
        checkpoint=None
        if yv is None:
            (folder/'model.pt').write_bytes(b'synthetic extended checkpoint')
            checkpoint={'name':'model.pt','sha256':e.base.sha256_file(folder/'model.pt')}
        return {'records':rows,'audit':rows[-1]['update_audit'],'target_mean':float(yt.mean()),
                'target_std':float(yt.std()),'checkpoint':checkpoint}


class ExtensionTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2);FakeRunner.calls=[]

    def test_lr_preserves_first40_and_holds_floor_without_restart(self):
        self.assertEqual(g.EPOCHS,tuple(range(0,81,5)))
        for ep in range(1,41):self.assertEqual(g.lr_factor(ep),oldg.lr_factor(ep))
        for ep in range(41,81):self.assertEqual(g.lr_factor(ep),.1)
        for ep in (0,81):
            with self.assertRaises(ValueError):g.lr_factor(ep)

    def test_actual_synthetic_prefix_both_arms_and_resume_beyond40(self):
        x=np.random.default_rng(9).normal(size=(20,4));ti,vi=np.arange(16),np.arange(16,20)
        yt=x[ti,0]*5+100;yv=x[vi,0]*5+100
        binding={'seed':42,'runtime':{'device':'cpu','synthetic':True}}
        with temporary() as root,redirect_stdout(io.StringIO()),patch.object(prior,'OUT',root/'original'),\
             patch.object(oldg,'make_model',side_effect=lambda torch:Tiny()),patch.object(oldg,'batch_images',side_effect=batch),\
             patch.object(g,'make_model',side_effect=lambda torch:Tiny()),patch.object(g,'batch_images',side_effect=batch):
            runner=g.Runner(torch,torch.device('cpu'),x,e.guard_record)
            oldrunner=oldg.Runner(torch,torch.device('cpu'),x)
            for mode in g.MODES:
                source=prior.OUT/mode
                original=prior.cell(source,binding,ti,yt,vi,yv,mode,40,12,oldrunner)
                full=e.cell(root/'full'/mode,binding,ti,yt,vi,yv,mode,80,12,runner,source)
                for a,b in zip(original['records'],full['records']):
                    for k in a:
                        if k!='last_epoch_seconds':self.assertEqual(a[k],b[k])
                self.assertEqual(full['records'][-1]['epoch'],80)
                self.assertEqual(full['audit']['encoder_updated'],mode=='partial')
                if mode!='partial':continue
                optimize=g.optimize;count=0
                def interrupt(*args,**kwargs):
                    nonlocal count
                    count+=1
                    if count==46:raise RuntimeError('stop after45')
                    return optimize(*args,**kwargs)
                with patch.object(g,'optimize',side_effect=interrupt):
                    with self.assertRaisesRegex(RuntimeError,'stop after45'):
                        e.cell(root/'resume',binding,ti,yt,vi,yv,mode,80,12,runner,source)
                self.assertTrue((root/'resume/progress.json').exists())
                resumed=e.cell(root/'resume',binding,ti,yt,vi,yv,mode,80,12,runner,source)
                for a,b in zip(full['records'],resumed['records']):
                    self.assertEqual(a['predictions'],b['predictions'])
                    self.assertEqual(a['update_audit'],b['update_audit'])
                self.assertFalse((root/'resume/progress.json').exists())
                expected=e.t.read_json(root/'full'/mode/'binding.json')
                bad=copy.deepcopy(full['records'][4]);bad['predictions'][0]+=1
                with self.assertRaisesRegex(ValueError,'prefix mismatch'):e.guard_record(bad,expected)
                poisoned=copy.deepcopy(expected);poisoned['train_idx'][0]=999
                with self.assertRaisesRegex(ValueError,'sample/seed'):e.read_reference(poisoned)
                (source/'completed.json').write_bytes(b'corrupted reference')
                with self.assertRaisesRegex(ValueError,'reference changed'):e.guard_record(full['records'][0],expected)

    def test_two_seed_pipeline_reuse_and_late_refits_preserve_old_artifacts(self):
        data,ref,plan=fixture();refs={s:ref for s in e.SEEDS};rt={'synthetic':True}
        with temporary() as root,redirect_stdout(io.StringIO()),patch.object(prior,'OUT',root/'original'):
            for seed in e.SEEDS:
                sub=prior.OUT/f'seed{seed}';binding=prior.binding_for(plan,rt,seed)
                p,s=prior.collect(sub,binding,data,ref,seed,FakeRunner());prior.save_results(sub,binding,data,ref,p,s,plan)
            oldhash={p:e.base.sha256_file(p) for p in prior.OUT.rglob('*') if p.is_file()}
            for late in (False,True):
                folder=root/str(late);runner=FakeExtended(late)
                with patch.object(e,'validate',return_value=(data,refs,plan)),patch.object(e,'verify_check',return_value=rt):
                    for seed in e.SEEDS:
                        sub=folder/f'seed{seed}';binding=e.binding_for(plan,rt,seed)
                        p,s=e.collect(sub,binding,data,ref,seed,runner);e.save_results(sub,binding,data,ref,p,s,plan)
                    self.assertEqual(len(runner.calls),80 if late else 60)
                    for c in runner.calls:
                        self.assertEqual(len(np.intersect1d(c['ti'],c['vi'])),0)
                        if len(c['ti'])==400:self.assertIsNone(c['yv'])
                    with patch.object(g,'Runner',side_effect=AssertionError('No fitting during verification')):
                        for seed in e.SEEDS:e.train(folder,seed,'cpu')
                        result=e.compare(folder)
                    for seed in e.SEEDS:
                        evidence=e.t.read_json(folder/f'seed{seed}'/'summary.json')['extension_vs40']
                        for v in evidence.values():
                            self.assertEqual(v['selected_past40'],5 if late else 0)
                            if not late:self.assertEqual(v['gain_vs_previous40'],0.)
                    self.assertIn('extension_vs40',result)
                    # Changing the outer targets of fold0 cannot change its fitted
                    # models, selection or cached predictions.
                    poison=data.y.copy();poison[:100]+=1e6
                    sub=folder/'seed42';binding=e.binding_for(plan,rt,42)
                    original=e.fit_fold(sub,binding,data.y,ref['folds'],42,0,e.unavailable)
                    poisoned=e.fit_fold(sub,binding,poison,ref['folds'],42,0,e.unavailable)
                    self.assertEqual(original,poisoned)
                    with (sub/'oof.csv').open('a') as f:f.write('changed')
                    with self.assertRaisesRegex(ValueError,'artifact changed'):e.compare(folder)
            self.assertEqual(oldhash,{p:e.base.sha256_file(p) for p in oldhash})

    def test_runtime_change_stops_before_synthetic_model(self):
        data,ref,plan=fixture();plan['frozen']['previous40']={'runtime':{'different':True}}
        with temporary() as root,patch.object(e,'validate',return_value=(data,{42:ref},plan)),\
             patch.object(e.t,'configure_device',return_value=(torch,torch.device('cpu'))),\
             patch.object(g,'make_model',side_effect=AssertionError('No model')):
            with self.assertRaisesRegex(ValueError,'same runtime'):e.check(root,'cpu')


if __name__=='__main__':unittest.main()
