"""Regression for real one-shot torch interop initialization; no vision models."""
import subprocess
import sys
from pathlib import Path
import unittest
from unittest.mock import patch
from scripts import resume_final_large_radio_d4 as recovery


class RuntimeRecoveryTests(unittest.TestCase):
    def test_real_torch_two_extractor_initializations_after_parallel_work(self):
        # Fresh interpreter is necessary: interop thread initialization is
        # process-global and cannot be undone by an in-process test fixture.
        code = r'''
import torch
from unittest.mock import patch
from scripts import resume_final_large_radio_d4 as r
f=r.final
class Model(torch.nn.Module):
    def forward(self,x): return x*x
def runtime(device):
    assert device=='cpu'
    t,d,_=f.e.configure(device)
    return t,d,{'synthetic':True}
original=f.exp.runtime
with patch.object(f.exp,'runtime',side_effect=runtime) as configure, patch.object(f.exp,'model_for',side_effect=lambda name:Model().eval()):
    restored=f.exp.runtime
    with r.shared_runtime() as session:
        first=f.D4Extractor('large','cpu',{'synthetic':True})
        a=torch.ones((64,64)); first.model(a) @ a
        first.close()
        second=f.D4Extractor('radio','cpu',{'synthetic':True})
        second.model(a) @ a
        second.close()
        assert configure.call_count==1 and session.calls==1
        assert torch.get_num_interop_threads()==1
        try: session('cuda')
        except ValueError: pass
        else: raise AssertionError('device drift accepted')
        torch.backends.cuda.matmul.allow_tf32=True
        try: session('cpu')
        except ValueError: pass
        else: raise AssertionError('runtime drift accepted')
        torch.backends.cuda.matmul.allow_tf32=False
    assert f.exp.runtime is restored
assert f.exp.runtime is original
print('Real torch initialized once across two extractor constructors; no competition forwards.')
'''
        result = subprocess.run([sys.executable, '-c', code], cwd=recovery.final.ROOT,
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout+'\n'+result.stderr)
        self.assertIn('Real torch initialized once', result.stdout)

    def test_scope_restored_on_encoder_failure(self):
        original = recovery.final.exp.runtime
        with self.assertRaisesRegex(RuntimeError, 'encoder failed'):
            with recovery.shared_runtime():
                self.assertIsNot(recovery.final.exp.runtime, original)
                raise RuntimeError('encoder failed')
        self.assertIs(recovery.final.exp.runtime, original)

    def test_run_has_no_training_and_checks_old_files(self):
        fake_state = (None, None, None, None, None, None, {'frozen': {}})
        evidence = {'model_alphas': {'large':1000,'radio':1000}, 'preserved_files': {'kept':'hash'}}
        with patch.object(sys, 'argv', ['resume', 'run']), \
             patch.object(recovery.final, 'validate', return_value=fake_state), \
             patch.object(recovery, 'addendum', return_value={}), \
             patch.object(recovery, 'verify_completed', return_value=evidence), \
             patch.object(recovery, 'verify_unchanged') as unchanged, \
             patch.object(recovery.final.e, 'write_json'), patch.object(recovery.final.e, 'file_ref', return_value={}), \
             patch.object(recovery.final, 'extract') as extract, patch.object(recovery.final, 'predict') as predict, \
             patch.object(recovery.final, 'train', side_effect=AssertionError('No refitting')):
            recovery.main()
            extract.assert_called_once_with(device='cuda'); predict.assert_called_once_with()
            unchanged.assert_called_once_with({'kept':'hash'})


if __name__ == '__main__': unittest.main()
