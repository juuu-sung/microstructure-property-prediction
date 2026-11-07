"""Synthetic/unit checks: no competition training, no GPU forwards."""
import contextlib
import io
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import mitra_v2_memory_guard as guard


class MemoryGuardTests(unittest.TestCase):
    def test_waits_until_ram_recovers_without_fitting(self):
        clock=[0.0];reads=iter([5,6,7]);clean=[]
        result=guard.wait_for_ram(7,available=lambda:next(reads),cleanup=lambda:clean.append(True),
            emit=lambda *a,**k:None,clock=lambda:clock[0],sleep=lambda s:clock.__setitem__(0,clock[0]+s),
            timeout=60,poll=15)
        self.assertEqual(result['wait_seconds'],30)
        self.assertEqual(result['polls'],2)
        self.assertEqual(len(clean),3)

    def test_timeout_is_bounded(self):
        clock=[0.0]
        with self.assertRaisesRegex(RuntimeError,'RAM wait timed out'):
            guard.wait_for_ram(10,available=lambda:7,cleanup=lambda:None,emit=lambda *a,**k:None,
                clock=lambda:clock[0],sleep=lambda s:clock.__setitem__(0,clock[0]+s),timeout=20,poll=15)
        self.assertEqual(clock[0],20)

    def test_requirement_preserves_existing_ag_threshold(self):
        estimate=int(10.776*guard.GIB)
        required=guard.required_for_child(estimate,2.0)
        self.assertGreaterEqual(required,estimate/1.8+.5*guard.GIB)
        self.assertEqual(required,6.5*guard.GIB)
        with self.assertRaises(ValueError):guard.required_for_child(estimate,3.0)

    def test_live_guard_still_checks_ram_after_wait(self):
        from autogluon.tabular.models.mitra.mitra_model import MitraModel
        from autogluon.common.utils.resource_utils import ResourceManager
        import torch
        original=MitraModel._validate_fit_memory_usage
        events=[]
        try:
            guard.install(events)
            subject=SimpleNamespace(params_aux={'max_memory_usage_ratio':2.0},
                estimate_memory_usage=lambda **k:int(10.776*guard.GIB))
            with (patch.object(ResourceManager,'get_available_virtual_mem',return_value=8*guard.GIB) as read,
                  contextlib.redirect_stdout(io.StringIO())):
                result=MitraModel._validate_fit_memory_usage(subject)
            self.assertGreaterEqual(read.call_count,2)
            self.assertEqual(result[1],8*guard.GIB)
            self.assertEqual(len(events),1)
            self.assertFalse(torch.cuda.is_initialized())
        finally:MitraModel._validate_fit_memory_usage=original

    def test_other_errors_are_not_hidden_or_retried(self):
        from autogluon.tabular.models.mitra.mitra_model import MitraModel
        from autogluon.common.utils.resource_utils import ResourceManager
        original=MitraModel._validate_fit_memory_usage
        def unexpected(*a,**k):raise ValueError('model/config bug')
        try:
            MitraModel._validate_fit_memory_usage=unexpected
            guard.install([])
            subject=SimpleNamespace(params_aux={'max_memory_usage_ratio':2.0},
                estimate_memory_usage=lambda **k:int(10.776*guard.GIB))
            with (patch.object(ResourceManager,'get_available_virtual_mem',return_value=8*guard.GIB),
                  contextlib.redirect_stdout(io.StringIO())):
                with self.assertRaisesRegex(ValueError,'model/config bug'):
                    MitraModel._validate_fit_memory_usage(subject)
        finally:MitraModel._validate_fit_memory_usage=original


if __name__=='__main__':unittest.main()
