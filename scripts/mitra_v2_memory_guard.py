"""Wait before Mitra starts a child; keep the existing AutoGluon RAM guard."""
import gc
import math
import time

GIB = 1024**3
POLICY = {'version':1, 'startup_available_gib':10.0, 'child_available_gib':6.5,
          'guard_headroom_gib':0.5, 'poll_seconds':15, 'timeout_seconds':1800,
          'ag_memory_ratio_unchanged':2.0}


def wait_for_ram(required, *, available, cleanup, emit=print,
                 clock=time.monotonic, sleep=time.sleep, timeout=1800, poll=15):
    start=clock();polls=0
    while True:
        cleanup()
        free=int(available())
        elapsed=clock()-start
        if free>=required:
            result={'required_bytes':int(required),'available_bytes':free,
                    'wait_seconds':elapsed,'polls':polls}
            emit(f'[RAM ready] available={free/GIB:.2f} GiB; required={required/GIB:.2f} GiB; waited={elapsed:.0f}s',flush=True)
            return result
        if elapsed>=timeout:
            raise RuntimeError(f'RAM wait timed out before training: available={free/GIB:.2f} GiB, '
                f'required={required/GIB:.2f} GiB. Close memory-heavy applications and rerun the resume command. '
                'Completed outer cells remain reusable.')
        emit(f'[RAM wait] available={free/GIB:.2f} GiB; required={required/GIB:.2f} GiB. '
             f'Close memory-heavy applications; retry in{poll}s. ({elapsed:.0f}/{timeout}s)',flush=True)
        polls+=1
        sleep(min(poll,timeout-elapsed))


def required_for_child(estimate, ratio, threshold=.9):
    if ratio != POLICY['ag_memory_ratio_unchanged'] or threshold<=0:
        raise ValueError('Unexpected AG guard configuration; refusing to weaken it')
    return max(int(POLICY['child_available_gib']*GIB),
               math.ceil(estimate/(threshold*ratio)+POLICY['guard_headroom_gib']*GIB))


def install(events):
    import psutil
    import torch
    from autogluon.tabular.models.mitra.mitra_model import MitraModel
    from autogluon.common.utils.resource_utils import ResourceManager
    from autogluon.core.utils.exceptions import NotEnoughMemoryError
    original=MitraModel._validate_fit_memory_usage
    if getattr(original,'_mitra_wait_guard',False):raise RuntimeError('RAM guard already installed')

    def cleanup():
        gc.collect()
        if torch.cuda.is_initialized():torch.cuda.empty_cache()

    def guarded(self,*args,**kwargs):
        if args:raise TypeError('Expected keyword-only AutoGluon memory validation')
        ratio=self.params_aux['max_memory_usage_ratio']
        estimate=kwargs.get('approx_mem_size_req')
        if estimate is None:estimate=self.estimate_memory_usage(**kwargs)
        required=required_for_child(estimate,ratio,kwargs.get('mem_error_threshold',.9))
        start=time.monotonic()
        while True:
            remaining=POLICY['timeout_seconds']-(time.monotonic()-start)
            if remaining<=0:raise RuntimeError('RAM guard retry timeout; no training was retried')
            event=wait_for_ram(required,available=ResourceManager.get_available_virtual_mem,
                cleanup=cleanup,timeout=remaining,poll=POLICY['poll_seconds'])
            events.append({'stage':'child_memory_guard','estimate_bytes':estimate,**event})
            try:
                # Re-read live RAM inside the unchanged upstream guard.
                check=dict(kwargs);check.pop('available_mem',None)
                return original(self,**check)
            except NotEnoughMemoryError:
                # Only this pre-fit check is retried; never retry a training
                # body or catch CUDA OOM / model errors here.
                print('[RAM wait] RAM fell during the guard check; waiting before this child starts.',flush=True)
                time.sleep(min(POLICY['poll_seconds'],remaining))
    guarded._mitra_wait_guard=True
    MitraModel._validate_fit_memory_usage=guarded
    return lambda:wait_for_ram(int(POLICY['startup_available_gib']*GIB),
        available=lambda:psutil.virtual_memory().available,cleanup=cleanup,
        timeout=POLICY['timeout_seconds'],poll=POLICY['poll_seconds'])
