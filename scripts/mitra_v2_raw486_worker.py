"""Uncompressed486 Mitra worker. Synthetic capacity check or explicit user-run fit."""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import sys
import time
if __package__:
    from . import mitra_v2_assets as assets
else:
    import mitra_v2_assets as assets

ROOT = assets.ROOT
OUT = ROOT/'reports/mitra_v2_raw486_runtime'
RECIPE = {'steps':50, 'bags':8, 'lr':1e-5, 'warmup_steps':10,
          'weight_decay':0.3, 'precision':'bfloat16', 'attention':'sdpa',
          'tune':'full', 'time_limit_seconds':86400, 'patience':50,
          'ag_max_memory_usage_ratio':2.0, 'persist_children':False}


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def environment():
    return {d.metadata['Name'].lower():d.version for d in importlib.metadata.distributions()}


def config(seed, steps):
    # Never inherit a user's previous Mitra ablation settings.
    for k in list(os.environ):
        if k.startswith('MITRA_'):
            del os.environ[k]
    os.environ.update({'MITRA_FT_STEPS':str(steps), 'MITRA_FT_ATTENTION':'sdpa',
        'MITRA_FT_FAST_LOOP':'1', 'MITRA_FT_PREFLIGHT':'1',
        'MITRA_REG_MAX_FEATURES':'0', 'MITRA_HELDOUT_IN_SUPPORT':'1',
        'MITRA_VAL_IN_SUPPORT':'0', 'MITRA_PREDICT_CHUNK':'0', 'HF_HUB_OFFLINE':'1',
        'HF_HUB_DISABLE_TELEMETRY':'1', 'DO_NOT_TRACK':'1',
        'CUBLAS_WORKSPACE_CONFIG':':4096:8', 'AG_DISABLE_TELEMETRY':'1'})
    temp = ROOT/'.task-tmp/mitra'
    temp.mkdir(parents=True, exist_ok=True)
    os.environ['TEMP'] = os.environ['TMP'] = str(temp)
    # tempfile may already have been imported; update its cached location.
    import tempfile
    tempfile.tempdir = str(temp)
    sys.path.insert(0, str(assets.BASE/'vendor/src'))
    import numpy as np
    import torch
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required; no silent CPU fallback')
    torch.cuda.set_device(0);torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError('BF16-capable GPU required')
    return np, torch


def install_audit(torch, audits):
    import mitra_finetune.hooks as hooks
    original = hooks.install_view
    active = [None]
    step = torch.optim.AdamW.step

    def counted_step(self, *args, **kwargs):
        result = step(self, *args, **kwargs)
        if active[0] is not None:
            active[0]['optimizer_steps'] += 1
            n = active[0]['optimizer_steps']
            if n == 1 or n % 10 == 0:
                print(f'[raw486 step] {n}; peak_allocated={torch.cuda.max_memory_allocated()/2**30:.3f} GiB', flush=True)
            if torch.cuda.max_memory_allocated() > torch.cuda.get_device_properties(0).total_memory:
                raise MemoryError('Raw486 exceeds physical GPU budget; context reduction forbidden')
        return result

    torch.optim.AdamW.step = counted_step

    def instrument(view):
        already = hasattr(hooks, 'ViewTrainer')
        original(view)
        if already:
            return
        cls = hooks.ViewTrainer
        train = cls.train

        def audited_train(self, x_train, y_train, x_val, y_val):
            hp = dict(self.cfg.hyperparams) if hasattr(self, 'cfg') else None
            if hp is None:
                hp = dict(self.config.hyperparams) if hasattr(self, 'config') else dict(self.hyperparams)
            record = {'train_rows':len(x_train), 'val_rows':len(x_val),
                'width':x_train.shape[1], 'optimizer_steps':0,
                'max_samples_support_before':hp.get('max_samples_support'),
                'precision':hp.get('precision'), 'configured_epochs':hp.get('max_epochs'),
                'lr':hp.get('lr'), 'warmup_steps':hp.get('warmup_steps'),
                'weight_decay':hp.get('weight_decay')}
            if hp.get('max_samples_support') != 20480:
                raise ValueError('Reduced support forbidden for raw486 comparison')
            active[0] = record
            start = time.perf_counter()
            try:
                result = train(self, x_train, y_train, x_val, y_val)
                record['preprocessor_dim_embedding'] = self.preprocessor.dim_embedding
                record['preprocessor_singular_columns'] = int(self.preprocessor.singular_features.sum())
                record['transformed_width'] = self.preprocessor.transform_X(x_train.copy()).shape[1]
                if record['preprocessor_dim_embedding'] is not None:
                    raise ValueError('Feature selection in bag preprocessor is forbidden')
                if record['transformed_width'] != 486-record['preprocessor_singular_columns']:
                    raise ValueError('Unexpected feature reduction inside bag')
                record['completed'] = True
                if torch.cuda.max_memory_allocated() > torch.cuda.get_device_properties(0).total_memory:
                    raise MemoryError('Raw486 exceeds physical GPU memory; do not rely on WDDM paging')
                return result
            except torch.cuda.OutOfMemoryError as exc:
                raise MemoryError('Raw486 capacity check failed; context reduction forbidden') from exc
            finally:
                record['seconds'] = time.perf_counter()-start
                record['epochs_observed'] = self._view_epoch
                record['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
                record['peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
                hp_after = self.cfg.hyperparams if hasattr(self, 'cfg') else (self.config.hyperparams if hasattr(self, 'config') else self.hyperparams)
                record['max_samples_support_after'] = hp_after.get('max_samples_support')
                audits.append(record);active[0] = None
                print('[experiment-audit] '+json.dumps(record, sort_keys=True), flush=True)
        cls.train = audited_train
    hooks.install_view = instrument


def install_local_runtime():
    """Use documented AG resource options without editing pinned vendor code."""
    import tempfile
    import psutil
    if psutil.virtual_memory().available < 6*1024**3:
        raise RuntimeError('At least6GiB free system RAM required for measured local configuration')
    from autogluon.features.generators import IdentityFeatureGenerator
    from tabarena.benchmark.exec_models.autogluon import AGSingleBagWrapper
    original = AGSingleBagWrapper.__init__
    def init(self,*args,**kwargs):
        kwargs['fit_kwargs'] = {**kwargs.get('fit_kwargs', {}), 'feature_generator': IdentityFeatureGenerator()}
        hp = dict(kwargs['model_hyperparameters'])
        hp['ag_args_fit'] = {**hp.get('ag_args_fit',{}),
            'ag.max_memory_usage_ratio':RECIPE['ag_max_memory_usage_ratio']}
        # Keep all50 steps observable; checkpoint selection still uses internal50.
        hp['patience'] = RECIPE['patience']
        kwargs['model_hyperparameters'] = hp
        kwargs['persist'] = False
        kwargs['init_kwargs'] = {**kwargs.get('init_kwargs',{}),
            'path':tempfile.mkdtemp(prefix='mitra_ag_',dir=ROOT/'.task-tmp/mitra')}
        return original(self,*args,**kwargs)
    AGSingleBagWrapper.__init__ = init
    # The next child's RAM guard otherwise sees the previous child's unused
    # CUDA allocator reservation (WDDM may back it with system memory).
    from autogluon.core.models.ensemble.fold_fitting_strategy import SequentialLocalFoldFittingStrategy
    import gc
    import torch
    original_fold = SequentialLocalFoldFittingStrategy._fit_fold_model
    def fit_fold(self,*args,**kwargs):
        gc.collect();torch.cuda.empty_cache()
        try:
            return original_fold(self,*args,**kwargs)
        finally:
            gc.collect();torch.cuda.empty_cache()
    SequentialLocalFoldFittingStrategy._fit_fold_model = fit_fold


def validate_audits(audits, steps, members):
    if members != 8 or len(audits) != 8:
        raise ValueError(f'Incomplete bagging: predictions={members}, trainers={len(audits)}; require8')
    for a in audits:
        if not a.get('completed') or a['optimizer_steps'] != steps or a['epochs_observed'] != steps:
            raise ValueError('Incomplete fine tuning; no truncated result accepted: '+str(a))
        if (a['train_rows'],a['val_rows'],a['width']) != (350,50,486):
            raise ValueError('Unexpected bag geometry: '+str(a))
        if a['max_samples_support_before'] != a['max_samples_support_after']:
            raise ValueError('OOM fallback changed training context; no reduced-context result accepted')
        if a['precision'] != RECIPE['precision'] or a['lr'] != RECIPE['lr']:
            raise ValueError('Runtime recipe differs: '+str(a))
        if a['preprocessor_dim_embedding'] is not None or a['transformed_width'] != 486-a['preprocessor_singular_columns']:
            raise ValueError('Unexpected internal feature selection')


def run(input_path, output_path, seed, synthetic=False):
    steps = 2 if synthetic else RECIPE['steps']
    pinned = assets.validate()
    np, torch = config(seed, steps)
    if synthetic:
        rng = np.random.default_rng(seed)
        x = rng.normal(size=(400,486)).astype('float32')
        q = rng.normal(size=(100,486)).astype('float32')
        y = (2*x[:,0]-x[:,1]+rng.normal(size=400)).astype('float64')
        input_hash = None
    else:
        with np.load(input_path, allow_pickle=False) as a:
            if set(a.files) != {'train_x','train_y','query_x'}:
                raise ValueError('Worker payload must contain training labels only')
            x,y,q = a['train_x'],a['train_y'],a['query_x']
        input_hash = assets.sha(input_path)
    if (x.shape,y.shape,q.shape) != ((400,486),(400,),(100,486)):
        raise ValueError('Require outer400/query100/486features')
    if any(not np.isfinite(a).all() for a in (x,y,q)):
        raise ValueError('Nonfinite input')
    import mitra_v2_memory_guard as guard
    events = []
    startup = guard.install(events)
    events.append({'stage':'worker_startup', **startup()})
    import mitra_v2_raw486_runtime as memory
    math_path = OUT/'math_check.json'
    math_check = json.loads(math_path.read_text(encoding='utf-8'))
    if not math_check['passed'] or math_check['runtime_sha256'] != assets.sha(memory.__file__):
        raise ValueError('Run check_mitra_raw486_math.py before GPU capacity test')
    memory.install()
    audits = []
    install_audit(torch, audits)
    install_local_runtime()
    from mitra_finetune import MitraFinetune
    free,total = torch.cuda.mem_get_info()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    estimator = MitraFinetune(str(assets.BASE), problem_type='regression',
        time_limit=RECIPE['time_limit_seconds'], device='cuda', random_state=seed,
        in_process=True, num_bag_folds=8)
    estimator.fit(x,y)
    if estimator._svd is not None or estimator._ksel_idx is not None:
        raise ValueError('Unexpected feature reduction; require uncompressed486')
    np.testing.assert_array_equal(np.asarray(estimator._fit_data[0]), x)
    # This call includes the one and only fine-tuning run; do not also call predict().
    dist = estimator.predict_distribution(q)
    torch.cuda.synchronize()
    validate_audits(audits, steps, dist.n_members)
    pred = np.asarray(dist.point_prediction, dtype='float64')
    if pred.shape != (100,) or not np.isfinite(pred).all():
        raise ValueError('Invalid Mitra predictions')
    np.testing.assert_allclose(dist.mean, pred, rtol=1e-4, atol=1e-4)
    result = {'predictions':pred.tolist(), 'input_sha256':input_hash,
        'seed':seed, 'synthetic':synthetic, 'competition_fits':0 if synthetic else 8,
        'competition_queries':0 if synthetic else 100, 'recipe':RECIPE,
        'steps_tested':steps,'audit':audits, 'bag_members':dist.n_members,
        'seconds':time.perf_counter()-started,
        'gpu':torch.cuda.get_device_name(0), 'torch_cuda':torch.version.cuda,
        'free_vram_before':free,'total_vram':total,
        'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
        'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
        'environment':environment(), 'assets':pinned,
        'worker_sha256':assets.sha(__file__), 'raw486_retained':True,
        'memory_guard':{'policy':guard.POLICY,'events':events},
        'memory_runtime':{'policy':memory.POLICY,'sha256':assets.sha(memory.__file__)},
        'math_check_sha256':assets.sha(math_path)}
    write(output_path,result)
    print(json.dumps({k:result[k] for k in ('synthetic','gpu','steps_tested','bag_members','seconds','peak_allocated_bytes','peak_reserved_bytes')}, sort_keys=True))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('environment','check','fit'))
    p.add_argument('--input', type=Path)
    p.add_argument('--output', type=Path, default=OUT/'synthetic_check.json')
    p.add_argument('--seed', type=int, default=314159)
    a=p.parse_args()
    if a.command=='environment':
        print(json.dumps(environment(),sort_keys=True))
    else:
        if a.command=='fit' and a.input is None: p.error('--input required')
        run(a.input,a.output,a.seed,a.command=='check')
