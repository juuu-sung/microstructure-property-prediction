"""Resume the archived final D4 run with one PyTorch initialization per process.

Keep all original sources, plan bindings and completed artifacts unchanged.
Only the process-local runtime initializer is shared between frozen encoders.
check verifies files only. run extracts missing features and writes the CSV;
it never calls train or fits a regression model.
"""
from __future__ import annotations
import argparse
import contextlib
from pathlib import Path
import random
import time
import numpy as np
if __package__:
    from . import final_large_radio_d4_submission as final
else:
    import final_large_radio_d4_submission as final

RECOVERY = final.OUT/'runtime_recovery'
ADDENDUM = RECOVERY/'runtime_addendum.json'


class RuntimeOnce:
    """Initialize interop threads once; reject device/settings drift on reuse."""
    def __init__(self, configure):
        self.configure = configure
        self.value = None
        self.request = None
        self.calls = 0

    def __call__(self, device):
        if self.value is None:
            self.value = self.configure(device)
            self.request = device
            self.calls += 1
        elif device != self.request:
            raise ValueError('Cannot switch device in the final D4 runtime session')
        else:
            torch, _, _ = self.value
            # Same seed as archived e.configure; do not reset thread pools.
            random.seed(123); np.random.seed(123); torch.manual_seed(123)
            if self.value[1].type == 'cuda': torch.cuda.manual_seed_all(123)
        torch, _, _ = self.value
        if (torch.get_num_threads() != 2 or torch.get_num_interop_threads() != 1
                or not torch.are_deterministic_algorithms_enabled()
                or not torch.backends.cudnn.deterministic or torch.backends.cudnn.benchmark
                or torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32):
            raise ValueError('Archived thread/determinism/TF32 settings changed')
        return self.value


@contextlib.contextmanager
def shared_runtime():
    original = final.exp.runtime
    session = RuntimeOnce(original)
    final.exp.runtime = session
    try:
        yield session
    finally:
        final.exp.runtime = original


def addendum(plan):
    expected = {'version': 1, 'original_plan': final.e.file_ref(final.OUT/'plan.json'),
        'original_plan_payload': final.e.digest(plan['frozen']),
        'recovery_source': final.e.file_ref(Path(__file__)),
        'policy': 'Initialize PyTorch once per extraction process; reuse device/runtime for both frozen encoders. Preserve seed123, threads2/interop1, deterministic execution, noTF32 and per-model precision.',
        'compatibility': 'Original sources, plan, numeric settings, tolerances and cache bindings are unchanged. No hash rebinding or skipped validation.',
        'scope': 'Verify existing heads and caches, extract missing test chunks and apply numeric heads. No train/model fitting/download/upload.'}
    if ADDENDUM.exists():
        if final.e.read_json(ADDENDUM) != expected: raise ValueError('Recovery runtime addendum changed')
    else: final.e.write_json(ADDENDUM, expected)
    return final.e.file_ref(ADDENDUM)


def verify_completed(state):
    data, features, raw, inner_s, _, _, plan = state
    result = {'model_alphas': {}, 'complete_test_encoders': [], 'pending_test_encoders': [],
              'preserved_files': {}, 'competition_fits': 0, 'competition_forwards': 0}
    for name in final.eng.NAMES:
        _, selection = final.load_head(final.OUT, plan, name, features[name], data.y, inner_s)
        result['model_alphas'][name] = selection['alpha']
        folder = final.OUT/'test_features'/name
        if (folder/'completed.json').exists():
            anchor = np.concatenate((raw[name][:final.ANCHORS], features[name][:final.ANCHORS]), axis=1)
            x = final.load_test_cache(final.OUT, plan, name, anchor)
            result['complete_test_encoders'].append(name)
            print(f'{name}: completed test{len(x)} verified; no forwards.', flush=True)
        else:
            # Validate every cached partial chunk with its original image binding.
            names = {f'{i:04d}.npz': i for i in range(len(plan['frozen']['test']['ids']))}
            for path in (folder/'chunks').glob('*.npz'):
                if path.name not in names: raise ValueError('Unexpected partial feature chunk')
                final.test_chunk(final.OUT, plan, name, names[path.name])
            if (folder/'anchor.npz').exists():
                anchor = np.concatenate((raw[name][:final.ANCHORS], features[name][:final.ANCHORS]), axis=1)
                final.anchor_record(final.OUT, plan, name, anchor)
            result['pending_test_encoders'].append(name)
    # Snapshot all existing artifacts except recovery receipts themselves.
    for path in final.OUT.rglob('*'):
        if path.is_file() and not path.is_relative_to(RECOVERY) and not path.name.endswith('.tmp'):
            result['preserved_files'][str(path)] = final.e.base.sha256_file(path)
    return result


def verify_unchanged(files):
    for path, digest in files.items():
        if final.e.base.sha256_file(path) != digest:
            raise ValueError('Previously saved artifact changed: '+path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('check', 'run'))
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    state = final.validate()
    evidence = addendum(state[-1]); before = verify_completed(state)
    receipt = RECOVERY/f'check_{time.time_ns()}.json'
    final.e.write_json(receipt, {'runtime_addendum': evidence, **before})
    print('Verified Ridge heads: '+str(before['model_alphas']), flush=True)
    if args.command == 'check':
        verify_unchanged(before['preserved_files'])
        print('Read-only recovery check complete; no fitting or model forwards.'); return
    with shared_runtime() as session:
        final.extract(device=args.device)
    final.predict()
    addendum(state[-1]); verify_unchanged(before['preserved_files'])
    final.e.write_json(RECOVERY/f'completed_{time.time_ns()}.json', {
        'runtime_addendum': evidence, 'before_check': final.e.file_ref(receipt),
        'runtime_initializations': session.calls, 'all_preexisting_artifacts_unchanged': True,
        'submission': final.e.file_ref(final.OUT/'submission/submission.csv'), 'new_regression_fits': 0})
    print('Recovery complete. Prior artifacts unchanged; separate D4 submission is ready.')


if __name__ == '__main__': main()
