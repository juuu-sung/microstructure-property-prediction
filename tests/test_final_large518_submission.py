"""Synthetic-only tests. Never fit competition data or load a vision checkpoint."""
from contextlib import contextmanager, redirect_stdout
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from threadpoolctl import threadpool_limits
from scripts import final_large518_submission as f


@contextmanager
def temporary():
    with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent, prefix='.large_final_') as p:
        yield Path(p)


def fixture(width=12, tests=1000):
    rng = np.random.default_rng(1234)
    x = rng.normal(size=(500, width)).astype(np.float32)
    x[:, 0] = np.arange(500)
    x[:, -1] = 1
    q = rng.normal(size=(tests, width)).astype(np.float32)
    q[:, 0] = np.arange(500, 500+tests)
    y = 190 + 3*x[:, 1].astype(np.float64)+rng.normal(size=500)
    data = SimpleNamespace(ids=np.array([f'TRAIN_{i+1:06d}' for i in range(500)]), y=y)
    ids = [f'TEST_{i+1:06d}' for i in range(tests)][::-1]
    runtime = {'device': 'cuda:0', 'gpu': 'synthetic', 'batch_size': 1, 'precision': 'float32', 'amp': False}
    plan = {'frozen': {'synthetic': True, 'train_runtime': runtime, 'anchor_ids': data.ids[:16].tolist(),
                      'test': {'ids': ids, 'rows': tests, 'images': []}}}
    previous = np.linspace(180., 210., tests)
    return data, x, q, previous, plan


def add_test_files(root, plan):
    (root/'reports').mkdir(exist_ok=True)
    for ident in plan['frozen']['test']['ids']:
        path = root/'open/test'/(ident+'.png')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(ident.encode())
        plan['frozen']['test']['images'].append({'id': ident, 'relative_path': 'test/'+ident+'.png', 'sha256': f.e.base.sha256_file(path)})


class FinalLargeTests(unittest.TestCase):
    def setUp(self):
        self.limit = threadpool_limits(limits=2); self.limit.__enter__()

    def tearDown(self):
        self.limit.__exit__(None, None, None)

    def test_full500_fit_export_predict1000_fixed_blend_and_resume(self):
        from sklearn.linear_model import Ridge
        data, x, q, previous, plan = fixture()
        fitted = []
        fit = Ridge.fit
        def traced(model, a, b, *args, **kwargs):
            fitted.append(len(a))
            return fit(model, a, b, *args, **kwargs)
        with temporary() as root, redirect_stdout(io.StringIO()):
            folder = root/'candidate'
            prior = root/'prior.csv'; prior.write_text('preserved submission', encoding='utf-8')
            before = prior.read_bytes()
            with patch.object(f, 'validate', return_value=(data, x, previous, plan)), patch.object(Ridge, 'fit', traced), \
                 patch.object(f, 'LargeExtractor', side_effect=AssertionError('Training must not run vision')):
                f.train(folder)
            self.assertEqual(len(fitted), 10)
            self.assertEqual(fitted.count(500), 1)
            self.assertEqual(set(fitted), {333, 334, 500})
            head, selection = f.load_head(folder, plan, data, x)
            np.testing.assert_allclose(head['x_mean'], x.astype(np.float64).mean(0), rtol=0, atol=1e-12)
            self.assertEqual(head['x_scale'][-1], 1.)
            reference = f.core.make_estimator(selection['alpha']).fit(x.astype(np.float64), data.y)
            np.testing.assert_allclose(f.shared.predict_head(head, q), reference.predict(q.astype(np.float64)), rtol=0, atol=1e-10)
            (folder/'test_features').mkdir()
            f.e.write_json(folder/'test_features/completed.json', {'synthetic': True})
            with patch.object(f, 'validate', return_value=(data, x, previous, plan)), \
                 patch.object(f, 'load_test_cache', return_value=q), patch.object(Ridge, 'fit', side_effect=AssertionError('No refitting')):
                f.train(folder)
                f.predict(folder)
                before_new = (folder/'submission/submission.csv').read_bytes()
                f.predict(folder)
            self.assertEqual(before_new, (folder/'submission/submission.csv').read_bytes())
            ids = plan['frozen']['test']['ids']
            expected = .5*previous+.5*f.shared.predict_head(head, q)
            f.shared.verify_submission(folder/'submission/submission.csv', ids, expected)
            self.assertEqual(before, prior.read_bytes())
            self.assertEqual(selection['n_train'], 500)
            self.assertFalse(selection['test_used_for_fit'])
            self.assertFalse(f.e.read_json(folder/'submission/validation.json')['uploaded'])
            with (folder/'model/metadata.json').open('a') as h: h.write('tamper')
            with self.assertRaises(ValueError): f.load_head(folder, plan, data, x)

    def test_selection_context_only_and_tie_rule_and_missing_cache(self):
        data, x, _, _, plan = fixture()
        calls = []
        def runner(tx, y, qx, indices):
            train_ids, query_ids = tx[:, 0].astype(int), qx[:, 0].astype(int)
            self.assertFalse(set(train_ids)&set(query_ids))
            np.testing.assert_array_equal(y, data.y[train_ids])
            self.assertEqual(tuple(indices), f.c.RIDGE_INDICES)
            calls.append(query_ids)
            return {'predictions': {str(i): np.full(len(qx), np.mean(y)).tolist() for i in indices}}
        with temporary() as p:
            value = f.select_alpha(p, {}, x, data.y, runner=runner)
            self.assertEqual(value['alpha'], 10000.)
            np.testing.assert_array_equal(np.sort(np.concatenate(calls)), np.arange(500))
            cached = f.select_alpha(p, {}, x, data.y, allow_fit=False, runner=f.c.forbidden)
            self.assertEqual(value, cached)
            with self.assertRaises(FileNotFoundError): f.select_alpha(p/'missing', {}, x, data.y, allow_fit=False)
            with self.assertRaises(ValueError): f.select_alpha(p, {'changed': True}, x, data.y, allow_fit=False)

    def test_scattering_source_is_entire_submitted_blend_with_sample_order(self):
        ids = ['TEST_000002', 'TEST_000001']
        base, tab = np.array([180., 200.]), np.array([200., 220.])
        values = .5*(base+tab)
        with temporary() as root:
            p = root/'old'; output = p/'submission'; output.mkdir(parents=True)
            plan = {'frozen': {'synthetic': True}}
            f.e.write_json(p/'plan.json', plan)
            path = output/'submission.csv'; path.write_text(f.shared.submission_text(ids, values), encoding='utf-8')
            sha = f.e.base.sha256_file(path)
            f.e.write_csv(output/'component_predictions.csv', [{'ID': i, 'base': base[j], 'tabpfn486': tab[j], 'hardness': values[j]} for j, i in enumerate(ids)])
            f.e.write_json(output/'validation.json', {'submission_sha256': sha, 'weights': {'base_hand185_resnet_dino': .5, 'tabpfn486': .5}})
            f.finish(output, {'plan': f.e.digest(plan['frozen']), 'stage': 'fixed486_full500_test1000'}, ('submission.csv', 'component_predictions.csv', 'validation.json'))
            with patch.object(f, 'SCATTERING', p), patch.object(f, 'SCATTERING_SHA', sha):
                actual, refs = f.read_scattering(ids)
                np.testing.assert_array_equal(actual, values)
                self.assertFalse(np.array_equal(actual, base))
                self.assertEqual(refs['submission.csv']['sha256'], sha)
                with self.assertRaises(ValueError): f.read_scattering(ids[::-1])
                path.write_text('changed', encoding='utf-8')
                with self.assertRaises(ValueError): f.read_scattering(ids)

    def test_anchor_failure_prevents_all_test_forwards(self):
        data, x, _, previous, plan = fixture(f.WIDTH, 3)
        calls = []
        class Fake:
            runtime = plan['frozen']['train_runtime']
            def __call__(self, split, ids):
                calls.append(split)
                return x[:len(ids)]+1
            def close(self): pass
        with temporary() as p, patch.object(f, 'validate', return_value=(data, x, previous, plan)), \
             patch.object(f, 'load_head', return_value=({}, {})), patch.object(f, 'LargeExtractor', return_value=Fake()):
            with self.assertRaisesRegex(ValueError, 'anchor mismatch'): f.extract(p)
            self.assertEqual(calls, ['train'])
            self.assertFalse((p/'test_features/anchor.npz').exists())
            self.assertFalse((p/'test_features/features.npz').exists())

    def test_extraction_interruption_resume_ids_chunks_and_no_rerun(self):
        data, x, q, previous, plan = fixture(f.WIDTH, 7)
        calls = []; failures = [True]
        class Fake:
            runtime = plan['frozen']['train_runtime']
            def __call__(self, split, ids):
                calls.append((split, list(ids)))
                if split == 'train':
                    return x[:len(ids)].copy()
                j = plan['frozen']['test']['ids'].index(ids[0])
                if j == 2 and failures[0]:
                    failures[0] = False
                    raise RuntimeError('Synthetic interruption')
                return q[j:j+1].copy()
            def close(self): pass
        with temporary() as root, redirect_stdout(io.StringIO()):
            add_test_files(root, plan); folder = root/'candidate'
            with patch.object(f, 'ROOT', root), patch.object(f, 'validate', return_value=(data, x, previous, plan)), \
                 patch.object(f, 'load_head', return_value=({}, {})), patch.object(f, 'LargeExtractor', return_value=Fake()):
                with self.assertRaises(RuntimeError): f.extract(folder)
                self.assertFalse((folder/'test_features/completed.json').exists())
                f.extract(folder)
                actual = f.load_test_cache(folder, plan, x)
                np.testing.assert_array_equal(actual, q)
                self.assertEqual(sum(split == 'train' for split, _ in calls), 1)
                self.assertEqual(sum(split == 'test' for split, _ in calls), 8)
                with patch.object(f, 'LargeExtractor', side_effect=AssertionError('Completed cache must not load model')):
                    f.extract(folder)
                with self.assertRaises(ValueError): f.load_test_cache(folder, plan, x+1)
                image = root/'open'/plan['frozen']['test']['images'][0]['relative_path']
                image.write_bytes(b'changed')
                with self.assertRaises(ValueError): f.load_test_cache(folder, plan, x)

    def test_chunk_tampering_and_invalid_features_rejected(self):
        _, _, q, _, plan = fixture(f.WIDTH, 2)
        with temporary() as root:
            add_test_files(root, plan)
            with patch.object(f, 'ROOT', root):
                folder = root/'candidate'
                with self.assertRaises(ValueError): f.test_chunk(folder, plan, 0, lambda *a: q[:1].astype(np.float64))
                with self.assertRaises(ValueError): f.test_chunk(folder, plan, 0, lambda *a: q[:1]*np.nan)
                with self.assertRaises(FileNotFoundError): f.test_chunk(folder, plan, 0)
                f.test_chunk(folder, plan, 0, lambda *a: q[:1])
                path = folder/'test_features/chunks/0000.npz'
                with np.load(path, allow_pickle=False) as a: payload = {k: a[k] for k in a.files}
                payload['x'][0, 0] += 1
                np.savez_compressed(path, **payload)
                with self.assertRaises(ValueError): f.test_chunk(folder, plan, 0)

    def test_destination_prior_preservation_output_integrity_and_wrong_inputs(self):
        for p in (f.SCATTERING, f.SCATTERING/'submission', f.c.OUT, f.ROOT/'open', f.ROOT/'models', f.ROOT):
            with self.assertRaises(ValueError): f.protect_destination(p)
        with temporary() as p, redirect_stdout(io.StringIO()):
            ids = ['TEST_000003', 'TEST_000001']
            old, new = np.array([200., 180.]), np.array([210., 170.])
            for bad_ids, a in (([ids[0], ids[0]], new), (ids, np.array([np.nan, 3.])), (ids, np.array([4.]))):
                with self.assertRaises(ValueError): f.write_submission(p/'bad', bad_ids, old, a, {})
            f.write_submission(p/'out', ids, old, new, {'fixed': True})
            f.shared.verify_submission(p/'out/submission.csv', ids, .5*old+.5*new)
            with self.assertRaises(ValueError): f.write_submission(p/'out', ids, old, new, {'fixed': False})
            with (p/'out/component_predictions.csv').open('a') as h: h.write('tamper')
            with self.assertRaises(ValueError): f.write_submission(p/'out', ids, old, new, {'fixed': True})

    def test_prepare_is_readonly_and_source_change_blocked(self):
        data, x, _, previous, plan = fixture()
        with temporary() as root, redirect_stdout(io.StringIO()):
            folder = root/'candidate'
            with patch.object(f, 'load_state', return_value=(data, x, previous, plan['frozen'])), \
                 patch.object(f.eng, 'fit_candidates', side_effect=AssertionError('No fitting')), \
                 patch.object(f, 'LargeExtractor', side_effect=AssertionError('No vision')):
                f.prepare(folder); before = (folder/'plan.json').read_bytes()
                f.prepare(folder); f.validate(folder)
                self.assertEqual(before, (folder/'plan.json').read_bytes())
            with patch.object(f, 'load_state', return_value=(data, x, previous, {'changed': True})):
                with self.assertRaises(ValueError): f.validate(folder)

    def test_driver_stages_and_fixed_scope(self):
        text = (f.ROOT/'scripts/run_final_large518.ps1').read_text()
        self.assertIn('$LASTEXITCODE -ne 0', text)
        for stage in ('prepare', 'train', 'extract', 'predict'):
            self.assertIn("'"+stage+"'", text)
        self.assertEqual(f.PROTOCOL['alpha_candidates'], [100., 1000., 10000.])
        self.assertEqual(f.PROTOCOL['blend'], {'preserved_scattering486': .5, 'large518_ridge': .5})
        self.assertEqual(sum(f.PROTOCOL['effective_weights'].values()), 1.)


if __name__ == '__main__': unittest.main()
