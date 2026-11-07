"""Synthetic-only leakage, ablation, cache and user-run pipeline checks."""
from contextlib import contextmanager, redirect_stdout
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from scripts import evaluate_micronet_tabpfn as e


@contextmanager
def temporary():
    parent = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory(dir=parent, prefix='.micronet_tabpfn_test_') as name:
        if Path(name).resolve().parent != parent:
            raise ValueError('Cleanup path outside tests')
        yield Path(name)


def fixture():
    rng = np.random.default_rng(27)
    x = rng.normal(size=(500, 386))
    x[:, 0] = np.arange(500)
    raw = {'fixed386': x, 'micronet864': rng.normal(size=(500, 864))}
    raw['micronet864'][:, 0] = 0.
    data = SimpleNamespace(ids=np.array([f'TRAIN_{i:06d}' for i in range(500)]),
        y=100+3*x[:, 1], noise=np.arange(500.))
    base = data.y + 4 + x[:, 2]
    ref = {'folds': np.repeat(np.arange(5), 100), 'base': base, 'tabpfn486': data.y+2,
        'current486': .5*(base+data.y+2), 'runtime': {'synthetic': True}}
    plan = {'frozen': {'common_failure_ids': list(data.ids[:16]), 'synthetic': True}}
    return data, raw, ref, plan


class FakePredictor:
    calls = []

    def __init__(self, *args):
        self.runtime = {'synthetic': True}

    def __call__(self, x, y, q, seed):
        self.calls.append({'train_shape': x.shape, 'query_shape': q.shape, 'seed': seed,
            'train_ids': x[:, 0].copy(), 'query_ids': q[:, 0].copy(), 'train_y': y.copy()})
        # Uses only context targets and query features, not held-out targets.
        return 100+3*q[:, 1]+.005*y.mean()+.03*q[:, -1], {'seconds': 0.}


class MicroNetTabPFNTests(unittest.TestCase):
    def setUp(self):
        FakePredictor.calls = []

    def test_pca_matches_sklearn_and_is_query_independent(self):
        _, raw, _, _ = fixture()
        train, query = raw['micronet864'][100:], raw['micronet864'][:100]
        params = e.fit_pca(train)
        e.validate_pca(params, 400)
        with threadpool_limits(limits=2):
            scaler = StandardScaler().fit(train)
            model = PCA(n_components=64, svd_solver='full', whiten=False).fit(scaler.transform(train))
            expected = model.transform(scaler.transform(query))
        np.testing.assert_allclose(e.project(query, params), expected, atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(params['scaler_mean'], train.mean(0), atol=1e-14)
        self.assertEqual(params['scaler_scale'][0], 1.)
        changed = raw['micronet864'].copy()
        changed[:100] += 1e6
        again = e.fit_pca(changed[100:])
        for key in params:
            np.testing.assert_array_equal(params[key], again[key])
        self.assertGreater(np.linalg.norm(e.project(changed[:100], params)-expected), 1.)

    def test_fold_scope_validation_label_poisoning_and_no_double_blend(self):
        data, raw, ref, _ = fixture()
        with temporary() as folder, redirect_stdout(io.StringIO()):
            vi, pred, _ = e.evaluate_fold(folder, {}, raw, data.y, ref['folds'], 42, 0, FakePredictor(), True)
            poisoned = data.y.copy()
            poisoned[vi] += 1e6
            with patch.object(e, 'fit_pca', side_effect=AssertionError('PCA must be cached')):
                _, again, _ = e.evaluate_fold(folder, {}, raw, poisoned, ref['folds'], 42, 0, e.forbidden, False)
            for key in pred:
                np.testing.assert_array_equal(pred[key], again[key])
        self.assertEqual(len(FakePredictor.calls), 2)
        for call, width in zip(FakePredictor.calls, (386, 450)):
            self.assertEqual(call['train_shape'], (400, width))
            self.assertEqual(call['query_shape'], (100, width))
            self.assertEqual(call['seed'], 51)
            self.assertFalse(set(call['train_ids']) & set(call['query_ids']))
            np.testing.assert_array_equal(call['train_y'], data.y[call['train_ids'].astype(int)])

    def test_pca_cache_changed_inputs_and_corruption_rejected(self):
        _, raw, _, _ = fixture()
        ti, vi = np.arange(100, 500), np.arange(100)
        with temporary() as folder:
            e.pca_cell(folder, {}, raw['micronet864'], ti, vi, True)
            changed = raw['micronet864'].copy()
            changed[0, 5] += .01
            with self.assertRaisesRegex(ValueError, 'cache binding'):
                e.pca_cell(folder, {}, changed, ti, vi, False)
            with (folder/'parameters.npz').open('ab') as f:
                f.write(b'corrupt')
            with self.assertRaisesRegex(ValueError, 'parameters changed'):
                e.pca_cell(folder, {}, raw['micronet864'], ti, vi, False)
        with temporary() as folder, patch.object(e, 'fit_pca', side_effect=AssertionError('No fit')):
            with self.assertRaisesRegex(ValueError, 'implicit fitting'):
                e.pca_cell(folder, {}, raw['micronet864'], ti, vi, False)

    def test_partial_resume_reuses_pca_and_completed_control(self):
        data, raw, ref, _ = fixture()
        def interrupted(x, y, q, seed):
            if x.shape[1] == 450:
                raise RuntimeError('Synthetic interruption')
            return FakePredictor()(x, y, q, seed)
        with temporary() as folder, redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, 'interruption'):
                e.evaluate_fold(folder, {}, raw, data.y, ref['folds'], 42, 0, interrupted, True)
            self.assertFalse((folder/'fold0_tabpfn450/completed.json').exists())
            with patch.object(e, 'fit_pca', side_effect=AssertionError('No second PCA fit')):
                e.evaluate_fold(folder, {}, raw, data.y, ref['folds'], 42, 0, FakePredictor(), True)
            self.assertEqual([c['train_shape'][1] for c in FakePredictor.calls], [386, 450])
            changed = data.y.copy()
            changed[100] += 1
            with self.assertRaisesRegex(ValueError, 'Completed cell changed'):
                e.evaluate_fold(folder, {}, raw, changed, ref['folds'], 42, 0, e.forbidden, False)

    def test_blend_formula_group_partition_and_standalone_comparison(self):
        data, _, ref, plan = fixture()
        predictions = {'tabpfn386': data.y+1, 'tabpfn450': data.y+.5}
        metrics, rows, diag, errors, detail = e.result_data(data, ref, predictions, plan['frozen']['common_failure_ids'])
        expected = .5*(ref['base']+predictions['tabpfn450'])
        np.testing.assert_array_equal([r[e.PRIMARY] for r in rows], expected)
        self.assertFalse(np.allclose(expected, .5*(ref['current486']+predictions['tabpfn450'])))
        groups = {r['group']: r for r in diag if r['model'] == e.PRIMARY}
        self.assertAlmostEqual(groups['common16']['sse']+groups['remaining484']['sse'], metrics[e.PRIMARY]['sse'])
        self.assertAlmostEqual(groups['middle80']['sse']+groups['tails20']['sse'], metrics[e.PRIMARY]['sse'])
        self.assertEqual(sum(r['common16'] for r in errors), 16)
        self.assertAlmostEqual(detail['standalone_gain_vs486'], 1.5)
        self.assertAlmostEqual(detail['standalone_gain_vs386'], .5)

    def test_fixed_twenty_contexts_end_to_end_read_only_resume_and_compare(self):
        data, raw, ref, plan = fixture()
        refs = {s: ref for s in e.SEEDS}
        with temporary() as folder, redirect_stdout(io.StringIO()), \
                patch.object(e, 'validate', return_value=(data, raw, refs, plan)), \
                patch.object(e.t, 'verify_weights', return_value=Path('synthetic.ckpt')), \
                patch.object(e.t, 'Predictor', FakePredictor):
            for seed in e.SEEDS:
                e.evaluate(folder, 'synthetic', seed)
            self.assertEqual(len(FakePredictor.calls), 20)
            self.assertEqual([c['train_shape'][1] for c in FakePredictor.calls], [386, 450]*10)
            for n, call in enumerate(FakePredictor.calls):
                self.assertEqual(call['seed'], e.SEEDS[n//10] + 1009*((n%10)//2) + 9)
            with patch.object(e, 'fit_pca', side_effect=AssertionError('No fit in verification')), \
                    patch.object(e.t, 'Predictor', side_effect=AssertionError('No model in verification')):
                for seed in e.SEEDS:
                    e.evaluate(folder, 'synthetic', seed)
                result = e.compare(folder)
            self.assertEqual(result['primary'], e.PRIMARY)
            self.assertEqual(len(result['per_seed']), 2)
            self.assertEqual(len(FakePredictor.calls), 20)
            summary = e.t.read_json(folder/'seed42/summary.json')
            self.assertFalse(summary['inner_search'])
            self.assertEqual(summary['archived486_context_fits_reused'], 5)
            with (folder/'seed42/oof.csv').open('a') as f:
                f.write('invalid')
            with self.assertRaises(ValueError):
                e.compare(folder)

    def test_runtime_mismatch_stops_before_any_competition_fit(self):
        data, raw, ref, plan = fixture()
        ref['runtime'] = {'different': True}
        with temporary() as folder, patch.object(e, 'validate', return_value=(data, raw, {42: ref}, plan)), \
                patch.object(e.t, 'verify_weights', return_value=Path('synthetic.ckpt')), \
                patch.object(e.t, 'Predictor', FakePredictor), \
                patch.object(e, 'fit_pca', side_effect=AssertionError('No PCA fit')):
            with self.assertRaisesRegex(ValueError, 'exact archived runtime'):
                e.evaluate(folder, 'synthetic', 42)
        self.assertEqual(FakePredictor.calls, [])

    def test_numerical_gate_does_not_depend_on_common16(self):
        records = [{'seed': seed, 'gain_vs_current486': .31, 'gain_vs_control386': .10,
                    'diagnosis': {'common16_improved': False}} for seed in e.SEEDS]
        self.assertTrue(e.aggregate(records)['passes_expansion_priority'])
        records[1]['gain_vs_control386'] = -.01
        self.assertFalse(e.aggregate(records)['all_seeds_improve_both'])

    def test_invalid_shape_and_overlap_rejected(self):
        _, raw, _, _ = fixture()
        with self.assertRaises(ValueError):
            e.fit_pca(raw['micronet864'][:64])
        with temporary() as folder:
            with self.assertRaises(ValueError):
                e.pca_cell(folder, {}, raw['micronet864'], np.arange(400), np.arange(100), True)


if __name__ == '__main__':
    unittest.main()
