"""Synthetic CPU tests; never fit or extract from competition images."""
from contextlib import contextmanager, redirect_stdout
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from scripts import efficientnet_micronet_experiment as e


@contextmanager
def temporary():
    parent = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory(dir=parent, prefix='.micronet_test_') as location:
        folder = Path(location).resolve()
        if folder.parent != parent:
            raise ValueError('Cleanup target must stay within tests')
        yield folder


def fixture(width=6):
    rng = np.random.default_rng(20261013)
    x = rng.normal(size=(500, width))
    y = 200 + 2*x[:, 0] + rng.normal(size=500)
    data = SimpleNamespace(ids=np.array([f'TRAIN_{i:06d}' for i in range(500)]), y=y, noise=x[:, 1])
    ref = {'current': y + 4, 'folds': np.repeat(np.arange(5), 100)}
    plan = {'frozen': {'common_failure_ids': data.ids[:16].tolist()}}
    return data, {n: x.copy() for n in e.REPS}, ref, plan


class MicroNetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_real_pinned_backbones_match_upstream_endpoints_and_do_not_update(self):
        torch.manual_seed(4)
        x = torch.randn(1, 3, 256, 256)
        hashes = []
        features = []
        for name in e.REPS:
            model = e.load_model(name)
            before = e.base.feature_state_hash(model)
            with torch.inference_mode():
                maps = e.forward_maps(model, x)
                upstream = model.extract_endpoints(x)
                for actual, key in zip(maps, ('reduction_4', 'reduction_5')):
                    torch.testing.assert_close(actual, upstream[key], rtol=0, atol=0)
                pooled = e.pool_maps(maps)
            self.assertEqual(pooled.shape, (1, 864))
            torch.testing.assert_close(pooled[:, :112], maps[0].mean((2, 3)), rtol=0, atol=0)
            torch.testing.assert_close(pooled[:, 112:224], maps[0].std((2, 3), correction=0), rtol=0, atol=0)
            self.assertEqual(before, e.base.feature_state_hash(model))
            self.assertFalse(any(p.requires_grad for p in model.parameters()))
            self.assertFalse(any(m.training for m in model.modules()))
            hashes.append(before)
            features.append(pooled)
        self.assertNotEqual(*hashes)
        self.assertFalse(torch.equal(*features))

    def test_missing_encoder_key_is_not_silently_ignored(self):
        from safetensors.torch import load_file
        state = load_file(str(e.assets.asset_path('imagenet')))
        state.pop('_conv_stem.weight')
        with patch('safetensors.torch.load_file', return_value=state):
            with self.assertRaises(RuntimeError):
                e.load_model('imagenet')

    def test_native_input_and_population_moments(self):
        maps = [torch.full((2, 112, 16, 16), 3.), torch.full((2, 320, 8, 8), -2.)]
        result = e.pool_maps(maps)
        self.assertTrue(torch.all(result[:, :112] == 3))
        self.assertTrue(torch.all(result[:, 112:224] == 0))
        self.assertTrue(torch.all(result[:, 224:544] == -2))
        self.assertTrue(torch.all(result[:, 544:] == 0))
        with self.assertRaises(ValueError):
            e.forward_maps(None, torch.zeros(1, 3, 224, 224))

    def test_outer_target_poison_cannot_change_selection_or_predictions(self):
        data, x, ref, _ = fixture()
        changed = data.y.copy()
        changed[:100] += 100000
        with threadpool_limits(limits=2):
            original = e.fit_fold(x, data.y, ref['folds'], 42, 0)
            poisoned = e.fit_fold(x, changed, ref['folds'], 42, 0)
        self.assertEqual(original, poisoned)

    def test_fit_count_train_subset_isolation_pooled_selection_and_tie_rule(self):
        data, x, ref, _ = fixture()
        calls = []
        for name in x:
            x[name][:100] = 100000  # a forbidden query marker
        class Constant:
            def fit(self, features, targets):
                self_outer.assertFalse(np.any(features == 100000))
                self_outer.assertTrue(set(targets).issubset(set(data.y[100:])))
                calls.append(len(targets))
                return self
            def predict(self, features):
                return np.full(len(features), 201.)
        self_outer = self
        result = e.fit_fold(x, data.y, ref['folds'], 42, 0, factory=lambda a, n: Constant())
        self.assertEqual(len(calls), 26)
        self.assertEqual(calls.count(400), 2)
        self.assertEqual(set(calls), {266, 267, 400})
        expected = float(np.mean((data.y[100:]-201)**2))
        for name in e.REPS:
            self.assertEqual(result['selection'][name]['alpha'], 3000)
            for candidate in result['candidates'][name]:
                self.assertAlmostEqual(candidate['inner_mse'], expected)
        model = e.core.make_estimator(100).fit(x['imagenet'][100:], data.y[100:])
        np.testing.assert_allclose(model.regressor_.steps[0][1].mean_, x['imagenet'][100:].mean(0))
        np.testing.assert_allclose(model.transformer_.mean_, [data.y[100:].mean()])

    def test_fold_resume_rejects_changed_binding_or_predictions(self):
        data, x, ref, _ = fixture()
        with temporary() as folder, threadpool_limits(limits=2):
            saved = e.fold_cell(folder, {'seed': 42}, x, data.y, ref['folds'], 42, 0)
            self.assertEqual(saved, e.fold_cell(folder, {'seed': 42}, x, data.y, ref['folds'], 42, 0, runner=e.unavailable))
            with self.assertRaises(ValueError):
                e.fold_cell(folder, {'seed': 2026}, x, data.y, ref['folds'], 42, 0, runner=e.unavailable)
            row = e.t.read_json(folder / 'fold0.json')
            row['result']['predictions']['micronet'][0] += 1
            e.t.write_json(folder / 'fold0.json', row)
            with self.assertRaises(ValueError):
                e.fold_cell(folder, {'seed': 42}, x, data.y, ref['folds'], 42, 0, runner=e.unavailable)

    def test_features_require_correct_ids_dtype_scope_hash_and_plan(self):
        data, _, _, plan = fixture()
        arrays = {'ids': data.ids, **{n: np.zeros((500, 864), np.float32) for n in e.REPS}}
        with temporary() as folder:
            path = folder / 'features/features.npz'
            with self.assertRaisesRegex(ValueError, 'Complete extract'):
                e.validate_feature_bundle(path, data, plan)
            path.parent.mkdir()
            def save():
                np.savez(path, **arrays)
                meta = {'plan': e.t.digest(plan['frozen']), 'file_sha256': e.base.sha256_file(path),
                        'arrays': {n: e.base.sha256_array(x) for n, x in arrays.items()},
                        'test_accessed': False, 'labels_used_for_features': False}
                e.t.write_json(path.with_name('metadata.json'), meta)
            save()
            e.validate_feature_bundle(path, data, plan)
            arrays['micronet'][0, 0] = 1
            np.savez(path, **arrays)
            with self.assertRaises(ValueError):
                e.validate_feature_bundle(path, data, plan)
            save()
            with self.assertRaises(ValueError):
                e.validate_feature_bundle(path, data, {'frozen': {'changed': True}})
            arrays['ids'] = data.ids[::-1]
            save()
            with self.assertRaises(AssertionError):
                e.validate_feature_bundle(path, data, plan)

    def test_blend_and_diagnostics_include_current486_and_common16(self):
        data, _, ref, plan = fixture()
        predictions = {'imagenet': data.y + 2, 'micronet': data.y.copy()}
        metrics, rows, diag, errors, detail = e.result_data(data, ref, predictions, plan['frozen']['common_failure_ids'])
        self.assertEqual(metrics[e.CONTROL]['rmse'], 3.)
        self.assertEqual(metrics[e.PRIMARY]['rmse'], 2.)
        self.assertEqual(metrics[e.PRIMARY]['gain_vs_current'], 2.)
        self.assertEqual(metrics[e.PRIMARY]['gain_vs_control'], 1.)
        self.assertEqual(detail['common16_improved_vs_current'], 16)
        self.assertEqual(len(rows), 500)
        self.assertEqual(sum(r['common16'] for r in errors), 16)
        counts = {r['group']: r['n'] for r in diag if r['model'] == e.PRIMARY}
        self.assertEqual(counts['remaining484'], 484)
        self.assertEqual(counts['middle80'], 400)
        self.assertEqual(counts['tails20'], 100)

    def test_artifact_completion_and_changed_report_rejected(self):
        data, _, ref, plan = fixture()
        predictions = {'imagenet': data.y + 2, 'micronet': data.y.copy()}
        with temporary() as folder:
            for f in range(5):
                e.t.write_json(folder / f'fold{f}.json', {'synthetic': True})
            with redirect_stdout(io.StringIO()):
                e.save_results(folder, data, ref, predictions, [], {'seed': 42}, plan)
            self.assertTrue(e.completed(folder, {'seed': 42}))
            (folder / 'common16.csv').write_text('changed', encoding='utf8')
            with self.assertRaises(ValueError):
                e.completed(folder, {'seed': 42})

    def test_gate_requires_both_seeds_both_controls_and_mean_not_prediction_average(self):
        rows = [{'seed': seed, 'gain_vs_current': .4, 'gain_vs_control': .1} for seed in e.SEEDS]
        self.assertTrue(e.aggregate(rows)['passes_expansion_priority'])
        rows[1]['gain_vs_control'] = -.01
        result = e.aggregate(rows)
        self.assertEqual(result['mean_gain_vs_current'], .4)
        self.assertFalse(result['passes_expansion_priority'])


if __name__ == '__main__':
    unittest.main()
