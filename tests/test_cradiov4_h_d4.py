"""Synthetic/meta-only H checks: no downloads or competition forwards/fits."""
import contextlib
import inspect
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
from scripts import cradiov4_h_engine as eng
from scripts import cradiov4_h_d4_experiment as exp


class HTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def test_official_native_architecture_meta_only_and_environment_stable(self):
        before, path = exp.e.environment(), list(sys.path)
        with patch('torch.hub.load', side_effect=AssertionError('No hub download')):
            model = eng.architecture(meta=True)
        self.assertTrue(all(p.is_meta for p in model.parameters()))
        self.assertEqual(sum(p.numel() for p in model.parameters()), 651645440)
        self.assertEqual((model.summary_dim, model.embed_dim, model.num_summary_tokens), (2560, 1280, 10))
        self.assertEqual(list(sys.path), path)
        self.assertEqual(exp.e.environment(), before)

    def test_h_pooling_population_std_and_prefix_exclusion(self):
        summary = torch.arange(eng.SUMMARY, dtype=torch.float32)[None]
        spatial = torch.arange(37*37, dtype=torch.float32)[None, :, None].expand(1, -1, eng.SPATIAL)
        x = eng.pool((summary, spatial))
        self.assertEqual(x.shape, (1, 5120))
        torch.testing.assert_close(x[:, :2560], summary)
        torch.testing.assert_close(x[:, 2560:3840], spatial.mean(1))
        torch.testing.assert_close(x[:, 3840:], spatial.std(1, correction=0))
        with self.assertRaises(ValueError): eng.pool((summary, torch.zeros(1, 1379, 1280)))
        with self.assertRaises(ValueError): eng.pool((summary[:, :-1], spatial))

    def test_h_d4_eight_views_and_replace_only_radio_quarter(self):
        raw = torch.arange(9, dtype=torch.float32).reshape(1, 1, 3, 3)
        calls = []
        def fake(name, model, x):
            calls.append(1)
            return x[:, 0, 0, 0, None].expand(-1, eng.WIDTH)
        with patch.object(eng, 'forward', side_effect=fake):
            identity, mean = eng.extract_views(None, raw)
        self.assertEqual(len(calls), 8)
        np.testing.assert_array_equal(identity, np.zeros((1, 5120)))
        np.testing.assert_array_equal(mean, np.full((1, 5120), 4.))
        s = np.arange(10, dtype=float); p = eng.compose(s, s+1, s+2, s+6)
        np.testing.assert_allclose(p[exp.PRIMARY]-p['current'], 1.)

    def test_context_only_selection_cache_resume_and_tamper(self):
        self.assertNotIn('query_y', inspect.signature(exp.fit_fold).parameters)
        x = np.arange(500, dtype=float)[:, None]; y = x[:400, 0]+1000
        calls = []
        def runner(ctx, target, query, indices):
            np.testing.assert_array_equal(target, ctx[:, 0]+1000)
            self.assertFalse(set(ctx[:, 0]) & set(query[:, 0]))
            self.assertLessEqual(len(ctx), 400)
            calls.append((len(ctx), len(query)))
            return {'predictions': {str(i): np.full(len(query), target.mean()+i).tolist() for i in indices}}
        with tempfile.TemporaryDirectory(dir=exp.ROOT/'.task-tmp', prefix='h-fold-') as tmp:
            args = (Path(tmp), {'synthetic': True}, x[:400], y, x[400:], np.zeros(400), 42)
            result, choice = exp.fit_fold(*args, runner=runner)
            self.assertEqual(len(calls), 4)
            self.assertEqual(sum(n == 400 for n, _ in calls), 1)
            with patch.object(eng.d4.ridge, 'fit_candidates', side_effect=AssertionError('Unexpected fitting')):
                replay, same_choice = exp.fit_fold(*args, allow_fit=False)
            np.testing.assert_array_equal(result, replay); self.assertEqual(choice, same_choice)
            self.assertEqual(choice['objective'], '.5S_inner+.5C_inner')
            p = Path(tmp)/'inner0.json'; obj = exp.e.read_json(p)
            obj['result']['predictions']['0'][0] += 1; exp.e.write_json(p, obj)
            with self.assertRaises(ValueError): exp.fit_fold(*args, allow_fit=False)

    def test_two_seed_exports_diagnostics_readonly_replay_and_comparison(self):
        rng = np.random.default_rng(6)
        ids = np.array([f'SYNTH_{i:06d}' for i in range(500)])
        data = SimpleNamespace(ids=ids, y=190+rng.normal(size=500)*15, noise=rng.uniform(size=500))
        folds = np.arange(500) % 5; s = np.full(500, 190.); l = s+1; c = s+2
        ref = {'folds': folds, 'scattering': s, 'large_d4': l, 'radio_d4': c,
               'current': .5*s+.25*l+.25*c, 'inner_scattering': {f: np.full(400, 190.) for f in range(5)}}
        features = rng.normal(size=(500, 3)).astype(np.float32)
        plan = {'frozen': {'common_failure_ids': ids[:16].tolist()}}
        calls = []
        def runner(ctx, target, query, indices):
            calls.append(1)
            return {'predictions': {str(i): np.full(len(query), target.mean()+i*.1).tolist() for i in indices}}
        with tempfile.TemporaryDirectory(dir=exp.ROOT/'.task-tmp', prefix='h-eval-') as tmp:
            state = (data, None, {s:ref for s in exp.SEEDS}, plan)
            with patch.object(exp, 'OUT', Path(tmp)), patch.object(exp, 'validate', return_value=state), \
                 patch.object(exp, 'load_features', return_value=(features, {'synthetic': True})), \
                 patch.object(eng.d4.ridge, 'fit_candidates', side_effect=runner), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(FileNotFoundError): exp.evaluate(42, allow_fit=False)
                self.assertEqual(len(calls), 0)
                first = [exp.evaluate(s) for s in exp.SEEDS]
                self.assertEqual(len(calls), 40)
                second = [exp.evaluate(s, allow_fit=False) for s in exp.SEEDS]
                self.assertEqual(first, second); self.assertEqual(len(calls), 40)
                d = first[0]['diagnosis'][exp.PRIMARY]
                self.assertEqual(d['groups']['common16']['n'], 16)
                self.assertEqual(d['groups']['remaining484']['n'], 484)
                result = exp.compare(state, second)
                self.assertFalse(result['automatic_adoption'])
                exp.compare(); self.assertEqual(len(calls), 40)
                with (Path(tmp)/'seed42/oof.csv').open() as f: self.assertIn(exp.PRIMARY, f.readline())
                p = Path(tmp)/'seed42/report.txt'; p.write_text('tampered')
                with self.assertRaises(ValueError): exp.evaluate(42, allow_fit=False)


if __name__ == '__main__': unittest.main()
