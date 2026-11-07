"""Synthetic checks for D4 grouping, selection, leakage boundaries and resume."""
import contextlib
import inspect
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
from scripts import large_radio_d4_engine as eng
from scripts import large_radio_d4_experiment as exp


class D4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_eight_pixel_permutations_and_image_level_pooling(self):
        raw = torch.arange(9, dtype=torch.float32).reshape(1, 1, 3, 3)
        views = list(eng.views(raw))
        self.assertEqual(len({v.numpy().tobytes() for v in views}), 8)
        for v in views:
            torch.testing.assert_close(v.sort(dim=-1).values.flatten().sort().values, raw.flatten().sort().values)
        torch.testing.assert_close(views[0], raw)
        calls = []
        def fake(name, model, x):
            calls.append(x.clone())
            return torch.stack([x[:, 0, 0, 0], x[:, 0, 0, 1]], 1)
        first, mean = eng.extract_views('large', None, raw, fake)
        self.assertEqual(len(calls), 8)
        self.assertEqual(mean.shape, (1, 2))
        np.testing.assert_array_equal(first, [[0, 1]])
        np.testing.assert_array_equal(mean, [[4, 4]])
        # Averaging the exact group is invariant to rotating/flipping the source.
        _, rotated = eng.extract_views('large', None, torch.rot90(raw, 1, (-2, -1)), fake)
        np.testing.assert_array_equal(mean, rotated)

    def test_preserved_distinct_alpha_objectives_and_tie(self):
        y = np.arange(6, dtype=float)
        inner = {'0': y.copy(), '1': y-2, '2': y-1}
        scattering = y+2
        self.assertEqual(eng.choose('large', inner, y, scattering)['selected']['index'], 0)
        self.assertEqual(eng.choose('radio', inner, y, scattering)['selected']['index'], 1)
        same = {str(i): y.copy() for i in eng.INDICES}
        self.assertEqual(eng.choose('radio', same, y, scattering)['selected']['alpha'], 10000)
        self.assertNotIn('query_y', inspect.signature(exp.fit_fold).parameters)

    def test_composition_replaces_only_fixed_quarters(self):
        a = np.arange(10, dtype=float)
        p = eng.compose(a, a+1, a+2, a+3, a+4)
        np.testing.assert_allclose(p['large_d4_blend']-p['current'], .5)
        np.testing.assert_allclose(p['radio_d4_blend']-p['current'], .5)
        np.testing.assert_allclose(p[eng.PRIMARY]-p['current'], 1.)

    def test_anchor_failure_and_payload_tamper_rejected(self):
        with tempfile.TemporaryDirectory(dir=exp.ROOT/'.task-tmp', prefix='d4-cache-') as tmp:
            p = Path(tmp)/'f.npz';ids = np.array(['syntheticA', 'syntheticB'])
            x = np.arange(12, dtype=np.float32).reshape(2, 6)
            exp.e.npz_save(p, ids, x, {'stage': 'synthetic'})
            result, _ = exp.e.npz_read(p, ids, 6, {'stage': 'synthetic'})
            np.testing.assert_array_equal(result, x)
            with self.assertRaises(ValueError):
                exp.e.npz_read(p, ids[::-1], 6)
            with self.assertRaises(AssertionError):
                eng.anchor('large', x+1, x)
            with np.load(p) as z:
                values = {k: z[k].copy() for k in z.files}
            values['x'][0, 0] += 1
            np.savez_compressed(p, **values)
            with self.assertRaises(ValueError):
                exp.e.npz_read(p, ids, 6)

    def test_fold_context_only_and_resume(self):
        x = np.arange(500, dtype=float)[:, None]; y = x[:400, 0]+1000
        calls = []
        def run(ctx, target, query, indices):
            np.testing.assert_array_equal(target, ctx[:, 0]+1000)
            self.assertFalse(set(ctx[:, 0]) & set(query[:, 0]))
            self.assertLessEqual(len(ctx), 400)
            calls.append((len(ctx), len(query), list(indices)))
            return {'predictions': {str(i): np.full(len(query), target.mean()+i).tolist() for i in indices}}
        with tempfile.TemporaryDirectory(dir=exp.ROOT/'.task-tmp', prefix='d4-fold-') as tmp:
            path = Path(tmp);train = {n: x[:400] for n in eng.NAMES};q = {n: x[400:] for n in eng.NAMES}
            result, chosen = exp.fit_fold(path, {'synthetic': True}, train, y, q, np.zeros(400), 42, True, run)
            self.assertEqual(len(calls), 8)
            self.assertEqual(sum(a == 400 for a, _, _ in calls), 2)
            with patch.object(eng.ridge, 'fit_candidates', side_effect=AssertionError('unexpected fitting')):
                reused, choices = exp.fit_fold(path, {'synthetic': True}, train, y, q, np.zeros(400), 42, False)
            self.assertEqual(chosen, choices)
            for name in result:
                np.testing.assert_array_equal(reused[name], result[name])
            cell = path/'large/inner0.json';obj = exp.e.read_json(cell)
            obj['result']['predictions']['0'][0] += 1;exp.e.write_json(cell, obj)
            with self.assertRaises(ValueError):
                exp.fit_fold(path, {'synthetic': True}, train, y, q, np.zeros(400), 42, False)

    def test_full_export_readonly_reconstruction_and_missing_no_fit(self):
        rng = np.random.default_rng(6)
        ids = np.array([f'SYNTH_{i:06d}' for i in range(500)])
        data = SimpleNamespace(ids=ids, y=190+rng.normal(size=500)*15, noise=rng.uniform(size=500))
        folds = np.arange(500) % 5
        s = np.full(500, 190.);l = s+1;c = s+2
        ref = {'folds': folds, 'scattering': s, 'large': l, 'radio': c, 'current': .5*s+.25*l+.25*c,
               'inner_scattering': {f: np.full(400, 190.) for f in range(5)}}
        features = {n: rng.normal(size=(500, 3)).astype(np.float32) for n in eng.NAMES}
        plan = {'frozen': {'common_failure_ids': ids[:16].tolist()}}
        calls = []
        def runner(ctx, y, q, indices):
            calls.append(1)
            return {'predictions': {str(i): np.full(len(q), y.mean()+i*.1).tolist() for i in indices}}
        with tempfile.TemporaryDirectory(dir=exp.ROOT/'.task-tmp', prefix='d4-eval-') as tmp:
            state = (data, features, {42: ref}, plan)
            with patch.object(exp, 'OUT', Path(tmp)), patch.object(exp, 'validate', return_value=state), \
                 patch.object(exp, 'load_features', return_value=(features, {'synthetic': True})), \
                 patch.object(eng.ridge, 'fit_candidates', side_effect=runner), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(FileNotFoundError):
                    exp.evaluate(42, allow_fit=False)
                self.assertEqual(len(calls), 0)
                first = exp.evaluate(42)
                self.assertEqual(len(calls), 40)
                second = exp.evaluate(42, allow_fit=False)
                self.assertEqual(first, second);self.assertEqual(len(calls), 40)
                self.assertEqual(first['diagnosis'][eng.PRIMARY]['groups']['common16']['n'], 16)
                self.assertEqual(first['diagnosis'][eng.PRIMARY]['groups']['remaining484']['n'], 484)
                with (Path(tmp)/'seed42/oof.csv').open() as f:
                    self.assertIn('both_d4_blend', f.readline())
                p = Path(tmp)/'seed42/report.txt';p.write_text('tampered')
                with self.assertRaises(ValueError):
                    exp.evaluate(42, allow_fit=False)


if __name__ == '__main__':unittest.main()
