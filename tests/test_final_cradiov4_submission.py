"""Synthetic arrays only; no competition fitting or real vision forwards."""
from __future__ import annotations
import contextlib
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from threadpoolctl import threadpool_limits
from scripts import final_cradiov4_submission as f


def temporary():
    return tempfile.TemporaryDirectory(dir=f.ROOT/'tests', prefix='.final_radio_')


def fixture(width=12, count=1000):
    rng = np.random.default_rng(20261015)
    x = rng.normal(size=(500, width)).astype(np.float32); x[:, 0] = np.arange(500); x[:, -1] = 1
    q = rng.normal(size=(count, width)).astype(np.float32); q[:, 0] = np.arange(500, 500+count)
    data = SimpleNamespace(ids=np.array([f'TRAIN_{i+1:06d}' for i in range(500)]), y=190+3*x[:, 1]+rng.normal(size=500))
    raw = {name: rng.normal(size=(500, 4)) for name in f.residual.eng.COMPONENTS}
    for a in raw.values(): a[:, 0] = np.arange(500)
    ids = [f'TEST_{i+1:06d}' for i in range(count)][::-1]
    runtime = {'synthetic': True}
    plan = {'frozen': {'synthetic': True, 'tabpfn_runtime': runtime, 'train_runtime': runtime, 'size': 592,
        'anchor_ids': data.ids[:16].tolist(), 'test': {'ids': ids, 'rows': count, 'images': []}}}
    s = np.linspace(180., 210., count); l = np.linspace(215., 185., count)
    return data, x, q, raw, s, l, plan


def add_files(root, plan):
    for ident in plan['frozen']['test']['ids']:
        path = root/'open/test'/(ident+'.png'); path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(ident.encode())
        plan['frozen']['test']['images'].append({'id': ident, 'relative_path': 'test/'+ident+'.png', 'sha256': f.e.base.sha256_file(path)})


class FakePredictor:
    runtime = {'synthetic': True}
    calls = []
    def __init__(self, *args): pass
    def __call__(self, tx, y, qx, seed):
        self.calls.append((tx.copy(), y.copy(), qx.copy(), seed))
        return np.full(len(qx), y.mean()), {'runtime': self.runtime, 'context_rows': len(tx), 'query_rows': len(qx)}


class FinalRadioTests(unittest.TestCase):
    def setUp(self):
        self.threads = threadpool_limits(limits=2); self.threads.__enter__()
    def tearDown(self):
        self.threads.__exit__(None, None, None)

    def test_full500_training_numeric_export_predict1000_and_no_refit(self):
        from sklearn.linear_model import Ridge
        data, x, q, raw, s, l, plan = fixture()
        fit = Ridge.fit; fitted = []
        def trace(model, tx, ty, *a, **kw):
            fitted.append(len(tx)); return fit(model, tx, ty, *a, **kw)
        FakePredictor.calls = []
        with temporary() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp); out = root/'reports/candidate'
            old = root/'prior.csv'; old.write_bytes(b'preserved submission'); old_bytes = old.read_bytes()
            state = (data, x, raw, s, l, plan)
            with patch.object(f, 'ROOT', root), patch.object(f, 'OUT', out), patch.object(f, 'validate', return_value=state), patch.object(f.t, 'Predictor', FakePredictor), patch.object(f.t, 'verify_weights', return_value=Path('synthetic')), patch.object(Ridge, 'fit', trace), patch.object(f, 'RadioExtractor', side_effect=AssertionError('No vision during train')):
                f.train(out)
            self.assertEqual(len(fitted), 127)
            self.assertEqual(fitted.count(500), 1)
            self.assertEqual(len(FakePredictor.calls), 3)
            for tx, ty, qx, seed in FakePredictor.calls:
                self.assertIn(len(tx), (333, 334)); self.assertEqual(len(tx)+len(qx), 500)
                ti, vi = tx[:, 0].astype(int), qx[:, 0].astype(int)
                self.assertFalse(set(ti)&set(vi)); np.testing.assert_array_equal(ty, data.y[ti])
            head, selected = f.load_head(out, plan, data, x, raw)
            np.testing.assert_allclose(head['x_mean'], x.astype(float).mean(0), rtol=0, atol=1e-12)
            self.assertEqual(head['x_scale'][-1], 1.)
            reference = f.core.make_estimator(selected['alpha']).fit(x.astype(float), data.y)
            np.testing.assert_allclose(f.shared.predict_head(head, q), reference.predict(q.astype(float)), rtol=0, atol=1e-10)
            f.e.write_json(out/'test_features/completed.json', {'synthetic': True})
            with patch.object(f, 'ROOT', root), patch.object(f, 'OUT', out), patch.object(f, 'validate', return_value=state), patch.object(f, 'load_test_cache', return_value=q), patch.object(f.t, 'Predictor', side_effect=AssertionError('No new context')), patch.object(Ridge, 'fit', side_effect=AssertionError('No Ridge refit')):
                f.train(out); f.predict(out)
                saved = (out/'submission/submission.csv').read_bytes()
                f.predict(out)
            self.assertEqual(saved, (out/'submission/submission.csv').read_bytes())
            expected = .5*s+.25*l+.25*f.shared.predict_head(head, q)
            f.shared.verify_submission(out/'submission/submission.csv', plan['frozen']['test']['ids'], expected)
            self.assertEqual(old_bytes, old.read_bytes())
            self.assertFalse(f.e.read_json(out/'submission/validation.json')['uploaded'])
            self.assertEqual(selected['n_train'], 500)
            cell = out/'model/cv/baseline/inner0/completed.json'
            cell.write_text(cell.read_text()+'changed')
            with self.assertRaises(ValueError): f.load_head(out, plan, data, x, raw)

    def test_alpha_blend_objective_ties_and_only_context_preprocessing(self):
        data, x, _, raw, _, _, _ = fixture()
        y = np.arange(500, dtype=float); seen = []
        def runner(tx, ty, qx, indices):
            ti, vi = tx[:, 0].astype(int), qx[:, 0].astype(int)
            self.assertFalse(set(ti)&set(vi)); np.testing.assert_array_equal(ty, y[ti])
            real = f.eng.ridge.fit_candidates(tx, ty, qx, indices)
            np.testing.assert_allclose(real['audit']['x_mean'], tx.astype(float).mean(0))
            self.assertAlmostEqual(real['audit']['y_mean'], ty.mean())
            self.assertFalse(real['audit']['query_labels_accepted'])
            seen.append(vi)
            return {'predictions': {'0': np.zeros(len(vi)).tolist(), '1': y[vi].tolist(), '2': (3*y[vi]).tolist()}}
        cross = np.repeat((2*y)[:, None], 4, axis=1)
        with temporary() as tmp, patch.object(f.residual, 'crossfit_components', return_value=cross):
            out = Path(tmp)
            selected = f.select_alpha(out, {}, x, y, raw, f.c.forbidden, runner=runner)
            self.assertEqual(selected['alpha'], 10000.)
            self.assertEqual(selected['choices']['standalone']['alpha'], 1000.)
            np.testing.assert_array_equal(np.sort(np.concatenate(seen)), np.arange(500))
            again = f.select_alpha(out, {}, x, y, raw, f.c.forbidden, False, f.c.forbidden)
            self.assertEqual(selected, again)
            with self.assertRaises(ValueError): f.select_alpha(out, {'changed': 1}, x, y, raw, f.c.forbidden, False, f.c.forbidden)
            with self.assertRaises(FileNotFoundError): f.select_alpha(out/'missing', {}, x, y, raw, f.c.forbidden, False, f.c.forbidden)
        tie = f.eng.choose({str(i): y for i in f.eng.RIDGE_INDICES}, y, y)
        self.assertEqual(tie['blend']['alpha'], 10000.)

    def test_anchor_failure_prevents_test_inference(self):
        data, x, _, raw, s, l, plan = fixture(f.WIDTH, 3); calls = []
        class Fake:
            runtime = plan['frozen']['train_runtime']; size = 592
            def __call__(self, split, ids): calls.append(split); return x[:len(ids)]+1
            def close(self): pass
        with temporary() as tmp, patch.object(f, 'validate', return_value=(data, x, raw, s, l, plan)), patch.object(f, 'load_head', return_value=({}, {})), patch.object(f, 'RadioExtractor', return_value=Fake()):
            with self.assertRaisesRegex(ValueError, 'anchor mismatch'): f.extract(Path(tmp))
            self.assertEqual(calls, ['train'])
            self.assertFalse((Path(tmp)/'test_features/anchor.npz').exists())

    def test_extraction_resume_source_and_feature_tamper(self):
        data, x, q, raw, s, l, plan = fixture(f.WIDTH, 7); calls = []; fail = [True]
        class Fake:
            runtime = plan['frozen']['train_runtime']; size = 592
            def __call__(self, split, ids):
                calls.append((split, list(ids)))
                if split == 'train': return x[:len(ids)].copy()
                j = plan['frozen']['test']['ids'].index(ids[0])
                if j == 2 and fail[0]: fail[0] = False; raise RuntimeError('Synthetic interruption')
                return q[j:j+1].copy()
            def close(self): pass
        with temporary() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp); out = root/'reports/candidate'; add_files(root, plan)
            with patch.object(f, 'ROOT', root), patch.object(f, 'validate', return_value=(data, x, raw, s, l, plan)), patch.object(f, 'load_head', return_value=({}, {})), patch.object(f, 'RadioExtractor', return_value=Fake()):
                with self.assertRaises(RuntimeError): f.extract(out)
                self.assertFalse((out/'test_features/completed.json').exists())
                f.extract(out)
                np.testing.assert_array_equal(f.load_test_cache(out, plan, x), q)
                self.assertEqual(sum(split == 'train' for split, _ in calls), 1)
                self.assertEqual(sum(split == 'test' for split, _ in calls), 8)
                with patch.object(f, 'RadioExtractor', side_effect=AssertionError('No repeat inference')): f.extract(out)
                with self.assertRaises(ValueError): f.load_test_cache(out, plan, x+1)
                chunk = out/'test_features/chunks/0000.npz'
                original = chunk.read_bytes()
                with np.load(chunk, allow_pickle=False) as a: payload = {k: a[k] for k in a.files}
                payload['x'][0, 0] += 1; np.savez_compressed(chunk, **payload)
                with self.assertRaises(ValueError): f.load_test_cache(out, plan, x)
                chunk.write_bytes(original)
                (root/'open'/plan['frozen']['test']['images'][0]['relative_path']).write_bytes(b'changed')
                with self.assertRaises(ValueError): f.load_test_cache(out, plan, x)

    def test_protected_output_and_exact_502525_not_old_submission_blend(self):
        for path in (f.previous.OUT, f.previous.SCATTERING, f.c.OUT, f.ROOT, f.ROOT/'open', f.ROOT/'scripts'):
            with self.assertRaises(ValueError): f.protect_destination(path)
        with temporary() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp); out = root/'reports/candidate'; ids = ['TEST_000002', 'TEST_000001']
            s, l, r = np.array([180., 200.]), np.array([210., 220.]), np.array([190., 170.])
            with patch.object(f, 'ROOT', root), patch.object(f, 'OUT', out):
                f.write_submission(out, ids, s, l, r, {'fixed': True})
                f.shared.verify_submission(out/'submission.csv', ids, .5*s+.25*l+.25*r)
                self.assertFalse(np.array_equal(.5*s+.25*l+.25*r, .75*(.5*s+.5*l)+.25*r))
                with self.assertRaises(ValueError): f.write_submission(out/'bad', ids, s, l, r*np.nan, {})
                with self.assertRaises(ValueError): f.write_submission(out/'bad', [ids[0]]*2, s, l, r, {})
                with self.assertRaises(ValueError): f.write_submission(out, ids, s, l, r, {'fixed': False})
                (out/'component_predictions.csv').write_text('tamper')
                with self.assertRaises(ValueError): f.write_submission(out, ids, s, l, r, {'fixed': True})

    def test_prepare_only_and_frozen_state_and_runner(self):
        data, x, _, raw, s, l, plan = fixture()
        with temporary() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp); out = root/'reports/candidate'
            state = (data, x, raw, s, l, plan['frozen'])
            with patch.object(f, 'ROOT', root), patch.object(f, 'OUT', out), patch.object(f, 'load_state', return_value=state), patch.object(f.t, 'Predictor', side_effect=AssertionError('No fitting')), patch.object(f.eng, 'load_model', side_effect=AssertionError('No inference')):
                f.prepare(out); before = (out/'plan.json').read_bytes(); f.prepare(out); f.validate(out)
                self.assertEqual(before, (out/'plan.json').read_bytes())
                changed = {**plan['frozen'], 'changed': True}
                with patch.object(f, 'load_state', return_value=(data, x, raw, s, l, changed)):
                    with self.assertRaises(ValueError): f.validate(out)
        text = (f.ROOT/'scripts/run_final_cradiov4.ps1').read_text()
        self.assertIn('$LASTEXITCODE -ne 0', text)
        for stage in ('prepare', 'train', 'extract', 'predict'): self.assertIn("'"+stage+"'", text)
        self.assertLess(text.index("'train'"), text.index("'extract'"))
        self.assertEqual(f.PROTOCOL['blend'], {'scattering486': .5, 'large518_ridge': .25, 'cradiov4_ridge': .25})
        self.assertEqual(sum(f.PROTOCOL['effective_weights'].values()), 1.)


if __name__ == '__main__': unittest.main()
