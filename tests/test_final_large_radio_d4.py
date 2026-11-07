"""Synthetic end-to-end finalization, scope/anchor/preservation/resume checks."""
import contextlib
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from threadpoolctl import threadpool_limits
from scripts import final_large_radio_d4_submission as f


class FinalD4Tests(unittest.TestCase):
    def setUp(self):
        self.thread_limit = threadpool_limits(limits=2); self.thread_limit.__enter__()

    def tearDown(self):
        self.thread_limit.__exit__(None, None, None)

    def test_full500_two_heads_to1000_rows_and_readonly_resume(self):
        from sklearn.linear_model import Ridge
        rng = np.random.default_rng(20261018)
        ids = np.array([f'TRAIN_{i+1:06d}' for i in range(500)])
        data = SimpleNamespace(ids=ids, y=rng.normal(200, 15, 500))
        widths = {'large': 7, 'radio': 9}
        x = {n: rng.normal(size=(500, w)).astype(np.float32) for n, w in widths.items()}
        raw = {n: rng.normal(size=v.shape).astype(np.float32) for n, v in x.items()}
        q = {n: rng.normal(size=(1000, w)).astype(np.float32) for n, w in widths.items()}
        inner_s = np.full(500, 199.); scattering = np.linspace(165, 230, 1000); current = scattering+2
        test_ids = [f'TEST_{i+1:06d}' for i in range(1000)][::-1]
        runtime = {'synthetic': True}
        anchors = {n: np.concatenate((raw[n][:16], x[n][:16]), axis=1) for n in widths}
        plan = {'frozen': {'test': {'ids': test_ids, 'images': []}, 'train_runtime': runtime,
            'anchor_ids': ids[:16].tolist(), 'anchor_sha256': {n: f.e.base.sha256_array(v) for n, v in anchors.items()}}}
        calls = []; fits = []; real_fit = Ridge.fit
        def trace(model, tx, ty, *a, **kw):
            fits.append(len(tx)); return real_fit(model, tx, ty, *a, **kw)
        class Extractor:
            def __init__(self, name, device, expected): self.name = name; self.runtime = expected
            def __call__(self, split, selected):
                calls.append((self.name, split, len(selected)))
                if split == 'train':
                    np.testing.assert_array_equal(selected, ids[:16]); return anchors[self.name].copy()
                i = test_ids.index(str(selected[0])); v = q[self.name][i:i+1]
                return np.concatenate((v+1, v), axis=1)
            def close(self): pass
        with tempfile.TemporaryDirectory(dir=f.ROOT/'tests', prefix='.final_d4_') as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp); out = root/'reports/candidate'; source = root/'open/test';source.mkdir(parents=True)
            prior = root/'old_submission.csv';prior.write_bytes(b'original'); prior_sha = f.e.base.sha256_file(prior)
            for ident in test_ids:
                p = source/(ident+'.png'); p.write_bytes(ident.encode())
                plan['frozen']['test']['images'].append({'id': ident, 'relative_path': 'test/'+ident+'.png', 'sha256': f.e.base.sha256_file(p)})
            state = (data, x, raw, inner_s, scattering, current, plan)
            with patch.object(f, 'ROOT', root), patch.object(f, 'OUT', out), patch.object(f, 'validate', return_value=state), \
                 patch.dict(f.eng.WIDTHS, widths), patch.object(Ridge, 'fit', trace), \
                 patch.object(f, 'D4Extractor', side_effect=AssertionError('No vision during training')):
                f.train(out)
                self.assertEqual(len(fits), 20);self.assertEqual(fits.count(500), 2)
                self.assertEqual(sorted(set(fits)), [333, 334, 500])
                f.train(out); self.assertEqual(len(fits), 20)
                with self.assertRaises(FileNotFoundError): f.predict(out)
                with patch.object(f, 'D4Extractor', Extractor): f.extract(out)
                self.assertEqual(len(calls), 2002)
                for n in widths:
                    self.assertEqual(next(c for c in calls if c[0] == n), (n, 'train', 16))
                f.predict(out)
                rows = f.core.read_rows(out/'submission/component_predictions.csv', {'ID', 'hardness'})
                self.assertEqual([r['ID'] for r in rows], test_ids)
                p = np.array([float(r['hardness']) for r in rows])
                expect = .5*scattering
                for n in widths:
                    head, sel = f.load_head(out, plan, n, x[n], data.y, inner_s)
                    expect += .25*f.shared.predict_head(head, q[n])
                    self.assertIn(sel['alpha'], [100, 1000, 10000])
                np.testing.assert_array_equal(p, expect)
                f.shared.verify_submission(out/'submission/submission.csv', test_ids, p)
                hashes = {n: f.e.base.sha256_file(out/'submission'/n) for n in ('submission.csv', 'completed.json')}
                f.train(out); f.extract(out); f.predict(out)
                self.assertEqual(len(fits), 20); self.assertEqual(len(calls), 2002)
                self.assertEqual(hashes, {n: f.e.base.sha256_file(out/'submission'/n) for n in hashes})
                self.assertEqual(prior_sha, f.e.base.sha256_file(prior))
                chunk = out/'test_features/large/chunks/0000.npz'
                chunk.write_bytes(b'tamper')
                with self.assertRaises(ValueError): f.predict(out)

    def test_cv_query_is_excluded_and_objectives_and_ties(self):
        x = np.arange(500, dtype=np.float64)[:, None]; y = x[:, 0]+1000; s = y+2
        calls = []
        def runner(ctx, target, query, indices):
            np.testing.assert_array_equal(target, ctx[:, 0]+1000)
            self.assertFalse(set(ctx[:, 0]) & set(query[:, 0]));self.assertLess(len(ctx), 500)
            self.assertEqual(list(indices), [0, 1, 2]);calls.append(len(ctx))
            truth = query[:, 0]+1000
            return {'predictions': {'0': truth.tolist(), '1': (truth-2).tolist(), '2': (truth-1).tolist()}}
        with tempfile.TemporaryDirectory(dir=f.ROOT/'tests', prefix='.d4_select_') as tmp:
            for name, expected in [('large', 10000), ('radio', 1000)]:
                out = Path(tmp)/name
                selected = f.select_alpha(out, {'synthetic': name}, name, x, y, s, True, runner)
                self.assertEqual(selected['alpha'], expected)
                reused = f.select_alpha(out, {'synthetic': name}, name, x, y, s, False, f.old.c.forbidden)
                self.assertEqual(selected, reused)
            self.assertEqual(len(calls), 6)
            cell = Path(tmp)/'large/inner0.json';obj = f.e.read_json(cell)
            obj['result']['predictions']['0'][0] += 1; f.e.write_json(cell, obj)
            with self.assertRaises(ValueError): f.select_alpha(Path(tmp)/'large', {'synthetic': 'large'}, 'large', x, y, s, False)
        same = {str(i): y.copy() for i in f.eng.INDICES}
        self.assertEqual(f.eng.choose('radio', same, y, s)['selected']['alpha'], 10000)

    def test_reused_contexts_forbid_fit_and_validate_hash(self):
        data = SimpleNamespace(y=np.arange(500, dtype=float)); cross = np.ones((500, 4))
        s = f.old.residual.eng.ensemble(cross); selection = {'inner_scattering_sha256': f.e.base.sha256_array(s)}
        with patch.object(f.old.residual, 'crossfit_components', return_value=cross) as mocked:
            actual = f.reuse_inner_scattering(data, {'synthetic': np.ones((500, 1))}, {'frozen': {}}, selection)
            np.testing.assert_array_equal(actual, s)
            self.assertIs(mocked.call_args.args[-1], False)
            self.assertIs(mocked.call_args.args[-2], f.old.c.forbidden)
            selection['inner_scattering_sha256'] = 'wrong'
            with self.assertRaises(ValueError): f.reuse_inner_scattering(data, {}, {'frozen': {}}, selection)

    def test_anchor_d4_mismatch_and_destination_guard(self):
        with tempfile.TemporaryDirectory(dir=f.ROOT/'tests', prefix='.d4_anchor_') as tmp:
            root = Path(tmp);out = root/'reports/candidate'; expected = np.zeros((16, 6), np.float32)
            plan = {'frozen': {'anchor_ids': [f'TRAIN_{i:06d}' for i in range(16)],
                'anchor_sha256': {'large': f.e.base.sha256_array(expected)}}}
            actual = expected.copy(); actual[:, 3:] = 1
            with patch.object(f, 'ROOT', root), patch.object(f, 'OUT', out), patch.dict(f.eng.WIDTHS, {'large': 3}):
                f.protect_destination(out/'submission')
                with self.assertRaises(ValueError): f.protect_destination(root/'reports/old')
                with self.assertRaises(AssertionError): f.anchor_record(out, plan, 'large', expected, lambda *_: actual)
                self.assertFalse((out/'test_features/large/anchor.npz').exists())

    def test_real_extractor_uses_identical_native_arithmetic(self):
        import torch
        from PIL import Image
        a = np.arange(256*256, dtype=np.uint8).reshape(256, 256)
        with tempfile.TemporaryDirectory(dir=f.ROOT/'tests', prefix='.d4_pixels_') as tmp:
            root = Path(tmp);path = root/'open/train/SYNTH.png';path.parent.mkdir(parents=True)
            Image.fromarray(a).save(path)
            for name in f.eng.NAMES:
                extractor = object.__new__(f.D4Extractor);extractor.name = name
                extractor.torch = torch; extractor.device = torch.device('cpu');extractor.model = object()
                seen = []
                def fake(n, model, raw):
                    seen.append(raw);return np.zeros((1, 2), np.float32), np.ones((1, 2), np.float32)
                with patch.object(f, 'ROOT', root), patch.object(f.eng, 'extract_views', side_effect=fake):
                    result = extractor('train', ['SYNTH'])
                expected = torch.from_numpy(a.astype(np.float32)/np.float32(255))[None, None] if name == 'large' else torch.from_numpy(a.copy()).float()[None, None]/255.
                torch.testing.assert_close(seen[0], expected, rtol=0, atol=0)
                np.testing.assert_array_equal(result, [[0, 0, 1, 1]])


if __name__ == '__main__': unittest.main()
