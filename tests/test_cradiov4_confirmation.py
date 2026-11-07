from __future__ import annotations
import contextlib
import inspect
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from threadpoolctl import threadpool_limits

from scripts import cradiov4_confirmation as c


def temporary():
    return tempfile.TemporaryDirectory(dir=c.ROOT/'reports', prefix='.cradio_confirmation_test_')


class SyntheticPredictor:
    calls = []
    runtime = {'synthetic': True}

    def __init__(self, *args):
        pass

    def __call__(self, x, y, q, seed):
        self.calls.append((len(x), len(q), seed))
        p = np.full(len(q), float(np.mean(y))) + .1*(q[:, 0]-x[:, 0].mean())
        return p, {'runtime': self.runtime, 'context_rows': len(x), 'query_rows': len(q)}


class ConfirmationTests(unittest.TestCase):
    def test_frozen_grid_formula_seed_scope_and_runner(self):
        self.assertEqual(c.SEEDS, (31415, 27182, 16180))
        self.assertEqual(c.PROTOCOL['alphas'], [100., 1000., 10000.])
        self.assertEqual(c.PRIMARY, 'radio_complementary_blend')
        self.assertEqual(c.CONTROL, 'large_retuned_complementary')
        a = np.arange(8, dtype=float)
        composed = c.eng.compose(a, 2*a, 3*a, 4*a)
        np.testing.assert_array_equal(composed[c.PRIMARY], 1.75*a)
        np.testing.assert_array_equal(composed[c.CONTROL], 2*a)
        # Original .5S+.5encoder selection, not standalone/final-mixture tuning.
        chosen = c.eng.choose({'0': 0*a, '1': a, '2': 3*a}, a, 2*a)
        self.assertEqual(chosen['blend']['index'], 0)
        self.assertEqual(chosen['standalone']['index'], 1)
        with self.assertRaises(ValueError): c.evaluate(seed=42)
        with self.assertRaises(ValueError): c.aggregate([])
        text = (c.ROOT/'scripts/run_cradiov4_confirmation.ps1').read_text()
        self.assertIn('$LASTEXITCODE -ne 0', text)
        self.assertIn('throw', text)
        self.assertIn('@(31415, 27182, 16180)', text)
        self.assertLess(text.index("'check'"), text.index("'evaluate'"))
        self.assertNotIn("'extract'", text)

    def test_nested_base_contexts_no_query_labels_resume_and_tamper(self):
        rng = np.random.default_rng(45)
        raw = {name: rng.normal(size=(500, 4)) for name in c.residual.eng.COMPONENTS}
        y = 190+rng.normal(size=500)
        indices = np.arange(100, 500)
        seen = []
        def fitted(tx, ty, qx, seed, predictor):
            seen.append((tx, ty, qx, seed))
            self.assertIn(len(ty), (266, 267))
            self.assertEqual(len(ty)+len(qx['tabpfn486']), 400)
            for name in tx:
                # Exact row membership proves outer100 are never in any context/query.
                universe = {tuple(v) for v in raw[name][indices]}
                train, query = {tuple(v) for v in tx[name]}, {tuple(v) for v in qx[name]}
                self.assertFalse(train & query)
                self.assertEqual(train | query, universe)
            return np.full((len(qx['tabpfn486']), 4), ty.mean()), {'synthetic': True}
        with temporary() as tmp, patch.object(c.residual.eng, 'fit_base', side_effect=fitted):
            folder = Path(tmp)
            a = c.residual.crossfit_components(folder, {'test': 1}, raw, indices, y[indices], 31415, 0, c.forbidden, True)
            self.assertEqual(len(seen), 3)
            self.assertEqual([row[-1] for row in seen], [31415, 31416, 31417])
            y[:100] = 1e9
            b = c.residual.crossfit_components(folder, {'test': 1}, raw, indices, y[indices], 31415, 0, c.forbidden, False)
            np.testing.assert_array_equal(a, b)
            self.assertEqual(len(seen), 3)
            cell = folder/'inner0/completed.json'
            saved = c.e.read_json(cell); saved['result']['components'][0][0] += 1
            c.e.write_json(cell, saved)
            with self.assertRaises(ValueError):
                c.residual.crossfit_components(folder, {'test': 1}, raw, indices, y[indices], 31415, 0, c.forbidden, False)
        self.assertNotIn('query_y', inspect.signature(c.original.fit_fold).parameters)

    def test_primary_diagnostics_and_group_tradeoff_are_not_replacement(self):
        ids = np.array([f'SYNTH_{i:06d}' for i in range(500)])
        y = np.linspace(140, 250, 500)
        data = SimpleNamespace(ids=ids, y=y, noise=np.arange(500))
        ref = {'current': y+2, 'folds': np.arange(500)%5}
        comp = y+1; comp[:16] = y[:16]+5
        pred = {'current': y+2, c.PRIMARY: comp, 'radio_replaced_blend': y+8,
                c.CONTROL: y+3, 'large_retuned_blend': y+4,
                'radio_ridge': y+6, 'large_retuned_ridge': y+7}
        plan = {'frozen': {'common_failure_ids': ids[:16].tolist()}}
        s, files = c.result_data(data, ref, pred, [], plan, 31415, {'synthetic': 1})
        self.assertEqual(s['primary'], c.PRIMARY)
        self.assertEqual(s['diagnosis'][c.PRIMARY]['common16_improved'], 0)
        for row in files['per_image_errors.csv']:
            i = int(row['ID'].split('_')[-1])
            self.assertEqual(row['candidate_sse_gain'], float(4-(comp[i]-y[i])**2))
        summaries = [dict(s, seed=seed) for seed in c.SEEDS]
        result = c.aggregate(summaries)
        self.assertTrue(result['all_seeds_improve_both'])
        self.assertFalse(result['all_seeds_nonworsening_vs_both_by_group']['common16'])
        self.assertEqual(result['conclusion'], 'overall_improves_but_subgroup_tradeoff')
        self.assertFalse(result['automatic_adoption'])
        self.assertFalse(result['historical_seeds_in_aggregate'])

    def test_synthetic_three_seed_pipeline_resume_readonly_compare_and_tamper(self):
        rng = np.random.default_rng(19); n = 500
        large = rng.normal(size=(n, 5)); radio = rng.normal(size=(n, 6))
        data = SimpleNamespace(ids=np.array([f'SYNTH_{i:06d}' for i in range(n)]),
            y=190+6*large[:, 0]+rng.normal(size=n)*4, noise=rng.random(n))
        raw = {name: rng.normal(size=(n, 4)) for name in c.residual.eng.COMPONENTS}
        folds = np.arange(n)%5; refs = {}
        plan = {'frozen': {'common_failure_ids': data.ids[:16].tolist()}, 'competition_fits_during_preparation': 0}
        with temporary() as tmp, threadpool_limits(limits=2), contextlib.redirect_stdout(io.StringIO()):
            folder = Path(tmp)/'experiment'
            for seed in c.SEEDS:
                anchor = np.full(n, np.nan); alphas = []
                for fold in range(5):
                    ti, vi = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
                    anchor[vi], choice = c.large_confirmation.fit_fold(Path(tmp)/f'anchor{seed}/{fold}', {},
                        large[ti], data.y[ti], large[vi], seed+fold)
                    alphas.append(choice['alpha'])
                scat = np.full(n, 190.)
                refs[seed] = {'folds': folds, 'large': anchor, 'large_alphas': alphas,
                    'scattering': scat, 'current': .5*scat+.5*anchor, 'runtime': SyntheticPredictor.runtime}
            state = (data, large, radio, raw, refs, plan)
            SyntheticPredictor.calls = []
            with patch.object(c, 'validate', return_value=state), patch.object(c.t, 'Predictor', SyntheticPredictor), patch.object(c.t, 'verify_weights', return_value=Path('synthetic')):
                c.check(folder=folder)
                c.check(folder=folder)  # Reuse synthetic check too.
                self.assertEqual(len(SyntheticPredictor.calls), 1)
                # Simulate an interruption after one real inner baseline cell.
                original_fit = c.residual.eng.fit_base
                counter = []
                def interrupted(*args):
                    counter.append(1)
                    if len(counter) == 2:
                        raise RuntimeError('synthetic interruption')
                    return original_fit(*args)
                with patch.object(c.residual.eng, 'fit_base', side_effect=interrupted):
                    with self.assertRaisesRegex(RuntimeError, 'synthetic interruption'):
                        c.evaluate(folder, c.SEEDS[0])
                preserved = folder/'seed31415/fold0/baseline/inner0/completed.json'
                old_hash = c.e.base.sha256_file(preserved)
                for seed in c.SEEDS:
                    c.evaluate(folder, seed)
                self.assertEqual(c.e.base.sha256_file(preserved), old_hash)
                self.assertEqual(len(SyntheticPredictor.calls), 46)  # 1synthetic +45contexts; no repeated completed cell.
                for seed in c.SEEDS:
                    self.assertTrue(c.completed(folder/f'seed{seed}', c.binding_for(plan, seed, refs[seed])))
            snapshots = {str(p): c.e.base.sha256_file(p) for p in folder.glob('seed*/**/*') if p.is_file()}
            with patch.object(c, 'validate', return_value=state), patch.object(c.t, 'Predictor', side_effect=AssertionError('No new TabPFN')), patch.object(c.residual.eng, 'fit_base', side_effect=AssertionError('No refit')), patch.object(c.eng.ridge, 'fit_candidates', side_effect=AssertionError('No Ridge refit')):
                result = c.compare(folder)
                self.assertEqual(result['seeds'], list(c.SEEDS))
                self.assertEqual(result['primary'], c.PRIMARY)
                self.assertFalse(result['independent_holdout'])
                self.assertEqual(snapshots, {str(p): c.e.base.sha256_file(p) for p in folder.glob('seed*/**/*') if p.is_file()})
                path = folder/'seed31415/oof.csv'
                path.write_text(path.read_text()+'bad\n')
                with self.assertRaises(ValueError): c.evaluate(folder, 31415, allow_fit=False)


if __name__ == '__main__':
    unittest.main()
