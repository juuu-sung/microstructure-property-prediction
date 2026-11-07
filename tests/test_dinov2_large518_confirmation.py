"""Synthetic-only confirmation, leakage, archive identity, and resume checks."""
from contextlib import contextmanager, redirect_stdout
import copy
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from threadpoolctl import threadpool_limits
from scripts import evaluate_dinov2_large518_confirmation as c


@contextmanager
def temporary():
    with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent, prefix='.dino_confirm_') as p:
        yield Path(p)


def fixture(n=150):
    rng = np.random.default_rng(84)
    x = rng.normal(size=(n, 12))
    x[:, 0] = np.arange(n)
    y = 190+4*x[:, 1]+rng.normal(size=n)
    data = SimpleNamespace(ids=np.array([f'SYNTH{i:06d}' for i in range(n)]), y=y, noise=rng.uniform(size=n))
    ref = {'current': y+rng.normal(size=n)*8, 'folds': np.arange(n)%5}
    plan = {'frozen': {'synthetic': True, 'common_failure_ids': data.ids[:16].tolist()}}
    return data, ref, x, plan


def fake_fit(tx, y, qx, indices):
    return {'predictions': {str(i): (np.full(len(qx), np.mean(y))+.1*i+qx[:, 1]).tolist() for i in indices},
            'audit': {'synthetic': True}}


class Large518ConfirmationTests(unittest.TestCase):
    def setUp(self):
        self.limit = threadpool_limits(limits=2)
        self.limit.__enter__()

    def tearDown(self):
        self.limit.__exit__(None, None, None)

    def test_fixed_candidates_and_driver_scope(self):
        self.assertEqual(c.SEEDS, (31415, 27182, 16180))
        self.assertEqual([c.eng.CANDIDATES[i]['alpha'] for i in c.RIDGE_INDICES], [10000, 1000, 100])
        text = (c.ROOT/'scripts/run_dinov2_large518_confirmation.ps1').read_text()
        self.assertIn('$LASTEXITCODE -ne 0', text)
        for forbidden in ('extract', 'setup_dinov2', '--device', 'cuda', 'download'):
            self.assertNotIn(forbidden, text)

    def test_context_only_scalers_and_nested_choice_match_independent_ridge(self):
        from sklearn.compose import TransformedTargetRegressor
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import Ridge
        data, ref, x, _ = fixture()
        ti, vi = np.flatnonzero(ref['folds'] != 0), np.flatnonzero(ref['folds'] == 0)
        tx, ty, qx = x[ti], data.y[ti], x[vi]
        inner = {i: np.full(len(ty), np.nan) for i in c.RIDGE_INDICES}
        def model(i):
            return TransformedTargetRegressor(regressor=make_pipeline(StandardScaler(), Ridge(alpha=c.eng.CANDIDATES[i]['alpha'])), transformer=StandardScaler())
        for a, b in c.eng.inner_splits(ty, 31415):
            for i in c.RIDGE_INDICES:
                inner[i][b] = model(i).fit(tx[a], ty[a]).predict(tx[b])
        best = min(c.RIDGE_INDICES, key=lambda i: (np.mean((inner[i]-ty)**2), i))
        expected = model(best).fit(tx, ty).predict(qx)
        with temporary() as p, patch.object(c.eng, 'PCA', side_effect=AssertionError('No PCA')), patch.object(c.eng, 'SVR', side_effect=AssertionError('No SVR')):
            actual, choice = c.fit_fold(p, {}, tx, ty, qx, 31415)
            changed = qx.copy(); changed[1:] += 1e7
            actual2, choice2 = c.fit_fold(p/'changed', {}, tx, ty, changed, 31415)
        self.assertEqual(choice['candidate_index'], best)
        self.assertEqual(choice, choice2)
        self.assertAlmostEqual(actual[0], actual2[0], places=8)
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-9)

    def test_inner_context_excludes_outer_and_inner_queries(self):
        data, ref, x, _ = fixture()
        ti, vi = np.flatnonzero(ref['folds'] != 0), np.flatnonzero(ref['folds'] == 0)
        calls = []
        def spy(tx, y, qx, indices):
            context, query = tx[:, 0].astype(int), qx[:, 0].astype(int)
            self.assertFalse(set(context) & set(query))
            self.assertTrue(set(context).issubset(set(ti)))
            np.testing.assert_array_equal(y, data.y[context])
            calls.append((context, query, indices))
            return fake_fit(tx, y, qx, indices)
        with temporary() as p:
            c.fit_fold(p, {}, x[ti], data.y[ti], x[vi], 31415, runner=spy)
        self.assertEqual(len(calls), 4)
        np.testing.assert_array_equal(np.sort(np.concatenate([q for _, q, _ in calls[:3]])), ti)
        self.assertTrue(all(tuple(indices) == c.RIDGE_INDICES for _, _, indices in calls[:3]))
        self.assertEqual(len(calls[-1][2]), 1)

    def test_outer_label_poisoning_and_current_predictions_do_not_enter_fits(self):
        data, ref, x, _ = fixture()
        ti, vi = np.flatnonzero(ref['folds'] != 0), np.flatnonzero(ref['folds'] == 0)
        poisoned = data.y.copy(); poisoned[vi] += 1e9
        with temporary() as p:
            a, ac = c.fit_fold(p/'a', {}, x[ti], data.y[ti], x[vi], 31415, runner=fake_fit)
            b, bc = c.fit_fold(p/'b', {}, x[ti], poisoned[ti], x[vi], 31415, runner=fake_fit)
        self.assertEqual(ac, bc)
        np.testing.assert_array_equal(a, b)

    def test_exact_tie_prefers_larger_alpha(self):
        data, ref, x, _ = fixture()
        ti, vi = np.flatnonzero(ref['folds'] != 0), np.flatnonzero(ref['folds'] == 0)
        def tied(tx, y, qx, indices):
            return {'predictions': {str(i): np.full(len(qx), np.mean(y)).tolist() for i in indices}}
        with temporary() as p:
            _, choice = c.fit_fold(p, {}, x[ti], data.y[ti], x[vi], 31415, runner=tied)
        self.assertEqual(choice['alpha'], 10000.)

    def test_interrupted_resume_and_cache_tamper(self):
        data, ref, x, _ = fixture()
        ti, vi = np.flatnonzero(ref['folds'] != 0), np.flatnonzero(ref['folds'] == 0)
        calls = []
        def interrupt(*args):
            calls.append(1)
            if len(calls) == 3:
                raise RuntimeError('Synthetic interruption')
            return fake_fit(*args)
        args = ({'fixed': 1}, x[ti], data.y[ti], x[vi], 31415)
        with temporary() as p:
            with self.assertRaises(RuntimeError): c.fit_fold(p, *args, runner=interrupt)
            with patch.object(c.eng, 'fit_candidates', wraps=fake_fit) as spy:
                a, ac = c.fit_fold(p, *args)
                self.assertEqual(spy.call_count, 2)
            b, bc = c.fit_fold(p, *args, allow_fit=False, runner=c.forbidden)
            np.testing.assert_array_equal(a, b); self.assertEqual(ac, bc)
            with self.assertRaises(ValueError): c.fit_fold(p, {'fixed': 2}, *args[1:], allow_fit=False)
            bad = c.e.read_json(p/'inner0.json'); bad['result']['predictions']['0'][0] += 1
            c.e.write_json(p/'inner0.json', bad)
            with self.assertRaises(ValueError): c.fit_fold(p, *args, allow_fit=False)
            with self.assertRaises(FileNotFoundError): c.fit_fold(p/'empty', *args, allow_fit=False)

    def test_reference_uses_matching_scattering_column_ids_and_folds(self):
        data, ref, _, _ = fixture(500)
        seed = 31415
        archive_plan = {'frozen': {'references': {str(seed): {'arrays': {'folds': c.e.base.sha256_array(ref['folds'])}}}}}
        binding = {'plan': c.e.digest(archive_plan['frozen']), 'seed': seed}
        summary = {'binding': binding, 'primary': 'scattering_replaced_blend',
                   'metrics': {'scattering_replaced_blend': c.core.metrics(data.y, ref['current'])}}
        with temporary() as p:
            folder = p/f'seed{seed}'
            rows = [{'ID': str(data.ids[i]), 'hardness': data.y[i], 'fold': int(ref['folds'][i]),
                     'current': -999., 'scattering_replaced_blend': float(ref['current'][i])} for i in range(500)]
            c.e.write_csv(folder/'oof.csv', rows)
            c.e.write_csv(p/f'folds_seed{seed}.csv', [{'ID': str(i), 'fold': int(f)} for i, f in zip(data.ids, ref['folds'])])
            c.e.write_json(folder/'summary.json', summary)
            for name in ('diagnostics.csv', 'per_image_errors.csv', 'report.md'):
                (folder/name).write_text('synthetic', encoding='utf-8')
            c.e.write_json(folder/'completed_evaluation.json', {'binding': binding, 'files':
                {n: c.e.base.sha256_file(folder/n) for n in ('oof.csv', 'summary.json', 'diagnostics.csv', 'per_image_errors.csv', 'report.md')}})
            actual, _ = c.archived_reference(data, archive_plan, seed, p)
            np.testing.assert_array_equal(actual['current'], ref['current'])
            np.testing.assert_array_equal(actual['folds'], ref['folds'])
            wrong = SimpleNamespace(ids=data.ids, y=data.y+1)
            with self.assertRaises(AssertionError): c.archived_reference(wrong, archive_plan, seed, p)
            (folder/'oof.csv').write_text('changed', encoding='utf-8')
            with self.assertRaises(ValueError): c.archived_reference(data, archive_plan, seed, p)

    def test_full_synthetic_pipeline_compare_is_readonly_and_subgroups_complete(self):
        data, ref, x, plan = fixture()
        refs = {s: ref for s in c.SEEDS}
        with temporary() as p, redirect_stdout(io.StringIO()):
            summaries = []
            for s in c.SEEDS:
                sub = p/f'seed{s}'
                predictions, choices = c.collect(sub, data, x, ref, c.binding_for(plan, s), s, runner=fake_fit)
                np.testing.assert_array_equal(predictions[c.PRIMARY], .5*ref['current']+.5*predictions['large518_ridge'])
                summary = c.save_results(sub, data, ref, predictions, choices, plan, s)
                summaries.append(summary)
                d = summary['diagnosis'][c.PRIMARY]
                self.assertAlmostEqual(d['common16_sse_gain']+d['remaining484_sse_gain'],
                    summary['metrics']['current']['sse']-summary['metrics'][c.PRIMARY]['sse'], places=8)
                for g in c.FOCUS:
                    self.assertIn(g, d['groups'])
            with patch.object(c, 'validate', return_value=(data, x, refs, plan)), patch.object(c.eng, 'fit_candidates', side_effect=AssertionError('No fitting')):
                result = c.compare(p)
                c.evaluate(p)
            self.assertEqual(result['seeds'], list(c.SEEDS))
            self.assertFalse(result['independent_holdout'])
            self.assertFalse(result['historical_seeds_in_aggregate'])
            self.assertFalse(result['automatic_adoption'])
            self.assertAlmostEqual(result['mean_gain_vs_current'], np.mean([s['metrics'][c.PRIMARY]['gain_vs_current'] for s in summaries]))
            (p/f'seed{c.SEEDS[0]}'/'common16.csv').write_text('tamper', encoding='utf-8')
            with self.assertRaises(ValueError): c.read_completed(p/f'seed{c.SEEDS[0]}', data, x, ref, plan, c.SEEDS[0])

    def test_aggregate_cannot_hide_subgroup_worsening_or_use_historical_seeds(self):
        data, ref, _, plan = fixture()
        p = {'current': ref['current'], 'large518_ridge': .5*(ref['current']+data.y), c.PRIMARY: .75*ref['current']+.25*data.y}
        summaries = [c.result_data(data, ref, p, [], plan, s)[0] for s in c.SEEDS]
        summaries[1]['diagnosis'][c.PRIMARY]['groups']['noiseQ4']['gain_vs_current'] = -.2
        r = c.aggregate(summaries)
        self.assertTrue(r['all_seeds_improve_overall'])
        self.assertFalse(r['all_seeds_nonworsening_by_group']['noiseQ4'])
        self.assertEqual(r['conclusion'], 'overall_improves_but_subgroup_tradeoff')
        with self.assertRaises(ValueError): c.aggregate(summaries[:2])
        bad = copy.deepcopy(summaries); bad[0]['seed'] = 42
        with self.assertRaises(ValueError): c.aggregate(bad)


if __name__ == '__main__':
    unittest.main()
