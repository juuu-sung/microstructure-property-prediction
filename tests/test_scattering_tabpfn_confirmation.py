"""Synthetic/mocked tests; no competition fits, image extraction or GPU use."""
import contextlib
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from scripts import evaluate_scattering_tabpfn_confirmation as c


@contextlib.contextmanager
def temporary():
    parent = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory(dir=parent, prefix='.scattering_confirmation_test_') as td:
        if Path(td).resolve().parent != parent:
            raise ValueError('Temporary cleanup path outside tests')
        yield Path(td)


def fixture():
    rng = np.random.default_rng(102)
    x = rng.normal(size=(500, 486)); x[:, 0] = np.arange(500)
    data = SimpleNamespace(ids=np.array([f'SYNTHETIC_{j:06d}' for j in range(500)]),
                           y=100+3*x[:, 1], noise=np.arange(500.))
    refs = {s: {'folds': (np.arange(500)+j) % 5, 'base': data.y+4, 'current': data.y+3,
                'tabpfn285': data.y+1, 'control': data.y+2.5, 'runtime': {'synthetic': True}}
            for j, s in enumerate(c.SEEDS)}
    plan = {'frozen': {'synthetic': True, 'common_failure_ids': data.ids[:16].tolist()}}
    return data, x, refs, plan


class FakePredictor:
    runtime = {'synthetic': True}
    def __init__(self): self.calls = []
    def __call__(self, x, y, q, seed):
        np.testing.assert_array_equal(y, 100+3*x[:, 1])
        if set(x[:, 0]) & set(q[:, 0]): raise AssertionError('Context/query overlap')
        self.calls.append((x.copy(), y.copy(), q.copy(), seed))
        return 100+3*q[:, 1], {'seconds': 0.}


class ConfirmationTests(unittest.TestCase):
    def test_three_seed_end_to_end_only15_fits_resume_without_predictor_and_tamper(self):
        data, x, refs, plan = fixture(); predictor = FakePredictor()
        with temporary() as folder, contextlib.redirect_stdout(io.StringIO()):
            with patch.object(c, 'validate', return_value=(data, x, refs, plan)), \
                 patch.object(c.t, 'Predictor', return_value=predictor), \
                 patch.object(c.t, 'verify_weights', return_value=Path('synthetic.ckpt')):
                for seed in c.SEEDS: c.evaluate_seed(folder, 'cuda', seed)
                self.assertEqual(len(predictor.calls), 15)
                for tx, ty, qx, model_seed in predictor.calls:
                    self.assertEqual((tx.shape, qx.shape), ((400,486),(100,486)))
                self.assertEqual([a[3] for a in predictor.calls], [s+1009*f+9 for s in c.SEEDS for f in range(5)])
                with patch.object(c.t, 'Predictor', side_effect=AssertionError('Resume must not construct a model')):
                    for seed in c.SEEDS: c.evaluate_seed(folder, 'cuda', seed)
                    c.compare(folder)
                result = c.t.read_json(folder/'confirmation_summary.json')
                self.assertEqual(result['seeds'], list(c.SEEDS))
                self.assertAlmostEqual(result['mean_gain_vs_current'], 1.)
                self.assertAlmostEqual(result['mean_gain_vs_control'], .5)
                self.assertTrue(result['all_seeds_improve_both'])
                self.assertFalse(result['historical_seeds_in_aggregate'])
                for seed in c.SEEDS:
                    s = c.read_completed(folder, data, x, refs[seed], plan, seed)
                    self.assertEqual(s['folds_improved_vs_control'], 5)
                    self.assertAlmostEqual(s['metrics'][c.PRIMARY]['rmse'], 2.)
                # Tampered summary is rejected even if its file hash is updated in the manifest.
                target = folder/f'seed{c.SEEDS[0]}'
                s = c.t.read_json(target/'summary.json'); s['metrics'][c.PRIMARY]['rmse'] = 0.
                c.t.write_json(target/'summary.json', s)
                done = c.t.read_json(target/'completed_evaluation.json')
                done['files']['summary.json'] = c.t.common.sha256_file(target/'summary.json')
                c.t.write_json(target/'completed_evaluation.json', done)
                with self.assertRaises(ValueError): c.compare(folder)

    def test_interrupted_run_reuses_completed_cells_and_isolates_validation_labels(self):
        data, x, refs, plan = fixture(); seed = c.SEEDS[0]; ref = refs[seed]
        predictor = FakePredictor(); binding = c.binding_for(plan, ref, seed)
        with temporary() as folder, contextlib.redirect_stdout(io.StringIO()):
            vi, first = c.experiment.evaluate_fold(folder/f'seed{seed}', binding, x, data.y,
                                                  ref['folds'], seed, 0, predictor)
            changed = data.y.copy(); changed[vi] += 99999
            _, second = c.experiment.evaluate_fold(folder/'isolation', binding, x, changed,
                                                    ref['folds'], seed, 0, predictor)
            np.testing.assert_array_equal(first, second)
            with patch.object(c, 'validate', return_value=(data, x, refs, plan)), \
                 patch.object(c.t, 'Predictor', return_value=predictor), \
                 patch.object(c.t, 'verify_weights', return_value=Path('synthetic.ckpt')):
                c.evaluate_seed(folder, 'cuda', seed)
            self.assertEqual(len(predictor.calls), 6)  #2 isolated calls,4 remaining outer folds

    def test_prepare_is_idempotent_rejects_changed_folds_and_sources(self):
        data, x, refs, plan = fixture()
        with temporary() as root, contextlib.redirect_stdout(io.StringIO()):
            folder = root/'out'
            with patch.object(c, 'load_inputs', return_value=(data,x,refs,plan['frozen'])), \
                 patch.object(c.t, 'Predictor', side_effect=AssertionError('No inference in preparation')):
                c.prepare(folder)
                sha = c.t.common.sha256_file(folder/'protocol.json')
                c.prepare(folder); c.validate(folder)
                self.assertEqual(sha, c.t.common.sha256_file(folder/'protocol.json'))
                with (folder/f'folds_seed{c.SEEDS[0]}.csv').open('a') as h: h.write('corrupt,99\n')
                with self.assertRaises(ValueError): c.validate(folder)
            with patch.object(c, 'load_inputs', return_value=(data,x,refs,{'changed':True})):
                with self.assertRaises(ValueError): c.validate(folder)

    def test_archived_reference_uses_matching_fold_and_both_controls_no_refits(self):
        data, x, refs, _ = fixture(); seed = c.SEEDS[0]; ref = refs[seed]
        p185, p285 = data.y+2, data.y+1
        choices = [{'fold': f, 'representation': 'morphology285'} for f in range(5)]
        rows = [{'ID': str(i), 'hardness': float(data.y[j]), 'fold': int(ref['folds'][j]),
                 'current': float(ref['base'][j]), 'tabpfn_existing185': float(p185[j]),
                 'tabpfn_morphology285': float(p285[j]), c.t.PRIMARY: float(p285[j]),
                 c.archive.PRIMARY: float(.5*(ref['base'][j]+p285[j]))} for j, i in enumerate(data.ids)]
        def cached(features, y, folds, seed, fold, folder, binding, predictor):
            with self.assertRaises(ValueError): predictor(None)
            vi = np.flatnonzero(folds == fold)
            return vi, {'existing185': p185[vi], 'morphology285': p285[vi], c.t.PRIMARY:p285[vi]}, choices[fold]
        with temporary() as root:
            folder = root/f'seed{seed}'; folder.mkdir()
            c.t.write_csv(folder/'oof.csv', list(reversed(rows)))
            c.t.write_json(folder/'completed_evaluation.json', {'synthetic':True})
            summary = {'binding': {'runtime':ref['runtime']}, 'selection': choices}
            with patch.object(c.archive, 'OUT', root), patch.object(c.archive, 'read_completed', return_value=summary), \
                 patch.object(c.t, 'evaluate_fold', side_effect=cached):
                actual = c.archived_reference(data, {}, {'current':ref['base'], 'folds':ref['folds']}, {}, seed)
                np.testing.assert_array_equal(actual['base'], ref['base'])
                np.testing.assert_array_equal(actual['tabpfn285'], p285)
                np.testing.assert_array_equal(actual['current'], actual['control'])
                rows[0]['fold'] = 99; c.t.write_csv(folder/'oof.csv', rows)
                with self.assertRaises(AssertionError):
                    c.archived_reference(data, {}, {'current':ref['base'], 'folds':ref['folds']}, {}, seed)

    def test_mismatched_runtime_and_arbitrary_seed_fail_before_fit(self):
        data, x, refs, plan = fixture(); predictor = FakePredictor(); predictor.runtime = {'wrong':True}
        with temporary() as folder, patch.object(c,'validate',return_value=(data,x,refs,plan)), \
             patch.object(c.t,'Predictor',return_value=predictor), \
             patch.object(c.t,'verify_weights',return_value=Path('synthetic.ckpt')):
            with self.assertRaises(ValueError): c.evaluate_seed(folder,'cpu',c.SEEDS[0])
            with self.assertRaises(ValueError): c.evaluate_seed(folder,'cuda',42)
            self.assertEqual(len(predictor.calls),0)

    def test_aggregate_requires_all_three_positive_against_both_no_historical(self):
        def summaries(gains, control_gains):
            return [{'seed':s,'primary':c.PRIMARY,'folds_improved_vs_current':3,'folds_improved_vs_control':3,
                     'metrics':{'current':{'rmse':14.},'fixed285_control_blend':{'rmse':14.-g+h},
                                c.PRIMARY:{'rmse':14.-g,'gain_vs_current':g,'gain_vs_control':h}}}
                    for s,g,h in zip(c.SEEDS,gains,control_gains,strict=True)]
        s = summaries([.1,.2,.15],[.05,.1,.08])
        self.assertTrue(c.aggregate(s)['all_seeds_improve_both'])
        self.assertFalse(c.aggregate(s)['passes_expansion_priority'])
        s = summaries([.4,.4,.4],[.1,.1,-.01])
        self.assertFalse(c.aggregate(s)['all_seeds_improve_both'])
        self.assertFalse(c.aggregate(s)['passes_expansion_priority'])
        s = summaries([.4,.4,.4],[.1,.1,.1])
        self.assertTrue(c.aggregate(s)['passes_expansion_priority'])
        with self.assertRaises(ValueError): c.aggregate(s[:2])
        s[0]['seed'] = 42
        with self.assertRaises(ValueError): c.aggregate(s)


if __name__ == '__main__': unittest.main()
