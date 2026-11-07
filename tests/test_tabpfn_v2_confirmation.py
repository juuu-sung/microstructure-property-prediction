"""Synthetic/mock confirmation tests; no competition fits or model inference."""
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from scripts import evaluate_tabpfn_v2_confirmation as c


def fixture():
    rng = np.random.default_rng(731)
    x = rng.normal(size=(500, 285))
    x[:, 0] = np.arange(500)
    data = SimpleNamespace(ids=np.array([f"SYNTHETIC_{i:03d}" for i in range(500)]),
                           y=x[:, 1].copy(), noise=np.arange(500.))
    features = {"existing185": x[:, :185].copy(), "morphology285": x.copy()}
    refs = {s: {"current": data.y+2, "folds": np.repeat(np.arange(5), 100)} for s in c.SEEDS}
    plan = {"frozen": {"synthetic": True, "common_failure_ids": data.ids[:17].tolist()}}
    return data, features, refs, plan


class FakePredictor:
    runtime = {"synthetic": True}

    def __init__(self):
        self.calls = 0

    def __call__(self, x, y, q, seed):
        self.calls += 1
        np.testing.assert_array_equal(y, x[:, 1])
        if set(x[:, 0]) & set(q[:, 0]):
            raise AssertionError("Context/query overlap")
        return q[:, 1]+(0 if x.shape[1] == 185 else .5), {"seconds": 0.}


class ConfirmationTests(unittest.TestCase):
    def test_baseline_is_augmented_same_fold_not_old_reference(self):
        data, _, _, _ = fixture()
        seed = c.SEEDS[0]
        folds = c.baseline.outer_folds(data.y, seed)
        rows = [{"ID": str(i), "hardness": str(data.y[j]), "fold": str(folds[j]),
                 c.baseline.AUGMENTED+"_nested": "1", c.baseline.RESNET+"_nested": "3",
                 c.baseline.DINO+"_nested": "5", c.baseline.PRIMARY: "3.5",
                 c.baseline.REFERENCE: "99999"} for j, i in enumerate(data.ids)]
        ref = c.baseline_reference(data, list(reversed(rows)), seed)
        np.testing.assert_array_equal(ref["current"], np.full(500, 3.5))
        for key, bad in (("fold", "99"), ("hardness", "9999"), (c.baseline.PRIMARY, "99999")):
            mutated = [dict(r) for r in rows]
            mutated[0][key] = bad
            with self.assertRaises(ValueError):
                c.baseline_reference(data, mutated, seed)

    def test_end_to_end_three_seeds_resume_and_corruption(self):
        data, features, refs, plan = fixture()
        predictor = FakePredictor()
        with tempfile.TemporaryDirectory() as td, contextlib.redirect_stdout(io.StringIO()):
            root = Path(td)
            c.t.write_json(root/"protocol.json", plan)
            with patch.object(c, "validate", return_value=(data, features, refs, plan)), \
                 patch.object(c.t, "Predictor", return_value=predictor), \
                 patch.object(c.t, "verify_weights", return_value=root/"synthetic.ckpt"):
                for seed in c.SEEDS:
                    c.evaluate_seed(root, "cpu", seed)
                self.assertEqual(predictor.calls, 120)
                c.evaluate_seed(root, "cpu", c.SEEDS[0])
                self.assertEqual(predictor.calls, 120)
                c.compare(root)
                result = c.t.read_json(root/"confirmation_summary.json")
                self.assertTrue(result["all_seeds_improve"])
                self.assertTrue(result["meets_priority_rule"])
                self.assertEqual(result["primary"], c.PRIMARY)
                self.assertAlmostEqual(result["mean_gain_vs_current"], 1.)
                s = c.read_completed(root, data, refs[c.SEEDS[0]], plan, c.SEEDS[0])
                self.assertAlmostEqual(s["metrics"][c.t.PRIMARY]["rmse"], 0.)
                self.assertAlmostEqual(s["metrics"][c.PRIMARY]["rmse"], 1.)
                self.assertEqual(s["folds_improved"], 5)
                with (root/f"seed{c.SEEDS[0]}/oof.csv").open("a") as f:
                    f.write("corrupt")
                with self.assertRaises(ValueError):
                    c.compare(root)

    def test_aggregate_requires_every_fixed_seed_and_reports_mixed_results(self):
        def summaries(gains):
            return [{"seed": s, "primary": c.PRIMARY, "folds_improved": 3,
                     "metrics": {"current": {"rmse": 14.}, c.PRIMARY: {"rmse": 14.-g, "gain_vs_current": g}}}
                    for s, g in zip(c.SEEDS, gains)]
        small = c.aggregate(summaries([.04, .06, .05]))
        self.assertTrue(small["all_seeds_improve"])
        self.assertFalse(small["meets_priority_rule"])
        self.assertAlmostEqual(small["mean_gain_vs_current"], .05)
        mixed = c.aggregate(summaries([.5, -.01, .3]))
        self.assertFalse(mixed["all_seeds_improve"])
        self.assertFalse(mixed["meets_priority_rule"])
        with self.assertRaises(ValueError):
            c.aggregate(summaries([.1, .2]))
        wrong = summaries([.1, .2, .3]); wrong[0]["seed"] = 42
        with self.assertRaises(ValueError):
            c.aggregate(wrong)
        wrong = summaries([.1, .2, .3]); wrong[0]["primary"] = c.t.PRIMARY
        with self.assertRaises(ValueError):
            c.aggregate(wrong)

    def test_preparation_is_idempotent_and_rejects_changed_folds_or_sources(self):
        data, features, refs, plan = fixture()
        with tempfile.TemporaryDirectory() as td, contextlib.redirect_stdout(io.StringIO()):
            root = Path(td)/"confirmation"
            with patch.object(c, "load_inputs", return_value=(data, features, refs, plan["frozen"])), \
                 patch.object(c.t, "Predictor", side_effect=AssertionError("No inference in preparation")):
                c.prepare(root)
                digest = c.t.common.sha256_file(root/"protocol.json")
                c.prepare(root)
                self.assertEqual(c.t.common.sha256_file(root/"protocol.json"), digest)
                c.validate(root)
                with (root/f"folds_seed{c.SEEDS[0]}.csv").open("a") as f:
                    f.write("corrupt,99\n")
                with self.assertRaises(ValueError):
                    c.validate(root)
            with patch.object(c, "load_inputs", return_value=(data, features, refs, {"changed": True})):
                with self.assertRaises(ValueError):
                    c.validate(root)


if __name__ == "__main__":
    unittest.main()
