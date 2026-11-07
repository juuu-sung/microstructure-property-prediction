"""Synthetic CPU tests; no real-data fitting, image inference, or GPU."""
import contextlib
import copy
from dataclasses import replace
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
from scripts import evaluate_local_structure_confirmation as e


class ConfirmationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(91)
        old, added = rng.normal(size=(50, 125)), rng.normal(size=(50, 60))
        features = {e.RAW: old, e.AUGMENTED: np.c_[old, added],
                    e.RESNET: rng.normal(size=(50, 12)), e.DINO: rng.normal(size=(50, 9))}
        y = 200 + 3*old[:, 0] + 2*added[:, 0] + rng.normal(size=50)
        cls.data = e.core.Inputs(np.array([f"TRAIN_{i:06d}" for i in range(50)]), y, features,
            e.outer_folds(y, e.SEEDS[0]), np.full(50, np.nan), rng.normal(size=50), {}, {})
        factory = e.core.make_estimator
        with contextlib.redirect_stdout(io.StringIO()), mock.patch.object(e.core, "make_estimator", wraps=factory) as calls:
            cls.baseline, cls.nested, cls.selections = e.predict_components(cls.data, e.SEEDS[0], 1)
        cls.fit_count = calls.call_count

    def test_exact_fit_budget_and_preserved_component_grids(self):
        self.assertEqual(self.fit_count, 250)
        self.assertEqual(set(self.nested), {n+"_nested" for n in e.GRIDS})
        self.assertEqual(len(self.selections), 5)
        for r in self.selections:
            self.assertNotIn("joint", r)
            self.assertEqual(len(r["candidates"]), 15)
            for n, alphas in e.GRIDS.items():
                candidates = [c for c in r["candidates"] if c["representation"] == n]
                self.assertEqual([c["alpha"] for c in candidates], list(alphas))
                self.assertEqual(r["per_representation"][n], min(candidates, key=lambda c: (c["inner_mse"], -c["alpha"])))

    def test_outer_labels_and_diagnostic_arrays_cannot_change_heldout_predictions(self):
        changed = copy.deepcopy(self.data)
        heldout = changed.folds == 0
        changed.y[heldout] += 1000
        changed.baseline[:] = 99999
        changed.noise *= -500
        with contextlib.redirect_stdout(io.StringIO()):
            baseline, nested, selected = e.predict_components(changed, e.SEEDS[0], 1)
        np.testing.assert_array_equal(baseline[heldout], self.baseline[heldout])
        for name in nested:
            np.testing.assert_array_equal(nested[name][heldout], self.nested[name][heldout])
        self.assertEqual(selected[0], self.selections[0])

    def test_fold_membership_is_deterministic_balanced_and_seeds_fixed(self):
        folds = [e.outer_folds(self.data.y, s) for s in e.SEEDS]
        for s, f in zip(e.SEEDS, folds):
            np.testing.assert_array_equal(f, e.outer_folds(self.data.y, s))
            self.assertEqual(np.bincount(f).tolist(), [10]*5)
        self.assertFalse(np.array_equal(folds[0], folds[1]))
        self.assertFalse(np.array_equal(folds[1], folds[2]))
        with self.assertRaises(ValueError):
            e.outer_folds(self.data.y, 42)

    def test_fixed_blends_use_new_components_and_replace_only_hand(self):
        result = e.blend(self.baseline, self.nested)
        rest = .25*self.nested[e.RESNET+"_nested"] + .5*self.nested[e.DINO+"_nested"]
        for name, hand in ((e.REFERENCE, self.baseline), (e.CONTROL, self.nested[e.RAW+"_nested"]),
                           (e.PRIMARY, self.nested[e.AUGMENTED+"_nested"])):
            np.testing.assert_array_equal(result[name], .25*hand+rest)
        with self.assertRaises(ValueError):
            e.blend(self.baseline, {e.RAW+"_nested": self.baseline})

    def test_prepare_and_validate_fit_nothing_and_validation_writes_nothing(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)/"confirmation"
            with mock.patch.object(e, "load_features", return_value=self.data), \
                 mock.patch.object(e.core, "make_estimator", side_effect=AssertionError("Unexpected fit")), \
                 mock.patch.object(e, "predict_components", side_effect=AssertionError("Unexpected fit")), \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(e.main(["--prepare", "--output-dir", str(path)]), 0)
                self.assertEqual({p.name for p in path.iterdir()}, {"protocol.json", *(f"folds_seed{s}.csv" for s in e.SEEDS)})
                before = {p.name: e.core.sha256(p) for p in path.iterdir()}
                self.assertEqual(e.main(["--validate-only", "--output-dir", str(path)]), 0)
                self.assertEqual(before, {p.name: e.core.sha256(p) for p in path.iterdir()})

    def test_frozen_fold_and_feature_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)/"confirmation"
            e.prepare(path, self.data)
            changed = copy.deepcopy(self.data)
            changed.features[e.AUGMENTED][0, 0] += .1
            with self.assertRaisesRegex(ValueError, "changed"):
                e.validate_plan(path, changed)
            f = path/f"folds_seed{e.SEEDS[0]}.csv"
            f.write_text(f.read_text()+"\n")
            with self.assertRaisesRegex(ValueError, "fold file"):
                e.validate_plan(path, self.data)

    def test_existing_outputs_and_input_directories_protected_before_fitting(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)/"confirmation"
            e.prepare(path, self.data)
            with self.assertRaisesRegex(ValueError, "already exists"):
                e.prepare(path, self.data)
            target = path/f"seed{e.SEEDS[1]}"
            target.mkdir()
            (target/"do_not_touch.txt").write_text("saved")
            with mock.patch.object(e, "load_features", return_value=self.data), mock.patch.object(e, "predict_components") as fit:
                with self.assertRaisesRegex(ValueError, "already exists"):
                    e.main(["--output-dir", str(path)])
                fit.assert_not_called()
            with self.assertRaisesRegex(ValueError, "protected"):
                e.prepare(e.ROOT/"open", self.data)

    def test_output_roundtrip_and_completion_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)/"confirmation"
            e.prepare(path, self.data)
            with mock.patch.object(e, "predict_components", return_value=(self.baseline, self.nested, self.selections)), contextlib.redirect_stdout(io.StringIO()):
                saved = e.run_seed(path, self.data, e.SEEDS[0], 1)
            self.assertEqual(e.read_completed(path, e.SEEDS[0], self.data), saved)
            self.assertFalse(saved["historical_oof_used"])
            dest = path/f"seed{e.SEEDS[0]}"
            with (dest/"oof.csv").open("a") as f:
                f.write("tampered")
            with self.assertRaisesRegex(ValueError, "changed"):
                e.read_completed(path, e.SEEDS[0], self.data)

    def test_no_aggregate_until_all_fixed_seeds_finish(self):
        with self.assertRaisesRegex(ValueError, "All three"):
            e.aggregate([])
        summaries = []
        for seed in e.SEEDS:
            summaries.append({"seed": seed, "diagnostics": {
                e.REFERENCE: {"overall": {"rmse": 14.}}, e.CONTROL: {"overall": {"rmse": 14.02}},
                e.PRIMARY: {"overall": {"rmse": 13.95, "delta_rmse_vs_current": -.05,
                    "delta_rmse_vs_tuned_control": -.07, "folds_improved_vs_current": 3, "folds_improved_vs_tuned_control": 4}}}})
        result = e.aggregate(summaries)
        self.assertTrue(result["all_seeds_improve_both"])
        self.assertFalse(result["meets_priority_rule"])
        self.assertAlmostEqual(result["mean_gain_vs_reference"], .05)
        self.assertFalse(result["historical_seeds_in_aggregate"])
        summaries[-1]["diagnostics"][e.PRIMARY]["overall"]["delta_rmse_vs_tuned_control"] = .01
        self.assertFalse(e.aggregate(summaries)["all_seeds_improve_both"])
        with self.assertRaisesRegex(ValueError, "All three"):
            e.aggregate(list(reversed(summaries)))

    def test_input_manifest_contains_no_historical_oof_or_test_files(self):
        paths = [str(p).lower() for p in e.input_paths().values()]
        self.assertTrue(all("oof" not in p and "test" not in p for p in paths))
        self.assertEqual(e.SEEDS, (31415, 27182, 16180))
        self.assertFalse(e.PROTOCOL["historical_oof_used"])

    def test_cli_import_has_no_torch(self):
        p = subprocess.run([sys.executable, "-c", "import sys; from scripts import evaluate_local_structure_confirmation; assert 'torch' not in sys.modules"],
                           cwd=e.ROOT, capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stderr)


if __name__ == "__main__":
    unittest.main()
