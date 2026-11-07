"""Synthetic-only checks. Never load the competition data or use a GPU."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts import evaluate_frozen_resnet as ev


class ConfigurationTests(unittest.TestCase):
    def test_alpha_defaults_are_unchanged_and_custom_cli_is_parsed(self):
        self.assertEqual(ev.validate_alphas(ev.parse_args([]).alphas), (10.0, 100.0, 1000.0))
        args = ev.parse_args(["--alphas", "10", "100", "1000", "3000", "10000", "30000"])
        self.assertEqual(ev.validate_alphas(args.alphas), (10.0, 100.0, 1000.0, 3000.0, 10000.0, 30000.0))

    def test_invalid_alpha_grids_are_rejected(self):
        for values in [(), (float("nan"),), (float("inf"),), (float("-inf"),), (0,), (-1,), (10, 10.0)]:
            with self.subTest(values=values):
                with self.assertRaisesRegex(ValueError, "--alphas"):
                    ev.validate_alphas(values)

    def test_invalid_alpha_and_seed_fail_before_reading_inputs(self):
        for args in [["--alphas", "nan"], ["--alphas", "0"], ["--alphas", "10", "10"], ["--seed", "-1"], ["--seed", str(2**32)]]:
            with self.subTest(args=args), patch.object(ev, "load_inputs") as loader:
                with self.assertRaises(ValueError):
                    ev.main(args)
                loader.assert_not_called()
        self.assertEqual(ev.validate_seed(0), 0)
        self.assertEqual(ev.validate_seed(2**32 - 5), 2**32 - 5)


class InputValidationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="frozen-eval-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.ids = np.array([f"TRAIN_{i:06d}" for i in range(50)])
        self.y = np.linspace(150, 250, 50)
        self.folds = np.arange(50) % 5
        self.train = [{"ID": i, "hardness": float(y)} for i, y in zip(self.ids, self.y)]
        self.fold_rows = [{"ID": i, "fold": int(f), "seed": 42} for i, f in zip(self.ids, self.folds)]
        self.baseline = [{"ID": i, "hardness": float(y), "fold": int(f), "ridge_alpha100": float(y + 1)} for i, y, f in zip(self.ids, self.y, self.folds)]
        self.audit = [{"ID": i, "hardness": float(y), "haar_hh_noise_proxy": j + 1.0} for j, (i, y) in enumerate(zip(self.ids, self.y))]
        for filename, rows in [("train.csv", self.train), ("folds.csv", self.fold_rows), ("baseline.csv", self.baseline), ("audit.csv", self.audit)]:
            self.write_csv(filename, rows)
        rng = np.random.default_rng(7)
        self.arrays = {"ids": self.ids[::-1], "final_gap": rng.normal(size=(50, 512)).astype(np.float32),
                       "multiscale_mean_std": rng.normal(size=(50, 1792)).astype(np.float32),
                       "metadata_json": np.array(json.dumps({"train_csv_sha256": ev.sha256(self.root / "train.csv")}))}
        self.write_npz()

    def write_csv(self, filename, rows):
        (self.root / filename).write_text(ev.csv_text(rows), encoding="utf-8")

    def write_npz(self):
        np.savez(self.root / "features.npz", **self.arrays)

    def load(self):
        return ev.load_inputs(self.root / "features.npz", self.root / "train.csv", self.root / "folds.csv",
                              self.root / "baseline.csv", self.root / "audit.csv", 42, expected_n=50)

    def test_valid_inputs_realign_features_by_id(self):
        data = self.load()
        np.testing.assert_array_equal(data.ids, self.ids)
        np.testing.assert_array_equal(data.features["final_gap"], self.arrays["final_gap"][::-1])

    def test_duplicate_ids_rejected(self):
        self.train[-1]["ID"] = self.train[0]["ID"]
        self.write_csv("train.csv", self.train)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.load()

    def test_missing_or_extra_ids_rejected(self):
        self.baseline[-1]["ID"] = "TRAIN_OTHER"
        self.write_csv("baseline.csv", self.baseline)
        with self.assertRaisesRegex(ValueError, "ID set"):
            self.load()

    def test_baseline_fold_and_target_mismatch_rejected(self):
        for key, value, message in [("fold", 4, "fold IDs"), ("hardness", 150.000001, "target values")]:
            with self.subTest(key=key):
                rows = [dict(row) for row in self.baseline]
                rows[0][key] = value
                self.write_csv("baseline.csv", rows)
                with self.assertRaisesRegex(ValueError, message):
                    self.load()

    def test_invalid_outer_folds_rejected(self):
        self.fold_rows[0]["fold"] = 9
        self.write_csv("folds.csv", self.fold_rows)
        with self.assertRaisesRegex(ValueError, "five equally"):
            self.load()

    def test_audit_tiny_csv_rounding_difference_allowed_and_recorded(self):
        self.audit[0]["hardness"] += 1e-12
        self.write_csv("audit.csv", self.audit)
        data = self.load()
        np.testing.assert_array_equal(data.y, self.y)
        alignment = data.provenance["audit_features"]["target_alignment"]
        self.assertGreater(alignment["max_absolute_difference"], 0)
        self.assertLess(alignment["max_absolute_difference"], 1e-10)
        self.assertEqual(alignment["rtol"], 0.0)

    def test_audit_meaningful_target_difference_rejected(self):
        self.audit[0]["hardness"] += 0.001
        self.write_csv("audit.csv", self.audit)
        with self.assertRaisesRegex(ValueError, "rounding tolerance"):
            self.load()

    def test_nonfinite_or_labelled_npz_rejected(self):
        self.arrays["final_gap"][0, 0] = np.nan
        self.write_npz()
        with self.assertRaisesRegex(ValueError, "NaN"):
            self.load()
        self.arrays["final_gap"][0, 0] = 0
        self.arrays["labels"] = self.y
        self.write_npz()
        with self.assertRaisesRegex(ValueError, "exactly these keys"):
            self.load()

    def test_metadata_train_hash_mismatch_rejected(self):
        self.arrays["metadata_json"] = np.array(json.dumps({"train_csv_sha256": "wrong"}))
        self.write_npz()
        with self.assertRaisesRegex(ValueError, "sha256"):
            self.load()

    def test_existing_output_requires_explicit_overwrite_before_fitting(self):
        output = self.root / "existing"
        output.mkdir()
        (output / "old.txt").write_text("preserve me", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "--overwrite"):
            ev.main(["--output-dir", str(output)])
        self.assertEqual((output / "old.txt").read_text(encoding="utf-8"), "preserve me")


class NestedEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(11)
        x = rng.normal(size=(50, 7))
        x[:, 0] = np.arange(50)  # Allows fit/predict sample-overlap checks.
        y = 170 + 4 * x[:, 1] + 2 * x[:, 2] + rng.normal(size=50)
        cls.data = ev.Inputs(np.array([f"TRAIN_{i:06d}" for i in range(50)]), y,
                             {"final_gap": x[:, :3], "multiscale_mean_std": x}, np.arange(50) % 5,
                             np.full(50, y.mean()), np.linspace(1, 10, 50), {}, {})
        cls.fit_sizes = []
        original_factory = ev.make_estimator
        class RecordingEstimator:
            def __init__(self, alpha):
                self.estimator = original_factory(alpha)
            def fit(self, features, target):
                self.ids = set(features[:, 0])
                cls.fit_sizes.append(len(features))
                self.estimator.fit(features, target)
                return self
            def predict(self, features):
                if self.ids & set(features[:, 0]):
                    raise AssertionError("A fitted sample appeared in validation prediction")
                return self.estimator.predict(features)
        with patch.object(ev, "make_estimator", RecordingEstimator):
            cls.predictions, cls.selections = ev.nested_predictions(cls.data, seed=42, threads=1)

    def test_no_fit_on_all_data_and_expected_fit_counts(self):
        self.assertEqual(len(self.fit_sizes), 100)  # 5 * (2 reps * 3 alpha * 3 inner + 2 outer).
        self.assertEqual(self.fit_sizes.count(40), 10)
        self.assertTrue(all(n < 50 for n in self.fit_sizes))
        self.assertEqual(set(self.predictions), {"final_gap_nested", "multiscale_mean_std_nested", "joint_nested"})
        self.assertTrue(all(np.isfinite(p).all() for p in self.predictions.values()))

    def test_joint_selection_uses_inner_scores(self):
        for record in self.selections:
            self.assertEqual(record["joint"]["inner_mse"], min(c["inner_mse"] for c in record["candidates"]))
            mask = self.data.folds == record["outer_fold"]
            np.testing.assert_array_equal(self.predictions["joint_nested"][mask],
                                          self.predictions[record["joint"]["representation"] + "_nested"][mask])

    def test_explicit_original_grid_preserves_default_predictions(self):
        predictions, selections = ev.nested_predictions(self.data, seed=42, threads=1, alphas=(10, 100, 1000))
        for name in self.predictions:
            np.testing.assert_array_equal(predictions[name], self.predictions[name])
        self.assertEqual(selections, self.selections)

    def test_extended_grid_propagates_to_search_and_reports(self):
        alphas = (10.0, 100.0, 1000.0, 3000.0, 10000.0, 30000.0)
        predictions, selections = ev.nested_predictions(self.data, seed=42, threads=1, alphas=alphas)
        for record in selections:
            self.assertEqual(len(record["candidates"]), 12)
            for representation in ev.REPRESENTATIONS:
                self.assertEqual(tuple(c["alpha"] for c in record["candidates"] if c["representation"] == representation), alphas)
        diagnostics, tables = ev.diagnose(self.data, predictions)
        with tempfile.TemporaryDirectory(prefix="frozen-eval-custom-grid-") as folder:
            out = Path(folder)
            ev.write_outputs(out, self.data, predictions, selections, diagnostics, tables, 42, 1, 0.0, alphas=alphas)
            summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["alpha_candidates"], list(alphas))
            self.assertIn("alpha 후보는 10, 100, 1000, 3000, 10000, 30000", (out / "report.md").read_text(encoding="utf-8"))
            with (out / "inner_search.csv").open(encoding="utf-8") as handle:
                search = list(csv.DictReader(handle))
            self.assertEqual(len(search), 60)
            self.assertEqual({float(row["alpha"]) for row in search}, set(alphas))

    def test_outer_validation_labels_cannot_affect_own_predictions(self):
        changed_y = self.data.y.copy()
        changed_y[self.data.folds == 0] += 1000
        changed = ev.Inputs(self.data.ids, changed_y, self.data.features, self.data.folds,
                            self.data.baseline, self.data.noise, {}, {})
        changed_predictions, changed_selections = ev.nested_predictions(changed, seed=42, threads=1)
        for name, pred in self.predictions.items():
            np.testing.assert_allclose(changed_predictions[name][self.data.folds == 0], pred[self.data.folds == 0])
        self.assertEqual(self.selections[0], changed_selections[0])

    def test_feature_and_target_scalers_fit_training_subset(self):
        estimator = ev.make_estimator(100)
        estimator.fit(self.data.features["final_gap"][:30], self.data.y[:30])
        np.testing.assert_allclose(estimator.regressor_.named_steps["standardscaler"].mean_, self.data.features["final_gap"][:30].mean(axis=0))
        np.testing.assert_allclose(estimator.transformer_.mean_, [self.data.y[:30].mean()])

    def test_diagnostics_and_reports_write_synthetic_results(self):
        diagnostics, tables = ev.diagnose(self.data, self.predictions)
        expected = np.sqrt(np.mean((self.predictions["joint_nested"] - self.data.y) ** 2))
        self.assertAlmostEqual(diagnostics["joint_nested"]["overall"]["rmse"], expected)
        self.assertEqual(len(tables["per_image_errors"]), 150)
        with tempfile.TemporaryDirectory(prefix="frozen-eval-report-") as folder:
            out = Path(folder)
            ev.write_outputs(out, self.data, self.predictions, self.selections, diagnostics, tables, 42, 1, 0.0)
            self.assertTrue({"summary.json", "report.md", "oof.csv", "per_image_errors.csv", "inner_search.csv"} <= {p.name for p in out.iterdir()})
            with (out / "oof.csv").open(encoding="utf-8") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 50)
            summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["n_train"], 50)


if __name__ == "__main__":
    unittest.main()
