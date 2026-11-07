"""Synthetic-only Kernel Ridge checks; no competition data or neural forward."""
import contextlib
import copy
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from scripts import evaluate_dinov2_kernel as kernel
from scripts import evaluate_frozen_resnet as core
try:
    from . import test_feature_fusion as fixture_module
except ImportError:
    import test_feature_fusion as fixture_module


class KernelArtifactTests(unittest.TestCase):
    def setUp(self):
        # Reuse the synthetic cache/old-OOF builder. These paths are all tempfiles;
        # its extra handcrafted cache lets us prove the new evaluator ignores it.
        self.fx = fixture_module.FusionArtifactTests()
        self.addCleanup(self.fx.doCleanups)
        self.fx.setUp()
        self.root = self.fx.root

    def load(self, search_profile="original"):
        return kernel.load_inputs(self.root / "features.npz", self.root / "train.csv", self.root / "folds.csv",
            self.root / "baseline.csv", self.root / "audit.csv", self.root / "frozen.csv", self.root / "dino.csv", 42,
            previous_dino_summary=self.root / "dino_summary.json", expected_n=50, search_profile=search_profile)

    def argv(self):
        return ["--features", str(self.root / "features.npz"), "--train-csv", str(self.root / "train.csv"),
            "--folds", str(self.root / "folds.csv"), "--baseline", str(self.root / "baseline.csv"),
            "--audit-features", str(self.root / "audit.csv"), "--previous-frozen-oof", str(self.root / "frozen.csv"),
            "--previous-dino-oof", str(self.root / "dino.csv"), "--previous-dino-summary", str(self.root / "dino_summary.json"),
            "--output-dir", str(self.root / "out")]

    def test_valid_cache_references_identical_aliases_no_hand_read_or_mutation(self):
        before = {p: core.sha256(p) for p in self.root.iterdir() if p.is_file()}
        with mock.patch.object(kernel.references_module, "load_handcrafted", side_effect=AssertionError("must not read hand cache")):
            data, refs = self.load()
        first = data.features[kernel.BANDWIDTH_FAMILIES[0]]
        np.testing.assert_array_equal(first, self.fx.dino_features)
        for values in data.features.values():
            self.assertIs(values, first)
            self.assertFalse(values.flags.writeable)
        expected = 0.5 * (0.5 * self.fx.baseline + 0.5 * self.fx.frozen) + 0.5 * self.fx.previous_dino
        np.testing.assert_array_equal(refs[kernel.CURRENT_REFERENCE], expected)
        self.assertFalse(data.provenance["kernel_inputs"]["reference_predictions_used_as_features"])
        self.assertFalse(data.provenance["kernel_inputs"]["handcrafted_feature_cache_read"])
        self.assertEqual(before, {path: core.sha256(path) for path in before})

    def test_lower_inputs_use_same_features_then_original_is_unchanged(self):
        before = {p: core.sha256(p) for p in self.root.iterdir() if p.is_file()}
        lower, lower_refs = self.load("lower")
        original, original_refs = self.load()
        self.assertEqual(tuple(lower.features), ("rbf_gamma0p01", "rbf_gamma0p03", "rbf_gamma0p1"))
        self.assertEqual(tuple(original.features), kernel.BANDWIDTH_FAMILIES)
        for values in lower.features.values():
            np.testing.assert_array_equal(values, original.features["rbf_gamma0p1"])
            self.assertIs(values, lower.features["rbf_gamma0p1"])
        for name in original_refs:
            np.testing.assert_array_equal(lower_refs[name], original_refs[name])
        self.assertEqual(kernel.ALPHAS, (0.1, 1.0, 10.0))
        self.assertEqual(kernel.GAMMA_MULTIPLIERS, {"rbf_gamma0p1": 0.1, "rbf_gamma1": 1.0, "rbf_gamma10": 10.0})
        self.assertEqual(before, {path: core.sha256(path) for path in before})

    def test_corrupt_cache_old_oof_or_summary_rejected(self):
        feature_path = self.root / "features.npz"
        original = feature_path.read_bytes()
        with np.load(feature_path, allow_pickle=False) as cache:
            arrays = {name: cache[name].copy() for name in cache.files}
        arrays["cls_patch_mean_std"][0, 400] += 1
        np.savez(feature_path, **arrays)
        with self.assertRaisesRegex(ValueError, "integrity hash"):
            self.load()
        feature_path.write_bytes(original)
        for column in ("fold", "hardness", "previous_blend_dino_equal", core.BASELINE_NAME):
            with self.subTest(column=column):
                changed = copy.deepcopy(self.fx.reference_rows)
                changed[0][column] += 1
                self.fx.save_rows("dino.csv", changed)
                with self.assertRaises(ValueError):
                    self.load()
        self.fx.save_reference()
        self.fx.summary["input_files"]["features"]["sha256"] = "0" * 64
        self.fx.save_summary()
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            self.load()

    def test_reference_changes_cannot_change_features_or_primary_selection(self):
        first, ref1 = self.load()
        self.fx.previous_dino += np.linspace(-2, 2, 50)
        self.fx.save_reference()
        second, ref2 = self.load()
        self.assertFalse(np.array_equal(ref1[kernel.CURRENT_REFERENCE], ref2[kernel.CURRENT_REFERENCE]))
        for family in kernel.BANDWIDTH_FAMILIES:
            np.testing.assert_array_equal(first.features[family], second.features[family])
        with contextlib.redirect_stdout(io.StringIO()):
            p1, s1 = kernel.nested_predictions(first, 42, 1)
            p2, s2 = kernel.nested_predictions(second, 42, 1)
        np.testing.assert_array_equal(p1[core.PRIMARY_NAME], p2[core.PRIMARY_NAME])
        self.assertEqual(s1, s2)

    def test_validate_only_has_no_fit_or_output_and_overwrite_protects_inputs(self):
        loader = kernel.load_inputs
        def small_loader(*args, **kwargs):
            return loader(*args, **kwargs, expected_n=50)
        for profile in ("original", "lower"):
            output = io.StringIO()
            with mock.patch.object(kernel, "load_inputs", side_effect=small_loader), \
                 mock.patch.object(kernel, "nested_predictions") as nested, mock.patch.object(kernel, "make_estimator") as factory, \
                 contextlib.redirect_stdout(output):
                self.assertEqual(kernel.main(self.argv() + ["--validate-only", "--search-profile", profile]), 0)
                nested.assert_not_called()
                factory.assert_not_called()
            recorded = json.loads(output.getvalue())
            self.assertEqual(recorded["search_profile"], profile)
            self.assertEqual(recorded["alpha_candidates"], list(kernel.search_parameters(profile)[0]))
            self.assertEqual(tuple(recorded["bandwidth_families"]), kernel.search_parameters(profile)[1])
        self.assertFalse((self.root / "out").exists())
        (self.root / "out").mkdir()
        old = self.root / "out/old.txt"
        old.write_text("preserve", encoding="utf-8")
        with mock.patch.object(kernel, "load_inputs") as loader:
            with self.assertRaisesRegex(ValueError, "--overwrite"):
                kernel.main(self.argv())
            loader.assert_not_called()
            with self.assertRaisesRegex(ValueError, "protected input"):
                kernel.main(self.argv() + ["--output-dir", str(self.root), "--overwrite"])
            loader.assert_not_called()
        self.assertEqual(old.read_text(encoding="utf-8"), "preserve")

    def test_seed_defaults_overrides_and_invalid_seed(self):
        for seed in (42, 2026):
            args = kernel.parse_args(["--seed", str(seed)])
            self.assertEqual(args.output_dir, kernel.ROOT / f"reports/dinov2_kernel/seed{seed}")
            self.assertEqual(args.baseline.name, f"benchmark_oof_seed{seed}.csv")
            self.assertEqual(args.folds.name, "folds.csv" if seed == 42 else "benchmark_folds_seed2026.csv")
            self.assertEqual(args.previous_dino_oof, kernel.ROOT / f"reports/frozen_dinov2_small/seed{seed}/oof.csv")
            self.assertEqual(args.previous_dino_summary, args.previous_dino_oof.with_name("summary.json"))
            self.assertEqual(args.search_profile, "original")
            lower = kernel.parse_args(["--seed", str(seed), "--search-profile", "lower"])
            self.assertEqual(lower.output_dir, kernel.ROOT / f"reports/dinov2_kernel_lower/seed{seed}")
            self.assertEqual(lower.folds, args.folds)
            self.assertEqual(lower.previous_dino_oof, args.previous_dino_oof)
        args = kernel.parse_args(["--seed", "2026", "--folds", "custom.csv", "--output-dir", "custom_out"])
        self.assertEqual(args.folds, Path("custom.csv"))
        self.assertEqual(args.output_dir, Path("custom_out"))
        lower = kernel.parse_args(["--search-profile", "lower", "--output-dir", "lower_custom"])
        self.assertEqual(lower.output_dir, Path("lower_custom"))
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            kernel.parse_args(["--search-profile", "arbitrary"])
        with mock.patch.object(kernel, "load_inputs") as loader:
            with self.assertRaisesRegex(ValueError, "seed"):
                kernel.main(self.argv() + ["--seed", "-1", "--validate-only"])
            loader.assert_not_called()


class KernelEstimatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(19)
        cls.X = rng.normal(size=(50, 1152))
        cls.X[:, 0] = np.arange(50) / 10
        cls.y = 200 + 8 * np.sin(cls.X[:, 0]) + 3 * cls.X[:, 1]
        cls.data = core.Inputs(np.array([f"TRAIN_{i+1:06d}" for i in range(50)]), cls.y,
            {family: cls.X for family in kernel.BANDWIDTH_FAMILIES}, np.arange(50) % 5,
            cls.y + np.sin(np.arange(50)), np.arange(50), {}, {})
        cls.calls = []
        original_factory = kernel.make_estimator
        class RecordingEstimator:
            def __init__(self, alpha, family):
                self.estimator = original_factory(alpha, family)
                self.alpha, self.family = alpha, family
            def fit(self, X, y):
                self.ids = set(X[:, 0])
                cls.calls.append((self.alpha, self.family, len(y)))
                self.estimator.fit(X, y)
                return self
            def predict(self, X):
                if self.ids & set(X[:, 0]):
                    raise AssertionError("Validation sample appeared in this estimator's fit")
                return self.estimator.predict(X)
        with mock.patch.object(kernel, "make_estimator", RecordingEstimator), contextlib.redirect_stdout(io.StringIO()):
            cls.predictions, cls.selections = kernel.nested_predictions(cls.data, 42, 1)

    def test_scalers_fit_train_only_gamma_is_fixed_and_inverse_is_original_units(self):
        train_x, train_y = self.X[:30], self.y[:30]
        for family in kernel.BANDWIDTH_FAMILIES:
            estimator = kernel.make_estimator(1, family).fit(train_x, train_y)
            steps = estimator.regressor_.named_steps
            np.testing.assert_allclose(steps["standardscaler"].mean_, train_x.mean(axis=0))
            np.testing.assert_allclose(estimator.transformer_.mean_, [train_y.mean()])
            transformed_y = estimator.transformer_.transform(train_y.reshape(-1, 1))
            self.assertTrue(np.any(transformed_y < 0) and np.any(transformed_y > 0))
            self.assertEqual(steps["kernelridge"].gamma, kernel.GAMMA_MULTIPLIERS[family] / 1152)
            self.assertEqual(steps["kernelridge"].kernel, "rbf")
            raw = estimator.regressor_.predict(self.X[30:])
            np.testing.assert_allclose(estimator.predict(self.X[30:]), raw * train_y.std() + train_y.mean())

    def test_far_input_returns_this_training_subset_target_mean(self):
        estimator = kernel.make_estimator(0.1, "rbf_gamma1").fit(self.X[:30], self.y[:30])
        far_x = np.full((1, 1152), 1e6)
        np.testing.assert_allclose(estimator.predict(far_x), [self.y[:30].mean()], rtol=0, atol=1e-12)
        self.assertNotAlmostEqual(float(estimator.predict(far_x)[0]), float(self.y.mean()), places=3)

    def test_nine_candidates_joint_inner_selection_and_expected_fit_count(self):
        self.assertEqual(len(self.calls), 150)  # 5*(3 bandwidths*3 alpha*3 inner + 3 outer).
        self.assertEqual(sum(n == 40 for _, _, n in self.calls), 15)
        self.assertTrue(all(n < 50 for _, _, n in self.calls))
        for record in self.selections:
            self.assertEqual(len(record["candidates"]), 9)
            self.assertEqual(record["joint"]["inner_mse"], min(c["inner_mse"] for c in record["candidates"]))
            for candidate in record["candidates"]:
                self.assertEqual(candidate["gamma"], kernel.GAMMA_MULTIPLIERS[candidate["representation"]] / 1152)
                self.assertAlmostEqual(candidate["inner_mse"], np.average(candidate["inner_fold_mse"], weights=record["inner_validation_sizes"]))
            mask = self.data.folds == record["outer_fold"]
            np.testing.assert_array_equal(self.predictions[core.PRIMARY_NAME][mask],
                self.predictions[record["joint"]["representation"] + "_nested"][mask])

    def test_outer_validation_target_change_cannot_affect_own_selection(self):
        changed = copy.copy(self.data)
        changed.y = self.y.copy()
        changed.y[changed.folds == 0] += 1000
        with contextlib.redirect_stdout(io.StringIO()):
            predictions, selections = kernel.nested_predictions(changed, 42, 1)
        self.assertEqual(selections[0], self.selections[0])
        for key in predictions:
            np.testing.assert_array_equal(predictions[key][changed.folds == 0], self.predictions[key][changed.folds == 0])

    def test_fixed_blend_search_gamma_fields_and_report_primary(self):
        references = {kernel.CURRENT_REFERENCE: self.y + np.sin(np.arange(50)),
            kernel.DINO_REFERENCE: self.y + np.cos(np.arange(50)), kernel.HAND_DINO_REFERENCE: self.y + 2 * np.sin(np.arange(50))}
        predictions = kernel.add_fixed_candidate(self.predictions, references)
        np.testing.assert_array_equal(predictions[kernel.FIXED_BLEND], 0.5 * references[kernel.CURRENT_REFERENCE] + 0.5 * self.predictions[core.PRIMARY_NAME])
        np.testing.assert_array_equal(predictions[core.PRIMARY_NAME], self.predictions[core.PRIMARY_NAME])
        diagnostics, tables = kernel.diagnose(self.data, predictions)
        expected_delta = diagnostics[core.PRIMARY_NAME]["overall"]["rmse"] - diagnostics[kernel.DINO_REFERENCE]["overall"]["rmse"]
        self.assertEqual(diagnostics[core.PRIMARY_NAME]["overall"]["delta_rmse_vs_dino_linear"], expected_delta)
        with tempfile.TemporaryDirectory(prefix="kernel-report-test-") as folder:
            out = Path(folder)
            kernel.write_outputs(out, self.data, predictions, self.selections, diagnostics, tables, 42, 1, 0.0)
            summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["primary_metric"], "joint_nested overall OOF RMSE")
            self.assertEqual(summary["alpha_candidates"], [0.1, 1.0, 10.0])
            self.assertFalse(summary["kernel_centering"])
            self.assertEqual(summary["source_feature_dimension"], 1152)
            with (out / "inner_search.csv").open(encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 45)
            self.assertEqual(sum(row["selected_joint"] == "True" for row in rows), 5)
            self.assertEqual(sum(row["selected_for_representation"] == "True" for row in rows), 15)
            for row in rows:
                self.assertEqual(float(row["gamma"]), float(row["gamma_multiplier"]) / 1152)
                self.assertIn("inner_fold2_mse", row)
            report = (out / "report.md").read_text(encoding="utf-8")
            self.assertIn("DINO 선형 대비", report)
            self.assertIn("해당 학습 fold의 평균", report)
            self.assertTrue((out / "target_region_metrics.csv").is_file())

    def test_nonidentical_or_invalid_feature_inputs_stop_before_fitting(self):
        for kind in ("different", "width", "nonfinite"):
            changed = copy.copy(self.data)
            changed.features = {family: self.X.copy() for family in kernel.BANDWIDTH_FAMILIES}
            if kind == "different":
                changed.features["rbf_gamma1"][0, 0] += 1
            elif kind == "width":
                changed.features = {family: value[:, :-1] for family, value in changed.features.items()}
            else:
                changed.features[kernel.BANDWIDTH_FAMILIES[0]][0, 0] = np.nan
            with mock.patch.object(kernel, "make_estimator") as factory:
                with self.assertRaises(ValueError):
                    kernel.nested_predictions(changed, 42, 1)
                factory.assert_not_called()

    def test_lower_nine_candidates_propagate_to_fit_report_and_preserve_shared_candidate(self):
        alphas, families = kernel.search_parameters("lower")
        lower_data = copy.copy(self.data)
        lower_data.features = {family: self.X for family in families}
        with mock.patch.object(kernel, "make_estimator", wraps=kernel.make_estimator) as factory, contextlib.redirect_stdout(io.StringIO()):
            predictions, selections = kernel.nested_predictions(lower_data, 42, 1, search_profile="lower")
        self.assertEqual(factory.call_count, 150)
        self.assertEqual({call.args for call in factory.call_args_list}, {(alpha, family) for alpha in alphas for family in families})
        for old, record in zip(self.selections, selections):
            self.assertEqual(len(record["candidates"]), 9)
            self.assertEqual({(c["alpha"], c["gamma_multiplier"]) for c in record["candidates"]},
                {(alpha, multiplier) for alpha in (0.01, 0.03, 0.1) for multiplier in (0.01, 0.03, 0.1)})
            shared = lambda rec: next(c for c in rec["candidates"] if c["alpha"] == 0.1 and c["gamma_multiplier"] == 0.1)
            self.assertEqual(shared(record), shared(old))
            key = lambda c: (c["inner_mse"], families.index(c["representation"]), -c["alpha"])
            self.assertEqual(record["joint"], min(record["candidates"], key=key))
        references = {kernel.CURRENT_REFERENCE: self.y + 1, kernel.DINO_REFERENCE: self.y + 2,
                      kernel.HAND_DINO_REFERENCE: self.y + 3}
        predictions = kernel.add_fixed_candidate(predictions, references, search_profile="lower")
        diagnostics, tables = kernel.diagnose(lower_data, predictions)
        with tempfile.TemporaryDirectory(prefix="kernel-lower-report-test-") as folder:
            out = Path(folder)
            with self.assertRaisesRegex(ValueError, "profile"):
                kernel.write_outputs(out, lower_data, predictions, selections, diagnostics, tables, 42, 1, 0.0)
            self.assertEqual(list(out.iterdir()), [])
            kernel.write_outputs(out, lower_data, predictions, selections, diagnostics, tables, 42, 1, 0.0, search_profile="lower")
            summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["search_profile"], "lower")
            self.assertEqual(summary["alpha_candidates"], [0.01, 0.03, 0.1])
            self.assertEqual(tuple(summary["bandwidth_families"]), families)
            self.assertEqual(summary["primary_metric"], "joint_nested overall OOF RMSE")
            with (out / "inner_search.csv").open(encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 45)
            self.assertEqual(sum(r["selected_joint"] == "True" for r in rows), 5)
            self.assertEqual(sum(r["selected_for_representation"] == "True" for r in rows), 15)
            self.assertEqual({float(r["alpha"]) for r in rows}, set(alphas))
            self.assertEqual({float(r["gamma_multiplier"]) for r in rows}, {0.01, 0.03, 0.1})
            for row in rows:
                self.assertEqual(float(row["gamma"]), float(row["gamma_multiplier"]) / 1152)
            report = (out / "report.md").read_text(encoding="utf-8")
            self.assertIn("**lower**", report)
            self.assertIn("alpha: 0.01, 0.03, 0.1", report)
            self.assertIn("gamma: 0.01/1152, 0.03/1152, 0.1/1152", report)
            self.assertNotIn("rbf_gamma10", report)
        # Explicit original remains numerically identical even after a lower run.
        with contextlib.redirect_stdout(io.StringIO()):
            original_predictions, original_selections = kernel.nested_predictions(self.data, 42, 1, search_profile="original")
        self.assertEqual(original_selections, self.selections)
        for name in self.predictions:
            np.testing.assert_array_equal(original_predictions[name], self.predictions[name])


if __name__ == "__main__":
    unittest.main()
