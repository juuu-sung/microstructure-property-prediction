"""Synthetic-only DINO evaluation checks; no images, weights, GPU, or real fitting."""
import contextlib
import copy
import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

from scripts import evaluate_frozen_dinov2 as dino
from scripts import evaluate_frozen_resnet as core


class CacheAndReferenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="dino-eval-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.ids = np.asarray([f"TRAIN_{index + 1:06d}" for index in range(50)])
        self.y = np.linspace(150, 250, 50)
        self.folds = np.arange(50) % 5
        self.baseline = self.y + np.sin(np.arange(50)) * 2
        self.frozen = self.y + np.cos(np.arange(50)) * 3
        self.write_rows("train.csv", [{"ID": ident, "hardness": float(y)} for ident, y in zip(self.ids, self.y)])
        self.write_rows("folds.csv", [{"ID": ident, "fold": int(fold), "seed": 42} for ident, fold in zip(self.ids, self.folds)])
        self.write_rows("baseline.csv", [{"ID": ident, "hardness": float(y), "fold": int(fold), "ridge_alpha100": float(pred)}
            for ident, y, fold, pred in zip(self.ids, self.y, self.folds, self.baseline)])
        self.write_rows("audit.csv", [{"ID": ident, "hardness": float(y), "haar_hh_noise_proxy": float(index)}
            for index, (ident, y) in enumerate(zip(self.ids, self.y))])
        self.reference_rows = [{"ID": ident, "hardness": float(y), "fold": int(fold),
            core.BASELINE_NAME: float(base), core.PRIMARY_NAME: float(frozen)}
            for ident, y, fold, base, frozen in zip(self.ids, self.y, self.folds, self.baseline, self.frozen)][::-1]
        self.write_rows("previous_oof.csv", self.reference_rows)
        rng = np.random.default_rng(21)
        self.cls = rng.normal(size=(50, 384)).astype(np.float32)
        self.combined = np.concatenate([self.cls, rng.normal(size=(50, 384)).astype(np.float32),
                                       np.abs(rng.normal(size=(50, 384))).astype(np.float32)], axis=1)
        self.feature_ids = self.ids[::-1]
        manifest = [{"id": str(ident), "relative_path": f"train/{ident}.png", "size_bytes": 100,
                     "sha256": "a" * 64} for ident in self.feature_ids]
        self.metadata = {
            "schema_version": 1, "rows": 50, "test_accessed": False, "labels_saved_in_npz": False,
            "train_csv_sha256": core.sha256(self.root / "train.csv"), "feature_state_sha256": "b" * 64,
            "training_image_manifest": manifest,
            "image_manifest_sha256": hashlib.sha256(dino.common.canonical_json(manifest).encode()).hexdigest(),
            "extraction_config": {**copy.deepcopy(dino.EXPECTED_CONFIG),
                "extractor_script_sha256": "c" * 64, "common_utility_sha256": "d" * 64},
        }
        self.write_cache()

    def write_rows(self, filename, rows):
        (self.root / filename).write_text(core.csv_text(rows), encoding="utf-8")

    def write_cache(self, update_hashes=True):
        if update_hashes:
            self.metadata["feature_array_sha256"] = {
                "cls_token": dino.common.sha256_array(self.cls),
                "cls_patch_mean_std": dino.common.sha256_array(self.combined),
            }
        np.savez(self.root / "features.npz", ids=self.feature_ids, cls_token=self.cls,
                 cls_patch_mean_std=self.combined, metadata_json=np.asarray(json.dumps(self.metadata)))
        (self.root / "metadata.json").write_text(json.dumps(self.metadata), encoding="utf-8")

    def load(self):
        return dino.load_inputs(self.root / "features.npz", self.root / "train.csv", self.root / "folds.csv",
            self.root / "baseline.csv", self.root / "audit.csv", self.root / "previous_oof.csv", 42, expected_n=50)

    def test_valid_cache_and_shuffled_reference_align_by_id(self):
        data, refs = self.load()
        np.testing.assert_array_equal(data.ids, self.ids)
        np.testing.assert_array_equal(data.features["cls_token"], self.cls[::-1])
        np.testing.assert_array_equal(refs[dino.FROZEN_REFERENCE], self.frozen)
        np.testing.assert_array_equal(refs[dino.PREVIOUS_REFERENCE], 0.5 * self.baseline + 0.5 * self.frozen)
        self.assertFalse(data.provenance["cache_validation"]["current_image_files_read"])
        self.assertIn("sha256", data.provenance["previous_frozen_oof"])

    def test_metadata_sidecar_and_array_hash_mismatches_rejected(self):
        (self.root / "metadata.json").write_text(json.dumps({**self.metadata, "rows": 49}), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "metadata.json differ"):
            self.load()
        self.cls[0, 0] += 1
        self.write_cache(update_hashes=False)
        with self.assertRaisesRegex(ValueError, "integrity hash"):
            self.load()

    def test_wrong_model_revision_or_preprocessing_rejected(self):
        original = copy.deepcopy(self.metadata)
        cases = [("model_id", "other/model"), ("revision", "0" * 40), ("input_shape", [3, 256, 256]),
                 ("padding", {"mode": "constant", "left": 5, "right": 5, "top": 5, "bottom": 5}),
                 ("pooling", "L2 normalized CLS"), ("dynamic_img_size", False),
                 ("num_register_tokens", 4), ("train_mode", True), ("amp", True)]
        for key, value in cases:
            with self.subTest(key=key):
                self.metadata = copy.deepcopy(original)
                self.metadata["extraction_config"][key] = value
                self.write_cache()
                with self.assertRaisesRegex(ValueError, key):
                    self.load()

    def test_missing_train_hash_or_tampered_manifest_rejected(self):
        original = copy.deepcopy(self.metadata)
        del self.metadata["train_csv_sha256"]
        self.write_cache()
        with self.assertRaisesRegex(ValueError, "train_csv_sha256"):
            self.load()
        self.metadata = original
        self.metadata["training_image_manifest"][0]["sha256"] = "f" * 64
        self.write_cache()
        with self.assertRaisesRegex(ValueError, "manifest failed"):
            self.load()

    def test_combined_cls_consistency_and_nonnegative_std(self):
        self.combined[0, 0] += 1
        self.write_cache()
        with self.assertRaisesRegex(ValueError, "exact CLS"):
            self.load()
        self.combined[:, :384] = self.cls
        self.combined[0, 768] = -1
        self.write_cache()
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            self.load()

    def test_reference_fold_target_baseline_or_ids_mismatch_rejected(self):
        for key, value in [("fold", 0), ("hardness", 999), (core.BASELINE_NAME, 999), ("ID", "TEST_000001")]:
            with self.subTest(key=key):
                rows = copy.deepcopy(self.reference_rows)
                rows[0][key] = value
                self.write_rows("previous_oof.csv", rows)
                with self.assertRaises(ValueError):
                    self.load()

    def test_validation_only_fits_nothing_and_writes_nothing(self):
        data, refs = self.load()
        output = self.root / "unused-output"
        with mock.patch.object(dino, "load_inputs", return_value=(data, refs)), \
             mock.patch.object(core, "nested_predictions", side_effect=AssertionError("Validation must not fit")) as fit, \
             contextlib.redirect_stdout(io.StringIO()) as capture:
            self.assertEqual(dino.main(["--validate-only", "--output-dir", str(output)]), 0)
        fit.assert_not_called()
        self.assertFalse(output.exists())
        self.assertEqual(json.loads(capture.getvalue())["estimators_fitted"], 0)

    def test_existing_output_rejected_before_cache_load(self):
        output = self.root / "existing"
        output.mkdir()
        (output / "keep.txt").write_text("original", encoding="utf-8")
        with mock.patch.object(dino, "load_inputs") as load:
            with self.assertRaisesRegex(ValueError, "--overwrite"):
                dino.main(["--output-dir", str(output)])
            load.assert_not_called()
        self.assertEqual((output / "keep.txt").read_text(encoding="utf-8"), "original")


class NestedDinoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(71)
        features = rng.normal(size=(50, 7))
        features[:, 0] = np.arange(50)
        target = 190 + 3 * features[:, 1] + rng.normal(size=50)
        cls.data = core.Inputs(np.asarray([f"TRAIN_{i + 1:06d}" for i in range(50)]), target,
            {"cls_token": features[:, :3], "cls_patch_mean_std": features}, np.arange(50) % 5,
            target + rng.normal(size=50), np.linspace(1, 10, 50), {}, {})
        cls.refs = {dino.FROZEN_REFERENCE: target + rng.normal(size=50)}
        cls.refs[dino.PREVIOUS_REFERENCE] = 0.5 * cls.data.baseline + 0.5 * cls.refs[dino.FROZEN_REFERENCE]
        cls.fit_sizes = []
        factory = core.make_estimator

        class RecordingEstimator:
            def __init__(self, alpha):
                self.estimator = factory(alpha)
            def fit(self, x, y):
                self.seen = set(x[:, 0])
                cls.fit_sizes.append(len(y))
                self.estimator.fit(x, y)
            def predict(self, x):
                if self.seen & set(x[:, 0]):
                    raise AssertionError("Validation sample used in estimator fit")
                return self.estimator.predict(x)

        with mock.patch.object(core, "make_estimator", RecordingEstimator), contextlib.redirect_stdout(io.StringIO()):
            cls.nested, cls.selections = core.nested_predictions(cls.data, 42, 1, dino.ALPHAS, representations=dino.REPRESENTATIONS)
        cls.predictions = dino.add_fixed_candidates(cls.data, cls.nested, cls.refs)

    def test_fixed_grid_joint_selection_and_no_outer_sample_fit(self):
        self.assertEqual(len(self.fit_sizes), 190)
        self.assertEqual(self.fit_sizes.count(40), 10)
        self.assertLess(max(self.fit_sizes), 50)
        for record in self.selections:
            self.assertEqual(len(record["candidates"]), 12)
            self.assertEqual(record["joint"]["inner_mse"], min(c["inner_mse"] for c in record["candidates"]))
            mask = self.data.folds == record["outer_fold"]
            np.testing.assert_array_equal(self.nested[core.PRIMARY_NAME][mask],
                self.nested[record["joint"]["representation"] + "_nested"][mask])

    def test_outer_validation_target_does_not_change_its_predictions(self):
        changed = copy.deepcopy(self.data)
        changed.y[changed.folds == 0] += 1000
        with contextlib.redirect_stdout(io.StringIO()):
            predictions, selections = core.nested_predictions(changed, 42, 1, dino.ALPHAS, representations=dino.REPRESENTATIONS)
        for name in predictions:
            np.testing.assert_allclose(predictions[name][changed.folds == 0], self.nested[name][changed.folds == 0])
        self.assertEqual(selections[0], self.selections[0])

    def test_predeclared_blends_preserve_primary_predictions(self):
        np.testing.assert_array_equal(self.predictions[core.PRIMARY_NAME], self.nested[core.PRIMARY_NAME])
        np.testing.assert_array_equal(self.predictions["baseline_dino_equal"],
                                      0.5 * self.data.baseline + 0.5 * self.nested[core.PRIMARY_NAME])
        np.testing.assert_allclose(self.predictions["previous_blend_dino_equal"],
                                  0.25 * self.data.baseline + 0.25 * self.refs[dino.FROZEN_REFERENCE] + 0.5 * self.nested[core.PRIMARY_NAME])

    def test_reports_have_reference_deltas_fixed_weights_and_primary_metric(self):
        diagnostics, tables = dino.diagnose(self.data, self.predictions)
        previous_rmse = core.metrics(self.data.y, self.refs[dino.PREVIOUS_REFERENCE])["rmse"]
        for row in tables["metrics"]:
            self.assertAlmostEqual(row["delta_rmse_vs_previous_blend"], row["rmse"] - previous_rmse)
        self.assertEqual(len(tables["residual_correlations"]), 64)
        self.assertTrue(tables["top20_error_changes_vs_previous_blend"])
        with tempfile.TemporaryDirectory(prefix="dino-report-test-") as folder:
            root = Path(folder)
            dino.write_outputs(root, self.data, self.predictions, self.selections, diagnostics, tables, 42, 1, 0.0)
            summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["primary_metric"], "joint_nested overall OOF RMSE")
            self.assertEqual(summary["alpha_candidates"], list(dino.ALPHAS))
            self.assertEqual(summary["fixed_blend_weights"], dino.FIXED_BLEND_WEIGHTS)
            report = (root / "report.md").read_text(encoding="utf-8")
            self.assertIn("DINOv2-small", report)
            self.assertIn("반복 연구", report)
            self.assertIn("25%", report)
            with (root / "oof.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 50)
            np.testing.assert_array_equal([float(row[core.PRIMARY_NAME]) for row in rows], self.nested[core.PRIMARY_NAME])

    def test_resnet_defaults_and_global_configuration_are_unchanged(self):
        self.assertEqual(core.REPRESENTATIONS, ("final_gap", "multiscale_mean_std"))
        self.assertEqual(core.FEATURE_DIMENSIONS, {"final_gap": 512, "multiscale_mean_std": 1792})
        self.assertEqual(core.parse_args([]).alphas, (10.0, 100.0, 1000.0))
        self.assertEqual(dino.parse_args([]).alphas, dino.ALPHAS)

    def test_extractor_metadata_contract_matches_evaluator_without_runtime(self):
        from scripts import extract_frozen_dinov2 as extractor
        args = extractor.build_parser().parse_args([])
        config = extractor.extraction_config(args, {}, "cpu", {})
        for key, value in dino.EXPECTED_CONFIG.items():
            self.assertEqual(config[key], value, key)
        self.assertEqual(extractor.FEATURE_DIMENSIONS, dino.FEATURE_DIMENSIONS)

    def test_import_has_no_torch_dependency_or_weight_loading(self):
        result = subprocess.run([sys.executable, "-c",
            "import sys; from scripts import evaluate_frozen_dinov2; assert 'torch' not in sys.modules; print('ok')"],
            cwd=core.ROOT, capture_output=True, text=True, timeout=20, check=True)
        self.assertEqual(result.stdout.strip(), "ok")


if __name__ == "__main__":
    unittest.main()
