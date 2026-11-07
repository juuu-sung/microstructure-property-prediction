"""Synthetic-only fusion checks. Never read competition data or run a neural net."""
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from scripts import evaluate_feature_fusion as fusion
from scripts import evaluate_frozen_dinov2 as dino
from scripts import evaluate_frozen_resnet as core


class FusionArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="feature-fusion-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.ids = np.asarray([f"TRAIN_{i+1:06d}" for i in range(50)])
        rng = np.random.default_rng(31)
        self.hand = rng.normal(size=(50, 125))
        self.cls = rng.normal(size=(50, 384)).astype(np.float32)
        self.dino_features = np.concatenate([self.cls, rng.normal(size=(50, 384)).astype(np.float32),
            np.abs(rng.normal(size=(50, 384))).astype(np.float32)], axis=1)
        self.y = 200 + 7 * self.hand[:, 0] + 2 * self.cls[:, 0] + rng.normal(size=50)
        self.folds = np.arange(50) % 5
        self.baseline = self.y + 2 * np.sin(np.arange(50))
        self.frozen = self.y + 3 * np.cos(np.arange(50))
        self.previous_dino = self.y + 4 * np.sin(np.arange(50) + 0.5)
        self.save_rows("train.csv", [{"ID": ident, "hardness": float(y)} for ident, y in zip(self.ids, self.y)])
        self.save_rows("folds.csv", [{"ID": ident, "fold": int(fold), "seed": 42} for ident, fold in zip(self.ids, self.folds)])
        self.save_rows("baseline.csv", [{"ID": ident, "hardness": float(y), "fold": int(fold), "ridge_alpha100": float(pred)}
            for ident, y, fold, pred in zip(self.ids, self.y, self.folds, self.baseline)])
        self.save_rows("audit.csv", [{"ID": ident, "hardness": float(y), "haar_hh_noise_proxy": float(i)}
            for i, (ident, y) in enumerate(zip(self.ids, self.y))])
        self.save_rows("frozen.csv", [{"ID": ident, "hardness": float(y), "fold": int(fold), core.BASELINE_NAME: float(base), core.PRIMARY_NAME: float(pred)}
            for ident, y, fold, base, pred in zip(self.ids, self.y, self.folds, self.baseline, self.frozen)])
        manifest = [{"id": str(ident), "relative_path": f"train/{ident}.png", "size_bytes": 100, "sha256": "a" * 64} for ident in self.ids[::-1]]
        self.metadata = {"schema_version": 1, "rows": 50, "test_accessed": False, "labels_saved_in_npz": False,
            "train_csv_sha256": core.sha256(self.root / "train.csv"), "feature_state_sha256": "b" * 64,
            "training_image_manifest": manifest,
            "image_manifest_sha256": hashlib.sha256(dino.common.canonical_json(manifest).encode()).hexdigest(),
            "extraction_config": {**copy.deepcopy(dino.EXPECTED_CONFIG), "extractor_script_sha256": "c" * 64, "common_utility_sha256": "d" * 64},
            "feature_array_sha256": {"cls_token": dino.common.sha256_array(self.cls[::-1]),
                                      "cls_patch_mean_std": dino.common.sha256_array(self.dino_features[::-1])}}
        np.savez(self.root / "features.npz", ids=self.ids[::-1], cls_token=self.cls[::-1], cls_patch_mean_std=self.dino_features[::-1],
                 metadata_json=np.asarray(json.dumps(self.metadata)))
        (self.root / "metadata.json").write_text(json.dumps(self.metadata), encoding="utf-8")
        self.names = fusion.expected_handcrafted_names()
        self.save_hand()
        (self.root / "definition.json").write_text(json.dumps({"n_train": 50, "n_features": 125, "feature_definition": self.names}), encoding="utf-8")
        self.save_reference()

    def save_rows(self, name, rows):
        (self.root / name).write_text(core.csv_text(rows), encoding="utf-8")

    def save_hand(self, values=None, names=None):
        np.savez(self.root / "hand.npz", ids=self.ids[::-1], X=self.hand[::-1] if values is None else values,
                 feature_names=np.asarray(self.names if names is None else names))

    def save_reference(self):
        prior = 0.5 * self.baseline + 0.5 * self.frozen
        current, hand_dino = 0.5 * prior + 0.5 * self.previous_dino, 0.5 * self.baseline + 0.5 * self.previous_dino
        rows = []
        for i, ident in enumerate(self.ids):
            rows.append({"ID": ident, "hardness": float(self.y[i]), "fold": int(self.folds[i]), core.BASELINE_NAME: float(self.baseline[i]),
                core.PRIMARY_NAME: float(self.previous_dino[i]), dino.FROZEN_REFERENCE: float(self.frozen[i]),
                dino.PREVIOUS_REFERENCE: float(prior[i]), "baseline_dino_equal": float(hand_dino[i]), "previous_blend_dino_equal": float(current[i])})
        self.reference_rows = rows[::-1]
        self.save_rows("dino.csv", self.reference_rows)
        sources = {"features": "features.npz", "train_csv": "train.csv", "folds": "folds.csv", "baseline": "baseline.csv", "previous_frozen_oof": "frozen.csv"}
        self.summary = {"seed": 42, "n_train": 50, "outer_splits": 5,
            "input_files": {key: {"sha256": core.sha256(self.root / filename)} for key, filename in sources.items()},
            "feature_dimensions": dino.FEATURE_DIMENSIONS, "feature_metadata": self.metadata,
            "diagnostics": {name: {"overall": core.metrics(self.y, pred)} for name, pred in {
                core.PRIMARY_NAME: self.previous_dino, "previous_blend_dino_equal": current, "baseline_dino_equal": hand_dino}.items()}}
        self.save_summary()

    def save_summary(self):
        (self.root / "dino_summary.json").write_text(json.dumps(self.summary), encoding="utf-8")

    def load(self):
        return fusion.load_inputs(self.root / "features.npz", self.root / "train.csv", self.root / "folds.csv",
            self.root / "baseline.csv", self.root / "audit.csv", self.root / "frozen.csv", self.root / "dino.csv",
            self.root / "hand.npz", self.root / "definition.json", 42,
            previous_dino_summary=self.root / "dino_summary.json", expected_n=50)

    def argv(self):
        return ["--features", str(self.root / "features.npz"), "--train-csv", str(self.root / "train.csv"),
            "--folds", str(self.root / "folds.csv"), "--baseline", str(self.root / "baseline.csv"),
            "--audit-features", str(self.root / "audit.csv"), "--previous-frozen-oof", str(self.root / "frozen.csv"),
            "--previous-dino-oof", str(self.root / "dino.csv"), "--previous-dino-summary", str(self.root / "dino_summary.json"),
            "--handcrafted-features", str(self.root / "hand.npz"), "--feature-definition", str(self.root / "definition.json"),
            "--output-dir", str(self.root / "out")]

    def test_artifacts_align_and_formulas_are_exact(self):
        before = {p: core.sha256(p) for p in self.root.iterdir() if p.is_file()}
        data, refs = self.load()
        np.testing.assert_array_equal(data.features["concat_standardized"], np.concatenate([self.hand, self.dino_features], axis=1))
        np.testing.assert_array_equal(refs[fusion.CURRENT_REFERENCE], 0.5 * (0.5 * self.baseline + 0.5 * self.frozen) + 0.5 * self.previous_dino)
        np.testing.assert_array_equal(refs[fusion.HAND_DINO_REFERENCE], 0.5 * self.baseline + 0.5 * self.previous_dino)
        self.assertFalse(data.features["concat_block_balanced"].flags.writeable)
        self.assertFalse(data.provenance["handcrafted_features"]["historical_provenance_verified"])
        self.assertEqual(before, {p: core.sha256(p) for p in before})

    def test_handcrafted_schema_dtype_nonfinite_and_names_rejected(self):
        for values, names in [(self.hand[::-1].astype(np.float32), self.names), (self.hand[::-1, :-1], self.names),
                              (self.hand[::-1], self.names[::-1])]:
            self.save_hand(values, names)
            with self.assertRaises(ValueError):
                self.load()
        bad = self.hand[::-1].copy()
        bad[0, 0] = np.inf
        self.save_hand(bad)
        with self.assertRaisesRegex(ValueError, "finite"):
            self.load()

    def test_reference_alignment_formula_and_summary_hash_rejected(self):
        for column in ("fold", "hardness", "previous_blend_dino_equal", core.BASELINE_NAME, dino.FROZEN_REFERENCE):
            with self.subTest(column=column):
                changed = copy.deepcopy(self.reference_rows)
                changed[0][column] += 1
                self.save_rows("dino.csv", changed)
                with self.assertRaises(ValueError):
                    self.load()
        self.save_reference()
        original = copy.deepcopy(self.summary)
        for source in ("features", "train_csv", "folds", "baseline", "previous_frozen_oof"):
            with self.subTest(source=source):
                self.summary = copy.deepcopy(original)
                self.summary["input_files"][source]["sha256"] = "0" * 64
                self.save_summary()
                with self.assertRaisesRegex(ValueError, "hash mismatch"):
                    self.load()

    def test_reference_changes_do_not_change_fusion_inputs_or_predictions(self):
        first, refs_before = self.load()
        self.previous_dino += np.linspace(-0.5, 0.5, 50)
        self.save_reference()
        second, refs_after = self.load()
        self.assertFalse(np.array_equal(refs_before[fusion.CURRENT_REFERENCE], refs_after[fusion.CURRENT_REFERENCE]))
        for rep in fusion.REPRESENTATIONS:
            np.testing.assert_array_equal(first.features[rep], second.features[rep])
        with contextlib.redirect_stdout(io.StringIO()):
            p1, s1 = core.nested_predictions(first, 42, 1, alphas=(10,), representations=fusion.REPRESENTATIONS, estimator_factory=fusion.make_estimator)
            p2, s2 = core.nested_predictions(second, 42, 1, alphas=(10,), representations=fusion.REPRESENTATIONS, estimator_factory=fusion.make_estimator)
        np.testing.assert_array_equal(p1[core.PRIMARY_NAME], p2[core.PRIMARY_NAME])
        self.assertEqual(s1, s2)

    def test_validate_only_fits_nothing_writes_nothing(self):
        loader = fusion.load_inputs
        def small_loader(*args, **kwargs):
            return loader(*args, **kwargs, expected_n=50)
        with mock.patch.object(fusion, "load_inputs", side_effect=small_loader), mock.patch.object(core, "nested_predictions") as nested, \
             mock.patch.object(fusion, "make_estimator") as factory, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(fusion.main(self.argv() + ["--validate-only"]), 0)
            nested.assert_not_called()
            factory.assert_not_called()
        self.assertFalse((self.root / "out").exists())

    def test_overwrite_protection_precedes_input_reads(self):
        (self.root / "out").mkdir()
        (self.root / "out/old.txt").write_text("preserve", encoding="utf-8")
        with mock.patch.object(fusion, "load_inputs") as loader:
            with self.assertRaisesRegex(ValueError, "--overwrite"):
                fusion.main(self.argv())
            loader.assert_not_called()
        self.assertEqual((self.root / "out/old.txt").read_text(encoding="utf-8"), "preserve")


class FusionEstimatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(5)
        cls.X = rng.normal(size=(50, 1277))
        cls.y = 200 + 7 * cls.X[:, 0] + 2 * cls.X[:, 125] + rng.normal(size=50)
        cls.data = core.Inputs(np.array([f"TRAIN_{i+1:06d}" for i in range(50)]), cls.y,
            {rep: cls.X for rep in fusion.REPRESENTATIONS}, np.arange(50) % 5, cls.y + 2, np.linspace(1, 10, 50), {}, {})
        cls.factory_calls = []
        def factory(alpha, representation):
            cls.factory_calls.append((alpha, representation))
            return fusion.make_estimator(alpha, representation)
        with contextlib.redirect_stdout(io.StringIO()):
            cls.predictions, cls.selections = core.nested_predictions(cls.data, 42, 1, alphas=(10, 100),
                representations=fusion.REPRESENTATIONS, estimator_factory=factory)

    def test_weights_survive_and_scalers_fit_training_subset_only(self):
        train = self.X[:30]
        for rep in fusion.REPRESENTATIONS:
            estimator = fusion.make_estimator(100, rep).fit(train, self.y[:30])
            blocks = estimator.regressor_.named_steps["blocks"]
            transformed = blocks.transform(train)
            weights = fusion.BLOCK_WEIGHTS if rep == "concat_block_balanced" else {"hand": 1.0, "dino": 1.0}
            np.testing.assert_allclose(transformed[:, :125].var(axis=0), weights["hand"] ** 2, rtol=1e-10, atol=1e-10)
            np.testing.assert_allclose(transformed[:, 125:].var(axis=0), weights["dino"] ** 2, rtol=1e-10, atol=1e-10)
            np.testing.assert_allclose(blocks.named_transformers_["hand"].mean_, train[:, :125].mean(axis=0))
            np.testing.assert_allclose(blocks.named_transformers_["dino"].mean_, train[:, 125:].mean(axis=0))
            np.testing.assert_allclose(estimator.transformer_.mean_, [self.y[:30].mean()])
            self.assertEqual(list(estimator.regressor_.named_steps), ["blocks", "ridge"])

    def test_factory_receives_representation_for_all_inner_and_outer_fits(self):
        self.assertEqual(len(self.factory_calls), 70)  # 5 * (2 reps * 2 alphas * 3 inner fits + 2 outer refits).
        for rep in fusion.REPRESENTATIONS:
            self.assertEqual(sum(name == rep for _, name in self.factory_calls), 35)
        self.assertTrue(all(np.isfinite(pred).all() for pred in self.predictions.values()))

    def test_outer_validation_labels_cannot_change_own_predictions(self):
        changed_y = self.y.copy()
        changed_y[self.data.folds == 0] += 1000
        changed = core.Inputs(self.data.ids, changed_y, self.data.features, self.data.folds, self.data.baseline, self.data.noise, {}, {})
        with contextlib.redirect_stdout(io.StringIO()):
            predictions, selections = core.nested_predictions(changed, 42, 1, alphas=(10, 100),
                representations=fusion.REPRESENTATIONS, estimator_factory=fusion.make_estimator)
        for name in self.predictions:
            np.testing.assert_array_equal(predictions[name][self.data.folds == 0], self.predictions[name][self.data.folds == 0])
        self.assertEqual(selections[0], self.selections[0])

    def test_fixed_candidate_reports_current_reference_and_primary_separately(self):
        references = {fusion.CURRENT_REFERENCE: self.y + 1, fusion.DINO_REFERENCE: self.y + 3, fusion.HAND_DINO_REFERENCE: self.y + 2.5}
        predictions = fusion.add_fixed_candidate(self.data, self.predictions, references)
        np.testing.assert_array_equal(predictions[fusion.FIXED_BLEND], 0.5 * references[fusion.CURRENT_REFERENCE] + 0.5 * self.predictions[core.PRIMARY_NAME])
        diagnostics, tables = fusion.diagnose(self.data, predictions)
        with tempfile.TemporaryDirectory(prefix="fusion-report-test-") as folder:
            out = Path(folder)
            fusion.write_outputs(out, self.data, predictions, self.selections, diagnostics, tables, 42, 1, 0.0, alphas=(10, 100))
            summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["primary_metric"], "joint_nested overall OOF RMSE")
            self.assertIn(fusion.HAND_DINO_REFERENCE, summary["diagnostics"])
            self.assertIn("target_regions", summary["diagnostics"][core.PRIMARY_NAME])
            self.assertIn("current_blend_fusion_equal", (out / "report.md").read_text(encoding="utf-8"))
            self.assertTrue((out / "target_region_metrics.csv").is_file())

    def test_default_core_factory_behavior_is_preserved(self):
        default_data = core.Inputs(self.data.ids, self.y, {"final_gap": self.X[:, :3], "multiscale_mean_std": self.X[:, :7]},
                                  self.data.folds, self.data.baseline, self.data.noise, {}, {})
        with contextlib.redirect_stdout(io.StringIO()):
            first, choices1 = core.nested_predictions(default_data, 42, 1, alphas=(10,))
            second, choices2 = core.nested_predictions(default_data, 42, 1, alphas=(10,), estimator_factory=lambda alpha, representation: core.make_estimator(alpha))
        for key in first:
            np.testing.assert_array_equal(first[key], second[key])
        self.assertEqual(choices1, choices2)

    def test_seed_defaults_and_manual_path_overrides(self):
        for seed in (42, 2026):
            args = fusion.parse_args(["--seed", str(seed)])
            self.assertEqual(args.baseline, fusion.ROOT / f"reports/benchmark_oof_seed{seed}.csv")
            self.assertEqual(args.previous_dino_oof, fusion.ROOT / f"reports/frozen_dinov2_small/seed{seed}/oof.csv")
            self.assertEqual(args.previous_dino_summary, args.previous_dino_oof.with_name("summary.json"))
            self.assertEqual(args.output_dir, fusion.ROOT / f"reports/feature_fusion/seed{seed}")
            self.assertEqual(args.folds.name, "folds.csv" if seed == 42 else "benchmark_folds_seed2026.csv")
        args = fusion.parse_args(["--seed", "2026", "--folds", "custom.csv", "--output-dir", "custom_out"])
        self.assertEqual(args.folds, Path("custom.csv"))
        self.assertEqual(args.output_dir, Path("custom_out"))


if __name__ == "__main__":
    unittest.main()
