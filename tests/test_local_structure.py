"""Synthetic CPU tests only; no project train feature extraction or fitting."""
import contextlib
import copy
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
from PIL import Image

from scripts import extract_local_structure as extract
from scripts import evaluate_local_structure as evaluate

core = evaluate.core


class FeatureTests(unittest.TestCase):
    def test_constant_image_finite_zero_and_invalid_inputs(self):
        self.assertEqual(len(extract.feature_names()), 60)
        self.assertEqual(len(set(extract.feature_names())), 60)
        np.testing.assert_array_equal(extract.local_features(np.full((256, 256), 0.5)), np.zeros(60))
        for a in (np.zeros((255, 256)), np.zeros((256, 256), dtype=np.float32), np.full((256, 256), np.nan), np.full((256, 256), 2.0)):
            with self.assertRaises(ValueError):
                extract.local_features(a)

    def test_axis_orientation_preserved_and_scalar_rotation_behavior(self):
        yy, xx = np.mgrid[:256, :256]
        a = 0.5 + 0.25 * np.sin(2 * np.pi * (xx + 2 * yy) / 32)
        raw = dict(zip(extract.feature_names(), extract.local_features(a)))
        rotated = dict(zip(extract.feature_names(), extract.local_features(np.rot90(a).copy())))
        reflected = dict(zip(extract.feature_names(), extract.local_features(a[:, ::-1].copy())))
        for i in range(3):
            self.assertGreater(abs(raw[f"tensor{i}_sin2"]), 0.2)
            for family in ("energy", "coherence"):
                for stat in ("mean", "std", "q10", "q50", "q90"):
                    self.assertAlmostEqual(raw[f"tensor{i}_{family}_{stat}"], rotated[f"tensor{i}_{family}_{stat}"], places=9)
            self.assertAlmostEqual(rotated[f"tensor{i}_cos2"], -raw[f"tensor{i}_cos2"], places=9)
            self.assertAlmostEqual(rotated[f"tensor{i}_sin2"], -raw[f"tensor{i}_sin2"], places=9)
            self.assertAlmostEqual(reflected[f"tensor{i}_sin2"], -raw[f"tensor{i}_sin2"], places=9)
            for order in (2, 4):
                self.assertAlmostEqual(rotated[f"tensor{i}_alignment{order}"], raw[f"tensor{i}_alignment{order}"], places=9)

    def test_weighted_quantiles_and_energy_scale(self):
        self.assertEqual(extract.weighted_quantiles(np.array([1., 5.]), np.zeros(2)), [0., 0., 0.])
        np.testing.assert_allclose(extract.weighted_quantiles(np.array([0., 1., 2.]), np.array([1., 2., 1.])), [0., 1., 2.])
        x = np.random.default_rng(91).uniform(0.2, 0.8, (256, 256))
        raw, half = extract.local_features(x), extract.local_features(x * 0.5)
        for i in range(3):
            np.testing.assert_allclose(half[i*18:i*18+5], raw[i*18:i*18+5] * 0.25, rtol=1e-10, atol=1e-14)

    def test_legacy_function_schema_without_path_or_environment_setup(self):
        before = list(sys.path)
        fn = extract.legacy_function()
        self.assertEqual(before, sys.path)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "synthetic.png"
            Image.fromarray(np.random.default_rng(17).integers(0, 256, (256, 256), dtype=np.uint8)).save(path)
            features = fn(path)
        self.assertEqual(list(features), extract.hand.expected_handcrafted_names())
        self.assertTrue(np.isfinite(list(features.values())).all())

    def test_extractor_end_to_end_on_two_synthetic_images_and_original_mismatch(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            data_dir = root / "synthetic_data"
            (data_dir / "train").mkdir(parents=True)
            ids = ["TRAIN_000001", "TRAIN_000002"]
            (data_dir / "train.csv").write_text("ID,hardness\nTRAIN_000001,190\nTRAIN_000002,200\n")
            fn = extract.legacy_function()
            records = []
            for j, ident in enumerate(ids):
                path = data_dir / "train" / (ident+".png")
                Image.fromarray(np.random.default_rng(j).integers(0, 256, (256, 256), dtype=np.uint8)).save(path)
                records.append(list(fn(path).values()))
            original = np.array(records)
            oldpath, definition = root / "old.npz", root / "definition.json"
            np.savez_compressed(oldpath, ids=np.array(ids), X=original, feature_names=np.array(extract.hand.expected_handcrafted_names()))
            definition.write_text(json.dumps({"n_train": 2, "n_features": 125, "feature_definition": extract.hand.expected_handcrafted_names()}))
            inventory = extract.common.training_inventory(data_dir, expected_count=2)
            output = root / "new"
            arguments = ["--data-dir", str(data_dir), "--original-features", str(oldpath), "--definition", str(definition), "--output-dir", str(output)]
            with mock.patch.object(extract.common, "training_inventory", return_value=inventory), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(extract.main(arguments), 0)
            reference = extract.original_inputs(oldpath, definition, ids)[1]
            local, meta = extract.validate_cache(output/"features.npz", ids, original, reference, inventory["train_csv_sha256"], inventory["image_manifest_sha256"])
            self.assertEqual(local.shape, (2, 60))
            self.assertEqual(meta["original_reextraction_check"]["max_absolute_difference"], 0.0)
            original[0, 0] += 0.1
            np.savez_compressed(oldpath, ids=np.array(ids), X=original, feature_names=np.array(extract.hand.expected_handcrafted_names()))
            failed = root / "failed"
            with mock.patch.object(extract.common, "training_inventory", return_value=inventory):
                with self.assertRaisesRegex(ValueError, "Original 125-feature mismatch"):
                    extract.main(arguments[:-1]+[str(failed)])
            self.assertFalse(failed.exists())


class CacheTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(prefix="local-structure-synthetic-")
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.ids = np.array([f"TRAIN_{i+1:06d}" for i in range(50)])
        self.old = np.random.default_rng(1).normal(size=(50, 125))
        self.local = np.random.default_rng(2).normal(size=(50, 60))
        self.original_path, self.definition = self.root / "original.npz", self.root / "definition.json"
        np.savez_compressed(self.original_path, ids=self.ids, X=self.old, feature_names=np.array(extract.hand.expected_handcrafted_names()))
        self.definition.write_text(json.dumps({"n_train": 50, "n_features": 125, "feature_definition": extract.hand.expected_handcrafted_names()}))
        self.reference = extract.original_inputs(self.original_path, self.definition, self.ids)[1]
        manifest = [{"id": i, "relative_path": f"train/{i}.png", "size_bytes": 1, "sha256": "b"*64} for i in self.ids]
        self.image_hash = hashlib.sha256(extract.common.canonical_json(manifest).encode()).hexdigest()
        self.meta = {"schema_version": 1, "rows": 50, "protocol": copy.deepcopy(extract.PROTOCOL), "source_sha256": extract.source_hashes(),
            "original_reference": self.reference, "train_csv_sha256": "a"*64, "image_manifest_sha256": self.image_hash,
            "training_image_manifest": manifest, "feature_array_sha256": {extract.RAW: extract.common.sha256_array(self.old), extract.LOCAL: extract.common.sha256_array(self.local)},
            "original_reextraction_check": {"passed": True, "max_absolute_difference": 0., "atol": extract.MATCH_ATOL, "rtol": 0.},
            "test_accessed": False, "labels_saved_in_npz": False}
        self.path = self.root / "cache/features.npz"
        self.save()

    def save(self):
        extract.save_cache(self.path.parent, self.ids, self.old, self.local, self.meta)

    def validate(self):
        return extract.validate_cache(self.path, self.ids, self.old, self.reference, "a"*64, self.image_hash)

    def test_cache_roundtrip_and_id_reordering(self):
        values, meta = self.validate()
        np.testing.assert_array_equal(values, self.local)
        meta = copy.deepcopy(meta)
        meta["feature_array_sha256"] = {extract.RAW: extract.common.sha256_array(self.old[::-1]), extract.LOCAL: extract.common.sha256_array(self.local[::-1])}
        extract.save_cache(self.path.parent, self.ids[::-1], self.old[::-1], self.local[::-1], meta)
        np.testing.assert_array_equal(self.validate()[0], self.local)

    def test_tampered_source_protocol_scope_manifest_and_agreement_rejected(self):
        original = copy.deepcopy(self.meta)
        for mutate in (lambda m: m["source_sha256"].update(extractor="x"), lambda m: m["protocol"].update(epsilon=1),
                       lambda m: m.update(test_accessed=True), lambda m: m["original_reextraction_check"].update(passed=False),
                       lambda m: m["training_image_manifest"][0].update(sha256="x"), lambda m: m.update(train_csv_sha256="x")):
            self.meta = copy.deepcopy(original)
            mutate(self.meta)
            self.save()
            with self.assertRaises(ValueError):
                self.validate()

    def test_array_and_sidecar_tampering_rejected(self):
        original = self.local.copy()
        self.local[0, 0] += 1
        self.save()
        with self.assertRaisesRegex(ValueError, "hash"):
            self.validate()
        self.local = original
        self.save()
        self.path.with_name("metadata.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "sidecar"):
            self.validate()

    def test_existing_results_and_protected_input_output_rejected(self):
        with mock.patch.object(evaluate, "load_prior") as loader:
            with self.assertRaisesRegex(ValueError, "already exists"):
                evaluate.main(["--output-dir", str(self.path.parent)])
            with self.assertRaisesRegex(ValueError, "protected"):
                evaluate.main(["--features", str(self.path), "--output-dir", str(self.root)])
            loader.assert_not_called()

    def test_readonly_modes_fit_nothing_write_nothing(self):
        data = core.Inputs(self.ids, np.arange(50.), {evaluate.RAW: self.old}, np.arange(50)%5, np.arange(50.), np.arange(50.),
            {"resnet_original": {"train_csv_sha256": "a"*64, "image_manifest_sha256": self.image_hash}, "hand_original_reference": self.reference}, {})
        before = {str(p): core.sha256(p) for p in self.root.rglob("*") if p.is_file()}
        for mode in ("--preflight", "--validate-only"):
            with mock.patch.object(evaluate, "load_prior", return_value=(copy.deepcopy(data), {evaluate.REFERENCE: data.y+1})), \
                 mock.patch.object(core, "make_estimator") as fit, mock.patch.object(evaluate, "nested_predictions") as nested, \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(evaluate.main(["--features", str(self.path), "--output-dir", str(self.root/"out"), mode]), 0)
                fit.assert_not_called()
                nested.assert_not_called()
        self.assertEqual(before, {str(p): core.sha256(p) for p in self.root.rglob("*") if p.is_file()})


class EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(42)
        x, local = rng.normal(size=(50, 125)), rng.normal(size=(50, 60))
        y = 200 + x[:, 0] * 3 + local[:, 0] * 2 + rng.normal(size=50)
        folds = np.arange(50) % 5
        baseline = np.zeros(50)
        for fold in range(5):
            m = core.make_estimator(100)
            m.fit(x[folds != fold], y[folds != fold])
            baseline[folds == fold] = m.predict(x[folds == fold])
        cls.data = core.Inputs(np.array([f"TRAIN_{i+1:06d}" for i in range(50)]), y,
            {evaluate.RAW: x, evaluate.AUGMENTED: np.concatenate([x, local], axis=1)}, folds, baseline, np.arange(50.), {}, {})
        with contextlib.redirect_stdout(io.StringIO()):
            cls.pred, cls.selection = evaluate.nested_predictions(cls.data, 42, 1)

    def refs(self):
        return {evaluate.REFERENCE: 0.25*self.data.baseline+0.25*(self.data.y+1)+0.5*(self.data.y+2),
                evaluate.previous.RAW_REFERENCE: self.data.y+1, "reference_dino_d4": self.data.y+2}

    def test_anchor_and_fit_budget_no_representation_gate(self):
        anchor, check = evaluate.fixed_anchor(self.data, 1)
        np.testing.assert_array_equal(anchor, self.data.baseline)
        self.assertEqual(check["status"], "matched")
        self.assertNotIn(core.PRIMARY_NAME, self.pred)
        self.assertTrue(all("joint" not in r and len(r["candidates"]) == 6 for r in self.selection))
        factory = core.make_estimator
        with mock.patch.object(core, "make_estimator", wraps=factory) as calls, contextlib.redirect_stdout(io.StringIO()):
            evaluate.fixed_anchor(self.data, 1)
            evaluate.nested_predictions(self.data, 42, 1)
        self.assertEqual(calls.call_count, 105)
        bad = copy.deepcopy(self.data)
        bad.baseline[0] += 0.01
        with self.assertRaisesRegex(ValueError, "anchor mismatch"):
            evaluate.fixed_anchor(bad, 1)

    def test_outer_labels_and_reference_arrays_do_not_affect_fold_selection(self):
        data = copy.deepcopy(self.data)
        data.y[data.folds == 0] += 50
        data.baseline += 1000
        data.noise *= -1
        with contextlib.redirect_stdout(io.StringIO()):
            predictions, selections = evaluate.nested_predictions(data, 42, 1)
        self.assertEqual(selections[0], self.selection[0])
        for name in predictions:
            np.testing.assert_array_equal(predictions[name][data.folds == 0], self.pred[name][data.folds == 0])

    def test_only_hand_quarter_replaced_with_separate_tuning_control(self):
        refs = self.refs()
        result = evaluate.add_fixed_blends(self.data, self.pred, refs, self.data.baseline)
        for name, rep in ((evaluate.PRIMARY, evaluate.AUGMENTED), (evaluate.CONTROL, evaluate.RAW)):
            expected = 0.25*self.pred[rep+"_nested"]+0.25*refs[evaluate.previous.RAW_REFERENCE]+0.5*refs["reference_dino_d4"]
            np.testing.assert_allclose(result[name], expected, rtol=0, atol=1e-12)
        unchanged = {n+"_nested": self.data.baseline for n in evaluate.REPRESENTATIONS}
        np.testing.assert_array_equal(evaluate.add_fixed_blends(self.data, unchanged, refs, self.data.baseline)[evaluate.PRIMARY], refs[evaluate.REFERENCE])

    def test_diagnostics_reporting_both_references_and_output_roundtrip(self):
        anchor, check = evaluate.fixed_anchor(self.data, 1)
        predictions = evaluate.add_fixed_blends(self.data, self.pred, self.refs(), anchor)
        diagnostics, tables = evaluate.diagnose(self.data, predictions)
        for label, name in (("current", evaluate.REFERENCE), ("tuned_control", evaluate.CONTROL)):
            m = diagnostics[evaluate.PRIMARY]["overall"]
            self.assertAlmostEqual(m["delta_rmse_vs_"+label], m["rmse"]-diagnostics[name]["overall"]["rmse"])
            for row in tables["fold_metrics"]:
                mask = self.data.folds == row["fold"]
                self.assertAlmostEqual(row["delta_rmse_vs_"+label], row["rmse"]-core.metrics(self.data.y[mask], predictions[name][mask])["rmse"])
        self.assertEqual(len(tables["replacement_errors"]), 50)
        with tempfile.TemporaryDirectory() as td:
            evaluate.write_outputs(Path(td), self.data, predictions, self.selection, diagnostics, tables, 42, 0., check, 1)
            summary = json.loads((Path(td)/"summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["estimator_fit_count"], 105)
            self.assertEqual(summary["feature_dimensions"], {evaluate.RAW: 125, evaluate.AUGMENTED: 185})
            self.assertIn("규제 탐색 대조군", (Path(td)/"report.md").read_text(encoding="utf-8"))
            self.assertTrue((Path(td)/"orientation_quartile_metrics.csv").exists())

    def test_no_torch_and_default_paths(self):
        p = subprocess.run([sys.executable, "-c", "import sys; from scripts import evaluate_local_structure; assert 'torch' not in sys.modules"], cwd=evaluate.ROOT, capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stderr)
        for seed in (42, 2026):
            self.assertEqual(evaluate.parse_args(["--seed", str(seed)]).output_dir, evaluate.ROOT/f"reports/local_structure/seed{seed}")


if __name__ == "__main__":
    unittest.main()
