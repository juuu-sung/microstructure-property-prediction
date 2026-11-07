"""Synthetic CPU-only spatial-pooling/cache/nested-evaluation checks."""
import contextlib
import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch
from scripts import extract_dinov2_spatial as extract
from scripts import evaluate_dinov2_spatial as evaluate
from scripts import evaluate_frozen_resnet as core
from tests import test_dinov2_learning_curve as fixtures


class PoolingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_exact_spatial_order_population_std_and_every_patch_once(self):
        tokens = torch.zeros((1, 362, 384), dtype=torch.float32)
        tokens[:, 0] = 99999
        grid = np.arange(361, dtype=np.float32).reshape(19, 19)
        tokens[:, 1:] = torch.from_numpy(grid.reshape(361, 1))
        global_features, local = extract.pool_regions(torch, tokens)
        self.assertEqual(tuple(local.shape), (1, 3072))
        coverage = np.zeros((19, 19), dtype=int)
        counts, expected = [], []
        for r0, r1 in ((0, 9), (9, 19)):
            for c0, c1 in ((0, 9), (9, 19)):
                region = grid[r0:r1, c0:c1]
                counts.append(region.size)
                coverage[r0:r1, c0:c1] += 1
                expected.extend([np.full(384, region.mean()), np.full(384, region.std(ddof=0))])
        np.testing.assert_allclose(local.numpy()[0], np.concatenate(expected), rtol=1e-6)
        np.testing.assert_array_equal(coverage, np.ones((19, 19)))
        self.assertEqual(counts, [81, 90, 90, 100])
        np.testing.assert_array_equal(global_features[:, :384].numpy(), np.full((1, 384), 99999))

    def test_position_changes_regions_but_not_global_summary(self):
        tokens = torch.zeros((1, 362, 384), dtype=torch.float32)
        tokens[:, 1:82] = 2
        changed = tokens.clone()
        changed[:, 1:] = torch.flip(tokens[:, 1:], dims=[1])
        g1, r1 = extract.pool_regions(torch, tokens)
        g2, r2 = extract.pool_regions(torch, changed)
        torch.testing.assert_close(g1, g2, atol=0, rtol=0)
        self.assertFalse(torch.equal(r1, r2))

    def test_invalid_tokens_and_changed_global_anchor_rejected(self):
        for bad in (torch.zeros(1, 361, 384), torch.zeros(1, 362, 384, dtype=torch.float64), torch.full((1, 362, 384), float("nan"))):
            with self.assertRaises(RuntimeError):
                extract.pool_regions(torch, bad)
        a = np.ones((3, 1152), dtype=np.float32)
        self.assertEqual(extract.check_global_match(np, a, a), 0)
        extract.check_global_match(np, a + 1e-6, a)
        with self.assertRaisesRegex(ValueError, "differ"):
            extract.check_global_match(np, a + 0.1, a)


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.InputAndCommandTests()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.setUp()
        self.root, self.fx = self.fixture.root, self.fixture.fx
        self.global_path = self.root / "features.npz"
        self.data, self.references, self.anchor = self.fixture.load()
        rng = np.random.default_rng(3)
        self.local = rng.normal(size=(50, 4, 2, 384)).astype(np.float32)
        self.local[:, :, 1] = np.abs(self.local[:, :, 1])
        self.local = self.local.reshape(50, 3072)
        self.output = self.root / "spatial"
        config = {**copy.deepcopy(self.fx.metadata["extraction_config"]), "pooling": extract.POOLING,
            "base_extractor_script_sha256": "a"*64}
        self.metadata = {"schema_version": 1, "rows": 50, "spatial_protocol": copy.deepcopy(extract.PROTOCOL),
            "test_accessed": False, "labels_saved_in_npz": False, "extraction_config": config,
            "global_reference": extract.global_reference(self.global_path),
            **{k: self.fx.metadata[k] for k in ("feature_state_sha256", "train_csv_sha256", "image_manifest_sha256")},
            "feature_array_sha256": {extract.LOCAL_NAME: extract.common.sha256_array(self.local)},
            "global_reextraction_check": {"passed": True, "atol": extract.MATCH_ATOL, "rtol": extract.MATCH_RTOL,
                "max_absolute_difference": 0.0}}
        self.save()

    def save(self, features=None):
        extract.atomic_save(np, self.output, self.fx.ids, self.local if features is None else features, self.metadata)

    def test_global_cache_reordering_and_spatial_roundtrip_exact_global_prefix(self):
        ids, features, metadata = extract.read_global_cache(np, self.global_path, self.root / "train.csv", expected_n=50)
        np.testing.assert_array_equal(ids, self.fx.ids)
        np.testing.assert_array_equal(features, self.fx.dino_features)
        self.assertEqual(metadata, self.fx.metadata)
        with mock.patch.object(core, "make_estimator") as fitting:
            data = evaluate.attach_spatial(self.data, self.output / "features.npz", self.global_path)
            fitting.assert_not_called()
        np.testing.assert_array_equal(data.features[evaluate.AUGMENTED][:, :1152], self.fx.dino_features)
        np.testing.assert_array_equal(data.features[evaluate.AUGMENTED][:, 1152:], self.local)
        self.assertEqual(data.features[evaluate.AUGMENTED].shape, (50, 4224))

    def test_corrupted_array_region_protocol_global_identity_and_sidecar_rejected(self):
        original = copy.deepcopy(self.metadata)
        for mutate in (lambda m: m["spatial_protocol"].update(row_edges=[0, 10, 19]),
                       lambda m: m.update(feature_state_sha256="f"*64),
                       lambda m: m["global_reference"].update(sha256="f"*64),
                       lambda m: m["global_reextraction_check"].update(passed=False),
                       lambda m: m["extraction_config"].update(revision="f"*40)):
            self.metadata = copy.deepcopy(original)
            mutate(self.metadata)
            self.save()
            with self.assertRaises(ValueError):
                extract.validate_spatial_cache(np, self.output / "features.npz", self.fx.ids, self.fx.metadata, self.global_path)
        self.metadata = original
        changed = self.local.copy(); changed[0, 0] += 1
        self.save(changed)
        with self.assertRaisesRegex(ValueError, "hash"):
            extract.validate_spatial_cache(np, self.output / "features.npz", self.fx.ids, self.fx.metadata, self.global_path)
        self.save()
        (self.output / "metadata.json").write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "sidecar"):
            extract.validate_spatial_cache(np, self.output / "features.npz", self.fx.ids, self.fx.metadata, self.global_path)

    def test_evaluator_validate_only_and_preflight_do_not_fit_or_write(self):
        args = ["--spatial-features", str(self.output / "features.npz"), "--global-features", str(self.global_path),
            "--output-dir", str(self.root / "evaluation")]
        for mode in ("--validate-only", "--preflight"):
            with mock.patch.object(evaluate, "load_prior", return_value=(copy.deepcopy(self.data), self.references, self.anchor)), \
                 mock.patch.object(core, "nested_predictions") as fitting, contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(evaluate.main(args + [mode]), 0)
                fitting.assert_not_called()
        self.assertFalse((self.root / "evaluation").exists())

    def test_protected_inputs_and_existing_outputs_rejected(self):
        with mock.patch.object(evaluate, "load_prior") as loader:
            with self.assertRaisesRegex(ValueError, "protected"):
                evaluate.main(["--global-features", str(self.global_path), "--output-dir", str(self.root), "--overwrite"])
            loader.assert_not_called()
        with mock.patch.object(extract, "read_global_cache") as loader:
            with self.assertRaisesRegex(ValueError, "protected"):
                extract.main(["--global-features", str(self.global_path), "--output-dir", str(self.root), "--overwrite"])
            with self.assertRaisesRegex(ValueError, "already exists"):
                extract.main(["--output-dir", str(self.output)])
            loader.assert_not_called()


class NestedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(16)
        original = rng.normal(size=(50, 8))
        augmented = np.concatenate([original, rng.normal(size=(50, 12))], axis=1)
        y = 200 + 8 * original[:, 0] + rng.normal(size=50)
        cls.data = core.Inputs(np.array([f"TRAIN_{i+1:06d}" for i in range(50)]), y,
            {evaluate.GLOBAL: original, evaluate.AUGMENTED: augmented}, np.arange(50) % 5, y+1, np.arange(50), {}, {})
        cls.alphas = evaluate.ALPHAS
        with contextlib.redirect_stdout(io.StringIO()):
            cls.predictions, cls.selections = core.nested_predictions(cls.data, 42, 1, alphas=cls.alphas, representations=evaluate.REPRESENTATIONS)
            old, old_selections = core.nested_predictions(cls.data, 42, 1, alphas=cls.alphas, representations=(evaluate.GLOBAL,))
        cls.anchor = {"predictions": old[core.PRIMARY_NAME], "prediction_column": evaluate.GLOBAL+"_nested",
            "selections": {r["outer_fold"]: {"winner": r["per_representation"][evaluate.GLOBAL], "candidates": r["candidates"],
                "inner_validation_sizes": r["inner_validation_sizes"]} for r in old_selections}}

    def test_original_global_exact_reproduction_and_wrong_anchor_rejected(self):
        result = evaluate.verify_global_anchor(self.data, self.predictions, self.selections, self.anchor)
        self.assertEqual(result["max_absolute_prediction_difference"], 0)
        self.assertEqual(result["max_absolute_inner_mse_difference"], 0)
        wrong = copy.deepcopy(self.anchor); wrong["predictions"][0] += 1
        with self.assertRaisesRegex(ValueError, "reproducibility"):
            evaluate.verify_global_anchor(self.data, self.predictions, self.selections, wrong)

    def test_outer_labels_do_not_affect_selection_or_predictions_for_that_fold(self):
        changed = copy.deepcopy(self.data)
        changed.y[changed.folds == 0] += 100
        with contextlib.redirect_stdout(io.StringIO()):
            predictions, selections = core.nested_predictions(changed, 42, 1, alphas=self.alphas, representations=evaluate.REPRESENTATIONS)
        self.assertEqual(selections[0], self.selections[0])
        for name in predictions:
            np.testing.assert_array_equal(predictions[name][self.data.folds == 0], self.predictions[name][self.data.folds == 0])

    def test_fixed_blend_keeps_25_25_50_and_fallback_reproduces_current(self):
        hand, resnet, old_dino = self.data.y+1, self.data.y+2, self.data.y+3
        current = 0.25*hand + 0.25*resnet + 0.5*old_dino
        refs = {evaluate.CURRENT: current, evaluate.DINO: old_dino, evaluate.refs_module.HAND_DINO_REFERENCE: 0.5*(hand+old_dino)}
        result = evaluate.add_fixed_candidate(self.predictions, refs)
        np.testing.assert_allclose(result[evaluate.FIXED_BLEND], 0.25*hand+0.25*resnet+0.5*self.predictions[core.PRIMARY_NAME], rtol=0, atol=1e-12)
        fallback = {**self.predictions, core.PRIMARY_NAME: old_dino}
        np.testing.assert_array_equal(evaluate.add_fixed_candidate(fallback, refs)[evaluate.FIXED_BLEND], current)
        diagnostics, tables = evaluate.diagnose(self.data, result)
        anchor_check = evaluate.verify_global_anchor(self.data, self.predictions, self.selections, self.anchor)
        with tempfile.TemporaryDirectory(prefix="spatial-synthetic-report-") as directory:
            evaluate.write_outputs(Path(directory), self.data, result, self.selections, diagnostics, tables, 42, 1, 0.0, anchor_check)
            report = (Path(directory)/"report.md").read_text(encoding="utf-8")
            self.assertIn("dino_replaced_blend", report)
            summary = json.loads((Path(directory)/"summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["candidate_count_per_inner_search"], 12)
            self.assertEqual(summary["global_anchor"]["status"], "matched")
            self.assertTrue((Path(directory)/"target_region_metrics.csv").is_file())

    def test_evaluator_imports_no_torch_and_cli_defaults(self):
        result = subprocess.run([sys.executable, "-c", "import sys; from scripts import evaluate_dinov2_spatial; assert 'torch' not in sys.modules"],
            cwd=evaluate.ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        for seed in (42, 2026):
            args = evaluate.parse_args(["--seed", str(seed)])
            self.assertEqual(args.output_dir, evaluate.ROOT/f"reports/dinov2_spatial/seed{seed}")
            self.assertIn(f"seed{seed}", str(args.previous_dino_oof))


if __name__ == "__main__":
    unittest.main()
