"""Synthetic CPU tests; never fit/forward on competition data or download."""
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
from scripts import extract_dinov2_d4 as extract
from scripts import evaluate_dinov2_d4 as evaluate
from tests import test_dinov2_learning_curve as fixtures

core = evaluate.core


class TransformTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_eight_unique_permutations_group_closure_and_no_pixel_change(self):
        x = torch.arange(25).reshape(1, 1, 5, 5).float()
        views = list(extract.d4_views(torch, x))
        enc = lambda t: tuple(t.flatten().tolist())
        orbit = {enc(t) for t in views}
        self.assertEqual(len(orbit), 8)
        torch.testing.assert_close(views[0], x, atol=0, rtol=0)
        for view in views:
            torch.testing.assert_close(view.flatten().sort().values, x.flatten(), atol=0, rtol=0)
            self.assertEqual({enc(v) for v in extract.d4_views(torch, view)}, orbit)
        with self.assertRaises(ValueError):
            list(extract.d4_views(torch, torch.zeros(1, 3, 2, 3)))

    def test_transform_and_symmetric_padding_commute(self):
        x = torch.randn(2, 3, 256, 256)
        padded_first = list(extract.d4_views(torch, extract.base.reflect_pad_input(torch, x)))
        for view, expected in zip(extract.d4_views(torch, x), padded_first):
            torch.testing.assert_close(extract.base.reflect_pad_input(torch, view), expected, atol=0, rtol=0)

    def test_pool_per_view_then_average_and_invariance_for_direction_sensitive_model(self):
        class FakeModel:
            def forward_features(self, x):
                # Deliberately position-sensitive tokens, to detect accidental
                # identity reuse, wrong aggregation, and missing group members.
                pixels = x[:, 0, 5:24, 5:24].reshape(len(x), 361, 1)
                patches = pixels.expand(-1, -1, 384).contiguous()
                cls = x[:, 0, 5, 5][:, None, None].expand(-1, 1, 384)
                return torch.cat([cls, patches], dim=1)
        x = torch.randn(2, 3, 256, 256)
        model = FakeModel()
        identity, result = extract.extract_batch(torch, model, x)
        per_view = [extract.base.pool_tokens(torch, model.forward_features(extract.base.reflect_pad_input(torch, v)))[1]
                    for v in extract.d4_views(torch, x)]
        expected = torch.stack(per_view).double().mean(0).float()
        torch.testing.assert_close(result, expected, atol=0, rtol=0)
        torch.testing.assert_close(identity, per_view[0], atol=0, rtol=0)
        self.assertFalse(torch.allclose(identity, result))
        for view in extract.d4_views(torch, x):
            _, other = extract.extract_batch(torch, model, view)
            torch.testing.assert_close(result, other, atol=1e-6, rtol=1e-6)
        self.assertEqual(result.shape, (2, 1152))
        self.assertTrue(torch.all(result[:, 768:] >= 0))


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.InputAndCommandTests()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.setUp()
        self.root, self.fx = self.fixture.root, self.fixture.fx
        self.global_path = self.root / "features.npz"
        self.data, self.references, self.anchor = self.fixture.load()
        self.values = np.random.default_rng(9).normal(size=(50, 1152)).astype(np.float32)
        self.values[:, 768:] = np.abs(self.values[:, 768:])
        self.output = self.root / "d4"
        config = {**copy.deepcopy(self.fx.metadata["extraction_config"]), "pooling": extract.POOLING,
                  "augmentation": copy.deepcopy(extract.PROTOCOL)}
        self.metadata = {"schema_version": 1, "rows": 50, "d4_protocol": copy.deepcopy(extract.PROTOCOL),
            "test_accessed": False, "labels_saved_in_npz": False, "offline_checkpoint_only": True,
            "extraction_config": config, "global_reference": extract.shared.global_reference(self.global_path),
            **{k: self.fx.metadata[k] for k in ("feature_state_sha256", "train_csv_sha256", "image_manifest_sha256")},
            "feature_array_sha256": {extract.FEATURE: extract.common.sha256_array(self.values)},
            "source_sha256": extract.source_hashes(), "identity_reextraction_check": {"passed": True,
                "atol": extract.shared.MATCH_ATOL, "rtol": extract.shared.MATCH_RTOL,
                "max_absolute_difference": 0.0, "evaluation_uses_original_global_cache": True}}
        self.save()
        self.partial_paths = {cap: self.root / f"partial{cap}.csv" for cap in (80, 160)}
        self.partial_rows = []
        for i, ident in enumerate(self.data.ids):
            old, current = self.references[evaluate.DINO][i], self.references[evaluate.CURRENT][i]
            self.partial_rows.append({"ID": ident, "hardness": self.data.y[i], "fold": self.data.folds[i],
                evaluate.CURRENT: current, evaluate.DINO: old, "ridge_head_only_nested": old,
                "ridge_partial_nested": old+1, "ridge_partial_replaced_blend": current+0.5})
        for path in self.partial_paths.values():
            path.write_text(core.csv_text(self.partial_rows), encoding="utf-8")

    def save(self, values=None, ids=None):
        extract.atomic_save(np, self.output, self.fx.ids if ids is None else ids,
            self.values if values is None else values, self.metadata)

    def test_cache_roundtrip_id_alignment_and_original_exactly_reused(self):
        reversed_values = self.values[::-1].copy()
        self.metadata["feature_array_sha256"] = {extract.FEATURE: extract.common.sha256_array(reversed_values)}
        self.save(reversed_values, self.fx.ids[::-1])
        data = evaluate.attach_features(self.data, self.output / "features.npz", self.global_path)
        np.testing.assert_array_equal(data.features[evaluate.D4], self.values)
        np.testing.assert_array_equal(data.features[evaluate.GLOBAL], self.fx.dino_features)
        self.assertEqual({x.shape for x in data.features.values()}, {(50, 1152)})

    def test_cache_corruption_and_stale_protocol_sources_rejected(self):
        original = copy.deepcopy(self.metadata)
        changes = [lambda m: m["d4_protocol"].update(views_per_image=4),
            lambda m: m["source_sha256"].update(extractor="f"*64),
            lambda m: m["global_reference"].update(sha256="f"*64),
            lambda m: m.update(feature_state_sha256="f"*64),
            lambda m: m["extraction_config"].update(augmentation=None),
            lambda m: m["identity_reextraction_check"].update(passed=False),
            lambda m: m.update(test_accessed=True)]
        for change in changes:
            self.metadata = copy.deepcopy(original)
            change(self.metadata)
            self.save()
            with self.assertRaises(ValueError):
                extract.validate_cache(np, self.output/"features.npz", self.fx.ids, self.fx.metadata, self.global_path)
        self.metadata = original
        changed = self.values.copy(); changed[0, 0] += 1
        self.save(changed)
        with self.assertRaisesRegex(ValueError, "hash"):
            extract.validate_cache(np, self.output/"features.npz", self.fx.ids, self.fx.metadata, self.global_path)
        self.save()
        (self.output/"metadata.json").write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "sidecar"):
            extract.validate_cache(np, self.output/"features.npz", self.fx.ids, self.fx.metadata, self.global_path)

    def test_partial_comparisons_aligned_and_corrupt_folds_or_formula_rejected(self):
        refs = evaluate.attach_partial_references(self.data, self.references, self.partial_paths)
        for name in evaluate.PARTIAL_REFERENCES.values():
            np.testing.assert_array_equal(refs[name], self.references[evaluate.CURRENT]+0.5)
        for column in ("fold", "hardness", evaluate.CURRENT, "ridge_head_only_nested", "ridge_partial_replaced_blend"):
            rows = copy.deepcopy(self.partial_rows); rows[0][column] += 1
            self.partial_paths[80].write_text(core.csv_text(rows), encoding="utf-8")
            with self.assertRaises(ValueError):
                evaluate.attach_partial_references(self.data, self.references, self.partial_paths)

    def test_preflight_validate_only_never_fit_or_write(self):
        refs = evaluate.attach_partial_references(self.data, self.references, self.partial_paths)
        args = ["--global-features", str(self.global_path), "--d4-features", str(self.output/"features.npz"),
                "--output-dir", str(self.root/"evaluation")]
        for mode in ("--preflight", "--validate-only"):
            with mock.patch.object(evaluate, "load_prior", return_value=(copy.deepcopy(self.data), refs, self.anchor)), \
                 mock.patch.object(core, "nested_predictions") as fitting, \
                 mock.patch.object(core, "make_estimator") as factory, contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(evaluate.main(args+[mode]), 0)
                fitting.assert_not_called(); factory.assert_not_called()
        self.assertFalse((self.root/"evaluation").exists())

    def test_protected_input_and_existing_output_abort_before_loading(self):
        with mock.patch.object(evaluate, "load_prior") as load:
            with self.assertRaisesRegex(ValueError, "protected"):
                evaluate.main(["--partial80-oof", str(self.partial_paths[80]), "--output-dir", str(self.root), "--overwrite"])
            load.assert_not_called()
        with mock.patch.object(extract.shared, "read_global_cache") as load:
            with self.assertRaisesRegex(ValueError, "protected"):
                extract.main(["--global-features", str(self.global_path), "--output-dir", str(self.root), "--overwrite"])
            with self.assertRaisesRegex(ValueError, "already exists"):
                extract.main(["--output-dir", str(self.output)])
            load.assert_not_called()


class NestedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(14)
        original, averaged = rng.normal(size=(50, 8)), rng.normal(size=(50, 8))
        y = 200 + 8 * original[:, 0] + 4 * averaged[:, 0] + rng.normal(size=50)
        cls.data = core.Inputs(np.array([f"TRAIN_{i+1:06d}" for i in range(50)]), y,
            {evaluate.GLOBAL: original, evaluate.D4: averaged}, np.arange(50)%5, y+1, np.arange(50), {}, {})
        with contextlib.redirect_stdout(io.StringIO()):
            cls.predictions, cls.selections = core.nested_predictions(cls.data, 42, 1, alphas=evaluate.ALPHAS, representations=evaluate.REPRESENTATIONS)
            old, records = core.nested_predictions(cls.data, 42, 1, alphas=evaluate.ALPHAS, representations=(evaluate.GLOBAL,))
        cls.anchor = {"predictions": old[core.PRIMARY_NAME], "prediction_column": evaluate.GLOBAL + "_nested", "selections": {r["outer_fold"]: {
            "winner": r["per_representation"][evaluate.GLOBAL], "candidates": r["candidates"],
            "inner_validation_sizes": r["inner_validation_sizes"]} for r in records}}

    def test_original_anchor_reproduced_and_wrong_anchor_rejected(self):
        checked = evaluate.prior.verify_global_anchor(self.data, self.predictions, self.selections, self.anchor)
        self.assertEqual(checked["max_absolute_prediction_difference"], 0)
        self.assertEqual(checked["max_absolute_inner_mse_difference"], 0)
        wrong = copy.deepcopy(self.anchor); wrong["predictions"][0] += 1
        with self.assertRaises(ValueError):
            evaluate.prior.verify_global_anchor(self.data, self.predictions, self.selections, wrong)

    def test_held_out_labels_cannot_affect_its_selection_or_prediction(self):
        changed = copy.deepcopy(self.data); changed.y[changed.folds == 0] += 100
        with contextlib.redirect_stdout(io.StringIO()):
            predictions, selections = core.nested_predictions(changed, 42, 1, alphas=evaluate.ALPHAS, representations=evaluate.REPRESENTATIONS)
        self.assertEqual(selections[0], self.selections[0])
        for name in predictions:
            np.testing.assert_array_equal(predictions[name][self.data.folds == 0], self.predictions[name][self.data.folds == 0])

    def test_fixed_weights_fallback_and_complete_synthetic_report(self):
        hand, resnet, dino = self.data.y+1, self.data.y+2, self.data.y+3
        current = 0.25*hand + 0.25*resnet + 0.5*dino
        refs = {evaluate.CURRENT: current, evaluate.DINO: dino,
            evaluate.prior.refs_module.HAND_DINO_REFERENCE: 0.5*(hand+dino),
            **{name: current+cap/10000 for cap, name in evaluate.PARTIAL_REFERENCES.items()}}
        result = evaluate.add_fixed_candidate(self.predictions, refs)
        np.testing.assert_allclose(result[evaluate.FIXED_BLEND], 0.25*hand+0.25*resnet+0.5*self.predictions[core.PRIMARY_NAME], atol=1e-12, rtol=0)
        fallback = {**self.predictions, core.PRIMARY_NAME: dino}
        np.testing.assert_array_equal(evaluate.add_fixed_candidate(fallback, refs)[evaluate.FIXED_BLEND], current)
        diagnostics, tables = evaluate.diagnose(self.data, result)
        checked = evaluate.prior.verify_global_anchor(self.data, self.predictions, self.selections, self.anchor)
        with tempfile.TemporaryDirectory(prefix="d4-synthetic-report-") as directory:
            evaluate.write_outputs(Path(directory), self.data, result, self.selections, diagnostics, tables, 42, 1, 0, checked)
            summary = json.loads((Path(directory)/"summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["candidate_count_per_inner_search"], 12)
            self.assertEqual(summary["estimator_fit_count"], 190)
            self.assertEqual(summary["global_anchor"]["status"], "matched")
            self.assertTrue((Path(directory)/"target_region_metrics.csv").is_file())
            self.assertIn("partial160", (Path(directory)/"report.md").read_text(encoding="utf-8"))

    def test_no_torch_import_in_evaluator_and_both_seed_defaults(self):
        result = subprocess.run([sys.executable, "-c", "import sys; from scripts import evaluate_dinov2_d4; assert 'torch' not in sys.modules"],
                                cwd=evaluate.ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        for seed in (42, 2026):
            args = evaluate.parse_args(["--seed", str(seed)])
            self.assertEqual(args.output_dir, evaluate.ROOT/f"reports/dinov2_d4/seed{seed}")
            self.assertIn(f"seed{seed}", str(args.partial160_oof))


if __name__ == "__main__":
    unittest.main()
