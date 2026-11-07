"""Synthetic-only leakage, freezing, optimization, cache and resume checks."""
import contextlib
import copy
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch
from scripts import train_partial_dinov2 as train
from scripts import extract_dinov2_prefix as prefix


class OptimizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_only_last_block_and_head_update_norm_stays_frozen(self):
        torch.manual_seed(12)
        block = torch.nn.Sequential(torch.nn.Linear(384, 384), torch.nn.GELU())
        norm = torch.nn.LayerNorm(384)
        original_block = copy.deepcopy(block.state_dict())
        original_norm = copy.deepcopy(norm.state_dict())
        model = train.make_regressor(torch, block, norm, "partial", np.zeros(1152, np.float32), np.ones(1152, np.float32))
        before = copy.deepcopy(model.state_dict())
        tokens = np.random.default_rng(42).normal(size=(3, 362, 384)).astype(np.float32)
        targets = np.array([1., -2., 3.])
        optimizer = torch.optim.AdamW(train.optimizer_groups(model, train.CONFIGS[3]))
        for _ in range(3):
            train.train_epoch(torch, model, tokens, np.arange(3), targets, optimizer, torch.device("cpu"), 2, 3, np.random.default_rng(4))
        self.assertTrue(any(not torch.equal(v, model.state_dict()[k]) for k, v in before.items() if k.startswith("block.")))
        self.assertFalse(torch.equal(before["head.weight"], model.head.weight))
        for key in before:
            if key.startswith("norm."):
                torch.testing.assert_close(before[key], model.state_dict()[key], rtol=0, atol=0)
        for k, v in original_block.items():
            torch.testing.assert_close(v, block.state_dict()[k], rtol=0, atol=0)
        for k, v in original_norm.items():
            torch.testing.assert_close(v, norm.state_dict()[k], rtol=0, atol=0)
        self.assertTrue(all(not m.training for m in model.modules()))

    def test_accumulation_matches_full_effective_batch_with_unequal_tail(self):
        rng = np.random.default_rng(33)
        x, y = rng.normal(size=(11, 5)).astype(np.float32), rng.normal(size=11)
        models = [train.make_regressor(torch, None, None, "head_only", np.zeros(5, np.float32), np.ones(5, np.float32)) for _ in range(2)]
        for model, batch in zip(models, (2, 5)):
            opt = torch.optim.SGD(model.parameters(), lr=.01)
            _, steps = train.train_epoch(torch, model, x, np.arange(11), y, opt, torch.device("cpu"), batch, 5, np.random.default_rng(7))
            self.assertEqual(steps, 3)
        torch.testing.assert_close(models[0].head.weight, models[1].head.weight, rtol=1e-6, atol=1e-8)
        torch.testing.assert_close(models[0].head.bias, models[1].head.bias, rtol=1e-6, atol=1e-8)

    def test_scaling_uses_fit_subset_and_constant_columns_are_safe(self):
        x = np.array([[1, 1], [3, 1], [99999, 1000]], np.float32)
        mean, scale, ym, ys = train.scaling(x[:2], np.array([2., 4.]))
        np.testing.assert_array_equal(mean, [2, 1])
        np.testing.assert_array_equal(scale, [1, 1])
        self.assertEqual((ym, ys), (3., 1.))
        with self.assertRaises(ValueError):
            train.scaling(x, np.ones(3))

    def test_real_dino_cached_suffix_matches_full_forward_and_gradients(self):
        import timm
        from timm.data import resolve_model_data_config
        torch.manual_seed(6)
        backbone, _ = prefix.base.build_model(torch, timm, resolve_model_data_config, pretrained=False)
        image = prefix.base.reflect_pad_input(torch, torch.randn(1, 3, 256, 256))
        tokens, original = prefix.capture_prefix(torch, backbone, image)
        model = train.make_regressor(torch, backbone.blocks[-1], backbone.norm, "partial",
            np.zeros(1152, np.float32), np.ones(1152, np.float32))
        with torch.no_grad():
            model.head.weight.normal_(std=.01)
        backbone.blocks[-1].requires_grad_(True)
        full_features = prefix.base.pool_tokens(torch, backbone.forward_features(image))[1]
        full_head = copy.deepcopy(model.head)
        full_pred, cached_pred = full_head(full_features).squeeze(-1), model(tokens)
        torch.testing.assert_close(full_features, original, rtol=0, atol=0)
        torch.testing.assert_close(full_pred, cached_pred, rtol=0, atol=0)
        (full_pred - .5).square().mean().backward()
        (cached_pred - .5).square().mean().backward()
        for (name, p), (other, q) in zip(backbone.blocks[-1].named_parameters(), model.block.named_parameters()):
            self.assertEqual(name, other)
            self.assertIsNotNone(p.grad)
            torch.testing.assert_close(p.grad, q.grad, rtol=1e-5, atol=1e-7)
        self.assertTrue(all(p.grad is None for b in backbone.blocks[:-1] for p in b.parameters()))
        self.assertTrue(all(p.grad is None for p in backbone.norm.parameters()))

    def test_real_runner_refit_has_no_outer_metrics_and_checkpoint_roundtrip(self):
        rng = np.random.default_rng(8)
        x = rng.normal(size=(12, 5)).astype(np.float32)
        args = SimpleNamespace(batch_size=3, effective_batch=5, run_fingerprint="synthetic", prefix_cache=Path("synthetic"))
        runner = train.TorchRunner(torch, None, None, x, None, torch.device("cpu"), args)
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            path = Path(tmp) / "head.pt"
            result = runner(np.arange(8), np.arange(8, dtype=float), np.arange(8, 12), None, train.CONFIGS[0], 3, 42, path)
            self.assertTrue(all("valid_mse" not in r for r in result["curve"]))
            saved = torch.load(path, map_location="cpu", weights_only=True)
            model = train.make_regressor(torch, None, None, "head_only", np.zeros(5, np.float32), np.ones(5, np.float32))
            model.load_state_dict(saved["state_dict"])
            pred = train.predict(torch, model, x, np.arange(8, 12), torch.device("cpu"), 3, saved["target_mean"], saved["target_std"])
            np.testing.assert_array_equal(pred, result["prediction"])


class NestedTests(unittest.TestCase):
    @staticmethod
    def runner(train_idx, train_y, valid_idx, valid_y, config, epochs, seed, checkpoint=None):
        curve = [{"epoch": e, "online_train_mse": 1., "optimizer_steps": 2,
            **({"valid_mse": float(np.mean((valid_y-train_y.mean())**2)) + (e-2)**2} if valid_y is not None else {})}
            for e in range(1, epochs+1)]
        return {"curve": curve, "prediction": np.full(len(valid_idx), train_y.mean()), "train_rmse": float(train_y.std())}

    def test_outer_labels_cannot_change_selection_memberships_or_predictions(self):
        y = np.random.default_rng(5).normal(size=100)
        folds = np.arange(100) % 5
        spy = mock.Mock(side_effect=self.runner)
        with contextlib.redirect_stdout(io.StringIO()):
            a = train.fit_outer_fold(y, folds, 42, 2, spy, 3)
            changed = y.copy(); changed[folds == 2] += 9999
            b = train.fit_outer_fold(changed, folds, 42, 2, self.runner, 3)
        self.assertEqual(a, b)
        self.assertEqual(spy.call_count, 14)
        for call in spy.call_args_list[:12]:
            self.assertTrue(np.all(folds[call.args[0]] != 2))
            self.assertTrue(np.all(folds[call.args[2]] != 2))
        for call in spy.call_args_list[12:]:
            self.assertIsNone(call.args[3])
            self.assertEqual(len(call.args[0]), 80)
            self.assertEqual(call.args[5], 2)
        for inner in range(3):
            members = [r for r in a["memberships"] if r["inner_fold"] == inner]
            self.assertEqual(len(members), 80)
            self.assertEqual(len({r["ID"] for r in members}), 80)

    def test_epoch_selection_is_weighted_pooled_not_average_best(self):
        curves = {}
        for c in train.CONFIGS:
            curves[c["name"]] = [[{"epoch": 1, "valid_mse": v}, {"epoch": 2, "valid_mse": 4.}]
                for v in (0., 7., 7.)]
        selected, candidates = train.select_candidates(curves, [50, 25, 25], 2)
        self.assertEqual(selected["partial"]["epoch"], 1)
        self.assertEqual(selected["partial"]["inner_mse"], 3.5)
        self.assertEqual(selected["head_only"]["config"], train.CONFIGS[0])
        broken = copy.deepcopy(curves); broken[train.CONFIGS[0]["name"]][0].pop()
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            train.select_candidates(broken, [50, 25, 25], 2)

    def test_fixed_blend_replaces_only_dino_half(self):
        refs = {train.evaluation.CURRENT: np.array([4., 5.]), train.evaluation.DINO: np.array([2., 8.])}
        preds = {train.PRIMARY: np.array([6., 4.]), train.CONTROL: refs[train.evaluation.DINO].copy()}
        out = train.add_fixed_predictions(preds, refs)
        np.testing.assert_array_equal(out[train.FIXED], [6, 3])
        np.testing.assert_array_equal(out[train.CONTROL_BLEND], refs[train.evaluation.CURRENT])


class CacheAndOutputTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_prefix_cache_integrity_protocol_and_source_checks(self):
        g = self.root / "global"; g.mkdir()
        (g / "features.npz").write_bytes(b"synthetic-global")
        (g / "metadata.json").write_text("{}")
        original = {k: str(i)*64 for i, k in enumerate(("feature_state_sha256", "train_csv_sha256", "image_manifest_sha256"), start=1)}
        ids = np.array(["TRAIN_000000", "TRAIN_000001"])
        tokens = np.zeros((2, 362, 384), np.float32)
        meta = {**original, "protocol": prefix.PROTOCOL, "ids": ids.tolist(), "test_accessed": False, "labels_saved": False,
            "global_reference": prefix.spatial.global_reference(g / "features.npz"), "sources": prefix.sources(),
            "extraction_config": prefix.spatial.dino.EXPECTED_CONFIG,
            "global_reextraction_check": {"passed": True, "atol": prefix.spatial.MATCH_ATOL, "rtol": prefix.spatial.MATCH_RTOL, "max_absolute_difference": 0.}}
        folder = self.root / "prefix"
        prefix.save_cache(np, folder, tokens, meta)
        actual, metadata = prefix.validate_cache(np, folder, ids, original, g / "features.npz")
        np.testing.assert_array_equal(actual, tokens)
        del actual  # Release Windows mmap before writing the synthetic file again.
        for key, value in (("ids", ids[::-1].tolist()), ("labels_saved", True), ("feature_state_sha256", "e"*64), ("sources", {})):
            bad = {**metadata, key: value}
            (folder / "metadata.json").write_text(json.dumps(bad))
            with self.assertRaises(ValueError):
                prefix.validate_cache(np, folder, ids, original, g / "features.npz")
        (folder / "metadata.json").write_text(json.dumps(metadata))
        with (folder / "tokens.npy").open("ab") as handle:
            handle.write(b"corruption")
        with self.assertRaisesRegex(ValueError, "hash"):
            prefix.validate_cache(np, folder, ids, original, g / "features.npz")

    def test_resume_requires_exact_contract_and_untampered_fold(self):
        folder = self.root / "run"
        run_id = train.prepare_output(folder, {"seed": 42}, False)
        self.assertEqual(train.prepare_output(folder, {"seed": 42}, True), run_id)
        with self.assertRaisesRegex(ValueError, "mismatch"):
            train.prepare_output(folder, {"seed": 2026}, True)
        f = folder / "fold0"; f.mkdir()
        result = {"outer_fold": 0, "outer_valid_indices": [0, 5]}
        train.write_json(f / "result.json", result)
        for name in ("head_only.pt", "partial.pt"):
            (f / name).write_bytes(b"synthetic")
        train.write_json(f / "done.json", {"run_fingerprint": run_id,
            "files": {n: train.core.sha256(f/n) for n in ("result.json", "head_only.pt", "partial.pt")}})
        self.assertEqual(train.read_completed_fold(f, run_id, 0, np.arange(10)%5), result)
        (f / "partial.pt").write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "hash"):
            train.read_completed_fold(f, run_id, 0, np.arange(10)%5)

    def test_preflight_and_validate_only_do_not_load_model_fit_or_write(self):
        data = SimpleNamespace(y=np.arange(50), ids=np.array([str(i) for i in range(50)]), metadata={}, provenance={})
        for mode in ("--preflight", "--validate-only"):
            with mock.patch.object(train.evaluation, "load_prior", return_value=(copy.deepcopy(data), {}, {})), \
                 mock.patch.object(prefix, "ensure_local_checkpoint", return_value=[]), \
                 mock.patch.object(prefix, "validate_cache", return_value=(np.zeros(1), {})), \
                 mock.patch.object(train.core, "sha256", return_value="a"*64), \
                 mock.patch.object(prefix.common, "import_runtime") as runtime, \
                 mock.patch.object(train, "fit_outer_fold") as fitting, contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(train.main([mode, "--output-dir", str(self.root / "absent")]), 0)
                runtime.assert_not_called(); fitting.assert_not_called()
                self.assertFalse((self.root / "absent").exists())

    def test_output_protects_inputs_and_existing_run(self):
        args = train.parse_args(["--output-dir", str(train.ROOT)])
        with self.assertRaisesRegex(ValueError, "protected"):
            train.protected_output(args)
        args.output_dir = self.root; (self.root / "old.txt").write_text("keep")
        with self.assertRaisesRegex(ValueError, "exists"):
            train.protected_output(args)

    def test_report_and_diagnostics_save_complete_oof(self):
        from scripts import evaluate_frozen_resnet as core
        rng = np.random.default_rng(12)
        y = rng.normal(size=50)
        data = core.Inputs(ids=np.array([f"TRAIN_{i:06d}" for i in range(50)]), y=y,
            folds=np.arange(50)%5, features={"cls_patch_mean_std": rng.normal(size=(50, 1152))},
            baseline=y+rng.normal(size=50), noise=rng.normal(size=50), metadata={}, provenance={})
        refs = {train.evaluation.CURRENT: y+rng.normal(size=50), train.evaluation.DINO: y+rng.normal(size=50),
            train.evaluation.refs_module.HAND_DINO_REFERENCE: y+rng.normal(size=50)}
        predictions = train.add_fixed_predictions({train.PRIMARY: y+rng.normal(size=50), train.CONTROL: y+rng.normal(size=50)}, refs)
        diagnostics, tables = train.evaluation.diagnose(data, predictions)
        with contextlib.redirect_stdout(io.StringIO()):
            records = [train.fit_outer_fold(y, data.folds, 42, f, NestedTests.runner, 3) for f in range(5)]
        args = train.parse_args(["--output-dir", str(self.root)])
        args.run_fingerprint = "synthetic"
        train.write_outputs(args, data, predictions, records, diagnostics, tables, {}, {}, 0.)
        saved = json.loads((self.root / "summary.json").read_text())
        self.assertEqual(saved["primary_metric"], "partial_nested pooled OOF RMSE")
        import csv
        with (self.root / "oof.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 50)
        self.assertIn(core.BASELINE_NAME, rows[0])
        self.assertTrue((self.root / "inner_curves.csv").exists())

    def test_main_assembles_all_folds_and_resume_does_not_train_again(self):
        rng = np.random.default_rng(22)
        tokens = rng.normal(size=(50, 362, 384)).astype(np.float32)
        global_features = prefix.base.pool_tokens(torch, torch.from_numpy(tokens))[1].numpy()
        y = rng.normal(size=50)
        data = train.core.Inputs(ids=np.array([f"TRAIN_{i:06d}" for i in range(50)]), y=y, folds=np.arange(50)%5,
            features={"cls_patch_mean_std": global_features}, baseline=y+.4, noise=rng.normal(size=50),
            metadata={"feature_state_sha256": "a"*64}, provenance={})
        refs = {train.evaluation.CURRENT: y+.2, train.evaluation.DINO: y+.3,
            train.evaluation.refs_module.HAND_DINO_REFERENCE: y+.25}
        cache = self.root / "cache"; cache.mkdir()
        (cache / "tokens.npy").write_bytes(b"synthetic mocked cache")
        (cache / "metadata.json").write_text("{}")
        backbone = torch.nn.Module()
        backbone.blocks = torch.nn.Sequential(*[torch.nn.Identity() for _ in range(12)])
        backbone.norm = torch.nn.Identity()
        def fake_runner(*args, **kwargs):
            result = NestedTests.runner(*args, **kwargs)
            checkpoint = kwargs.get("checkpoint")
            if checkpoint is not None:
                checkpoint.write_bytes(b"synthetic mocked checkpoint")
            return result
        with mock.patch.object(train.evaluation, "load_prior", side_effect=lambda args: (copy.deepcopy(data), refs, {})), \
             mock.patch.object(prefix, "validate_cache", return_value=(tokens, {})), \
             mock.patch.object(prefix, "ensure_local_checkpoint", return_value=[]), \
             mock.patch.object(prefix.common, "import_runtime", return_value=(np, torch, None, None, {"synthetic": True})), \
             mock.patch.object(prefix.common, "configure_runtime", return_value=(torch.device("cpu"), {})), \
             mock.patch.object(prefix.base, "build_model", side_effect=lambda *a: (copy.deepcopy(backbone), {})), \
             mock.patch.object(prefix.common, "feature_state_hash", return_value="a"*64), \
             mock.patch.object(train, "TorchRunner", return_value=fake_runner), \
             mock.patch.object(train, "fit_outer_fold", wraps=train.fit_outer_fold) as fitting, contextlib.redirect_stdout(io.StringIO()):
            args = ["--device", "cpu", "--max-epochs", "3", "--prefix-cache", str(cache), "--output-dir", str(self.root / "out")]
            self.assertEqual(train.main(args), 0)
            self.assertEqual(fitting.call_count, 5)
            before = (self.root / "out/oof.csv").read_bytes()
            self.assertEqual(train.main(args + ["--resume"]), 0)
            self.assertEqual(fitting.call_count, 5)
            self.assertEqual(before, (self.root / "out/oof.csv").read_bytes())


if __name__ == "__main__":
    unittest.main()
