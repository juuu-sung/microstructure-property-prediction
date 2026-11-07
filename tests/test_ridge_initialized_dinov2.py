"""Synthetic-only checks; no competition fitting, pretrained model or GPU."""
import contextlib
import copy
import csv
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch
from scripts import train_ridge_initialized_dinov2 as train


class OptimizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        rng = np.random.default_rng(17)
        self.x = rng.normal(size=(27, 19)).astype(np.float32)
        self.x[:, -1] = 4
        self.y = 30 + 5 * rng.normal(size=27)
        self.it, self.iv = np.arange(21), np.arange(21, 27)
        self.est, self.state = train.fit_ridge(self.x, self.it, self.y[self.it])

    def model(self):
        return train.make_regressor(torch, None, None, "head_only", self.state)

    def test_ridge_prediction_and_explicit_objective_stationary_with_unpenalized_bias(self):
        model = self.model()
        y = (self.y[self.it] - self.state["target_mean"]) / self.state["target_std"]
        pred = model(torch.from_numpy(self.x)).detach().numpy() * self.state["target_std"] + self.state["target_mean"]
        np.testing.assert_allclose(pred, self.est.predict(self.x.astype(np.float64)), rtol=0, atol=1e-12)
        loss = (model(torch.from_numpy(self.x[self.it])) - torch.as_tensor(y)).square().mean()
        loss = loss + train.ALPHA / len(y) * model.head.weight.square().sum()
        loss.backward()
        self.assertLess(float(model.head.weight.grad.abs().max()), 1e-12)
        self.assertLess(float(model.head.bias.grad.abs().max()), 1e-12)
        np.testing.assert_allclose(self.state["mean"], self.x[self.it].astype(np.float64).mean(0), atol=0, rtol=0)
        self.assertEqual(self.state["scale"][-1], 1)

    def test_full_subset_accumulation_matches_independent_objective_including_unequal_tail(self):
        target = (self.y[self.it] - self.state["target_mean"]) / self.state["target_std"]
        models = [self.model() for _ in range(3)]
        for model in models:
            with torch.no_grad():
                model.head.weight.add_(0.05)
                model.head.bias.add_(0.15)
        for model, batch in zip(models[:2], (4, 13)):
            opt = torch.optim.SGD(model.head.parameters(), lr=0.01)
            stats = train.train_epoch(torch, model, self.x, self.it, target, opt, None, torch.device("cpu"), batch)
            self.assertEqual(stats["optimizer_steps"], 1)
        model = models[2]
        z = model.standardized_features(torch.from_numpy(self.x[self.it]))
        objective = (model.head(z).squeeze(-1) - torch.as_tensor(target)).square().mean()
        objective += train.ALPHA / len(target) * model.head.weight.square().sum()
        objective.backward()
        lr = min(.01, .5 / (float(z.square().sum()) / len(target) + 1 + train.ALPHA / len(target)))
        with torch.no_grad():
            for p in model.head.parameters():
                p.add_(p.grad, alpha=-lr)
        self.assertAlmostEqual(stats["objective_before_step"], float(objective.detach()), places=12)
        for actual in models[:2]:
            for a, b in zip(actual.head.parameters(), model.head.parameters()):
                torch.testing.assert_close(a, b, rtol=0, atol=1e-15)

    def test_head_control_stays_at_ridge_and_checkpoint_roundtrip(self):
        args = SimpleNamespace(batch_size=4, run_fingerprint="synthetic", prefix_cache=Path("synthetic"))
        runner = train.TorchRunner(torch, None, None, self.x, None, torch.device("cpu"), args)
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            path = Path(folder) / "head.pt"
            result = runner(self.it, self.y[self.it], self.iv, None, train.CONFIGS[0], 80, 42, path)
            self.assertLess(result["final_prediction_max_change_from_ridge"], 1e-12)
            self.assertTrue(all("valid_mse" not in r for r in result["curve"]))
            self.assertEqual(sum(r["optimizer_steps"] for r in result["curve"]), 80)
            saved = torch.load(path, map_location="cpu", weights_only=True)
            model = self.model()
            model.load_state_dict(saved["state_dict"])
            pred = train.previous.predict(torch, model, self.x, self.iv, torch.device("cpu"), 4,
                saved["target_mean"], saved["target_std"])
            np.testing.assert_array_equal(pred, result["prediction"])

    def test_epoch_zero_returns_canonical_ridge_with_zero_optimizer_steps(self):
        args = SimpleNamespace(batch_size=4, run_fingerprint="synthetic", prefix_cache=Path("synthetic"))
        runner = train.TorchRunner(torch, None, None, self.x, None, torch.device("cpu"), args)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "zero.pt"
            result = runner(self.it, self.y[self.it], self.iv, self.y[self.iv], train.CONFIGS[0], 0, 42, path)
            np.testing.assert_array_equal(result["prediction"], self.est.predict(self.x[self.iv].astype(np.float64)))
            self.assertEqual(len(result["curve"]), 1)
            self.assertEqual(result["curve"][0]["optimizer_steps"], 0)
            saved = torch.load(path, map_location="cpu", weights_only=True)
            np.testing.assert_array_equal(saved["canonical_epoch0_predictions"], result["prediction"])

    def test_partial_initialization_and_update_preserve_norm_and_block_template(self):
        torch.manual_seed(9)
        block = torch.nn.Sequential(torch.nn.Linear(384, 384), torch.nn.GELU())
        norm = torch.nn.LayerNorm(384)
        tokens = np.random.default_rng(7).normal(size=(9, 362, 384)).astype(np.float32)
        with torch.no_grad():
            x = train.prefix.base.pool_tokens(torch, norm(block(torch.from_numpy(tokens))))[1].numpy()
        y = np.arange(9, dtype=float)
        args = SimpleNamespace(batch_size=3, run_fingerprint="synthetic", prefix_cache=Path("synthetic"))
        before = copy.deepcopy(block.state_dict())
        runner = train.TorchRunner(torch, block, norm, x, tokens, torch.device("cpu"), args)
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            path = Path(folder) / "partial.pt"
            result = runner(np.arange(6), y[:6], np.arange(6, 9), None, train.CONFIGS[1], 2, 42, path)
            self.assertLess(result["epoch0_max_absolute_difference"], 1e-4)
            saved = torch.load(path, map_location="cpu", weights_only=True)["state_dict"]
            self.assertTrue(any(not torch.equal(saved["block." + k], v) for k, v in before.items()))
            for k, v in norm.state_dict().items():
                torch.testing.assert_close(saved["norm." + k], v, rtol=0, atol=0)
            for k, v in before.items():
                torch.testing.assert_close(block.state_dict()[k], v, rtol=0, atol=0)


class NestedTests(unittest.TestCase):
    @staticmethod
    def runner(train_idx, train_y, valid_idx, valid_y, config, epochs, seed, checkpoint=None):
        p = np.full(len(valid_idx), train_y.mean())
        curve = [{"epoch": e, "optimizer_steps": int(e > 0),
            **({"valid_mse": float(np.mean((p - valid_y)**2)) + e} if valid_y is not None else {}),
            **({"objective_before_step": 1.} if e else {})} for e in range(epochs + 1)]
        if checkpoint is not None:
            checkpoint.write_bytes(b"synthetic checkpoint")
        return {"curve": curve, "prediction": p, "ridge_prediction": p.copy(), "train_rmse": float(train_y.std()),
            "epoch0_max_absolute_difference": 0., "final_prediction_max_change_from_ridge": 0.}

    def test_outer_labels_never_reach_selection_and_epoch_zero_refit_is_allowed(self):
        y = np.random.default_rng(8).normal(size=100)
        folds = np.arange(100) % 5
        spy = mock.Mock(side_effect=self.runner)
        with contextlib.redirect_stdout(io.StringIO()):
            a = train.fit_outer_fold(y, folds, 42, 2, spy, 3)
            changed = y.copy(); changed[folds == 2] += 123456
            b = train.fit_outer_fold(changed, folds, 42, 2, self.runner, 3)
        self.assertEqual(a, b)
        self.assertEqual(spy.call_count, 11)
        for call in spy.call_args_list[:9]:
            self.assertTrue(np.all(folds[call.args[0]] != 2))
            self.assertTrue(np.all(folds[call.args[2]] != 2))
        for call in spy.call_args_list[9:]:
            self.assertIsNone(call.args[3])
            self.assertEqual(call.args[5], 0)
        self.assertTrue(all(c["epoch"] == 0 for c in a["selected"].values()))

    def test_weighted_selection_roundoff_ties_and_invalid_curves(self):
        curves = {c["name"]: [[{"epoch": 0, "valid_mse": v}, {"epoch": 1, "valid_mse": 4.}]
            for v in (0., 7., 7.)] for c in train.CONFIGS}
        selected, _ = train.select_candidates(curves, [50, 25, 25], 1)
        self.assertEqual(selected["partial"]["inner_mse"], 3.5)
        self.assertEqual(selected["partial"]["epoch"], 0)
        for rows in curves.values():
            for row in rows:
                row[1]["valid_mse"] = row[0]["valid_mse"] - 1e-9 if row[0]["valid_mse"] else 0
        self.assertEqual(train.select_candidates(curves, [50, 25, 25], 1)[0]["partial"]["epoch"], 0)
        curves[train.CONFIGS[0]["name"]][0].pop()
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            train.select_candidates(curves, [50, 25, 25], 1)

    def test_anchor_policy_rejects_incompatible_prior_and_blend_fallback_is_exact(self):
        dino = np.array([14., 3.])
        refs = {train.evaluation.DINO: dino, train.evaluation.CURRENT: np.array([6., 8.])}
        anchor = {"selections": {i: {"winner": {"alpha": 1000}} for i in range(5)}, "predictions": dino.copy()}
        train.validate_anchor_policy(anchor, refs)
        pred = train.add_fixed_predictions({train.PRIMARY: dino, train.CONTROL: dino}, refs)
        np.testing.assert_array_equal(pred[train.FIXED], refs[train.evaluation.CURRENT])
        anchor["selections"][1]["winner"]["alpha"] = 3000
        with self.assertRaisesRegex(ValueError, "alpha"):
            train.validate_anchor_policy(anchor, refs)

    def test_epoch_zero_anchor_mismatch_is_fatal(self):
        y, folds = np.arange(100, dtype=float), np.arange(100) % 5
        anchor = {"selections": {0: {"inner_validation_sizes": [27, 27, 26], "winner": {"inner_fold_mse": [-1., -1., -1.]}}}}
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "anchor mismatch"):
            train.fit_outer_fold(y, folds, 42, 0, self.runner, 2, anchor=anchor)


class PipelineTests(unittest.TestCase):
    def test_validation_modes_never_fit_import_runtime_or_write(self):
        anchor = {"selections": {i: {"winner": {"alpha": 1000}} for i in range(5)}, "predictions": np.zeros(50)}
        data = SimpleNamespace(y=np.zeros(50), ids=np.arange(50), metadata={}, provenance={})
        refs = {train.evaluation.DINO: np.zeros(50)}
        with tempfile.TemporaryDirectory() as tmp, \
             mock.patch.object(train.evaluation, "load_prior", return_value=(data, refs, anchor)), \
             mock.patch.object(train.prefix, "ensure_local_checkpoint", return_value=[]), \
             mock.patch.object(train.prefix, "validate_cache", return_value=(None, {})), \
             mock.patch.object(train.core, "sha256", return_value="synthetic"), \
             mock.patch.object(train, "fit_ridge", side_effect=AssertionError("fit forbidden")), \
             mock.patch.object(train.prefix.common, "import_runtime", side_effect=AssertionError("runtime forbidden")), \
             contextlib.redirect_stdout(io.StringIO()):
            for mode in ("--preflight", "--validate-only"):
                self.assertEqual(train.main([mode, "--output-dir", str(Path(tmp) / "absent")]), 0)
                self.assertFalse((Path(tmp) / "absent").exists())

    def test_main_assembles_diagnostics_and_resumes_without_any_new_fits(self):
        rng = np.random.default_rng(22)
        tokens = rng.normal(size=(50, 362, 384)).astype(np.float32)
        features = train.prefix.base.pool_tokens(torch, torch.from_numpy(tokens))[1].numpy()
        y, folds = rng.normal(size=50), np.arange(50) % 5
        data = train.core.Inputs(ids=np.array([f"TRAIN_{i:06d}" for i in range(50)]), y=y, folds=folds,
            features={"cls_patch_mean_std": features}, baseline=y + .4, noise=rng.normal(size=50),
            metadata={"feature_state_sha256": "a" * 64}, provenance={})
        # Build a synthetic prior for the fake runner, including inner epoch0 checks.
        with contextlib.redirect_stdout(io.StringIO()):
            prior = [train.fit_outer_fold(y, folds, 42, f, NestedTests.runner, 2) for f in range(5)]
        dino = np.zeros(50)
        selections = {}
        for r in prior:
            dino[r["outer_valid_indices"]] = r["predictions"][train.PRIMARY]
            selections[r["outer_fold"]] = {"winner": {"alpha": 1000,
                "inner_fold_mse": r["selected"]["partial"]["inner_fold_mse"]}, "inner_validation_sizes": r["inner_validation_sizes"]}
        anchor = {"predictions": dino, "selections": selections}
        refs = {train.evaluation.CURRENT: y + .2, train.evaluation.DINO: dino,
            train.evaluation.refs_module.HAND_DINO_REFERENCE: y + .25}
        backbone = torch.nn.Module()
        backbone.blocks = torch.nn.Sequential(*[torch.nn.Identity() for _ in range(12)])
        backbone.norm = torch.nn.Identity()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); cache = root / "cache"; cache.mkdir()
            (cache / "tokens.npy").write_bytes(b"synthetic cache")
            (cache / "metadata.json").write_text("{}")
            with mock.patch.object(train.evaluation, "load_prior", side_effect=lambda a: (copy.deepcopy(data), refs, anchor)), \
                 mock.patch.object(train.prefix, "validate_cache", return_value=(tokens, {})), \
                 mock.patch.object(train.prefix, "ensure_local_checkpoint", return_value=[]), \
                 mock.patch.object(train.prefix.common, "import_runtime", return_value=(np, torch, None, None, {})), \
                 mock.patch.object(train.prefix.common, "configure_runtime", return_value=(torch.device("cpu"), {})), \
                 mock.patch.object(train.prefix.base, "build_model", side_effect=lambda *a: (copy.deepcopy(backbone), {})), \
                 mock.patch.object(train.prefix.common, "feature_state_hash", return_value="a" * 64), \
                 mock.patch.object(train, "TorchRunner", return_value=NestedTests.runner), \
                 mock.patch.object(train, "fit_outer_fold", wraps=train.fit_outer_fold) as fitting, contextlib.redirect_stdout(io.StringIO()):
                args = ["--device", "cpu", "--max-epochs", "2", "--prefix-cache", str(cache), "--output-dir", str(root / "out")]
                self.assertEqual(train.main(args), 0)
                self.assertEqual(fitting.call_count, 5)
                before = (root / "out/oof.csv").read_bytes()
                self.assertEqual(train.main(args + ["--resume"]), 0)
                self.assertEqual(fitting.call_count, 5)
                self.assertEqual(before, (root / "out/oof.csv").read_bytes())
                summary = json.loads((root / "out/summary.json").read_text())
                self.assertEqual(len(summary["epoch_zero_selections"]), 10)
                self.assertEqual(summary["primary_metric"], train.PRIMARY + " pooled OOF RMSE")
                with (root / "out/inner_curves.csv").open() as handle:
                    rows = list(csv.DictReader(handle))
                self.assertEqual(rows[0]["objective_before_step"], "")
                self.assertEqual(rows[1]["objective_before_step"], "1.0")
                with self.assertRaisesRegex(ValueError, "mismatch"):
                    train.main(args + ["--resume", "--batch-size", "4"])
                (root / "out/fold0/partial.pt").write_bytes(b"corrupt")
                with self.assertRaisesRegex(ValueError, "hash mismatch"):
                    train.main(args + ["--resume"])


if __name__ == "__main__":
    unittest.main()
