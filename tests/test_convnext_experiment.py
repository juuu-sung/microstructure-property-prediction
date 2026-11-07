"""Synthetic CPU tests only; no competition fits, image features or downloads."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
import torch

from scripts import convnext_experiment as c


class TinyNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 1), torch.nn.GroupNorm(1, 4))
        self.stages = torch.nn.Sequential(*[torch.nn.Sequential(torch.nn.Conv2d(4, 4, 1), torch.nn.GELU()) for _ in range(4)])
        self.head = torch.nn.Sequential(torch.nn.LayerNorm(4), torch.nn.Dropout(.1), torch.nn.Linear(4, 1))

    def forward(self, x):
        return self.head(self.stages(self.stem(x)).mean((2, 3))).squeeze(-1)


class ConvNeXtTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        c.configure_runtime("cpu", 7, threads=2)

    def test_bare_cuda_resolves_current_device_before_set_device(self):
        with patch.object(torch.cuda, "is_available", return_value=True), \
             patch.object(torch.cuda, "current_device", return_value=1) as current, \
             patch.object(torch.cuda, "device_count", return_value=2), \
             patch.object(torch.cuda, "set_device") as setter, patch.object(c, "seed_all") as seeder:
            _, device = c.configure_runtime("cuda", 123, threads=2)
            self.assertEqual(device, torch.device("cuda:1"))
            current.assert_called_once_with()
            setter.assert_called_once_with(torch.device("cuda:1"))
            seeder.assert_called_once_with(torch, 123, torch.device("cuda:1"))

    def test_explicit_cuda_index_is_preserved(self):
        with patch.object(torch.cuda, "is_available", return_value=True), \
             patch.object(torch.cuda, "current_device") as current, \
             patch.object(torch.cuda, "device_count", return_value=2), \
             patch.object(torch.cuda, "set_device") as setter, patch.object(c, "seed_all"):
            _, device = c.configure_runtime("cuda:0", 123, threads=2)
            self.assertEqual(device, torch.device("cuda:0"))
            current.assert_not_called()
            setter.assert_called_once_with(torch.device("cuda:0"))

    def test_invalid_cuda_index_fails_before_setting_or_seeding(self):
        with patch.object(torch.cuda, "is_available", return_value=True), \
             patch.object(torch.cuda, "device_count", return_value=1), \
             patch.object(torch.cuda, "set_device") as setter, patch.object(c, "seed_all") as seeder:
            with self.assertRaisesRegex(ValueError, "index 1 does not exist"):
                c.configure_runtime("cuda:1", 123, threads=2)
            setter.assert_not_called()
            seeder.assert_not_called()

    def test_unavailable_cuda_has_no_cpu_fallback(self):
        with patch.object(torch.cuda, "is_available", return_value=False), \
             patch.object(torch.cuda, "current_device") as current, \
             patch.object(torch.cuda, "set_device") as setter:
            with self.assertRaisesRegex(RuntimeError, "no silent CPU fallback"):
                c.configure_runtime("cuda", 123, threads=2)
            current.assert_not_called()
            setter.assert_not_called()

    def test_native_preprocessing_and_d4(self):
        pixels = np.arange(2*8*8, dtype=np.uint8).reshape(2, 8, 8)
        tensor = c.batch_images(torch, pixels, [0, 1], torch.device("cpu"))
        self.assertEqual(tuple(tensor.shape), (2, 3, 8, 8))
        recovered = tensor.numpy()*np.array(c.STD)[None, :, None, None]+np.array(c.MEAN)[None, :, None, None]
        np.testing.assert_allclose(recovered[:, 0], pixels/255., atol=1e-7)
        x = torch.arange(9).reshape(1, 3, 3)
        views = [c.d4_view(torch, x, i) for i in range(8)]
        self.assertEqual(len({tuple(v.flatten().tolist()) for v in views}), 8)
        for v in views:
            torch.testing.assert_close(v.flatten().sort().values, x.flatten().sort().values)

    def test_all_parameters_mapped_and_warmup_freezes_backbone(self):
        model = TinyNetwork()
        c.phase_mode(model, True)
        model(torch.randn(3, 3, 8, 8)).sum().backward()
        self.assertTrue(all(p.grad is None for name, p in model.named_parameters() if not name.startswith("head.")))
        c.phase_mode(model, False)
        groups = c.parameter_groups(model, False)
        ids = [id(p) for g in groups for p in g["params"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(ids), {id(p) for p in model.parameters()})
        for g in groups:
            for p in g["params"]:
                if p.ndim == 1:
                    self.assertEqual(g["weight_decay"], 0.)
        lrs = {g["base_lr"] for g in groups}
        self.assertEqual(lrs, {3e-6, 6e-6, 1e-5, 3e-5, 3e-4})

    def test_scheduler_is_not_compressed_for_early_refit(self):
        self.assertEqual(c.lr_factor(1), 1.)
        self.assertEqual(c.lr_factor(6), 1.)
        self.assertGreater(c.lr_factor(30), .8)
        self.assertAlmostEqual(c.lr_factor(100), 0.)
        self.assertTrue(all(c.lr_factor(i) >= c.lr_factor(i+1) for i in range(6, 100)))

    def test_inner_split_and_selection(self):
        ti = np.arange(100, 500)
        y = np.sin(ti)
        tr, va = c.split_inner(ti, y, 42, 0)
        self.assertEqual((len(tr), len(va)), (320, 80))
        self.assertFalse(np.intersect1d(tr, va).size)
        self.assertEqual(set(ti), set(tr)|set(va))
        again = c.split_inner(ti, y, 42, 0)
        np.testing.assert_array_equal(tr, again[0])
        curves = {p: {"curve": [{"epoch": e, "valid_rmse": 10.} for e in range(1, 101)]} for p in c.PROFILES}
        self.assertEqual(c.select_profile(curves), {"profile": "none", "epoch": 1, "inner_rmse": 10.})
        curves["d4"]["curve"][25]["valid_rmse"] = 9.
        self.assertEqual(c.select_profile(curves)["epoch"], 26)
        curves["none"]["curve"].pop()
        with self.assertRaises(ValueError):
            c.select_profile(curves)

    def test_outer_label_perturbation_cannot_change_neural_selection(self):
        folds = np.repeat(np.arange(5), 100)
        y = np.linspace(10, 40, 500)
        logs = []
        def runner(ti, ty, vi, vy, profile, epochs, seed, folder):
            logs.append((ti.copy(), ty.copy(), vi.copy(), None if vy is None else vy.copy(), profile, epochs, seed))
            if vy is None:
                return {"prediction": np.repeat(ty.mean(), len(vi)).tolist()}
            return {"curve": [{"epoch": e, "valid_rmse": float(np.mean(ty)+np.mean(vy))+(e-17)**2} for e in range(1, 101)]}
        a = c.neural_fold(runner, y, folds, 42, 0, Path("unused"))
        first = copy.deepcopy(logs)
        y[:100] = -999999.
        logs.clear()
        b = c.neural_fold(runner, y, folds, 42, 0, Path("unused"))
        np.testing.assert_array_equal(a[1], b[1])
        self.assertEqual(a[2], b[2])
        self.assertIsNone(logs[-1][3])
        for old, new in zip(first, logs):
            for i in (0, 1, 2):
                np.testing.assert_array_equal(old[i], new[i])

    def test_ridge_scales_only_fit_rows(self):
        rng = np.random.default_rng(41)
        x, y = rng.normal(size=(40, 8)), rng.normal(size=40)
        ti, vi, tr, va = np.arange(30), np.arange(30, 40), np.arange(20), np.arange(20, 30)
        a = c.ridge_fold(x, y, ti, vi, tr, va)
        y[vi] += 1000
        b = c.ridge_fold(x, y, ti, vi, tr, va)
        np.testing.assert_array_equal(a[0], b[0])
        self.assertEqual(a[1], b[1])

    def test_accumulation_matches_large_batch_even_with_short_last_group(self):
        # No stochastic layer in this check: isolates sample-weighted accumulation.
        torch.manual_seed(2)
        model = TinyNetwork()
        model.head[1] = torch.nn.Identity()
        other = copy.deepcopy(model)
        pixels = np.random.default_rng(2).integers(0, 255, size=(11, 8, 8), dtype=np.uint8)
        y = np.linspace(-1, 1, 11)
        for m, batch, accum in ((model, 2, 4), (other, 8, 1)):
            c.phase_mode(m, False)
            optimizer = torch.optim.SGD(m.parameters(), lr=.02)
            scaler = torch.amp.GradScaler("cpu", enabled=False)
            result = c.optimize_epoch(torch, m, pixels, np.arange(11), y, torch.device("cpu"), optimizer,
                                      scaler, batch, accum, "none", np.random.default_rng(5), False)
            self.assertEqual(result[1:], (2, 0))
        for a, b in zip(model.parameters(), other.parameters()):
            torch.testing.assert_close(a, b, atol=2e-7, rtol=2e-6)

    def test_resume_restores_optimizer_rng_and_rejects_changed_settings(self):
        pixels = np.random.default_rng(2).integers(0, 255, size=(12, 8, 8), dtype=np.uint8)
        ti, vi = np.arange(10), np.arange(10, 12)
        y = np.linspace(10., 20., 10)
        with tempfile.TemporaryDirectory() as td, patch.object(c, "make_model", side_effect=lambda *a: TinyNetwork()):
            root = Path(td)
            run = c.Runner(torch, torch.device("cpu"), pixels, None, "synthetic", 2, 4)
            full = run(ti, y, vi, None, "d4", 7, 99, root/"full")
            original = c.optimize_epoch
            count = [0]
            def fail_after_saved_checkpoint(*args, **kwargs):
                count[0] += 1
                if count[0] == 6:
                    raise RuntimeError("synthetic interruption")
                return original(*args, **kwargs)
            with patch.object(c, "optimize_epoch", side_effect=fail_after_saved_checkpoint):
                with self.assertRaisesRegex(RuntimeError, "synthetic interruption"):
                    run(ti, y, vi, None, "d4", 7, 99, root/"resumed")
            self.assertTrue((root/"resumed/last.pt").exists())
            resumed = run(ti, y, vi, None, "d4", 7, 99, root/"resumed")
            np.testing.assert_array_equal(full["prediction"], resumed["prediction"])
            for a, b in zip(full["curve"], resumed["curve"]):
                self.assertEqual({k:v for k,v in a.items() if k != "seconds"}, {k:v for k,v in b.items() if k != "seconds"})
            self.assertFalse((root/"resumed/last.pt").exists())
            cached = run(ti, y, vi, None, "d4", 7, 99, root/"resumed")
            self.assertEqual(cached, resumed)
            with self.assertRaises(ValueError):
                run(ti, y, vi, None, "none", 7, 99, root/"resumed")
            (root/"resumed/curve.json").write_text("[]")
            with self.assertRaises(ValueError):
                run(ti, y, vi, None, "d4", 7, 99, root/"resumed")

    def test_nonfinite_and_overlap_rejected(self):
        with self.assertRaises(ValueError):
            c.scaling(np.ones(8))
        with self.assertRaises(ValueError):
            c.scaling(np.array([1, float("nan")]))
        run = c.Runner(torch, torch.device("cpu"), None, None, "synthetic")
        with tempfile.TemporaryDirectory() as td, self.assertRaisesRegex(ValueError, "overlap"):
            run(np.arange(4), np.arange(4.), np.arange(2), None, "none", 3, 2, Path(td))

    def test_actual_convnext_shape_gradient_and_no_unused_parameters(self):
        model = c.make_model(torch)
        c.phase_mode(model, False)
        x = torch.randn(1, 3, 256, 256)
        self.assertEqual(tuple(model.features(x).shape), (1, 2688))
        model(x).square().sum().backward()
        for name, p in model.named_parameters():
            self.assertIsNotNone(p.grad, name)
            self.assertTrue(torch.isfinite(p.grad).all(), name)
        for prefix in ("stem.", "stages.0.", "stages.1.", "stages.2.", "stages.3.", "head."):
            self.assertTrue(any(torch.count_nonzero(p.grad) for n,p in model.named_parameters() if n.startswith(prefix)), prefix)

    def test_report_and_two_seed_audit_recompute_saved_oof(self):
        rng = np.random.default_rng(91)
        data = SimpleNamespace(ids=np.array([f"TRAIN_{i:06d}" for i in range(500)]),
                               y=rng.normal(100, 20, 500), noise=rng.normal(size=500))
        refs = {"current": data.y+rng.normal(0, 10, 500), "folds": np.repeat(np.arange(5), 100)}
        plan = {"frozen": {"synthetic": True}}
        binding = {"plan": c.fingerprint(plan["frozen"])}
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for seed in c.SEEDS:
                dest = root/f"seed{seed}"
                dest.mkdir()
                c.report(dest, data, refs, data.y+(refs["current"]-data.y)*.5,
                         refs["current"], [], [], seed, binding)
            with patch.object(c, "validate", return_value=(data, plan, None)), patch.object(c.previous, "load_references", return_value=refs):
                c.compare(root)
                value = c.read_json(root/"comparison.json")
                self.assertTrue(value["passes_numeric_priority_gate"])
                self.assertGreater(value["mean_primary_gain"], 4.)
                with (root/"seed42/oof.csv").open("a") as handle:
                    handle.write("corruption")
                with self.assertRaisesRegex(ValueError, "changed"):
                    c.compare(root)


if __name__ == "__main__":
    unittest.main()
