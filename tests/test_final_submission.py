"""Synthetic CPU/PNG tests only; no real fitting, pretrained models or GPU."""
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
from PIL import Image
import torch
from scripts import final_submission as f


def synthetic_data():
    rng = np.random.default_rng(707)
    features = {n: rng.normal(size=(50, width)).astype(np.float64 if n == f.HAND else np.float32) for n, width in f.WIDTHS.items()}
    features[f.HAND][:, -1] = 1.0
    y = 210 + 3*features[f.HAND][:, 0] + rng.normal(size=50)
    return f.core.Inputs(np.asarray([f"TRAIN_{i:06d}" for i in range(50)]), y, features,
        np.arange(50)%5, np.zeros(50), np.zeros(50), {}, {})


class FakeResNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("scale", torch.tensor(1.))
        self.eval()

    def forward(self, x):
        a = torch.nn.functional.adaptive_avg_pool2d(x.mean(1, keepdim=True), (4, 4))
        return [a.repeat(1, c, 1, 1)*self.scale for c in (128, 256, 512)]


class FakeDino(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("scale", torch.tensor(1.))
        self.shapes = []
        self.eval()

    def forward_features(self, x):
        self.shapes.append(tuple(x.shape))
        tokens = torch.nn.functional.adaptive_avg_pool2d(x.mean(1, keepdim=True), (19, 19)).flatten(2).transpose(1, 2)
        tokens = torch.cat([tokens.mean(1, keepdim=True), tokens], dim=1)
        return tokens.repeat(1, 1, 384)*self.scale


class FinalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.data = synthetic_data()
        factory = f.core.make_estimator
        cls.fit_sizes = []
        def traced(alpha):
            model = factory(alpha)
            original_fit = model.fit
            def fit(x, y, **kwargs):
                cls.fit_sizes.append(len(y))
                return original_fit(x, y, **kwargs)
            model.fit = fit
            return model
        with mock.patch.object(f.core, "make_estimator", side_effect=traced), contextlib.redirect_stdout(io.StringIO()):
            cls.heads, cls.selection = f.fit_heads(cls.data, 1)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.folder = self.root/"result"
        self.folder.mkdir()
        self.plan = {"train": f.train_binding(self.data), "anchor_ids": self.data.ids[:16].tolist(),
            "test": {"ids": ["TEST_000002", "TEST_000001"], "rows": 2}, "checkpoints": {}}
        f.write_json(self.folder/f.PLAN_NAME, self.plan)

    def tearDown(self):
        self.temp.cleanup()

    def save_model(self):
        with mock.patch.object(f, "validate_plan", return_value=self.plan), \
             mock.patch.object(f, "fit_heads", return_value=(self.heads, self.selection)), contextlib.redirect_stdout(io.StringIO()):
            f.train(self.folder, self.data, 1)

    def test_fit_budget_train_only_sizes_and_fixed_grids(self):
        self.assertEqual(len(self.fit_sizes), 39)
        self.assertEqual(self.fit_sizes.count(50), 3)
        self.assertEqual(set(self.fit_sizes), {33, 34, 50})
        self.assertEqual(self.selection["cv_validation_sizes"], [17, 17, 16])
        self.assertEqual(self.selection["cv_seed"], 31415)
        for n in f.COMPONENTS:
            records = [r for r in self.selection["candidates"] if r["component"] == n]
            self.assertEqual([r["alpha"] for r in records], list(f.GRIDS[n]))
            self.assertEqual(self.selection["selected"][n], min(records, key=lambda r: (r["pooled_mse"], -r["alpha"])))
            for r in records:
                self.assertAlmostEqual(r["pooled_mse"], np.average(r["fold_mse"], weights=[17,17,16]), places=10)

    def test_export_matches_sklearn_on_unseen_inputs_and_constant_column(self):
        rng = np.random.default_rng(12)
        for n in f.COMPONENTS:
            x = self.data.features[n].astype(np.float64)
            model = f.core.make_estimator(self.selection["selected"][n]["alpha"])
            model.fit(x, self.data.y)
            novel = rng.normal(size=(7, f.WIDTHS[n]))*3+2
            np.testing.assert_allclose(f.predict_head(self.heads[n], novel), model.predict(novel), atol=1e-10, rtol=0)
        self.assertEqual(self.heads[f.HAND]["x_scale"][-1], 1.)

    def test_train_command_never_reads_test_or_imports_runtime(self):
        with mock.patch.object(f, "load_training", return_value=self.data), \
             mock.patch.object(f, "validate_plan", return_value=self.plan), \
             mock.patch.object(f, "test_inventory", side_effect=AssertionError("Training read test")), \
             mock.patch.object(f.common, "import_runtime", side_effect=AssertionError("Training imported torch")), \
             mock.patch.object(f, "fit_heads", return_value=(self.heads, self.selection)), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(f.main(["train", "--output-dir", str(self.folder)]), 0)
        loaded, meta = f.load_heads(self.folder, self.plan)
        self.assertFalse(meta["test_used_for_fit"])
        for n in loaded:
            np.testing.assert_array_equal(f.predict_head(loaded[n], self.data.features[n]), f.predict_head(self.heads[n], self.data.features[n]))

    def test_model_archive_tampering_and_existing_output_rejected(self):
        self.save_model()
        with mock.patch.object(f, "validate_plan", return_value=self.plan), mock.patch.object(f, "fit_heads") as fit:
            with self.assertRaisesRegex(ValueError, "already exists"):
                f.train(self.folder, self.data, 1)
            fit.assert_not_called()
        path = self.folder/"model/metadata.json"
        meta = json.loads(path.read_text())
        meta["weights"][f.HAND] = .5
        f.write_json(path, meta)
        with self.assertRaisesRegex(ValueError, "sidecar"):
            f.load_heads(self.folder, self.plan)

    def make_images(self):
        root = self.root/"open"
        (root/"test").mkdir(parents=True)
        gray = np.tile(np.arange(256, dtype=np.uint8), (256, 1))
        for ident in self.plan["test"]["ids"]:
            Image.fromarray(gray).save(root/"test"/(ident+".png"))
        (root/"sample_submission.csv").write_text("ID,hardness\nTEST_000002,123456\nTEST_000001,-999\n", encoding="utf-8")
        return root, gray

    def test_template_order_placeholder_independence_and_inventory_hash(self):
        root, _ = self.make_images()
        inv = f.test_inventory(root, 2)
        self.assertEqual(inv["ids"], ["TEST_000002", "TEST_000001"])
        plan = {"test": inv}
        f.validate_test_identity(plan, root)
        (root/"sample_submission.csv").write_text("ID,hardness\nTEST_000001,0\nTEST_000002,0\n")
        with self.assertRaisesRegex(ValueError, "changed"):
            f.validate_test_identity(plan, root)

    def test_duplicate_missing_extra_and_bad_image_rejected(self):
        root, gray = self.make_images()
        template = root/"sample_submission.csv"
        for text in ("ID,hardness\nTEST_000001,0\nTEST_000001,0\n", "hardness,ID\n0,TEST_000001\n0,TEST_000002\n",
                     "ID,hardness\n../outside,0\nTEST_000002,0\n"):
            template.write_text(text)
            with self.assertRaises(ValueError):
                f.test_inventory(root, 2)
        template.write_text("ID,hardness\nTEST_000002,0\nTEST_000001,0\n")
        Image.fromarray(gray).save(root/"test/TEST_999999.png")
        with self.assertRaisesRegex(ValueError, "inventory"):
            f.test_inventory(root, 2)
        (root/"test/TEST_999999.png").unlink()
        Image.fromarray(gray[:128]).save(root/"test/TEST_000001.png")
        with self.assertRaisesRegex(ValueError, "grayscale"):
            f.test_inventory(root, 2)

    def test_normalization_identical_to_train_dataset_without_resize(self):
        root, gray = self.make_images()
        (root/"train").mkdir()
        Image.fromarray(gray).save(root/"train/TRAIN_000001.png")
        original = f.common.TrainImageDataset(root, ["TRAIN_000001"], f.common.EXPECTED_MEAN, f.common.EXPECTED_STD)[0]
        actual = f.Images(root, "test", ["TEST_000001"])[0]
        np.testing.assert_array_equal(actual.numpy(), original.numpy())
        self.assertEqual(tuple(actual.shape), (3,256,256))
        with self.assertRaisesRegex(ValueError, "IDs"):
            f.Images(root, "test", ["TRAIN_000001"])

    def test_cpu_feature_schema_and_train_anchor_rejection(self):
        root, _ = self.make_images()
        values = f.cpu_features(root, "test", self.plan["test"]["ids"])
        self.assertEqual(values.shape, (2,185))
        self.assertEqual(values.dtype, np.float64)
        self.assertTrue(f.require_anchor(values, values.copy(), f.HAND)["passed"])
        changed = values.copy()
        changed[0,0] += .001
        with self.assertRaisesRegex(ValueError, "anchor mismatch"):
            f.require_anchor(changed, values, f.HAND)

    def test_frozen_neural_paths_single_resnet_and_eight_dino_views(self):
        root, _ = self.make_images()
        dataset = f.Images(root, "test", self.plan["test"]["ids"])
        for name, model in ((f.RESNET, FakeResNet()), (f.DINO, FakeDino())):
            before = f.common.feature_state_hash(model)
            with contextlib.redirect_stdout(io.StringIO()):
                batched = f.neural_features(torch, model, name, dataset, torch.device("cpu"), 2)
            self.assertEqual(batched.shape, (2, f.WIDTHS[name]))
            self.assertEqual(before, f.common.feature_state_hash(model))
            self.assertFalse(model.training)
            if name == f.DINO:
                self.assertEqual(model.shapes, [(2,3,266,266)]*8)
            with contextlib.redirect_stdout(io.StringIO()):
                one = f.neural_features(torch, model, name, dataset, torch.device("cpu"), 1)
            np.testing.assert_allclose(batched, one, atol=1e-6, rtol=1e-6)

    def test_csv_preserves_order_and_does_not_clip_or_add_index(self):
        ids = self.plan["test"]["ids"]
        values = np.array([-123.45678901234567, 9876.543210987654])
        path = self.root/"submission.csv"
        path.write_text(f.submission_text(ids, values), encoding="utf-8", newline="")
        f.verify_submission(path, ids, values)
        self.assertEqual(path.read_text().splitlines()[0], "ID,hardness")
        with self.assertRaisesRegex(ValueError, "order"):
            f.verify_submission(path, ids[::-1], values)
        for bad in (np.array([np.nan, 1.]), np.ones(3)):
            with self.assertRaises(ValueError):
                f.submission_text(ids, bad)

    def test_fixed_weights_and_prediction_is_read_only_for_parameters(self):
        features = {n: self.data.features[n][:2] for n in f.COMPONENTS}
        before = {n: {k: v.copy() for k,v in h.items()} for n,h in self.heads.items()}
        values, components = f.blended_predictions(self.heads, features)
        np.testing.assert_array_equal(values, .25*components[f.HAND]+.25*components[f.RESNET]+.5*components[f.DINO])
        for n, h in self.heads.items():
            for k, v in h.items():
                np.testing.assert_array_equal(v, before[n][k])

    def test_predict_end_to_end_requires_matching_test_cache(self):
        root, _ = self.make_images()
        self.plan["test"] = f.test_inventory(root, 2)
        self.plan["train"]["metadata"] = {"resnet": {"feature_state_sha256": "a"*64}, "dino_global": {"feature_state_sha256": "b"*64}}
        f.write_json(self.folder/f.PLAN_NAME, self.plan)
        self.save_model()
        arrays = {"ids": np.asarray(self.plan["test"]["ids"]), **{n: self.data.features[n][:2] for n in f.COMPONENTS}}
        meta = {"kind": "frozen_test_features", "protocol_sha256": f.core.sha256(self.folder/f.PLAN_NAME),
            "heads_sha256": f.core.sha256(self.folder/"model/heads.npz"), "test_binding_sha256": f.digest(self.plan["test"]),
            "test_used_for_fit": False, "anchor_ids": self.plan["anchor_ids"],
            "feature_states": {f.RESNET: "a"*64, f.DINO: "b"*64},
            "anchor_checks": {n: {"passed": True, "atol": 1e-10 if n==f.HAND else 1e-5, "rtol": 0 if n==f.HAND else 1e-6} for n in f.COMPONENTS},
            "array_sha256": {k: f.common.sha256_array(v) for k,v in arrays.items()}}
        f.save_archive(self.folder/"test_features", arrays, meta)
        with mock.patch.object(f, "ROOT", self.root), mock.patch.object(f, "validate_plan", return_value=self.plan), \
             mock.patch.object(f.core, "make_estimator", side_effect=AssertionError("Prediction must not fit")), \
             mock.patch.object(f.common, "import_runtime", side_effect=AssertionError("Prediction must not use GPU")), contextlib.redirect_stdout(io.StringIO()):
            f.predict(self.folder, self.data)
            with self.assertRaisesRegex(ValueError, "already exists"):
                f.predict(self.folder, self.data)
        saved = json.loads((self.folder/"submission/validation.json").read_text())
        self.assertEqual(saved["rows"], 2)
        self.assertFalse(saved["uploaded"])
        self.assertTrue(saved["sample_order_matches"])
        changed = copy.deepcopy(self.plan)
        changed["test"]["ids"].reverse()
        with self.assertRaisesRegex(ValueError, "provenance"):
            f.load_test_features(self.folder, changed)

    def test_prepare_output_cannot_overlap_original_inputs(self):
        with self.assertRaisesRegex(ValueError, "protected"):
            f.local.protect_output(f.ROOT, f.protected_paths(self.data))

    def test_extraction_orchestration_checks_both_states_and_anchors_before_cache(self):
        root, _ = self.make_images()
        res, din = FakeResNet(), FakeDino()
        data = copy.deepcopy(self.data)
        data.metadata = {"resnet": {"feature_state_sha256": f.common.feature_state_hash(res)},
                         "dino_global": {"feature_state_sha256": f.common.feature_state_hash(din)}}
        self.plan["train"] = f.train_binding(data)
        self.plan["test"] = f.test_inventory(root, 2)
        f.write_json(self.folder/f.PLAN_NAME, self.plan)
        self.save_model()
        calls = []
        def cpu(data_dir, split, ids):
            calls.append((f.HAND, split, len(ids)))
            return data.features[f.HAND][:len(ids)].copy()
        def neural(torch_arg, model, name, dataset, device, batch):
            split = dataset.paths[0].parent.name
            calls.append((name, split, len(dataset)))
            return data.features[name][:len(dataset)].copy()
        args = f.parse_args(["extract", "--device", "cpu", "--output-dir", str(self.folder)])
        with mock.patch.object(f, "ROOT", self.root), mock.patch.object(f, "validate_plan", return_value=self.plan), \
             mock.patch.object(f, "checkpoint_references", return_value={}), \
             mock.patch.object(f.common, "import_runtime", return_value=(np, torch, None, None, {})), \
             mock.patch.object(f.common, "configure_runtime", return_value=(torch.device("cpu"), {})), \
             mock.patch.object(f.resnet, "build_model", return_value=res), \
             mock.patch.object(f.dino.base, "build_model", return_value=(din, {})), \
             mock.patch.object(f, "cpu_features", side_effect=cpu), mock.patch.object(f, "neural_features", side_effect=neural), \
             mock.patch.object(f.core, "make_estimator", side_effect=AssertionError("Test extraction must not fit")), \
             contextlib.redirect_stdout(io.StringIO()):
            f.extract(self.folder, data, args)
        self.assertEqual(calls, [(f.HAND,"train",16),(f.HAND,"test",2), (f.RESNET,"train",16),(f.RESNET,"test",2),
                                (f.DINO,"train",16),(f.DINO,"test",2)])
        loaded, meta = f.load_test_features(self.folder, self.plan)
        self.assertTrue(all(c["passed"] for c in meta["anchor_checks"].values()))
        self.assertFalse(meta["test_used_for_fit"])
        for n in f.COMPONENTS:
            np.testing.assert_array_equal(loaded[n], data.features[n][:2])

    def test_import_and_help_without_torch_runtime(self):
        result = subprocess.run([sys.executable, "-c", "import sys; from scripts import final_submission as f; assert 'torch' not in sys.modules; f.parse_args(['train'])"],
            cwd=f.ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
