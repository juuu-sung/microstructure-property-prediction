"""Tiny synthetic CPU checks; never read competition images or download weights."""
import copy
import csv
import json
from pathlib import Path
import tempfile
import contextlib
import io
import unittest
from unittest import mock

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from scripts import train_partial_resnet as trainer


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 4, 1, bias=False), nn.BatchNorm2d(4), nn.ReLU())
        self.layer2 = nn.Sequential(nn.Conv2d(4, 4, 1, bias=False), nn.BatchNorm2d(4), nn.ReLU())
        self.layer3 = nn.Sequential(nn.Conv2d(4, 4, 1, bias=False), nn.BatchNorm2d(4), nn.ReLU())
        self.layer4 = nn.Sequential(nn.Conv2d(4, 4, 1, bias=False), nn.BatchNorm2d(4), nn.ReLU())

    def forward(self, images):
        second = self.layer2(self.stem(images))
        third = self.layer3(second)
        return second, third, self.layer4(third)


class TinyRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = TinyBackbone()
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(24, 1))
        self._finetune = False

    def forward(self, images):
        pieces = []
        for feature in self.backbone(images):
            pieces.extend([feature.mean((2, 3)), feature.std((2, 3), correction=0)])
        return self.head(torch.cat(pieces, dim=1)).flatten()


class ScalarRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(1, 1, bias=False)
        self.head = nn.Linear(1, 1, bias=False)
        self._finetune = False
        with torch.no_grad():
            self.backbone.weight.fill_(1.0)
            self.head.weight.fill_(0.1)
        self.backbone.requires_grad_(False)

    def forward(self, value):
        return self.head(self.backbone(value)).flatten()


class PartialTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(71)
        self.device = torch.device("cpu")

    def test_accumulation_groups_use_actual_sample_count(self):
        self.assertEqual(trainer.accumulation_group_samples([4, 4, 2], 2), [8, 8, 2])
        self.assertEqual(trainer.accumulation_group_samples([4, 4, 2], 4), [10, 10, 10])
        self.assertEqual(trainer.accumulation_group_samples([4, 4, 2], 1), [4, 4, 2])

    def test_accumulation_matches_effective_batches_including_short_final_batch(self):
        x = torch.tensor([[0.1], [0.2], [0.4], [0.8], [0.3], [0.5], [0.9]])
        y = torch.tensor([0.3, -0.1, 0.2, 0.1, -0.2, 0.2, -0.3])
        actual = ScalarRegressor()
        reference = copy.deepcopy(actual)
        optimizer = torch.optim.SGD(actual.head.parameters(), lr=0.2)
        reference_optimizer = torch.optim.SGD(reference.head.parameters(), lr=0.2)
        loader = DataLoader(TensorDataset(x, y), batch_size=2, shuffle=False)
        result = trainer.train_epoch(torch, actual, loader, optimizer, self.device, 0.0, 1.0, 2, amp=False)
        for start, end in ((0, 4), (4, 7)):
            reference_optimizer.zero_grad()
            ((reference(x[start:end]) - y[start:end]) ** 2).mean().backward()
            reference_optimizer.step()
        torch.testing.assert_close(actual.head.weight, reference.head.weight, rtol=1e-6, atol=1e-7)
        self.assertEqual(result["optimizer_steps"], 2)

    def test_warmup_and_partial_finetune_preserve_frozen_weights_and_bn(self):
        model = TinyRegressor()
        images = torch.randn(6, 3, 4, 4)
        targets = torch.linspace(170, 230, 6)
        loader = DataLoader(TensorDataset(images, targets), batch_size=2, shuffle=False)
        trainer.configure_trainable(torch, model, finetune=False)
        self.assertTrue(all(not p.requires_grad for p in model.backbone.parameters()))
        self.assertTrue(all(p.requires_grad for p in model.head.parameters()))
        before = {name: value.detach().clone() for name, value in model.state_dict().items()}
        optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.05)
        trainer.train_epoch(torch, model, loader, optimizer, self.device, 200.0, 20.0, 2, amp=False)
        for name, value in model.backbone.state_dict().items():
            torch.testing.assert_close(value, before["backbone." + name], rtol=0, atol=0)
        self.assertTrue(any(not torch.equal(value, before["head." + name]) for name, value in model.head.state_dict().items()))

        trainer.configure_trainable(torch, model, finetune=True)
        self.assertTrue(any(p.requires_grad for p in model.backbone.layer4.parameters()))
        for name, parameter in model.backbone.named_parameters():
            if not name.startswith("layer4."):
                self.assertFalse(parameter.requires_grad, name)
        for module in model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                self.assertTrue(all(not p.requires_grad for p in module.parameters()))
        before = {name: value.detach().clone() for name, value in model.state_dict().items()}
        optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.05)
        trainer.train_epoch(torch, model, loader, optimizer, self.device, 200.0, 20.0, 2, amp=False)
        for name, value in model.state_dict().items():
            if name.startswith("backbone.") and not name.startswith("backbone.layer4."):
                torch.testing.assert_close(value, before[name], rtol=0, atol=0)
            if "running_" in name or "num_batches_tracked" in name:
                torch.testing.assert_close(value, before[name], rtol=0, atol=0)
        self.assertTrue(any(not torch.equal(value, before["backbone.layer4." + name])
                            for name, value in model.backbone.layer4.state_dict().items()))
        self.assertTrue(all(not module.training for module in model.modules()
                            if isinstance(module, nn.modules.batchnorm._BatchNorm)))

    def test_training_uses_fold_target_scale_and_fp32_loss(self):
        x = torch.tensor([[0.2], [0.4], [0.8]], dtype=torch.float32)
        y = torch.tensor([170.0, 210.0, 220.0], dtype=torch.float64)
        actual = ScalarRegressor()
        reference = copy.deepcopy(actual)
        optimizer = torch.optim.SGD(actual.head.parameters(), lr=0.2)
        reference_optimizer = torch.optim.SGD(reference.head.parameters(), lr=0.2)
        loader = DataLoader(TensorDataset(x, y), batch_size=3, shuffle=False)
        trainer.train_epoch(torch, actual, loader, optimizer, self.device, 200.0, 17.0, 1, amp=False)
        reference_optimizer.zero_grad()
        normalized = (y.to(torch.float32) - 200.0) / 17.0
        ((reference(x) - normalized) ** 2).mean().backward()
        reference_optimizer.step()
        self.assertEqual(actual.head.weight.dtype, torch.float32)
        torch.testing.assert_close(actual.head.weight, reference.head.weight, rtol=1e-6, atol=1e-7)

    def test_evaluation_restores_original_units_and_uses_eval_mode(self):
        model = TinyRegressor()
        trainer.configure_trainable(torch, model, finetune=False)
        images = torch.randn(5, 3, 4, 4)
        targets = torch.linspace(180, 220, 5, dtype=torch.float64)
        loader = DataLoader(TensorDataset(images, targets), batch_size=2, shuffle=False)
        model.eval()
        with torch.no_grad():
            expected = model(images).numpy().astype(np.float64) * 17.0 + 200.0
        model.train()
        prediction, rmse = trainer.evaluate(torch, model, loader, self.device, 200.0, 17.0)
        np.testing.assert_allclose(prediction, expected, atol=1e-5, rtol=1e-7)
        self.assertAlmostEqual(rmse, np.sqrt(np.mean((prediction - targets.numpy()) ** 2)))
        self.assertEqual(prediction.dtype, np.float64)
        self.assertTrue(all(not module.training for module in model.modules()))
        second, _ = trainer.evaluate(torch, model, loader, self.device, 200.0, 17.0)
        np.testing.assert_array_equal(prediction, second)

    def test_checkpoint_payload_is_detached_from_later_updates(self):
        model = TinyRegressor()
        payload = trainer.checkpoint_payload(torch, model, 200.0, 17.0, 3, 0,
                                             ["TRAIN_000001"], ["TRAIN_000002"], {})
        snapshot = {name: value.clone() for name, value in payload["model_state"].items()}
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(1.0)
        for name, value in payload["model_state"].items():
            self.assertEqual(value.device.type, "cpu")
            torch.testing.assert_close(value, snapshot[name], rtol=0, atol=0)
        with tempfile.TemporaryDirectory(prefix="partial-checkpoint-test-") as folder:
            path = Path(folder) / "best.pt"
            trainer.save_checkpoint(torch, path, payload)
            restored = torch.load(path, map_location="cpu", weights_only=True)
            for name, value in restored["model_state"].items():
                torch.testing.assert_close(value, snapshot[name], rtol=0, atol=0)

    def test_actual_timm_architecture_layer4_names_without_weights_or_forward(self):
        import timm
        from timm.data import resolve_model_data_config
        with mock.patch.object(timm, "create_model", wraps=timm.create_model) as create, \
             mock.patch("timm.models._hub.hf_hub_download", side_effect=AssertionError("Tests must not download")):
            model = trainer.build_model(torch, timm, resolve_model_data_config, pretrained=False)
        create.assert_called_once_with(trainer.extraction.MODEL_NAME, pretrained=False,
                                       features_only=True, out_indices=(2, 3, 4))
        self.assertEqual(model.head.in_features, 1792)
        self.assertEqual(model.backbone.feature_info.channels(), [128, 256, 512])
        self.assertEqual({name for name, parameter in model.named_parameters() if parameter.requires_grad},
                         {"head.weight", "head.bias"})
        trainer.configure_trainable(torch, model, finetune=True)
        trainable = [name for name, parameter in model.backbone.named_parameters() if parameter.requires_grad]
        self.assertEqual(len(trainable), 5)
        self.assertTrue(all(trainer.is_layer4(name) for name in trainable))
        self.assertTrue(all(not module.training and not any(p.requires_grad for p in module.parameters())
                            for module in model.modules() if isinstance(module, nn.modules.batchnorm._BatchNorm)))


class FoldValidationTests(unittest.TestCase):
    def setUp(self):
        self.ids = [f"TRAIN_{index + 1:06d}" for index in range(500)]
        self.rows = [{"ID": ident, "fold": str(index % 5), "seed": "42"}
                     for index, ident in enumerate(self.ids)]

    def test_folds_align_by_id_and_each_split_is_disjoint(self):
        folds = np.asarray(trainer.validate_fold_rows(self.ids, self.rows[::-1], 42))
        np.testing.assert_array_equal(folds, np.arange(500) % 5)
        for fold in range(5):
            train_ids = {ident for ident, value in zip(self.ids, folds) if value != fold}
            valid_ids = {ident for ident, value in zip(self.ids, folds) if value == fold}
            self.assertEqual(len(train_ids), 400)
            self.assertEqual(len(valid_ids), 100)
            self.assertFalse(train_ids & valid_ids)
            self.assertEqual(train_ids | valid_ids, set(self.ids))

    def test_bad_fold_artifacts_are_rejected(self):
        cases = []
        duplicate = copy.deepcopy(self.rows)
        duplicate[-1]["ID"] = duplicate[0]["ID"]
        cases.append(duplicate)
        extra = copy.deepcopy(self.rows)
        extra[-1]["ID"] = "TEST_000001"
        cases.append(extra)
        seed = copy.deepcopy(self.rows)
        seed[0]["seed"] = "43"
        cases.append(seed)
        imbalance = copy.deepcopy(self.rows)
        imbalance[0]["fold"] = "1"
        cases.append(imbalance)
        label = copy.deepcopy(self.rows)
        label[0]["fold"] = "5"
        cases.append(label)
        cases.append(self.rows[:-1])
        for rows in cases:
            with self.subTest(first=rows[0], last=rows[-1], count=len(rows)):
                with self.assertRaises(ValueError):
                    trainer.validate_fold_rows(self.ids, rows, 42)


class MonitoringAndAggregationTests(unittest.TestCase):
    def test_run_fold_restores_best_checkpoint_instead_of_last_epoch(self):
        args = trainer.build_parser().parse_args([
            "--device", "cpu", "--warmup-epochs", "1", "--epochs", "2",
            "--patience", "10", "--batch-size", "2", "--threads", "1",
        ])
        ids = [f"TRAIN_{index + 1:06d}" for index in range(10)]
        targets = [float(index + 170) for index in range(10)]
        folds = [index % 5 for index in range(10)]
        counter, evaluated_weights = [0], []

        def synthetic_dataset(_root, _ids, values):
            return TensorDataset(torch.zeros(len(values), 3, 4, 4), torch.tensor(values, dtype=torch.float64))

        def synthetic_model(*_args, **_kwargs):
            model = TinyRegressor()
            trainer.configure_trainable(torch, model, finetune=False)
            return model

        def fake_train(_torch, model, _loader, optimizer, *_args, **_kwargs):
            counter[0] += 1
            with torch.no_grad():
                model.head[-1].weight.fill_(counter[0])
            optimizer.step()
            return {"std_target_mse": 0.5, "optimizer_steps": 1}

        def fake_evaluate(_torch, model, loader, *_args):
            weight = int(model.head[-1].weight[0, 0].detach().item())
            evaluated_weights.append(weight)
            delta = {1: 3.0, 2: 1.0, 3: 2.0}[weight]
            return loader.dataset.tensors[1].numpy() + delta, delta

        with tempfile.TemporaryDirectory(prefix="partial-best-checkpoint-test-") as folder:
            root = Path(folder)
            with mock.patch.object(trainer, "LabeledTrainDataset", side_effect=synthetic_dataset), \
                 mock.patch.object(trainer, "build_model", side_effect=synthetic_model), \
                 mock.patch.object(trainer, "train_epoch", side_effect=fake_train), \
                 mock.patch.object(trainer, "evaluate", side_effect=fake_evaluate), \
                 mock.patch.object(trainer.extraction, "feature_state_hash", return_value="tiny-state"), \
                 mock.patch.object(trainer.extraction, "ensure_inputs_unchanged") as unchanged, \
                 contextlib.redirect_stdout(io.StringIO()):
                trainer.run_fold(np, torch, None, None, args, root, root, {"ids": ids}, targets, folds,
                                 0, {"config_hash": "synthetic-config"}, torch.device("cpu"))
            unchanged.assert_called_once()
            self.assertEqual(evaluated_weights, [1, 2, 3, 2, 2])
            checkpoint = torch.load(root / "fold0/best.pt", map_location="cpu", weights_only=True)
            self.assertEqual(checkpoint["epoch"], 2)
            self.assertEqual(checkpoint["val_rmse"], 1.0)
            self.assertFalse(set(checkpoint["train_ids"]) & set(checkpoint["validation_ids"]))
            self.assertEqual(set(checkpoint["validation_ids"]), {ids[0], ids[5]})
            self.assertAlmostEqual(checkpoint["target_mean"], np.mean([y for y, f in zip(targets, folds) if f != 0]))
            self.assertAlmostEqual(checkpoint["target_std"], np.std([y for y, f in zip(targets, folds) if f != 0]))
            metrics = json.loads((root / "fold0/metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(metrics["best_epoch"], 2)
            self.assertEqual(metrics["val_rmse"], 1.0)
            with (root / "fold0/oof_fold.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertTrue(all(float(row["prediction"]) - float(row["hardness"]) == 1.0 for row in rows))

    def test_warmup_does_not_consume_patience_and_finetune_resets_phase_counter(self):
        state = trainer.update_monitor(None, 10.0, 1, "warmup", 2)
        for epoch in range(2, 6):
            state = trainer.update_monitor(state, 11.0, epoch, "warmup", 2)
            self.assertFalse(state["stop"])
            self.assertEqual(state["bad_finetune_epochs"], 0)
        state["bad_finetune_epochs"] = 99
        state = trainer.update_monitor(state, 12.0, 6, "finetune", 2)
        self.assertEqual(state["bad_finetune_epochs"], 1)
        self.assertFalse(state["stop"])
        state = trainer.update_monitor(state, 11.0, 7, "finetune", 2)
        self.assertTrue(state["stop"])
        self.assertEqual(state["best_epoch"], 1)
        self.assertEqual(state["best_phase"], "warmup")
        self.assertEqual(state["best_rmse"], 10.0)
        state = trainer.update_monitor(state, 9.0, 8, "finetune", 2)
        self.assertTrue(state["improved"])
        self.assertFalse(state["stop"])
        self.assertEqual(state["bad_finetune_epochs"], 0)
        self.assertEqual(state["best_epoch"], 8)
        self.assertEqual(state["best_phase"], "finetune")

    def make_completed_fold(self, root, fold):
        folder = root / f"fold{fold}"
        folder.mkdir()
        trainer.write_json(folder / "status.json", {"status": "complete", "config_hash": "synthetic-config"})
        trainer.write_json(folder / "metrics.json", {
            "fold": fold, "config_hash": "synthetic-config", "pretrained_state_sha256": "same-tiny-backbone",
            "best_epoch": 2, "best_phase": "finetune", "val_rmse": float(fold + 1),
            "train_rmse_at_best_checkpoint": 0.5,
        })
        trainer.write_csv(folder / "oof_fold.csv", [
            {"ID": f"TRAIN_{index + 1:06d}", "hardness": float(index + 170),
             "fold": fold, "prediction": float(index + 170 + fold + 1)}
            for index in range(10) if index % 5 == fold
        ])

    def aggregate(self, root):
        trainer.aggregate_results(root, [f"TRAIN_{index + 1:06d}" for index in range(10)],
                                  [float(index + 170) for index in range(10)],
                                  [index % 5 for index in range(10)], {"config_hash": "synthetic-config"})
        return json.loads((root / "summary.json").read_text(encoding="utf-8"))

    def test_partial_coverage_never_claims_full_oof(self):
        with tempfile.TemporaryDirectory(prefix="partial-aggregate-test-") as folder:
            root = Path(folder)
            self.make_completed_fold(root, 0)
            running = root / "fold1"
            running.mkdir()
            trainer.write_json(running / "status.json", {"status": "running", "config_hash": "synthetic-config"})
            summary = self.aggregate(root)
            self.assertEqual(summary["status"], "partial_development")
            self.assertEqual(summary["coverage_n"], 2)
            self.assertEqual(summary["total_train_n"], 10)
            self.assertEqual(summary["completed_folds"], [0])
            self.assertIsNone(summary["development_oof_rmse"])
            self.assertAlmostEqual(summary["partial_heldout_rmse"], 1.0)
            with (root / "oof.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([r["ID"] for r in rows], ["TRAIN_000001", "TRAIN_000006"])

    def test_invalidated_completed_folds_replace_stale_full_score(self):
        with tempfile.TemporaryDirectory(prefix="partial-stale-results-test-") as folder:
            root = Path(folder)
            for fold in range(5):
                self.make_completed_fold(root, fold)
            self.assertEqual(self.aggregate(root)["status"], "complete_5fold_development")
            for fold in range(5):
                trainer.write_json(root / f"fold{fold}/status.json",
                                   {"status": "pending", "config_hash": "synthetic-config"})
            summary = self.aggregate(root)
            self.assertEqual(summary["status"], "no_completed_folds")
            self.assertEqual(summary["coverage_n"], 0)
            self.assertEqual(summary["completed_folds"], [])
            self.assertIsNone(summary["development_oof_rmse"])
            self.assertIsNone(summary["partial_heldout_rmse"])
            with (root / "oof.csv").open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(list(reader), [])
                self.assertEqual(reader.fieldnames, ["ID", "hardness", "fold", "prediction"])
            self.assertNotIn("**3.316625**", (root / "report.md").read_text(encoding="utf-8"))

    def test_all_completed_folds_produce_pooled_original_unit_rmse(self):
        with tempfile.TemporaryDirectory(prefix="partial-full-aggregate-test-") as folder:
            root = Path(folder)
            for fold in range(5):
                self.make_completed_fold(root, fold)
            summary = self.aggregate(root)
            self.assertEqual(summary["status"], "complete_5fold_development")
            self.assertEqual(summary["coverage_n"], 10)
            self.assertIsNone(summary["partial_heldout_rmse"])
            self.assertAlmostEqual(summary["development_oof_rmse"], np.sqrt(11.0))
            self.assertIn("not unbiased nested CV", summary["evaluation_note"])

    def test_aggregation_rejects_wrong_fold_or_target_or_duplicate_ids(self):
        for column, replacement in (("fold", "1"), ("hardness", "999"), ("ID", "TRAIN_000006")):
            with self.subTest(column=column), tempfile.TemporaryDirectory(prefix="partial-bad-oof-test-") as folder:
                root = Path(folder)
                self.make_completed_fold(root, 0)
                path = root / "fold0/oof_fold.csv"
                with path.open(newline="", encoding="utf-8") as handle:
                    rows = list(csv.DictReader(handle))
                rows[0][column] = replacement
                trainer.write_csv(path, rows)
                with self.assertRaises(ValueError):
                    self.aggregate(root)

    def test_aggregation_rejects_mixed_configs_or_pretrained_states(self):
        for key, replacement in (("config_hash", "other-config"), ("pretrained_state_sha256", "other-state")):
            with self.subTest(key=key), tempfile.TemporaryDirectory(prefix="partial-mixed-config-test-") as folder:
                root = Path(folder)
                for fold in (0, 1):
                    self.make_completed_fold(root, fold)
                path = root / "fold1/metrics.json"
                metrics = json.loads(path.read_text(encoding="utf-8"))
                metrics[key] = replacement
                trainer.write_json(path, metrics)
                with self.assertRaises(ValueError):
                    self.aggregate(root)


if __name__ == "__main__":
    unittest.main()
