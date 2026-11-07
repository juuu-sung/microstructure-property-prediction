"""Focused synthetic-input checks; no torch, real images, downloads, or training."""
import json
import contextlib
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import extract_frozen_resnet as extract


class ExtractionSafetyTests(unittest.TestCase):
    def test_cpu_or_unavailable_cuda_does_not_initialize_cudnn(self):
        for requested in ("cpu", "cuda"):
            with self.subTest(device=requested):
                torch = mock.Mock()
                torch.device.return_value = SimpleNamespace(type=requested)
                torch.cuda.is_available.return_value = False
                torch.backends.cudnn.version.side_effect = ValueError("min() iterable argument is empty")
                args = SimpleNamespace(seed=42, threads=1, device=requested)
                if requested == "cpu":
                    _, info = extract.configure_runtime(torch, mock.Mock(), args)
                    self.assertIsNone(info["cudnn_version"])
                else:
                    with self.assertRaisesRegex(RuntimeError, "CUDA is unavailable"):
                        extract.configure_runtime(torch, mock.Mock(), args)
                torch.backends.cudnn.version.assert_not_called()

    def make_data(self, root):
        (root / "train").mkdir()
        (root / "train.csv").write_text("ID,hardness\nTRAIN_000002,200\nTRAIN_000001,190\n", encoding="utf-8")
        for number in (1, 2):
            Image.new("L", (256, 256), color=number).save(root / "train" / f"TRAIN_{number:06d}.png")
        return extract.training_inventory(root, expected_count=2)

    def test_inventory_order_and_input_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inventory = self.make_data(root)
            self.assertEqual(inventory["ids"], ["TRAIN_000002", "TRAIN_000001"])
            extract.ensure_inputs_unchanged(root, inventory)
            Image.new("L", (256, 256), color=42).save(root / "train/TRAIN_000001.png")
            with self.assertRaisesRegex(RuntimeError, "changed during extraction"):
                extract.ensure_inputs_unchanged(root, inventory)

    def test_rejects_invalid_modes_and_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_data(root)
            Image.new("RGB", (256, 256)).save(root / "train/TRAIN_000001.png")
            with self.assertRaisesRegex(ValueError, "grayscale L PNG"):
                extract.training_inventory(root, expected_count=2)
            (root / "train.csv").write_text("ID,hardness\nTRAIN_000001,200\nTRAIN_000001,190\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unique"):
                extract.training_inventory(root, expected_count=2)

    def test_requires_competition_sample_count_and_valid_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_data(root)
            with self.assertRaisesRegex(ValueError, "Expected 500 training rows; found 2"):
                extract.training_inventory(root)
        for argv in (["--check-only", "--batch-size", "0"], ["--revision", "main"]):
            with mock.patch.object(extract, "import_runtime") as runtime, contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as result:
                    extract.main(argv)
                self.assertEqual(result.exception.code, 2)
                runtime.assert_not_called()

    def test_cache_integrity_configuration_and_interrupted_pair(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inventory = self.make_data(root)
            config = {"model_id": extract.HF_REPO, "revision": extract.DEFAULT_REVISION}
            final = np.zeros((2, 512), dtype=np.float32)
            multi = np.ones((2, 1792), dtype=np.float32)
            metadata = {"rows": 2, "extraction_config": config,
                        "train_csv_sha256": inventory["train_csv_sha256"],
                        "image_manifest_sha256": inventory["image_manifest_sha256"],
                        "feature_state_sha256": "a" * 64,
                        "feature_array_sha256": {"final_gap": extract.sha256_array(final),
                                                  "multiscale_mean_std": extract.sha256_array(multi)}}
            output = root / "output"
            extract.atomic_save(np, output, inventory["ids"], final, multi, metadata)
            self.assertEqual(extract.verify_cached(np, output, inventory, config), metadata)
            with self.assertRaisesRegex(ValueError, "configuration/runtime/script"):
                extract.verify_cached(np, output, inventory, {**config, "revision": "b" * 40})
            tampered = final.copy()
            tampered[0, 0] = 1
            extract.atomic_save(np, output, inventory["ids"], tampered, multi, metadata)
            with self.assertRaisesRegex(ValueError, "integrity hash"):
                extract.verify_cached(np, output, inventory, config)
            (output / "metadata.json").write_text(json.dumps({**metadata, "rows": 3}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "metadata.json differ"):
                extract.verify_cached(np, output, inventory, config)


if __name__ == "__main__":
    unittest.main()
