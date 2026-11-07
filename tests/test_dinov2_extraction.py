"""Synthetic CPU/token/cache checks; no pretrained model, GPU, or real data."""
import contextlib
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
import torch

from scripts import extract_frozen_dinov2 as extract


class TokenAndPaddingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_reflection_preserves_every_original_pixel(self):
        original = torch.arange(3 * 256 * 256, dtype=torch.float32).reshape(3, 256, 256)
        padded = extract.reflect_pad_input(torch, original)
        self.assertEqual(tuple(padded.shape), (3, 266, 266))
        torch.testing.assert_close(padded[:, 5:-5, 5:-5], original, rtol=0, atol=0)
        expected = np.pad(original.numpy(), ((0, 0), (5, 5), (5, 5)), mode="reflect")
        np.testing.assert_array_equal(padded.numpy(), expected)
        batched = extract.reflect_pad_input(torch, original.unsqueeze(0))
        torch.testing.assert_close(batched[0], padded, rtol=0, atol=0)

    def test_padding_rejects_wrong_size_channel_or_dtype(self):
        for sample in [torch.zeros(3, 224, 224), torch.zeros(1, 256, 256), torch.zeros(3, 256, 256, dtype=torch.float64)]:
            with self.subTest(shape=sample.shape, dtype=sample.dtype):
                with self.assertRaisesRegex(ValueError, "FP32"):
                    extract.reflect_pad_input(torch, sample)

    def test_dataset_normalizes_replicated_gray_without_resize(self):
        with tempfile.TemporaryDirectory(prefix="dino-synthetic-image-") as folder:
            root = Path(folder)
            (root / "train").mkdir()
            gray = np.tile(np.arange(256, dtype=np.uint8), (256, 1))
            Image.fromarray(gray).save(root / "train/TRAIN_000001.png")
            dataset = extract.TrainPaddedDataset(root, ["TRAIN_000001"])
            image = dataset[0]
            expected = np.repeat(gray[None].astype(np.float32) / np.float32(255), 3, axis=0)
            expected -= np.asarray(extract.common.EXPECTED_MEAN, dtype=np.float32)[:, None, None]
            expected /= np.asarray(extract.common.EXPECTED_STD, dtype=np.float32)[:, None, None]
            np.testing.assert_array_equal(image[:, 5:-5, 5:-5].numpy(), expected)
            self.assertEqual(image.dtype, torch.float32)

    def test_pooling_order_excludes_cls_from_patch_statistics(self):
        tokens = torch.zeros((2, 362, 384), dtype=torch.float32)
        tokens[:, 0] = torch.arange(384, dtype=torch.float32) + 1000
        spatial = torch.arange(361, dtype=torch.float32) - 180
        tokens[0, 1:] = spatial[:, None]
        tokens[1, 1:] = 2 * spatial[:, None] + 5
        cls, combined = extract.pool_tokens(torch, tokens)
        self.assertEqual(tuple(cls.shape), (2, 384))
        self.assertEqual(tuple(combined.shape), (2, 1152))
        torch.testing.assert_close(combined[:, :384], cls, rtol=0, atol=0)
        np.testing.assert_allclose(combined[0, 384:768].numpy(), 0, atol=1e-7)
        np.testing.assert_allclose(combined[1, 384:768].numpy(), 5, atol=1e-7)
        expected_std = np.arange(-180, 181, dtype=np.float64).std(ddof=0)
        np.testing.assert_allclose(combined[0, 768:].numpy(), expected_std, rtol=1e-6)
        np.testing.assert_allclose(combined[1, 768:].numpy(), expected_std * 2, rtol=1e-6)

    def test_pooling_rejects_register_tokens_wrong_dtype_and_nonfinite_values(self):
        cases = [torch.zeros(1, 366, 384), torch.zeros(1, 362, 768), torch.zeros(1, 362, 384, dtype=torch.float64)]
        for value in (float("nan"), float("inf")):
            tokens = torch.zeros(1, 362, 384)
            tokens[0, 100, 0] = value
            cases.append(tokens)
        for tokens in cases:
            with self.subTest(shape=tokens.shape, dtype=tokens.dtype):
                with self.assertRaises(RuntimeError):
                    extract.pool_tokens(torch, tokens)


class CacheAndCliTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="dino-cache-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.output = self.root / "output"
        self.ids = ["TRAIN_000002", "TRAIN_000001"]
        manifest = [{"id": ident, "relative_path": f"train/{ident}.png", "size_bytes": 123, "sha256": "b" * 64} for ident in self.ids]
        self.inventory = {"ids": self.ids, "train_csv_sha256": "a" * 64, "images": manifest,
                          "image_manifest_sha256": hashlib.sha256(extract.common.canonical_json(manifest).encode()).hexdigest()}
        self.args = extract.build_parser().parse_args(["--device", "cpu"])
        self.config = extract.extraction_config(self.args, {"timm": "synthetic-test"}, "cpu", {})
        self.cls = np.zeros((2, 384), dtype=np.float32)
        self.combined = np.ones((2, 1152), dtype=np.float32)
        self.metadata = {"schema_version": 1, "rows": 2, "extraction_config": self.config,
            "train_csv_sha256": self.inventory["train_csv_sha256"], "image_manifest_sha256": self.inventory["image_manifest_sha256"],
            "training_image_manifest": manifest, "feature_state_sha256": "c" * 64,
            "feature_array_sha256": {"cls_token": extract.common.sha256_array(self.cls), "cls_patch_mean_std": extract.common.sha256_array(self.combined)},
            "test_accessed": False, "labels_saved_in_npz": False}
        self.save()

    def save(self, ids=None, cls=None, combined=None, metadata=None):
        extract.atomic_save(np, self.output, self.ids if ids is None else ids,
            self.cls if cls is None else cls, self.combined if combined is None else combined,
            self.metadata if metadata is None else metadata)

    def verify(self, config=None, inventory=None):
        return extract.verify_cached(np, self.output, self.inventory if inventory is None else inventory,
                                      self.config if config is None else config)

    def test_cache_roundtrip_and_exact_schema(self):
        self.assertEqual(self.verify(), self.metadata)
        with np.load(self.output / "features.npz", allow_pickle=False) as archive:
            self.assertEqual(set(archive.files), extract.FEATURE_KEYS)
            self.assertEqual(archive["metadata_json"].shape, ())
            self.assertEqual(archive["ids"].dtype.kind, "U")
            self.assertEqual(archive["ids"].tolist(), self.ids)

    def test_configuration_source_hash_and_input_changes_are_rejected(self):
        for field in ("revision", "extractor_script_sha256", "common_utility_sha256"):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "configuration/runtime/source"):
                self.verify(config={**self.config, field: "changed"})
        with self.assertRaisesRegex(ValueError, "train_csv_sha256"):
            self.verify(inventory={**self.inventory, "train_csv_sha256": "d" * 64})

    def test_array_hash_nonfinite_dtype_and_id_order_are_rejected(self):
        changed = self.cls.copy()
        changed[0, 0] = 1
        self.save(cls=changed)
        with self.assertRaisesRegex(ValueError, "integrity hash"):
            self.verify()
        changed[0, 0] = np.nan
        self.save(cls=changed)
        with self.assertRaisesRegex(ValueError, "finite"):
            self.verify()
        self.save(cls=self.cls.astype(np.float64))
        with self.assertRaisesRegex(ValueError, "dtype"):
            self.verify()
        self.save(ids=self.ids[::-1])
        with self.assertRaisesRegex(ValueError, "IDs"):
            self.verify()

    def test_interrupted_metadata_pair_and_extra_labels_are_rejected(self):
        (self.output / "metadata.json").write_text(json.dumps({**self.metadata, "rows": 3}), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "metadata.json differ"):
            self.verify()
        self.save()
        np.savez(self.output / "features.npz", ids=np.array(self.ids), cls_token=self.cls,
                 cls_patch_mean_std=self.combined, metadata_json=np.array(json.dumps(self.metadata)), labels=np.zeros(2))
        with self.assertRaisesRegex(ValueError, "keys"):
            self.verify()

    def test_invalid_cli_rejected_before_runtime_or_data_access(self):
        for argv in (["--batch-size", "0"], ["--revision", "main"], ["--seed", "-1"], ["--check-only", "--reuse"]):
            with self.subTest(argv=argv), mock.patch.object(extract.common, "import_runtime") as runtime, mock.patch.object(extract.common, "training_inventory") as inventory:
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                    extract.main(argv)
                self.assertEqual(error.exception.code, 2)
                runtime.assert_not_called()
                inventory.assert_not_called()

    def test_existing_output_requires_explicit_overwrite_before_data_access(self):
        with mock.patch.object(extract.common, "import_runtime") as runtime, mock.patch.object(extract.common, "training_inventory") as inventory:
            with self.assertRaisesRegex(ValueError, "already exist"):
                extract.main(["--output-dir", str(self.output)])
            runtime.assert_not_called()
            inventory.assert_not_called()

    def test_reuse_loads_no_model_or_weights(self):
        with mock.patch.object(extract.common, "training_inventory", return_value=self.inventory), \
             mock.patch.object(extract.common, "import_runtime", return_value=(np, None, None, None, {"timm": "synthetic-test"})), \
             mock.patch.object(extract.common, "configure_runtime", return_value=("cpu", {})), \
             mock.patch.object(extract, "build_model") as builder, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(extract.main(["--reuse", "--device", "cpu", "--output-dir", str(self.output)]), 0)
            builder.assert_not_called()

    def test_validate_only_never_imports_runtime(self):
        with mock.patch.object(extract.common, "training_inventory", return_value=self.inventory), \
             mock.patch.object(extract.common, "import_runtime") as runtime, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(extract.main(["--validate-only"]), 0)
            runtime.assert_not_called()

    def test_module_import_does_not_import_torch(self):
        result = subprocess.run([sys.executable, "-c", "import sys; from scripts import extract_frozen_dinov2; assert 'torch' not in sys.modules"],
                                cwd=extract.ROOT, capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
