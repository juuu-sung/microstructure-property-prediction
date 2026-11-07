"""User-run, train-only frozen DINOv2-small features; no model fitting.

Original 256x256 grayscale images are repeated to RGB, ImageNet-normalized, and
reflect-padded by five pixels per side to 266x266. No resize, crop, or augmentation.
--validate-only does not import torch. --check-only uses random weights and a
synthetic input, without reading competition images or downloading anything.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sys
import time
import uuid

if __package__:
    from . import extract_frozen_resnet as common
else:
    import extract_frozen_resnet as common

ROOT = common.ROOT
MODEL_NAME = "vit_small_patch14_dinov2.lvd142m"
HF_REPO = "timm/vit_small_patch14_dinov2.lvd142m"
DEFAULT_REVISION = "4610ca143709d58a633b6397a74412c2c3842454"
SCHEMA_VERSION = 1
FEATURE_KEYS = {"ids", "cls_token", "cls_patch_mean_std", "metadata_json"}
FEATURE_DIMENSIONS = {"cls_token": 384, "cls_patch_mean_std": 1152}
PADDING = {"mode": "reflect", "left": 5, "right": 5, "top": 5, "bottom": 5}
POOLING = "normalized CLS; concat normalized CLS,patch mean,patch std; correction=0"


def reflect_pad_input(torch, inputs):
    if inputs.ndim not in (3, 4) or tuple(inputs.shape[-3:]) != (3, 256, 256) or inputs.dtype != torch.float32:
        raise ValueError("Expected FP32 CHW or NCHW image with trailing shape (3,256,256)")
    padded = torch.nn.functional.pad(inputs, (5, 5, 5, 5), mode="reflect")
    if tuple(padded.shape[-2:]) != (266, 266):
        raise RuntimeError("Unexpected padded image shape")
    return padded


class TrainPaddedDataset(common.TrainImageDataset):
    def __init__(self, data_dir, ids):
        super().__init__(data_dir, ids, common.EXPECTED_MEAN, common.EXPECTED_STD)
    def __getitem__(self, index):
        import torch
        return reflect_pad_input(torch, super().__getitem__(index))


def pool_tokens(torch, tokens):
    if not isinstance(tokens, torch.Tensor) or tokens.ndim != 3 or tuple(tokens.shape[1:]) != (362, 384):
        raise RuntimeError("Expected normalized DINOv2 tokens with shape (batch,362,384): one CLS and 361 patches")
    if tokens.dtype != torch.float32 or not torch.isfinite(tokens).all().item():
        raise RuntimeError("Tokens must be finite FP32")
    cls = tokens[:, 0, :]
    patches = tokens[:, 1:, :]
    combined = torch.cat([cls, patches.mean(dim=1), patches.std(dim=1, correction=0)], dim=1)
    if combined.shape[1] != 1152 or not torch.isfinite(combined).all().item():
        raise RuntimeError("Invalid pooled DINOv2 features")
    return cls, combined


def validate_model(torch, model, official):
    checks = {
        "embedding_dim": int(model.embed_dim) == 384,
        "prefix_tokens": int(model.num_prefix_tokens) == 1,
        "register_tokens": int(model.num_reg_tokens) == 0 and model.reg_token is None,
        "class_token": model.cls_token is not None and tuple(model.cls_token.shape) == (1, 1, 384),
        "patch_size": tuple(model.patch_embed.patch_size) == (14, 14),
        "stored_patch_grid": tuple(model.patch_embed.grid_size) == (37, 37),
        "stored_positional_embedding": model.pos_embed is not None and tuple(model.pos_embed.shape) == (1, 1370, 384),
        "dynamic_img_size": model.dynamic_img_size is True,
        "dynamic_img_pad": model.patch_embed.dynamic_img_pad is False,
        "non_strict_image_size": model.patch_embed.strict_img_size is False,
        "post_transformer_layernorm": isinstance(model.norm, torch.nn.LayerNorm),
        "normalization_mean": list(official["mean"]) == common.EXPECTED_MEAN,
        "normalization_std": list(official["std"]) == common.EXPECTED_STD,
        "official_input_size": list(official["input_size"]) == [3, 518, 518],
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"Unexpected DINOv2 configuration: {', '.join(failed)}")


def build_model(torch, timm, resolve_config, revision=DEFAULT_REVISION, pretrained=True):
    identifier = f"hf-hub:{HF_REPO}@{revision}" if pretrained else MODEL_NAME
    # Keep checkpoint positional parameters on the native 37x37 grid. timm
    # interpolates them to 19x19 during forward, without resizing the image.
    model = timm.create_model(identifier, pretrained=pretrained, num_classes=0,
                              img_size=518, dynamic_img_size=True, dynamic_img_pad=False)
    official = resolve_config(model)
    validate_model(torch, model, official)
    model.requires_grad_(False)
    model.eval()
    if any(module.training for module in model.modules()) or any(p.requires_grad for p in model.parameters()):
        raise RuntimeError("DINOv2 extractor was not fully frozen")
    return model, official


def extraction_config(args, versions, device, device_info):
    return {"schema_version": SCHEMA_VERSION, "model_id": HF_REPO, "revision": args.revision,
        "input_shape": [3, 266, 266], "original_image_shape": [1, 256, 256], "padding": dict(PADDING),
        "normalization_mean": common.EXPECTED_MEAN, "normalization_std": common.EXPECTED_STD,
        "grayscale_handling": "L -> float32/255 -> repeat RGB -> fixed ImageNet normalization -> reflect pad",
        "resize": None, "crop": None, "augmentation": None, "dtype": "float32", "amp": False,
        "dynamic_img_size": True, "dynamic_img_pad": False, "patch_size": [14, 14], "patch_grid": [19, 19],
        "num_prefix_tokens": 1, "num_register_tokens": 0, "embedding_dim": 384, "token_count": 362,
        "pooling": POOLING, "token_source": "model.forward_features, after final LayerNorm",
        "positional_embedding": {"stored_grid": [37, 37], "forward_grid": [19, 19],
            "implementation": "timm resample_abs_pos_embed via dynamic_img_size",
            "interpolation": "bicubic", "antialias": True, "stored_parameter_changed": False},
        "train_mode": False, "seed": args.seed, "batch_size": args.batch_size, "num_workers": args.num_workers,
        "threads": args.threads, "device": str(device), "gpu_name": device_info.get("gpu_name"),
        "deterministic_algorithms": True, "cudnn_benchmark": False, "tf32": False, "versions": versions,
        "extractor_script_sha256": common.sha256_file(Path(__file__)),
        "common_utility_sha256": common.sha256_file(Path(common.__file__))}


def verify_cached(np, output_dir, inventory, config):
    npz_path, json_path = output_dir / "features.npz", output_dir / "metadata.json"
    if not npz_path.is_file() or not json_path.is_file():
        raise ValueError("--reuse requires both features.npz and metadata.json")
    with np.load(npz_path, allow_pickle=False) as cached:
        if len(cached.files) != 4 or set(cached.files) != FEATURE_KEYS:
            raise ValueError("Cached NPZ has unexpected keys")
        raw_metadata = cached["metadata_json"]
        if raw_metadata.shape != () or raw_metadata.dtype.kind != "U":
            raise ValueError("Cached metadata_json must be a Unicode scalar")
        metadata = json.loads(raw_metadata.item())
        if not isinstance(metadata, dict) or metadata != json.loads(json_path.read_text(encoding="utf-8")):
            raise ValueError("Cached NPZ metadata and metadata.json differ")
        if metadata.get("extraction_config") != config:
            raise ValueError("Cached configuration/runtime/source hashes differ; use a new directory or --overwrite")
        if metadata.get("schema_version") != SCHEMA_VERSION or metadata.get("rows") != len(inventory["ids"]):
            raise ValueError("Cached schema version or row count differs")
        if metadata.get("test_accessed") is not False or metadata.get("labels_saved_in_npz") is not False:
            raise ValueError("Cached train-only/no-label provenance is missing")
        for key in ("train_csv_sha256", "image_manifest_sha256"):
            if metadata.get(key) != inventory[key]:
                raise ValueError(f"Cached {key} differs from current training data")
        if metadata.get("training_image_manifest") != inventory["images"]:
            raise ValueError("Cached image manifest differs from current training images")
        ids = cached["ids"]
        if ids.ndim != 1 or ids.dtype.kind != "U" or ids.tolist() != inventory["ids"] or len(set(ids)) != len(ids):
            raise ValueError("Cached IDs do not match unique train CSV order")
        if not re.fullmatch(r"[0-9a-f]{64}", metadata.get("feature_state_sha256", "")):
            raise ValueError("Cached frozen model-state hash is missing or invalid")
        for name, width in FEATURE_DIMENSIONS.items():
            values = cached[name]
            if values.dtype != np.float32 or values.shape != (len(ids), width) or not np.isfinite(values).all():
                raise ValueError(f"Cached {name} has invalid dtype, shape, or finite values")
            if common.sha256_array(values) != metadata.get("feature_array_sha256", {}).get(name):
                raise ValueError(f"Cached {name} failed its array integrity hash")
    return metadata


def atomic_save(np, output_dir, ids, cls_token, cls_patch_mean_std, metadata):
    output_dir.mkdir(parents=True, exist_ok=True)
    nonce = uuid.uuid4().hex
    npz_tmp = output_dir / f".features-{nonce}.npz.tmp"
    json_tmp = output_dir / f".metadata-{nonce}.json.tmp"
    try:
        with npz_tmp.open("wb") as handle:
            np.savez_compressed(handle, ids=np.asarray(ids, dtype=str), cls_token=cls_token,
                cls_patch_mean_std=cls_patch_mean_std, metadata_json=np.asarray(common.canonical_json(metadata)))
            handle.flush()
            os.fsync(handle.fileno())
        with json_tmp.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(npz_tmp, output_dir / "features.npz")
        os.replace(json_tmp, output_dir / "metadata.json")
    finally:
        for path in (npz_tmp, json_tmp):
            if path.exists():
                path.unlink()


def cached_checkpoint_files(revision):
    from huggingface_hub import try_to_load_from_cache
    candidates = []
    for name in ("config.json", "model.safetensors", "pytorch_model.bin"):
        path = try_to_load_from_cache(HF_REPO, name, revision=revision)
        if isinstance(path, str) and Path(path).is_file():
            candidates.append({"filename": name, "cache_path": str(Path(path).resolve()),
                               "sha256": common.sha256_file(path), "size_bytes": Path(path).stat().st_size})
    return candidates


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default="open", help="Absolute path or project-relative path")
    parser.add_argument("--output-dir", default="reports/frozen_dinov2_small")
    parser.add_argument("--revision", default=DEFAULT_REVISION, help="Full lowercase 40-character HF commit, no branches/tags")
    parser.add_argument("--device", default="cuda", help="cuda, cuda:N, or cpu; no automatic CPU fallback")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    checks = parser.add_mutually_exclusive_group()
    checks.add_argument("--validate-only", action="store_true", help="Read/hash train data only; no torch/GPU/download")
    checks.add_argument("--check-only", action="store_true", help="Random-weight synthetic forward only; no data/download")
    reuse = parser.add_mutually_exclusive_group()
    reuse.add_argument("--reuse", action="store_true", help="Verify exact configuration, source/input hashes, and feature cache")
    reuse.add_argument("--overwrite", action="store_true", help="Explicitly replace features.npz/metadata.json after success")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.num_workers < 0 or args.threads < 1 or not 0 <= args.seed < 2**32:
        parser.error("batch-size/threads must be positive, workers nonnegative, seed in [0,2**32)")
    if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        parser.error("revision must be a full lowercase 40-character commit hash")
    if (args.validate_only or args.check_only) and (args.reuse or args.overwrite):
        parser.error("Check modes cannot be combined with --reuse/--overwrite")
    data_dir, output_dir = common.project_path(args.data_dir), common.project_path(args.output_dir)
    if args.validate_only:
        inventory = common.training_inventory(data_dir)
        print(json.dumps({"status": "training_data_valid", "rows": len(inventory["ids"]),
            "train_csv_sha256": inventory["train_csv_sha256"], "image_manifest_sha256": inventory["image_manifest_sha256"],
            "test_accessed": False, "torch_imported": False}, indent=2))
        return 0
    inventory = None
    if not args.check_only:
        if output_dir.exists() and not output_dir.is_dir():
            raise ValueError("Output path must be a directory")
        if not args.reuse and not args.overwrite and any((output_dir / name).exists() for name in ("features.npz", "metadata.json")):
            raise ValueError("Output features already exist; use --reuse, a new directory, or explicit --overwrite")
        print("Validating train-only image inventory...", flush=True)
        inventory = common.training_inventory(data_dir)
    np, torch, timm, resolve_config, versions = common.import_runtime()
    device, device_info = common.configure_runtime(torch, np, args)
    config = extraction_config(args, versions, device, device_info)
    if args.reuse:
        metadata = verify_cached(np, output_dir, inventory, config)
        print(f"Verified cached DINOv2 features ({metadata['rows']} rows); no model was loaded or downloaded.")
        return 0
    print("Creating random-weight DINOv2 for synthetic check..." if args.check_only else f"Loading pinned pretrained DINOv2 {HF_REPO}@{args.revision} ...", flush=True)
    model, official = build_model(torch, timm, resolve_config, args.revision, pretrained=not args.check_only)
    model = model.to(device=device, dtype=torch.float32)
    state_before = common.feature_state_hash(model)
    if args.check_only:
        with torch.inference_mode():
            sample = reflect_pad_input(torch, torch.zeros((1, 3, 256, 256), device=device, dtype=torch.float32))
            tokens = model.forward_features(sample)
            cls, combined = pool_tokens(torch, tokens)
        if common.feature_state_hash(model) != state_before:
            raise RuntimeError("Frozen random model changed state during synthetic forward")
        print(json.dumps({"status": "synthetic_forward_ok", "padded_shape": list(sample.shape),
            "token_shape": list(tokens.shape), "cls_token_shape": list(cls.shape),
            "cls_patch_mean_std_shape": list(combined.shape), "pretrained": False, "data_accessed": False,
            "features_saved": False, "model_state_unchanged": True}, indent=2))
        return 0
    dataset = TrainPaddedDataset(data_dir, inventory["ids"])
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda", worker_init_fn=common.seed_worker,
        generator=torch.Generator().manual_seed(args.seed))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started, processed, cls_parts, combined_parts = time.perf_counter(), 0, [], []
    with torch.inference_mode():
        for step, inputs in enumerate(loader, start=1):
            inputs = inputs.to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda")
            tokens = model.forward_features(inputs)
            cls, combined = pool_tokens(torch, tokens)
            cls_parts.append(cls.cpu().numpy().copy())
            combined_parts.append(combined.cpu().numpy().copy())
            processed += len(inputs)
            if step == 1 or step % 5 == 0 or processed == len(dataset):
                print(f"Extracted {processed}/{len(dataset)} train images; elapsed={time.perf_counter()-started:.1f}s", flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    cls_array = np.concatenate(cls_parts).astype(np.float32, copy=False)
    combined_array = np.concatenate(combined_parts).astype(np.float32, copy=False)
    if cls_array.shape != (len(dataset), 384) or combined_array.shape != (len(dataset), 1152):
        raise RuntimeError("Unexpected final feature shapes")
    common.ensure_inputs_unchanged(data_dir, inventory)
    state_after = common.feature_state_hash(model)
    if state_before != state_after:
        raise RuntimeError("Model state changed during frozen extraction; no cache saved")
    metadata = {"schema_version": SCHEMA_VERSION, "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Train-only frozen DINOv2 tokens; no fitting, labels in NPZ, or test access", "rows": len(dataset),
        "extraction_config": config, "train_csv_sha256": inventory["train_csv_sha256"],
        "image_manifest_sha256": inventory["image_manifest_sha256"], "training_image_manifest": inventory["images"],
        "feature_state_sha256": state_after,
        "feature_array_sha256": {"cls_token": common.sha256_array(cls_array), "cls_patch_mean_std": common.sha256_array(combined_array)},
        "official_model_data_config": official, "official_resize_crop_applied": False,
        "checkpoint": {"repository": HF_REPO, "revision": args.revision,
            "model_card_url": f"https://huggingface.co/{HF_REPO}/blob/{args.revision}/README.md",
            "model_card_license": "apache-2.0" if args.revision == DEFAULT_REVISION else "verify override revision",
            "upstream_license_url": "https://github.com/facebookresearch/dinov2#license",
            "pretrained_config_license": model.pretrained_cfg.get("license"),
            "pretrained_config": dict(model.pretrained_cfg),
            "license_provenance_note": "The pinned model card/upstream state Apache-2.0; an archived HF config may contain a legacy cc-by-nc-4.0 field. Preserve both source values.",
            "cached_candidate_files": cached_checkpoint_files(args.revision),
            "identity_note": "feature_state_sha256 hashes actual loaded model state; cache files are provenance candidates, not asserted load traces."},
        "runtime": device_info, "elapsed_extraction_seconds": elapsed,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None,
        "test_accessed": False, "labels_saved_in_npz": False}
    atomic_save(np, output_dir, inventory["ids"], cls_array, combined_array, metadata)
    print(f"Saved {output_dir / 'features.npz'} and metadata.json; shapes={cls_array.shape}/{combined_array.shape}; seconds={elapsed:.1f}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted. No complete new cache is claimed; --reuse verifies any existing artifacts.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
