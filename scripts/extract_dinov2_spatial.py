"""User-run frozen DINOv2 2x2 patch pooling; cached weights only, train only.

--validate-only checks original cache, image inventory and checkpoint files without
torch/GPU. --check-only --device cpu uses random weights and synthetic input only.
Stores regional features separately; existing global features are never replaced.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import uuid

if __package__:
    from . import extract_frozen_dinov2 as base
    from . import evaluate_frozen_dinov2 as dino
else:
    import extract_frozen_dinov2 as base
    import evaluate_frozen_dinov2 as dino

common = base.common
ROOT = common.ROOT
LOCAL_NAME = "patch_2x2_mean_std"
LOCAL_DIM = 3072
POOLING = "row-major 2x2 disjoint regions; each region mean then population std; CLS excluded"
PROTOCOL = {"patch_grid": [19, 19], "row_edges": [0, 9, 19], "column_edges": [0, 9, 19],
    "region_order": ["top_left", "top_right", "bottom_left", "bottom_right"],
    "region_token_counts": [81, 90, 90, 100], "within_region": ["mean", "std"],
    "std_correction": 0, "embedding_dim": 384, "feature_dimension": LOCAL_DIM,
    "disjoint": True, "cls_included": False, "token_order": "row-major after CLS",
    "pooling": POOLING}
MATCH_ATOL, MATCH_RTOL = 1e-5, 1e-6


def pool_regions(torch, tokens):
    # The shared check also returns a global anchor for the same forward pass.
    _, global_features = base.pool_tokens(torch, tokens)
    grid = tokens[:, 1:].reshape(tokens.shape[0], 19, 19, 384)
    parts = []
    for r0, r1 in ((0, 9), (9, 19)):
        for c0, c1 in ((0, 9), (9, 19)):
            region = grid[:, r0:r1, c0:c1].reshape(tokens.shape[0], -1, 384)
            parts.extend((region.mean(dim=1), region.std(dim=1, correction=0)))
    regional = torch.cat(parts, dim=1)
    if regional.shape != (tokens.shape[0], LOCAL_DIM) or not torch.isfinite(regional).all().item():
        raise RuntimeError("Invalid 2x2 pooled features")
    return global_features, regional


def read_global_cache(np, path, train_csv, *, expected_n=500):
    """Validate original feature provenance, returning train.csv order; no torch."""
    core = dino.core
    ids = np.asarray([r["ID"] for r in core.read_rows(train_csv, {"ID", "hardness"})])
    if len(ids) != expected_n or any(not ident.startswith("TRAIN_") for ident in ids):
        raise ValueError("Unexpected training row count or ID namespace")
    with np.load(path, allow_pickle=False) as archive:
        if len(archive.files) != 4 or set(archive.files) != base.FEATURE_KEYS:
            raise ValueError("Original global cache has unexpected keys")
        cached_ids = archive["ids"]
        if cached_ids.dtype.kind != "U" or cached_ids.shape != ids.shape or len(set(cached_ids)) != len(ids) or set(cached_ids) != set(ids):
            raise ValueError("Original global cache IDs differ from train.csv")
        for name, width in base.FEATURE_DIMENSIONS.items():
            values = archive[name]
            if values.shape != (len(ids), width) or values.dtype != np.float32 or not np.isfinite(values).all():
                raise ValueError("Invalid original global feature array")
        raw = archive["metadata_json"]
        if raw.shape != () or raw.dtype.kind != "U":
            raise ValueError("Original metadata must be a Unicode scalar")
        metadata = json.loads(raw.item())
        lookup = {ident: i for i, ident in enumerate(cached_ids)}
        features = archive["cls_patch_mean_std"][[lookup[i] for i in ids]].copy()
    holder = SimpleNamespace(ids=ids, metadata=metadata,
        provenance={"train_csv": {"sha256": core.sha256(train_csv)}})
    dino.validate_feature_metadata(holder, path, path.with_name("metadata.json"))
    return ids, features, metadata


def global_reference(path):
    return {"path": str(path.resolve()), "sha256": common.sha256_file(path),
        "metadata_path": str(path.with_name("metadata.json").resolve()),
        "metadata_sha256": common.sha256_file(path.with_name("metadata.json"))}


def check_inventory(inventory, original):
    for key in ("train_csv_sha256", "image_manifest_sha256"):
        if inventory[key] != original[key]:
            raise ValueError(f"Current training images/CSV differ from original DINO cache: {key}")


def check_global_match(np, current, previous):
    if current.shape != previous.shape or not np.isfinite(current).all() or not np.allclose(current, previous, atol=MATCH_ATOL, rtol=MATCH_RTOL):
        raise ValueError("Re-extracted global features differ from original DINO cache; no spatial cache saved")
    return float(np.max(np.abs(current - previous)))


def validate_spatial_cache(np, path, ids, original_metadata, original_path):
    """Read-only cache validator shared with the evaluator; no image/model access."""
    with np.load(path, allow_pickle=False) as archive:
        if len(archive.files) != 3 or set(archive.files) != {"ids", LOCAL_NAME, "metadata_json"}:
            raise ValueError("Spatial NPZ must contain ids, regional features and metadata only")
        cached_ids = archive["ids"]
        values, raw = archive[LOCAL_NAME], archive["metadata_json"]
        if cached_ids.dtype.kind != "U" or cached_ids.shape != ids.shape or len(set(cached_ids)) != len(ids) or set(cached_ids) != set(ids):
            raise ValueError("Spatial cache IDs differ")
        if values.dtype != np.float32 or values.shape != (len(ids), LOCAL_DIM) or not np.isfinite(values).all():
            raise ValueError("Spatial features must be finite FP32 with width 3072")
        if np.any(values.reshape(len(ids), 4, 2, 384)[:, :, 1] < 0):
            raise ValueError("Regional standard deviations must be nonnegative")
        if raw.shape != () or raw.dtype.kind != "U":
            raise ValueError("Spatial metadata must be a Unicode scalar")
        metadata = json.loads(raw.item())
        if metadata != json.loads(path.with_name("metadata.json").read_text(encoding="utf-8")):
            raise ValueError("Spatial metadata sidecar mismatch")
        if (metadata.get("schema_version") != 1 or metadata.get("rows") != len(ids) or
                metadata.get("spatial_protocol") != PROTOCOL or metadata.get("test_accessed") is not False or
                metadata.get("labels_saved_in_npz") is not False):
            raise ValueError("Spatial cache protocol or train-only scope mismatch")
        for key in ("feature_state_sha256", "train_csv_sha256", "image_manifest_sha256"):
            if metadata.get(key) != original_metadata[key]:
                raise ValueError(f"Spatial/global source mismatch: {key}")
        for key in ("sha256", "metadata_sha256"):
            if metadata.get("global_reference", {}).get(key) != global_reference(original_path)[key]:
                raise ValueError("Spatial cache references a different original global cache")
        config = metadata.get("extraction_config", {})
        for key, expected in {**dino.EXPECTED_CONFIG, "pooling": POOLING}.items():
            if config.get(key) != expected:
                raise ValueError(f"Spatial extraction configuration mismatch: {key}")
        for key in ("extractor_script_sha256", "base_extractor_script_sha256", "common_utility_sha256"):
            dino.require_sha256(config.get(key), key)
        check = metadata.get("global_reextraction_check", {})
        maximum = check.get("max_absolute_difference")
        if (check.get("passed") is not True or check.get("atol") != MATCH_ATOL or check.get("rtol") != MATCH_RTOL
                or not isinstance(maximum, (float, int)) or not np.isfinite(maximum) or maximum < 0):
            raise ValueError("Missing global re-extraction verification")
        if metadata.get("feature_array_sha256") != {LOCAL_NAME: common.sha256_array(values)}:
            raise ValueError("Regional array integrity hash mismatch")
        lookup = {ident: i for i, ident in enumerate(cached_ids)}
        result = values[[lookup[i] for i in ids]].copy()
    return result, metadata


def atomic_save(np, output_dir, ids, features, metadata):
    output_dir.mkdir(parents=True, exist_ok=True)
    nonce = uuid.uuid4().hex
    tmp_npz, tmp_json = output_dir / f".{nonce}.npz.tmp", output_dir / f".{nonce}.json.tmp"
    try:
        with tmp_npz.open("wb") as handle:
            np.savez_compressed(handle, ids=np.asarray(ids, dtype=str), **{LOCAL_NAME: features},
                metadata_json=np.asarray(common.canonical_json(metadata)))
        tmp_json.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp_npz, output_dir / "features.npz")
        os.replace(tmp_json, output_dir / "metadata.json")
    finally:
        for path in (tmp_npz, tmp_json):
            if path.exists():
                path.unlink()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "open")
    parser.add_argument("--global-features", type=Path, default=ROOT / "reports/frozen_dinov2_small/features.npz")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/dinov2_spatial/features")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--validate-only", action="store_true")
    modes.add_argument("--check-only", action="store_true")
    modes.add_argument("--reuse", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    args.revision = base.DEFAULT_REVISION
    if args.batch_size < 1 or args.num_workers < 0 or args.threads < 1 or not 0 <= args.seed < 2**32:
        parser.error("Invalid batch size, workers, threads or seed")
    if args.overwrite and (args.check_only or args.validate_only or args.reuse):
        parser.error("--overwrite is incompatible with check/validate/reuse")
    return args


def main(argv=None):
    args = parse_args(argv)
    # Set before importing huggingface_hub or timm: no network requests, even for metadata.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import numpy as np
    if not (args.check_only or args.validate_only or args.reuse):
        if args.output_dir.resolve() in (args.global_features.resolve().parent, args.data_dir.resolve()):
            raise ValueError("Output would overwrite a protected input directory")
        if args.output_dir.exists() and (not args.output_dir.is_dir() or (any(args.output_dir.iterdir()) and not args.overwrite)):
            raise ValueError("Output already exists; use --reuse or a new output directory")
    if not args.check_only:
        ids, original_global, original_metadata = read_global_cache(np, args.global_features, args.data_dir / "train.csv")
        reference_before = global_reference(args.global_features)
        inventory = common.training_inventory(args.data_dir)
        check_inventory(inventory, original_metadata)
        if args.reuse:
            validate_spatial_cache(np, args.output_dir / "features.npz", ids, original_metadata, args.global_features)
            print("Verified cached spatial features; no model loaded.")
            return 0
        checkpoints = base.cached_checkpoint_files(args.revision)
        names = {item["filename"] for item in checkpoints}
        if "config.json" not in names or not names.intersection({"model.safetensors", "pytorch_model.bin"}):
            raise ValueError("Pinned DINO checkpoint is absent from local cache; offline extraction cannot proceed")
        if args.validate_only:
            print(json.dumps({"status": "original_cache_images_and_local_checkpoint_valid", "rows": len(ids),
                "regional_dimension": LOCAL_DIM, "augmented_dimension": 1152 + LOCAL_DIM,
                "torch_imported": False, "model_loaded": False, "test_accessed": False, "output_written": False}, indent=2))
            return 0
    np, torch, timm, resolve_config, versions = common.import_runtime()
    device, device_info = common.configure_runtime(torch, np, args)
    model, official = base.build_model(torch, timm, resolve_config, args.revision, pretrained=not args.check_only)
    model = model.to(device=device, dtype=torch.float32)
    state_before = common.feature_state_hash(model)
    if args.check_only:
        with torch.inference_mode():
            inputs = base.reflect_pad_input(torch, torch.zeros((1, 3, 256, 256), device=device))
            global_features, regions = pool_regions(torch, model.forward_features(inputs))
        if common.feature_state_hash(model) != state_before:
            raise RuntimeError("Frozen synthetic model changed state")
        print(json.dumps({"status": "synthetic_forward_ok", "global_shape": list(global_features.shape),
            "regional_shape": list(regions.shape), "pretrained": False, "data_accessed": False, "output_written": False}))
        return 0
    if state_before != original_metadata["feature_state_sha256"]:
        raise ValueError("Loaded frozen model differs from original DINO weights; no cache saved")
    dataset = base.TrainPaddedDataset(args.data_dir, ids.tolist())
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda", worker_init_fn=common.seed_worker,
        generator=torch.Generator().manual_seed(args.seed))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started, processed, parts, max_difference = time.perf_counter(), 0, [], 0.0
    with torch.inference_mode():
        for step, inputs in enumerate(loader, start=1):
            inputs = inputs.to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda")
            global_features, regions = pool_regions(torch, model.forward_features(inputs))
            actual = global_features.cpu().numpy()
            max_difference = max(max_difference, check_global_match(np, actual, original_global[processed:processed + len(actual)]))
            parts.append(regions.cpu().numpy().copy())
            processed += len(actual)
            if step == 1 or step % 5 == 0 or processed == len(ids):
                print(f"Extracted {processed}/{len(ids)} spatial features; elapsed={time.perf_counter()-started:.1f}s", flush=True)
    features = np.concatenate(parts).astype(np.float32, copy=False)
    common.ensure_inputs_unchanged(args.data_dir, inventory)
    if global_reference(args.global_features) != reference_before or common.feature_state_hash(model) != state_before:
        raise ValueError("Global cache or frozen model changed during extraction; no cache saved")
    config = base.extraction_config(args, versions, device, device_info)
    config.update(pooling=POOLING, extractor_script_sha256=common.sha256_file(Path(__file__)),
        base_extractor_script_sha256=common.sha256_file(Path(base.__file__)))
    metadata = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(), "rows": len(ids),
        "spatial_protocol": PROTOCOL, "extraction_config": config, "global_reference": reference_before,
        "train_csv_sha256": inventory["train_csv_sha256"], "image_manifest_sha256": inventory["image_manifest_sha256"],
        "feature_state_sha256": state_before, "feature_array_sha256": {LOCAL_NAME: common.sha256_array(features)},
        "global_reextraction_check": {"passed": True, "atol": MATCH_ATOL, "rtol": MATCH_RTOL,
            "max_absolute_difference": max_difference, "evaluation_uses_original_global_cache": True},
        "test_accessed": False, "labels_saved_in_npz": False, "offline_checkpoint_only": True,
        "cached_checkpoint_files": checkpoints, "elapsed_extraction_seconds": time.perf_counter()-started}
    atomic_save(np, args.output_dir, ids, features, metadata)
    print(f"Saved {args.output_dir / 'features.npz'}; regional={features.shape}; max global difference={max_difference:.3g}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, RuntimeError, KeyError) as exc:
        print(f"Spatial extraction stopped: {exc}", file=sys.stderr)
        raise SystemExit(2)
