"""User-run frozen DINO D4 feature averaging, using local weights and train only.

Eight exact pixel permutations (four right-angle rotations, with/without a
horizontal flip) are pooled separately, then averaged into one 1152-D row.
--validate-only reads/hashes existing inputs without torch or model inference.
--check-only --device cpu uses random weights and synthetic pixels only.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
import uuid

if __package__:
    from . import extract_dinov2_spatial as shared
else:
    import extract_dinov2_spatial as shared

base, common, dino = shared.base, shared.common, shared.dino
ROOT = common.ROOT
FEATURE = "d4_mean_cls_patch_mean_std"
WIDTH = 1152
POOLING = "mean over 8 D4 views of per-view CLS,patch mean,patch population std"
PROTOCOL = {
    "view_order": ["r0", "r90", "r180", "r270", "flip_r0", "flip_r90", "flip_r180", "flip_r270"],
    "transforms": "horizontal flip first if applicable, then torch.rot90(k, dims=(-2,-1))",
    "transform_stage": "256x256 normalized RGB before symmetric reflect padding",
    "interpolation": None, "random_augmentation": False, "views_per_image": 8,
    "pool_each_view": base.POOLING, "view_reduction": "arithmetic mean; FP64 accumulation then FP32 storage",
    "feature_dimension": WIDTH, "training_rows_per_image": 1,
}


def source_hashes():
    return {name: common.sha256_file(Path(module)) for name, module in {
        "extractor": __file__, "spatial_utilities": shared.__file__, "base_extractor": base.__file__,
        "common_utilities": common.__file__, "global_validator": dino.__file__}.items()}


def d4_views(torch, inputs):
    if inputs.ndim != 4 or inputs.shape[-1] != inputs.shape[-2]:
        raise ValueError("D4 requires square NCHW images")
    for flip in (False, True):
        image = torch.flip(inputs, dims=(-1,)) if flip else inputs
        for k in range(4):
            yield torch.rot90(image, k, dims=(-2, -1)).contiguous()


def extract_batch(torch, model, inputs):
    parts = []
    for view in d4_views(torch, inputs):
        padded = base.reflect_pad_input(torch, view)
        _, pooled = base.pool_tokens(torch, model.forward_features(padded))
        parts.append(pooled)
    views = torch.stack(parts, dim=0)
    averaged = views.to(torch.float64).mean(dim=0).to(torch.float32)
    if averaged.shape != (len(inputs), WIDTH) or not torch.isfinite(averaged).all().item():
        raise RuntimeError("Invalid D4 features")
    return views[0], averaged


def validate_cache(np, path, ids, original_metadata, original_path):
    with np.load(path, allow_pickle=False) as archive:
        if len(archive.files) != 3 or set(archive.files) != {"ids", FEATURE, "metadata_json"}:
            raise ValueError("D4 cache must contain IDs, one feature array and metadata only")
        cached_ids, values, raw = archive["ids"], archive[FEATURE], archive["metadata_json"]
        if (cached_ids.dtype.kind != "U" or cached_ids.shape != ids.shape or
                len(set(cached_ids)) != len(ids) or set(cached_ids) != set(ids)):
            raise ValueError("D4 cache IDs differ")
        if values.dtype != np.float32 or values.shape != (len(ids), WIDTH) or not np.isfinite(values).all():
            raise ValueError("D4 features must be finite FP32 with width 1152")
        if np.any(values[:, 768:] < 0):
            raise ValueError("Mean of patch standard deviations must be nonnegative")
        if raw.shape != () or raw.dtype.kind != "U":
            raise ValueError("D4 metadata must be a Unicode scalar")
        metadata = json.loads(raw.item())
        if metadata != json.loads(path.with_name("metadata.json").read_text(encoding="utf-8")):
            raise ValueError("D4 metadata sidecar mismatch")
        if (metadata.get("schema_version") != 1 or metadata.get("rows") != len(ids) or
                metadata.get("d4_protocol") != PROTOCOL or metadata.get("test_accessed") is not False or
                metadata.get("labels_saved_in_npz") is not False or metadata.get("offline_checkpoint_only") is not True):
            raise ValueError("D4 protocol or train-only scope mismatch")
        for key in ("feature_state_sha256", "train_csv_sha256", "image_manifest_sha256"):
            if metadata.get(key) != original_metadata[key]:
                raise ValueError(f"D4/global source mismatch: {key}")
        reference = shared.global_reference(original_path)
        for key in ("sha256", "metadata_sha256"):
            if metadata.get("global_reference", {}).get(key) != reference[key]:
                raise ValueError("D4 cache references a different original global cache")
        config = metadata.get("extraction_config", {})
        expected = {**dino.EXPECTED_CONFIG, "pooling": POOLING, "augmentation": PROTOCOL}
        for key, value in expected.items():
            if config.get(key) != value:
                raise ValueError(f"D4 extraction configuration mismatch: {key}")
        if metadata.get("source_sha256") != source_hashes():
            raise ValueError("D4 extraction sources changed; use a new cache")
        check = metadata.get("identity_reextraction_check", {})
        difference = check.get("max_absolute_difference")
        if (check.get("passed") is not True or check.get("atol") != shared.MATCH_ATOL or
                check.get("rtol") != shared.MATCH_RTOL or not isinstance(difference, (int, float)) or
                not np.isfinite(difference) or difference < 0 or
                check.get("evaluation_uses_original_global_cache") is not True):
            raise ValueError("Missing identity re-extraction verification")
        if metadata.get("feature_array_sha256") != {FEATURE: common.sha256_array(values)}:
            raise ValueError("D4 array integrity hash mismatch")
        lookup = {ident: i for i, ident in enumerate(cached_ids)}
        return values[[lookup[i] for i in ids]].copy(), metadata


def atomic_save(np, output, ids, features, metadata):
    output.mkdir(parents=True, exist_ok=True)
    nonce = uuid.uuid4().hex
    temporary = [output / f".{nonce}.npz.tmp", output / f".{nonce}.json.tmp"]
    try:
        with temporary[0].open("wb") as handle:
            np.savez_compressed(handle, ids=np.asarray(ids, dtype=str), **{FEATURE: features},
                                metadata_json=np.asarray(common.canonical_json(metadata)))
        temporary[1].write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        for source, name in zip(temporary, ("features.npz", "metadata.json")):
            os.replace(source, output / name)
    finally:
        for path in temporary:
            if path.exists():
                path.unlink()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "open")
    parser.add_argument("--global-features", type=Path, default=ROOT / "reports/frozen_dinov2_small/features.npz")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/dinov2_d4/features")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    modes = parser.add_mutually_exclusive_group()
    for flag in ("validate-only", "check-only", "reuse"):
        modes.add_argument("--" + flag, action="store_true")
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
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import numpy as np
    if not (args.check_only or args.validate_only or args.reuse):
        output = args.output_dir.resolve()
        if output == args.global_features.resolve().parent or output.is_relative_to(args.data_dir.resolve()):
            raise ValueError("Output would overwrite a protected input directory")
        if output.exists() and (not output.is_dir() or (any(output.iterdir()) and not args.overwrite)):
            raise ValueError("Output already exists; use --reuse or a new directory")
    if not args.check_only:
        ids, original, original_metadata = shared.read_global_cache(np, args.global_features, args.data_dir / "train.csv")
        reference_before = shared.global_reference(args.global_features)
        inventory = common.training_inventory(args.data_dir)
        shared.check_inventory(inventory, original_metadata)
        if args.reuse:
            validate_cache(np, args.output_dir / "features.npz", ids, original_metadata, args.global_features)
            print("Verified cached D4 features; no model loaded.")
            return 0
        checkpoints = base.cached_checkpoint_files(args.revision)
        names = {entry["filename"] for entry in checkpoints}
        if "config.json" not in names or not names.intersection({"model.safetensors", "pytorch_model.bin"}):
            raise ValueError("Pinned checkpoint absent from local cache; no downloads attempted")
        if args.validate_only:
            print(json.dumps({"status": "original_cache_images_and_local_checkpoint_valid", "rows": len(ids),
                "views_per_image": 8, "feature_dimension": WIDTH, "torch_imported": "torch" in sys.modules,
                "model_loaded": False, "test_accessed": False, "output_written": False}, indent=2))
            return 0
    np, torch, timm, resolve_config, versions = common.import_runtime()
    device, info = common.configure_runtime(torch, np, args)
    model, _ = base.build_model(torch, timm, resolve_config, args.revision, pretrained=not args.check_only)
    model = model.to(device=device, dtype=torch.float32)
    state = common.feature_state_hash(model)
    if args.check_only:
        with torch.inference_mode():
            sample = torch.randn(1, 3, 256, 256, device=device)
            identity, averaged = extract_batch(torch, model, sample)
        if common.feature_state_hash(model) != state:
            raise RuntimeError("Frozen synthetic model changed state")
        print(json.dumps({"status": "synthetic_forward_ok", "identity_shape": list(identity.shape),
            "d4_shape": list(averaged.shape), "pretrained": False, "data_accessed": False, "output_written": False}))
        return 0
    if state != original_metadata["feature_state_sha256"]:
        raise ValueError("Frozen model differs from original DINO weights; no cache saved")
    hashes_before = source_hashes()
    dataset = common.TrainImageDataset(args.data_dir, ids.tolist(), common.EXPECTED_MEAN, common.EXPECTED_STD)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda", worker_init_fn=common.seed_worker,
        generator=torch.Generator().manual_seed(args.seed))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start, processed, parts, maximum = time.perf_counter(), 0, [], 0.0
    with torch.inference_mode():
        for step, inputs in enumerate(loader, start=1):
            inputs = inputs.to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda")
            identity, averaged = extract_batch(torch, model, inputs)
            actual = identity.cpu().numpy()
            maximum = max(maximum, shared.check_global_match(np, actual, original[processed:processed + len(actual)]))
            parts.append(averaged.cpu().numpy().copy())
            processed += len(inputs)
            if step == 1 or step % 5 == 0 or processed == len(ids):
                print(f"Extracted {processed}/{len(ids)} D4 train images (8 views each); elapsed={time.perf_counter()-start:.1f}s", flush=True)
    features = np.concatenate(parts).astype(np.float32, copy=False)
    common.ensure_inputs_unchanged(args.data_dir, inventory)
    if (shared.global_reference(args.global_features) != reference_before or
            common.feature_state_hash(model) != state or source_hashes() != hashes_before):
        raise ValueError("Inputs, sources or model changed during extraction; no cache saved")
    config = base.extraction_config(args, versions, device, info)
    config.update(pooling=POOLING, augmentation=PROTOCOL, extractor_script_sha256=hashes_before["extractor"])
    metadata = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(), "rows": len(ids),
        "d4_protocol": PROTOCOL, "extraction_config": config, "global_reference": reference_before,
        "train_csv_sha256": inventory["train_csv_sha256"], "image_manifest_sha256": inventory["image_manifest_sha256"],
        "feature_state_sha256": state, "feature_array_sha256": {FEATURE: common.sha256_array(features)},
        "source_sha256": hashes_before, "identity_reextraction_check": {"passed": True,
            "atol": shared.MATCH_ATOL, "rtol": shared.MATCH_RTOL, "max_absolute_difference": maximum,
            "evaluation_uses_original_global_cache": True},
        "test_accessed": False, "labels_saved_in_npz": False, "offline_checkpoint_only": True,
        "cached_checkpoint_files": checkpoints, "elapsed_extraction_seconds": time.perf_counter()-start}
    atomic_save(np, args.output_dir, ids, features, metadata)
    validate_cache(np, args.output_dir / "features.npz", ids, original_metadata, args.global_features)
    print(f"Saved {args.output_dir / 'features.npz'}; D4={features.shape}; max identity difference={maximum:.3g}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, RuntimeError, KeyError) as exc:
        print(f"D4 extraction stopped: {exc}", file=sys.stderr)
        raise SystemExit(2)
