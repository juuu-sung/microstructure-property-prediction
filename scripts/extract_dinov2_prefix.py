"""User-run frozen DINO prefix cache for last-block fine-tuning.

Cache the input to blocks[11], not labels, with the exact original image protocol.
--validate-only reads/hashes train inputs and local checkpoint; no torch/forward.
--check-only uses a random model and synthetic CPU input; no competition images.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
import uuid

if __package__:
    from . import extract_dinov2_spatial as spatial
else:
    import extract_dinov2_spatial as spatial

base, common = spatial.base, spatial.common
ROOT = common.ROOT
PROTOCOL = {"schema_version": 1, "token_source": "input to blocks.11 (after blocks.10)",
    "frozen_prefix_blocks": 11, "last_block_index": 11, "total_blocks": 12,
    "shape_per_image": [362, 384], "dtype": "float32", "augmentation": None}


def sources():
    return {name: {"path": str(Path(module).resolve()), "sha256": common.sha256_file(module)}
        for name, module in {"prefix_extractor": __file__, "base_extractor": base.__file__,
            "spatial_validation": spatial.__file__, "common": common.__file__}.items()}


def ensure_local_checkpoint():
    records = base.cached_checkpoint_files(base.DEFAULT_REVISION)
    names = {r["filename"] for r in records}
    if "config.json" not in names or not names.intersection({"model.safetensors", "pytorch_model.bin"}):
        raise ValueError("Pinned DINO checkpoint missing locally; downloads are disabled")
    return records


def capture_prefix(torch, model, inputs):
    if len(model.blocks) != 12:
        raise ValueError("Expected 12 DINO blocks")
    captured = []
    hook = model.blocks[-1].register_forward_pre_hook(lambda module, args: captured.append(args[0].detach().clone()))
    try:
        with torch.no_grad():
            tokens = model.forward_features(inputs)
            _, pooled = base.pool_tokens(torch, tokens)
    finally:
        hook.remove()
    if len(captured) != 1 or tuple(captured[0].shape[1:]) != (362, 384):
        raise RuntimeError("Unexpected prefix hook execution or shape")
    return captured[0], pooled


def validate_cache(np, folder, ids, original_metadata, global_path):
    metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("protocol") != PROTOCOL or metadata.get("ids") != list(ids):
        raise ValueError("Prefix cache protocol/ordered IDs mismatch")
    if metadata.get("test_accessed") is not False or metadata.get("labels_saved") is not False:
        raise ValueError("Prefix cache must be train-only and label-free")
    for key in ("feature_state_sha256", "train_csv_sha256", "image_manifest_sha256"):
        if metadata.get(key) != original_metadata[key]:
            raise ValueError(f"Prefix/global mismatch: {key}")
    if metadata.get("global_reference") != spatial.global_reference(global_path):
        raise ValueError("Prefix cache references a different original cache")
    if metadata.get("sources") != sources():
        raise ValueError("Prefix extraction source files changed; use a fresh cache directory")
    config = metadata.get("extraction_config", {})
    for key, value in spatial.dino.EXPECTED_CONFIG.items():
        if config.get(key) != value:
            raise ValueError(f"Prefix image/model configuration mismatch: {key}")
    check = metadata.get("global_reextraction_check", {})
    if (check.get("passed") is not True or check.get("atol") != spatial.MATCH_ATOL or
            check.get("rtol") != spatial.MATCH_RTOL or not isinstance(check.get("max_absolute_difference"), (int, float)) or
            not np.isfinite(check["max_absolute_difference"]) or check["max_absolute_difference"] < 0):
        raise ValueError("Prefix cache lacks global reproduction check")
    path = folder / "tokens.npy"
    if common.sha256_file(path) != metadata.get("tokens_file_sha256"):
        raise ValueError("Prefix token file hash mismatch")
    tokens = np.load(path, mmap_mode="r", allow_pickle=False)
    if tokens.shape != (len(ids), 362, 384) or tokens.dtype != np.float32 or not np.isfinite(tokens).all():
        raise ValueError("Invalid prefix token array")
    return tokens, metadata


def save_cache(np, folder, tokens, metadata):
    folder.mkdir(parents=True, exist_ok=True)
    token_tmp = folder / (".tokens-" + uuid.uuid4().hex + ".tmp")
    meta_tmp = folder / (".metadata-" + uuid.uuid4().hex + ".tmp")
    try:
        with token_tmp.open("wb") as handle:
            np.save(handle, tokens, allow_pickle=False)
        metadata = {**metadata, "tokens_file_sha256": common.sha256_file(token_tmp)}
        meta_tmp.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        os.replace(token_tmp, folder / "tokens.npy")
        os.replace(meta_tmp, folder / "metadata.json")
    finally:
        for p in (token_tmp, meta_tmp):
            if p.exists():
                p.unlink()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=ROOT / "open")
    p.add_argument("--global-features", type=Path, default=ROOT / "reports/frozen_dinov2_small/features.npz")
    p.add_argument("--output-dir", type=Path, default=ROOT / "reports/dinov2_partial/prefix")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--threads", type=int, default=2)
    modes = p.add_mutually_exclusive_group()
    for mode in ("validate-only", "check-only", "reuse"):
        modes.add_argument("--" + mode, action="store_true")
    args = p.parse_args(argv)
    args.seed, args.num_workers, args.revision = 42, 0, base.DEFAULT_REVISION
    if args.batch_size < 1 or args.threads < 1:
        p.error("Positive batch size and threads required")
    if args.check_only and args.device != "cpu":
        p.error("--check-only requires --device cpu")
    return args


def main(argv=None):
    args = parse_args(argv)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import numpy as np
    if not (args.check_only or args.validate_only or args.reuse):
        if args.data_dir.resolve().is_relative_to(args.output_dir.resolve()) or args.output_dir.resolve().is_relative_to(args.data_dir.resolve()) or args.global_features.resolve().is_relative_to(args.output_dir.resolve()):
            raise ValueError("Output contains protected inputs or is inside the data directory")
        if args.output_dir.exists() and (not args.output_dir.is_dir() or any(args.output_dir.iterdir())):
            raise ValueError("Output exists; use --reuse or a new directory")
    if not args.check_only:
        ids, original, original_metadata = spatial.read_global_cache(np, args.global_features, args.data_dir / "train.csv")
        inventory = common.training_inventory(args.data_dir)
        spatial.check_inventory(inventory, original_metadata)
        reference_before = spatial.global_reference(args.global_features)
        if args.reuse:
            validate_cache(np, args.output_dir, ids, original_metadata, args.global_features)
            print("Prefix cache verified; no model loaded.")
            return 0
        checkpoint = ensure_local_checkpoint()
        if args.validate_only:
            print(json.dumps({"status": "global_cache_images_local_checkpoint_valid", "rows": len(ids),
                "prefix_bytes": len(ids)*362*384*4, "torch_imported": False, "forward_or_fit": False, "output_written": False}))
            return 0
    np, torch, timm, resolve_config, versions = common.import_runtime()
    device, device_info = common.configure_runtime(torch, np, args)
    model, official = base.build_model(torch, timm, resolve_config, pretrained=not args.check_only)
    state = common.feature_state_hash(model)
    if not args.check_only and state != original_metadata["feature_state_sha256"]:
        raise ValueError("Local pretrained model differs from original DINO cache")
    model.to(device)
    if args.check_only:
        x = base.reflect_pad_input(torch, torch.randn(2, 3, 256, 256))
        tokens, pooled = capture_prefix(torch, model, x)
        with torch.no_grad():
            restored = base.pool_tokens(torch, model.norm(model.blocks[-1](tokens)))[1]
        torch.testing.assert_close(restored, pooled, rtol=0, atol=0)
        assert state == common.feature_state_hash(model)
        print("Synthetic CPU prefix/suffix roundtrip matched exactly; no images or pretrained weights used.")
        return 0
    dataset = base.TrainPaddedDataset(args.data_dir, ids)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    tokens = np.empty((len(ids), 362, 384), dtype=np.float32)
    actual_global = np.empty_like(original)
    start, offset = time.perf_counter(), 0
    for x in loader:
        prefix, pooled = capture_prefix(torch, model, x.to(device))
        end = offset + len(x)
        tokens[offset:end] = prefix.cpu().numpy()
        actual_global[offset:end] = pooled.cpu().numpy()
        offset = end
        if offset == len(ids) or offset % 80 == 0:
            print(f"Cached prefix {offset}/{len(ids)}; elapsed={time.perf_counter()-start:.1f}s", flush=True)
    difference = spatial.check_global_match(np, actual_global, original)
    common.ensure_inputs_unchanged(args.data_dir, inventory)
    if common.feature_state_hash(model) != state or reference_before != spatial.global_reference(args.global_features):
        raise ValueError("Frozen state or reference cache changed during extraction")
    metadata = {"protocol": PROTOCOL, "ids": list(ids), "test_accessed": False, "labels_saved": False,
        "feature_state_sha256": state, "train_csv_sha256": inventory["train_csv_sha256"],
        "image_manifest_sha256": inventory["image_manifest_sha256"], "global_reference": reference_before,
        "sources": sources(), "checkpoint": checkpoint, "versions": versions,
        "extraction_config": base.extraction_config(args, versions, device, device_info),
        "global_reextraction_check": {"passed": True, "atol": spatial.MATCH_ATOL, "rtol": spatial.MATCH_RTOL,
            "max_absolute_difference": difference}, "elapsed_seconds": time.perf_counter()-start}
    save_cache(np, args.output_dir, tokens, metadata)
    print(f"Saved {args.output_dir}; prefix shape={tokens.shape}; global max difference={difference:g}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"Prefix extraction stopped: {exc}", file=sys.stderr)
        raise SystemExit(2)
