"""Extract fixed, ImageNet-pretrained ResNet18 features from TRAIN images only.

No fit, augmentation, resize/crop, test-image access, or label-dependent operation
is performed. Run --validate-only before installing torch, or --check-only for a
synthetic forward check that does not download pretrained weights.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "resnet18.a1_in1k"
HF_REPO = "timm/resnet18.a1_in1k"
DEFAULT_REVISION = "491b427b45c94c7fb0e78b5474cc919aff584bbf"
EXPECTED_MEAN = [0.485, 0.456, 0.406]
EXPECTED_STD = [0.229, 0.224, 0.225]
EXPECTED_CHANNELS = [128, 256, 512]
FEATURE_KEYS = {"ids", "final_gap", "multiscale_mean_std", "metadata_json"}
SCHEMA_VERSION = 1


def project_path(value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def canonical_json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(array):
    # Include shape/dtype to distinguish identical bytes with different layouts.
    digest = hashlib.sha256()
    digest.update(canonical_json({"shape": list(array.shape), "dtype": str(array.dtype)}).encode())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def training_inventory(data_dir, expected_count=500):
    """Validate only train.csv and train/*.png; preserve CSV order exactly."""
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is missing. Use the project .venv-microstructure interpreter and install the requirements.") from exc
    csv_path = data_dir / "train.csv"
    csv_hash = sha256_file(csv_path)
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["ID", "hardness"]:
            raise ValueError(f"Expected train.csv columns ID,hardness; found {reader.fieldnames!r}.")
        rows = list(reader)
    if len(rows) != expected_count:
        raise ValueError(f"Expected {expected_count} training rows; found {len(rows)}.")
    ids = [row["ID"] for row in rows]
    if len(set(ids)) != len(ids) or any(not re.fullmatch(r"TRAIN_\d{6}", ident or "") for ident in ids):
        raise ValueError("Training IDs must be unique and match TRAIN_000001 format.")
    for row in rows:
        try:
            finite_target = math.isfinite(float(row["hardness"]))
        except (TypeError, ValueError):
            finite_target = False
        if not finite_target:
            raise ValueError(f"Non-finite or invalid hardness for {row['ID']}.")
    train_dir = data_dir / "train"
    png_names = {p.name for p in train_dir.iterdir() if p.is_file() and p.suffix.lower() == ".png"}
    expected_names = {f"{ident}.png" for ident in ids}
    if png_names != expected_names:
        raise ValueError(f"Training PNG inventory mismatch: missing={sorted(expected_names-png_names)[:5]}, extra={sorted(png_names-expected_names)[:5]}.")
    manifest = []
    for ident in ids:
        path = train_dir / f"{ident}.png"
        with Image.open(path) as image:
            if image.format != "PNG" or image.mode != "L" or image.size != (256, 256):
                raise ValueError(f"{path.name}: expected 256x256 grayscale L PNG, found {image.format}/{image.mode}/{image.size}.")
            image.load()  # Decode now so corrupt image data cannot pass metadata validation.
        manifest.append({"id": ident, "relative_path": f"train/{ident}.png", "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    if sha256_file(csv_path) != csv_hash:
        raise RuntimeError("train.csv changed during validation. Retry with unchanged input files.")
    return {"ids": ids, "train_csv_sha256": csv_hash, "images": manifest,
            "image_manifest_sha256": hashlib.sha256(canonical_json(manifest).encode()).hexdigest()}


def ensure_inputs_unchanged(data_dir, inventory):
    if sha256_file(data_dir / "train.csv") != inventory["train_csv_sha256"]:
        raise RuntimeError("train.csv changed during extraction; no features were saved.")
    for entry in inventory["images"]:
        if sha256_file(data_dir / entry["relative_path"]) != entry["sha256"]:
            raise RuntimeError(f"{entry['relative_path']} changed during extraction; no features were saved.")


class TrainImageDataset:
    """Top-level, pickleable dataset for Windows DataLoader worker processes."""

    def __init__(self, data_dir, ids, mean, std):
        self.data_dir = str(data_dir)
        self.ids = list(ids)
        self.mean = list(mean)
        self.std = list(std)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        import numpy as np
        from PIL import Image
        import torch
        path = Path(self.data_dir) / "train" / f"{self.ids[index]}.png"
        with Image.open(path) as image:
            if image.format != "PNG" or image.mode != "L" or image.size != (256, 256):
                raise ValueError(f"Input changed or has unsupported image format: {path.name}.")
            gray = np.array(image, dtype=np.float32) / np.float32(255.0)
        rgb = np.repeat(gray[None, :, :], 3, axis=0)
        rgb -= np.asarray(self.mean, dtype=np.float32)[:, None, None]
        rgb /= np.asarray(self.std, dtype=np.float32)[:, None, None]
        return torch.from_numpy(np.ascontiguousarray(rgb))


def seed_worker(worker_id):
    import numpy as np
    import torch
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def import_runtime():
    try:
        import numpy as np
        import torch
        import timm
        from timm.data import resolve_model_data_config
    except (ImportError, OSError, RuntimeError) as exc:
        raise RuntimeError(
            "Cannot import the extraction runtime. In VS Code select .venv-microstructure\\Scripts\\python.exe "
            "and install the project requirements. CUDA torch and torchvision must be matching builds. "
            f"Original error: {exc}"
        ) from exc
    versions = {"python": platform.python_version()}
    for package in ("torch", "torchvision", "timm", "numpy", "Pillow", "huggingface-hub", "safetensors"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return np, torch, timm, resolve_model_data_config, versions


def configure_runtime(torch, np, args):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("--device must be cpu, cuda, or cuda:N.")
    cuda_available = torch.cuda.is_available()
    # Some CUDA wheel builds query the minimum GPU capability in cudnn.version().
    # A CPU-only check must also work when no CUDA devices are visible.
    cudnn_version = torch.backends.cudnn.version() if device.type == "cuda" and cuda_available else None
    info = {"requested_device": args.device, "torch_cuda_build": torch.version.cuda,
            "cuda_available": cuda_available, "cudnn_version": cudnn_version}
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. Use the CUDA PyTorch environment and a compatible NVIDIA driver, or pass --device cpu explicitly. No CPU fallback was performed.")
        index = torch.cuda.current_device() if device.index is None else device.index
        if index >= torch.cuda.device_count() or index < 0:
            raise ValueError(f"CUDA device index {index} does not exist.")
        device = torch.device(f"cuda:{index}")
        torch.cuda.set_device(device)
        torch.cuda.manual_seed_all(args.seed)
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        info.update({"resolved_device": str(device), "gpu_name": torch.cuda.get_device_name(device),
                     "compute_capability": list(torch.cuda.get_device_capability(device)),
                     "free_vram_bytes_before_model": free_bytes, "total_vram_bytes": total_bytes})
    else:
        info["resolved_device"] = "cpu"
    return device, info


def pool_feature_maps(torch, maps):
    if len(maps) != 3 or [int(item.shape[1]) for item in maps] != EXPECTED_CHANNELS:
        raise RuntimeError(f"Unexpected feature maps: {[list(item.shape) for item in maps]}.")
    if any(item.ndim != 4 or item.dtype != torch.float32 for item in maps):
        raise RuntimeError("Expected FP32 NCHW feature maps.")
    # Order: layer2 mean/std, layer3 mean/std, layer4 mean/std.
    pieces = []
    for item in maps:
        pieces.extend([item.mean(dim=(2, 3)), item.std(dim=(2, 3), correction=0)])
    final_gap = maps[-1].mean(dim=(2, 3))
    multiscale = torch.cat(pieces, dim=1)
    if final_gap.shape[1] != 512 or multiscale.shape[1] != 1792:
        raise RuntimeError("Unexpected pooled feature dimensions.")
    if not torch.isfinite(final_gap).all().item() or not torch.isfinite(multiscale).all().item():
        raise RuntimeError("Non-finite features were produced; extraction stopped.")
    return final_gap, multiscale


def feature_state_hash(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(sha256_array(array).encode("ascii"))
    return digest.hexdigest()


def extraction_config(args, versions, device, device_info):
    return {"schema_version": SCHEMA_VERSION, "model_id": HF_REPO, "revision": args.revision,
            "feature_indices": [2, 3, 4], "channels": EXPECTED_CHANNELS,
            "input_shape": [3, 256, 256], "normalization_mean": EXPECTED_MEAN, "normalization_std": EXPECTED_STD,
            "grayscale_handling": "L -> float32/255 -> repeat channels -> fixed ImageNet normalization",
            "resize": None, "crop": None, "augmentation": None, "dtype": "float32", "amp": False,
            "pooling": "final layer4 spatial mean; concat layer2 mean,std,layer3 mean,std,layer4 mean,std; correction=0",
            "train_mode": False, "seed": args.seed, "batch_size": args.batch_size, "num_workers": args.num_workers,
            "threads": args.threads, "device": str(device), "gpu_name": device_info.get("gpu_name"),
            "deterministic_algorithms": True, "cudnn_benchmark": False, "tf32": False,
            "versions": versions, "extractor_script_sha256": sha256_file(Path(__file__))}


def verify_cached(np, output_dir, inventory, config):
    features_path, metadata_path = output_dir / "features.npz", output_dir / "metadata.json"
    if not features_path.is_file() or not metadata_path.is_file():
        raise ValueError("--reuse requires both features.npz and metadata.json. Use a new output directory or --overwrite.")
    with np.load(features_path, allow_pickle=False) as cached:
        if set(cached.files) != FEATURE_KEYS:
            raise ValueError("Cached NPZ has an unexpected schema.")
        metadata_value = cached["metadata_json"]
        if metadata_value.shape != () or metadata_value.dtype.kind != "U":
            raise ValueError("Cached metadata_json must be a Unicode scalar.")
        metadata = json.loads(str(metadata_value.item()))
        sidecar = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata != sidecar:
            raise ValueError("Cached NPZ metadata and metadata.json differ; cache is incomplete or stale.")
        if metadata.get("extraction_config") != config:
            raise ValueError("Cached extraction configuration/runtime/script differs. Use a new directory or --overwrite.")
        for key in ("train_csv_sha256", "image_manifest_sha256"):
            if metadata.get(key) != inventory[key]:
                raise ValueError(f"Cached {key} differs from the current training data.")
        ids = cached["ids"]
        if ids.ndim != 1 or ids.dtype.kind != "U" or ids.tolist() != inventory["ids"]:
            raise ValueError("Cached training IDs do not match CSV order.")
        state_hash = metadata.get("feature_state_sha256", "")
        if not re.fullmatch(r"[0-9a-f]{64}", state_hash):
            raise ValueError("Cached feature-state identity is missing or invalid.")
        for name, width in (("final_gap", 512), ("multiscale_mean_std", 1792)):
            array = cached[name]
            if array.shape != (len(ids), width) or array.dtype != np.float32 or not np.isfinite(array).all():
                raise ValueError(f"Cached {name} is malformed.")
            if sha256_array(array) != metadata.get("feature_array_sha256", {}).get(name):
                raise ValueError(f"Cached {name} failed its integrity hash.")
    return metadata


def atomic_save(np, output_dir, ids, final_gap, multiscale, metadata):
    output_dir.mkdir(parents=True, exist_ok=True)
    nonce = uuid.uuid4().hex
    temp_npz = output_dir / f".features-{nonce}.npz.tmp"
    temp_json = output_dir / f".metadata-{nonce}.json.tmp"
    try:
        with temp_npz.open("wb") as handle:
            np.savez_compressed(handle, ids=np.asarray(ids, dtype=str), final_gap=final_gap,
                                multiscale_mean_std=multiscale, metadata_json=np.asarray(canonical_json(metadata)))
            handle.flush()
            os.fsync(handle.fileno())
        with temp_json.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        # Each file is atomic. Reuse rejects a mismatched pair after an interrupted commit.
        os.replace(temp_npz, output_dir / "features.npz")
        os.replace(temp_json, output_dir / "metadata.json")
    finally:
        for path in (temp_npz, temp_json):
            if path.exists():
                path.unlink()


def cached_checkpoint_files(revision):
    from huggingface_hub import try_to_load_from_cache
    candidates = []
    for name in ("config.json", "model.safetensors", "pytorch_model.bin"):
        path = try_to_load_from_cache(HF_REPO, name, revision=revision)
        if isinstance(path, str) and Path(path).is_file():
            candidates.append({"filename": name, "cache_path": str(Path(path).resolve()),
                               "sha256": sha256_file(path), "size_bytes": Path(path).stat().st_size})
    return candidates


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default="open", help="Absolute path or path relative to the project root.")
    parser.add_argument("--output-dir", default="reports/frozen_resnet18", help="Absolute path or project-relative directory.")
    parser.add_argument("--revision", default=DEFAULT_REVISION, help="Full 40-character Hugging Face commit hash; branches/tags are rejected.")
    parser.add_argument("--device", default="cuda", help="cuda, cuda:N, or cpu (no automatic CPU fallback).")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    checks = parser.add_mutually_exclusive_group()
    checks.add_argument("--validate-only", action="store_true", help="Validate/hash train data only; do not import torch or download anything.")
    checks.add_argument("--check-only", action="store_true", help="Check a synthetic forward with random weights; no input data read, pretrained download, or saved features.")
    reuse = parser.add_mutually_exclusive_group()
    reuse.add_argument("--reuse", action="store_true", help="Reuse only a complete cache matching every input hash and extraction setting.")
    reuse.add_argument("--overwrite", action="store_true", help="Replace this output directory's features.npz/metadata.json after successful extraction.")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.num_workers < 0 or args.threads < 1:
        parser.error("batch-size/threads must be positive, and num-workers must be nonnegative.")
    if args.seed < 0 or args.seed >= 2**32:
        parser.error("seed must be in [0, 2**32).")
    if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        parser.error("revision must be a full lowercase 40-character commit hash.")
    if (args.validate_only or args.check_only) and (args.reuse or args.overwrite):
        parser.error("Check modes cannot be combined with --reuse/--overwrite.")
    data_dir, output_dir = project_path(args.data_dir), project_path(args.output_dir)
    if args.validate_only:
        inventory = training_inventory(data_dir)
        print(json.dumps({"status": "training_data_valid", "rows": len(inventory["ids"]),
                          "train_csv_sha256": inventory["train_csv_sha256"],
                          "image_manifest_sha256": inventory["image_manifest_sha256"],
                          "test_accessed": False, "torch_imported": False}, indent=2))
        return 0
    inventory = None
    if not args.check_only:
        if not args.reuse and not args.overwrite and any((output_dir / name).exists() for name in ("features.npz", "metadata.json")):
            raise ValueError("Output features already exist. Use --reuse to verify them, a new directory, or --overwrite.")
        print("Validating training image inventory (test files are not read)...", flush=True)
        inventory = training_inventory(data_dir)
    np, torch, timm, resolve_config, versions = import_runtime()
    device, device_info = configure_runtime(torch, np, args)
    print(json.dumps({"versions": versions, "runtime": device_info}, indent=2), flush=True)
    config = extraction_config(args, versions, device, device_info)
    if args.reuse:
        metadata = verify_cached(np, output_dir, inventory, config)
        print(f"Verified existing features: {output_dir / 'features.npz'} (rows={metadata['rows']}). No model was loaded or weights downloaded.")
        return 0
    model_id = MODEL_NAME if args.check_only else f"hf-hub:{HF_REPO}@{args.revision}"
    print("Creating random-weight model for synthetic check (no download)..." if args.check_only else f"Loading pretrained checkpoint {model_id} ...", flush=True)
    model = timm.create_model(model_id, pretrained=not args.check_only, features_only=True, out_indices=(2, 3, 4))
    if list(model.feature_info.channels()) != EXPECTED_CHANNELS:
        raise RuntimeError(f"Unexpected architecture channels: {model.feature_info.channels()}.")
    official_config = resolve_config(model)
    if list(official_config["mean"]) != EXPECTED_MEAN or list(official_config["std"]) != EXPECTED_STD:
        raise RuntimeError("Checkpoint normalization differs from the fixed experiment configuration. Do not silently reuse this cache.")
    model.requires_grad_(False)
    model.eval()
    model = model.to(device=device, dtype=torch.float32)
    if any(module.training for module in model.modules()):
        raise RuntimeError("Feature extractor unexpectedly contains a training-mode module.")
    if args.check_only:
        with torch.inference_mode():
            sample = torch.zeros((1, 3, 256, 256), device=device, dtype=torch.float32)
            final, multiscale = pool_feature_maps(torch, model(sample))
        print(json.dumps({"status": "synthetic_forward_ok", "final_gap_shape": list(final.shape),
                          "multiscale_mean_std_shape": list(multiscale.shape), "pretrained": False,
                          "data_accessed": False, "features_saved": False}, indent=2))
        return 0
    state_hash_before = feature_state_hash(model)
    dataset = TrainImageDataset(data_dir, inventory["ids"], official_config["mean"], official_config["std"])
    generator = torch.Generator().manual_seed(args.seed)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False,
                                        num_workers=args.num_workers, pin_memory=device.type == "cuda",
                                        worker_init_fn=seed_worker, generator=generator)
    final_parts, multiscale_parts = [], []
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    processed = 0
    with torch.inference_mode():
        for step, inputs in enumerate(loader, start=1):
            inputs = inputs.to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda")
            final, multiscale = pool_feature_maps(torch, model(inputs))
            final_parts.append(final.cpu().numpy())
            multiscale_parts.append(multiscale.cpu().numpy())
            processed += len(inputs)
            if step == 1 or step % 5 == 0 or processed == len(dataset):
                print(f"Extracted {processed}/{len(dataset)} train images; elapsed={time.perf_counter()-start:.1f}s", flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    final_gap = np.concatenate(final_parts, axis=0).astype(np.float32, copy=False)
    multiscale = np.concatenate(multiscale_parts, axis=0).astype(np.float32, copy=False)
    if final_gap.shape != (len(dataset), 512) or multiscale.shape != (len(dataset), 1792):
        raise RuntimeError("Final feature arrays have unexpected shapes.")
    ensure_inputs_unchanged(data_dir, inventory)
    state_hash_after = feature_state_hash(model)
    if state_hash_before != state_hash_after:
        raise RuntimeError("Model state changed during frozen extraction; no features were saved.")
    metadata = {
        "schema_version": SCHEMA_VERSION, "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "train only; frozen ImageNet features; no training or target-dependent transformation",
        "rows": len(dataset), "extraction_config": config,
        "train_csv_sha256": inventory["train_csv_sha256"], "image_manifest_sha256": inventory["image_manifest_sha256"],
        "training_image_manifest": inventory["images"], "feature_state_sha256": state_hash_after,
        "feature_array_sha256": {"final_gap": sha256_array(final_gap), "multiscale_mean_std": sha256_array(multiscale)},
        "official_model_data_config": official_config, "official_resize_crop_applied": False,
        "checkpoint": {"repository": HF_REPO, "revision": args.revision,
                       "model_card_url": f"https://huggingface.co/{HF_REPO}/blob/{args.revision}/README.md",
                       "model_card_license": "apache-2.0" if args.revision == DEFAULT_REVISION else "verify model card for override revision",
                       "cached_candidate_files": cached_checkpoint_files(args.revision),
                       "identity_note": "feature_state_sha256 hashes the actual loaded feature-extractor state; cached files are provenance candidates, not asserted load traces."},
        "runtime": device_info, "elapsed_extraction_seconds": elapsed,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None,
        "test_accessed": False, "labels_saved_in_npz": False,
    }
    atomic_save(np, output_dir, inventory["ids"], final_gap, multiscale, metadata)
    print(f"Saved {output_dir / 'features.npz'} and {output_dir / 'metadata.json'}", flush=True)
    print(f"Feature shapes: final_gap={final_gap.shape}, multiscale_mean_std={multiscale.shape}; extraction time={elapsed:.1f}s.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted. No completed output is claimed; use --reuse to verify any existing cache.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        if "out of memory" in str(exc).lower():
            print("Close other GPU workloads or rerun with --batch-size 8 (or 4). Existing completed outputs were not silently replaced.", file=sys.stderr)
        raise SystemExit(1)
