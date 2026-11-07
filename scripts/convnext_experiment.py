"""Train-only ConvNeXt experiment. User runs extract/sanity/train; prepare never fits.

The runner accepts outer image indices but never their labels. All profile/epoch
selection is inside a predeclared 320/80 split of each original 400-image fold.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np

if __package__:
    from . import evaluate_morphology as previous
    from . import extract_frozen_resnet as common
else:
    import evaluate_morphology as previous
    import extract_frozen_resnet as common

ROOT, core = common.ROOT, previous.core
OUT = ROOT / "reports/convnext_full"
MODEL = "convnext_tiny.fb_in22k_ft_in1k"
REPO = "timm/" + MODEL
REVISION = "bc48a87f119bae1b06418f5340bfa97ef75d4609"
WEIGHT_SHA = "77b619e187de71bce58721689da0c009e84cc20677293d7d43105307e34d868a"
MEAN, STD = common.EXPECTED_MEAN, common.EXPECTED_STD
SEEDS = (42, 2026)
PROFILES = ("none", "d4")
ALPHAS = (10., 100., 1000.)
PROTOCOL = {
    "version": 1, "model": REPO, "revision": REVISION, "weight_sha256": WEIGHT_SHA,
    "input": "L256 -> repeat3 /255 -> fixed ImageNet mean/std; no resize/crop",
    "mean": MEAN, "std": STD, "stages": [2, 3, 4], "channels": [192, 384, 768],
    "pooling": "spatial mean/std correction=0 in FP32; concatenated2688",
    "head": "LayerNorm2688 -> Dropout0.1 -> Linear1, default torch initialization",
    "backbone_drop_path": 0., "seeds": list(SEEDS), "outer": "existing5 folds",
    "inner": "one StratifiedShuffleSplit320/80 on outer400 rank-deciles; random_state=seed+1009*fold",
    "run_seed": "seed+1009*fold+100000 for both inner profiles, +200000 for outer refit",
    "profiles": list(PROFILES), "validation": "identity1view, no augmentation or TTA",
    "selection": "lowest inner80 MSE across two100epoch curves; ties earlier epoch then none",
    "warmup_epochs": 5, "warmup_head_lr": .001, "max_epochs": 100,
    "full_lrs": {"stem_stage1": 3e-6, "stage2": 6e-6, "stage3": 1e-5, "stage4": 3e-5, "head": 3e-4},
    "optimizer": "AdamW, reset once at full-training boundary", "weight_decay": .01,
    "no_decay": "bias, normalization and other1D parameters (including layer-scale gamma)",
    "schedule": "full epochs6..100: 0.5*(1+cos(pi*(epoch-6)/94)); same horizon at outer refit",
    "loss": "MSE, y population mean/std fitted on each training subset", "clip_norm": 1.,
    "batch": "default4x4; pre-run memory fallback2x8, fixed per experiment run; effective16",
    "amp": "CUDA float16 with GradScaler; CPU FP32",
    "tf32": False, "deterministic_algorithms": True,
    "ridge_alphas": list(ALPHAS), "ridge_selection": "same320/80 split; ties larger alpha; X/y scaling fit inside training subset",
    "primary": "convnext_nested", "secondary": "0.5*current185_ensemble+0.5*convnext_nested",
    "control": "frozen2688 + Ridge; differs in head/preprocessing/optimizer too, not pure backbone ablation",
    "checkpoint": "last optimizer/RNG checkpoint every5epochs; completed runs reused; outer final weights retained",
    "priority": "both seeds improve; mean primary RMSE gain>=0.5; inspect tails/common failures; not significance",
    "scope": "train-only; no test, outer-label selection, global weight fitting, full500 fit or submission",
}


def fingerprint(value):
    return hashlib.sha256(common.canonical_json(value).encode("utf-8")).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_csv(path, rows):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def versions():
    return {n: importlib.metadata.version(n) for n in
            ("torch", "torchvision", "timm", "numpy", "scipy", "scikit-learn", "Pillow", "safetensors", "huggingface-hub")}


def payload(data):
    references = {}
    for seed in SEEDS:
        r = previous.load_references(seed, data)
        references[str(seed)] = {"files": r["files"], "folds": r["folds"].tolist(),
                                 "current_sha256": common.sha256_array(r["current"])}
    inventory = common.training_inventory(ROOT / "open")
    if inventory["ids"] != data.ids.tolist():
        raise ValueError("Train IDs differ")
    return {"protocol": PROTOCOL, "sources": {**previous.sources(), Path(__file__).name: {
        "path": str(Path(__file__).resolve()), "sha256": common.sha256_file(__file__)}},
        "versions": versions(), "training": previous.train_binding(data), "inventory": inventory,
        "references": references}


def ensure_weights(folder, download=False):
    path = folder / "weights/model.safetensors"
    if not path.exists():
        if not download:
            raise FileNotFoundError("Run prepare to download the pinned public weights")
        from huggingface_hub import hf_hub_download
        for name in ("config.json", "README.md", "model.safetensors"):
            hf_hub_download(REPO, name, revision=REVISION, local_dir=folder / "weights")
    if common.sha256_file(path) != WEIGHT_SHA:
        raise ValueError("Pinned weight SHA256 mismatch")
    cfg = read_json(path.with_name("config.json"))["pretrained_cfg"]
    if cfg["mean"] != MEAN or cfg["std"] != STD:
        raise ValueError("Pretrained normalization differs")
    return path


def prepare(folder):
    data = previous.load_base()
    frozen = payload(data)
    ensure_weights(folder, download=True)
    path = folder / "protocol.json"
    if path.exists():
        if read_json(path)["frozen"] != frozen:
            raise ValueError("Existing protocol differs; use a new experiment directory")
        print("Prepared protocol already matches; unchanged", flush=True)
        return
    write_json(path, {"frozen": frozen, "training_during_preparation": False,
                      "created_unix": time.time(), "checkpoint_sha256": WEIGHT_SHA})
    print(f"Prepared {path}; train500 IDs/images and both references validated; no fitting", flush=True)


def validate(folder):
    data = previous.load_base()
    plan = read_json(folder / "protocol.json")
    if plan["frozen"] != payload(data):
        raise ValueError("Protocol/source/input/environment changed; do not reuse old results")
    weights = ensure_weights(folder)
    return data, plan, weights


def configure_runtime(device_name, seed, threads=4):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    torch.set_num_threads(threads)
    device = torch.device(device_name)
    if device.type not in ("cuda", "cpu"):
        raise ValueError("Only CUDA or CPU supported")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no silent CPU fallback")
    if device.type == "cuda":
        # set_device requires an explicit index; torch.device('cuda') has None.
        index = torch.cuda.current_device() if device.index is None else device.index
        if not 0 <= index < torch.cuda.device_count():
            raise ValueError(f"CUDA device index {index} does not exist")
        device = torch.device("cuda", index)
        torch.cuda.set_device(device)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    seed_all(torch, seed, device)
    return torch, device


def seed_all(torch, seed, device):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def make_model(torch, weights=None):
    import timm
    base = timm.create_model(MODEL, pretrained=False, drop_path_rate=0.)
    if weights is not None:
        from safetensors.torch import load_file
        # Strictly load the full official model before dropping its classifier.
        base.load_state_dict(load_file(str(weights), device="cpu"), strict=True)

    class Regressor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.stem, self.stages = base.stem, base.stages
            self.head = torch.nn.Sequential(torch.nn.LayerNorm(2688), torch.nn.Dropout(.1), torch.nn.Linear(2688, 1))

        def features(self, x):
            x = self.stem(x)
            pieces = []
            for i, stage in enumerate(self.stages):
                x = stage(x)
                if i > 0:
                    if x.ndim != 4 or x.shape[1] != (192, 384, 768)[i-1]:
                        raise RuntimeError("Unexpected ConvNeXt stage shape")
                    # Keep moment computation stable while retaining gradients.
                    f = x.float()
                    pieces.extend((f.mean((2, 3)), f.std((2, 3), correction=0)))
            return torch.cat(pieces, dim=1)

        def forward(self, x):
            return self.head(self.features(x)).squeeze(-1)

    model = Regressor()
    if any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in model.modules()):
        raise RuntimeError("Unexpected BatchNorm")
    return model


def phase_mode(model, warmup):
    model.train()
    model.stem.requires_grad_(not warmup)
    model.stages.requires_grad_(not warmup)
    model.head.requires_grad_(True)
    if warmup:
        model.stem.eval()
        model.stages.eval()


def parameter_groups(model, warmup):
    groups = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("head."):
            lr = .001 if warmup else .0003
        elif name.startswith(("stem.", "stages.0.")):
            lr = 3e-6
        elif name.startswith("stages.1."):
            lr = 6e-6
        elif name.startswith("stages.2."):
            lr = 1e-5
        elif name.startswith("stages.3."):
            lr = 3e-5
        else:
            raise ValueError("Unmapped parameter: " + name)
        decay = .01 if p.ndim > 1 and not name.endswith(".bias") else 0.
        key = (lr, decay)
        groups.setdefault(key, {"params": [], "lr": lr, "base_lr": lr, "weight_decay": decay})["params"].append(p)
    return list(groups.values())


def lr_factor(epoch):
    if not 1 <= epoch <= 100:
        raise ValueError("Epoch must be1..100")
    return 1. if epoch <= 5 else .5 * (1. + math.cos(math.pi * (epoch-6) / 94))


def load_pixels(ids):
    from PIL import Image
    images = []
    for ident in ids:
        with Image.open(ROOT / "open/train" / f"{ident}.png") as im:
            if im.mode != "L" or im.size != (256, 256):
                raise ValueError("Image format changed")
            images.append(np.array(im, dtype=np.uint8))
    return np.stack(images)


def batch_images(torch, pixels, indices, device, rng=None):
    x = torch.from_numpy(pixels[np.asarray(indices)]).to(device=device, dtype=torch.float32).unsqueeze(1) / 255.
    if rng is not None:
        views = rng.integers(0, 8, size=len(indices))
        x = torch.stack([d4_view(torch, item, int(view)) for item, view in zip(x, views)])
    x = x.expand(-1, 3, -1, -1)
    mean = torch.tensor(MEAN, device=device)[None, :, None, None]
    std = torch.tensor(STD, device=device)[None, :, None, None]
    return (x - mean) / std


def d4_view(torch, x, view):
    if not 0 <= view < 8:
        raise ValueError("Invalid D4 view")
    x = torch.rot90(x, view % 4, dims=(-2, -1))
    return x.flip(-1) if view >= 4 else x


def amp_context(torch, device):
    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda")


def predict(torch, model, pixels, indices, device, batch, ym, ys):
    model.eval()
    parts = []
    with torch.inference_mode():
        for start in range(0, len(indices), batch):
            x = batch_images(torch, pixels, indices[start:start+batch], device)
            with amp_context(torch, device):
                p = model(x)
            parts.append(p.float().cpu().numpy().astype(np.float64)*ys + ym)
    values = np.concatenate(parts)
    if not np.isfinite(values).all():
        raise RuntimeError("Nonfinite prediction")
    return values


def optimize_epoch(torch, model, pixels, indices, standardized_y, device, optimizer, scaler,
                   batch, accumulation, profile, rng, warmup):
    phase_mode(model, warmup)
    order = rng.permutation(len(indices))
    total, steps, skipped = 0., 0, 0
    effective = batch*accumulation
    for g in range(0, len(order), effective):
        group = order[g:g+effective]
        optimizer.zero_grad(set_to_none=True)
        for start in range(0, len(group), batch):
            local = group[start:start+batch]
            x = batch_images(torch, pixels, indices[local], device, rng if profile == "d4" else None)
            y = torch.as_tensor(standardized_y[local], dtype=torch.float32, device=device)
            with amp_context(torch, device):
                prediction = model(x)
                sse = (prediction.float()-y).square().sum()
            if not torch.isfinite(sse).item():
                raise RuntimeError("Nonfinite training loss")
            scaler.scale(sse/len(group)).backward()
            total += sse.detach().item()
        scaler.unscale_(optimizer)
        # GradScaler skips overflowing FP16 updates; count them explicitly.
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=device.type != "cuda")
        before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        overflow = scaler.get_scale() < before
        skipped += int(overflow)
        steps += int(not overflow)
        if not overflow and not math.isfinite(float(norm)):
            raise RuntimeError("Nonfinite gradient without a skipped AMP update")
    return total/len(indices), steps, skipped


def scaling(y):
    y = np.asarray(y, dtype=np.float64)
    ym, ys = float(y.mean()), float(y.std(ddof=0))
    if not np.isfinite(y).all() or ys <= 0:
        raise ValueError("Invalid/constant training targets")
    return ym, ys


def split_inner(outer_train, train_targets, seed, fold):
    from sklearn.model_selection import StratifiedShuffleSplit
    if len(outer_train) != 400 or len(train_targets) != 400:
        raise ValueError("Expected400 training examples")
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=80, random_state=seed+1009*fold)
    tr, va = next(splitter.split(np.zeros(400), core.rank_groups(train_targets, 10)))
    return np.asarray(outer_train)[tr], np.asarray(outer_train)[va]


def select_profile(results):
    candidates = []
    for rank, profile in enumerate(PROFILES):
        curve = results[profile]["curve"]
        if [r["epoch"] for r in curve] != list(range(1, 101)):
            raise ValueError("Inner curve must contain all100epochs")
        for row in curve:
            if not math.isfinite(row["valid_rmse"]):
                raise ValueError("Nonfinite inner score")
            candidates.append((row["valid_rmse"], row["epoch"], rank, profile))
    rmse, epoch, _, profile = min(candidates)
    return {"profile": profile, "epoch": epoch, "inner_rmse": rmse}


def atomic_torch_save(torch, path, value):
    tmp = path.with_name(path.name + ".tmp")
    torch.save(value, tmp)
    os.replace(tmp, path)


def completed_result(folder, binding):
    path = folder / "completed.json"
    if not path.exists():
        return None
    value = read_json(path)
    if value["binding"] != binding or value["result_sha256"] != fingerprint(value["result"]):
        raise ValueError("Completed run binding/hash differs")
    for name, sha in value["files"].items():
        if common.sha256_file(folder/name) != sha:
            raise ValueError("Completed artifact changed: " + name)
    return value["result"]


class Runner:
    """No outer labels in this interface; valid_y exists only for internal selection."""
    def __init__(self, torch, device, pixels, weights, plan_hash, batch=4, accumulation=4):
        self.torch, self.device, self.pixels, self.weights = torch, device, pixels, weights
        self.plan_hash, self.batch, self.accumulation = plan_hash, batch, accumulation
        self.runtime = {"device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                        "amp": "float16" if device.type == "cuda" else "disabled", "threads": torch.get_num_threads()}

    def __call__(self, train_idx, train_y, valid_idx, valid_y, profile, epochs, seed, folder, *, purpose="experiment"):
        torch, device = self.torch, self.device
        if profile not in PROFILES or not 1 <= epochs <= 100:
            raise ValueError("Invalid profile/epochs")
        if purpose != "sanity" and np.intersect1d(train_idx, valid_idx).size:
            raise ValueError("Train/validation overlap")
        ym, ys = scaling(train_y)
        binding = {"plan": self.plan_hash, "train": np.asarray(train_idx).tolist(), "valid": np.asarray(valid_idx).tolist(),
                   "train_y": common.sha256_array(np.asarray(train_y)), "valid_y": None if valid_y is None else common.sha256_array(np.asarray(valid_y)),
                   "profile": profile, "epochs": epochs, "seed": seed, "batch": self.batch, "accumulation": self.accumulation,
                   "runtime": self.runtime, "purpose": purpose}
        folder.mkdir(parents=True, exist_ok=True)
        old = completed_result(folder, binding)
        if old is not None:
            print(f"Reused completed run: {folder.name}", flush=True)
            return old
        marker = folder / "binding.json"
        if marker.exists() and read_json(marker) != binding:
            raise ValueError("Interrupted run settings changed; use a separate experiment directory")
        write_json(marker, binding)
        seed_all(torch, seed, device)
        model = make_model(torch, self.weights).to(device)
        rng = np.random.default_rng(seed)
        scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
        history, start_epoch, optimizer = [], 1, None
        resume = folder / "last.pt"
        if resume.exists():
            # Local self-produced checkpoint only. Never load untrusted pickle checkpoints.
            state = torch.load(resume, map_location=device, weights_only=False)
            if state["binding"] != binding:
                raise ValueError("Resume checkpoint binding differs")
            model.load_state_dict(state["model"], strict=True)
            phase_mode(model, state["epoch"] <= 5)
            optimizer = torch.optim.AdamW(parameter_groups(model, state["epoch"] <= 5))
            optimizer.load_state_dict(state["optimizer"])
            scaler.load_state_dict(state["scaler"])
            rng.bit_generator.state = state["numpy_rng"]
            torch.set_rng_state(state["torch_rng"].cpu())
            if device.type == "cuda":
                torch.cuda.set_rng_state(state["cuda_rng"].cpu(), device)
            history, start_epoch = state["curve"], state["epoch"]+1
            print(f"Resumed {folder.name} after epoch{start_epoch-1}", flush=True)
            del state
        target = (np.asarray(train_y)-ym)/ys
        for epoch in range(start_epoch, epochs+1):
            started = time.perf_counter()
            warmup = epoch <= 5
            phase_mode(model, warmup)
            if optimizer is None or epoch == 6:
                optimizer = torch.optim.AdamW(parameter_groups(model, warmup))
            for group in optimizer.param_groups:
                group["lr"] = group["base_lr"]*lr_factor(epoch)
            loss, steps, skipped = optimize_epoch(torch, model, self.pixels, train_idx, target, device, optimizer,
                                                 scaler, self.batch, self.accumulation, profile, rng, warmup)
            train_pred = predict(torch, model, self.pixels, train_idx, device, self.batch, ym, ys)
            train_rmse = float(np.sqrt(np.mean((train_pred-train_y)**2)))
            valid_rmse = None
            if valid_y is not None:
                val = predict(torch, model, self.pixels, valid_idx, device, self.batch, ym, ys)
                valid_rmse = float(np.sqrt(np.mean((val-valid_y)**2)))
            row = {"epoch": epoch, "phase": "head" if warmup else "full", "train_rmse": train_rmse,
                   "valid_rmse": valid_rmse, "augmented_standardized_mse": loss, "optimizer_steps": steps,
                   "amp_skipped_steps": skipped, "lr": [g["lr"] for g in optimizer.param_groups],
                   "seconds": time.perf_counter()-started}
            history.append(row)
            write_json(folder/"curve.json", history)
            print(f"{folder.name} epoch={epoch:03d}/{epochs} train={train_rmse:.5f} inner={valid_rmse} steps={steps} skipped={skipped} seconds={row['seconds']:.1f}", flush=True)
            if epoch % 5 == 0 or epoch == epochs:
                atomic_torch_save(torch, resume, {"binding": binding, "epoch": epoch, "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(), "numpy_rng": rng.bit_generator.state,
                    "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
                    "curve": history})
        result = {"curve": history, "target_mean": ym, "target_std": ys, "prediction": None,
                  "outer_labels_used": False, "runtime": self.runtime}
        files = {}
        if valid_y is None:
            result["prediction"] = predict(torch, model, self.pixels, valid_idx, device, self.batch, ym, ys).tolist()
            final_path = folder / "model.pt"
            atomic_torch_save(torch, final_path, {"model": model.state_dict(), "binding": binding, "target_mean": ym, "target_std": ys})
            files["model.pt"] = common.sha256_file(final_path)
        write_json(folder/"curve.json", history)
        files["curve.json"] = common.sha256_file(folder/"curve.json")
        write_json(folder/"completed.json", {"binding": binding, "result": result, "result_sha256": fingerprint(result), "files": files})
        # Delete only this run's temporary optimizer checkpoint after an atomic completion record.
        if resume.exists():
            resume.unlink()
        del model, optimizer, scaler
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return result


def extract(folder, device_name, batch):
    data, plan, weights = validate(folder)
    plan_hash = fingerprint(plan["frozen"])
    path = folder / "features.npz"
    if path.exists():
        load_features(folder, data, plan_hash)
        print("Verified existing frozen ConvNeXt cache; unchanged", flush=True)
        return
    torch, device = configure_runtime(device_name, 0)
    model = make_model(torch, weights).to(device).eval().requires_grad_(False)
    pixels = load_pixels(data.ids)
    parts = []
    started = time.perf_counter()
    # Frozen Ridge features use FP32. Neural AMP is separately disclosed.
    with torch.inference_mode():
        for start in range(0, len(data.ids), batch):
            idx = np.arange(start, min(start+batch, len(data.ids)))
            f = model.features(batch_images(torch, pixels, idx, device)).cpu().numpy()
            parts.append(f)
            if start == 0 or (start//batch)%10 == 0:
                print(f"Frozen ConvNeXt: {start+len(idx)}/500", flush=True)
    x = np.concatenate(parts)
    if x.shape != (500, 2688) or not np.isfinite(x).all():
        raise ValueError("Invalid frozen features")
    common.ensure_inputs_unchanged(ROOT/"open", plan["frozen"]["inventory"])
    if plan["frozen"] != payload(data):
        raise ValueError("Frozen inputs changed while extracting")
    meta = {"plan": plan_hash, "feature_sha256": common.sha256_array(x), "ids_sha256": common.sha256_array(data.ids),
            "shape": list(x.shape), "seconds": time.perf_counter()-started, "dtype": "float32", "test_accessed": False,
            "weight_sha256": WEIGHT_SHA, "device": str(device), "batch": batch}
    tmp = path.with_name("features.tmp.npz")
    np.savez_compressed(tmp, ids=data.ids, features=x, metadata_json=common.canonical_json(meta))
    os.replace(tmp, path)
    write_json(folder/"feature_metadata.json", meta)
    print(f"Saved {path}; shape={x.shape}", flush=True)


def load_features(folder, data, plan_hash):
    with np.load(folder/"features.npz", allow_pickle=False) as z:
        meta = json.loads(z["metadata_json"].item())
        x, ids = z["features"].copy(), z["ids"].copy()
    if meta["plan"] != plan_hash or meta["feature_sha256"] != common.sha256_array(x) or meta["ids_sha256"] != common.sha256_array(ids):
        raise ValueError("Frozen feature binding/hash mismatch")
    if not np.array_equal(ids, data.ids) or x.shape != (500, 2688) or x.dtype != np.float32 or not np.isfinite(x).all():
        raise ValueError("Frozen feature ID/shape/dtype/value mismatch")
    if meta != read_json(folder/"feature_metadata.json"):
        raise ValueError("Frozen feature metadata mismatch")
    return x


def ridge_fold(x, y, ti, vi, inner_train, inner_valid):
    from threadpoolctl import threadpool_limits
    records = []
    with threadpool_limits(limits=4):
        for alpha in ALPHAS:
            model = core.make_estimator(alpha)
            model.fit(x[inner_train], y[inner_train])
            pred = model.predict(x[inner_valid])
            records.append({"alpha": alpha, "inner_mse": float(np.mean((pred-y[inner_valid])**2))})
        choice = min(records, key=lambda r: (r["inner_mse"], -r["alpha"]))
        model = core.make_estimator(choice["alpha"])
        model.fit(x[ti], y[ti])
        return np.asarray(model.predict(x[vi])), {"choice": choice, "search": records}


def neural_fold(runner, y, folds, seed, fold, folder):
    ti, vi = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
    tr, va = split_inner(ti, y[ti], seed, fold)
    results = {p: runner(tr, y[tr], va, y[va], p, 100, seed+1009*fold+100000,
                         folder/f"fold{fold}_inner_{p}") for p in PROFILES}
    choice = select_profile(results)
    fitted = runner(ti, y[ti], vi, None, choice["profile"], choice["epoch"], seed+1009*fold+200000,
                    folder/f"fold{fold}_outer")
    return vi, np.asarray(fitted["prediction"]), {"fold": fold, "choice": choice, "inner_train": tr.tolist(), "inner_valid": va.tolist()}


def report(folder, data, refs, cnn, ridge, selections, ridge_selection, seed, binding):
    predictions = {"current": refs["current"], "convnext_frozen_ridge": ridge,
                   "convnext_nested": cnn, "current_convnext_equal": .5*(refs["current"]+cnn)}
    if any(p.shape != data.y.shape or not np.isfinite(p).all() for p in predictions.values()):
        raise ValueError("Incomplete/nonfinite OOF")
    base = core.metrics(data.y, refs["current"])["rmse"]
    metrics = {n: {**core.metrics(data.y, p), "gain_vs_current": base-core.metrics(data.y, p)["rmse"]}
               for n, p in predictions.items()}
    worst = set(np.argsort((refs["current"]-data.y)**2, kind="stable")[-20:].tolist())
    deciles, noise = core.rank_groups(data.y, 10), core.rank_groups(data.noise, 4)
    rows = [{"ID": str(data.ids[i]), "hardness": float(data.y[i]), "fold": int(refs["folds"][i]),
             **{n: float(p[i]) for n, p in predictions.items()}} for i in range(len(data.ids))]
    write_csv(folder/"oof.csv", rows)
    diagnostics = []
    for name, pred in predictions.items():
        metrics[name]["worst20_overlap"] = len(worst & set(np.argsort((pred-data.y)**2, kind="stable")[-20:].tolist()))
        masks = {"tail20": (deciles == 0) | (deciles == 9), "middle80": (deciles > 0) & (deciles < 9), "baseline_worst20": np.isin(np.arange(500), list(worst))}
        masks.update({f"fold{f}": refs["folds"] == f for f in range(5)})
        masks.update({f"noiseQ{q+1}": noise == q for q in range(4)})
        for region, mask in masks.items():
            diagnostics.append({"model": name, "group": region, **core.metrics(data.y[mask], pred[mask])})
    write_csv(folder/"diagnostics.csv", diagnostics)
    errors = [{"ID": str(data.ids[i]), "hardness": float(data.y[i]), "current_error": float(refs["current"][i]-data.y[i]),
               "convnext_error": float(cnn[i]-data.y[i]), "blend_error": float(predictions["current_convnext_equal"][i]-data.y[i])}
              for i in sorted(range(500), key=lambda j: -(refs["current"][j]-data.y[j])**2)]
    write_csv(folder/"per_image_errors.csv", errors)
    summary = {"seed": seed, "binding": binding, "metrics": metrics, "neural_selection": selections, "ridge_selection": ridge_selection,
               "primary": PROTOCOL["primary"], "outer_label_selection": False, "independent_holdout": False,
               "limitations": "Same500 repeatedly researched; single80-image inner holdout; not Public score; frozen Ridge uses different head/scaler/FP32"}
    write_json(folder/"summary.json", summary)
    lines = [f"# ConvNeXt full fine-tuning · seed{seed}", "", "| Model | OOF RMSE | Gain vs current |", "|---|---:|---:|"]
    lines += [f"| {n} | {m['rmse']:.6f} | {m['gain_vs_current']:+.6f} |" for n, m in metrics.items()]
    lines += ["", "Primary: convnext_nested. Fixed secondary: current_convnext_equal.",
              "Internal profile/epoch selection only; outer100 labels are used only after predictions.",
              "Same500 images, not an independent holdout or an estimate of Public leaderboard score.", "", "## Internal selections"]
    lines += [f"- fold{s['fold']}: {s['choice']['profile']}, epoch{s['choice']['epoch']}, inner RMSE{s['choice']['inner_rmse']:.6f}" for s in selections]
    (folder/"report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    outputs = ("oof.csv", "diagnostics.csv", "per_image_errors.csv", "summary.json", "report.md")
    write_json(folder/"completed_evaluation.json", {"binding": binding, "files": {n: common.sha256_file(folder/n) for n in outputs}})
    for name in ("convnext_frozen_ridge", "convnext_nested", "current_convnext_equal"):
        print(f"{name}: RMSE={metrics[name]['rmse']:.6f}; gain vs current={metrics[name]['gain_vs_current']:+.6f}", flush=True)
    print(f"Report: {folder/'report.md'}", flush=True)


def compare(folder):
    """Audit saved OOF and aggregate seed differences; never fit or average OOF predictions."""
    data, plan, _ = validate(folder)
    plan_hash = fingerprint(plan["frozen"])
    results = []
    for seed in SEEDS:
        dest = folder/f"seed{seed}"
        done = read_json(dest/"completed_evaluation.json")
        if done["binding"]["plan"] != plan_hash:
            raise ValueError("Evaluation plan differs")
        for name, sha in done["files"].items():
            if common.sha256_file(dest/name) != sha:
                raise ValueError("Evaluation output changed: " + name)
        refs = previous.load_references(seed, data)
        rows = core.align_rows(core.read_rows(dest/"oof.csv", {"ID", "hardness", "fold", "convnext_nested", "current_convnext_equal", "current"}), data.ids, dest/"oof.csv")
        np.testing.assert_array_equal(core.finite_column(rows, "hardness", dest), data.y)
        np.testing.assert_array_equal([int(r["fold"]) for r in rows], refs["folds"])
        current = core.finite_column(rows, "current", dest)
        pred = core.finite_column(rows, "convnext_nested", dest)
        blend = core.finite_column(rows, "current_convnext_equal", dest)
        np.testing.assert_allclose(current, refs["current"], atol=1e-10, rtol=0)
        np.testing.assert_allclose(blend, .5*(current+pred), atol=1e-10, rtol=0)
        rmse = core.metrics(data.y, current)["rmse"]
        results.append({"seed": seed, "current_rmse": rmse, "primary_rmse": core.metrics(data.y, pred)["rmse"],
                        "primary_gain": rmse-core.metrics(data.y, pred)["rmse"], "secondary_gain": rmse-core.metrics(data.y, blend)["rmse"]})
    mean_gain = float(np.mean([r["primary_gain"] for r in results]))
    value = {"per_seed": results, "mean_primary_gain": mean_gain,
             "passes_numeric_priority_gate": all(r["primary_gain"] > 0 for r in results) and mean_gain >= .5,
             "additional_review_required": "tails/common failures in diagnostics and per_image_errors; numeric gate alone is not adoption",
             "aggregation": "mean per-seed OOF RMSE differences; same500; no independent inference or prediction averaging"}
    write_json(folder/"comparison.json", value)
    print(json.dumps(value, ensure_ascii=False, indent=2), flush=True)


def train(folder, device_name, seed, batch, accumulation):
    data, plan, weights = validate(folder)
    plan_hash = fingerprint(plan["frozen"])
    x = load_features(folder, data, plan_hash)
    refs = previous.load_references(seed, data)
    dest = folder / f"seed{seed}"
    dest.mkdir(parents=True, exist_ok=True)
    binding = {"plan": plan_hash, "seed": seed, "batch": batch, "accumulation": accumulation,
               "device": device_name, "features_sha256": common.sha256_file(folder/"features.npz")}
    done = dest/"completed_evaluation.json"
    if done.exists():
        record = read_json(done)
        if record["binding"] != binding or any(common.sha256_file(dest/n) != sha for n, sha in record["files"].items()):
            raise ValueError("Completed evaluation changed")
        print(f"Verified complete evaluation; {dest/'report.md'}", flush=True)
        if all((folder/f"seed{s}/completed_evaluation.json").exists() for s in SEEDS):
            compare(folder)
        return
    torch, device = configure_runtime(device_name, seed)
    runner = Runner(torch, device, load_pixels(data.ids), weights, plan_hash, batch, accumulation)
    cnn, ridge = np.full(500, np.nan), np.full(500, np.nan)
    choices, ridge_choices = [], []
    for fold in range(5):
        ti, vi = np.flatnonzero(refs["folds"] != fold), np.flatnonzero(refs["folds"] == fold)
        tr, va = split_inner(ti, data.y[ti], seed, fold)
        ridge_path = dest/f"fold{fold}_ridge.json"
        if ridge_path.exists():
            saved = read_json(ridge_path)
            if saved["binding"] != binding or saved["sha256"] != fingerprint(saved["result"]):
                raise ValueError("Ridge fold cache changed")
            result = saved["result"]
            rp, rc = np.asarray(result["prediction"]), result["selection"]
        else:
            rp, rc = ridge_fold(x, data.y, ti, vi, tr, va)
            result = {"prediction": rp.tolist(), "selection": rc}
            write_json(ridge_path, {"binding": binding, "result": result, "sha256": fingerprint(result)})
        ridge[vi] = rp
        ridge_choices.append({"fold": fold, **rc})
        vi, pred, choice = neural_fold(runner, data.y, refs["folds"], seed, fold, dest)
        cnn[vi] = pred
        choices.append(choice)
        print(f"Outer{fold}: internally selected {choice['choice']}; predictions stored, no outer score used", flush=True)
    common.ensure_inputs_unchanged(ROOT/"open", plan["frozen"]["inventory"])
    # Recheck pinned sources, references and dependencies before publishing the OOF report.
    if plan["frozen"] != payload(data):
        raise ValueError("Frozen inputs changed while training")
    report(dest, data, refs, cnn, ridge, choices, ridge_choices, seed, binding)
    if all((folder/f"seed{s}/completed_evaluation.json").exists() for s in SEEDS):
        compare(folder)


def sanity(folder, device_name, batch, accumulation):
    data, plan, weights = validate(folder)
    refs = previous.load_references(42, data)
    ti = np.flatnonzero(refs["folds"] != 0)
    tr, _ = split_inner(ti, data.y[ti], 42, 0)
    idx = np.random.default_rng(314159).choice(tr, size=32, replace=False)
    torch, device = configure_runtime(device_name, 314159)
    runner = Runner(torch, device, load_pixels(data.ids), weights, fingerprint(plan["frozen"]), batch, accumulation)
    result = runner(idx, data.y[idx], idx, data.y[idx], "none", 100, 314159, folder/"sanity32", purpose="sanity")
    first, last = result["curve"][0]["train_rmse"], result["curve"][-1]["train_rmse"]
    print(f"SANITY only: train32 RMSE epoch1={first:.6f}, epoch100={last:.6f}. Not validation/generalization performance.", flush=True)


def check(device_name):
    torch, device = configure_runtime(device_name, 123)
    model = make_model(torch).to(device)
    phase_mode(model, False)
    x = torch.randn(1, 3, 256, 256, device=device)
    with amp_context(torch, device):
        loss = (model(x).float()-1).square().mean()
    loss.backward()
    for prefix in ("stem.", "stages.0.", "stages.1.", "stages.2.", "stages.3.", "head."):
        values = [p.grad for name, p in model.named_parameters() if name.startswith(prefix)]
        if not values or any(g is None or not torch.isfinite(g).all() for g in values) or not any(torch.count_nonzero(g).item() for g in values):
            raise RuntimeError("Missing/nonfinite gradients in " + prefix)
    print(f"Synthetic forward/backward passed on {device}; all stages connected; params={sum(p.numel() for p in model.parameters())}; no competition images/weights", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "validate", "check", "extract", "sanity", "train", "compare"))
    parser.add_argument("--output-dir", type=common.project_path, default=OUT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, choices=SEEDS, default=42)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--accumulation", type=int, default=4)
    args = parser.parse_args(argv)
    if args.batch_size <= 0 or args.accumulation <= 0 or args.batch_size*args.accumulation != 16:
        parser.error("Training effective batch must remain16 (default4x4, memory fallback2x8)")
    if args.command == "prepare":
        prepare(args.output_dir)
    elif args.command == "validate":
        validate(args.output_dir)
        print("Protocol, images, references, source hashes, environment and weights matched; no fitting")
    elif args.command == "check":
        check(args.device)
    elif args.command == "extract":
        extract(args.output_dir, args.device, args.batch_size)
    elif args.command == "sanity":
        sanity(args.output_dir, args.device, args.batch_size, args.accumulation)
    elif args.command == "compare":
        compare(args.output_dir)
    else:
        train(args.output_dir, args.device, args.seed, args.batch_size, args.accumulation)


if __name__ == "__main__":
    main()
