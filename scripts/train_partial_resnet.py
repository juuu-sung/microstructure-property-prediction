"""Train-only ResNet18 layer4/head development experiment, run by the user.

Outer validation selects the best epoch and early stopping, so reported OOF is
DEVELOPMENT validation, not an unbiased nested-CV estimate. No test inference,
submission, or final fit on all 500 images is implemented. --validate-only checks
train inputs and folds without importing torch, using a GPU, or downloading.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

if __package__:
    from . import extract_frozen_resnet as extraction
else:
    import extract_frozen_resnet as extraction

ROOT = extraction.ROOT
REVISION = extraction.DEFAULT_REVISION
DEVELOPMENT_NOTE = "Development OOF: outer validation chooses checkpoint/early stopping; not unbiased nested CV."


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def write_csv(path, rows, fieldnames=None):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def validate_fold_rows(ids, rows, seed):
    if len(rows) != len(ids) or len({row.get("ID") for row in rows}) != len(rows):
        raise ValueError("Fold CSV has duplicate or missing IDs")
    by_id = {row["ID"]: row for row in rows}
    if set(by_id) != set(ids):
        raise ValueError("Fold CSV IDs must exactly match train.csv")
    try:
        folds = [int(by_id[ident]["fold"]) for ident in ids]
        if any("seed" in row and int(row["seed"]) != seed for row in rows):
            raise ValueError("Fold CSV seed does not match --seed")
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid fold/seed values: {exc}") from exc
    if len(ids) % 5 or set(folds) != set(range(5)) or any(folds.count(f) != len(ids) // 5 for f in range(5)):
        raise ValueError("Expected five equal outer folds labelled 0..4")
    return folds


def load_training_inputs(data_dir, folds_path, seed, expected_count=500):
    inventory = extraction.training_inventory(data_dir, expected_count=expected_count)
    with (data_dir / "train.csv").open(newline="", encoding="utf-8-sig") as handle:
        train = list(csv.DictReader(handle))
    with folds_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        header = reader.fieldnames or []
        if len(header) != len(set(header)) or not {"ID", "fold"} <= set(header):
            raise ValueError("Fold CSV requires unique ID/fold headers")
        folds = validate_fold_rows(inventory["ids"], list(reader), seed)
    y = [float(row["hardness"]) for row in train]
    if [row["ID"] for row in train] != inventory["ids"] or not all(math.isfinite(value) for value in y):
        raise ValueError("train.csv changed after validation or has invalid targets")
    if extraction.sha256_file(data_dir / "train.csv") != inventory["train_csv_sha256"]:
        raise ValueError("train.csv changed during validation")
    return inventory, y, folds


class LabeledTrainDataset:
    def __init__(self, data_dir, ids, targets):
        self.images = extraction.TrainImageDataset(data_dir, ids, extraction.EXPECTED_MEAN, extraction.EXPECTED_STD)
        self.targets = list(targets)
        if len(self.images) != len(self.targets):
            raise ValueError("Image/target count mismatch")
    def __len__(self):
        return len(self.targets)
    def __getitem__(self, index):
        return self.images[index], self.targets[index]


def is_layer4(name):
    return name == "layer4" or name.startswith("layer4.") or name.startswith("layer4_")


def build_model(torch, timm, resolve_config, pooling="multiscale", pretrained=True):
    if pooling not in ("multiscale", "gap"):
        raise ValueError("pooling must be multiscale or gap")
    identifier = f"hf-hub:{extraction.HF_REPO}@{REVISION}" if pretrained else extraction.MODEL_NAME
    backbone = timm.create_model(identifier, pretrained=pretrained, features_only=True, out_indices=(2, 3, 4))
    if list(backbone.feature_info.channels()) != extraction.EXPECTED_CHANNELS:
        raise RuntimeError("Unexpected ResNet18 feature channels")
    official = resolve_config(backbone)
    if list(official["mean"]) != extraction.EXPECTED_MEAN or list(official["std"]) != extraction.EXPECTED_STD:
        raise RuntimeError("Pretrained normalization does not match the fixed protocol")
    class PartialRegressor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = backbone
            self.head = torch.nn.Linear(1792 if pooling == "multiscale" else 512, 1)
            self.pooling = pooling
            self._finetune = False
        def forward(self, inputs):
            maps = [feature.float() for feature in self.backbone(inputs)]
            gap, multiscale = extraction.pool_feature_maps(torch, maps)
            return self.head(multiscale if self.pooling == "multiscale" else gap).squeeze(-1)
    model = PartialRegressor()
    configure_trainable(torch, model, finetune=False)
    return model


def configure_trainable(torch, model, finetune):
    model.backbone.requires_grad_(False)
    model.head.requires_grad_(True)
    last_parameters = [(name, parameter) for name, parameter in model.backbone.named_parameters() if is_layer4(name)]
    if not last_parameters:
        raise RuntimeError("Could not identify layer4 parameters in the feature backbone")
    if finetune:
        for _, parameter in last_parameters:
            parameter.requires_grad_(True)
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()
            for parameter in module.parameters(recurse=False):
                parameter.requires_grad_(False)
    model._finetune = bool(finetune)
    enforce_training_mode(torch, model, training=True)


def enforce_training_mode(torch, model, training=True):
    model.train(training)
    model.backbone.eval()
    if training and getattr(model, "_finetune", False):
        for name, module in model.backbone.named_children():
            if is_layer4(name):
                module.train()
    model.head.train(training)
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()
            if any(parameter.requires_grad for parameter in module.parameters(recurse=False)):
                raise RuntimeError("BatchNorm affine parameters unexpectedly require gradients")
    for name, parameter in model.backbone.named_parameters():
        if parameter.requires_grad and (not getattr(model, "_finetune", False) or not is_layer4(name)):
            raise RuntimeError(f"Unexpected trainable backbone parameter: {name}")


def parameter_groups(model, head_lr=3e-4, backbone_lr=1e-5, weight_decay=1e-4):
    groups = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        family = "head" if name.startswith("head.") else "layer4"
        decay = weight_decay if parameter.ndim > 1 and not name.endswith("bias") else 0.0
        key = (family, decay)
        groups.setdefault(key, {"params": [], "lr": head_lr if family == "head" else backbone_lr,
                                "weight_decay": decay, "name": f"{family}_{'decay' if decay else 'no_decay'}"})["params"].append(parameter)
    if not groups:
        raise ValueError("No trainable parameters")
    return list(groups.values())


def accumulation_group_samples(batch_sizes, accumulation_steps):
    if accumulation_steps < 1 or not batch_sizes or any(size < 1 for size in batch_sizes):
        raise ValueError("Positive batch sizes and accumulation_steps are required")
    totals = []
    for start in range(0, len(batch_sizes), accumulation_steps):
        group = batch_sizes[start:start + accumulation_steps]
        totals.extend([sum(group)] * len(group))
    return totals


def train_epoch(torch, model, loader, optimizer, device, target_mean, target_std,
                accumulation_steps, amp=False, scaler=None):
    if target_std <= 0 or not math.isfinite(target_std):
        raise ValueError("Training target standard deviation must be positive and finite")
    enforce_training_mode(torch, model, training=True)
    n = len(loader.dataset)
    if loader.drop_last or not loader.batch_size:
        raise ValueError("Use a fixed batch_size and drop_last=False")
    batch_sizes = [min(loader.batch_size, n - start) for start in range(0, n, loader.batch_size)]
    denominators = accumulation_group_samples(batch_sizes, accumulation_steps)
    scaler = scaler or torch.amp.GradScaler("cuda", enabled=amp)
    optimizer.zero_grad(set_to_none=True)
    total_sse, seen, optimizer_steps = 0.0, 0, 0
    for step, (inputs, targets) in enumerate(loader):
        inputs = inputs.to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda")
        targets = targets.to(device=device, dtype=torch.float32)
        standardized = (targets - target_mean) / target_std
        with torch.autocast(device_type=device.type, enabled=amp, dtype=torch.float16 if device.type == "cuda" else torch.bfloat16):
            prediction = model(inputs)
            batch_sse = (prediction.float() - standardized).square().sum()
            loss = batch_sse / denominators[step]
        if not torch.isfinite(loss).item():
            raise RuntimeError("Nonfinite training loss")
        scaler.scale(loss).backward()
        if (step + 1) % accumulation_steps == 0 or step + 1 == len(batch_sizes):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1
        total_sse += float(batch_sse.detach().item())
        seen += len(targets)
    if seen != n:
        raise RuntimeError("Training loader did not cover each requested sample once")
    return {"std_target_mse": total_sse / seen, "optimizer_steps": optimizer_steps}


def evaluate(torch, model, loader, device, target_mean, target_std):
    import numpy as np
    enforce_training_mode(torch, model, training=False)
    predictions, targets_all = [], []
    with torch.inference_mode():
        for inputs, targets in loader:
            inputs = inputs.to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda")
            prediction = model(inputs).float().cpu().numpy().astype(np.float64) * target_std + target_mean
            predictions.append(prediction)
            targets_all.append(targets.cpu().numpy().astype(np.float64))
    prediction, target = np.concatenate(predictions), np.concatenate(targets_all)
    if not np.isfinite(prediction).all():
        raise RuntimeError("Nonfinite validation predictions")
    return prediction, float(np.sqrt(np.mean((prediction - target) ** 2)))


def update_monitor(best, rmse, epoch, phase, patience):
    if not math.isfinite(rmse) or phase not in ("warmup", "finetune") or patience < 1:
        raise ValueError("Invalid checkpoint monitoring input")
    state = dict(best or {"best_rmse": math.inf, "best_epoch": None, "bad_finetune_epochs": 0})
    improved = rmse < state["best_rmse"]
    if phase == "warmup" or state.get("phase") != phase:
        state["bad_finetune_epochs"] = 0
    if improved:
        state.update(best_rmse=rmse, best_epoch=epoch, best_phase=phase, bad_finetune_epochs=0)
    elif phase == "finetune":
        state["bad_finetune_epochs"] += 1
    state.update(improved=improved, phase=phase, stop=phase == "finetune" and state["bad_finetune_epochs"] >= patience)
    return state


def checkpoint_payload(torch, model, mean, std, epoch, fold, train_ids, val_ids, config):
    return {"model_state": {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()},
            "head_state": {name: tensor.detach().cpu().clone() for name, tensor in model.head.state_dict().items()},
            "target_mean": float(mean), "target_std": float(std), "epoch": epoch, "fold": fold,
            "train_ids": list(train_ids), "validation_ids": list(val_ids), "revision": REVISION,
            "config": config, "evaluation_note": DEVELOPMENT_NOTE, "resume_supported": False}


def save_checkpoint(torch, path, payload):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def make_loader(torch, dataset, args, seed, shuffle):
    return torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle, drop_last=False,
        num_workers=args.num_workers, pin_memory=args.device.startswith("cuda"),
        worker_init_fn=extraction.seed_worker, generator=torch.Generator().manual_seed(seed))


def run_fold(np, torch, timm, resolve_config, args, data_dir, output_dir, inventory, y, folds, fold, config, device):
    fold_dir = output_dir / f"fold{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    write_json(fold_dir / "status.json", {"status": "running", "config_hash": config["config_hash"]})
    aggregate_results(output_dir, inventory["ids"], y, folds, config)
    fold_seed = args.seed + fold
    random.seed(fold_seed)
    np.random.seed(fold_seed)
    torch.manual_seed(fold_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(fold_seed)
    train_idx = [i for i, value in enumerate(folds) if value != fold]
    valid_idx = [i for i, value in enumerate(folds) if value == fold]
    ids = inventory["ids"]
    train_ids, val_ids = [ids[i] for i in train_idx], [ids[i] for i in valid_idx]
    mean, std = float(np.mean([y[i] for i in train_idx])), float(np.std([y[i] for i in train_idx]))
    if not math.isfinite(std) or std <= 0:
        raise ValueError("Fold training targets have zero or invalid standard deviation")
    train_dataset = LabeledTrainDataset(data_dir, train_ids, [y[i] for i in train_idx])
    valid_dataset = LabeledTrainDataset(data_dir, val_ids, [y[i] for i in valid_idx])
    train_loader = make_loader(torch, train_dataset, args, fold_seed, True)
    valid_loader = make_loader(torch, valid_dataset, args, fold_seed, False)
    model = build_model(torch, timm, resolve_config, args.pooling, pretrained=True)
    pretrained_hash = extraction.feature_state_hash(model.backbone)
    model = model.to(device=device, dtype=torch.float32)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)
    monitor, history, epoch, start = None, [], 0, time.perf_counter()
    for phase, phase_epochs in [("warmup", args.warmup_epochs), ("finetune", args.epochs)]:
        configure_trainable(torch, model, finetune=phase == "finetune")
        optimizer = torch.optim.AdamW(parameter_groups(model), foreach=False)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs) if phase == "finetune" else None
        for phase_epoch in range(1, phase_epochs + 1):
            epoch += 1
            epoch_start = time.perf_counter()
            learning_rates = {group["name"]: group["lr"] for group in optimizer.param_groups}
            training = train_epoch(torch, model, train_loader, optimizer, device, mean, std, args.accumulation_steps, args.amp, scaler)
            _, val_rmse = evaluate(torch, model, valid_loader, device, mean, std)
            monitor = update_monitor(monitor, val_rmse, epoch, phase, args.patience)
            if monitor["improved"]:
                payload = checkpoint_payload(torch, model, mean, std, epoch, fold, train_ids, val_ids, config)
                payload.update(phase=phase, val_rmse=val_rmse, pretrained_state_sha256=pretrained_hash)
                save_checkpoint(torch, fold_dir / "best.pt", payload)
            history.append({"epoch": epoch, "phase": phase, "phase_epoch": phase_epoch, **training,
                "val_rmse": val_rmse, "best_val_rmse": monitor["best_rmse"], "best_epoch": monitor["best_epoch"],
                "bad_finetune_epochs": monitor["bad_finetune_epochs"], "head_lr": learning_rates["head_decay"],
                "layer4_lr": learning_rates.get("layer4_decay", 0.0), "seconds": time.perf_counter() - epoch_start})
            write_csv(fold_dir / "history.csv", history)
            print(f"fold={fold} {phase}={phase_epoch}/{phase_epochs} loss={training['std_target_mse']:.5f} val={val_rmse:.5f} best={monitor['best_rmse']:.5f} steps={training['optimizer_steps']} time={history[-1]['seconds']:.1f}s", flush=True)
            if scheduler is not None:
                scheduler.step()
            if monitor["stop"]:
                break
    checkpoint = torch.load(fold_dir / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    prediction, val_rmse = evaluate(torch, model, valid_loader, device, mean, std)
    train_eval_loader = make_loader(torch, train_dataset, args, fold_seed, False)
    _, train_rmse = evaluate(torch, model, train_eval_loader, device, mean, std)
    extraction.ensure_inputs_unchanged(data_dir, inventory)
    write_csv(fold_dir / "oof_fold.csv", [{"ID": ident, "hardness": y[index], "fold": fold, "prediction": float(pred)} for ident, index, pred in zip(val_ids, valid_idx, prediction)])
    result = {"fold": fold, "train_n": len(train_ids), "validation_n": len(val_ids), "val_rmse": val_rmse,
        "train_rmse_at_best_checkpoint": train_rmse, "best_epoch": monitor["best_epoch"], "best_phase": monitor["best_phase"],
        "epochs_run": epoch, "target_mean": mean, "target_std": std, "elapsed_seconds": time.perf_counter() - start,
        "pretrained_state_sha256": pretrained_hash, "config_hash": config["config_hash"], "config": config,
        "evaluation_note": DEVELOPMENT_NOTE}
    write_json(fold_dir / "metrics.json", result)
    write_json(fold_dir / "status.json", {"status": "complete", "config_hash": config["config_hash"]})
    del model, optimizer, train_loader, valid_loader, train_eval_loader
    if device.type == "cuda":
        torch.cuda.empty_cache()


def aggregate_results(output_dir, ids, y, fold_assignments, config):
    completed, predictions = [], {}
    for fold in range(5):
        folder = output_dir / f"fold{fold}"
        if not (folder / "status.json").is_file():
            continue
        status = json.loads((folder / "status.json").read_text(encoding="utf-8"))
        if status["status"] != "complete":
            continue
        metrics = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
        if status["config_hash"] != config["config_hash"] or metrics["config_hash"] != config["config_hash"]:
            raise ValueError("Cannot aggregate folds with different configurations; use a new output directory")
        with (folder / "oof_fold.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        expected = {ident: float(target) for ident, target, assigned in zip(ids, y, fold_assignments) if assigned == fold}
        if len(rows) != len(expected) or {r["ID"] for r in rows} != set(expected):
            raise ValueError(f"fold{fold} OOF IDs are missing or duplicated")
        for row in rows:
            ident, prediction = row["ID"], float(row["prediction"])
            if ident in predictions or int(row["fold"]) != fold or float(row["hardness"]) != expected[ident] or not math.isfinite(prediction):
                raise ValueError("Invalid fold OOF alignment or prediction")
            predictions[ident] = prediction
        completed.append(metrics)
    if completed and len({m["pretrained_state_sha256"] for m in completed}) != 1:
        raise ValueError("Loaded pretrained backbone states differ across folds")
    rows = [{"ID": ident, "hardness": target, "fold": fold, "prediction": predictions[ident]} for ident, target, fold in zip(ids, y, fold_assignments) if ident in predictions]
    full = len(rows) == len(ids) and len(completed) == 5
    rmse = math.sqrt(sum((r["prediction"] - r["hardness"]) ** 2 for r in rows) / len(rows)) if rows else None
    summary = {"status": "complete_5fold_development" if full else "partial_development" if rows else "no_completed_folds", "coverage_n": len(rows),
        "total_train_n": len(ids), "completed_folds": [m["fold"] for m in completed], "evaluation_note": DEVELOPMENT_NOTE,
        "development_oof_rmse": rmse if full else None, "partial_heldout_rmse": None if full else rmse,
        "config": config, "fold_metrics": completed, "no_test_access": True, "no_all_data_final_fit": True}
    write_csv(output_dir / "oof.csv", rows, fieldnames=["ID", "hardness", "fold", "prediction"])
    write_json(output_dir / "summary.json", summary)
    title = "5-fold development OOF" if full else f"부분 development 검증 ({len(rows)}/{len(ids)}장)"
    score_line = f"검증 RMSE: **{rmse:.6f}**" if rmse is not None else "현재 설정에서 완료된 fold가 없어 검증 점수를 표시하지 않습니다."
    lines = [f"# {title}", "", score_line, "",
        "각 outer validation에서 최선 epoch와 조기 종료를 선택했습니다. 이 점수는 개발용 진단이며 nested CV처럼 선택 편향이 제거된 점수가 아닙니다.", "",
        "| fold | best epoch | phase | validation RMSE | train RMSE at best |", "|---|---:|---|---:|---:|"]
    lines += [f"| {m['fold']} | {m['best_epoch']} | {m['best_phase']} | {m['val_rmse']:.5f} | {m['train_rmse_at_best_checkpoint']:.5f} |" for m in completed]
    lines += ["", "전체 원본 256×256, 무증강, 고정 ImageNet 정규화를 사용했습니다. target 평균·표준편차는 각 fold의 학습 400장에서만 추정했습니다.",
        "BatchNorm 통계와 affine 파라미터는 고정했습니다. head warmup 후 layer4와 head만 AdamW로 학습했습니다.",
        "정규화 target의 MSE로 학습하고, 원래 경도 단위의 전체 제곱오차를 합쳐 RMSE를 계산했습니다.",
        "test 추론, 제출 파일 생성, 전체 500장 최종 학습은 수행하지 않았습니다.", ""]
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default="open")
    parser.add_argument("--folds-file", default="reports/folds.csv")
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--output-dir", default="reports/partial_resnet18/seed42")
    parser.add_argument("--pooling", choices=("multiscale", "gap"), default="multiscale")
    parser.add_argument("--epochs", type=int, default=60, help="Maximum finetuning epochs, after warmup")
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=10, help="Non-improving finetune epochs; warmup does not consume patience")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--accumulation-steps", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true", help="Optional CUDA mixed precision; default is FP32")
    parser.add_argument("--validate-only", action="store_true", help="Read/validate train data and fold CSV only; no torch, GPU, or downloads")
    parser.add_argument("--overwrite", action="store_true", help="Explicitly replace results only for the requested fold directories")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if any(getattr(args, key) < 1 for key in ("epochs", "patience", "batch_size", "accumulation_steps", "threads")) or args.warmup_epochs < 0 or args.num_workers < 0:
        parser.error("Epochs/patience/batch/accumulation/threads must be positive; warmup/workers must be nonnegative")
    if not 0 <= args.seed <= 2**32 - 5 or not args.folds or len(set(args.folds)) != len(args.folds) or any(f not in range(5) for f in args.folds):
        parser.error("seed must be 0..4294967291; folds must be unique values in 0..4")
    if args.amp and not args.device.startswith("cuda"):
        parser.error("--amp requires CUDA; default FP32 also supports CPU")
    data_dir, output_dir = extraction.project_path(args.data_dir), extraction.project_path(args.output_dir)
    folds_path = extraction.project_path(args.folds_file)
    if not args.validate_only:
        if output_dir.exists() and not output_dir.is_dir():
            raise ValueError("--output-dir must be a directory")
        for fold in args.folds:
            folder = output_dir / f"fold{fold}"
            if folder.exists() and not folder.is_dir():
                raise ValueError(f"Fold output path is not a directory: {folder}")
            if folder.exists() and (not folder.is_dir() or any(folder.iterdir())) and not args.overwrite:
                raise ValueError(f"{folder} already contains results; use a new output directory or explicit --overwrite")
    inventory, y, folds = load_training_inputs(data_dir, folds_path, args.seed)
    if args.validate_only:
        print(json.dumps({"status": "validated", "n_train": len(y), "fold_counts": {f: folds.count(f) for f in range(5)},
                          "train_csv_sha256": inventory["train_csv_sha256"], "requested_folds": args.folds,
                          "torch_imported": False, "test_accessed": False}, indent=2))
        return 0
    np, torch, timm, resolve_config, versions = extraction.import_runtime()
    device, device_info = extraction.configure_runtime(torch, np, args)
    config = {"model": extraction.HF_REPO, "revision": REVISION, "pooling": args.pooling,
        "train_csv_sha256": inventory["train_csv_sha256"], "image_manifest_sha256": inventory["image_manifest_sha256"],
        "folds_sha256": extraction.sha256_file(folds_path), "seed": args.seed, "epochs": args.epochs,
        "warmup_epochs": args.warmup_epochs, "patience": args.patience, "batch_size": args.batch_size,
        "accumulation_steps": args.accumulation_steps, "head_lr": 3e-4, "layer4_lr": 1e-5, "weight_decay": 1e-4,
        "warmup_scheduler": None, "finetune_scheduler": "cosine over requested finetune epochs",
        "optimizer_reset_after_warmup": True, "amp": args.amp, "deterministic": True, "tf32": False,
        "augmentation": None, "resize_crop": None, "input_size": [3, 256, 256], "normalization_mean": extraction.EXPECTED_MEAN,
        "normalization_std": extraction.EXPECTED_STD, "batchnorm": "eval; affine frozen", "versions": versions,
        "device": str(device), "threads": args.threads, "num_workers": args.num_workers,
        "script_sha256": extraction.sha256_file(Path(__file__)), "evaluation_note": DEVELOPMENT_NOTE}
    config["config_hash"] = hashlib.sha256(extraction.canonical_json(config).encode()).hexdigest()
    for fold in set(range(5)) - set(args.folds):
        metrics_path = output_dir / f"fold{fold}" / "metrics.json"
        if metrics_path.exists() and json.loads(metrics_path.read_text(encoding="utf-8")).get("config_hash") != config["config_hash"]:
            raise ValueError("Existing unrequested folds use another config; choose a new output directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "run_config.json", {"config": config, "runtime": device_info, "requested_folds": args.folds,
                                               "started_utc": datetime.now(timezone.utc).isoformat()})
    # Invalidate requested old results before an overwrite can be interrupted.
    # Preserve their files for recovery, but never report them as current scores.
    for fold in args.folds:
        folder = output_dir / f"fold{fold}"
        folder.mkdir(parents=True, exist_ok=True)
        write_json(folder / "status.json", {"status": "pending", "config_hash": config["config_hash"]})
    aggregate_results(output_dir, inventory["ids"], y, folds, config)
    print(DEVELOPMENT_NOTE, flush=True)
    for fold in args.folds:
        run_fold(np, torch, timm, resolve_config, args, data_dir, output_dir, inventory, y, folds, fold, config, device)
        aggregate_results(output_dir, inventory["ids"], y, folds, config)
    print(f"Saved development results: {output_dir / 'report.md'}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted. Running fold is incomplete; completed folds remain separately recorded.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
