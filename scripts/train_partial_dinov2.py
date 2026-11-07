"""User-run DINO last-block/head fine-tuning with nested epoch/LR selection.

Primary: partial_nested. Control: head_only_nested. Outer labels never enter
training, checkpoint or epoch selection. Uses a label-free frozen-prefix cache.
--preflight needs old artifacts only; --validate-only also checks prefix cache.
Neither validation mode imports torch, fits models, writes files, or downloads.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

if __package__:
    from . import extract_dinov2_prefix as prefix
    from . import evaluate_dinov2_spatial as evaluation
else:
    import extract_dinov2_prefix as prefix
    import evaluate_dinov2_spatial as evaluation

core, np, ROOT = evaluation.core, evaluation.np, evaluation.ROOT
FAMILIES = ("head_only", "partial")
CONFIGS = (
    {"name": "head_lr3e-4", "family": "head_only", "head_lr": 3e-4, "block_lr": 0.0},
    {"name": "head_lr1e-3", "family": "head_only", "head_lr": 1e-3, "block_lr": 0.0},
    {"name": "partial_lr3e-6", "family": "partial", "head_lr": 3e-4, "block_lr": 3e-6},
    {"name": "partial_lr1e-5", "family": "partial", "head_lr": 1e-3, "block_lr": 1e-5},
)
PRIMARY, CONTROL = "partial_nested", "head_only_nested"
FIXED, CONTROL_BLEND = "partial_replaced_blend", "head_only_replaced_blend"
PROTOCOL = {"schema_version": 1, "primary": PRIMARY, "families": list(FAMILIES), "configs": list(CONFIGS),
    "outer_splits": 5, "inner_splits": 3, "epoch_selection": "minimum pooled inner MSE; earliest epoch then config order for ties",
    "optimizer": "AdamW", "weight_decay_matrix_only": 1e-4, "clip_grad_norm": 1.0,
    "head": "1152 -> 1 linear, zero initialization", "feature_scaling": "StandardScaler on frozen global features of fit subset; fixed during optimization",
    "target_scaling": "population mean/std of fit subset", "pooling": prefix.base.POOLING,
    "trainable_backbone": "blocks.11 only; final norm and blocks.0..10 frozen",
    "augmentation": None, "dropout": "disabled via eval mode (gradients still enabled)",
    "amp": False, "tf32": False, "scheduler": None, "patience": None,
    "fixed_blend": "current + 0.5 * (partial_nested - old_DINO)",
    "outer_labels_used_during_fit": False}


def write_json(path, value):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def fingerprint(value):
    return hashlib.sha256(prefix.common.canonical_json(value).encode("utf-8")).hexdigest()


def verify_input_files(value):
    if isinstance(value, dict):
        for path_key, hash_key in (("path", "sha256"), ("summary_path", "summary_sha256"), ("metadata_path", "metadata_sha256")):
            if path_key in value and hash_key in value and core.sha256(Path(value[path_key])) != value[hash_key]:
                raise ValueError("Input/source changed during training: " + value[path_key])
        for item in value.values():
            verify_input_files(item)
    elif isinstance(value, list):
        for item in value:
            verify_input_files(item)


def scaling(global_features, targets):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(np.asarray(global_features, dtype=np.float64))
    y = np.asarray(targets, dtype=np.float64)
    ym, ys = float(y.mean()), float(y.std(ddof=0))
    if not np.isfinite(y).all() or not math.isfinite(ys) or ys <= 0:
        raise ValueError("Nonfinite or constant training targets")
    return scaler.mean_.astype(np.float32), scaler.scale_.astype(np.float32), ym, ys


def make_regressor(torch, block, norm, family, feature_mean, feature_scale):
    if family not in FAMILIES:
        raise ValueError("Unknown model family")
    class Regressor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.family = family
            if family == "partial":
                self.block = copy.deepcopy(block).requires_grad_(True)
                self.norm = copy.deepcopy(norm).requires_grad_(False)
            self.head = torch.nn.Linear(len(feature_mean), 1)
            torch.nn.init.zeros_(self.head.weight)
            torch.nn.init.zeros_(self.head.bias)
            self.register_buffer("feature_mean", torch.as_tensor(feature_mean).clone())
            self.register_buffer("feature_scale", torch.as_tensor(feature_scale).clone())
        def forward(self, x):
            if self.family == "partial":
                x = prefix.base.pool_tokens(torch, self.norm(self.block(x)))[1]
            return self.head((x - self.feature_mean) / self.feature_scale).squeeze(-1)
    model = Regressor().eval()
    for name, p in model.named_parameters():
        if p.requires_grad != (name.startswith("head.") or name.startswith("block.")):
            raise RuntimeError("Unexpected trainable parameters: " + name)
    return model


def optimizer_groups(model, config):
    groups = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        lr = config["head_lr"] if name.startswith("head.") else config["block_lr"]
        decay = 1e-4 if parameter.ndim > 1 else 0.0
        groups.setdefault((lr, decay), {"params": [], "lr": lr, "weight_decay": decay})["params"].append(parameter)
    return list(groups.values())


def predict(torch, model, features, indices, device, batch_size, y_mean, y_std):
    model.eval()
    parts = []
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            x = torch.from_numpy(np.array(features[indices[start:start+batch_size]], copy=True)).to(device)
            parts.append(model(x).cpu().numpy().astype(np.float64) * y_std + y_mean)
    result = np.concatenate(parts)
    if not np.isfinite(result).all():
        raise RuntimeError("Nonfinite predictions")
    return result


def train_epoch(torch, model, features, indices, targets, optimizer, device, batch_size, effective_batch, rng):
    """Sample-weighted accumulation, including unequal final microbatches."""
    model.eval()  # Frozen inference behavior; eval does not disable autograd.
    order = rng.permutation(len(indices))
    total, steps = 0.0, 0
    for group_start in range(0, len(order), effective_batch):
        group = order[group_start:group_start+effective_batch]
        optimizer.zero_grad(set_to_none=True)
        for start in range(0, len(group), batch_size):
            local = group[start:start+batch_size]
            x = torch.from_numpy(np.array(features[indices[local]], copy=True)).to(device)
            y = torch.from_numpy(np.asarray(targets[local], dtype=np.float32)).to(device)
            sse = (model(x) - y).square().sum()
            if not torch.isfinite(sse).item():
                raise RuntimeError("Nonfinite training loss")
            (sse / len(group)).backward()
            total += float(sse.detach())
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0, error_if_nonfinite=True)
        optimizer.step()
        steps += 1
    return total / len(indices), steps


class TorchRunner:
    """Receives only fitting labels and (for inner search) inner validation labels."""
    def __init__(self, torch, block, norm, global_features, tokens, device, args):
        self.torch, self.block, self.norm = torch, block, norm
        self.global_features, self.tokens, self.device, self.args = global_features, tokens, device, args

    def __call__(self, train_idx, train_y, valid_idx, valid_y, config, epochs, seed, checkpoint=None):
        torch = self.torch
        started = time.perf_counter()
        torch.manual_seed(seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        mean, scale, ym, ys = scaling(self.global_features[train_idx], train_y)
        model = make_regressor(torch, self.block, self.norm, config["family"], mean, scale).to(self.device)
        norm_before = {k: v.detach().clone() for k, v in model.state_dict().items() if k.startswith("norm.")}
        optimizer = torch.optim.AdamW(optimizer_groups(model, config))
        features = self.tokens if config["family"] == "partial" else self.global_features
        standardized_y = (np.asarray(train_y, dtype=np.float64) - ym) / ys
        rng, curve = np.random.default_rng(seed), []
        for epoch in range(1, epochs+1):
            online_loss, steps = train_epoch(torch, model, features, train_idx, standardized_y, optimizer,
                self.device, self.args.batch_size, self.args.effective_batch, rng)
            row = {"epoch": epoch, "online_train_mse": online_loss * ys**2, "optimizer_steps": steps}
            if valid_y is not None:
                pred = predict(torch, model, features, valid_idx, self.device, self.args.batch_size, ym, ys)
                row["valid_mse"] = float(np.mean((pred - valid_y)**2))
            curve.append(row)
            if epoch == 1 or epoch % 20 == 0 or epoch == epochs:
                suffix = f" inner_RMSE={math.sqrt(row['valid_mse']):.4f}" if valid_y is not None else " refit; outer labels withheld"
                print(f"  {config['name']} epoch={epoch}/{epochs}{suffix}", flush=True)
        for key, value in norm_before.items():
            if not torch.equal(value, model.state_dict()[key]):
                raise RuntimeError("Frozen final LayerNorm changed")
        train_pred = predict(torch, model, features, train_idx, self.device, self.args.batch_size, ym, ys)
        final_pred = predict(torch, model, features, valid_idx, self.device, self.args.batch_size, ym, ys)
        result = {"curve": curve, "prediction": final_pred, "train_rmse": float(np.sqrt(np.mean((train_pred-train_y)**2))),
            "target_mean": ym, "target_std": ys, "elapsed_seconds": time.perf_counter()-started,
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)}
        if checkpoint is not None:
            saved = {"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "config": config, "epoch": epochs, "target_mean": ym, "target_std": ys,
                "train_indices": train_idx.tolist(), "validation_indices": valid_idx.tolist(), "training_seed": seed,
                "protocol": PROTOCOL, "run_fingerprint": self.args.run_fingerprint,
                "cache_folder": str(self.args.prefix_cache.resolve()), "revision": prefix.base.DEFAULT_REVISION}
            tmp = checkpoint.with_name(checkpoint.name + ".tmp")
            torch.save(saved, tmp)
            os.replace(tmp, checkpoint)
        del model, optimizer
        return result


def select_candidates(curves, sizes, max_epochs, configs=CONFIGS):
    """Pooled MSE for each common epoch; no average of per-split best scores."""
    candidates = []
    for config in configs:
        runs = curves[config["name"]]
        if len(runs) != 3 or any([r["epoch"] for r in rows] != list(range(1, max_epochs+1)) for rows in runs):
            raise ValueError("Incomplete inner learning curves")
        for epoch in range(1, max_epochs+1):
            scores = [float(rows[epoch-1]["valid_mse"]) for rows in runs]
            if not np.isfinite(scores).all() or min(scores) < 0:
                raise ValueError("Invalid inner score")
            candidates.append({"config": config, "epoch": epoch, "inner_fold_mse": scores,
                "inner_mse": float(np.average(scores, weights=sizes))})
    order = {c["name"]: i for i, c in enumerate(configs)}
    selected = {family: min([r for r in candidates if r["config"]["family"] == family],
        key=lambda r: (r["inner_mse"], r["epoch"], order[r["config"]["name"]])) for family in FAMILIES}
    return selected, candidates


def fit_outer_fold(y, folds, seed, fold, runner, max_epochs, folder=None, ids=None):
    from sklearn.model_selection import StratifiedKFold
    train_idx, valid_idx = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
    train_y = y[train_idx].copy()  # Outer validation labels are not passed onward.
    splits = list(StratifiedKFold(3, shuffle=True, random_state=seed+fold).split(train_y, core.rank_groups(train_y, 10)))
    sizes = [len(v) for _, v in splits]
    curves, logs, memberships = {}, [], []
    for inner, (it, iv) in enumerate(splits):
        memberships.extend({"outer_fold": fold, "inner_fold": inner, "role": role, "ID": str(ids[i]) if ids is not None else str(i)}
            for role, indices in (("train", train_idx[it]), ("valid", train_idx[iv])) for i in indices)
    for config in CONFIGS:
        runs = []
        for inner, (it, iv) in enumerate(splits):
            run_seed = (seed + 1009*fold + 31*inner) % 2**32
            print(f"Outer {fold}, inner {inner}, {config['name']}", flush=True)
            result = runner(train_idx[it], train_y[it], train_idx[iv], train_y[iv], config, max_epochs, run_seed)
            runs.append(result["curve"])
            logs.extend({"outer_fold": fold, "inner_fold": inner, "config": config["name"], **r} for r in result["curve"])
        curves[config["name"]] = runs
    selected, candidates = select_candidates(curves, sizes, max_epochs)
    predictions, refits = {}, {}
    for family in FAMILIES:
        choice = selected[family]
        checkpoint = folder / (family + ".pt") if folder is not None else None
        result = runner(train_idx, train_y, valid_idx, None, choice["config"], choice["epoch"],
            (seed + 1009*fold + 997) % 2**32, checkpoint=checkpoint)
        pred = np.asarray(result["prediction"], dtype=np.float64)
        if pred.shape != (len(valid_idx),) or not np.isfinite(pred).all():
            raise ValueError("Invalid outer prediction shape/values")
        predictions[family+"_nested"] = pred.tolist()
        refits[family] = {k: v for k, v in result.items() if k not in ("prediction",)}
        print(f"Outer {fold}: {family} selected {choice['config']['name']} epoch={choice['epoch']}" +
            (" [EPOCH LIMIT SELECTED]" if choice["epoch"] == max_epochs else ""), flush=True)
    return {"outer_fold": fold, "inner_seed": seed+fold, "outer_train_indices": train_idx.tolist(),
        "outer_valid_indices": valid_idx.tolist(), "inner_validation_sizes": sizes,
        "selected": selected, "candidates": candidates, "inner_curves": logs, "memberships": memberships,
        "predictions": predictions, "refits": refits}


def add_fixed_predictions(predictions, references):
    return {**predictions, **references,
        FIXED: references[evaluation.CURRENT] + 0.5*(predictions[PRIMARY] - references[evaluation.DINO]),
        CONTROL_BLEND: references[evaluation.CURRENT] + 0.5*(predictions[CONTROL] - references[evaluation.DINO])}


def protected_output(args):
    inputs = [args.global_features, args.prefix_cache / "tokens.npy", args.train_csv, args.audit_features,
        args.folds, args.baseline, args.previous_frozen_oof, args.previous_dino_oof, args.previous_dino_summary]
    if any(p.resolve().is_relative_to(args.output_dir.resolve()) for p in inputs) or args.output_dir.resolve().is_relative_to(args.train_csv.resolve().parent):
        raise ValueError("Output would contain protected input/reference data")
    if args.output_dir.exists() and (not args.output_dir.is_dir() or (any(args.output_dir.iterdir()) and not args.resume)):
        raise ValueError("Output exists; use --resume for exactly the same run or a new output directory")


def prepare_output(folder, contract, resume):
    manifest = {"fingerprint": fingerprint(contract), "contract": contract}
    if folder.exists() and any(folder.iterdir()):
        if not resume or not (folder / "run_manifest.json").is_file():
            raise ValueError("Cannot resume output without its run manifest")
        if json.loads((folder / "run_manifest.json").read_text(encoding="utf-8")) != manifest:
            raise ValueError("Resume configuration/source/input/runtime mismatch; choose a new output directory")
    else:
        folder.mkdir(parents=True, exist_ok=True)
        write_json(folder / "run_manifest.json", manifest)
    return manifest["fingerprint"]


def read_completed_fold(folder, run_id, fold, folds):
    done = folder / "done.json"
    if not done.exists():
        return None
    info = json.loads(done.read_text(encoding="utf-8"))
    if info.get("run_fingerprint") != run_id or set(info.get("files", {})) != {"result.json", "head_only.pt", "partial.pt"}:
        raise ValueError("Invalid completed-fold manifest")
    if any(core.sha256(folder / name) != sha for name, sha in info["files"].items()):
        raise ValueError("Completed-fold file hash mismatch")
    result = json.loads((folder / "result.json").read_text(encoding="utf-8"))
    if result["outer_fold"] != fold or result["outer_valid_indices"] != np.flatnonzero(folds == fold).tolist():
        raise ValueError("Completed-fold assignment mismatch")
    return result


def build_report(summary):
    lines = ["# DINO 마지막 block + head · 중첩 검증", "",
        "주 결과는 partial_nested이며 head_only_nested는 대조군입니다. outer 정답은 epoch·학습률·checkpoint 선택에 사용하지 않았습니다.", "",
        "| 모델 | OOF RMSE | 현재 조합 대비 Δ | 기존 DINO 대비 Δ |", "|---|---:|---:|---:|"]
    for name, value in summary["diagnostics"].items():
        m = value["overall"]
        lines.append(f"| {name} | {m['rmse']:.6f} | {m['delta_rmse_vs_current_blend']:+.6f} | {m['delta_rmse_vs_dino_linear']:+.6f} |")
    lines += ["", "| fold | family | head LR | block LR | epoch | inner RMSE | outer 학습 RMSE |", "|---|---|---:|---:|---:|---:|---:|"]
    for r in summary["selections"]:
        for family, choice in r["selected"].items():
            c = choice["config"]
            lines.append(f"| {r['outer_fold']} | {family} | {c['head_lr']:g} | {c['block_lr']:g} | {choice['epoch']} | {math.sqrt(choice['inner_mse']):.6f} | {r['refits'][family]['train_rmse']:.6f} |")
    lines += ["", "- partial_replaced_blend = 기존 조합 + 0.5×(partial_nested − 기존 DINO). 기존 수작업25%·ResNet25% 유지.",
        "- head_only_replaced_blend는 대조군 조합 진단입니다. 사후 최저 OOF를 주 지표로 바꾸지 않습니다.",
        "- 각 내부 학습은 지정한 모든 epoch를 실행합니다. 같은 epoch의 3개 inner MSE를 표본 수로 가중해 LR·epoch를 선택하고 outer400장을 처음부터 재적합합니다.",
        "- 마지막 LayerNorm은 고정하고 마지막 Transformer block과 linear head만 갱신합니다. dropout·증강·AMP·TF32는 사용하지 않습니다.",
        "- head와 partial은 같은 두 head LR 후보를 쓰며 partial block LR은 head LR의1/100입니다.",
        "- 전역 특징 X 통계와 y 통계는 각 적합 부분에서만 계산합니다. X 통계는 초기 frozen 특징에서 구한 뒤 학습 중 고정합니다.",
        "- online_train_mse는 epoch 중 갱신 전후 모델이 섞인 평균입니다. 최종 학습 RMSE는 refit 종료 후 다시 측정합니다.",
        f"- 최대 epoch: {summary['max_epochs']}; 상한 선택: {summary['epoch_limit_selections']}. 상한 선택/곡선이 계속 개선될 때만 학습량 재검토.",
        "- seed당60개 inner 학습 + 10개 outer 재적합. prefix는 두 seed가 공유하는 고정·정답 미사용 변환입니다.",
        "- 두 seed는 같은500장의 재분할입니다. 반복 연구의 선택 편향이 남으며 독립 검증이나 대회 점수가 아닙니다.", ""]
    return "\n".join(lines)


def write_outputs(args, data, predictions, records, diagnostics, tables, runtime, cache_metadata, elapsed):
    selections = [{k: v for k, v in r.items() if k not in ("inner_curves", "memberships", "predictions", "candidates")} for r in records]
    summary = {"created_utc": datetime.now(timezone.utc).isoformat(), "experiment": "dinov2_last_block_nested",
        "primary_metric": PRIMARY + " pooled OOF RMSE", "protocol": PROTOCOL, "seed": args.seed,
        "max_epochs": args.max_epochs, "batch_size": args.batch_size, "effective_batch": args.effective_batch,
        "estimator_run_count": 70, "inner_training_runs": 60, "outer_refits": 10,
        "selections": selections, "diagnostics": diagnostics, "input_files": data.provenance,
        "prefix_metadata": cache_metadata, "runtime": runtime, "run_fingerprint": args.run_fingerprint,
        "elapsed_seconds_this_invocation": elapsed,
        "epoch_limit_selections": [{"fold": r["outer_fold"], "family": f} for r in records for f in FAMILIES if r["selected"][f]["epoch"] == args.max_epochs]}
    oof = [{"ID": str(ident), "hardness": float(data.y[i]), "fold": int(data.folds[i]), core.BASELINE_NAME: float(data.baseline[i]),
        **{k: float(v[i]) for k, v in predictions.items()}} for i, ident in enumerate(data.ids)]
    inner = [{"outer_fold": r["outer_fold"], "config": c["config"]["name"], "family": c["config"]["family"],
        "epoch": c["epoch"], "inner_mse": c["inner_mse"], **{f"inner_fold{j}_mse": v for j, v in enumerate(c["inner_fold_mse"])},
        "selected": c == r["selected"][c["config"]["family"]]} for r in records for c in r["candidates"]]
    outputs = {"oof": oof, "inner_search": inner, "inner_curves": [x for r in records for x in r["inner_curves"]],
        "inner_memberships": [x for r in records for x in r["memberships"]], **tables}
    for name, rows in outputs.items():
        tmp = args.output_dir / (name + ".csv.tmp")
        tmp.write_text(core.csv_text(rows), encoding="utf-8", newline="")
        os.replace(tmp, args.output_dir / (name + ".csv"))
    (args.output_dir / "report.md").write_text(build_report(summary), encoding="utf-8")
    write_json(args.output_dir / "summary.json", summary)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--effective-batch", type=int, default=32)
    p.add_argument("--max-epochs", type=int, default=80)
    p.add_argument("--prefix-cache", type=Path, default=ROOT / "reports/dinov2_partial/prefix")
    p.add_argument("--global-features", type=Path, default=ROOT / "reports/frozen_dinov2_small/features.npz")
    p.add_argument("--train-csv", type=Path, default=ROOT / "open/train.csv")
    p.add_argument("--audit-features", type=Path, default=ROOT / "reports/audit_train_features.csv")
    for flag in ("folds", "baseline", "previous-frozen-oof", "previous-dino-oof", "previous-dino-summary", "output-dir"):
        p.add_argument("--" + flag, type=Path)
    modes = p.add_mutually_exclusive_group()
    modes.add_argument("--preflight", action="store_true")
    modes.add_argument("--validate-only", action="store_true")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args(argv)
    core.validate_seed(args.seed)
    if min(args.threads, args.batch_size, args.effective_batch, args.max_epochs) < 1 or args.effective_batch < args.batch_size:
        p.error("Positive settings required; effective-batch must be >= batch-size")
    defaults = evaluation.parse_args(["--seed", str(args.seed)])
    for name in ("folds", "baseline", "previous_frozen_oof", "previous_dino_oof"):
        if getattr(args, name) is None:
            setattr(args, name, getattr(defaults, name))
    args.previous_dino_summary = args.previous_dino_summary or args.previous_dino_oof.with_name("summary.json")
    args.output_dir = args.output_dir or ROOT / f"reports/dinov2_partial/seed{args.seed}"
    return args


def main(argv=None):
    args = parse_args(argv)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    start = time.perf_counter()
    if not (args.preflight or args.validate_only):
        protected_output(args)
    data, references, anchor = evaluation.load_prior(args)
    checkpoint_files = prefix.ensure_local_checkpoint()
    if args.preflight:
        print(json.dumps({"status": "prior_artifacts_and_checkpoint_valid", "seed": args.seed, "rows": len(data.y),
            "prefix_checked": False, "torch_imported": False, "fit_or_forward": False, "output_written": False}))
        return 0
    tokens, metadata = prefix.validate_cache(np, args.prefix_cache, data.ids, data.metadata, args.global_features)
    data.provenance["prefix_tokens"] = {"path": str((args.prefix_cache / "tokens.npy").resolve()), "sha256": core.sha256(args.prefix_cache / "tokens.npy"),
        "metadata_path": str((args.prefix_cache / "metadata.json").resolve()), "metadata_sha256": core.sha256(args.prefix_cache / "metadata.json")}
    data.provenance["partial_sources"] = {name: {"path": str(Path(path).resolve()), "sha256": core.sha256(Path(path))}
        for name, path in {"trainer": __file__, "prefix": prefix.__file__, "diagnostics": evaluation.__file__}.items()}
    if args.validate_only:
        print(json.dumps({"status": "all_artifacts_valid", "seed": args.seed, "rows": len(data.y),
            "torch_imported": False, "fit_or_forward": False, "output_written": False}))
        return 0
    _, torch, timm, resolve_config, versions = prefix.common.import_runtime()
    device, device_info = prefix.common.configure_runtime(torch, np, args)
    print(f"Device={device} ({device_info.get('gpu_name') or 'CPU'}); FP32; batch={args.batch_size}; effective_batch={args.effective_batch}; max_epochs={args.max_epochs}", flush=True)
    backbone, _ = prefix.base.build_model(torch, timm, resolve_config)
    if prefix.common.feature_state_hash(backbone) != data.metadata["feature_state_sha256"] or len(backbone.blocks) != 12:
        raise ValueError("Initial backbone must exactly match original frozen DINO")
    block, norm = copy.deepcopy(backbone.blocks[-1]).cpu(), copy.deepcopy(backbone.norm).cpu()
    del backbone
    global_features = data.features["cls_patch_mean_std"].astype(np.float32)
    # Verify all prefix tokens reconstruct the reference representation before any fit.
    block_device, norm_device = copy.deepcopy(block).to(device).eval(), copy.deepcopy(norm).to(device).eval()
    restored = []
    with torch.no_grad():
        for offset in range(0, len(tokens), args.batch_size):
            x = torch.from_numpy(np.array(tokens[offset:offset+args.batch_size], copy=True)).to(device)
            restored.append(prefix.base.pool_tokens(torch, norm_device(block_device(x)))[1].cpu().numpy())
    difference = prefix.spatial.check_global_match(np, np.concatenate(restored), global_features)
    del block_device, norm_device, restored
    print(f"Initial prefix/suffix global anchor matched; max difference={difference:g}", flush=True)
    runtime = {"versions": versions, "device": str(device), "gpu_name": device_info.get("gpu_name"),
        "checkpoint_files": checkpoint_files, "initial_global_max_difference": difference}
    contract = {"protocol": PROTOCOL, "seed": args.seed, "max_epochs": args.max_epochs,
        "batch_size": args.batch_size, "effective_batch": args.effective_batch, "threads": args.threads,
        "input_files": data.provenance, "runtime": runtime}
    args.run_fingerprint = prepare_output(args.output_dir, contract, args.resume)
    runner = TorchRunner(torch, block, norm, global_features, tokens, device, args)
    predictions = {name: np.full(len(data.y), np.nan) for name in (CONTROL, PRIMARY)}
    records = []
    for fold in range(5):
        folder = args.output_dir / f"fold{fold}"
        folder.mkdir(exist_ok=True)
        record = read_completed_fold(folder, args.run_fingerprint, fold, data.folds)
        if record is None:
            record = fit_outer_fold(data.y, data.folds, args.seed, fold, runner, args.max_epochs, folder, data.ids)
            write_json(folder / "result.json", record)
            write_json(folder / "done.json", {"run_fingerprint": args.run_fingerprint,
                "files": {name: core.sha256(folder / name) for name in ("result.json", "head_only.pt", "partial.pt")}})
        else:
            print(f"Outer {fold}: verified completed fold restored", flush=True)
        valid = record["outer_valid_indices"]
        for name in predictions:
            predictions[name][valid] = record["predictions"][name]
        records.append(record)
    if not all(np.isfinite(v).all() for v in predictions.values()):
        raise RuntimeError("Incomplete OOF predictions")
    verify_input_files(data.provenance)
    # Outer labels first enter metrics here, after every training decision is fixed.
    predictions = add_fixed_predictions(predictions, references)
    diagnostics, tables = evaluation.diagnose(data, predictions)
    for name, d in diagnostics.items():
        d["overall"]["delta_rmse_vs_head_only"] = d["overall"]["rmse"] - diagnostics[CONTROL]["overall"]["rmse"]
    for row in tables["metrics"]:
        row["delta_rmse_vs_head_only"] = diagnostics[row["model"]]["overall"]["delta_rmse_vs_head_only"]
    write_outputs(args, data, predictions, records, diagnostics, tables, runtime, metadata, time.perf_counter()-start)
    for name in (PRIMARY, CONTROL, FIXED, CONTROL_BLEND):
        m = diagnostics[name]["overall"]
        print(f"{name}: RMSE={m['rmse']:.6f}; delta vs current={m['delta_rmse_vs_current_blend']:+.6f}; delta vs DINO={m['delta_rmse_vs_dino_linear']:+.6f}")
    print(f"Report: {args.output_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"DINO fine-tuning stopped: {exc}", file=sys.stderr)
        raise SystemExit(2)
