"""User-run Ridge-initialized DINO last-block experiment (no downloads).

Fixed alpha=1000, exact fit-subset Ridge initialization, explicit head L2,
full-training-set gradient accumulation and nested epoch-zero fallback.
--preflight/--validate-only never fit, import torch, forward or write output.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
import time

if __package__:
    from . import train_partial_dinov2 as previous
else:
    import train_partial_dinov2 as previous

prefix, evaluation, core, np, ROOT = previous.prefix, previous.evaluation, previous.core, previous.np, previous.ROOT
write_json, fingerprint = previous.write_json, previous.fingerprint
FAMILIES = ("head_only", "partial")
CONFIGS = (
    {"name": "ridge_head_control", "family": "head_only", "block_lr": 0.0},
    {"name": "ridge_partial_lr1e-6", "family": "partial", "block_lr": 1e-6},
    {"name": "ridge_partial_lr3e-6", "family": "partial", "block_lr": 3e-6},
)
ALPHA = 1000.0
PRIMARY, CONTROL = "ridge_partial_nested", "ridge_head_only_nested"
FIXED, CONTROL_BLEND = "ridge_partial_replaced_blend", "ridge_head_only_replaced_blend"
MODEL_NAMES = {"partial": PRIMARY, "head_only": CONTROL}
PROTOCOL = {
    "schema_version": 1, "primary": PRIMARY, "configs": list(CONFIGS), "alpha": ALPHA,
    "alpha_policy": "fixed before this experiment from prior research; never reselected on an outer fold",
    "objective": "mean standardized-target squared error + alpha/n_fit * sum(head.weight**2); bias unpenalized",
    "initialization": "sklearn StandardScaler + Ridge(alpha=1000, solver=auto), y StandardScaler; fit subset only",
    "feature_scaling": "fit-subset frozen-global statistics, fixed throughout optimization",
    "optimizer_head": "SGD without momentum/decay; lr=min(0.01, 0.5/(mean squared augmented standardized-feature norm + alpha/n_fit))",
    "optimizer_block": "Adam defaults, weight_decay=0; last-block gradient norm clipped at 1",
    "effective_batch": "entire fit subset, one optimizer update per epoch; microbatches only bound memory",
    "precision": "FP32 backbone and pooling, FP64 normalization/head/loss; AMP/TF32 disabled",
    "trainable_backbone": "blocks.11 only; final LayerNorm and blocks.0..10 frozen",
    "pooling": prefix.base.POOLING, "outer_splits": 5, "inner_splits": 3,
    "epoch_selection": "pooled inner MSE, epochs 0..max; ties prefer earlier epoch then config order",
    "epoch_zero": "canonical sklearn Ridge predictions after torch initialization equivalence check",
    "augmentation": None, "dropout": "disabled via eval mode", "scheduler": None,
    "outer_labels_used_during_fit": False,
    "fixed_blend": "current + 0.5 * (ridge_partial_nested - old_DINO)",
}


def fit_ridge(global_features, train_idx, train_y):
    """Called only by the user-run trainer or synthetic tests, never validation modes."""
    y = np.asarray(train_y, dtype=np.float64)
    if len(y) != len(train_idx) or not np.isfinite(y).all() or y.std() <= 0:
        raise ValueError("Invalid/constant fit targets")
    estimator = core.make_estimator(ALPHA).fit(np.asarray(global_features[train_idx], dtype=np.float64), y)
    scaler = estimator.regressor_.named_steps["standardscaler"]
    ridge = estimator.regressor_.named_steps["ridge"]
    state = {"mean": scaler.mean_.copy(), "scale": scaler.scale_.copy(),
        "weight": np.asarray(ridge.coef_, dtype=np.float64).reshape(1, -1).copy(),
        "bias": np.asarray(ridge.intercept_, dtype=np.float64).reshape(1).copy(),
        "target_mean": float(estimator.transformer_.mean_[0]), "target_std": float(estimator.transformer_.scale_[0])}
    return estimator, state


def make_regressor(torch, block, norm, family, state):
    if family not in FAMILIES:
        raise ValueError("Unknown family")
    class Regressor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            if family == "partial":
                self.block = copy.deepcopy(block).float().requires_grad_(True)
                self.norm = copy.deepcopy(norm).float().requires_grad_(False)
            self.head = torch.nn.Linear(len(state["mean"]), 1, dtype=torch.float64)
            with torch.no_grad():
                self.head.weight.copy_(torch.as_tensor(state["weight"]))
                self.head.bias.copy_(torch.as_tensor(state["bias"]))
            self.register_buffer("feature_mean", torch.as_tensor(state["mean"], dtype=torch.float64).clone())
            self.register_buffer("feature_scale", torch.as_tensor(state["scale"], dtype=torch.float64).clone())

        def standardized_features(self, x):
            if family == "partial":
                x = prefix.base.pool_tokens(torch, self.norm(self.block(x)))[1]
            return (x.double() - self.feature_mean) / self.feature_scale

        def forward(self, x):
            return self.head(self.standardized_features(x)).squeeze(-1)
    model = Regressor().eval()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad != (name.startswith("head.") or name.startswith("block.")):
            raise RuntimeError("Unexpected trainable parameter: " + name)
    return model


def train_epoch(torch, model, features, indices, targets, head_optimizer, block_optimizer, device, batch_size):
    """One deterministic full-subset gradient, explicit penalty applied exactly once.

    The head step uses a conservative curvature bound for the current features:
    lambda_max(Z_aug.T Z_aug/n + diag(alpha/n,...,0)) <= mean||Z_aug||^2 + alpha/n.
    This keeps the fixed-feature optimum stationary without Adam amplifying tiny
    head roundoff gradients. The joint nonlinear block step has no such guarantee.
    """
    model.eval()
    head_optimizer.zero_grad(set_to_none=True)
    if block_optimizer is not None:
        block_optimizer.zero_grad(set_to_none=True)
    n = len(indices)
    sse_total, feature_norm_total = 0.0, 0.0
    for start in range(0, n, batch_size):
        local = slice(start, start + batch_size)
        x = torch.from_numpy(np.array(features[indices[local]], copy=True)).to(device)
        y = torch.as_tensor(targets[local], dtype=torch.float64, device=device)
        z = model.standardized_features(x)
        sse = (model.head(z).squeeze(-1) - y).square().sum()
        if not torch.isfinite(sse).item():
            raise RuntimeError("Nonfinite data loss")
        (sse / n).backward()
        sse_total += float(sse.detach())
        feature_norm_total += float(z.detach().square().sum()) + len(x)  # includes intercept column
    penalty = (ALPHA / n) * model.head.weight.square().sum()
    penalty.backward()
    head_grad = math.sqrt(sum(float(p.grad.square().sum()) for p in model.head.parameters()))
    head_lr = min(0.01, 0.5 / (feature_norm_total / n + ALPHA / n))
    if not math.isfinite(head_grad + head_lr + float(penalty.detach())):
        raise RuntimeError("Nonfinite objective/gradient")
    for group in head_optimizer.param_groups:
        group["lr"] = head_lr
    block_grad = 0.0
    if block_optimizer is not None:
        block_grad = float(torch.nn.utils.clip_grad_norm_(model.block.parameters(), 1.0, error_if_nonfinite=True))
    head_optimizer.step()
    if block_optimizer is not None:
        block_optimizer.step()
    return {"data_mse_standardized": sse_total / n, "l2_penalty": float(penalty.detach()),
        "objective_before_step": sse_total / n + float(penalty.detach()), "head_lr": head_lr,
        "head_gradient_norm": head_grad, "block_gradient_norm_before_clip": block_grad, "optimizer_steps": 1}


class TorchRunner:
    def __init__(self, torch, block, norm, global_features, tokens, device, args):
        self.torch, self.block, self.norm = torch, block, norm
        self.global_features, self.tokens, self.device, self.args = global_features, tokens, device, args

    def __call__(self, train_idx, train_y, valid_idx, valid_y, config, epochs, seed, checkpoint=None):
        torch, started = self.torch, time.perf_counter()
        torch.manual_seed(seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        estimator, state = fit_ridge(self.global_features, train_idx, train_y)
        model = make_regressor(torch, self.block, self.norm, config["family"], state).to(self.device)
        features = self.tokens if config["family"] == "partial" else self.global_features
        ym, ys = state["target_mean"], state["target_std"]
        def predict(indices):
            return previous.predict(torch, model, features, indices, self.device, self.args.batch_size, ym, ys)
        ridge_pred = estimator.predict(np.asarray(self.global_features[valid_idx], dtype=np.float64))
        ridge_train = estimator.predict(np.asarray(self.global_features[train_idx], dtype=np.float64))
        checks = [np.max(np.abs(predict(valid_idx) - ridge_pred)), np.max(np.abs(predict(train_idx) - ridge_train))]
        if max(checks) > 1e-4:
            raise ValueError(f"Ridge initialization prediction mismatch: {max(checks):g}")
        row0 = {"epoch": 0, "optimizer_steps": 0,
            **({"valid_mse": float(np.mean((ridge_pred - valid_y)**2))} if valid_y is not None else {})}
        curve = [row0]
        frozen = {k: v.detach().clone() for k, v in model.state_dict().items() if k.startswith("norm.")}
        head_optimizer = torch.optim.SGD(model.head.parameters(), lr=0.01, momentum=0, weight_decay=0)
        block_optimizer = (torch.optim.Adam(model.block.parameters(), lr=config["block_lr"], weight_decay=0)
            if config["family"] == "partial" else None)
        targets = (np.asarray(train_y, dtype=np.float64) - ym) / ys
        for epoch in range(1, epochs + 1):
            stats = train_epoch(torch, model, features, train_idx, targets, head_optimizer, block_optimizer,
                self.device, self.args.batch_size)
            row = {"epoch": epoch, **stats}
            if valid_y is not None:
                row["valid_mse"] = float(np.mean((predict(valid_idx) - valid_y)**2))
            curve.append(row)
            if epoch == 1 or epoch % 20 == 0 or epoch == epochs:
                suffix = f" inner_RMSE={math.sqrt(row['valid_mse']):.6f}" if valid_y is not None else " refit; outer labels withheld"
                print(f"  {config['name']} epoch={epoch}/{epochs}{suffix}", flush=True)
        if any(not torch.equal(v, model.state_dict()[k]) for k, v in frozen.items()):
            raise RuntimeError("Frozen final LayerNorm changed")
        final_pred = ridge_pred.copy() if epochs == 0 else predict(valid_idx)
        train_pred = ridge_train if epochs == 0 else predict(train_idx)
        result = {"curve": curve, "prediction": final_pred, "ridge_prediction": ridge_pred,
            "train_rmse": float(np.sqrt(np.mean((train_pred - train_y)**2))),
            "epoch0_max_absolute_difference": float(max(checks)),
            "final_prediction_max_change_from_ridge": float(np.max(np.abs(final_pred - ridge_pred))),
            "elapsed_seconds": time.perf_counter() - started, "target_mean": ym, "target_std": ys,
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)}
        if checkpoint is not None:
            saved = {"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "config": config, "epoch": epochs, "target_mean": ym, "target_std": ys,
                "train_indices": train_idx.tolist(), "validation_indices": valid_idx.tolist(), "training_seed": seed,
                "protocol": PROTOCOL, "run_fingerprint": self.args.run_fingerprint,
                "cache_folder": str(self.args.prefix_cache.resolve()), "revision": prefix.base.DEFAULT_REVISION,
                "canonical_epoch0_predictions": ridge_pred.tolist() if epochs == 0 else None,
                "canonical_epoch0_ids_are_validation_indices": True}
            tmp = checkpoint.with_name(checkpoint.name + ".tmp")
            torch.save(saved, tmp)
            os.replace(tmp, checkpoint)
        return result


def select_candidates(curves, sizes, max_epochs):
    candidates = []
    for config in CONFIGS:
        runs = curves[config["name"]]
        if len(runs) != 3 or any([r["epoch"] for r in rows] != list(range(max_epochs + 1)) for rows in runs):
            raise ValueError("Incomplete inner learning curves")
        for epoch in range(max_epochs + 1):
            scores = [float(rows[epoch]["valid_mse"]) for rows in runs]
            if not np.isfinite(scores).all() or min(scores) < 0:
                raise ValueError("Invalid inner score")
            candidates.append({"config": config, "epoch": epoch, "inner_fold_mse": scores,
                "inner_mse": float(np.average(scores, weights=sizes))})
    order = {c["name"]: i for i, c in enumerate(CONFIGS)}
    selected = {}
    for family in FAMILIES:
        rows = [r for r in candidates if r["config"]["family"] == family]
        best = min(r["inner_mse"] for r in rows)
        # Resolve roundoff-only differences in favor of fewer updates, predeclared.
        selected[family] = min([r for r in rows if r["inner_mse"] <= best + 1e-8],
            key=lambda r: (r["epoch"], order[r["config"]["name"]]))
    return selected, candidates


def validate_anchor_policy(anchor, references):
    if any(r["winner"]["alpha"] != ALPHA for r in anchor["selections"].values()):
        raise ValueError("Prior DINO anchor must select alpha=1000 in every outer fold")
    if not np.allclose(anchor["predictions"], references[evaluation.DINO], rtol=0, atol=1e-7):
        raise ValueError("Global Ridge anchor must match the current DINO component")


def fit_outer_fold(y, folds, seed, fold, runner, max_epochs, folder=None, ids=None, anchor=None):
    from sklearn.model_selection import StratifiedKFold
    train_idx, valid_idx = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
    train_y = y[train_idx].copy()
    splits = list(StratifiedKFold(3, shuffle=True, random_state=seed + fold).split(train_y, core.rank_groups(train_y, 10)))
    sizes = [len(iv) for _, iv in splits]
    curves, logs, checks, memberships = {}, [], [], []
    for inner, (it, iv) in enumerate(splits):
        memberships.extend({"outer_fold": fold, "inner_fold": inner, "role": role, "ID": str(ids[i]) if ids is not None else str(i)}
            for role, indices in (("train", train_idx[it]), ("valid", train_idx[iv])) for i in indices)
    for config in CONFIGS:
        runs = []
        for inner, (it, iv) in enumerate(splits):
            print(f"Outer {fold}, inner {inner}, {config['name']}", flush=True)
            result = runner(train_idx[it], train_y[it], train_idx[iv], train_y[iv], config, max_epochs,
                (seed + 1009 * fold + 31 * inner) % 2**32)
            if anchor is not None:
                old = anchor["selections"][fold]
                if sizes != old["inner_validation_sizes"] or abs(result["curve"][0]["valid_mse"] - old["winner"]["inner_fold_mse"][inner]) > 1e-8:
                    raise ValueError("Epoch-zero inner Ridge anchor mismatch")
            runs.append(result["curve"])
            checks.append({"config": config["name"], "inner_fold": inner,
                "epoch0_max_absolute_difference": result["epoch0_max_absolute_difference"],
                "final_prediction_max_change_from_ridge": result["final_prediction_max_change_from_ridge"]})
            logs.extend({"outer_fold": fold, "inner_fold": inner, "config": config["name"], **r} for r in result["curve"])
        curves[config["name"]] = runs
    selected, candidates = select_candidates(curves, sizes, max_epochs)
    predictions, refits = {}, {}
    for family in FAMILIES:
        choice = selected[family]
        result = runner(train_idx, train_y, valid_idx, None, choice["config"], choice["epoch"],
            (seed + 1009 * fold + 997) % 2**32, checkpoint=folder / (family + ".pt") if folder is not None else None)
        if anchor is not None and not np.allclose(result["ridge_prediction"], anchor["predictions"][valid_idx], rtol=0, atol=1e-7):
            raise ValueError("Outer Ridge initialization anchor mismatch")
        pred = np.asarray(result["prediction"], dtype=np.float64)
        if pred.shape != (len(valid_idx),) or not np.isfinite(pred).all():
            raise ValueError("Invalid outer prediction shape/values")
        predictions[MODEL_NAMES[family]] = pred.tolist()
        refits[family] = {k: v for k, v in result.items() if k not in ("prediction", "ridge_prediction")}
        print(f"Outer {fold}: {family} selected {choice['config']['name']} epoch={choice['epoch']}" +
            (" [NO UPDATE: RIDGE]" if choice["epoch"] == 0 else " [EPOCH LIMIT SELECTED]" if choice["epoch"] == max_epochs else ""), flush=True)
    return {"outer_fold": fold, "inner_seed": seed + fold, "outer_train_indices": train_idx.tolist(),
        "outer_valid_indices": valid_idx.tolist(), "inner_validation_sizes": sizes, "selected": selected,
        "candidates": candidates, "inner_curves": logs, "memberships": memberships, "initialization_checks": checks,
        "predictions": predictions, "refits": refits, "ridge_anchor_matched": anchor is not None}


def add_fixed_predictions(predictions, references):
    return {**predictions, **references,
        FIXED: references[evaluation.CURRENT] + 0.5 * (predictions[PRIMARY] - references[evaluation.DINO]),
        CONTROL_BLEND: references[evaluation.CURRENT] + 0.5 * (predictions[CONTROL] - references[evaluation.DINO])}


def build_report(summary):
    lines = ["# Ridge 초기화 + 동일 head L2 · DINO 마지막 block", "",
        "주 결과는 ridge_partial_nested, 대조군은 ridge_head_only_nested입니다. epoch0은 기존 Ridge로 복귀하는 선택입니다.", "",
        "| 모델 | OOF RMSE | 현재 조합 대비 Δ | 기존 DINO 대비 Δ |", "|---|---:|---:|---:|"]
    for name, value in summary["diagnostics"].items():
        m = value["overall"]
        lines.append(f"| {name} | {m['rmse']:.6f} | {m['delta_rmse_vs_current_blend']:+.6f} | {m['delta_rmse_vs_dino_linear']:+.6f} |")
    lines += ["", "| fold | family | block LR | epoch | inner RMSE | train RMSE |", "|---|---|---:|---:|---:|---:|"]
    for r in summary["selections"]:
        for family, choice in r["selected"].items():
            lines.append(f"| {r['outer_fold']} | {family} | {choice['config']['block_lr']:g} | {choice['epoch']} | {math.sqrt(choice['inner_mse']):.6f} | {r['refits'][family]['train_rmse']:.6f} |")
    lines += ["", "- alpha=1000을 이번 실험 전에 고정. 각 적합 부분의 X/y scaler와 Ridge 해만 사용합니다.",
        "- 목적함수는 표준화 y의 MSE + (1000/n_fit) × head.weight 제곱합. bias와 block에는 weight decay가 없습니다.",
        "- backbone/pooling FP32, head/표준화 FP64. 모든 학습 표본의 기울기를 모아 epoch당 한 번 갱신합니다.",
        "- head는 현재 특징의 곡률 상한으로 제한한 SGD, block은 Adam(1e-6 또는 3e-6)입니다. head-only에 같은 head 규칙을 적용합니다.",
        "- epoch0의 torch 예측은 sklearn Ridge와 최대 절대차 1e-4 이내인지 확인합니다. epoch0을 선택한 OOF는 sklearn 값을 보존합니다.",
        "- epoch0 내부 MSE/외부 예측을 이전 Ridge와 대조. 3개 inner의 같은 epoch MSE를 표본 수로 가중해 선택합니다. 1e-8 이내 동률은 이른 epoch 우선.",
        "- outer 정답은 재적합 및 epoch·LR 선택에 전달하지 않습니다. inner의 epoch0 fallback이 outer 개선을 보장하지는 않습니다.",
        f"- 최대 epoch={summary['max_epochs']}; epoch0 선택={summary['epoch_zero_selections']}; 상한 선택={summary['epoch_limit_selections']}.",
        "- objective_before_step은 해당 epoch의 갱신 직전 전체 학습 목적함수입니다. epoch E의 valid_mse는 E번 갱신 후 값입니다.",
        "- seed당45 inner 실행+10 outer 재적합. 같은500장 반복 연구이며 독립 검증/대회 점수가 아닙니다.", ""]
    return "\n".join(lines)


def write_outputs(args, data, predictions, records, diagnostics, tables, runtime, metadata, elapsed):
    selections = [{k: v for k, v in r.items() if k not in ("inner_curves", "memberships", "predictions", "candidates")} for r in records]
    summary = {"created_utc": datetime.now(timezone.utc).isoformat(), "experiment": "dinov2_ridge_initialized",
        "primary_metric": PRIMARY + " pooled OOF RMSE", "protocol": PROTOCOL, "seed": args.seed,
        "max_epochs": args.max_epochs, "batch_size": args.batch_size, "inner_training_runs": 45, "outer_refits": 10,
        "selections": selections, "diagnostics": diagnostics, "input_files": data.provenance,
        "prefix_metadata": metadata, "runtime": runtime, "run_fingerprint": args.run_fingerprint,
        "elapsed_seconds_this_invocation": elapsed,
        "epoch_zero_selections": [{"fold": r["outer_fold"], "family": f} for r in records for f in FAMILIES if r["selected"][f]["epoch"] == 0],
        "epoch_limit_selections": [{"fold": r["outer_fold"], "family": f} for r in records for f in FAMILIES if r["selected"][f]["epoch"] == args.max_epochs]}
    oof = [{"ID": str(ident), "hardness": float(data.y[i]), "fold": int(data.folds[i]), core.BASELINE_NAME: float(data.baseline[i]),
        **{k: float(v[i]) for k, v in predictions.items()}} for i, ident in enumerate(data.ids)]
    search = [{"outer_fold": r["outer_fold"], "config": c["config"]["name"], "family": c["config"]["family"],
        "epoch": c["epoch"], "inner_mse": c["inner_mse"], **{f"inner_fold{j}_mse": v for j, v in enumerate(c["inner_fold_mse"])},
        "selected": c == r["selected"][c["config"]["family"]]} for r in records for c in r["candidates"]]
    outputs = {"oof": oof, "inner_search": search, "inner_curves": [x for r in records for x in r["inner_curves"]],
        "inner_memberships": [x for r in records for x in r["memberships"]], **tables}
    for name, rows in outputs.items():
        # Epoch0 deliberately lacks optimizer diagnostics; CSV requires a union schema.
        keys = list(dict.fromkeys(k for row in rows for k in row))
        uniform = [{k: row.get(k, "") for k in keys} for row in rows]
        tmp = args.output_dir / (name + ".csv.tmp")
        tmp.write_text(core.csv_text(uniform), encoding="utf-8", newline="")
        os.replace(tmp, args.output_dir / (name + ".csv"))
    (args.output_dir / "report.md").write_text(build_report(summary), encoding="utf-8")
    write_json(args.output_dir / "summary.json", summary)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=8)
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
    if min(args.threads, args.batch_size, args.max_epochs) < 1:
        p.error("Positive threads, batch-size and max-epochs required")
    defaults = evaluation.parse_args(["--seed", str(args.seed)])
    for name in ("folds", "baseline", "previous_frozen_oof", "previous_dino_oof"):
        if getattr(args, name) is None:
            setattr(args, name, getattr(defaults, name))
    args.previous_dino_summary = args.previous_dino_summary or args.previous_dino_oof.with_name("summary.json")
    args.output_dir = args.output_dir or ROOT / f"reports/dinov2_ridge_initialized/seed{args.seed}"
    return args


def main(argv=None):
    args = parse_args(argv)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    start = time.perf_counter()
    if not (args.preflight or args.validate_only):
        previous.protected_output(args)
    data, references, anchor = evaluation.load_prior(args)
    validate_anchor_policy(anchor, references)
    checkpoint_files = prefix.ensure_local_checkpoint()
    if args.preflight:
        print(json.dumps({"status": "prior_artifacts_and_checkpoint_valid", "seed": args.seed,
            "alpha": ALPHA, "prefix_checked": False, "torch_imported": False, "fit_or_forward": False, "output_written": False}))
        return 0
    tokens, metadata = prefix.validate_cache(np, args.prefix_cache, data.ids, data.metadata, args.global_features)
    data.provenance["prefix_tokens"] = {"path": str((args.prefix_cache / "tokens.npy").resolve()), "sha256": core.sha256(args.prefix_cache / "tokens.npy"),
        "metadata_path": str((args.prefix_cache / "metadata.json").resolve()), "metadata_sha256": core.sha256(args.prefix_cache / "metadata.json")}
    data.provenance["ridge_initialized_sources"] = {name: {"path": str(Path(path).resolve()), "sha256": core.sha256(Path(path))}
        for name, path in {"trainer": __file__, "previous_helpers": previous.__file__, "prefix": prefix.__file__, "diagnostics": evaluation.__file__}.items()}
    if args.validate_only:
        print(json.dumps({"status": "all_artifacts_valid", "seed": args.seed, "rows": len(data.y), "alpha": ALPHA,
            "torch_imported": False, "fit_or_forward": False, "output_written": False}))
        return 0
    _, torch, timm, resolve_config, versions = prefix.common.import_runtime()
    device, device_info = prefix.common.configure_runtime(torch, np, args)
    print(f"Device={device}; microbatch={args.batch_size}; effective batch=entire fit subset; max_epochs={args.max_epochs}", flush=True)
    backbone, _ = prefix.base.build_model(torch, timm, resolve_config)
    if prefix.common.feature_state_hash(backbone) != data.metadata["feature_state_sha256"] or len(backbone.blocks) != 12:
        raise ValueError("Initial backbone must exactly match original frozen DINO")
    block, norm = copy.deepcopy(backbone.blocks[-1]).cpu(), copy.deepcopy(backbone.norm).cpu()
    del backbone
    global_features = data.features["cls_patch_mean_std"].astype(np.float32)
    block_device, norm_device = copy.deepcopy(block).to(device).eval(), copy.deepcopy(norm).to(device).eval()
    restored = []
    with torch.no_grad():
        for offset in range(0, len(tokens), args.batch_size):
            x = torch.from_numpy(np.array(tokens[offset:offset + args.batch_size], copy=True)).to(device)
            restored.append(prefix.base.pool_tokens(torch, norm_device(block_device(x)))[1].cpu().numpy())
    difference = prefix.spatial.check_global_match(np, np.concatenate(restored), global_features)
    del block_device, norm_device, restored
    print(f"Initial prefix/suffix global anchor matched; max difference={difference:g}", flush=True)
    runtime = {"versions": versions, "device": str(device), "gpu_name": device_info.get("gpu_name"),
        "checkpoint_files": checkpoint_files, "initial_global_max_difference": difference}
    contract = {"protocol": PROTOCOL, "seed": args.seed, "max_epochs": args.max_epochs,
        "batch_size": args.batch_size, "threads": args.threads, "input_files": data.provenance, "runtime": runtime}
    args.run_fingerprint = previous.prepare_output(args.output_dir, contract, args.resume)
    runner = TorchRunner(torch, block, norm, global_features, tokens, device, args)
    predictions = {name: np.full(len(data.y), np.nan) for name in (CONTROL, PRIMARY)}
    records = []
    for fold in range(5):
        folder = args.output_dir / f"fold{fold}"
        folder.mkdir(exist_ok=True)
        record = previous.read_completed_fold(folder, args.run_fingerprint, fold, data.folds)
        if record is None:
            record = fit_outer_fold(data.y, data.folds, args.seed, fold, runner, args.max_epochs, folder, data.ids, anchor)
            write_json(folder / "result.json", record)
            write_json(folder / "done.json", {"run_fingerprint": args.run_fingerprint,
                "files": {name: core.sha256(folder / name) for name in ("result.json", "head_only.pt", "partial.pt")}})
        else:
            print(f"Outer {fold}: verified completed fold restored", flush=True)
        for name in predictions:
            predictions[name][record["outer_valid_indices"]] = record["predictions"][name]
        records.append(record)
    if not all(np.isfinite(v).all() for v in predictions.values()):
        raise RuntimeError("Incomplete OOF predictions")
    previous.verify_input_files(data.provenance)
    predictions = add_fixed_predictions(predictions, references)
    diagnostics, tables = evaluation.diagnose(data, predictions)
    write_outputs(args, data, predictions, records, diagnostics, tables, runtime, metadata, time.perf_counter() - start)
    for name in (PRIMARY, CONTROL, FIXED, CONTROL_BLEND):
        m = diagnostics[name]["overall"]
        print(f"{name}: RMSE={m['rmse']:.6f}; delta vs current={m['delta_rmse_vs_current_blend']:+.6f}; delta vs DINO={m['delta_rmse_vs_dino_linear']:+.6f}")
    print(f"Report: {args.output_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"Ridge-initialized DINO stopped: {exc}", file=sys.stderr)
        raise SystemExit(2)
