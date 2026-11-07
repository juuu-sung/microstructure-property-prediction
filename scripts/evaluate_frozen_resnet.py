"""Evaluate cached, frozen train features with nested Ridge selection.

This entry point never imports torch, reads images/test data, downloads weights,
fits on all 500 samples, or creates a submission. Run it yourself after extraction.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone

try:
    import numpy as np
except ModuleNotFoundError:
    np = None  # --help remains available before the environment is installed.

ROOT = Path(__file__).resolve().parents[1]
REPRESENTATIONS = ("final_gap", "multiscale_mean_std")
FEATURE_DIMENSIONS = {"final_gap": 512, "multiscale_mean_std": 1792}
ALPHAS = (10.0, 100.0, 1000.0)
BASELINE_NAME = "baseline_ridge_alpha100"
PRIMARY_NAME = "joint_nested"


def validate_alphas(values) -> tuple[float, ...]:
    """Preserve a finite positive unique candidate grid; do not silently fix it."""
    try:
        alphas = tuple(float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise ValueError("--alphas must be a nonempty sequence of finite positive numbers") from exc
    if not alphas or any(not math.isfinite(alpha) or alpha <= 0 for alpha in alphas):
        raise ValueError("--alphas must contain finite positive numbers and cannot be empty")
    if len(set(alphas)) != len(alphas):
        raise ValueError("--alphas must contain unique values; duplicate candidates are not allowed")
    return alphas


def validate_seed(seed: int) -> int:
    # Each inner split uses seed + outer_fold, so reserve room for offsets 0..4.
    if not isinstance(seed, int) or not 0 <= seed <= 2**32 - 5:
        raise ValueError("--seed must be an integer between 0 and 4294967291 (inner folds add offsets 0..4)")
    return seed


@dataclass
class Inputs:
    ids: np.ndarray
    y: np.ndarray
    features: dict[str, np.ndarray]
    folds: np.ndarray
    baseline: np.ndarray
    noise: np.ndarray
    metadata: dict
    provenance: dict


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_rows(path: Path, required: set[str]) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        header = reader.fieldnames or []
        if len(header) != len(set(header)) or not required <= set(header):
            raise ValueError(f"{path}: duplicate headers or missing columns {sorted(required - set(header))}")
        rows = list(reader)
    if not rows or any(None in row or any(row.get(k) in (None, "") for k in required) for row in rows):
        raise ValueError(f"{path}: empty file, ragged row, or missing required value")
    ids = [r["ID"] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path}: duplicate IDs")
    return rows


def align_rows(rows: list[dict], ids: np.ndarray, source: Path) -> list[dict]:
    lookup = {r["ID"]: r for r in rows}
    if set(lookup) != set(ids):
        raise ValueError(f"{source}: ID set must exactly match train.csv (missing/extra IDs)")
    return [lookup[i] for i in ids]


def finite_column(rows: list[dict], name: str, source: Path) -> np.ndarray:
    try:
        values = np.asarray([float(r[name]) for r in rows], dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source}: invalid numeric column {name}") from exc
    if not np.isfinite(values).all():
        raise ValueError(f"{source}: nonfinite values in {name}")
    return values


def integer_column(rows: list[dict], name: str, source: Path) -> np.ndarray:
    try:
        return np.asarray([int(r[name]) for r in rows], dtype=np.int64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source}: {name} must contain integers") from exc


def load_inputs(features_path: Path, train_path: Path, folds_path: Path,
                baseline_path: Path, audit_path: Path, seed: int,
                expected_n: int = 500, *, feature_dimensions=None) -> Inputs:
    """Validate and align every artifact to train.csv; expected_n is relaxed only in unit tests."""
    train = read_rows(train_path, {"ID", "hardness"})
    ids = np.asarray([r["ID"] for r in train])
    if len(ids) != expected_n or expected_n < 50 or expected_n % 5:
        raise ValueError(f"train.csv must contain {expected_n} unique rows; supported count is >=50 and divisible by 5")
    if any(not i.startswith("TRAIN_") for i in ids):
        raise ValueError("train.csv contains IDs outside the TRAIN_ namespace")
    y = finite_column(train, "hardness", train_path)
    if np.std(y) == 0:
        raise ValueError("Training targets must vary")
    fold_rows = align_rows(read_rows(folds_path, {"ID", "fold"}), ids, folds_path)
    folds = integer_column(fold_rows, "fold", folds_path)
    if set(folds) != set(range(5)) or any(np.sum(folds == f) != expected_n // 5 for f in range(5)):
        raise ValueError("Outer folds must be five equally sized folds labelled 0,1,2,3,4")
    if "seed" in fold_rows[0] and not np.all(integer_column(fold_rows, "seed", folds_path) == seed):
        raise ValueError("--seed must match the seed recorded in the fixed folds CSV")
    baseline_rows = align_rows(read_rows(baseline_path, {"ID", "hardness", "fold", "ridge_alpha100"}), ids, baseline_path)
    if not np.array_equal(integer_column(baseline_rows, "fold", baseline_path), folds):
        raise ValueError("Baseline fold IDs do not exactly match the requested outer folds")
    if not np.array_equal(finite_column(baseline_rows, "hardness", baseline_path), y):
        raise ValueError("Baseline target values do not exactly match train.csv")
    baseline = finite_column(baseline_rows, "ridge_alpha100", baseline_path)
    audit_rows = align_rows(read_rows(audit_path, {"ID", "hardness", "haar_hh_noise_proxy"}), ids, audit_path)
    audit_y = finite_column(audit_rows, "hardness", audit_path)
    audit_max_target_difference = float(np.max(np.abs(audit_y - y)))
    # Audit features were exported separately through Pandas CSV serialization.
    # Permit only negligible decimal round-trip differences here. The baseline
    # remains an exact match, and model targets always come from train.csv (y).
    if not np.allclose(audit_y, y, rtol=0.0, atol=1e-10):
        raise ValueError("Audit target values differ from train.csv beyond CSV rounding tolerance (atol=1e-10)")
    noise = finite_column(audit_rows, "haar_hh_noise_proxy", audit_path)
    dimensions = FEATURE_DIMENSIONS if feature_dimensions is None else dict(feature_dimensions)
    if not dimensions or any(not isinstance(name, str) or name in ("ids", "metadata_json") or
                             not isinstance(width, int) or width < 1 for name, width in dimensions.items()):
        raise ValueError("Feature dimensions must map feature names to positive integer widths")
    expected_keys = {"ids", "metadata_json", *dimensions}
    with np.load(features_path, allow_pickle=False) as archive:
        if len(archive.files) != len(expected_keys) or set(archive.files) != expected_keys:
            raise ValueError(f"NPZ must have exactly these keys, without labels: {sorted(expected_keys)}")
        feature_ids = archive["ids"]
        if feature_ids.dtype.kind != "U" or feature_ids.shape != (expected_n,):
            raise ValueError("NPZ ids must be a one-dimensional Unicode array matching train row count")
        if len(set(feature_ids)) != expected_n or set(feature_ids) != set(ids):
            raise ValueError("NPZ contains duplicate, missing, or extra IDs")
        order_lookup = {sample_id: i for i, sample_id in enumerate(feature_ids)}
        order = [order_lookup[sample_id] for sample_id in ids]
        features = {}
        for name, dimension in dimensions.items():
            values = archive[name]
            if values.dtype != np.dtype("float32") or values.shape != (expected_n, dimension):
                raise ValueError(f"NPZ {name} must be float32 with shape ({expected_n}, {dimension})")
            if not np.isfinite(values).all():
                raise ValueError(f"NPZ {name} contains NaN or infinity")
            features[name] = values[order].copy()
        raw_metadata = archive["metadata_json"]
        if raw_metadata.dtype.kind != "U" or raw_metadata.shape != ():
            raise ValueError("NPZ metadata_json must be a Unicode scalar")
        metadata = json.loads(raw_metadata.item())
        if not isinstance(metadata, dict):
            raise ValueError("NPZ metadata_json must encode a JSON object")
    train_hash = sha256(train_path)
    if "train_csv_sha256" in metadata and metadata["train_csv_sha256"] != train_hash:
        raise ValueError("Feature extraction train_csv_sha256 does not match the current train.csv")
    paths = {"features": features_path, "train_csv": train_path, "folds": folds_path,
             "baseline": baseline_path, "audit_features": audit_path}
    provenance = {k: {"path": str(p.resolve()), "sha256": train_hash if k == "train_csv" else sha256(p)} for k, p in paths.items()}
    provenance["audit_features"]["target_alignment"] = {
        "rtol": 0.0, "atol": 1e-10, "max_absolute_difference": audit_max_target_difference,
        "reason": "Allow only decimal round-trip differences from the separately serialized audit CSV.",
        "model_target_source": "train.csv only; audit targets are never used for fitting or scoring",
    }
    return Inputs(ids, y, features, folds, baseline, noise, metadata, provenance)


def rank_groups(values: np.ndarray, count: int) -> np.ndarray:
    return np.argsort(np.argsort(values, kind="stable"), kind="stable") * count // len(values)


def make_estimator(alpha: float):
    from sklearn.compose import TransformedTargetRegressor
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    # Both scalers are fitted afresh by every .fit call, including inner folds.
    return TransformedTargetRegressor(
        regressor=make_pipeline(StandardScaler(), Ridge(alpha=alpha, solver="auto")),
        transformer=StandardScaler(),
    )


def nested_predictions(data: Inputs, seed: int, threads: int = 2,
                       alphas=ALPHAS, *, representations=None, estimator_factory=None) -> tuple[dict, list]:
    alphas = validate_alphas(alphas)
    seed = validate_seed(seed)
    representations = REPRESENTATIONS if representations is None else tuple(representations)
    if not representations or len(set(representations)) != len(representations) or any(name not in data.features for name in representations):
        raise ValueError("Representations must be unique names present in the input features")
    if estimator_factory is not None and not callable(estimator_factory):
        raise ValueError("estimator_factory must be callable or None")
    create_estimator = estimator_factory if estimator_factory is not None else lambda alpha, representation: make_estimator(alpha)
    from sklearn.model_selection import StratifiedKFold
    from threadpoolctl import threadpool_limits
    predictions = {f"{name}_nested": np.full(len(data.y), np.nan) for name in representations}
    predictions[PRIMARY_NAME] = np.full(len(data.y), np.nan)
    selections = []
    with threadpool_limits(limits=threads):
        for outer_fold in range(5):
            outer_train = np.flatnonzero(data.folds != outer_fold)
            outer_valid = np.flatnonzero(data.folds == outer_fold)
            train_y = data.y[outer_train]
            if len(train_y) < 30:
                raise ValueError("Each outer training set needs at least 30 samples for inner target deciles")
            inner_seed = seed + outer_fold
            strata = rank_groups(train_y, 10)
            inner_splits = list(StratifiedKFold(n_splits=3, shuffle=True, random_state=inner_seed).split(train_y, strata))
            candidates = []
            for representation in representations:
                train_x = data.features[representation][outer_train].astype(np.float64)
                for alpha in alphas:
                    inner_pred = np.full(len(outer_train), np.nan)
                    fold_mse = []
                    for inner_train, inner_valid in inner_splits:
                        estimator = create_estimator(alpha, representation)
                        estimator.fit(train_x[inner_train], train_y[inner_train])
                        inner_pred[inner_valid] = estimator.predict(train_x[inner_valid])
                        fold_mse.append(float(np.mean((inner_pred[inner_valid] - train_y[inner_valid]) ** 2)))
                    if not np.isfinite(inner_pred).all():
                        raise ValueError("Inner predictions contain nonfinite values")
                    candidates.append({"representation": representation, "alpha": alpha,
                                       "inner_mse": float(np.mean((inner_pred - train_y) ** 2)),
                                       "inner_fold_mse": fold_mse})
            # Deterministic ties: declared representation order, then stronger regularization.
            key = lambda c: (c["inner_mse"], representations.index(c["representation"]), -c["alpha"])
            per_representation = {}
            for representation in representations:
                winner = min((c for c in candidates if c["representation"] == representation), key=key)
                estimator = create_estimator(winner["alpha"], representation)
                estimator.fit(data.features[representation][outer_train].astype(np.float64), train_y)
                outer_pred = estimator.predict(data.features[representation][outer_valid].astype(np.float64))
                if not np.isfinite(outer_pred).all():
                    raise ValueError("Outer predictions contain nonfinite values")
                predictions[f"{representation}_nested"][outer_valid] = outer_pred
                per_representation[representation] = dict(winner)
            joint_winner = min(candidates, key=key)
            predictions[PRIMARY_NAME][outer_valid] = predictions[f"{joint_winner['representation']}_nested"][outer_valid]
            selections.append({"outer_fold": outer_fold, "outer_train_n": len(outer_train),
                               "outer_valid_n": len(outer_valid), "inner_seed": inner_seed,
                               "inner_validation_sizes": [len(valid) for _, valid in inner_splits],
                               "candidates": candidates, "per_representation": per_representation,
                               "joint": dict(joint_winner)})
            print(f"Outer fold {outer_fold}: joint={joint_winner['representation']} alpha={joint_winner['alpha']:g}; outer predictions stored", flush=True)
    if any(not np.isfinite(p).all() for p in predictions.values()):
        raise ValueError("OOF predictions are incomplete or nonfinite")
    return predictions, selections


def metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    error = pred - y
    return {"n": len(y), "rmse": float(np.sqrt(np.mean(error ** 2))),
            "mae": float(np.mean(np.abs(error))), "bias_pred_minus_target": float(error.mean()),
            "sse": float(np.sum(error ** 2)), "target_std": float(y.std()),
            "prediction_std": float(pred.std()), "prediction_min": float(pred.min()),
            "prediction_max": float(pred.max())}


def correlation(a: np.ndarray, b: np.ndarray):
    return float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 and np.std(b) > 0 else None


def diagnose(data: Inputs, predictions: dict) -> tuple[dict, dict[str, list]]:
    all_predictions = {BASELINE_NAME: data.baseline, **predictions}
    target_bins, noise_bins = rank_groups(data.y, 10), rank_groups(data.noise, 4)
    tables = {name: [] for name in ("metrics", "fold_metrics", "target_decile_metrics", "noise_quartile_metrics", "residual_correlations", "top20_error_changes", "per_image_errors")}
    baseline_error = data.baseline - data.y
    baseline_worst = set(np.argsort(baseline_error ** 2, kind="stable")[-20:])
    baseline_rmse = metrics(data.y, data.baseline)["rmse"]
    diagnostics = {}
    for name, pred in all_predictions.items():
        error = pred - data.y
        values = metrics(data.y, pred)
        worst = set(np.argsort(error ** 2, kind="stable")[-20:])
        values.update({"delta_rmse_vs_baseline": values["rmse"] - baseline_rmse,
                       "residual_correlation_with_baseline": correlation(error, baseline_error),
                       "top20_sse_fraction": float(np.sum(error[list(worst)] ** 2) / values["sse"]) if values["sse"] else 0.0,
                       "top20_overlap_with_baseline": len(worst & baseline_worst)})
        tables["metrics"].append({"model": name, **values})
        diagnostics[name] = {"overall": values, "target_deciles": [], "noise_quartiles": []}
        improved = 0
        for fold in range(5):
            mask = data.folds == fold
            fold_values = metrics(data.y[mask], pred[mask])
            delta = fold_values["rmse"] - metrics(data.y[mask], data.baseline[mask])["rmse"]
            improved += int(delta < 0)
            tables["fold_metrics"].append({"model": name, "fold": fold, **fold_values, "delta_rmse_vs_baseline": delta})
        values["folds_improved_vs_baseline"] = improved
        tables["metrics"][-1]["folds_improved_vs_baseline"] = improved
        for label, bins, count, table_name, output_key in [("decile", target_bins, 10, "target_decile_metrics", "target_deciles"), ("quartile", noise_bins, 4, "noise_quartile_metrics", "noise_quartiles")]:
            for group in range(count):
                mask = bins == group
                group_values = metrics(data.y[mask], pred[mask])
                group_values.update({"sse_fraction": group_values["sse"] / values["sse"] if values["sse"] else 0.0,
                                     "target_min": float(data.y[mask].min()), "target_max": float(data.y[mask].max())})
                if label == "quartile":
                    group_values.update({"noise_min": float(data.noise[mask].min()), "noise_max": float(data.noise[mask].max())})
                row = {"model": name, label: group + 1, **group_values}
                tables[table_name].append(row)
                diagnostics[name][output_key].append(row)
        if name != BASELINE_NAME:
            for index, sample_id in enumerate(data.ids):
                tables["per_image_errors"].append({"model": name, "ID": str(sample_id), "fold": int(data.folds[index]),
                    "hardness": float(data.y[index]), "prediction": float(pred[index]), "baseline_prediction": float(data.baseline[index]),
                    "error_pred_minus_target": float(error[index]), "baseline_error": float(baseline_error[index]),
                    "squared_error": float(error[index] ** 2), "baseline_squared_error": float(baseline_error[index] ** 2),
                    "delta_squared_error": float(error[index] ** 2 - baseline_error[index] ** 2),
                    "target_decile": int(target_bins[index] + 1), "noise_proxy_quartile": int(noise_bins[index] + 1)})
            ranks = np.empty(len(error), dtype=int)
            ranks[np.argsort(-(error ** 2), kind="stable")] = np.arange(1, len(error) + 1)
            for index in sorted(worst | baseline_worst, key=lambda i: (ranks[i], data.ids[i])):
                tables["top20_error_changes"].append({"model": name, "ID": str(data.ids[index]), "fold": int(data.folds[index]),
                    "hardness": float(data.y[index]), "prediction": float(pred[index]), "baseline_prediction": float(data.baseline[index]),
                    "error": float(error[index]), "baseline_error": float(baseline_error[index]),
                    "delta_squared_error": float(error[index] ** 2 - baseline_error[index] ** 2),
                    "current_error_rank": int(ranks[index]), "in_current_top20": index in worst, "in_baseline_top20": index in baseline_worst})
    for first, pred_a in all_predictions.items():
        for second, pred_b in all_predictions.items():
            tables["residual_correlations"].append({"model_a": first, "model_b": second, "correlation": correlation(pred_a - data.y, pred_b - data.y)})
    return diagnostics, tables


def csv_text(rows: list[dict]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def build_markdown(summary: dict, report_title="고정 ResNet18 특징 + nested Ridge 검증") -> str:
    diagnostics = summary["diagnostics"]
    alpha_text = ", ".join(f"{alpha:g}" for alpha in summary["alpha_candidates"])
    lines = [f"# {report_title}", "",
             "주 평가값은 **joint_nested의 전체 OOF RMSE**입니다. 각 outer 학습 부분의 inner 3-fold에서 표현과 alpha를 함께 선택한 뒤 outer validation에서 한 번 평가했습니다.", "",
             "두 표현별 점수는 비교 진단입니다. 이 두 outer 점수 중 낮은 것을 다시 고른 결과를 선택 편향 없는 주 점수로 해석하지 않습니다.", "",
             "| 모델 | 전체 OOF RMSE | 기준선 대비 ΔRMSE | 개선 fold 수 |", "|---|---:|---:|---:|"]
    for name, result in diagnostics.items():
        m = result["overall"]
        lines.append(f"| {name} | {m['rmse']:.5f} | {m['delta_rmse_vs_baseline']:+.5f} | {m['folds_improved_vs_baseline']}/5 |")
    lines.extend(["", "ΔRMSE가 음수이면 같은 fold의 기존 Ridge 기준선보다 개선된 것입니다. fold 점수 평균이 아니라 모든 OOF 제곱오차의 평균에 제곱근을 적용했습니다.", "",
                  "## Outer fold별 inner 선택", "", "| outer fold | 선택 표현 | alpha | inner OOF MSE |", "|---|---|---:|---:|"])
    for record in summary["selections"]:
        winner = record["joint"]
        lines.append(f"| {record['outer_fold']} | {winner['representation']} | {winner['alpha']:g} | {winner['inner_mse']:.5f} |")
    lines.extend(["", "## 양끝 경도와 잡음 대용지표 진단", "", "| 모델 | 최저 10% RMSE | 최고 10% RMSE | 낮은 잡음 Q1 RMSE | 높은 잡음 Q4 RMSE |", "|---|---:|---:|---:|---:|"])
    for name, result in diagnostics.items():
        target, noise = result["target_deciles"], result["noise_quartiles"]
        lines.append(f"| {name} | {target[0]['rmse']:.3f} | {target[-1]['rmse']:.3f} | {noise[0]['rmse']:.3f} | {noise[-1]['rmse']:.3f} |")
    lines.extend(["", "잡음 대용지표는 실제 미세구조를 함께 반영할 수 있습니다. 위 구간은 사후 오류 진단이며 추론 시 모델을 선택하는 기준이 아닙니다.", "",
                  "## 해석과 실행 범위", "",
                  f"- alpha 후보는 {alpha_text}입니다. 입력과 정답 StandardScaler는 매 inner 학습 fold 안에서 새로 적합합니다.",
                  "- outer fold는 제공된 고정 CSV를 그대로 사용하며 기준선의 ID, 정답, fold가 정확히 일치할 때만 실행됩니다.",
                  "- inner 선택으로 인한 누출은 막지만, 동일한 500장을 반복해 실험하는 전체 연구 과정의 선택 편향까지 제거하지는 않습니다.",
                  "- 기존 기준선은 이전 CV 비교에서 선택된 모델입니다. 독립적인 holdout 결과 또는 대회 Public/Private 점수가 아닙니다.",
                  "- 이 실행은 train의 cached features와 정답만 사용합니다. test 읽기, 전체 500장 최종 재학습, 제출 파일 생성은 수행하지 않습니다.",
                  "- 예측 clipping, calibration, tail 가중 또는 앙상블을 적용하지 않았습니다.",
                  "- top20_error_changes.csv는 기준선 또는 현재 모델의 최대 오차 20장에 속하는 ID의 합집합을 보여 줍니다.", "",
                  "상세 선택, 버전, 입력 SHA256, 추출 metadata는 summary.json에 기록됩니다. raw OOF는 oof.csv, 구간별 signed error와 SSE 비중은 진단 CSV를 확인하세요.", ""])
    return "\n".join(lines)


def write_outputs(output_dir: Path, data: Inputs, predictions: dict, selections: list,
                  diagnostics: dict, tables: dict, seed: int, threads: int, elapsed: float,
                  alphas=ALPHAS, *, summary_extras=None, report_builder=None):
    alphas = validate_alphas(alphas)
    import scipy
    import sklearn
    from threadpoolctl import threadpool_info
    summary = {"created_utc": datetime.now(timezone.utc).isoformat(), "primary_metric": "joint_nested overall OOF RMSE",
               "seed": seed, "threads": threads, "elapsed_seconds": elapsed, "n_train": len(data.y),
               "outer_splits": 5, "inner_splits": 3, "alpha_candidates": list(alphas),
               "feature_dimensions": {k: v.shape[1] for k, v in data.features.items()},
               "selection_objective": "pooled inner out-of-fold MSE in original hardness units",
               "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__},
               "threadpool_libraries": threadpool_info(), "input_files": data.provenance,
               "feature_metadata": data.metadata, "selections": selections, "diagnostics": diagnostics,
               "scope": "Train-only nested OOF evaluation; no all-data final fitting; no test or submission.",
               "limitations": ["Repeated research on the same dataset can still cause selection bias.", "Baseline was selected from prior CV comparisons.", "Noise proxy can mix image noise with true structure."]}
    if summary_extras:
        if set(summary_extras) & set(summary):
            raise ValueError("Additional summary fields cannot replace the evaluation's primary results")
        summary.update(summary_extras)
    oof_rows = [{"ID": str(sample_id), "hardness": float(data.y[i]), "fold": int(data.folds[i]), BASELINE_NAME: float(data.baseline[i]),
                 **{name: float(pred[i]) for name, pred in predictions.items()}} for i, sample_id in enumerate(data.ids)]
    selection_rows = []
    for record in selections:
        for candidate in record["candidates"]:
            representation = candidate["representation"]
            selection_rows.append({"outer_fold": record["outer_fold"], "inner_seed": record["inner_seed"],
                "representation": representation, "alpha": candidate["alpha"], "inner_mse": candidate["inner_mse"],
                **{f"inner_fold{j}_mse": score for j, score in enumerate(candidate["inner_fold_mse"])},
                "selected_for_representation": candidate == record["per_representation"][representation],
                "selected_joint": candidate == record["joint"]})
    outputs = {"summary.json": json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False),
               "report.md": (report_builder or build_markdown)(summary), "oof.csv": csv_text(oof_rows),
               "inner_search.csv": csv_text(selection_rows)}
    outputs.update({f"{name}.csv": csv_text(rows) for name, rows in tables.items()})
    for name, content in outputs.items():
        destination = output_dir / name
        temporary = output_dir / (name + f".{os.getpid()}.tmp")
        temporary.write_text(content, encoding="utf-8", newline="")
        os.replace(temporary, destination)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", type=Path, default=ROOT / "reports/frozen_resnet18/features.npz")
    parser.add_argument("--train-csv", type=Path, default=ROOT / "open/train.csv")
    parser.add_argument("--folds", type=Path, default=ROOT / "reports/folds.csv")
    parser.add_argument("--baseline", type=Path, default=ROOT / "reports/benchmark_oof_seed42.csv")
    parser.add_argument("--audit-features", type=Path, default=ROOT / "reports/audit_train_features.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/frozen_resnet18/seed42")
    parser.add_argument("--seed", type=int, default=42, help="Must match the fixed fold CSV when its seed column exists")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--alphas", nargs="+", type=float, default=ALPHAS,
                        help="Finite positive unique Ridge alpha candidates (default: 10 100 1000)")
    parser.add_argument("--overwrite", action="store_true", help="Explicitly allow replacing evaluation result files; other files are preserved")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    alphas = validate_alphas(args.alphas)
    seed = validate_seed(args.seed)
    if np is None:
        raise ValueError("NumPy is not installed. Install the experiment requirements in your virtual environment first.")
    if args.threads < 1:
        raise ValueError("--threads must be positive")
    if args.output_dir.exists() and (not args.output_dir.is_dir() or any(args.output_dir.iterdir())) and not args.overwrite:
        raise ValueError(f"Output directory already contains files: {args.output_dir}. Use a new directory or explicit --overwrite.")
    if args.output_dir.exists() and not args.output_dir.is_dir():
        raise ValueError("--output-dir must be a directory")
    start = time.perf_counter()
    data = load_inputs(args.features, args.train_csv, args.folds, args.baseline, args.audit_features, seed)
    print(f"Validated {len(data.y)} train IDs; both feature arrays, outer folds, baseline, and audit match.", flush=True)
    predictions, selections = nested_predictions(data, seed, args.threads, alphas=alphas)
    diagnostics, tables = diagnose(data, predictions)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_outputs(args.output_dir, data, predictions, selections, diagnostics, tables, seed, args.threads, time.perf_counter() - start, alphas=alphas)
    primary = diagnostics[PRIMARY_NAME]["overall"]
    print(f"PRIMARY joint_nested OOF RMSE: {primary['rmse']:.6f} (delta vs baseline {primary['delta_rmse_vs_baseline']:+.6f})")
    print(f"Report: {(args.output_dir / 'report.md').resolve()}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, KeyError) as exc:
        print(f"Evaluation stopped: {exc}", file=sys.stderr)
        raise SystemExit(2)
