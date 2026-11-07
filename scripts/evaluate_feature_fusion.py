"""User-run train-only nested Ridge evaluation of handcrafted+DINO features.

Two prespecified feature scalings, six alpha values, inner 3-fold selection and
fixed outer 5-fold evaluation. No torch/images/test/downloads/final all-data fit.
--validate-only reads and validates existing artifacts without fitting or saving.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

if __package__:
    from . import evaluate_frozen_resnet as core
    from . import evaluate_frozen_dinov2 as dino
else:
    import evaluate_frozen_resnet as core
    import evaluate_frozen_dinov2 as dino

np = core.np
ROOT = core.ROOT
HAND_DIM, DINO_DIM = 125, 1152
TOTAL_DIM = HAND_DIM + DINO_DIM
REPRESENTATIONS = ("concat_standardized", "concat_block_balanced")
ALPHAS = dino.ALPHAS
BLOCK_WEIGHTS = {"hand": math.sqrt(TOTAL_DIM / (2 * HAND_DIM)), "dino": math.sqrt(TOTAL_DIM / (2 * DINO_DIM))}
CURRENT_REFERENCE = "reference_current_blend"
DINO_REFERENCE = "reference_dino_joint"
HAND_DINO_REFERENCE = "reference_hand_dino_equal"
FIXED_BLEND = "current_blend_fusion_equal"
HAND_PROVENANCE_LIMITATION = "Historical handcrafted cache has no embedded image/source hashes. Current file/array hashes and feature-name checks identify this snapshot; they do not prove which image/source versions originally generated it."


def expected_handcrafted_names():
    """The fixed 125-column schema only; never import the old feature generator."""
    names = [f"intensity_{name}" for name in ("mean", "std", "skew", "kurtosis")]
    names += [f"intensity_q{q}" for q in (1, 5, 10, 25, 50, 75, 90, 95, 99)]
    names += [f"intensity_hist{j:02d}" for j in range(12)]
    for threshold in (32, 64, 96, 128, 160, 192):
        names += [f"dark{threshold}_fraction", f"dark{threshold}_count"]
        names += [f"dark{threshold}_area_{name}" for name in ("median", "q90", "max")]
    for sigma in (0.6, 1.2, 2.4, 4.8):
        names += [f"scale{sigma}_{name}" for name in ("residual_std", "residual_abs", "gradient_mean", "gradient_std", "gradient_q90", "laplacian_abs")]
        if sigma == 2.4:
            names += ["orientation_anisotropy", "orientation_cos2", "orientation_sin2"]
    names += [f"fft_band{j}" for j in range(7)] + ["fft_centroid", "fft_radius_rms", "fft_cos2", "fft_sin2"]
    for lag in (1, 2, 4, 8, 16, 32):
        names += [f"autocorrelation{lag}_mean", f"autocorrelation{lag}_xy_difference"]
    for lag in (1, 4, 12):
        names += [f"glcm{lag}_{name}" for name in ("contrast", "homogeneity", "energy", "entropy")]
    for size in (8, 16, 32, 64):
        names += [f"block{size}_mean_std", f"block{size}_mean_q90q10"]
    assert len(names) == HAND_DIM and len(set(names)) == HAND_DIM
    return names


def load_handcrafted(path, definition_path, ids):
    definition = json.loads(definition_path.read_text(encoding="utf-8"))
    with np.load(path, allow_pickle=False) as cache:
        if len(cache.files) != 3 or set(cache.files) != {"ids", "X", "feature_names"}:
            raise ValueError("Handcrafted NPZ must contain exactly ids, X, feature_names; no labels")
        cache_ids, values, names = cache["ids"], cache["X"], cache["feature_names"]
        if cache_ids.dtype.kind != "U" or cache_ids.shape != (len(ids),) or len(set(cache_ids)) != len(ids) or set(cache_ids) != set(ids):
            raise ValueError("Handcrafted IDs must be unique Unicode IDs exactly matching train.csv")
        if values.dtype != np.float64 or values.shape != (len(ids), HAND_DIM) or not np.isfinite(values).all():
            raise ValueError("Handcrafted X must be finite float64 with exactly 125 columns")
        if names.dtype.kind != "U" or names.shape != (HAND_DIM,) or len(set(names)) != HAND_DIM:
            raise ValueError("Handcrafted feature names must be 125 unique Unicode values")
        if names.tolist() != definition.get("feature_definition") or names.tolist() != expected_handcrafted_names():
            raise ValueError("Handcrafted feature-name order differs from the historical JSON or fixed 125-column schema")
        if definition.get("n_features") != HAND_DIM or definition.get("n_train") != len(ids):
            raise ValueError("Handcrafted definition row/feature counts do not match")
        index = {ident: i for i, ident in enumerate(cache_ids)}
        aligned = values[[index[ident] for ident in ids]].copy()
        provenance = {"path": str(path.resolve()), "sha256": core.sha256(path),
            "raw_array_sha256": dino.common.sha256_array(values), "aligned_array_sha256": dino.common.sha256_array(aligned),
            "feature_names": names.tolist(), "definition_path": str(definition_path.resolve()),
            "definition_sha256": core.sha256(definition_path), "historical_provenance_verified": False,
            "limitation": HAND_PROVENANCE_LIMITATION}
    return aligned, provenance


def load_previous_dino(data, frozen_references, previous_path, summary_path, seed):
    required = {"ID", "hardness", "fold", core.BASELINE_NAME, core.PRIMARY_NAME,
                dino.FROZEN_REFERENCE, dino.PREVIOUS_REFERENCE, "baseline_dino_equal", "previous_blend_dino_equal"}
    rows = core.align_rows(core.read_rows(previous_path, required), data.ids, previous_path)
    if not np.array_equal(core.integer_column(rows, "fold", previous_path), data.folds):
        raise ValueError("Previous DINO OOF folds do not match")
    if not np.array_equal(core.finite_column(rows, "hardness", previous_path), data.y):
        raise ValueError("Previous DINO OOF targets do not exactly match train.csv")
    previous_joint = core.finite_column(rows, core.PRIMARY_NAME, previous_path)
    current = 0.5 * frozen_references[dino.PREVIOUS_REFERENCE] + 0.5 * previous_joint
    hand_dino = 0.5 * data.baseline + 0.5 * previous_joint
    expected = {core.BASELINE_NAME: data.baseline, **frozen_references,
                "baseline_dino_equal": hand_dino, "previous_blend_dino_equal": current}
    for column, values in expected.items():
        if not np.array_equal(core.finite_column(rows, column, previous_path), values):
            raise ValueError(f"Previous DINO OOF reference/formula mismatch: {column}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("seed") != seed or summary.get("n_train") != len(data.ids) or summary.get("outer_splits") != 5:
        raise ValueError("Previous DINO summary seed/row/fold count does not match")
    for source in ("features", "train_csv", "folds", "baseline", "previous_frozen_oof"):
        old_hash = summary.get("input_files", {}).get(source, {}).get("sha256")
        if old_hash != data.provenance[source]["sha256"]:
            raise ValueError(f"Previous DINO summary input hash mismatch: {source}")
    if summary.get("feature_dimensions") != dino.FEATURE_DIMENSIONS or summary.get("feature_metadata") != data.metadata:
        raise ValueError("Previous DINO summary feature dimensions/metadata do not match the current DINO cache")
    for name, values in {core.PRIMARY_NAME: previous_joint, "previous_blend_dino_equal": current,
                         "baseline_dino_equal": hand_dino}.items():
        recorded = summary.get("diagnostics", {}).get(name, {}).get("overall", {}).get("rmse")
        if not isinstance(recorded, (int, float)) or not math.isclose(recorded, core.metrics(data.y, values)["rmse"], rel_tol=0, abs_tol=1e-10):
            raise ValueError(f"Previous DINO summary/OOF RMSE mismatch: {name}")
    data.provenance["previous_dino_oof"] = {"path": str(previous_path.resolve()), "sha256": core.sha256(previous_path),
        "summary_path": str(summary_path.resolve()), "summary_sha256": core.sha256(summary_path),
        "targets_folds_baseline_resnet_and_fixed_formulas_exactly_match": True,
        "summary_input_hashes_and_dino_cache_verified": True,
        "source_version_note": "Past evaluator source hashes are historical provenance, not required to equal the current evaluator code."}
    return {CURRENT_REFERENCE: current, DINO_REFERENCE: previous_joint, HAND_DINO_REFERENCE: hand_dino}


def load_inputs(features_path, train_path, folds_path, baseline_path, audit_path, previous_frozen_path,
                previous_dino_path, handcrafted_path, definition_path, seed, *, metadata_path=None,
                previous_dino_summary=None, expected_n=500):
    data, frozen_references = dino.load_inputs(features_path, train_path, folds_path, baseline_path,
        audit_path, previous_frozen_path, seed, metadata_path=metadata_path, expected_n=expected_n)
    summary_path = previous_dino_summary or previous_dino_path.with_name("summary.json")
    references = load_previous_dino(data, frozen_references, previous_dino_path, summary_path, seed)
    hand, provenance = load_handcrafted(handcrafted_path, definition_path, data.ids)
    combined = np.concatenate([hand, data.features["cls_patch_mean_std"].astype(np.float64)], axis=1)
    combined.flags.writeable = False
    data.features = {representation: combined for representation in REPRESENTATIONS}
    data.provenance["handcrafted_features"] = provenance
    data.provenance["feature_fusion_layout"] = {"column_order": ["handcrafted_125", "dino_cls_patch_mean_std_1152"],
        "total_columns": TOTAL_DIM, "audit_features_used_as_model_inputs": False,
        "block_weights_after_within_fold_standardization": BLOCK_WEIGHTS}
    data.provenance["evaluation_sources"] = {name: {"path": str(Path(module_path).resolve()), "sha256": core.sha256(Path(module_path))}
        for name, module_path in {"feature_fusion": __file__, "nested_core": core.__file__, "dino_validator": dino.__file__}.items()}
    return data, references


def make_estimator(alpha, representation):
    from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    if representation not in REPRESENTATIONS:
        raise ValueError("Unknown feature-fusion representation")
    weights = BLOCK_WEIGHTS if representation == "concat_block_balanced" else None
    blocks = ColumnTransformer([
        ("hand", StandardScaler(), slice(0, HAND_DIM)),
        ("dino", StandardScaler(), slice(HAND_DIM, TOTAL_DIM)),
    ], transformer_weights=weights, remainder="drop", sparse_threshold=0.0)
    # No later X StandardScaler: it would cancel the fixed block weights.
    return TransformedTargetRegressor(regressor=Pipeline([("blocks", blocks), ("ridge", Ridge(alpha=alpha, solver="auto"))]),
                                      transformer=StandardScaler())


def add_fixed_candidate(data, predictions, references):
    expected = {f"{name}_nested" for name in REPRESENTATIONS} | {core.PRIMARY_NAME}
    if set(predictions) != expected or set(references) != {CURRENT_REFERENCE, DINO_REFERENCE, HAND_DINO_REFERENCE}:
        raise ValueError("Unexpected fusion predictions or reference columns")
    return {**predictions, **references, FIXED_BLEND: 0.5 * references[CURRENT_REFERENCE] + 0.5 * predictions[core.PRIMARY_NAME]}


def diagnose(data, predictions):
    diagnostics, tables = core.diagnose(data, predictions)
    all_predictions = {core.BASELINE_NAME: data.baseline, **predictions}
    reference = predictions[CURRENT_REFERENCE]
    reference_error = reference - data.y
    reference_rmse = core.metrics(data.y, reference)["rmse"]
    hand_dino_rmse = core.metrics(data.y, predictions[HAND_DINO_REFERENCE])["rmse"]
    for row in tables["metrics"]:
        pred = all_predictions[row["model"]]
        extra = {"delta_rmse_vs_current_blend": row["rmse"] - reference_rmse,
                 "delta_rmse_vs_hand_dino_equal": row["rmse"] - hand_dino_rmse,
                 "folds_improved_vs_current_blend": sum(core.metrics(data.y[data.folds == f], pred[data.folds == f])["rmse"]
                     < core.metrics(data.y[data.folds == f], reference[data.folds == f])["rmse"] for f in range(5)),
                 "residual_correlation_with_current_blend": core.correlation(pred - data.y, reference_error)}
        row.update(extra)
        diagnostics[row["model"]]["overall"].update(extra)
    for table_name, group_key in (("fold_metrics", "fold"), ("target_decile_metrics", "decile"), ("noise_quartile_metrics", "quartile")):
        base = {r[group_key]: r["rmse"] for r in tables[table_name] if r["model"] == CURRENT_REFERENCE}
        for row in tables[table_name]:
            row["delta_rmse_vs_current_blend"] = row["rmse"] - base[row[group_key]]
    indices = {str(ident): i for i, ident in enumerate(data.ids)}
    for row in tables["per_image_errors"]:
        i = indices[row["ID"]]
        row.update(current_blend_prediction=float(reference[i]), current_blend_squared_error=float(reference_error[i] ** 2),
                   delta_squared_error_vs_current_blend=float(row["squared_error"] - reference_error[i] ** 2))
    target_bins = core.rank_groups(data.y, 10)
    regions = {"low10": target_bins == 0, "high10": target_bins == 9,
               "tails20": (target_bins == 0) | (target_bins == 9), "middle80": (target_bins > 0) & (target_bins < 9)}
    tables["target_region_metrics"] = []
    for name, pred in all_predictions.items():
        diagnostics[name]["target_regions"] = {}
        for region, mask in regions.items():
            values = core.metrics(data.y[mask], pred[mask])
            values["delta_rmse_vs_current_blend"] = values["rmse"] - core.metrics(data.y[mask], reference[mask])["rmse"]
            diagnostics[name]["target_regions"][region] = values
            tables["target_region_metrics"].append({"model": name, "region": region, **values})
    previous_worst = set(np.argsort(reference_error ** 2, kind="stable")[-20:])
    tables["top20_error_changes_vs_current_blend"] = []
    for name, pred in all_predictions.items():
        if name == CURRENT_REFERENCE:
            continue
        errors = pred - data.y
        ranks = np.empty(len(data.y), dtype=int)
        ranks[np.argsort(-(errors ** 2), kind="stable")] = np.arange(1, len(data.y) + 1)
        worst = set(np.argsort(errors ** 2, kind="stable")[-20:])
        diagnostics[name]["overall"]["top20_overlap_with_current_blend"] = len(worst & previous_worst)
        for i in sorted(worst | previous_worst, key=lambda index: ranks[index]):
            tables["top20_error_changes_vs_current_blend"].append({"model": name, "ID": str(data.ids[i]),
                "fold": int(data.folds[i]), "hardness": float(data.y[i]), "prediction": float(pred[i]),
                "current_blend_prediction": float(reference[i]), "error": float(errors[i]),
                "current_blend_error": float(reference_error[i]), "delta_squared_error_vs_current_blend": float(errors[i] ** 2 - reference_error[i] ** 2),
                "current_error_rank": int(ranks[i]), "in_current_top20": i in worst, "in_reference_top20": i in previous_worst})
    return diagnostics, tables


def build_markdown(summary):
    lines = ["# 수작업 특징 + DINO 특징 결합 실험", "",
        "주 평가값은 **fusion joint_nested OOF RMSE**입니다. outer 학습 부분의 inner 3-fold에서 두 특징 스케일링 방식과 alpha를 공동 선택했습니다.", "",
        "현재 비교 기준은 수작업 Ridge 25% + frozen ResNet18 25% + DINO 50%입니다. 기존 수작업 Ridge+DINO 50:50 행은 같은 두 정보원을 예측 단계에서 합쳤을 때의 참고값입니다.", "",
        "| 모델 | OOF RMSE | 현재 25:25:50 대비 Δ | 개선 fold |", "|---|---:|---:|---:|"]
    for name, result in summary["diagnostics"].items():
        value = result["overall"]
        lines.append(f"| {name} | {value['rmse']:.6f} | {value['delta_rmse_vs_current_blend']:+.6f} | {value['folds_improved_vs_current_blend']}/5 |")
    lines += ["", "Δ가 음수이면 개선입니다. 전체 OOF 제곱오차를 합쳐 RMSE를 계산했습니다.", "",
        "## 고정한 실험 설계", "",
        "- 입력은 수작업 125개 + DINO CLS/patch 평균/표준편차 1152개, 총 1277개입니다. audit 특징은 모델 입력에 넣지 않습니다.",
        "- concat_standardized: 각 블록의 모든 열을 해당 학습 fold에서 표준화합니다.",
        f"- concat_block_balanced: 같은 표준화 후 수작업 블록 ×{BLOCK_WEIGHTS['hand']:.8f}, DINO 블록 ×{BLOCK_WEIGHTS['dino']:.8f}를 적용한 뒤 바로 Ridge로 넘깁니다.",
        "- 블록 배율은 sqrt(1277/(2×블록차원))입니다. 열 개수에 따른 표준화 분산 규모를 근사적으로 맞추며, 상수열·열 상관 때문에 실제 정보 기여가 같다는 뜻은 아닙니다.",
        "- X/y의 학습형 전처리는 매 inner 학습 부분에서만 적합합니다. 블록 가중치 뒤 X 재표준화는 없습니다.",
        f"- alpha 후보: {', '.join(f'{alpha:g}' for alpha in summary['alpha_candidates'])}.",
        "- current_blend_fusion_equal은 현재 기준 50% + fusion joint 50%의 사전 고정 진단 한 개입니다. 가중치 학습·탐색은 없으며 주 평가값을 대체하지 않습니다.", "",
        "## Outer fold별 inner 선택", "", "| fold | 방식 | alpha | inner MSE |", "|---|---|---:|---:|"]
    for record in summary["selections"]:
        winner = record["joint"]
        lines.append(f"| {record['outer_fold']} | {winner['representation']} | {winner['alpha']:g} | {winner['inner_mse']:.6f} |")
    lines += ["", "## 경도 꼬리와 잡음 구간", "",
        "| 모델 | 양끝 20% RMSE | 중간 80% RMSE | 잡음 Q1 RMSE | 잡음 Q4 RMSE |", "|---|---:|---:|---:|---:|"]
    for name, result in summary["diagnostics"].items():
        region, noise = result["target_regions"], result["noise_quartiles"]
        lines.append(f"| {name} | {region['tails20']['rmse']:.4f} | {region['middle80']['rmse']:.4f} | {noise[0]['rmse']:.4f} | {noise[-1]['rmse']:.4f} |")
    lines += ["", "## 검증 범위와 한계", "",
        "- 이전 DINO OOF/summary의 seed·train/fold/feature 캐시 해시와 기준선·ResNet·고정 평균 수식을 대조했습니다.",
        "- 수작업 캐시는 과거 생성 당시 이미지·소스 해시를 내장하지 않았습니다. 현재 파일/배열 해시, 125개 특징 이름·순서·스키마는 확인했지만 과거 생성 출처 전체가 검증됐다는 뜻은 아닙니다.",
        "- 신경망 forward, 이미지 재읽기, test 접근, 전체 500장 최종 적합, 제출 생성은 수행하지 않습니다.",
        "- 두 seed 모두 같은 500장을 재사용합니다. 이번 nested 절차도 반복 연구와 이전 모델 선택의 편향까지 제거하지 않습니다.",
        "- 잡음 지표는 실제 미세구조도 반영할 수 있습니다. 큰 오차나 꼬리 구간의 이미지를 제거·보정하지 않았습니다.",
        "- oof.csv, inner_search.csv, fold/target/noise 진단 CSV와 summary.json에 설정·참조·코드·입력 해시를 저장했습니다.", ""]
    return "\n".join(lines)


def write_outputs(output_dir, data, predictions, selections, diagnostics, tables, seed, threads, elapsed, alphas=ALPHAS):
    core.write_outputs(output_dir, data, predictions, selections, diagnostics, tables, seed, threads, elapsed,
        alphas=alphas, report_builder=build_markdown, summary_extras={
            "experiment": "handcrafted_dino_feature_fusion", "representations": list(REPRESENTATIONS),
            "primary_interpretation": "joint_nested is feature-fusion nested OOF; reference and fixed-blend rows are diagnostics",
            "block_dimensions": {"hand": HAND_DIM, "dino": DINO_DIM}, "balanced_block_weights": BLOCK_WEIGHTS,
            "block_weight_limitation": "Nominal column-count balance only; correlated/constant columns prevent equal-information or exact-energy guarantees.",
            "current_reference_weights": {"hand_ridge": 0.25, "frozen_resnet": 0.25, "dino_joint": 0.5},
            "fixed_diagnostic_weights": {CURRENT_REFERENCE: 0.5, core.PRIMARY_NAME: 0.5},
            "handcrafted_provenance_limitation": HAND_PROVENANCE_LIMITATION,
            "research_selection_note": "The experiment and reference were selected after prior results on these same images; nested CV does not undo research-level selection.",
        })


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--features", "--dino-features", dest="features", type=Path, default=ROOT / "reports/frozen_dinov2_small/features.npz")
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--handcrafted-features", type=Path, default=ROOT / "reports/benchmark_features.npz")
    parser.add_argument("--feature-definition", type=Path, default=ROOT / "reports/benchmark_seed42.json")
    parser.add_argument("--train-csv", type=Path, default=ROOT / "open/train.csv")
    parser.add_argument("--audit-features", type=Path, default=ROOT / "reports/audit_train_features.csv")
    parser.add_argument("--folds", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--previous-frozen-oof", type=Path)
    parser.add_argument("--previous-dino-oof", type=Path)
    parser.add_argument("--previous-dino-summary", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--alphas", nargs="+", type=float, default=ALPHAS)
    parser.add_argument("--validate-only", action="store_true", help="Validate existing artifacts only; no fitting and no output writes")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    defaults = {"folds": ROOT / ("reports/folds.csv" if args.seed == 42 else f"reports/benchmark_folds_seed{args.seed}.csv"),
                "baseline": ROOT / f"reports/benchmark_oof_seed{args.seed}.csv",
                "previous_frozen_oof": ROOT / f"reports/frozen_resnet18/seed{args.seed}/oof.csv",
                "previous_dino_oof": ROOT / f"reports/frozen_dinov2_small/seed{args.seed}/oof.csv",
                "output_dir": ROOT / f"reports/feature_fusion/seed{args.seed}"}
    for name, default in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, default)
    if args.previous_dino_summary is None:
        args.previous_dino_summary = args.previous_dino_oof.with_name("summary.json")
    return args


def main(argv=None):
    args = parse_args(argv)
    seed, alphas = core.validate_seed(args.seed), core.validate_alphas(args.alphas)
    if np is None or args.threads < 1:
        raise ValueError("NumPy and positive --threads are required")
    if not args.validate_only:
        if args.output_dir.exists() and not args.output_dir.is_dir():
            raise ValueError("--output-dir must be a directory")
        if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
            raise ValueError("Output directory contains files; use a new directory or explicit --overwrite")
        protected = {path.resolve() for path in (args.previous_dino_oof, args.previous_frozen_oof, args.previous_dino_summary)}
        if any((args.output_dir / name).resolve() in protected for name in ("oof.csv", "summary.json")):
            raise ValueError("Output directory would overwrite a reference OOF/summary; choose a new directory")
    started = time.perf_counter()
    data, references = load_inputs(args.features, args.train_csv, args.folds, args.baseline, args.audit_features,
        args.previous_frozen_oof, args.previous_dino_oof, args.handcrafted_features, args.feature_definition, seed,
        metadata_path=args.metadata, previous_dino_summary=args.previous_dino_summary)
    if args.validate_only:
        print(json.dumps({"status": "validated", "seed": seed, "n_train": len(data.y), "features": TOTAL_DIM,
            "representations": list(REPRESENTATIONS), "reference_current_blend_rmse": core.metrics(data.y, references[CURRENT_REFERENCE])["rmse"],
            "reference_hand_dino_equal_rmse": core.metrics(data.y, references[HAND_DINO_REFERENCE])["rmse"],
            "estimators_fitted": 0, "output_written": False, "images_or_test_read": False,
            "handcrafted_historical_provenance_verified": False}, indent=2))
        return 0
    predictions, selections = core.nested_predictions(data, seed, args.threads, alphas=alphas,
        representations=REPRESENTATIONS, estimator_factory=make_estimator)
    predictions = add_fixed_candidate(data, predictions, references)
    diagnostics, tables = diagnose(data, predictions)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_outputs(args.output_dir, data, predictions, selections, diagnostics, tables, seed, args.threads,
                  time.perf_counter() - started, alphas=alphas)
    for name in (core.PRIMARY_NAME, FIXED_BLEND):
        result = diagnostics[name]["overall"]
        print(f"{'PRIMARY' if name == core.PRIMARY_NAME else 'FIXED DIAGNOSTIC'} {name}: RMSE={result['rmse']:.6f}; delta vs current={result['delta_rmse_vs_current_blend']:+.6f}")
    print(f"Report: {(args.output_dir / 'report.md').resolve()}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError) as exc:
        print(f"Feature fusion evaluation stopped: {exc}", file=sys.stderr)
        raise SystemExit(2)
