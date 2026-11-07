"""User-run, train-only nested RBF Kernel Ridge on fixed DINO features.

The fixed 1152-dimensional CLS/patch mean/std representation is reused for all
nine alpha/gamma candidates. No images, torch, test, weights or all-data fitting.
Named profiles: original (default) or lower (separate default output directory).
--validate-only checks saved artifacts without fitting or creating output files.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

if __package__:
    from . import evaluate_frozen_resnet as core
    from . import evaluate_frozen_dinov2 as dino
    from . import evaluate_feature_fusion as references_module
else:
    import evaluate_frozen_resnet as core
    import evaluate_frozen_dinov2 as dino
    import evaluate_feature_fusion as references_module

np = core.np
ROOT = core.ROOT
FEATURE_NAME = "cls_patch_mean_std"
FEATURE_DIM = 1152
ALPHAS = (0.1, 1.0, 10.0)
GAMMA_MULTIPLIERS = {"rbf_gamma0p1": 0.1, "rbf_gamma1": 1.0, "rbf_gamma10": 10.0}
BANDWIDTH_FAMILIES = tuple(GAMMA_MULTIPLIERS)
LOWER_ALPHAS = (0.01, 0.03, 0.1)
LOWER_GAMMA_MULTIPLIERS = {"rbf_gamma0p01": 0.01, "rbf_gamma0p03": 0.03, "rbf_gamma0p1": 0.1}
SEARCH_PROFILES = {"original": (ALPHAS, BANDWIDTH_FAMILIES),
                   "lower": (LOWER_ALPHAS, tuple(LOWER_GAMMA_MULTIPLIERS))}
ALL_GAMMA_MULTIPLIERS = {**GAMMA_MULTIPLIERS, **LOWER_GAMMA_MULTIPLIERS}
CURRENT_REFERENCE = references_module.CURRENT_REFERENCE
DINO_REFERENCE = references_module.DINO_REFERENCE
HAND_DINO_REFERENCE = references_module.HAND_DINO_REFERENCE
FIXED_BLEND = "current_blend_kernel_equal"


def search_parameters(search_profile="original"):
    if search_profile not in SEARCH_PROFILES:
        raise ValueError(f"Unknown search profile: {search_profile}; choose original or lower")
    return SEARCH_PROFILES[search_profile]


def gamma_parameters(family):
    if family not in ALL_GAMMA_MULTIPLIERS:
        raise ValueError(f"Unknown RBF bandwidth family: {family}")
    multiplier = ALL_GAMMA_MULTIPLIERS[family]
    return {"gamma_multiplier": multiplier, "gamma": multiplier / FEATURE_DIM}


def load_inputs(features_path, train_path, folds_path, baseline_path, audit_path,
                previous_frozen_path, previous_dino_path, seed, *, metadata_path=None,
                previous_dino_summary=None, expected_n=500, search_profile="original"):
    _, families = search_parameters(search_profile)
    data, frozen_references = dino.load_inputs(features_path, train_path, folds_path, baseline_path,
        audit_path, previous_frozen_path, seed, metadata_path=metadata_path, expected_n=expected_n)
    references = references_module.load_previous_dino(data, frozen_references, previous_dino_path,
        previous_dino_summary or previous_dino_path.with_name("summary.json"), seed)
    features = data.features[FEATURE_NAME]
    features.flags.writeable = False
    # The aliases vary only kernel bandwidth. There is no feature selection here.
    data.features = {family: features for family in families}
    data.provenance["kernel_inputs"] = {
        "feature_name": FEATURE_NAME, "feature_dimension": FEATURE_DIM,
        "aligned_feature_array_sha256": dino.common.sha256_array(features),
        "identical_feature_array_for_all_bandwidth_families": True,
        "reference_predictions_used_as_features": False,
        "handcrafted_feature_cache_read": False, "audit_features_used_as_model_inputs": False,
    }
    data.provenance["evaluation_sources"] = {name: {"path": str(Path(path).resolve()), "sha256": core.sha256(Path(path))}
        for name, path in {"kernel_evaluator": __file__, "nested_core": core.__file__,
                           "dino_validator": dino.__file__, "reference_validator_and_diagnostics": references_module.__file__}.items()}
    return data, references


def make_estimator(alpha, family):
    from sklearn.compose import TransformedTargetRegressor
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    alpha = core.validate_alphas([alpha])[0]
    return TransformedTargetRegressor(
        regressor=Pipeline([("standardscaler", StandardScaler()),
            ("kernelridge", KernelRidge(alpha=alpha, kernel="rbf", gamma=gamma_parameters(family)["gamma"]))]),
        transformer=StandardScaler(),
    )


def nested_predictions(data, seed, threads=2, *, search_profile="original"):
    alphas, families = search_parameters(search_profile)
    if set(data.features) != set(families):
        raise ValueError("Expected exactly the three fixed RBF bandwidth families")
    first = data.features[families[0]]
    if first.shape != (len(data.y), FEATURE_DIM) or not np.isfinite(first).all():
        raise ValueError("Kernel input must be finite with exactly 1152 columns")
    if any(not np.array_equal(first, data.features[family]) for family in families[1:]):
        raise ValueError("All bandwidth families must use identical DINO features")
    predictions, selections = core.nested_predictions(data, seed, threads, alphas=alphas,
        representations=families, estimator_factory=make_estimator)
    for record in selections:
        # All three locations get the same metadata so shared writer equality
        # checks for selected candidates remain exact.
        for candidate in [*record["candidates"], *record["per_representation"].values(), record["joint"]]:
            candidate.update(gamma_parameters(candidate["representation"]))
    return predictions, selections


def add_fixed_candidate(predictions, references, *, search_profile="original"):
    _, families = search_parameters(search_profile)
    expected = {family + "_nested" for family in families} | {core.PRIMARY_NAME}
    if set(predictions) != expected or set(references) != {CURRENT_REFERENCE, DINO_REFERENCE, HAND_DINO_REFERENCE}:
        raise ValueError("Unexpected kernel predictions or reference columns")
    return {**predictions, **references,
            FIXED_BLEND: 0.5 * references[CURRENT_REFERENCE] + 0.5 * predictions[core.PRIMARY_NAME]}


def diagnose(data, predictions):
    diagnostics, tables = references_module.diagnose(data, predictions)
    linear_rmse = diagnostics[DINO_REFERENCE]["overall"]["rmse"]
    for row in tables["metrics"]:
        delta = row["rmse"] - linear_rmse
        row["delta_rmse_vs_dino_linear"] = delta
        diagnostics[row["model"]]["overall"]["delta_rmse_vs_dino_linear"] = delta
    return diagnostics, tables


def build_markdown(summary):
    alpha_text = ", ".join(f"{alpha:g}" for alpha in summary["alpha_candidates"])
    gamma_text = ", ".join(f"{parameters['gamma_multiplier']:g}/1152" for parameters in summary["bandwidth_families"].values())
    family_text = ", ".join(summary["bandwidth_families"])
    lines = ["# 고정 DINO 특징의 RBF Kernel Ridge", "",
        f"탐색 프로필: **{summary['search_profile']}**. alpha 후보: {alpha_text}. gamma 후보: {gamma_text}.", "",
        "주 평가값은 **joint_nested 전체 OOF RMSE**입니다. 같은 1152개 DINO 특징에 대해 outer 학습 부분의 inner 3-fold에서 alpha와 gamma 9개 조합을 공동 선택했습니다.", "",
        f"{family_text}은 서로 다른 특징 표현이 아니라 같은 입력의 대역폭 설정입니다. 대역폭별 outer OOF는 진단용이며, 그중 최저 점수를 사후 선택해 주 평가값을 바꾸지 않습니다.", "",
        "| 모델 | OOF RMSE | 현재 25:25:50 대비 Δ | DINO 선형 대비 Δ | 개선 fold |", "|---|---:|---:|---:|---:|"]
    for name, result in summary["diagnostics"].items():
        m = result["overall"]
        lines.append(f"| {name} | {m['rmse']:.6f} | {m['delta_rmse_vs_current_blend']:+.6f} | {m['delta_rmse_vs_dino_linear']:+.6f} | {m['folds_improved_vs_current_blend']}/5 |")
    lines += ["", "현재 기준은 수작업 Ridge 25% + frozen ResNet18 25% + DINO 선형 Ridge 50%입니다. Δ가 음수면 개선입니다. fold RMSE 평균이 아닌 전체 OOF SSE로 계산합니다.", "",
        "## 고정한 학습 설계", "",
        "- 입력: 기존 cls_patch_mean_std 1152개. 이미지 재추출, 수작업 특징 결합, PCA, 추가 특징 선택은 없습니다.",
        "- X StandardScaler → RBF KernelRidge, y StandardScaler는 TransformedTargetRegressor로 적용하며 모든 scaler는 해당 학습 fold에서만 적합합니다.",
        f"- alpha: {alpha_text}. gamma: {gamma_text}. K(x,z)=exp(-gamma × ||x-z||²).",
        "- Kernel Ridge의 alpha는 커널 행렬의 스케일에 적용되므로 이전 선형 Ridge alpha=1000과 숫자만 비교할 수 없습니다. gamma 분모 1152는 고정이며 거리 통계로 다시 추정하지 않습니다.",
        "- kernel centering과 별도 intercept는 없습니다. y를 각 학습 fold 평균으로 중심화하므로, 학습점과의 커널값이 모두 0에 가까운 입력은 역변환 후 해당 학습 fold의 평균 경도로 돌아갑니다. 전체 500장 평균을 사용하지 않습니다.",
        "- inner 선택 기준: 원래 경도 단위로 복원한 pooled inner OOF MSE. 기존 outer 5-fold, inner seed=seed+outer_fold를 사용합니다.",
        "- inner MSE가 정확히 같으면 작은 gamma multiplier, 그다음 큰 alpha를 선택합니다. 프로필별 후보는 사전에 고정한 9개입니다.",
        "- current_blend_kernel_equal은 기존 기준 50% + kernel joint 50%의 사전 고정 진단 한 개입니다. 가중치 탐색은 없습니다.", "",
        "## Outer fold별 inner 선택", "", "| fold | gamma multiplier | 실제 gamma | alpha | inner MSE |", "|---|---:|---:|---:|---:|"]
    for record in summary["selections"]:
        w = record["joint"]
        lines.append(f"| {record['outer_fold']} | {w['gamma_multiplier']:g} | {w['gamma']:.10g} | {w['alpha']:g} | {w['inner_mse']:.6f} |")
    lines += ["", "## 경도와 잡음 구간", "", "| 모델 | 꼬리 20% | 중간 80% | 잡음 Q1 | 잡음 Q4 |", "|---|---:|---:|---:|---:|"]
    for name, result in summary["diagnostics"].items():
        regions, noise = result["target_regions"], result["noise_quartiles"]
        lines.append(f"| {name} | {regions['tails20']['rmse']:.4f} | {regions['middle80']['rmse']:.4f} | {noise[0]['rmse']:.4f} | {noise[-1]['rmse']:.4f} |")
    lines += ["", "## 검증 범위와 한계", "",
        "- DINO 캐시 metadata·배열 해시, 이전 OOF/summary의 seed·입력 해시·ID·타깃·fold·고정 조합 수식을 대조했습니다. 이전 예측과 audit 특징은 kernel 입력에 넣지 않았습니다.",
        "- 같은 500장에 대한 반복 연구의 선택 편향은 nested CV로 모두 제거되지 않습니다. 두 seed는 독립 holdout이 아닙니다.",
        "- 잡음 대용지표는 미세구조도 반영할 수 있습니다. 구간별 오차로 레이블 오류나 원인을 확정하지 않습니다.",
        "- 신경망 forward, 이미지/test 읽기, 가중치 다운로드, 전체 500장 최종 적합, 제출 생성은 수행하지 않습니다.",
        "- summary.json과 inner_search.csv에 실제 gamma·alpha·선택 기록을, oof.csv와 진단 CSV에 기준 대비 결과를 저장합니다.", ""]
    return "\n".join(lines)


def write_outputs(output_dir, data, predictions, selections, diagnostics, tables, seed, threads, elapsed, *, search_profile="original"):
    alphas, families = search_parameters(search_profile)
    expected_candidates = {(family, alpha) for family in families for alpha in alphas}
    if set(data.features) != set(families) or any(
        len(record["candidates"]) != 9 or {(c["representation"], c["alpha"]) for c in record["candidates"]} != expected_candidates
        for record in selections
    ):
        raise ValueError("Output profile does not match evaluated bandwidth/alpha candidates")
    # Use the shared writer's table extension point to retain every existing
    # inner_search field and add explicit bandwidth values, without changing core.
    search = []
    for record in selections:
        for candidate in record["candidates"]:
            family = candidate["representation"]
            search.append({"outer_fold": record["outer_fold"], "inner_seed": record["inner_seed"],
                "representation": family, "bandwidth_family": family,
                "alpha": candidate["alpha"], **gamma_parameters(family), "inner_mse": candidate["inner_mse"],
                **{f"inner_fold{j}_mse": value for j, value in enumerate(candidate["inner_fold_mse"])},
                "selected_for_representation": candidate == record["per_representation"][family],
                "selected_joint": candidate == record["joint"]})
    core.write_outputs(output_dir, data, predictions, selections, diagnostics, {**tables, "inner_search": search},
        seed, threads, elapsed, alphas=alphas, report_builder=build_markdown, summary_extras={
            "experiment": "fixed_dinov2_rbf_kernel_ridge",
            "search_profile": search_profile,
            "source_feature_name": FEATURE_NAME, "source_feature_dimension": FEATURE_DIM,
            "representation_key_semantics": "Bandwidth families of one identical DINO feature matrix; no feature-representation selection.",
            "bandwidth_families": {family: gamma_parameters(family) for family in families},
            "gamma_formula": "gamma_multiplier / 1152", "kernel_formula": "exp(-gamma * squared_euclidean_distance(StandardScaler(X)))",
            "kernel_centering": False, "separate_intercept": False,
            "target_transform": "StandardScaler fitted on each training subset; inverse transform restores original hardness units.",
            "far_kernel_prediction": "As all train/query kernel values approach zero, prediction approaches that training subset's target mean.",
            "candidate_count_per_inner_search": 9,
            "primary_interpretation": "joint_nested selects alpha and gamma jointly using pooled inner OOF MSE; bandwidth-only OOF and blends are diagnostic.",
            "current_reference_weights": {"hand_ridge": 0.25, "frozen_resnet": 0.25, "dino_linear_joint": 0.5},
            "fixed_diagnostic_weights": {CURRENT_REFERENCE: 0.5, core.PRIMARY_NAME: 0.5},
            "research_selection_note": "This nonlinear candidate follows prior results on the same dataset; nested CV cannot undo research-level selection.",
        })


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--search-profile", choices=tuple(SEARCH_PROFILES), default="original",
                        help="Fixed nine-candidate grid; lower uses alpha and gamma multipliers 0.01, 0.03, 0.1")
    parser.add_argument("--features", type=Path, default=ROOT / "reports/frozen_dinov2_small/features.npz")
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--train-csv", type=Path, default=ROOT / "open/train.csv")
    parser.add_argument("--audit-features", type=Path, default=ROOT / "reports/audit_train_features.csv")
    for flag in ("folds", "baseline", "previous-frozen-oof", "previous-dino-oof", "previous-dino-summary", "output-dir"):
        parser.add_argument(f"--{flag}", type=Path)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--validate-only", action="store_true", help="Check existing artifacts only; no fitting and no output files")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    output_stem = "dinov2_kernel" if args.search_profile == "original" else "dinov2_kernel_lower"
    defaults = {"folds": ROOT / ("reports/folds.csv" if args.seed == 42 else f"reports/benchmark_folds_seed{args.seed}.csv"),
        "baseline": ROOT / f"reports/benchmark_oof_seed{args.seed}.csv",
        "previous_frozen_oof": ROOT / f"reports/frozen_resnet18/seed{args.seed}/oof.csv",
        "previous_dino_oof": ROOT / f"reports/frozen_dinov2_small/seed{args.seed}/oof.csv",
        "output_dir": ROOT / f"reports/{output_stem}/seed{args.seed}"}
    for name, default in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, default)
    if args.previous_dino_summary is None:
        args.previous_dino_summary = args.previous_dino_oof.with_name("summary.json")
    return args


def main(argv=None):
    args = parse_args(argv)
    seed = core.validate_seed(args.seed)
    alphas, families = search_parameters(args.search_profile)
    if np is None or args.threads < 1:
        raise ValueError("NumPy and positive --threads are required")
    if not args.validate_only:
        if args.output_dir.exists() and not args.output_dir.is_dir():
            raise ValueError("--output-dir must be a directory")
        if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
            raise ValueError("Output directory contains files; use a new directory or explicit --overwrite")
        protected = {path.resolve() for path in (args.features, args.train_csv, args.folds, args.baseline,
            args.audit_features, args.previous_frozen_oof, args.previous_dino_oof, args.previous_dino_summary,
            args.metadata or args.features.with_name("metadata.json"))}
        # Every evaluator output is directly inside this directory. Refuse an
        # input's parent even with --overwrite, including custom input filenames.
        if any(path.parent == args.output_dir.resolve() for path in protected):
            raise ValueError("Output directory contains a protected input/reference; choose a new directory")
    started = time.perf_counter()
    data, references = load_inputs(args.features, args.train_csv, args.folds, args.baseline, args.audit_features,
        args.previous_frozen_oof, args.previous_dino_oof, seed, metadata_path=args.metadata,
        previous_dino_summary=args.previous_dino_summary, search_profile=args.search_profile)
    if args.validate_only:
        print(json.dumps({"status": "validated", "seed": seed, "n_train": len(data.y), "source_feature": FEATURE_NAME,
            "feature_dimension": FEATURE_DIM, "search_profile": args.search_profile, "alpha_candidates": alphas,
            "bandwidth_families": {family: gamma_parameters(family) for family in families},
            "reference_current_blend_rmse": core.metrics(data.y, references[CURRENT_REFERENCE])["rmse"],
            "estimators_fitted": 0, "output_written": False, "images_or_test_read": False,
            "handcrafted_feature_cache_read": False}, indent=2))
        return 0
    print(f"Search profile={args.search_profile}; alpha={list(alphas)}; gamma multipliers={[gamma_parameters(family)['gamma_multiplier'] for family in families]}", flush=True)
    predictions, selections = nested_predictions(data, seed, args.threads, search_profile=args.search_profile)
    predictions = add_fixed_candidate(predictions, references, search_profile=args.search_profile)
    diagnostics, tables = diagnose(data, predictions)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_outputs(args.output_dir, data, predictions, selections, diagnostics, tables, seed, args.threads,
                  time.perf_counter() - started, search_profile=args.search_profile)
    for name in (core.PRIMARY_NAME, FIXED_BLEND):
        value = diagnostics[name]["overall"]
        print(f"{'PRIMARY' if name == core.PRIMARY_NAME else 'FIXED DIAGNOSTIC'} {name}: RMSE={value['rmse']:.6f}; delta vs current={value['delta_rmse_vs_current_blend']:+.6f}")
    print(f"Report: {(args.output_dir / 'report.md').resolve()}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError) as exc:
        print(f"DINO kernel evaluation stopped: {exc}", file=sys.stderr)
        raise SystemExit(2)
