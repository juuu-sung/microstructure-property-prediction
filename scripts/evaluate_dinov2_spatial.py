"""User-run nested Ridge: original DINO global features vs global + 2x2 regions.

--preflight checks prior artifacts before regional extraction; --validate-only
also requires the new spatial cache. Both fit zero estimators and write nothing.
No torch, images, weights, test data or all-data final fit in this evaluator.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

if __package__:
    from . import evaluate_dinov2_learning_curve as prior
    from . import extract_dinov2_spatial as spatial
else:
    import evaluate_dinov2_learning_curve as prior
    import extract_dinov2_spatial as spatial

core, refs_module = prior.core, prior.references_module
np, ROOT = core.np, core.ROOT
GLOBAL = "cls_patch_mean_std"
AUGMENTED = "global_plus_2x2"
REPRESENTATIONS = (GLOBAL, AUGMENTED)
ALPHAS = prior.ALPHAS
CURRENT = refs_module.CURRENT_REFERENCE
DINO = refs_module.DINO_REFERENCE
FIXED_BLEND = "dino_replaced_blend"


def load_prior(args, *, expected_n=500):
    return prior.load_inputs(args.global_features, args.train_csv, args.folds, args.baseline,
        args.audit_features, args.previous_frozen_oof, args.previous_dino_oof, args.seed,
        previous_dino_summary=args.previous_dino_summary, expected_n=expected_n)


def attach_spatial(data, path, global_path):
    local, metadata = spatial.validate_spatial_cache(np, path, data.ids, data.metadata, global_path)
    original = data.features[GLOBAL]
    data.features = {GLOBAL: original, AUGMENTED: np.concatenate([original, local], axis=1)}
    assert np.array_equal(data.features[AUGMENTED][:, :1152], original)
    data.provenance["spatial_features"] = {"path": str(path.resolve()), "sha256": core.sha256(path),
        "metadata_path": str(path.with_name("metadata.json").resolve()),
        "metadata_sha256": core.sha256(path.with_name("metadata.json")),
        "global_block_exactly_reused": True, "references_used_as_inputs": False}
    data.provenance["spatial_evaluation_sources"] = {name: {"path": str(Path(module).resolve()), "sha256": core.sha256(Path(module))}
        for name, module in {"evaluator": __file__, "spatial_cache_validator": spatial.__file__}.items()}
    data.metadata = {"global": data.metadata, "spatial": metadata}
    return data


def verify_global_anchor(data, predictions, selections, anchor):
    records = {}
    for record in selections:
        fold = record["outer_fold"]
        valid = np.flatnonzero(data.folds == fold)
        records[fold] = {"winner": record["per_representation"][GLOBAL],
            "candidates": [c for c in record["candidates"] if c["representation"] == GLOBAL],
            "inner_validation_sizes": record["inner_validation_sizes"], "validation_indices": valid,
            "validation_predictions": predictions[GLOBAL + "_nested"][valid]}
    return prior.validate_anchor(records, data, anchor)


def add_fixed_candidate(predictions, references):
    if set(predictions) != {core.PRIMARY_NAME, *(name + "_nested" for name in REPRESENTATIONS)}:
        raise ValueError("Unexpected spatial predictions")
    # Keep hand25/resnet25/dino50. Only substitute the DINO component.
    replaced = references[CURRENT] + 0.5 * (predictions[core.PRIMARY_NAME] - references[DINO])
    return {**predictions, **references, FIXED_BLEND: replaced}


def diagnose(data, predictions):
    diagnostics, tables = refs_module.diagnose(data, predictions)
    reference_rmse = diagnostics[DINO]["overall"]["rmse"]
    for row in tables["metrics"]:
        delta = row["rmse"] - reference_rmse
        row["delta_rmse_vs_dino_linear"] = delta
        diagnostics[row["model"]]["overall"]["delta_rmse_vs_dino_linear"] = delta
    return diagnostics, tables


def build_markdown(summary):
    lines = ["# DINO 2×2 지역 특징 + nested Ridge", "",
        "기존 전역1152개와 전역+지역4224개 중 표현·alpha를 inner CV로 공동 선택한 joint_nested가 주 지표입니다. 개별 표현 OOF를 사후 선택하지 않습니다.", "",
        "| 모델 | OOF RMSE | 현재 조합 대비 Δ | 기존 DINO 대비 Δ | 현재 조합보다 개선 fold |",
        "|---|---:|---:|---:|---:|"]
    for name, value in summary["diagnostics"].items():
        m = value["overall"]
        lines.append(f"| {name} | {m['rmse']:.6f} | {m['delta_rmse_vs_current_blend']:+.6f} | {m['delta_rmse_vs_dino_linear']:+.6f} | {m['folds_improved_vs_current_blend']}/5 |")
    lines += ["", "## Fold 내부 선택", "", "| fold | 표현 | alpha | inner MSE |", "|---|---|---:|---:|"]
    for record in summary["selections"]:
        w = record["joint"]
        lines.append(f"| {record['outer_fold']} | {w['representation']} | {w['alpha']:g} | {w['inner_mse']:.6f} |")
    lines += ["", "## 해석", "",
        "- cls_patch_mean_std_nested는 기존 전역 특징 재현 검사, global_plus_2x2_nested는 지역 추가 후보의 별도 진단입니다.",
        "- dino_replaced_blend는 수작업25% + ResNet25% + joint_nested50%입니다. 기존 조합과 새 모델을 다시50:50 평균한 것이 아닙니다. 가중치를 탐색하지 않습니다.",
        "- 19×19 patch를 행/열[0,9,19] 경계로 분할해81/90/90/100개를 중복 없이 모두 사용합니다. 순서는 좌상·우상·좌하·우하이며 각 지역마다384개 평균 뒤384개 population 표준편차를 붙입니다.",
        "- 원본 전역 캐시1152개를 그대로 사용하고 지역3072개만 추가합니다. 전처리·가중치·이미지 manifest 일치와 재추출 전역값의 허용 오차를 확인했습니다.",
        "- X/y StandardScaler는 각 inner 학습 fold에서만 적합합니다. 전역/지역 블록별 추가 가중, PCA, calibration, clipping을 사용하지 않습니다.",
        "- 2표현×6alpha×3inner×5outer + 2표현×5재적합 =190회 적합/seed입니다. 표현 동점은 전역, alpha 동점은 큰 alpha를 우선합니다.",
        f"- 기존 전역 OOF/inner 점수 재현: {summary['global_anchor']['status']}; 최대 예측 차이 {summary['global_anchor']['max_absolute_prediction_difference']:.3g}.",
        "- 지역별 요약도 전역 attention 이후 token입니다. 위치 불변성이 보장되지 않으며 차원 증가와 위치 의존성이 과적합을 늘릴 수 있습니다.",
        "- 두 seed는 동일500장의 재분할입니다. 반복 연구의 선택 편향이 남습니다. 이 실험만으로 공간 정보 부족 또는 라벨 잡음을 확정하지 않습니다.",
        "- 현재 조합은 결과 검토까지 유지합니다. 양끝 경도·잡음 구간·잔차 상관·큰 오류 변화는 함께 저장한 진단 CSV에서 확인합니다.",
        "- evaluator는 이미지·test·torch·가중치 다운로드·전체500장 최종 적합·제출 생성을 수행하지 않습니다.", ""]
    return "\n".join(lines)


def write_outputs(output_dir, data, predictions, selections, diagnostics, tables, seed, threads, elapsed, anchor_check):
    core.write_outputs(output_dir, data, predictions, selections, diagnostics, tables, seed, threads, elapsed,
        alphas=ALPHAS, report_builder=build_markdown, summary_extras={"experiment": "dinov2_spatial_2x2",
            "global_anchor": anchor_check, "spatial_protocol": spatial.PROTOCOL,
            "representations": list(REPRESENTATIONS), "estimator_fit_count": 190,
            "candidate_count_per_inner_search": 12,
            "fixed_diagnostic": {"name": FIXED_BLEND, "weights": {"hand_ridge": 0.25, "frozen_resnet": 0.25, "joint_nested": 0.5},
                "formula": "reference_current_blend + 0.5 * (joint_nested - reference_dino_joint)"},
            "selection_note": "Joint inner choice includes original global features as fallback; individual OOF and fixed blend remain diagnostics."})


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--global-features", type=Path, default=ROOT / "reports/frozen_dinov2_small/features.npz")
    parser.add_argument("--spatial-features", type=Path, default=ROOT / "reports/dinov2_spatial/features/features.npz")
    parser.add_argument("--train-csv", type=Path, default=ROOT / "open/train.csv")
    parser.add_argument("--audit-features", type=Path, default=ROOT / "reports/audit_train_features.csv")
    for flag in ("folds", "baseline", "previous-frozen-oof", "previous-dino-oof", "previous-dino-summary", "output-dir"):
        parser.add_argument("--" + flag, type=Path)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--preflight", action="store_true")
    modes.add_argument("--validate-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    defaults = {"folds": ROOT / ("reports/folds.csv" if args.seed == 42 else f"reports/benchmark_folds_seed{args.seed}.csv"),
        "baseline": ROOT / f"reports/benchmark_oof_seed{args.seed}.csv",
        "previous_frozen_oof": ROOT / f"reports/frozen_resnet18/seed{args.seed}/oof.csv",
        "previous_dino_oof": ROOT / f"reports/frozen_dinov2_small/seed{args.seed}/oof.csv",
        "output_dir": ROOT / f"reports/dinov2_spatial/seed{args.seed}"}
    for name, value in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    if args.previous_dino_summary is None:
        args.previous_dino_summary = args.previous_dino_oof.with_name("summary.json")
    return args


def main(argv=None):
    args = parse_args(argv)
    seed = core.validate_seed(args.seed)
    if np is None or args.threads < 1:
        raise ValueError("NumPy and positive --threads required")
    if not (args.preflight or args.validate_only):
        protected = [args.global_features, args.spatial_features, args.train_csv, args.audit_features,
            args.folds, args.baseline, args.previous_frozen_oof, args.previous_dino_oof, args.previous_dino_summary]
        if any(path.resolve().parent == args.output_dir.resolve() for path in protected):
            raise ValueError("Output directory contains a protected input/reference")
        if args.output_dir.exists() and (not args.output_dir.is_dir() or (any(args.output_dir.iterdir()) and not args.overwrite)):
            raise ValueError("Output directory contains results; use a new directory or explicit --overwrite")
    started = time.perf_counter()
    data, references, anchor = load_prior(args)
    if args.preflight:
        print(json.dumps({"status": "prior_artifacts_valid", "seed": seed, "rows": len(data.y),
            "spatial_cache_checked": False, "spatial_extraction_required": True,
            "estimators_fitted": 0, "output_written": False}, indent=2))
        return 0
    data = attach_spatial(data, args.spatial_features, args.global_features)
    if args.validate_only:
        print(json.dumps({"status": "all_artifacts_valid", "seed": seed, "rows": len(data.y),
            "feature_dimensions": {name: x.shape[1] for name, x in data.features.items()},
            "estimators_fitted": 0, "output_written": False}, indent=2))
        return 0
    predictions, selections = core.nested_predictions(data, seed, args.threads, alphas=ALPHAS, representations=REPRESENTATIONS)
    anchor_check = verify_global_anchor(data, predictions, selections, anchor)
    predictions = add_fixed_candidate(predictions, references)
    diagnostics, tables = diagnose(data, predictions)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_outputs(args.output_dir, data, predictions, selections, diagnostics, tables, seed, args.threads,
        time.perf_counter()-started, anchor_check)
    print(f"Global anchor: {anchor_check['status']}")
    for name in (core.PRIMARY_NAME, AUGMENTED + "_nested", FIXED_BLEND):
        m = diagnostics[name]["overall"]
        print(f"{name}: RMSE={m['rmse']:.6f}; delta vs current={m['delta_rmse_vs_current_blend']:+.6f}; delta vs DINO={m['delta_rmse_vs_dino_linear']:+.6f}")
    print(f"Report: {(args.output_dir / 'report.md').resolve()}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError) as exc:
        print(f"Spatial evaluation stopped: {exc}", file=sys.stderr)
        raise SystemExit(2)
