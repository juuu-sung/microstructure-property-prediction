"""User-run nested Ridge: original DINO vs eight-view D4 averaged features.

Both representations have 1152 columns. Inner CV alone selects representation
and alpha. --preflight and --validate-only never fit or write. No torch, image
inference, downloads, test data, submission, or final all-data fitting.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

if __package__:
    from . import evaluate_dinov2_spatial as prior
    from . import extract_dinov2_d4 as extract
else:
    import evaluate_dinov2_spatial as prior
    import extract_dinov2_d4 as extract

core, np, ROOT = prior.core, prior.np, prior.ROOT
GLOBAL, D4 = prior.GLOBAL, extract.FEATURE
REPRESENTATIONS = (GLOBAL, D4)
ALPHAS = prior.ALPHAS
CURRENT, DINO = prior.CURRENT, prior.DINO
FIXED_BLEND = "dino_replaced_blend"
PARTIAL_REFERENCES = {80: "reference_ridge_partial80_blend", 160: "reference_ridge_partial160_blend"}


def attach_partial_references(data, references, paths):
    """Previously reviewed candidates are comparison-only, never model inputs."""
    result = dict(references)
    required = {"ID", "hardness", "fold", CURRENT, DINO, "ridge_head_only_nested",
                "ridge_partial_nested", "ridge_partial_replaced_blend"}
    records = {}
    for cap, path in paths.items():
        rows = core.align_rows(core.read_rows(path, required), data.ids, path)
        if (not np.array_equal(core.finite_column(rows, "hardness", path), data.y) or
                not np.array_equal(core.integer_column(rows, "fold", path), data.folds)):
            raise ValueError("Previous partial OOF labels or folds differ")
        for column, expected in ((CURRENT, references[CURRENT]), (DINO, references[DINO]),
                                  ("ridge_head_only_nested", references[DINO])):
            if not np.allclose(core.finite_column(rows, column, path), expected, rtol=0, atol=1e-7):
                raise ValueError(f"Previous partial reference mismatch: {column}")
        partial = core.finite_column(rows, "ridge_partial_nested", path)
        blend = core.finite_column(rows, "ridge_partial_replaced_blend", path)
        expected = references[CURRENT] + 0.5 * (partial - references[DINO])
        if not np.allclose(blend, expected, rtol=0, atol=1e-7):
            raise ValueError("Previous partial blend formula mismatch")
        result[PARTIAL_REFERENCES[cap]] = blend
        records[str(cap)] = {"path": str(path.resolve()), "sha256": core.sha256(path),
            "comparison_only": True, "ids_labels_folds_and_blend_verified": True}
    data.provenance["previous_partial_oof"] = records
    return result


def load_prior(args, *, expected_n=500):
    data, references, anchor = prior.load_prior(args, expected_n=expected_n)
    references = attach_partial_references(data, references, {80: args.partial80_oof, 160: args.partial160_oof})
    return data, references, anchor


def attach_features(data, path, original_path):
    features, metadata = extract.validate_cache(np, path, data.ids, data.metadata, original_path)
    data.features = {GLOBAL: data.features[GLOBAL], D4: features}
    data.provenance["d4_features"] = {"path": str(path.resolve()), "sha256": core.sha256(path),
        "metadata_path": str(path.with_name("metadata.json").resolve()),
        "metadata_sha256": core.sha256(path.with_name("metadata.json")),
        "original_features_exactly_reused": True, "references_used_as_inputs": False}
    data.provenance["d4_evaluation_sources"] = {name: {"path": str(Path(module).resolve()), "sha256": core.sha256(Path(module))}
        for name, module in {"evaluator": __file__, "d4_extractor": extract.__file__, "anchor_utilities": prior.__file__}.items()}
    data.metadata = {"global": data.metadata, "d4": metadata}
    return data


def add_fixed_candidate(predictions, references):
    if set(predictions) != {core.PRIMARY_NAME, *(name + "_nested" for name in REPRESENTATIONS)}:
        raise ValueError("Unexpected D4 predictions")
    return {**predictions, **references,
        FIXED_BLEND: references[CURRENT] + 0.5 * (predictions[core.PRIMARY_NAME] - references[DINO])}


def diagnose(data, predictions):
    diagnostics, tables = prior.diagnose(data, predictions)
    for cap, name in PARTIAL_REFERENCES.items():
        previous = diagnostics[name]["overall"]["rmse"]
        for row in tables["metrics"]:
            key = f"delta_rmse_vs_partial{cap}_blend"
            row[key] = row["rmse"] - previous
            diagnostics[row["model"]]["overall"][key] = row[key]
    return diagnostics, tables


def build_markdown(summary):
    lines = ["# DINO D4 방향 평균 + nested Ridge", "",
        "주 지표는 원본/D4 표현과 alpha를 inner CV로 공동 선택한 joint_nested입니다. 고정25:25:50 교체 조합과 개별 표현은 사전 지정 진단입니다.", "",
        "| 모델 | OOF RMSE | 기존 조합 Δ | partial80 조합 Δ | partial160 조합 Δ |",
        "|---|---:|---:|---:|---:|"]
    for name, value in summary["diagnostics"].items():
        m = value["overall"]
        lines.append(f"| {name} | {m['rmse']:.6f} | {m['delta_rmse_vs_current_blend']:+.6f} | {m['delta_rmse_vs_partial80_blend']:+.6f} | {m['delta_rmse_vs_partial160_blend']:+.6f} |")
    lines += ["", "## Fold 내부 선택", "", "| fold | 표현 | alpha | inner MSE |", "|---|---|---:|---:|"]
    for record in summary["selections"]:
        w = record["joint"]
        lines.append(f"| {record['outer_fold']} | {w['representation']} | {w['alpha']:g} | {w['inner_mse']:.6f} |")
    lines += ["", "## 남은 약점의 변화", "",
        "| 모델 | 경도 양끝20% | 중간80% | 잡음 대용지표 Q4 | 기존 최대 오차20장과 공통 수 |",
        "|---|---:|---:|---:|---:|"]
    for name, value in summary["diagnostics"].items():
        r, n = value["target_regions"], value["noise_quartiles"]
        overlap = value["overall"].get("top20_overlap_with_current_blend", 20)
        lines.append(f"| {name} | {r['tails20']['rmse']:.4f} | {r['middle80']['rmse']:.4f} | {n[-1]['rmse']:.4f} | {overlap} |")
    lines += ["", "## 고정한 절차와 해석", "",
        "- 원본과 D4 모두1152차원입니다. 각 이미지의0/90/180/270도, 좌우반사 후 같은4회전에서 각각 CLS·patch 평균·population 표준편차를 구한 뒤8개 벡터를 평균합니다. 서로 다른 view의 token을 먼저 합쳐 표준편차를 구하지 않습니다.",
        "- 학습 표본은500장 그대로이며8개 view는 독립 표본이 아닙니다. 변환은256×256에서 픽셀 위치만 치환하고 각 view에 기존5pixel reflect padding을 적용합니다. 입력 크기 변경·크롭·블러·색 증강이 없습니다.",
        "- D4의8개 변환에 대해서만 특징 불변성을 기대할 수 있습니다(수치 오차 제외). 임의 각도 회전·이동에 대한 불변성이나 경도 라벨 불변성은 보장하지 않습니다. 유용한 방향 의존 신호를 평균하면서 성능이 나빠질 가능성도 검증합니다.",
        "- 5outer×3inner와 기존6alpha를 유지합니다. 원본 특징도 후보이므로 내부 검증에서 D4가 불리하면 원본을 선택할 수 있습니다. 같은 inner 점수는 원본, 그다음 큰 alpha를 우선합니다.",
        "- X/y StandardScaler를 매 inner 학습 부분에서만 적합합니다. 2표현×6alpha×3inner×5outer +2표현×5재적합 =190회 Ridge 적합/seed입니다. CNN/ViT 가중치는 고정이고 epoch 학습은 없습니다.",
        "- dino_replaced_blend = 기존 조합 +0.5×(joint_nested−기존 DINO)입니다. 수작업25/ResNet25를 유지하며 가중치 탐색·사후 fold 교체·clipping을 하지 않습니다.",
        f"- 원본 전역 OOF/inner 점수 재현: {summary['global_anchor']['status']}; 최대 예측 차이 {summary['global_anchor']['max_absolute_prediction_difference']:.3g}.",
        "- partial80/160 OOF는 이전 검토 결과를 비교용으로 읽습니다. ID·정답·분할·기준예측·조합 수식과 현재 파일 해시를 확인하며 이 실행에서 그 모델들의 checkpoint를 다시 감사하거나 재학습하지 않습니다.",
        "- 기존 조합 대비 두 seed의 전체오차·fold·꼬리·중간·잡음·큰 실패 이미지 변화를 함께 검토합니다. 개별 표현 또는 seed별 좋은 결과만 골라 주 결과로 바꾸지 않습니다. 원본 fallback도 외부 성능 비악화를 보장하지 않습니다.",
        "- 두 seed는 동일500장의 재분할입니다. 반복 연구의 선택 편향이 남으며 독립 검증이나 대회 점수가 아닙니다. 잡음 지표는 구조 신호도 포함할 수 있습니다.",
        "- 고정 교체 조합이 두 seed에서 일관된 이득을 보이지 않으면 이 실험을 확장하지 않습니다. 단독 성능이 나빠도 조합의 보완성은 평가할 수 있습니다. 관측 이득이 극히 작으면 후보만 보존하고 확정 개선으로 해석하지 않습니다.", ""]
    return "\n".join(lines)


def write_outputs(output_dir, data, predictions, selections, diagnostics, tables, seed, threads, elapsed, anchor):
    core.write_outputs(output_dir, data, predictions, selections, diagnostics, tables, seed, threads, elapsed,
        alphas=ALPHAS, report_builder=build_markdown, summary_extras={"experiment": "dinov2_d4_mean",
            "global_anchor": anchor, "d4_protocol": extract.PROTOCOL, "representations": list(REPRESENTATIONS),
            "estimator_fit_count": 190, "candidate_count_per_inner_search": 12,
            "fixed_diagnostic": {"name": FIXED_BLEND, "weights": {"hand_ridge": 0.25, "frozen_resnet": 0.25, "joint_nested": 0.5}},
            "partial_comparison_references": PARTIAL_REFERENCES})


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--global-features", type=Path, default=ROOT / "reports/frozen_dinov2_small/features.npz")
    parser.add_argument("--d4-features", type=Path, default=ROOT / "reports/dinov2_d4/features/features.npz")
    parser.add_argument("--train-csv", type=Path, default=ROOT / "open/train.csv")
    parser.add_argument("--audit-features", type=Path, default=ROOT / "reports/audit_train_features.csv")
    for flag in ("folds", "baseline", "previous-frozen-oof", "previous-dino-oof", "previous-dino-summary",
                 "partial80-oof", "partial160-oof", "output-dir"):
        parser.add_argument("--" + flag, type=Path)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--preflight", action="store_true")
    modes.add_argument("--validate-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    defaults = prior.parse_args(["--seed", str(args.seed)])
    for name in ("folds", "baseline", "previous_frozen_oof", "previous_dino_oof"):
        if getattr(args, name) is None:
            setattr(args, name, getattr(defaults, name))
    if args.previous_dino_summary is None:
        args.previous_dino_summary = args.previous_dino_oof.with_name("summary.json")
    for cap, directory in ((80, "dinov2_ridge_initialized"), (160, "dinov2_ridge_initialized_160")):
        if getattr(args, f"partial{cap}_oof") is None:
            setattr(args, f"partial{cap}_oof", ROOT / f"reports/{directory}/seed{args.seed}/oof.csv")
    if args.output_dir is None:
        args.output_dir = ROOT / f"reports/dinov2_d4/seed{args.seed}"
    return args


def main(argv=None):
    args = parse_args(argv)
    seed = core.validate_seed(args.seed)
    if np is None or args.threads < 1:
        raise ValueError("NumPy and positive --threads required")
    if not (args.preflight or args.validate_only):
        protected = [args.global_features, args.d4_features, args.train_csv, args.audit_features, args.folds,
            args.baseline, args.previous_frozen_oof, args.previous_dino_oof, args.previous_dino_summary,
            args.partial80_oof, args.partial160_oof]
        if any(path.resolve().parent == args.output_dir.resolve() for path in protected):
            raise ValueError("Output directory contains a protected input/reference")
        if args.output_dir.exists() and (not args.output_dir.is_dir() or (any(args.output_dir.iterdir()) and not args.overwrite)):
            raise ValueError("Output already exists; use a new directory or explicit --overwrite")
    started = time.perf_counter()
    data, references, anchor = load_prior(args)
    if args.preflight:
        print(json.dumps({"status": "prior_artifacts_valid", "seed": seed, "rows": len(data.y),
            "d4_extraction_required": True, "d4_cache_checked": False,
            "partial_comparison_rmse": {str(cap): core.metrics(data.y, references[name])["rmse"] for cap, name in PARTIAL_REFERENCES.items()},
            "estimators_fitted": 0, "output_written": False}, indent=2))
        return 0
    data = attach_features(data, args.d4_features, args.global_features)
    if args.validate_only:
        print(json.dumps({"status": "all_artifacts_valid", "seed": seed, "rows": len(data.y),
            "feature_dimensions": {name: x.shape[1] for name, x in data.features.items()},
            "estimators_fitted": 0, "output_written": False}, indent=2))
        return 0
    predictions, selections = core.nested_predictions(data, seed, args.threads, alphas=ALPHAS, representations=REPRESENTATIONS)
    anchor_check = prior.verify_global_anchor(data, predictions, selections, anchor)
    predictions = add_fixed_candidate(predictions, references)
    diagnostics, tables = diagnose(data, predictions)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_outputs(args.output_dir, data, predictions, selections, diagnostics, tables, seed, args.threads,
                  time.perf_counter()-started, anchor_check)
    print(f"Global anchor: {anchor_check['status']}")
    for name in (core.PRIMARY_NAME, D4 + "_nested", FIXED_BLEND):
        m = diagnostics[name]["overall"]
        print(f"{name}: RMSE={m['rmse']:.6f}; delta vs current={m['delta_rmse_vs_current_blend']:+.6f}; "
              f"delta vs partial80={m['delta_rmse_vs_partial80_blend']:+.6f}; delta vs partial160={m['delta_rmse_vs_partial160_blend']:+.6f}")
    print(f"Report: {(args.output_dir / 'report.md').resolve()}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError) as exc:
        print(f"D4 evaluation stopped: {exc}", file=sys.stderr)
        raise SystemExit(2)
