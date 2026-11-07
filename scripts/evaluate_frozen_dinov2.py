"""Train-only nested Ridge evaluation of cached frozen DINOv2-small features.

No torch import, image/test access, downloads, all-data final fitting, or
submission generation. --validate-only validates caches and references without
fitting. The two blend candidates have fixed weights chosen before this run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import time

if __package__:
    from . import evaluate_frozen_resnet as core
    from . import extract_frozen_resnet as common
else:
    import evaluate_frozen_resnet as core
    import extract_frozen_resnet as common

np = core.np
ROOT = core.ROOT
REPRESENTATIONS = ("cls_token", "cls_patch_mean_std")
FEATURE_DIMENSIONS = {"cls_token": 384, "cls_patch_mean_std": 1152}
ALPHAS = (10.0, 100.0, 1000.0, 3000.0, 10000.0, 30000.0)
MODEL_ID = "timm/vit_small_patch14_dinov2.lvd142m"
REVISION = "4610ca143709d58a633b6397a74412c2c3842454"
FROZEN_REFERENCE = "reference_frozen_resnet18"
PREVIOUS_REFERENCE = "reference_previous_equal_blend"
FIXED_BLEND_WEIGHTS = {
    PREVIOUS_REFERENCE: {core.BASELINE_NAME: 0.5, FROZEN_REFERENCE: 0.5},
    "baseline_dino_equal": {core.BASELINE_NAME: 0.5, core.PRIMARY_NAME: 0.5},
    "previous_blend_dino_equal": {core.BASELINE_NAME: 0.25, FROZEN_REFERENCE: 0.25, core.PRIMARY_NAME: 0.5},
}
EXPECTED_CONFIG = {
    "schema_version": 1, "model_id": MODEL_ID, "revision": REVISION,
    "input_shape": [3, 266, 266], "original_image_shape": [1, 256, 256],
    "padding": {"mode": "reflect", "left": 5, "right": 5, "top": 5, "bottom": 5},
    "resize": None, "crop": None, "augmentation": None,
    "normalization_mean": common.EXPECTED_MEAN, "normalization_std": common.EXPECTED_STD,
    "dynamic_img_size": True, "dynamic_img_pad": False, "patch_size": [14, 14],
    "patch_grid": [19, 19], "num_prefix_tokens": 1, "num_register_tokens": 0, "embedding_dim": 384,
    "token_count": 362, "token_source": "model.forward_features, after final LayerNorm",
    "pooling": "normalized CLS; concat normalized CLS,patch mean,patch std; correction=0",
    "positional_embedding": {"stored_grid": [37, 37], "forward_grid": [19, 19],
        "implementation": "timm resample_abs_pos_embed via dynamic_img_size",
        "interpolation": "bicubic", "antialias": True, "stored_parameter_changed": False},
    "dtype": "float32", "amp": False, "train_mode": False,
}


def require_sha256(value, label):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"DINO metadata requires a valid SHA256: {label}")


def validate_feature_metadata(data, features_path: Path, metadata_path: Path):
    """Check recorded extraction provenance without opening any image files."""
    metadata = data.metadata
    sidecar = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata != sidecar:
        raise ValueError("DINO NPZ metadata and metadata.json differ; cache is incomplete or stale")
    config = metadata.get("extraction_config")
    if not isinstance(config, dict):
        raise ValueError("DINO extraction_config is missing")
    for key, expected in EXPECTED_CONFIG.items():
        if key not in config or config[key] != expected:
            raise ValueError(f"DINO extraction configuration mismatch: {key}")
    if (metadata.get("schema_version") != 1 or metadata.get("rows") != len(data.ids)
            or metadata.get("test_accessed") is not False or metadata.get("labels_saved_in_npz") is not False):
        raise ValueError("DINO metadata schema/row count/train-only scope does not match")
    if metadata.get("train_csv_sha256") != data.provenance["train_csv"]["sha256"]:
        raise ValueError("DINO train_csv_sha256 does not match train.csv")
    for key in ("image_manifest_sha256", "feature_state_sha256"):
        require_sha256(metadata.get(key), key)
    for key in ("extractor_script_sha256", "common_utility_sha256"):
        require_sha256(config.get(key), key)
    manifest = metadata.get("training_image_manifest")
    if not isinstance(manifest, list) or len(manifest) != len(data.ids):
        raise ValueError("DINO training image manifest is missing or has the wrong size")
    for entry in manifest:
        if not isinstance(entry, dict) or not isinstance(entry.get("id"), str):
            raise ValueError("DINO training image manifest has an invalid entry")
        if (entry.get("relative_path") != f"train/{entry['id']}.png"
                or not isinstance(entry.get("size_bytes"), int) or entry["size_bytes"] < 1):
            raise ValueError("DINO training image manifest has an invalid path or size")
        require_sha256(entry.get("sha256"), "training image")
    digest = hashlib.sha256(common.canonical_json(manifest).encode()).hexdigest()
    if digest != metadata["image_manifest_sha256"]:
        raise ValueError("DINO training image manifest failed its integrity hash")
    with np.load(features_path, allow_pickle=False) as archive:
        if [entry["id"] for entry in manifest] != archive["ids"].tolist():
            raise ValueError("DINO training manifest and cached feature ID order differ")
        hashes = metadata.get("feature_array_sha256")
        if not isinstance(hashes, dict) or set(hashes) != set(FEATURE_DIMENSIONS):
            raise ValueError("DINO feature array hashes are missing or have unexpected keys")
        for name in REPRESENTATIONS:
            require_sha256(hashes[name], name)
            if common.sha256_array(archive[name]) != hashes[name]:
                raise ValueError(f"DINO {name} failed its integrity hash")
        if not np.array_equal(archive["cls_token"], archive["cls_patch_mean_std"][:, :384]):
            raise ValueError("DINO combined representation must begin with the exact CLS features")
        if np.any(archive["cls_patch_mean_std"][:, 768:] < 0):
            raise ValueError("DINO patch standard deviation features must be nonnegative")
    data.provenance["feature_metadata_sidecar"] = {"path": str(metadata_path.resolve()), "sha256": core.sha256(metadata_path)}
    data.provenance["cache_validation"] = {
        "metadata_sidecar_matches_npz": True, "feature_array_hashes_verified": True,
        "model_and_preprocessing_match_fixed_protocol": True,
        "image_manifest_hash_verified": True, "current_image_files_read": False,
        "note": "Recorded image manifest checked; current image pixels are not re-read by this evaluator.",
    }


def load_inputs(features_path: Path, train_path: Path, folds_path: Path, baseline_path: Path,
                audit_path: Path, previous_frozen_path: Path, seed: int, *, metadata_path=None,
                expected_n=500):
    data = core.load_inputs(features_path, train_path, folds_path, baseline_path, audit_path, seed,
                            expected_n=expected_n, feature_dimensions=FEATURE_DIMENSIONS)
    metadata_path = Path(metadata_path) if metadata_path is not None else features_path.with_name("metadata.json")
    validate_feature_metadata(data, features_path, metadata_path)
    rows = core.align_rows(core.read_rows(previous_frozen_path,
        {"ID", "hardness", "fold", core.BASELINE_NAME, core.PRIMARY_NAME}), data.ids, previous_frozen_path)
    if not np.array_equal(core.integer_column(rows, "fold", previous_frozen_path), data.folds):
        raise ValueError("Previous frozen ResNet OOF fold IDs do not match")
    if not np.array_equal(core.finite_column(rows, "hardness", previous_frozen_path), data.y):
        raise ValueError("Previous frozen ResNet OOF targets do not exactly match train.csv")
    if not np.array_equal(core.finite_column(rows, core.BASELINE_NAME, previous_frozen_path), data.baseline):
        raise ValueError("Previous frozen ResNet OOF baseline predictions do not match the selected baseline")
    frozen = core.finite_column(rows, core.PRIMARY_NAME, previous_frozen_path)
    references = {FROZEN_REFERENCE: frozen, PREVIOUS_REFERENCE: 0.5 * data.baseline + 0.5 * frozen}
    data.provenance["previous_frozen_oof"] = {
        "path": str(previous_frozen_path.resolve()), "sha256": core.sha256(previous_frozen_path),
        "prediction_column": core.PRIMARY_NAME, "targets_and_folds_exactly_match": True,
        "baseline_predictions_exactly_match": True,
    }
    return data, references


def add_fixed_candidates(data, nested_predictions, references):
    expected = {f"{name}_nested" for name in REPRESENTATIONS} | {core.PRIMARY_NAME}
    if set(nested_predictions) != expected or set(references) != {FROZEN_REFERENCE, PREVIOUS_REFERENCE}:
        raise ValueError("Unexpected nested prediction or reference keys")
    joint = nested_predictions[core.PRIMARY_NAME]
    return {
        **nested_predictions, **references,
        "baseline_dino_equal": 0.5 * data.baseline + 0.5 * joint,
        "previous_blend_dino_equal": 0.5 * references[PREVIOUS_REFERENCE] + 0.5 * joint,
    }


def diagnose(data, predictions):
    diagnostics, tables = core.diagnose(data, predictions)
    all_predictions = {core.BASELINE_NAME: data.baseline, **predictions}
    previous = predictions[PREVIOUS_REFERENCE]
    previous_error = previous - data.y
    previous_rmse = core.metrics(data.y, previous)["rmse"]
    for row in tables["metrics"]:
        name = row["model"]
        delta = row["rmse"] - previous_rmse
        improved = sum(core.metrics(data.y[data.folds == fold], all_predictions[name][data.folds == fold])["rmse"]
                       < core.metrics(data.y[data.folds == fold], previous[data.folds == fold])["rmse"] for fold in range(5))
        extra = {"delta_rmse_vs_previous_blend": delta, "folds_improved_vs_previous_blend": int(improved)}
        row.update(extra)
        diagnostics[name]["overall"].update(extra)
    previous_groups = {}
    for table_name, group_key in (("fold_metrics", "fold"), ("target_decile_metrics", "decile"), ("noise_quartile_metrics", "quartile")):
        previous_groups[table_name] = {row[group_key]: row["rmse"] for row in tables[table_name] if row["model"] == PREVIOUS_REFERENCE}
        for row in tables[table_name]:
            row["delta_rmse_vs_previous_blend"] = row["rmse"] - previous_groups[table_name][row[group_key]]
    by_id = {str(ident): index for index, ident in enumerate(data.ids)}
    for row in tables["per_image_errors"]:
        i = by_id[row["ID"]]
        row.update(previous_blend_prediction=float(previous[i]), previous_blend_squared_error=float(previous_error[i] ** 2),
                   delta_squared_error_vs_previous_blend=float(row["squared_error"] - previous_error[i] ** 2))
    previous_worst = set(np.argsort(previous_error ** 2, kind="stable")[-20:])
    tables["top20_error_changes_vs_previous_blend"] = []
    for name, prediction in all_predictions.items():
        if name == PREVIOUS_REFERENCE:
            continue
        error = prediction - data.y
        current_worst = set(np.argsort(error ** 2, kind="stable")[-20:])
        ranks = np.empty(len(error), dtype=int)
        ranks[np.argsort(-(error ** 2), kind="stable")] = np.arange(1, len(error) + 1)
        for i in sorted(current_worst | previous_worst, key=lambda index: (ranks[index], data.ids[index])):
            tables["top20_error_changes_vs_previous_blend"].append({
                "model": name, "ID": str(data.ids[i]), "fold": int(data.folds[i]), "hardness": float(data.y[i]),
                "prediction": float(prediction[i]), "previous_blend_prediction": float(previous[i]),
                "error": float(error[i]), "previous_blend_error": float(previous_error[i]),
                "delta_squared_error_vs_previous_blend": float(error[i] ** 2 - previous_error[i] ** 2),
                "current_error_rank": int(ranks[i]), "in_current_top20": i in current_worst,
                "in_previous_blend_top20": i in previous_worst,
            })
    return diagnostics, tables


def build_markdown(summary):
    diagnostics = summary["diagnostics"]
    reference_rmse = diagnostics[PREVIOUS_REFERENCE]["overall"]["rmse"]
    lines = ["# 고정 DINOv2-small 특징 + nested Ridge 검증", "",
        "주 평가값은 **joint_nested의 전체 OOF RMSE**입니다. 각 outer 학습 부분의 inner 3-fold에서 표현과 alpha를 함께 선택했습니다. 표현별 결과와 고정 앙상블은 별도 진단이며 주 평가값을 대체하지 않습니다.", "",
        f"기존 수작업 Ridge와 frozen ResNet18의 50:50 평균을 동일 ID·정답·fold로 재확인한 기준 RMSE: **{reference_rmse:.6f}**.", "",
        "| 모델 | OOF RMSE | 수작업 Ridge 대비 Δ | 기존 50:50 대비 Δ | 기존 50:50 대비 개선 fold |",
        "|---|---:|---:|---:|---:|"]
    for name, result in diagnostics.items():
        m = result["overall"]
        lines.append(f"| {name} | {m['rmse']:.6f} | {m['delta_rmse_vs_baseline']:+.6f} | {m['delta_rmse_vs_previous_blend']:+.6f} | {m['folds_improved_vs_previous_blend']}/5 |")
    lines += ["", "Δ가 음수이면 해당 기준보다 개선된 것입니다. 모든 OOF 제곱오차를 합쳐 RMSE를 계산했습니다.", "",
              "## 사전에 고정한 평균", "",
              "- baseline_dino_equal: 수작업 Ridge 50% + DINO joint_nested 50%.",
              "- previous_blend_dino_equal: 기존 50:50 평균 50% + DINO joint_nested 50%, 즉 수작업 Ridge 25% + frozen ResNet18 25% + DINO 50%.",
              "- 가중치 최적화·clipping·calibration·tail 가중은 수행하지 않았습니다. 두 평균 중 좋은 결과를 사후 선택하면 추가 선택 편향이 생깁니다.", "",
              "## Fold 내부 선택", "", "| outer fold | 표현 | alpha | inner OOF MSE |", "|---|---|---:|---:|"]
    for record in summary["selections"]:
        winner = record["joint"]
        lines.append(f"| {record['outer_fold']} | {winner['representation']} | {winner['alpha']:g} | {winner['inner_mse']:.6f} |")
    lines += ["", "## 경도 양끝과 영상 잡음 대용지표", "",
              "| 모델 | 최저 10% RMSE | 최고 10% RMSE | 잡음 Q1 RMSE | 잡음 Q4 RMSE |", "|---|---:|---:|---:|---:|"]
    for name, result in diagnostics.items():
        target, noise = result["target_deciles"], result["noise_quartiles"]
        lines.append(f"| {name} | {target[0]['rmse']:.3f} | {target[-1]['rmse']:.3f} | {noise[0]['rmse']:.3f} | {noise[-1]['rmse']:.3f} |")
    alpha_text = ", ".join(f"{alpha:g}" for alpha in summary["alpha_candidates"])
    lines += ["", "## 해석과 재현", "",
        f"- alpha 후보: {alpha_text}. X와 y의 StandardScaler는 매 inner 학습 부분 안에서 새로 적합합니다.",
        "- 256×256 원본에 각 변 5픽셀 reflect padding만 적용한 266×266 표현입니다. 최종 LayerNorm 이후 CLS와 361개 patch token의 평균·population 표준편차를 사용합니다.",
        "- 기존 기준선과 기존 50:50 평균은 같은 데이터의 이전 연구에서 선택된 결과입니다. 이번 nested 선택도 반복 연구 전체의 선택 편향을 제거하지 않습니다.",
        "- 이 점수는 독립 holdout 또는 대회 Public/Private 점수가 아닙니다. 여러 fold와 잔차·경도 양끝·잡음 구간을 함께 확인하세요.",
        "- 잡음 대용지표는 실제 미세구조도 반영할 수 있으므로 잡음만의 효과로 단정하지 않습니다.",
        "- 캐시의 metadata sidecar, 모델 revision, 전처리, 배열/CSV/기록된 이미지 manifest 해시를 검사했습니다. 현재 이미지 파일은 이 평가기에서 읽지 않습니다.",
        "- test, 전체 500장 최종 적합, 제출 생성, 신경망 학습을 수행하지 않습니다.", "",
        "fold_metrics.csv, target_decile_metrics.csv, noise_quartile_metrics.csv에 기존 평균 대비 Δ를 기록했습니다. residual_correlations.csv는 잔차 상관, top20_error_changes_vs_previous_blend.csv는 큰 오류 이미지의 변화, oof.csv는 원단위 예측입니다. 상세 설정·고정 가중치·입력 경로와 SHA256은 summary.json에 있습니다.", ""]
    return "\n".join(lines)


def write_outputs(output_dir, data, predictions, selections, diagnostics, tables, seed, threads, elapsed, alphas=ALPHAS):
    core.write_outputs(output_dir, data, predictions, selections, diagnostics, tables, seed, threads, elapsed,
        alphas=alphas, report_builder=build_markdown, summary_extras={
            "experiment": "frozen_dinov2_small", "representations": list(REPRESENTATIONS),
            "fixed_blend_weights": FIXED_BLEND_WEIGHTS,
            "reference_prediction": {"name": PREVIOUS_REFERENCE, "rmse": diagnostics[PREVIOUS_REFERENCE]["overall"]["rmse"]},
            "blend_selection_note": "Two weights-fixed diagnostic candidates; primary metric remains joint_nested. No blend weight fitting.",
            "research_selection_note": "Model family and reference blend were chosen after previous experiments on the same 500 samples; nested CV does not undo that selection.",
        })


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", type=Path, default=ROOT / "reports/frozen_dinov2_small/features.npz")
    parser.add_argument("--metadata", type=Path, help="Defaults to metadata.json beside --features")
    parser.add_argument("--train-csv", type=Path, default=ROOT / "open/train.csv")
    parser.add_argument("--folds", type=Path, default=ROOT / "reports/folds.csv")
    parser.add_argument("--baseline", type=Path, default=ROOT / "reports/benchmark_oof_seed42.csv")
    parser.add_argument("--audit-features", type=Path, default=ROOT / "reports/audit_train_features.csv")
    parser.add_argument("--previous-frozen-oof", type=Path, default=ROOT / "reports/frozen_resnet18/seed42/oof.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/frozen_dinov2_small/seed42")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--alphas", type=float, nargs="+", default=ALPHAS)
    parser.add_argument("--validate-only", action="store_true", help="Validate train feature cache/references only; no estimator is fitted")
    parser.add_argument("--overwrite", action="store_true", help="Explicitly replace evaluation output files")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    alphas, seed = core.validate_alphas(args.alphas), core.validate_seed(args.seed)
    if np is None:
        raise ValueError("NumPy is missing; install the project environment first")
    if args.threads < 1:
        raise ValueError("--threads must be positive")
    if not args.validate_only:
        if args.output_dir.exists() and not args.output_dir.is_dir():
            raise ValueError("--output-dir must be a directory")
        if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
            raise ValueError("Output directory contains files; use a new directory or explicit --overwrite")
    start = time.perf_counter()
    data, references = load_inputs(args.features, args.train_csv, args.folds, args.baseline, args.audit_features,
                                  args.previous_frozen_oof, seed, metadata_path=args.metadata)
    if args.validate_only:
        print(json.dumps({"status": "validated", "n_train": len(data.y), "feature_dimensions": FEATURE_DIMENSIONS,
                          "reference_previous_equal_blend_rmse": core.metrics(data.y, references[PREVIOUS_REFERENCE])["rmse"],
                          "estimators_fitted": 0, "torch_imported_by_evaluator": False, "images_or_test_read": False}, indent=2))
        return 0
    predictions, selections = core.nested_predictions(data, seed, args.threads, alphas=alphas, representations=REPRESENTATIONS)
    predictions = add_fixed_candidates(data, predictions, references)
    diagnostics, tables = diagnose(data, predictions)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_outputs(args.output_dir, data, predictions, selections, diagnostics, tables, seed, args.threads,
                  time.perf_counter() - start, alphas=alphas)
    primary = diagnostics[core.PRIMARY_NAME]["overall"]
    print(f"PRIMARY joint_nested OOF RMSE: {primary['rmse']:.6f}; delta vs previous fixed blend {primary['delta_rmse_vs_previous_blend']:+.6f}")
    for name in ("baseline_dino_equal", "previous_blend_dino_equal"):
        result = diagnostics[name]["overall"]
        print(f"FIXED {name} OOF RMSE: {result['rmse']:.6f}; delta vs previous fixed blend {result['delta_rmse_vs_previous_blend']:+.6f}")
    print(f"Report: {(args.output_dir / 'report.md').resolve()}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError) as exc:
        print(f"DINO evaluation stopped: {exc}", file=sys.stderr)
        raise SystemExit(2)
