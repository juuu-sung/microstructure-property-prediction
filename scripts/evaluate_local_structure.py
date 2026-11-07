"""User-run nested Ridge: replace only handcrafted25%, keeping ResNet/DINO fixed.

--preflight/--validate-only never fit or write. No torch, image reads, test,
downloads, learned stacking, representation gating, or ensemble-weight search.
"""
from __future__ import annotations

import argparse
import contextlib
from datetime import datetime, timezone
import importlib.metadata
import io
import json
import os
from pathlib import Path
import sys
import time

if __package__:
    from . import evaluate_resnet_d4 as previous
    from . import extract_local_structure as extract
else:
    import evaluate_resnet_d4 as previous
    import extract_local_structure as extract

core, np, ROOT = previous.core, previous.np, previous.ROOT
RAW, AUGMENTED = extract.RAW, extract.AUGMENTED
REPRESENTATIONS = (RAW, AUGMENTED)
ALPHAS = (10.0, 100.0, 1000.0)
REFERENCE = previous.REFERENCE
PRIMARY = "structure_replaced_blend"
CONTROL = "hand_tuned_control_blend"
ANCHOR = "hand_alpha100_anchor"


def load_prior(args, *, expected_n=500):
    prior_args = previous.parse_args(["--seed", str(args.seed)])
    data, all_refs, _ = previous.load_prior(prior_args, expected_n=expected_n)
    original, reference = extract.original_inputs(args.original_features, args.definition, data.ids)
    data.features = {RAW: original}
    data.metadata["hand_original_reference"] = reference
    data.provenance["hand_original"] = reference
    data.provenance["local_structure_evaluation_sources"] = {name: extract.file_reference(Path(path)) for name, path in {
        "evaluator": __file__, "extractor": extract.__file__, "nested_core": core.__file__, "prior_loader": previous.__file__}.items()}
    refs = {n: all_refs[n] for n in (REFERENCE, previous.RAW_REFERENCE, "reference_dino_d4")}
    expected = 0.25 * data.baseline + 0.25 * refs[previous.RAW_REFERENCE] + 0.5 * refs["reference_dino_d4"]
    np.testing.assert_allclose(refs[REFERENCE], expected, atol=1e-10, rtol=0)
    return data, refs


def attach_features(data, path):
    oldmeta = data.metadata["resnet_original"]
    values, metadata = extract.validate_cache(path, data.ids, data.features[RAW], data.metadata["hand_original_reference"],
        oldmeta["train_csv_sha256"], oldmeta["image_manifest_sha256"])
    data.features[AUGMENTED] = np.concatenate([data.features[RAW], values], axis=1)
    data.metadata["local_structure"] = metadata
    data.provenance["local_structure_features"] = {**extract.file_reference(path),
        "metadata_path": str(path.with_name("metadata.json").resolve()), "metadata_sha256": core.sha256(path.with_name("metadata.json"))}
    return data


def fixed_anchor(data, threads):
    from threadpoolctl import threadpool_limits
    predictions = np.full(len(data.y), np.nan)
    with threadpool_limits(limits=threads):
        for fold in range(5):
            train, valid = data.folds != fold, data.folds == fold
            model = core.make_estimator(100.0)
            model.fit(data.features[RAW][train], data.y[train])
            predictions[valid] = model.predict(data.features[RAW][valid])
    if not np.isfinite(predictions).all():
        raise ValueError("Incomplete/nonfinite alpha100 anchor")
    difference = float(np.max(np.abs(predictions - data.baseline)))
    if difference > 1e-7:
        raise ValueError(f"Original handcrafted alpha100 OOF anchor mismatch: {difference}")
    return predictions, {"status": "matched", "max_absolute_prediction_difference": difference, "atol": 1e-7,
                         "fit_count": 5, "alpha": 100.0, "target_source": "train.csv"}


def nested_predictions(data, seed, threads):
    predictions, by_rep = {}, {}
    for name in REPRESENTATIONS:
        with contextlib.redirect_stdout(io.StringIO()):
            pred, records = core.nested_predictions(data, seed, threads, alphas=ALPHAS, representations=(name,))
        predictions[name + "_nested"] = pred[name + "_nested"]
        by_rep[name] = records
    selections = []
    for fold in range(5):
        source = by_rep[RAW][fold]
        record = {k: source[k] for k in ("outer_fold", "outer_train_n", "outer_valid_n", "inner_seed", "inner_validation_sizes")}
        record["candidates"] = [c for name in REPRESENTATIONS for c in by_rep[name][fold]["candidates"]]
        record["per_representation"] = {name: by_rep[name][fold]["per_representation"][name] for name in REPRESENTATIONS}
        selections.append(record)
        print(f"Outer fold {fold}: original alpha={record['per_representation'][RAW]['alpha']:g}; augmented alpha={record['per_representation'][AUGMENTED]['alpha']:g}; representations fixed", flush=True)
    return predictions, selections


def add_fixed_blends(data, nested, refs, anchor):
    if set(nested) != {n + "_nested" for n in REPRESENTATIONS}:
        raise ValueError("Unexpected prediction set")
    return {**nested, **refs, ANCHOR: anchor,
        PRIMARY: refs[REFERENCE] + 0.25 * (nested[AUGMENTED + "_nested"] - data.baseline),
        CONTROL: refs[REFERENCE] + 0.25 * (nested[RAW + "_nested"] - data.baseline)}


def diagnose(data, predictions):
    diagnostics, tables = core.diagnose(data, predictions)
    all_predictions = {core.BASELINE_NAME: data.baseline, **predictions}
    target_bins = core.rank_groups(data.y, 10)
    orientation = data.features[RAW][:, extract.hand.expected_handcrafted_names().index("orientation_anisotropy")]
    orientation_bins = core.rank_groups(orientation, 4)
    regions = {"low10": target_bins == 0, "high10": target_bins == 9,
        "tails20": (target_bins == 0) | (target_bins == 9), "middle80": (target_bins > 0) & (target_bins < 9)}
    for table_name, diag_key, masks, group_key in (
        ("target_region_metrics", "target_regions", regions, "region"),
        ("orientation_quartile_metrics", "orientation_quartiles", {str(q + 1): orientation_bins == q for q in range(4)}, "quartile")):
        tables[table_name] = []
        for name, pred in all_predictions.items():
            summaries = {}
            for group, mask in masks.items():
                m = core.metrics(data.y[mask], pred[mask])
                summaries[group] = m
                tables[table_name].append({"model": name, group_key: group, **m})
            diagnostics[name][diag_key] = summaries
    for name, diag in diagnostics.items():
        p = all_predictions[name]
        for label, ref in (("current", REFERENCE), ("tuned_control", CONTROL)):
            diag["overall"]["delta_rmse_vs_" + label] = diag["overall"]["rmse"] - diagnostics[ref]["overall"]["rmse"]
            diag["overall"]["folds_improved_vs_" + label] = sum(core.metrics(data.y[data.folds == f], p[data.folds == f])["rmse"] <
                core.metrics(data.y[data.folds == f], predictions[ref][data.folds == f])["rmse"] for f in range(5))
    for row in tables["metrics"]:
        row.update(diagnostics[row["model"]]["overall"])
    for table_name, key in (("fold_metrics", "fold"), ("target_decile_metrics", "decile"),
                           ("noise_quartile_metrics", "quartile"), ("target_region_metrics", "region"),
                           ("orientation_quartile_metrics", "quartile")):
        for label, ref in (("current", REFERENCE), ("tuned_control", CONTROL)):
            reference = {r[key]: r["rmse"] for r in tables[table_name] if r["model"] == ref}
            for row in tables[table_name]:
                row["delta_rmse_vs_" + label] = row["rmse"] - reference[row[key]]
    top = {name: set(np.argsort(-(predictions[name] - data.y) ** 2, kind="stable")[:20]) for name in (REFERENCE, CONTROL, PRIMARY)}
    tables["replacement_errors"] = [{"ID": str(ident), "fold": int(data.folds[i]), "hardness": float(data.y[i]),
        "reference_prediction": float(predictions[REFERENCE][i]), "control_prediction": float(predictions[CONTROL][i]),
        "new_prediction": float(predictions[PRIMARY][i]),
        "delta_squared_error_vs_current": float((predictions[PRIMARY][i]-data.y[i])**2-(predictions[REFERENCE][i]-data.y[i])**2),
        "delta_squared_error_vs_tuned_control": float((predictions[PRIMARY][i]-data.y[i])**2-(predictions[CONTROL][i]-data.y[i])**2),
        "in_reference_top20": i in top[REFERENCE], "in_control_top20": i in top[CONTROL], "in_new_top20": i in top[PRIMARY],
        "orientation_quartile": int(orientation_bins[i]+1)} for i, ident in enumerate(data.ids)]
    diagnostics[PRIMARY]["top20_overlap"] = {"current": len(top[PRIMARY] & top[REFERENCE]), "tuned_control": len(top[PRIMARY] & top[CONTROL])}
    return diagnostics, tables


def build_markdown(summary):
    d = summary["diagnostics"]
    lines = ["# 국소 구조60개 + 수작업125개 — 수작업25%만 교체", "",
        "주 결과는 structure_replaced_blend입니다. 현재 기준과 alpha 탐색 대조군을 모두 비교합니다.", "",
        "| 모델 | OOF RMSE | 현재 기준 Δ | 규제 탐색 대조군 Δ |", "|---|---:|---:|---:|"]
    for name in (core.BASELINE_NAME, RAW+"_nested", AUGMENTED+"_nested", REFERENCE, CONTROL, PRIMARY):
        m = d[name]["overall"]
        lines.append(f"| {name} | {m['rmse']:.6f} | {m['delta_rmse_vs_current']:+.6f} | {m['delta_rmse_vs_tuned_control']:+.6f} |")
    lines += ["", "## 내부 규제 선택", "", "| fold | 기존125개 alpha | 보강185개 alpha |", "|---|---:|---:|"]
    for r in summary["selections"]:
        lines.append(f"| {r['outer_fold']} | {r['per_representation'][RAW]['alpha']:g} | {r['per_representation'][AUGMENTED]['alpha']:g} |")
    lines += ["", "## 약점", "", "| 조합 | 양끝20% | 중간80% | 잡음Q4 | 방향성Q4 |", "|---|---:|---:|---:|---:|"]
    for name in (REFERENCE, CONTROL, PRIMARY):
        v = d[name]
        lines.append(f"| {name} | {v['target_regions']['tails20']['rmse']:.6f} | {v['target_regions']['middle80']['rmse']:.6f} | {v['noise_quartiles'][-1]['rmse']:.6f} | {v['orientation_quartiles']['4']['rmse']:.6f} |")
    lines += ["", "## 고정한 절차", "",
        "- 국소 특징60개: sigma/rho=(0.6,2),(1.2,4),(2.4,8), 각18개 + 인접 크기 간6개. 기존 명암·절대 방향을 보존하며 특징·관찰 크기를 점수로 고르지 않습니다.",
        "- 각각 고정한125/185개 표현에서 alpha10/100/1000만 inner3-fold pooled MSE로 선택합니다. X/y 표준화는 해당 학습 부분에서만 적합합니다. 표현 joint 선택은 없습니다.",
        "- 새 조합=현재 기준+0.25×(보강 수작업−기존alpha100), 대조군=현재 기준+0.25×(기존125개 내부선택−기존alpha100). ResNet/DINO D4 예측은 변경하지 않습니다.",
        "- 기존 OOF는 비교·고정 평균에만 사용하며 학습 입력이나 내부 alpha 선택에 사용하지 않습니다. stacking/가중치 탐색/보정/꼬리 가중이 없습니다.",
        f"- 기존 alpha100 OOF anchor: {summary['hand_anchor']['status']}; 최대 차이 {summary['hand_anchor']['max_absolute_prediction_difference']:.3g}.",
        "- 적합105회/seed: 2표현×3alpha×3inner×5outer + 2표현×5재적합 + alpha100 anchor5회. 신경망 학습·GPU·test·전체500장 최종 적합 없음.",
        "- 두 seed에서 현재 기준과 규제 탐색 대조군을 모두 이겨야 특징 추가 이득을 인정합니다. 두 seed 모두 개선하고 현재 기준 대비 평균0.10 이상 개선한 후보를 우선합니다. 이는 통계적 유의성 기준이 아닙니다.",
        "- 같은500장 반복연구의 선택 편향이 남습니다. 방향성·잡음·정답 구간은 진단용이며 인과 효과나 추론 라우팅 기준이 아닙니다.", ""]
    return "\n".join(lines)


def write_outputs(folder, data, predictions, selections, diagnostics, tables, seed, elapsed, anchor, threads):
    summary = {"created_utc": datetime.now(timezone.utc).isoformat(), "experiment": "local_structure_hand_replacement",
        "seed": seed, "n_train": len(data.y), "primary_model": PRIMARY, "comparison_reference": REFERENCE, "tuning_control": CONTROL,
        "representations": list(REPRESENTATIONS), "representation_selection": False, "alpha_candidates": list(ALPHAS),
        "outer_splits": 5, "inner_splits": 3, "estimator_fit_count": 105, "threads": threads, "elapsed_seconds": elapsed,
        "feature_dimensions": {n: x.shape[1] for n, x in data.features.items()}, "protocol": extract.PROTOCOL,
        "input_files": data.provenance, "feature_metadata": data.metadata, "hand_anchor": anchor, "selections": selections,
        "diagnostics": diagnostics, "fixed_weights": {"hand": 0.25, "original_resnet": 0.25, "dino_d4": 0.5},
        "versions": {n: importlib.metadata.version(n) for n in ("numpy", "scipy", "scikit-learn")},
        "scope": "Train-only fixed representations and nested alpha selection; no test or final all-data fit."}
    oof = [{"ID": str(ident), "hardness": float(data.y[i]), "fold": int(data.folds[i]), core.BASELINE_NAME: float(data.baseline[i]),
            **{n: float(p[i]) for n, p in predictions.items()}} for i, ident in enumerate(data.ids)]
    search = [{"outer_fold": r["outer_fold"], "inner_seed": r["inner_seed"], "representation": c["representation"], "alpha": c["alpha"],
        "inner_mse": c["inner_mse"], **{f"inner_fold{j}_mse": v for j, v in enumerate(c["inner_fold_mse"])},
        "selected_for_representation": c == r["per_representation"][c["representation"]]} for r in selections for c in r["candidates"]]
    outputs = {"summary.json": json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), "report.md": build_markdown(summary),
        "oof.csv": core.csv_text(oof), "inner_search.csv": core.csv_text(search), **{n+".csv": core.csv_text(rows) for n, rows in tables.items()}}
    folder.mkdir(parents=True, exist_ok=True)
    for name, content in outputs.items():
        temporary = folder / (name + f".{os.getpid()}.tmp")
        temporary.write_text(content, encoding="utf-8", newline="")
        os.replace(temporary, folder / name)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--original-features", type=Path, default=ROOT / "reports/benchmark_features.npz")
    p.add_argument("--definition", type=Path, default=ROOT / "reports/benchmark_seed42.json")
    p.add_argument("--features", type=Path, default=ROOT / "reports/local_structure/features/features.npz")
    p.add_argument("--output-dir", type=Path)
    modes = p.add_mutually_exclusive_group()
    modes.add_argument("--preflight", action="store_true")
    modes.add_argument("--validate-only", action="store_true")
    args = p.parse_args(argv)
    if args.output_dir is None:
        args.output_dir = ROOT / f"reports/local_structure/seed{args.seed}"
    return args


def main(argv=None):
    args = parse_args(argv)
    core.validate_seed(args.seed)
    if args.threads < 1:
        raise ValueError("--threads must be positive")
    readonly = args.preflight or args.validate_only
    if not readonly:
        extract.protect_output(args.output_dir, [args.features, args.original_features, args.definition, Path(__file__), ROOT / "open/train.csv"])
    start = time.perf_counter()
    data, refs = load_prior(args)
    if not readonly:
        # Include all implicit saved reference inputs in output protection.
        def paths(value):
            if isinstance(value, dict):
                for k, v in value.items():
                    if k.endswith("path") and isinstance(v, str):
                        yield Path(v)
                    else:
                        yield from paths(v)
            elif isinstance(value, list):
                for v in value:
                    yield from paths(v)
        extract.protect_output(args.output_dir, list(paths(data.provenance)))
    if args.preflight:
        print(json.dumps({"status": "prior_artifacts_valid", "seed": args.seed, "rows": len(data.y), "reference_rmse": core.metrics(data.y, refs[REFERENCE])["rmse"],
            "local_cache_checked": False, "estimators_fitted": 0, "output_written": False}))
        return 0
    data = attach_features(data, args.features)
    if args.validate_only:
        print(json.dumps({"status": "all_artifacts_valid", "feature_dimensions": {n: x.shape[1] for n, x in data.features.items()},
            "estimators_fitted": 0, "output_written": False}))
        return 0
    anchor_predictions, check = fixed_anchor(data, args.threads)
    print(f"Handcrafted alpha100 anchor: {check['status']}", flush=True)
    nested, selections = nested_predictions(data, args.seed, args.threads)
    predictions = add_fixed_blends(data, nested, refs, anchor_predictions)
    diagnostics, tables = diagnose(data, predictions)
    previous.verify_hashes(data.provenance)
    write_outputs(args.output_dir, data, predictions, selections, diagnostics, tables, args.seed, time.perf_counter()-start, check, args.threads)
    for name in (CONTROL, PRIMARY):
        m = diagnostics[name]["overall"]
        print(f"{'PRIMARY' if name == PRIMARY else 'CONTROL'} {name}: RMSE={m['rmse']:.6f}; delta vs current={m['delta_rmse_vs_current']:+.6f}; delta vs tuned control={m['delta_rmse_vs_tuned_control']:+.6f}")
    print(f"Report: {args.output_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError) as exc:
        print(f"Local structure evaluation stopped: {exc}", file=sys.stderr)
        raise SystemExit(2)
