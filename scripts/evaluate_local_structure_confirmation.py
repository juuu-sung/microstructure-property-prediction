"""User-run confirmation on three prespecified splits using cached features.

--prepare freezes folds/protocol without fitting. --validate-only is read-only.
Default execution refits every regression component on each new outer split.
No historical OOF reuse, image inference, torch, test data or weight search.
"""
from __future__ import annotations

import argparse
import contextlib
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import sys
import time

if __package__:
    from . import evaluate_local_structure as previous
    from . import extract_resnet_d4 as resnet
    from . import extract_dinov2_d4 as dino
else:
    import evaluate_local_structure as previous
    import extract_resnet_d4 as resnet
    import extract_dinov2_d4 as dino

core, np, ROOT = previous.core, previous.np, previous.ROOT
local = previous.extract
SEEDS = (31415, 27182, 16180)
RAW, AUGMENTED = previous.RAW, previous.AUGMENTED
RESNET, DINO = resnet.RAW, dino.FEATURE
REFERENCE, CONTROL, PRIMARY = previous.REFERENCE, previous.CONTROL, previous.PRIMARY
GRIDS = {RAW: (10., 100., 1000.), AUGMENTED: (10., 100., 1000.),
         RESNET: (10., 100., 1000.), DINO: (10., 100., 1000., 3000., 10000., 30000.)}
DIMENSIONS = {RAW: 125, AUGMENTED: 185, RESNET: 1792, DINO: 1152}
FIT_COUNT = 250  # 4 nested regressors: 245; fixed-alpha hand baseline: 5.
PROTOCOL = {
    "version": 1, "seeds": list(SEEDS), "feature_dimensions": DIMENSIONS,
    "alpha_grids": {k: list(v) for k, v in GRIDS.items()},
    "fixed_hand_baseline_alpha": 100., "weights": {"hand": .25, "original_resnet": .25, "dino_d4": .5},
    "outer": "5-fold StratifiedKFold; target rank deciles; shuffle=True; random_state=seed",
    "inner": "3-fold StratifiedKFold; deciles recomputed only on outer training labels; random_state=seed+fold",
    "estimator": "X StandardScaler -> Ridge(solver=auto); y StandardScaler, refit inside every training fold",
    "selection": "per-representation pooled inner MSE; exact ties choose larger alpha; no representation selection",
    "components": "all regression components refitted on the same new folds; frozen image features reused",
    "historical_oof_used": False, "primary": PRIMARY, "comparators": [REFERENCE, CONTROL],
    "decision": "direction supported only if all three seeds improve both comparators; priority if also mean reference gain >=0.10",
    "priority_mean_gain": .10, "stop_rule": "report all three; no replacing seeds or tuning after results",
    "interpretation": "same 500 previously studied images; split sensitivity check, not independent holdout or significance test",
    "local_feature_protocol": local.PROTOCOL, "resnet_feature_protocol": resnet.ORIGINAL_CONFIG,
    "dino_feature_protocol": dino.PROTOCOL, "fit_count_per_seed": FIT_COUNT,
}


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def versions():
    return {n: importlib.metadata.version(n) for n in ("numpy", "scipy", "scikit-learn")}


def source_references():
    paths = {Path(p).resolve() for p in ( __file__, previous.__file__, core.__file__, local.__file__,
        local.hand.__file__, local.common.__file__, resnet.__file__, dino.__file__, dino.shared.__file__,
        dino.base.__file__, dino.dino.__file__)}
    return {p.name: local.file_reference(p) for p in sorted(paths)}


def input_paths():
    return {"train": ROOT / "open/train.csv", "noise_audit": ROOT / "reports/audit_train_features.csv",
        "hand": ROOT / "reports/benchmark_features.npz", "hand_schema": ROOT / "reports/benchmark_seed42.json",
        "local": ROOT / "reports/local_structure/features/features.npz",
        "resnet": ROOT / "reports/frozen_resnet18/features.npz",
        "dino_global": ROOT / "reports/frozen_dinov2_small/features.npz",
        "dino_d4": ROOT / "reports/dinov2_d4/features/features.npz"}


def load_features(*, expected_n=500):
    """Read label-free caches directly, never any historical OOF files."""
    paths = input_paths()
    refs = {k: local.file_reference(p) for k, p in paths.items()}
    refs.update({k + "_metadata": local.file_reference(paths[k].with_name("metadata.json"))
                 for k in ("local", "resnet", "dino_global", "dino_d4")})
    train = core.read_rows(paths["train"], {"ID", "hardness"})
    ids = np.asarray([r["ID"] for r in train])
    y = core.finite_column(train, "hardness", paths["train"])
    if len(ids) != expected_n or expected_n < 50 or expected_n % 5 or np.std(y) == 0:
        raise ValueError("Expected 500 unique training IDs and varying targets")
    res_ids, res, res_meta = resnet.read_original(np, paths["resnet"], paths["train"], expected_n=expected_n)
    din_ids, _, din_meta = dino.shared.read_global_cache(np, paths["dino_global"], paths["train"], expected_n=expected_n)
    if not np.array_equal(ids, res_ids) or not np.array_equal(ids, din_ids):
        raise ValueError("Cache IDs differ")
    if res_meta["image_manifest_sha256"] != din_meta["image_manifest_sha256"]:
        raise ValueError("ResNet and DINO image manifests differ")
    din, d4_meta = dino.validate_cache(np, paths["dino_d4"], ids, din_meta, paths["dino_global"])
    hand, hand_ref = local.original_inputs(paths["hand"], paths["hand_schema"], ids)
    added, local_meta = local.validate_cache(paths["local"], ids, hand, hand_ref,
        refs["train"]["sha256"], res_meta["image_manifest_sha256"])
    audit = core.align_rows(core.read_rows(paths["noise_audit"], {"ID", "hardness", "haar_hh_noise_proxy"}), ids, paths["noise_audit"])
    if not np.allclose(core.finite_column(audit, "hardness", paths["noise_audit"]), y, atol=1e-10, rtol=0):
        raise ValueError("Audit labels differ; only train.csv supplies fitting/scoring labels")
    noise = core.finite_column(audit, "haar_hh_noise_proxy", paths["noise_audit"])
    features = {RAW: hand, AUGMENTED: np.concatenate([hand, added], axis=1), RESNET: res, DINO: din}
    data = core.Inputs(ids, y, features, np.full(len(y), -1), np.full(len(y), np.nan), noise,
        {"hand_reference": hand_ref, "local": local_meta, "resnet": res_meta, "dino_global": din_meta, "dino_d4": d4_meta}, refs)
    check_references(refs)
    return data


def check_references(refs):
    for r in refs.values():
        if core.sha256(Path(r["path"])) != r["sha256"]:
            raise ValueError(f"Frozen input/source changed: {r['path']}")


def outer_folds(y, seed):
    from sklearn.model_selection import StratifiedKFold
    if seed not in SEEDS:
        raise ValueError("Only the three prespecified seeds are permitted")
    folds = np.full(len(y), -1, dtype=np.int64)
    bins = core.rank_groups(y, 10)
    for fold, (_, valid) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(np.zeros(len(y)), bins)):
        folds[valid] = fold
    if set(folds) != set(range(5)) or any(np.sum(folds == f) != len(y)//5 for f in range(5)):
        raise ValueError("Invalid fold counts")
    return folds


def fold_text(data, seed):
    folds, bins = outer_folds(data.y, seed), core.rank_groups(data.y, 10)
    return core.csv_text([{"ID": str(i), "fold": int(f), "seed": seed, "target_quantile": int(b)}
                          for i, f, b in zip(data.ids, folds, bins)])


def frozen_payload(data):
    return {"protocol": PROTOCOL, "n_train": len(data.y), "sources": source_references(), "input_files": data.provenance,
        "versions": versions(), "feature_metadata": data.metadata,
        "aligned_feature_hashes": {k: local.common.sha256_array(v) for k, v in data.features.items()},
        "fold_sha256": {str(s): hashlib.sha256(fold_text(data, s).encode()).hexdigest() for s in SEEDS}}


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8", newline="")


def prepare(folder, data):
    local.protect_output(folder, [Path(r["path"]) for r in {**data.provenance, **source_references()}.values()])
    plan = {"created_utc": datetime.now(timezone.utc).isoformat(), "frozen": frozen_payload(data),
            "prepared_without_fitting": True, "actual_fits_during_preparation": 0}
    folder.mkdir(parents=True, exist_ok=True)
    for seed in SEEDS:
        (folder / f"folds_seed{seed}.csv").write_text(fold_text(data, seed), encoding="utf-8", newline="")
    write_json(folder / "protocol.json", plan)
    return plan


def validate_plan(folder, data):
    path = folder / "protocol.json"
    plan = json.loads(path.read_text(encoding="utf-8"))
    if plan.get("frozen") != frozen_payload(data) or plan.get("prepared_without_fitting") is not True:
        raise ValueError("Prepared protocol, inputs, sources, environment or folds changed")
    for seed in SEEDS:
        if core.sha256(folder / f"folds_seed{seed}.csv") != plan["frozen"]["fold_sha256"][str(seed)]:
            raise ValueError("Prepared fold file changed")
    return plan


def predict_components(data, seed, threads):
    """All scalers/heads see only the new split's training samples."""
    from threadpoolctl import threadpool_limits
    baseline = np.full(len(data.y), np.nan)
    with threadpool_limits(limits=threads):
        for fold in range(5):
            train, valid = data.folds != fold, data.folds == fold
            model = core.make_estimator(100.)
            model.fit(data.features[RAW][train], data.y[train])
            baseline[valid] = model.predict(data.features[RAW][valid])
    nested, selections = {}, []
    for name, alphas in GRIDS.items():
        with contextlib.redirect_stdout(io.StringIO()):
            pred, selected = core.nested_predictions(data, seed, threads, alphas=alphas, representations=(name,))
        nested[name + "_nested"] = pred[name + "_nested"]
        for r in selected:
            if name == RAW:
                selections.append({k: r[k] for k in ("outer_fold", "outer_train_n", "outer_valid_n", "inner_seed", "inner_validation_sizes")})
                selections[-1].update(candidates=[], per_representation={})
            dest = selections[r["outer_fold"]]
            dest["candidates"].extend(r["candidates"])
            dest["per_representation"][name] = r["per_representation"][name]
        print(f"seed={seed}: {name} nested regression completed", flush=True)
    if not np.isfinite(baseline).all() or any(not np.isfinite(p).all() for p in nested.values()):
        raise ValueError("Incomplete OOF predictions")
    return baseline, nested, selections


def blend(baseline, nested):
    if set(nested) != {name + "_nested" for name in GRIDS}:
        raise ValueError("Need all four newly refitted component predictions")
    res, din = nested[RESNET + "_nested"], nested[DINO + "_nested"]
    other = .25 * res + .5 * din
    return {**nested, REFERENCE: .25 * baseline + other,
        CONTROL: .25 * nested[RAW + "_nested"] + other,
        PRIMARY: .25 * nested[AUGMENTED + "_nested"] + other}


def report_text(summary):
    lines = [f"# 국소 구조 추가 분할 확인 — seed {summary['seed']}", "",
        "모든 회귀기를 이 seed의 새 fold에서 재적합. 기존 OOF 예측 재사용 없음.", "",
        "| 모델 | OOF RMSE | 기준 Δ | 규제 대조군 Δ |", "|---|---:|---:|---:|"]
    for name in (REFERENCE, CONTROL, PRIMARY):
        m = summary["diagnostics"][name]["overall"]
        lines.append(f"| {name} | {m['rmse']:.6f} | {m['delta_rmse_vs_current']:+.6f} | {m['delta_rmse_vs_tuned_control']:+.6f} |")
    lines += ["", "단일 seed만 보고 판정하지 않으며, 세 seed 완료 후 상위 폴더 report.md를 확인한다.",
        "같은500장의 분할 민감도 확인이며 독립 holdout이나 통계적 유의성 검정이 아니다.",
        "각 표준화·alpha 선택은 학습 fold 내부에서만 수행. 네 표현 고정, alpha 후보·25:25:50 가중치 유지.", ""]
    return "\n".join(lines)


def run_seed(folder, data, seed, threads):
    dest = folder / f"seed{seed}"
    local.protect_output(dest, [folder / "protocol.json", *input_paths().values()])
    validate_plan(folder, data)
    start = time.perf_counter()
    current = replace(data, folds=outer_folds(data.y, seed), baseline=np.full(len(data.y), np.nan))
    baseline, nested, selections = predict_components(current, seed, threads)
    current = replace(current, baseline=baseline)
    predictions = blend(baseline, nested)
    diagnostics, tables = previous.diagnose(current, predictions)
    validate_plan(folder, data)
    summary = {"experiment": "local_structure_confirmation", "seed": seed, "n_train": len(data.y),
        "protocol_sha256": core.sha256(folder / "protocol.json"), "fold_sha256": core.sha256(folder / f"folds_seed{seed}.csv"),
        "fit_count": FIT_COUNT, "threads": threads, "elapsed_seconds": time.perf_counter()-start,
        "selections": selections, "diagnostics": diagnostics, "historical_oof_used": False,
        "created_utc": datetime.now(timezone.utc).isoformat()}
    oof = [{"ID": str(i), "hardness": float(current.y[j]), "fold": int(current.folds[j]), core.BASELINE_NAME: float(baseline[j]),
            **{n: float(p[j]) for n, p in predictions.items()}} for j, i in enumerate(current.ids)]
    search = [{"outer_fold": r["outer_fold"], "inner_seed": r["inner_seed"], "representation": c["representation"],
        "alpha": c["alpha"], "inner_mse": c["inner_mse"], **{f"inner_fold{j}_mse": v for j, v in enumerate(c["inner_fold_mse"])},
        "selected": c == r["per_representation"][c["representation"]]} for r in selections for c in r["candidates"]]
    dest.mkdir(parents=True, exist_ok=True)
    write_json(dest / "summary.json", summary)
    outputs = {"oof.csv": core.csv_text(oof), "inner_search.csv": core.csv_text(search), "report.md": report_text(summary),
               **{n + ".csv": core.csv_text(rows) for n, rows in tables.items()}}
    for name, text in outputs.items():
        (dest / name).write_text(text, encoding="utf-8", newline="")
    write_json(dest / "completed.json", {"seed": seed, "protocol_sha256": summary["protocol_sha256"],
        "files": {p.name: core.sha256(p) for p in sorted(dest.iterdir()) if p.is_file()}})
    m = diagnostics[PRIMARY]["overall"]
    print(f"PRIMARY seed={seed}: RMSE={m['rmse']:.6f}; delta vs reference={m['delta_rmse_vs_current']:+.6f}; delta vs control={m['delta_rmse_vs_tuned_control']:+.6f}", flush=True)
    return summary


def read_completed(folder, seed, data):
    dest = folder / f"seed{seed}"
    complete = json.loads((dest / "completed.json").read_text(encoding="utf-8"))
    if complete["seed"] != seed or complete["protocol_sha256"] != core.sha256(folder / "protocol.json"):
        raise ValueError("Completion belongs to another protocol/seed")
    files = {p.name: core.sha256(p) for p in dest.iterdir() if p.is_file() and p.name != "completed.json"}
    if complete["files"] != files:
        raise ValueError("Completed outputs missing or changed")
    summary = json.loads((dest / "summary.json").read_text(encoding="utf-8"))
    if summary["seed"] != seed or summary["n_train"] != len(data.y) or summary["historical_oof_used"] is not False:
        raise ValueError("Saved result scope differs")
    if summary["protocol_sha256"] != complete["protocol_sha256"] or summary["fold_sha256"] != core.sha256(folder / f"folds_seed{seed}.csv"):
        raise ValueError("Saved result protocol/fold differs")
    rows = core.align_rows(core.read_rows(dest / "oof.csv", {"ID", "hardness", "fold"}), data.ids, dest)
    folds = outer_folds(data.y, seed)
    if not np.array_equal(core.integer_column(rows, "fold", dest), folds) or not np.array_equal(core.finite_column(rows, "hardness", dest), data.y):
        raise ValueError("Completed OOF labels/folds differ")
    baseline = core.finite_column(rows, core.BASELINE_NAME, dest)
    nested = {n + "_nested": core.finite_column(rows, n + "_nested", dest) for n in GRIDS}
    predictions = blend(baseline, nested)
    for name, pred in predictions.items():
        if not np.array_equal(pred, core.finite_column(rows, name, dest)):
            raise ValueError("Saved fixed blend differs")
    diagnostics, _ = previous.diagnose(replace(data, baseline=baseline, folds=folds), predictions)
    if diagnostics != summary["diagnostics"]:
        raise ValueError("Saved metrics differ")
    return summary


def aggregate(summaries):
    if [s["seed"] for s in summaries] != list(SEEDS):
        raise ValueError("All three prespecified seeds, in order, are required")
    per_seed = []
    for s in summaries:
        d = s["diagnostics"]
        m = d[PRIMARY]["overall"]
        per_seed.append({"seed": s["seed"], "reference_rmse": d[REFERENCE]["overall"]["rmse"],
            "control_rmse": d[CONTROL]["overall"]["rmse"], "candidate_rmse": m["rmse"],
            "gain_vs_reference": -m["delta_rmse_vs_current"], "gain_vs_control": -m["delta_rmse_vs_tuned_control"],
            "folds_improved_vs_reference": m["folds_improved_vs_current"],
            "folds_improved_vs_control": m["folds_improved_vs_tuned_control"]})
    mean_gain = float(np.mean([r["gain_vs_reference"] for r in per_seed]))
    supported = all(r["gain_vs_reference"] > 0 and r["gain_vs_control"] > 0 for r in per_seed)
    return {"seeds": list(SEEDS), "per_seed": per_seed,
        "mean_gain_vs_reference": mean_gain, "mean_gain_vs_control": float(np.mean([r["gain_vs_control"] for r in per_seed])),
        "all_seeds_improve_both": supported, "meets_priority_rule": supported and mean_gain >= .10,
        "conclusion": "direction_supported" if supported else "mixed_or_not_supported",
        "aggregation": "mean of per-seed OOF RMSE differences; no averaging predictions, no independent-sample inference",
        "historical_seeds_in_aggregate": False, "independent_holdout": False}


def summarize(folder, data):
    validate_plan(folder, data)
    summaries = [read_completed(folder, seed, data) for seed in SEEDS]
    result = aggregate(summaries)
    result["protocol_sha256"] = core.sha256(folder / "protocol.json")
    write_json(folder / "confirmation_summary.json", result)
    (folder / "comparison.csv").write_text(core.csv_text(result["per_seed"]), encoding="utf-8", newline="")
    lines = ["# 국소 구조 — 세 추가 분할 확인", "", "| seed | 기준 | 규제 대조군 | 후보 | 기준 대비 개선 | 대조군 대비 개선 |",
        "|---|---:|---:|---:|---:|---:|"]
    for r in result["per_seed"]:
        lines.append(f"| {r['seed']} | {r['reference_rmse']:.6f} | {r['control_rmse']:.6f} | {r['candidate_rmse']:.6f} | {r['gain_vs_reference']:+.6f} | {r['gain_vs_control']:+.6f} |")
    lines += ["", f"기준 대비 평균 개선: {result['mean_gain_vs_reference']:+.6f}",
        f"세 seed 모두 두 비교 기준 개선: {result['all_seeds_improve_both']}",
        f"동시에 평균 개선0.10 이상인 우선 기준 충족: {result['meets_priority_rule']}",
        "", "세 seed를 모두 포함했다. seed42/2026은 과거 탐색 결과로 분리하며 이번 평균에 넣지 않는다.",
        "같은500장을 재분할한 민감도 확인이다. 독립 holdout·독립 표본1500개·통계적 유의성으로 해석하지 않는다.",
        "새 결과에 맞춰 특징·가중치·seed를 교체하지 않는다. 세 seed 예측을 평균한 다른 모델의 점수도 아니다.", ""]
    (folder / "report.md").write_text("\n".join(lines), encoding="utf-8", newline="")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Report: {folder / 'report.md'}")
    return result


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=ROOT / "reports/local_structure_confirmation")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--seed", type=int, choices=SEEDS, help="Optional one of the fixed seeds; default runs all three")
    modes = p.add_mutually_exclusive_group()
    for flag in ("prepare", "validate-only", "summarize"):
        modes.add_argument("--" + flag, action="store_true")
    args = p.parse_args(argv)
    if args.threads < 1 or (args.seed is not None and (args.prepare or args.validate_only or args.summarize)):
        p.error("Positive threads required; --seed only applies to fitting")
    return args


def main(argv=None):
    args = parse_args(argv)
    data = load_features()
    if args.prepare:
        prepare(args.output_dir, data)
        print(f"Prepared seeds={list(SEEDS)}; estimators_fitted=0; folds/protocol saved to {args.output_dir}")
        return 0
    validate_plan(args.output_dir, data)
    if args.validate_only:
        print(json.dumps({"status": "frozen_protocol_and_caches_valid", "seeds": list(SEEDS), "rows": len(data.y),
            "feature_dimensions": {k: v.shape[1] for k, v in data.features.items()}, "estimators_fitted": 0, "output_written": False}))
        return 0
    if args.summarize:
        summarize(args.output_dir, data)
        return 0
    selected = SEEDS if args.seed is None else (args.seed,)
    # Check all destinations before spending time on any fitting.
    for seed in selected:
        local.protect_output(args.output_dir / f"seed{seed}", [args.output_dir / "protocol.json", *input_paths().values()])
    for seed in selected:
        run_seed(args.output_dir, data, seed, args.threads)
    if all((args.output_dir / f"seed{s}/completed.json").exists() for s in SEEDS):
        summarize(args.output_dir, data)
    else:
        print("Waiting for the remaining prespecified seeds; no aggregate conclusion yet.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError) as exc:
        print(f"Confirmation stopped: {exc}", file=sys.stderr)
        raise SystemExit(2)
