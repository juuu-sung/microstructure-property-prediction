"""User-run fixed TabPFN-v2 50:50 confirmation on three additional splits.

Built with PriorLabs-TabPFN. No new model, features, weights or tuning grid.
prepare/validate only inspect caches; evaluate is the user-run GPU command.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

if __package__:
    from . import evaluate_tabpfn_v2 as t
    from . import evaluate_local_structure_confirmation as baseline
else:
    import evaluate_tabpfn_v2 as t
    import evaluate_local_structure_confirmation as baseline

OUT = t.ROOT/"reports/tabpfn_v2_confirmation"
BASE = t.ROOT/"reports/local_structure_confirmation"
SEEDS = (31415, 27182, 16180)
PRIMARY = t.SECONDARY  # The already fixed blend, not a newly selected model.
FILES = ("oof.csv", "summary.json", "diagnostics.csv", "per_image_errors.csv", "report.md")
PROTOCOL = {
    "version": 1, "seeds": list(SEEDS), "primary": PRIMARY,
    "weights": {"current": .5, "tabpfn_joint_nested": .5},
    "current": "185-feature Ridge25% + original ResNet25% + DINO D4 Ridge50%; audited same-fold saved predictions",
    "baseline_source_column": baseline.PRIMARY,
    "baseline_reuse": "only the matching additional-seed outer OOF; no seed42/2026 predictions in the blend",
    "tabpfn": t.PROTOCOL,
    "protocol_inheritance": "TabPFN settings/inner selection unchanged; this outer seed list and blend primary supersede exploratory reporting fields",
    "outer": "reuse exact archived5-fold target-rank-stratified partitions for each additional seed",
    "inner": "unchanged3-fold pooled MSE selects185/285; tie185; seed+fold; no outer-score selection",
    "context_fits_per_seed": 40, "context_fits_total": 120,
    "neural_gradient_training": False, "image_extraction": False, "test_access": False,
    "historical_results_in_aggregate": False,
    "decision": "direction supported only if all three blend gains>0; priority additionally mean gain>=0.10; diagnostic review before adoption",
    "priority_mean_gain": .10,
    "stop": "complete all three; no new seeds, blend weights, features or hyperparameters based on these results",
    "interpretation": "same500 studied images; these seeds previously used for local-structure validation; not independent holdout or significance",
    "diagnostics": "tails20/middle80,noise quartiles,folds,current-worst20,fixed17 common historical failures; no removal or tuning by groups",
}


def same(a, b, label):
    if not np.array_equal(a, b):
        raise ValueError(label+" differs")


def baseline_reference(data, rows, seed):
    """Bind the adopted augmented ensemble, never the older125-feature comparator."""
    path = BASE/f"seed{seed}/oof.csv"
    rows = t.core.align_rows(rows, data.ids, path)
    same(t.core.finite_column(rows, "hardness", path), data.y, "Reference labels")
    folds = baseline.outer_folds(data.y, seed)
    same(t.core.integer_column(rows, "fold", path), folds, "Reference folds")
    components = {n: t.core.finite_column(rows, n+"_nested", path)
                  for n in (baseline.AUGMENTED, baseline.RESNET, baseline.DINO)}
    reconstructed = .25*components[baseline.AUGMENTED]+(.25*components[baseline.RESNET]+.5*components[baseline.DINO])
    current = t.core.finite_column(rows, baseline.PRIMARY, path)
    same(current, reconstructed, "Reference25:25:50 blend")
    return {"current": current, "folds": folds}


def load_inputs():
    data, features, _, source_plan = t.validate(t.OUT)
    base_data = baseline.load_features()
    base_plan = baseline.validate_plan(BASE, base_data)
    same(data.ids, base_data.ids, "Baseline IDs")
    same(data.y, base_data.y, "Baseline targets")
    same(features["existing185"], base_data.features[baseline.AUGMENTED], "Baseline185 features")
    same(data.noise, base_data.noise, "Noise audit")
    refs, files = {}, {}
    for seed in SEEDS:
        summary = baseline.read_completed(BASE, seed, base_data)
        path = BASE/f"seed{seed}/oof.csv"
        refs[seed] = baseline_reference(data, t.core.read_rows(path, {"ID", "hardness", "fold"}), seed)
        same(t.core.metrics(data.y, refs[seed]["current"])["rmse"],
             summary["diagnostics"][baseline.PRIMARY]["overall"]["rmse"], "Baseline RMSE")
        # Also validate the saved inner Ridge selections, without refitting.
        if len(summary["selections"]) != 5:
            raise ValueError("Incomplete baseline selections")
        for r in summary["selections"]:
            for n, alphas in baseline.GRIDS.items():
                candidates = [c for c in r["candidates"] if c["representation"] == n]
                if sorted(c["alpha"] for c in candidates) != sorted(alphas):
                    raise ValueError("Baseline alpha grid differs")
                if r["per_representation"][n] != min(candidates, key=lambda c: (c["inner_mse"], -c["alpha"])):
                    raise ValueError("Baseline selection differs")
        names = [BASE/f"folds_seed{seed}.csv", BASE/f"seed{seed}/completed.json", path,
                 BASE/f"seed{seed}/summary.json"]
        files[str(seed)] = [t.file_ref(p) for p in names]
    review = t.read_json(t.OUT/"review_results.json")
    common_ids = review["baseline_common_worst20_ids"]
    if len(common_ids) != 17 or len(set(common_ids)) != 17 or not set(common_ids).issubset(set(data.ids)):
        raise ValueError("Expected17 fixed historical failures")
    frozen = {"protocol": PROTOCOL, "source": t.file_ref(__file__),
              "baseline_sources": baseline.source_references(),
              "source_plan": t.file_ref(t.OUT/"protocol.json"), "source_plan_payload": t.digest(source_plan["frozen"]),
              "baseline_plan": t.file_ref(BASE/"protocol.json"), "baseline_plan_payload": t.digest(base_plan["frozen"]),
              "historical_review": t.file_ref(t.OUT/"review_results.json"), "common_failure_ids": common_ids,
              "reference_files": files, "feature_arrays": {n: t.common.sha256_array(x) for n, x in features.items()},
              "reference_arrays": {str(s): {n: t.common.sha256_array(x) for n, x in r.items()} for s, r in refs.items()}}
    return data, features, refs, frozen


def fold_rows(data, ref):
    return [{"ID": str(i), "fold": int(f)} for i, f in zip(data.ids, ref["folds"], strict=True)]


def prepare(folder=OUT):
    data, _, refs, frozen = load_inputs()
    if (folder/"protocol.json").exists():
        validate(folder)
        print("Existing confirmation preparation matches; unchanged")
        return
    if folder.exists() and any(folder.iterdir()):
        raise ValueError("Nonempty unprepared destination; refusing overwrite")
    folder.mkdir(parents=True, exist_ok=True)
    for seed, ref in refs.items():
        t.write_csv(folder/f"folds_seed{seed}.csv", fold_rows(data, ref))
    t.write_json(folder/"protocol.json", {"frozen": frozen, "created_unix": time.time(), "competition_fits_during_preparation": 0})
    print(f"Prepared seeds={list(SEEDS)}; no fitting or model inference; {folder}")


def validate(folder=OUT):
    data, features, refs, frozen = load_inputs()
    plan = t.read_json(folder/"protocol.json")
    if plan["frozen"] != frozen or plan["competition_fits_during_preparation"] != 0:
        raise ValueError("Frozen confirmation inputs/settings/sources changed")
    for seed, ref in refs.items():
        path = folder/f"folds_seed{seed}.csv"
        r = t.core.read_rows(path, {"ID", "fold"})
        if r != [{k: str(v) for k, v in row.items()} for row in fold_rows(data, ref)]:
            raise ValueError("Confirmation fold file changed")
    return data, features, refs, plan


def build_results(data, ref, predictions, choices, binding, seed, failure_ids):
    pred = {"current": ref["current"], **{"tabpfn_"+n: predictions[n] for n in t.REPS},
            t.PRIMARY: predictions[t.PRIMARY], PRIMARY: .5*(ref["current"]+predictions[t.PRIMARY])}
    if any(p.shape != data.y.shape or not np.isfinite(p).all() for p in pred.values()):
        raise ValueError("Invalid/incomplete OOF predictions")
    if len(choices) != 5 or [c["fold"] for c in choices] != list(range(5)):
        raise ValueError("Incomplete inner selection records")
    for c in choices:
        if set(c["inner_mse"]) != set(t.REPS) or not np.isfinite(list(c["inner_mse"].values())).all():
            raise ValueError("Invalid inner selection scores")
        expected = min(t.REPS, key=lambda n: (c["inner_mse"][n], t.REPS.index(n)))
        if c["representation"] != expected:
            raise ValueError("Incorrect pooled inner selection")
        vi = ref["folds"] == c["fold"]
        same(pred[t.PRIMARY][vi], predictions[expected][vi], "Selected outer predictions")
    rmse = t.core.metrics(data.y, ref["current"])["rmse"]
    metrics = {n: {**t.core.metrics(data.y, p), "gain_vs_current": rmse-t.core.metrics(data.y, p)["rmse"]} for n, p in pred.items()}
    bins, noise = t.core.rank_groups(data.y, 10), t.core.rank_groups(data.noise, 4)
    worst = np.argsort((ref["current"]-data.y)**2, kind="stable")[-20:]
    masks = {"tail20": (bins == 0)|(bins == 9), "middle80": (bins > 0)&(bins < 9),
             "current_worst20": np.isin(np.arange(len(data.y)), worst), "historical_common17": np.isin(data.ids, failure_ids)}
    masks.update({f"fold{f}": ref["folds"] == f for f in range(5)})
    masks.update({f"noiseQ{q+1}": noise == q for q in range(4)})
    if int(masks["historical_common17"].sum()) != 17:
        raise ValueError("Historical failure IDs not aligned")
    groups, diag = {}, []
    for n, p in pred.items():
        groups[n] = {}
        for g, mask in masks.items():
            m = t.core.metrics(data.y[mask], p[mask])
            groups[n][g] = m["rmse"]
            diag.append({"model": n, "group": g, **m})
    summary = {"seed": seed, "binding": binding, "primary": PRIMARY, "metrics": metrics, "groups": groups,
               "selection": choices, "cells": 40, "baseline_reused_same_fold": True, "test_accessed": False,
               "no_gradient_training": True, "independent_holdout": False,
               "folds_improved": sum(groups[PRIMARY][f"fold{f}"] < groups["current"][f"fold{f}"] for f in range(5)),
               "attribution": "Built with PriorLabs-TabPFN"}
    oof = [{"ID": str(i), "hardness": float(data.y[j]), "fold": int(ref["folds"][j]),
            **{n: float(p[j]) for n, p in pred.items()}} for j, i in enumerate(data.ids)]
    error = [{"ID": str(data.ids[i]), "hardness": float(data.y[i]),
              "current_error": float(pred["current"][i]-data.y[i]), "tabpfn_error": float(pred[t.PRIMARY][i]-data.y[i]),
              "blend_error": float(pred[PRIMARY][i]-data.y[i]),
              "gain_squared_error": float((pred["current"][i]-data.y[i])**2-(pred[PRIMARY][i]-data.y[i])**2)}
             for i in np.argsort(-(pred["current"]-data.y)**2, kind="stable")]
    return summary, oof, diag, error


def report_text(s):
    lines = [f"# Fixed TabPFN-v2 confirmation - seed{s['seed']}", "", "Built with PriorLabs-TabPFN.", "",
             "| Model | OOF RMSE | Gain vs current |", "|---|---:|---:|"]
    lines += [f"| {n} | {m['rmse']:.6f} | {m['gain_vs_current']:+.6f} |" for n, m in s["metrics"].items()]
    lines += ["", f"Primary: {PRIMARY}; unchanged current50% + internally selected TabPFN50%.",
              "Current baseline reuses audited predictions fitted on these exact outer folds.",
              "40 TabPFN context-fit/predict cells; no gradient training or test access.",
              "Complete all three fixed seeds before deciding; same500, not independent holdout.",
              "No historical seed42/2026 scores in the confirmation aggregate.", ""]
    return "\n".join(lines)


def expected_files():
    return set(FILES)|{f"fold{f}_{r}_{stage}/completed.json" for f in range(5) for r in t.REPS
                       for stage in ("inner0", "inner1", "inner2", "outer")}


def read_completed(folder, data, ref, plan, seed, runtime=None):
    target = folder/f"seed{seed}"
    s = t.read_json(target/"summary.json")
    binding = s["binding"]
    if binding["plan"] != t.digest(plan["frozen"]) or binding["seed"] != seed or (runtime is not None and binding["runtime"] != runtime):
        raise ValueError("Completed experiment/runtime differs")
    done = t.read_json(target/"completed_evaluation.json")
    if set(done["files"]) != expected_files() or not t.verify_completed(target, binding):
        raise ValueError("Incomplete confirmation outputs")
    path = target/"oof.csv"
    rows = t.core.align_rows(t.core.read_rows(path, {"ID", "hardness", "fold"}), data.ids, path)
    same(t.core.finite_column(rows, "hardness", path), data.y, "OOF labels")
    same(t.core.integer_column(rows, "fold", path), ref["folds"], "OOF folds")
    same(t.core.finite_column(rows, "current", path), ref["current"], "OOF current")
    predictions = {n: t.core.finite_column(rows, "tabpfn_"+n, path) for n in t.REPS}
    predictions[t.PRIMARY] = t.core.finite_column(rows, t.PRIMARY, path)
    rebuilt, _, _, _ = build_results(data, ref, predictions, s["selection"], binding, seed, plan["frozen"]["common_failure_ids"])
    if s != rebuilt:
        raise ValueError("Saved summary does not match OOF")
    same(t.core.finite_column(rows, PRIMARY, path), .5*(ref["current"]+predictions[t.PRIMARY]), "OOF50:50 blend")
    return s


def evaluate_seed(folder, device, seed):
    if seed not in SEEDS:
        raise ValueError("Only the fixed three seeds are allowed")
    data, features, refs, plan = validate(folder)
    predictor = t.Predictor(t.verify_weights(t.OUT), device)
    target = folder/f"seed{seed}"
    if (target/"completed_evaluation.json").exists():
        read_completed(folder, data, refs[seed], plan, seed, predictor.runtime)
        print(f"Completed seed{seed} verified; no refitting")
        return
    target.mkdir(parents=True, exist_ok=True)
    binding = {"plan": t.digest(plan["frozen"]), "seed": seed, "runtime": predictor.runtime}
    pred = {n: np.full(len(data.y), np.nan) for n in (*t.REPS, t.PRIMARY)}
    choices = []
    for fold in range(5):
        vi, outputs, choice = t.evaluate_fold(features, data.y, refs[seed]["folds"], seed, fold, target, binding, predictor)
        for n, p in outputs.items():
            pred[n][vi] = p
        choices.append(choice)
    validate(folder)  # Do not publish results if source inputs changed mid-run.
    summary, oof, diag, errors = build_results(data, refs[seed], pred, choices, binding, seed, plan["frozen"]["common_failure_ids"])
    for name, values in (("oof.csv", oof), ("diagnostics.csv", diag), ("per_image_errors.csv", errors)):
        t.write_csv(target/name, values)
    t.write_json(target/"summary.json", summary)
    (target/"report.md").write_text(report_text(summary), encoding="utf-8")
    hashes = {n: t.common.sha256_file(target/n) for n in sorted(expected_files())}
    t.write_json(target/"completed_evaluation.json", {"binding": binding, "files": hashes})
    m = summary["metrics"][PRIMARY]
    print(f"PRIMARY seed={seed}: RMSE={m['rmse']:.6f}; gain vs current={m['gain_vs_current']:+.6f}", flush=True)
    print("Report:", target/"report.md")


def aggregate(summaries):
    if [s["seed"] for s in summaries] != list(SEEDS) or any(s["primary"] != PRIMARY for s in summaries):
        raise ValueError("All three fixed seeds and the fixed blend primary are required")
    rows = [{"seed": s["seed"], "current_rmse": s["metrics"]["current"]["rmse"],
             "candidate_rmse": s["metrics"][PRIMARY]["rmse"], "gain_vs_current": s["metrics"][PRIMARY]["gain_vs_current"],
             "folds_improved": s["folds_improved"]} for s in summaries]
    mean = float(np.mean([r["gain_vs_current"] for r in rows]))
    supported = all(r["gain_vs_current"] > 0 for r in rows)
    return {"seeds": list(SEEDS), "primary": PRIMARY, "per_seed": rows, "mean_gain_vs_current": mean,
            "all_seeds_improve": supported, "meets_priority_rule": supported and mean >= PROTOCOL["priority_mean_gain"],
            "conclusion": "direction_supported" if supported else "mixed_or_not_supported",
            "additional_review_required": "tails/noise/common failures; numerical gates are not automatic adoption",
            "aggregation": "mean of three per-seed OOF RMSE differences; no averaging predictions or independent-sample inference",
            "historical_seeds_in_aggregate": False, "independent_holdout": False}


def compare(folder=OUT):
    data, _, refs, plan = validate(folder)
    result = aggregate([read_completed(folder, data, refs[s], plan, s) for s in SEEDS])
    result["protocol_sha256"] = t.common.sha256_file(folder/"protocol.json")
    t.write_json(folder/"confirmation_summary.json", result)
    t.write_csv(folder/"comparison.csv", result["per_seed"])
    lines = ["# Fixed TabPFN-v2 blend - three split confirmation", "", "Built with PriorLabs-TabPFN.", "",
             "| Seed | Current | Candidate50:50 | Gain |", "|---|---:|---:|---:|"]
    lines += [f"| {r['seed']} | {r['current_rmse']:.6f} | {r['candidate_rmse']:.6f} | {r['gain_vs_current']:+.6f} |" for r in result["per_seed"]]
    lines += ["", f"Mean gain: {result['mean_gain_vs_current']:+.6f}; all improve: {result['all_seeds_improve']}; priority>=0.10: {result['meets_priority_rule']}.",
              "Inspect tails/noise/common failures before adoption. Current submission is not changed.",
              "Same500; not independent holdout, not1500 independent samples; no seed42/2026 in this aggregate.", ""]
    (folder/"report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("Report:", folder/"report.md")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("prepare", "validate", "evaluate", "compare"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, choices=SEEDS)
    args = p.parse_args(argv)
    if args.seed is not None and args.command != "evaluate":
        p.error("--seed only applies to evaluate")
    if args.command == "prepare":
        prepare()
    elif args.command == "validate":
        validate()
        print("Confirmation protocol, caches and same-fold references validated; no fitting or inference")
    elif args.command == "compare":
        compare()
    else:
        for seed in SEEDS if args.seed is None else (args.seed,):
            evaluate_seed(OUT, args.device, seed)
        if all((OUT/f"seed{s}/completed_evaluation.json").exists() for s in SEEDS):
            compare()
        else:
            print("Remaining fixed seeds required before aggregate conclusion")


if __name__ == "__main__":
    main()
