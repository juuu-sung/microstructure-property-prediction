"""Review completed confirmation outputs without fitting or extracting features."""
from __future__ import annotations

import csv
from dataclasses import replace
import json
from pathlib import Path
import sys
from unittest import mock

import evaluate_local_structure_confirmation as e

core, np = e.core, e.np
OUT = e.ROOT / "reports/local_structure_confirmation"


def csv_rows(path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def audit_seed(data, seed):
    summary = e.read_completed(OUT, seed, data)
    folder = OUT / f"seed{seed}"
    assert summary["fit_count"] == 250
    selections, search, choices = summary["selections"], [], []
    assert len(selections) == 5 and {r["outer_fold"] for r in selections} == set(range(5))
    for r in selections:
        fold = r["outer_fold"]
        assert (r["outer_train_n"], r["outer_valid_n"], r["inner_seed"]) == (400, 100, seed+fold)
        assert r["inner_validation_sizes"] == [134, 133, 133] and "joint" not in r
        assert len(r["candidates"]) == 15
        assert {(c["representation"], c["alpha"]) for c in r["candidates"]} == {(n, a) for n, aa in e.GRIDS.items() for a in aa}
        for name in e.GRIDS:
            winner = min((c for c in r["candidates"] if c["representation"] == name), key=lambda c: (c["inner_mse"], -c["alpha"]))
            assert winner == r["per_representation"][name]
        choices.append({"fold": fold, **{n: r["per_representation"][n]["alpha"] for n in e.GRIDS}})
        for c in r["candidates"]:
            assert len(c["inner_fold_mse"]) == 3 and np.isfinite([c["inner_mse"], *c["inner_fold_mse"]]).all()
            np.testing.assert_allclose(c["inner_mse"], np.average(c["inner_fold_mse"], weights=r["inner_validation_sizes"]), atol=1e-9, rtol=0)
            search.append({"outer_fold": fold, "inner_seed": r["inner_seed"], "representation": c["representation"],
                "alpha": c["alpha"], "inner_mse": c["inner_mse"], **{f"inner_fold{j}_mse": v for j, v in enumerate(c["inner_fold_mse"])},
                "selected": c == r["per_representation"][c["representation"]]})
    raw = core.align_rows(core.read_rows(folder / "oof.csv", {"ID", "hardness", "fold"}), data.ids, folder)
    baseline = core.finite_column(raw, core.BASELINE_NAME, folder)
    nested = {n+"_nested": core.finite_column(raw, n+"_nested", folder) for n in e.GRIDS}
    predictions = e.blend(baseline, nested)
    current = replace(data, baseline=baseline, folds=e.outer_folds(data.y, seed))
    diagnostics, tables = e.previous.diagnose(current, predictions)
    for name, records in {**tables, "inner_search": search}.items():
        assert csv_rows(folder / f"{name}.csv") == list(csv.DictReader(core.csv_text(records).splitlines())), name
    assert (folder / "report.md").read_text(encoding="utf-8") == e.report_text(summary)
    per_image = tables["replacement_errors"]
    change = predictions[e.PRIMARY] - data.y
    ref_error = predictions[e.REFERENCE] - data.y
    delta_sse = change**2 - ref_error**2
    bins = core.rank_groups(data.y, 10)
    tail = (bins == 0) | (bins == 9)
    top = lambda p: sorted(str(data.ids[i]) for i in np.argsort(-(p-data.y)**2, kind="stable")[:20])
    groups = {}
    for name in (e.REFERENCE, e.CONTROL, e.PRIMARY):
        d = diagnostics[name]
        groups[name] = {"tails20": d["target_regions"]["tails20"]["rmse"],
            "middle80": d["target_regions"]["middle80"]["rmse"], "noise_q4": d["noise_quartiles"][-1]["rmse"],
            "orientation_q4": d["orientation_quartiles"]["4"]["rmse"]}
    return summary, {"seed": seed, "alpha_choices": choices,
        "component_rmse": {n: d["overall"]["rmse"] for n, d in diagnostics.items()}, "groups": groups,
        "fold_deltas": [r for r in tables["fold_metrics"] if r["model"] == e.PRIMARY],
        "improved_images": int(np.sum(delta_sse < 0)), "worsened_images": int(np.sum(delta_sse > 0)),
        "net_sse_change": float(delta_sse.sum()), "tails20_sse_change": float(delta_sse[tail].sum()),
        "middle80_sse_change": float(delta_sse[~tail].sum()),
        "reference_worst20": top(predictions[e.REFERENCE]), "candidate_worst20": top(predictions[e.PRIMARY]),
        "top20_overlap": diagnostics[e.PRIMARY]["top20_overlap"], "per_image": per_image,
        "evaluation_seconds": summary["elapsed_seconds"]}


def main():
    original = [p for p in OUT.rglob("*") if p.is_file() and p.name != "review_results.json"]
    historical = e.ROOT / "reports/local_structure/review_results.json"
    before = {str(p): core.sha256(p) for p in [*original, historical]}
    with mock.patch.object(core, "make_estimator", side_effect=AssertionError("No fitting in review")), \
         mock.patch.object(core, "nested_predictions", side_effect=AssertionError("No fitting in review")), \
         mock.patch.object(e, "predict_components", side_effect=AssertionError("No fitting in review")), \
         mock.patch.object(e.local.Image, "open", side_effect=AssertionError("No image reads")), \
         mock.patch.object(e.local, "local_features", side_effect=AssertionError("No new features")):
        data = e.load_features()
        plan = e.validate_plan(OUT, data)
        checked = [audit_seed(data, seed) for seed in e.SEEDS]
        aggregate = e.aggregate([s for s, _ in checked])
        aggregate["protocol_sha256"] = core.sha256(OUT / "protocol.json")
        assert aggregate == json.loads((OUT / "confirmation_summary.json").read_text(encoding="utf-8"))
        assert csv_rows(OUT / "comparison.csv") == list(csv.DictReader(core.csv_text(aggregate["per_seed"]).splitlines()))
    assert "torch" not in sys.modules
    assert before == {str(p): core.sha256(Path(p)) for p in before}
    details = [r for _, r in checked]
    common_old = set.intersection(*(set(r["reference_worst20"]) for r in details))
    common_new = set.intersection(*(set(r["candidate_worst20"]) for r in details))
    prior = json.loads(historical.read_text(encoding="utf-8"))
    prior18 = set(prior["repeated_failures"]["previous_common_worst20"])
    maps = [{r["ID"]: r["delta_squared_error_vs_current"] for r in d["per_image"]} for d in details]
    repeated = {"new_three_seeds_common_reference_worst20": sorted(common_old),
        "new_three_seeds_common_candidate_worst20": sorted(common_new),
        "common_reference_cases_improved_in_all_three": sorted(i for i in common_old if all(m[i] < 0 for m in maps)),
        "common_reference_cases_remaining_in_all_three": sorted(common_old & common_new),
        "historical_18_improved_in_all_three": sorted(i for i in prior18 if all(m[i] < 0 for m in maps)),
        "historical_18_remaining_in_all_three": sorted(prior18 & common_new)}
    result = {"scope": "Saved results only; no fitting, image reads, inference, torch, GPU, test or tuning.",
        "review_script_sha256": core.sha256(Path(__file__)), "preserved_files": before,
        "verified_frozen_input_count": len(plan["frozen"]["input_files"]),
        "verified_frozen_source_count": len(plan["frozen"]["sources"]), "aggregate": aggregate,
        "details": details, "repeated_failures": repeated}
    e.write_json(OUT / "review_results.json", result)
    compact = {**result, "details": [{k: v for k, v in r.items() if k not in ("per_image", "fold_deltas")} for r in details]}
    compact["preserved_files"] = len(before)
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
