"""Audit saved local-structure outputs; no feature extraction or model fitting."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
from unittest import mock

import numpy as np
import evaluate_local_structure as evaluate
from review_dinov2_spatial import verify_hashes, rows

core = evaluate.core
OUT = core.ROOT / "reports/local_structure"


def audit(seed):
    args = evaluate.parse_args(["--seed", str(seed)])
    folder = args.output_dir
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    hashes = verify_hashes(summary["input_files"])
    data, refs = evaluate.load_prior(args)
    data = evaluate.attach_features(data, args.features)
    assert data.metadata == summary["feature_metadata"]
    assert data.provenance == summary["input_files"]
    assert summary["seed"] == seed and summary["n_train"] == 500
    assert summary["alpha_candidates"] == list(evaluate.ALPHAS)
    assert summary["feature_dimensions"] == {evaluate.RAW: 125, evaluate.AUGMENTED: 185}
    assert summary["protocol"] == evaluate.extract.PROTOCOL
    assert summary["primary_model"] == evaluate.PRIMARY
    assert summary["comparison_reference"] == evaluate.REFERENCE
    assert summary["tuning_control"] == evaluate.CONTROL
    assert summary["representation_selection"] is False
    assert (summary["outer_splits"], summary["inner_splits"], summary["estimator_fit_count"]) == (5, 3, 105)
    assert summary["fixed_weights"] == {"hand": 0.25, "original_resnet": 0.25, "dino_d4": 0.5}
    raw = core.align_rows(core.read_rows(folder / "oof.csv", {"ID", "hardness", "fold"}), data.ids, folder)
    for column, expected in (("hardness", data.y), ("fold", data.folds), (core.BASELINE_NAME, data.baseline)):
        np.testing.assert_array_equal(core.finite_column(raw, column, folder), expected)
    nested = {n + "_nested": core.finite_column(raw, n + "_nested", folder) for n in evaluate.REPRESENTATIONS}
    anchor = core.finite_column(raw, evaluate.ANCHOR, folder)
    difference = float(np.max(np.abs(anchor - data.baseline)))
    assert difference <= 1e-7
    check = {"status": "matched", "max_absolute_prediction_difference": difference, "atol": 1e-7,
             "fit_count": 5, "alpha": 100.0, "target_source": "train.csv"}
    assert check == summary["hand_anchor"]
    predictions = evaluate.add_fixed_blends(data, nested, refs, anchor)
    for name, pred in predictions.items():
        np.testing.assert_array_equal(pred, core.finite_column(raw, name, folder))
    selections = summary["selections"]
    assert len(selections) == 5 and {r["outer_fold"] for r in selections} == set(range(5))
    fold_details, search = [], []
    for r in selections:
        fold, candidates = r["outer_fold"], r["candidates"]
        assert r["outer_train_n"] == 400 and r["outer_valid_n"] == 100
        assert r["inner_seed"] == seed + fold and r["inner_validation_sizes"] == [134, 133, 133]
        assert "joint" not in r and len(candidates) == 6
        assert {(c["representation"], c["alpha"]) for c in candidates} == {(n, a) for n in evaluate.REPRESENTATIONS for a in evaluate.ALPHAS}
        for name in evaluate.REPRESENTATIONS:
            assert r["per_representation"][name] == min((c for c in candidates if c["representation"] == name), key=lambda c: (c["inner_mse"], -c["alpha"]))
        for c in candidates:
            assert len(c["inner_fold_mse"]) == 3
            assert np.isfinite([c["inner_mse"], *c["inner_fold_mse"]]).all()
            np.testing.assert_allclose(c["inner_mse"], np.average(c["inner_fold_mse"], weights=r["inner_validation_sizes"]), rtol=0, atol=1e-9)
            search.append({"outer_fold": fold, "inner_seed": r["inner_seed"], "representation": c["representation"],
                "alpha": c["alpha"], "inner_mse": c["inner_mse"], **{f"inner_fold{j}_mse": v for j, v in enumerate(c["inner_fold_mse"])},
                "selected_for_representation": c == r["per_representation"][c["representation"]]})
        valid = data.folds == fold
        score = lambda p: core.metrics(data.y[valid], p[valid])["rmse"]
        fold_details.append({"fold": fold,
            "original_alpha": r["per_representation"][evaluate.RAW]["alpha"],
            "augmented_alpha": r["per_representation"][evaluate.AUGMENTED]["alpha"],
            "reference_rmse": score(refs[evaluate.REFERENCE]), "control_rmse": score(predictions[evaluate.CONTROL]),
            "new_blend_rmse": score(predictions[evaluate.PRIMARY]),
            "delta_vs_current": score(predictions[evaluate.PRIMARY]) - score(refs[evaluate.REFERENCE]),
            "delta_vs_control": score(predictions[evaluate.PRIMARY]) - score(predictions[evaluate.CONTROL])})
    diagnostics, tables = evaluate.diagnose(data, predictions)
    assert diagnostics == summary["diagnostics"]
    for name, records in {**tables, "inner_search": search}.items():
        assert rows(folder / f"{name}.csv") == list(csv.DictReader(core.csv_text(records).splitlines())), name
    assert (folder / "report.md").read_text(encoding="utf-8") == evaluate.build_markdown(summary)
    old_error, new_error = [predictions[n] - data.y for n in (evaluate.REFERENCE, evaluate.PRIMARY)]
    other_error = (0.25 * refs[evaluate.previous.RAW_REFERENCE] + 0.5 * refs["reference_dino_d4"]) / 0.75 - data.y
    new_hand_error = nested[evaluate.AUGMENTED + "_nested"] - data.y
    decompositions = {}
    for label, hand, blend in (("current", data.baseline, refs[evaluate.REFERENCE]),
                              ("tuned_control", nested[evaluate.RAW + "_nested"], predictions[evaluate.CONTROL])):
        old_hand_error = hand - data.y
        own = float(0.25 ** 2 * np.mean(new_hand_error ** 2 - old_hand_error ** 2))
        cross = float(2 * 0.25 * 0.75 * np.mean((new_hand_error - old_hand_error) * other_error))
        delta = float(np.mean(new_error ** 2 - (blend - data.y) ** 2))
        np.testing.assert_allclose(own + cross, delta, rtol=0, atol=1e-10)
        decompositions[label] = {"weighted_component_mse_change": own, "cross_error_change": cross, "total_blend_mse_change": delta}
    top20 = lambda error: {str(data.ids[i]) for i in np.argsort(-error ** 2, kind="stable")[:20]}
    old_top, new_top = top20(old_error), top20(new_error)
    changes = new_error ** 2 - old_error ** 2
    bins = core.rank_groups(data.y, 10)
    tails = (bins == 0) | (bins == 9)
    correlations = {name: core.correlation(pred - data.y, other_error) for name, pred in
                    (("original", nested[evaluate.RAW + "_nested"]), ("augmented", nested[evaluate.AUGMENTED + "_nested"]))}
    return {"seed": seed, "verified_input_and_source_hashes": hashes, "verified_oof_rows": len(data.ids),
        "hand_anchor": check, "original_reextraction_check": data.metadata["local_structure"]["original_reextraction_check"],
        "extraction_seconds": data.metadata["local_structure"]["elapsed_seconds"],
        "user_evaluation_seconds": summary["elapsed_seconds"], "fold_details": fold_details, "diagnostics": diagnostics,
        "mse_decomposition": decompositions, "hand_other_residual_correlations": correlations,
        "current_worst20": sorted(old_top), "new_worst20": sorted(new_top), "top20_overlap": len(old_top & new_top),
        "improved_images": int(np.sum(changes < 0)), "worsened_images": int(np.sum(changes > 0)),
        "unchanged_images": int(np.sum(changes == 0)), "net_sse_change": float(changes.sum()),
        "tails20_sse_change": float(changes[tails].sum()), "middle80_sse_change": float(changes[~tails].sum()),
        "per_image_blend_change": tables["replacement_errors"]}


def main():
    original = [p for p in OUT.rglob("*") if p.is_file() and p.name != "review_results.json"]
    before = {str(p): core.sha256(p) for p in original}
    with mock.patch.object(core, "make_estimator", side_effect=AssertionError("No fitting in review")), \
         mock.patch.object(core, "nested_predictions", side_effect=AssertionError("No fitting in review")), \
         mock.patch.object(evaluate, "nested_predictions", side_effect=AssertionError("No fitting in review")), \
         mock.patch.object(evaluate, "fixed_anchor", side_effect=AssertionError("No refitting anchor")), \
         mock.patch.object(evaluate.extract, "local_features", side_effect=AssertionError("No new features")), \
         mock.patch.object(evaluate.extract, "legacy_function", side_effect=AssertionError("No old feature extraction")):
        seeds = {str(s): audit(s) for s in (42, 2026)}
    assert "torch" not in sys.modules
    assert before == {str(p): core.sha256(p) for p in original}
    common = set(seeds["42"]["current_worst20"]) & set(seeds["2026"]["current_worst20"])
    changes = {s: {r["ID"]: r["delta_squared_error_vs_current"] for r in v["per_image_blend_change"]} for s, v in seeds.items()}
    repeated = {"previous_common_worst20": sorted(common),
        "improved_in_both": sorted(i for i in common if changes["42"][i] < 0 and changes["2026"][i] < 0),
        "remain_in_both_worst20": sorted(common & set(seeds["42"]["new_worst20"]) & set(seeds["2026"]["new_worst20"]))}
    gains = [-v["diagnostics"][evaluate.PRIMARY]["overall"]["delta_rmse_vs_current"] for v in seeds.values()]
    control_gains = [-v["diagnostics"][evaluate.PRIMARY]["overall"]["delta_rmse_vs_tuned_control"] for v in seeds.values()]
    decision = {"mean_rmse_gain_vs_current": float(np.mean(gains)), "mean_rmse_gain_vs_control": float(np.mean(control_gains)),
        "beats_both_references_in_both_seeds": all(g > 0 for g in gains + control_gains),
        "predeclared_priority_gain": 0.10, "meets_priority_gain": np.mean(gains) >= 0.10,
        "status": "Small-improvement candidate; keep protocol frozen and established reference for further comparisons."}
    decision["meets_priority_gain"] = bool(decision["meets_priority_gain"])
    result = {"scope": "Saved arithmetic/cache/source audit only; no fitting, feature extraction, torch, GPU, test, downloads, or weight search.",
        "review_script_sha256": core.sha256(Path(__file__)), "original_files_preserved": before,
        "seeds": seeds, "repeated_failures": repeated, "decision": decision}
    (OUT / "review_results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    for s, r in seeds.items():
        print(json.dumps({k: r[k] for k in ("seed", "verified_input_and_source_hashes", "hand_anchor", "original_reextraction_check",
            "extraction_seconds", "user_evaluation_seconds", "fold_details", "mse_decomposition", "hand_other_residual_correlations",
            "top20_overlap", "improved_images", "worsened_images", "net_sse_change", "tails20_sse_change", "middle80_sse_change")}, indent=2))
    print(json.dumps({"original_files_preserved": len(before), "repeated_failures": repeated, "decision": decision}, indent=2))


if __name__ == "__main__":
    main()
