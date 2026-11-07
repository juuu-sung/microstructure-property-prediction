"""Arithmetic/provenance audit of saved spatial results; no fitting or forward."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from unittest import mock

import numpy as np
import evaluate_dinov2_spatial as evaluate

core = evaluate.core
ROOT = core.ROOT
OUT = ROOT / "reports/dinov2_spatial"


def verify_hashes(value):
    count = 0
    if isinstance(value, dict):
        for pkey, hkey in (("path", "sha256"), ("summary_path", "summary_sha256"), ("metadata_path", "metadata_sha256")):
            if pkey in value and hkey in value:
                assert core.sha256(Path(value[pkey])) == value[hkey], value[pkey]
                count += 1
        count += sum(verify_hashes(v) for v in value.values())
    elif isinstance(value, list):
        count += sum(verify_hashes(v) for v in value)
    return count


def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def audit(seed):
    args = evaluate.parse_args(["--seed", str(seed)])
    folder = args.output_dir
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    hashes = verify_hashes(summary["input_files"])
    data, references, anchor = evaluate.load_prior(args)
    data = evaluate.attach_spatial(data, args.spatial_features, args.global_features)
    assert summary["feature_metadata"] == data.metadata
    assert summary["seed"] == seed and summary["n_train"] == 500
    assert summary["alpha_candidates"] == list(evaluate.ALPHAS)
    assert summary["feature_dimensions"] == {evaluate.GLOBAL: 1152, evaluate.AUGMENTED: 4224}
    raw = core.align_rows(core.read_rows(folder / "oof.csv", {"ID", "hardness", "fold"}), data.ids, folder / "oof.csv")
    np.testing.assert_array_equal(core.finite_column(raw, "hardness", folder), data.y)
    np.testing.assert_array_equal(core.integer_column(raw, "fold", folder), data.folds)
    np.testing.assert_array_equal(core.finite_column(raw, core.BASELINE_NAME, folder), data.baseline)
    nested = {name: core.finite_column(raw, name, folder) for name in
        (evaluate.GLOBAL+"_nested", evaluate.AUGMENTED+"_nested", core.PRIMARY_NAME)}
    predictions = evaluate.add_fixed_candidate(nested, references)
    for name, prediction in predictions.items():
        np.testing.assert_array_equal(prediction, core.finite_column(raw, name, folder))
    selections = summary["selections"]
    assert len(selections) == 5 and {r["outer_fold"] for r in selections} == set(range(5))
    fold_comparison, search = [], []
    for r in selections:
        fold, candidates = r["outer_fold"], r["candidates"]
        assert r["outer_train_n"] == 400 and r["outer_valid_n"] == 100
        assert r["inner_seed"] == seed + fold and r["inner_validation_sizes"] == [134, 133, 133]
        assert len(candidates) == 12
        assert {(c["representation"], c["alpha"]) for c in candidates} == {(name, a) for name in evaluate.REPRESENTATIONS for a in evaluate.ALPHAS}
        key = lambda c: (c["inner_mse"], evaluate.REPRESENTATIONS.index(c["representation"]), -c["alpha"])
        assert r["joint"] == min(candidates, key=key)
        for name in evaluate.REPRESENTATIONS:
            assert r["per_representation"][name] == min([c for c in candidates if c["representation"] == name], key=key)
        valid = data.folds == fold
        np.testing.assert_array_equal(nested[core.PRIMARY_NAME][valid], nested[r["joint"]["representation"]+"_nested"][valid])
        for c in candidates:
            np.testing.assert_allclose(c["inner_mse"], np.average(c["inner_fold_mse"], weights=r["inner_validation_sizes"]), atol=1e-9, rtol=0)
            search.append({"outer_fold": fold, "inner_seed": r["inner_seed"], "representation": c["representation"],
                "alpha": c["alpha"], "inner_mse": c["inner_mse"], **{f"inner_fold{j}_mse": score for j, score in enumerate(c["inner_fold_mse"])},
                "selected_for_representation": c == r["per_representation"][c["representation"]], "selected_joint": c == r["joint"]})
        global_choice, local_choice = [r["per_representation"][name] for name in evaluate.REPRESENTATIONS]
        outer_global, outer_local = [core.metrics(data.y[valid], nested[name+"_nested"][valid])["rmse"] for name in evaluate.REPRESENTATIONS]
        old_blend, new_blend = [core.metrics(data.y[valid], predictions[name][valid])["rmse"] for name in (evaluate.CURRENT, evaluate.FIXED_BLEND)]
        fold_comparison.append({"fold": fold, "joint_representation": r["joint"]["representation"],
            "global_alpha": global_choice["alpha"], "spatial_alpha": local_choice["alpha"],
            "inner_global_mse": global_choice["inner_mse"], "inner_spatial_mse": local_choice["inner_mse"],
            "inner_spatial_minus_global_mse": local_choice["inner_mse"]-global_choice["inner_mse"],
            "outer_global_rmse": outer_global, "outer_spatial_rmse": outer_local,
            "outer_spatial_minus_global_rmse": outer_local-outer_global,
            "blend_delta_rmse": new_blend-old_blend,
            "blend_predictions_exactly_unchanged": bool(np.array_equal(predictions[evaluate.CURRENT][valid], predictions[evaluate.FIXED_BLEND][valid]))})
    anchor_check = evaluate.verify_global_anchor(data, nested, selections, anchor)
    assert anchor_check == summary["global_anchor"]
    diagnostics, tables = evaluate.diagnose(data, predictions)
    assert diagnostics == summary["diagnostics"]
    for name, data_rows in {**tables, "inner_search": search}.items():
        assert rows(folder / (name+".csv")) == list(csv.DictReader(core.csv_text(data_rows).splitlines())), name
    assert summary["estimator_fit_count"] == 190 and summary["candidate_count_per_inner_search"] == 12
    old_error = references[evaluate.CURRENT] - data.y
    new_error = predictions[evaluate.FIXED_BLEND] - data.y
    old_worst = set(np.argsort(-np.abs(old_error), kind="stable")[:20])
    new_worst = set(np.argsort(-np.abs(new_error), kind="stable")[:20])
    models = (evaluate.CURRENT, evaluate.DINO, evaluate.AUGMENTED+"_nested", core.PRIMARY_NAME, evaluate.FIXED_BLEND)
    return {"seed": seed, "verified_input_and_source_hashes": hashes, "verified_oof_rows": 500,
        "global_anchor": anchor_check, "extraction_global_match": data.metadata["spatial"]["global_reextraction_check"],
        "estimator_fit_count_in_user_run": summary["estimator_fit_count"], "fold_comparison": fold_comparison,
        "diagnostics": {name: diagnostics[name] for name in models},
        "residual_correlation_spatial_vs_global": core.correlation(nested[evaluate.AUGMENTED+"_nested"]-data.y, nested[evaluate.GLOBAL+"_nested"]-data.y),
        "residual_correlation_new_vs_current_blend": core.correlation(new_error, old_error),
        "current_worst20": [str(data.ids[i]) for i in sorted(old_worst)],
        "new_worst20": [str(data.ids[i]) for i in sorted(new_worst)],
        "new_worst20_overlap_with_current": len(old_worst & new_worst),
        "per_image_blend_change": [{"ID": str(ident), "old_squared_error": float(old_error[i]**2),
            "new_squared_error": float(new_error[i]**2), "delta_squared_error": float(new_error[i]**2-old_error[i]**2)} for i, ident in enumerate(data.ids)]}


if __name__ == "__main__":
    original_files = [p for folder in ("features", "seed42", "seed2026") for p in (OUT/folder).iterdir() if p.is_file()]
    before = {str(p): core.sha256(p) for p in original_files}
    with mock.patch.object(core, "make_estimator", side_effect=AssertionError("Review must never fit a model")):
        seeds = {str(seed): audit(seed) for seed in (42, 2026)}
    assert before == {str(p): core.sha256(p) for p in original_files}
    common = set(seeds["42"]["current_worst20"]) & set(seeds["2026"]["current_worst20"])
    changes = {seed: {r["ID"]: r["delta_squared_error"] for r in result["per_image_blend_change"]} for seed, result in seeds.items()}
    repeated = {"previous_common_worst20_count": len(common), "ids": sorted(common),
        "improved_in_both_seeds": sorted(i for i in common if changes["42"][i] < 0 and changes["2026"][i] < 0),
        "remain_in_new_worst20_both_seeds": sorted(common & set(seeds["42"]["new_worst20"]) & set(seeds["2026"]["new_worst20"]))}
    output = {"scope": "Stored OOF arithmetic/cache provenance only; estimator fitting explicitly disabled; no images, GPU, test or downloads.",
        "original_files_preserved": before, "seeds": seeds, "repeated_failures": repeated}
    (OUT / "review_results.json").write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    for seed, result in seeds.items():
        print(json.dumps({k: result[k] for k in ("seed", "verified_input_and_source_hashes", "global_anchor", "extraction_global_match",
            "fold_comparison", "residual_correlation_spatial_vs_global", "residual_correlation_new_vs_current_blend", "new_worst20_overlap_with_current")}, indent=2))
        for name, d in result["diagnostics"].items():
            print(name, json.dumps({"rmse": d["overall"]["rmse"], "tails20": d["target_regions"]["tails20"]["rmse"],
                "middle80": d["target_regions"]["middle80"]["rmse"], "noise_q4": d["noise_quartiles"][-1]["rmse"]}))
    print(json.dumps(repeated, indent=2))
