"""Audit saved D4 results; arithmetic only, no estimator fitting or forward.

The all-D4 replacement blend is explicitly POST-HOC and never replaces the
prespecified joint_nested result. Original experiment artifacts are hash-guarded.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
from unittest import mock

import numpy as np
import evaluate_dinov2_d4 as evaluate
from review_dinov2_spatial import verify_hashes, rows

core = evaluate.core
OUT = core.ROOT / "reports/dinov2_d4"
POSTHOC = "posthoc_d4_only_replaced_blend"


def audit(seed):
    args = evaluate.parse_args(["--seed", str(seed)])
    folder = args.output_dir
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    hashes = verify_hashes(summary["input_files"])
    data, references, anchor = evaluate.load_prior(args)
    data = evaluate.attach_features(data, args.d4_features, args.global_features)
    assert data.metadata == summary["feature_metadata"]
    assert data.provenance == summary["input_files"]
    assert summary["seed"] == seed and summary["n_train"] == 500
    assert summary["alpha_candidates"] == list(evaluate.ALPHAS)
    assert summary["feature_dimensions"] == {name: 1152 for name in evaluate.REPRESENTATIONS}
    assert summary["d4_protocol"] == evaluate.extract.PROTOCOL
    assert summary["estimator_fit_count"] == 190 and summary["candidate_count_per_inner_search"] == 12
    raw = core.align_rows(core.read_rows(folder / "oof.csv", {"ID", "hardness", "fold"}), data.ids, folder)
    for column, expected in (("hardness", data.y), ("fold", data.folds), (core.BASELINE_NAME, data.baseline)):
        np.testing.assert_array_equal(core.finite_column(raw, column, folder), expected)
    nested = {name: core.finite_column(raw, name, folder) for name in
              (evaluate.GLOBAL+"_nested", evaluate.D4+"_nested", core.PRIMARY_NAME)}
    predictions = evaluate.add_fixed_candidate(nested, references)
    for name, pred in predictions.items():
        np.testing.assert_array_equal(pred, core.finite_column(raw, name, folder))
    selections = summary["selections"]
    assert len(selections) == 5 and {r["outer_fold"] for r in selections} == set(range(5))
    fold_details, search = [], []
    posthoc = references[evaluate.CURRENT] + 0.5*(nested[evaluate.D4+"_nested"]-references[evaluate.DINO])
    for r in selections:
        fold, candidates = r["outer_fold"], r["candidates"]
        assert r["outer_train_n"] == 400 and r["outer_valid_n"] == 100
        assert r["inner_seed"] == seed+fold and r["inner_validation_sizes"] == [134,133,133]
        assert len(candidates) == 12
        assert {(c["representation"],c["alpha"]) for c in candidates} == {(n,a) for n in evaluate.REPRESENTATIONS for a in evaluate.ALPHAS}
        key = lambda c: (c["inner_mse"], evaluate.REPRESENTATIONS.index(c["representation"]), -c["alpha"])
        assert r["joint"] == min(candidates, key=key)
        for name in evaluate.REPRESENTATIONS:
            assert r["per_representation"][name] == min((c for c in candidates if c["representation"] == name), key=key)
        valid = data.folds == fold
        np.testing.assert_array_equal(nested[core.PRIMARY_NAME][valid], nested[r["joint"]["representation"]+"_nested"][valid])
        for c in candidates:
            assert len(c["inner_fold_mse"]) == 3
            assert np.isfinite([c["inner_mse"],*c["inner_fold_mse"]]).all()
            np.testing.assert_allclose(c["inner_mse"], np.average(c["inner_fold_mse"], weights=r["inner_validation_sizes"]), rtol=0, atol=1e-9)
            search.append({"outer_fold":fold,"inner_seed":r["inner_seed"],"representation":c["representation"],
                "alpha":c["alpha"],"inner_mse":c["inner_mse"], **{f"inner_fold{j}_mse":v for j,v in enumerate(c["inner_fold_mse"])},
                "selected_for_representation":c == r["per_representation"][c["representation"]],"selected_joint":c == r["joint"]})
        global_choice, d4_choice = [r["per_representation"][n] for n in evaluate.REPRESENTATIONS]
        score = lambda p: core.metrics(data.y[valid],p[valid])["rmse"]
        fold_details.append({"fold":fold,"selected_representation":r["joint"]["representation"],
            "global_alpha":global_choice["alpha"],"d4_alpha":d4_choice["alpha"],
            "inner_global_rmse":global_choice["inner_mse"]**0.5,"inner_d4_rmse":d4_choice["inner_mse"]**0.5,
            "outer_global_rmse":score(nested[evaluate.GLOBAL+"_nested"]),"outer_d4_rmse":score(nested[evaluate.D4+"_nested"]),
            "original_blend_rmse":score(references[evaluate.CURRENT]),"joint_blend_rmse":score(predictions[evaluate.FIXED_BLEND]),
            "blend_delta_rmse":score(predictions[evaluate.FIXED_BLEND])-score(references[evaluate.CURRENT]),
            "blend_exactly_unchanged":bool(np.array_equal(predictions[evaluate.FIXED_BLEND][valid],references[evaluate.CURRENT][valid])),
            "posthoc_d4_only_blend_rmse":score(posthoc)})
    anchor_check = evaluate.prior.verify_global_anchor(data, nested, selections, anchor)
    assert anchor_check == summary["global_anchor"]
    diagnostics, tables = evaluate.diagnose(data, predictions)
    assert diagnostics == summary["diagnostics"]
    for name, records in {**tables,"inner_search":search}.items():
        assert rows(folder / f"{name}.csv") == list(csv.DictReader(core.csv_text(records).splitlines())), name
    assert (folder/"report.md").read_text(encoding="utf-8") == evaluate.build_markdown(summary)
    post_diagnostics, _ = evaluate.diagnose(data, {**predictions,POSTHOC:posthoc})
    old_error = references[evaluate.CURRENT]-data.y
    new_error = predictions[evaluate.FIXED_BLEND]-data.y
    old_worst = set(np.argsort(-(old_error**2),kind="stable")[:20])
    new_worst = set(np.argsort(-(new_error**2),kind="stable")[:20])
    per_image = [{"ID":str(ident),"old_squared_error":float(old_error[i]**2),
        "new_squared_error":float(new_error[i]**2),"delta_squared_error":float(new_error[i]**2-old_error[i]**2)} for i,ident in enumerate(data.ids)]
    changes = np.array([r["delta_squared_error"] for r in per_image])
    return {"seed":seed,"verified_input_and_source_hashes":hashes,"verified_oof_rows":len(data.ids),
        "global_anchor":anchor_check,"identity_reextraction_check":data.metadata["d4"]["identity_reextraction_check"],
        "extraction_seconds":data.metadata["d4"]["elapsed_extraction_seconds"],
        "user_evaluation_seconds":summary["elapsed_seconds"],"fold_details":fold_details,"diagnostics":diagnostics,
        "posthoc_d4_only_blend":{"status":"posthoc_exploratory_not_primary", "no_additional_fit":True,
            "formula":"reference_current_blend + 0.5 * (d4_mean_cls_patch_mean_std_nested - reference_dino_joint)",
            "diagnostics":post_diagnostics[POSTHOC]},
        "residual_correlation_d4_vs_global":core.correlation(nested[evaluate.D4+"_nested"]-data.y,nested[evaluate.GLOBAL+"_nested"]-data.y),
        "residual_correlation_new_vs_old_blend":core.correlation(new_error,old_error),
        "current_worst20":[str(data.ids[i]) for i in sorted(old_worst)],"new_worst20":[str(data.ids[i]) for i in sorted(new_worst)],
        "improved_images":int(np.sum(changes<0)),"worsened_images":int(np.sum(changes>0)),"unchanged_images":int(np.sum(changes==0)),
        "net_sse_change":float(changes.sum()),"five_largest_improvements":sorted(per_image,key=lambda r:r["delta_squared_error"])[:5],
        "five_largest_regressions":sorted(per_image,key=lambda r:r["delta_squared_error"],reverse=True)[:5],"per_image_blend_change":per_image}


def main():
    original = [p for directory in ("features","seed42","seed2026") for p in (OUT/directory).iterdir() if p.is_file()]
    before = {str(p):core.sha256(p) for p in original}
    with mock.patch.object(core,"make_estimator",side_effect=AssertionError("No fitting in review")), \
         mock.patch.object(core,"nested_predictions",side_effect=AssertionError("No nested fitting in review")), \
         mock.patch.object(evaluate.extract.base,"build_model",side_effect=AssertionError("No model in review")):
        seeds = {str(seed):audit(seed) for seed in (42,2026)}
    assert "torch" not in sys.modules
    assert before == {str(p):core.sha256(p) for p in original}
    common = set(seeds["42"]["current_worst20"]) & set(seeds["2026"]["current_worst20"])
    changes = {s:{r["ID"]:r["delta_squared_error"] for r in v["per_image_blend_change"]} for s,v in seeds.items()}
    repeated = {"previous_common_worst20":sorted(common),
        "improved_in_both":sorted(i for i in common if changes["42"][i]<0 and changes["2026"][i]<0),
        "remain_in_both_worst20":sorted(common & set(seeds["42"]["new_worst20"]) & set(seeds["2026"]["new_worst20"]))}
    result = {"scope":"Arithmetic and cache/source provenance only; no fitting, torch, image inference, GPU, test or downloads.",
        "review_script_sha256":core.sha256(Path(__file__)),"original_files_preserved":before,"seeds":seeds,"repeated_failures":repeated}
    (OUT/"review_results.json").write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    for seed,r in seeds.items():
        print(json.dumps({k:r[k] for k in ("seed","verified_input_and_source_hashes","fold_details",
            "improved_images","worsened_images","unchanged_images","net_sse_change",
            "residual_correlation_d4_vs_global","residual_correlation_new_vs_old_blend")},indent=2))
        d=r["posthoc_d4_only_blend"]["diagnostics"]
        print("POSTHOC ONLY",json.dumps({"seed":seed,"rmse":d["overall"]["rmse"],
            "tails20":d["target_regions"]["tails20"]["rmse"],"middle80":d["target_regions"]["middle80"]["rmse"],
            "noise_q4":d["noise_quartiles"][-1]["rmse"]}))
    print(json.dumps({"original_files_preserved":len(before),"repeated_failures":repeated},indent=2))


if __name__ == "__main__":
    main()
