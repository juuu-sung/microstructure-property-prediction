"""Audit saved ResNet D4 results and error complementarity; never fit/infer."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
from unittest import mock

import numpy as np
import evaluate_resnet_d4 as evaluate
from review_dinov2_spatial import verify_hashes, rows

core = evaluate.core
OUT = core.ROOT / "reports/resnet_d4"


def audit(seed):
    args = evaluate.parse_args(["--seed", str(seed)])
    folder = args.output_dir
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    hashes = verify_hashes(summary["input_files"])
    data, refs, anchor = evaluate.load_prior(args)
    data = evaluate.attach_features(data, args.resnet_d4_features, args.original_features)
    assert data.metadata == summary["feature_metadata"]
    assert data.provenance == summary["input_files"]
    assert summary["seed"] == seed and summary["n_train"] == 500
    assert summary["alpha_candidates"] == list(evaluate.ALPHAS)
    assert summary["feature_dimensions"] == {n: 1792 for n in evaluate.REPRESENTATIONS}
    assert summary["protocol"] == evaluate.extract.PROTOCOL
    assert summary["primary_model"] == evaluate.PRIMARY
    assert summary["comparison_reference"] == evaluate.REFERENCE
    assert summary["representation_selection"] is False
    assert (summary["outer_splits"], summary["inner_splits"], summary["estimator_fit_count"]) == (5, 3, 100)
    assert summary["fixed_weights"] == {"hand": 0.25, "resnet_d4": 0.25, "dino_d4": 0.5}
    raw = core.align_rows(core.read_rows(folder / "oof.csv", {"ID", "hardness", "fold"}), data.ids, folder)
    for column, expected in (("hardness", data.y), ("fold", data.folds), (core.BASELINE_NAME, data.baseline)):
        np.testing.assert_array_equal(core.finite_column(raw, column, folder), expected)
    nested = {n + "_nested": core.finite_column(raw, n + "_nested", folder) for n in evaluate.REPRESENTATIONS}
    predictions = evaluate.add_fixed_blend(nested, refs)
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
            "raw_alpha": r["per_representation"][evaluate.RAW]["alpha"], "d4_alpha": r["per_representation"][evaluate.D4]["alpha"],
            "raw_resnet_rmse": score(nested[evaluate.RAW + "_nested"]), "d4_resnet_rmse": score(nested[evaluate.D4 + "_nested"]),
            "reference_rmse": score(refs[evaluate.REFERENCE]), "new_blend_rmse": score(predictions[evaluate.PRIMARY]),
            "blend_delta_rmse": score(predictions[evaluate.PRIMARY]) - score(refs[evaluate.REFERENCE])})
    check = evaluate.verify_anchor(data, nested, selections, anchor)
    assert check == summary["resnet_anchor"]
    diagnostics, tables = evaluate.diagnose(data, predictions)
    assert diagnostics == summary["diagnostics"]
    for name, records in {**tables, "inner_search": search}.items():
        assert rows(folder / f"{name}.csv") == list(csv.DictReader(core.csv_text(records).splitlines())), name
    assert (folder / "report.md").read_text(encoding="utf-8") == evaluate.build_markdown(summary)
    old_error, new_error = [predictions[n] - data.y for n in (evaluate.REFERENCE, evaluate.PRIMARY)]
    r, rd = [nested[n + "_nested"] - data.y for n in evaluate.REPRESENTATIONS]
    other_error = (0.25 * data.baseline + 0.5 * refs["reference_dino_d4"]) / 0.75 - data.y
    own = float(0.25 ** 2 * np.mean(rd ** 2 - r ** 2))
    cross = float(2 * 0.25 * 0.75 * np.mean((rd - r) * other_error))
    delta_mse = float(np.mean(new_error ** 2 - old_error ** 2))
    np.testing.assert_allclose(own + cross, delta_mse, rtol=0, atol=1e-10)
    top20 = lambda error: {str(data.ids[i]) for i in np.argsort(-error ** 2, kind="stable")[:20]}
    old_top, new_top = top20(old_error), top20(new_error)
    per_image = tables["resnet_replacement_errors"]
    changes = np.array([r["delta_squared_error"] for r in per_image])
    def correlations(residual):
        return {name: core.correlation(residual, p - data.y) for name, p in
                (("hand", data.baseline), ("dino_d4", refs["reference_dino_d4"]),
                 ("other_components_normalized", other_error + data.y))}
    return {"seed": seed, "verified_input_and_source_hashes": hashes, "verified_oof_rows": len(data.ids),
        "resnet_anchor": check, "identity_reextraction_check": data.metadata["resnet_d4"]["identity_reextraction_check"],
        "extraction_seconds": data.metadata["resnet_d4"]["elapsed_extraction_seconds"],
        "user_evaluation_seconds": summary["elapsed_seconds"], "fold_details": fold_details, "diagnostics": diagnostics,
        "mse_decomposition": {"formula": "Delta MSE = 0.25^2 E[rd^2-r^2] + 2*0.25*0.75 E[(rd-r)*u]; u=(0.25 hand+0.5 DINO_D4)/0.75-y",
            "weighted_component_mse_change": own, "cross_error_change": cross, "total_blend_mse_change": delta_mse},
        "correlations_raw_resnet": correlations(r), "correlations_d4_resnet": correlations(rd),
        "current_worst20": sorted(old_top), "new_worst20": sorted(new_top), "top20_overlap": len(old_top & new_top),
        "improved_images": int(np.sum(changes < 0)), "worsened_images": int(np.sum(changes > 0)),
        "unchanged_images": int(np.sum(changes == 0)), "net_sse_change": float(changes.sum()),
        "five_largest_improvements": sorted(per_image, key=lambda r: r["delta_squared_error"])[:5],
        "five_largest_regressions": sorted(per_image, key=lambda r: r["delta_squared_error"], reverse=True)[:5],
        "per_image_blend_change": per_image}


def main():
    original = [p for d in ("features", "seed42", "seed2026") for p in (OUT / d).iterdir() if p.is_file()]
    before = {str(p): core.sha256(p) for p in original}
    with mock.patch.object(core, "make_estimator", side_effect=AssertionError("No fitting in review")), \
         mock.patch.object(core, "nested_predictions", side_effect=AssertionError("No nested fitting in review")), \
         mock.patch.object(evaluate, "nested_predictions", side_effect=AssertionError("No fitting in review")), \
         mock.patch.object(evaluate.extract, "build_model", side_effect=AssertionError("No model in review")):
        seeds = {str(s): audit(s) for s in (42, 2026)}
    assert "torch" not in sys.modules
    assert before == {str(p): core.sha256(p) for p in original}
    common = set(seeds["42"]["current_worst20"]) & set(seeds["2026"]["current_worst20"])
    changes = {s: {r["ID"]: r["delta_squared_error"] for r in v["per_image_blend_change"]} for s, v in seeds.items()}
    repeated = {"previous_common_worst20": sorted(common),
        "improved_in_both": sorted(i for i in common if changes["42"][i] < 0 and changes["2026"][i] < 0),
        "remain_in_both_worst20": sorted(common & set(seeds["42"]["new_worst20"]) & set(seeds["2026"]["new_worst20"]))}
    result = {"scope": "Saved arithmetic/cache/source audit only; no fitting, torch, image inference, GPU, test, downloads, or weight search.",
        "review_script_sha256": core.sha256(Path(__file__)), "original_files_preserved": before, "seeds": seeds, "repeated_failures": repeated}
    (OUT / "review_results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    for s, r in seeds.items():
        print(json.dumps({k: r[k] for k in ("seed", "verified_input_and_source_hashes", "resnet_anchor", "identity_reextraction_check",
            "extraction_seconds", "user_evaluation_seconds", "fold_details", "mse_decomposition", "correlations_raw_resnet",
            "correlations_d4_resnet", "top20_overlap", "improved_images", "worsened_images", "net_sse_change")}, indent=2))
    print(json.dumps({"original_files_preserved": len(before), "repeated_failures": repeated}, indent=2))


if __name__ == "__main__":
    main()
