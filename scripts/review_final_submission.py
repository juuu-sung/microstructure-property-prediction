"""Audit saved final heads and predictions without refitting or image inference."""
from __future__ import annotations

from contextlib import ExitStack
import csv
import json
from pathlib import Path
import sys
from unittest import mock

import final_submission as f

np = f.np
OUT = f.ROOT / "reports/final_ensemble"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def stats(values):
    return {"n": len(values), "min": float(values.min()), "max": float(values.max()),
            "mean": float(values.mean()), "std": float(values.std()),
            "quantiles_1_5_25_50_75_95_99": np.quantile(values, [.01,.05,.25,.5,.75,.95,.99]).tolist()}


def main():
    original = [p for p in OUT.rglob("*") if p.is_file() and p.name != "review_results.json"]
    before = {str(p): f.core.sha256(p) for p in original}
    preparation = read_json(OUT/"preparation.json")
    preserved = preparation["protected_original_files"]
    assert all(f.core.sha256(Path(p)) == h for p,h in preserved.items())
    with ExitStack() as stack:
        for obj, attr in ((f.core,"make_estimator"), (f,"fit_heads"), (f,"cpu_features"),
                          (f,"neural_features"), (f.common,"import_runtime")):
            stack.enter_context(mock.patch.object(obj, attr, side_effect=AssertionError("Review cannot fit/extract")))
        data = f.load_training()
        plan = f.validate_plan(OUT, data)
        f.validate_test_identity(plan, f.ROOT/"open")
        assert f.checkpoint_references() == plan["checkpoints"]
        heads, model_meta = f.load_heads(OUT, plan)
        features, feature_meta = f.load_test_features(OUT, plan)
        selection = read_json(OUT/"model/selection.json")
        assert selection == model_meta["selection"]
        assert selection["n_train"] == 500 and selection["fit_count"] == 39
        assert selection["cv_seed"] == f.SEED and selection["cv_validation_sizes"] == [167,167,166]
        candidates = selection["candidates"]
        assert len(candidates) == 12
        assert {(r["component"],r["alpha"]) for r in candidates} == {(n,a) for n in f.COMPONENTS for a in f.GRIDS[n]}
        for row in candidates:
            assert len(row["fold_mse"]) == 3 and np.isfinite(row["fold_mse"]).all()
            np.testing.assert_allclose(row["pooled_mse"], np.average(row["fold_mse"], weights=selection["cv_validation_sizes"]), atol=1e-9,rtol=0)
        summaries, predictions, train_predictions = {}, {}, {}
        for name in f.COMPONENTS:
            records = [r for r in candidates if r["component"] == name]
            winner = min(records, key=lambda r:(r["pooled_mse"],-r["alpha"]))
            assert selection["selected"][name] == winner
            head = heads[name]
            x = data.features[name].astype(np.float64)
            # Verify all-500 scalers and the saved solution's optimality equation.
            # This computes residuals only: no solve, optimization, or model fitting.
            expected_scale = x.std(axis=0)
            expected_scale[np.ptp(x,axis=0)==0] = 1.
            np.testing.assert_allclose(head["x_mean"],x.mean(axis=0),atol=1e-10,rtol=1e-12)
            np.testing.assert_allclose(head["x_scale"],expected_scale,atol=1e-10,rtol=1e-12)
            np.testing.assert_allclose(head["y_mean"],data.y.mean(),atol=1e-10,rtol=0)
            np.testing.assert_allclose(head["y_scale"],data.y.std(),atol=1e-10,rtol=0)
            scaled_x = (x-head["x_mean"])/head["x_scale"]
            scaled_y = (data.y-head["y_mean"])/head["y_scale"]
            fitted = scaled_x @ head["coef"] + head["intercept"]
            residual = fitted-scaled_y
            gradient = scaled_x.T @ residual + winner["alpha"]*head["coef"]
            relative = float(np.linalg.norm(gradient)/(1+np.linalg.norm(scaled_x.T@scaled_y)+np.linalg.norm(winner["alpha"]*head["coef"])))
            assert relative < 1e-9 and abs(float(residual.mean())) < 1e-10
            test_x = (features[name].astype(np.float64)-head["x_mean"])/head["x_scale"]
            # Row-wise dot product independently checks the production matmul path.
            predictions[name] = (np.einsum("ij,j->i",test_x,head["coef"])+head["intercept"])*head["y_scale"]+head["y_mean"]
            train_predictions[name] = fitted*head["y_scale"]+head["y_mean"]
            summaries[name] = {"alpha":winner["alpha"],"weight":f.WEIGHTS[name],
                "selection_cv_rmse":float(np.sqrt(winner["pooled_mse"])),
                "scalers_match_all_500":True,"ridge_optimality_relative_residual":relative,
                "test_predictions":stats(predictions[name])}
        values = .25*predictions[f.HAND]+.25*predictions[f.RESNET]+.5*predictions[f.DINO]
        train_values = .25*train_predictions[f.HAND]+.25*train_predictions[f.RESNET]+.5*train_predictions[f.DINO]
        template_ids = plan["test"]["ids"]
        submission_path = OUT/"submission/submission.csv"
        assert f.submission_ids(submission_path) == template_ids
        rows = read_rows(submission_path)
        saved = np.array([float(r["hardness"]) for r in rows])
        assert np.isfinite(saved).all()
        np.testing.assert_allclose(saved,values,atol=1e-10,rtol=0)
        component_rows = read_rows(OUT/"submission/component_predictions.csv")
        assert [r["ID"] for r in component_rows] == template_ids
        for name in f.COMPONENTS:
            np.testing.assert_allclose([float(r[name]) for r in component_rows],predictions[name],atol=1e-10,rtol=0)
        np.testing.assert_array_equal([float(r["hardness"]) for r in component_rows],saved)
        stored_blend = .25*np.array([float(r[f.HAND]) for r in component_rows])+.25*np.array([float(r[f.RESNET]) for r in component_rows])+.5*np.array([float(r[f.DINO]) for r in component_rows])
        np.testing.assert_array_equal(stored_blend,saved)
        validation = read_json(OUT/"submission/validation.json")
        for key,path in (("protocol_sha256",OUT/f.PLAN_NAME),("heads_sha256",OUT/"model/heads.npz"),
                         ("test_features_sha256",OUT/"test_features/features.npz"),("submission_sha256",submission_path)):
            assert validation[key] == f.core.sha256(path)
        assert validation["rows"] == 1000 and validation["columns"] == ["ID","hardness"]
        assert validation["sample_order_matches"] and validation["all_finite"]
        assert validation["clipping_or_calibration"] is False and validation["uploaded"] is False
        for key,value in (("min",saved.min()),("max",saved.max()),("mean",saved.mean()),("std",saved.std())):
            np.testing.assert_allclose(validation["prediction_"+key],value,atol=1e-12,rtol=0)
        component_matrix = np.column_stack([predictions[n] for n in f.COMPONENTS])
        spread = np.ptp(component_matrix,axis=1)
        disagreement_cases = [{"ID":template_ids[i],"component_range":float(spread[i]),
            **{n:float(predictions[n][i]) for n in f.COMPONENTS},"hardness":float(saved[i])}
            for i in np.argsort(-spread,kind="stable")[:3]]
        assert "torch" not in sys.modules
        result = {"review_date_kst":"2026-10-09","status":"saved_submission_verified",
            "submission_path":str(submission_path),"submission_sha256":f.core.sha256(submission_path),
            "components":summaries,"test_prediction_distribution":stats(saved),
            "train_label_distribution":stats(data.y),"same_heads_train_prediction_distribution":stats(train_values),
            "outside_train_label_range":int(np.sum((saved<data.y.min())|(saved>data.y.max()))),
            "resnet_predictions_above_train_label_max":int(np.sum(predictions[f.RESNET]>data.y.max())),
            "largest_component_disagreements":disagreement_cases,
            "unique_predictions":len(np.unique(saved)),"independent_prediction_max_abs_difference":float(np.max(np.abs(saved-values))),
            "weights_match_exactly":True,"train_anchor_checks":feature_meta["anchor_checks"],
            "feature_extraction_seconds":feature_meta["elapsed_seconds"],
            "model_selection_cv_scope":selection["cv_score_scope"],
            "test_rmse_available":False,"leaderboard_submission_status":"not checked; local generation did not upload",
            "fits_during_review":0,"feature_extractions_during_review":0,"torch_imported":False,
            "protected_final_files":before,"preserved_historical_files":len(preserved)}
    assert before == {p:f.core.sha256(Path(p)) for p in before}
    assert all(f.core.sha256(Path(p))==h for p,h in preserved.items())
    f.write_json(OUT/"review_results.json",result)
    printable = {k:v for k,v in result.items() if k not in {"protected_final_files","same_heads_train_prediction_distribution"}}
    print(json.dumps(printable,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
