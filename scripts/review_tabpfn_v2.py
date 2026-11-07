"""Audit saved TabPFN-v2 results without fitting or running model inference."""
from __future__ import annotations

import csv
import numpy as np
from sklearn.model_selection import StratifiedKFold

import evaluate_tabpfn_v2 as t


def close(a, b):
    np.testing.assert_allclose(a, b, rtol=0, atol=1e-9)


def rows(path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def cell(folder, binding, rep, x, y, ti, vi, seed):
    assert not np.intersect1d(ti, vi).size
    saved = t.read_json(folder/"completed.json")
    assert saved["binding"] == {
        "experiment": binding, "representation": rep,
        "train_idx": ti.tolist(), "query_idx": vi.tolist(), "seed": seed,
        "train_y_sha256": t.common.sha256_array(y[ti]),
        "train_x_sha256": t.common.sha256_array(x[ti]),
        "query_x_sha256": t.common.sha256_array(x[vi]),
    }
    result = saved["result"]
    assert saved["result_sha256"] == t.digest(result)
    p = np.asarray(result["prediction"], dtype=np.float64)
    assert p.shape == (len(vi),) and np.isfinite(p).all()
    m = result["metadata"]
    assert (m["context_rows"], m["query_rows"], m["columns"]) == (len(ti), len(vi), x.shape[1])
    assert m["runtime"] == binding["runtime"]
    assert m["resolved_config"]["MAX_NUMBER_OF_FEATURES"] == 500
    assert m["resolved_config"]["MIN_UNIQUE_FOR_NUMERICAL_FEATURES"] == 0
    close(m["context_mean"], y[ti].mean())
    close(m["context_std"], y[ti].std())
    return p


def main():
    data, features, refs, plan = t.validate(t.OUT)
    protected = t.read_json(t.OUT/"preparation_audit.json")["protected_files"].copy()
    records, predictions, worst = [], {}, {}
    for name in ("protocol.json", "comparison.json", "preparation_audit.json"):
        path = t.OUT/name
        protected[str(path)] = t.common.sha256_file(path)
    for seed in t.SEEDS:
        folder, ref = t.OUT/f"seed{seed}", refs[seed]
        summary = t.read_json(folder/"summary.json")
        binding = summary["binding"]
        assert binding["plan"] == t.digest(plan["frozen"]) and binding["seed"] == seed
        assert binding["runtime"] == {"device": "cuda:0", "gpu": "NVIDIA GeForce RTX 5060",
                                      "precision": "float32", "threads": 4, "telemetry": False, "hf_offline": True}
        assert t.verify_completed(folder, binding)
        completed = t.read_json(folder/"completed_evaluation.json")
        assert len(completed["files"]) == 45  # 40 cells and five aggregate outputs
        protected.update({str(folder/n): sha for n, sha in completed["files"].items()})
        protected[str(folder/"completed_evaluation.json")] = t.common.sha256_file(folder/"completed_evaluation.json")
        assert summary["cells"] == 40 and summary["no_gradient_training"] and not summary["test_accessed"]
        assert summary["primary"] == t.PRIMARY and summary["secondary"] == t.SECONDARY
        path = folder/"oof.csv"
        oof = t.core.align_rows(t.core.read_rows(path, {"ID", "hardness", "fold"}), data.ids, path)
        np.testing.assert_array_equal(t.core.finite_column(oof, "hardness", path), data.y)
        np.testing.assert_array_equal([int(r["fold"]) for r in oof], ref["folds"])
        pred = {n: t.core.finite_column(oof, n, path) for n in summary["metrics"]}
        close(pred["current"], ref["current"])
        for n, p in ref["controls"].items():
            close(pred[n], p)
        close(pred[t.SECONDARY], .5*(pred["current"]+pred[t.PRIMARY]))
        for fold in range(5):
            ti, vi = np.flatnonzero(ref["folds"] != fold), np.flatnonzero(ref["folds"] == fold)
            assert (len(ti), len(vi)) == (400, 100)
            splits = list(StratifiedKFold(3, shuffle=True, random_state=seed+fold).split(ti, t.core.rank_groups(data.y[ti], 10)))
            inner = {rep: np.full(400, np.nan) for rep in t.REPS}
            saved_splits = []
            for rep in t.REPS:
                for j, (tr, va) in enumerate(splits):
                    p = cell(folder/f"fold{fold}_{rep}_inner{j}", binding, rep, features[rep], data.y,
                             ti[tr], ti[va], seed+1009*fold+j)
                    inner[rep][va] = p
                    saved_splits.append({"representation": rep, "inner_fold": j, "train_idx": ti[tr].tolist(),
                                         "valid_idx": ti[va].tolist(), "valid_mse": float(np.mean((p-data.y[ti[va]])**2))})
                p = cell(folder/f"fold{fold}_{rep}_outer", binding, rep, features[rep], data.y, ti, vi, seed+1009*fold+9)
                close(pred["tabpfn_"+rep][vi], p)
            choice, scores = t.select_representation(inner, data.y[ti])
            assert summary["selection"][fold] == {"fold": fold, "representation": choice, "inner_mse": scores, "inner_splits": saved_splits}
            close(pred[t.PRIMARY][vi], pred["tabpfn_"+choice][vi])
        bins, noise = t.core.rank_groups(data.y, 10), t.core.rank_groups(data.noise, 4)
        worst[seed] = {n: set(np.argsort((p-data.y)**2, kind="stable")[-20:].tolist()) for n, p in pred.items()}
        masks = {"tail20": (bins == 0)|(bins == 9), "middle80": (bins > 0)&(bins < 9),
                 "baseline_worst20": np.isin(np.arange(500), list(worst[seed]["current"]))}
        masks.update({f"fold{f}": ref["folds"] == f for f in range(5)})
        masks.update({f"noiseQ{q+1}": noise == q for q in range(4)})
        diag = {(r["model"], r["group"]): r for r in rows(folder/"diagnostics.csv")}
        assert len(diag) == len(pred)*len(masks)
        baseline = t.core.metrics(data.y, pred["current"])["rmse"]
        for n, p in pred.items():
            m = t.core.metrics(data.y, p)
            for k, v in {**m, "gain_vs_current": baseline-m["rmse"]}.items():
                close(summary["metrics"][n][k], v)
            for g, mask in masks.items():
                gm = t.core.metrics(data.y[mask], p[mask])
                close(summary["groups"][n][g], gm["rmse"])
                for k, v in gm.items():
                    close(float(diag[n, g][k]), v)
        errors = rows(folder/"per_image_errors.csv")
        order = np.argsort(-(pred["current"]-data.y)**2, kind="stable")
        assert [r["ID"] for r in errors] == [str(data.ids[i]) for i in order]
        close([float(r["hardness"]) for r in errors], data.y[order])
        for column, model in (("current_error", "current"), ("tabpfn_error", t.PRIMARY), ("blend_error", t.SECONDARY)):
            close([float(r[column]) for r in errors], (pred[model]-data.y)[order])
        gains = (pred["current"]-data.y)**2-(pred[t.SECONDARY]-data.y)**2
        top = np.argsort(-gains, kind="stable")[:5]
        records.append({"seed": seed, "metrics": summary["metrics"], "groups": summary["groups"],
                        "choices": [{"fold": r["fold"], "representation": r["representation"]} for r in summary["selection"]],
                        "folds_improved": sum(summary["groups"][t.SECONDARY][f"fold{f}"] < summary["groups"]["current"][f"fold{f}"] for f in range(5)),
                        "residual_correlation_current_tabpfn": float(np.corrcoef(pred["current"]-data.y, pred[t.PRIMARY]-data.y)[0, 1]),
                        "images_improved": int((gains > 0).sum()), "net_sse_gain": float(gains.sum()),
                        "largest5_positive_sse_gain": float(gains[top].sum()),
                        "largest5_gain_ids": [str(data.ids[i]) for i in top]})
        predictions[seed] = pred
    cmp = t.read_json(t.OUT/"comparison.json")
    for c, r in zip(cmp["per_seed"], records, strict=True):
        assert c["seed"] == r["seed"]
        for key, n, metric in (("current_rmse", "current", "rmse"), ("primary_rmse", t.PRIMARY, "rmse"),
                               ("primary_gain", t.PRIMARY, "gain_vs_current"), ("secondary_gain", t.SECONDARY, "gain_vs_current")):
            close(c[key], r["metrics"][n][metric])
    primary = [r["metrics"][t.PRIMARY]["gain_vs_current"] for r in records]
    close(cmp["mean_primary_gain"], np.mean(primary))
    assert cmp["passes_numeric_priority_gate"] == (all(v > 0 for v in primary) and np.mean(primary) >= .5)
    shared = worst[42]["current"] & worst[2026]["current"]
    indices = sorted(shared)
    common = {}
    for seed, pred in predictions.items():
        common[seed] = {"retained_in_blend_worst20": len(shared & worst[seed][t.SECONDARY]),
                        "metrics": {n: t.core.metrics(data.y[indices], pred[n][indices]) for n in ("current", t.PRIMARY, t.SECONDARY)}}
    assert all(t.common.sha256_file(path) == sha for path, sha in protected.items())
    result = {"review_date_kst": "2026-10-10", "scope": "saved results arithmetic only; no context fitting, GPU inference or test images",
              "checks": "protocol/environment/features/weights;80 cells and disjoint partitions; context scaling; pooled inner choice;500 OOF IDs/targets/folds; all metrics/diagnostics/per-image errors;50:50; protected hashes",
              "per_seed": records, "mean_secondary_gain": float(np.mean([r["metrics"][t.SECONDARY]["gain_vs_current"] for r in records])),
              "baseline_common_worst20_ids": [str(data.ids[i]) for i in indices], "common_worst_diagnostics": common,
              "decision": "standalone not adopted; fixed secondary blend is a modest candidate pending locked confirmation; existing submission retained",
              "independent_holdout": False, "protected_files": protected, "review_script_sha256": t.common.sha256_file(__file__)}
    t.write_json(t.OUT/"review_results.json", result)
    print("Audit passed:80 cells, inner selections,500 OOF rows per seed, all diagnostics and protected hashes.")
    print("Mean secondary RMSE gain:", result["mean_secondary_gain"])
    for r in records:
        print(r["seed"], {k: r[k] for k in ("folds_improved", "residual_correlation_current_tabpfn", "images_improved", "net_sse_gain", "largest5_positive_sse_gain")})
    print("Common baseline worst20:", len(indices), common)
    print("Saved", t.OUT/"review_results.json")


if __name__ == "__main__":
    main()
