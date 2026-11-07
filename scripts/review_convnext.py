"""Audit completed ConvNeXt artifacts and learning curves; never train/infer."""
from __future__ import annotations
import csv
import math
from pathlib import Path
import numpy as np
if __package__:
    from . import convnext_experiment as c
else:
    import convnext_experiment as c


def table(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def assert_close(actual, expected):
    np.testing.assert_allclose(actual, expected, atol=1e-9, rtol=1e-12)


def audit_run(path, plan_hash, train_idx, train_y, valid_idx, valid_y, profile, epochs, seed, purpose="experiment"):
    stored = c.read_json(path/"completed.json")
    binding = stored["binding"]
    result = c.completed_result(path, binding)  # verifies result and curve/checkpoint file hashes
    expected = {"plan": plan_hash, "train": train_idx.tolist(), "valid": valid_idx.tolist(),
                "train_y": c.common.sha256_array(train_y), "valid_y": None if valid_y is None else c.common.sha256_array(valid_y),
                "profile": profile, "epochs": epochs, "seed": seed, "batch": 4, "accumulation": 4,
                "runtime": binding["runtime"], "purpose": purpose}
    assert binding == expected and c.read_json(path/"binding.json") == expected
    assert binding["runtime"]["device"] == "cuda:0" and binding["runtime"]["amp"] == "float16"
    assert result["runtime"] == binding["runtime"] and result["outer_labels_used"] is False
    assert_close([result["target_mean"], result["target_std"]], c.scaling(train_y))
    curve = result["curve"]
    assert curve == c.read_json(path/"curve.json")
    assert [r["epoch"] for r in curve] == list(range(1, epochs+1))
    for row in curve:
        epoch = row["epoch"]
        assert row["phase"] == ("head" if epoch <= 5 else "full")
        assert all(math.isfinite(row[k]) and row[k] >= 0 for k in ("train_rmse", "augmented_standardized_mse", "seconds"))
        assert row["optimizer_steps"]+row["amp_skipped_steps"] == math.ceil(len(train_idx)/16)
        assert row["optimizer_steps"] >= 0 and row["amp_skipped_steps"] >= 0
        if valid_y is None:
            assert row["valid_rmse"] is None
        else:
            assert math.isfinite(row["valid_rmse"]) and row["valid_rmse"] >= 0
        base = [.001]*2 if epoch <= 5 else [v for v in (3e-6, 6e-6, 1e-5, 3e-5, 3e-4) for _ in range(2)]
        assert_close(row["lr"], np.asarray(base)*c.lr_factor(epoch))
    if valid_y is None:
        assert np.asarray(result["prediction"]).shape == valid_idx.shape
        assert np.isfinite(result["prediction"]).all()
    else:
        assert result["prediction"] is None
    return result


def curve_summary(curve):
    best = min(curve, key=lambda r: (r["valid_rmse"], r["epoch"])) if curve[0]["valid_rmse"] is not None else curve[-1]
    return {"epochs": len(curve), "first_train_rmse": curve[0]["train_rmse"], "best_epoch": best["epoch"],
            "train_at_best": best["train_rmse"], "best_inner_rmse": best["valid_rmse"],
            "last_train_rmse": curve[-1]["train_rmse"], "last_inner_rmse": curve[-1]["valid_rmse"],
            "skipped_epochs": [{"epoch": r["epoch"], "skipped": r["amp_skipped_steps"]} for r in curve if r["amp_skipped_steps"]],
            "optimizer_steps": sum(r["optimizer_steps"] for r in curve), "amp_skipped_steps": sum(r["amp_skipped_steps"] for r in curve),
            "recorded_epoch_seconds": sum(r["seconds"] for r in curve)}


def main():
    root = c.OUT
    data, plan, _ = c.validate(root)
    plan_hash = c.fingerprint(plan["frozen"])
    c.load_features(root, data, plan_hash)
    records, curve_rows, common_worst = [], [], {}
    protected = {}
    for seed in c.SEEDS:
        folder = root/f"seed{seed}"
        refs = c.previous.load_references(seed, data)
        done, summary = c.read_json(folder/"completed_evaluation.json"), c.read_json(folder/"summary.json")
        assert done["binding"] == summary["binding"]
        assert done["binding"] == {"plan": plan_hash, "seed": seed, "batch": 4, "accumulation": 4,
                                    "device": "cuda", "features_sha256": c.common.sha256_file(root/"features.npz")}
        for name, sha in done["files"].items():
            assert c.common.sha256_file(folder/name) == sha
            protected[str(folder/name)] = sha
        protected[str(folder/"completed_evaluation.json")] = c.common.sha256_file(folder/"completed_evaluation.json")
        rows = c.core.align_rows(c.core.read_rows(folder/"oof.csv", {"ID", "hardness", "fold"}), data.ids, folder/"oof.csv")
        np.testing.assert_array_equal(c.core.finite_column(rows, "hardness", folder), data.y)
        np.testing.assert_array_equal([int(r["fold"]) for r in rows], refs["folds"])
        preds = {name: c.core.finite_column(rows, name, folder) for name in summary["metrics"]}
        assert_close(preds["current"], refs["current"])
        assert_close(preds["current_convnext_equal"], .5*(refs["current"]+preds["convnext_nested"]))
        base = c.core.metrics(data.y, refs["current"])["rmse"]
        worst = set(np.argsort((refs["current"]-data.y)**2, kind="stable")[-20:].tolist())
        for name, pred in preds.items():
            m = c.core.metrics(data.y, pred)
            for key, val in m.items():
                assert_close(summary["metrics"][name][key], val)
            assert_close(summary["metrics"][name]["gain_vs_current"], base-m["rmse"])
            overlap = len(worst & set(np.argsort((pred-data.y)**2, kind="stable")[-20:].tolist()))
            assert summary["metrics"][name]["worst20_overlap"] == overlap
        bins, noise = c.core.rank_groups(data.y, 10), c.core.rank_groups(data.noise, 4)
        masks = {"tail20": (bins == 0)|(bins == 9), "middle80": (bins > 0)&(bins < 9), "baseline_worst20": np.isin(np.arange(500), list(worst))}
        masks.update({f"fold{f}": refs["folds"] == f for f in range(5)})
        masks.update({f"noiseQ{q+1}": noise == q for q in range(4)})
        diagnostics = table(folder/"diagnostics.csv")
        assert len(diagnostics) == len(preds)*len(masks)
        groups = {}
        for row in diagnostics:
            mask = masks[row["group"]]
            m = c.core.metrics(data.y[mask], preds[row["model"]][mask])
            for key, val in m.items():
                assert_close(float(row[key]), val)
            groups.setdefault(row["model"], {})[row["group"]] = m["rmse"]
        selections = []
        for fold in range(5):
            ti, vi = np.flatnonzero(refs["folds"] != fold), np.flatnonzero(refs["folds"] == fold)
            tr, va = c.split_inner(ti, data.y[ti], seed, fold)
            inner = {}
            for profile in c.PROFILES:
                path = folder/f"fold{fold}_inner_{profile}"
                result = audit_run(path, plan_hash, tr, data.y[tr], va, data.y[va], profile, 100, seed+1009*fold+100000)
                inner[profile] = result
                curve_rows.append({"seed": seed, "fold": fold, "profile": profile, **curve_summary(result["curve"])})
            choice = c.select_profile(inner)
            assert summary["neural_selection"][fold] == {"fold": fold, "choice": choice, "inner_train": tr.tolist(), "inner_valid": va.tolist()}
            result = audit_run(folder/f"fold{fold}_outer", plan_hash, ti, data.y[ti], vi, None, choice["profile"], choice["epoch"], seed+1009*fold+200000)
            assert_close(preds["convnext_nested"][vi], result["prediction"])
            curve_rows.append({"seed": seed, "fold": fold, "profile": "outer", **curve_summary(result["curve"])})
            selections.append({"fold": fold, **choice})
            saved = c.read_json(folder/f"fold{fold}_ridge.json")
            assert saved["binding"] == done["binding"] and saved["sha256"] == c.fingerprint(saved["result"])
            selection = saved["result"]["selection"]
            assert [v["alpha"] for v in selection["search"]] == list(c.ALPHAS)
            assert selection["choice"] == min(selection["search"], key=lambda v:(v["inner_mse"], -v["alpha"]))
            assert summary["ridge_selection"][fold] == {"fold": fold, **selection}
            assert_close(preds["convnext_frozen_ridge"][vi], saved["result"]["prediction"])
        # Freeze all completed run records/curves/checkpoint hashes without loading model tensors.
        for path in folder.glob("fold*/completed.json"):
            for name, sha in c.read_json(path)["files"].items():
                protected[str(path.parent/name)] = sha
            protected[str(path)] = c.common.sha256_file(path)
        common_worst[seed] = {n: set(np.argsort((p-data.y)**2, kind="stable")[-20:].tolist()) for n,p in preds.items()}
        records.append({"seed": seed, "metrics": summary["metrics"], "groups": groups, "selections": selections,
                        "folds_improved": {n: sum(groups[n][f"fold{f}"] < groups["current"][f"fold{f}"] for f in range(5)) for n in preds},
                        "frozen_vs_finetuned_delta": summary["metrics"]["convnext_nested"]["rmse"]-summary["metrics"]["convnext_frozen_ridge"]["rmse"],
                        "residual_correlations": {n: float(np.corrcoef(preds[n]-data.y, preds["current"]-data.y)[0,1]) for n in preds}})
    comparison = c.read_json(root/"comparison.json")
    gains = [r["metrics"]["convnext_nested"]["gain_vs_current"] for r in records]
    assert_close(comparison["mean_primary_gain"], np.mean(gains))
    assert comparison["passes_numeric_priority_gate"] == (all(v > 0 for v in gains) and np.mean(gains) >= .5)
    for row, r in zip(comparison["per_seed"], records):
        assert row["seed"] == r["seed"]
        assert_close(row["current_rmse"], r["metrics"]["current"]["rmse"])
        assert_close(row["primary_rmse"], r["metrics"]["convnext_nested"]["rmse"])
        assert_close(row["primary_gain"], r["metrics"]["convnext_nested"]["gain_vs_current"])
        assert_close(row["secondary_gain"], r["metrics"]["current_convnext_equal"]["gain_vs_current"])
    refs = c.previous.load_references(42, data)
    ti = np.flatnonzero(refs["folds"] != 0)
    tr, _ = c.split_inner(ti, data.y[ti], 42, 0)
    idx = np.random.default_rng(314159).choice(tr, size=32, replace=False)
    sanity = audit_run(root/"sanity32", plan_hash, idx, data.y[idx], idx, data.y[idx], "none", 100, 314159, purpose="sanity")
    baseline_common = common_worst[42]["current"] & common_worst[2026]["current"]
    retain = {n: sorted(baseline_common & common_worst[42][n] & common_worst[2026][n]) for n in ("convnext_nested", "current_convnext_equal")}
    previous_protected = c.read_json(c.ROOT/"reports/morphology/review_results.json")["protected_files"]
    protected.update(previous_protected)
    assert all(c.common.sha256_file(p) == sha for p,sha in protected.items())
    result = {"review_date_kst": "2026-10-10", "scope": "saved data/OOF/curve/hash arithmetic only; no fit, GPU inference or test images",
              "checks": "protocol, features,500 OOF IDs/targets/folds, metrics, diagnostics,50:50,30run bindings and file hashes, inner choices, scaler, LR, step counts, outer predictions, sanity32",
              "ridge_limit": "stored inner search selection checked; no Ridge refit or reconstruction of unsaved inner predictions",
              "per_seed": records, "inner_and_outer_curves": curve_rows, "sanity": curve_summary(sanity["curve"]),
              "baseline_common_worst20_count": len(baseline_common), "retained_common_worst20": {n:[str(data.ids[i]) for i in ids] for n,ids in retain.items()},
              "decision": "not_adopted; retain existing ensemble and submission; do not extend epochs based on these curves",
              "protected_files": protected, "review_script_sha256": c.common.sha256_file(__file__)}
    c.write_json(root/"review_results.json", result)
    print("Audit passed:30 neural runs,500 OOF rows per seed, metrics, inner selection, curves and protected hashes.")
    print("Sanity:", result["sanity"])
    print("Common baseline worst20:",len(baseline_common),"retained",{n:len(ids) for n,ids in retain.items()})
    print("Saved", root/"review_results.json")


if __name__ == "__main__":
    main()
