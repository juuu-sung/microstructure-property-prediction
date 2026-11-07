"""Read-only arithmetic/checkpoint audit of completed DINO partial runs.

No model building, fitting, forward, GPU, download or image access. Writes only
reports/dinov2_partial/review_results.json; original results remain unchanged.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from safetensors import safe_open
from sklearn.model_selection import StratifiedKFold

import train_partial_dinov2 as train
from review_dinov2_spatial import verify_hashes

core, evaluation, prefix = train.core, train.evaluation, train.prefix
OUT = core.ROOT / "reports/dinov2_partial"


def csv_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def check_checkpoint(path, family, record, data, run_id, original):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    idx = record["outer_train_indices"]
    choice = record["selected"][family]
    assert saved["config"] == choice["config"] and saved["epoch"] == choice["epoch"]
    assert saved["train_indices"] == idx and saved["validation_indices"] == record["outer_valid_indices"]
    assert saved["run_fingerprint"] == run_id and saved["protocol"] == train.PROTOCOL
    assert saved["revision"] == prefix.base.DEFAULT_REVISION
    y = data.y[idx]
    assert saved["target_mean"] == float(y.mean()) and saved["target_std"] == float(y.std())
    state = saved["state_dict"]
    x = data.features["cls_patch_mean_std"][idx].astype(np.float64)
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale == 0] = 1.
    np.testing.assert_allclose(state["feature_mean"].numpy(), mean.astype(np.float32), rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(state["feature_scale"].numpy(), scale.astype(np.float32), rtol=1e-6, atol=1e-7)
    assert all(torch.isfinite(v).all() for v in state.values())
    assert torch.count_nonzero(state["head.weight"]).item() > 0
    assert saved["training_seed"] == (record["inner_seed"]-record["outer_fold"] + 1009*record["outer_fold"]+997) % 2**32
    result = {"head_weight_l2": float(torch.linalg.vector_norm(state["head.weight"])),
        "head_bias": float(state["head.bias"].item()), "target_mean": saved["target_mean"], "target_std": saved["target_std"]}
    if family == "partial":
        original_sq, delta_sq, changed, count = 0., 0., 0, 0
        for name, tensor in state.items():
            if name.startswith("norm."):
                assert torch.equal(tensor, original[name]), name
            elif name.startswith("block."):
                previous = original["blocks.11." + name[len("block."):]]
                delta = tensor.double() - previous.double()
                delta_sq += float(delta.square().sum())
                original_sq += float(previous.double().square().sum())
                count += 1
                changed += int(not torch.equal(tensor, previous))
            else:
                assert name in {"feature_mean", "feature_scale", "head.weight", "head.bias"}
        assert changed > 0 and count > 0
        result.update(block_parameter_tensors=count, block_parameter_tensors_changed=changed,
            block_relative_l2_change=math.sqrt(delta_sq/original_sq), final_norm_exactly_unchanged=True)
    else:
        assert set(state) == {"feature_mean", "feature_scale", "head.weight", "head.bias"}
    return result


def audit(seed):
    args = train.parse_args(["--seed", str(seed)])
    folder = args.output_dir
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((folder / "run_manifest.json").read_text(encoding="utf-8"))
    run_id = train.fingerprint(manifest["contract"])
    assert run_id == manifest["fingerprint"] == summary["run_fingerprint"]
    assert summary["seed"] == seed and summary["protocol"] == train.PROTOCOL
    assert summary["max_epochs"] == 80 and summary["batch_size"] == 8 and summary["effective_batch"] == 32
    assert summary["estimator_run_count"] == 70 and summary["inner_training_runs"] == 60 and summary["outer_refits"] == 10
    assert summary["input_files"] == manifest["contract"]["input_files"]
    hashes = verify_hashes(summary["input_files"])
    data, refs, anchor = evaluation.load_prior(args)
    tokens, metadata = prefix.validate_cache(np, args.prefix_cache, data.ids, data.metadata, args.global_features)
    assert metadata == summary["prefix_metadata"]
    del tokens
    for item in summary["runtime"]["checkpoint_files"]:
        assert core.sha256(Path(item["cache_path"])) == item["sha256"]
    safepath = next(r["cache_path"] for r in summary["runtime"]["checkpoint_files"] if r["filename"] == "model.safetensors")
    with safe_open(safepath, framework="pt", device="cpu") as handle:
        original = {k: handle.get_tensor(k) for k in handle.keys() if k.startswith(("blocks.11.", "norm."))}
    oof = core.align_rows(core.read_rows(folder / "oof.csv", {"ID", "hardness", "fold", train.PRIMARY, train.CONTROL}), data.ids, folder)
    np.testing.assert_array_equal(core.finite_column(oof, "hardness", folder), data.y)
    np.testing.assert_array_equal(core.integer_column(oof, "fold", folder), data.folds)
    np.testing.assert_array_equal(core.finite_column(oof, core.BASELINE_NAME, folder), data.baseline)
    nested = {name: core.finite_column(oof, name, folder) for name in (train.CONTROL, train.PRIMARY)}
    predictions = train.add_fixed_predictions(nested, refs)
    for name, value in predictions.items():
        np.testing.assert_array_equal(value, core.finite_column(oof, name, folder))
    records, epoch_stats, fold_stats, ckpt_stats = [], [], [], []
    optimizer_steps, fit_epochs = 0, 0
    for fold in range(5):
        record = train.read_completed_fold(folder / f"fold{fold}", run_id, fold, data.folds)
        assert record is not None
        assert record["inner_seed"] == seed+fold
        ti, vi = np.flatnonzero(data.folds != fold), np.flatnonzero(data.folds == fold)
        assert record["outer_train_indices"] == ti.tolist()
        for name in nested:
            np.testing.assert_array_equal(record["predictions"][name], nested[name][vi])
        splits = list(StratifiedKFold(3, shuffle=True, random_state=seed+fold).split(data.y[ti], core.rank_groups(data.y[ti], 10)))
        sizes = [len(v) for _, v in splits]
        assert sizes == record["inner_validation_sizes"]
        members = [{"outer_fold": fold, "inner_fold": j, "role": role, "ID": str(data.ids[i])}
            for j, (it, iv) in enumerate(splits) for role, indices in (("train", ti[it]), ("valid", ti[iv])) for i in indices]
        assert record["memberships"] == members
        curves = {}
        for config in train.CONFIGS:
            curves[config["name"]] = []
            for j in range(3):
                rows = [r for r in record["inner_curves"] if r["config"] == config["name"] and r["inner_fold"] == j]
                assert len(rows) == 80 and all(r["outer_fold"] == fold for r in rows)
                assert all(r["optimizer_steps"] == math.ceil((400-sizes[j])/32) for r in rows)
                assert all(np.isfinite([r["online_train_mse"], r["valid_mse"]]).all() and min(r["online_train_mse"], r["valid_mse"]) >= 0 for r in rows)
                optimizer_steps += sum(r["optimizer_steps"] for r in rows)
                fit_epochs += len(rows)
                curves[config["name"]].append(rows)
        selected, candidates = train.select_candidates(curves, sizes, 80)
        assert selected == record["selected"] and candidates == record["candidates"]
        for config in train.CONFIGS:
            candidates_config = [c for c in candidates if c["config"] == config]
            best = min(candidates_config, key=lambda c: (c["inner_mse"], c["epoch"]))
            series = []
            for e in (1, best["epoch"], 20, 40, 60, 80):
                c = candidates_config[e-1]
                online = np.average([r[e-1]["online_train_mse"] for r in curves[config["name"]]], weights=[400-v for v in sizes])
                series.append({"epoch": e, "inner_rmse": math.sqrt(c["inner_mse"]), "online_train_rmse": math.sqrt(online)})
            epoch_stats.append({"fold": fold, "config": config["name"], "family": config["family"],
                "best_epoch": best["epoch"], "best_inner_rmse": math.sqrt(best["inner_mse"]),
                "epoch80_inner_rmse": math.sqrt(candidates_config[-1]["inner_mse"]), "series": series})
        fold_values = {n: core.metrics(data.y[vi], p[vi])["rmse"] for n, p in predictions.items()}
        fold_stats.append({"fold": fold, "rmse": fold_values,
            "partial_minus_head": fold_values[train.PRIMARY]-fold_values[train.CONTROL],
            "partial_minus_dino": fold_values[train.PRIMARY]-fold_values[evaluation.DINO],
            "blend_minus_current": fold_values[train.FIXED]-fold_values[evaluation.CURRENT],
            "original_ridge_inner_rmse": math.sqrt(anchor["selections"][fold]["winner"]["inner_mse"]),
            "selected": selected})
        for family in train.FAMILIES:
            refit = record["refits"][family]
            epoch = selected[family]["epoch"]
            assert [r["epoch"] for r in refit["curve"]] == list(range(1, epoch+1))
            assert all("valid_mse" not in r and r["optimizer_steps"] == 13 for r in refit["curve"])
            optimizer_steps += sum(r["optimizer_steps"] for r in refit["curve"])
            fit_epochs += epoch
            assert refit["target_mean"] == float(data.y[ti].mean()) and refit["target_std"] == float(data.y[ti].std())
            ckpt_stats.append({"fold": fold, "family": family, "train_rmse": refit["train_rmse"],
                **check_checkpoint(folder / f"fold{fold}" / (family+".pt"), family, record, data, run_id, original)})
        records.append(record)
    compact = [{k: v for k, v in r.items() if k not in ("inner_curves", "memberships", "predictions", "candidates")} for r in records]
    assert compact == summary["selections"]
    diagnostics, tables = evaluation.diagnose(data, predictions)
    for name, d in diagnostics.items():
        d["overall"]["delta_rmse_vs_head_only"] = d["overall"]["rmse"] - diagnostics[train.CONTROL]["overall"]["rmse"]
    for row in tables["metrics"]:
        row["delta_rmse_vs_head_only"] = diagnostics[row["model"]]["overall"]["delta_rmse_vs_head_only"]
    assert diagnostics == summary["diagnostics"]
    inner = [{"outer_fold": r["outer_fold"], "config": c["config"]["name"], "family": c["config"]["family"],
        "epoch": c["epoch"], "inner_mse": c["inner_mse"], **{f"inner_fold{j}_mse": v for j, v in enumerate(c["inner_fold_mse"])},
        "selected": c == r["selected"][c["config"]["family"]]} for r in records for c in r["candidates"]]
    tables.update(inner_search=inner, inner_curves=[v for r in records for v in r["inner_curves"]],
        inner_memberships=[v for r in records for v in r["memberships"]])
    for name, rows in tables.items():
        assert csv_rows(folder / (name+".csv")) == list(csv.DictReader(core.csv_text(rows).splitlines())), name
    assert summary["epoch_limit_selections"] == []
    worst = {name: [str(data.ids[i]) for i in np.argsort(-np.abs(predictions[name]-data.y), kind="stable")[:20]]
        for name in (evaluation.CURRENT, train.FIXED)}
    checkpoint_train = {family: math.sqrt(np.mean([r["train_rmse"]**2 for r in ckpt_stats if r["family"] == family])) for family in train.FAMILIES}
    return {"seed": seed, "verified_input_source_hashes": hashes, "oof_rows": len(data.y),
        "global_reconstruction_max_difference_recorded": summary["runtime"]["initial_global_max_difference"],
        "elapsed_seconds": summary["elapsed_seconds_this_invocation"], "inner_curve_rows": len(tables["inner_curves"]),
        "total_fit_epochs_including_refits": fit_epochs, "total_optimizer_steps": optimizer_steps,
        "metrics": {name: d["overall"] for name, d in diagnostics.items()},
        "target_regions": {name: d["target_regions"] for name, d in diagnostics.items()},
        "noise_q4": {name: d["noise_quartiles"][-1] for name, d in diagnostics.items()},
        "train_pooled_rmse": checkpoint_train, "folds": fold_stats, "epoch_curves": epoch_stats, "checkpoints": ckpt_stats,
        "residual_correlation_partial_vs_dino": core.correlation(predictions[train.PRIMARY]-data.y, predictions[evaluation.DINO]-data.y),
        "worst20": worst, "worst20_overlap": len(set(worst[evaluation.CURRENT]) & set(worst[train.FIXED])),
        "blend_delta_squared_error_by_id": {str(ident): float((predictions[train.FIXED][i]-data.y[i])**2-(predictions[evaluation.CURRENT][i]-data.y[i])**2) for i, ident in enumerate(data.ids)}}


if __name__ == "__main__":
    files = [p for folder in ("prefix", "seed42", "seed2026") for p in (OUT / folder).rglob("*") if p.is_file()]
    before = {str(p): core.sha256(p) for p in files}
    with mock.patch.object(train, "TorchRunner", side_effect=AssertionError("Audit must not train")), \
         mock.patch.object(prefix.base, "build_model", side_effect=AssertionError("Audit must not build or run a model")), \
         mock.patch.object(core, "make_estimator", side_effect=AssertionError("Audit must not fit an estimator")):
        seeds = {str(seed): audit(seed) for seed in (42, 2026)}
    assert before == {str(p): core.sha256(p) for p in files}
    common = set(seeds["42"]["worst20"][evaluation.CURRENT]) & set(seeds["2026"]["worst20"][evaluation.CURRENT])
    failures = {"previous_common_worst20": sorted(common),
        "improved_both": sorted(i for i in common if all(s["blend_delta_squared_error_by_id"][i] < 0 for s in seeds.values())),
        "still_worst20_both": sorted(common & set(seeds["42"]["worst20"][train.FIXED]) & set(seeds["2026"]["worst20"][train.FIXED]))}
    result = {"scope": "Stored predictions/curves/CPU checkpoint tensors only; no training/forward/GPU/images/test/downloads",
        "original_files_preserved": before, "seeds": seeds, "repeated_failures": failures}
    (OUT / "review_results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Verified {len(files)} original files; both runs and 20 checkpoints audited without fitting or forward.")
    for seed, s in seeds.items():
        print(json.dumps({"seed": seed, "source_hashes": s["verified_input_source_hashes"], "elapsed_seconds": s["elapsed_seconds"],
            "fit_epochs": s["total_fit_epochs_including_refits"], "optimizer_steps": s["total_optimizer_steps"], "train_rmse": s["train_pooled_rmse"],
            "metrics": {n: {k: s["metrics"][n][k] for k in ("rmse", "delta_rmse_vs_current_blend", "delta_rmse_vs_head_only")} for n in (train.PRIMARY, train.CONTROL, train.FIXED)},
            "worst20_overlap": s["worst20_overlap"], "residual_correlation": s["residual_correlation_partial_vs_dino"]}, indent=2))
        for c in s["epoch_curves"]:
            print(json.dumps({k: c[k] for k in ("fold", "config", "best_epoch", "best_inner_rmse", "epoch80_inner_rmse")}))
    print(json.dumps(failures, indent=2))
