"""Arithmetic/provenance and CPU-checkpoint audit; never fit/forward/download.

Writes only a new review_results.json. Original training files are hash-guarded.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from safetensors import safe_open
from sklearn.model_selection import StratifiedKFold

import train_ridge_initialized_dinov2 as train
from review_dinov2_spatial import verify_hashes

core, evaluation, prefix = train.core, train.evaluation, train.prefix
OUT = core.ROOT / "reports/dinov2_ridge_initialized"


def checkpoint_audit(path, family, record, data, run_id, original, initial_head):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    choice, ti, vi = record["selected"][family], record["outer_train_indices"], record["outer_valid_indices"]
    assert saved["config"] == choice["config"] and saved["epoch"] == choice["epoch"]
    assert saved["train_indices"] == ti and saved["validation_indices"] == vi
    assert saved["run_fingerprint"] == run_id and saved["protocol"] == train.PROTOCOL
    assert saved["revision"] == prefix.base.DEFAULT_REVISION
    assert saved["target_mean"] == float(data.y[ti].mean()) and saved["target_std"] == float(data.y[ti].std())
    state = saved["state_dict"]
    assert all(torch.isfinite(v).all() for v in state.values())
    for key in ("feature_mean", "feature_scale", "head.weight", "head.bias"):
        assert state[key].dtype == torch.float64
    x = data.features["cls_patch_mean_std"][ti].astype(np.float64)
    scale = x.std(axis=0); scale[scale == 0] = 1.
    np.testing.assert_allclose(state["feature_mean"].numpy(), x.mean(axis=0), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(state["feature_scale"].numpy(), scale, rtol=1e-12, atol=1e-12)
    assert saved["training_seed"] == (record["inner_seed"] - record["outer_fold"] + 1009*record["outer_fold"] + 997) % 2**32
    if choice["epoch"] == 0:
        np.testing.assert_array_equal(saved["canonical_epoch0_predictions"], record["predictions"][train.MODEL_NAMES[family]])
    else:
        assert saved["canonical_epoch0_predictions"] is None
    result = {"fold": record["outer_fold"], "family": family, "epoch": choice["epoch"],
        "head_weight_l2": float(torch.linalg.vector_norm(state["head.weight"]))}
    if family == "partial":
        assert torch.equal(state["feature_mean"], initial_head["feature_mean"])
        assert torch.equal(state["feature_scale"], initial_head["feature_scale"])
        delta_sq, original_sq, changed, count = 0., 0., 0, 0
        for key, tensor in state.items():
            if key.startswith("norm."):
                assert torch.equal(tensor, original[key])
            elif key.startswith("block."):
                old = original["blocks.11." + key[len("block."):]]
                delta_sq += float((tensor.double() - old.double()).square().sum())
                original_sq += float(old.double().square().sum())
                changed += int(not torch.equal(tensor, old)); count += 1
        assert count == 14 and (changed == 0 if choice["epoch"] == 0 else changed > 0)
        if choice["epoch"] == 0:
            assert all(torch.equal(state[k], initial_head[k]) for k in ("head.weight", "head.bias"))
        result.update(block_tensors=count, changed_block_tensors=changed,
            block_relative_l2_change=math.sqrt(delta_sq/original_sq), frozen_norm_exactly_unchanged=True,
            head_weight_change_l2=float(torch.linalg.vector_norm(state["head.weight"] - initial_head["head.weight"])))
    else:
        assert set(state) == {"feature_mean", "feature_scale", "head.weight", "head.bias"}
        assert choice["epoch"] == 0
    return result


def audit(seed, output_root=OUT, max_epochs=80):
    args = train.parse_args(["--seed", str(seed), "--max-epochs", str(max_epochs),
        "--output-dir", str(output_root / f"seed{seed}")])
    folder = args.output_dir
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((folder / "run_manifest.json").read_text(encoding="utf-8"))
    run_id = train.fingerprint(manifest["contract"])
    assert run_id == manifest["fingerprint"] == summary["run_fingerprint"]
    assert summary["seed"] == seed and summary["protocol"] == train.PROTOCOL
    assert summary["max_epochs"] == max_epochs and summary["inner_training_runs"] == 45 and summary["outer_refits"] == 10
    assert summary["input_files"] == manifest["contract"]["input_files"]
    hashes = verify_hashes(summary["input_files"])
    data, refs, anchor = evaluation.load_prior(args)
    train.validate_anchor_policy(anchor, refs)
    tokens, metadata = prefix.validate_cache(np, args.prefix_cache, data.ids, data.metadata, args.global_features)
    assert metadata == summary["prefix_metadata"]
    del tokens
    for r in summary["runtime"]["checkpoint_files"]:
        assert core.sha256(Path(r["cache_path"])) == r["sha256"]
    safepath = next(r["cache_path"] for r in summary["runtime"]["checkpoint_files"] if r["filename"] == "model.safetensors")
    with safe_open(safepath, framework="pt", device="cpu") as handle:
        original = {k: handle.get_tensor(k) for k in handle.keys() if k.startswith(("blocks.11.", "norm."))}
    rows = core.align_rows(core.read_rows(folder / "oof.csv", {"ID", "hardness", "fold", train.PRIMARY, train.CONTROL}), data.ids, folder)
    np.testing.assert_array_equal(core.finite_column(rows, "hardness", folder), data.y)
    np.testing.assert_array_equal(core.integer_column(rows, "fold", folder), data.folds)
    np.testing.assert_array_equal(core.finite_column(rows, core.BASELINE_NAME, folder), data.baseline)
    predictions = train.add_fixed_predictions({n: core.finite_column(rows, n, folder) for n in (train.CONTROL, train.PRIMARY)}, refs)
    for name, p in predictions.items():
        np.testing.assert_array_equal(core.finite_column(rows, name, folder), p)
    np.testing.assert_allclose(predictions[train.CONTROL], refs[evaluation.DINO], rtol=0, atol=1e-7)
    records, curve_stats, fold_stats, ckpt_stats = [], [], [], []
    optimizer_steps, refit_steps = 0, 0
    for fold in range(5):
        record = train.previous.read_completed_fold(folder / f"fold{fold}", run_id, fold, data.folds)
        assert record is not None and record["ridge_anchor_matched"] is True and record["inner_seed"] == seed + fold
        ti, vi = np.flatnonzero(data.folds != fold), np.flatnonzero(data.folds == fold)
        assert record["outer_train_indices"] == ti.tolist()
        for name in (train.PRIMARY, train.CONTROL):
            np.testing.assert_array_equal(record["predictions"][name], predictions[name][vi])
        splits = list(StratifiedKFold(3, shuffle=True, random_state=seed + fold).split(data.y[ti], core.rank_groups(data.y[ti], 10)))
        sizes = [len(v) for _, v in splits]
        assert sizes == record["inner_validation_sizes"]
        members = [{"outer_fold": fold, "inner_fold": j, "role": role, "ID": str(data.ids[i])}
            for j, (it, iv) in enumerate(splits) for role, indices in (("train", ti[it]), ("valid", ti[iv])) for i in indices]
        assert record["memberships"] == members
        curves = {}
        for config in train.CONFIGS:
            runs = [[r for r in record["inner_curves"] if r["config"] == config["name"] and r["inner_fold"] == j] for j in range(3)]
            for j, run in enumerate(runs):
                assert [r["epoch"] for r in run] == list(range(max_epochs + 1))
                assert abs(run[0]["valid_mse"] - anchor["selections"][fold]["winner"]["inner_fold_mse"][j]) < 1e-8
                for r in run:
                    assert r["optimizer_steps"] == int(r["epoch"] > 0) and np.isfinite(r["valid_mse"]) and r["valid_mse"] >= 0
                    if r["epoch"]:
                        assert 0 < r["head_lr"] <= .01 and r["l2_penalty"] >= 0
                        assert abs(r["objective_before_step"] - r["data_mse_standardized"] - r["l2_penalty"]) < 1e-12
                    optimizer_steps += r["optimizer_steps"]
            curves[config["name"]] = runs
        selected, candidates = train.select_candidates(curves, sizes, max_epochs)
        assert selected == record["selected"] and candidates == record["candidates"]
        for config in train.CONFIGS:
            cs = [c for c in candidates if c["config"] == config]
            best = min(cs, key=lambda c: (c["inner_mse"], c["epoch"]))
            sample_epochs = sorted({0, 1, best["epoch"], max_epochs, *range(20, max_epochs + 1, 20)})
            series = [{"epoch": e, "inner_rmse": math.sqrt(cs[e]["inner_mse"])} for e in sample_epochs]
            curve_stats.append({"fold": fold, "config": config["name"], "family": config["family"],
                "best_epoch_without_tie_tolerance": best["epoch"], "series": series})
        initial_head = torch.load(folder / f"fold{fold}/head_only.pt", map_location="cpu", weights_only=True)["state_dict"]
        for family in train.FAMILIES:
            refit, epoch = record["refits"][family], selected[family]["epoch"]
            assert [r["epoch"] for r in refit["curve"]] == list(range(epoch + 1))
            assert all("valid_mse" not in r and r["optimizer_steps"] == int(r["epoch"] > 0) for r in refit["curve"])
            refit_steps += epoch
            ckpt_stats.append(checkpoint_audit(folder / f"fold{fold}/{family}.pt", family, record, data, run_id, original, initial_head))
        assert len(record["initialization_checks"]) == 9
        assert max(r["epoch0_max_absolute_difference"] for r in record["initialization_checks"]) <= 1e-4
        fold_metrics = {n: core.metrics(data.y[vi], p[vi])["rmse"] for n, p in predictions.items()}
        fold_stats.append({"fold": fold, "partial_epoch": selected["partial"]["epoch"], "partial_block_lr": selected["partial"]["config"]["block_lr"],
            "partial_delta_vs_dino": fold_metrics[train.PRIMARY] - fold_metrics[evaluation.DINO],
            "blend_delta_vs_current": fold_metrics[train.FIXED] - fold_metrics[evaluation.CURRENT]})
        records.append(record)
    assert optimizer_steps == 45 * max_epochs
    assert summary["selections"] == [{k: v for k, v in r.items() if k not in ("inner_curves", "memberships", "predictions", "candidates")} for r in records]
    for key, epoch in (("epoch_zero_selections", 0), ("epoch_limit_selections", max_epochs)):
        assert summary[key] == [{"fold": r["outer_fold"], "family": f} for r in records for f in train.FAMILIES if r["selected"][f]["epoch"] == epoch]
    diagnostics, tables = evaluation.diagnose(data, predictions)
    assert diagnostics == summary["diagnostics"]
    tables.update(inner_curves=[v for r in records for v in r["inner_curves"]], inner_memberships=[v for r in records for v in r["memberships"]],
        inner_search=[{"outer_fold": r["outer_fold"], "config": c["config"]["name"], "family": c["config"]["family"],
            "epoch": c["epoch"], "inner_mse": c["inner_mse"], **{f"inner_fold{j}_mse": v for j, v in enumerate(c["inner_fold_mse"])},
            "selected": c == r["selected"][c["config"]["family"]]} for r in records for c in r["candidates"]])
    for name, values in tables.items():
        keys = list(dict.fromkeys(k for r in values for k in r))
        uniform = [{k: r.get(k, "") for k in keys} for r in values]
        with (folder / (name + ".csv")).open(encoding="utf-8-sig", newline="") as f:
            assert list(csv.DictReader(f)) == list(csv.DictReader(core.csv_text(uniform).splitlines())), name
    delta = (predictions[train.FIXED] - data.y)**2 - (predictions[evaluation.CURRENT] - data.y)**2
    worst = {n: [str(data.ids[i]) for i in np.argsort(-np.abs(predictions[n] - data.y), kind="stable")[:20]] for n in (evaluation.CURRENT, train.FIXED)}
    return {"seed": seed, "max_epochs": max_epochs, "verified_input_source_hashes": hashes, "metrics": {n: d["overall"] for n, d in diagnostics.items()},
        "elapsed_minutes": summary["elapsed_seconds_this_invocation"] / 60, "inner_optimizer_steps": optimizer_steps, "outer_optimizer_steps": refit_steps,
        "global_max_difference_recorded": summary["runtime"]["initial_global_max_difference"],
        "max_epoch0_prediction_difference": max(c["epoch0_max_absolute_difference"] for r in records for c in r["initialization_checks"]),
        "head_control_max_prediction_drift_after_training": max(c["final_prediction_max_change_from_ridge"] for r in records for c in r["initialization_checks"] if c["config"] == train.CONFIGS[0]["name"]),
        "folds": fold_stats, "curves": curve_stats, "checkpoints": ckpt_stats,
        "regions": {n: {region: d["rmse"] for region, d in diagnostics[n]["target_regions"].items()} for n in (evaluation.CURRENT, train.FIXED)},
        "noise_q4": {n: diagnostics[n]["noise_quartiles"][-1]["rmse"] for n in (evaluation.CURRENT, train.FIXED)},
        "blend_delta_squared_error_sum": float(delta.sum()), "improved_samples": int((delta < 0).sum()), "unchanged_samples": int((delta == 0).sum()),
        "partial_dino_residual_correlation": core.correlation(predictions[train.PRIMARY] - data.y, predictions[evaluation.DINO] - data.y),
        "worst20": worst, "worst20_overlap": len(set(worst[evaluation.CURRENT]) & set(worst[train.FIXED])),
        "delta_squared_error_by_id": {str(ident): float(delta[i]) for i, ident in enumerate(data.ids)},
        "largest_improvements": [{"ID": str(data.ids[i]), "delta_sse": float(delta[i])} for i in np.argsort(delta)[:5]],
        "largest_worsenings": [{"ID": str(data.ids[i]), "delta_sse": float(delta[i])} for i in np.argsort(delta)[-5:]]}


def compare_previous(seed, current_root, previous_root):
    """Verify the longer run changes only the cap and reproduces the old prefix."""
    folders = [root / f"seed{seed}" for root in (previous_root, current_root)]
    summaries = [json.loads((f / "summary.json").read_text(encoding="utf-8")) for f in folders]
    manifests = [json.loads((f / "run_manifest.json").read_text(encoding="utf-8")) for f in folders]
    for s, m in zip(summaries, manifests):
        assert s["run_fingerprint"] == m["fingerprint"] == train.fingerprint(m["contract"])
    old_cap, new_cap = [s["max_epochs"] for s in summaries]
    assert 0 < old_cap < new_cap
    assert {k: v for k, v in manifests[0]["contract"].items() if k != "max_epochs"} == {
        k: v for k, v in manifests[1]["contract"].items() if k != "max_epochs"}
    oofs = [core.read_rows(f / "oof.csv", {"ID", "hardness", "fold", train.PRIMARY, train.FIXED}) for f in folders]
    assert [[r[k] for k in ("ID", "hardness", "fold")] for r in oofs[0]] == [[r[k] for k in ("ID", "hardness", "fold")] for r in oofs[1]]
    folds = np.array([int(r["fold"]) for r in oofs[0]])
    y = np.array([float(r["hardness"]) for r in oofs[0]])
    predictions = [{n: np.array([float(r[n]) for r in rows]) for n in (train.PRIMARY, train.CONTROL, train.FIXED)} for rows in oofs]
    changes, curve_rows, identical_states = [], 0, 0
    for fold in range(5):
        records = [train.previous.read_completed_fold(f / f"fold{fold}", m["fingerprint"], fold, folds) for f, m in zip(folders, manifests)]
        a, b = records
        assert a["memberships"] == b["memberships"]
        truncated = [r for r in b["inner_curves"] if r["epoch"] <= old_cap]
        assert a["inner_curves"] == truncated, (seed, fold, "early training curves differ")
        curve_rows += len(truncated)
        for family in train.FAMILIES:
            old, new = a["selected"][family], b["selected"][family]
            saved = [torch.load(f / f"fold{fold}/{family}.pt", map_location="cpu", weights_only=True) for f in folders]
            same_state = all(torch.equal(saved[0]["state_dict"][k], v) for k, v in saved[1]["state_dict"].items())
            identical_states += int(same_state)
            if old == new:
                assert same_state
                assert a["predictions"][train.MODEL_NAMES[family]] == b["predictions"][train.MODEL_NAMES[family]]
            else:
                idx = folds == fold
                changes.append({"fold": fold, "family": family, "old_epoch": old["epoch"], "new_epoch": new["epoch"],
                    "old_inner_rmse": math.sqrt(old["inner_mse"]), "new_inner_rmse": math.sqrt(new["inner_mse"]),
                    "old_train_rmse": a["refits"][family]["train_rmse"], "new_train_rmse": b["refits"][family]["train_rmse"],
                    "outer_rmse": {n: [core.metrics(y[idx], p[n][idx])["rmse"] for p in predictions] for n in (train.PRIMARY, train.FIXED)}})
    return {"previous_max_epochs": old_cap, "new_max_epochs": new_cap, "only_contract_change_is_epoch_cap": True,
        "identical_early_curve_rows": curve_rows, "identical_checkpoint_states": identical_states, "selection_changes": changes,
        "metrics": {n: {"previous_rmse": core.metrics(y, predictions[0][n])["rmse"],
            "new_rmse": core.metrics(y, predictions[1][n])["rmse"],
            "delta_rmse": core.metrics(y, predictions[1][n])["rmse"] - core.metrics(y, predictions[0][n])["rmse"],
            "changed_predictions": int(np.count_nonzero(predictions[0][n] != predictions[1][n]))} for n in predictions[0]}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-dir", type=Path, default=OUT)
    parser.add_argument("--max-epochs", type=int, default=80)
    parser.add_argument("--compare-with", type=Path)
    cli = parser.parse_args()
    if cli.max_epochs < 1:
        parser.error("max-epochs must be positive")
    output_root = cli.reports_dir.resolve()
    roots = [output_root] + ([cli.compare_with.resolve()] if cli.compare_with else [])
    files = [p for root in roots for seed in (42, 2026) for p in (root / f"seed{seed}").rglob("*") if p.is_file()]
    files += [prefix.ROOT / f"reports/dinov2_partial/prefix/{name}" for name in ("tokens.npy", "metadata.json")]
    before = {str(p): core.sha256(p) for p in files}
    with mock.patch.object(train, "TorchRunner", side_effect=AssertionError("No training")), \
         mock.patch.object(train, "fit_ridge", side_effect=AssertionError("No fitting")), \
         mock.patch.object(core, "make_estimator", side_effect=AssertionError("No estimator")), \
         mock.patch.object(prefix.base, "build_model", side_effect=AssertionError("No model or forward")):
        seeds = {str(s): audit(s, output_root, cli.max_epochs) for s in (42, 2026)}
        comparisons = {str(s): compare_previous(s, output_root, cli.compare_with.resolve()) for s in (42, 2026)} if cli.compare_with else None
    assert before == {str(p): core.sha256(p) for p in files}
    common = set(seeds["42"]["worst20"][evaluation.CURRENT]) & set(seeds["2026"]["worst20"][evaluation.CURRENT])
    repeated = {"previous_common": sorted(common),
        "improved_both": sorted(i for i in common if all(s["delta_squared_error_by_id"][i] < 0 for s in seeds.values())),
        "still_worst20_both": sorted(common & set(seeds["42"]["worst20"][train.FIXED]) & set(seeds["2026"]["worst20"][train.FIXED]))}
    out = {"scope": "Stored predictions/curves/CPU checkpoint tensors; no fit/forward/GPU/download", "original_files_preserved": before,
        "seeds": seeds, "repeated_failures": repeated, "comparison_with_previous": comparisons}
    train.write_json(output_root / "review_results.json", out)
    print(f"Verified {len(files)} original files and 20 current checkpoints. No fitting or forward.")
    for seed, s in seeds.items():
        print(json.dumps({k: v for k, v in s.items() if k not in ("curves", "checkpoints", "metrics", "delta_squared_error_by_id", "worst20")}, ensure_ascii=False))
        print(json.dumps({"seed": seed, "selected_low_lr_curves": [c for c in s["curves"] if c["config"] == "ridge_partial_lr1e-6"]}))
    print(json.dumps(repeated))
    if comparisons:
        print(json.dumps(comparisons, indent=2))
