"""Audit saved learning-curve outputs; never fit an estimator or read images."""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import evaluate_dinov2_learning_curve as curve

ROOT = curve.ROOT
OUT = ROOT / "reports/dinov2_learning_curve"


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def close(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-9)


def verify_hashes(obj):
    count = 0
    if isinstance(obj, dict):
        for path_key, hash_key in (("path", "sha256"), ("summary_path", "summary_sha256")):
            if path_key in obj and hash_key in obj:
                assert curve.core.sha256(Path(obj[path_key])) == obj[hash_key], obj[path_key]
                count += 1
        count += sum(verify_hashes(value) for value in obj.values())
    elif isinstance(obj, list):
        count += sum(verify_hashes(value) for value in obj)
    return count


def review(seed):
    directory = OUT / f"seed{seed}"
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    hash_count = verify_hashes(summary["input_files"])
    args = curve.parse_args(["--seed", str(seed)])
    data, references, anchor = curve.load_inputs(args.features, args.train_csv, args.folds,
        args.baseline, args.audit_features, args.previous_frozen_oof, args.previous_dino_oof, seed,
        previous_dino_summary=args.previous_dino_summary)
    expected = curve.build_memberships(data.folds, seed)
    memberships = defaultdict(list)
    for row in read_csv(directory / "memberships.csv"):
        key = tuple(int(row[k]) for k in ("outer_fold", "repeat", "train_size"))
        index = int(row["source_row_index"])
        assert row["ID"] == str(data.ids[index])
        memberships[key].append(index)
    assert set(memberships) == set(expected)
    for key in expected:
        np.testing.assert_array_equal(memberships[key], expected[key])

    oof = defaultdict(list)
    for row in read_csv(directory / "oof.csv"):
        oof[int(row["repeat"]), int(row["train_size"])].append(row)
    predictions, records, full = {}, [], {}
    assert len(oof) == 12 and len(summary["selections"]) == 60
    for key, rows in oof.items():
        assert len(rows) == 500
        assert [r["ID"] for r in rows] == list(data.ids)
        close([float(r["hardness"]) for r in rows], data.y)
        np.testing.assert_array_equal([int(r["fold"]) for r in rows], data.folds)
        prediction = np.array([float(r["prediction"]) for r in rows])
        assert np.isfinite(prediction).all()
        close([float(r["error"]) for r in rows], prediction - data.y)
        predictions[key] = prediction
    for record in summary["selections"]:
        fold, repeat, size = [record[k] for k in ("outer_fold", "repeat", "train_size")]
        indices = np.flatnonzero(data.folds == fold)
        pred = predictions[repeat, size][indices]
        assert record["inner_seed"] == seed + fold
        assert record["inner_validation_sizes"] == [len(x) for x in np.array_split(np.arange(size), 3)]
        assert record["train_metrics"]["n"] == size
        subset_y = data.y[expected[fold, repeat, size]]
        for name, val in (("mean", subset_y.mean()), ("std", subset_y.std()),
                          ("min", subset_y.min()), ("max", subset_y.max())):
            close(record["subset_target_" + name], val)
        assert {c["alpha"] for c in record["candidates"]} == set(curve.ALPHAS)
        assert len(record["candidates"]) == 6
        for candidate in record["candidates"]:
            close(candidate["inner_mse"], np.average(candidate["inner_fold_mse"],
                weights=record["inner_validation_sizes"]))
        assert record["winner"] == min(record["candidates"], key=lambda c: (c["inner_mse"], -c["alpha"]))
        actual = curve.core.metrics(data.y[indices], pred)
        for name in ("rmse", "mae", "bias_pred_minus_target", "sse"):
            close(actual[name], record["validation_metrics"][name])
        close(record["train_metrics"]["rmse"], np.sqrt(record["train_metrics"]["sse"] / size))
        enriched = {**record, "validation_indices": indices, "validation_predictions": pred}
        records.append(enriched)
        if size == 400:
            np.testing.assert_array_equal(pred, anchor["predictions"][indices])
            if repeat == 0:
                full[fold] = enriched
    anchor_check = curve.validate_anchor(full, data, anchor)
    assert anchor_check == summary["anchor_check"]
    tables = curve.summarize(data, records)
    for name, values in tables.items():
        saved = read_csv(directory / (name + ".csv"))
        regenerated = list(csv.DictReader(curve.core.csv_text(values).splitlines()))
        assert saved == regenerated, name
        if name in summary:
            assert summary[name] == values, name
    assert summary["estimator_fit_count"] == curve.planned_fit_count() == 950
    bins = curve.core.rank_groups(data.y, 10)
    groups = {"all": np.ones(500, dtype=bool), "low10": bins == 0, "high10": bins == 9,
        "tails20": (bins == 0) | (bins == 9), "middle80": (bins > 0) & (bins < 9)}
    noise = curve.core.rank_groups(data.noise, 4)
    groups.update({f"noise_q{q+1}": noise == q for q in range(4)})
    subgroup = {}
    for name, mask in groups.items():
        rmses = {size: [float(np.sqrt(np.mean((predictions[r, size][mask] - data.y[mask]) ** 2)))
                       for r in range(3)] for size in curve.TRAIN_SIZES}
        subgroup[name] = {"n": int(mask.sum()), "rmse_mean": {str(n): float(np.mean(v)) for n, v in rmses.items()},
            "delta_300_to_400_by_repeat": (np.array(rmses[400]) - rmses[300]).tolist(),
            "delta_300_to_400_mean": float(np.mean(rmses[400]) - np.mean(rmses[300]))}
    alpha_counts = {str(size): dict(Counter(str(r["winner"]["alpha"]) for r in records
        if r["train_size"] == size and not r["full_size_reused"])) for size in curve.TRAIN_SIZES}
    folded = {(r["repeat"], r["train_size"], r["outer_fold"]): r["validation_metrics"]["rmse"] for r in records}
    fold_deltas = [[folded[r, 400, f] - folded[r, 300, f] for f in range(5)] for r in range(3)]
    curve_rows = tables["curve_summary"]
    transitions = [{"from": before["train_size"], "to": after["train_size"],
        "delta_rmse": after["oof_rmse_mean"] - before["oof_rmse_mean"]}
        for before, after in zip(curve_rows, curve_rows[1:])]
    errors400 = predictions[0, 400] - data.y
    mean_sse300 = np.mean([(predictions[r, 300] - data.y)**2 for r in range(3)], axis=0)
    gain = mean_sse300 - errors400**2
    worst = np.argsort(-np.abs(errors400))[:20]
    return {"seed": seed, "validated_input_and_source_hashes": hash_count,
        "verified_membership_sets": len(memberships), "verified_oof_rows": 6000,
        "anchor_check": anchor_check, "estimator_fits_in_recorded_run": 950,
        "elapsed_seconds_in_recorded_run": summary["elapsed_seconds"],
        "curve": curve_rows, "transitions": transitions, "alpha_counts_unique_fits": alpha_counts,
        "paired_300_to_400": tables["paired_300_to_400"], "fold_deltas_300_to_400": fold_deltas,
        "fold_repeat_pairs_improved": int(np.sum(np.array(fold_deltas) < 0)), "fold_repeat_pairs_total": 15,
        "subgroups": subgroup, "train_validation_gap_at_400": curve_rows[-1]["oof_rmse_mean"] - curve_rows[-1]["train_resubstitution_rmse_mean"],
        "current_ensemble_rmse": curve.core.metrics(data.y, references[curve.references_module.CURRENT_REFERENCE])["rmse"],
        "tail_share_of_sse_400": float(np.sum(errors400[groups["tails20"]]**2) / np.sum(errors400**2)),
        "images_improved_in_squared_error_vs_300_repeat_average": int(np.sum(gain > 0)),
        "worst20_at_400": [{"ID": str(data.ids[i]), "hardness": float(data.y[i]), "prediction": float(predictions[0, 400][i]),
                            "error": float(errors400[i]), "squared_error_reduction_vs_300_repeat_mean": float(gain[i])} for i in worst]}


if __name__ == "__main__":
    source_files = [p for seed in (42, 2026) for p in (OUT / f"seed{seed}").iterdir() if p.is_file()]
    before = {str(p): curve.core.sha256(p) for p in source_files}
    results = {str(seed): review(seed) for seed in (42, 2026)}
    assert before == {str(p): curve.core.sha256(p) for p in source_files}
    output = {"scope": "Saved-output arithmetic and provenance audit only; estimator fits=0; no images, GPU, download or test data.",
        "original_run_files_preserved": before, "seeds": results,
        "interpretation_limit": "Same 500 rows, two splits, overlapping repeats; descriptive sensitivity of fixed DINO+Ridge, not independent replication or an asymptotic error estimate."}
    (OUT / "review_results.json").write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    for seed, result in results.items():
        print(json.dumps({k: result[k] for k in ("seed", "validated_input_and_source_hashes", "anchor_check", "elapsed_seconds_in_recorded_run",
            "transitions", "alpha_counts_unique_fits", "fold_repeat_pairs_improved", "subgroups", "train_validation_gap_at_400", "tail_share_of_sse_400")}, indent=2))
