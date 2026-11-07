"""Read existing train-only OOF artifacts; no fitting or prediction adjustment."""
import csv
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def read_csv(name):
    with (REPORTS / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def rank_groups(values, n):
    return np.argsort(np.argsort(values, kind="stable"), kind="stable") * n // len(values)


def group_metrics(y, pred, mask, total_sse):
    error = pred[mask] - y[mask]
    return {
        "n": int(mask.sum()),
        "target_mean": float(y[mask].mean()),
        "target_min": float(y[mask].min()),
        "target_max": float(y[mask].max()),
        "prediction_mean": float(pred[mask].mean()),
        "signed_error_mean_pred_minus_target": float(error.mean()),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "mae": float(np.mean(np.abs(error))),
        "sse_fraction": float(np.sum(error ** 2) / total_sse),
    }


def main():
    audit = {row["ID"]: row for row in read_csv("audit_train_features.csv")}
    output = {
        "scope": "Existing train-only OOF and audit feature CSVs. No training, calibration, clipping, image reading, or test use.",
        "model": "ridge_alpha100",
        "residual_sign": "prediction minus true hardness; positive means overprediction",
        "groups": "Equal-count target deciles and noise-proxy quartiles by stable rank. Descriptive only.",
        "noise_proxy": "haar_hh_noise_proxy; proxy can mix genuine fine structure and image noise, not ground truth image quality",
        "limitations": ["Same 500 original images under both seeds: agreement is not independent validation.", "Model was selected from earlier CV comparisons; post-hoc diagnostics are hypothesis generation.", "Target-defined bins are error audits, not inference-time routing labels."],
        "seeds": {},
    }
    error_vectors, top_ids = {}, {}
    reference_ids = None
    for seed in [42, 2026]:
        rows = read_csv(f"benchmark_oof_seed{seed}.csv")
        ids = [r["ID"] for r in rows]
        assert len(ids) == 500 and set(ids) == set(audit)
        if reference_ids is not None:
            assert ids == reference_ids
        reference_ids = ids
        y = np.array([float(r["hardness"]) for r in rows])
        pred = np.array([float(r["ridge_alpha100"]) for r in rows])
        noise = np.array([float(audit[i]["haar_hh_noise_proxy"]) for i in ids])
        assert np.allclose(y, [float(audit[i]["hardness"]) for i in ids])
        err = pred - y
        se = err ** 2
        target_bins, noise_bins = rank_groups(y, 10), rank_groups(noise, 4)
        worst = np.argsort(se)[-20:][::-1]
        top_ids[seed] = set(np.array(ids)[worst])
        error_vectors[seed] = err
        seed_result = {
            "overall": group_metrics(y, pred, np.ones(len(y), dtype=bool), se.sum()),
            "target_std": float(y.std()), "prediction_std": float(pred.std()),
            "prediction_to_target_std_ratio": float(pred.std() / y.std()),
            "prediction_min": float(pred.min()), "prediction_max": float(pred.max()),
            "prediction_outside_observed_target_range_count": int(((pred < y.min()) | (pred > y.max())).sum()),
            "residual_target_correlation": float(np.corrcoef(err, y)[0, 1]),
            "residual_noise_correlation": float(np.corrcoef(err, noise)[0, 1]),
            "squared_error_noise_correlation": float(np.corrcoef(se, noise)[0, 1]),
            "target_deciles": [], "noise_proxy_quartiles": [],
            "target_tails_combined": group_metrics(y, pred, (target_bins == 0) | (target_bins == 9), se.sum()),
            "target_middle_eight_deciles": group_metrics(y, pred, (target_bins > 0) & (target_bins < 9), se.sum()),
            "top20_sse_fraction": float(se[worst].sum() / se.sum()),
            "top20_target_tail_count": int(((target_bins[worst] == 0) | (target_bins[worst] == 9)).sum()),
            "top20": [{"ID": ids[i], "target": float(y[i]), "prediction": float(pred[i]), "signed_error": float(err[i]), "target_decile": int(target_bins[i] + 1), "noise_proxy_quartile": int(noise_bins[i] + 1)} for i in worst],
        }
        for j in range(10):
            seed_result["target_deciles"].append({"decile": j + 1, **group_metrics(y, pred, target_bins == j, se.sum())})
        for j in range(4):
            mask = noise_bins == j
            seed_result["noise_proxy_quartiles"].append({"quartile": j + 1, "noise_proxy_min": float(noise[mask].min()), "noise_proxy_max": float(noise[mask].max()), **group_metrics(y, pred, mask, se.sum())})
        output["seeds"][str(seed)] = seed_result
    overlap = top_ids[42] & top_ids[2026]
    output["cross_seed"] = {"residual_correlation": float(np.corrcoef(error_vectors[42], error_vectors[2026])[0, 1]), "top20_overlap_count": len(overlap), "top20_overlap_fraction": len(overlap) / 20, "top20_overlap_ids": sorted(overlap)}
    (REPORTS / "training_diagnostics.json").write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    compact = {"cross_seed": output["cross_seed"]}
    for seed, result in output["seeds"].items():
        compact[seed] = {k: result[k] for k in ["target_std", "prediction_std", "prediction_to_target_std_ratio", "prediction_min", "prediction_max", "target_tails_combined", "target_middle_eight_deciles", "top20_sse_fraction", "top20_target_tail_count", "noise_proxy_quartiles"]}
        compact[seed]["deciles"] = [{k: row[k] for k in ["decile", "signed_error_mean_pred_minus_target", "rmse", "sse_fraction"]} for row in result["target_deciles"]]
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
