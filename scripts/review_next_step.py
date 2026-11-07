"""Descriptive diagnostics on existing OOF/features, without fitting or inference."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    paths = [ROOT / "reports/benchmark_features.npz", ROOT / "reports/audit_train_features.csv",
             ROOT / "reports/resnet_d4/review_results.json"]
    paths += [ROOT / f"reports/resnet_d4/seed{s}/oof.csv" for s in (42, 2026)]
    before = {str(p): sha(p) for p in paths}
    cache = np.load(paths[0], allow_pickle=False)
    ids = cache["ids"].astype(str)
    assert len(ids) == 500 and len(set(ids)) == 500
    names = cache["feature_names"].astype(str).tolist()
    x = cache["X"]
    audit = {r["ID"]: r for r in read_csv(paths[1])}
    assert set(audit) == set(ids)
    target_rows = {r["ID"]: r for r in read_csv(paths[3])}
    y = np.array([float(target_rows[i]["hardness"]) for i in ids])
    # The older pandas-written descriptive audit rounded some CSV floats.
    np.testing.assert_allclose([float(audit[i]["hardness"]) for i in ids], y, rtol=0, atol=1e-10)
    noise = np.array([float(audit[i]["haar_hh_noise_proxy"]) for i in ids])
    groups = {"noise_proxy": noise, "orientation_anisotropy": x[:, names.index("orientation_anisotropy")],
              "orientation_sin2": x[:, names.index("orientation_sin2")]}
    rank = np.argsort(np.argsort(y, kind="stable"), kind="stable")
    tails = (rank < 50) | (rank >= 450)
    seeds, errors = {}, {}
    for seed, path in zip((42, 2026), paths[3:]):
        by_id = {r["ID"]: r for r in read_csv(path)}
        assert set(by_id) == set(ids)
        np.testing.assert_array_equal([float(by_id[i]["hardness"]) for i in ids], y)
        p = np.array([float(by_id[i]["reference_dino_d4_fixed_blend"]) for i in ids])
        e = p - y
        errors[seed] = e
        worst = set(np.argsort(-e ** 2, kind="stable")[:20])
        group_results = {}
        for name, values in groups.items():
            ranks = np.argsort(np.argsort(values, kind="stable"), kind="stable")
            records = []
            for quartile in range(4):
                mask = ranks // 125 == quartile
                records.append({"quartile": quartile + 1, "n": int(mask.sum()),
                    "rmse": float(np.sqrt(np.mean(e[mask] ** 2))), "tail_n": int(np.sum(mask & tails)),
                    "tail_rmse": float(np.sqrt(np.mean(e[mask & tails] ** 2))),
                    "worst20_n": len(worst & set(np.flatnonzero(mask))),
                    "signed_bias": float(e[mask].mean())})
            group_results[name] = records
        seeds[str(seed)] = {"rmse": float(np.sqrt(np.mean(e ** 2))),
            "prediction_std": float(p.std()), "target_std": float(y.std()),
            "residual_prediction_correlation": float(np.corrcoef(e, p)[0, 1]),
            "tail_sse_fraction": float(np.sum(e[tails] ** 2) / np.sum(e ** 2)),
            "tail_n_in_worst20": sum(bool(tails[i]) for i in worst), "groups": group_results}
    result = {"scope": "Post-hoc descriptive arithmetic only: no model fit, new feature extraction, calibration, GPU, images, test, or downloads.",
        "reference": "hand25 + rawResNet25 + fixedDINO_D4_50", "input_sha256": before,
        "review_script_sha256": sha(Path(__file__)), "seeds": seeds,
        "cross_seed_residual_correlation": float(np.corrcoef(errors[42], errors[2026])[0, 1]),
        "cross_seed_prediction_difference_rms": float(np.sqrt(np.mean((errors[42] - errors[2026]) ** 2))),
        "limitations": "Same500 reused; groups selected after earlier diagnostics and confounded by target composition; these associations establish neither causes nor an irreducible error floor."}
    assert before == {str(p): sha(p) for p in paths}
    folder = ROOT / "reports/next_step_review"
    folder.mkdir(exist_ok=True)
    (folder / "diagnostics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in ("input_sha256",)}, indent=2))


if __name__ == "__main__":
    main()
