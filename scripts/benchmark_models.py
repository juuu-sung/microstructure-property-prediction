"""Bounded CPU benchmark. Reads ONLY open/train.csv and open/train/*.png.

The fixed feature specification and model registry precede score inspection.
All learned preprocessing is fitted inside each training fold.
"""
from __future__ import annotations

import os
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "2"
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ".ml-deps"))
import argparse
import csv
import json
import platform
import time
from datetime import datetime, timezone
import numpy as np
from PIL import Image
import scipy
from scipy import ndimage
import sklearn
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from threadpoolctl import threadpool_limits


def features(path: Path) -> dict[str, float]:
    a = np.asarray(Image.open(path).convert("L"), dtype=np.float64) / 255.0
    f = {}
    def put(name, value):
        f[name] = float(value)
    sd = a.std()
    z = (a - a.mean()) / max(sd, 1e-8)
    for name, value in [("mean", a.mean()), ("std", sd), ("skew", (z ** 3).mean()), ("kurtosis", (z ** 4).mean())]:
        put("intensity_" + name, value)
    for q in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        put(f"intensity_q{q}", np.percentile(a, q))
    hist = np.histogram(a, bins=12, range=(0, 1))[0] / a.size
    for j, value in enumerate(hist):
        put(f"intensity_hist{j:02d}", value)
    # Observable dark regions: deliberately not labelled as physical pores.
    for threshold in [32, 64, 96, 128, 160, 192]:
        mask = a < threshold / 255.0
        labels, _ = ndimage.label(mask, structure=np.ones((3, 3)))
        sizes = np.bincount(labels.ravel())[1:]
        sizes = sizes[sizes >= 3]
        put(f"dark{threshold}_fraction", mask.mean())
        put(f"dark{threshold}_count", len(sizes) / a.size)
        for name, value in [("median", np.median(sizes) if len(sizes) else 0), ("q90", np.percentile(sizes, 90) if len(sizes) else 0), ("max", sizes.max() if len(sizes) else 0)]:
            put(f"dark{threshold}_area_{name}", np.log1p(value))
    for sigma in [0.6, 1.2, 2.4, 4.8]:
        smooth = ndimage.gaussian_filter(a, sigma, mode="reflect")
        residual = a - smooth
        gx = ndimage.sobel(smooth, axis=1, mode="reflect") / 8.0
        gy = ndimage.sobel(smooth, axis=0, mode="reflect") / 8.0
        gradient = np.hypot(gx, gy)
        for name, value in [("residual_std", residual.std()), ("residual_abs", np.abs(residual).mean()), ("gradient_mean", gradient.mean()), ("gradient_std", gradient.std()), ("gradient_q90", np.percentile(gradient, 90)), ("laplacian_abs", np.abs(ndimage.laplace(smooth)).mean())]:
            put(f"scale{sigma}_{name}", value)
        if sigma == 2.4:
            xx, yy, xy = (gx * gx).mean(), (gy * gy).mean(), (gx * gy).mean()
            total = max(xx + yy, 1e-12)
            put("orientation_anisotropy", np.sqrt((xx - yy)**2 + 4 * xy**2) / total)
            put("orientation_cos2", (xx - yy) / total)
            put("orientation_sin2", 2 * xy / total)
    # Windowed spectral summary, with radial scale and directional information.
    window = np.hanning(a.shape[0])[:, None] * np.hanning(a.shape[1])[None, :]
    power = np.abs(np.fft.fft2((a - a.mean()) * window)) ** 2
    fy, fx = np.meshgrid(np.fft.fftfreq(a.shape[0]), np.fft.fftfreq(a.shape[1]), indexing="ij")
    radius = np.hypot(fx, fy)
    power[0, 0] = 0
    total_power = max(power.sum(), 1e-12)
    bands = [0, 1/128, 1/64, 1/32, 1/16, 1/8, 1/4, 1]
    for j, (lo, hi) in enumerate(zip(bands[:-1], bands[1:])):
        put(f"fft_band{j}", power[(radius >= lo) & (radius < hi)].sum() / total_power)
    put("fft_centroid", (power * radius).sum() / total_power)
    put("fft_radius_rms", np.sqrt((power * radius**2).sum() / total_power))
    theta = np.arctan2(fy, fx)
    put("fft_cos2", (power * np.cos(2 * theta)).sum() / total_power)
    put("fft_sin2", (power * np.sin(2 * theta)).sum() / total_power)
    for lag in [1, 2, 4, 8, 16, 32]:
        ac_x = (z[:, :-lag] * z[:, lag:]).mean()
        ac_y = (z[:-lag, :] * z[lag:, :]).mean()
        put(f"autocorrelation{lag}_mean", (ac_x + ac_y) / 2)
        put(f"autocorrelation{lag}_xy_difference", abs(ac_x - ac_y))
    quantized = np.clip((a * 16).astype(int), 0, 15)
    indices = np.arange(16)
    distance_matrix = indices[:, None] - indices[None, :]
    for lag in [1, 4, 12]:
        pairs_x = quantized[:, :-lag].ravel() * 16 + quantized[:, lag:].ravel()
        pairs_y = quantized[:-lag, :].ravel() * 16 + quantized[lag:, :].ravel()
        counts = np.bincount(np.concatenate([pairs_x, pairs_y]), minlength=256).reshape(16, 16)
        matrix = counts + counts.T
        matrix = matrix / max(matrix.sum(), 1)
        put(f"glcm{lag}_contrast", (matrix * distance_matrix**2).sum())
        put(f"glcm{lag}_homogeneity", (matrix / (1 + distance_matrix**2)).sum())
        put(f"glcm{lag}_energy", (matrix**2).sum())
        put(f"glcm{lag}_entropy", -(matrix * np.log(matrix + 1e-15)).sum())
    for size in [8, 16, 32, 64]:
        blocks = a.reshape(256 // size, size, 256 // size, size).mean(axis=(1, 3))
        put(f"block{size}_mean_std", blocks.std())
        put(f"block{size}_mean_q90q10", np.percentile(blocks, 90) - np.percentile(blocks, 10))
    return f


def model_registry(n_features):
    def scaled(model):
        return TransformedTargetRegressor(regressor=make_pipeline(StandardScaler(), model), transformer=StandardScaler())
    return {
        "dummy_mean": DummyRegressor(strategy="mean"),
        "ridge_alpha10": scaled(Ridge(alpha=10)),
        "ridge_alpha100": scaled(Ridge(alpha=100)),
        "svr_rbf_C3": scaled(SVR(C=3, epsilon=0.05, gamma="scale")),
        "svr_rbf_C10": scaled(SVR(C=10, epsilon=0.05, gamma="scale")),
        "kernelridge_rbf": scaled(KernelRidge(alpha=0.1, kernel="rbf", gamma=1/n_features)),
        "extratrees_leaf2": ExtraTreesRegressor(n_estimators=400, min_samples_leaf=2, max_features=1.0, n_jobs=2, random_state=42),
        "extratrees_leaf4": ExtraTreesRegressor(n_estimators=400, min_samples_leaf=4, max_features=0.7, n_jobs=2, random_state=42),
        "histgb_leaf7": HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=7, min_samples_leaf=20, l2_regularization=10, early_stopping=False, random_state=42),
    }


def rmse(y, pred):
    return float(np.sqrt(np.mean((y - pred) ** 2)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--models", nargs="*", default=None)
    args = parser.parse_args()
    start = time.perf_counter()
    output = ROOT / "reports"
    output.mkdir(exist_ok=True)
    with (ROOT / "open" / "train.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    ids = [r["ID"] for r in rows]
    y = np.array([float(r["hardness"]) for r in rows])
    assert len(ids) == 500 and len(set(ids)) == 500
    cache = output / "benchmark_features.npz"
    if cache.exists():
        data = np.load(cache)
        assert np.array_equal(data["ids"], ids)
        X, feature_names = data["X"], list(data["feature_names"])
    else:
        records = []
        for i, sample_id in enumerate(ids):
            records.append(features(ROOT / "open" / "train" / f"{sample_id}.png"))
            if (i + 1) % 100 == 0:
                print(f"Extracted {i + 1}/{len(ids)} train images", flush=True)
        feature_names = list(records[0])
        X = np.array([[record[key] for key in feature_names] for record in records])
        np.savez_compressed(cache, ids=ids, X=X, feature_names=feature_names)
    assert np.isfinite(X).all()
    # Ten target quantile bins are used only to distribute target ranges across folds.
    rank = np.argsort(np.argsort(y, kind="stable"), kind="stable")
    strata = rank * 10 // len(y)
    splits = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed).split(X, strata))
    fold_ids = np.full(len(ids), -1, dtype=int)
    for fold, (_, valid) in enumerate(splits):
        fold_ids[valid] = fold
    fold_path = output / ("folds.csv" if args.seed == 42 else f"benchmark_folds_seed{args.seed}.csv")
    with fold_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ID", "fold", "seed", "target_quantile"])
        writer.writerows(zip(ids, fold_ids, [args.seed] * len(ids), strata))
    registry = model_registry(X.shape[1])
    names = args.models or list(registry)
    assert set(names) <= registry.keys()
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "n_train": len(y), "n_features": X.shape[1], "seed": args.seed,
        "validation": "5-fold shuffled target-decile stratification; no test reads; identical folds across models",
        "learned_preprocessing": "feature and target StandardScaler fitted within each training fold only",
        "feature_definition": feature_names,
        "data_scope": "open/train.csv and open/train/*.png only; no outlier removal",
        "limitations": ["Single small dataset; model selection on these folds gives optimistic selected-model estimates.", "No hidden test/public/private score evaluated.", "No latent-generator grouping proven; exact duplicates reportedly absent.", "Gradient and dark-region features are observable image statistics, not validated physical segmentation."],
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__},
        "models": {}
    }
    predictions = {}
    for name in names:
        model_start = time.perf_counter()
        oof = np.empty_like(y)
        fold_scores, train_scores = [], []
        with threadpool_limits(limits=2):
            for train, valid in splits:
                model = clone(registry[name])
                model.fit(X[train], y[train])
                oof[valid] = model.predict(X[valid])
                fold_scores.append(rmse(y[valid], oof[valid]))
                train_scores.append(rmse(y[train], model.predict(X[train])))
        result = {"oof_rmse": rmse(y, oof), "fold_rmse": fold_scores,
                  "fold_rmse_std": float(np.std(fold_scores)), "train_rmse": train_scores,
                  "seconds": time.perf_counter() - model_start,
                  "low_target_decile_rmse": rmse(y[strata == 0], oof[strata == 0]),
                  "high_target_decile_rmse": rmse(y[strata == 9], oof[strata == 9]),
                  "configuration": str(registry[name])}
        summary["models"][name] = result
        predictions[name] = oof
        print(f"{name}: OOF RMSE={result['oof_rmse']:.5f}; folds={np.round(fold_scores, 4).tolist()}; elapsed={result['seconds']:.1f}s", flush=True)
        summary["elapsed_seconds"] = time.perf_counter() - start
        (output / f"benchmark_seed{args.seed}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output / f"benchmark_oof_seed{args.seed}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ID", "hardness", "fold"] + list(predictions))
        for i, sample_id in enumerate(ids):
            writer.writerow([sample_id, y[i], fold_ids[i]] + [pred[i] for pred in predictions.values()])
    errors = np.stack([pred-y for pred in predictions.values()], axis=1)
    summary["residual_correlation"] = {"model_order": list(predictions), "matrix": np.corrcoef(errors, rowvar=False).tolist()}
    summary["elapsed_seconds"] = time.perf_counter() - start
    (output / f"benchmark_seed{args.seed}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Completed in {summary['elapsed_seconds']:.1f}s; {X.shape[1]} features; no test data read", flush=True)


if __name__ == "__main__":
    main()
