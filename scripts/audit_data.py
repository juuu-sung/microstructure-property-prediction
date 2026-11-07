"""Read-only training data audit; no test pixels or targets are read."""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "open"
OUT = ROOT / "reports"


def correlations(features, target):
    result = []
    for name, values in features.items():
        a = np.asarray(values, dtype=np.float64)
        if not np.isfinite(a).all() or np.std(a) == 0:
            continue
        result.append({"feature": name,
                       "pearson": float(np.corrcoef(a, target)[0, 1]),
                       "spearman": float(np.corrcoef(pd.Series(a).rank(), pd.Series(target).rank())[0, 1])})
    return sorted(result, key=lambda x: -abs(x["pearson"]))


def features_for(arr):
    a = arr.astype(np.float64)
    gx = np.diff(a, axis=1)
    gy = np.diff(a, axis=0)
    counts = np.bincount(arr.ravel(), minlength=256)
    p = counts[counts > 0] / a.size
    result = {
        "mean": float(a.mean()), "std": float(a.std()),
        "entropy_bits": float(-(p * np.log2(p)).sum()),
        "gradient_abs_mean": float((abs(gx).mean() + abs(gy).mean()) / 2),
        "gradient_energy": float((np.square(gx).mean() + np.square(gy).mean()) / 2),
        "gradient_x_to_y": float(np.square(gx).mean() / max(np.square(gy).mean(), 1e-10)),
        "saturated_black_fraction": float((a == 0).mean()),
        "saturated_white_fraction": float((a == 255).mean()),
    }
    # Robust HH Haar estimate: exploratory image-noise proxy, also affected by
    # fine morphology and non-white noise; not a calibrated acquisition measure.
    hh = (a[::2, ::2] - a[::2, 1::2] - a[1::2, ::2] + a[1::2, 1::2]) / 2
    sigma = float(np.median(np.abs(hh - np.median(hh))) / 0.67448975)
    result["haar_hh_noise_proxy"] = sigma
    result["variance_minus_noise_proxy"] = float(np.sqrt(max(a.var() - sigma * sigma, 0)))
    lap = a[1:-1, :-2] + a[1:-1, 2:] + a[:-2, 1:-1] + a[2:, 1:-1] - 4 * a[1:-1, 1:-1]
    result["laplacian_energy"] = float(np.square(lap).mean())
    for q in (1, 5, 10, 25, 50, 75, 90, 95, 99):
        result[f"percentile_{q}"] = float(np.percentile(a, q))
    for threshold in (32, 64, 96, 128, 160, 192, 224):
        result[f"fraction_below_{threshold}"] = float((a < threshold).mean())
    centered = a - a.mean()
    variance = float(np.square(centered).mean())
    for lag in (1, 2, 4, 8, 16, 32):
        result[f"autocorr_lag_{lag}"] = float((np.mean(centered[:, :-lag] * centered[:, lag:]) + np.mean(centered[:-lag] * centered[lag:])) / (2 * max(variance, 1e-10)))
    for scale in (2, 4, 8):
        b = a.reshape(a.shape[0] // scale, scale, a.shape[1] // scale, scale).mean(axis=(1, 3))
        result[f"block_{scale}_std"] = float(b.std())
        result[f"block_{scale}_gradient"] = float((np.abs(np.diff(b, axis=0)).mean() + np.abs(np.diff(b, axis=1)).mean()) / 2)
    return result


def font(size):
    candidates = [Path("C:/Windows/Fonts/arial.ttf"), Path("C:/Windows/Fonts/malgun.ttf")]
    for f in candidates:
        if f.exists():
            return ImageFont.truetype(str(f), size)
    return ImageFont.load_default()


def sheet(df, arrays, indices, path, title, extra=None, columns=5):
    cell_w, cell_h = 276, 306
    rows = (len(indices) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * cell_w + 20, rows * cell_h + 68), "#eff2f4")
    draw = ImageDraw.Draw(canvas)
    draw.text((16, 14), title, font=font(20), fill="#101820")
    for k, i in enumerate(indices):
        x = 10 + (k % columns) * cell_w
        y = 58 + (k // columns) * cell_h
        canvas.paste(Image.fromarray(arrays[i]).convert("RGB"), (x, y))
        label = f"{df.iloc[i]['ID']}   y={df.iloc[i]['hardness']:.2f}"
        draw.text((x, y + 260), label, font=font(13), fill="#101820")
        if extra:
            draw.text((x, y + 278), extra[i], font=font(12), fill="#555555")
    canvas.save(path)


def main():
    OUT.mkdir(exist_ok=True)
    df = pd.read_csv(DATA / "train.csv")
    sample = pd.read_csv(DATA / "sample_submission.csv")
    paths = list((DATA / "train").glob("*"))
    path_map = {p.stem: p for p in paths if p.is_file()}
    image_meta, arrays, feature_rows = [], [], []
    file_groups, pixel_groups = defaultdict(list), defaultdict(list)
    for row in df.itertuples(index=False):
        path = path_map[str(row.ID)]
        with Image.open(path) as image:
            arr = np.asarray(image)
            image_meta.append({"ID": row.ID, "mode": image.mode, "shape": list(arr.shape), "dtype": str(arr.dtype), "format": image.format})
        arrays.append(arr)
        file_groups[hashlib.sha256(path.read_bytes()).hexdigest()].append(row.ID)
        pixel_groups[hashlib.sha256(arr.tobytes()).hexdigest()].append(row.ID)
        feature_rows.append(features_for(arr))
    features = pd.DataFrame(feature_rows)
    target = df["hardness"].to_numpy()
    quality_extra = {i: f"gradient={features.iloc[i]['gradient_abs_mean']:.2f}; std={features.iloc[i]['std']:.2f}" for i in range(len(df))}
    order = np.argsort(target)
    selected = order[np.linspace(0, len(order) - 1, 25).round().astype(int)].tolist()
    sheet(df, arrays, selected, OUT / "audit_target_contact_sheet.png", "TRAIN: target-ordered examples (exploratory selection; no causal inference)", quality_extra)
    quality_order = np.argsort(features["gradient_abs_mean"].to_numpy())
    quality = np.r_[quality_order[:5], quality_order[-5:]].tolist()
    sheet(df, arrays, quality, OUT / "audit_gradient_contact_sheet.png", "TRAIN: lowest / highest gradient magnitude (not a pure blur measure)", quality_extra)
    # Only file inventory and submission schema are examined for test.
    test_paths = [p for p in (DATA / "test").iterdir() if p.is_file()]
    test_stems = {p.stem for p in test_paths}
    id_numbers = pd.to_numeric(df["ID"].str.extract(r"(\d+)$", expand=False)).to_numpy()
    duplicate_targets = df[df["hardness"].duplicated(keep=False)]
    summary = {
        "scope": "All image features, plots, hashes and statistics use TRAIN ONLY. Test uses filename inventory and sample_submission schema only.",
        "training_rows": len(df), "training_columns": list(df.columns), "dtypes": {k: str(v) for k, v in df.dtypes.items()},
        "missing_values": df.isna().sum().to_dict(), "duplicate_ids": int(df["ID"].duplicated().sum()),
        "duplicate_target_row_count": len(duplicate_targets),
        "target": {k: float(v) for k, v in df["hardness"].describe(percentiles=[.01, .05, .1, .25, .5, .75, .9, .95, .99]).items()},
        "target_skew": float(df["hardness"].skew()), "target_excess_kurtosis": float(df["hardness"].kurt()),
        "constant_mean_training_rmse_descriptive_only": float(np.std(target)),
        "csv_missing_images": sorted(set(df["ID"]) - set(path_map)),
        "extra_training_images": sorted(set(path_map) - set(df["ID"])),
        "training_image_file_count": len(path_map),
        "image_formats": dict(Counter(x["format"] for x in image_meta)),
        "image_modes": dict(Counter(x["mode"] for x in image_meta)),
        "image_shapes": dict(Counter(str(x["shape"]) for x in image_meta)),
        "image_dtypes": dict(Counter(x["dtype"] for x in image_meta)),
        "exact_file_duplicate_groups": [v for v in file_groups.values() if len(v) > 1],
        "exact_pixel_duplicate_groups": [v for v in pixel_groups.values() if len(v) > 1],
        "id_target_pearson": float(np.corrcoef(id_numbers, target)[0, 1]),
        "id_target_spearman": float(np.corrcoef(pd.Series(id_numbers).rank(), df["hardness"].rank())[0, 1]),
        "training_feature_correlations_exploratory": correlations(features, target),
        "feature_summary": features.describe().to_dict(),
        "test_inventory_only": {"file_count": len(test_paths), "extensions": dict(Counter(p.suffix for p in test_paths)),
                                 "sample_submission_rows": len(sample), "sample_submission_columns": list(sample.columns),
                                 "sample_submission_duplicate_ids": int(sample["ID"].duplicated().sum()),
                                 "sample_ids_missing_test_file": sorted(set(sample["ID"]) - test_stems),
                                 "test_files_not_in_sample": sorted(test_stems - set(sample["ID"]))},
        "interpretation_limits": ["Feature correlations are exploratory associations on the whole training set, not cross-validation performance or causal evidence.",
                                  "Target-sorted samples were deliberately selected for visual inspection; apparent trends require held-out validation.",
                                  "Exact duplicate check does not establish absence of related images or latent generator families.",
                                  "Gradient magnitude depends on morphology and contrast as well as sharpness; do not equate it with imaging quality.",
                                  "Global target standard deviation is a descriptive in-sample constant baseline, not OOF RMSE."],
    }
    (OUT / "audit_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.concat([df, features], axis=1).to_csv(OUT / "audit_train_features.csv", index=False)
    pd.DataFrame(image_meta).to_csv(OUT / "audit_image_metadata.csv", index=False)
    print(json.dumps({k: summary[k] for k in ["training_rows", "target", "target_skew", "target_excess_kurtosis", "missing_values", "exact_file_duplicate_groups", "image_modes", "image_shapes", "id_target_pearson", "id_target_spearman", "test_inventory_only"]}, ensure_ascii=False, indent=2))
    print(json.dumps(summary["training_feature_correlations_exploratory"][:15], indent=2))


if __name__ == "__main__":
    main()
