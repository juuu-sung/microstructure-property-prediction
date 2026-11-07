"""User-run CPU train-only local tensor extraction; no model fitting or test.

The 60-dimensional specification is fixed before any score inspection. Original
125 features are re-extracted for agreement, then the exact old cache is retained.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
from scipy import ndimage

if __package__:
    from . import evaluate_feature_fusion as hand
    from . import extract_frozen_resnet as common
else:
    import evaluate_feature_fusion as hand
    import extract_frozen_resnet as common

ROOT = common.ROOT
RAW, LOCAL, AUGMENTED = "hand_original", "local_structure", "hand_plus_structure"
SCALES = ((0.6, 2.0), (1.2, 4.0), (2.4, 8.0))
EPS = 1e-12
MATCH_ATOL = 1e-10
LEGACY = ROOT / "scripts/benchmark_models.py"


def feature_names():
    names = []
    for i in range(3):
        names += [f"tensor{i}_{family}_{stat}" for family in ("energy", "coherence")
                  for stat in ("mean", "std", "q10", "q50", "q90")]
        names += [f"tensor{i}_{name}" for name in
                  ("cos2", "sin2", "cos4", "sin4", "alignment2", "alignment4", "coherence_above0p5", "coherence_above0p8")]
    for i in range(2):
        names += [f"cross{i}_{name}" for name in ("log_mean_energy_ratio", "coherence_change_rms", "axis_agreement")]
    return names


PROTOCOL = {
    "version": 1, "input": "256x256 grayscale uint8 / 255, float64; no resize, D4, crop, or per-image contrast normalization",
    "scale_pairs_sigma_rho": [list(x) for x in SCALES], "gaussian_mode": "reflect", "gaussian_truncate": 4.0,
    "gradient": "Sobel x/y of Gaussian_sigma(image), divided by 8",
    "tensor": "Gaussian_rho(gx*gx), Gaussian_rho(gy*gy), Gaussian_rho(gx*gy)",
    "energy": "T=Jxx+Jyy", "coherence": "D/(T+epsilon); D=sqrt((Jxx-Jyy)^2+4*Jxy^2)",
    "direction": "gradient-normal axis theta=0.5*atan2(2Jxy,Jxx-Jyy); absolute image coordinates retained",
    "direction_moments": "sum(D*cos(k*theta))/sum(T), sum(D*sin(k*theta))/sum(T), k=2,4; alignment=hypot(moments)",
    "energy_stats": "unweighted mean, population std, linear percentiles10/50/90",
    "coherence_stats": "energy-weighted mean, population std, midpoint-CDF quantiles10/50/90, and mass fractions >0.5/>0.8",
    "cross_scale": "adjacent coarse/fine log mean-energy ratio; sqrt(Tfine*Tcoarse)-weighted coherence-change RMS; sqrt(Dfine*Dcoarse)-weighted cos(2*(thetafine-thetacoarse))",
    "epsilon": EPS, "zero_mass": "if total weight<=epsilon*pixel_count, weighted summaries are zero",
    "features_per_scale": 18, "cross_scale_features": 6, "local_dimensions": 60, "total_dimensions": 185,
    "feature_names": feature_names(), "legacy_function": "unchanged benchmark_models.features definition only; no module setup or main executed",
    "legacy_agreement_atol": MATCH_ATOL, "legacy_agreement_rtol": 0.0,
    "labels_used_for_features": False, "test_accessed": False,
}


def source_hashes():
    return {name: common.sha256_file(path) for name, path in {
        "extractor": __file__, "legacy_features": LEGACY, "schema_validator": hand.__file__, "inventory_utilities": common.__file__}.items()}


def weighted_mean(values, weights):
    total = float(weights.sum())
    return float(np.sum(values * weights) / total) if total > EPS * weights.size else 0.0


def weighted_quantiles(values, weights):
    values, weights = values.ravel(), weights.ravel()
    keep = weights > 0
    values, weights = values[keep], weights[keep]
    if float(weights.sum()) <= EPS * keep.size:
        return [0.0, 0.0, 0.0]
    index = np.argsort(values, kind="stable")
    values, weights = values[index], weights[index]
    cdf = (np.cumsum(weights) - 0.5 * weights) / weights.sum()
    return np.interp([0.1, 0.5, 0.9], cdf, values).tolist()


def tensor_fields(a, sigma, rho):
    smooth = ndimage.gaussian_filter(a, sigma, mode="reflect", truncate=4.0)
    gx = ndimage.sobel(smooth, axis=1, mode="reflect") / 8.0
    gy = ndimage.sobel(smooth, axis=0, mode="reflect") / 8.0
    xx, yy, xy = [ndimage.gaussian_filter(v, rho, mode="reflect", truncate=4.0)
                  for v in (gx * gx, gy * gy, gx * gy)]
    energy = xx + yy
    magnitude = np.minimum(np.hypot(xx - yy, 2 * xy), energy)
    coherence = magnitude / (energy + EPS)
    theta = 0.5 * np.arctan2(2 * xy, xx - yy)
    return energy, magnitude, coherence, theta


def local_features(a):
    a = np.asarray(a)
    if a.shape != (256, 256) or a.dtype != np.float64 or not np.isfinite(a).all() or a.min() < 0 or a.max() > 1:
        raise ValueError("Expected finite float64 grayscale 256x256 in [0,1]")
    result, fields = [], []
    for sigma, rho in SCALES:
        energy, magnitude, coherence, theta = tensor_fields(a, sigma, rho)
        fields.append((energy, magnitude, coherence, theta))
        result += [float(energy.mean()), float(energy.std()), *np.percentile(energy, [10, 50, 90]).tolist()]
        cm = weighted_mean(coherence, energy)
        result += [cm, weighted_mean((coherence - cm) ** 2, energy) ** 0.5, *weighted_quantiles(coherence, energy)]
        # D-weighted moments retain absolute gradient-normal axis orientation.
        mass = float(energy.sum())
        moments = [float(np.sum(magnitude * trig(k * theta)) / mass) if mass > EPS * energy.size else 0.0
                   for k in (2, 4) for trig in (np.cos, np.sin)]
        result += moments + [float(np.hypot(*moments[:2])), float(np.hypot(*moments[2:]))]
        result += [weighted_mean(coherence > cutoff, energy) for cutoff in (0.5, 0.8)]
    for fine, coarse in zip(fields[:-1], fields[1:]):
        tf, df, cf, af = fine
        tc, dc, cc, ac = coarse
        result += [float(np.log((tc.mean() + EPS) / (tf.mean() + EPS))),
                   weighted_mean((cc - cf) ** 2, np.sqrt(tf * tc)) ** 0.5,
                   weighted_mean(np.cos(2 * (af - ac)), np.sqrt(df * dc))]
    values = np.array(result, dtype=np.float64)
    if values.shape != (60,) or not np.isfinite(values).all():
        raise ValueError("Local structure feature schema/nonfinite error")
    return values


def legacy_function():
    """Reuse the exact legacy function, without its old .ml-deps/sys.path setup."""
    tree = ast.parse(LEGACY.read_text(encoding="utf-8"), filename=str(LEGACY))
    found = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "features"]
    if len(found) != 1 or found[0].decorator_list:
        raise ValueError("Legacy feature function changed")
    namespace = {"np": np, "ndimage": ndimage, "Image": Image, "Path": Path}
    exec(compile(ast.Module(body=found, type_ignores=[]), str(LEGACY), "exec"), namespace)
    return namespace["features"]


def file_reference(path):
    return {"path": str(path.resolve()), "sha256": common.sha256_file(path)}


def original_inputs(original_path, definition_path, ids):
    values, provenance = hand.load_handcrafted(original_path, definition_path, np.asarray(ids))
    return values, {"features": file_reference(original_path), "definition": file_reference(definition_path),
                    "aligned_array_sha256": common.sha256_array(values), "historical_limit": provenance["limitation"]}


def validate_cache(path, ids, original, original_reference, train_hash, image_hash):
    with np.load(path, allow_pickle=False) as cache:
        if set(cache.files) != {"ids", RAW, LOCAL, "feature_names", "metadata_json"}:
            raise ValueError("Unexpected cache keys or labels")
        meta = json.loads(str(cache["metadata_json"].item()))
        if meta != json.loads(path.with_name("metadata.json").read_text(encoding="utf-8")):
            raise ValueError("Cache/sidecar metadata mismatch")
        cache_ids, old, local = cache["ids"], cache[RAW], cache[LOCAL]
        if cache_ids.dtype.kind != "U" or cache_ids.shape != (len(ids),) or len(set(cache_ids)) != len(ids) or set(cache_ids) != set(ids):
            raise ValueError("Cache ID mismatch")
        if cache["feature_names"].tolist() != feature_names():
            raise ValueError("Local feature names/order mismatch")
        for name, arr, width in ((RAW, old, 125), (LOCAL, local, 60)):
            if arr.dtype != np.float64 or arr.shape != (len(ids), width) or not np.isfinite(arr).all():
                raise ValueError(f"Invalid {name} shape/dtype/values")
            if meta.get("feature_array_sha256", {}).get(name) != common.sha256_array(arr):
                raise ValueError(f"{name} array hash mismatch")
        if (meta.get("schema_version") != 1 or meta.get("rows") != len(ids) or meta.get("protocol") != PROTOCOL
                or meta.get("source_sha256") != source_hashes() or meta.get("original_reference") != original_reference
                or meta.get("train_csv_sha256") != train_hash or meta.get("image_manifest_sha256") != image_hash
                or meta.get("test_accessed") is not False or meta.get("labels_saved_in_npz") is not False):
            raise ValueError("Cache protocol/source/reference/scope mismatch")
        manifest = meta.get("training_image_manifest", [])
        if ({m.get("id") for m in manifest} != set(ids) or len(manifest) != len(ids)
                or hashlib.sha256(common.canonical_json(manifest).encode()).hexdigest() != image_hash):
            raise ValueError("Training manifest mismatch")
        check = meta.get("original_reextraction_check", {})
        difference = check.get("max_absolute_difference", float("nan"))
        if check.get("passed") is not True or check.get("atol") != MATCH_ATOL or check.get("rtol") != 0 or not 0 <= difference <= MATCH_ATOL:
            raise ValueError("Original re-extraction agreement missing/failed")
        index = {ident: i for i, ident in enumerate(cache_ids)}
        order = [index[i] for i in ids]
        if not np.array_equal(old[order], original):
            raise ValueError("Original 125 cached features changed")
        return local[order].copy(), meta


def save_cache(folder, ids, original, local, metadata):
    folder.mkdir(parents=True, exist_ok=True)
    npz_tmp = folder / f"features.{os.getpid()}.tmp.npz"
    meta_tmp = folder / f"metadata.{os.getpid()}.tmp.json"
    np.savez_compressed(npz_tmp, ids=np.asarray(ids), **{RAW: original, LOCAL: local},
                        feature_names=np.asarray(feature_names()), metadata_json=np.array(common.canonical_json(metadata)))
    meta_tmp.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(npz_tmp, folder / "features.npz")
    os.replace(meta_tmp, folder / "metadata.json")


def protect_output(folder, inputs):
    resolved = folder.resolve()
    if resolved.is_relative_to((ROOT / "open").resolve()) or any(p.resolve().is_relative_to(resolved) for p in inputs):
        raise ValueError("Output overlaps protected inputs")
    if folder.exists() and (not folder.is_dir() or any(folder.iterdir())):
        raise ValueError("Output already exists; use a new empty --output-dir")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=ROOT / "open")
    p.add_argument("--original-features", type=Path, default=ROOT / "reports/benchmark_features.npz")
    p.add_argument("--definition", type=Path, default=ROOT / "reports/benchmark_seed42.json")
    p.add_argument("--output-dir", type=Path, default=ROOT / "reports/local_structure/features")
    p.add_argument("--preflight", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if not args.preflight:
        protect_output(args.output_dir, [args.original_features, args.definition, args.data_dir / "train.csv", LEGACY, Path(__file__)])
    start = time.perf_counter()
    sources = source_hashes()
    inventory = common.training_inventory(args.data_dir)
    original, reference = original_inputs(args.original_features, args.definition, inventory["ids"])
    if args.preflight:
        print(json.dumps({"status": "train_inventory_and_original_cache_valid", "rows": len(original),
                          "new_dimensions": 60, "total_dimensions": 185, "features_extracted": 0, "estimators_fitted": 0, "output_written": False}))
        return 0
    legacy = legacy_function()
    local = np.empty((len(original), 60), dtype=np.float64)
    max_difference = 0.0
    for i, ident in enumerate(inventory["ids"]):
        path = args.data_dir / "train" / f"{ident}.png"
        reproduced = legacy(path)
        if list(reproduced) != hand.expected_handcrafted_names():
            raise ValueError("Original feature schema changed")
        difference = float(np.max(np.abs(np.array(list(reproduced.values())) - original[i])))
        if not np.isfinite(difference) or difference > MATCH_ATOL:
            raise ValueError(f"Original 125-feature mismatch at {ident}: {difference}; no output saved")
        max_difference = max(max_difference, difference)
        with Image.open(path) as image:
            local[i] = local_features(np.asarray(image, dtype=np.float64) / 255.0)
        if (i + 1) % 50 == 0:
            print(f"Extracted {i+1}/{len(original)} train images; local=60; elapsed={time.perf_counter()-start:.1f}s", flush=True)
    common.ensure_inputs_unchanged(args.data_dir, inventory)
    if sources != source_hashes() or reference != original_inputs(args.original_features, args.definition, inventory["ids"])[1]:
        raise ValueError("Input/source changed during extraction")
    metadata = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(), "rows": len(original),
        "protocol": PROTOCOL, "source_sha256": sources, "original_reference": reference,
        "train_csv_sha256": inventory["train_csv_sha256"], "image_manifest_sha256": inventory["image_manifest_sha256"],
        "training_image_manifest": inventory["images"], "feature_array_sha256": {RAW: common.sha256_array(original), LOCAL: common.sha256_array(local)},
        "original_reextraction_check": {"passed": True, "max_absolute_difference": max_difference, "atol": MATCH_ATOL, "rtol": 0.0},
        "test_accessed": False, "labels_saved_in_npz": False, "elapsed_seconds": time.perf_counter()-start,
        "versions": {n: importlib.metadata.version(n) for n in ("numpy", "scipy", "Pillow")}}
    save_cache(args.output_dir, inventory["ids"], original, local, metadata)
    print(f"Saved {args.output_dir / 'features.npz'}; original=({len(original)},125), local=({len(original)},60), combined=({len(original)},185); max original difference={max_difference:.3g}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError) as exc:
        print(f"Local structure extraction stopped: {exc}", file=sys.stderr)
        raise SystemExit(2)
