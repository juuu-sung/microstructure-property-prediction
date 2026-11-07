"""Frozen local-structure/ResNet/DINO-D4 submission pipeline, run by the user.

prepare: freeze inputs/configuration, without fitting or feature extraction.
train: choose alpha within train only, refit on all train, export numeric heads.
extract: offline frozen test features, after train-anchor reproducibility checks.
predict: apply saved heads, preserve sample_submission ID order, write local CSV.
No uploading, test fitting, clipping, weight search, or network model downloads.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import re
import sys
import time

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

if __package__:
    from . import evaluate_local_structure_confirmation as confirmation
else:
    import evaluate_local_structure_confirmation as confirmation

core, np, ROOT = confirmation.core, confirmation.np, confirmation.ROOT
local, resnet, dino = confirmation.local, confirmation.resnet, confirmation.dino
common = local.common
HAND, RESNET, DINO = confirmation.AUGMENTED, confirmation.RESNET, confirmation.DINO
COMPONENTS = (HAND, RESNET, DINO)
WIDTHS = {HAND: 185, RESNET: 1792, DINO: 1152}
GRIDS = {n: confirmation.GRIDS[n] for n in COMPONENTS}
WEIGHTS = {HAND: .25, RESNET: .25, DINO: .5}
SEED = 31415
ANCHORS = 16
PLAN_NAME = "plan.json"
PROTOCOL = {
    "version": 1, "components": list(COMPONENTS), "dimensions": WIDTHS, "weights": WEIGHTS,
    "alpha_candidates": {n: list(a) for n, a in GRIDS.items()}, "selection_seed": SEED,
    "selection": "3-fold StratifiedKFold, shuffle=True; train target rank deciles; pooled MSE; ties choose larger alpha",
    "final_fit": "Each selected pipeline refitted on all 500 training rows; X/y StandardScaler fit on train only",
    "fit_count": 39, "cv_score_scope": "hyperparameter-selection diagnostic, not unbiased final-model validation",
    "resnet": "original single view; multiscale means/population std; no D4",
    "dino": "D4 mean CLS/patch mean/population std; existing reflect-pad/normalization",
    "hand": "original125 plus fixed local60; original brightness and absolute orientation retained",
    "anchor": {"count": ANCHORS, "choice": "first16 train.csv IDs, no target-based selection",
        "hand_atol": 1e-10, "hand_rtol": 0., "neural_atol": 1e-5, "neural_rtol": 1e-6},
    "test_scope": "1000 unlabeled images, frozen inference only; no scaler/PCA/SSL/alpha fitting",
    "output": "sample_submission order, ID and hardness only, float64 predictions, no clipping/rounding calibration",
    "upload": False, "downloads": False,
}


def digest(value):
    return hashlib.sha256(confirmation.canonical(value).encode()).hexdigest()


def write_json(path, value):
    confirmation.write_json(path, value)


def runtime_versions():
    return {n: importlib.metadata.version(n) for n in
        ("numpy", "scipy", "scikit-learn", "Pillow", "torch", "torchvision", "timm", "huggingface-hub", "safetensors")}


def sources():
    refs = confirmation.source_references()
    for path in (Path(__file__), Path(confirmation.__file__), local.LEGACY):
        refs[path.name] = local.file_reference(path)
    return refs


def load_training():
    data = confirmation.load_features()
    confirmation.validate_plan(ROOT / "reports/local_structure_confirmation", data)
    common.ensure_inputs_unchanged(ROOT/"open", {"train_csv_sha256": data.metadata["resnet"]["train_csv_sha256"],
        "images": data.metadata["resnet"]["training_image_manifest"]})
    return data


def train_binding(data):
    return {"rows": len(data.ids), "input_files": data.provenance, "metadata": data.metadata,
        "ids_sha256": common.sha256_array(data.ids), "target_sha256": common.sha256_array(data.y),
        "feature_hashes": {n: common.sha256_array(data.features[n]) for n in COMPONENTS}}


def submission_ids(path, expected_count=1000):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["ID", "hardness"]:
            raise ValueError("Submission must have exactly ID,hardness columns in that order")
        rows = list(reader)
    if len(rows) != expected_count or any(None in r or r.get("hardness") is None for r in rows):
        raise ValueError("Submission row count or CSV structure differs")
    ids = [r["ID"] for r in rows]
    if len(set(ids)) != len(ids) or any(not re.fullmatch(r"TEST_\d{6}", s or "") for s in ids):
        raise ValueError("Submission IDs must be unique TEST_000001-style IDs")
    # Placeholder hardness values are deliberately not used as model targets.
    return ids


def test_inventory(data_dir, expected_count=1000):
    from PIL import Image
    sample = data_dir / "sample_submission.csv"
    sample_hash = core.sha256(sample)
    ids = submission_ids(sample, expected_count)
    expected = {i+".png" for i in ids}
    found = {p.name for p in (data_dir/"test").iterdir() if p.is_file() and p.suffix.lower() == ".png"}
    if found != expected:
        raise ValueError("Test PNG inventory differs from submission IDs")
    images = []
    for ident in ids:
        path = data_dir / "test" / (ident+".png")
        with Image.open(path) as im:
            if (im.format, im.mode, im.size) != ("PNG", "L", (256, 256)):
                raise ValueError(f"Expected grayscale 256x256 PNG: {ident}")
            im.verify()
        images.append({"id": ident, "relative_path": "test/"+ident+".png",
                       "size_bytes": path.stat().st_size, "sha256": core.sha256(path)})
    if core.sha256(sample) != sample_hash:
        raise ValueError("Submission template changed during inventory")
    return {"ids": ids, "rows": len(ids), "sample_sha256": sample_hash,
            "images": images, "image_manifest_sha256": digest(images)}


def checkpoint_references():
    result = {}
    for name, module in ((RESNET, common), (DINO, dino.base)):
        files = module.cached_checkpoint_files(module.DEFAULT_REVISION)
        names = {r["filename"] for r in files}
        if "config.json" not in names or not names.intersection({"model.safetensors", "pytorch_model.bin"}):
            raise ValueError(f"Pinned offline checkpoint is missing: {name}; no download attempted")
        result[name] = {"repository": module.HF_REPO, "revision": module.DEFAULT_REVISION, "files": files}
    return result


def protected_paths(data=None):
    refs = sources()
    if data is not None:
        refs = {**refs, **data.provenance}
    return [Path(r["path"]) for r in refs.values()] + [ROOT/"open/sample_submission.csv"]


def prepare(folder, data):
    local.protect_output(folder, protected_paths(data))
    history_dir = ROOT / "reports/local_structure_confirmation"
    summaries = [confirmation.read_completed(history_dir, s, data) for s in confirmation.SEEDS]
    decision = confirmation.aggregate(summaries)
    if not decision["all_seeds_improve_both"]:
        raise ValueError("Confirmation evidence no longer matches the accepted candidate")
    inventory = test_inventory(ROOT/"open")
    if set(data.ids) & set(inventory["ids"]):
        raise ValueError("Train/test IDs overlap")
    plan = {"created_utc": datetime.now(timezone.utc).isoformat(), "protocol": PROTOCOL,
        "sources": sources(), "versions": runtime_versions(), "train": train_binding(data),
        "test": inventory, "anchor_ids": data.ids[:ANCHORS].tolist(), "checkpoints": checkpoint_references(),
        "confirmation": {"protocol": local.file_reference(history_dir/"protocol.json"),
            "summary": local.file_reference(history_dir/"confirmation_summary.json"), "decision": decision},
        "actual_fits_during_preparation": 0, "features_extracted_during_preparation": 0}
    folder.mkdir(parents=True, exist_ok=True)
    write_json(folder/PLAN_NAME, plan)
    return plan


def validate_plan(folder, data):
    plan = json.loads((folder/PLAN_NAME).read_text(encoding="utf-8"))
    if (plan["protocol"] != PROTOCOL or plan["sources"] != sources() or plan["versions"] != runtime_versions()
            or plan["train"] != train_binding(data) or plan["anchor_ids"] != data.ids[:ANCHORS].tolist()):
        raise ValueError("Frozen final protocol, sources, versions or training data changed")
    confirmation.check_references({k: plan["confirmation"][k] for k in ("protocol", "summary")})
    return plan


def validate_test_identity(plan, data_dir):
    if test_inventory(data_dir, plan["test"]["rows"]) != plan["test"]:
        raise ValueError("Test images or sample_submission changed after preparation")


def pack_head(model):
    x = model.regressor_.named_steps["standardscaler"]
    ridge = model.regressor_.named_steps["ridge"]
    y = model.transformer_
    return {"x_mean": np.asarray(x.mean_, dtype=np.float64), "x_scale": np.asarray(x.scale_, dtype=np.float64),
        "coef": np.asarray(ridge.coef_, dtype=np.float64).reshape(-1),
        "intercept": np.asarray(ridge.intercept_, dtype=np.float64).reshape(()),
        "y_mean": np.asarray(y.mean_[0], dtype=np.float64), "y_scale": np.asarray(y.scale_[0], dtype=np.float64)}


def predict_head(head, features):
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != len(head["coef"]) or not np.isfinite(x).all():
        raise ValueError("Invalid input feature matrix")
    return (((x-head["x_mean"])/head["x_scale"]) @ head["coef"] + head["intercept"])*head["y_scale"] + head["y_mean"]


def fit_heads(data, threads):
    """Accepts training data only. Test caches/IDs/values are not arguments."""
    from sklearn.model_selection import StratifiedKFold
    from threadpoolctl import threadpool_limits
    strata = core.rank_groups(data.y, 10)
    splits = list(StratifiedKFold(3, shuffle=True, random_state=SEED).split(data.y, strata))
    heads, selected, candidates = {}, {}, []
    with threadpool_limits(limits=threads):
        for name in COMPONENTS:
            x = data.features[name].astype(np.float64)
            records = []
            for alpha in GRIDS[name]:
                pred = np.full(len(data.y), np.nan)
                mses = []
                for train, valid in splits:
                    model = core.make_estimator(alpha)
                    model.fit(x[train], data.y[train])
                    pred[valid] = model.predict(x[valid])
                    mses.append(float(np.mean((pred[valid]-data.y[valid])**2)))
                if not np.isfinite(pred).all():
                    raise ValueError("Incomplete training CV predictions")
                records.append({"component": name, "alpha": alpha, "pooled_mse": float(np.mean((pred-data.y)**2)),
                                "fold_mse": mses})
            winner = min(records, key=lambda r: (r["pooled_mse"], -r["alpha"]))
            model = core.make_estimator(winner["alpha"])
            model.fit(x, data.y)
            head = pack_head(model)
            exported = predict_head(head, x)
            np.testing.assert_allclose(exported, model.predict(x), rtol=0, atol=1e-10)
            heads[name] = head
            selected[name] = dict(winner)
            candidates.extend(records)
            print(f"Final {name}: train-only CV alpha={winner['alpha']:g}; refitted on {len(data.y)} rows", flush=True)
    return heads, {"selected": selected, "candidates": candidates, "cv_seed": SEED,
        "cv_validation_sizes": [len(v) for _, v in splits], "fit_count": 39,
        "cv_score_scope": PROTOCOL["cv_score_scope"], "n_train": len(data.y)}


def save_archive(folder, arrays, metadata, filename="features.npz"):
    folder.mkdir(parents=True, exist_ok=True)
    with (folder/filename).open("wb") as f:
        np.savez_compressed(f, **arrays, metadata_json=np.asarray(confirmation.canonical(metadata)))
    write_json(folder/"metadata.json", metadata)


def train(folder, data, threads):
    plan = validate_plan(folder, data)
    output = folder/"model"
    local.protect_output(output, [folder/PLAN_NAME, *protected_paths(data)])
    heads, selection = fit_heads(data, threads)
    validate_plan(folder, data)
    arrays = {n+"__"+k: v for n, head in heads.items() for k, v in head.items()}
    metadata = {"kind": "numeric_ridge_heads", "protocol_sha256": core.sha256(folder/PLAN_NAME),
        "training_binding_sha256": digest(plan["train"]), "weights": WEIGHTS, "selection": selection,
        "array_sha256": {k: common.sha256_array(v) for k, v in arrays.items()}, "test_used_for_fit": False,
        "created_utc": datetime.now(timezone.utc).isoformat()}
    save_archive(output, arrays, metadata, "heads.npz")
    write_json(output/"selection.json", selection)
    print(f"Saved final heads: {output/'heads.npz'}; test accessed for fitting=False")


def read_archive(folder, filename):
    with np.load(folder/filename, allow_pickle=False) as saved:
        meta = json.loads(saved["metadata_json"].item())
        if meta != json.loads((folder/"metadata.json").read_text(encoding="utf-8")):
            raise ValueError("Archive/sidecar mismatch")
        arrays = {k: saved[k].copy() for k in saved.files if k != "metadata_json"}
    if meta["array_sha256"] != {k: common.sha256_array(v) for k, v in arrays.items()}:
        raise ValueError("Archive array hash mismatch")
    return arrays, meta


def load_heads(folder, plan):
    arrays, meta = read_archive(folder/"model", "heads.npz")
    if (meta["kind"] != "numeric_ridge_heads" or meta["protocol_sha256"] != core.sha256(folder/PLAN_NAME)
            or meta["training_binding_sha256"] != digest(plan["train"]) or meta["weights"] != WEIGHTS
            or meta["test_used_for_fit"] is not False):
        raise ValueError("Head provenance differs")
    expected = {n+"__"+k for n in COMPONENTS for k in ("x_mean", "x_scale", "coef", "intercept", "y_mean", "y_scale")}
    if set(arrays) != expected:
        raise ValueError("Unexpected head arrays")
    heads = {}
    for n in COMPONENTS:
        head = {k.split("__", 1)[1]: v for k, v in arrays.items() if k.startswith(n+"__")}
        for k, value in head.items():
            shape = (WIDTHS[n],) if k in ("x_mean", "x_scale", "coef") else ()
            if value.dtype != np.float64 or value.shape != shape or not np.isfinite(value).all():
                raise ValueError("Invalid exported head parameter")
        if np.any(head["x_scale"] <= 0) or head["y_scale"] <= 0:
            raise ValueError("Invalid exported scale")
        if meta["selection"]["selected"][n]["alpha"] not in GRIDS[n]:
            raise ValueError("Unexpected selected alpha")
        heads[n] = head
    return heads, meta


def read_gray(path):
    from PIL import Image
    with Image.open(path) as im:
        if (im.format, im.mode, im.size) != ("PNG", "L", (256, 256)):
            raise ValueError(f"Unsupported image: {path}")
        return np.array(im, dtype=np.uint8)


def normalized_gray(path):
    gray = read_gray(path).astype(np.float32)/np.float32(255.)
    rgb = np.repeat(gray[None, :, :], 3, axis=0)
    rgb -= np.asarray(common.EXPECTED_MEAN, dtype=np.float32)[:, None, None]
    rgb /= np.asarray(common.EXPECTED_STD, dtype=np.float32)[:, None, None]
    return np.ascontiguousarray(rgb)


class Images:
    def __init__(self, data_dir, split, ids):
        if split not in ("train", "test"):
            raise ValueError("Unsupported split")
        prefix = "TRAIN" if split == "train" else "TEST"
        if any(not re.fullmatch(prefix+r"_\d{6}", str(i)) for i in ids):
            raise ValueError("IDs outside selected split")
        self.paths = [Path(data_dir)/split/(str(i)+".png") for i in ids]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        import torch
        return torch.from_numpy(normalized_gray(self.paths[index]))


def cpu_features(data_dir, split, ids):
    legacy = local.legacy_function()
    result = np.empty((len(ids), WIDTHS[HAND]), dtype=np.float64)
    for j, ident in enumerate(ids):
        path = data_dir/split/(str(ident)+".png")
        gray = read_gray(path)
        old = legacy(path)
        if list(old) != local.hand.expected_handcrafted_names():
            raise ValueError("Handcrafted schema changed")
        result[j] = np.r_[np.asarray(list(old.values())), local.local_features(gray.astype(np.float64)/255.)]
        if (j+1) % 100 == 0:
            print(f"Handcrafted {split}: {j+1}/{len(ids)}", flush=True)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite handcrafted features")
    return result


def require_anchor(actual, expected, name):
    atol, rtol = (1e-10, 0.) if name == HAND else (1e-5, 1e-6)
    if actual.shape != expected.shape or not np.isfinite(actual).all() or not np.allclose(actual, expected, atol=atol, rtol=rtol):
        raise ValueError(f"Training feature anchor mismatch: {name}; no test cache saved")
    return {"passed": True, "max_absolute_difference": float(np.max(np.abs(actual-expected))), "atol": atol, "rtol": rtol}


def neural_features(torch, model, name, dataset, device, batch_size):
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0,
        pin_memory=device.type == "cuda", generator=torch.Generator().manual_seed(SEED))
    parts, processed = [], 0
    with torch.inference_mode():
        for inputs in loader:
            inputs = inputs.to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda")
            if name == RESNET:
                _, pooled = common.pool_feature_maps(torch, model(inputs))
            elif name == DINO:
                _, pooled = dino.extract_batch(torch, model, inputs)
            else:
                raise ValueError("Unknown neural feature representation")
            parts.append(pooled.cpu().numpy())
            processed += len(inputs)
            if processed % (batch_size*10) == 0 or processed == len(dataset):
                print(f"{name}: {processed}/{len(dataset)} images", flush=True)
    features = np.concatenate(parts)
    if features.shape != (len(dataset), WIDTHS[name]) or features.dtype != np.float32 or not np.isfinite(features).all():
        raise ValueError("Invalid neural feature output")
    return features


def extract(folder, data, args):
    plan = validate_plan(folder, data)
    load_heads(folder, plan)  # Freeze train-only selection before looking at test features.
    output = folder/"test_features"
    local.protect_output(output, [folder/PLAN_NAME, folder/"model/heads.npz", *protected_paths(data)])
    validate_test_identity(plan, ROOT/"open")
    if checkpoint_references() != plan["checkpoints"]:
        raise ValueError("Offline checkpoint files changed")
    start = time.perf_counter()
    _, torch, timm, resolve_config, runtime = common.import_runtime()
    device, info = common.configure_runtime(torch, np, args)
    anchor_ids, ids = plan["anchor_ids"], plan["test"]["ids"]
    anchor_index = [list(data.ids).index(i) for i in anchor_ids]
    checks = {}
    anchor_hand = cpu_features(ROOT/"open", "train", anchor_ids)
    checks[HAND] = require_anchor(anchor_hand, data.features[HAND][anchor_index], HAND)
    features = {HAND: cpu_features(ROOT/"open", "test", ids)}
    expected_states = {RESNET: data.metadata["resnet"]["feature_state_sha256"],
                       DINO: data.metadata["dino_global"]["feature_state_sha256"]}
    for name in (RESNET, DINO):
        model = (resnet.build_model(torch, timm, resolve_config, pretrained=True) if name == RESNET else
                 dino.base.build_model(torch, timm, resolve_config, pretrained=True)[0]).to(device=device, dtype=torch.float32)
        if any(m.training for m in model.modules()) or any(p.requires_grad for p in model.parameters()):
            raise ValueError("Backbone must stay frozen and in eval mode")
        if common.feature_state_hash(model) != expected_states[name]:
            raise ValueError("Checkpoint state differs from train feature extractor")
        anchor = neural_features(torch, model, name, Images(ROOT/"open", "train", anchor_ids), device, args.batch_size)
        checks[name] = require_anchor(anchor, data.features[name][anchor_index], name)
        features[name] = neural_features(torch, model, name, Images(ROOT/"open", "test", ids), device, args.batch_size)
        if common.feature_state_hash(model) != expected_states[name]:
            raise ValueError("Frozen model state changed during inference")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    validate_test_identity(plan, ROOT/"open")
    validate_plan(folder, data)
    arrays = {"ids": np.asarray(ids), **features}
    metadata = {"kind": "frozen_test_features", "protocol_sha256": core.sha256(folder/PLAN_NAME),
        "heads_sha256": core.sha256(folder/"model/heads.npz"), "test_binding_sha256": digest(plan["test"]),
        "feature_states": expected_states, "anchor_ids": anchor_ids, "anchor_checks": checks,
        "array_sha256": {k: common.sha256_array(v) for k, v in arrays.items()}, "runtime": runtime, "device": info,
        "batch_size": args.batch_size, "elapsed_seconds": time.perf_counter()-start,
        "test_used_for_fit": False, "created_utc": datetime.now(timezone.utc).isoformat()}
    save_archive(output, arrays, metadata)
    print(f"Saved test features: {output/'features.npz'}; all train anchors matched")


def load_test_features(folder, plan):
    arrays, meta = read_archive(folder/"test_features", "features.npz")
    if (meta["kind"] != "frozen_test_features" or meta["protocol_sha256"] != core.sha256(folder/PLAN_NAME)
            or meta["heads_sha256"] != core.sha256(folder/"model/heads.npz")
            or meta["test_binding_sha256"] != digest(plan["test"]) or meta["test_used_for_fit"] is not False
            or meta["anchor_ids"] != plan["anchor_ids"]):
        raise ValueError("Test cache provenance changed")
    if set(arrays) != {"ids", *COMPONENTS} or arrays["ids"].dtype.kind != "U" or arrays["ids"].tolist() != plan["test"]["ids"]:
        raise ValueError("Test cache IDs/order/keys differ")
    expected_states = {RESNET: plan["train"]["metadata"]["resnet"]["feature_state_sha256"],
                       DINO: plan["train"]["metadata"]["dino_global"]["feature_state_sha256"]}
    if meta["feature_states"] != expected_states:
        raise ValueError("Test backbone states differ from train")
    for n in COMPONENTS:
        a = arrays[n]
        if a.shape != (plan["test"]["rows"], WIDTHS[n]) or a.dtype != (np.float64 if n == HAND else np.float32) or not np.isfinite(a).all():
            raise ValueError("Invalid test feature array")
        check = meta["anchor_checks"][n]
        if check["passed"] is not True or check["atol"] != (1e-10 if n == HAND else 1e-5) or check["rtol"] != (0 if n == HAND else 1e-6):
            raise ValueError("Missing matching train-anchor check")
    return {n: arrays[n] for n in COMPONENTS}, meta


def blended_predictions(heads, features):
    components = {n: predict_head(heads[n], features[n]) for n in COMPONENTS}
    predicted = sum(WEIGHTS[n]*components[n] for n in COMPONENTS)
    if not np.isfinite(predicted).all():
        raise ValueError("Nonfinite predictions")
    return predicted, components


def submission_text(ids, predictions):
    predictions = np.asarray(predictions)
    if predictions.shape != (len(ids),) or not np.isfinite(predictions).all() or len(set(ids)) != len(ids):
        raise ValueError("Invalid submission predictions/IDs")
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer)
    writer.writerow(["ID", "hardness"])
    writer.writerows((ident, repr(float(p))) for ident, p in zip(ids, predictions))
    return buffer.getvalue()


def verify_submission(path, ids, predictions):
    if submission_ids(path, len(ids)) != list(ids):
        raise ValueError("Submission IDs/order changed")
    rows = core.read_rows(path, {"ID", "hardness"})
    actual = core.finite_column(rows, "hardness", path)
    if not np.array_equal(actual, predictions):
        raise ValueError("CSV prediction round-trip differs")


def predict(folder, data):
    plan = validate_plan(folder, data)
    output = folder/"submission"
    local.protect_output(output, [folder/PLAN_NAME, folder/"model/heads.npz", folder/"test_features/features.npz", *protected_paths(data)])
    validate_test_identity(plan, ROOT/"open")
    heads, _ = load_heads(folder, plan)
    features, _ = load_test_features(folder, plan)
    values, components = blended_predictions(heads, features)
    output.mkdir(parents=True, exist_ok=True)
    path = output/"submission.csv"
    path.write_text(submission_text(plan["test"]["ids"], values), encoding="utf-8", newline="")
    verify_submission(path, plan["test"]["ids"], values)
    rows = [{"ID": ident, **{n: float(p[i]) for n, p in components.items()}, "hardness": float(values[i])}
            for i, ident in enumerate(plan["test"]["ids"])]
    (output/"component_predictions.csv").write_text(core.csv_text(rows), encoding="utf-8", newline="")
    report = {"protocol_sha256": core.sha256(folder/PLAN_NAME), "rows": len(values), "columns": ["ID", "hardness"],
        "sample_order_matches": True, "all_finite": True, "clipping_or_calibration": False, "uploaded": False,
        "prediction_min": float(values.min()), "prediction_max": float(values.max()),
        "prediction_mean": float(values.mean()), "prediction_std": float(values.std()),
        "heads_sha256": core.sha256(folder/"model/heads.npz"), "test_features_sha256": core.sha256(folder/"test_features/features.npz"),
        "submission_sha256": core.sha256(path), "created_utc": datetime.now(timezone.utc).isoformat()}
    write_json(output/"validation.json", report)
    print(f"Submission ready: {path}; rows={len(values)}; ID order/finite values/CSV round-trip verified")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=("prepare", "validate", "train", "extract", "predict"))
    p.add_argument("--output-dir", type=Path, default=ROOT/"reports/final_ensemble")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--threads", type=int, default=2)
    args = p.parse_args(argv)
    args.seed = SEED
    if args.batch_size < 1 or args.threads < 1:
        p.error("batch-size and threads must be positive")
    return args


def main(argv=None):
    args = parse_args(argv)
    data = load_training()
    if args.stage == "prepare":
        plan = prepare(args.output_dir, data)
        print(f"Prepared final pipeline: train={len(data.ids)}, test={plan['test']['rows']}; fits=0, features_extracted=0")
    elif args.stage == "validate":
        plan = validate_plan(args.output_dir, data)
        validate_test_identity(plan, ROOT/"open")
        if checkpoint_references() != plan["checkpoints"]:
            raise ValueError("Offline checkpoints changed")
        print("Final plan, train caches, test inventory and offline checkpoints valid; no fit/inference/output")
    elif args.stage == "train":
        train(args.output_dir, data, args.threads)
    elif args.stage == "extract":
        extract(args.output_dir, data, args)
    elif args.stage == "predict":
        predict(args.output_dir, data)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError, RuntimeError) as exc:
        print(f"Final pipeline stopped: {exc}", file=sys.stderr)
        raise SystemExit(2)
