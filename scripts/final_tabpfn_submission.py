"""User-run final TabPFN-v2 context and fixed50:50 submission candidate.

Built with PriorLabs-TabPFN. prepare/validate never fit or extract features.
run performs train-only selection, conditional morphology extraction and inference.
Original submission is read-only. No upload, calibration or new tuning.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import os
from pathlib import Path
import time

import numpy as np
from sklearn.model_selection import StratifiedKFold

if __package__:
    from . import evaluate_tabpfn_v2_confirmation as confirmation
    from . import final_submission as original
else:
    import evaluate_tabpfn_v2_confirmation as confirmation
    import final_submission as original

t = confirmation.t
morph = t.previous.extract
ROOT = t.ROOT
BASE = ROOT/"reports/final_ensemble"
OUT = ROOT/"reports/final_tabpfn_blend"
SEED = original.SEED  #31415, inherited from the already frozen original final pipeline.
ANCHORS = 16
BASE_SHA = "525f8806085d001a2ee570db3dab97c1d5e9ba599b8d671ee5b8e5e309d84125"
BASE_FILES = ("plan.json", "model/heads.npz", "model/metadata.json", "model/selection.json",
              "test_features/features.npz", "test_features/metadata.json", "submission/submission.csv",
              "submission/validation.json", "submission/component_predictions.csv")
PROTOCOL = {
    "version": 1, "attribution": "Built with PriorLabs-TabPFN", "selection_seed": SEED,
    "selection": "train500 only, StratifiedKFold3 target-rank deciles shuffle seed31415; pooled MSE selects185/285; exact ties185",
    "predictor_settings": "unchanged validated local TabPFN-v2,8 estimators,mean output,FP32,no TF32,offline,telemetry disabled",
    "context_seeds": "selection seed+inner_fold; final full500 seed+9; no seed ensemble",
    "cv_prediction_count": 6, "final_context_count": 1,
    "selection_score_scope": "representation-selection diagnostic; not unbiased final model or submission validation",
    "test": "all1000 queries in sample order, one predict call using all500 labeled context rows; no test labels or test fitting",
    "test_features": "reuse validated existing185; only if285 selected compute same fixed morphology100",
    "anchor": "first16 train IDs without label selection, morphology float64 atol1e-10 rtol0 before test extraction",
    "blend": {"original_submission": .5, "tabpfn": .5}, "original_submission_sha256": BASE_SHA,
    "output": "separate ID,hardness CSV in original sample order; no clipping, calibration, rounding or upload",
    "resume": "same binding/runtime only; completed selection cells/test cache/final prediction verified and reused",
}


def check_refs(refs):
    for ref in refs:
        if t.common.sha256_file(ref["path"]) != ref["sha256"]:
            raise ValueError("Frozen file changed: "+str(ref["path"]))


def train_state():
    data, features, refs, cp = confirmation.validate()
    summaries = [confirmation.read_completed(confirmation.OUT, data, refs[s], cp, s) for s in confirmation.SEEDS]
    # Bind reviewed confirmation evidence; never use it to tune final parameters.
    review = t.read_json(confirmation.OUT/"review_results.json")
    result = t.read_json(confirmation.OUT/"confirmation_summary.json")
    recomputed = confirmation.aggregate(summaries)
    recomputed["protocol_sha256"] = t.common.sha256_file(confirmation.OUT/"protocol.json")
    if review["confirmation"] != result or result != recomputed or result["all_seeds_improve"] is not True:
        raise ValueError("Reviewed confirmation evidence differs")
    legacy_data = original.load_training()
    bp = original.validate_plan(BASE, legacy_data)
    confirmation.same(data.ids, legacy_data.ids, "Final train IDs")
    confirmation.same(data.y, legacy_data.y, "Final train labels")
    confirmation.same(features["existing185"], legacy_data.features[original.HAND], "Final185 features")
    frozen = {"protocol": PROTOCOL, "source": t.file_ref(__file__), "baseline_source": t.file_ref(original.__file__),
              "confirmation_plan": t.file_ref(confirmation.OUT/"protocol.json"),
              "confirmation_payload": t.digest(cp["frozen"]),
              "confirmation_summary": t.file_ref(confirmation.OUT/"confirmation_summary.json"),
              "confirmation_review": t.file_ref(confirmation.OUT/"review_results.json"),
              "train_ids": data.ids.tolist(), "anchor_ids": data.ids[:ANCHORS].tolist(),
              "target_sha256": t.common.sha256_array(data.y),
              "train_feature_hashes": {n: t.common.sha256_array(x) for n, x in features.items()},
              "baseline_files": [t.file_ref(BASE/n) for n in BASE_FILES],
              "test_binding": bp["test"], "morphology_protocol": morph.PROTOCOL,
              "morphology_sources": morph.source_refs(), "morphology_versions": morph.versions()}
    if frozen["baseline_files"][6]["sha256"] != BASE_SHA:
        raise ValueError("Original submitted CSV changed")
    return data, features, bp, frozen


def baseline_test(bp):
    """Read/verify existing test caches and arithmetic; no neural inference or fit."""
    original.validate_test_identity(bp, ROOT/"open")
    features, _ = original.load_test_features(BASE, bp)
    heads, _ = original.load_heads(BASE, bp)
    path = BASE/"submission/submission.csv"
    if original.submission_ids(path) != bp["test"]["ids"] or t.common.sha256_file(path) != BASE_SHA:
        raise ValueError("Original submission IDs/hash changed")
    rows = t.core.read_rows(path, {"ID", "hardness"})
    values = t.core.finite_column(rows, "hardness", path)
    recomputed, _ = original.blended_predictions(heads, features)
    np.testing.assert_allclose(values, recomputed, atol=1e-10, rtol=0)
    return features[original.HAND], values


def prepare(folder=OUT):
    _, _, bp, frozen = train_state()
    baseline_test(bp)
    if (folder/"plan.json").exists():
        validate(folder)
        print("Existing final TabPFN plan matched; unchanged")
        return
    if folder.exists() and any(folder.iterdir()):
        raise ValueError("Nonempty unprepared final destination; refusing overwrite")
    folder.mkdir(parents=True, exist_ok=True)
    t.write_json(folder/"plan.json", {"frozen": frozen, "created_unix": time.time(), "fits_during_preparation": 0,
                                    "new_features_during_preparation": 0})
    print(f"Prepared {folder}; train500/test1000; no fitting or new feature extraction")


def validate(folder=OUT):
    data, features, bp, frozen = train_state()
    plan = t.read_json(folder/"plan.json")
    if plan["frozen"] != frozen or plan["fits_during_preparation"] != 0 or plan["new_features_during_preparation"] != 0:
        raise ValueError("Final TabPFN plan/inputs/settings changed")
    return data, features, bp, plan


def selection_files():
    return {"selection.json"}|{f"{r}_inner{j}/completed.json" for r in t.REPS for j in range(3)}


def compute_selection(folder, data, features, binding, predictor):
    """No test features, IDs, template values or baseline predictions as arguments."""
    splits = list(StratifiedKFold(3, shuffle=True, random_state=SEED).split(data.y, t.core.rank_groups(data.y, 10)))
    predictions = {n: np.full(len(data.y), np.nan) for n in t.REPS}
    records = []
    for rep in t.REPS:
        for j, (ti, vi) in enumerate(splits):
            p = t.run_cell(folder/f"{rep}_inner{j}", binding, rep, features[rep], data.y[ti], ti, vi, SEED+j, predictor)
            predictions[rep][vi] = p
            records.append({"representation": rep, "inner_fold": j, "train_idx": ti.tolist(), "valid_idx": vi.tolist(),
                            "valid_mse": float(np.mean((p-data.y[vi])**2))})
    choice, scores = t.select_representation(predictions, data.y)
    return {"binding": binding, "representation": choice, "pooled_mse": scores,
            "cv_prediction_sha256": {n: t.common.sha256_array(p) for n, p in predictions.items()},
            "splits": records, "seed": SEED, "final_context_seed": SEED+9, "context_rows": len(data.y),
            "test_used_for_selection": False, "cv_score_scope": PROTOCOL["selection_score_scope"]}


def missing_cell(*args):
    raise ValueError("Incomplete saved selection; run select/run first")


def read_selection(folder, data, features, plan, runtime=None):
    path = folder/"selection"
    saved = t.read_json(path/"selection.json")
    binding = saved["binding"]
    if binding["plan"] != t.digest(plan["frozen"]) or binding["stage"] != "train_only_selection" or (runtime is not None and binding["runtime"] != runtime):
        raise ValueError("Selection plan/runtime differs")
    if not t.verify_completed(path, binding) or set(t.read_json(path/"completed_evaluation.json")["files"]) != selection_files():
        raise ValueError("Selection completion missing/changed")
    with contextlib.redirect_stdout(io.StringIO()):
        rebuilt = compute_selection(path, data, features, binding, missing_cell)
    if saved != rebuilt:
        raise ValueError("Saved selection differs from pooled train-only predictions")
    return saved


def select_stage(folder, device):
    data, features, _, plan = validate(folder)
    predictor = t.Predictor(t.verify_weights(t.OUT), device)
    path = folder/"selection"
    if (path/"completed_evaluation.json").exists():
        saved = read_selection(folder, data, features, plan, predictor.runtime)
        print("Final selection verified:", saved["representation"], "; no refitting")
        return
    binding = {"plan": t.digest(plan["frozen"]), "stage": "train_only_selection", "runtime": predictor.runtime}
    saved = compute_selection(path, data, features, binding, predictor)
    validate(folder)
    t.write_json(path/"selection.json", saved)
    t.write_json(path/"completed_evaluation.json", {"binding": binding,
        "files": {n: t.common.sha256_file(path/n) for n in sorted(selection_files())}})
    print("Final train-only representation:", saved["representation"], "; pooled MSE:", saved["pooled_mse"])
    print("Selection scores are diagnostic, not unbiased validation; test not used for selection")


def morphology_anchor(actual, expected):
    if actual.shape != expected.shape or not np.isfinite(actual).all() or not np.allclose(actual, expected, atol=1e-10, rtol=0):
        raise ValueError("Morphology train anchor mismatch; refusing test extraction")
    return {"passed": True, "max_absolute_difference": float(np.max(np.abs(actual-expected))), "atol": 1e-10, "rtol": 0.}


def build_test_features(rep, base185, ids, anchor_ids, expected_anchor, image_reader, extractor):
    """Unlabeled deterministic features only;185 path does not open any image."""
    if base185.shape != (len(ids), 185) or not np.isfinite(base185).all() or rep not in t.REPS:
        raise ValueError("Invalid test feature inputs")
    if rep == "existing185":
        return base185.copy(), {"required": False, "reason": "existing validated185 cache reused"}
    anchor = np.stack([extractor(image_reader("train", ident)) for ident in anchor_ids])
    check = morphology_anchor(anchor, expected_anchor)
    additional = []
    for i, ident in enumerate(ids):
        a = np.asarray(extractor(image_reader("test", ident)), dtype=np.float64)
        if a.shape != (100,) or not np.isfinite(a).all():
            raise ValueError("Invalid test morphology row")
        additional.append(a)
        if (i+1) % 100 == 0:
            print(f"Test morphology: {i+1}/{len(ids)}", flush=True)
    return np.concatenate([base185, np.stack(additional)], axis=1), check


def test_cache_binding(folder, plan, selection):
    return {"plan": t.digest(plan["frozen"]), "selection_sha256": t.common.sha256_file(folder/"selection/selection.json"),
            "representation": selection["representation"], "source_test_features_sha256": t.common.sha256_file(BASE/"test_features/features.npz"),
            "test_identity": original.digest(plan["frozen"]["test_binding"])}


def read_test_cache(folder, plan, selection):
    path = folder/"test_features"
    binding = test_cache_binding(folder, plan, selection)
    done = t.read_json(path/"completed.json")
    if done["binding"] != binding or set(done["files"]) != {"features.npz", "metadata.json"}:
        raise ValueError("Test cache provenance differs")
    check_refs([{"path": path/n, "sha256": h} for n, h in done["files"].items()])
    arrays, meta = original.read_archive(path, "features.npz")
    ids = plan["frozen"]["test_binding"]["ids"]
    if set(arrays) != {"ids", "features"} or arrays["ids"].tolist() != ids or meta["binding"] != binding:
        raise ValueError("Test cache ID order/arrays/binding differ")
    x = arrays["features"]
    width = 185 if selection["representation"] == "existing185" else 285
    if x.shape != (len(ids), width) or x.dtype != np.float64 or not np.isfinite(x).all() or meta["test_used_for_fit"] is not False:
        raise ValueError("Invalid test feature matrix")
    if width == 285 and meta["morphology_anchor"].get("passed") is not True:
        raise ValueError("Missing morphology train anchor")
    return x


def extract_stage(folder):
    data, features, bp, plan = validate(folder)
    selection = read_selection(folder, data, features, plan)
    base185, _ = baseline_test(bp)
    path = folder/"test_features"
    if (path/"completed.json").exists():
        read_test_cache(folder, plan, selection)
        print("Final test feature cache verified; no extraction")
        return
    ids = bp["test"]["ids"]
    anchor_ids = plan["frozen"]["anchor_ids"]
    indices = [data.ids.tolist().index(i) for i in anchor_ids]
    x, anchor = build_test_features(selection["representation"], base185, ids, anchor_ids,
        features["morphology285"][indices, 185:],
        lambda split, ident: morph.read_image(ROOT/"open"/split/(str(ident)+".png")), morph.morphology_features)
    original.validate_test_identity(bp, ROOT/"open")
    validate(folder)
    arrays = {"ids": np.asarray(ids), "features": x}
    binding = test_cache_binding(folder, plan, selection)
    meta = {"binding": binding, "array_sha256": {n: t.common.sha256_array(v) for n, v in arrays.items()},
            "morphology_anchor": anchor, "test_used_for_fit": False, "labels_saved": False}
    original.save_archive(path, arrays, meta)
    t.write_json(path/"completed.json", {"binding": binding, "files": {n: t.common.sha256_file(path/n) for n in ("features.npz", "metadata.json")}})
    print(f"Final test features ready: {x.shape}; morphology anchor={anchor}")


def write_submission(output, ids, baseline, tabpfn, binding):
    if len(set(ids)) != len(ids) or baseline.shape != (len(ids),) or tabpfn.shape != baseline.shape or not np.isfinite([baseline, tabpfn]).all():
        raise ValueError("Invalid final prediction arrays/IDs")
    values = .5*(baseline+tabpfn)
    files = {"submission.csv", "component_predictions.csv", "validation.json"}
    if (output/"completed.json").exists():
        done = t.read_json(output/"completed.json")
        if done["binding"] != binding or set(done["files"]) != files:
            raise ValueError("Completed submission binding differs")
        check_refs([{"path": output/n, "sha256": h} for n, h in done["files"].items()])
        original.verify_submission(output/"submission.csv", ids, values)
        print("Final candidate submission verified; unchanged")
        return
    output.mkdir(parents=True, exist_ok=True)
    path = output/"submission.csv"
    tmp = output/"submission.csv.tmp"
    tmp.write_text(original.submission_text(ids, values), encoding="utf-8", newline="")
    os.replace(tmp, path)
    original.verify_submission(path, ids, values)
    t.write_csv(output/"component_predictions.csv", [{"ID": ident, "original": float(baseline[i]),
                "tabpfn": float(tabpfn[i]), "hardness": float(values[i])} for i, ident in enumerate(ids)])
    delta = values-baseline
    report = {"binding": binding, "rows": len(ids), "columns": ["ID", "hardness"],
              "sample_order_matches": True, "all_finite": True, "csv_round_trip_exact": True,
              "weights": PROTOCOL["blend"], "clipping_or_calibration": False, "uploaded": False,
              "prediction_min": float(values.min()), "prediction_max": float(values.max()),
              "prediction_mean": float(values.mean()), "prediction_std": float(values.std()),
              "mean_absolute_change_vs_original": float(np.abs(delta).mean()), "max_absolute_change_vs_original": float(np.abs(delta).max()),
              "submission_sha256": t.common.sha256_file(path), "attribution": PROTOCOL["attribution"]}
    t.write_json(output/"validation.json", report)
    t.write_json(output/"completed.json", {"binding": binding, "files": {n: t.common.sha256_file(output/n) for n in files}})
    print(f"Submission ready: {path}; rows={len(ids)}; original retained;50:50/ID order/finite/CSV round-trip verified")


def predict_stage(folder, device):
    data, features, bp, plan = validate(folder)
    predictor = t.Predictor(t.verify_weights(t.OUT), device)
    selection = read_selection(folder, data, features, plan, predictor.runtime)
    _, baseline = baseline_test(bp)
    test_x = read_test_cache(folder, plan, selection)
    rep = selection["representation"]
    binding = {"plan": t.digest(plan["frozen"]), "stage": "full500_test_prediction", "runtime": predictor.runtime,
               "selection_sha256": t.common.sha256_file(folder/"selection/selection.json"),
               "test_features_sha256": t.common.sha256_file(folder/"test_features/features.npz")}
    x = np.concatenate([features[rep], test_x], axis=0)
    pred = t.run_cell(folder/"inference", binding, rep, x, data.y, np.arange(len(data.y)),
                      np.arange(len(data.y), len(x)), SEED+9, predictor)
    validate(folder)
    original.validate_test_identity(bp, ROOT/"open")
    binding = {**binding, "inference_sha256": t.common.sha256_file(folder/"inference/completed.json"),
               "original_submission_sha256": t.common.sha256_file(BASE/"submission/submission.csv"), "representation": rep}
    if binding["original_submission_sha256"] != BASE_SHA:
        raise ValueError("Original submission changed; refusing blend")
    write_submission(folder/"submission", bp["test"]["ids"], baseline, pred, binding)


def check(device):
    """Actual local model on random synthetic500/1000 rows only; no competition I/O."""
    predictor = t.Predictor(t.verify_weights(t.OUT), device)
    rng = np.random.default_rng(271828)
    x = rng.normal(size=(1500, 285))
    x[:, 0] = 0.
    x[:, 1] = rng.integers(0, 3, len(x))
    y = 190+3*x[:500, 2]+rng.normal(size=500)
    result = []
    for width in (185, 285):
        p, meta = predictor(x[:500, :width], y, x[500:, :width], SEED+9)
        result.append({"width": width, "shape": list(p.shape), **meta})
        print(f"Synthetic500-context/1000-query {width}: finite={np.isfinite(p).all()}; seconds={meta['seconds']:.1f}; peak={meta['peak_gpu_allocated_bytes']}")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "validate", "select", "extract", "predict", "run", "check"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    if args.stage == "prepare":
        prepare()
    elif args.stage == "validate":
        _, _, bp, _ = validate()
        baseline_test(bp)
        print("Final plan/train/test caches/original submission validated; no fitting or extraction")
    elif args.stage == "check":
        check(args.device)
    else:
        if args.stage in ("select", "run"):
            select_stage(OUT, args.device)
        if args.stage in ("extract", "run"):
            extract_stage(OUT)
        if args.stage in ("predict", "run"):
            predict_stage(OUT, args.device)


if __name__ == "__main__":
    main()
