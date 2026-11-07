"""User-run local TabPFN-v2 comparison on existing185/morphology285 features.

Built with PriorLabs-TabPFN. Pinned weights/license: reports/tabpfn_v2/weights.
No neural gradient updates, test data, global preprocessing, or OOF weight fitting.
"""
from __future__ import annotations
import argparse
import csv
import dataclasses
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import time

# Set before any TabPFN/common-utils import. Evaluation uses an explicit local file.
os.environ["TABPFN_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
if __package__:
    from . import evaluate_morphology as previous
    from . import extract_frozen_resnet as common
else:
    import evaluate_morphology as previous
    import extract_frozen_resnet as common

ROOT, core = common.ROOT, previous.core
OUT = ROOT/"reports/tabpfn_v2"
REPO = "Prior-Labs/TabPFN-v2-reg"
REVISION = "4972a65a1b30806315c6f92499959ffbfc69a673"
CHECKPOINT = "tabpfn-v2-regressor.ckpt"
WEIGHT_SHA = "2ab5a07d5c41dfe6db9aa7ae106fc6de898326c2765be66505a07e2868c10736"
LICENSE_SHA = "08b1851b4b6136b5fb0cb124d3258e0d91e3e1623f1be5904b77e4d270d112d8"
SEEDS = (42, 2026)
REPS = ("existing185", "morphology285")
PRIMARY = "tabpfn_joint_nested"
SECONDARY = "current_tabpfn_equal"
CONFIG_OVERRIDE = {"MIN_UNIQUE_FOR_NUMERICAL_FEATURES": 0}
PROTOCOL = {
    "version": 1, "package_version": "6.4.1", "model_version": "V2", "model_repo": REPO,
    "revision": REVISION, "checkpoint": CHECKPOINT, "checkpoint_sha256": WEIGHT_SHA,
    "license_sha256": LICENSE_SHA, "attribution": "Built with PriorLabs-TabPFN",
    "representations": list(REPS), "seeds": list(SEEDS),
    "outer": "original5 outer folds;400 context/100 held out; same500 IDs as current ensemble",
    "inner": "StratifiedKFold3, target rank deciles within outer400, shuffle, random_state=seed+fold",
    "selection": "lowest pooled inner MSE for185 vs285; ties existing185; no other hyperparameter search",
    "model_seed": "seed+1009*fold+inner_fold; full outer400 uses +9; identical across representations",
    "predictor": "create_default_for_version(V2), explicit local checkpoint, output_type=mean",
    "n_estimators": 8, "softmax_temperature": .9, "average_before_softmax": False,
    "fit_mode": "fit_preprocessors", "memory_saving_mode": True, "n_preprocessing_jobs": 1,
    "inference_precision": "torch.float32, no AMP", "tf32": False, "torch_threads": 4,
    "deterministic_algorithms": True, "categorical_features_indices": [],
    "inference_config_override": CONFIG_OVERRIDE,
    "preprocessing": "raw numeric features; TabPFN v2 internal transforms/SVD fitted on each context subset; no external scaler/PCA or target calibration",
    "scope": "train-only; fit means in-context preprocessing, no gradient training; no test or full500 refit",
    "fit_count_per_seed": 40, "fit_breakdown": "5*(2representations*3inner+2outer); joint reuses selected outer prediction",
    "primary": PRIMARY, "secondary": "0.5*current185_ensemble +0.5*tabpfn_joint_nested",
    "fixed_diagnostics": "185 and285 individual OOF; saved same-fold Ridge/CatBoost controls; no post-hoc adoption by minimum OOF",
    "priority_rule": "primary improves both seeds and mean RMSE gain>=0.5; inspect tails/common failures; not significance",
    "telemetry": "disabled before import; HF offline; no external inference service",
    "resume": "per completed context-fit/predict result; same inputs/runtime only; no failed-cell outputs",
}


def digest(value):
    return hashlib.sha256(common.canonical_json(value).encode("utf-8")).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def write_csv(path, rows):
    tmp = path.with_name(path.name+".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def file_ref(path):
    return {"path": str(Path(path).resolve()), "sha256": common.sha256_file(path)}


def verify_weights(folder):
    path = folder/"weights"/CHECKPOINT
    if common.sha256_file(path) != WEIGHT_SHA or common.sha256_file(path.parent/"LICENSE.txt") != LICENSE_SHA:
        raise ValueError("Pinned TabPFN-v2 weight/license hash mismatch")
    return path


def environment():
    return {d.metadata["Name"].lower().replace("_", "-"): d.version for d in importlib.metadata.distributions()}


def runtime_source_hashes():
    # Bind installed implementation, including inference/preprocessing and telemetry.
    refs = {}
    for dist_name, module in (("tabpfn", "tabpfn"), ("tabpfn-common-utils", "tabpfn_common_utils")):
        root = Path(importlib.metadata.distribution(dist_name).locate_file(module))
        refs[dist_name] = {p.relative_to(root).as_posix(): common.sha256_file(p) for p in sorted(root.rglob("*.py"))}
    return refs


def load_inputs():
    data = previous.load_base()
    old_path = ROOT/"reports/morphology/protocol.json"
    old_plan = previous.validate_plan(old_path.parent, data)
    features, _ = previous.load_morphology(ROOT/"reports/morphology/features/features.npz", data, old_plan, old_path)
    if features[REPS[0]].shape != (500,185) or features[REPS[1]].shape != (500,285):
        raise ValueError("Unexpected feature dimensions")
    if not all(np.isfinite(x).all() for x in features.values()):
        raise ValueError("Nonfinite input features")
    refs = {seed: previous.load_references(seed, data) for seed in SEEDS}
    for seed, ref in refs.items():
        folder = ROOT/f"reports/morphology/seed{seed}"
        # Existing outputs are comparison controls only; never model inputs.
        done = read_json(folder/"completed.json")
        if any(common.sha256_file(folder/n) != sha for n,sha in done.items()):
            raise ValueError("Saved Ridge/CatBoost control output changed")
        path = folder/"oof.csv"
        rows = core.align_rows(core.read_rows(path, {"ID", "hardness", "fold"}), data.ids, path)
        np.testing.assert_array_equal(core.finite_column(rows, "hardness", path), data.y)
        np.testing.assert_array_equal([int(r["fold"]) for r in rows], ref["folds"])
        ref["controls"] = {n: core.finite_column(rows, n, path) for n in
                           ("ridge_existing", "ridge_morphology", "catboost_existing", "catboost_morphology")}
        np.testing.assert_allclose(ref["controls"]["ridge_existing"], ref["hand"], rtol=0, atol=1e-10)
        ref["control_files"] = {n: file_ref(folder/n) for n in ("oof.csv", "summary.json", "completed.json")}
    return data, features, refs


def payload(data, features, refs, folder):
    if importlib.metadata.version("tabpfn") != "6.4.1":
        raise ValueError("Use pinned TabPFN package6.4.1 with model V2")
    verify_weights(folder)
    return {"protocol": PROTOCOL, "sources": {**previous.sources(), Path(__file__).name: file_ref(__file__),
            "requirements": file_ref(ROOT/"requirements-tabpfn.txt")}, "runtime_sources": runtime_source_hashes(),
            "environment": environment(), "training": previous.train_binding(data),
            "feature_arrays": {n: common.sha256_array(x) for n,x in features.items()},
            "feature_files": [file_ref(ROOT/"reports/morphology/features/features.npz"), file_ref(ROOT/"reports/morphology/features/metadata.json")],
            "references": {str(seed): {"files": ref["files"], "control_files": ref["control_files"],
                "arrays": {n: common.sha256_array(ref[n]) for n in ("folds","current","hand","resnet","dino")}} for seed,ref in refs.items()}}


def prepare(folder):
    data, features, refs = load_inputs()
    frozen = payload(data, features, refs, folder)
    path = folder/"protocol.json"
    if path.exists():
        if read_json(path)["frozen"] != frozen:
            raise ValueError("Existing protocol differs; use a new experiment folder")
        print("Existing preparation matches; unchanged")
        return
    write_json(path, {"frozen": frozen, "created_unix": time.time(), "competition_fits_during_preparation": 0})
    print(f"Prepared {path};500 aligned IDs;185/285 features; no competition fit or inference")


def validate(folder):
    data, features, refs = load_inputs()
    plan = read_json(folder/"protocol.json")
    if plan["frozen"] != payload(data, features, refs, folder):
        raise ValueError("Source, dependency, inputs or protocol changed; refusing reuse")
    return data, features, refs, plan


def configure_device(name, seed):
    import torch
    torch.set_num_threads(4)
    device = torch.device(name)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; no automatic CPU fallback")
        index = torch.cuda.current_device() if device.index is None else device.index
        if not 0 <= index < torch.cuda.device_count():
            raise ValueError("Invalid CUDA device index")
        device = torch.device("cuda", index)
        torch.cuda.set_device(device)
    elif device.type != "cpu":
        raise ValueError("Only CUDA/CPU supported")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    return torch, device


def make_model(weights, device, seed):
    import torch
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion
    return TabPFNRegressor.create_default_for_version(ModelVersion.V2,
        model_path=weights, device=str(device), random_state=seed, n_estimators=8,
        softmax_temperature=.9, average_before_softmax=False, fit_mode="fit_preprocessors",
        memory_saving_mode=True, inference_precision=torch.float32, n_preprocessing_jobs=1,
        categorical_features_indices=[], inference_config=CONFIG_OVERRIDE.copy(),
        ignore_pretraining_limits=False, differentiable_input=False)


class Predictor:
    """Only fitting y is accepted. No query labels or baseline predictions."""
    def __init__(self, weights, device_name):
        self.torch, self.device = configure_device(device_name, 0)
        self.weights = weights
        self.runtime = {"device": str(self.device), "gpu": self.torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else None,
                        "precision": "float32", "threads": 4, "telemetry": False, "hf_offline": True}

    def __call__(self, train_x, train_y, query_x, seed):
        started = time.perf_counter()
        if self.device.type == "cuda":
            self.torch.cuda.reset_peak_memory_stats(self.device)
        model = make_model(self.weights, self.device, seed)
        try:
            # Internal transforms are fitted by fit() on the context subset only.
            model.fit(train_x.copy(), train_y.copy())
            pred = np.asarray(model.predict(query_x.copy(), output_type="mean"), dtype=np.float64).reshape(-1)
            if pred.shape != (len(query_x),) or not np.isfinite(pred).all():
                raise ValueError("Invalid TabPFN prediction")
            cfg = dataclasses.asdict(model.inference_config_)
            if cfg["MAX_NUMBER_OF_FEATURES"] != 500 or cfg["MIN_UNIQUE_FOR_NUMERICAL_FEATURES"] != 0:
                raise ValueError("Unexpected v2 inference configuration")
            from tabpfn.preprocessing.datamodel import FeatureModality
            if model.inferred_feature_schema_.indices_for(FeatureModality.CATEGORICAL):
                raise ValueError("Expected numeric feature inputs")
            metadata = {"seconds": time.perf_counter()-started, "context_rows": len(train_x), "query_rows": len(query_x),
                        "columns": train_x.shape[1], "resolved_config": cfg, "context_mean": float(model.y_train_mean_),
                        "context_std": float(model.y_train_std_), "runtime": self.runtime,
                        "peak_gpu_allocated_bytes": int(self.torch.cuda.max_memory_allocated(self.device)) if self.device.type == "cuda" else None}
            return pred, metadata
        finally:
            del model
            gc.collect()
            if self.device.type == "cuda":
                self.torch.cuda.empty_cache()


def run_cell(folder, binding_base, representation, x, y_train, train_idx, query_idx, seed, predictor):
    if np.intersect1d(train_idx, query_idx).size:
        raise ValueError("Context/query overlap")
    binding = {"experiment": binding_base, "representation": representation,
               "train_idx": train_idx.tolist(), "query_idx": query_idx.tolist(), "seed": seed,
               "train_y_sha256": common.sha256_array(y_train),
               "train_x_sha256": common.sha256_array(x[train_idx]), "query_x_sha256": common.sha256_array(x[query_idx])}
    path = folder/"completed.json"
    if path.exists():
        saved = read_json(path)
        if saved["binding"] != binding or saved["result_sha256"] != digest(saved["result"]):
            raise ValueError("Completed cell changed: " + str(folder))
        pred = np.asarray(saved["result"]["prediction"], dtype=np.float64)
        if pred.shape != (len(query_idx),) or not np.isfinite(pred).all():
            raise ValueError("Invalid cached predictions")
        print(f"Reused {folder.name}", flush=True)
        return pred
    pred, metadata = predictor(x[train_idx], y_train, x[query_idx], seed)
    pred = np.asarray(pred, dtype=np.float64)
    if pred.shape != (len(query_idx),) or not np.isfinite(pred).all():
        raise ValueError("Invalid predictor output; no completion record written")
    result = {"prediction": pred.tolist(), "metadata": metadata}
    write_json(path, {"binding": binding, "result": result, "result_sha256": digest(result)})
    print(f"Completed {folder.name}: context={len(train_idx)} query={len(query_idx)} features={x.shape[1]}; {metadata['seconds']:.1f}s", flush=True)
    return pred


def select_representation(inner_predictions, train_y):
    scores = {n: float(np.mean((inner_predictions[n]-train_y)**2)) for n in REPS}
    if not all(np.isfinite(scores[n]) for n in REPS):
        raise ValueError("Incomplete inner predictions")
    choice = min(REPS, key=lambda n:(scores[n], REPS.index(n)))
    return choice, scores


def evaluate_fold(features, y, folds, seed, fold, folder, binding, predictor):
    from sklearn.model_selection import StratifiedKFold
    ti, vi = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
    inner = list(StratifiedKFold(3, shuffle=True, random_state=seed+fold).split(ti, core.rank_groups(y[ti],10)))
    predictions = {n: np.full(len(ti), np.nan) for n in REPS}
    records = []
    for rep in REPS:
        for j,(tr,va) in enumerate(inner):
            pred = run_cell(folder/f"fold{fold}_{rep}_inner{j}", binding, rep, features[rep], y[ti[tr]], ti[tr], ti[va],
                            seed+1009*fold+j, predictor)
            predictions[rep][va] = pred
            records.append({"representation": rep, "inner_fold": j, "train_idx": ti[tr].tolist(), "valid_idx": ti[va].tolist(),
                            "valid_mse": float(np.mean((pred-y[ti[va]])**2))})
    choice, scores = select_representation(predictions, y[ti])
    outputs = {}
    for rep in REPS:
        outputs[rep] = run_cell(folder/f"fold{fold}_{rep}_outer", binding, rep, features[rep], y[ti], ti, vi,
                                seed+1009*fold+9, predictor)
    outputs[PRIMARY] = outputs[choice].copy()
    selection = {"fold": fold, "representation": choice, "inner_mse": scores, "inner_splits": records}
    print(f"Outer{fold}: representation={choice}; pooled inner RMSE={np.sqrt(scores[choice]):.6f}; outer predictions saved", flush=True)
    return vi, outputs, selection


def summarize(folder, data, refs, predictions, choices, binding, seed):
    all_predictions = {"current": refs["current"], **refs["controls"],
                       "tabpfn_existing185": predictions[REPS[0]], "tabpfn_morphology285": predictions[REPS[1]],
                       PRIMARY: predictions[PRIMARY], SECONDARY: .5*(refs["current"]+predictions[PRIMARY])}
    if any(p.shape != data.y.shape or not np.isfinite(p).all() for p in all_predictions.values()):
        raise ValueError("Incomplete OOF")
    baseline = core.metrics(data.y, refs["current"])["rmse"]
    metrics = {n: {**core.metrics(data.y, p), "gain_vs_current": baseline-core.metrics(data.y,p)["rmse"]} for n,p in all_predictions.items()}
    rows = [{"ID": str(data.ids[i]), "hardness": float(data.y[i]), "fold": int(refs["folds"][i]),
             **{n: float(p[i]) for n,p in all_predictions.items()}} for i in range(len(data.ids))]
    write_csv(folder/"oof.csv", rows)
    groups, diagnostics = {}, []
    bins, noise = core.rank_groups(data.y,10), core.rank_groups(data.noise,4)
    baseline_worst = np.argsort((refs["current"]-data.y)**2,kind="stable")[-20:]
    masks = {"tail20": (bins==0)|(bins==9), "middle80": (bins>0)&(bins<9), "baseline_worst20": np.isin(np.arange(len(data.y)),baseline_worst)}
    masks.update({f"fold{f}":refs["folds"]==f for f in range(5)})
    masks.update({f"noiseQ{q+1}":noise==q for q in range(4)})
    for name,p in all_predictions.items():
        groups[name] = {}
        for group,mask in masks.items():
            m = core.metrics(data.y[mask],p[mask])
            groups[name][group] = m["rmse"]
            diagnostics.append({"model":name,"group":group,**m})
    write_csv(folder/"diagnostics.csv",diagnostics)
    err_rows = [{"ID":str(data.ids[i]),"hardness":float(data.y[i]), "current_error":float(refs["current"][i]-data.y[i]),
                 "tabpfn_error":float(predictions[PRIMARY][i]-data.y[i]),"blend_error":float(all_predictions[SECONDARY][i]-data.y[i])}
                for i in np.argsort(-(refs["current"]-data.y)**2,kind="stable")]
    write_csv(folder/"per_image_errors.csv",err_rows)
    summary = {"seed":seed,"binding":binding,"metrics":metrics,"groups":groups,"selection":choices,
               "primary":PRIMARY,"secondary":SECONDARY,"cells":40,"independent_holdout":False,
               "no_gradient_training":True,"test_accessed":False,"attribution":"Built with PriorLabs-TabPFN"}
    write_json(folder/"summary.json",summary)
    lines = [f"# TabPFN-v2 · seed{seed}","","Built with PriorLabs-TabPFN.","", "| Model | OOF RMSE | Gain vs current |","|---|---:|---:|"]
    lines += [f"| {n} | {m['rmse']:.6f} | {m['gain_vs_current']:+.6f} |" for n,m in metrics.items()]
    lines += ["",f"Primary: {PRIMARY}. Fixed secondary: current50%+selected TabPFN50%.",
              "Features185/285 selected using inner3-fold pooled MSE. Individual scores are fixed diagnostics, not post-hoc selection.",
              "Same500 reused; not independent holdout or a Public leaderboard score. No neural gradient updates or test use.",
              "Ridge/CatBoost controls reuse audited predictions on the same outer folds; no new fitting.","", "## Inner representation choices"]
    lines += [f"- fold{r['fold']}: {r['representation']}; {r['inner_mse']}" for r in choices]
    (folder/"report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    files = {p.relative_to(folder).as_posix():common.sha256_file(p) for p in folder.glob("fold*/completed.json")}
    files.update({n:common.sha256_file(folder/n) for n in ("oof.csv","summary.json","diagnostics.csv","per_image_errors.csv","report.md")})
    write_json(folder/"completed_evaluation.json",{"binding":binding,"files":files})
    for name in ("tabpfn_existing185","tabpfn_morphology285",PRIMARY,SECONDARY):
        print(f"{name}: RMSE={metrics[name]['rmse']:.6f}; gain vs current={metrics[name]['gain_vs_current']:+.6f}",flush=True)
    print(f"Report: {folder/'report.md'}",flush=True)


def verify_completed(folder, binding):
    path = folder/"completed_evaluation.json"
    if not path.exists():
        return False
    value = read_json(path)
    if value["binding"] != binding or any(common.sha256_file(folder/n)!=sha for n,sha in value["files"].items()):
        raise ValueError("Completed evaluation binding/output changed")
    return True


def evaluate(folder, device_name, seed):
    data,features,refs,plan = validate(folder)
    predictor = Predictor(verify_weights(folder),device_name)
    binding = {"plan":digest(plan["frozen"]),"seed":seed,"runtime":predictor.runtime}
    target = folder/f"seed{seed}"
    target.mkdir(parents=True,exist_ok=True)
    if verify_completed(target,binding):
        print(f"Completed evaluation verified; {target/'report.md'}",flush=True)
    else:
        predictions = {n:np.full(len(data.ids),np.nan) for n in (*REPS,PRIMARY)}
        choices = []
        for fold in range(5):
            idx,outputs,choice = evaluate_fold(features,data.y,refs[seed]["folds"],seed,fold,target,binding,predictor)
            for n,p in outputs.items():
                predictions[n][idx]=p
            choices.append(choice)
        if payload(data,features,refs,folder) != plan["frozen"]:
            raise ValueError("Inputs changed during evaluation")
        summarize(target,data,refs[seed],predictions,choices,binding,seed)
    if all((folder/f"seed{s}/completed_evaluation.json").exists() for s in SEEDS):
        compare(folder)


def compare(folder):
    data,features,refs,plan = validate(folder)
    records = []
    for seed in SEEDS:
        path = folder/f"seed{seed}"
        summary = read_json(path/"summary.json")
        if summary["binding"]["plan"] != digest(plan["frozen"]) or not verify_completed(path,summary["binding"]):
            raise ValueError("Evaluation plan differs")
        rows = core.align_rows(core.read_rows(path/"oof.csv",{"ID","hardness","fold",PRIMARY,SECONDARY,"current"}),data.ids,path)
        np.testing.assert_array_equal(core.finite_column(rows,"hardness",path),data.y)
        np.testing.assert_array_equal([int(r["fold"]) for r in rows],refs[seed]["folds"])
        pred = core.finite_column(rows,PRIMARY,path)
        blend = core.finite_column(rows,SECONDARY,path)
        np.testing.assert_allclose(core.finite_column(rows,"current",path),refs[seed]["current"],atol=1e-10,rtol=0)
        np.testing.assert_allclose(blend,.5*(pred+refs[seed]["current"]),atol=1e-10,rtol=0)
        rmse = core.metrics(data.y,refs[seed]["current"])["rmse"]
        records.append({"seed":seed,"current_rmse":rmse,"primary_rmse":core.metrics(data.y,pred)["rmse"],
                        "primary_gain":rmse-core.metrics(data.y,pred)["rmse"],"secondary_gain":rmse-core.metrics(data.y,blend)["rmse"]})
    mean = float(np.mean([r["primary_gain"] for r in records]))
    value = {"per_seed":records,"mean_primary_gain":mean,
             "passes_numeric_priority_gate":all(r["primary_gain"]>0 for r in records) and mean>=.5,
             "additional_review_required":"tails/common errors; numeric gate is not automatic adoption",
             "aggregation":"mean of per-seed RMSE differences; no prediction averaging or independent-sample inference"}
    write_json(folder/"comparison.json",value)
    print(json.dumps(value,ensure_ascii=False,indent=2),flush=True)


def check(folder,device):
    # No competition data or prior OOF is loaded by this command.
    predictor = Predictor(verify_weights(folder),device)
    rng = np.random.default_rng(314159)
    # Exercise the actual400/100 outer dimensions on GPU, lighter40/8 CPU smoke test.
    context, queries = (400,100) if predictor.device.type == "cuda" else (40,8)
    x = rng.normal(size=(context+queries,285))
    x[:,0] = rng.integers(0,3,size=len(x))  # low-cardinality numeric count column
    x[:,1] = 0.                       # constant feature
    y = 100+3*x[:,2]-2*x[:,3]+rng.normal(size=len(x))
    checks = []
    for columns in (185,285):
        p,meta = predictor(x[:context,:columns],y[:context],x[context:,:columns],314159)
        checks.append({"features":columns,"prediction_shape":list(p.shape),"finite":bool(np.isfinite(p).all()),"metadata":meta})
        print(f"Synthetic {columns} features passed on {predictor.device}; {meta['seconds']:.1f}s",flush=True)
    write_json(folder/f"synthetic_check_{predictor.device.type}.json",{"checks":checks,"competition_data_used":False,"gradient_updates":0})


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command",choices=("prepare","validate","check","evaluate","compare"))
    p.add_argument("--seed",type=int,choices=SEEDS,default=42)
    p.add_argument("--device",default="cuda")
    p.add_argument("--output-dir",type=common.project_path,default=OUT)
    args = p.parse_args(argv)
    if args.command == "prepare": prepare(args.output_dir)
    elif args.command == "validate":
        validate(args.output_dir)
        print("Inputs, protocol, sources, environment and v2 weights match; no fit/inference")
    elif args.command == "check": check(args.output_dir,args.device)
    elif args.command == "compare": compare(args.output_dir)
    else: evaluate(args.output_dir,args.device,args.seed)


if __name__ == "__main__":
    main()
