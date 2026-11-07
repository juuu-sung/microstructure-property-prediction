"""User-run 2x2 nested ablation: existing185/new285 x Ridge/CatBoost, train only."""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import importlib.metadata
import json
from pathlib import Path
import sys
import time

if __package__:
    from . import extract_morphology as extract
    from . import evaluate_local_structure_confirmation as confirmation
else:
    import extract_morphology as extract
    import evaluate_local_structure_confirmation as confirmation

core,np,ROOT=confirmation.core,confirmation.np,confirmation.ROOT
OUT=extract.OUT
SEEDS=(42,2026)
EXISTING="existing185"
AUGMENTED="morphology285"
NAMES=("ridge_existing","ridge_morphology","catboost_existing","catboost_morphology")
PRIMARY="catboost_morphology_blend"
RIDGE_CONFIGS=[{"alpha":a} for a in (10.,100.,1000.)]
CAT_CONFIGS=[{"depth":d,"l2_leaf_reg":l} for d in (3,5) for l in (3.,10.)]
CAT_FIXED={"iterations":600,"learning_rate":.035,"loss_function":"RMSE","eval_metric":"RMSE",
    "boosting_type":"Plain","grow_policy":"SymmetricTree","bootstrap_type":"Bayesian",
    "bagging_temperature":1.,"random_strength":1.,"border_count":64,"task_type":"CPU",
    "allow_writing_files":False,"verbose":False,"use_best_model":False}
PROTOCOL={"version":1,"seeds":list(SEEDS),"feature_dimensions":{EXISTING:185,AUGMENTED:285},
    "morphology":extract.PROTOCOL,"ridge_configs":RIDGE_CONFIGS,"catboost_configs":CAT_CONFIGS,"catboost_fixed":CAT_FIXED,
    "outer":"existing seed42/2026 5-fold CSV, exactly identical to current OOF reference",
    "inner":"3-fold stratified train-label rank deciles, shuffle=True random_state=seed+outer_fold",
    "catboost_seed":"seed+10*outer_fold+inner_fold; final outer refit uses +9; identical across representations",
    "preprocessing":"Ridge: X/y StandardScaler inside each fit; CatBoost: no external fitted transform, numeric features only",
    "selection":"pooled inner MSE separately per cell; exact ties favor larger alpha or shallower depth then larger l2",
    "iteration_selection":"600 fixed; no eval_set, no early stopping, no best-epoch choice on outer labels",
    "weights":{"hand":.25,"original_resnet":.25,"dino_d4":.5},"reference":"current local-structure185 blend, not first handcrafted125 blend",
    "neural_components":"reuse verified predictions on exactly the same outer folds; never regression input or inner selection target",
    "primary":PRIMARY,"secondary":"ridge_morphology_blend; all four cells reported without picking best outer score as primary",
    "fit_count_per_seed":230,"fit_breakdown":{"ridge":100,"catboost":130},
    "priority_rule":"primary improves both current and CatBoost-existing control in both seeds; mean gain vs current>=0.30",
    "priority_mean_gain":.30,"interpretation":"investment threshold, not significance; reused500 and label-free preview carry research selection bias",
    "scope":"train-only; no test, calibration, final500 refit, model-weight search, GPU, or submission mutation"}


def load_base():
    data=confirmation.load_features()
    confirmation.validate_plan(ROOT/"reports/local_structure_confirmation",data)
    return data


def load_references(seed,data):
    if seed not in SEEDS:raise ValueError("Only fixed seeds42/2026")
    old=confirmation.previous
    args=old.parse_args(["--seed",str(seed)])
    legacy,refs=old.load_prior(args)
    old.attach_features(legacy,ROOT/"reports/local_structure/features/features.npz")
    np.testing.assert_array_equal(legacy.ids,data.ids);np.testing.assert_array_equal(legacy.y,data.y)
    np.testing.assert_array_equal(legacy.features[old.AUGMENTED],data.features[confirmation.AUGMENTED])
    path=ROOT/f"reports/local_structure/seed{seed}/oof.csv"
    rows=core.align_rows(core.read_rows(path,{"ID","hardness","fold",old.PRIMARY,old.AUGMENTED+"_nested"}),data.ids,path)
    np.testing.assert_array_equal(core.finite_column(rows,"hardness",path),data.y)
    np.testing.assert_array_equal([int(r["fold"]) for r in rows],legacy.folds)
    hand=core.finite_column(rows,old.AUGMENTED+"_nested",path)
    resnet=refs[old.previous.RAW_REFERENCE];dino=refs["reference_dino_d4"]
    current=core.finite_column(rows,old.PRIMARY,path)
    np.testing.assert_allclose(current,.25*hand+.25*resnet+.5*dino,atol=1e-10,rtol=0)
    files={"oof":extract.previous.file_reference(path),"summary":extract.previous.file_reference(path.with_name("summary.json"))}
    return {"folds":legacy.folds,"hand":hand,"resnet":resnet,"dino":dino,"current":current,"files":files}


def versions():
    return {n:importlib.metadata.version(n) for n in ("numpy","scipy","scikit-learn","Pillow","scikit-image","catboost","pandas")}


def sources():
    paths={Path(p).resolve() for p in (__file__,extract.__file__,confirmation.__file__)}
    return {**confirmation.source_references(),**{p.name:extract.previous.file_reference(p) for p in paths}}


def train_binding(data):
    return {"inputs":data.provenance,"ids_sha256":extract.common.sha256_array(data.ids),
        "target_sha256":extract.common.sha256_array(data.y),"existing185_sha256":extract.common.sha256_array(data.features[confirmation.AUGMENTED])}


def payload(data):
    preview=OUT/"preview_v2/preview.json"
    visual=OUT/"visual_review.json"
    p=json.loads(preview.read_text(encoding="utf-8"))
    if p["protocol"]!=extract.PROTOCOL or p["sources"]!=extract.source_refs() or p["versions"]!=extract.versions():
        raise ValueError("Preview is stale; no regression permitted")
    v=json.loads(visual.read_text(encoding="utf-8"))
    if v["preview_sha256"]!=core.sha256(preview) or v["decision"]!="proceed_as_region_proxies":
        raise ValueError("Visual review is missing or stale")
    refs={}
    for seed in SEEDS:
        r=load_references(seed,data)
        refs[str(seed)]={"files":r["files"],"arrays":{n:extract.common.sha256_array(r[n]) for n in ("folds","hand","resnet","dino","current")}}
    return {"protocol":PROTOCOL,"sources":sources(),"versions":versions(),"train_binding":train_binding(data),
        "preview":extract.previous.file_reference(preview),"visual_review":extract.previous.file_reference(visual),"references":refs}


def prepare(folder,data):
    path=folder/"protocol.json"
    if path.exists():raise ValueError("Protocol exists; refusing overwrite")
    folder.mkdir(parents=True,exist_ok=True)
    p={"created_utc":datetime.now(timezone.utc).isoformat(),"frozen":payload(data),"fits_during_preparation":0}
    extract.write_json(path,p);return p


def validate_plan(folder,data):
    p=json.loads((folder/"protocol.json").read_text(encoding="utf-8"))
    if p["frozen"]!=payload(data) or p["fits_during_preparation"]!=0:raise ValueError("Frozen inputs/sources/environment/protocol changed")
    return p


def load_morphology(path,data,plan,plan_path):
    with np.load(path,allow_pickle=False) as z:
        if set(z.files)!={"ids","morphology","feature_names","metadata_json"}:raise ValueError("Unexpected feature cache keys")
        meta=json.loads(z["metadata_json"].item())
        if meta!=json.loads(path.with_name("metadata.json").read_text(encoding="utf-8")):raise ValueError("Feature sidecar mismatch")
        arrays={n:z[n].copy() for n in ("ids","morphology","feature_names")}
    x=arrays["morphology"]
    if (meta["protocol"]!=extract.PROTOCOL or meta["sources"]!=extract.source_refs() or meta["versions"]!=extract.versions()
        or meta["plan_sha256"]!=core.sha256(plan_path) or meta["train_binding"]!=plan["frozen"]["train_binding"]
        or meta["test_accessed"] is not False or meta["labels_saved"] is not False
        or meta["array_sha256"]!={k:extract.common.sha256_array(v) for k,v in arrays.items()}):raise ValueError("Feature provenance mismatch")
    if not np.array_equal(arrays["ids"],data.ids) or arrays["feature_names"].tolist()!=extract.feature_names():raise ValueError("Feature IDs/schema differ")
    if x.shape!=(len(data.ids),100) or x.dtype!=np.float64 or not np.isfinite(x).all():raise ValueError("Invalid morphology array")
    old=data.features[confirmation.AUGMENTED]
    return {EXISTING:old,AUGMENTED:np.concatenate([old,x],axis=1)},meta


def model_spec(name):
    return ("ridge" if name.startswith("ridge") else "catboost",AUGMENTED if name.endswith("morphology") else EXISTING)


def make_model(family,config,seed,threads):
    if family=="ridge":return core.make_estimator(config["alpha"])
    if family!="catboost":raise ValueError("Unknown regressor")
    from catboost import CatBoostRegressor
    return CatBoostRegressor(**CAT_FIXED,**config,random_seed=seed,thread_count=threads)


def choice_key(family,record):
    c=record["config"]
    return (record["inner_mse"],-c["alpha"]) if family=="ridge" else (record["inner_mse"],c["depth"],-c["l2_leaf_reg"])


def nested(features,y,folds,seed,threads,*,factory=make_model):
    from sklearn.model_selection import StratifiedKFold
    from threadpoolctl import threadpool_limits
    predictions={n:np.full(len(y),np.nan) for n in NAMES};search=[];selections=[];fits=0
    with threadpool_limits(limits=threads):
        for fold in range(5):
            ti=np.flatnonzero(folds!=fold);vi=np.flatnonzero(folds==fold);ty=y[ti]
            inner=list(StratifiedKFold(3,shuffle=True,random_state=seed+fold).split(ty,core.rank_groups(ty,10)))
            selected={}
            for name in NAMES:
                family,representation=model_spec(name);x=features[representation]
                records=[]
                for config in RIDGE_CONFIGS if family=="ridge" else CAT_CONFIGS:
                    ip=np.full(len(ti),np.nan);mses=[]
                    for j,(tr,va) in enumerate(inner):
                        model=factory(family,config,seed+10*fold+j,threads)
                        model.fit(x[ti[tr]],ty[tr]);fits+=1
                        ip[va]=np.asarray(model.predict(x[ti[va]])).reshape(-1)
                        mses.append(float(np.mean((ip[va]-ty[va])**2)))
                    if not np.isfinite(ip).all():raise ValueError("Nonfinite inner prediction")
                    records.append({"model":name,"outer_fold":fold,"config":dict(config),"inner_mse":float(np.mean((ip-ty)**2)),"inner_fold_mse":mses})
                winner=min(records,key=lambda r:choice_key(family,r))
                model=factory(family,winner["config"],seed+10*fold+9,threads)
                model.fit(x[ti],ty);fits+=1
                predictions[name][vi]=np.asarray(model.predict(x[vi])).reshape(-1)
                selected[name]=winner;search.extend(records)
            selections.append({"outer_fold":fold,"outer_train_n":len(ti),"outer_valid_n":len(vi),
                "inner_seed":seed+fold,"inner_validation_sizes":[len(v) for _,v in inner],"selected":selected})
            print(f"Outer {fold}: four fixed cells completed; fits={fits}/230",flush=True)
    if fits!=230 or any(not np.isfinite(p).all() for p in predictions.values()):raise ValueError("Incomplete nested regression")
    return predictions,selections,search


def add_blends(predictions,refs):
    if set(predictions)!=set(NAMES):raise ValueError("Need four cells")
    difference=float(np.max(np.abs(predictions["ridge_existing"]-refs["hand"])))
    if difference>1e-7:raise ValueError(f"Existing185 Ridge anchor differs: {difference}")
    other=.25*refs["resnet"]+.5*refs["dino"]
    blended={n+"_blend":.25*p+other for n,p in predictions.items()}
    np.testing.assert_allclose(blended["ridge_existing_blend"],refs["current"],atol=1e-7,rtol=0)
    return {"current_blend":refs["current"],**predictions,**blended},difference


def diagnose(data,predictions,folds):
    target=core.rank_groups(data.y,10);noise=core.rank_groups(data.noise,4)
    current=predictions["current_blend"];metrics=[];groups=[];errors=[]
    masks={**{f"fold{j}":folds==j for j in range(5)},"low10":target==0,"high10":target==9,
        "tails20":(target==0)|(target==9),"middle80":(target>0)&(target<9),**{f"noise_q{j+1}":noise==j for j in range(4)}}
    for name,p in predictions.items():
        m=core.metrics(data.y,p);m.update(model=name,delta_vs_current=m["rmse"]-core.metrics(data.y,current)["rmse"]);metrics.append(m)
        for group,mask in masks.items():groups.append({"model":name,"group":group,**core.metrics(data.y[mask],p[mask])})
    worst=set(np.argsort(-(current-data.y)**2,kind="stable")[:20])
    for i,ident in enumerate(data.ids):
        errors.append({"ID":str(ident),"fold":int(folds[i]),"hardness":float(data.y[i]),"in_current_worst20":i in worst,
            **{n+"_delta_sse":float((p[i]-data.y[i])**2-(current[i]-data.y[i])**2) for n,p in predictions.items() if n.endswith("_blend")}})
    return metrics,groups,errors


def report_text(summary):
    lines=[f"# Morphology 2x2 ablation — seed {summary['seed']}","", "PRIMARY: catboost_morphology_blend. All cells prespecified; lower is better.","",
        "| model | OOF RMSE | delta vs current |","|---|---:|---:|"]
    for r in summary["metrics"]:lines.append(f"| {r['model']} | {r['rmse']:.6f} | {r['delta_vs_current']:+.6f} |")
    lines += ["","Existing185 Ridge anchor matched. Neural OOF reused only on identical folds.",
        "New feature effect: compare morphology vs existing WITHIN the same regressor. Regressor effect: compare CatBoost vs Ridge WITHIN the same feature set.",
        "CPU230 fits/seed. No early stopping/eval_set, test, final500 fit, calibration or new submission.",
        "Same500 previously researched; no independent-holdout or statistical significance claim.",
        "Region features are image-derived proxies; noisy images can merge grains. No physical segmentation ground truth is available.",""]
    return "\n".join(lines)


def run(folder,data,plan,seed,threads):
    output=folder/f"seed{seed}"
    extract.previous.protect_output(output,[folder/"protocol.json",folder/"features/features.npz",ROOT/"open/train.csv"])
    feature_hash=core.sha256(folder/"features/features.npz")
    features,_=load_morphology(folder/"features/features.npz",data,plan,folder/"protocol.json")
    refs=load_references(seed,data);start=time.perf_counter()
    predicted,selections,search=nested(features,data.y,refs["folds"],seed,threads)
    predictions,anchor=add_blends(predicted,refs)
    metrics,groups,errors=diagnose(data,predictions,refs["folds"])
    validate_plan(folder,data)
    if core.sha256(folder/"features/features.npz")!=feature_hash:raise ValueError("Feature cache changed during evaluation")
    summary={"seed":seed,"protocol_sha256":core.sha256(folder/"protocol.json"),"features_sha256":feature_hash,
        "metrics":metrics,"primary":PRIMARY,"selections":selections,"anchor_max_abs_difference":anchor,"fit_count":230,
        "elapsed_seconds":time.perf_counter()-start,"scope":PROTOCOL["scope"],"versions":versions()}
    oof=[{"ID":str(ident),"hardness":float(data.y[i]),"fold":int(refs["folds"][i]),**{n:float(p[i]) for n,p in predictions.items()}} for i,ident in enumerate(data.ids)]
    output.mkdir(parents=True,exist_ok=True)
    extract.write_json(output/"summary.json",summary);extract.write_json(output/"inner_search.json",search)
    for n,rows in (("oof",oof),("metrics",metrics),("group_metrics",groups),("per_image_errors",errors)):
        (output/(n+".csv")).write_text(core.csv_text(rows),encoding="utf-8",newline="")
    (output/"report.md").write_text(report_text(summary),encoding="utf-8")
    extract.write_json(output/"completed.json",{p.name:core.sha256(p) for p in output.iterdir() if p.is_file()})
    m={r["model"]:r for r in metrics}
    print(f"Existing185 Ridge anchor matched: {anchor:.3g}")
    for name in ("ridge_morphology_blend","catboost_existing_blend",PRIMARY):
        print(f"{name}: RMSE={m[name]['rmse']:.6f}; delta vs current={m[name]['delta_vs_current']:+.6f}")
    print(f"Report: {output/'report.md'}")


def aggregate(records):
    rows=[]
    for record in records:
        m={r["model"]:r for r in record["metrics"]}
        rows.append({"seed":record["seed"],"current_rmse":m["current_blend"]["rmse"],
            "primary_rmse":m[PRIMARY]["rmse"],"gain_vs_current":m["current_blend"]["rmse"]-m[PRIMARY]["rmse"],
            "gain_vs_catboost_control":m["catboost_existing_blend"]["rmse"]-m[PRIMARY]["rmse"],
            "ridge_feature_gain":m["ridge_existing_blend"]["rmse"]-m["ridge_morphology_blend"]["rmse"]})
    if len(rows)!=2 or {r["seed"] for r in rows}!=set(SEEDS):raise ValueError("Both seeds required")
    mean=float(np.mean([r["gain_vs_current"] for r in rows]));direction=all(r["gain_vs_current"]>0 and r["gain_vs_catboost_control"]>0 for r in rows)
    return {"per_seed":rows,"mean_primary_gain_vs_current":mean,"all_seeds_improve_both":direction,
        "meets_priority_rule":direction and mean>=.30,"aggregation":"mean per-seed RMSE differences, not prediction averaging or independent-sample inference"}


def summarize(folder):
    records=[]
    for seed in SEEDS:
        p=folder/f"seed{seed}";manifest=json.loads((p/"completed.json").read_text(encoding="utf-8"))
        if any(core.sha256(p/n)!=h for n,h in manifest.items()):raise ValueError("Completed output changed")
        record=json.loads((p/"summary.json").read_text(encoding="utf-8"))
        if record["protocol_sha256"]!=core.sha256(folder/"protocol.json") or record["features_sha256"]!=core.sha256(folder/"features/features.npz"):raise ValueError("Mixed protocols/caches")
        records.append(record)
    result=aggregate(records);extract.write_json(folder/"comparison.json",result)
    (folder/"comparison.csv").write_text(core.csv_text(result["per_seed"]),encoding="utf-8",newline="")
    print(json.dumps(result,indent=2));return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed",type=int,choices=SEEDS,default=42);p.add_argument("--threads",type=int,default=2)
    p.add_argument("--root-dir",type=Path,default=OUT)
    mode=p.add_mutually_exclusive_group();mode.add_argument("--prepare",action="store_true");mode.add_argument("--validate-only",action="store_true");mode.add_argument("--summarize",action="store_true")
    args=p.parse_args(argv)
    if args.threads<1:p.error("threads must be positive")
    data=load_base()
    if args.prepare:
        prepare(args.root_dir,data);print("Prepared fixed 2x2 protocol; fits=0, full-train feature extractions=0");return
    plan=validate_plan(args.root_dir,data)
    if args.validate_only:
        present=(args.root_dir/"features/features.npz").exists()
        if present:load_morphology(args.root_dir/"features/features.npz",data,plan,args.root_dir/"protocol.json")
        print(f"Frozen protocol/references valid; morphology cache present={present}; fits=0, writes=0");return
    if args.summarize:summarize(args.root_dir);return
    run(args.root_dir,data,plan,args.seed,args.threads)
    if all((args.root_dir/f"seed{s}/completed.json").exists() for s in SEEDS):summarize(args.root_dir)


if __name__=="__main__":
    try:main()
    except (ValueError,OSError,KeyError,RuntimeError) as exc:
        print(f"Morphology evaluation stopped: {exc}",file=sys.stderr);raise SystemExit(2)
