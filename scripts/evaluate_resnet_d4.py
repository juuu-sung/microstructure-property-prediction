"""User-run fixed ResNet D4 replacement with DINO D4 held constant.

ResNet representation is prespecified, not selected against raw by inner CV.
Only Ridge alpha is selected, separately for raw control and D4. Both checking
modes fit zero estimators. No images, torch, downloads, test, or submissions.
"""
from __future__ import annotations

import argparse
import contextlib
from datetime import datetime, timezone
import io
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

if __package__:
    from . import evaluate_dinov2_d4 as previous
    from . import extract_resnet_d4 as extract
else:
    import evaluate_dinov2_d4 as previous
    import extract_resnet_d4 as extract

core,np,ROOT=previous.core,previous.np,previous.ROOT
RAW,D4=extract.RAW,extract.FEATURE
REPRESENTATIONS=(RAW,D4)
ALPHAS=core.ALPHAS
REFERENCE="reference_dino_d4_fixed_blend"
RAW_REFERENCE="reference_original_resnet"
PRIMARY="resnet_d4_replaced_blend"


def verify_hashes(value):
    if isinstance(value,dict):
        for pk,hk in (("path","sha256"),("metadata_path","metadata_sha256"),("summary_path","summary_sha256")):
            if pk in value and hk in value and core.sha256(Path(value[pk]))!=value[hk]:
                raise ValueError(f"Saved input/source changed: {value[pk]}")
        for v in value.values(): verify_hashes(v)
    elif isinstance(value,list):
        for v in value: verify_hashes(v)


def load_oof(path,data,columns):
    rows=core.align_rows(core.read_rows(path,{"ID","hardness","fold",*columns}),data.ids,path)
    if (not np.array_equal(core.finite_column(rows,"hardness",path),data.y) or
            not np.array_equal(core.integer_column(rows,"fold",path),data.folds)):
        raise ValueError("Reference OOF labels/folds differ")
    return {name:core.finite_column(rows,name,path) for name in columns}


def load_dino_reference(data,references,oof_path,summary_path,seed):
    summary=json.loads(summary_path.read_text(encoding="utf-8"))
    if (summary.get("seed")!=seed or summary.get("n_train")!=len(data.ids) or summary.get("experiment")!="dinov2_d4_mean" or
            summary.get("feature_metadata")!=data.metadata or summary.get("d4_protocol")!=previous.extract.PROTOCOL):
        raise ValueError("Prior DINO D4 summary protocol/metadata mismatch")
    verify_hashes(summary["input_files"])
    column=previous.D4+"_nested"
    saved=load_oof(oof_path,data,[column,core.PRIMARY_NAME,previous.FIXED_BLEND,*references])
    for name,pred in references.items():
        if not np.array_equal(saved[name],pred): raise ValueError(f"Prior DINO D4 reference changed: {name}")
    for name in (column,core.PRIMARY_NAME,previous.FIXED_BLEND):
        if abs(core.metrics(data.y,saved[name])["rmse"]-summary["diagnostics"][name]["overall"]["rmse"])>1e-8:
            raise ValueError("Prior DINO D4 OOF/summary metric mismatch")
    np.testing.assert_allclose(saved[previous.FIXED_BLEND],references[previous.CURRENT]+0.5*(saved[core.PRIMARY_NAME]-references[previous.DINO]),atol=1e-7,rtol=0)
    refs={**references,"reference_dino_d4":saved[column],
        "reference_previous_joint_blend":saved[previous.FIXED_BLEND],
        REFERENCE:references[previous.CURRENT]+0.5*(saved[column]-references[previous.DINO])}
    data.provenance["fixed_dino_d4_reference"]={"path":str(oof_path.resolve()),"sha256":core.sha256(oof_path),
        "summary_path":str(summary_path.resolve()),"summary_sha256":core.sha256(summary_path),
        "prediction_column":column,"refitted":False,"representation_fixed_after_prior_exploratory_results":True}
    return refs


def load_resnet_anchor(data,path,summary_path,seed):
    saved=load_oof(path,data,[RAW+"_nested",core.PRIMARY_NAME,core.BASELINE_NAME])
    if not np.array_equal(saved[core.BASELINE_NAME],data.baseline): raise ValueError("Original ResNet baseline changed")
    if not np.array_equal(saved[RAW+"_nested"],saved[core.PRIMARY_NAME]):
        raise ValueError("Original ResNet reference must equal multiscale OOF on every fold")
    summary=json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("seed")!=seed or summary.get("alpha_candidates")!=list(ALPHAS) or summary.get("inner_splits")!=3:
        raise ValueError("Original ResNet seed/alpha/inner protocol changed")
    verify_hashes(summary["input_files"])
    if summary.get("feature_metadata")!=data.metadata["resnet_original"]:
        raise ValueError("Original ResNet summary/cache metadata differs")
    if abs(core.metrics(data.y,saved[RAW+"_nested"])["rmse"]-summary["diagnostics"][RAW+"_nested"]["overall"]["rmse"])>1e-8:
        raise ValueError("Original ResNet OOF/summary metric mismatch")
    records=summary["selections"]
    if len(records)!=5 or {r["outer_fold"] for r in records}!=set(range(5)):
        raise ValueError("Original ResNet needs five unique outer folds")
    anchor={"predictions":saved[RAW+"_nested"],"prediction_column":RAW+"_nested","selections":{}}
    for r in records:
        fold=r["outer_fold"]; count=int(np.sum(data.folds!=fold))
        sizes=[count//3+int(i<count%3) for i in range(3)]
        if (r["inner_seed"]!=seed+fold or r["outer_train_n"]!=count or r["outer_valid_n"]!=len(data.y)-count or
                r["inner_validation_sizes"]!=sizes or r["joint"]["representation"]!=RAW):
            raise ValueError("Original ResNet selection/fold assignment mismatch")
        candidates=[c for c in r["candidates"] if c["representation"]==RAW]
        if len(candidates)!=len(ALPHAS) or {c["alpha"] for c in candidates}!=set(ALPHAS):
            raise ValueError("Original ResNet alpha grid differs")
        for c in candidates:
            if (len(c["inner_fold_mse"])!=3 or not np.isfinite([c["inner_mse"],*c["inner_fold_mse"]]).all() or
                    abs(c["inner_mse"]-np.average(c["inner_fold_mse"],weights=sizes))>1e-8):
                raise ValueError("Original ResNet pooled inner score mismatch")
        winner=min(candidates,key=lambda c:(c["inner_mse"],-c["alpha"]))
        if winner!=r["per_representation"][RAW]: raise ValueError("Original ResNet alpha selection differs")
        anchor["selections"][fold]={"winner":winner,"candidates":candidates,"inner_validation_sizes":sizes}
    data.provenance["original_resnet_oof"]={"path":str(path.resolve()),"sha256":core.sha256(path),
        "summary_path":str(summary_path.resolve()),"summary_sha256":core.sha256(summary_path)}
    return anchor


def load_prior(args,*,expected_n=500):
    dino_args=previous.parse_args(["--seed",str(args.seed)])
    dino_args.previous_frozen_oof=args.original_oof
    data,refs,_=previous.load_prior(dino_args,expected_n=expected_n)
    data=previous.attach_features(data,dino_args.d4_features,dino_args.global_features)
    refs=load_dino_reference(data,refs,args.dino_d4_oof,args.dino_d4_summary,args.seed)
    ids,values,metadata=extract.read_original(np,args.original_features,dino_args.train_csv,expected_n=expected_n)
    if not np.array_equal(ids,data.ids) or metadata["image_manifest_sha256"]!=data.metadata["global"]["image_manifest_sha256"]:
        raise ValueError("ResNet/DINO train identity mismatch")
    data.features={RAW:values}; data.metadata={"resnet_original":metadata,"fixed_dino":data.metadata}
    data.provenance["original_resnet_features"]=extract.reference(args.original_features)
    anchor=load_resnet_anchor(data,args.original_oof,args.original_summary,args.seed)
    refs[RAW_REFERENCE]=anchor["predictions"]
    np.testing.assert_allclose(refs[REFERENCE],0.25*data.baseline+0.25*refs[RAW_REFERENCE]+0.5*refs["reference_dino_d4"],atol=1e-7,rtol=0)
    data.provenance["resnet_d4_evaluation_sources"]={n:{"path":str(Path(p).resolve()),"sha256":core.sha256(Path(p))}
        for n,p in {"evaluator":__file__,"resnet_extractor":extract.__file__}.items()}
    return data,refs,anchor


def attach_features(data,path,original_path):
    values,metadata=extract.validate_cache(np,path,data.ids,data.metadata["resnet_original"],original_path)
    data.features[D4]=values; data.metadata["resnet_d4"]=metadata
    data.provenance["resnet_d4_features"]=extract.reference(path)
    return data


def nested_predictions(data,seed,threads):
    # One representation per call: no raw/D4 gate or joint candidate selection.
    predictions={}; by_rep={}
    for name in REPRESENTATIONS:
        with contextlib.redirect_stdout(io.StringIO()):
            pred,records=core.nested_predictions(data,seed,threads,alphas=ALPHAS,representations=(name,))
        predictions[name+"_nested"]=pred[name+"_nested"]
        by_rep[name]=records
    selections=[]
    for fold in range(5):
        source=by_rep[RAW][fold]
        record={k:source[k] for k in ("outer_fold","outer_train_n","outer_valid_n","inner_seed","inner_validation_sizes")}
        record["candidates"]=[c for name in REPRESENTATIONS for c in by_rep[name][fold]["candidates"]]
        record["per_representation"]={name:by_rep[name][fold]["per_representation"][name] for name in REPRESENTATIONS}
        selections.append(record)
        print(f"Outer fold {fold}: raw alpha={record['per_representation'][RAW]['alpha']:g}; D4 alpha={record['per_representation'][D4]['alpha']:g}; representation fixed",flush=True)
    return predictions,selections


def verify_anchor(data,predictions,selections,anchor):
    records={r["outer_fold"]:{"winner":r["per_representation"][RAW],
        "candidates":[c for c in r["candidates"] if c["representation"]==RAW],
        "inner_validation_sizes":r["inner_validation_sizes"],"validation_indices":np.flatnonzero(data.folds==r["outer_fold"]),
        "validation_predictions":predictions[RAW+"_nested"][data.folds==r["outer_fold"]]} for r in selections}
    return previous.prior.prior.validate_anchor(records,data,anchor)


def add_fixed_blend(predictions,references):
    if set(predictions)!={name+"_nested" for name in REPRESENTATIONS}: raise ValueError("Unexpected ResNet predictions")
    return {**predictions,**references,PRIMARY:references[REFERENCE]+0.25*(predictions[D4+"_nested"]-references[RAW_REFERENCE])}


def diagnose(data,predictions):
    diagnostics,tables=previous.diagnose(data,predictions)
    for row in tables["metrics"]:
        m=diagnostics[row["model"]]["overall"]
        for label,name in (("dino_d4_fixed_blend",REFERENCE),("original_resnet",RAW_REFERENCE)):
            key="delta_rmse_vs_"+label
            row[key]=m[key]=row["rmse"]-diagnostics[name]["overall"]["rmse"]
    reference=predictions[REFERENCE]
    for table,key in (("fold_metrics","fold"),("target_decile_metrics","decile"),
                      ("noise_quartile_metrics","quartile"),("target_region_metrics","region")):
        reference_scores={r[key]:r["rmse"] for r in tables[table] if r["model"]==REFERENCE}
        for row in tables[table]:
            row["delta_rmse_vs_dino_d4_fixed_blend"]=row["rmse"]-reference_scores[row[key]]
    old_error,new_error=reference-data.y,predictions[PRIMARY]-data.y
    old_top=set(np.argsort(-old_error**2,kind="stable")[:20]); new_top=set(np.argsort(-new_error**2,kind="stable")[:20])
    all_predictions={core.BASELINE_NAME:data.baseline,**predictions}
    for row in tables["metrics"]:
        pred=all_predictions[row["model"]]; error=pred-data.y
        top=set(np.argsort(-error**2,kind="stable")[:20])
        extra={"top20_overlap_with_dino_d4_fixed_blend":len(old_top&top),
            "residual_correlation_with_dino_d4_fixed_blend":core.correlation(old_error,error),
            "folds_improved_vs_dino_d4_fixed_blend":sum(core.metrics(data.y[data.folds==f],pred[data.folds==f])["rmse"]<
                core.metrics(data.y[data.folds==f],reference[data.folds==f])["rmse"] for f in range(5))}
        row.update(extra); diagnostics[row["model"]]["overall"].update(extra)
    tables["resnet_replacement_errors"]=[{"ID":str(ident),"fold":int(data.folds[i]),"hardness":float(data.y[i]),
        "reference_prediction":float(reference[i]),"new_prediction":float(predictions[PRIMARY][i]),
        "delta_squared_error":float(new_error[i]**2-old_error[i]**2),"in_reference_top20":i in old_top,"in_new_top20":i in new_top}
        for i,ident in enumerate(data.ids)]
    return diagnostics,tables


def build_markdown(summary):
    lines=["# ResNet D4 — DINO D4 고정, ResNet25%만 교체","",
        "사전 지정 주 결과는 resnet_d4_replaced_blend입니다. 원본/D4 표현 선택은 없고 각각의 alpha만 inner CV로 선택합니다.","",
        "| 모델 | OOF RMSE | DINO D4 고정 기준 Δ | 원래25:25:50 Δ |","|---|---:|---:|---:|"]
    for name,d in summary["diagnostics"].items():
        m=d["overall"]; lines.append(f"| {name} | {m['rmse']:.6f} | {m['delta_rmse_vs_dino_d4_fixed_blend']:+.6f} | {m['delta_rmse_vs_current_blend']:+.6f} |")
    lines += ["","## 내부 alpha 선택","","| fold | 원본 alpha | D4 alpha |","|---|---:|---:|"]
    for r in summary["selections"]:
        lines.append(f"| {r['outer_fold']} | {r['per_representation'][RAW]['alpha']:g} | {r['per_representation'][D4]['alpha']:g} |")
    lines += ["","## 약점 변화","","| 모델 | 양끝20% | 중간80% | 잡음Q4 |","|---|---:|---:|---:|"]
    for name in (previous.CURRENT,REFERENCE,PRIMARY):
        d=summary["diagnostics"][name]
        lines.append(f"| {name} | {d['target_regions']['tails20']['rmse']:.6f} | {d['target_regions']['middle80']['rmse']:.6f} | {d['noise_quartiles'][-1]['rmse']:.6f} |")
    lines += ["","## 해석과 재현","",
        "- 기준 = 수작업25% + 원본 ResNet25% + 고정 DINO D4 50%. 새 조합은 ResNet25%만 D4로 교체합니다. DINO OOF를 재학습하거나 고르지 않습니다.",
        "- 원본/D4 각각1792차원. layer2/3/4 평균·population 표준편차를 view별 계산 후8개 벡터를 평균합니다. 원본은256×256이며 별도 padding·보간·크롭을 하지 않습니다.",
        "- 원본/D4의 특징 종류를 multiscale로 사전 고정하고 alpha10/100/1000만 각 inner 학습 부분에서 선택합니다. 원본을 선택한 이전 joint OOF가 multiscale OOF와 정확히 같은지 확인합니다.",
        "- 5outer/3inner, X/y 표준화는 inner 학습 부분에서만 적합합니다. seed별(2표현×3alpha×3inner×5outer)+(2표현×5재적합)=100회 Ridge 적합입니다. 신경망·BatchNorm은 고정입니다.",
        f"- 원본 ResNet anchor: {summary['resnet_anchor']['status']}, 최대 예측 차이 {summary['resnet_anchor']['max_absolute_prediction_difference']:.3g}.",
        "- 이 기준은 앞선 DINO D4 결과를 본 뒤 고정한 후보입니다. 같은500장의 반복 연구 선택 편향이 남으며 두 seed는 독립 검증이 아닙니다. 원래 기준과partial80/160 및이전joint 조합도 비교용으로 보존합니다.",
        "- 개선 여부는 두seed의 주 조합을 고정 DINO D4 기준과 비교합니다. fold별 사후 교체·ensemble 가중치 탐색·clipping은 하지 않습니다. 원래 기준만 이기고 DINO D4 기준보다 나쁘면 ResNet 변경의 이득은 없습니다.",
        "- 두seed에서 이득이 일관되지 않으면 ResNet D4를 채택하지 않습니다. 극히 작은 차이는 후보 수준으로만 기록합니다. 잡음 지표는 실제 구조도 포함할 수 있습니다.",""]
    return "\n".join(lines)


def write_outputs(folder,data,predictions,selections,diagnostics,tables,seed,elapsed,anchor,*,threads=2):
    summary={"created_utc":datetime.now(timezone.utc).isoformat(),"experiment":"resnet_d4_with_fixed_dino_d4",
        "seed":seed,"n_train":len(data.y),"primary_metric":PRIMARY+" overall OOF RMSE","primary_model":PRIMARY,
        "comparison_reference":REFERENCE,"representation_selection":False,"alpha_candidates":list(ALPHAS),
        "outer_splits":5,"inner_splits":3,"estimator_fit_count":100,"elapsed_seconds":elapsed,
        "threads":threads,"versions":{name:importlib.metadata.version(name) for name in ("numpy","scipy","scikit-learn")},
        "feature_dimensions":{n:x.shape[1] for n,x in data.features.items()},"protocol":extract.PROTOCOL,
        "input_files":data.provenance,"feature_metadata":data.metadata,"resnet_anchor":anchor,
        "selections":selections,"diagnostics":diagnostics,"fixed_weights":{"hand":0.25,"resnet_d4":0.25,"dino_d4":0.5},
        "scope":"Train-only nested Ridge; fixed representations, no test, no all-data final fit."}
    oof=[{"ID":str(ident),"hardness":float(data.y[i]),"fold":int(data.folds[i]),core.BASELINE_NAME:float(data.baseline[i]),
          **{n:float(p[i]) for n,p in predictions.items()}} for i,ident in enumerate(data.ids)]
    search=[{"outer_fold":r["outer_fold"],"inner_seed":r["inner_seed"],"representation":c["representation"],"alpha":c["alpha"],
        "inner_mse":c["inner_mse"],**{f"inner_fold{j}_mse":v for j,v in enumerate(c["inner_fold_mse"])},
        "selected_for_representation":c==r["per_representation"][c["representation"]]} for r in selections for c in r["candidates"]]
    outputs={"summary.json":json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False),"report.md":build_markdown(summary),
        "oof.csv":core.csv_text(oof),"inner_search.csv":core.csv_text(search),**{f"{n}.csv":core.csv_text(v) for n,v in tables.items()}}
    folder.mkdir(parents=True,exist_ok=True)
    for name,content in outputs.items():
        temporary=folder/(name+f".{os.getpid()}.tmp"); temporary.write_text(content,encoding="utf-8",newline="")
        os.replace(temporary,folder/name)


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed",type=int,default=42); p.add_argument("--threads",type=int,default=2)
    p.add_argument("--original-features",type=Path,default=ROOT/"reports/frozen_resnet18/features.npz")
    p.add_argument("--resnet-d4-features",type=Path,default=ROOT/"reports/resnet_d4/features/features.npz")
    for name in ("original-oof","original-summary","dino-d4-oof","dino-d4-summary","output-dir"): p.add_argument("--"+name,type=Path)
    modes=p.add_mutually_exclusive_group(); modes.add_argument("--preflight",action="store_true"); modes.add_argument("--validate-only",action="store_true")
    p.add_argument("--overwrite",action="store_true"); args=p.parse_args(argv)
    for name,path in {"original_oof":ROOT/f"reports/frozen_resnet18/seed{args.seed}/oof.csv",
        "dino_d4_oof":ROOT/f"reports/dinov2_d4/seed{args.seed}/oof.csv","output_dir":ROOT/f"reports/resnet_d4/seed{args.seed}"}.items():
        if getattr(args,name) is None: setattr(args,name,path)
    if args.original_summary is None: args.original_summary=args.original_oof.with_name("summary.json")
    if args.dino_d4_summary is None: args.dino_d4_summary=args.dino_d4_oof.with_name("summary.json")
    return args


def main(argv=None):
    args=parse_args(argv); core.validate_seed(args.seed)
    if np is None or args.threads<1: raise ValueError("NumPy and positive threads required")
    if not (args.preflight or args.validate_only):
        if args.output_dir.exists() and (not args.output_dir.is_dir() or (any(args.output_dir.iterdir()) and not args.overwrite)):
            raise ValueError("Output already exists; choose a new directory")
        # Protect all previous source/results including paths of implicit references.
        dino_args=previous.parse_args(["--seed",str(args.seed)])
        paths=[v for ns in (args,dino_args) for n,v in vars(ns).items() if isinstance(v,Path) and n!="output_dir"]
        paths.append(dino_args.output_dir/"summary.json")
        if any(v.resolve().parent==args.output_dir.resolve() for v in paths): raise ValueError("Output overlaps protected inputs")
    start=time.perf_counter(); data,refs,anchor=load_prior(args)
    if args.preflight:
        print(json.dumps({"status":"prior_artifacts_valid","seed":args.seed,"rows":len(data.y),
            "comparison_reference_rmse":core.metrics(data.y,refs[REFERENCE])["rmse"],"resnet_d4_cache_checked":False,
            "estimators_fitted":0,"output_written":False})); return 0
    data=attach_features(data,args.resnet_d4_features,args.original_features)
    if args.validate_only:
        print(json.dumps({"status":"all_artifacts_valid","seed":args.seed,"feature_dimensions":{n:x.shape[1] for n,x in data.features.items()},
            "estimators_fitted":0,"output_written":False})); return 0
    predictions,selections=nested_predictions(data,args.seed,args.threads)
    check=verify_anchor(data,predictions,selections,anchor)
    predictions=add_fixed_blend(predictions,refs); diagnostics,tables=diagnose(data,predictions)
    verify_hashes(data.provenance)
    write_outputs(args.output_dir,data,predictions,selections,diagnostics,tables,args.seed,time.perf_counter()-start,check,threads=args.threads)
    print(f"ResNet anchor: {check['status']}")
    for name in (D4+"_nested",PRIMARY):
        m=diagnostics[name]["overall"]
        print(f"{'PRIMARY' if name==PRIMARY else 'COMPONENT'} {name}: RMSE={m['rmse']:.6f}; delta vs DINO D4 fixed blend={m['delta_rmse_vs_dino_d4_fixed_blend']:+.6f}; delta vs original blend={m['delta_rmse_vs_current_blend']:+.6f}")
    print(f"Report: {(args.output_dir/'report.md').resolve()}")
    return 0


if __name__=="__main__":
    try: raise SystemExit(main())
    except (ValueError,OSError,KeyError) as exc:
        print(f"ResNet D4 evaluation stopped: {exc}",file=sys.stderr); raise SystemExit(2)
