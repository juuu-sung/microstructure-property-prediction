"""User-run train-only ResNet18 D4 mean features; local checkpoint only.

--validate-only hashes original inputs/cache/checkpoint, without torch or forward.
--check-only --device cpu uses random weights and synthetic input only.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import uuid

if __package__:
    from . import extract_frozen_resnet as base
    from . import extract_dinov2_d4 as geometry
    from . import evaluate_frozen_resnet as core
else:
    import extract_frozen_resnet as base
    import extract_dinov2_d4 as geometry
    import evaluate_frozen_resnet as core

ROOT = base.ROOT
RAW = "multiscale_mean_std"
FEATURE = "d4_mean_multiscale_mean_std"
WIDTH = 1792
MATCH_ATOL, MATCH_RTOL = 1e-5, 1e-6
ORIGINAL_CONFIG = {"schema_version": 1, "model_id": base.HF_REPO, "revision": base.DEFAULT_REVISION,
    "feature_indices": [2,3,4], "channels": [128,256,512], "input_shape": [3,256,256],
    "normalization_mean": base.EXPECTED_MEAN, "normalization_std": base.EXPECTED_STD,
    "resize": None, "crop": None, "augmentation": None, "dtype": "float32", "amp": False,
    "train_mode": False,
    "pooling": "final layer4 spatial mean; concat layer2 mean,std,layer3 mean,std,layer4 mean,std; correction=0"}
PROTOCOL = {"view_order": geometry.PROTOCOL["view_order"], "transforms": geometry.PROTOCOL["transforms"],
    "input_shape": [3,256,256], "padding": None, "interpolation": None, "views_per_image": 8,
    "pool_each_view": "layer2 mean,std; layer3 mean,std; layer4 mean,std; population std",
    "view_reduction": "arithmetic mean; FP64 accumulation then FP32 storage",
    "feature_dimension": WIDTH, "training_rows_per_image": 1, "batchnorm": "eval; running statistics frozen"}


def source_hashes():
    return {n: base.sha256_file(Path(p)) for n,p in
            {"extractor":__file__, "resnet_utilities":base.__file__, "d4_geometry":geometry.__file__}.items()}


def reference(path):
    return {"path":str(path.resolve()), "sha256":base.sha256_file(path),
        "metadata_path":str(path.with_name("metadata.json").resolve()),
        "metadata_sha256":base.sha256_file(path.with_name("metadata.json"))}


def validate_values(np, values, count):
    if values.dtype != np.float32 or values.shape != (count,WIDTH) or not np.isfinite(values).all():
        raise ValueError("Expected finite FP32 (n,1792) ResNet features")
    if any(np.any(values[:,a:b]<0) for a,b in ((128,256),(512,768),(1280,1792))):
        raise ValueError("Population standard deviations must be nonnegative")


def read_original(np, path, train_csv, *, expected_n=500):
    ids = np.asarray([r["ID"] for r in core.read_rows(train_csv,{"ID","hardness"})])
    if len(ids)!=expected_n or any(not ident.startswith("TRAIN_") for ident in ids):
        raise ValueError("Unexpected training row count or ID namespace")
    with np.load(path,allow_pickle=False) as cache:
        if len(cache.files)!=4 or set(cache.files)!=base.FEATURE_KEYS:
            raise ValueError("Unexpected original ResNet cache schema")
        saved, values, gap, raw = cache["ids"], cache[RAW], cache["final_gap"], cache["metadata_json"]
        if saved.dtype.kind!="U" or saved.shape!=ids.shape or len(set(saved))!=len(ids) or set(saved)!=set(ids):
            raise ValueError("Original ResNet IDs differ")
        validate_values(np,values,len(ids))
        if gap.dtype!=np.float32 or gap.shape!=(len(ids),512) or not np.array_equal(gap,values[:,768:1280]):
            raise ValueError("Original GAP does not match layer4 mean block")
        if raw.shape!=() or raw.dtype.kind!="U":
            raise ValueError("Original metadata must be a Unicode scalar")
        metadata=json.loads(raw.item())
        if metadata!=json.loads(path.with_name("metadata.json").read_text(encoding="utf-8")):
            raise ValueError("Original ResNet metadata sidecar mismatch")
        if (metadata.get("schema_version")!=1 or metadata.get("rows")!=len(ids) or
                metadata.get("test_accessed") is not False or metadata.get("labels_saved_in_npz") is not False):
            raise ValueError("Original ResNet scope mismatch")
        for key,value in ORIGINAL_CONFIG.items():
            if metadata.get("extraction_config",{}).get(key)!=value:
                raise ValueError(f"Original ResNet configuration mismatch: {key}")
        if metadata.get("train_csv_sha256")!=base.sha256_file(train_csv):
            raise ValueError("Original ResNet train.csv changed")
        geometry.dino.require_sha256(metadata.get("feature_state_sha256"),"ResNet state")
        manifest=metadata.get("training_image_manifest",[])
        if [m.get("id") for m in manifest]!=saved.tolist() or any(m.get("relative_path")!=f"train/{m['id']}.png" for m in manifest):
            raise ValueError("Original ResNet image manifest order or paths differ")
        digest=hashlib.sha256(base.canonical_json(manifest).encode()).hexdigest()
        if digest!=metadata.get("image_manifest_sha256"):
            raise ValueError("Original ResNet image manifest hash mismatch")
        if metadata.get("feature_array_sha256")!={"final_gap":base.sha256_array(gap),RAW:base.sha256_array(values)}:
            raise ValueError("Original ResNet array hash mismatch")
        lookup={ident:i for i,ident in enumerate(saved)}
        return ids, values[[lookup[i] for i in ids]].copy(), metadata


def validate_cache(np,path,ids,original_metadata,original_path):
    with np.load(path,allow_pickle=False) as cache:
        if len(cache.files)!=3 or set(cache.files)!={"ids",FEATURE,"metadata_json"}:
            raise ValueError("Unexpected ResNet D4 cache schema")
        saved,values,raw=cache["ids"],cache[FEATURE],cache["metadata_json"]
        if saved.dtype.kind!="U" or saved.shape!=ids.shape or len(set(saved))!=len(ids) or set(saved)!=set(ids):
            raise ValueError("ResNet D4 IDs differ")
        validate_values(np,values,len(ids))
        if raw.shape!=() or raw.dtype.kind!="U":
            raise ValueError("D4 metadata must be a Unicode scalar")
        metadata=json.loads(raw.item())
        if metadata!=json.loads(path.with_name("metadata.json").read_text(encoding="utf-8")):
            raise ValueError("ResNet D4 metadata sidecar mismatch")
        if (metadata.get("schema_version")!=1 or metadata.get("rows")!=len(ids) or metadata.get("protocol")!=PROTOCOL or
                metadata.get("test_accessed") is not False or metadata.get("labels_saved_in_npz") is not False or
                metadata.get("offline_checkpoint_only") is not True):
            raise ValueError("ResNet D4 protocol/scope mismatch")
        for key in ("feature_state_sha256","train_csv_sha256","image_manifest_sha256"):
            if metadata.get(key)!=original_metadata[key]:
                raise ValueError(f"ResNet D4/original source mismatch: {key}")
        original=reference(original_path)
        if any(metadata.get("original_reference",{}).get(k)!=original[k] for k in ("sha256","metadata_sha256")):
            raise ValueError("ResNet D4 references a different original cache")
        expected={**ORIGINAL_CONFIG,"augmentation":PROTOCOL,"pooling":PROTOCOL["pool_each_view"]+"; D4 mean"}
        for key,value in expected.items():
            if metadata.get("extraction_config",{}).get(key)!=value:
                raise ValueError(f"ResNet D4 configuration mismatch: {key}")
        if metadata.get("source_sha256")!=source_hashes():
            raise ValueError("ResNet D4 sources changed; use a new cache")
        check=metadata.get("identity_reextraction_check",{}); difference=check.get("max_absolute_difference")
        if (check.get("passed") is not True or check.get("atol")!=MATCH_ATOL or check.get("rtol")!=MATCH_RTOL or
                not isinstance(difference,(int,float)) or not np.isfinite(difference) or difference<0):
            raise ValueError("Missing ResNet identity re-extraction verification")
        if metadata.get("feature_array_sha256")!={FEATURE:base.sha256_array(values)}:
            raise ValueError("ResNet D4 array hash mismatch")
        lookup={ident:i for i,ident in enumerate(saved)}
        return values[[lookup[i] for i in ids]].copy(), metadata


def build_model(torch,timm,resolve_config,*,pretrained):
    name=f"hf-hub:{base.HF_REPO}@{base.DEFAULT_REVISION}" if pretrained else base.MODEL_NAME
    model=timm.create_model(name,pretrained=pretrained,features_only=True,out_indices=(2,3,4))
    config=resolve_config(model)
    if (list(model.feature_info.channels())!=base.EXPECTED_CHANNELS or list(config["mean"])!=base.EXPECTED_MEAN or
            list(config["std"])!=base.EXPECTED_STD):
        raise ValueError("ResNet architecture or normalization changed")
    model.requires_grad_(False); model.eval()
    if any(m.training for m in model.modules()) or any(p.requires_grad for p in model.parameters()):
        raise ValueError("ResNet including BatchNorm must be completely frozen")
    return model


def extract_batch(torch,model,inputs):
    if inputs.ndim!=4 or inputs.shape[1:]!=(3,256,256) or inputs.dtype!=torch.float32:
        raise ValueError("Expected FP32 NCHW (batch,3,256,256)")
    parts=[]
    for view in geometry.d4_views(torch,inputs):
        _, pooled=base.pool_feature_maps(torch,model(view))
        parts.append(pooled)
    return parts[0],torch.stack(parts).double().mean(dim=0).float()


def atomic_save(np,output,ids,values,metadata):
    output.mkdir(parents=True,exist_ok=True)
    nonce=uuid.uuid4().hex; temporary=[output/f".{nonce}.npz.tmp",output/f".{nonce}.json.tmp"]
    try:
        with temporary[0].open("wb") as f:
            np.savez_compressed(f,ids=np.asarray(ids,dtype=str),**{FEATURE:values},metadata_json=np.asarray(base.canonical_json(metadata)))
        temporary[1].write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding="utf-8")
        for p,name in zip(temporary,("features.npz","metadata.json")):
            os.replace(p,output/name)
    finally:
        for p in temporary:
            if p.exists(): p.unlink()


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir",type=Path,default=ROOT/"open")
    p.add_argument("--original-features",type=Path,default=ROOT/"reports/frozen_resnet18/features.npz")
    p.add_argument("--output-dir",type=Path,default=ROOT/"reports/resnet_d4/features")
    p.add_argument("--device",default="cuda"); p.add_argument("--batch-size",type=int,default=16)
    p.add_argument("--threads",type=int,default=2); p.add_argument("--num-workers",type=int,default=0)
    p.add_argument("--seed",type=int,default=42)
    modes=p.add_mutually_exclusive_group()
    for flag in ("validate-only","check-only","reuse"): modes.add_argument("--"+flag,action="store_true")
    p.add_argument("--overwrite",action="store_true")
    args=p.parse_args(argv); args.revision=base.DEFAULT_REVISION
    if args.batch_size<1 or args.threads<1 or args.num_workers<0 or not 0<=args.seed<2**32:
        p.error("Invalid batch size, threads, workers or seed")
    if args.overwrite and (args.validate_only or args.check_only or args.reuse): p.error("Cannot overwrite in check/reuse modes")
    return args


def main(argv=None):
    args=parse_args(argv)
    os.environ["HF_HUB_OFFLINE"]="1"; os.environ["TRANSFORMERS_OFFLINE"]="1"
    import numpy as np
    if not (args.validate_only or args.check_only or args.reuse):
        out=args.output_dir.resolve()
        if out==args.original_features.resolve().parent or out.is_relative_to(args.data_dir.resolve()):
            raise ValueError("Output would overwrite protected inputs")
        if out.exists() and (not out.is_dir() or (any(out.iterdir()) and not args.overwrite)):
            raise ValueError("Output already exists; use --reuse or a new directory")
    if not args.check_only:
        ids,original,metadata=read_original(np,args.original_features,args.data_dir/"train.csv")
        ref=reference(args.original_features)
        inventory=base.training_inventory(args.data_dir)
        for key in ("train_csv_sha256","image_manifest_sha256"):
            if inventory[key]!=metadata[key]: raise ValueError("Training inventory changed")
        if args.reuse:
            validate_cache(np,args.output_dir/"features.npz",ids,metadata,args.original_features)
            print("ResNet D4 cache verified; no model loaded."); return 0
        checkpoints=base.cached_checkpoint_files(base.DEFAULT_REVISION); names={c["filename"] for c in checkpoints}
        if "config.json" not in names or not names.intersection({"model.safetensors","pytorch_model.bin"}):
            raise ValueError("Pinned ResNet checkpoint absent locally; no downloads attempted")
        if args.validate_only:
            print(json.dumps({"status":"original_cache_images_checkpoint_valid","rows":len(ids),"feature_dimension":WIDTH,
                "model_loaded":False,"torch_imported":"torch" in sys.modules,"output_written":False})); return 0
    np,torch,timm,resolve_config,versions=base.import_runtime()
    device,info=base.configure_runtime(torch,np,args)
    model=build_model(torch,timm,resolve_config,pretrained=not args.check_only).to(device=device,dtype=torch.float32)
    state=base.feature_state_hash(model)
    if args.check_only:
        with torch.inference_mode():
            identity,averaged=extract_batch(torch,model,torch.randn(1,3,256,256,device=device))
        if base.feature_state_hash(model)!=state: raise ValueError("Frozen model/BatchNorm state changed")
        print(json.dumps({"status":"synthetic_forward_ok","identity_shape":list(identity.shape),"d4_shape":list(averaged.shape),
            "model_state_unchanged":True,"pretrained":False,"data_accessed":False,"output_written":False})); return 0
    if state!=metadata["feature_state_sha256"]: raise ValueError("Loaded ResNet state differs from original cache")
    sources=source_hashes()
    dataset=base.TrainImageDataset(args.data_dir,ids.tolist(),base.EXPECTED_MEAN,base.EXPECTED_STD)
    loader=torch.utils.data.DataLoader(dataset,batch_size=args.batch_size,shuffle=False,drop_last=False,
        num_workers=args.num_workers,pin_memory=device.type=="cuda",worker_init_fn=base.seed_worker,
        generator=torch.Generator().manual_seed(args.seed))
    if device.type=="cuda": torch.cuda.synchronize(device)
    start=time.perf_counter(); processed=0; parts=[]; maximum=0.
    with torch.inference_mode():
        for step,inputs in enumerate(loader,start=1):
            inputs=inputs.to(device=device,dtype=torch.float32,non_blocking=device.type=="cuda")
            identity,averaged=extract_batch(torch,model,inputs)
            actual=identity.cpu().numpy(); previous=original[processed:processed+len(actual)]
            if not np.allclose(actual,previous,atol=MATCH_ATOL,rtol=MATCH_RTOL):
                raise ValueError("Original ResNet identity features did not reproduce; no cache saved")
            maximum=max(maximum,float(np.max(np.abs(actual-previous))))
            parts.append(averaged.cpu().numpy().copy()); processed+=len(inputs)
            if step==1 or step%5==0 or processed==len(ids):
                print(f"Extracted {processed}/{len(ids)} ResNet D4 images (8 views each); elapsed={time.perf_counter()-start:.1f}s",flush=True)
    values=np.concatenate(parts); validate_values(np,values,len(ids))
    base.ensure_inputs_unchanged(args.data_dir,inventory)
    if reference(args.original_features)!=ref or source_hashes()!=sources or base.feature_state_hash(model)!=state:
        raise ValueError("Input, source, model or BatchNorm state changed")
    config=base.extraction_config(args,versions,device,info)
    config.update(augmentation=PROTOCOL,pooling=PROTOCOL["pool_each_view"]+"; D4 mean",extractor_script_sha256=sources["extractor"])
    saved={"schema_version":1,"created_utc":datetime.now(timezone.utc).isoformat(),"rows":len(ids),"protocol":PROTOCOL,
        "extraction_config":config,"original_reference":ref,"source_sha256":sources,
        **{k:metadata[k] for k in ("train_csv_sha256","image_manifest_sha256","feature_state_sha256")},
        "feature_array_sha256":{FEATURE:base.sha256_array(values)},"identity_reextraction_check":{
            "passed":True,"atol":MATCH_ATOL,"rtol":MATCH_RTOL,"max_absolute_difference":maximum},
        "test_accessed":False,"labels_saved_in_npz":False,"offline_checkpoint_only":True,
        "cached_checkpoint_files":checkpoints,"elapsed_extraction_seconds":time.perf_counter()-start}
    atomic_save(np,args.output_dir,ids,values,saved)
    validate_cache(np,args.output_dir/"features.npz",ids,metadata,args.original_features)
    print(f"Saved {args.output_dir/'features.npz'}; D4={values.shape}; max identity difference={maximum:.3g}")
    return 0


if __name__=="__main__":
    try: raise SystemExit(main())
    except (ValueError,RuntimeError,OSError,KeyError) as exc:
        print(f"ResNet D4 extraction stopped: {exc}",file=sys.stderr); raise SystemExit(2)
