"""Label-free watershed/dark-region proxies. Preview 25 train images; user extracts all 500."""
from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi
from skimage import filters, measure, morphology, segmentation, restoration

if __package__:
    from . import extract_local_structure as previous
else:
    import extract_local_structure as previous

ROOT, common = previous.ROOT, previous.common
OUT = ROOT/"reports/morphology"
SCALES = (1.0, 2.0)
REGION_NAMES = ["log_count", "log_interior_count", "border_area_fraction", "largest_area_fraction",
    "boundary_fraction", "log_marker_count", "within_variance_ratio",
    *["log_area_"+s for s in ("mean","std","q10","q50","q90")],
    "diameter_mean", "diameter_q50", "inverse_sqrt_diameter_mean",
    *["log_axis_ratio_"+s for s in ("mean","q50","q90")],
    "eccentricity_mean", "solidity_mean", "circularity_mean", "within_std_mean", "between_mean_std",
    "area_cv", *["interior_log_area_"+s for s in ("q10","q50","q90")], "boundary_barrier_mean"]
DARK_NAMES = ["fraction", "log_count", "largest_area_fraction", "border_area_fraction",
    *["log_area_"+s for s in ("mean","std","q10","q50","q90")],
    *["log_axis_ratio_"+s for s in ("mean","q50","q90")],
    *["circularity_"+s for s in ("mean","q50","q90")],
    "solidity_mean", "eccentricity_mean", "extent_mean", "opened_fraction", "log_opened_count"]


def feature_names():
    return ([f"watershed_s{s:g}_{n}" for s in SCALES for n in REGION_NAMES]
        + [f"dark_t{t}_{n}" for t in (25,50) for n in DARK_NAMES]
        + ["cross_log_count_ratio", "cross_log_area_ratio", "cross_log_axis_ratio_difference", "cross_boundary_iou"])


PROTOCOL = {
    "version":1,"dimensions":100,"combined_dimensions":285,"feature_names":feature_names(),
    "input":"256x256 uint8 grayscale; float64/255; no resizing or learned preprocessing",
    "denoising":"per-image orthonormal 2x2 Haar HH MAD/0.6744897501960817 noise proxy; fast nonlocal means h=1.1*sigma, patch_size5, patch_distance6, sigma correction; no learned model",
    "contrast":"per-image denoised p2/p98 normalization clipped to [0,1]; original185 retained separately; constant span<=1e-8 gives zeros",
    "regions":{"sigmas":list(SCALES),"smoothing":"Gaussian reflect truncate4",
        "gradient":"barrier=Sobel/p95 + black_tophat(disk3)/p95; divide by barrier p95 then clip [0,1]; zero denominators bounded1e-12",
        "markers":"h_minima(barrier,h=0.20,8-connectivity); retain single-pixel minima; fallback argmin",
        "watershed":"4-connected gradient watershed, compactness0, no watershed line; all pixels covered",
        "units":"pixels; region proxies, not validated physical grain labels; border regions flagged separately"},
    "dark":{"sigma":1.2,"thresholds":[.25,.50],"connectivity":8,"minimum_area":5,
        "opening":"binary opening disk1; count components >=5 pixels",
        "interpretation":"dark connected objects, potentially including grain boundaries, phases and pores; no semantic labels"},
    "statistics":"population std, linear quantiles; empty summaries=0; log1p area/count; log clipped major/minor ratio[1,20]; circularity clipped[0,1] using Crofton perimeter",
    "selection":"fixed before regression scores; both scales concatenated, no score-selected segmentation",
    "labels_used":False,"test_accessed":False,
}
assert len(REGION_NAMES)==28 and len(DARK_NAMES)==20 and len(feature_names())==100


def versions():
    return {n:importlib.metadata.version(n) for n in ("numpy","scipy","Pillow","scikit-image")}


def source_refs():
    return {Path(p).name:previous.file_reference(Path(p)) for p in (__file__,previous.__file__,common.__file__)}


def write_json(path, value):
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding="utf-8")


def read_image(path):
    with Image.open(path) as im:
        if (im.mode,im.size,im.format)!=("L",(256,256),"PNG"):
            raise ValueError("Expected 256x256 grayscale PNG")
        return np.array(im,dtype=np.uint8)


def normalize(gray):
    gray=np.asarray(gray)
    if gray.shape!=(256,256) or gray.dtype!=np.uint8:
        raise ValueError("Expected uint8 256x256")
    raw=gray.astype(np.float64)/255.
    hh=(raw[::2,::2]-raw[::2,1::2]-raw[1::2,::2]+raw[1::2,1::2])/2
    sigma=float(np.median(np.abs(hh-np.median(hh)))/0.6744897501960817)
    if sigma>1e-8:
        raw=restoration.denoise_nl_means(raw,h=1.1*sigma,sigma=sigma,fast_mode=True,
            patch_size=5,patch_distance=6,preserve_range=True,channel_axis=None)
    low,high=np.quantile(raw,[.02,.98])
    return np.clip((raw-low)/(high-low),0,1) if high-low>1e-8 else np.zeros_like(raw)


def filter_components(mask, minimum):
    labels,_=ndi.label(mask,structure=np.ones((3,3)))
    counts=np.bincount(labels.ravel())
    keep=counts>=minimum;keep[0]=False
    return ndi.label(keep[labels],structure=np.ones((3,3)))[0]


def region_map(a,sigma):
    smooth=ndi.gaussian_filter(a,sigma,mode="reflect",truncate=4.)
    grad=filters.sobel(smooth,mode="reflect")
    dark=morphology.black_tophat(smooth,footprint=morphology.disk(3))
    grad=grad/max(float(np.quantile(grad,.95)),1e-12)+dark/max(float(np.quantile(dark,.95)),1e-12)
    p95=float(np.quantile(grad,.95))
    if p95<1e-12:
        labels=np.ones(a.shape,dtype=np.int32)
        return labels,np.zeros_like(a),1
    grad=np.clip(grad/p95,0,1)
    minima=morphology.h_minima(grad,.20,footprint=np.ones((3,3),dtype=bool))
    markers=ndi.label(minima,structure=np.ones((3,3)))[0]
    count=int(markers.max())
    if count==0:
        markers.flat[int(np.argmin(grad))]=1;count=1
    labels=segmentation.watershed(grad,markers,connectivity=1,compactness=0,watershed_line=False).astype(np.int32)
    return labels,grad,count


def summaries(values):
    v=np.asarray(values,dtype=np.float64)
    return [float(v.mean()),float(v.std()),*np.quantile(v,[.1,.5,.9]).tolist()] if len(v) else [0.]*5


def property_arrays(labels,a):
    regions=measure.regionprops(labels,intensity_image=a)
    area=np.array([r.area for r in regions],dtype=np.float64)
    ratio=np.array([np.log(np.clip(r.axis_major_length/max(r.axis_minor_length,1e-8),1,20)) for r in regions])
    circular=np.array([np.clip(4*np.pi*r.area/max(r.perimeter_crofton**2,1e-8),0,1) for r in regions])
    border=np.array([r.bbox[0]==0 or r.bbox[1]==0 or r.bbox[2]==a.shape[0] or r.bbox[3]==a.shape[1] for r in regions],dtype=bool)
    return regions,area,ratio,circular,border


def region_features(labels,grad,a,marker_count):
    regions,area,ratio,circular,border=property_arrays(labels,a)
    boundary=segmentation.find_boundaries(labels,connectivity=1,mode="inner")
    std=np.array([r.intensity_std for r in regions])
    diameter=np.sqrt(4*area/np.pi)
    logarea=summaries(np.log1p(area));axis=summaries(ratio)
    interior=summaries(np.log1p(area[~border]))
    values=[np.log1p(len(area)),np.log1p(np.sum(~border)),area[border].sum()/a.size,area.max()/a.size,
        boundary.mean(),np.log1p(marker_count),float(np.sum(area*std**2)/a.size/max(float(a.var()),1e-12)),
        *logarea,diameter.mean(),np.median(diameter),np.mean(1/np.sqrt(np.maximum(diameter,1e-8))),
        axis[0],axis[3],axis[4],np.mean([r.eccentricity for r in regions]),np.mean([r.solidity for r in regions]),
        circular.mean(),std.mean(),np.std([r.intensity_mean for r in regions]),area.std()/max(area.mean(),1e-8),
        *interior[2:],float(grad[boundary].mean()) if boundary.any() else 0.]
    return values,boundary,{"regions":len(area),"interior_regions":int(np.sum(~border)),"median_area":float(np.median(area)),
        "largest_area_fraction":float(area.max()/a.size),"within_variance_ratio":float(values[6])}


def dark_features(a,threshold):
    smooth=ndi.gaussian_filter(a,1.2,mode="reflect",truncate=4.)
    labels=filter_components(smooth<threshold,5)
    regions,area,ratio,circular,border=property_arrays(labels,a)
    logarea=summaries(np.log1p(area));axis=summaries(ratio);circ=summaries(circular)
    mask=labels>0
    opened=filter_components(ndi.binary_opening(mask,structure=morphology.disk(1)),5)
    values=[mask.mean(),np.log1p(len(area)),area.max()/a.size if len(area) else 0.,area[border].sum()/a.size,
        *logarea,axis[0],axis[3],axis[4],circ[0],circ[3],circ[4],
        *[float(np.mean([getattr(r,p) for r in regions])) if regions else 0. for p in ("solidity","eccentricity","extent")],
        (opened>0).mean(),np.log1p(opened.max())]
    return values,mask


def morphology_features(gray,*,return_maps=False):
    a=normalize(gray);values=[];maps=[];boundaries=[];qc=[]
    for sigma in SCALES:
        labels,grad,count=region_map(a,sigma)
        v,b,q=region_features(labels,grad,a,count)
        values.extend(v);maps.append(labels);boundaries.append(b);qc.append(q)
    masks=[]
    for t in (.25,.5):
        v,mask=dark_features(a,t);values.extend(v);masks.append(mask)
    values.extend([np.log((qc[1]["regions"]+1)/(qc[0]["regions"]+1)),
        np.log((np.exp(values[28+7])+1e-12)/(np.exp(values[7])+1e-12)),
        values[28+15]-values[15],np.sum(boundaries[0]&boundaries[1])/max(np.sum(boundaries[0]|boundaries[1]),1)])
    result=np.asarray(values,dtype=np.float64)
    if result.shape!=(100,) or not np.isfinite(result).all():
        raise ValueError("Invalid morphology features")
    if return_maps:
        return result,{"regions":maps,"dark":masks,"qc":qc}
    return result


def preview_selection(rows):
    """Uses only ID and two unlabeled audit columns. Never reads hardness."""
    ids=[r["ID"] for r in rows]
    rank=lambda v:np.argsort(np.argsort(v,kind="stable"),kind="stable")/max(len(v)-1,1)
    noise=rank([float(r["haar_hh_noise_proxy"]) for r in rows])
    contrast=rank([float(r["variance_minus_noise_proxy"]) for r in rows])
    used=set();selected=[]
    for ni in range(5):
        for ci in range(5):
            distance=(noise-(ni+.5)/5)**2+(contrast-(ci+.5)/5)**2
            j=min((j for j in range(len(rows)) if j not in used),key=lambda j:(distance[j],ids[j]))
            used.add(j)
            selected.append({"ID":ids[j],"noise_grid":ni+1,"contrast_grid":ci+1,
                "noise_rank":float(noise[j]),"contrast_rank":float(contrast[j])})
    return selected


def preview(folder):
    previous.protect_output(folder,[ROOT/"open/train.csv",Path(__file__)])
    audit=ROOT/"reports/audit_train_features.csv"
    with audit.open(encoding="utf-8-sig",newline="") as h:rows=list(csv.DictReader(h))
    selected=preview_selection(rows)
    folder.mkdir(parents=True,exist_ok=True)
    font=ImageFont.truetype("C:/Windows/Fonts/arial.ttf",16)
    records=[];started=time.perf_counter()
    for page in range(5):
        sheet=Image.new("RGB",(1068,1500),"#f6f7f9");draw=ImageDraw.Draw(sheet)
        draw.text((12,8),f"TRAIN ONLY | noise grid {page+1}/5 | No targets used | regions are proxies",fill="black",font=font)
        for column,title in enumerate(("Original","Watershed sigma 1","Watershed sigma 2","Dark: red<.25 / yellow<.50")):
            draw.text((12+column*264,36),title,fill="black",font=font)
        for row,record in enumerate(selected[page*5:page*5+5]):
            gray=read_image(ROOT/"open/train"/(record["ID"]+".png"))
            _,detail=morphology_features(gray,return_maps=True)
            imgs=[np.repeat(gray[...,None],3,axis=2)]
            for labels in detail["regions"]:
                im=imgs[0].copy();im[segmentation.find_boundaries(labels,mode="inner")]=[0,220,130];imgs.append(im)
            im=imgs[0].copy();im[detail["dark"][1]]=(.6*im[detail["dark"][1]]+.4*np.array([255,210,0])).astype(np.uint8)
            im[detail["dark"][0]]=(.5*im[detail["dark"][0]]+.5*np.array([255,30,30])).astype(np.uint8);imgs.append(im)
            y=64+row*286
            for col,im in enumerate(imgs):sheet.paste(Image.fromarray(im),(12+col*264,y))
            draw.text((12,y+260),f"{record['ID']} | contrast grid {record['contrast_grid']} | regions={detail['qc'][0]['regions']}/{detail['qc'][1]['regions']}",fill="black",font=font)
            records.append({**record,"image_sha256":common.sha256_file(ROOT/"open/train"/(record["ID"]+".png")),"qc":detail["qc"]})
        sheet.save(folder/f"preview_{page+1}.png")
    write_json(folder/"preview.json",{"protocol":PROTOCOL,"sources":source_refs(),"versions":versions(),
        "selection":"nearest unused to 5x5 noise/contrast rank grid, ID tie-break, no targets",
        "audit_sha256":common.sha256_file(audit),"records":records,"images_segmented":25,
        "regression_fits":0,"test_accessed":False,"elapsed_seconds":time.perf_counter()-started})
    print(f"Previewed 25 train images: {folder}; no targets, fitting or test images used")


def extract(folder,plan_path):
    # Evaluation protocol prepared before extraction; it binds the exact source/preview.
    if __package__:
        from . import evaluate_morphology as evaluation
    else:
        import evaluate_morphology as evaluation
    data=evaluation.load_base();plan=evaluation.validate_plan(plan_path.parent,data)
    previous.protect_output(folder,[plan_path,*[Path(r["path"]) for r in data.provenance.values()]])
    common.ensure_inputs_unchanged(ROOT/"open",{"train_csv_sha256":data.metadata["resnet"]["train_csv_sha256"],"images":data.metadata["resnet"]["training_image_manifest"]})
    start=time.perf_counter();features=[]
    for i,ident in enumerate(data.ids):
        features.append(morphology_features(read_image(ROOT/"open/train"/(str(ident)+".png"))))
        if (i+1)%50==0:print(f"Extracted {i+1}/{len(data.ids)} train morphology images; elapsed={time.perf_counter()-start:.1f}s",flush=True)
    x=np.stack(features)
    evaluation.validate_plan(plan_path.parent,data)
    common.ensure_inputs_unchanged(ROOT/"open",{"train_csv_sha256":data.metadata["resnet"]["train_csv_sha256"],"images":data.metadata["resnet"]["training_image_manifest"]})
    arrays={"ids":data.ids,"morphology":x,"feature_names":np.asarray(feature_names())}
    meta={"protocol":PROTOCOL,"sources":source_refs(),"versions":versions(),"rows":len(x),
        "plan_sha256":common.sha256_file(plan_path),"train_binding":plan["frozen"]["train_binding"],
        "array_sha256":{k:common.sha256_array(v) for k,v in arrays.items()},"test_accessed":False,"labels_saved":False,
        "elapsed_seconds":time.perf_counter()-start}
    folder.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(folder/"features.npz",**arrays,metadata_json=np.asarray(common.canonical_json(meta)))
    write_json(folder/"metadata.json",meta)
    print(f"Saved {folder/'features.npz'}; morphology={x.shape}; combined=(500,285)")


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview",action="store_true")
    parser.add_argument("--output-dir",type=Path)
    parser.add_argument("--plan",type=Path,default=OUT/"protocol.json")
    args=parser.parse_args(argv)
    if args.preview:preview(args.output_dir or OUT/"preview_v2")
    else:extract(args.output_dir or OUT/"features",args.plan)


if __name__=="__main__":
    main()
