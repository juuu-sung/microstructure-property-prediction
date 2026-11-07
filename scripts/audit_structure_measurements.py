"""Small, label-free measurement audit. No regression, GPU, test images or full extraction.

prepare freezes 25 IDs and a single diagnostic profile before image inspection.
run evaluates it on 15 development / 10 check images without parameter tuning.
All physical interpretations remain hypotheses; compact dark objects are NOT pore labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi
from skimage import measure, morphology, segmentation

if __package__:
    from . import extract_morphology as old
else:
    import extract_morphology as old

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/structure_measurement_audit"
AUDIT = ROOT / "reports/audit_train_features.csv"
PRIOR = ROOT / "reports/morphology/preview_v2/preview.json"
COLS = ["ID", "haar_hh_noise_proxy", "variance_minus_noise_proxy",
        "gradient_x_to_y", "autocorr_lag_4", "autocorr_lag_16", "fraction_below_64"]
PROFILE = {
    "version": 1,
    "selection": "Exclude previous25; noise quintiles, five rank-space farthest-point representatives each. First nearest feature median; ties ID. Structure coordinates: contrast, gradient X/Y, autocorrelation4/16, dark fraction64. No targets.",
    "split": "Per quintile, SHA256(ID+measurement-v1) sorted: first3 development, last2 check. One fixed profile; no fitting/tuning on either subset.",
    "normalization": "Unchanged prior per-image NLM and p2/p98; shared preprocessing can bias all diagnostic branches.",
    "line": "Gaussian sigma1; black_tophat disk3 >0.08; remove connected objects under8px, subtract compact objects dilated disk1; no watershed. Six symmetric H/V transects at32,64,96,159,191,223. Runs>12px excluded as unresolved broad/parallel objects; edge-censored intervals excluded. Spacing is dark-line spacing, NOT calibrated grain diameter.",
    "compact": "Gaussian sigma1 normalized intensity<0.25; binary opening disk2; 8-connected area12..512; axis ratio<=2, Crofton circularity>=0.45; non-border; local disk3 annulus median minus component median>=0.12. Compact dark candidate, not pore truth.",
    "granulometry": "Dark mask<0.25 binary opened with disks1,2,4,8,16, recorded surviving area/whole image. Descriptive size response, not a semantic phase estimate.",
    "interiors": "Prior watershed sigma1/2; Euclidean distance>=2px from boundaries. Interior median intensity and max interior radius used to diagnose strip regions, not to relabel ground truth.",
    "perturbations": ["rot90", "flip_lr", "contrast_0.85_plus19", "gaussian_noise_sd3_uint8", "gaussian_blur_sigma0.6"],
    "perturbation_rng": "SHA256(ID+noise-v1) first8 hex digits, numpy default_rng",
    "visual": "All25 whole images and fixed center128x128 ROI; no hardness/OOF displayed. Visual reviewer may abstain; no expert segmentation ground truth.",
    "stability_screen": "Check subset only: for contrast and noise, >=8/10 images with pooled line spacing relative change<=0.15 AND >=8/10 with aligned line-mask tolerant F1>=0.8 for each perturbation. Missing spacing fails. These practical gates cannot establish semantic accuracy.",
    "scope": {"images": 25, "regression_fits": 0, "labels_loaded": False, "test_accessed": False},
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def select(rows, excluded):
    """Selection reads only these unlabeled columns, regardless of extra input columns."""
    rows = rows[COLS].sort_values("ID", kind="stable").reset_index(drop=True)
    if rows.ID.duplicated().any() or len(rows) < 25:
        raise ValueError("Need unique train IDs")
    ranks = rows[COLS[1:]].rank(method="first", pct=True).to_numpy()
    bins = np.minimum((np.arange(len(rows)) * 5 // len(rows)), 4)
    noise_order = np.argsort(rows.haar_hh_noise_proxy.to_numpy(), kind="stable")
    group = np.empty(len(rows), int); group[noise_order] = bins
    result = []
    for q in range(5):
        available = [j for j in range(len(rows)) if group[j] == q and rows.ID[j] not in excluded]
        chosen = []
        for k in range(5):
            if not chosen:
                j = min(available, key=lambda j: (float(np.sum((ranks[j, 1:] - .5) ** 2)), rows.ID[j]))
            else:
                j = min(available, key=lambda j: (-float(np.min(np.sum((ranks[chosen, 1:] - ranks[j, 1:]) ** 2, axis=1))), rows.ID[j]))
            chosen.append(j); available.remove(j)
        split_order = sorted(chosen, key=lambda j: hashlib.sha256((rows.ID[j] + "measurement-v1").encode()).hexdigest())
        for j in chosen:
            r = {c: (str(rows.loc[j, c]) if c == "ID" else float(rows.loc[j, c])) for c in COLS}
            r.update(noise_quintile=q+1, subset="development" if j in split_order[:3] else "check")
            result.append(r)
    return result


def prepare(folder):
    if folder.exists():
        raise ValueError("Use a new output folder; audit artifacts must not be overwritten")
    rows = pd.read_csv(AUDIT, usecols=COLS)  # hardness is deliberately not parsed
    prior = json.loads(PRIOR.read_text(encoding="utf-8"))
    selected = select(rows, {r["ID"] for r in prior["records"]})
    sources = [Path(__file__), Path(old.__file__), ROOT/"scripts/extract_local_structure.py",
               ROOT/"scripts/extract_frozen_resnet.py", AUDIT, PRIOR,
               ROOT/"reports/morphology/features/features.npz",
               ROOT/"reports/final_tabpfn_blend/submission/submission.csv"]
    for r in selected:
        if not r["ID"].startswith("TRAIN_") or not r["ID"][6:].isdigit():
            raise ValueError("Only train IDs accepted")
        r["image_sha256"] = sha(ROOT/"open/train"/(r["ID"]+".png"))
    plan = {"profile": PROFILE, "sources": {str(p.relative_to(ROOT)): sha(p) for p in sources},
            "versions": old.versions(), "records": selected}
    folder.mkdir(parents=True)
    save(folder/"protocol.json", plan)
    print(f"Frozen25: {folder/'protocol.json'}; development15/check10; no labels parsed")


def validate(folder):
    plan = json.loads((folder/"protocol.json").read_text(encoding="utf-8"))
    if plan["profile"] != PROFILE or plan["versions"] != old.versions():
        raise ValueError("Profile/runtime changed")
    for p, digest in plan["sources"].items():
        if sha(ROOT/p) != digest:
            raise ValueError(f"Source changed: {p}")
    for r in plan["records"]:
        if sha(ROOT/"open/train"/(r["ID"]+".png")) != r["image_sha256"]:
            raise ValueError("Image changed")
    return plan


def runs_centers(line, max_width=12):
    """Count one run once; reject broad and edge-touching runs."""
    z = np.diff(np.r_[False, np.asarray(line, bool), False].astype(int))
    starts, ends = np.flatnonzero(z == 1), np.flatnonzero(z == -1)
    return np.array([(a+b-1)/2 for a,b in zip(starts, ends)
                     if a > 0 and b < len(line) and b-a <= max_width])


def intercepts(mask):
    positions = [32,64,96,159,191,223]
    intervals = []; info = {}
    for axis in ("h", "v"):
        values=[]; crossings=0
        for k in positions:
            centers = runs_centers(mask[k,:] if axis=="h" else mask[:,k])
            crossings += len(centers)
            values.extend(np.diff(centers).tolist())
        info[axis+"_crossings"] = crossings
        info[axis+"_spacing"] = float(np.mean(values)) if values else None
        intervals.extend(values)
    info["spacing"] = float(np.mean(intervals)) if intervals else None
    info["interval_count"] = len(intervals)
    return info


def compact_mask(a):
    dark = ndi.gaussian_filter(a,1.,mode="reflect") < .25
    opened = ndi.binary_opening(dark, structure=morphology.disk(2))
    labels = old.filter_components(opened, 12)
    chosen = np.zeros_like(dark); count=0
    for r in measure.regionprops(labels, intensity_image=a):
        if r.area > 512 or 0 in r.bbox[:2] or r.bbox[2]==256 or r.bbox[3]==256:
            continue
        aspect = r.axis_major_length/max(r.axis_minor_length,1e-8)
        circ = 4*np.pi*r.area/max(r.perimeter_crofton**2,1e-8)
        region = labels == r.label
        ring = ndi.binary_dilation(region, structure=morphology.disk(3)) & ~region
        contrast = float(np.median(a[ring])-np.median(a[region])) if ring.any() else 0.
        if aspect<=2 and circ>=.45 and contrast>=.12:
            chosen |= region; count+=1
    return chosen, dark, count


def maps_and_metrics(gray):
    a = old.normalize(gray)
    constant = float(np.ptp(a)) < 1e-8
    compact,dark,count = compact_mask(a) if not constant else (np.zeros_like(a,bool),np.zeros_like(a,bool),0)
    smooth = ndi.gaussian_filter(a,1.,mode="reflect")
    response = morphology.black_tophat(smooth,footprint=morphology.disk(3))
    line = old.filter_components(response>.08,8)>0
    line &= ~ndi.binary_dilation(compact,structure=morphology.disk(1))
    measurements = {"compact_count": count,"compact_area": float(compact.mean()),
                    "dark_area": float(dark.mean()),"line_area":float(line.mean()),
                    **{"line_"+k:v for k,v in intercepts(line).items()}}
    for radius in (1,2,4,8,16):
        measurements[f"dark_open_r{radius}"] = float(ndi.binary_opening(dark,structure=morphology.disk(radius)).mean())
    labels_all=[]; boundaries=[]
    for sigma in (1.,2.):
        labels,_,_ = old.region_map(a,sigma)
        boundary = segmentation.find_boundaries(labels,mode="inner")
        distance=ndi.distance_transform_edt(~boundary)
        props=measure.regionprops(labels,intensity_image=a)
        thin=0; interior_values=[]; dark_area=0
        # Group indexing avoids scanning full image once for every region.
        for p in props:
            sl=p.slice; inside=(labels[sl]==p.label); dist=distance[sl]
            thin += int(float(dist[inside].max())<2.5)
            core = inside & (dist>=2)
            if core.any():
                value=float(np.median(a[sl][core])); interior_values.append(value)
                if value<.25:dark_area += p.area
        prefix=f"ws{int(sigma)}_"
        measurements.update({prefix+"regions":len(props),prefix+"thin_fraction":thin/len(props),
                             prefix+"dark_core_region_area":float(dark_area/a.size),
                             prefix+"core_median_intensity":float(np.median(interior_values)) if interior_values else None,
                             prefix+"spacing":intercepts(boundary)["spacing"]})
        labels_all.append(labels); boundaries.append(boundary)
    return measurements, {"normalized":a,"line":line,"compact":compact,"dark":dark,
                          "ws1":boundaries[0],"ws2":boundaries[1],"labels":labels_all}


def tolerant_f1(a,b,tolerance=1):
    if not a.any() and not b.any():return 1.
    if not a.any() or not b.any():return 0.
    structure=np.ones((2*tolerance+1,)*2,bool)
    precision=float(np.mean(ndi.binary_dilation(a,structure=structure)[b]))
    recall=float(np.mean(ndi.binary_dilation(b,structure=structure)[a]))
    return 2*precision*recall/(precision+recall) if precision+recall else 0.


def perturb(gray,name,ident):
    if name=="rot90":return np.rot90(gray).copy()
    if name=="flip_lr":return gray[:,::-1].copy()
    if name=="contrast_0.85_plus19":a=.85*gray.astype(float)+19
    elif name=="gaussian_blur_sigma0.6":a=ndi.gaussian_filter(gray.astype(float),.6,mode="reflect")
    elif name=="gaussian_noise_sd3_uint8":
        seed=int(hashlib.sha256((ident+"noise-v1").encode()).hexdigest()[:8],16)
        a=gray.astype(float)+np.random.default_rng(seed).normal(0,3,gray.shape)
    else:raise ValueError(name)
    return np.clip(np.rint(a),0,255).astype(np.uint8)


def align(mask,name):
    if name=="rot90":return np.rot90(mask,-1).copy()
    if name=="flip_lr":return mask[:,::-1].copy()
    return mask


def relative(a,b):
    return abs(a-b)/max(abs(a),1e-8) if a is not None and b is not None else None


def color(gray,mask,rgb):
    im=np.repeat(gray[...,None],3,axis=2)
    im[mask]=.35*im[mask]+.65*np.asarray(rgb)
    return Image.fromarray(im)


def render_page(folder,records,images,maps,page):
    font=ImageFont.truetype("C:/Windows/Fonts/arial.ttf",15)
    sheet=Image.new("RGB",(1332,1490),"#f5f6f8");d=ImageDraw.Draw(sheet)
    titles=["Original","Old watershed sigma1","New dark lines (cyan)","Compact red / other yellow","Center ROI: original | lines"]
    for c,t in enumerate(titles):d.text((8+c*264,10),t,fill="black",font=font)
    for row,r in enumerate(records):
        ident=r["ID"];g=images[ident];m=maps[ident];y=42+row*289
        objects=color(g,m["dark"] & ~m["compact"],(250,200,0))
        objects=np.asarray(objects).copy();objects[m["compact"]]=[255,40,60]
        roi=Image.new("RGB",(256,256));roi.paste(Image.fromarray(g[64:192,64:192]).convert("RGB"),(0,64))
        roi.paste(color(g,m["line"],(0,230,250)).crop((64,64,192,192)),(128,64))
        panels=[Image.fromarray(g).convert("RGB"),color(g,m["ws1"],(50,240,120)),color(g,m["line"],(0,230,250)),Image.fromarray(objects),roi]
        for c,im in enumerate(panels):sheet.paste(im,(8+c*264,y))
        d.text((8,y+258),f"{ident} | noise Q{r['noise_quintile']} | {r['subset']} | label-free",fill="black",font=font)
    sheet.save(folder/f"review_{page+1}.png")


def run(folder):
    plan=validate(folder)
    if (folder/"measurements.csv").exists():raise ValueError("Completed measurements exist; preserve them")
    records=[]; comparisons=[]; images={}; maps={}; arrays={}
    for i,r in enumerate(plan["records"]):
        ident=r["ID"];gray=old.read_image(ROOT/"open/train"/(ident+".png"))
        base,m=maps_and_metrics(gray);images[ident]=gray;maps[ident]=m
        records.append({**r,**base})
        for key in ("line","compact","dark","ws1","ws2","normalized"):
            arrays[ident+"_"+key]=m[key]
        for name in PROFILE["perturbations"]:
            candidate,other=maps_and_metrics(perturb(gray,name,ident))
            c={"ID":ident,"noise_quintile":r["noise_quintile"],"subset":r["subset"],"perturbation":name}
            for key in ("line","ws1","ws2","compact"):
                c[key+"_tolerant_f1"]=tolerant_f1(m[key],align(other[key],name))
            for key in ("line_spacing","compact_count","compact_area","ws1_regions","ws2_regions","dark_open_r4","dark_open_r8"):
                c[key+"_relative_change"]=relative(base[key],candidate[key])
            comparisons.append(c)
        print(f"Measured {i+1}/25 {ident} ({r['subset']}); no fitting",flush=True)
    validate(folder)
    df=pd.DataFrame(records);st=pd.DataFrame(comparisons)
    df.to_csv(folder/"measurements.csv",index=False);st.to_csv(folder/"stability.csv",index=False)
    np.savez_compressed(folder/"maps.npz",**arrays)
    for page in range(5):render_page(folder,plan["records"][page*5:page*5+5],images,maps,page)
    screens=[]
    for name in ("contrast_0.85_plus19","gaussian_noise_sd3_uint8"):
        a=st[(st.subset=="check") & (st.perturbation==name)]
        screens.append({"perturbation":name,"n":len(a),
                        "spacing_pass":int((a.line_spacing_relative_change<=.15).sum()),
                        "line_map_pass":int((a.line_tolerant_f1>=.8).sum()),
                        "median_line_spacing_change":float(a.line_spacing_relative_change.median()),
                        "median_line_map_f1":float(a.line_tolerant_f1.median())})
    summary={"protocol_sha256":sha(folder/"protocol.json"),"source_integrity":"matched before and after",
             "scope":PROFILE["scope"],"screens":screens,
             "passes_stability_screen":all(x["spacing_pass"]>=8 and x["line_map_pass"]>=8 for x in screens),
             "semantic_accuracy":"not established; requires visual review; no expert masks available",
             "run_artifacts":{p.name:sha(p) for p in folder.iterdir() if p.is_file() and p.name!="summary.json"}}
    save(folder/"summary.json",summary)
    print(json.dumps(summary["screens"],indent=2));print("Stability screen:",summary["passes_stability_screen"])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("command",choices=["prepare","run","validate"])
    p.add_argument("--output-dir",type=Path,default=OUT)
    a=p.parse_args()
    if a.command=="prepare":prepare(a.output_dir)
    elif a.command=="run":run(a.output_dir)
    else:validate(a.output_dir);print("All source/image hashes and profile match")


if __name__=="__main__":main()
