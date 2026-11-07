"""Mixed boundary measurement and explicit abstention, CPU preview only.

Development15 reuse prior images; check10 exclude both prior preview sets.
No hardness parsing, model fitting, test images, GPU or full500 extraction.
"""
from __future__ import annotations
import argparse
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage import feature, morphology
from PIL import Image, ImageDraw, ImageFont
if __package__:
    from . import audit_structure_measurements as v1
else:
    import audit_structure_measurements as v1

ROOT=v1.ROOT
OUT=ROOT/'reports/structure_measurement_v2'
SIGMAS=(1.,1.6)
PROFILE={
    'version':2,'units':'Pixels; mixed visible-boundary proxy, not physical grain diameter',
    'input':'uint8 256x256 grayscale /255, no NLM, no percentile normalization for boundary detection',
    'noise':'2x2 Haar HH MAD /0.6744897501960817; a proxy not calibrated noise truth',
    'edge':'Canny sigma1/1.6, reflect; high=max0.04,4*noise*Gaussian-Sobel kernel L2; low=0.5high; retain fine edges within2px Chebyshev of coarse edges',
    'valley':'Gaussian sigma1/1.6 black_tophat disk3; threshold=max0.006,3*noise*Gaussian kernel L2; fine support within2px of coarse; min8 connected pixels, skeletonized',
    'combine':'Step edges within3px Chebyshev of valley skeleton removed to avoid two flanks of same dark line; skeletonize union, min8px components; remove outer2px border. No forced closure or watershed.',
    'measurement':'All256 H and256 V transects. Reject edge-touching runs or width>12. Merge runs separated by<=2 empty pixels; omit frame-censored intervals. Directional and pooled mean/median spacing with interval counts.',
    'abstention':{'min_intervals':128,'min_usable_transect_fraction':.5,'min_occupied_tiles':8,'tile_size':64,'min_tile_pixels':8,'min_scale_f1':.65,'min_structure_noise_ratio':1.5,'max_boundary_area':.20},
    'output':'Raw spacing kept for diagnostics. Reported spacing null if any abstention rule fails; reasons explicit. Observed proxy does NOT imply semantic correctness or calibrated confidence. No zero imputation.',
    'auxiliary':'Unchanged v1 NLM-normalized dark opening area radii1,2,4,8,16 and compact_dark mask. Compact candidates remain semantically ambiguous; no pore labels.',
    'perturbations':v1.PROFILE['perturbations'],
    'development':'Prior measurement audit development15 only; sparse raw-image visual references fixed before v2 outputs; may diagnose but not assert independent accuracy.',
    'check_selection':'Exclude all prior morphology preview25 and measurement audit25. Apply previous target-free25 selector, retain its two check representatives per noise quintile => new10. Settings frozen before viewing check images.',
    'screen':'For each mild contrast/noise perturbation, at least8/10 have same abstention status; among base observed proxies at least80% remain observed with spacing change<=15% and map tolerantF1>=0.8. At least4 base observed proxies required; otherwise insufficient evidence. Visual review remains mandatory, not automatic regression adoption.',
    'labels_used':False,'test_accessed':False,'regression_fits':0,
}


def noise_sigma(raw):
    hh=(raw[::2,::2]-raw[::2,1::2]-raw[1::2,::2]+raw[1::2,1::2])/2
    return float(np.median(abs(hh-np.median(hh)))/.6744897501960817)


@lru_cache(None)
def noise_gains(sigma):
    impulse=np.zeros((65,65));impulse[32,32]=1
    g=ndi.gaussian_filter(impulse,sigma,mode='reflect')
    return float(np.linalg.norm(g)),float(np.linalg.norm(ndi.sobel(g,axis=0)))


def clean(mask,minimum=8):
    return v1.old.filter_components(mask,minimum)>0


def centers(row):
    z=np.diff(np.r_[False,row,False].astype(int))
    starts=np.flatnonzero(z==1);ends=np.flatnonzero(z==-1)
    merged=[]
    for start,end in zip(starts,ends):
        if merged and start-merged[-1][1]<=2:merged[-1][1]=end
        else:merged.append([start,end])
    return np.array([(s+e-1)/2 for s,e in merged if s>0 and e<len(row) and e-s<=12])


def all_intercepts(mask):
    values=[];crossings=0;usable=0;result={}
    for name,arr in [('h',mask),('v',mask.T)]:
        intervals=[];used=0
        for row in arr:
            c=centers(row);crossings+=len(c)
            if len(c)>=2:used+=1
            intervals.extend(np.diff(c).tolist())
        result[name+'_spacing']=float(np.mean(intervals)) if intervals else None
        result[name+'_intervals']=len(intervals)
        result[name+'_usable_transects']=used
        usable+=used;values.extend(intervals)
    result.update(raw_spacing=float(np.mean(values)) if values else None,
                  median_spacing=float(np.median(values)) if values else None,
                  intervals=len(values),crossings=crossings,usable_transect_fraction=usable/512)
    return result


def quality(metrics):
    p=PROFILE['abstention'];reasons=[]
    for key,limit in [('intervals',p['min_intervals']),('usable_transect_fraction',p['min_usable_transect_fraction']),
                      ('occupied_tiles',p['min_occupied_tiles']),('scale_f1',p['min_scale_f1']),
                      ('structure_noise_ratio',p['min_structure_noise_ratio'])]:
        if metrics[key]<limit:reasons.append('low_'+key)
    if metrics['boundary_area']>p['max_boundary_area']:reasons.append('excessive_boundary_area')
    if metrics['raw_spacing'] is None:reasons.append('no_spacing')
    return {'status':'abstain' if reasons else 'observed_proxy','reasons':reasons,
            'reported_spacing':None if reasons else metrics['raw_spacing']}


def detect(gray):
    if np.asarray(gray).shape!=(256,256) or gray.dtype!=np.uint8:raise ValueError('Expected256x256 uint8')
    raw=gray.astype(float)/255;noise=noise_sigma(raw)
    edges=[];valleys=[]
    for sigma in SIGMAS:
        gg,eg=noise_gains(sigma)
        high=max(.04,4*noise*eg)
        edges.append(feature.canny(raw,sigma=sigma,low_threshold=.5*high,high_threshold=high,mode='reflect'))
        sm=ndi.gaussian_filter(raw,sigma,mode='reflect')
        response=morphology.black_tophat(sm,footprint=morphology.disk(3))
        valleys.append(clean(response>max(.006,3*noise*gg)))
    neighborhood=np.ones((5,5),bool)
    valley=valleys[0] & ndi.binary_dilation(valleys[1],structure=neighborhood)
    valley=morphology.skeletonize(clean(valley))
    step=edges[0] & ndi.binary_dilation(edges[1],structure=neighborhood)
    step &= ~ndi.binary_dilation(valley,structure=np.ones((7,7),bool))
    combined=clean(morphology.skeletonize(valley|step))
    combined[:2]=False;combined[-2:]=False;combined[:,:2]=False;combined[:,-2:]=False
    valley &= combined;step &= combined
    fine=edges[0]|morphology.skeletonize(valleys[0])
    coarse=edges[1]|morphology.skeletonize(valleys[1])
    structural=np.sqrt(max(float(raw.var())-noise**2,0))
    tiles=[combined[y:y+64,x:x+64].sum() for y in range(0,256,64) for x in range(0,256,64)]
    m={**all_intercepts(combined),'noise_proxy':noise,'structure_noise_ratio':float(structural/max(noise,1e-8)),
       'occupied_tiles':sum(t>=8 for t in tiles),'scale_f1':v1.tolerant_f1(fine,coarse,2),
       'boundary_area':float(combined.mean()),'valley_area':float(valley.mean()),'step_area':float(step.mean())}
    m.update(quality(m))
    return m,{'boundary':combined,'valley':valley,'step':step}


def baseline(gray):
    norm=v1.old.normalize(gray)
    if np.ptp(norm)<1e-8:compact=dark=np.zeros_like(norm,bool)
    else:compact,dark,_=v1.compact_mask(norm)
    smooth=ndi.gaussian_filter(norm,1.,mode='reflect')
    line=clean(morphology.black_tophat(smooth,footprint=morphology.disk(3))>.08)
    line &= ~ndi.binary_dilation(compact,structure=morphology.disk(1))
    metrics={'v1_'+k:v for k,v in all_intercepts(line).items()}
    for radius in (1,2,4,8,16):metrics[f'dark_open_r{radius}']=float(ndi.binary_opening(dark,structure=morphology.disk(radius)).mean())
    return metrics,{'v1_line':line,'dark':dark,'compact':compact}


def prepare():
    if (OUT/'protocol.json').exists():raise ValueError('Protocol exists; preserve audit')
    prior=v1.validate(v1.OUT)
    earlier=json.loads(v1.PRIOR.read_text(encoding='utf-8'))
    excluded={r['ID'] for r in prior['records']}|{r['ID'] for r in earlier['records']}
    rows=pd.read_csv(v1.AUDIT,usecols=v1.COLS)
    check=[r for r in v1.select(rows,excluded) if r['subset']=='check']
    dev=[dict(r,subset='development') for r in prior['records'] if r['subset']=='development']
    records=dev+check
    assert len(check)==10 and len(records)==25 and len({r['ID'] for r in records})==25
    for r in records:r['image_sha256']=v1.sha(ROOT/'open/train'/(r['ID']+'.png'))
    paths=[Path(__file__),Path(v1.__file__),Path(v1.old.__file__),v1.OUT/'protocol.json',
           OUT/'visual_references.json',OUT/'development_raw_rois.png',v1.AUDIT,v1.PRIOR,
           ROOT/'reports/final_tabpfn_blend/submission/submission.csv']
    OUT.mkdir(exist_ok=True)
    v1.save(OUT/'protocol.json',{'profile':PROFILE,'versions':v1.old.versions(),'records':records,
                              'sources':{str(p.relative_to(ROOT)):v1.sha(p) for p in paths}})
    print('Frozen development15/check10; fresh check IDs exclude earlier50; no labels parsed')


def validate():
    plan=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))
    if plan['profile']!=PROFILE or plan['versions']!=v1.old.versions():raise ValueError('Profile/runtime changed')
    v1.validate(v1.OUT)
    for p,h in plan['sources'].items():
        if v1.sha(ROOT/p)!=h:raise ValueError('Changed source: '+p)
    for r in plan['records']:
        if v1.sha(ROOT/'open/train'/(r['ID']+'.png'))!=r['image_sha256']:raise ValueError('Changed image')
    return plan


def render(folder,records,raw,maps,metrics):
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',15)
    for page in range((len(records)+4)//5):
        sheet=Image.new('RGB',(1068,1535),'#f5f6f8');d=ImageDraw.Draw(sheet)
        for c,t in enumerate(['Original','v1 dark lines (cyan)','v2 valley cyan / step orange','Quality: green observed / red abstain']):d.text((8+264*c,12),t,fill='black',font=font)
        for row,r in enumerate(records[page*5:page*5+5]):
            ident=r['ID'];g=raw[ident];m=maps[ident];q=metrics[ident];y=44+row*294
            combined=np.repeat(g[...,None],3,axis=2);combined[m['valley']]=[0,230,240];combined[m['step']]=[255,145,25]
            rgb=(60,245,100) if q['status']=='observed_proxy' else (255,65,85)
            panels=[Image.fromarray(g).convert('RGB'),v1.color(g,m['v1_line'],(0,230,240)),Image.fromarray(combined),v1.color(g,m['boundary'],rgb)]
            for c,im in enumerate(panels):sheet.paste(im,(8+c*264,y))
            text=f"{ident} | Q{r['noise_quintile']} | {q['status']} | intervals={q['intervals']} | coverage={q['usable_transect_fraction']:.2f} | scaleF1={q['scale_f1']:.2f}"
            d.text((8,y+258),text,fill='black',font=font)
            d.text((8,y+276),'Reasons: '+(', '.join(q['reasons']) or 'Numeric observability only; semantics unverified'),fill='black',font=font)
        sheet.save(folder/f'review_{page+1}.png')


def run(subset):
    plan=validate();folder=OUT/subset
    if folder.exists():raise ValueError('Completed/partial output exists; preserve it')
    if subset=='check' and not (OUT/'development/review_decision.json').exists():raise ValueError('Review development first')
    folder.mkdir();records=[r for r in plan['records'] if r['subset']==subset]
    measurements=[];stability=[];arrays={};raws={};allmaps={};allmetrics={}
    for r in records:
        ident=r['ID'];gray=v1.old.read_image(ROOT/'open/train'/(ident+'.png'))
        m,maps=detect(gray);old,oldmaps=baseline(gray);m.update(old);maps.update(oldmaps)
        raws[ident]=gray;allmaps[ident]=maps;allmetrics[ident]=m
        measurements.append({**r,**m,'reasons':';'.join(m['reasons'])})
        for key,value in maps.items():arrays[ident+'_'+key]=value
        for variant in PROFILE['perturbations']:
            g=v1.perturb(gray,variant,ident);other,om=detect(g);old_other,old_om=baseline(g)
            aligned=v1.align(om['boundary'],variant)
            st={'ID':ident,'noise_quintile':r['noise_quintile'],'variant':variant,'base_status':m['status'],
                'variant_status':other['status'],'status_same':m['status']==other['status'],
                'v2_map_f1':v1.tolerant_f1(maps['boundary'],aligned),
                'v1_map_f1':v1.tolerant_f1(maps['v1_line'],v1.align(old_om['v1_line'],variant)),
                'v2_spacing_change':v1.relative(m['raw_spacing'],other['raw_spacing']),
                'v1_spacing_change':v1.relative(m['v1_raw_spacing'],old_other['v1_raw_spacing']),
                'dark_open_r4_change':v1.relative(m['dark_open_r4'],old_other['dark_open_r4'])}
            stability.append(st)
        print(f"{subset} {ident}: {m['status']}, raw={m['raw_spacing']}, reported={m['reported_spacing']}",flush=True)
    validate()
    pd.DataFrame(measurements).to_csv(folder/'measurements.csv',index=False)
    pd.DataFrame(stability).to_csv(folder/'stability.csv',index=False)
    np.savez_compressed(folder/'maps.npz',**arrays)
    render(folder,records,raws,allmaps,allmetrics)
    screens=[];nobs=sum(m['status']=='observed_proxy' for m in allmetrics.values())
    for variant in ('contrast_0.85_plus19','gaussian_noise_sd3_uint8'):
        rows=[r for r in stability if r['variant']==variant]
        obs=[r for r in rows if r['base_status']=='observed_proxy']
        passed=sum(r['variant_status']=='observed_proxy' and r['v2_spacing_change'] is not None and r['v2_spacing_change']<=.15 and r['v2_map_f1']>=.8 for r in obs)
        screens.append({'variant':variant,'status_same':sum(r['status_same'] for r in rows),'observed_n':nobs,'observed_stable_n':passed})
    summary={'subset':subset,'n':len(records),'observed_n':nobs,'abstain_n':len(records)-nobs,'screens':screens,
             'check_screen_pass':subset=='check' and nobs>=4 and all(s['status_same']>=8 and s['observed_stable_n']>=.8*nobs for s in screens),
             'semantic_validity':'unverified; visual review required','protocol_sha256':v1.sha(OUT/'protocol.json'),
             'artifacts':{p.name:v1.sha(p) for p in folder.iterdir() if p.is_file()}}
    v1.save(folder/'summary.json',summary);print(json.dumps(summary,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['prepare','development','check','validate'])
    args=p.parse_args()
    if args.command=='prepare':prepare()
    elif args.command=='validate':validate();print('Source/image/profile hashes matched')
    else:run(args.command)


if __name__=='__main__':main()
