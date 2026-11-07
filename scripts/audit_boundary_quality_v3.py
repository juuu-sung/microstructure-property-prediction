"""Quality diagnostics around the IMMUTABLE v2 boundary detector.

CPU preview27 only: known development15 + two prior borderline cases + fresh10.
No hardness parsing, regression, test images, GPU, or full-dataset extraction.
Repeatability is never described as semantic segmentation accuracy.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from PIL import Image,ImageDraw,ImageFont
if __package__:
    from . import audit_boundary_measurements_v2 as detector
else:
    import audit_boundary_measurements_v2 as detector

ROOT=detector.ROOT
OUT=ROOT/'reports/boundary_quality_v3'
INTERNAL=('contrast_compress','contrast_expand','noise11','noise29','rot90','flip_lr')
EXTERNAL=('check_contrast','noise20261011','noise20261012')
PROFILE={
    'version':3,'scope':'quality wrapper only; v2 detector source and original maps unchanged',
    'axes':['observation_amount','repeatability','spatial_evidence','semantic_validity'],
    'observation':'Existing v2 limits:128 intervals,0.5 usable transects,8 occupied64px tiles, boundary fraction<=0.20; no space => unavailable. No SNR veto.',
    'internal_probes':list(INTERNAL),
    'internal_definition':'uint8 rounding/clipping; .85*x+19,1.05*x-6.375; Gaussian sd3 with fixed seeds11/29; rot90 and flip_lr inverse-aligned. No image IDs used in RNG. All6 required.',
    'repeatability':'minimum of all6 aligned1px-tolerant map F1 and base cross-scale F1; worst spacing relative change across6; report every probe and observation failure',
    'bands':'repeatable if support>=0.85 and spacing change<=0.10; unstable if support<0.70 or spacing change>0.20 or a probe lacks observations; between these => borderline. Buffer around old0.8/0.15 replaces brittle binary classification; empirical engineering limits, not calibrated probabilities.',
    'spatial_evidence':'Variance after Gaussian sigma2 divided by maximum of3 pixel-permutation variances (seeds101,307,911); outer8px omitted, raw/255. <=2 null-like,2..4 indeterminate,>=4 structured. Descriptive spatial organization evidence, not a p-value or grain evidence. No synthetic training.',
    'snr':'Retain v2 structure/noise ratio as diagnostic only; no direct status threshold at1.5',
    'semantics':'Always unverified_mixed_boundaries; physical_grain_size=null, pore_fraction=null regardless of repeatability. No ground-truth accuracy claim.',
    'outputs':'observed_spacing when available and not null-like; recommended_proxy_spacing only repeatable+structured. Other outputs null, never zero-imputed. Internal probe range is sensitivity range, not confidence interval.',
    'external_checks':list(EXTERNAL),
    'external_definition':'Re-run entire quality wrapper on .9*x+12.75 and independent Gaussian sd3 seeds20261011/20261012. These are not internal probe seeds. Record exact, adjacent and severe status changes; do not tune after check.',
    'samples':'Historical15 from v2 development plus previously known033/319; fresh10 chosen with old target-free selector after excluding all prior preview60. Historical results are development evidence, fresh10 are measurement diagnostics, not regression validation.',
    'decision':'No automatic regression promotion. Report fresh status changes, repeatability support, null-fixture outcomes and visual limits even if unfavorable; no all-rejected success claim.',
    'labels_used':False,'regression_fits':0,'test_accessed':False,
}


def perturb(gray,name):
    if name=='rot90':return np.rot90(gray).copy()
    if name=='flip_lr':return gray[:,::-1].copy()
    raw=gray.astype(float)
    if name=='contrast_compress':value=.85*raw+19
    elif name=='contrast_expand':value=1.05*raw-6.375
    elif name=='check_contrast':value=.9*raw+12.75
    elif name.startswith('noise'):value=raw+np.random.default_rng(int(name[5:])).normal(0,3,gray.shape)
    else:raise ValueError(name)
    return np.clip(np.rint(value),0,255).astype(np.uint8)


def observation(metrics):
    limits=detector.PROFILE['abstention'];why=[]
    for name,cutoff in [('intervals',limits['min_intervals']),('usable_transect_fraction',limits['min_usable_transect_fraction']),('occupied_tiles',limits['min_occupied_tiles'])]:
        if metrics[name]<cutoff:why.append('low_'+name)
    if metrics['boundary_area']>limits['max_boundary_area']:why.append('excessive_boundary_area')
    if metrics['raw_spacing'] is None:why.append('missing_spacing')
    return {'available':not why,'reasons':why}


def spatial_evidence(gray):
    raw=gray.astype(float)/255
    signal=float(ndi.gaussian_filter(raw,2.,mode='reflect')[8:-8,8:-8].var())
    nulls=[]
    for seed in (101,307,911):
        shuffled=np.random.default_rng(seed).permutation(raw.ravel()).reshape(raw.shape)
        nulls.append(float(ndi.gaussian_filter(shuffled,2.,mode='reflect')[8:-8,8:-8].var()))
    ratio=signal/max(max(nulls),1e-12)
    status='null_like' if ratio<=2 else ('indeterminate' if ratio<4 else 'structured')
    return {'status':status,'ratio':ratio,'smoothed_variance':signal,'permuted_variances':nulls}


def decide(base,probes,spatial):
    if len(probes)!=len(INTERNAL) or {p['name'] for p in probes}!=set(INTERNAL):raise ValueError('All fixed probes required exactly once')
    obs=observation(base)
    support=min([base['scale_f1']]+[p['map_f1'] for p in probes])
    delta=max(p['spacing_change'] for p in probes) if all(p['spacing_change'] is not None for p in probes) else None
    all_available=all(p['observation']['available'] for p in probes)
    if not all_available or delta is None or support<.70 or delta>.20:repeat='unstable'
    elif support>=.85 and delta<=.10:repeat='repeatable'
    else:repeat='borderline'
    if not obs['available']:status='unavailable'
    elif spatial['status']=='null_like':status='null_like'
    elif repeat=='unstable':status='unstable_proxy'
    elif repeat=='repeatable' and spatial['status']=='structured':status='repeatable_proxy'
    else:status='borderline_proxy'
    values=[base['raw_spacing']]+[p['raw_spacing'] for p in probes]
    finite=[float(v) for v in values if v is not None and np.isfinite(v)]
    return {'status':status,'observation':obs,'repeatability':repeat,'support_min':support,'spacing_change_max':delta,
            'all_probes_observable':all_available,'spatial_evidence':spatial,
            'structure_noise_ratio_diagnostic':base['structure_noise_ratio'],
            'raw_spacing':base['raw_spacing'],
            'observed_spacing':base['raw_spacing'] if obs['available'] and spatial['status']!='null_like' else None,
            'recommended_proxy_spacing':base['raw_spacing'] if status=='repeatable_proxy' else None,
            'sensitivity_min':min(finite) if finite else None,'sensitivity_max':max(finite) if finite else None,
            'semantic_validity':'unverified_mixed_boundaries','physical_grain_size':None,'pore_fraction':None}


def assess(gray):
    base,maps=detector.detect(gray);probes=[]
    for name in INTERNAL:
        m,p=detector.detect(perturb(gray,name))
        aligned=detector.v1.align(p['boundary'],name)
        probes.append({'name':name,'map_f1':detector.v1.tolerant_f1(maps['boundary'],aligned),
                       'spacing_change':detector.v1.relative(base['raw_spacing'],m['raw_spacing']),
                       'raw_spacing':m['raw_spacing'],'observation':observation(m),
                       'v2_status':m['status'],'structure_noise_ratio':m['structure_noise_ratio']})
    q=decide(base,probes,spatial_evidence(gray))
    native_base={key:(value.item() if isinstance(value,np.generic) else value) for key,value in base.items()}
    q.update(probes=probes,v2_status=base['status'],v2_metrics=native_base)
    return q,maps


def transition(a,b):
    if a==b:return 'same'
    if {a,b}<={'repeatable_proxy','borderline_proxy','unstable_proxy'}:
        ranks={'repeatable_proxy':2,'borderline_proxy':1,'unstable_proxy':0}
        return 'adjacent' if abs(ranks[a]-ranks[b])==1 else 'severe'
    return 'severe'


def prepare():
    if OUT.exists():raise ValueError('Use untouched output folder')
    prior=detector.validate()
    a=json.loads((detector.v1.OUT/'protocol.json').read_text(encoding='utf-8'))
    m=json.loads(detector.v1.PRIOR.read_text(encoding='utf-8'))
    excluded={r['ID'] for plan in (prior,a,m) for r in plan['records']}
    rows=pd.read_csv(detector.v1.AUDIT,usecols=detector.v1.COLS)
    fresh=[dict(r,subset='fresh') for r in detector.v1.select(rows,excluded) if r['subset']=='check']
    history=[dict(r,subset='historical') for r in prior['records'] if r['subset']=='development' or r['ID'] in ('TRAIN_000033','TRAIN_000319')]
    records=history+fresh
    if len(history)!=17 or len(fresh)!=10 or {r['ID'] for r in fresh}&excluded:raise ValueError('Selection mismatch')
    for r in records:r['image_sha256']=detector.v1.sha(ROOT/'open/train'/(r['ID']+'.png'))
    paths=[Path(__file__),Path(detector.__file__),Path(detector.v1.__file__),
           detector.OUT/'completion_manifest.json',detector.OUT/'protocol.json',detector.v1.AUDIT,
           ROOT/'reports/final_tabpfn_blend/submission/submission.csv']
    OUT.mkdir()
    detector.v1.save(OUT/'protocol.json',{'profile':PROFILE,'records':records,'excluded_count':len(excluded),
                                        'versions':detector.v1.old.versions(),'sources':{str(p.relative_to(ROOT)):detector.v1.sha(p) for p in paths}})
    print(f'Frozen historical17/fresh10; excluded earlier{len(excluded)}; no targets parsed')


def validate():
    detector.validate()
    plan=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))
    if plan['profile']!=PROFILE or plan['versions']!=detector.v1.old.versions():raise ValueError('Profile/runtime changed')
    for path,digest in plan['sources'].items():
        if detector.v1.sha(ROOT/path)!=digest:raise ValueError('Source changed '+path)
    for r in plan['records']:
        if detector.v1.sha(ROOT/'open/train'/(r['ID']+'.png'))!=r['image_sha256']:raise ValueError('Image changed')
    return plan


def render(folder,records,gray,masks,qualities):
    colors={'repeatable_proxy':(40,220,140),'borderline_proxy':(255,165,25),'unstable_proxy':(250,60,75),'unavailable':(180,100,210),'null_like':(180,100,210)}
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',15)
    for page in range((len(records)+4)//5):
        sheet=Image.new('RGB',(1048,1515),'#f5f6f8');d=ImageDraw.Draw(sheet)
        d.text((8,12),'Original | Fixed v2 boundary: color = repeatability, NOT grain accuracy | Separate quality axes',font=font,fill='black')
        for row,r in enumerate(records[page*5:page*5+5]):
            ident=r['ID'];q=qualities[ident];y=42+row*294
            sheet.paste(Image.fromarray(gray[ident]).convert('RGB'),(8,y))
            sheet.paste(detector.v1.color(gray[ident],masks[ident],colors[q['status']]),(272,y))
            info=[ident+f" | Q{r['noise_quintile']}",q['status'],f"Map support min: {q['support_min']:.3f}",
                  f"Spacing worst change: {q['spacing_change_max']:.1%}" if q['spacing_change_max'] is not None else 'Spacing: missing',
                  f"Spatial / shuffled variance: {q['spatial_evidence']['ratio']:.2f}",
                  f"Old SNR proxy: {q['structure_noise_ratio_diagnostic']:.3f} (no veto)",
                  'Semantic validity: UNVERIFIED', 'Physical grain size: not reported']
            for j,t in enumerate(info):d.text((544,y+12+27*j),t,font=font,fill='black')
            d.text((8,y+259),f"{ident} | observation={q['observation']['available']} | v2={q['v2_status']} -> v3={q['status']}",font=font,fill='black')
        sheet.save(folder/f'review_{page+1}.png')


def run(subset):
    plan=validate();folder=OUT/subset
    if folder.exists():raise ValueError('Preserve existing output')
    if subset=='fresh':
        review=json.loads((OUT/'historical/review_decision.json').read_text(encoding='utf-8'))
        if review.get('decision')!='proceed_fixed_fresh_check':raise ValueError('Historical review not approved')
    folder.mkdir();records=[r for r in plan['records'] if r['subset']==subset]
    result=[];checks=[];base_arrays={};gray={};masks={};qs={}
    for r in records:
        ident=r['ID'];g=detector.v1.old.read_image(ROOT/'open/train'/(ident+'.png'))
        q,m=assess(g)
        if subset=='historical':
            prior_subset='check' if ident in ('TRAIN_000033','TRAIN_000319') else 'development'
            with np.load(detector.OUT/prior_subset/'maps.npz',allow_pickle=False) as old:
                np.testing.assert_array_equal(m['boundary'],old[ident+'_boundary'])
        result.append({'ID':ident,'noise_quintile':r['noise_quintile'],'subset':subset,**q})
        gray[ident]=g;masks[ident]=m['boundary'];qs[ident]=q;base_arrays[ident]=m['boundary']
        for name in EXTERNAL:
            eq,_=assess(perturb(g,name))
            checks.append({'ID':ident,'external':name,'base_status':q['status'],'external_status':eq['status'],
                           'transition':transition(q['status'],eq['status']),
                           'support_base':q['support_min'],'support_external':eq['support_min'],
                           'spacing_change_base':q['spacing_change_max'],'spacing_change_external':eq['spacing_change_max'],
                           'spatial_ratio_external':eq['spatial_evidence']['ratio'],
                           'v2_base':q['v2_status'],'v2_external':eq['v2_status'],
                           'semantic_validity_external':eq['semantic_validity']})
        print(f"{subset} {ident}: {q['v2_status']} -> {q['status']}; support={q['support_min']:.3f}",flush=True)
    validate()
    detector.v1.save(folder/'quality_details.json',result)
    flats=[]
    for r in result:
        flats.append({k:v for k,v in r.items() if not isinstance(v,(dict,list))}|{
            'observation_available':r['observation']['available'],'observation_reasons':';'.join(r['observation']['reasons']),
            'spatial_ratio':r['spatial_evidence']['ratio'],'spatial_status':r['spatial_evidence']['status'],
            'intervals':r['v2_metrics']['intervals'],'coverage':r['v2_metrics']['usable_transect_fraction']})
    pd.DataFrame(flats).to_csv(folder/'measurements.csv',index=False)
    pd.DataFrame(checks).to_csv(folder/'external_checks.csv',index=False)
    np.savez_compressed(folder/'maps.npz',**base_arrays)
    render(folder,records,gray,masks,qs)
    transitions={k:sum(r['transition']==k for r in checks) for k in ('same','adjacent','severe')}
    counts=pd.Series([r['status'] for r in result]).value_counts().to_dict()
    summary={'subset':subset,'n':len(result),'status_counts':counts,'external_comparisons':len(checks),'transitions':transitions,
             'v2_flips':sum(r['v2_base']!=r['v2_external'] for r in checks),'physical_accuracy':'unverified for every image',
             'historical_detector_anchor':'exact masks matched' if subset=='historical' else 'same frozen detector source',
             'automatic_regression_adoption':False,'protocol_sha256':detector.v1.sha(OUT/'protocol.json'),
             'artifacts':{p.name:detector.v1.sha(p) for p in folder.iterdir() if p.is_file()}}
    detector.v1.save(folder/'summary.json',summary);print(json.dumps(summary,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['prepare','historical','fresh','validate'])
    args=p.parse_args()
    if args.command=='prepare':prepare()
    elif args.command=='validate':validate();print('Frozen detector, inputs, previous report and profile match')
    else:run(args.command)


if __name__=='__main__':main()
