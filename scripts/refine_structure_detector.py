"""Classical proposal refinement; no fitting, labels, test images or human-mask writes.

Development cases Q01/Q09/Q24. Freeze one profile before assessing other nine
previously seen pilot images. These are regression checks, not independent accuracy.
"""
from __future__ import annotations
import argparse
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi
from skimage import feature, measure, morphology, restoration, segmentation

if __package__:
    from . import auto_structure_review as previous
else:
    import auto_structure_review as previous

pilot=previous.pilot; quality=previous.quality; old=quality.detector
ROOT=pilot.ROOT; OUT=ROOT/'reports/structure_detector_refinement'
TEMPLATE=ROOT/'scripts/refine_structure_detector.html'
DEVELOPMENT=('C01','C09','C12')
PROFILE={
 'version':1,'development_cases':list(DEVELOPMENT),'development_rois':['Q01','Q09','Q24'],
 'checks':'Other9 already-seen pilot images; not independent holdout. No tuning after their results.',
 'denoise':'Raw/255 NLM h=1.1*Haar sigma, patch_size5, patch_distance6, sigma correction; no percentile rescaling for new edges.',
 'step_proposal':'Canny sigma1/1.6 high=max0.012,2.5*post-NLM-Haar-sigma*Gaussian-Sobel kernel L2; low=0.5high. Fine edges supported within2px of coarse. Discard within3px of old valley; no closure.',
 'raw_support':'At each proposed edge use denoised gradient normal; average raw samples at normal offsets2,3,4 and tangent offsets-3..3 on each side (bilinear). Absolute difference>=max0.010,2.5*raw-noise*sqrt(2/21). Engineering evidence threshold, NOT calibrated probability. Keep>=8 connected pixels; no additions within1px of old boundary.',
 'null_screen':'Add no edges when unchanged v3 spatial organization ratio<4. Remove outer5px of new proposals only. Old boundary preserved in mixed_boundary.',
 'objects':'Reassess old compact components, all also retained in old_compact. Shape measured on intersecting pre-opening dark component (avoid opening-induced roundness): area<=640, aspect<=1.35, Crofton circularity>=0.85, solidity>=0.94, no frame contact. Raw component median versus outside7px annulus excluding first2px: depth>=max0.18,2.5*noise). Passing = round_deep_candidate; all others=ambiguous_compact, NOT polygon or pore truth.',
 'contours':'Separate round-deep-object outlines (2px band) from mixed boundary into structural_proxy. This is optional candidate separation, not a new grain label. Mixed union retained for comparison.',
 'evaluation':'Same six fixed contrast/noise/rotation/reflection probes for before and after; report worst tolerantF1 and spacing sensitivity, counts added/separated/ambiguous. No accuracy gain claim without independent masks.',
 'labels_used':False,'model_fits':0,'test_accessed':False,'training_approved':False,
}


def denoise(gray):
    raw=gray.astype(float)/255.; sigma=old.noise_sigma(raw)
    if sigma<=1e-8:return raw,sigma
    clean=restoration.denoise_nl_means(raw,h=1.1*sigma,sigma=sigma,fast_mode=True,patch_size=5,patch_distance=6,preserve_range=True,channel_axis=None)
    return clean,sigma


def raw_step_evidence(raw,clean,candidates,noise):
    out=np.zeros(raw.shape,float); yy,xx=np.nonzero(candidates)
    if not len(yy):return out
    smooth=ndi.gaussian_filter(clean,1.,mode='reflect')
    gy=ndi.sobel(smooth,axis=0);gx=ndi.sobel(smooth,axis=1)
    norm=np.maximum(np.hypot(gy[yy,xx],gx[yy,xx]),1e-12)
    ny=gy[yy,xx]/norm;nx=gx[yy,xx]/norm
    positive=np.zeros(len(yy));negative=np.zeros(len(yy))
    for n in (2,3,4):
        for t in range(-3,4):
            for sign,values in ((1,positive),(-1,negative)):
                coords=np.array([yy+sign*n*ny-t*nx,xx+sign*n*nx+t*ny])
                values+=ndi.map_coordinates(raw,coords,order=1,mode='reflect')
    out[yy,xx]=abs(positive-negative)/21
    return out


def classify_objects(raw,dark,compact,noise):
    labels,n=ndi.label(compact,structure=np.ones((3,3),bool))
    prelabels,_=ndi.label(dark,structure=np.ones((3,3),bool))
    preprops={r.label:r for r in measure.regionprops(prelabels)}
    certain=np.zeros_like(compact);ambiguous=np.zeros_like(compact);records=[]
    for i in range(1,n+1):
        region=labels==i; overlap=prelabels[region];overlap=overlap[overlap>0]
        ident=int(np.bincount(overlap).argmax()) if overlap.size else 0
        reasons=[]
        if ident:
            r=preprops[ident];shape=prelabels==ident
            aspect=r.axis_major_length/max(r.axis_minor_length,1e-8)
            circularity=4*np.pi*r.area/max(r.perimeter_crofton**2,1e-8)
            solidity=r.solidity
            if r.area>640:reasons.append('large_or_connected_dark_region')
            if aspect>1.35:reasons.append('elongated')
            if circularity<.85:reasons.append('nonround_before_opening')
            if solidity<.94:reasons.append('irregular_before_opening')
            if 0 in r.bbox[:2] or r.bbox[2]==256 or r.bbox[3]==256:reasons.append('frame_contact')
        else:
            shape=region;aspect=circularity=solidity=None;reasons.append('missing_unopened_component')
        ring=ndi.binary_dilation(shape,structure=morphology.disk(7)) & ~ndi.binary_dilation(shape,structure=morphology.disk(2))
        depth=float(np.median(raw[ring])-np.median(raw[region])) if ring.any() else 0.
        if depth<max(.18,2.5*noise):reasons.append('insufficient_raw_darkness')
        (ambiguous if reasons else certain)[region]=True
        records.append({'component':i,'pixels':int(region.sum()),'raw_depth':depth,'aspect_before_opening':aspect,
            'circularity_before_opening':circularity,'solidity_before_opening':solidity,'reasons':reasons,
            'category':'ambiguous_compact' if reasons else 'round_deep_candidate','semantic_truth':None})
    return certain,ambiguous,records


def detect(gray):
    if gray.shape!=(256,256) or gray.dtype!=np.uint8:raise ValueError('Expected native256 uint8')
    base,b=old.detect(gray);_,aux=old.baseline(gray)
    raw=gray.astype(float)/255.; clean,noise=denoise(gray)
    post_noise=old.noise_sigma(clean)
    edges=[]
    for sigma in (1.,1.6):
        high=max(.012,2.5*post_noise*old.noise_gains(sigma)[1])
        edges.append(feature.canny(clean,sigma=sigma,low_threshold=.5*high,high_threshold=high,mode='reflect'))
    proposal=edges[0] & ndi.binary_dilation(edges[1],structure=np.ones((5,5),bool))
    proposal &= ~ndi.binary_dilation(b['valley'],structure=np.ones((7,7),bool))
    proposal[:5]=False;proposal[-5:]=False;proposal[:,:5]=False;proposal[:,-5:]=False
    evidence=raw_step_evidence(raw,clean,proposal,noise)
    supported=proposal & (evidence>=max(.010,2.5*noise*np.sqrt(2/21)))
    supported=old.clean(supported)
    spatial=quality.spatial_evidence(gray)
    if spatial['ratio']<4:supported[:]=False
    additions=supported & ~ndi.binary_dilation(b['boundary'],structure=np.ones((3,3),bool))
    mixed=b['boundary']|additions
    round_deep,ambiguous,objects=classify_objects(raw,aux['dark'],aux['compact'],noise)
    contour=ndi.binary_dilation(segmentation.find_boundaries(round_deep,mode='inner'),structure=morphology.disk(2))
    structural=mixed & ~contour
    maps={'before':b['boundary'],'mixed_boundary':mixed,'structural_proxy':structural,'added':additions,
        'weak_rejected':proposal & ~supported,'object_contour':mixed & contour,'dark':aux['dark'],
        'old_compact':aux['compact'],'round_deep':round_deep,'ambiguous_compact':ambiguous,'raw_step_contrast':evidence}
    metrics={'added_pixels':int(additions.sum()),'old_boundary_pixels':int(b['boundary'].sum()),
        'mixed_boundary_pixels':int(mixed.sum()),'separated_contour_pixels':int((mixed & contour).sum()),
        'round_deep_count':sum(not r['reasons'] for r in objects),'ambiguous_count':sum(bool(r['reasons']) for r in objects),
        'old_compact_count':len(objects),'raw_noise_proxy':noise,'post_nlm_noise_proxy':post_noise,
        'spatial_ratio':spatial['ratio'],'objects':objects,'semantic_truth':None,'training_approved':False}
    return metrics,maps


def diagnostics(gray):
    metrics,maps=detect(gray); before=[];after=[];added=[];objects=[]
    old_spacing=old.all_intercepts(maps['before'])['raw_spacing'];new_spacing=old.all_intercepts(maps['mixed_boundary'])['raw_spacing']
    for name in quality.INTERNAL:
        _,other=detect(quality.perturb(gray,name))
        other={k:old.v1.align(v,name) for k,v in other.items()}
        before.append(other['before']);after.append(other['mixed_boundary']);added.append(other['added']);objects.append(other['round_deep'])
    for prefix,key,probes in [('before','before',before),('after','mixed_boundary',after),('added','added',added),('round_deep','round_deep',objects)]:
        values=[previous.overlap(maps[key],v,'f1' if prefix!='round_deep' else 'iou') for v in probes]
        metrics[prefix+'_min_agreement']=min((v for v in values if v is not None),default=None)
    metrics['before_spacing']=old_spacing;metrics['after_spacing']=new_spacing
    metrics['before_spacing_change_max']=max((old.v1.relative(old_spacing,old.all_intercepts(v)['raw_spacing']) or 0 for v in before),default=None) if old_spacing is not None and all(old.all_intercepts(v)['raw_spacing'] is not None for v in before) else None
    metrics['after_spacing_change_max']=max((old.v1.relative(new_spacing,old.all_intercepts(v)['raw_spacing']) or 0 for v in after),default=None) if new_spacing is not None and all(old.all_intercepts(v)['raw_spacing'] is not None for v in after) else None
    maps['added_support']=np.mean([ndi.binary_dilation(v,structure=np.ones((3,3),bool)) for v in added],axis=0).astype(np.float32)
    return metrics,maps,before,after


def panels(gray,m):
    raw=np.repeat(gray[:,:,None],3,axis=2)
    def mark(img,mask,c):img[mask]=(.3*img[mask]+.7*np.array(c)).astype(np.uint8)
    before=raw.copy();mark(before,m['before'],[0,230,235])
    after=raw.copy();mark(after,m['before'],[0,230,235]);mark(after,m['added'],[80,255,80]);mark(after,m['object_contour'],[240,95,205])
    oldobjects=raw.copy();mark(oldobjects,m['dark'] & ~m['old_compact'],[80,135,255]);mark(oldobjects,m['old_compact'],[255,60,170])
    newobjects=raw.copy();mark(newobjects,m['dark'] & ~m['old_compact'],[80,135,255]);mark(newobjects,m['ambiguous_compact'],[255,170,30]);mark(newobjects,m['round_deep'],[255,60,170])
    return {'raw':raw,'before':before,'after':after,'old_objects':oldobjects,'new_objects':newobjects}


def validate():
    previous.verify();p=pilot.read(OUT/'protocol.json')
    if p['profile']!=PROFILE:raise ValueError('Frozen profile changed')
    for name,sha in p['sources'].items():
        if pilot.sha(name)!=sha:raise ValueError('Frozen source changed: '+name)
    if p['versions']!=old.v1.old.versions():raise ValueError('Runtime changed')
    for stage in ('development','check'):
        if (OUT/stage/'completion.json').exists():
            for name,sha in pilot.read(OUT/stage/'completion.json')['outputs'].items():
                if pilot.sha(OUT/stage/name)!=sha:raise ValueError('Output changed: '+name)
    return p


def prepare():
    if (OUT/'protocol.json').exists():validate();return
    if OUT.exists() and any(OUT.iterdir()):raise ValueError('Nonempty output')
    previous.verify();human=pilot.validate()
    sources=[Path(__file__),TEMPLATE,Path(previous.__file__),previous.OUT/'protocol.json',previous.OUT/'proposals.npz',pilot.OUT/'protocol.json']
    protected=pilot.read(ROOT/'reports/final_scattering_blend/preparation_audit.json')['protected_files']
    assert all(pilot.sha(k)==v for k,v in protected.items())
    pilot.write(OUT/'protocol.json',{'profile':PROFILE,'versions':old.v1.old.versions(),'sources':{str(p):pilot.sha(p) for p in sources},
        'records':human['frozen']['records'],'tasks':[t for t in human['frozen']['tasks'] if t['repeat_of'] is None],
        'protected_before':protected,'created_unix':time.time()})


def stage(name):
    prepare();p=validate();folder=OUT/name
    if (folder/'completion.json').exists():print(name+' verified; no rerun');return
    if folder.exists():raise ValueError('Preserve partial stage')
    if name=='check' and not (OUT/'development/completion.json').exists():raise ValueError('Complete development first')
    folder.mkdir();cases={};rois=[];arrays={};images={};imagepanels={}
    with np.load(previous.OUT/'proposals.npz',allow_pickle=False) as baseline:
        for i,r in enumerate(p['records']):
            case=f'C{i+1:02d}'
            if (case in DEVELOPMENT)!=(name=='development'):continue
            g=old.v1.old.read_image(ROOT/'open/train'/f"{r['ID']}.png")
            q,m,before,after=diagnostics(g)
            np.testing.assert_array_equal(m['before'],baseline[case+'_boundary'])
            np.testing.assert_array_equal(m['old_compact'],baseline[case+'_compact'])
            cases[case]=q;ps=panels(g,m);imagepanels[case]=ps;images[case]={k:previous.png_uri(v) for k,v in ps.items()}
            for k,v in m.items():arrays[case+'_'+k]=v
            for t in p['tasks']:
                if t['case']!=case:continue
                x,y,w,h=t['rect'];sl=np.s_[y:y+h,x:x+w]
                row={k:t[k] for k in ('code','case','rect')}
                for label,key,probes in [('before','before',before),('after','mixed_boundary',after)]:
                    vals=[previous.overlap(m[key][sl],v[sl],'f1') for v in probes]
                    row[label+'_f1_min']=min((v for v in vals if v is not None),default=None)
                    row[label+'_pixels']=int(m[key][sl].sum())
                for key in ('added','old_compact','round_deep','ambiguous_compact','object_contour'):
                    row[key+'_pixels']=int(m[key][sl].sum())
                rois.append(row)
            print(f'{name} {case}: added={q["added_pixels"]}; compact {q["old_compact_count"]} -> round/deep {q["round_deep_count"]} + ambiguous {q["ambiguous_count"]}',flush=True)
    rois.sort(key=lambda r:r['code']);pilot.write(folder/'diagnostics.json',{'cases':cases,'rois':rois})
    np.savez_compressed(folder/'maps.npz',**arrays)
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',14)
    tasks=[r for r in rois if r['code'] in PROFILE['development_rois']] if name=='development' else [r for r in rois if int(r['code'][1:])<=12]
    for part in range((len(tasks)+2)//3):
        selected=tasks[part*3:part*3+3];sheet=Image.new('RGB',(1296,36+len(selected)*290),'#f5f6f7');d=ImageDraw.Draw(sheet)
        d.text((8,8),'Original | Before boundary | Added GREEN / separated PINK | Old compact | Round/deep PINK / ambiguous ORANGE',font=font,fill='black')
        for j,r in enumerate(selected):
            x,y,w,h=r['rect'];sl=np.s_[y:y+h,x:x+w];yy=36+j*290
            for k,key in enumerate(('raw','before','after','old_objects','new_objects')):
                im=Image.fromarray(imagepanels[r['case']][key][sl]).resize((256,256),Image.Resampling.NEAREST);sheet.paste(im,(k*260,yy))
            d.text((8,yy+260),f'{r["code"]}: +{r["added_pixels"]} edge pixels; compact pixels {r["old_compact_pixels"]} -> {r["round_deep_pixels"]} candidate / {r["ambiguous_compact_pixels"]} ambiguous. Accuracy UNVERIFIED',font=font,fill='black')
        sheet.save(folder/f'comparison_{part+1}.png')
    pilot.write(folder/'view_data.json',{'cases':cases,'rois':rois,'images':images})
    validate();assert all(pilot.sha(k)==v for k,v in p['protected_before'].items())
    pilot.write(folder/'completion.json',{'stage':name,'count':len(cases),'source':'classical_auto_proposal','ground_truth':False,
        'outputs':{f.name:pilot.sha(f) for f in folder.iterdir() if f.is_file()}})
    build_view()


def build_view():
    payload={'cases':{},'rois':[],'images':{}}
    for stage in ('development','check'):
        f=OUT/stage/'view_data.json'
        if not f.exists():continue
        d=pilot.read(f);payload['cases'].update(d['cases']);payload['images'].update(d['images']);payload['rois']+=d['rois']
    payload['rois'].sort(key=lambda r:r['code'])
    s=json.dumps(payload,ensure_ascii=False,allow_nan=False).replace('</','<\\/')
    (OUT/'index.html').write_text(TEMPLATE.read_text(encoding='utf-8').replace('__DATA__',s),encoding='utf-8')


def serve(port):
    validate()
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_GET(self):
            if self.path!='/' or self.headers.get('Host')!=f'127.0.0.1:{self.server.server_port}':self.send_error(404);return
            raw=(OUT/'index.html').read_bytes();self.send_response(200);self.send_header('Content-Type','text/html; charset=utf-8');self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
    server=ThreadingHTTPServer(('127.0.0.1',port),Handler)
    print(f'Refinement comparison: http://127.0.0.1:{port}/',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('development','check','validate','serve'));p.add_argument('--port',type=int,default=8768);a=p.parse_args()
    if a.command in ('development','check'):stage(a.command)
    elif a.command=='serve':serve(a.port)
    else:validate();print('Frozen refinement verified')
