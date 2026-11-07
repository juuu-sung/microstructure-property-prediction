"""Read-only classical automatic review of the12 already selected train images.

No learning, hardness parsing, test images, human annotation writes or pseudo-label adoption.
The immutable v2 detector and v3 diagnostic rules are reused without tuning.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi

if __package__:
    from . import structure_annotation_pilot as pilot
    from . import audit_boundary_quality_v3 as quality
else:
    import structure_annotation_pilot as pilot
    import audit_boundary_quality_v3 as quality

ROOT = pilot.ROOT
OUT = ROOT / 'reports/structure_auto_review'
TEMPLATE = ROOT / 'scripts/auto_structure_review.html'
PROFILE = {
    'version': 1, 'scope': 'Same12 target-blind pilot train images;24 primary ROIs; no repeats counted as new samples',
    'source': 'classical_auto_proposal', 'ground_truth': False, 'training_approved': False,
    'boundary': 'Unchanged v2 mixed step/valley detector + unchanged v3 six-probe quality assessment',
    'auxiliary': 'Unchanged v2 baseline: per-image NLM/p2-p98 normalized dark mask and compact dark candidates',
    'probes': list(quality.INTERNAL),
    'pixel_support': 'Among six inverse-aligned probes, presence within1 Chebyshev pixel. Base boundary with>=5/6 support cyan; other base pixels orange; newly appearing pixels not shown as truth.',
    'region_support': 'Exact-pixel dark membership fraction across six probes. Dark/compact masks remain ambiguous candidates. No inferred bright-interior or physical pore labels.',
    'roi_diagnostics': 'Crop full-image masks to128x128; minimum nonempty tolerant boundary F1 and exact dark IoU over probes. Empty/empty is null, not perfect detection. No crop detector recalibration.',
    'alerts': 'ROI boundary F1<0.85, dark IoU<0.80, or unavailable signal; display whole-image v3 state separately. Descriptive review flags, no accuracy threshold.',
    'human_reference': 'Independent raw-first human annotations remain needed for accuracy. Viewing auto overlays may bias later annotations; they are not independent references.',
    'labels_loaded': False, 'model_fits': 0, 'test_accessed': False,
}
STATUS_KO = {'repeatable_proxy':'반복 안정', 'borderline_proxy':'경계 사례', 'unstable_proxy':'변형에 불안정',
             'unavailable':'관측량 부족', 'null_like':'공간 구조 근거 부족'}


def overlap(a, b, kind):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    if not a.any() and not b.any(): return None
    if kind == 'iou': return float((a & b).sum() / (a | b).sum())
    if kind == 'f1': return quality.detector.v1.tolerant_f1(a, b)
    raise ValueError(kind)


def summarize_roi(maps, boundary_probes, dark_probes, rect):
    x,y,w,h = rect
    if w!=128 or h!=128 or x<0 or y<0 or x+w>256 or y+h>256: raise ValueError('Native128 ROI required')
    sl = np.s_[y:y+h,x:x+w]
    b,d,c = (maps[k][sl] for k in ('boundary','dark','compact'))
    fs = [overlap(b, v[sl], 'f1') for v in boundary_probes]
    js = [overlap(d, v[sl], 'iou') for v in dark_probes]
    fmin = min((v for v in fs if v is not None), default=None)
    jmin = min((v for v in js if v is not None), default=None)
    flags = []
    if not b.any(): flags.append('경계 후보 없음: 구조 부재로 단정하지 않음')
    elif fmin is not None and fmin<.85: flags.append('경계 위치가 변형에 민감')
    if not d.any(): flags.append('어두운 영역 후보 없음: 미검출 가능')
    if jmin is not None and jmin<.80: flags.append('어두운 영역 면적·위치가 변형에 민감')
    return {'boundary_pixels':int(b.sum()), 'boundary_support_ge5of6_fraction':float((maps['boundary_support'][sl][b]>=5/6-1e-7).mean()) if b.any() else None,
        'boundary_f1_min':fmin, 'boundary_f1_by_probe':fs, 'dark_iou_min':jmin, 'dark_iou_by_probe':js,
        'dark_area_fraction':float(d.mean()), 'compact_area_fraction':float(c.mean()),
        'horizontal_center_crossings':len(quality.detector.centers(b[64])),
        'vertical_center_crossings':len(quality.detector.centers(b[:,64])),
        'review_flags':flags, 'semantic_validity':'unverified', 'ground_truth':False}


def assess(gray):
    base,maps = quality.detector.detect(gray)
    _,aux = quality.detector.baseline(gray)
    maps.update({k:aux[k] for k in ('dark','compact')})
    probes=[]; bp=[]; dp=[]; cp=[]
    for name in quality.INTERNAL:
        g=quality.perturb(gray,name)
        m,b=quality.detector.detect(g)
        _,a=quality.detector.baseline(g)
        b=quality.detector.v1.align(b['boundary'],name)
        d=quality.detector.v1.align(a['dark'],name)
        c=quality.detector.v1.align(a['compact'],name)
        bp.append(b); dp.append(d); cp.append(c)
        probes.append({'name':name,'map_f1':quality.detector.v1.tolerant_f1(maps['boundary'],b),
            'spacing_change':quality.detector.v1.relative(base['raw_spacing'],m['raw_spacing']),
            'raw_spacing':m['raw_spacing'],'observation':quality.observation(m)})
    q=quality.decide(base,probes,quality.spatial_evidence(gray))
    q['probes']=probes
    maps['boundary_support']=np.mean([ndi.binary_dilation(b,structure=np.ones((3,3),bool)) for b in bp],axis=0).astype(np.float32)
    maps['dark_support']=np.mean(dp,axis=0).astype(np.float32)
    maps['compact_support']=np.mean(cp,axis=0).astype(np.float32)
    return q,maps,bp,dp


def panels(gray,maps):
    original=np.repeat(gray[:,:,None],3,axis=2)
    def tint(img,mask,rgb,alpha=.70):
        img[mask]=np.rint((1-alpha)*img[mask]+alpha*np.array(rgb)).astype(np.uint8)
    boundary=original.copy()
    stable=maps['boundary'] & (maps['boundary_support']>=5/6-1e-7)
    tint(boundary,stable,[0,230,235]); tint(boundary,maps['boundary'] & ~stable,[255,130,35])
    regions=original.copy(); tint(regions,maps['dark'] & ~maps['compact'],[90,145,255],.55)
    tint(regions,maps['compact'],[255,75,155],.70)
    risk=original.copy()
    # Only proposal disagreement, never a calibrated error probability or ignore label.
    dark_variable=(maps['dark_support']>0) & (maps['dark_support']<1)
    tint(risk,dark_variable,[245,190,40],.65)
    tint(risk,ndi.binary_dilation(maps['boundary'] & ~stable),[255,75,65],.80)
    return {'original':original,'boundary':boundary,'regions':regions,'sensitivity':risk}


def png_uri(array):
    b=io.BytesIO(); Image.fromarray(array).save(b,format='PNG')
    return 'data:image/png;base64,'+base64.b64encode(b.getvalue()).decode()


def contact_sheets(folder,records,outputs):
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',15)
    for page in range(3):
        sheet=Image.new('RGB',(1064,1260),'#f3f5f6'); draw=ImageDraw.Draw(sheet)
        draw.text((10,8),'AUTO PROPOSALS ONLY | Original | Boundary | Dark / compact | Sensitivity',font=font,fill='black')
        for row,(case,rec) in enumerate(records[page*4:page*4+4]):
            y=48+row*300; p,q=outputs[case]
            for col,key in enumerate(('original','boundary','regions','sensitivity')):
                sheet.paste(Image.fromarray(p[key]),(8+264*col,y))
            draw.text((8,y+261),f'{case}: {q["status"]}; min support={q["support_min"]:.3f}; semantic correctness UNVERIFIED',font=font,fill='black')
        sheet.save(folder/f'overview_{page+1}.png')


def verify(folder=OUT):
    p=pilot.read(folder/'protocol.json')
    if p['profile']!=PROFILE: raise ValueError('Auto review profile changed')
    pilot.validate()
    for path,h in p['sources'].items():
        if pilot.sha(path)!=h: raise ValueError('Frozen source changed: '+path)
    if (folder/'completion.json').exists():
        for path,h in pilot.read(folder/'completion.json')['outputs'].items():
            if pilot.sha(folder/path)!=h: raise ValueError('Saved output changed: '+path)
    return p


def run():
    human=pilot.validate()
    quality.validate()
    if (OUT/'completion.json').exists(): verify(); print('Completed automatic review verified; no recomputation'); return
    if OUT.exists() and any(OUT.iterdir()): raise ValueError('Preserve partial auto review folder')
    protected=pilot.read(ROOT/'reports/final_scattering_blend/preparation_audit.json')['protected_files']
    assert all(pilot.sha(k)==v for k,v in protected.items())
    source_paths=[Path(__file__),TEMPLATE,pilot.OUT/'protocol.json',Path(pilot.__file__),
        Path(quality.__file__),Path(quality.detector.__file__),Path(quality.detector.v1.__file__),Path(quality.detector.v1.old.__file__)]
    pilot.write(OUT/'protocol.json',{'profile':PROFILE,'human_project_id':human['project_id'],
        'sources':{str(p):pilot.sha(p) for p in source_paths},'versions':quality.detector.v1.old.versions(),
        'created_unix':time.time(),'human_annotation_visibility':'Auto overlays exposed after this run; later marks may be assisted, not independent raw-only references'})
    rows=[]; arrays={}; images={}; details={}; outputs={}; records=[]; start=time.monotonic()
    for i,rec in enumerate(human['frozen']['records']):
        case=f'C{i+1:02d}'; records.append((case,rec))
        g=quality.detector.v1.old.read_image(ROOT/'open/train'/f"{rec['ID']}.png")
        q,m,bp,dp=assess(g); details[case]=q
        p=panels(g,m); outputs[case]=(p,q); images[case]={k:png_uri(v) for k,v in p.items()}
        for key,value in m.items(): arrays[case+'_'+key]=value
        for t in human['frozen']['tasks']:
            if t['case']==case and t['repeat_of'] is None:
                rows.append({k:t[k] for k in ('code','case','rect')} | summarize_roi(m,bp,dp,t['rect']))
        print(f"Automatic review {i+1}/12 {case}: {q['status']}; support={q['support_min']:.3f}; no training",flush=True)
    rows.sort(key=lambda r:r['code'])
    summary={'images':12,'primary_rois':24,'image_status_counts':dict(Counter(q['status'] for q in details.values())),
        'rois_with_review_flags':sum(bool(r['review_flags']) for r in rows),
        'rois_boundary_f1_below085':sum(r['boundary_f1_min'] is not None and r['boundary_f1_min']<.85 for r in rows),
        'rois_dark_iou_below080':sum(r['dark_iou_min'] is not None and r['dark_iou_min']<.80 for r in rows),
        'semantic_accuracy_measured':False,'human_approved_masks':0,'model_fits':0,'labels_loaded':False,'test_accessed':False,
        'seconds':time.monotonic()-start}
    pilot.write(OUT/'diagnostics.json',{'summary':summary,'images':details,'rois':rows})
    np.savez_compressed(OUT/'proposals.npz',**arrays)
    payload={'summary':summary,'images':images,'qualities':details,'rois':rows,'status_names':STATUS_KO}
    html=TEMPLATE.read_text(encoding='utf-8').replace('__REVIEW_DATA__',json.dumps(payload,ensure_ascii=False,allow_nan=False).replace('</','<\\/'))
    (OUT/'index.html').write_text(html,encoding='utf-8')
    contact_sheets(OUT,records,outputs)
    verify()
    assert all(pilot.sha(k)==v for k,v in protected.items())
    pilot.write(OUT/'completion.json',{'summary':summary,'protected_files_verified':len(protected),
        'outputs':{p.name:pilot.sha(p) for p in OUT.iterdir() if p.is_file()},'source':'classical_auto_proposal','training_approved':False})
    print(json.dumps(summary,ensure_ascii=False,indent=2))


def serve(port=8767):
    verify()
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args): pass
        def do_GET(self):
            if self.headers.get('Host')!=f'127.0.0.1:{self.server.server_port}' or self.path not in ('/','/index.html'):
                self.send_error(404); return
            raw=(OUT/'index.html').read_bytes()
            self.send_response(200); self.send_header('Content-Type','text/html; charset=utf-8')
            self.send_header('Content-Length',str(len(raw))); self.send_header('Cache-Control','no-store'); self.end_headers(); self.wfile.write(raw)
    server=ThreadingHTTPServer(('127.0.0.1',port),Handler)
    print(f'Automatic review (read only): http://127.0.0.1:{server.server_port}/',flush=True)
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally: server.server_close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('command',choices=('run','validate','serve')); p.add_argument('--port',type=int,default=8767)
    a=p.parse_args()
    if a.command=='run': run()
    elif a.command=='validate': verify(); print('Automatic review verified')
    else: serve(a.port)
