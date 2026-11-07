"""Local, target-blind human annotation pilot. No model or automatic labels.

prepare selects12 train images from cached quality descriptors only.
serve binds127.0.0.1 and saves user annotations separately with history.
"""
from __future__ import annotations
import argparse
import base64
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import threading
import time
from urllib.parse import urlsplit
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'reports/structure_annotation_pilot'
TEMPLATE = ROOT/'scripts/structure_annotation_pilot.html'
COLS = ['ID','haar_hh_noise_proxy','variance_minus_noise_proxy','gradient_x_to_y','autocorr_lag_4','autocorr_lag_16','fraction_below_64']
PROFILE = {
    'version':1, 'selection':'Sort500 IDs; noise rank quartiles125 each. In global rank coordinates of five structure proxies, choose closest to0.5, then twice farthest minimum distance to already chosen; ID tie break. No target/OOF/common16 filtering.',
    'tasks':'First12: center128 ROIs, cycling noise quartiles for each selection round. Next12: one corner128 each, SHA256(ID+corner-v1) mod4. Last4: exact center repeats of first4, previous marks hidden.',
    'annotation':'Human only. Region0 unreviewed,1 bright interior,2 dark polygon interior,3 compact dark object,255 unobservable. Boundary0 unreviewed,1 visible centerline,2 known nonboundary,255 unobservable. No physical phase/pore truth asserted.',
    'scope':'12 train images,24 primary ROIs+4 repeats. Raw native pixels, no resize/denoise. No fitting/test access or automatic training masks.',
    'pilot':'Start with first4 only. Assess time/coverage; repeats separately before expansion. No obligation to annotate all28.',
    'review_gate_proposal':'Low/medium noise: repeat boundaryF1 at2px>=0.85, transect count difference<=10%, dark-area ratio difference<=3percentage points on adequately reviewed comparable areas; report coverage/unknowns. Practical screen, not accuracy proof.'}


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    tmp.replace(path)


def select(rows):
    rows=rows[COLS].sort_values('ID',kind='stable').reset_index(drop=True)
    if len(rows)!=500 or rows.ID.duplicated().any() or not np.isfinite(rows[COLS[1:]].to_numpy()).all():
        raise ValueError('Expected500 unique finite train records')
    ranks=rows[COLS[2:]].rank(method='first',pct=True).to_numpy()
    noise_order=np.argsort(rows.haar_hh_noise_proxy.to_numpy(),kind='stable')
    quartile=np.empty(500,int);quartile[noise_order]=np.repeat(np.arange(4),125)
    selected=[]
    for q in range(4):
        available=list(np.flatnonzero(quartile==q));chosen=[]
        for k in range(3):
            if not chosen:
                j=min(available,key=lambda j:(float(np.sum((ranks[j]-.5)**2)),rows.ID[j]))
            else:
                j=min(available,key=lambda j:(-float(np.min(np.sum((ranks[chosen]-ranks[j])**2,axis=1))),rows.ID[j]))
            chosen.append(j);available.remove(j)
            selected.append({'ID':str(rows.ID[j]),'noise_quartile':q+1,'selection_round':k,
                'descriptors':{c:float(rows.loc[j,c]) for c in COLS[1:]}})
    return sorted(selected,key=lambda x:(x['selection_round'],x['noise_quartile']))


def tasks_for(records):
    result=[]
    for i,r in enumerate(records):
        result.append({'code':f'Q{i+1:02d}','case':f'C{i+1:02d}','ID':r['ID'],'rect':[64,64,128,128],'repeat_of':None})
    corners=[[0,0,128,128],[128,0,128,128],[0,128,128,128],[128,128,128,128]]
    for i,r in enumerate(records):
        k=int(hashlib.sha256((r['ID']+'corner-v1').encode()).hexdigest()[:8],16)%4
        result.append({'code':f'Q{i+13:02d}','case':f'C{i+1:02d}','ID':r['ID'],'rect':corners[k],'repeat_of':None})
    order=sorted(range(4),key=lambda i:hashlib.sha256((records[i]['ID']+'repeat-v1').encode()).hexdigest())
    for i,j in enumerate(order):
        result.append({**result[j],'code':f'Q{i+25:02d}','repeat_of':result[j]['code']})
    return result


def prepare(folder=OUT):
    folder=Path(folder)
    if (folder/'protocol.json').exists():
        validate(folder);print('Prepared pilot verified; unchanged');return
    if folder.exists() and any(folder.iterdir()):raise ValueError('Nonempty unprepared destination')
    source=ROOT/'reports/audit_train_features.csv'
    rows=pd.read_csv(source,usecols=COLS)  # hardness is deliberately not parsed
    records=select(rows)
    sources={str(source):sha(source),str(Path(__file__).resolve()):sha(__file__),str(TEMPLATE):sha(TEMPLATE)}
    for r in records:
        if not r['ID'].startswith('TRAIN_') or not r['ID'][6:].isdigit():raise ValueError('Only train images')
        p=ROOT/'open/train'/f"{r['ID']}.png"
        with Image.open(p) as im:
            if im.size!=(256,256):raise ValueError('Native256 required')
        sources[str(p)]=sha(p)
    frozen={'profile':PROFILE,'records':records,'tasks':tasks_for(records),'sources':sources}
    write(folder/'protocol.json',{'project_id':digest(frozen),'frozen':frozen,'created_unix':time.time(),
        'targets_loaded':False,'labels_generated':False,'model_fits':0,'test_accessed':False})
    print(f'Prepared12 train images /24 primary ROIs+4 repeats at {folder}; no target values, models or labels')


def validate(folder=OUT):
    p=read(Path(folder)/'protocol.json')
    if p['project_id']!=digest(p['frozen']) or p['frozen']['profile']!=PROFILE:raise ValueError('Protocol changed')
    for name,s in p['frozen']['sources'].items():
        if sha(name)!=s:raise ValueError('Frozen input/source changed: '+name)
    return p


def public_pack(folder=OUT, demo=False):
    if demo:
        im=Image.new('L',(256,256),180);d=ImageDraw.Draw(im)
        d.rectangle((40,40,120,150),fill=100);d.ellipse((150,80,174,104),fill=25)
        d.line((70,0,170,255),fill=55,width=2);b=io.BytesIO();im.save(b,format='PNG')
        return {'project_id':'demo-v1','demo':True,'images':{'DEMO':'data:image/png;base64,'+base64.b64encode(b.getvalue()).decode()},
            'tasks':[{'code':'연습01','case':'DEMO','rect':[64,64,128,128]}]}
    p=validate(folder);images={}
    for i,r in enumerate(p['frozen']['records']):
        images[f'C{i+1:02d}']='data:image/png;base64,'+base64.b64encode((ROOT/'open/train'/f"{r['ID']}.png").read_bytes()).decode()
    return {'project_id':p['project_id'],'demo':False,'images':images,
        'tasks':[{k:t[k] for k in ('code','case','rect')} for t in p['frozen']['tasks']]}


def decode_rle(runs,allowed):
    if not isinstance(runs,list) or len(runs)>16384:raise ValueError('Invalid mask RLE')
    a=[]
    for pair in runs:
        if not isinstance(pair,list) or len(pair)!=2:raise ValueError('Invalid mask run')
        value,count=pair
        if type(value) is not int or value not in allowed or type(count) is not int or count<1 or count>16384:
            raise ValueError('Invalid class or run length')
        a.extend([value]*count)
        if len(a)>16384:raise ValueError('Mask exceeds128x128')
    if len(a)!=16384:raise ValueError('Mask must be128x128')
    return np.asarray(a,dtype=np.uint8).reshape(128,128)


def validate_annotations(doc,plan):
    if not isinstance(doc,dict) or doc.get('version')!=1 or doc.get('project_id')!=plan['project_id']:
        raise ValueError('Wrong annotation project/version')
    if set(doc)-{'version','project_id','tasks','updated_at','source'}:raise ValueError('Unexpected top-level fields')
    if doc.get('source')!='human_local_annotation':raise ValueError('Human annotation source required')
    if not isinstance(doc.get('tasks'),list) or len(doc['tasks'])>28:raise ValueError('Invalid tasks')
    expected={t['code'] for t in plan['frozen']['tasks']};seen=set()
    for t in doc['tasks']:
        if set(t)!={'code','region','boundary','notes','observation','status','h_count','v_count','seconds'}:raise ValueError('Unexpected task schema')
        if t['code'] not in expected or t['code'] in seen:raise ValueError('Unknown/duplicate task')
        seen.add(t['code'])
        r=decode_rle(t['region'],{0,1,2,3,255});b=decode_rle(t['boundary'],{0,1,2,255})
        if not isinstance(t['notes'],str) or len(t['notes'])>4000:raise ValueError('Invalid notes')
        if t['observation'] not in ('','clear','partial','unobservable'):raise ValueError('Invalid observation')
        if t['status'] not in ('draft','done'):raise ValueError('Invalid status')
        if t['status']=='done' and (not t['observation'] or (not np.any(r) and not np.any(b) and t['observation']!='unobservable')):
            raise ValueError('Completed task requires observation and marks or explicit unobservable')
        for key in ('h_count','v_count'):
            if t[key] is not None and (type(t[key]) is not int or not 0<=t[key]<=256):raise ValueError('Invalid transect count')
        if type(t['seconds']) not in (float,int) or not np.isfinite(t['seconds']) or not 0<=t['seconds']<=604800:raise ValueError('Invalid time')
    if not isinstance(doc.get('updated_at'),str) or len(doc['updated_at'])>80:raise ValueError('Invalid date')
    return doc


def save_annotations(folder,doc,revision):
    folder=Path(folder);plan=validate(folder);validate_annotations(doc,plan)
    path=folder/'annotations/latest.json'
    current=sha(path) if path.exists() else None
    if current!=revision:raise FileExistsError('Another saved version exists. Export a backup and reload before merging.')
    if path.exists():
        old=read(path)
        if old!=doc:write(folder/'annotations/history'/f'{time.time_ns()}_{current[:12]}.json',old)
    write(path,doc)
    return {'revision':sha(path),'path':str(path),'done':sum(t['status']=='done' for t in doc['tasks'])}


def make_server(folder=OUT,port=8766):
    folder=Path(folder);validate(folder);lock=threading.Lock()
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def send_json(self,status,value):
            raw=json.dumps(value,ensure_ascii=False,allow_nan=False).encode();self.send_response(status)
            self.send_header('Content-Type','application/json; charset=utf-8');self.send_header('Cache-Control','no-store')
            self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
        def safe_origin(self):
            host=f'127.0.0.1:{self.server.server_port}'
            return self.headers.get('Host')==host and self.headers.get('Origin',f'http://{host}')==f'http://{host}'
        def do_GET(self):
            if not self.safe_origin():self.send_json(403,{'error':'Local origin required'});return
            path=urlsplit(self.path).path
            try:
                if path in ('/','/demo'):
                    raw=TEMPLATE.read_bytes();self.send_response(200);self.send_header('Content-Type','text/html; charset=utf-8')
                    self.send_header('Content-Length',str(len(raw)));self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(raw)
                elif path=='/api/project':self.send_json(200,public_pack(folder))
                elif path=='/api/demo':self.send_json(200,public_pack(folder,True))
                elif path=='/api/annotations':
                    p=folder/'annotations/latest.json'
                    self.send_json(200,{'revision':sha(p) if p.exists() else None,'document':read(p) if p.exists() else None})
                else:self.send_json(404,{'error':'Not found'})
            except Exception as exc:self.send_json(400,{'error':str(exc)})
        def do_POST(self):
            if not self.safe_origin() or self.headers.get('X-Annotation-Pilot')!='1':self.send_json(403,{'error':'Local annotation client required'});return
            if self.path!='/api/annotations':self.send_json(404,{'error':'Not found'});return
            try:
                n=int(self.headers.get('Content-Length','0'))
                if n<1 or n>6*1024*1024:raise ValueError('Invalid request size')
                msg=json.loads(self.rfile.read(n))
                with lock:reply=save_annotations(folder,msg['document'],msg.get('revision'))
                self.send_json(200,reply)
            except FileExistsError as exc:self.send_json(409,{'error':str(exc)})
            except Exception as exc:self.send_json(400,{'error':str(exc)})
    return ThreadingHTTPServer(('127.0.0.1',port),Handler)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('prepare','serve','validate'))
    p.add_argument('--port',type=int,default=8766);a=p.parse_args()
    if a.command=='prepare':prepare()
    elif a.command=='validate':validate();print('Frozen pilot verified; no annotations generated')
    else:
        server=make_server(port=a.port);print(f'Human annotation: http://127.0.0.1:{server.server_port}/ ; practice: /demo ; Ctrl+C stops. No model training.',flush=True)
        try:server.serve_forever()
        except KeyboardInterrupt:pass
        finally:server.server_close()


if __name__=='__main__':main()
