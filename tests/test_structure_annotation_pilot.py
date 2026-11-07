import copy
import http.client
import json
from pathlib import Path
import shutil
import tempfile
import threading
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from PIL import Image
from scripts import structure_annotation_pilot as p


class PilotTests(unittest.TestCase):
    def setUp(self):
        self.folder=Path(tempfile.mkdtemp(prefix='pilot-test-',dir=Path(__file__).parent))
        self.plan={'project_id':'fixture','frozen':{'tasks':[{'code':'Q01'}]}}
    def tearDown(self):
        assert self.folder.resolve().parent==Path(__file__).resolve().parent
        shutil.rmtree(self.folder)
    def frame(self):
        rng=np.random.default_rng(17)
        return pd.DataFrame({'ID':[f'TRAIN_{i:06}' for i in range(1,501)],**{c:rng.random(500) for c in p.COLS[1:]}})
    def doc(self):
        return {'version':1,'project_id':'fixture','source':'human_local_annotation','updated_at':'2026-10-12T00:00:00Z',
            'tasks':[{'code':'Q01','region':[[0,16384]],'boundary':[[0,16384]],'notes':'',
                'observation':'','status':'draft','h_count':None,'v_count':None,'seconds':0}]}
    def test_target_blind_selection_and_repeats(self):
        frame=self.frame();a=p.select(frame)
        frame['hardness']=np.arange(500)*999;frame['prediction']=-np.arange(500)
        self.assertEqual(a,p.select(frame.sample(frac=1,random_state=9)))
        self.assertEqual([r['noise_quartile'] for r in a[:4]],[1,2,3,4])
        self.assertEqual(len({r['ID'] for r in a}),12)
        tasks=p.tasks_for(a);self.assertEqual(len(tasks),28)
        for t in tasks[24:]:
            original=next(x for x in tasks[:4] if x['code']==t['repeat_of'])
            self.assertEqual(t['rect'],original['rect']);self.assertEqual(t['ID'],original['ID'])
        with self.assertRaises(ValueError):p.select(frame.iloc[:499])
    def test_masks_keep_unknown_distinct_and_validate_lengths(self):
        a=p.decode_rle([[0,100],[255,100],[1,16184]],{0,1,255})
        self.assertEqual(a.shape,(128,128));self.assertEqual((a==255).sum(),100)
        for runs in ([[1,16383]],[[7,16384]],[[True,16384]],[[0,16385]],[[0,0]],[[1,16384.0]]):
            with self.assertRaises(ValueError):p.decode_rle(runs,{0,1,255})
    def test_completion_and_import_safety(self):
        doc=self.doc();p.validate_annotations(doc,self.plan)
        doc['tasks'][0]['status']='done'
        with self.assertRaises(ValueError):p.validate_annotations(doc,self.plan)
        doc['tasks'][0]['observation']='unobservable';p.validate_annotations(doc,self.plan)
        for key,value in [('code','../escape'),('seconds',float('nan')),('h_count',-1),('region',[[8,16384]])]:
            bad=copy.deepcopy(doc);bad['tasks'][0][key]=value
            with self.assertRaises(ValueError):p.validate_annotations(bad,self.plan)
        bad=copy.deepcopy(doc);bad['tasks']*=2
        with self.assertRaises(ValueError):p.validate_annotations(bad,self.plan)
    def test_save_revisions_and_history(self):
        doc=self.doc()
        with patch.object(p,'validate',return_value=self.plan):
            first=p.save_annotations(self.folder,doc,None)
            doc['tasks'][0]['notes']='human note'
            with self.assertRaises(FileExistsError):p.save_annotations(self.folder,doc,None)
            second=p.save_annotations(self.folder,doc,first['revision'])
            self.assertNotEqual(first['revision'],second['revision'])
            history=list((self.folder/'annotations/history').glob('*.json'))
            self.assertEqual(len(history),1);self.assertEqual(p.read(history[0])['tasks'][0]['notes'],'')
    def test_prepare_pack_has_no_targets_or_ids(self):
        (self.folder/'reports').mkdir();(self.folder/'open/train').mkdir(parents=True)
        frame=self.frame();frame['hardness']=123.456789
        frame.to_csv(self.folder/'reports/audit_train_features.csv',index=False)
        for record in p.select(frame):Image.new('L',(256,256),127).save(self.folder/'open/train'/f"{record['ID']}.png")
        out=self.folder/'pilot'
        with patch.object(p,'ROOT',self.folder):
            p.prepare(out);p.prepare(out);pack=p.public_pack(out)
            text=json.dumps(pack)
            self.assertNotIn('TRAIN_',text);self.assertNotIn('hardness',text);self.assertNotIn('123.456789',text)
            self.assertNotIn('repeat_of',text);self.assertFalse((out/'annotations').exists())
            plan=p.validate(out);image=self.folder/'open/train'/f"{plan['frozen']['records'][0]['ID']}.png"
            image.write_bytes(b'changed')
            with self.assertRaises(ValueError):p.validate(out)
    def test_http_loopback_and_post_validation(self):
        with patch.object(p,'validate',return_value=self.plan),patch.object(p,'public_pack',return_value={'project_id':'fixture'}):
            server=p.make_server(self.folder,0);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
            def request(method,path,body=None,headers=None):
                conn=http.client.HTTPConnection('127.0.0.1',server.server_port,timeout=5)
                conn.request(method,path,body=body,headers=headers or {});r=conn.getresponse();result=(r.status,r.read());conn.close();return result
            try:
                self.assertEqual(request('GET','/api/project')[0],200)
                self.assertEqual(request('GET','/../open/train.csv')[0],404)
                self.assertEqual(request('GET','/api/project',headers={'Origin':'https://example.com'})[0],403)
                body=json.dumps({'document':self.doc(),'revision':None})
                self.assertEqual(request('POST','/api/annotations',body)[0],403)
                result=request('POST','/api/annotations',body,{'X-Annotation-Pilot':'1','Content-Type':'application/json'})
                self.assertEqual(result[0],200)
                self.assertEqual(request('GET','/api/annotations')[0],200)
            finally:server.shutdown();thread.join();server.server_close()


if __name__=='__main__':unittest.main()
