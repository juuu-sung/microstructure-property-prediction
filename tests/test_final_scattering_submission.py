"""Synthetic/mocked final pipeline tests; no competition fitting or extraction."""
import contextlib
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from scripts import final_scattering_submission as f


@contextlib.contextmanager
def temporary():
    parent = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory(dir=parent,prefix='.final_scattering_test_') as td:
        if Path(td).resolve().parent != parent:raise ValueError('Cleanup target outside tests')
        yield Path(td)


def fixture():
    rng = np.random.default_rng(492)
    x = rng.normal(size=(1500,486));x[:,0]=np.arange(1500)
    data = SimpleNamespace(ids=np.array([f'TRAIN_{j:06d}' for j in range(500)]),y=190+3*x[:500,1])
    ids = [f'TEST_{j:06d}' for j in range(1000,0,-1)]
    images = [{'id':i,'relative_path':f'test/{i}.png','sha256':'mock'} for i in ids]
    frozen = {'synthetic':True,'runtime':{'synthetic':True},'resolved_config':{'synthetic':True},
              'anchor_ids':data.ids[:16].tolist(),'test_binding':{'ids':ids,'rows':1000,'images':images}}
    return data,x[:500].copy(),x[500:,:285].copy(),x[500:,285:].copy(),np.linspace(180,220,1000),np.linspace(190,210,1000),{'frozen':frozen}


class FakePredictor:
    runtime = {'synthetic':True}
    def __init__(self):self.calls=[]
    def __call__(self,x,y,q,seed):
        np.testing.assert_array_equal(y,190+3*x[:,1])
        if np.any(x[:,0]>=500) or set(x[:,0])&set(q[:,0]):raise AssertionError('Test entered context')
        self.calls.append((x.copy(),y.copy(),q.copy(),seed))
        return 190+3*q[:,1]+.001*q[:,0],{'seconds':0.,'peak_gpu_allocated_bytes':0,
            'context_rows':len(x),'query_rows':len(q),'columns':x.shape[1],
            'context_mean':float(y.mean()),'context_std':float(y.std()),
            'runtime':self.runtime,'resolved_config':{'synthetic':True}}


class FinalScatteringTests(unittest.TestCase):
    def test_full500_1000_one_call_correct_base_blend_sample_order_resume_and_tamper(self):
        data,x,tx,added,base,old,plan=fixture();predictor=FakePredictor()
        with temporary() as root,contextlib.redirect_stdout(io.StringIO()):
            folder=root/'candidate';(folder/'test_features').mkdir(parents=True)
            f.t.write_json(folder/'test_features/completed.json',{'mock':True})
            prior=root/'previous.csv';prior.write_bytes(b'preserved previous submission')
            before=prior.read_bytes()
            with patch.object(f,'validate',return_value=(data,x,tx,base,old,plan)), \
                 patch.object(f,'load_test_cache',return_value=np.c_[tx,added]), \
                 patch.object(f.t,'Predictor',return_value=predictor), \
                 patch.object(f.t,'verify_weights',return_value=Path('mock.ckpt')):
                f.predict_stage(folder,'cuda')
                self.assertEqual(len(predictor.calls),1)
                a,y,b,seed=predictor.calls[0]
                self.assertEqual((a.shape,b.shape),((500,486),(1000,486)))
                self.assertEqual(seed,31424)
                expected=.5*(base+190+3*tx[:,1]+.001*tx[:,0])
                path=folder/'submission/submission.csv'
                rows=f.t.core.read_rows(path,{'ID','hardness'})
                saved=f.t.core.finite_column(rows,'hardness',path)
                np.testing.assert_allclose(saved,expected,rtol=0,atol=1e-12)
                self.assertEqual(f.original.submission_ids(path),plan['frozen']['test_binding']['ids'])
                self.assertFalse(np.allclose(saved,.5*(old+190+3*tx[:,1]+.001*tx[:,0])))
                with patch.object(f.t,'Predictor',side_effect=AssertionError('No model on completed inference')):
                    f.predict_stage(folder,'cuda')
                self.assertEqual(prior.read_bytes(),before)
                with (folder/'submission/component_predictions.csv').open('a') as h:h.write('corrupt')
                with self.assertRaises(ValueError):f.predict_stage(folder,'cuda')

    def test_anchor_failure_before_test_extraction_and_saved_anchor_validation(self):
        data,x,_,_,_,_,plan=fixture()
        seen=[]
        def read(path):seen.append(str(path));return int(path.stem.split('_')[1])
        def correct(j):return x[j,285:].copy(),{'mock':True}
        with temporary() as folder,patch.object(f.sf,'read_gray',side_effect=read):
            with self.assertRaisesRegex(ValueError,'anchor mismatch'):
                f.anchor_record(folder,plan,x[:16,285:],lambda j:(x[j,285:]+1,{}))
            self.assertTrue(all('train' in p for p in seen))
            self.assertFalse((folder/'test_features/anchor.json').exists())
            f.anchor_record(folder,plan,x[:16,285:],correct)
            seen.clear();f.anchor_record(folder,plan,x[:16,285:])
            self.assertEqual(seen,[])
            with self.assertRaises(ValueError):f.anchor_record(folder,plan,x[:16,285:]+1)

    def test_test_cell_resume_corruption_and_image_binding(self):
        with temporary() as root:
            image=root/'open/test/TEST_000001.png';image.parent.mkdir(parents=True);image.write_bytes(b'mock image')
            item={'id':'TEST_000001','relative_path':'test/TEST_000001.png','sha256':f.t.common.sha256_file(image)}
            folder=root/'candidate';plan={'frozen':{'mock':True}};calls=[]
            def extract(a):calls.append(a);return np.arange(201,dtype=np.float64),{'mock':True}
            with patch.object(f,'ROOT',root),patch.object(f.sf,'read_gray',return_value='mock'):
                a=f.test_cell(folder,plan,item,extract);b=f.test_cell(folder,plan,item)
                np.testing.assert_array_equal(a,b);self.assertEqual(len(calls),1)
                p=folder/'test_features/cells/TEST_000001.json';saved=f.t.read_json(p)
                saved['result']['features'][0]+=1;f.t.write_json(p,saved)
                with self.assertRaises(ValueError):f.test_cell(folder,plan,item)
                image.write_bytes(b'changed')
                with self.assertRaises(ValueError):f.test_cell(root/'other',plan,item,extract)

    def test_extract_preserves285_and_caches_all1000_rows_and_anchors(self):
        data,x,tx,added,base,old,plan=fixture();read_count=[];extract_count=[]
        ids=plan['frozen']['test_binding']['ids'];test_index={v:j for j,v in enumerate(ids)}
        def read(path):
            read_count.append(path)
            return ('train',int(path.stem.split('_')[1])) if path.parent.name=='train' else ('test',test_index[path.stem])
        def extract(key):
            extract_count.append(key);split,j=key
            return (x[j,285:].copy() if split=='train' else added[j].copy()),{'mock':True}
        with temporary() as root,contextlib.redirect_stdout(io.StringIO()):
            for item in plan['frozen']['test_binding']['images']:
                path=root/'open'/item['relative_path'];path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(item['id'].encode())
                item['sha256']=f.t.common.sha256_file(path)
            folder=root/'candidate'
            with patch.object(f,'ROOT',root),patch.object(f,'validate',return_value=(data,x,tx,base,old,plan)), \
                 patch.object(f.sf,'read_gray',side_effect=read),patch.object(f.sf,'Extractor',return_value=extract):
                f.extract_stage(folder)
                self.assertEqual(len(extract_count),1016)
                out=f.load_test_cache(folder,plan,x,tx)
                np.testing.assert_array_equal(out,np.c_[tx,added])
                with patch.object(f.sf,'Extractor',side_effect=AssertionError('No extraction when complete')):
                    f.extract_stage(folder)
                self.assertEqual(len(extract_count),1016)
                with self.assertRaises(AssertionError):f.load_test_cache(folder,plan,x,tx+1.)
                changed={'frozen':dict(plan['frozen'],changed=True)}
                with self.assertRaises(ValueError):f.load_test_cache(folder,changed,x,tx)

    def test_prepare_idempotent_and_frozen_state_changes_rejected(self):
        data,x,tx,_,base,old,plan=fixture()
        with temporary() as root,contextlib.redirect_stdout(io.StringIO()):
            folder=root/'candidate'
            with patch.object(f,'load_inputs',return_value=(data,x,tx,base,old,plan['frozen'])), \
                 patch.object(f.t,'Predictor',side_effect=AssertionError('No inference in prepare')), \
                 patch.object(f.sf,'Extractor',side_effect=AssertionError('No extraction in prepare')):
                f.prepare(folder);before=(folder/'plan.json').read_bytes();f.prepare(folder)
                self.assertEqual(before,(folder/'plan.json').read_bytes())
            with patch.object(f,'load_inputs',return_value=(data,x,tx,base,old,{'changed':True})):
                with self.assertRaises(ValueError):f.validate(folder)

    def test_runtime_mismatch_blocks_prediction(self):
        data,x,tx,added,base,old,plan=fixture();p=FakePredictor();p.runtime={'wrong':True}
        with temporary() as folder,patch.object(f,'validate',return_value=(data,x,tx,base,old,plan)), \
             patch.object(f,'load_test_cache',return_value=np.c_[tx,added]), \
             patch.object(f.t,'Predictor',return_value=p),patch.object(f.t,'verify_weights',return_value=Path('mock.ckpt')):
            f.t.write_json(folder/'test_features/completed.json',{'mock':True})
            with self.assertRaises(ValueError):f.predict_stage(folder,'cpu')
            self.assertEqual(len(p.calls),0)

    def test_submission_rejects_protected_paths_bad_arrays_and_changed_binding(self):
        for path in (f.previous.BASE,f.previous.OUT/'submission',f.ROOT/'open',f.ROOT):
            with self.assertRaises(ValueError):f.protect_destination(path)
        with temporary() as root,contextlib.redirect_stdout(io.StringIO()):
            ids=['TEST_000002','TEST_000001'];base=np.array([180.,200.]);pred=np.array([190.,220.]);old=np.array([185.,210.])
            with self.assertRaises(ValueError):f.write_submission(root,ids,base,np.array([np.nan,0]),old,{})
            with self.assertRaises(ValueError):f.write_submission(root,[ids[0]]*2,base,pred,old,{})
            f.write_submission(root,ids,base,pred,old,{'plan':'one'})
            with self.assertRaises(ValueError):f.write_submission(root,ids,base,pred,old,{'plan':'two'})


if __name__=='__main__':unittest.main()
