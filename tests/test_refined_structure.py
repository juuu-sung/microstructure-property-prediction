"""Synthetic isolation, measurement absence, comparison arithmetic and cache tests."""
from contextlib import contextmanager
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
from scripts import evaluate_refined_structure as e
from scripts import refined_structure_features as f


@contextmanager
def temporary():
    parent=Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory(dir=parent,prefix='.refined_test_') as name:
        if Path(name).resolve().parent!=parent:raise ValueError('Temporary cleanup outside tests')
        yield Path(name)


def fixture():
    rng=np.random.default_rng(17);base=rng.normal(size=(500,486));base[:,0]=np.arange(500)
    extra=np.zeros((500,14),dtype=np.float64);extra[:,f.NULLABLE]=np.nan;extra[:,f.FLAGS]=1.
    extra[::3,1]=15.;extra[::3,2]=0.
    data=SimpleNamespace(ids=np.array([f'TRAIN_{i:06d}' for i in range(500)]),y=100+base[:,2],noise=np.arange(500.))
    ref={'folds':np.repeat(np.arange(5),100),'base':data.y+4,'tabpfn486':data.y+2,
         'current':data.y+3,'runtime':{'synthetic':True}}
    return data,base,{r:extra.copy() for r in e.REPS},ref,{'frozen':{'common_failure_ids':data.ids[:16].tolist()}}


class FakePredictor:
    calls=[]
    def __init__(self,*args):self.runtime={'synthetic':True}
    def __call__(self,x,y,q,seed):
        self.calls.append((x.copy(),y.copy(),q.copy(),seed))
        if not np.isfinite(x).all() or not np.isfinite(q).all():raise AssertionError('Not imputed')
        return 100+q[:,2]+.01*y.mean(),{'seconds':0.}


class RefinedTests(unittest.TestCase):
    def setUp(self):FakePredictor.calls=[]

    def test_summary_geometry_missing_and_abstention(self):
        blank={k:np.zeros((256,256),bool) for k in f.MAP_KEYS}
        a=f.summarize(blank,[blank]*6,False)
        self.assertTrue(np.isnan(a[list(f.NULLABLE)]).all())
        self.assertEqual(a[11],0.);self.assertEqual(a[5],0.)
        m={k:v.copy() for k,v in blank.items()}
        m['before'][:,[32,96,160,224]]=True;m['structural_proxy'][:]=m['before']
        m['old_compact'][50:55,50:55]=True;m['ambiguous_compact'][:]=m['old_compact']
        m['dark'][:]=m['old_compact']
        old=f.summarize(m,[m]*6,False);new=f.summarize(m,[m]*6,True)
        self.assertEqual(old[1],64.);self.assertEqual(old[2],0.)
        self.assertEqual(old[5],.5);self.assertEqual(old[11],1.)
        self.assertEqual(old[12],1.);self.assertEqual(old[7],25/65536)
        self.assertEqual(new[7],0.);self.assertEqual(new[8],25/65536)
        self.assertTrue(np.isnan(new[12]));self.assertEqual(new[13],1.)
        other={k:v.copy() for k,v in m.items()};other['old_compact'][:]=False
        unstable=f.summarize(m,[other]*6,False)
        self.assertEqual(unstable[12],0.);self.assertEqual(unstable[13],0.)

    def test_context_only_imputation_keeps486_and_flags(self):
        _,base,extra,_,_=fixture();x=np.c_[base,extra['refined14']]
        query=x[:100].copy();query[:,487]=1e9;query[:,488]=0.
        a,b,meta=f.impute_context(x[100:],query)
        np.testing.assert_array_equal(a[:,:486],x[100:,:486])
        np.testing.assert_array_equal(b[:,:486],query[:,:486])
        self.assertEqual(meta['context_medians'][0]['median'],15.)
        self.assertTrue(meta['context_medians'][1]['all_missing_context'])
        self.assertTrue(np.all(a[:,490]==1.));self.assertTrue(np.all(a[:,489]==0.))
        self.assertFalse(meta['query_used_for_fit'])
        bad=x.copy();bad[:,488]=1.
        with self.assertRaises(ValueError):f.impute_context(bad[:400],bad[400:])

    def test_frozen_detector_synthetic_extraction_no_labels(self):
        gray=np.full((256,256),128,np.uint8)
        rows,maps=f.extract(gray)
        self.assertEqual(set(rows),set(e.REPS))
        for row in rows.values():f.validate_matrix(row[None,:])
        self.assertEqual(maps['before'].sum(),0)
        self.assertEqual(maps['structural_proxy'].sum(),0)
        self.assertEqual(len(rows['refined14']),14)

    def test_outer_query_labels_never_passed_and_cache_binding(self):
        data,base,extra,ref,_=fixture();matrices={r:np.c_[base,extra[r]] for r in e.REPS}
        with temporary() as folder:
            predictor=e.CandidatePredictor(FakePredictor())
            vi,pa=e.evaluate_fold(folder/'a',{},matrices,data.y,ref['folds'],42,0,predictor)
            y=data.y.copy();y[:100]+=10000
            _,pb=e.evaluate_fold(folder/'b',{},matrices,y,ref['folds'],42,0,predictor)
            for rep in e.REPS:np.testing.assert_array_equal(pa[rep],pb[rep])
            for x,yy,q,seed in FakePredictor.calls:
                self.assertEqual((x.shape,q.shape),((400,500),(100,500)))
                self.assertEqual(seed,51);self.assertFalse(set(x[:,0])&set(q[:,0]))
                np.testing.assert_array_equal(yy,data.y[x[:,0].astype(int)])
            e.evaluate_fold(folder/'a',{},matrices,data.y,ref['folds'],42,0,predictor)
            self.assertEqual(len(FakePredictor.calls),4)
            with self.assertRaises(ValueError):e.evaluate_fold(folder/'a',{},matrices,data.y+1,ref['folds'],42,0,predictor)

    def test_two_seed_mock_end_to_end_resume_and_tamper(self):
        data,base,extra,ref,plan=fixture()
        with temporary() as folder:
            e.t.write_json(folder/'features/metadata.json',{'mock':True})
            with patch.object(e,'validate',return_value=(data,base,{s:ref for s in e.SEEDS},plan)), \
                 patch.object(e,'load_features',return_value=(extra,{})), \
                 patch.object(e.t,'Predictor',FakePredictor),patch.object(e.t,'verify_weights',return_value=Path('fake.ckpt')):
                for s in e.SEEDS:e.evaluate(folder,'cuda',s)
                self.assertEqual(len(FakePredictor.calls),20)
                for s in e.SEEDS:e.evaluate(folder,'cuda',s)
                self.assertEqual(len(FakePredictor.calls),20)
                self.assertFalse(e.t.read_json(folder/'comparison.json')['all_seeds_improve_both'])
                with (folder/'seed42/oof.csv').open('a') as out:out.write('tamper')
                with self.assertRaises(ValueError):e.compare(folder)

    def test_blend_common16_sse_and_gate(self):
        data,_,_,ref,plan=fixture();pred={'legacy14':data.y+2,'refined14':data.y+1}
        metrics,rows,diag,errors=e.result_data(data,ref,pred,plan['frozen']['common_failure_ids'])
        self.assertEqual(metrics[e.PRIMARY]['rmse'],2.5)
        self.assertEqual(metrics[e.PRIMARY]['gain_vs_current'],.5)
        self.assertEqual(metrics[e.PRIMARY]['gain_vs_control'],.5)
        groups={r['group']:r for r in diag if r['model']==e.PRIMARY}
        self.assertEqual(groups['common16']['n'],16)
        self.assertEqual(groups['common16']['sse']+groups['remaining484']['sse'],500*2.5**2)
        self.assertEqual(sum(r['gain_squared_error_vs_current'] for r in errors),500*(9-6.25))
        records=[{'seed':s,'gain_vs_current':.31,'gain_vs_control':.02} for s in e.SEEDS]
        self.assertTrue(e.aggregate(records)['passes_expansion_priority'])
        records[1]['gain_vs_control']=-.01
        self.assertFalse(e.aggregate(records)['all_seeds_improve_both'])

    def test_extraction_anchor_cache_and_source_corruption(self):
        _,_,extra,_,_=fixture();maps={k:np.zeros((256,256),bool) for k in f.MAP_KEYS}
        with temporary() as root:
            source=root/'open/train/TRAIN_000000.png';source.parent.mkdir(parents=True)
            Image.fromarray(np.full((256,256),128,np.uint8)).save(source)
            item={'ID':'TRAIN_000000','sha256':e.t.common.sha256_file(source)}
            anchors={item['ID']:{k:e.t.common.sha256_array(v) for k,v in maps.items()}}
            plan={'frozen':{'anchors':anchors}};calls=[]
            def fake(gray):
                calls.append(1);return {r:extra[r][0] for r in e.REPS},maps
            with patch.object(e,'ROOT',root):
                e.extraction_cell(root/'out',plan,item,fake)
                e.extraction_cell(root/'out',plan,item,e.forbidden)
                self.assertEqual(len(calls),1)
                wrong={'frozen':{'anchors':{item['ID']:{k:'wrong' for k in f.MAP_KEYS}}}}
                with self.assertRaises(ValueError):e.extraction_cell(root/'new',wrong,item,fake)
                cell=root/'out/features/cells/TRAIN_000000.json';saved=e.t.read_json(cell)
                saved['result']['features']['refined14'][0]=99;e.t.write_json(cell,saved)
                with self.assertRaises(ValueError):e.extraction_cell(root/'out',plan,item,fake)
                source.write_bytes(b'changed')
                with self.assertRaises(ValueError):e.extraction_cell(root/'else',plan,item,fake)


if __name__=='__main__':unittest.main()
