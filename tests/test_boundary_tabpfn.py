"""Synthetic/mocked predictive ablation tests; no competition fits or GPU."""
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from scripts import evaluate_boundary_tabpfn as b


@contextmanager
def temporary():
    parent=Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory(dir=parent,prefix='.boundary_test_') as td:
        if Path(td).resolve().parent!=parent:raise ValueError('Temporary cleanup path outside tests')
        yield td


def fixture():
    rng=np.random.default_rng(14);base=rng.normal(size=(500,285));base[:,0]=np.arange(500)
    added=np.zeros((500,37),dtype=np.float64);added[:,:7]=rng.uniform(1,20,(500,7))
    added[::3,0]=np.nan;added[:,4]=np.nan;added[:,30:]=np.isnan(added[:,:7]);added[:,25]=1.
    data=SimpleNamespace(ids=np.array([f'TRAIN_{i:06d}' for i in range(500)]),y=100+base[:,1]*3,noise=np.arange(500.))
    folds=np.repeat(np.arange(5),100)
    ref={'folds':folds,'base':data.y+4,'current':data.y+3,'tabpfn285':data.y+1,'control':data.y+2.5,'runtime':{'synthetic':True}}
    plan={'frozen':{'common_failure_ids':list(data.ids[:16]),'mock':True}}
    return data,base,added,ref,plan


class FakePredictor:
    calls=[]
    def __init__(self,*args):self.runtime={'synthetic':True}
    def __call__(self,x,y,q,seed):
        self.calls.append((x.copy(),y.copy(),q.copy(),seed))
        return 100+3*q[:,1]+.01*y.mean(),{'seconds':0.}


class BoundaryTabPFNTests(unittest.TestCase):
    def setUp(self):FakePredictor.calls=[]

    def test_real_synthetic_grid_noise_and_missing_schema(self):
        grid=np.full((256,256),180,np.uint8);grid[16::32,:]=40;grid[:,16::32]=40
        q,_=b.quality.assess(grid);row=b.vector(q)
        self.assertEqual(row.shape,(37,));self.assertAlmostEqual(row[0],32,delta=1)
        blank,_=b.quality.assess(np.full((256,256),128,np.uint8));missing=b.vector(blank)
        self.assertTrue(np.isnan(missing[:5]).all());self.assertTrue((missing[30:35]==1).all())
        self.assertIsNone(q['physical_grain_size']);self.assertFalse(any('grain_size' in n for n in b.NAMES))

    def test_context_only_medians_missing_flags_and_all_empty_column(self):
        data,base,added,_,_=fixture();x=np.c_[base,added]
        a,q,m=b.impute_context(x[:400],x[400:])
        changed=x[400:].copy();changed[:,285:292]=1e8;changed[:,315:322]=0
        _,_,m2=b.impute_context(x[:400],changed)
        self.assertEqual(m,m2);self.assertEqual(m['all_missing_context_columns'],[289])
        self.assertEqual(m['context_medians'][0],np.nanmedian(x[:400,285]))
        np.testing.assert_array_equal(a[:,:285],x[:400,:285]);np.testing.assert_array_equal(a[:,315:],x[:400,315:])
        self.assertTrue(np.isfinite(a).all() and np.isfinite(q).all());self.assertTrue((a[:,289]==0).all())
        invalid=x.copy();invalid[0,315]=1-invalid[0,315]
        with self.assertRaises(ValueError):b.impute_context(invalid[:400],invalid[400:])

    def test_outer_labels_isolated_and_completed_cell_resume(self):
        data,base,added,ref,_=fixture();x=np.c_[base,added]
        with temporary() as td:
            folder=Path(td);p=b.CandidatePredictor(FakePredictor())
            vi,a=b.evaluate_fold(folder/'a',{},x,data.y,ref['folds'],42,0,p)
            y2=data.y.copy();y2[:100]+=99999
            vj,c=b.evaluate_fold(folder/'c',{},x,y2,ref['folds'],42,0,p)
            np.testing.assert_array_equal(a,c);self.assertEqual(len(FakePredictor.calls),2)
            for fit_x,fit_y,query,seed in FakePredictor.calls:
                self.assertEqual((len(fit_x),len(query)),(400,100));self.assertEqual(seed,51)
                self.assertFalse(set(fit_x[:,0])&set(query[:,0]))
                np.testing.assert_array_equal(fit_y,data.y[fit_x[:,0].astype(int)])
            b.evaluate_fold(folder/'a',{},x,data.y,ref['folds'],42,0,p)
            self.assertEqual(len(FakePredictor.calls),2)
            with self.assertRaises(ValueError):b.evaluate_fold(folder/'a',{},x,data.y+1,ref['folds'],42,0,p)

    def test_exact_half_blend_and_control_are_distinct(self):
        data,_,added,ref,plan=fixture();pred=data.y+1
        metrics,rows,diag,errors=b.result_data(data,ref,added,pred,plan['frozen']['common_failure_ids'])
        self.assertAlmostEqual(metrics[b.PRIMARY]['rmse'],2.5)
        self.assertAlmostEqual(metrics[b.PRIMARY]['gain_vs_current'],.5)
        self.assertAlmostEqual(metrics[b.PRIMARY]['gain_vs_control'],0.)
        self.assertAlmostEqual(rows[0][b.PRIMARY],.5*(ref['base'][0]+pred[0]))
        self.assertAlmostEqual(errors[0]['gain_squared_error_vs_current'],9-6.25)
        self.assertTrue(any(r['group']=='historical_common_worst20' and r['n']==16 for r in diag))

    def test_two_seed_mock_evaluation_resume_aggregate_and_tamper(self):
        data,base,added,ref,plan=fixture()
        with temporary() as td:
            folder=Path(td);(folder/'features').mkdir();b.t.write_json(folder/'features/metadata.json',{'mock':True})
            with patch.object(b,'validate',return_value=(data,base,{s:ref for s in b.SEEDS},plan)), \
                 patch.object(b,'load_features',return_value=(added,{})),patch.object(b.t,'Predictor',FakePredictor), \
                 patch.object(b.t,'verify_weights',return_value=Path('mock.ckpt')):
                for seed in b.SEEDS:b.evaluate(folder,'cuda',seed)
                self.assertEqual(len(FakePredictor.calls),10)
                for seed in b.SEEDS:b.evaluate(folder,'cuda',seed)
                self.assertEqual(len(FakePredictor.calls),10)
                comparison=b.t.read_json(folder/'comparison.json');self.assertEqual(len(comparison['per_seed']),2)
                self.assertFalse(comparison['all_seeds_improve_both'])
                self.assertGreater(comparison['mean_gain_vs_current'],0)
                self.assertLess(comparison['mean_gain_vs_control'],0)
                with (folder/'seed42/oof.csv').open('a') as f:f.write('corruption')
                with self.assertRaises(ValueError):b.compare(folder)

    def test_extraction_cell_resume_anchor_and_image_change(self):
        from PIL import Image
        gray=np.full((256,256),180,np.uint8);gray[16::32,:]=40;gray[:,16::32]=40
        q,m=b.quality.assess(gray)
        with temporary() as td:
            root=Path(td);source=root/'open/train/TRAIN_000001.png';source.parent.mkdir(parents=True);Image.fromarray(gray).save(source)
            image={'ID':'TRAIN_000001','sha256':b.t.common.sha256_file(source)}
            plan={'frozen':{'quality_anchor_hashes':{'TRAIN_000001':b.t.digest(q)}}}
            with patch.object(b,'ROOT',root),patch.object(b.quality,'assess',return_value=(q,m)) as assessor:
                a,s=b.extraction_cell(root/'out',plan,image);c,_=b.extraction_cell(root/'out',plan,image)
                self.assertEqual(assessor.call_count,1);np.testing.assert_array_equal(a,c)
                wrong={'frozen':{'quality_anchor_hashes':{'TRAIN_000001':'wrong'}}}
                with self.assertRaises(ValueError):b.extraction_cell(root/'other',wrong,image)
                source.write_bytes(b'changed')
                with self.assertRaises(ValueError):b.extraction_cell(root/'third',plan,image)

    def test_gate_requires_both_comparators_both_seeds(self):
        rows=[{'seed':s,'gain_vs_current':.2,'gain_vs_control':.01} for s in b.SEEDS]
        self.assertTrue(b.aggregate(rows)['meets_priority_rule'])
        rows[1]['gain_vs_control']=-.01
        self.assertFalse(b.aggregate(rows)['all_seeds_improve_both'])
        with self.assertRaises(ValueError):b.aggregate(rows[:1])


if __name__=='__main__':unittest.main()
