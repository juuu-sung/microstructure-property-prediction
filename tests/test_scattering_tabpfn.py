"""Synthetic/mocked numerical, validation-isolation, cache and ablation checks."""
from contextlib import contextmanager
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
from scripts import evaluate_scattering_tabpfn as b
from scripts import scattering_features as sf


@contextmanager
def temporary():
    parent=Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory(dir=parent,prefix='.scattering_test_') as td:
        if Path(td).resolve().parent != parent:
            raise ValueError('Temporary cleanup path outside tests')
        yield Path(td)


def fixture():
    rng=np.random.default_rng(12);base=rng.normal(size=(500,285));base[:,0]=np.arange(500)
    added=rng.uniform(size=(500,201)).astype(np.float64)
    data=SimpleNamespace(ids=np.array([f'TRAIN_{i:06d}' for i in range(500)]),
                         y=100+base[:,1]*3,noise=np.arange(500.))
    ref={'folds':np.repeat(np.arange(5),100),'base':data.y+4,'current':data.y+3,
         'tabpfn285':data.y+1,'control':data.y+2.5,'runtime':{'synthetic':True}}
    return data,base,added,ref,{'frozen':{'common_failure_ids':list(data.ids[:16]),'mock':True}}


class FakePredictor:
    calls=[]
    def __init__(self,*args): self.runtime={'synthetic':True}
    def __call__(self,x,y,q,seed):
        self.calls.append((x.copy(),y.copy(),q.copy(),seed))
        return 100+3*q[:,1]+.01*y.mean()+.2,{'seconds':0.}


class ScatteringTests(unittest.TestCase):
    def setUp(self): FakePredictor.calls=[]

    def test_metadata_mapping_and_normalization_math_independent_of_order(self):
        paths=[{'j':(),'theta':(),'coef':np.full((8,8),.5)}]
        for j in range(5):
            for theta in range(8):
                paths.append({'j':(j,),'theta':(theta,),'coef':np.full((8,8),j+theta+1.)})
        for j1,j2 in sf.PAIRS:
            for u in range(8):
                for v in range(8):
                    value=j1*1000+j2*100+(u+1.)*(v+1.)
                    paths.append({'j':(j1,j2),'theta':(u,v),'coef':np.full((8,8),value)})
        row,detail=sf.summarize(paths)
        np.testing.assert_array_equal(row,sf.summarize(paths[::-1])[0])
        self.assertEqual(len(row),201);self.assertEqual(detail['path_count'],681)
        self.assertEqual(row[0],.5);self.assertEqual(row[1],1.);self.assertEqual(row[40],12.)
        for k,(a,c) in enumerate(sf.PAIRS):
            for d in range(8):
                mean=np.mean([a*1000+c*100+(u+1.)*((u+d)%8+1.) for u in range(8)])
                self.assertEqual(row[41+8*k+d],mean)
                self.assertAlmostEqual(row[121+8*k+d],mean/(a+4.5+1e-6))
        with self.assertRaises(ValueError): sf.summarize(paths[:-1])
        with self.assertRaises(ValueError): sf.summarize(paths+[paths[0]])

    def test_real_library_synthetic_blank_repeat_perturbation_and_environment(self):
        before=b.t.environment()
        report=sf.synthetic_check()
        self.assertEqual(report['competition_images_used'],0)
        self.assertTrue(report['exact_repeat']);self.assertEqual(report['features'],201)
        self.assertEqual(before,b.t.environment())
        self.assertNotIn('kymatio',before)
        self.assertTrue(all(np.isfinite(v) for d in report['perturbations_relative_l2'].values() for v in d.values()))

    def test_real_library_retains_absolute_orientation_and_rejects_resize(self):
        extractor=sf.Extractor()
        vertical=np.full((256,256),180,np.uint8);vertical[:,16::32]=40
        a,_=extractor(vertical);c,_=extractor(vertical.T.copy())
        self.assertGreater(np.linalg.norm(a[1:41]-c[1:41]),.001)
        with self.assertRaises(ValueError):extractor(vertical[:128])
        with self.assertRaises(ValueError):extractor(vertical.astype(float))

    def test_outer_validation_labels_never_passed_and_resume(self):
        data,base,added,ref,_=fixture();x=np.c_[base,added]
        with temporary() as folder:
            vi,a=b.evaluate_fold(folder/'a',{},x,data.y,ref['folds'],42,0,FakePredictor())
            changed=data.y.copy();changed[:100]+=99999
            _,c=b.evaluate_fold(folder/'b',{},x,changed,ref['folds'],42,0,FakePredictor())
            np.testing.assert_array_equal(a,c)
            for tx,ty,q,seed in FakePredictor.calls:
                self.assertEqual((tx.shape,q.shape),((400,486),(100,486)))
                self.assertEqual(seed,51);self.assertFalse(set(tx[:,0])&set(q[:,0]))
                np.testing.assert_array_equal(ty,data.y[tx[:,0].astype(int)])
                np.testing.assert_array_equal(tx[:,:285],base[tx[:,0].astype(int)])
            b.evaluate_fold(folder/'a',{},x,data.y,ref['folds'],42,0,FakePredictor())
            self.assertEqual(len(FakePredictor.calls),2)
            with self.assertRaises(ValueError):b.evaluate_fold(folder/'a',{},x,data.y+1,ref['folds'],42,0,FakePredictor())

    def test_exact_blend_sse_partition_and_both_comparators(self):
        data,_,_,ref,plan=fixture()
        metrics,rows,diag,errors=b.result_data(data,ref,data.y+1,plan['frozen']['common_failure_ids'])
        self.assertAlmostEqual(metrics[b.PRIMARY]['rmse'],2.5)
        self.assertAlmostEqual(metrics[b.PRIMARY]['gain_vs_current'],.5)
        self.assertAlmostEqual(metrics[b.PRIMARY]['gain_vs_control'],0.)
        self.assertAlmostEqual(rows[0][b.PRIMARY],.5*(ref['base'][0]+data.y[0]+1))
        groups={r['group']:r for r in diag if r['model']==b.PRIMARY}
        self.assertEqual(groups['common16']['n'],16);self.assertEqual(groups['remaining484']['n'],484)
        self.assertAlmostEqual(groups['common16']['sse']+groups['remaining484']['sse'],500*2.5**2)
        self.assertAlmostEqual(sum(r['gain_squared_error_vs_current'] for r in errors),500*(9-6.25))

    def test_two_seed_mock_end_to_end_resume_and_tamper(self):
        data,base,added,ref,plan=fixture()
        with temporary() as folder:
            b.t.write_json(folder/'features/metadata.json',{'mock':True})
            with patch.object(b,'validate',return_value=(data,base,{s:ref for s in b.SEEDS},plan)), \
                 patch.object(b,'load_features',return_value=(added,{})), \
                 patch.object(b.t,'Predictor',FakePredictor),patch.object(b.t,'verify_weights',return_value=Path('mock.ckpt')):
                for s in b.SEEDS:b.evaluate(folder,'cuda',s)
                self.assertEqual(len(FakePredictor.calls),10)
                for s in b.SEEDS:b.evaluate(folder,'cuda',s)
                self.assertEqual(len(FakePredictor.calls),10)
                result=b.t.read_json(folder/'comparison.json')
                self.assertFalse(result['all_seeds_improve_both'])
                self.assertGreater(result['mean_gain_vs_current'],0)
                self.assertLess(result['mean_gain_vs_control'],0)
                with (folder/'seed42/oof.csv').open('a') as f:f.write('tamper')
                with self.assertRaises(ValueError):b.compare(folder)

    def test_extraction_cell_source_binding_resume_and_corruption(self):
        with temporary() as root:
            source=root/'open/train/TRAIN_000001.png';source.parent.mkdir(parents=True)
            Image.fromarray(np.full((256,256),128,np.uint8)).save(source)
            item={'ID':'TRAIN_000001','sha256':b.t.common.sha256_file(source)}
            plan={'frozen':{'mock':True}};calls=[]
            def fake(image):
                calls.append(image)
                return np.arange(201,dtype=np.float64),{'synthetic':True}
            with patch.object(b,'ROOT',root):
                a=b.extraction_cell(root/'out',plan,item,fake)
                c=b.extraction_cell(root/'out',plan,item,fake)
                np.testing.assert_array_equal(a,c);self.assertEqual(len(calls),1)
                with self.assertRaises(ValueError):b.extraction_cell(root/'out',{'frozen':{'changed':True}},item,fake)
                cell=root/'out/features/cells/TRAIN_000001.json';saved=b.t.read_json(cell)
                saved['result']['features'][0]+=1;b.t.write_json(cell,saved)
                with self.assertRaises(ValueError):b.extraction_cell(root/'out',plan,item,fake)
                source.write_bytes(b'changed')
                with self.assertRaises(ValueError):b.extraction_cell(root/'new',plan,item,fake)

    def test_gate_requires_two_positive_controls_and_point30(self):
        rows=[{'seed':s,'gain_vs_current':.31,'gain_vs_control':.1} for s in b.SEEDS]
        self.assertTrue(b.aggregate(rows)['passes_expansion_priority'])
        rows[1]['gain_vs_current']=.01
        self.assertFalse(b.aggregate(rows)['passes_expansion_priority'])
        rows[1]['gain_vs_control']=-.01
        self.assertFalse(b.aggregate(rows)['all_seeds_improve_both'])
        with self.assertRaises(ValueError):b.aggregate(rows[:1])


if __name__=='__main__':unittest.main()
