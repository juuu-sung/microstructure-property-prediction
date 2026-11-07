"""Synthetic CPU checks; never fit on competition rows or open test images."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
from scripts import extract_morphology as x
from scripts import evaluate_morphology as e


class MeanModel:
    def __init__(self,records,family,config,seed):
        self.records,self.family,self.config,self.seed=records,family,config,seed
    def fit(self,a,y):
        self.mean=float(np.mean(y));self.records.append((self.family,dict(self.config),a.shape,len(y),self.seed))
        return self
    def predict(self,a):return np.full(len(a),self.mean)


class MorphologyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng=np.random.default_rng(87)
        cls.y=rng.normal(200,15,50)
        old=rng.normal(size=(50,185));added=rng.normal(size=(50,100))
        cls.features={e.EXISTING:old,e.AUGMENTED:np.c_[old,added]}
        cls.folds=np.arange(50)%5

    def test_feature_schema_finite_constant_and_no_mutation(self):
        gray=np.full((256,256),127,dtype=np.uint8);before=gray.copy()
        result,detail=x.morphology_features(gray,return_maps=True)
        self.assertEqual(result.shape,(100,));self.assertEqual(result.dtype,np.float64)
        self.assertTrue(np.isfinite(result).all());self.assertEqual(len(set(x.feature_names())),100)
        self.assertEqual([q['regions'] for q in detail['qc']],[1,1])
        np.testing.assert_array_equal(gray,before)
        with self.assertRaises(ValueError):x.morphology_features(gray.astype(float))

    def test_partition_grid_separates_all_64_cell_interiors(self):
        gray=np.full((256,256),180,dtype=np.uint8)
        gray[::32,:]=40;gray[:,::32]=40
        f,m=x.morphology_features(gray,return_maps=True)
        self.assertTrue(np.isfinite(f).all())
        for labels in m['regions']:
            # Thin dark lines can themselves produce boundary-strip regions.
            # Test separation of known cell interiors, not a physical grain-count claim.
            self.assertEqual(len(np.unique(labels[16::32,16::32])),64)
            for row in range(8):
                for col in range(8):
                    self.assertEqual(len(np.unique(labels[row*32+10:row*32+22,col*32+10:col*32+22])),1)
            self.assertTrue(np.all(labels>0));self.assertEqual(labels.shape,gray.shape)

    def test_single_pixel_minima_are_retained_as_markers(self):
        a=np.tile(np.linspace(0,1,256),(256,1))
        minima=np.zeros((256,256),bool)
        minima[40,40]=minima[120,120]=minima[200,200]=True
        with mock.patch.object(x.morphology,'h_minima',return_value=minima):
            labels,_,count=x.region_map(a,1.)
        self.assertEqual(count,3);self.assertEqual(len(np.unique(labels)),3)
        self.assertEqual(len(np.unique(labels[minima])),3)

    def test_empty_dark_region_and_component_geometry(self):
        values,mask=x.dark_features(np.ones((256,256)),.25)
        self.assertFalse(mask.any());np.testing.assert_array_equal(values,np.zeros(20))
        a=np.ones((256,256));a[40:70,45:75]=0;a[120:155,150:165]=0
        values,mask=x.dark_features(a,.25)
        self.assertGreater(values[0],0);self.assertAlmostEqual(values[1],np.log1p(2))
        self.assertTrue(np.isfinite(values).all())

    def test_preview_selection_independent_of_labels(self):
        rows=[{'ID':f'TRAIN_{i:06d}','haar_hh_noise_proxy':i%17,'variance_minus_noise_proxy':(i*7)%41,'hardness':i} for i in range(100)]
        before=x.preview_selection(rows)
        for r in rows:r['hardness']='forbidden target value'
        self.assertEqual(before,x.preview_selection(rows));self.assertEqual(len({r['ID'] for r in before}),25)
        self.assertTrue(all('hardness' not in r for r in before))

    def test_nested_budget_grids_sizes_and_ties(self):
        records=[]
        factory=lambda family,cfg,seed,threads:MeanModel(records,family,cfg,seed)
        with contextlib.redirect_stdout(io.StringIO()):
            p,chosen,search=e.nested(self.features,self.y,self.folds,42,2,factory=factory)
        self.assertEqual(len(records),230)
        self.assertEqual(sum(r[0]=='ridge' for r in records),100)
        self.assertEqual(sum(r[0]=='catboost' for r in records),130)
        self.assertEqual(set(r[3] for r in records),{26,27,40})
        self.assertEqual(set(r[2][1] for r in records),{185,285})
        self.assertEqual(len(search),70);self.assertEqual(set(p),set(e.NAMES))
        for record in chosen:
            for name,choice in record['selected'].items():
                expected={'alpha':1000.} if name.startswith('ridge') else {'depth':3,'l2_leaf_reg':10.}
                self.assertEqual(choice['config'],expected)
                np.testing.assert_allclose(choice['inner_mse'],np.average(choice['inner_fold_mse'],weights=record['inner_validation_sizes']),atol=1e-10)

    def test_outer_valid_labels_cannot_affect_same_fold_predictions(self):
        factory=lambda family,cfg,seed,threads:MeanModel([],family,cfg,seed)
        changed=self.y.copy();changed[self.folds==0]+=10000
        with contextlib.redirect_stdout(io.StringIO()):
            p,c,_=e.nested(self.features,self.y,self.folds,42,2,factory=factory)
            p2,c2,_=e.nested(self.features,changed,self.folds,42,2,factory=factory)
        for name in e.NAMES:np.testing.assert_array_equal(p[name][self.folds==0],p2[name][self.folds==0])
        self.assertEqual(c[0],c2[0])

    def test_real_cpu_catboost_and_ridge_predict_unseen_synthetic_rows(self):
        for family,config in [('ridge',{'alpha':100.}),('catboost',{'depth':3,'l2_leaf_reg':10.})]:
            model=e.make_model(family,config,42,2)
            model.fit(self.features[e.EXISTING][:40,:8],self.y[:40])
            p=model.predict(self.features[e.EXISTING][40:,:8])
            self.assertEqual(np.asarray(p).shape,(10,));self.assertTrue(np.isfinite(p).all())
            if family=='catboost':
                self.assertEqual(model.tree_count_,600)
                self.assertFalse(model.get_params()['allow_writing_files'])

    def test_anchor_and_fixed_blends(self):
        pred={n:self.y+j for j,n in enumerate(e.NAMES)}
        refs={'hand':pred['ridge_existing'],'resnet':self.y+5,'dino':self.y-3}
        refs['current']=.25*refs['hand']+.25*refs['resnet']+.5*refs['dino']
        p,d=e.add_blends(pred,refs);self.assertEqual(d,0)
        np.testing.assert_array_equal(p[e.PRIMARY],.25*pred['catboost_morphology']+.25*refs['resnet']+.5*refs['dino'])
        bad=copy.deepcopy(refs);bad['hand']+=1
        with self.assertRaises(ValueError):e.add_blends(pred,bad)

    def test_cache_roundtrip_tamper_and_id_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp);plan_path=path/'protocol.json';plan_path.write_text('{}',encoding='utf-8')
            ids=np.array([f'TRAIN_{i:06d}' for i in range(50)])
            data=SimpleNamespace(ids=ids,features={e.confirmation.AUGMENTED:self.features[e.EXISTING]})
            plan={'frozen':{'train_binding':{'synthetic':True}}}
            arrays={'ids':ids,'morphology':self.features[e.AUGMENTED][:,185:].copy(),'feature_names':np.asarray(x.feature_names())}
            meta={'protocol':x.PROTOCOL,'sources':x.source_refs(),'versions':x.versions(),'plan_sha256':e.core.sha256(plan_path),
                'train_binding':plan['frozen']['train_binding'],'test_accessed':False,'labels_saved':False,
                'array_sha256':{k:x.common.sha256_array(v) for k,v in arrays.items()}}
            def save():
                np.savez_compressed(path/'features.npz',**arrays,metadata_json=np.asarray(x.common.canonical_json(meta)))
                x.write_json(path/'metadata.json',meta)
            save();loaded,_=e.load_morphology(path/'features.npz',data,plan,plan_path)
            np.testing.assert_array_equal(loaded[e.AUGMENTED],self.features[e.AUGMENTED])
            arrays['morphology'][0,0]+=1;save()
            with self.assertRaises(ValueError):e.load_morphology(path/'features.npz',data,plan,plan_path)
            arrays['morphology'][0,0]-=1
            arrays['ids']=ids[::-1];meta['array_sha256']['ids']=x.common.sha256_array(arrays['ids']);save()
            with self.assertRaises(ValueError):e.load_morphology(path/'features.npz',data,plan,plan_path)

    def test_synthetic_output_workflow_and_summary_without_real_fit(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);(folder/'features').mkdir();(folder/'protocol.json').write_text('{}')
            (folder/'features/features.npz').write_bytes(b'synthetic')
            ids=np.array([f'TRAIN_{i:06d}' for i in range(50)])
            data=SimpleNamespace(ids=ids,y=self.y,noise=np.arange(50))
            factory=lambda family,cfg,seed,threads:MeanModel([],family,cfg,seed)
            actual_nested=e.nested
            with contextlib.redirect_stdout(io.StringIO()):pred,_,_=actual_nested(self.features,self.y,self.folds,42,2,factory=factory)
            refs={'folds':self.folds,'hand':pred['ridge_existing'],'resnet':self.y,'dino':self.y}
            refs['current']=.25*refs['hand']+.75*self.y
            with mock.patch.object(e,'load_morphology',return_value=(self.features,{})),mock.patch.object(e,'load_references',return_value=refs),\
                mock.patch.object(e,'validate_plan',return_value={}),mock.patch.object(e,'nested',side_effect=lambda *a:actual_nested(*a,factory=factory)),\
                contextlib.redirect_stdout(io.StringIO()):
                for seed in e.SEEDS:e.run(folder,data,{},seed,2)
                result=e.summarize(folder)
                with self.assertRaises(ValueError):e.run(folder,data,{},42,2)
            self.assertEqual(len(result['per_seed']),2)
            self.assertTrue((folder/'seed42/group_metrics.csv').is_file())

    def test_aggregate_requires_both_fixed_seeds_and_priority(self):
        def r(seed,gain):
            scores={'current_blend':14.,e.PRIMARY:14.-gain,'catboost_existing_blend':14.1,'ridge_existing_blend':14.,'ridge_morphology_blend':13.9}
            return {'seed':seed,'metrics':[{'model':n,'rmse':v} for n,v in scores.items()]}
        self.assertTrue(e.aggregate([r(42,.4),r(2026,.5)])['meets_priority_rule'])
        self.assertFalse(e.aggregate([r(42,.05),r(2026,.06)])['meets_priority_rule'])
        with self.assertRaises(ValueError):e.aggregate([r(42,.5)])

    def test_readonly_prepare_validate_do_not_fit_or_extract(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)/'plan'
            with mock.patch.object(e,'payload',return_value={'test':'synthetic'}),\
                mock.patch.object(e,'make_model',side_effect=AssertionError('No fitting')),\
                mock.patch.object(x,'morphology_features',side_effect=AssertionError('No extraction')):
                e.prepare(folder,None);before=(folder/'protocol.json').read_bytes()
                e.validate_plan(folder,None)
                self.assertEqual(before,(folder/'protocol.json').read_bytes())
                with self.assertRaises(ValueError):e.prepare(folder,None)
                with mock.patch.object(e,'payload',return_value={'changed':True}):
                    with self.assertRaises(ValueError):e.validate_plan(folder,None)


if __name__=='__main__':unittest.main()
