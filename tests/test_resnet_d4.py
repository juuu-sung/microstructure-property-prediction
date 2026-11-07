"""Synthetic CPU-only tests for fixed ResNet D4 replacement; no real fitting."""
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch
from scripts import extract_resnet_d4 as extract
from scripts import evaluate_resnet_d4 as evaluate

core=evaluate.core


class PoolTests(unittest.TestCase):
    def test_per_view_multiscale_pool_then_mean_is_d4_invariant(self):
        torch.set_num_threads(1)
        class Fake:
            def __call__(self,x):
                return [x[:,0:1,:n,:n].expand(-1,c,-1,-1).contiguous() for n,c in ((32,128),(16,256),(8,512))]
        x=torch.randn(2,3,256,256); model=Fake()
        identity,averaged=extract.extract_batch(torch,model,x)
        expected=torch.stack([extract.base.pool_feature_maps(torch,model(v))[1] for v in extract.geometry.d4_views(torch,x)]).double().mean(0).float()
        torch.testing.assert_close(averaged,expected,atol=0,rtol=0)
        torch.testing.assert_close(identity,extract.base.pool_feature_maps(torch,model(x))[1],atol=0,rtol=0)
        self.assertFalse(torch.allclose(identity,averaged))
        for view in extract.geometry.d4_views(torch,x):
            torch.testing.assert_close(extract.extract_batch(torch,model,view)[1],averaged,atol=1e-6,rtol=1e-6)
        extract.validate_values(np,averaged.numpy(),2)
        for bad in (torch.zeros(1,3,224,224),torch.zeros(1,3,256,256,dtype=torch.float64)):
            with self.assertRaises(ValueError): extract.extract_batch(torch,model,bad)


class CacheTests(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory(prefix="resnet-d4-synthetic-"); self.addCleanup(tmp.cleanup)
        self.root=Path(tmp.name); self.ids=np.array([f"TRAIN_{i+1:06d}" for i in range(50)])
        self.y=np.arange(50,dtype=float)+180; self.train=self.root/"train.csv"
        self.train.write_text(core.csv_text([{"ID":i,"hardness":y} for i,y in zip(self.ids,self.y)]),encoding="utf-8")
        rng=np.random.default_rng(26); self.raw=rng.normal(size=(50,1792)).astype(np.float32)
        for a,b in ((128,256),(512,768),(1280,1792)): self.raw[:,a:b]=np.abs(self.raw[:,a:b])
        manifest=[{"id":i,"relative_path":f"train/{i}.png","size_bytes":1,"sha256":"a"*64} for i in self.ids]
        self.metadata={"schema_version":1,"rows":50,"extraction_config":copy.deepcopy(extract.ORIGINAL_CONFIG),
            "test_accessed":False,"labels_saved_in_npz":False,"train_csv_sha256":core.sha256(self.train),
            "feature_state_sha256":"b"*64,"training_image_manifest":manifest,
            "image_manifest_sha256":hashlib.sha256(extract.base.canonical_json(manifest).encode()).hexdigest(),
            "feature_array_sha256":{"final_gap":extract.base.sha256_array(self.raw[:,768:1280]),extract.RAW:extract.base.sha256_array(self.raw)}}
        self.original=self.root/"original/features.npz"
        extract.base.atomic_save(np,self.original.parent,self.ids,self.raw[:,768:1280],self.raw,self.metadata)
        self.values=self.raw*0.9; self.path=self.root/"d4/features.npz"
        self.d4meta={"schema_version":1,"rows":50,"protocol":copy.deepcopy(extract.PROTOCOL),
            "extraction_config":{**extract.ORIGINAL_CONFIG,"augmentation":extract.PROTOCOL,"pooling":extract.PROTOCOL["pool_each_view"]+"; D4 mean"},
            "original_reference":extract.reference(self.original),"source_sha256":extract.source_hashes(),
            **{k:self.metadata[k] for k in ("feature_state_sha256","train_csv_sha256","image_manifest_sha256")},
            "feature_array_sha256":{extract.FEATURE:extract.base.sha256_array(self.values)},
            "identity_reextraction_check":{"passed":True,"atol":extract.MATCH_ATOL,"rtol":extract.MATCH_RTOL,"max_absolute_difference":0.},
            "test_accessed":False,"labels_saved_in_npz":False,"offline_checkpoint_only":True}
        self.save()
        self.data=core.Inputs(self.ids,self.y,{extract.RAW:self.raw},np.arange(50)%5,self.y+1,np.arange(50),
            {"resnet_original":self.metadata},{})

    def save(self,values=None,ids=None):
        extract.atomic_save(np,self.path.parent,self.ids if ids is None else ids,self.values if values is None else values,self.d4meta)

    def test_original_and_d4_cache_alignment_and_roundtrip(self):
        ids,values,metadata=extract.read_original(np,self.original,self.train,expected_n=50)
        np.testing.assert_array_equal(ids,self.ids); np.testing.assert_array_equal(values,self.raw)
        reversed_values=self.values[::-1].copy()
        self.d4meta["feature_array_sha256"]={extract.FEATURE:extract.base.sha256_array(reversed_values)}
        self.save(reversed_values,self.ids[::-1])
        attached=evaluate.attach_features(self.data,self.path,self.original)
        np.testing.assert_array_equal(attached.features[extract.FEATURE],self.values)
        np.testing.assert_array_equal(attached.features[extract.RAW],self.raw)

    def test_d4_cache_protocol_sources_identity_and_array_tampering_rejected(self):
        original=copy.deepcopy(self.d4meta)
        for change in (lambda m:m["protocol"].update(padding=5),lambda m:m["source_sha256"].update(extractor="c"*64),
            lambda m:m["identity_reextraction_check"].update(passed=False),lambda m:m.update(test_accessed=True),
            lambda m:m["original_reference"].update(sha256="c"*64),lambda m:m.update(feature_state_sha256="c"*64)):
            self.d4meta=copy.deepcopy(original); change(self.d4meta); self.save()
            with self.assertRaises(ValueError): extract.validate_cache(np,self.path,self.ids,self.metadata,self.original)
        self.d4meta=original; altered=self.values.copy(); altered[0,0]+=1; self.save(altered)
        with self.assertRaisesRegex(ValueError,"hash"): extract.validate_cache(np,self.path,self.ids,self.metadata,self.original)

    def test_original_cache_corruption_rejected(self):
        bad=copy.deepcopy(self.metadata); bad["extraction_config"]["resize"]=224
        extract.base.atomic_save(np,self.original.parent,self.ids,self.raw[:,768:1280],self.raw,bad)
        with self.assertRaisesRegex(ValueError,"configuration"): extract.read_original(np,self.original,self.train,expected_n=50)
        extract.base.atomic_save(np,self.original.parent,self.ids,self.raw[:,768:1280]+1,self.raw,self.metadata)
        with self.assertRaisesRegex(ValueError,"GAP"): extract.read_original(np,self.original,self.train,expected_n=50)

    def test_read_only_modes_fit_nothing_and_write_nothing(self):
        refs={evaluate.REFERENCE:self.y+2}
        args=["--original-features",str(self.original),"--resnet-d4-features",str(self.path),"--output-dir",str(self.root/"out")]
        before={str(p):core.sha256(p) for p in self.root.rglob("*") if p.is_file()}
        for mode in ("--preflight","--validate-only"):
            with mock.patch.object(evaluate,"load_prior",return_value=(copy.deepcopy(self.data),refs,{})), \
                 mock.patch.object(core,"make_estimator") as fit, mock.patch.object(evaluate,"nested_predictions") as nested, \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(evaluate.main(args+[mode]),0); fit.assert_not_called(); nested.assert_not_called()
        self.assertFalse((self.root/"out").exists())
        self.assertEqual(before,{str(p):core.sha256(p) for p in self.root.rglob("*") if p.is_file()})

    def test_output_protection_before_loading(self):
        with mock.patch.object(evaluate,"load_prior") as loader:
            with self.assertRaisesRegex(ValueError,"protected"):
                evaluate.main(["--original-features",str(self.original),"--output-dir",str(self.original.parent),"--overwrite"])
            loader.assert_not_called()
        with mock.patch.object(extract,"read_original") as loader:
            with self.assertRaisesRegex(ValueError,"protected"):
                extract.main(["--original-features",str(self.original),"--output-dir",str(self.original.parent),"--overwrite"])
            with self.assertRaisesRegex(ValueError,"already exists"):
                extract.main(["--output-dir",str(self.path.parent)])
            loader.assert_not_called()


class NestedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng=np.random.default_rng(45); x=rng.normal(size=(50,8)); z=rng.normal(size=(50,8))
        y=200+5*x[:,0]+2*z[:,0]+rng.normal(size=50)
        cls.data=core.Inputs(np.array([f"TRAIN_{i+1:06d}" for i in range(50)]),y,{evaluate.RAW:x,evaluate.D4:z},np.arange(50)%5,y+1,np.arange(50),{"resnet_original":{}},{})
        with contextlib.redirect_stdout(io.StringIO()):
            cls.predictions,cls.selections=evaluate.nested_predictions(cls.data,42,1)
            cls.raw_old,cls.raw_records=core.nested_predictions(cls.data,42,1,alphas=evaluate.ALPHAS,representations=(evaluate.RAW,))
        cls.anchor={"predictions":cls.raw_old[core.PRIMARY_NAME],"prediction_column":evaluate.RAW+"_nested","selections":{
            r["outer_fold"]:{"candidates":r["candidates"],"winner":r["per_representation"][evaluate.RAW],
            "inner_validation_sizes":r["inner_validation_sizes"]} for r in cls.raw_records}}

    def references(self):
        hand=self.data.baseline; resnet=self.anchor["predictions"]; dino=self.data.y+3; d4=self.data.y+2
        current=0.25*hand+0.25*resnet+0.5*dino
        return {evaluate.previous.CURRENT:current,evaluate.previous.DINO:dino,
            evaluate.previous.prior.refs_module.HAND_DINO_REFERENCE:0.5*(hand+dino),
            **{n:current+cap/10000 for cap,n in evaluate.previous.PARTIAL_REFERENCES.items()},
            evaluate.RAW_REFERENCE:resnet,"reference_dino_d4":d4,evaluate.REFERENCE:0.25*hand+0.25*resnet+0.5*d4}

    def test_raw_anchor_exact_and_no_representation_gate(self):
        check=evaluate.verify_anchor(self.data,self.predictions,self.selections,self.anchor)
        self.assertEqual(check["max_absolute_prediction_difference"],0)
        self.assertEqual(check["max_absolute_inner_mse_difference"],0)
        self.assertNotIn(core.PRIMARY_NAME,self.predictions)
        self.assertTrue(all("joint" not in r for r in self.selections))
        bad=copy.deepcopy(self.anchor); bad["predictions"][0]+=1
        with self.assertRaises(ValueError): evaluate.verify_anchor(self.data,self.predictions,self.selections,bad)

    def test_outer_labels_do_not_change_their_predictions_or_alpha(self):
        data=copy.deepcopy(self.data); data.y[data.folds==0]+=100
        with contextlib.redirect_stdout(io.StringIO()): pred,sel=evaluate.nested_predictions(data,42,1)
        self.assertEqual(sel[0],self.selections[0])
        for name in pred:
            np.testing.assert_array_equal(pred[name][data.folds==0],self.predictions[name][data.folds==0])

    def test_only_quarter_resnet_changes_and_dino_constant(self):
        refs=self.references(); result=evaluate.add_fixed_blend(self.predictions,refs)
        expected=0.25*self.data.baseline+0.25*self.predictions[evaluate.D4+"_nested"]+0.5*refs["reference_dino_d4"]
        np.testing.assert_allclose(result[evaluate.PRIMARY],expected,atol=1e-12,rtol=0)
        fallback={**self.predictions,evaluate.D4+"_nested":refs[evaluate.RAW_REFERENCE]}
        np.testing.assert_array_equal(evaluate.add_fixed_blend(fallback,refs)[evaluate.PRIMARY],refs[evaluate.REFERENCE])
        changed={**refs,"reference_dino_d4":refs["reference_dino_d4"]+100}
        # Fixed reference is the only DINO-bearing term used for replacement;
        # the evaluator validates its formula before fitting.
        np.testing.assert_array_equal(evaluate.add_fixed_blend(self.predictions,changed)[evaluate.PRIMARY],result[evaluate.PRIMARY])

    def test_report_and_diagnostics_use_new_reference_not_old_baseline(self):
        predictions=evaluate.add_fixed_blend(self.predictions,self.references())
        diagnostics,tables=evaluate.diagnose(self.data,predictions)
        ref=diagnostics[evaluate.REFERENCE]["overall"]["rmse"]
        self.assertAlmostEqual(diagnostics[evaluate.PRIMARY]["overall"]["delta_rmse_vs_dino_d4_fixed_blend"],
            diagnostics[evaluate.PRIMARY]["overall"]["rmse"]-ref)
        for row in tables["fold_metrics"]:
            mask=self.data.folds==row["fold"]
            self.assertAlmostEqual(row["delta_rmse_vs_dino_d4_fixed_blend"],row["rmse"]-core.metrics(self.data.y[mask],predictions[evaluate.REFERENCE][mask])["rmse"])
        deciles=core.rank_groups(self.data.y,10)
        regions={"low10":deciles==0,"high10":deciles==9,"tails20":(deciles==0)|(deciles==9),"middle80":(deciles>0)&(deciles<9)}
        for row in tables["target_region_metrics"]:
            mask=regions[row["region"]]
            self.assertAlmostEqual(row["delta_rmse_vs_dino_d4_fixed_blend"],row["rmse"]-core.metrics(self.data.y[mask],predictions[evaluate.REFERENCE][mask])["rmse"])
        check=evaluate.verify_anchor(self.data,self.predictions,self.selections,self.anchor)
        with tempfile.TemporaryDirectory() as td:
            evaluate.write_outputs(Path(td),self.data,predictions,self.selections,diagnostics,tables,42,0.,check,threads=1)
            summary=json.loads((Path(td)/"summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["primary_model"],evaluate.PRIMARY)
            self.assertFalse(summary["representation_selection"])
            self.assertEqual(summary["estimator_fit_count"],100)
            self.assertTrue((Path(td)/"resnet_replacement_errors.csv").is_file())

    def test_saved_anchor_references_and_fold_corruption(self):
        with tempfile.TemporaryDirectory() as td:
            folder=Path(td); path=folder/"oof.csv"; summary_path=folder/"summary.json"
            rows=[{"ID":i,"hardness":self.data.y[j],"fold":int(self.data.folds[j]),core.BASELINE_NAME:self.data.baseline[j],
                evaluate.RAW+"_nested":self.anchor["predictions"][j],core.PRIMARY_NAME:self.anchor["predictions"][j]} for j,i in enumerate(self.data.ids)]
            path.write_text(core.csv_text(rows),encoding="utf-8")
            summary={"seed":42,"alpha_candidates":list(evaluate.ALPHAS),"inner_splits":3,"input_files":{},"feature_metadata":{},
                "diagnostics":{evaluate.RAW+"_nested":{"overall":core.metrics(self.data.y,self.anchor["predictions"])}},"selections":self.raw_records}
            summary_path.write_text(json.dumps(summary),encoding="utf-8")
            loaded=evaluate.load_resnet_anchor(copy.deepcopy(self.data),path,summary_path,42)
            np.testing.assert_array_equal(loaded["predictions"],self.anchor["predictions"])
            rows[0]["fold"]=(rows[0]["fold"]+1)%5; path.write_text(core.csv_text(rows),encoding="utf-8")
            with self.assertRaises(ValueError): evaluate.load_resnet_anchor(copy.deepcopy(self.data),path,summary_path,42)

    def test_prior_dino_d4_is_fixed_and_mismatched_oof_rejected(self):
        refs={k:v for k,v in self.references().items() if k not in (evaluate.REFERENCE,evaluate.RAW_REFERENCE,"reference_dino_d4")}
        column=evaluate.previous.D4+"_nested"; d4=self.data.y+2
        saved={**refs,column:d4,core.PRIMARY_NAME:d4,evaluate.previous.FIXED_BLEND:refs[evaluate.previous.CURRENT]+0.5*(d4-refs[evaluate.previous.DINO])}
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/"oof.csv"; sp=Path(td)/"summary.json"
            rows=[{"ID":i,"hardness":self.data.y[j],"fold":int(self.data.folds[j]),**{n:p[j] for n,p in saved.items()}} for j,i in enumerate(self.data.ids)]
            path.write_text(core.csv_text(rows),encoding="utf-8")
            summary={"seed":42,"n_train":50,"experiment":"dinov2_d4_mean","feature_metadata":self.data.metadata,
                "d4_protocol":evaluate.previous.extract.PROTOCOL,"input_files":{},
                "diagnostics":{n:{"overall":core.metrics(self.data.y,p)} for n,p in saved.items()}}
            sp.write_text(json.dumps(summary),encoding="utf-8")
            loaded=evaluate.load_dino_reference(copy.deepcopy(self.data),refs,path,sp,42)
            np.testing.assert_array_equal(loaded["reference_dino_d4"],d4)
            rows[0][column]+=1; path.write_text(core.csv_text(rows),encoding="utf-8")
            with self.assertRaisesRegex(ValueError,"metric mismatch"): evaluate.load_dino_reference(copy.deepcopy(self.data),refs,path,sp,42)

    def test_evaluator_imports_no_torch_and_seed_paths(self):
        r=subprocess.run([sys.executable,"-c","import sys; from scripts import evaluate_resnet_d4; assert 'torch' not in sys.modules"],cwd=evaluate.ROOT,capture_output=True,text=True)
        self.assertEqual(r.returncode,0,r.stderr)
        for seed in (42,2026):
            a=evaluate.parse_args(["--seed",str(seed)])
            self.assertEqual(a.output_dir,evaluate.ROOT/f"reports/resnet_d4/seed{seed}")
            self.assertIn(f"seed{seed}",str(a.dino_d4_oof))


if __name__=="__main__": unittest.main()
