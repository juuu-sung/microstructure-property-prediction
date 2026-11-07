"""Synthetic-only learning-curve invariants; no real images/cache fitting."""
import contextlib
import copy
import csv
import inspect
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
from scripts import evaluate_dinov2_learning_curve as curve
from scripts import evaluate_frozen_resnet as core
try:
    from . import test_feature_fusion as fixture_module
except ImportError:
    import test_feature_fusion as fixture_module


class InputAndCommandTests(unittest.TestCase):
    def setUp(self):
        self.fx = fixture_module.FusionArtifactTests()
        self.addCleanup(self.fx.doCleanups)
        self.fx.setUp()
        self.root = self.fx.root
        for row in self.fx.reference_rows:
            row[curve.FEATURE_NAME + "_nested"] = row[core.PRIMARY_NAME]
        self.fx.save_rows("dino.csv", self.fx.reference_rows)
        self.fx.summary["diagnostics"][curve.FEATURE_NAME + "_nested"] = {"overall": core.metrics(self.fx.y, self.fx.previous_dino)}
        self.fx.summary.update(alpha_candidates=list(curve.ALPHAS), inner_splits=3, selections=[])
        for fold in range(5):
            candidates = [{"representation": curve.FEATURE_NAME, "alpha": a, "inner_mse": float(10+i),
                           "inner_fold_mse": [float(10+i)]*3} for i, a in enumerate(curve.ALPHAS)]
            self.fx.summary["selections"].append({"outer_fold": fold, "inner_seed": 42+fold,
                "outer_train_n": 40, "outer_valid_n": 10, "inner_validation_sizes": [14,13,13],
                "candidates": candidates, "per_representation": {curve.FEATURE_NAME: candidates[0]}})
        self.fx.save_summary()

    def load(self):
        return curve.load_inputs(self.root / "features.npz", self.root / "train.csv", self.root / "folds.csv",
            self.root / "baseline.csv", self.root / "audit.csv", self.root / "frozen.csv", self.root / "dino.csv", 42,
            previous_dino_summary=self.root / "dino_summary.json", expected_n=50)

    def argv(self):
        return ["--features", str(self.root / "features.npz"), "--train-csv", str(self.root / "train.csv"),
            "--folds", str(self.root / "folds.csv"), "--baseline", str(self.root / "baseline.csv"),
            "--audit-features", str(self.root / "audit.csv"), "--previous-frozen-oof", str(self.root / "frozen.csv"),
            "--previous-dino-oof", str(self.root / "dino.csv"), "--previous-dino-summary", str(self.root / "dino_summary.json"),
            "--output-dir", str(self.root / "out")]

    def test_anchor_schema_and_cache_validated_without_fit_or_mutation(self):
        before = {p: core.sha256(p) for p in self.root.iterdir() if p.is_file()}
        with mock.patch.object(core, "make_estimator") as factory:
            data, references, anchor = self.load()
            factory.assert_not_called()
        self.assertEqual(set(data.features), {curve.FEATURE_NAME})
        np.testing.assert_array_equal(data.features[curve.FEATURE_NAME], self.fx.dino_features)
        np.testing.assert_array_equal(anchor["predictions"], self.fx.previous_dino)
        self.assertEqual(len(anchor["selections"]), 5)
        self.assertEqual(before, {p: core.sha256(p) for p in before})
        self.assertIn(curve.references_module.CURRENT_REFERENCE, references)

    def test_bad_anchor_alpha_scores_and_prediction_rejected(self):
        original = copy.deepcopy(self.fx.summary)
        mutations = [lambda s: s.update(alpha_candidates=[1000]),
                     lambda s: s["selections"][0].update(inner_seed=100),
                     lambda s: s["selections"][0]["candidates"][0].update(inner_mse=500),
                     lambda s: s["diagnostics"][curve.FEATURE_NAME+"_nested"]["overall"].update(rmse=1000)]
        for change in mutations:
            self.fx.summary = copy.deepcopy(original)
            change(self.fx.summary)
            self.fx.save_summary()
            with self.assertRaises(ValueError):
                self.load()

    def test_validate_only_memberships_in_memory_and_output_protection(self):
        loader, membership_builder = curve.load_inputs, curve.build_memberships
        def small_loader(*args, **kwargs): return loader(*args, **kwargs, expected_n=50)
        def small_memberships(folds, seed): return membership_builder(folds, seed, train_sizes=(10,20,30,40))
        captured = io.StringIO()
        with mock.patch.object(curve, "load_inputs", side_effect=small_loader), \
             mock.patch.object(curve, "build_memberships", side_effect=small_memberships), \
             mock.patch.object(curve, "evaluate_curve") as fitting, mock.patch.object(core, "make_estimator") as factory, \
             contextlib.redirect_stdout(captured):
            self.assertEqual(curve.main(self.argv()+["--validate-only"]), 0)
            fitting.assert_not_called()
            factory.assert_not_called()
        info = json.loads(captured.getvalue())
        self.assertEqual(info["membership_sets"], 60)
        self.assertEqual(info["estimators_fitted"], 0)
        self.assertEqual(info["planned_estimator_fits"], 950)
        self.assertFalse(info["sampling_uses_labels"])
        self.assertFalse((self.root/"out").exists())
        (self.root/"out").mkdir()
        (self.root/"out/keep.txt").write_text("keep",encoding="utf-8")
        with mock.patch.object(curve, "load_inputs") as load:
            with self.assertRaisesRegex(ValueError, "overwrite"):
                curve.main(self.argv())
            with self.assertRaisesRegex(ValueError, "protected input"):
                curve.main(self.argv()+["--output-dir",str(self.root),"--overwrite"])
            load.assert_not_called()

    def test_seed_defaults_and_cli_has_no_arbitrary_size_or_grid(self):
        for seed in (42,2026):
            args = curve.parse_args(["--seed",str(seed)])
            self.assertEqual(args.output_dir,curve.ROOT/f"reports/dinov2_learning_curve/seed{seed}")
            self.assertEqual(args.folds.name,"folds.csv" if seed==42 else "benchmark_folds_seed2026.csv")
        for option in ("--repeats", "--train-sizes", "--alphas"):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                curve.parse_args([option,"2"])


class CurveCalculationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(12)
        cls.X = rng.normal(size=(100,8))
        cls.X[:,0] = np.arange(100)/100
        cls.y = 200+8*cls.X[:,1]+rng.normal(size=100)*3
        cls.folds = np.arange(100)%5
        cls.sizes,cls.repeats,cls.alphas = (30,50,70,80),3,(10.0,100.0)
        cls.data = core.Inputs(np.array([f"TRAIN_{i+1:06d}" for i in range(100)]),cls.y,
            {curve.FEATURE_NAME:cls.X},cls.folds,cls.y+np.sin(np.arange(100)),np.arange(100),{}, {})
        cls.memberships = curve.build_memberships(cls.folds,42,train_sizes=cls.sizes,repeats=cls.repeats)
        with contextlib.redirect_stdout(io.StringIO()):
            old_predictions,old_selections = core.nested_predictions(cls.data,42,1,alphas=cls.alphas,representations=(curve.FEATURE_NAME,))
        cls.anchor = {"predictions":old_predictions[core.PRIMARY_NAME],"prediction_column":curve.FEATURE_NAME+"_nested",
            "selections":{r["outer_fold"]:{"winner":r["per_representation"][curve.FEATURE_NAME],
                "candidates":r["candidates"],"inner_validation_sizes":r["inner_validation_sizes"]} for r in old_selections}}
        original_factory = core.make_estimator
        cls.fit_sizes=[]
        class RecordingEstimator:
            def __init__(self,alpha): self.estimator=original_factory(alpha)
            def fit(self,X,y):
                cls.fit_sizes.append(len(y));self.estimator.fit(X,y);return self
            def predict(self,X): return self.estimator.predict(X)
        with mock.patch.object(core,"make_estimator",RecordingEstimator),contextlib.redirect_stdout(io.StringIO()):
            cls.records,cls.anchor_check=curve.evaluate_curve(cls.data,cls.memberships,42,1,anchor=cls.anchor,
                train_sizes=cls.sizes,repeats=cls.repeats,alphas=cls.alphas)
        cls.tables=curve.summarize(cls.data,cls.records,train_sizes=cls.sizes,repeats=cls.repeats)

    def test_label_independent_nested_membership_disjoint_validation_and_canonical_full_order(self):
        self.assertNotIn("y",inspect.signature(curve.build_memberships).parameters)
        same=curve.build_memberships(self.folds,42,train_sizes=self.sizes,repeats=self.repeats)
        for fold in range(5):
            validation=set(np.flatnonzero(self.folds==fold))
            for repeat in range(3):
                previous=set()
                for size in self.sizes:
                    indices=self.memberships[(fold,repeat,size)]
                    np.testing.assert_array_equal(indices,same[(fold,repeat,size)])
                    self.assertEqual(len(indices),size)
                    self.assertTrue(np.all(indices[:-1]<indices[1:]))
                    self.assertFalse(validation&set(indices))
                    self.assertTrue(previous<=set(indices));previous=set(indices)
                np.testing.assert_array_equal(self.memberships[(fold,repeat,80)],np.flatnonzero(self.folds!=fold))
            self.assertFalse(np.array_equal(self.memberships[(fold,0,30)],self.memberships[(fold,1,30)]))

    def test_exact_anchor_fit_count_and_full_size_reuse(self):
        self.assertEqual(curve.planned_fit_count(),950)
        self.assertEqual(len(self.fit_sizes),350) # 5*(3*3+1)*(2*3+1), relaxed synthetic alpha grid.
        self.assertTrue(all(n<100 for n in self.fit_sizes))
        self.assertEqual(self.fit_sizes.count(80),5)
        self.assertEqual(self.anchor_check["max_absolute_prediction_difference"],0)
        self.assertEqual(self.anchor_check["max_absolute_inner_mse_difference"],0)
        for fold in range(5):
            full=[r for r in self.records if r["outer_fold"]==fold and r["train_size"]==80]
            self.assertEqual(len({r["fit_instance"] for r in full}),1)
            self.assertEqual([r["full_size_reused"] for r in full],[False,True,True])
            self.assertIs(full[0]["validation_predictions"],full[2]["validation_predictions"])
        search=self.tables["inner_search"]
        self.assertEqual(len(search),100) # 50 independently fitted blocks, two alpha candidates.
        self.assertEqual(len([r for r in search if r["train_size"]==80]),10)
        self.assertTrue(all(r["repeat"]==0 for r in search if r["train_size"]==80))

    def test_unselected_and_outer_labels_cannot_change_subset_fit(self):
        indices=self.memberships[(0,0,30)];valid=np.flatnonzero(self.folds==0)
        changed=copy.copy(self.data);changed.y=self.y.copy()
        changed.y[np.setdiff1d(np.arange(100),indices)]+=500
        first=curve.fit_subset(self.data,indices,valid,42,alphas=self.alphas)
        second=curve.fit_subset(changed,indices,valid,42,alphas=self.alphas)
        self.assertEqual(first["candidates"],second["candidates"])
        self.assertEqual(first["winner"],second["winner"])
        self.assertEqual(first["train_metrics"],second["train_metrics"])
        np.testing.assert_array_equal(first["validation_predictions"],second["validation_predictions"])

    def test_mismatching_anchor_aborts_before_any_smaller_subset_fit(self):
        wrong=copy.deepcopy(self.anchor);wrong["predictions"][0]+=0.01
        with mock.patch.object(curve,"fit_subset",wraps=curve.fit_subset) as fitting,contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError,"anchor reproducibility"):
                curve.evaluate_curve(self.data,self.memberships,42,1,anchor=wrong,
                    train_sizes=self.sizes,repeats=self.repeats,alphas=self.alphas)
        self.assertEqual(fitting.call_count,5)
        self.assertTrue(all(len(call.args[1])==80 for call in fitting.call_args_list))
        full={r["outer_fold"]:r for r in self.records if r["train_size"]==80 and r["repeat"]==0}
        wrong=copy.deepcopy(self.anchor);wrong["selections"][0]["candidates"][0]["inner_mse"]+=0.01
        with self.assertRaisesRegex(ValueError,"anchor reproducibility"):
            curve.validate_anchor(full,self.data,wrong)
        wrong=copy.deepcopy(self.anchor);wrong["selections"][0]["winner"]["alpha"]=-1
        with self.assertRaisesRegex(ValueError,"selected alpha"):
            curve.validate_anchor(full,self.data,wrong)

    def test_pooled_repeat_rmse_then_mean_not_prediction_ensemble_and_zero_full_std(self):
        group=[r for r in self.tables["curve_by_repeat"] if r["train_size"]==30]
        summary=next(r for r in self.tables["curve_summary"] if r["train_size"]==30)
        self.assertEqual(summary["oof_rmse_mean"],np.mean([r["oof_rmse"] for r in group]))
        self.assertEqual(summary["oof_rmse_std"],np.std([r["oof_rmse"] for r in group],ddof=1))
        allpred=[]
        for repeat in range(3):
            rows=[r for r in self.tables["oof"] if r["repeat"]==repeat and r["train_size"]==30]
            pred=np.array([r["prediction"] for r in rows]);allpred.append(pred)
            self.assertAlmostEqual(group[repeat]["oof_rmse"],np.sqrt(np.mean((pred-self.y)**2)))
        ensemble_rmse=np.sqrt(np.mean((np.mean(allpred,axis=0)-self.y)**2))
        self.assertGreater(abs(summary["oof_rmse_mean"]-ensemble_rmse),1e-5)
        full=next(r for r in self.tables["curve_summary"] if r["train_size"]==80)
        self.assertEqual(full["oof_rmse_std"],0)
        self.assertEqual(full["independently_fitted_subset_repeats"],1)
        for row in self.tables["paired_300_to_400"]:
            rep=row["repeat"]
            rmse=lambda n:next(r["oof_rmse"] for r in self.tables["curve_by_repeat"] if r["repeat"]==rep and r["train_size"]==n)
            self.assertEqual(row["delta_oof_rmse"],rmse(80)-rmse(70))

    def test_numeric_outputs_survive_optional_plot_skip(self):
        references={curve.references_module.CURRENT_REFERENCE:self.y+1}
        with tempfile.TemporaryDirectory(prefix="learning-curve-report-") as directory:
            out=Path(directory)
            with mock.patch.object(curve,"save_plot",return_value={"status":"skipped","reason":"matplotlib unavailable"}):
                summary=curve.write_outputs(out,self.data,references,self.memberships,self.records,self.anchor_check,
                    self.tables,42,1,0.0,train_sizes=self.sizes,repeats=self.repeats,alphas=self.alphas)
            self.assertEqual(summary["estimator_fit_count"],350)
            self.assertFalse(summary["sampling"]["target_statistics_used_for_selection"])
            self.assertEqual(summary["plot"]["status"],"skipped")
            expected={"summary.json","report.md","memberships.csv","inner_search.csv","fold_metrics.csv","oof.csv",
                      "curve_by_repeat.csv","curve_summary.csv","paired_300_to_400.csv"}
            self.assertTrue(expected<={p.name for p in out.iterdir()})
            with (out/"oof.csv").open(encoding="utf-8") as f:
                self.assertEqual(len(list(csv.DictReader(f))),100*4*3)
            report=(out/"report.md").read_text(encoding="utf-8")
            self.assertIn("앙상블 점수가 아닙니다",report)
            self.assertIn("불확실성 0을 뜻하지 않습니다",report)


if __name__=="__main__":
    unittest.main()
