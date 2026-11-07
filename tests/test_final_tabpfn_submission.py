"""Synthetic and mock tests only; no competition fits, extraction or inference."""
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from scripts import final_tabpfn_submission as f


class FakePredictor:
    runtime = {"synthetic": True}

    def __init__(self):
        self.calls = []

    def __call__(self, x, y, q, seed):
        np.testing.assert_array_equal(y, x[:, 2])
        if np.any(x[:, 0] >= 500) or set(x[:, 0]) & set(q[:, 0]):
            raise AssertionError("Query rows entered context")
        self.calls.append((x.copy(), y.copy(), q.copy(), seed))
        return q[:, 2]+.002*q[:, 0]+(.5 if x.shape[1] == 285 else 0.), {"seconds": 0.}


def fixture():
    rng = np.random.default_rng(104)
    x = rng.normal(size=(1500, 285))
    x[:, 0] = np.arange(1500)
    ids = np.array([f"TRAIN_{i:06d}" for i in range(500)])
    data = SimpleNamespace(ids=ids, y=x[:500, 2].copy())
    features = {"existing185": x[:500, :185].copy(), "morphology285": x[:500].copy()}
    test_ids = [f"TEST_{i:06d}" for i in range(1000, 0, -1)]
    bp = {"test": {"ids": test_ids, "rows": 1000}}
    plan = {"frozen": {"synthetic": True, "test_binding": bp["test"], "anchor_ids": ids[:16].tolist()}}
    return data, features, x[500:, :185].copy(), bp, plan


class FinalTabPFNTests(unittest.TestCase):
    def test_selection_never_loads_test_and_uses_pooled_train_cv(self):
        data, features, _, bp, plan = fixture()
        predictor = FakePredictor()
        with tempfile.TemporaryDirectory() as td, contextlib.redirect_stdout(io.StringIO()):
            root = Path(td)
            with patch.object(f, "validate", return_value=(data, features, bp, plan)), \
                 patch.object(f, "baseline_test", side_effect=AssertionError("Selection accessed test values")), \
                 patch.object(f, "read_test_cache", side_effect=AssertionError("Selection accessed test cache")), \
                 patch.object(f.t, "Predictor", return_value=predictor), patch.object(f.t, "verify_weights", return_value=root/"synthetic.ckpt"):
                f.select_stage(root, "cpu")
                self.assertEqual(len(predictor.calls), 6)
                self.assertEqual([len(c[0]) for c in predictor.calls], [333, 333, 334]*2)
                s = f.read_selection(root, data, features, plan)
                self.assertEqual(s["representation"], "existing185")
                self.assertFalse(s["test_used_for_selection"])
                self.assertAlmostEqual(s["pooled_mse"]["existing185"], np.mean((.002*np.arange(500))**2))
                for rep in f.t.REPS:
                    r = [r for r in s["splits"] if r["representation"] == rep]
                    self.assertAlmostEqual(s["pooled_mse"][rep], np.average([a["valid_mse"] for a in r], weights=[len(a["valid_idx"]) for a in r]))
                self.assertEqual([a[3] for a in predictor.calls], [f.SEED+j for j in range(3)]*2)
                f.select_stage(root, "cpu")
                self.assertEqual(len(predictor.calls), 6)
                x2 = {n: x.copy() for n, x in features.items()}; x2["existing185"][0, 2] += 1.
                with self.assertRaises(ValueError):
                    f.read_selection(root, data, x2, plan)

    def test_existing185_path_never_extracts_images(self):
        base = np.ones((3, 185))
        fail = lambda *a: self.fail("Unexpected image read/extraction")
        x, check = f.build_test_features("existing185", base, ["a", "b", "c"], [], None, fail, fail)
        np.testing.assert_array_equal(x, base)
        self.assertFalse(check["required"])

    def test_morphology_anchor_checked_before_any_test_image(self):
        reads = []
        def reader(split, ident):
            reads.append((split, ident))
            return float(ident)
        extractor = lambda v: np.full(100, v)
        base = np.ones((2, 185))
        anchors = np.stack([np.full(100, i) for i in range(16)])
        x, check = f.build_test_features("morphology285", base, [20, 21], list(range(16)), anchors, reader, extractor)
        self.assertTrue(check["passed"])
        self.assertEqual(x.shape, (2, 285))
        np.testing.assert_array_equal(x[:, :185], base)
        np.testing.assert_array_equal(x[:, 185:], np.stack([np.full(100, 20), np.full(100, 21)]))
        self.assertEqual(reads[:16], [("train", i) for i in range(16)])
        reads.clear()
        with self.assertRaisesRegex(ValueError, "anchor mismatch"):
            f.build_test_features("morphology285", base, [20, 21], list(range(16)), anchors+1, reader, extractor)
        self.assertTrue(all(split == "train" for split, _ in reads))

    def test_full_pipeline_fixed_blend_unsorted_ids_resume_original_preserved(self):
        data, features, tx, bp, plan = fixture()
        predictor = FakePredictor()
        baseline = np.linspace(180., 220., 1000)
        with tempfile.TemporaryDirectory() as td, contextlib.redirect_stdout(io.StringIO()):
            root, base = Path(td)/"candidate", Path(td)/"original"
            (base/"test_features").mkdir(parents=True)
            (base/"test_features/features.npz").write_bytes(b"synthetic cached feature binding")
            (base/"submission").mkdir()
            original_path = base/"submission/submission.csv"
            original_path.write_text(f.original.submission_text(bp["test"]["ids"], baseline), encoding="utf-8", newline="")
            before = original_path.read_bytes()
            original_hash = f.t.common.sha256_file(original_path)
            with patch.object(f, "BASE", base), patch.object(f, "BASE_SHA", original_hash), \
                 patch.object(f, "validate", return_value=(data, features, bp, plan)), \
                 patch.object(f, "baseline_test", return_value=(tx, baseline)), \
                 patch.object(f.original, "validate_test_identity"), patch.object(f.t, "Predictor", return_value=predictor), \
                 patch.object(f.t, "verify_weights", return_value=base/"synthetic.ckpt"):
                f.select_stage(root, "cpu")
                f.extract_stage(root)
                f.predict_stage(root, "cpu")
                self.assertEqual(len(predictor.calls), 7)
                last = predictor.calls[-1]
                self.assertEqual((last[0].shape, last[2].shape), ((500, 185), (1000, 185)))
                self.assertEqual(last[3], f.SEED+9)
                expected = .5*(baseline+tx[:, 2]+.002*tx[:, 0])
                # Addition grouping is not identical to the production expression.
                saved = f.t.core.finite_column(f.t.core.read_rows(root/"submission/submission.csv", {"ID", "hardness"}), "hardness", root)
                np.testing.assert_allclose(saved, expected, atol=1e-12, rtol=0)
                self.assertEqual(f.original.submission_ids(root/"submission/submission.csv"), bp["test"]["ids"])
                self.assertEqual(original_path.read_bytes(), before)
                f.select_stage(root, "cpu"); f.extract_stage(root); f.predict_stage(root, "cpu")
                self.assertEqual(len(predictor.calls), 7)
                self.assertEqual(original_path.read_bytes(), before)
                with (root/"submission/component_predictions.csv").open("a") as h:
                    h.write("corrupt")
                with self.assertRaises(ValueError):
                    f.predict_stage(root, "cpu")

    def test_submission_rejects_bad_values_duplicate_ids_and_changed_binding(self):
        with tempfile.TemporaryDirectory() as td, contextlib.redirect_stdout(io.StringIO()):
            root = Path(td)
            ids = ["TEST_000002", "TEST_000001"]
            for a in (np.array([1., np.nan]), np.array([1.])):
                with self.assertRaises(ValueError):
                    f.write_submission(root, ids, a, np.ones(2), {})
            with self.assertRaises(ValueError):
                f.write_submission(root, [ids[0]]*2, np.ones(2), np.ones(2), {})
            f.write_submission(root, ids, np.array([1., 4.]), np.array([3., 8.]), {"plan": "one"})
            with self.assertRaises(ValueError):
                f.write_submission(root, ids, np.array([1., 4.]), np.array([3., 8.]), {"plan": "two"})


if __name__ == "__main__":
    unittest.main()
