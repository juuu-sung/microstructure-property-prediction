"""CPU/synthetic-only tests for grouped images, nested selection and resume integrity."""
from contextlib import contextmanager, redirect_stdout
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from scripts import micronet_multiview_experiment as e


@contextmanager
def temporary():
    parent = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory(dir=parent, prefix='.multiview_test_') as loc:
        folder = Path(loc).resolve()
        if folder.parent != parent:
            raise ValueError('Temporary cleanup must stay under tests')
        yield folder


def fixture():
    rng = np.random.default_rng(8171)
    x = rng.normal(size=(500, 5, 6)).astype(np.float32)
    # Every view carries the source-image index, allowing grouped split checks.
    x[:, :, 0] = np.arange(500)[:, None]
    y = 200 + rng.normal(size=500)*10
    folds = np.repeat(np.arange(5), 100)
    data = SimpleNamespace(y=y, ids=np.array([f'SYNTHETIC_{i:06d}' for i in range(500)]), noise=x[:, 0, 1])
    ref = {'folds': folds, 'current': y+4, 'frozen_micronet_ridge': y+5}
    return x, data, ref


def fake_run(xtrain, ytrain, xquery, *, seed, device, epochs, query_y=None, checkpoint=None, log=''):
    records = []
    for epoch in epochs:
        pred = np.full(len(xquery), float(ytrain.mean()) + .001*epoch)
        row = {'epoch': epoch, 'predictions': pred.tolist(), 'train_sse': 9.*len(ytrain), 'train_n': len(ytrain)}
        if query_y is not None:
            row.update(valid_sse=float(np.sum((pred-query_y)**2)), valid_n=len(query_y))
        records.append(row)
    artifact = None
    if checkpoint is not None:
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b'synthetic-checkpoint')
        artifact = e.t.file_ref(checkpoint)
    return {'records': records, 'checkpoint': artifact}


class MultiViewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_native_views_cover_image_once_without_resize_or_reorder(self):
        x = torch.arange(3*256*256).reshape(1, 3, 256, 256)
        views = e.image_views(x)
        self.assertIs(views[0], x)
        self.assertTrue(all(v.shape == (1, 3, 128, 128) for v in views[1:]))
        joined = torch.cat((torch.cat(views[1:3], -1), torch.cat(views[3:5], -1)), -2)
        torch.testing.assert_close(joined, x, rtol=0, atol=0)
        with self.assertRaises(ValueError):
            e.image_views(x[:, :, :128, :128])

    def test_real_weights_native128_and256_match_independent_endpoints(self):
        torch.manual_seed(5)
        model = e.old.load_model('micronet')
        state = e.base.feature_state_hash(model)
        alternate = e.old.import_encoder().from_name('efficientnet-b0', image_size=128)
        del alternate._fc
        alternate.load_state_dict(model.state_dict(), strict=True)
        alternate.eval().requires_grad_(False)
        with torch.inference_mode():
            for size in (128, 256):
                x = torch.randn(1, 3, size, size)
                actual = e.native_maps(model, x)
                endpoints = (alternate if size == 128 else model).extract_endpoints(x)
                for got, key in zip(actual, ('reduction_4', 'reduction_5')):
                    torch.testing.assert_close(got, endpoints[key], rtol=0, atol=0)
                self.assertEqual(e.native_features(model, x).shape, (1, 864))
        self.assertEqual(e.base.feature_state_hash(model), state)
        self.assertFalse(any(p.requires_grad for p in model.parameters()))
        self.assertFalse(any(m.training for m in model.modules()))

    def test_global_anchor_allows_roundoff_but_rejects_wrong_extraction(self):
        x = np.ones((4, 864), np.float32)
        self.assertGreater(e.assert_global_anchor(x+1e-6, x)['max_abs'], 0)
        with self.assertRaises(ValueError):
            e.assert_global_anchor(x+.01, x)

    def test_outer_targets_and_all_queries_cannot_change_inner_selection(self):
        x, data, ref = fixture()
        calls = []
        def checked(*args, **kwargs):
            xt, yt, xq = args
            train_ids = set(xt[:, 0, 0].astype(int))
            query_ids = set(xq[:, 0, 0].astype(int))
            self.assertFalse(train_ids & query_ids)
            self.assertTrue(all(i >= 100 for i in train_ids))
            np.testing.assert_array_equal(yt, data.y[xt[:, 0, 0].astype(int)])
            self.assertTrue(np.all(xt[:, :, 0] == xt[:, :1, 0]))
            if kwargs['query_y'] is None:
                self.assertEqual(len(yt), 400)
            else:
                self.assertTrue(all(i >= 100 for i in query_ids))
            calls.append(len(yt))
            return fake_run(*args, **kwargs)
        first = e.fit_fold(x, data.y, ref['folds'], 42, 0, device=torch.device('cpu'), run=checked)
        self.assertEqual(len(calls), 8)
        self.assertEqual(calls.count(400), 2)
        poison = data.y.copy(); poison[:100] += 1e6
        second = e.fit_fold(x, poison, ref['folds'], 42, 0, device=torch.device('cpu'), run=fake_run)
        self.assertEqual(first, second)
        x_poison = x.copy(); x_poison[:100] += 1e6
        third = e.fit_fold(x_poison, data.y, ref['folds'], 42, 0, device=torch.device('cpu'), run=fake_run)
        self.assertEqual(first['selection'], third['selection'])

    def test_pooled_sse_selection_not_mean_fold_rmse_and_earlier_ties(self):
        runs = []
        for n, a, b in [(1, 25., 0.), (9, 0., 16.)]:
            runs.append({'records': [{'epoch': 0, 'valid_n': n, 'valid_sse': a, 'train_n': 10, 'train_sse': 10.},
                                     {'epoch': 5, 'valid_n': n, 'valid_sse': b, 'train_n': 10, 'train_sse': 10.}]})
        chosen, curve = e.select_epoch(runs, (0, 5))
        self.assertEqual(chosen['epoch'], 5)
        self.assertAlmostEqual(curve[0]['inner_mse'], 2.5)
        for run in runs:
            run['records'][1]['valid_sse'] = run['records'][0]['valid_sse']
        self.assertEqual(e.select_epoch(runs, (0, 5))[0]['epoch'], 0)

    def test_training_learns_synthetic_image_target_and_saved_checkpoint_reproduces(self):
        rng = np.random.default_rng(721)
        x = rng.normal(size=(180, 5, 6)).astype(np.float32)
        y = 210 + 10*x[:, :, 0].mean(1)
        with temporary() as folder:
            result = e.trajectory(x[:140], y[:140], x[140:], seed=6, device=torch.device('cpu'),
                                  epochs=(0, 40), query_y=y[140:], checkpoint=folder/'head.pt')
            self.assertLess(result['records'][-1]['valid_sse'], .65*result['records'][0]['valid_sse'])
            saved = torch.load(folder/'head.pt', weights_only=True)
            model = e.make_head(torch, 6).eval(); model.load_state_dict(saved['state_dict'])
            s = saved['scaling']
            q = ((torch.tensor(x[140:]).double()-s['mean'])/s['scale']).float()
            with torch.inference_mode():
                pred = model(q).double()*s['yscale']+s['ymean']
            np.testing.assert_allclose(pred.numpy(), result['records'][-1]['predictions'], rtol=0, atol=1e-6)
            np.testing.assert_allclose(s['mean'], x[:140].astype(float).reshape(-1, 6).mean(0), rtol=0, atol=0)
            # Poisoning only query labels changes diagnostics, never fitted predictions or scaling.
            poisoned = e.trajectory(x[:140], y[:140], x[140:], seed=6, device=torch.device('cpu'),
                                    epochs=(0, 40), query_y=y[140:]+1e5)
            self.assertEqual(result['scaling'], poisoned['scaling'])
            self.assertEqual(result['records'][-1]['predictions'], poisoned['records'][-1]['predictions'])

    def test_control_and_candidate_share_nominal_parameters_and_initialize_at_mean(self):
        torch.manual_seed(11); a = e.make_head(torch, 6).eval()
        torch.manual_seed(11); b = e.make_head(torch, 6).eval()
        self.assertEqual(e.base.feature_state_hash(a), e.base.feature_state_hash(b))
        self.assertEqual(sum(p.numel() for p in a.parameters()), sum(p.numel() for p in b.parameters()))
        self.assertEqual(a(torch.randn(8, 1, 6)).shape, (8,))
        self.assertTrue(torch.all(b(torch.randn(8, 5, 6)) == 0))

    def test_resume_reuses_trajectories_and_detects_tampered_or_missing_checkpoint(self):
        x, data, ref = fixture()
        with temporary() as folder, redirect_stdout(io.StringIO()):
            def runner(*args, **kwargs):
                return e.fit_fold(*args, **kwargs, run=fake_run)
            a = e.fold_cell(folder, {'synthetic': True}, x, data.y, ref['folds'], 42, 0, torch.device('cpu'), runner=runner)
            b = e.fold_cell(folder, {'synthetic': True}, x, data.y, ref['folds'], 42, 0, torch.device('cpu'), runner=e.old.unavailable)
            self.assertEqual(a, b)
            with self.assertRaises(ValueError):
                e.fold_cell(folder, {'synthetic': False}, x, data.y, ref['folds'], 42, 0, torch.device('cpu'))
            checkpoint = folder/'cells/fold0/global_only/refit.pt'
            checkpoint.write_bytes(b'changed')
            with self.assertRaises(ValueError):
                e.fold_cell(folder, {'synthetic': True}, x, data.y, ref['folds'], 42, 0, torch.device('cpu'))

    def test_feature_schema_identity_and_metadata_hash(self):
        rng = np.random.default_rng(55)
        data = SimpleNamespace(ids=np.array(['A', 'B', 'C']))
        arrays = {'ids': data.ids, 'views': rng.normal(size=(3, 5, 864)).astype(np.float32)}
        plan = {'frozen': {'upstream_state_hash': 'abc'}}
        with temporary() as folder, patch.object(e.old, 'OUT', folder/'archive'):
            path = folder/'features/features.npz'; path.parent.mkdir()
            archived = e.old.OUT/'features/features.npz'; archived.parent.mkdir(parents=True)
            np.savez(archived, micronet=arrays['views'][:, 0])
            np.savez(path, **arrays)
            meta = {'plan': e.t.digest(plan['frozen']), 'file_sha256': e.base.sha256_file(path),
                    'encoder_state': 'abc', 'labels_used': False, 'test_accessed': False, 'encoder_updates': 0,
                    'arrays': {n: e.base.sha256_array(a) for n, a in arrays.items()}}
            e.t.write_json(path.with_name('metadata.json'), meta)
            actual, _ = e.load_features(folder, data, plan)
            np.testing.assert_array_equal(actual, arrays['views'])
            meta['labels_used'] = True; e.t.write_json(path.with_name('metadata.json'), meta)
            with self.assertRaises(ValueError):
                e.load_features(folder, data, plan)

    def test_complete_synthetic_pipeline_metrics_and_manifest(self):
        x, data, ref = fixture()
        plan = {'frozen': {'common_failure_ids': data.ids[:16].tolist()}}
        with temporary() as folder, redirect_stdout(io.StringIO()):
            def runner(*args, **kwargs):
                return e.fit_fold(*args, **kwargs, run=fake_run)
            binding = {'seed': 42, 'synthetic': True}
            pred, choices = e.collect(folder, binding, data, ref, x, 42, torch.device('cpu'), runner=runner)
            e.save_results(folder, data, ref, pred, choices, binding, plan)
            self.assertTrue(e.completed(folder, binding))
            metrics, rows, diag, errors, detail = e.result_data(data, ref, pred, plan['frozen']['common_failure_ids'])
            np.testing.assert_allclose([r[e.PRIMARY] for r in rows], .5*ref['current']+.5*pred['global_patches'])
            lookup = {(r['model'], r['group']): r for r in diag}
            self.assertAlmostEqual(lookup[e.PRIMARY, 'all']['sse'],
                                   lookup[e.PRIMARY, 'common16']['sse']+lookup[e.PRIMARY, 'remaining484']['sse'])
            (folder/'cells/fold0/global_only/refit.pt').write_bytes(b'tampered')
            with self.assertRaises(ValueError):
                e.completed(folder, binding)


if __name__ == '__main__':
    unittest.main()
