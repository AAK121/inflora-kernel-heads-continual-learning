"""Unit tests for methods/kernel_head.py and the post-hoc evaluation tool.

Run from the repository root with either:
    python -m pytest tests -v
    python -m unittest discover -s tests -v
No dataset, pretrained weights or GPU are needed (CUDA tests are skipped without a GPU).
"""
import math
import os
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from methods.kernel_head import (AnalyticKernelHead, KernelHeadMixin, LinearMap,  # noqa: E402
                                 RFFGaussianMap, RandomReLUMap, _orthogonal_gaussian,
                                 build_head, head_metrics)


def blobs(n_per_class, classes, dim, sep=4.0, seed=0, centers_seed=123):
    """Gaussian blobs; class centers depend only on the class id (stable across calls)."""
    g = torch.Generator().manual_seed(seed)
    gc = torch.Generator().manual_seed(centers_seed)
    centers = torch.randn(max(classes) + 1, dim, generator=gc) * sep
    xs, ys = [], []
    for c in classes:
        xs.append(centers[c] + torch.randn(n_per_class, dim, generator=g))
        ys.append(torch.full((n_per_class,), c, dtype=torch.long))
    return torch.cat(xs), torch.cat(ys)


def rings(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    r = torch.cat([torch.full((n,), 1.0), torch.full((n,), 3.0)]) + 0.1 * torch.randn(2 * n, generator=g)
    a = 2 * math.pi * torch.rand(2 * n, generator=g)
    x = torch.stack([r * torch.cos(a), r * torch.sin(a)], dim=1)
    y = torch.cat([torch.zeros(n), torch.ones(n)]).long()
    return x, y


def xor(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    centers = torch.tensor([[1., 1.], [-1., -1.], [1., -1.], [-1., 1.]])
    lab = torch.tensor([0, 0, 1, 1])
    idx = torch.randint(0, 4, (n,), generator=g)
    return centers[idx] + 0.15 * torch.randn(n, 2, generator=g), lab[idx]


def fit_acc(head, x, y, xt, yt):
    head.fit_task(x, y, int(y.max()) + 1)
    return (head.predict(xt).cpu() == yt).float().mean().item()


class TestFeatureMaps(unittest.TestCase):

    def test_rff_has_unit_norm(self):
        m = RFFGaussianMap(16, 512, input_norm='none', bandwidth=0.7)
        m.fit(torch.randn(10, 16))
        h = m(torch.randn(100, 16))
        self.assertEqual(h.shape, (100, 512))
        self.assertTrue(torch.allclose(h.norm(dim=1), torch.ones(100), atol=1e-5))

    def test_rff_approximates_gaussian_kernel(self):
        for orthogonal in (True, False):
            m = RFFGaussianMap(16, 16384, input_norm='none', bandwidth=2.0, orthogonal=orthogonal, seed=1)
            m.fit(torch.zeros(2, 16))
            z = torch.randn(40, 16)
            h = m(z).double()
            exact = torch.exp(-torch.cdist(z.double(), z.double()) ** 2 / (2 * 2.0 ** 2))
            err = (h @ h.t() - exact).abs().max().item()
            self.assertLess(err, 0.05, 'orthogonal={} max kernel error {}'.format(orthogonal, err))

    def test_orthogonal_blocks(self):
        omega = _orthogonal_gaussian(48, 16, torch.Generator().manual_seed(0))
        self.assertEqual(omega.shape, (48, 16))
        for b in range(3):
            block = omega[16 * b:16 * (b + 1)]
            u = block / block.norm(dim=1, keepdim=True)
            self.assertTrue(torch.allclose(u @ u.t(), torch.eye(16), atol=1e-4))

    def test_median_bandwidth(self):
        z = torch.randn(300, 8)
        m = RFFGaussianMap(8, 64, input_norm='none', bandwidth_mult=2.0)
        m.fit(z)
        self.assertAlmostEqual(float(m.sigma), 2.0 * torch.pdist(z).median().item(), places=4)

    def test_map_is_frozen_after_first_fit(self):
        m = RFFGaussianMap(8, 64, input_norm='center_l2')
        m.fit(torch.randn(200, 8) + 5.0)
        sigma, mean = float(m.sigma), m.mean.clone()
        m.fit(torch.randn(200, 8) * 10)
        self.assertEqual(float(m.sigma), sigma)
        self.assertTrue(torch.equal(m.mean, mean))

    def test_input_normalizations(self):
        x = torch.randn(50, 8) + 3.0
        m = LinearMap(8, 'center_l2')
        m.fit(x)
        z = m.preprocess(x)
        self.assertTrue(torch.allclose(z.norm(dim=1), torch.ones(50), atol=1e-5))
        self.assertTrue(torch.allclose(m.mean, x.mean(0)))
        self.assertTrue(torch.equal(LinearMap(8, 'none').preprocess(x), x))

    def test_relu_map(self):
        m = RandomReLUMap(8, 100, input_norm='l2')
        h = m(torch.randn(20, 8))
        self.assertEqual(h.shape, (20, 100))
        self.assertTrue((h >= 0).all())

    def test_seed_reproducibility(self):
        a, b = RFFGaussianMap(8, 64, seed=3), RFFGaussianMap(8, 64, seed=3)
        self.assertTrue(torch.equal(a.omega, b.omega))
        self.assertFalse(torch.equal(a.omega, RFFGaussianMap(8, 64, seed=4).omega))


class TestAnalyticHead(unittest.TestCase):

    def test_matches_closed_form_ridge(self):
        x, y = blobs(30, [0, 1, 2], 6)
        head = AnalyticKernelHead(LinearMap(6, 'none'), ridge=0.5)
        head.fit_task(x, y, 3)
        H = np.hstack([x.numpy().astype(np.float64), np.ones((len(x), 1))])
        Y = np.eye(3)[y.numpy()]
        W = np.linalg.solve(H.T @ H + 0.5 * np.eye(7), H.T @ Y)
        self.assertTrue(np.allclose(head.W.numpy(), W, atol=1e-4))

    def test_order_invariance(self):
        xa, ya = blobs(20, [0, 1], 8, seed=1)
        xb, yb = blobs(20, [2, 3], 8, seed=2)

        def make():
            return AnalyticKernelHead(RFFGaussianMap(8, 128, input_norm='none', bandwidth=3.0), ridge=1.0)

        h1, h2, h3 = make(), make(), make()
        for h in (h1, h2, h3):
            h.feature_map.fit(xa)
        h1.update(xa, ya, 2); h1.update(xb, yb, 4)
        h2.update(xb, yb, 4); h2.update(xa, ya, 4)
        h3.update(torch.cat([xa, xb]), torch.cat([ya, yb]), 4)
        for h in (h2, h3):
            self.assertTrue(torch.allclose(h1.G, h.G))
            self.assertTrue(torch.allclose(h1.Q, h.Q))

    def test_class_growth(self):
        head = AnalyticKernelHead(LinearMap(8, 'none'), ridge=1.0)
        x1, y1 = blobs(20, [0, 1], 8)
        x2, y2 = blobs(20, [2, 3, 4], 8)
        head.fit_task(x1, y1, 2)
        self.assertEqual(head.logits(x1).shape, (40, 2))
        head.fit_task(x2, y2, 5)
        self.assertEqual(head.logits(x1).shape, (40, 5))
        self.assertEqual(head.num_classes, 5)

    def test_rbf_separates_rings_but_linear_does_not(self):
        x, y = rings(200, seed=0)
        xt, yt = rings(200, seed=1)
        lin = fit_acc(AnalyticKernelHead(LinearMap(2, 'none'), ridge=1e-3), x, y, xt, yt)
        rbf = fit_acc(AnalyticKernelHead(RFFGaussianMap(2, 512, input_norm='none'), ridge=1e-3), x, y, xt, yt)
        self.assertLess(lin, 0.75)
        self.assertGreater(rbf, 0.95)

    def test_relu_solves_xor(self):
        x, y = xor(400, seed=0)
        xt, yt = xor(400, seed=1)
        lin = fit_acc(AnalyticKernelHead(LinearMap(2, 'none'), ridge=1e-3), x, y, xt, yt)
        relu = fit_acc(AnalyticKernelHead(RandomReLUMap(2, 256, input_norm='none'), ridge=1e-3), x, y, xt, yt)
        self.assertLess(lin, 0.75)
        self.assertGreater(relu, 0.95)

    def test_auto_ridge_matches_direct_solves(self):
        x, y = blobs(40, [0, 1, 2], 10, sep=1.0)
        grid = [1e-3, 1.0, 1e3]
        head = AnalyticKernelHead(RFFGaussianMap(10, 256, input_norm='l2'), ridge='val', ridge_grid=grid)
        head.feature_map.fit(x)
        fit_idx, val_idx = head._split(y)  # same generator state select_ridge will use
        lam = head.select_ridge(x, y, 3)
        self.assertIn(lam, grid)
        H = head._lift(x[fit_idx]); Y = torch.nn.functional.one_hot(y[fit_idx], 3).double()
        Hv = head._lift(x[val_idx])
        for l in grid:
            W = torch.linalg.solve(H.t() @ H + l * torch.eye(256, dtype=torch.float64), H.t() @ Y)
            acc = ((Hv @ W).argmax(1) == y[val_idx]).double().mean().item()
            self.assertAlmostEqual(head.history[-1]['val_acc_per_lambda'][l], acc, places=6)

    def test_gcv_from_sums_matches_explicit(self):
        x, y = blobs(25, [0, 1, 2, 3], 10, sep=1.0)
        head = AnalyticKernelHead(RFFGaussianMap(10, 64, input_norm='l2'), ridge='gcv', ridge_grid=[1e-2, 1.0, 1e2])
        head.feature_map.fit(x)
        head.update(x, y, 4)
        H = head._lift(x)
        Y = torch.nn.functional.one_hot(y, 4).double()
        n = float(len(y))
        for lam, score in head.gcv_scores().items():
            S = H @ torch.linalg.solve(H.t() @ H + lam * torch.eye(64, dtype=torch.float64), H.t())
            rss = ((Y - S @ Y) ** 2).sum().item()
            expected = (rss / n) / (1 - torch.trace(S).item() / n) ** 2
            self.assertAlmostEqual(score, expected, delta=1e-6 * max(1.0, expected))

    def test_gcv_when_features_outnumber_samples(self):
        """D > n with large unnormalized features: GCV must match the exact (dual) value and
        never select a lambda whose degrees of freedom exceed n (this used to happen)."""
        x, y = blobs(30, [0, 1, 2, 3, 4], 32, sep=0.6)
        x = x * 30.0 / x.norm(dim=1, keepdim=True).mean()  # raw-feature scale, as with input_norm='none'
        grid = [10.0 ** (k / 2.0) for k in range(-16, 17)]
        head = AnalyticKernelHead(RandomReLUMap(32, 1000, input_norm='none'), ridge='gcv', ridge_grid=grid)
        head.feature_map.fit(x)
        head.update(x, y, 5)  # n = 150 < D = 1000
        H = head._lift(x)
        K = H @ H.t()  # dual form: S = K (K + l I)^-1 is exact for D > n
        Y = torch.nn.functional.one_hot(y, 5).double()
        n = float(len(y))
        scores = head.gcv_scores()
        e = torch.linalg.eigvalsh(K).clamp_min(0)
        for lam in grid:
            if lam < 1e-8 * float(e[-1]):
                continue  # the explicit inverse is itself unreliable for extremely small lambda
            S = K @ torch.linalg.inv(K + lam * torch.eye(len(y), dtype=torch.float64))
            rss = ((Y - S @ Y) ** 2).sum().item()
            expected = n * rss / (n - torch.trace(S).item()) ** 2
            self.assertAlmostEqual(scores[lam], expected, delta=1e-4 * max(1.0, expected), msg='lambda={}'.format(lam))
        # selection in the regime that broke the old formula (first task, D = 3000 >> n = 300):
        # it chose a lambda with df == n, i.e. exact interpolation of the training labels
        x, y = blobs(60, [0, 1, 2, 3, 4], 64, sep=0.6, seed=0)
        x = x * 30.0 / x.norm(dim=1, keepdim=True).mean()
        head = build_head({'map': 'relu', 'dim': 3000, 'input_norm': 'none'}, 64)
        head.fit_task(x, y, 5)
        e = torch.linalg.eigvalsh(head.G).clamp_min(0)
        df = float((e / (e + float(head.ridge))).sum())
        self.assertLess(df, len(y) - 1.0, 'GCV selected an interpolating lambda={}'.format(float(head.ridge)))

    def test_gcv_is_order_invariant_and_default(self):
        xa, ya = blobs(30, [0, 1], 12, seed=1)
        xb, yb = blobs(30, [2, 3], 12, seed=2)
        cfg = {'map': 'relu', 'dim': 256, 'input_norm': 'none'}
        h1, h2 = build_head(cfg, 12), build_head(cfg, 12)
        self.assertEqual(h1.ridge_mode, 'gcv')
        h1.feature_map.fit(xa); h2.feature_map.fit(xa)
        h1.fit_task(xa, ya, 2); h1.fit_task(xb, yb, 4)
        h2.fit_task(xb, yb, 4); h2.fit_task(xa, ya, 4)
        self.assertEqual(float(h1.ridge), float(h2.ridge))
        self.assertTrue(torch.allclose(h1.logits(xa), h2.logits(xa), atol=1e-5))

    def test_invalid_ridge_mode(self):
        with self.assertRaises(ValueError):
            build_head({'map': 'linear', 'ridge': 'cv'}, 4)

    def test_incremental_tasks_with_auto_ridge(self):
        head = build_head({'map': 'rbf', 'dim': 1024, 'input_norm': 'l2', 'ridge': 'val'}, 32)
        accs = []
        for t, classes in enumerate([[0, 1, 2], [3, 4, 5], [6, 7, 8]]):
            x, y = blobs(60, classes, 32, seed=t)
            head.fit_task(x, y, max(classes) + 1)
            xt, yt = blobs(30, list(range(max(classes) + 1)), 32, seed=100 + t)
            accs.append((head.predict(xt) == yt).float().mean().item())
        self.assertGreater(min(accs), 0.9, accs)
        self.assertEqual(len(head.history), 3)

    def test_state_dict_roundtrip(self):
        x, y = blobs(30, [0, 1, 2, 3], 12)
        cfg = {'map': 'rbf', 'dim': 128, 'ridge': 1.0}
        a = build_head(cfg, 12)
        a.fit_task(x, y, 4)
        b = build_head(cfg, 12)
        b.load_state_dict(a.state_dict())
        self.assertTrue(torch.allclose(a.logits(x), b.logits(x)))
        self.assertTrue(b.feature_map.is_fitted)

    def test_float32_tiny_ridge_does_not_crash(self):
        x, y = blobs(10, [0, 1], 4)
        head = AnalyticKernelHead(RandomReLUMap(4, 256, input_norm='none'), ridge=1e-12, dtype=torch.float32)
        head.fit_task(x, y, 2)  # rank-deficient G: Cholesky may fail, eigh fallback must work
        self.assertTrue(torch.isfinite(head.W).all())

    def test_build_head_configs(self):
        for cfg, d in (({'map': 'linear'}, 9), ({'map': 'relu', 'dim': 50}, 50), ({'map': 'rbf', 'dim': 64}, 64)):
            self.assertEqual(build_head(cfg, 8).feature_map.out_dim, d)
        with self.assertRaises(ValueError):
            build_head({'map': 'rbf', 'dim': 63}, 8)
        with self.assertRaises(ValueError):
            build_head({'map': 'unknown'}, 8)

    @unittest.skipUnless(torch.cuda.is_available(), 'needs CUDA')
    def test_cuda_matches_cpu(self):
        x, y = blobs(30, [0, 1, 2], 16)
        cfg = {'map': 'rbf', 'dim': 256, 'ridge': 1.0}
        cpu, gpu = build_head(cfg, 16), build_head(cfg, 16, torch.device('cuda'))
        cpu.fit_task(x, y, 3)
        gpu.fit_task(x, y, 3)
        self.assertTrue(torch.allclose(cpu.logits(x), gpu.logits(x).cpu(), atol=1e-4))


class TestMetrics(unittest.TestCase):

    def test_head_metrics(self):
        y = np.array([0, 1, 2, 3, 4, 5])
        pred = np.array([0, 1, 3, 3, 5, 5])  # 2 -> 3 (same task), 4 -> 5 (same task)
        m = head_metrics(pred, y, known_classes=4, class_num=2)
        self.assertAlmostEqual(m['total'], 66.67)
        self.assertEqual(m['task_acc'], 1.0)
        self.assertEqual(m['00-01'], 100.0)
        self.assertEqual(m['02-03'], 50.0)
        self.assertEqual(m['new'], 50.0)


# --- integration with a stub learner (no ViT, no dataset) -------------------

class _StubNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(20, 16)

    def extract_vector(self, x):
        return self.proj(x)


class _ListDataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x, self.y = x, y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return i, self.x[i], self.y[i]


class _FakeDataManager(object):
    def __init__(self):
        self.train = blobs(40, list(range(6)), 20, seed=0)
        self.test = blobs(20, list(range(6)), 20, seed=1)

    def get_dataset(self, indices, source, mode):
        x, y = self.train if source == 'train' else self.test
        mask = torch.zeros_like(y, dtype=torch.bool)
        for c in indices:
            mask |= y == int(c)
        return _ListDataset(x[mask], y[mask])


class _StubLearner(KernelHeadMixin):
    def __init__(self):
        self._network = _StubNet()
        self._device = torch.device('cpu')
        self._known_classes = 0
        self._total_classes = 0
        self._cur_task = -1
        self.class_num = 2
        self.num_workers = 0
        self._init_kernel_head({'embd_dim': 16, 'kernel_head': {'map': 'rbf', 'dim': 512, 'batch_size': 32}})


class TestMixinIntegration(unittest.TestCase):

    def test_three_tasks(self):
        from torch.utils.data import DataLoader
        torch.manual_seed(0)
        learner, dm = _StubLearner(), _FakeDataManager()
        for t in range(3):
            learner._cur_task = t
            learner._total_classes = learner._known_classes + 2
            learner._kh_after_task(dm)
            test = dm.get_dataset(range(learner._total_classes), 'test', 'test')
            metrics = learner._kh_eval(DataLoader(test, batch_size=16))
            learner._known_classes = learner._total_classes
        self.assertEqual(len(learner._kh_curve['top1']), 3)
        self.assertEqual(learner._kh.num_classes, 6)
        self.assertGreater(metrics['total'], 90.0)


class TestPosthocTool(unittest.TestCase):

    def test_evaluate_head_on_fake_cache(self):
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tools'))
        from eval_kernel_heads import evaluate_head, summarize
        steps, known = [], 0
        for t in range(3):
            total = known + 2
            tr = blobs(40, list(range(known, total)), 16, seed=t)
            te = blobs(20, list(range(total)), 16, seed=10 + t)
            steps.append({'known': known, 'total': total, 'train_f': tr[0], 'train_y': tr[1],
                          'test_f': te[0], 'test_y': te[1], 'base_pred': te[1].clone()})
            known = total
        cache = {'meta': {'class_num': 2}, 'steps': steps}
        curve = evaluate_head(cache, {'name': 'rbf', 'map': 'rbf', 'dim': 256})
        self.assertEqual(len(curve['top1']), 3)
        self.assertEqual(curve['base_top1'], [100.0, 100.0, 100.0])
        self.assertGreater(curve['top1'][-1], 90.0)
        self.assertEqual(summarize([curve])['n_seeds'], 1)

    def test_saved_head_reloads_with_identical_predictions(self):
        import tempfile
        from eval_kernel_heads import evaluate_head
        from methods.kernel_head import load_head
        steps, known = [], 0
        for t in range(2):
            total = known + 2
            tr = blobs(40, list(range(known, total)), 16, seed=t)
            te = blobs(20, list(range(total)), 16, seed=10 + t)
            steps.append({'known': known, 'total': total, 'train_f': tr[0], 'train_y': tr[1],
                          'test_f': te[0], 'test_y': te[1], 'base_pred': te[1].clone()})
            known = total
        cache = {'meta': {'class_num': 2, 'seed': 0, 'dataset': 'fake', 'ckpt_dir': 'x'}, 'steps': steps}
        d = tempfile.mkdtemp()
        for name, cfg, keep in (('relu', {'map': 'relu', 'dim': 256, 'input_norm': 'none'}, False),
                                ('rbf', {'map': 'rbf', 'dim': 256, 'input_norm': 'center_l2'}, True)):
            cfg['name'] = name
            curve = evaluate_head(cache, cfg, save_dir=d, keep_gram=keep)
            head = load_head(os.path.join(d, name, 'seed0.pt'))
            acc = (head.predict(steps[-1]['test_f']).cpu() == steps[-1]['test_y']).float().mean().item() * 100
            self.assertAlmostEqual(acc, curve['top1'][-1], places=1)
            self.assertEqual(head.W.shape, (256, 4))
            ckpt = torch.load(os.path.join(d, name, 'seed0.pt'))
            self.assertEqual('G' in ckpt['state'], keep)
            self.assertEqual(ckpt['meta']['n_classes'], 4)
            self.assertNotIn('model', ckpt)

    def test_sweep_bundles_model_with_head_per_folder(self):
        import json
        import tempfile
        from eval_kernel_heads import run_sweep
        from methods.kernel_head import load_head
        steps, known = [], 0
        for t in range(2):
            total = known + 2
            tr = blobs(30, list(range(known, total)), 8, seed=t)
            te = blobs(10, list(range(total)), 8, seed=10 + t)
            steps.append({'known': known, 'total': total, 'train_f': tr[0], 'train_y': tr[1],
                          'test_f': te[0], 'test_y': te[1], 'base_pred': te[1].clone()})
            known = total
        d = tempfile.mkdtemp()
        caches, bundle = [], {}
        for seed in (0, 1):
            caches.append({'meta': {'class_num': 2, 'seed': seed, 'dataset': 'fake', 'ckpt_dir': 'x'}, 'steps': steps})
            model_path = os.path.join(d, 'task_1_seed{}.pth'.format(seed))
            torch.save({'w': torch.full((3,), float(seed))}, model_path)  # stand-in for a SiNet state dict
            bundle[seed] = (model_path, {'total_sessions': 2, 'seed': [seed]})
        heads = [{'name': 'lin', 'map': 'linear'}, {'name': 'relu', 'map': 'relu', 'dim': 64}]
        res = run_sweep(caches, heads, save_dir=d, bundle=bundle)
        for name in ('lin', 'relu'):
            for seed in (0, 1):
                ckpt = torch.load(os.path.join(d, name, 'seed{}.pt'.format(seed)))
                self.assertTrue(torch.equal(ckpt['model']['w'], torch.full((3,), float(seed))))
                self.assertEqual(ckpt['model_config']['seed'], [seed])
                self.assertEqual(ckpt['config']['name'], name)
                load_head(os.path.join(d, name, 'seed{}.pt'.format(seed)))  # head part still loads alone
            with open(os.path.join(d, name, 'summary.json')) as fh:
                summary = json.load(fh)
            self.assertEqual(sorted(summary['seeds']), ['0', '1'])
            self.assertEqual(summary['summary']['final_per_seed'], res['heads'][name]['summary']['final_per_seed'])


class TestSweepScript(unittest.TestCase):

    def setUp(self):
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tools'))

    def test_make_train_config(self):
        from run_kernel_sweep import make_train_config
        base = {'model_name': 'InfLoRA_KH', 'kernel_head': {'map': 'relu'}, 'init_epoch': 20, 'epochs': 20,
                'seed': [0, 1, 2], 'total_sessions': 10}
        cfg = make_train_config(base, 5, 1, '0')
        self.assertEqual(cfg['model_name'], 'InfLoRA')
        self.assertNotIn('kernel_head', cfg)
        self.assertEqual((cfg['init_epoch'], cfg['epochs'], cfg['seed'], cfg['device']), (5, 5, [1], '0'))
        self.assertEqual(cfg['log_tag'], 'kh_base_e5')
        self.assertFalse(cfg['debug'])
        self.assertEqual(base['epochs'], 20)  # input not mutated
        smoke = make_train_config(base, 1, 0, '0', smoke=True)
        self.assertTrue(smoke['debug'])
        self.assertEqual(smoke['log_tag'], 'kh_base_e1_smoke')

    def test_has_all_checkpoints(self):
        import tempfile
        from run_kernel_sweep import has_all_checkpoints
        d = tempfile.mkdtemp()
        for t in range(2):
            open(os.path.join(d, 'task_{}.pth'.format(t)), 'w').close()
        self.assertTrue(has_all_checkpoints(d, 2))
        self.assertFalse(has_all_checkpoints(d, 3))

    def test_run_sweep_and_table(self):
        from eval_kernel_heads import format_table, run_sweep
        steps, known = [], 0
        for t in range(2):
            total = known + 2
            tr = blobs(30, list(range(known, total)), 8, seed=t)
            te = blobs(10, list(range(total)), 8, seed=10 + t)
            steps.append({'known': known, 'total': total, 'train_f': tr[0], 'train_y': tr[1],
                          'test_f': te[0], 'test_y': te[1], 'base_pred': te[1].clone()})
            known = total
        cache = {'meta': {'class_num': 2}, 'steps': steps}
        heads = [{'name': 'lin', 'map': 'linear'}, {'name': 'bad', 'map': 'rbf', 'dim': 63}]
        res = run_sweep([cache, cache], heads)
        self.assertEqual(res['heads']['lin']['summary']['n_seeds'], 2)
        self.assertIn('error', res['heads']['bad'])  # a failing head is reported, not fatal
        table = format_table(res)
        self.assertIn('InfLoRA (concat heads)', table)
        self.assertIn('FAILED', table)


class TestRepoIntegration(unittest.TestCase):
    """Needs the full training environment (timm etc.); skipped if imports fail."""

    def test_factory_registration(self):
        try:
            from utils import factory
            from methods.inflora_kh import InfLoRA_KH
        except Exception as e:  # pragma: no cover - environment dependent
            self.skipTest('training dependencies not importable: {}'.format(e))
        import inspect
        self.assertIn("'inflora_kh': InfLoRA_KH", inspect.getsource(factory.get_model))

    def test_dual_gpm_with_installed_numpy(self):
        """InfLoRA's DualGPM must keep numpy bases (numpy>=2 returns tensors if given tensors)."""
        try:
            from methods.inflora import InfLoRA
        except Exception as e:  # pragma: no cover - environment dependent
            self.skipTest('training dependencies not importable: {}'.format(e))
        from types import SimpleNamespace
        rng = np.random.RandomState(0)
        state = SimpleNamespace(lamb=0.95, lame=1.0, total_sessions=3, feature_list=[], project_type=[])
        for t in range(3):
            state._cur_task = t
            mats = []
            for _ in range(2):  # two "layers"
                x = rng.randn(16, 200) * np.linspace(3, 0.1, 16)[:, None]
                mats.append(x @ x.T / 200)
            InfLoRA.update_DualGPM(state, mats)
            for f in state.feature_list:
                self.assertIsInstance(f, np.ndarray)
                Uf = torch.Tensor(np.dot(f, f.transpose()))  # what InfLoRA._train does next
                self.assertEqual(tuple(Uf.shape), (16, 16))
                self.assertTrue(np.allclose(f.T @ f, np.eye(f.shape[1]), atol=1e-6))


if __name__ == '__main__':
    unittest.main()
