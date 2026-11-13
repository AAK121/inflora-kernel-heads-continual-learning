"""Tests for kernel_bt/ (Phase 0 of the kernelised B_t). Tiny synthetic tensors only: no data, no weights.

Run from the repo root:  python -m pytest tests/test_kernel_bt.py -v
"""
import math
import os
import sys
import types
import unittest

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from kernel_bt.diagnostics import energy_fraction, linearization, novelty, residual_fraction  # noqa: E402
from kernel_bt.dualgpm import DualGPM, inflora_threshold  # noqa: E402
from kernel_bt.lifts import KernelLift  # noqa: E402
from kernel_bt import phase0  # noqa: E402

f64 = torch.float64


def low_rank_moment(p, rank, n, g, shift=None):
    basis = torch.linalg.qr(torch.randn(p, rank, generator=g, dtype=f64))[0]
    X = torch.randn(n, rank, generator=g, dtype=f64) @ basis.t() + 0.05 * torch.randn(n, p, generator=g, dtype=f64)
    if shift is not None:
        X = X + shift
    return X, X.t() @ X / n


class TestLift(unittest.TestCase):

    def setUp(self):
        g = torch.Generator().manual_seed(0)
        self.x = torch.randn(400, 16, generator=g, dtype=f64) + 2.0

    def test_rff_unit_norm_and_kernel(self):
        lf = KernelLift('rbf', dim=8192, sigma_mult=1.0, seed=1).calibrate(self.x)
        phi = lf.phi(self.x[:50])
        self.assertTrue(torch.allclose(phi.norm(dim=1), torch.ones(50, dtype=f64), atol=1e-12))
        z = lf.z(self.x[:50])
        err = (phi @ phi.t() - lf.exact_kernel(z, z)).abs().max().item()
        self.assertLess(err, 6 / math.sqrt(4096))

    def test_relu_kernel(self):
        lf = KernelLift('relu', dim=20000, seed=1).calibrate(self.x)
        phi = lf.phi(self.x[:30])
        z = lf.z(self.x[:30])
        self.assertLess((phi @ phi.t() - lf.exact_kernel(z, z)).abs().max().item(), 0.08)

    def test_frozen_and_reproducible(self):
        a = KernelLift('rbf', dim=64, seed=3).calibrate(self.x)
        b = KernelLift('rbf', dim=64, seed=3).calibrate(self.x)
        before = a.phi(self.x[:5]).clone()
        a.phi(self.x + 10)  # using the lift never changes it
        self.assertTrue(torch.equal(before, a.phi(self.x[:5])))
        self.assertTrue(torch.equal(before, b.phi(self.x[:5])))

    def test_hybrid_alpha_and_pre_projection(self):
        lf = KernelLift('rbf', dim=64, gamma=0.25, pre_dim=4, seed=0).calibrate(self.x)
        self.assertEqual(lf.S.shape, (4, 16))
        psi = lf.psi(self.x)
        ratio = (psi[:, 16:] ** 2).sum(1).mean() / (self.x ** 2).sum(1).mean()
        self.assertAlmostEqual(ratio.item(), 0.25, places=6)

    def test_bad_args(self):
        with self.assertRaises(ValueError):
            KernelLift('poly')
        with self.assertRaises(ValueError):
            KernelLift('rbf', dim=65)


class TestDualGPM(unittest.TestCase):

    def _tasks(self, p=32, n_tasks=6):
        g = torch.Generator().manual_seed(1)
        return [low_rank_moment(p, 6, 300, g)[1] for _ in range(n_tasks)]

    def test_matches_inflora_update_dualgpm(self):
        try:
            from methods.inflora import InfLoRA
        except Exception as e:  # pragma: no cover - environment dependent
            self.skipTest('methods.inflora not importable here: {}'.format(e))
        tasks, T = self._tasks(), 6
        ref = types.SimpleNamespace(lamb=0.95, lame=1.0, total_sessions=T, feature_list=[], project_type=[])
        ours = DualGPM(32)
        modes = set()
        for t, C in enumerate(tasks):
            ref._cur_task = t
            InfLoRA.update_DualGPM(ref, [C.numpy().copy()])
            ours.update(C, inflora_threshold(0.95, 1.0, t, T))
            U = torch.from_numpy(np.asarray(ref.feature_list[0]))
            P_ref = U @ U.t() if ref.project_type[0] == 'remove' else torch.eye(32, dtype=f64) - U @ U.t()
            P_ours = torch.eye(32, dtype=f64) - ours.free_part(torch.eye(32, dtype=f64))
            self.assertEqual(ours.mode, ref.project_type[0])
            self.assertTrue(torch.allclose(P_ref, P_ours, atol=1e-6), 'task {}'.format(t))
            modes.add(ours.mode)
        self.assertEqual(modes, {'remove', 'retain'})  # the remove -> retain switch was exercised

    def test_top_free_is_orthogonal_to_protected(self):
        tasks = self._tasks()
        g = DualGPM(32)
        g.update(tasks[0], 0.97)
        B = g.top_free(tasks[1], 5)
        P = torch.eye(32, dtype=f64) - g.free_part(torch.eye(32, dtype=f64))
        self.assertLess((B @ P).abs().max().item(), 1e-8)
        self.assertTrue(torch.allclose(B @ B.t(), torch.eye(5, dtype=f64), atol=1e-8))

    def test_first_task_fix_and_cap(self):
        C = torch.eye(8, dtype=f64)  # flat spectrum: task 1 wants ~all directions
        fixed, buggy = DualGPM(8), DualGPM(8, fix_first_task=False)
        fixed.update(C, 0.99)
        buggy.update(C, 0.99)
        self.assertEqual(fixed.mode, 'retain')
        self.assertEqual(fixed.protected_dim, 7)          # 7 protected, basis = the 1 free direction
        self.assertEqual(buggy.basis.shape[1], 7)         # original: protected basis labelled 'retain'
        capped = DualGPM(8, kmax=2)
        capped.update(C, 0.99)
        self.assertEqual((capped.mode, capped.protected_dim), ('remove', 2))


class TestMetrics(unittest.TestCase):

    def test_energy_and_residual_fractions(self):
        X = torch.tensor([[1.0, 0, 0], [0, 2.0, 0]], dtype=f64)
        B = torch.tensor([[1.0, 0, 0]], dtype=f64)
        self.assertAlmostEqual(energy_fraction(X, B), 1 / 5)
        g = DualGPM(3)
        g.basis, g.mode = B.t(), 'remove'
        self.assertAlmostEqual(residual_fraction(g, X), 4 / 5)
        self.assertAlmostEqual(novelty(g, X.t() @ X), 4 / 5)

    def test_linearization_recovers_affine_kernel(self):
        g = torch.Generator().manual_seed(0)
        Z = torch.nn.functional.normalize(torch.randn(60, 10, generator=g, dtype=f64), dim=1)
        K = 0.3 + 0.5 * Z @ Z.t() + 0.2 * torch.eye(60, dtype=f64)
        fit = linearization(K, Z)
        self.assertAlmostEqual(fit['a'], 0.3, places=8)
        self.assertAlmostEqual(fit['b'], 0.5, places=8)
        self.assertLess(fit['residual_spectral'], 1e-8)

    def test_locality_mechanism(self):
        """Two tasks along the SAME direction but in different places: linear novelty is small after protecting
        task 1, kernel novelty is large (Lemma 'Separated inputs are nearly orthogonal in feature space')."""
        g = torch.Generator().manual_seed(0)
        e1 = torch.zeros(12, dtype=f64)
        e1[0] = 1.0
        old = 4 * e1 + 0.3 * torch.randn(800, 12, generator=g, dtype=f64)
        new = -4 * e1 + 0.3 * torch.randn(800, 12, generator=g, dtype=f64)
        lin, ker = DualGPM(12), DualGPM(512)
        lf = KernelLift('rbf', dim=512, sigma_mult=0.5, seed=0).calibrate(old)
        lin.update(old.t() @ old / 800, 0.95)
        Fo, Fn = lf.phi(old), lf.phi(new)
        ker.update(Fo.t() @ Fo / 800, 0.95)
        rho_lin = novelty(lin, new.t() @ new / 800)
        rho_ker = novelty(ker, Fn.t() @ Fn / 800)
        self.assertLess(rho_lin, 0.2)
        self.assertGreater(rho_ker, rho_lin + 0.4)


def fake_cache(n_tasks=3, layers=(1, 2), d=16, n_tr=300, n_te=120, seed=0):
    g = torch.Generator().manual_seed(seed)
    tasks = []
    for t in range(n_tasks):
        shift = torch.randn(d, generator=g, dtype=f64)
        blocks = {}
        for l in layers:
            X, C = low_rank_moment(d, 4, n_tr, g, shift)
            Xte = low_rank_moment(d, 4, n_te, g, shift)[0]
            blocks[l] = {'C': C.float(), 'n': n_tr, 'train': X.half(), 'test': Xte.half()}
        tasks.append({'task': t, 'classes': [2 * t, 2 * t + 2], 'layers': blocks})
    return tasks


class TestPhase0(unittest.TestCase):

    def test_analyze_and_summary_on_fake_cache(self):
        specs = phase0.lift_specs(['rbf', 'relu'], [0.5, 1.0], [0, 4])
        self.assertEqual(len(specs), 6)
        opts = {'rff_dim': 64, 'rank': 2, 'kmax_phi': 8, 'angles_k': 3, 'pool_per_task': 50,
                'structure_tokens': 80, 'structure_per_task': 20}
        res = phase0.analyze(fake_cache(), 3, [1, 2], specs, opts)
        lin = res['layers']['1']['linear']
        self.assertEqual([r['task'] for r in lin], [1, 2, 3])
        for key in ('rho', 'nu', 'iota', 'prot_old_heldout', 'overlap_cross'):
            self.assertIn(key, lin[1])
        kt = res['layers']['2']['lifts']['rbf_c0.5_q4']['per_task'][2]
        for key in ('nu', 'iota', 'nu_hybrid', 'iota_hybrid', 'kernel_share', 'nn_within', 'nn_between'):
            self.assertTrue(0 <= kt[key] or key.startswith('nn'), key)
        self.assertIn('linearization', res['layers']['1']['lifts']['relu'])
        self.assertIn('go', res['verdict'])
        text = phase0.format_summary(res)
        self.assertIn('Verdict', text)
        import json
        json.dumps(res)  # serialisable

    def test_verdict_rule(self):
        def rows(nu, iota, key_nu='nu', key_io='iota'):
            return [{'task': t, key_nu: nu, key_io: iota} for t in (2, 3, 4)]
        res = {str(l): {'linear': rows(0.10, 0.01),
                        'lifts': {'a': {'per_task': rows(0.20 if l < 3 else 0.11, 0.015, 'nu_hybrid', 'iota_hybrid')},
                                  'b': {'per_task': rows(0.30, 0.10, 'nu_hybrid', 'iota_hybrid')}}}
               for l in (1, 2, 3, 4)}
        v = phase0.verdict(res, 4, phase0.DEFAULT_OPTS)
        self.assertEqual(v['per_lift']['a']['layers_passed'], ['1', '2'])
        self.assertTrue(v['per_lift']['a']['go'])     # 2 of 4 layers = half
        self.assertFalse(v['per_lift']['b']['go'])    # leakage too high everywhere
        self.assertTrue(v['go'])

    def test_run_split_moments_and_sampling(self):
        """Extraction hooks: exact second moment over ALL tokens, and the token sample, from a toy encoder."""
        class Enc(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = torch.nn.ModuleList([torch.nn.Module() for _ in range(2)])
                for b in self.blocks:
                    b.norm1 = torch.nn.Identity()

            def forward(self, x, task):
                for b in self.blocks:
                    x = b.norm1(x) * 2
                return x

        class DS(torch.utils.data.Dataset):
            def __init__(self, data):
                self.data = data

            def __len__(self):
                return len(self.data)

            def __getitem__(self, i):
                return i, self.data[i], 0

        g = torch.Generator().manual_seed(0)
        data = torch.randn(10, 5, 3, generator=g)
        enc, captured = Enc(), {}
        for l, b in zip((1, 2), enc.blocks):
            b.norm1.register_forward_hook(lambda m, i, o, l=l: captured.__setitem__(l, o))
        out = phase0._run_split(enc, captured, DS(data), [1, 2], 20, True, torch.Generator().manual_seed(0),
                                'cpu', 4, 0)
        flat = data.reshape(-1, 3).double()
        self.assertTrue(torch.allclose(out[1]['C'].double(), flat.t() @ flat / 50, atol=1e-5))
        self.assertTrue(torch.allclose(out[2]['C'].double(), 4 * flat.t() @ flat / 50, atol=1e-5))
        self.assertEqual(out[1]['n'], 50)
        self.assertEqual(tuple(out[1]['tokens'].shape), (20, 3))


if __name__ == '__main__':
    unittest.main()
