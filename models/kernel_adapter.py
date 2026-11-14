"""Kernel feature map between InfLoRA's two factors: delta_t(x) = A_t psi_t(B_t x).

B_t (code: lora_A) stays InfLoRA's frozen linear projection onto the free subspace. psi_t lifts the r
coordinates u = B_t x to m random features, and A_t (code: lora_B, now dim x m) is trained on them, so each
task's adapter is a random-feature kernel machine on its own subspace instead of a linear map.

psi(0) = 0 for every kind, so an input with B_t x = 0 still gets delta_t(x) = 0: InfLoRA's
interference argument carries over, with leakage bounded by Lip(psi) * ||B_t x||.

Kinds (each normalised so that E||psi(u)||^2 ~= ||u||^2, i.e. the same scale as the linear adapter):
  linear  psi(u) = Omega u / sqrt(m)                         control: still rank r overall
  relu    psi(u) = sqrt(2/m) relu(Omega u)                   arc-cosine kernel, scale-free
  rff     psi(u) = s sqrt(2/m) [cos(Omega u / s) - 1, sin(Omega u / s)]
          centred random Fourier features of the Gaussian kernel with bandwidth s = sigma_mult * sqrt(E||u||^2)
          (s is set per task from the input second moment when B_t is chosen; see calibrate)

Output scale (psi_scale): Adam moves every entry of A_t by ~lr, so with m columns instead of r the effective
update of the composed rank-r map grows ~sqrt(m/r) per step. psi_scale multiplies psi's output; under Adam that
is the same as multiplying the A_t step by psi_scale. 'auto' = sqrt(r/m) matches InfLoRA's step size.
"""
import itertools
import math

import torch
import torch.nn as nn

KINDS = ('linear', 'relu', 'rff')
_draw_index = itertools.count()  # distinct Omega per TaskFeatureMap (layer, k/v) within a run


class TaskFeatureMap(nn.Module):
    """One frozen psi per task. Omega and sigma are buffers, so they are saved in the checkpoints."""

    def __init__(self, kind, r, m, n_tasks, sigma_mult=1.0, scale=1.0):
        super().__init__()
        if kind not in KINDS:
            raise ValueError(f'unknown psi kind {kind!r}; expected one of {KINDS}')
        if kind == 'rff' and m % 2:
            raise ValueError('rff needs an even psi_dim (cos and sin halves)')
        self.kind, self.r, self.m, self.sigma_mult, self.scale = kind, r, m, sigma_mult, scale
        n_freq = m // 2 if kind == 'rff' else m
        # Own generator, derived from the run seed (trainer._set_random): reproducible per run, and the
        # global RNG is left alone, so data order and LoRA init match the plain InfLoRA run exactly.
        g = torch.Generator().manual_seed((torch.initial_seed() * 1_000_003 + next(_draw_index)) % 2 ** 63)
        self.register_buffer('omega', torch.randn(n_tasks, n_freq, r, generator=g))
        self.register_buffer('sigma', torch.ones(n_tasks))

    @torch.no_grad()
    def calibrate(self, task, proj, second_moment):
        """Set task's bandwidth from E||u||^2 = tr(B C B^T), with C the uncentered input second moment."""
        c = second_moment.to(proj.device, proj.dtype)
        mean_sq = torch.einsum('rd,de,re->', proj, c, proj).clamp_min(1e-12)
        self.sigma[task] = self.sigma_mult * mean_sq.sqrt()

    def forward(self, u):
        """u: (..., T, r) coordinates for tasks 0..T-1 -> (..., T, m)."""
        return self.scale * self._features(u)

    def _features(self, u):
        n_tasks = u.shape[-2]
        z = torch.einsum('...tr,tfr->...tf', u, self.omega[:n_tasks])
        if self.kind == 'linear':
            return z / math.sqrt(self.m)
        if self.kind == 'relu':
            return math.sqrt(2.0 / self.m) * torch.relu(z)
        s = self.sigma[:n_tasks, None]
        zs = z / s
        return (s * math.sqrt(2.0 / self.m)) * torch.cat([torch.cos(zs) - 1.0, torch.sin(zs)], dim=-1)

    def extra_repr(self):
        return f'kind={self.kind}, r={self.r}, m={self.m}, sigma_mult={self.sigma_mult}, scale={self.scale:.4g}'


def psi_config(args):
    """Read the psi options from the run args; None means plain InfLoRA."""
    kind = str(args.get('psi', 'none')).lower()
    if kind == 'none':
        return None
    m = int(args.get('psi_dim', 128))
    scale = args.get('psi_scale', 1.0)
    scale = math.sqrt(int(args.get('rank', 10)) / m) if str(scale).lower() == 'auto' else float(scale)
    return {'kind': kind, 'm': m, 'sigma_mult': float(args.get('psi_sigma', 1.0)), 'scale': scale}


def psi_tag(args):
    """Suffix for log directories so psi runs don't overwrite the baseline."""
    cfg = psi_config(args)
    if cfg is None:
        return ''
    tag = f"_psi-{cfg['kind']}{cfg['m']}"
    tag += f"_s{cfg['sigma_mult']:g}" if cfg['kind'] == 'rff' else ''
    scale = args.get('psi_scale', 1.0)
    if str(scale).lower() == 'auto':
        tag += '_xauto'
    elif float(scale) != 1.0:
        tag += f'_x{float(scale):g}'
    return tag
