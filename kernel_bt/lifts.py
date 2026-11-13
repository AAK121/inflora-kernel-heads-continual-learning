"""Frozen kernel lift for one attention block (report/kernel_inflora.tex, Sec. "The kernel feature map").

    z(x)   = S (x - mu) / ||S (x - mu)||                 normalised input, S = I or top-q PCs
    phi(z) = [cos(Omega z / sigma), sin(Omega z / sigma)] / sqrt(D/2)    orthogonal RFF ('rbf')
           = sqrt(2/D) relu(G z)                                         random ReLU ('relu')
    psi(x) = [x; alpha * phi(z(x))]                       hybrid lift, alpha^2 = gamma * E||x||^2

mu, S, sigma and alpha are estimated on task-1 training tokens by calibrate() and then frozen:
every protected basis and every stored B_s is expressed in these coordinates.
"""
import math

import torch


def orthogonal_gaussian(rows, cols, generator, dtype=torch.float64):
    """Orthogonal random features (Yu et al. 2016): stacked random orthogonal blocks with chi row norms,
    so every row is marginally N(0, I)."""
    blocks = []
    for _ in range(math.ceil(rows / cols)):
        q, r = torch.linalg.qr(torch.randn(cols, cols, generator=generator, dtype=dtype))
        q = q * torch.sign(torch.diagonal(r)).unsqueeze(0)
        norms = torch.randn(cols, cols, generator=generator, dtype=dtype).norm(dim=1)
        blocks.append(norms.unsqueeze(1) * q)
    return torch.cat(blocks, dim=0)[:rows]


def median_distance(z, max_samples=2000, generator=None):
    if len(z) > max_samples:
        z = z[torch.randperm(len(z), generator=generator)[:max_samples].to(z.device)]
    d = torch.cdist(z, z)
    iu = torch.triu_indices(len(z), len(z), offset=1, device=z.device)
    return d[iu[0], iu[1]].median().item()


class KernelLift:
    """phi and psi for one block. Not an nn.Module on purpose: Phase 0 only needs tensors."""

    def __init__(self, kind='rbf', dim=2048, sigma_mult=1.0, pre_dim=0, gamma=1.0, seed=0):
        if kind not in ('rbf', 'relu'):
            raise ValueError('kind must be rbf or relu, got {}'.format(kind))
        if kind == 'rbf' and dim % 2:
            raise ValueError('rbf needs an even dim, got {}'.format(dim))
        self.kind, self.dim, self.sigma_mult = kind, dim, sigma_mult
        self.pre_dim, self.gamma, self.seed = pre_dim, gamma, seed
        self.mu = self.S = self.Omega = None
        self.sigma = self.alpha = None

    @property
    def name(self):
        tag = 'rbf_c{:g}'.format(self.sigma_mult) if self.kind == 'rbf' else 'relu'
        return tag + ('_q{}'.format(self.pre_dim) if self.pre_dim else '')

    def calibrate(self, x):
        """x: task-1 training tokens (n, d). Sets mu, S, Omega, sigma, alpha; afterwards the lift is frozen."""
        x = x.double()
        g = torch.Generator().manual_seed(self.seed)
        self.mu = x.mean(dim=0)
        xc = x - self.mu
        if self.pre_dim and self.pre_dim < x.shape[1]:
            _, evecs = torch.linalg.eigh(xc.t() @ xc)
            self.S = evecs[:, -self.pre_dim:].flip(1).t().contiguous()  # (q, d), top-q principal directions
        else:
            self.S = None
        q = self.S.shape[0] if self.S is not None else x.shape[1]
        if self.kind == 'rbf':
            self.Omega = orthogonal_gaussian(self.dim // 2, q, g).to(x.device)
            self.sigma = self.sigma_mult * median_distance(self.z(x), generator=g)
        else:
            self.Omega = torch.randn(self.dim, q, generator=g, dtype=torch.float64).to(x.device)
            self.sigma = float('nan')
        self.alpha = math.sqrt(self.gamma * (x ** 2).sum(dim=1).mean().item())
        return self

    def to(self, device):
        for k in ('mu', 'S', 'Omega'):
            v = getattr(self, k)
            if v is not None:
                setattr(self, k, v.to(device))
        return self

    def z(self, x):
        h = x.to(self.mu.dtype) - self.mu
        if self.S is not None:
            h = h @ self.S.t()
        return h / h.norm(dim=1, keepdim=True).clamp_min(1e-12)

    def phi(self, x):
        u = self.z(x) @ self.Omega.t()
        if self.kind == 'rbf':
            u = u / self.sigma
            return torch.cat([torch.cos(u), torch.sin(u)], dim=1) / math.sqrt(self.dim // 2)
        return torch.relu(u) * math.sqrt(2.0 / self.dim)

    def psi(self, x):
        return torch.cat([x.to(self.mu.dtype), self.alpha * self.phi(x)], dim=1)

    def exact_kernel(self, za, zb):
        """The kernel phi approximates (D -> infinity), on normalised inputs z:
        rbf: exp(-||z - z'||^2 / (2 sigma^2));  relu: arc-cosine order 1, (sin t + (pi - t) cos t) / pi."""
        c = (za @ zb.t()).clamp(-1, 1)
        if self.kind == 'rbf':
            return torch.exp(-(1 - c) / self.sigma ** 2)
        t = torch.arccos(c)
        return (torch.sin(t) + (math.pi - t) * c) / math.pi
