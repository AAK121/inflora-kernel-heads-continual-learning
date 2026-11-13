"""Phase 0 metrics (report/kernel_inflora.tex, Sec. "Phase 0"). Rows of every token matrix are tokens.

    novelty        rho   = tr((I-P) C) / tr(C)                  energy of the new task outside M
    usable signal  nu    = ||X_new_heldout B^T||^2 / ||X_new_heldout||^2    (B: orthonormal rows)
    leakage        iota  = ||X_old_heldout B^T||^2 / ||X_old_heldout||^2
    protection     ||(I-P) X_old||^2 / ||X_old||^2           kernel block: the GP posterior variance
    overlap        mean cos^2 of principal angles between top-k second-moment subspaces
"""
import torch


def energy_fraction(X, B):
    """Share of the energy of tokens X (n, p) inside the row space of B (r, p), rows orthonormal."""
    return ((X @ B.t()) ** 2).sum().item() / (X ** 2).sum().clamp_min(1e-300).item()


def residual_fraction(gpm, X):
    """Share of the energy of tokens X outside the protected subspace of `gpm`."""
    return (gpm.free_part(X.t()) ** 2).sum().item() / (X ** 2).sum().clamp_min(1e-300).item()


def novelty(gpm, C):
    return torch.trace(gpm.free_part(C)).item() / torch.trace(C).clamp_min(1e-300).item()


def top_subspace(X, k):
    """Orthonormal basis (p, k) of the top-k uncentered principal directions of tokens X."""
    k = min(k, *X.shape)
    if min(X.shape) <= k + 10:
        return torch.linalg.svd(X, full_matrices=False)[2][:k].t()
    return torch.svd_lowrank(X, q=k + 10, niter=4, M=None)[2][:, :k]


def subspace_overlap(Xa, Xb, k):
    """Mean squared cosine of the principal angles between the top-k subspaces of Xa and Xb (1 = identical)."""
    Ua, Ub = top_subspace(Xa, k), top_subspace(Xb, k)
    return ((Ua.t() @ Ub) ** 2).sum().item() / min(Ua.shape[1], Ub.shape[1])


def effective_rank(X):
    """(tr S)^2 / tr(S^2) of the centered covariance S."""
    Xc = X - X.mean(dim=0)
    S = Xc.t() @ Xc
    return (torch.trace(S) ** 2 / (S ** 2).sum()).item()


def nearest_neighbour_median(Q, R, chunk=2048):
    """Median over rows of Q of the distance to the nearest row of R."""
    nn = [torch.cdist(Q[i:i + chunk], R).min(dim=1).values for i in range(0, len(Q), chunk)]
    return torch.cat(nn).median().item()


def cosine_structure(z_by_task):
    """Within- vs between-task cosines of unit vectors; gap = (E d_b^2 - E d_w^2) / E d_w^2, d^2 = 2(1 - c)."""
    within, between = [], []
    for i, zi in enumerate(z_by_task):
        c = zi @ zi.t()
        iu = torch.triu_indices(len(zi), len(zi), offset=1, device=zi.device)
        within.append(c[iu[0], iu[1]])
        for zj in z_by_task[i + 1:]:
            between.append((zi @ zj.t()).flatten())
    w = torch.cat(within)
    out = {'cos_within_mean': w.mean().item(), 'cos_within_std': w.std().item()}
    if between:
        b = torch.cat(between)
        dw, db = 2 * (1 - w.mean()), 2 * (1 - b.mean())
        out.update({'cos_between_mean': b.mean().item(), 'cos_between_std': b.std().item(),
                    'gap_ratio': ((db - dw) / dw.clamp_min(1e-12)).item()})
    return out


def linearization(K, Z):
    """Least-squares fit K ~ a 11^T + b Z Z^T + v I (Frobenius); returns the coefficients and the residual.
    A spectral residual near 0 means the kernel is effectively affine in the cosine plus a private part."""
    n = len(K)
    basis = [torch.ones_like(K), Z @ Z.t(), torch.eye(n, dtype=K.dtype, device=K.device)]
    G = torch.tensor([[(u * v).sum().item() for v in basis] for u in basis], dtype=torch.float64)
    h = torch.tensor([(u * K).sum().item() for u in basis], dtype=torch.float64)
    a, b, v = torch.linalg.lstsq(G, h.unsqueeze(1)).solution.squeeze(1).tolist()
    R = K - (a * basis[0] + b * basis[1] + v * basis[2])
    spec = lambda m: torch.linalg.eigvalsh(m).abs().max().item()  # noqa: E731
    return {'a': a, 'b': b, 'upsilon': v,
            'residual_spectral': spec(R) / spec(K),
            'residual_fro': (R.norm() / K.norm()).item()}
