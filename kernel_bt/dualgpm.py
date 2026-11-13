"""InfLoRA's DualGPM (methods/inflora.py, update_DualGPM) in torch, for a space of any dimension p.

Same rule as the original, including its "energy" = squared singular values of the second-moment
matrix C, and the remove/retain switch at p/2. Two additions (report/kernel_inflora.tex):
  - fix_first_task: if task 1 selects k >= p/2 directions, the original stores the PROTECTED basis
    labelled 'retain' (= free basis); we store its complement instead. Never triggered at p = 768.
  - kmax: cap on the number of directions protected per task (for the flat RFF spectrum tail).

Left singular vectors of (I-P)C are computed as eigenvectors of (I-P)C^2(I-P), whose eigenvalues are
the squared singular values the original uses. The same spectrum gives B_t (top_free) and the update.
"""
import torch


def inflora_threshold(lamb, lame, task_index, n_tasks):
    """Threshold used after learning task `task_index` (0-based), as in update_DualGPM."""
    return (lame - lamb) * task_index / n_tasks + lamb


def _eigh_desc(m):
    evals, evecs = torch.linalg.eigh(m)
    return evals.flip(0).clamp_min(0), evecs.flip(1)


def _exclusive_cumsum(v):
    return torch.cumsum(v, 0) - v


class DualGPM:
    def __init__(self, dim, kmax=None, fix_first_task=True):
        self.dim, self.kmax, self.fix_first_task = dim, kmax, fix_first_task
        self.basis, self.mode = None, None   # mode 'remove': basis spans M; 'retain': basis spans M-perp

    @property
    def protected_dim(self):
        if self.basis is None:
            return 0
        return self.basis.shape[1] if self.mode == 'remove' else self.dim - self.basis.shape[1]

    def free_part(self, a):
        """(I - P) a for a matrix a with `dim` rows."""
        if self.basis is None:
            return a
        b = self.basis.to(a.dtype)
        coef = b.t() @ a
        return a - b @ coef if self.mode == 'remove' else b @ coef

    def free_spectrum(self, C):
        """(squared singular values, left singular vectors) of (I - P) C, in descending order."""
        h = self.free_part(C)
        return _eigh_desc(h @ h.t())

    def top_free(self, C, r, spectrum=None):
        """InfLoRA's B_t without the 1/sqrt(3) scale: (r, dim) with orthonormal rows in M-perp."""
        _, U = spectrum if spectrum is not None else self.free_spectrum(C)
        return U[:, :r].t().contiguous()

    def _cap(self, r):
        return min(r, self.kmax) if self.kmax else r

    def update(self, C, threshold, spectrum=None):
        """Protect the new task's second moment C (dim x dim). Returns the number of directions moved."""
        p = C.shape[0]
        total = (C ** 2).sum()
        if self.basis is None:
            lam, U = _eigh_desc(C)
            r = max(int((torch.cumsum(lam ** 2 / total, 0) < threshold).sum()), 1)
            r = self._cap(r)
            if r < p / 2:
                self.basis, self.mode = U[:, :r], 'remove'
            else:
                self.basis = U[:, r:] if self.fix_first_task else U[:, :r]
                self.mode = 'retain'
            moved = r
        else:
            s2, U = spectrum if spectrum is not None else self.free_spectrum(C)
            ratio = s2 / total
            if self.mode == 'remove':
                before = (total - s2.sum()) / total + _exclusive_cumsum(ratio)
                r = self._cap(int((before < threshold).sum()))
                if r:
                    self.basis = torch.cat([self.basis.to(U.dtype), U[:, :r]], dim=1)[:, :p]
            else:
                before = s2.sum() / total - _exclusive_cumsum(ratio)
                r = self._cap(int((before >= 1 - threshold).sum()))
                if r:
                    V = self.basis.to(U.dtype)
                    act = V - U[:, :r] @ (U[:, :r].t() @ V)
                    _, W = _eigh_desc(act @ act.t())
                    self.basis = W[:, :V.shape[1] - r]
            moved = r
        if self.mode == 'remove' and self.basis.shape[1] > p / 2:
            m = self.basis.shape[1]
            eye = torch.eye(p, dtype=self.basis.dtype, device=self.basis.device)
            _, W = _eigh_desc(eye - self.basis @ self.basis.t())
            self.basis, self.mode = W[:, :p - m], 'retain'
        return moved
