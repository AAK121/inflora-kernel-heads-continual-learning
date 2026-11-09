"""Exemplar-free analytic kernel classifier head for class-incremental learning.

The head replaces InfLoRA's concatenated per-task linear heads at evaluation time.
Backbone features f (CLS token, 768-d) are lifted with a fixed feature map
h = phi(f) in R^D, and two running sums are accumulated over all tasks:

    G = sum h h^T      (D x D)
    Q = sum h y^T      (D x C, y one-hot over all classes seen so far)

Prediction is ridge regression onto one-hot targets (RanPAC-style):

    W = (G + lambda I)^-1 Q,   logits = phi(f) W

G and Q are sums, so the head does not depend on task order and stores no
exemplars. The feature map is calibrated once (first task) and then frozen.

Only torch is required; the code sticks to APIs available in torch 1.10.
"""
import logging
import math
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

EPS = 1e-12
DEFAULT_RIDGE_GRID = [10.0 ** (k / 2.0) for k in range(-16, 17)]  # 1e-8 ... 1e8, half-decade steps
RIDGE_MODES = ('gcv', 'val')


# ---------------------------------------------------------------------------
# Feature maps
# ---------------------------------------------------------------------------

class FeatureMap(nn.Module):
    """Base class: optional centering / L2 normalization, then a fixed lift.

    input_norm: 'none' | 'l2' | 'center_l2'. The centering mean is estimated
    in fit() (first task only) and frozen afterwards.
    """

    def __init__(self, in_dim: int, input_norm: str = 'l2'):
        super().__init__()
        if input_norm not in ('none', 'l2', 'center_l2'):
            raise ValueError('Unknown input_norm {}'.format(input_norm))
        self.in_dim = in_dim
        self.input_norm = input_norm
        self.register_buffer('mean', torch.zeros(in_dim))
        self.register_buffer('fitted', torch.zeros((), dtype=torch.uint8))

    @property
    def out_dim(self) -> int:
        raise NotImplementedError

    @property
    def is_fitted(self) -> bool:
        return bool(self.fitted.item())

    def preprocess(self, f: torch.Tensor) -> torch.Tensor:
        if self.input_norm == 'center_l2':
            f = f - self.mean
        if self.input_norm in ('l2', 'center_l2'):
            f = f / f.norm(dim=1, keepdim=True).clamp_min(EPS)
        return f

    @torch.no_grad()
    def fit(self, f: torch.Tensor) -> None:
        """Calibrate on first-task features. No-op once fitted (the map is frozen)."""
        if self.is_fitted:
            return
        f = f.to(self.mean.device, torch.float32)
        if self.input_norm == 'center_l2':
            self.mean.copy_(f.mean(dim=0))
        self._fit(self.preprocess(f))
        self.fitted.fill_(1)

    def _fit(self, z: torch.Tensor) -> None:
        pass

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        return self.lift(self.preprocess(f))

    def lift(self, z: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class LinearMap(FeatureMap):
    """No lift: h = [z, 1]. The ablation that shows what the nonlinearity adds."""

    @property
    def out_dim(self) -> int:
        return self.in_dim + 1

    def lift(self, z):
        return torch.cat([z, torch.ones_like(z[:, :1])], dim=1)


class RandomReLUMap(FeatureMap):
    """RanPAC random projection: h = ReLU(z W), W ~ N(0, 1)^{in_dim x D}.

    In expectation h^T h' is the order-1 arc-cosine kernel (Cho & Saul, 2009), up to scale.
    """

    def __init__(self, in_dim: int, dim: int, input_norm: str = 'l2', seed: int = 0):
        super().__init__(in_dim, input_norm)
        g = torch.Generator().manual_seed(seed)
        self.register_buffer('W', torch.randn(in_dim, dim, generator=g))

    @property
    def out_dim(self) -> int:
        return self.W.shape[1]

    def lift(self, z):
        return F.relu(z @ self.W)


def _orthogonal_gaussian(rows: int, cols: int, g: torch.Generator) -> torch.Tensor:
    """Orthogonal random features (Yu et al., 2016): stacked orthogonal blocks whose
    rows are rescaled by chi(cols)-distributed norms, so each row is marginally N(0, I)."""
    blocks = []
    remaining = rows
    while remaining > 0:
        q, r = torch.linalg.qr(torch.randn(cols, cols, generator=g))
        q = q * torch.sign(torch.diagonal(r)).unsqueeze(0)  # Haar-uniform orthogonal matrix
        norms = torch.randn(cols, cols, generator=g).norm(dim=1)  # chi(cols) samples
        blocks.append(q.t() * norms.unsqueeze(1))
        remaining -= cols
    return torch.cat(blocks, dim=0)[:rows]


class RFFGaussianMap(FeatureMap):
    """Random Fourier features of the Gaussian kernel exp(-||z - z'||^2 / (2 sigma^2)).

    h = sqrt(2 / D) [cos(Omega z / sigma), sin(Omega z / sigma)], so ||h|| = 1 exactly.
    sigma = bandwidth_mult * median pairwise distance of first-task features,
    unless an absolute `bandwidth` is given.
    """

    def __init__(self, in_dim: int, dim: int, input_norm: str = 'l2', seed: int = 0,
                 orthogonal: bool = True, bandwidth_mult: float = 1.0,
                 bandwidth: Optional[float] = None, median_samples: int = 2000):
        super().__init__(in_dim, input_norm)
        if dim % 2 != 0:
            raise ValueError('RFF dim must be even, got {}'.format(dim))
        g = torch.Generator().manual_seed(seed)
        half = dim // 2
        omega = _orthogonal_gaussian(half, in_dim, g) if orthogonal else torch.randn(half, in_dim, generator=g)
        self.register_buffer('omega', omega)
        self.register_buffer('sigma', torch.tensor(float(bandwidth) if bandwidth else 1.0))
        self.bandwidth_mult = bandwidth_mult
        self.fixed_bandwidth = bandwidth is not None
        self.median_samples = median_samples
        self._seed = seed

    @property
    def out_dim(self) -> int:
        return 2 * self.omega.shape[0]

    def _fit(self, z):
        if self.fixed_bandwidth:
            return
        g = torch.Generator().manual_seed(self._seed + 1)
        idx = torch.randperm(z.shape[0], generator=g)[: self.median_samples].to(z.device)
        med = torch.pdist(z[idx]).median()
        self.sigma.fill_(float(self.bandwidth_mult * med.clamp_min(EPS)))

    def lift(self, z):
        proj = z @ (self.omega.t() / self.sigma)
        scale = math.sqrt(1.0 / self.omega.shape[0])
        return scale * torch.cat([torch.cos(proj), torch.sin(proj)], dim=1)


def build_feature_map(cfg: Dict, in_dim: int) -> FeatureMap:
    kind = cfg.get('map', 'rbf').lower()
    norm = cfg.get('input_norm', 'l2')
    seed = int(cfg.get('seed', 0))
    if kind == 'linear':
        return LinearMap(in_dim, norm)
    if kind == 'relu':
        return RandomReLUMap(in_dim, int(cfg.get('dim', 10000)), norm, seed)
    if kind == 'rbf':
        return RFFGaussianMap(in_dim, int(cfg.get('dim', 10000)), norm, seed,
                              orthogonal=bool(cfg.get('orthogonal', True)),
                              bandwidth_mult=float(cfg.get('bandwidth_mult', 1.0)),
                              bandwidth=cfg.get('bandwidth'))
    raise ValueError('Unknown feature map {}'.format(kind))


# ---------------------------------------------------------------------------
# Analytic head
# ---------------------------------------------------------------------------

class AnalyticKernelHead(nn.Module):
    """Ridge-regression head on a fixed feature map, updated from running sums.

    ridge: a float (fixed lambda), or a selection rule over `ridge_grid`, re-run after every task:
      'gcv' (default, alias 'auto'): generalized cross-validation computed from G and Q alone,
            so it uses the data of ALL tasks seen so far and needs no stored samples;
      'val': RanPAC-style accuracy on a held-out split of the NEW task only (the split
            is folded back into G, Q afterwards). Can be unstable when tasks are small.
    """

    def __init__(self, feature_map: FeatureMap, ridge='auto',
                 ridge_grid: Optional[Sequence[float]] = None, val_fraction: float = 0.2,
                 dtype: torch.dtype = torch.float64, chunk_size: int = 2048, seed: int = 0):
        super().__init__()
        if ridge == 'auto':
            ridge = 'gcv'
        if isinstance(ridge, str) and ridge not in RIDGE_MODES:
            raise ValueError('ridge must be a number or one of {}, got {}'.format(RIDGE_MODES, ridge))
        self.feature_map = feature_map
        self.ridge_mode = ridge
        self.ridge_grid = list(ridge_grid) if ridge_grid else list(DEFAULT_RIDGE_GRID)
        self.val_fraction = val_fraction
        self.chunk_size = chunk_size
        self.seed = seed
        self.dtype = dtype
        D = feature_map.out_dim
        self.register_buffer('G', torch.zeros(D, D, dtype=dtype))
        self.register_buffer('Q', torch.zeros(D, 0, dtype=dtype))
        self.register_buffer('W', torch.zeros(D, 0))
        self.register_buffer('ridge', torch.tensor(1.0 if isinstance(ridge, str) else float(ridge), dtype=torch.float64))
        self.register_buffer('n_seen', torch.zeros((), dtype=torch.long))
        self.history = []  # per-task diagnostics

    # -- helpers ----------------------------------------------------------
    @property
    def device(self) -> torch.device:
        return self.G.device

    @property
    def num_classes(self) -> int:
        return self.Q.shape[1]

    def _lift(self, f: torch.Tensor) -> torch.Tensor:
        return self.feature_map(f.to(self.device, torch.float32)).to(self.dtype)

    def _chunks(self, n: int):
        for s in range(0, n, self.chunk_size):
            yield slice(s, min(s + self.chunk_size, n))

    def _expand_classes(self, num_classes: int) -> None:
        C = self.num_classes
        if num_classes > C:
            pad = torch.zeros(self.Q.shape[0], num_classes - C, dtype=self.dtype, device=self.device)
            self.Q = torch.cat([self.Q, pad], dim=1)

    def _stats(self, f: torch.Tensor, y: torch.Tensor, num_classes: int) -> Tuple[torch.Tensor, torch.Tensor]:
        D = self.G.shape[0]
        G = torch.zeros(D, D, dtype=self.dtype, device=self.device)
        Q = torch.zeros(D, num_classes, dtype=self.dtype, device=self.device)
        for sl in self._chunks(f.shape[0]):
            h = self._lift(f[sl])
            Y = F.one_hot(y[sl].to(self.device).long(), num_classes).to(self.dtype)
            G += h.t() @ h
            Q += h.t() @ Y
        return G, Q

    def _split(self, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Stratified split of indices into (fit, val); every class keeps >= 1 fit sample."""
        g = torch.Generator().manual_seed(self.seed + int(self.n_seen.item()))
        y = y.cpu()
        fit_idx, val_idx = [], []
        for c in torch.unique(y):
            idx = torch.nonzero(y == c, as_tuple=False).flatten()
            idx = idx[torch.randperm(len(idx), generator=g)]
            n_val = int(round(self.val_fraction * len(idx)))
            n_val = min(n_val, len(idx) - 1)
            val_idx.append(idx[:n_val])
            fit_idx.append(idx[n_val:])
        return torch.cat(fit_idx), torch.cat(val_idx)

    # -- public API -------------------------------------------------------
    @torch.no_grad()
    def update(self, f: torch.Tensor, y: torch.Tensor, num_classes: Optional[int] = None) -> None:
        """Add a batch of (features, labels) to the running sums."""
        num_classes = max(num_classes or 0, int(y.max().item()) + 1, self.num_classes)
        self._expand_classes(num_classes)
        G, Q = self._stats(f, y, num_classes)
        self.G += G
        self.Q += Q
        self.n_seen += f.shape[0]

    @torch.no_grad()
    def select_ridge(self, f: torch.Tensor, y: torch.Tensor, num_classes: int) -> float:
        """Pick lambda by accuracy (over all classes seen so far) on a held-out split of the new task."""
        fit_idx, val_idx = self._split(y)
        if len(val_idx) == 0:
            logging.info('Kernel head: no validation samples, keeping lambda={}'.format(float(self.ridge)))
            return float(self.ridge)
        num_classes = max(num_classes, self.num_classes)
        G_fit, Q_fit = self._stats(f[fit_idx], y[fit_idx], num_classes)
        G_fit += self.G
        Q_fit[:, : self.num_classes] += self.Q
        evals, evecs = torch.linalg.eigh(G_fit)
        evals = evals.clamp_min(0)
        P = evecs.t() @ Q_fit  # D x C
        correct = torch.zeros(len(self.ridge_grid), dtype=torch.long)
        f_val, y_val = f[val_idx], y[val_idx].to(self.device)
        for sl in self._chunks(f_val.shape[0]):
            T = self._lift(f_val[sl]) @ evecs  # chunk x D
            for i, lam in enumerate(self.ridge_grid):
                pred = (T @ (P / (evals + lam).unsqueeze(1))).argmax(dim=1)
                correct[i] += (pred == y_val[sl]).sum().cpu()
        accs = correct.double() / len(val_idx)
        best = accs.max()
        # ties -> the largest lambda (most regularized)
        lam = max(l for l, a in zip(self.ridge_grid, accs.tolist()) if a == best.item())
        self.history.append({'val_acc_per_lambda': dict(zip(self.ridge_grid, accs.tolist())), 'lambda': lam})
        return lam

    @torch.no_grad()
    def gcv_scores(self) -> Dict[float, float]:
        """Generalized cross-validation for multi-output ridge, from the running sums only.

        With G = V diag(e) V^T, P = V^T Q, p_i = ||P_i||^2, n samples and one-hot targets
        (so ||Y||_F^2 = n):
            RSS(l) = n - 2 sum_i p_i/(e_i+l) + sum_i e_i p_i/(e_i+l)^2   (= ||Y - H W_l||_F^2)
            df(l)  = sum_i e_i/(e_i+l)                                   (= tr of the hat matrix)
            GCV(l) = (RSS/n) / (1 - df/n)^2
        """
        n = float(self.n_seen.item())
        evals, evecs = torch.linalg.eigh(self.G)
        evals = evals.clamp_min(0)
        p = ((evecs.t() @ self.Q) ** 2).sum(dim=1)
        scores = {}
        for lam in self.ridge_grid:
            d = evals + lam
            rss = n - 2 * (p / d).sum() + (evals * p / d ** 2).sum()
            df = (evals / d).sum()
            denom = (1 - df / n) ** 2
            scores[lam] = float(rss / n / denom) if denom > 1e-12 else float('inf')
        return scores

    @torch.no_grad()
    def solve(self) -> None:
        D = self.G.shape[0]
        lam = float(self.ridge)
        A = self.G + lam * torch.eye(D, dtype=self.dtype, device=self.device)
        try:
            L = torch.linalg.cholesky(A)
            self.W = torch.cholesky_solve(self.Q, L).float()
        except RuntimeError:  # numerically indefinite (tiny lambda / float32): fall back to eigh
            evals, evecs = torch.linalg.eigh(self.G)
            inv = 1.0 / (evals.clamp_min(0) + lam)
            self.W = (evecs @ (inv.unsqueeze(1) * (evecs.t() @ self.Q))).float()

    @torch.no_grad()
    def fit_task(self, f: torch.Tensor, y: torch.Tensor, num_classes: int) -> float:
        """Everything that happens after a task: calibrate the map (first task only),
        choose lambda, add the task to the running sums, and re-solve W. Returns lambda."""
        if not self.feature_map.is_fitted:
            self.feature_map.fit(f)
        if self.ridge_mode == 'val':
            self.ridge.fill_(self.select_ridge(f, y, num_classes))
        self.update(f, y, num_classes)
        if self.ridge_mode == 'gcv':
            scores = self.gcv_scores()
            lam = min(scores, key=scores.get)
            self.history.append({'gcv_per_lambda': scores, 'lambda': lam})
            if lam in (min(self.ridge_grid), max(self.ridge_grid)):
                logging.warning('Kernel head: GCV picked lambda=%g at the edge of the grid; widen ridge_grid', lam)
            self.ridge.fill_(lam)
        self.solve()
        return float(self.ridge)

    @torch.no_grad()
    def logits(self, f: torch.Tensor) -> torch.Tensor:
        out = []
        for sl in self._chunks(f.shape[0]):
            h = self.feature_map(f[sl].to(self.device, torch.float32))
            out.append(h @ self.W)
        return torch.cat(out, dim=0)

    @torch.no_grad()
    def predict(self, f: torch.Tensor) -> torch.Tensor:
        return self.logits(f).argmax(dim=1)

    def load_state_dict(self, state_dict, strict: bool = True):
        # Q and W change shape as classes are added; resize before loading.
        for k in ('Q', 'W'):
            if k in state_dict:
                setattr(self, k, torch.empty_like(state_dict[k], device=self.device))
        return super().load_state_dict(state_dict, strict)


def build_head(cfg: Optional[Dict], in_dim: int, device=None) -> AnalyticKernelHead:
    cfg = dict(cfg or {})
    dtype = torch.float64 if cfg.get('dtype', 'float64') == 'float64' else torch.float32
    ridge = cfg.get('ridge', 'gcv')
    head = AnalyticKernelHead(build_feature_map(cfg, in_dim), ridge=ridge if isinstance(ridge, str) else float(ridge),
                              ridge_grid=cfg.get('ridge_grid'), val_fraction=float(cfg.get('val_fraction', 0.2)),
                              dtype=dtype, chunk_size=int(cfg.get('chunk_size', 2048)), seed=int(cfg.get('seed', 0)))
    if device is not None:
        head.to(device)
    return head


# ---------------------------------------------------------------------------
# Evaluation helpers (shared by the online method and the post-hoc tools)
# ---------------------------------------------------------------------------

def head_metrics(pred: np.ndarray, y: np.ndarray, known_classes: int, class_num: int) -> Dict:
    """Same grouping as utils.toolkit.accuracy, plus task-identification accuracy."""
    out = {'total': float(np.around((pred == y).mean() * 100, 2))}
    for c in range(0, int(y.max()) + 1, class_num):
        m = (y >= c) & (y < c + class_num)
        out['{}-{}'.format(str(c).rjust(2, '0'), str(c + class_num - 1).rjust(2, '0'))] = \
            float(np.around((pred[m] == y[m]).mean() * 100, 2))
    old, new = y < known_classes, y >= known_classes
    out['old'] = float(np.around((pred[old] == y[old]).mean() * 100, 2)) if old.any() else 0.0
    out['new'] = float(np.around((pred[new] == y[new]).mean() * 100, 2))
    out['task_acc'] = float(np.around((pred // class_num == y // class_num).mean(), 4))
    return out


@torch.no_grad()
def extract_features(network: nn.Module, loader, device) -> Tuple[torch.Tensor, torch.Tensor]:
    """CLS features (via network.extract_vector) and labels for every batch of `loader`, on CPU."""
    net = network.module if isinstance(network, nn.DataParallel) else network
    net.eval()
    feats, labels = [], []
    for _, inputs, targets in loader:
        feats.append(net.extract_vector(inputs.to(device)).float().cpu())
        labels.append(torch.as_tensor(targets).long())
    return torch.cat(feats), torch.cat(labels)


class KernelHeadMixin(object):
    """Adds the analytic kernel head to an InfLoRA-style learner.

    Expects the learner to define: _network, _device, _known_classes,
    _total_classes, class_num, num_workers, test_loader.
    """

    def _init_kernel_head(self, args: Dict) -> None:
        self._kh_cfg = dict(args.get('kernel_head', {}))
        self._kh = build_head(self._kh_cfg, int(args.get('embd_dim', 768)), self._device)
        self._kh_batch = int(self._kh_cfg.get('batch_size', 128))
        self._kh_curve = {'top1': [], 'task': [], 'lambda': []}
        logging.info('Kernel head: {} (feature dim {})'.format(self._kh_cfg, self._kh.feature_map.out_dim))

    def _kh_after_task(self, data_manager) -> None:
        from torch.utils.data import DataLoader
        dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),
                                           source='train', mode='test')
        loader = DataLoader(dataset, batch_size=self._kh_batch, shuffle=False, num_workers=self.num_workers)
        f, y = extract_features(self._network, loader, self._device)
        lam = self._kh.fit_task(f, y, self._total_classes)
        self._kh_curve['lambda'].append(lam)
        sigma = getattr(self._kh.feature_map, 'sigma', None)
        logging.info('Kernel head: task {} added {} samples, lambda={:g}{}'.format(
            self._cur_task, len(y), lam, '' if sigma is None else ', sigma={:.4f}'.format(float(sigma))))

    def _kh_eval(self, loader) -> Dict:
        f, y = extract_features(self._network, loader, self._device)
        pred = self._kh.predict(f).cpu().numpy()
        metrics = head_metrics(pred, y.numpy(), self._known_classes, self.class_num)
        self._kh_curve['top1'].append(metrics['total'])
        self._kh_curve['task'].append(metrics['task_acc'])
        logging.info('KH: {}'.format(metrics))
        logging.info('KH top1 curve: {}'.format(self._kh_curve['top1']))
        logging.info('KH top1 task curve: {}'.format(self._kh_curve['task']))
        logging.info('KH lambda curve: {}'.format(self._kh_curve['lambda']))
        return metrics
