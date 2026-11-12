"""Kernel head + CLIP zero-shot text classifier, fused (option 2).

    score_c(f) = (1 - beta) * kernel_head_c(f) + beta * softmax_c(logit_scale * cos(f, text_c))

over the classes seen so far. The ridge head regresses one-hot targets, so its scores are on
roughly the same [0, 1] scale as the zero-shot probabilities.

beta:
  0      -> kernel head only;   1 -> zero-shot CLIP only;   a number -> fixed mix;
  'auto' -> after each task, fit a copy of the head on 80% of the new task (plus all old
            statistics) and pick the beta in beta_grid with the best accuracy on the other 20%,
            scored over all classes seen so far. No old samples are needed.
"""
import copy
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from methods.kernel_head import build_head  # noqa: E402

FUSION_KEYS = ('text_fusion', 'beta_grid', 'fusion_val_fraction')
DEFAULT_BETA_GRID = [round(0.1 * i, 1) for i in range(11)]


def zero_shot_probs(f, text_emb, logit_scale, total):
    f = f / f.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return torch.softmax(logit_scale * f @ text_emb[:total].t(), dim=1)


class TextFusionHead(object):

    def __init__(self, cfg, in_dim, text_emb, logit_scale, device=None):
        cfg = dict(cfg)
        self.beta_mode = cfg.get('text_fusion', 'auto')
        if not (self.beta_mode == 'auto' or isinstance(self.beta_mode, (int, float))):
            raise ValueError("text_fusion must be a number in [0, 1] or 'auto', got {}".format(self.beta_mode))
        self.beta_grid = list(cfg.get('beta_grid', DEFAULT_BETA_GRID))
        self.val_fraction = float(cfg.get('fusion_val_fraction', 0.2))
        head_cfg = {k: v for k, v in cfg.items() if k not in FUSION_KEYS}
        self.device = device
        self.head = build_head(head_cfg, in_dim, device)
        self.text_emb = text_emb.float().to(self.head.device)
        self.logit_scale = float(logit_scale)
        self.beta = 0.0 if self.beta_mode == 'auto' else float(self.beta_mode)
        self.total = 0
        self.history = []

    @property
    def feature_map(self):
        return self.head.feature_map

    def _scores(self, head, f, total, beta):
        f = f.to(self.head.device, torch.float32)
        zs = zero_shot_probs(f, self.text_emb, self.logit_scale, total)
        if beta >= 1.0:
            return zs
        return (1.0 - beta) * head.logits(f)[:, :total] + beta * zs

    @torch.no_grad()
    def _select_beta(self, f, y, total):
        fit_idx, val_idx = self.head._split(y)  # stratified, every class keeps fit samples
        if len(val_idx) == 0:
            return self.beta
        probe = copy.deepcopy(self.head)
        probe.fit_task(f[fit_idx], y[fit_idx], total)
        yv = y[val_idx].to(self.head.device)
        accs = {b: (self._scores(probe, f[val_idx], total, b).argmax(1) == yv).float().mean().item()
                for b in self.beta_grid}
        del probe
        best = max(accs.values())
        winners = sorted(b for b, a in accs.items() if a == best)
        beta = winners[len(winners) // 2]  # middle of the tied range
        self.history.append({'val_acc_per_beta': accs, 'beta': beta})
        return beta

    @torch.no_grad()
    def fit_task(self, f, y, total):
        """Returns the ridge lambda (same contract as AnalyticKernelHead.fit_task)."""
        if self.beta_mode == 'auto':
            self.beta = self._select_beta(f, y, total)
        self.total = total
        return self.head.fit_task(f, y, total)

    @torch.no_grad()
    def predict(self, f):
        return self._scores(self.head, f, self.total, self.beta).argmax(dim=1)
