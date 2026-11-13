"""Phase 0 of the kernelised B_t (report/kernel_inflora.tex): offline go/no-go diagnostics, no training.

Step 1 (extract, GPU forward passes only). The frozen ViT-B/16 (InfLoRA's backbone, adapters off) is run
over every task of the class-incremental split from --config. For each chosen block it caches:
  - C_t, the exact uncentered second moment of the k/v inputs over ALL training tokens (InfLoRA's cur_matrix),
  - a random sample of training tokens and of held-out (test) tokens of the task.
Step 2 (analyse). Tasks are replayed in order. Per block, linear InfLoRA's DualGPM runs on C_t. For every lift
(RBF at several bandwidths, ReLU), a kernel DualGPM runs on the lifted sampled tokens. Before each update the
script measures novelty, usable signal (nu), leakage (iota), protection of old held-out tokens,
principal-angle overlaps and nearest-neighbour scales, for linear, kernel-only and hybrid B_t.
Step 3. Dataset structure (effective rank, cosines, linearization residual) and the go/no-go verdict.

Usage (from the repo root; see kernel_bt/README.md and kernel_bt/phase0.sbatch):
  python kernel_bt/phase0.py --config configs/mimg10_inflora.json --data_path /path/to/imagenet-r
  python kernel_bt/phase0.py --config configs/mimg10_inflora.json --data_path ... --init_cls 4 --increment 4
"""
import argparse
import json
import logging
import math
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from kernel_bt.diagnostics import (cosine_structure, effective_rank, energy_fraction, linearization,  # noqa: E402
                                   nearest_neighbour_median, novelty, residual_fraction, subspace_overlap)
from kernel_bt.dualgpm import DualGPM, inflora_threshold  # noqa: E402
from kernel_bt.lifts import KernelLift  # noqa: E402

DEFAULT_OPTS = {
    'rank': 10, 'lamb': 0.98, 'lame': 1.0,              # linear block: as in the InfLoRA config
    'lamb_phi': 0.90, 'lame_phi': 0.95, 'kmax_phi': 64,  # kernel block (design note defaults)
    'rff_dim': 2048, 'gamma': 1.0,
    'angles_k': 20, 'pool_per_task': 1000, 'pool_cap': 10000,
    'structure_tokens': 1500, 'structure_per_task': 200, 'structure_max_tasks': 20,
    'delta_nu': 0.05, 'delta_iota': 0.01, 'seed': 0,
}


def lift_specs(lifts, sigma_mults, pre_dims):
    specs = []
    for q in pre_dims:
        for kind in lifts:
            for c in (sigma_mults if kind == 'rbf' else [None]):
                specs.append({'kind': kind, 'sigma_mult': c if c is not None else 1.0, 'pre_dim': q})
    return specs


# ---------------------------------------------------------------------------
# Step 1: extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def _run_split(enc, captured, dataset, layers, n_tokens, with_moment, gen, device, batch_size, num_workers):
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    moments, samples, n_rows, per_image = {}, {l: [] for l in layers}, 0, None
    for _, x, _ in loader:
        enc(x.to(device), -1)  # task_id -1: no LoRA branch, the frozen pretrained ViT
        B, N, d = captured[layers[0]].shape
        if per_image is None:
            per_image = min(N, math.ceil(n_tokens / len(dataset)))
        idx = torch.rand(B, N, generator=gen).argsort(dim=1)[:, :per_image].to(device)
        for l in layers:
            h = captured[l]
            if with_moment:
                flat = h.reshape(-1, d).double()
                moments[l] = moments.get(l, 0) + flat.t() @ flat
            samples[l].append(torch.gather(h, 1, idx.unsqueeze(-1).expand(-1, -1, d)).reshape(-1, d).half().cpu())
        n_rows += B * N
    out = {}
    for l in layers:
        s = torch.cat(samples[l])
        s = s[torch.randperm(len(s), generator=gen)[:n_tokens]]
        out[l] = {'tokens': s}
        if with_moment:
            out[l]['C'] = (moments[l] / n_rows).float().cpu()
            out[l]['n'] = n_rows
    return out


def extract(cfg, layers, cache_dir, device, train_tokens, heldout_tokens, seed, batch_size, num_workers):
    """Writes cache_dir/task_XXX.pt for every task (skips existing ones). Returns the number of tasks."""
    from models.sinet_inflora import SiNet
    from utils.data_manager import DataManager

    torch.manual_seed(seed)  # as trainer._set_random before building the DataManager
    dm = DataManager(cfg['dataset'], cfg['shuffle'], seed, cfg['init_cls'], cfg['increment'], cfg)
    n_tasks = dm.nb_tasks
    os.makedirs(cache_dir, exist_ok=True)
    meta = {'dataset': cfg['dataset'], 'init_cls': cfg['init_cls'], 'increment': cfg['increment'],
            'n_tasks': n_tasks, 'layers': layers, 'train_tokens': train_tokens,
            'heldout_tokens': heldout_tokens, 'seed': seed}
    meta_path = os.path.join(cache_dir, 'meta.json')
    if os.path.exists(meta_path):
        with open(meta_path) as fh:
            old = json.load(fh)
        if old != meta:
            raise SystemExit('Cache {} was built with different settings:\n  {}\nvs now\n  {}\nUse another --out.'
                             .format(cache_dir, old, meta))
    with open(meta_path, 'w') as fh:
        json.dump(meta, fh, indent=1)

    todo = [t for t in range(n_tasks) if not os.path.exists(os.path.join(cache_dir, 'task_{:03d}.pt'.format(t)))]
    if not todo:
        logging.info('Cache complete: %s', cache_dir)
        return n_tasks
    net = SiNet({'total_sessions': n_tasks, 'rank': cfg.get('rank', 10), 'init_cls': cfg['init_cls'],
                 'embd_dim': cfg.get('embd_dim', 768)})
    enc = net.image_encoder.to(device).eval()
    captured = {}

    def hook(l):
        def fn(_module, _inp, out):
            captured[l] = out
        return fn

    handles = [enc.blocks[l - 1].norm1.register_forward_hook(hook(l)) for l in layers]
    starts = np.cumsum([0] + [dm.get_task_size(t) for t in range(n_tasks)])
    for t in todo:
        t0 = time.time()
        classes = np.arange(starts[t], starts[t + 1])
        gen = torch.Generator().manual_seed(seed * 100003 + t)
        tr = _run_split(enc, captured, dm.get_dataset(classes, source='train', mode='test'), layers,
                        train_tokens, True, gen, device, batch_size, num_workers)
        te = _run_split(enc, captured, dm.get_dataset(classes, source='test', mode='test'), layers,
                        heldout_tokens, False, gen, device, batch_size, num_workers)
        task = {'task': t, 'classes': [int(classes[0]), int(classes[-1]) + 1],
                'layers': {l: {'C': tr[l]['C'], 'n': tr[l]['n'], 'train': tr[l]['tokens'], 'test': te[l]['tokens']}
                           for l in layers}}
        torch.save(task, os.path.join(cache_dir, 'task_{:03d}.pt'.format(t)))
        logging.info('Extracted task %d/%d (classes %d-%d) in %.0fs', t + 1, n_tasks, classes[0], classes[-1],
                     time.time() - t0)
    for h in handles:
        h.remove()
    del enc, net
    if torch.device(device).type == 'cuda':
        torch.cuda.empty_cache()
    return n_tasks


def iter_cache(cache_dir, n_tasks):
    for t in range(n_tasks):
        yield torch.load(os.path.join(cache_dir, 'task_{:03d}.pt'.format(t)), map_location='cpu')


# ---------------------------------------------------------------------------
# Step 2: sequential analysis
# ---------------------------------------------------------------------------

def _pool(chunks, cap, gen, device):
    x = torch.cat(chunks)
    if len(x) > cap:
        x = x[torch.randperm(len(x), generator=gen)[:cap]]
    return x.to(device=device, dtype=torch.float64)


def _top_rows(h, r):
    evals, evecs = torch.linalg.eigh(h @ h.t())
    return evecs.flip(1)[:, :r].t().contiguous()


def _late_mean(rows, key, n_tasks):
    vals = [r[key] for r in rows if r['task'] > n_tasks / 2 and r['task'] >= 2 and key in r]
    return float(np.mean(vals)) if vals else float('nan')


def analyze(tasks, n_tasks, layers, specs, opts=None, device='cpu', progress=None):
    """tasks: iterable of per-task dicts in the cache format, in task order. Returns the results dict."""
    o = dict(DEFAULT_OPTS, **(opts or {}))
    f64 = torch.float64
    gen = torch.Generator().manual_seed(o['seed'])
    lin = {l: DualGPM(None) for l in layers}
    lifts = {l: [KernelLift(dim=o['rff_dim'], gamma=o['gamma'], seed=o['seed'] * 1000 + l, **s) for s in specs]
             for l in layers}
    kgpm = {l: [DualGPM(o['rff_dim'], kmax=o['kmax_phi']) for _ in specs] for l in layers}
    pool_tr = {l: [] for l in layers}
    pool_te = {l: [] for l in layers}
    res = {str(l): {'linear': [], 'lifts': {}} for l in layers}

    for t0, task in enumerate(tasks):
        tic = time.time()
        t = t0 + 1
        thr_x = inflora_threshold(o['lamb'], o['lame'], t0, n_tasks)
        thr_phi = inflora_threshold(o['lamb_phi'], o['lame_phi'], t0, n_tasks)
        for l in layers:
            blk = task['layers'][l]
            C = blk['C'].to(device=device, dtype=f64)
            Xtr = blk['train'].to(device=device, dtype=f64)
            Xte = blk['test'].to(device=device, dtype=f64)
            d = C.shape[0]
            g = lin[l]
            g.dim = d
            if t0 == 0:
                for lf in lifts[l]:
                    lf.calibrate(Xtr)
            old_tr = _pool(pool_tr[l], o['pool_cap'], gen, device) if t0 else None
            old_te = _pool(pool_te[l], o['pool_cap'], gen, device) if t0 else None

            row = {'task': t, 'dim': g.protected_dim}
            spec_x = None
            if t0:
                spec_x = g.free_spectrum(C)
                B = g.top_free(C, o['rank'], spec_x)
                row.update({'rho': novelty(g, C), 'nu': energy_fraction(Xte, B), 'iota': energy_fraction(old_te, B),
                            'prot_old_train': residual_fraction(g, old_tr),
                            'prot_old_heldout': residual_fraction(g, old_te),
                            'overlap_cross': subspace_overlap(old_tr, Xtr, o['angles_k']),
                            'overlap_within': subspace_overlap(Xtr[0::2], Xtr[1::2], o['angles_k'])})

            for lf, kg in zip(lifts[l], kgpm[l]):
                Ftr = lf.phi(Xtr)
                Cphi = Ftr.t() @ Ftr / len(Ftr)
                krow = {'task': t, 'dim': kg.protected_dim}
                spec_k = None
                if t0:
                    Fte, Fold_tr, Fold_te = lf.phi(Xte), lf.phi(old_tr), lf.phi(old_te)
                    spec_k = kg.free_spectrum(Cphi)
                    Bk = kg.top_free(Cphi, o['rank'], spec_k)
                    # hybrid psi = [x; alpha phi], block-diagonal protection, one joint top-r
                    Ptr = torch.cat([Xtr, lf.alpha * Ftr], 1)
                    Cpsi = Ptr.t() @ Ptr / len(Ptr)
                    Bh = _top_rows(torch.cat([g.free_part(Cpsi[:d]), kg.free_part(Cpsi[d:])], 0), o['rank'])
                    Pte = torch.cat([Xte, lf.alpha * Fte], 1)
                    Pold = torch.cat([old_te, lf.alpha * Fold_te], 1)
                    zte, ztr, zold = lf.z(Xte), lf.z(Xtr), lf.z(old_tr)
                    krow.update({
                        'rho_train': novelty(kg, Cphi), 'rho_heldout': residual_fraction(kg, Fte),
                        'nu': energy_fraction(Fte, Bk), 'iota': energy_fraction(Fold_te, Bk),
                        'prot_old_train': residual_fraction(kg, Fold_tr),
                        'prot_old_heldout': residual_fraction(kg, Fold_te),
                        'overlap_cross': subspace_overlap(Fold_tr, Ftr, o['angles_k']),
                        'overlap_within': subspace_overlap(Ftr[0::2], Ftr[1::2], o['angles_k']),
                        'nu_hybrid': energy_fraction(Pte, Bh), 'iota_hybrid': energy_fraction(Pold, Bh),
                        'kernel_share': (Bh[:, d:] ** 2).sum().item() / o['rank'],
                        'nn_within': nearest_neighbour_median(zte, ztr),
                        'nn_between': nearest_neighbour_median(zte, zold)})
                kg.update(Cphi, thr_phi, spec_k)
                krow['dim_after'] = kg.protected_dim
                entry = res[str(l)]['lifts'].setdefault(lf.name, {'sigma': lf.sigma, 'alpha': lf.alpha,
                                                                   'per_task': []})
                entry['per_task'].append(krow)

            g.update(C, thr_x, spec_x)
            row['dim_after'] = g.protected_dim
            res[str(l)]['linear'].append(row)
            k = min(o['pool_per_task'], len(Xtr), len(Xte))
            pool_tr[l].append(blk['train'][:k])
            pool_te[l].append(blk['test'][:k])
        if progress:
            progress(t, n_tasks, time.time() - tic)

    # Step 3: structure of the pooled tokens and linearization of each kernel
    for l in layers:
        chunks = pool_tr[l]
        step = max(1, math.ceil(len(chunks) / o['structure_max_tasks']))
        per_task = [c[:o['structure_per_task']].to(device=device, dtype=f64) for c in chunks[::step]]
        sample = _pool(chunks, o['structure_tokens'], gen, device)
        st = {'effective_rank': effective_rank(sample)}
        seen_q = set()
        for lf in lifts[l]:
            if lf.pre_dim not in seen_q:
                seen_q.add(lf.pre_dim)
                st['cosines_q{}'.format(lf.pre_dim or 'full')] = cosine_structure([lf.z(c) for c in per_task])
            z = lf.z(sample)
            res[str(l)]['lifts'][lf.name]['linearization'] = linearization(lf.exact_kernel(z, z), z)
        res[str(l)]['structure'] = st
    return {'opts': o, 'n_tasks': n_tasks, 'layers': {str(l): res[str(l)] for l in layers},
            'verdict': verdict(res, n_tasks, o)}


def verdict(res, n_tasks, o):
    """Design-note rule: GO if some lift has, for late tasks (t > T/2) in at least half of the layers,
    nu_hybrid >= nu_linear + delta_nu and iota_hybrid <= iota_linear + delta_iota."""
    names = sorted({n for r in res.values() for n in r['lifts']})
    per_lift = {}
    for name in names:
        passed = []
        for l, r in res.items():
            k = r['lifts'][name]['per_task']
            nu_h, nu_l = _late_mean(k, 'nu_hybrid', n_tasks), _late_mean(r['linear'], 'nu', n_tasks)
            io_h, io_l = _late_mean(k, 'iota_hybrid', n_tasks), _late_mean(r['linear'], 'iota', n_tasks)
            if nu_h >= nu_l + o['delta_nu'] and io_h <= io_l + o['delta_iota']:
                passed.append(l)
        per_lift[name] = {'layers_passed': passed, 'go': 2 * len(passed) >= len(res) and len(passed) > 0}
    return {'go': any(v['go'] for v in per_lift.values()), 'per_lift': per_lift,
            'rule': 'late-task mean nu_hyb >= nu_lin + {delta_nu} and iota_hyb <= iota_lin + {delta_iota} '
                    'in >= half the layers'.format(**o)}


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _f(v, w=6, p=3):
    return ('{:>%d.%df}' % (w, p)).format(v) if isinstance(v, float) and not math.isnan(v) else ' ' * (w - 1) + '-'


def format_summary(results):
    T, L = results['n_tasks'], results['layers']
    out = ['Phase 0: kernelised B_t go/no-go  ({} tasks; late tasks = t > {:g})'.format(T, T / 2), '']
    out.append('1) Linear InfLoRA saturation, per block (rows) and task (columns)')
    for key, title in (('rho', 'novelty rho_t = new-task energy outside M'), ('nu', 'usable signal nu_t'),
                       ('dim', 'protected dim(M) before the task (of 768)')):
        out.append('   ' + title)
        out.append('   blk ' + ''.join('{:>6}'.format('t' + str(r['task'])) for r in L[next(iter(L))]['linear']))
        for l, r in L.items():
            cells = [_f(float(x.get(key, float('nan')))) if key != 'dim' else '{:>6}'.format(x[key])
                     for x in r['linear']]
            out.append('   {:>3} '.format(l) + ''.join(cells))
        out.append('')

    out.append('2) Late-task means: linear vs kernel-only vs hybrid B_t (higher nu, lower iota is better)')
    hdr = ('   {:<14}{:>4} {:>6}{:>6}{:>6} {:>6}{:>6}{:>6} {:>6} {:>6}{:>6} {:>6}{:>6} {:>6}{:>6} {:>6}  pass'
           .format('lift', 'blk', 'nuLin', 'nuKer', 'nuHyb', 'ioLin', 'ioKer', 'ioHyb', 'kshr',
                   'pOldL', 'pOldK', 'ovrL', 'ovrK', 's/dW', 'dB/s', 'linRs'))
    out.append(hdr)
    out.append('   ' + '-' * (len(hdr) - 3))
    names = sorted({n for r in L.values() for n in r['lifts']})
    for name in names:
        passed = results['verdict']['per_lift'][name]['layers_passed']
        for l, r in L.items():
            k, lin = r['lifts'][name], r['linear']
            kt = k['per_task']
            ratio = lambda rows, a, b: _late_mean(rows, a, T) / max(_late_mean(rows, b, T), 1e-12)  # noqa: E731
            sig = k['sigma']
            sdw = sig / _late_mean(kt, 'nn_within', T) if not math.isnan(sig) else float('nan')
            dbs = _late_mean(kt, 'nn_between', T) / sig if not math.isnan(sig) else float('nan')
            out.append('   {:<14}{:>4} {}{}{} {}{}{} {} {}{} {}{} {}{} {}  {}'.format(
                name, l, _f(_late_mean(lin, 'nu', T)), _f(_late_mean(kt, 'nu', T)), _f(_late_mean(kt, 'nu_hybrid', T)),
                _f(_late_mean(lin, 'iota', T)), _f(_late_mean(kt, 'iota', T)), _f(_late_mean(kt, 'iota_hybrid', T)),
                _f(_late_mean(kt, 'kernel_share', T)),
                _f(_late_mean(lin, 'prot_old_heldout', T)), _f(_late_mean(kt, 'prot_old_heldout', T)),
                _f(ratio(lin, 'overlap_cross', 'overlap_within')), _f(ratio(kt, 'overlap_cross', 'overlap_within')),
                _f(sdw, p=2), _f(dbs, p=2), _f(k['linearization']['residual_spectral']),
                'yes' if l in passed else ''))
        out.append('')
    out.append('   nu*/io*: usable signal / leakage;  kshr: share of hybrid B_t in the kernel block;')
    out.append('   pOld*: energy of held-out OLD tokens outside M (lower = better protected);')
    out.append('   ovr*: old-vs-new subspace overlap / same-task-halves overlap (lower = tasks more separated);')
    out.append('   s/dW = sigma / NN distance within the new task, dB/s = NN distance to old tasks / sigma')
    out.append('   (useful regime: s/dW >~ 1 and dB/s > 1);  linRs: linearization residual (near 0 = effectively linear).')
    out.append('')
    out.append('3) Token structure per block')
    for l, r in L.items():
        st = r['structure']
        cos = {k: v for k, v in st.items() if k.startswith('cosines')}
        parts = ['r_eff {:.1f}'.format(st['effective_rank'])]
        for k, v in cos.items():
            parts.append('{}: within {:.3f}+-{:.3f}, between {}, gap {}'.format(
                k.replace('cosines_', ''), v['cos_within_mean'], v['cos_within_std'],
                '{:.3f}'.format(v['cos_between_mean']) if 'cos_between_mean' in v else '-',
                '{:.3f}'.format(v['gap_ratio']) if 'gap_ratio' in v else '-'))
        out.append('   blk {:>2}: '.format(l) + '; '.join(parts))
    out.append('')
    v = results['verdict']
    out.append('4) Verdict: {}   (rule: {})'.format('GO' if v['go'] else 'NO-GO', v['rule']))
    for name, pl in v['per_lift'].items():
        out.append('   {:<14} passes in blocks: {}'.format(name, ', '.join(pl['layers_passed']) or 'none'))
    return '\n'.join(out)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', default='configs/mimg10_inflora.json', help='InfLoRA dataset config (split, data_path)')
    p.add_argument('--data_path', default=None)
    p.add_argument('--init_cls', type=int, default=None, help='override the split, e.g. 4 for ImageNet-R/50')
    p.add_argument('--increment', type=int, default=None)
    p.add_argument('--layers', type=int, nargs='+', default=list(range(1, 13)), help='1-based ViT blocks')
    p.add_argument('--lifts', nargs='+', default=['rbf', 'relu'], choices=['rbf', 'relu'])
    p.add_argument('--sigma_mults', type=float, nargs='+', default=[0.25, 0.5, 1.0, 2.0, 4.0])
    p.add_argument('--pre_dims', type=int, nargs='+', default=[0], help='0 = no pre-projection; e.g. 0 128')
    p.add_argument('--rff_dim', type=int, default=DEFAULT_OPTS['rff_dim'])
    p.add_argument('--gamma', type=float, default=DEFAULT_OPTS['gamma'])
    p.add_argument('--lamb_phi', type=float, default=DEFAULT_OPTS['lamb_phi'])
    p.add_argument('--lame_phi', type=float, default=DEFAULT_OPTS['lame_phi'])
    p.add_argument('--kmax_phi', type=int, default=DEFAULT_OPTS['kmax_phi'])
    p.add_argument('--train_tokens', type=int, default=5000, help='sampled training tokens per task and block')
    p.add_argument('--heldout_tokens', type=int, default=2000, help='sampled test tokens per task and block')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='0', help='GPU index or cpu')
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--out', default=None, help='default kbt_runs/phase0_<dataset>_<T>tasks')
    p.add_argument('--extract_only', action='store_true')
    a = p.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    with open(a.config) as fh:
        cfg = json.load(fh)
    for key in ('data_path', 'init_cls', 'increment'):
        if getattr(a, key) is not None:
            cfg[key] = getattr(a, key)
    n_classes = {'imagenet_r': 200, 'cifar100': 100, 'domainnet': 345}.get(cfg['dataset'].lower())
    n_guess = 1 + (n_classes - cfg['init_cls']) // cfg['increment'] if n_classes else 0
    out = a.out or os.path.join('kbt_runs', 'phase0_{}_{}tasks'.format(cfg['dataset'], n_guess or 'x'))
    os.makedirs(out, exist_ok=True)
    logging.getLogger().addHandler(logging.FileHandler(os.path.join(out, 'phase0.log')))
    device = torch.device('cuda:{}'.format(a.device) if torch.cuda.is_available() and a.device != 'cpu' else 'cpu')
    logging.info('Phase 0: %s, split %d+%d, blocks %s, device %s, out %s',
                 cfg['dataset'], cfg['init_cls'], cfg['increment'], a.layers, device, out)

    cache = os.path.join(out, 'cache')
    n_tasks = extract(cfg, a.layers, cache, device, a.train_tokens, a.heldout_tokens, a.seed,
                      a.batch_size, a.num_workers)
    if a.extract_only:
        return
    specs = lift_specs(a.lifts, a.sigma_mults, a.pre_dims)
    opts = {'rank': cfg.get('rank', 10), 'lamb': cfg.get('lamb', 0.98), 'lame': cfg.get('lame', 1.0),
            'lamb_phi': a.lamb_phi, 'lame_phi': a.lame_phi, 'kmax_phi': a.kmax_phi,
            'rff_dim': a.rff_dim, 'gamma': a.gamma, 'seed': a.seed}
    logging.info('Analysing %d tasks x %d blocks x %d lifts', n_tasks, len(a.layers), len(specs))
    results = analyze(iter_cache(cache, n_tasks), n_tasks, a.layers, specs, opts, device,
                      progress=lambda t, T, s: logging.info('Analysed task %d/%d in %.0fs', t, T, s))
    results['run'] = {'config': a.config, 'dataset': cfg['dataset'], 'init_cls': cfg['init_cls'],
                      'increment': cfg['increment'], 'layers': a.layers, 'specs': specs,
                      'train_tokens': a.train_tokens, 'heldout_tokens': a.heldout_tokens}
    with open(os.path.join(out, 'phase0.json'), 'w') as fh:
        json.dump(results, fh, indent=1)
    text = format_summary(results)
    with open(os.path.join(out, 'phase0_summary.txt'), 'w') as fh:
        fh.write(text + '\n')
    print(text)
    logging.info('Wrote %s/phase0.json and phase0_summary.txt', out)


if __name__ == '__main__':
    main()
