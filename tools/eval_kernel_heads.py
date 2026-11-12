"""Sweep kernel-head configurations over cached features (from tools/extract_features.py).

Replays the class-incremental sequence: after step t the head absorbs task t's
training features, then it is evaluated on the test images of all classes seen so far.
The result is identical to what InfLoRA_KH would log online, but needs no retraining,
and every head in the sweep reuses the same features.

Example:
  python tools/eval_kernel_heads.py --features features/cifar100_seed*.pt \
      --heads configs/kernel_head_sweep.json --device 0 --out results/cifar100_heads.json
"""
import argparse
import glob
import json
import logging
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from methods.kernel_head import build_head, head_metrics  # noqa: E402


def evaluate_head(cache, head_cfg, device=None):
    """Run one head config over one cached sequence. Returns per-step curves."""
    class_num = cache['meta']['class_num']
    in_dim = cache['steps'][0]['train_f'].shape[1]
    head = build_head(head_cfg, in_dim, device)
    curve = {'top1': [], 'task': [], 'old': [], 'new': [], 'lambda': [], 'base_top1': []}
    for step in cache['steps']:
        lam = head.fit_task(step['train_f'], step['train_y'], step['total'])
        y = step['test_y'].numpy()
        m = head_metrics(head.predict(step['test_f']).cpu().numpy(), y, step['known'], class_num)
        base = head_metrics(step['base_pred'].numpy(), y, step['known'], class_num)
        for k in ('top1', 'task', 'old', 'new'):
            curve[k].append(m['total' if k == 'top1' else ('task_acc' if k == 'task' else k)])
        curve['lambda'].append(lam)
        curve['base_top1'].append(base['total'])
    sigma = getattr(head.feature_map, 'sigma', None)
    curve['sigma'] = None if sigma is None else float(sigma)
    return curve


def summarize(curves):
    final = np.array([c['top1'][-1] for c in curves])
    avg = np.array([np.mean(c['top1']) for c in curves])
    task = np.array([c['task'][-1] for c in curves])
    return {'final_mean': float(final.mean()), 'final_std': float(final.std()),
            'avg_mean': float(avg.mean()), 'task_acc_final': float(task.mean()), 'n_seeds': len(curves)}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--features', nargs='+', required=True, help='cache files (one per seed); globs allowed')
    p.add_argument('--heads', required=True, help='JSON list of head configs, each with a "name"')
    p.add_argument('--only', nargs='*', default=None, help='run only these head names')
    p.add_argument('--device', default='0', help="GPU id, or 'cpu'")
    p.add_argument('--out', default=None)
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

    device = torch.device('cpu' if a.device == 'cpu' else 'cuda:{}'.format(a.device))
    paths = sorted({f for pat in a.features for f in glob.glob(pat)})
    if not paths:
        sys.exit('no feature files match {}'.format(a.features))
    caches = [torch.load(f, map_location='cpu') for f in paths]
    with open(a.heads) as fh:
        heads = json.load(fh)
    results = run_sweep(caches, heads, device, only=a.only)
    results['files'] = paths
    print(format_table(results))
    if a.out:
        save_json(results, a.out)


def run_sweep(caches, heads, device=None, only=None):
    """Evaluate every head config on every cached seed. Returns a JSON-serializable dict."""
    if only:
        heads = [h for h in heads if h['name'] in only]
    base = summarize([{'top1': evaluate_baseline(c), 'task': [0.0]} for c in caches])
    logging.info('InfLoRA concatenated heads: final %.2f +- %.2f, avg %.2f (%d seeds)',
                 base['final_mean'], base['final_std'], base['avg_mean'], base['n_seeds'])
    results = {'baseline': base, 'heads': {}}
    for cfg in heads:
        t0 = time.time()
        try:
            curves = [evaluate_head(c, cfg, device) for c in caches]
        except (RuntimeError, ValueError) as e:  # e.g. GPU out of memory or a bad config: report it and keep going
            logging.error('%s failed: %s', cfg['name'], e)
            results['heads'][cfg['name']] = {'config': cfg, 'error': str(e)}
            if device is not None and torch.device(device).type == 'cuda':
                torch.cuda.empty_cache()
            continue
        s = summarize(curves)
        s['seconds'] = round(time.time() - t0, 1)
        results['heads'][cfg['name']] = {'config': cfg, 'summary': s, 'curves': curves}
        logging.info('%-28s final %.2f +- %.2f | avg %.2f | task-id %.4f | lambda %s | %.0fs',
                     cfg['name'], s['final_mean'], s['final_std'], s['avg_mean'], s['task_acc_final'],
                     curves[0]['lambda'], s['seconds'])
    return results


def format_table(results, baseline_label='InfLoRA (concat heads)'):
    base = results['baseline']
    lines = ['{:<28} {:>14} {:>8} {:>9}'.format('head', 'final', 'avg', 'task-id'),
             '{:<28} {:>8.2f} +- {:<4.2f} {:>8.2f} {:>9}'.format(
                 baseline_label, base['final_mean'], base['final_std'], base['avg_mean'], '-')]
    ok = [(n, r['summary']) for n, r in results['heads'].items() if 'summary' in r]
    for name, s in sorted(ok, key=lambda r: -r[1]['final_mean']):
        lines.append('{:<28} {:>8.2f} +- {:<4.2f} {:>8.2f} {:>9.4f}'.format(
            name, s['final_mean'], s['final_std'], s['avg_mean'], s['task_acc_final']))
    for name, r in results['heads'].items():
        if 'error' in r:
            lines.append('{:<28} FAILED: {}'.format(name, r['error'][:80]))
    return '\n'.join(lines)


def save_json(results, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w') as fh:
        json.dump(results, fh, indent=1)
    logging.info('saved %s', path)


def evaluate_baseline(cache):
    return [float(np.around((s['base_pred'] == s['test_y']).float().mean().item() * 100, 2)) for s in cache['steps']]


if __name__ == '__main__':
    main()
