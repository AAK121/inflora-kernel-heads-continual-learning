"""CLIP baseline + kernel heads + text fusion, in one command (no training).

1. Encode every image once with frozen CLIP (default ViT-B/16) and every class name with the
   frozen text encoder (prompt ensemble); cache as <out>/features.pt (reused if present).
2. Replay the class-incremental sequence of the given dataset config and evaluate:
     - CLIP zero-shot over the classes seen so far      (baseline row)
     - kernel heads on the CLIP image embedding         ("text_fusion": 0)
     - kernel head + zero-shot text classifier, fused   ("text_fusion": "auto" / a number)
   -> <out>/clip_sweep.txt and clip_sweep.json

Examples (from the repo root):
  python clip/run_clip_sweep.py --config configs/mimg10_inflora.json --data_path /path/to/imagenet-r --device 0
  python clip/run_clip_sweep.py --config configs/cifar100_inflora.json --device 0

Offline machine: put the CLIP checkpoint at ~/.cache/clip/ViT-B-16.pt (the file the loader
would download), or pass --clip_model /path/to/ViT-B-16.pt.
"""
import argparse
import json
import logging
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, os.path.join(ROOT, 'tools'), HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from methods.kernel_head import head_metrics  # noqa: E402
from fusion import TextFusionHead  # noqa: E402


def evaluate_fused(cache, cfg, device=None):
    """One head config over one cached CLIP sequence. Same curve keys as tools/eval_kernel_heads."""
    meta = cache['meta']
    head = TextFusionHead(cfg, cache['steps'][0]['train_f'].shape[1], meta['text_emb'], meta['logit_scale'], device)
    curve = {'top1': [], 'task': [], 'old': [], 'new': [], 'lambda': [], 'beta': [], 'base_top1': []}
    for step in cache['steps']:
        lam = head.fit_task(step['train_f'], step['train_y'], step['total'])
        y = step['test_y'].numpy()
        m = head_metrics(head.predict(step['test_f']).cpu().numpy(), y, step['known'], meta['class_num'])
        base = head_metrics(step['base_pred'].numpy(), y, step['known'], meta['class_num'])
        curve['top1'].append(m['total'])
        curve['task'].append(m['task_acc'])
        curve['old'].append(m['old'])
        curve['new'].append(m['new'])
        curve['lambda'].append(lam)
        curve['beta'].append(head.beta)
        curve['base_top1'].append(base['total'])
    return curve


def run(cache, heads, device=None, only=None):
    from eval_kernel_heads import evaluate_baseline, summarize
    if only:
        heads = [h for h in heads if h['name'] in only]
    base_curve = evaluate_baseline(cache)
    results = {'baseline': summarize([{'top1': base_curve, 'task': [0.0]}]), 'baseline_curve': base_curve, 'heads': {}}
    for cfg in heads:
        t0 = time.time()
        try:
            curve = evaluate_fused(cache, cfg, device)
        except (RuntimeError, ValueError) as e:
            logging.error('%s failed: %s', cfg['name'], e)
            results['heads'][cfg['name']] = {'config': cfg, 'error': str(e)}
            continue
        s = summarize([curve])
        s['seconds'] = round(time.time() - t0, 1)
        results['heads'][cfg['name']] = {'config': cfg, 'summary': s, 'curves': [curve]}
        logging.info('%-28s final %.2f | avg %.2f | task-id %.4f | beta %s | %.0fs',
                     cfg['name'], s['final_mean'], s['avg_mean'], s['task_acc_final'], curve['beta'], s['seconds'])
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', default='configs/mimg10_inflora.json', help='dataset config (dataset, data_path, task split)')
    p.add_argument('--data_path', default=None, help="overrides the config's data_path")
    p.add_argument('--clip_model', default='ViT-B/16', help='CLIP name (ViT-B/16, ViT-B/32, ...) or a local .pt path')
    p.add_argument('--class_names', default=None, help='optional JSON: list in label order, or {wnid: name}')
    p.add_argument('--templates', default='ensemble', choices=['ensemble', 'single'])
    p.add_argument('--heads', default=os.path.join('clip', 'clip_heads.json'))
    p.add_argument('--only', nargs='*', default=None)
    p.add_argument('--seed', type=int, default=0, help='DataManager seed (class order / first-time ImageNet-R split)')
    p.add_argument('--device', default='0')
    p.add_argument('--out', default=None, help='default: kh_runs/clip_<dataset>_<model>')
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--num_workers', type=int, default=8)
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

    if a.data_path:
        a.data_path = os.path.abspath(a.data_path)
    if os.path.isfile(a.clip_model):
        a.clip_model = os.path.abspath(a.clip_model)
    os.chdir(ROOT)
    with open(a.config) as fh:
        cfg = json.load(fh)
    if a.data_path:
        cfg['data_path'] = a.data_path
    device = torch.device('cuda:{}'.format(a.device) if torch.cuda.is_available() and a.device != 'cpu' else 'cpu')
    tag = os.path.splitext(os.path.basename(a.clip_model))[0].replace('/', '-')
    out = a.out or os.path.join('kh_runs', 'clip_{}_{}'.format(cfg['dataset'], tag))
    os.makedirs(out, exist_ok=True)

    cache_path = os.path.join(out, 'features.pt')
    if os.path.exists(cache_path):
        logging.info('using cached CLIP features %s', cache_path)
        cache = torch.load(cache_path, map_location='cpu')
    else:
        from clip_features import build_clip_cache
        t0 = time.time()
        cache = build_clip_cache(cfg, a.clip_model, device, a.class_names, a.templates,
                                 a.batch_size, a.num_workers, a.seed)
        torch.save(cache, cache_path)
        logging.info('encoded %d train / %d test images in %.0fs -> %s',
                     sum(len(s['train_y']) for s in cache['steps']), len(cache['steps'][-1]['test_y']),
                     time.time() - t0, cache_path)

    with open(a.heads) as fh:
        heads = json.load(fh)
    results = run(cache, heads, device, a.only)
    results['run'] = {'config': a.config, 'clip_model': a.clip_model, 'templates': cache['meta']['templates'],
                      'heads_file': a.heads, 'data_path': cfg['data_path']}
    from eval_kernel_heads import format_table, save_json
    save_json(results, os.path.join(out, 'clip_sweep.json'))
    header = 'CLIP {} on {} ({} tasks), prompts={}\n'.format(
        cache['meta']['clip_model'], cfg['dataset'], len(cache['steps']), cache['meta']['templates'])
    table = format_table(results, baseline_label='CLIP zero-shot')
    with open(os.path.join(out, 'clip_sweep.txt'), 'w') as fh:
        fh.write(header + table + '\n')
    print('\n' + header + table)


if __name__ == '__main__':
    main()
