"""Fit the kernel head on top of ONE frozen checkpoint (e.g. the final task_9.pth).

Unlike tools/extract_features.py, which loads task_<t>.pth for every task t, this loads a
single checkpoint, runs every train/test image through it once, and then replays the
task sequence with the head only: after step t the head has absorbed the training
features of tasks 0..t, and it is tested on all classes seen so far.

Note: every feature comes from the final backbone, which has already been trained on all
tasks. So this is a post-hoc head on a finished model, not the class-incremental protocol
(that is what extract_features.py + eval_kernel_heads.py measure); only the final number
is meaningful, the intermediate steps are optimistic.

Example:
  python tools/fit_head_on_final.py --config configs/mimg10_inflora.json \
      --ckpt logs/.../kh_base_e10/0/task_9.pth --only relu10k_none --device 0 \
      --save_heads checkpoints/final_heads
"""
import argparse
import json
import logging
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_manager import DataManager  # noqa: E402
from models.sinet_inflora import SiNet  # noqa: E402
from extract_features import run  # noqa: E402  (tools/ is on sys.path when run as a script)
from eval_kernel_heads import format_table, run_sweep, save_json  # noqa: E402


def extract_final(cfg, ckpt, seed, device, batch_size=128, num_workers=8):
    """Features of every train and test image under one checkpoint, split into the task sequence.

    Returns a cache in the same format as tools/extract_features.py, so the sweep code reuses it."""
    cfg = dict(cfg)
    cfg['seed'] = seed
    torch.manual_seed(seed)
    dm = DataManager(cfg['dataset'], cfg['shuffle'], seed, cfg['init_cls'], cfg['increment'], cfg)
    n_tasks = dm.nb_tasks
    n_classes = sum(dm.get_task_size(t) for t in range(n_tasks))

    net = SiNet(cfg).to(device).eval()
    net.load_state_dict(torch.load(ckpt, map_location='cpu'))
    net.numtask = n_tasks

    def loader(source):
        ds = dm.get_dataset(np.arange(0, n_classes), source=source, mode='test')
        return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    logging.info('extracting train features with %s', ckpt)
    train = run(net, loader('train'), device, False)
    logging.info('extracting test features')
    test = run(net, loader('test'), device, False)
    with torch.no_grad():  # InfLoRA's own heads, as the baseline
        base_logits = torch.cat([net.interface2(test['f'][s:s + 4096].to(device)).float().cpu()
                                 for s in range(0, len(test['y']), 4096)])
    del net
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    cache = {'meta': {'ckpt_dir': ckpt, 'seed': seed, 'dataset': cfg['dataset'], 'class_num': cfg['init_cls'],
                      'increments': [dm.get_task_size(t) for t in range(n_tasks)]},
             'steps': []}
    known = 0
    for t in range(n_tasks):
        total = known + dm.get_task_size(t)
        tr = (train['y'] >= known) & (train['y'] < total)
        te = test['y'] < total
        cache['steps'].append({'known': known, 'total': total,
                               'train_f': train['f'][tr], 'train_y': train['y'][tr],
                               'test_f': test['f'][te], 'test_y': test['y'][te],
                               'base_pred': base_logits[te, :total].argmax(dim=1)})
        known = total
    logging.info('%d train / %d test images, %d classes, %d tasks', len(train['y']), len(test['y']), n_classes, n_tasks)
    return cache


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', required=True, help='the JSON config the checkpoint was trained with')
    p.add_argument('--ckpt', required=True, help='the frozen checkpoint, e.g. .../0/task_9.pth')
    p.add_argument('--heads', default='configs/kernel_head_sweep.json', help='JSON list of head configs')
    p.add_argument('--only', nargs='*', default=None, help='run only these head names (default: all)')
    p.add_argument('--seed', type=int, default=None, help='seed of that run (default: first seed in config)')
    p.add_argument('--device', default='0', help="GPU id, or 'cpu'")
    p.add_argument('--save_heads', default=None, help='folder to save each fitted head in')
    p.add_argument('--keep_gram', action='store_true', help='also save G, Q so the head can keep learning')
    p.add_argument('--save_features', default=None, help='also save the feature cache (reusable by eval_kernel_heads.py)')
    p.add_argument('--out', default=None, help='JSON with all curves')
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--num_workers', type=int, default=8)
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

    with open(a.config) as fh:
        cfg = json.load(fh)
    seed = a.seed if a.seed is not None else (cfg['seed'][0] if isinstance(cfg['seed'], list) else cfg['seed'])
    device = torch.device('cpu' if a.device == 'cpu' else 'cuda:{}'.format(a.device))

    if a.save_features and os.path.exists(a.save_features):
        logging.info('reusing features from %s', a.save_features)
        cache = torch.load(a.save_features, map_location='cpu')
    else:
        cache = extract_final(cfg, a.ckpt, seed, device, a.batch_size, a.num_workers)
        cache['meta']['config'] = a.config
        if a.save_features:
            os.makedirs(os.path.dirname(os.path.abspath(a.save_features)), exist_ok=True)
            torch.save(cache, a.save_features)
            logging.info('saved %s', a.save_features)

    with open(a.heads) as fh:
        heads = json.load(fh)
    results = run_sweep([cache], heads, device, only=a.only, save_dir=a.save_heads, keep_gram=a.keep_gram)
    results['files'] = [a.ckpt]
    print(format_table(results, baseline_label='InfLoRA heads (final ckpt)'))
    if a.out:
        save_json(results, a.out)


if __name__ == '__main__':
    main()
