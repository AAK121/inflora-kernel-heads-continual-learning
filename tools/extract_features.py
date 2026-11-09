"""Cache CLS features from saved InfLoRA checkpoints, for post-hoc kernel-head evaluation.

trainer.py saves the network after every task as <logdir>/<seed>/task_<t>.pth.
For each task t this script loads task_<t>.pth and stores:
  * features + labels of task t's TRAINING images (test-time transform), which is
    exactly what the online kernel head absorbs after task t;
  * features + labels of the TEST images of all classes seen so far;
  * the predictions of InfLoRA's own concatenated heads on those test images (baseline).

Example:
  python tools/extract_features.py --config configs/cifar100_inflora.json \
      --ckpt_dir logs/cifar100/10_10_sip/InfLoRA/adam/10/0.95_1.0-0.0005/0 \
      --seed 0 --device 0 --out features/cifar100_seed0.pt
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


@torch.no_grad()
def run(net, loader, device, with_baseline):
    feats, labels, preds = [], [], []
    for _, inputs, targets in loader:
        f = net.extract_vector(inputs.to(device))
        feats.append(f.float().cpu())
        labels.append(torch.as_tensor(targets).long())
        if with_baseline:
            preds.append(net.interface2(f).argmax(dim=1).cpu())
    out = {'f': torch.cat(feats), 'y': torch.cat(labels)}
    if with_baseline:
        out['base_pred'] = torch.cat(preds)
    return out


def extract_cache(cfg, ckpt_dir, seed, device, batch_size=128, num_workers=8):
    """Build the feature cache for one trained run (cfg = the run's config dict)."""
    cfg = dict(cfg)
    cfg['seed'] = seed
    torch.manual_seed(seed)
    dm = DataManager(cfg['dataset'], cfg['shuffle'], seed, cfg['init_cls'], cfg['increment'], cfg)
    net = SiNet(cfg).to(device).eval()
    cache = {'meta': {'ckpt_dir': ckpt_dir, 'seed': seed, 'dataset': cfg['dataset'],
                      'class_num': cfg['init_cls'], 'increments': [dm.get_task_size(t) for t in range(dm.nb_tasks)]},
             'steps': []}

    def loader(ds):
        return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    known = 0
    for t in range(dm.nb_tasks):
        path = os.path.join(ckpt_dir, 'task_{}.pth'.format(t))
        if not os.path.exists(path):
            logging.warning('missing %s, stopping after %d tasks', path, t)
            break
        net.load_state_dict(torch.load(path, map_location='cpu'))
        net.numtask = t + 1
        total = known + dm.get_task_size(t)
        train = run(net, loader(dm.get_dataset(np.arange(known, total), source='train', mode='test')), device, False)
        test = run(net, loader(dm.get_dataset(np.arange(0, total), source='test', mode='test')), device, True)
        base_acc = (test['base_pred'] == test['y']).float().mean().item() * 100
        logging.info('task %d: %d train / %d test features, InfLoRA head acc %.2f', t, len(train['y']), len(test['y']), base_acc)
        cache['steps'].append({'known': known, 'total': total,
                               'train_f': train['f'], 'train_y': train['y'],
                               'test_f': test['f'], 'test_y': test['y'], 'base_pred': test['base_pred']})
        known = total
    del net
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return cache


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', required=True, help='the JSON config the checkpoints were trained with')
    p.add_argument('--ckpt_dir', required=True, help='directory holding task_<t>.pth')
    p.add_argument('--seed', type=int, default=None, help='seed of that run (default: first seed in config)')
    p.add_argument('--device', default='0', help="GPU id, or 'cpu'")
    p.add_argument('--out', required=True)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--num_workers', type=int, default=8)
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

    with open(a.config) as fh:
        cfg = json.load(fh)
    seed = a.seed if a.seed is not None else (cfg['seed'][0] if isinstance(cfg['seed'], list) else cfg['seed'])
    device = torch.device('cpu' if a.device == 'cpu' else 'cuda:{}'.format(a.device))
    cache = extract_cache(cfg, a.ckpt_dir, seed, device, a.batch_size, a.num_workers)
    cache['meta']['config'] = a.config
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    torch.save(cache, a.out)
    logging.info('saved %s', a.out)


if __name__ == '__main__':
    main()
