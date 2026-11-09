"""One command: train a small InfLoRA base model, then sweep kernel heads on its features.

Steps (each is skipped if its output already exists, so the script can be re-run safely):
  1. Train plain InfLoRA with fewer epochs per task (--epochs). trainer.py saves
     task_<t>.pth after every task under logs/.../<log_tag>/<seed>/.
  2. Extract features from those checkpoints, one cache per seed:
     <out>/features_seed<k>.pt
  3. Evaluate every head in --heads on all seeds, compared with InfLoRA's own
     concatenated heads: <out>/sweep.json and <out>/sweep.txt.

Examples:
  # quick pipeline check (1 epoch, training loop capped at ~10 batches per epoch)
  python tools/run_kernel_sweep.py --smoke --device 0

  # small base model on CIFAR-100 (5 epochs per task instead of 20, 1 seed)
  python tools/run_kernel_sweep.py --config configs/cifar100_inflora.json --epochs 5 --seeds 0 --device 0

  # ImageNet-R (10 tasks) from a dataset folder elsewhere, 2 seeds
  python tools/run_kernel_sweep.py --config configs/mimg10_inflora.json --data_path /data/imagenet-r \
      --epochs 10 --seeds 0 1 --device 0
"""
import argparse
import copy
import json
import logging
import os
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'tools'))


def make_train_config(cfg, epochs, seed, device, smoke=False):
    """Base-model config: plain InfLoRA, shortened schedule, one seed, its own log folder."""
    cfg = copy.deepcopy(cfg)
    if cfg.get('model_name') == 'InfLoRA_KH':  # the base model is plain InfLoRA
        cfg['model_name'] = 'InfLoRA'
    cfg.pop('kernel_head', None)
    cfg['init_epoch'] = epochs
    cfg['epochs'] = epochs
    cfg['seed'] = [seed]
    cfg['device'] = str(device)
    cfg['debug'] = bool(smoke)
    cfg['log_tag'] = 'kh_base_e{}{}'.format(epochs, '_smoke' if smoke else '')
    return cfg


def checkpoint_dir(cfg, seed):
    from trainer import _logdir
    return os.path.join(_logdir(cfg), str(seed))


def has_all_checkpoints(ckpt_dir, n_tasks):
    return all(os.path.exists(os.path.join(ckpt_dir, 'task_{}.pth'.format(t))) for t in range(n_tasks))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', default='configs/cifar100_inflora.json', help='InfLoRA config to train the base model with')
    p.add_argument('--epochs', type=int, default=5, help='epochs per task (the full configs use 20 / 50)')
    p.add_argument('--seeds', type=int, nargs='+', default=[0])
    p.add_argument('--device', default='0', help='single GPU id')
    p.add_argument('--heads', default='configs/kernel_head_sweep.json')
    p.add_argument('--only', nargs='*', default=None, help='only these head names from --heads')
    p.add_argument('--out', default=None, help='output folder (default: kh_runs/<dataset>_e<epochs>[_smoke])')
    p.add_argument('--smoke', action='store_true', help='1 epoch and capped training loop: checks the pipeline only')
    p.add_argument('--extract_batch_size', type=int, default=128)
    p.add_argument('--data_path', default=None, help="dataset folder; overrides the config's data_path")
    a = p.parse_args()

    if a.data_path:
        a.data_path = os.path.abspath(a.data_path)  # resolve before changing directory
    os.chdir(ROOT)  # trainer and data paths are relative to the repository root
    with open(a.config) as fh:
        base_cfg = json.load(fh)
    if a.data_path:
        base_cfg['data_path'] = a.data_path
    epochs = 1 if a.smoke else a.epochs
    out = a.out or os.path.join('kh_runs', '{}_e{}{}'.format(base_cfg['dataset'], epochs, '_smoke' if a.smoke else ''))
    os.makedirs(out, exist_ok=True)
    device = torch.device('cuda:{}'.format(a.device) if torch.cuda.is_available() else 'cpu')
    timings = {}

    # 1. train (trainer.py configures logging to stdout + the run's log file)
    from trainer import train
    ckpt_dirs = {}
    t0 = time.time()
    for seed in a.seeds:
        cfg = make_train_config(base_cfg, epochs, seed, a.device, a.smoke)
        ckpt_dirs[seed] = checkpoint_dir(cfg, seed)
        if has_all_checkpoints(ckpt_dirs[seed], cfg['total_sessions']):
            print('[run_kernel_sweep] seed {}: checkpoints found in {}, skipping training'.format(seed, ckpt_dirs[seed]))
            continue
        cfg['config'] = a.config
        train(cfg)
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    timings['train_s'] = round(time.time() - t0)
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

    # 2. extract features
    from extract_features import extract_cache
    caches = []
    t0 = time.time()
    for seed in a.seeds:
        path = os.path.join(out, 'features_seed{}.pt'.format(seed))
        if os.path.exists(path):
            logging.info('[run_kernel_sweep] seed %d: using cached features %s', seed, path)
            caches.append(torch.load(path, map_location='cpu'))
            continue
        cfg = make_train_config(base_cfg, epochs, seed, a.device, a.smoke)
        cache = extract_cache(cfg, ckpt_dirs[seed], seed, device, a.extract_batch_size, cfg.get('num_workers', 8))
        if len(cache['steps']) != cfg['total_sessions']:
            raise RuntimeError('expected {} tasks in {}, found {}'.format(cfg['total_sessions'], ckpt_dirs[seed], len(cache['steps'])))
        torch.save(cache, path)
        caches.append(cache)
    timings['extract_s'] = round(time.time() - t0)

    # 3. sweep heads
    from eval_kernel_heads import format_table, run_sweep, save_json
    with open(a.heads) as fh:
        heads = json.load(fh)
    t0 = time.time()
    results = run_sweep(caches, heads, device, only=a.only)
    timings['sweep_s'] = round(time.time() - t0)
    results['run'] = {'config': a.config, 'epochs_per_task': epochs, 'seeds': a.seeds, 'smoke': a.smoke,
                      'checkpoints': ckpt_dirs, 'heads_file': a.heads, 'timings': timings}
    save_json(results, os.path.join(out, 'sweep.json'))
    table = format_table(results)
    header = 'config={} epochs/task={} seeds={}{}\n'.format(a.config, epochs, a.seeds, ' (SMOKE: not a real result)' if a.smoke else '')
    with open(os.path.join(out, 'sweep.txt'), 'w') as fh:
        fh.write(header + table + '\n')
    print('\n' + header + table)
    print('\nresults in {}/ (sweep.txt, sweep.json); timings {}'.format(out, timings))


if __name__ == '__main__':
    main()
