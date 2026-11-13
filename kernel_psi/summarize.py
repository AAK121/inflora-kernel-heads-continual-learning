"""Collect InfLoRA / psi-adapter results from logs/ into one table.

    python kernel_psi/summarize.py              # all InfLoRA runs under logs/
    python kernel_psi/summarize.py --logs logs/ImageNet_R

final = accuracy on all classes after the last task; avg = mean of the per-task accuracies (both from the
'CNN top1 curve' line that trainer.py logs after every task). Mean +- std over seeds.
"""
import argparse
import ast
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

CURVE = re.compile(r'CNN top1 curve: (\[.*\])')


def read_curve(log):
    curve = None
    for line in Path(log).read_text(errors='replace').splitlines():
        m = CURVE.search(line)
        if m:
            curve = ast.literal_eval(m.group(1))
    return curve


def collect(root):
    """{(dataset, split, variant): {seed: curve}} for logs/<dataset>/<split>/InfLoRA/<optim>/<rank>/<run>/<seed>.log"""
    runs = defaultdict(dict)
    for log in sorted(Path(root).rglob('*.log')):
        parts = log.parts
        if 'InfLoRA' not in parts:
            continue
        i = parts.index('InfLoRA')
        if len(parts) != i + 5:
            continue
        dataset, split, run = parts[i - 2], parts[i - 1], parts[i + 3]
        variant = run.split('_psi-', 1)[1] if '_psi-' in run else 'InfLoRA'
        curve = read_curve(log)
        if curve:
            runs[(dataset, split, variant)][log.stem] = curve
    return runs


def order(variant):
    return (variant != 'InfLoRA', not variant.startswith('linear'), variant)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--logs', default='logs')
    args = p.parse_args()
    runs = collect(args.logs)
    if not runs:
        print(f'no InfLoRA logs with a "CNN top1 curve" line under {args.logs}')
        return
    print(f"{'dataset':<12} {'split':<12} {'variant':<14} {'seeds':<7} {'tasks':<6} {'final':>14} {'avg':>14}")
    for key in sorted(runs, key=lambda k: (k[0], k[1], order(k[2]))):
        by_seed = runs[key]
        n_tasks = sorted({len(c) for c in by_seed.values()})
        final = np.array([c[-1] for c in by_seed.values()])
        avg = np.array([np.mean(c) for c in by_seed.values()])
        tasks = '/'.join(map(str, n_tasks))  # differing counts = some seeds unfinished
        print(f'{key[0]:<12} {key[1]:<12} {key[2]:<14} {",".join(sorted(by_seed)):<7} {tasks:<6} '
              f'{final.mean():8.2f} +-{final.std():4.2f} {avg.mean():8.2f} +-{avg.std():4.2f}')


if __name__ == '__main__':
    main()
