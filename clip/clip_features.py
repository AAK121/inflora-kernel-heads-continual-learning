"""Frozen CLIP features, class-name text embeddings and the zero-shot baseline, cached in
the same per-step format as tools/extract_features.py, so the kernel-head code reads it unchanged.

CLIP is never trained here, so every image is encoded once and each class-incremental
step is a slice of those features:
  step t:  train_f / train_y = task t's training images
           test_f  / test_y  = test images of all classes seen so far (0 .. total-1)
           base_pred         = zero-shot CLIP prediction among the classes seen so far
The class-name text embeddings (prompt-ensembled) are stored in meta['text_emb'] for the
text-fusion heads in clip/fusion.py.
"""
import json
import logging
import os
import re
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Generic prompts plus rendition-style prompts (ImageNet-R is art, cartoons, sketches, ...).
TEMPLATES = {
    'single': ['a photo of a {}.'],
    'ensemble': [
        'a photo of a {}.', 'a bad photo of a {}.', 'a photo of the large {}.', 'a photo of the small {}.',
        'a drawing of a {}.', 'a painting of a {}.', 'a sketch of a {}.', 'a cartoon {}.',
        'a rendition of a {}.', 'art of the {}.', 'a sculpture of a {}.', 'a tattoo of a {}.',
        'an embroidered {}.', 'a toy {}.', 'a plastic {}.', 'graffiti of a {}.',
    ],
}


# ---------------------------------------------------------------------------
# Class names, in the DataManager's label order
# ---------------------------------------------------------------------------

def parse_wnid_names(text):
    """{wnid: name} from lines like 'n01443537 goldfish' (as in ImageNet-R's README.txt)."""
    names = {}
    for line in text.splitlines():
        m = re.match(r'\s*(n\d{8})[\s,:]+(.+?)\s*$', line)
        if m:
            names[m.group(1)] = m.group(2).split(',')[0].strip()
    return names


def _clean(name):
    return name.replace('_', ' ').strip()


def original_class_names(dataset, data_path, override=None):
    """Names indexed by the dataset's ORIGINAL label (before DataManager's class_order)."""
    if override:
        with open(override) as fh:
            obj = json.load(fh)
        if isinstance(obj, dict) and dataset.lower() == 'imagenet_r':
            wnids = sorted(d for d in os.listdir(os.path.join(data_path, 'train'))
                           if os.path.isdir(os.path.join(data_path, 'train', d)))
            return [_clean(obj[w]) for w in wnids]
        if isinstance(obj, dict):
            return [_clean(obj[str(i)] if str(i) in obj else obj[i]) for i in range(len(obj))]
        return [_clean(n) for n in obj]

    name = dataset.lower()
    if name == 'cifar100':
        from torchvision import datasets
        return [_clean(c) for c in datasets.CIFAR100(data_path, train=False, download=False).classes]
    if name == 'domainnet':
        from utils.class_names import domainnet_classnames
        return [_clean(domainnet_classnames[i]) for i in range(len(domainnet_classnames))]
    if name == 'imagenet_r':
        train_dir = os.path.join(data_path, 'train')
        wnids = sorted(d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d)))
        readme = os.path.join(data_path, 'README.txt')
        table = {}
        if os.path.exists(readme):
            with open(readme, encoding='utf8', errors='ignore') as fh:
                table = parse_wnid_names(fh.read())
        missing = [w for w in wnids if w not in table]
        if missing:
            raise RuntimeError(
                'No class names for {} ImageNet-R folders (e.g. {}). Expected "<wnid> <name>" lines in {}. '
                'Pass --class_names with a JSON file mapping wnid -> name.'.format(len(missing), missing[:3], readme))
        return [_clean(table[w]) for w in wnids]
    raise ValueError('No class-name source for dataset {}; pass --class_names'.format(dataset))


def ordered_class_names(original_names, class_order):
    """class_order[i] is the original label of new label i (utils/data_manager.py)."""
    return [original_names[c] for c in class_order]


# ---------------------------------------------------------------------------
# Building the cache
# ---------------------------------------------------------------------------

def make_steps(train_f, train_y, test_f, test_y, increments, text_emb, logit_scale):
    """Slice once-computed features into class-incremental steps (same keys as the InfLoRA cache)."""
    steps, known = [], 0
    for inc in increments:
        total = known + inc
        tr = (train_y >= known) & (train_y < total)
        te = test_y < total
        logits = logit_scale * test_f[te] @ text_emb[:total].t()
        steps.append({'known': known, 'total': total,
                      'train_f': train_f[tr].clone(), 'train_y': train_y[tr].clone(),
                      'test_f': test_f[te].clone(), 'test_y': test_y[te].clone(),
                      'base_pred': logits.argmax(dim=1)})
        known = total
    return steps


class _ImageList(Dataset):
    def __init__(self, data, targets, transform, use_path):
        self.data, self.targets, self.transform, self.use_path = data, targets, transform, use_path

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        from PIL import Image
        img = Image.open(self.data[i]).convert('RGB') if self.use_path else Image.fromarray(self.data[i])
        return self.transform(img), int(self.targets[i])


@torch.no_grad()
def encode_images(model, preprocess, data, targets, use_path, device, batch_size, num_workers):
    loader = DataLoader(_ImageList(data, targets, preprocess, use_path), batch_size=batch_size,
                        shuffle=False, num_workers=num_workers)
    feats, labels = [], []
    for x, y in loader:
        f = model.encode_image(x.to(device)).float()
        feats.append((f / f.norm(dim=1, keepdim=True)).cpu())
        labels.append(y)
    return torch.cat(feats), torch.cat(labels).long()


@torch.no_grad()
def encode_class_names(model, names, templates, device):
    from models.clip import clip as clip_lib
    embs = []
    for name in names:
        t = model.encode_text(clip_lib.tokenize([tpl.format(name) for tpl in templates]).to(device)).float()
        t = t / t.norm(dim=1, keepdim=True)
        e = t.mean(dim=0)
        embs.append(e / e.norm())
    return torch.stack(embs).cpu()


def build_clip_cache(cfg, clip_model, device, class_names_file=None, templates='ensemble',
                     batch_size=256, num_workers=8, seed=0):
    """cfg: a dataset config dict (dataset, data_path, init_cls, increment, shuffle)."""
    from models.clip import clip as clip_lib
    from utils.data_manager import DataManager

    torch.manual_seed(seed)  # as trainer.py does before building the DataManager (ImageNet-R split)
    dm = DataManager(cfg['dataset'], cfg['shuffle'], seed, cfg['init_cls'], cfg['increment'], cfg)
    names = ordered_class_names(original_class_names(cfg['dataset'], cfg['data_path'], class_names_file),
                                dm._class_order)
    logging.info('%d classes, e.g. %s', len(names), names[:5])

    model, preprocess = clip_lib.load(clip_model, device=device, jit=False)
    model.eval()
    n_cls = len(names)
    tr_data, tr_y, _ = dm.get_dataset(np.arange(n_cls), source='train', mode='test', ret_data=True)
    te_data, te_y, _ = dm.get_dataset(np.arange(n_cls), source='test', mode='test', ret_data=True)
    train_f, train_y = encode_images(model, preprocess, tr_data, tr_y, dm.use_path, device, batch_size, num_workers)
    test_f, test_y = encode_images(model, preprocess, te_data, te_y, dm.use_path, device, batch_size, num_workers)
    text_emb = encode_class_names(model, names, TEMPLATES[templates], device)
    logit_scale = model.logit_scale.exp().item()
    increments = [dm.get_task_size(t) for t in range(dm.nb_tasks)]
    del model
    if torch.device(device).type == 'cuda':
        torch.cuda.empty_cache()
    return {'meta': {'dataset': cfg['dataset'], 'class_num': cfg['init_cls'], 'increments': increments,
                     'clip_model': clip_model, 'templates': templates, 'class_names': names,
                     'text_emb': text_emb, 'logit_scale': logit_scale, 'seed': seed},
            'steps': make_steps(train_f, train_y, test_f, test_y, increments, text_emb, logit_scale)}
