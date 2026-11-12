"""Tests for clip/ (CLIP baseline, kernel heads on CLIP features, text fusion).

Run from the repo root:  python -m pytest tests -v
No CLIP weights or datasets are needed; the tokenizer test is skipped if ftfy/regex are missing.
"""
import json
import os
import shutil
import sys
import tempfile
import unittest

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, 'clip'), os.path.join(ROOT, 'tools')):
    if p not in sys.path:
        sys.path.insert(0, p)

from clip_features import make_steps, ordered_class_names, original_class_names, parse_wnid_names  # noqa: E402
from fusion import TextFusionHead, zero_shot_probs  # noqa: E402


def clip_like(n_classes=6, per_class=40, dim=32, text_quality=0.5, seed=0):
    """Unit-norm image features around class centers; text embeddings = noisy copies of the centers."""
    g = torch.Generator().manual_seed(seed)
    centers = torch.nn.functional.normalize(torch.randn(n_classes, dim, generator=g), dim=1)
    text = torch.nn.functional.normalize(centers + (1 - text_quality) * torch.randn(n_classes, dim, generator=g), dim=1)

    def sample(n, s):
        gg = torch.Generator().manual_seed(s)
        y = torch.arange(n_classes).repeat_interleave(n)
        f = torch.nn.functional.normalize(centers[y] + 0.35 * torch.randn(len(y), dim, generator=gg), dim=1)
        return f, y

    return sample(per_class, seed + 1), sample(per_class // 2, seed + 2), text


class TestClassNames(unittest.TestCase):

    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_parse_wnid_names(self):
        table = parse_wnid_names('ImageNet-R readme\nn01443537 goldfish\nn01484850 great white shark, white shark\nnot a row\n')
        self.assertEqual(table, {'n01443537': 'goldfish', 'n01484850': 'great white shark'})

    def _imagenet_r(self, readme):
        for w in ('n01484850', 'n01443537'):
            os.makedirs(os.path.join(self.dir, 'train', w))
        if readme is not None:
            with open(os.path.join(self.dir, 'README.txt'), 'w') as fh:
                fh.write(readme)

    def test_imagenet_r_names_follow_sorted_folders(self):
        self._imagenet_r('n01484850 great white shark\nn01443537 goldfish\n')
        self.assertEqual(original_class_names('imagenet_r', self.dir), ['goldfish', 'great white shark'])

    def test_imagenet_r_missing_names_raise(self):
        self._imagenet_r('n01443537 goldfish\n')
        with self.assertRaises(RuntimeError):
            original_class_names('imagenet_r', self.dir)

    def test_override_json(self):
        self._imagenet_r(None)
        path = os.path.join(self.dir, 'names.json')
        with open(path, 'w') as fh:
            json.dump({'n01443537': 'gold_fish', 'n01484850': 'shark'}, fh)
        self.assertEqual(original_class_names('imagenet_r', self.dir, path), ['gold fish', 'shark'])
        with open(path, 'w') as fh:
            json.dump(['a', 'b_c'], fh)
        self.assertEqual(original_class_names('cifar100', self.dir, path), ['a', 'b c'])

    def test_ordered_names(self):
        self.assertEqual(ordered_class_names(['a', 'b', 'c'], [2, 0, 1]), ['c', 'a', 'b'])


class TestSteps(unittest.TestCase):

    def test_make_steps(self):
        (tr_f, tr_y), (te_f, te_y), text = clip_like()
        steps = make_steps(tr_f, tr_y, te_f, te_y, [2, 2, 2], text, 100.0)
        self.assertEqual([(s['known'], s['total']) for s in steps], [(0, 2), (2, 4), (4, 6)])
        for s in steps:
            self.assertTrue(((s['train_y'] >= s['known']) & (s['train_y'] < s['total'])).all())
            self.assertTrue((s['test_y'] < s['total']).all())
            self.assertTrue((s['base_pred'] < s['total']).all())  # zero-shot only over seen classes
        self.assertEqual(len(steps[-1]['test_y']), len(te_y))


class TestFusion(unittest.TestCase):

    def setUp(self):
        (self.f, self.y), (self.ft, self.yt), self.text = clip_like(text_quality=0.6)

    def _fit(self, beta):
        head = TextFusionHead({'map': 'rbf', 'dim': 512, 'input_norm': 'center_l2', 'text_fusion': beta}, 32, self.text, 100.0)
        head.fit_task(self.f, self.y, 6)
        return head

    def test_beta_extremes(self):
        zs = zero_shot_probs(self.ft, self.text, 100.0, 6).argmax(1)
        self.assertTrue(torch.equal(self._fit(1.0).predict(self.ft), zs))
        pure = self._fit(0.0)
        self.assertTrue(torch.equal(pure.predict(self.ft), pure.head.predict(self.ft)))

    def test_auto_beta(self):
        head = self._fit('auto')
        self.assertIn(head.beta, head.beta_grid)
        self.assertEqual(len(head.history), 1)
        acc = lambda p: (p == self.yt).float().mean().item()  # noqa: E731
        worst = min(acc(self._fit(0.0).predict(self.ft)), acc(self._fit(1.0).predict(self.ft)))
        self.assertGreaterEqual(acc(head.predict(self.ft)), worst - 1e-6)

    def test_bad_beta(self):
        with self.assertRaises(ValueError):
            TextFusionHead({'map': 'linear', 'text_fusion': 'sometimes'}, 32, self.text, 100.0)


class TestClipSweep(unittest.TestCase):

    def test_run_on_fake_cache(self):
        from run_clip_sweep import run
        from eval_kernel_heads import format_table
        (tr_f, tr_y), (te_f, te_y), text = clip_like()
        cache = {'meta': {'class_num': 2, 'text_emb': text, 'logit_scale': 100.0},
                 'steps': make_steps(tr_f, tr_y, te_f, te_y, [2, 2, 2], text, 100.0)}
        heads = [{'name': 'lin', 'map': 'linear', 'input_norm': 'center_l2', 'text_fusion': 0},
                 {'name': 'rbf+text', 'map': 'rbf', 'dim': 256, 'input_norm': 'center_l2', 'text_fusion': 'auto'},
                 {'name': 'bad', 'map': 'rbf', 'dim': 255}]
        res = run(cache, heads)
        self.assertEqual(len(res['heads']['rbf+text']['curves'][0]['beta']), 3)
        self.assertIn('error', res['heads']['bad'])
        self.assertIn('CLIP zero-shot', format_table(res, baseline_label='CLIP zero-shot'))

    def test_heads_file_is_valid(self):
        with open(os.path.join(ROOT, 'clip', 'clip_heads.json')) as fh:
            heads = json.load(fh)
        self.assertEqual(len({h['name'] for h in heads}), len(heads))
        text = torch.nn.functional.normalize(torch.randn(4, 16), dim=1)
        for h in heads:
            cfg = dict(h)
            if 'dim' in cfg:
                cfg['dim'] = 64  # same config, small enough for a quick construction check
            TextFusionHead(cfg, 16, text, 100.0)


class TestClipTokenizer(unittest.TestCase):

    def test_tokenize(self):
        try:
            from models.clip import clip as clip_lib
        except Exception as e:  # pragma: no cover - environment dependent
            self.skipTest('CLIP code not importable (needs ftfy, regex): {}'.format(e))
        tokens = clip_lib.tokenize(['a photo of a goldfish.', 'a sketch of a great white shark.'])
        self.assertEqual(tuple(tokens.shape), (2, 77))


if __name__ == '__main__':
    unittest.main()
