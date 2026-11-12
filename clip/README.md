# CLIP baseline, kernel heads and text fusion

Frozen OpenAI CLIP (default ViT-B/16). Nothing is trained: no LoRA in either encoder, and no gradient steps.

```
image ──► frozen CLIP image encoder ──► f (512-d, unit norm) ──► kernel head (methods/kernel_head.py)
class names ──► frozen CLIP text encoder (prompt ensemble, once per class) ──► t_c (512-d)

zero-shot  (baseline):  argmax_c  100·cos(f, t_c)                          over classes seen so far
kernel head:            argmax_c  ridge(φ(f))_c                            running sums, λ by GCV
fusion (option 2):      argmax_c  (1−β)·ridge(φ(f))_c + β·softmax(100·cos(f, t_c))
```

β is either fixed or `"auto"`. With `"auto"`, after each task a copy of the head is fit on 80% of the new task, together with all old statistics. The β with the best accuracy on the remaining 20%, scored over all classes seen so far, is kept. No old images are needed.

## Run (from the repo root)

```bash
python clip/run_clip_sweep.py --config configs/mimg10_inflora.json --data_path /path/to/imagenet-r --device 0
python clip/run_clip_sweep.py --config configs/cifar100_inflora.json --device 0
```

- The config only supplies the dataset, `data_path` and the task split, so the tasks match the InfLoRA runs.
- Images are encoded once and cached in `kh_runs/clip_<dataset>_ViT-B-16/features.pt`. Re-runs, for example with another `--heads` file, take seconds to minutes.
- Results go to `kh_runs/clip_<dataset>_ViT-B-16/clip_sweep.txt` (table) and `clip_sweep.json` (per-task curves, λ and β).
- Options:
  - `--clip_model ViT-B/32` or a `.pt` path
  - `--templates single` for "a photo of a {}." only
  - `--only <head names>`
  - `--class_names names.json`

## Offline machine

The loader looks for `~/.cache/clip/ViT-B-16.pt` (about 350 MB) before downloading. Download it elsewhere and copy it there:

```bash
wget https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt
scp ViT-B-16.pt <user>@<remote>:~/.cache/clip/ViT-B-16.pt
```

You can also pass the file directly with `--clip_model /path/to/ViT-B-16.pt`.

## Class names

| Dataset | Source |
|---|---|
| CIFAR-100 | torchvision's CIFAR-100 class list |
| DomainNet | `utils/class_names.py` |
| ImageNet-R | `<data_path>/README.txt` from the official tar, read as lines of the form `n01443537 goldfish` |

If the README is missing or different, pass `--class_names` with a JSON file mapping `{wnid: name}`.

## Heads compared (`clip/clip_heads.json`)

Each head runs with no text, with automatic fusion (`+text`), and one with fixed β = 0.5:
- linear ridge (no lift)
- random ReLU (D = 10k)
- RBF random Fourier features (D = 5k)

Inputs are centered and L2-normalized, since CLIP features are compared by angle.

## Comparing with the InfLoRA runs

CLIP's web-scale pretraining is a very different starting point from InfLoRA's ImageNet-21k ViT. Compare methods **within** a backbone:

- here: kernel head and fusion vs. CLIP zero-shot;
- in `tools/run_kernel_sweep.py`: kernel head vs. InfLoRA's own heads.

Comparing numbers across the two backbones mostly measures the backbone.

Tests: `tests/test_clip.py` (no weights or data needed).
