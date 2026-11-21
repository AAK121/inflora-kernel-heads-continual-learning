# Where Adapters Work: Kernel Heads for Exemplar-Free Continual Learning

Class-incremental learning on top of **InfLoRA** (CVPR 2024). We show that most of InfLoRA's error comes from choosing the wrong task at test time, not from weak features. An **exemplar-free, closed-form kernel ridge head** on the same frozen features raises final ImageNet-R accuracy by **about 4.5 points**, and a fused kernel + text head on frozen CLIP reaches **83.55%**.

![PyTorch](https://img.shields.io/badge/PyTorch-ViT--B%2F16-EE4C2C?logo=pytorch)
![Continual learning](https://img.shields.io/badge/setting-class--incremental%2C%20exemplar--free-blue)

## Overview

**Continual learning:** a model learns tasks one after another, without access to old data, and must not forget earlier tasks. In the **class-incremental** setting no task ID is given at test time, so the model has to choose among every class seen so far.

**InfLoRA** freezes a pre-trained ViT-B/16 and trains one low-rank branch per task. Each branch is projected so it cannot interfere with earlier tasks. Its features are strong, but each task gets its own linear head trained only on that task's classes. At test time the heads' logits are simply concatenated, and they were never calibrated against each other.

**The diagnosis.**
- On CIFAR-100, InfLoRA scores 98.65% when the task ID is given but 86.33% without it.
- On ImageNet-R (10 tasks), about 22% of test images are routed to the wrong task. That accounts for roughly 85% of all errors: 22.1 of 25.9 points.

So the bottleneck is the decision rule, not the representation.

## Method

```
frozen ViT-B/16 + InfLoRA adapters (trained as usual) ──► f  (768-d CLS feature)
                                                            │
                     φ(f): fixed random-feature lift (ReLU or RBF, D = 5k–10k)
                                                            │
     running sums after each task (no stored images):   G += φ(f)φ(f)ᵀ ,  Q += φ(f) yᵀ
     closed-form head:                                   W  = (G + λI)⁻¹ Q ,  logits = φ(f) W
     λ chosen after every task by generalized cross-validation (GCV), using only G, Q and n
```

- **Kernel classifier head** (`methods/kernel_head.py`, `methods/inflora_kh.py`). This head replaces the per-task linear heads with one ridge-regression head that scores all classes on the same scale. It needs no exemplars, no gradient steps and no change to InfLoRA's training. The random-ReLU variant follows RanPAC.
- **Kernel head on frozen CLIP** (`clip/`). The same head on frozen CLIP ViT-B/16 image features, optionally fused with zero-shot text logits. The fusion weight β is picked on a 20% hold-out of each new task.
- **Kernel inside the adapter** (branch `kernel-adapter`). This puts a kernel lift in place of, or between, InfLoRA's B and A matrices. It is a negative result: it gave no gain.
- **Geometric score and subspace pruning** (team report). Also explored, but their code is not part of this repository:
  - a training-free nearest-prototype "proto-energy" rule;
  - prototype-guided pruning of DualGPM's protected subspace to free capacity for new tasks.

## Results

ImageNet-R (200 classes, 10 tasks of 20 classes), ViT-B/16 pre-trained on ImageNet-21k, final class-incremental accuracy.

**Decision rules on the same InfLoRA features** (team report):

| Decision head | Final acc. (%) |
| --- | --- |
| InfLoRA concatenated per-task heads | 74.2 |
| Proto-energy (nearest prototype, training-free) | 75–76 |
| **RBF / ReLU kernel ridge head (ours)** | **78.5–78.8** |

**Head sweep, single seed, 10 epochs per task** (`checkpoints/kernel_sweep/sweep.txt`):

| Head | Final | Task-ID acc. |
| --- | --- | --- |
| InfLoRA (concat heads) | 73.78 | n/a |
| Random ReLU, D = 10k | **80.02** | 0.848 |
| Random ReLU, D = 10k, centered + L2 | 79.83 | 0.847 |
| RBF random Fourier features, D = 10k | 79.43 | 0.844 |
| Linear ridge (no lift) | 77.28 | 0.832 |

**Frozen CLIP ViT-B/16, no training at all** (`kh_runs/clip_ImageNet_R_ViT-B-16/clip_sweep.txt`):

| Head | Final | Avg. incremental |
| --- | --- | --- |
| CLIP zero-shot | 77.20 | 80.38 |
| ReLU kernel head | 81.20 | 83.10 |
| **ReLU kernel head + text fusion** | **83.55** | **86.07** |

Reproduction of the InfLoRA baselines: CIFAR-100/10 gave 86.33% (paper: 86.51%) and ImageNet-R/10 gave 75.34% (paper: 75.65%).

## Findings

- **Most class-incremental errors are wrong-task errors.** About 75–85% of mistakes pick the wrong task, not the wrong class within the right task.
- **A better head beats a better adapter.** A closed-form kernel head on unchanged features gave the largest gain (about +4.5 points). Putting the kernel inside the adapter gave nothing.
- **The lift matters.** Lifting with random ReLU or RBF features beat a linear ridge head by about 2.7 points, and GCV chose λ without any validation data.
- **Text helps on CLIP.** Fusing zero-shot text logits added 2.35 points on top of the kernel head on frozen CLIP.
- **Training-free geometry helps a little.** Scoring by prototype energy added 1.5–1.7 points over the concatenated heads.
- **Subspace pruning is a trade-off.** Windowed pruning freed DualGPM capacity and helped the newest task (+6.8 points on task 19), but cost final accuracy on ImageNet-R. On DomainNet it freed about 40% of protected dimensions at parity accuracy (73.80% → 73.91%).

## Repository structure

```
.
├── methods/
│   ├── inflora.py              # InfLoRA learner (DualGPM-designed LoRA branches)
│   ├── kernel_head.py          # feature lifts, analytic ridge head, GCV
│   ├── inflora_kh.py           # InfLoRA + kernel head (model_name "InfLoRA_KH")
│   └── inflora_ca*.py, sprompt_*.py, ...   # baselines from the InfLoRA codebase
├── clip/                       # frozen-CLIP features, kernel heads, text fusion, sweep
├── tools/                      # feature extraction, head sweeps, one-command pipeline
├── models/, dataloaders/, utils/   # ViT-B/16 + LoRA, datasets (from InfLoRA)
├── configs/                    # InfLoRA, kernel-head and sweep configs
├── tests/                      # unit tests, no data or GPU needed
├── checkpoints/kernel_sweep/, kh_runs/, logs/   # result tables and training logs
├── KERNEL_HEAD.md              # detailed kernel-head guide
├── docs/final_presentation.pdf # team's final presentation
└── docs/INFLORA_README.md      # original InfLoRA README (datasets, baseline commands)
```

Branches:
- `main`: InfLoRA plus the kernel head and the CLIP experiments.
- `kernel-adapter`: the kernel-inside-the-adapter experiments, with a LaTeX write-up in `report/`.
- `inflora-original`: the upstream InfLoRA code.

## How to run

```bash
conda env create -f environment.yaml && conda activate python3_8    # or: uv sync (pyproject.toml)
python -m pytest tests -v                                            # 46 tests; 5 need a GPU or the full training env

# one command: train a short InfLoRA base model, extract features, sweep all heads
python tools/run_kernel_sweep.py --smoke --device 0                  # pipeline check
python tools/run_kernel_sweep.py --config configs/mimg10_inflora.json --epochs 10 --seeds 0 --device 0

# train InfLoRA with the kernel head online
python main.py --device 0 --config configs/mimg10_inflora_kh.json

# frozen-CLIP kernel heads and text fusion
python clip/run_clip_sweep.py --config configs/mimg10_inflora.json --data_path /path/to/imagenet-r --device 0
```

Dataset setup for CIFAR-100, ImageNet-R and DomainNet is described in [docs/INFLORA_README.md](docs/INFLORA_README.md). More options are in [KERNEL_HEAD.md](KERNEL_HEAD.md) and [clip/README.md](clip/README.md).

## Team and acknowledgements

- **Team (GNR 650 final project):** Anshul Kumar Yadav, Rishi Raj Vishwakarma, Samridh Agarwal, Sujith Kumar A, Yash Sharma, Deepak Mewada, Dheeraj Mishra, Alok Ashok Kale, Eeshan Narula.
- Built for **GNR650: Advanced Topics in Deep Learning for Image Analysis**, Indian Institute of Technology Bombay, under the guidance of **Prof. Biplab Banerjee**.
- The base code is the official [InfLoRA implementation](https://github.com/liangyanshuo/InfLoRA) by Yan-Shuo Liang and Wu-Jun Li (MIT License, see `LICENSE`). Its commit history is preserved here.

## References

1. Y.-S. Liang, W.-J. Li. *InfLoRA: Interference-Free Low-Rank Adaptation for Continual Learning.* CVPR 2024.
2. M. D. McDonnell et al. *RanPAC: Random Projections and Pre-trained Models for Continual Learning.* NeurIPS 2023.
3. A. Rahimi, B. Recht. *Random Features for Large-Scale Kernel Machines.* NeurIPS 2007.
4. G. H. Golub, M. Heath, G. Wahba. *Generalized Cross-Validation as a Method for Choosing a Good Ridge Parameter.* Technometrics 1979.
5. A. Radford et al. *Learning Transferable Visual Models from Natural Language Supervision (CLIP).* ICML 2021.
6. G. Saha, I. Garg, K. Roy. *Gradient Projection Memory for Continual Learning.* ICLR 2021.
7. E. J. Hu et al. *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR 2022.
8. A. Dosovitskiy et al. *An Image is Worth 16x16 Words (ViT).* ICLR 2021.
9. J. Snell, K. Swersky, R. Zemel. *Prototypical Networks for Few-shot Learning.* NeurIPS 2017.
10. D. Hendrycks et al. *The Many Faces of Robustness (ImageNet-R).* ICCV 2021.
11. G. M. van de Ven, A. S. Tolias. *Three Scenarios for Continual Learning.* arXiv:1904.07734, 2019.
