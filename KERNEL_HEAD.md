# Kernel classifier head for InfLoRA

InfLoRA trains one linear head per task, only on that task's classes, and concatenates their logits at test time. Most of its error comes from picking the wrong task: on CIFAR-100, accuracy is 98.65% when the task ID is given but 86.33% overall.

This code adds an exemplar-free closed-form head that scores all classes on one scale:

```
h = φ(f)                       # f = 768-d CLS feature, φ = fixed random-feature map
G += h hᵀ ,  Q += h yᵀ          # running sums after each task (no stored images)
W  = (G + λI)⁻¹ Q ,  logits = φ(f) W
```

- λ is re-selected after every task by GCV, which needs only `G`, `Q` and the sample count.
- InfLoRA's training is unchanged; the original heads are still used to train the adapters.
- The log reports both heads: `CNN ...` is InfLoRA's own head, `KH ...` is the kernel head.

**Default:** RanPAC-style random ReLU features, `D = 10000`, no input normalization, λ chosen by GCV.

- RBF random Fourier features and a no-lift linear head are included in the sweep.

## Files

| File | What it is |
|---|---|
| `methods/kernel_head.py` | Feature maps (`linear`, `relu`, `rbf`), the analytic head, GCV, and the mixin used by the learner |
| `methods/inflora_kh.py` | `InfLoRA_KH` = InfLoRA + kernel head (`model_name: "InfLoRA_KH"`) |
| `tools/extract_features.py` | Caches features from saved checkpoints (`task_<t>.pth`) |
| `tools/eval_kernel_heads.py` | Sweeps head configs over cached features |
| `tools/run_kernel_sweep.py` | One command: train a small base model → extract → sweep |
| `configs/*_inflora_kh*.json` | Training configs (`_smoke` = 1 epoch, 1 seed) |
| `configs/kernel_head_sweep.json` | Head variants compared by the sweep |
| `tests/test_kernel_head.py` | Unit and integration tests (no data or GPU needed) |

## 1. Run the tests

```bash
conda activate python3_8            # the environment from environment.yaml
cd <repo root>
python -m pytest tests -v           # or: python -m unittest discover -s tests -v
```

Expect all tests to pass. `test_cuda_matches_cpu` runs only if a GPU is visible. `test_factory_registration` runs only if the full training environment (timm, sklearn, ...) imports.

## 2. One command: train a small base model and run the whole sweep

[tools/run_kernel_sweep.py](tools/run_kernel_sweep.py) does three steps:

1. Trains plain InfLoRA with a shortened schedule. Checkpoints go to `logs/.../kh_base_e<epochs>/<seed>/`.
2. Extracts features from the checkpoints.
3. Evaluates every head in `configs/kernel_head_sweep.json`.

Every step is skipped if its output already exists, so an interrupted run can simply be restarted.

```bash
# 1) pipeline check first: 1 epoch, training loop capped at ~10 batches (numbers are meaningless)
python tools/run_kernel_sweep.py --smoke --device 0

# 2) the real (small) run: CIFAR-100, 5 epochs per task instead of 20, 1 seed
python tools/run_kernel_sweep.py --config configs/cifar100_inflora.json --epochs 5 --seeds 0 --device 0

# optional: ImageNet-R, or more seeds
python tools/run_kernel_sweep.py --config configs/mimg10_inflora.json --epochs 10 --seeds 0 1 --device 0
```

Results go to `kh_runs/<dataset>_e<epochs>/`:
- `sweep.txt` is the table, with InfLoRA's own head as the first row.
- `sweep.json` holds per-task curves, the chosen λ per task, and timings.
- `features_seed<k>.pt` are the cached features. Re-running with another `--heads` file reuses them.

A shortened base model is weaker than the paper's, so compare the kernel heads with the `InfLoRA (concat heads)` row from **the same run**, not with the paper's numbers.

## 2a. Evaluate heads on existing checkpoints (manual steps)

`trainer.py` already saves `logs/<...>/<seed>/task_<t>.pth` after every task. If those files are on the machine, no retraining is needed:

```bash
# one feature cache per seed (GPU; downloads the ViT weights like training does)
python tools/extract_features.py --config configs/cifar100_inflora.json \
    --ckpt_dir logs/cifar100/10_10_sip/InfLoRA/adam/10/0.95_1.0-0.0005/0 \
    --seed 0 --device 0 --out features/cifar100_seed0.pt

# compare all head variants, plus InfLoRA's own head, on all cached seeds
python tools/eval_kernel_heads.py --features "features/cifar100_seed*.pt" \
    --heads configs/kernel_head_sweep.json --device 0 --out results/cifar100_heads.json
```

The script prints a table of final accuracy (mean ± std over seeds), average incremental accuracy, and task-identification accuracy. Use `--only relu10k_none rbf5k_centerl2_m1` to run a subset.

For ImageNet-R, use `configs/mimg10_inflora.json` and that run's checkpoint directory.

## 2b. Train with the kernel head online

```bash
# smoke test: 1 epoch per task, 1 seed, short training loop
python main.py --device 0 --config configs/cifar100_inflora_kh_smoke.json

# full runs (5 seeds)
python main.py --device 0 --config configs/cifar100_inflora_kh.json
python main.py --device 0 --config configs/mimg10_inflora_kh.json
```

Logs go to `logs/<dataset>/<...>/InfLoRA_KH/...`, and a `log_tag` key in the config adds a subfolder. Look for these lines:

```
KH: {'total': ..., 'old': ..., 'new': ..., 'task_acc': ...}
KH top1 curve: [...]
KH lambda curve: [...]
```

Compare `KH top1 curve` with `CNN top1 curve` in the same log.

## Config keys (`"kernel_head": {...}`)

| Key | Default | Options |
|---|---|---|
| `map` | `relu` | `relu` (random ReLU / arc-cosine), `rbf` (Gaussian RFF), `linear` (no lift) |
| `dim` | 10000 | feature dimension D (even for `rbf`); `G` takes D²×8 bytes, i.e. 800 MB at 10k |
| `input_norm` | `none` | `none`, `l2`, `center_l2` (mean from task 1, then frozen) |
| `bandwidth_mult` | 1.0 | `rbf` only: σ = mult × median pairwise distance on task 1 |
| `ridge` | `gcv` | `gcv` (all tasks, from the sums), `val` (RanPAC: 20% hold-out of the new task), or a number |
| `ridge_grid` | 1e-8 … 1e8 | half-decade grid; a warning is logged if the chosen λ sits on the edge |
| `seed`, `dtype`, `batch_size`, `chunk_size` | 0, `float64`, 128, 2048 | |

**Cost.** Per task: one forward pass over the new task's training set, one `D×D` eigendecomposition (for GCV) and one Cholesky solve. With D = 10000 in float64 on a GPU, that is roughly tens of seconds per task, depending on the GPU.

## Numbers to beat (ViT-B/16 IN-21k, final accuracy)

| Benchmark | InfLoRA (paper) | Our InfLoRA re-run | InfLoRA+CA (paper) |
|---|---|---|---|
| CIFAR-100 / 10 | 86.51 | 86.33 | 91.59 |
| ImageNet-R / 10 | 75.65 | 75.34 | 79.78 |

InfLoRA+CA (`configs/*_infloraca.json`) is the authors' own classifier fix, so it is the fair target for a better head.
