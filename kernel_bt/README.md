# Kernelised B_t: Phase 0 (go/no-go, no training)

The method and every metric below are defined in `report/kernel_inflora.tex`. The literature background is in `agents/00_survey.md`.

**Question.** Can a frozen kernel lift ψ(x) = [x; α·φ(z(x))] give InfLoRA's B_t usable room that linear InfLoRA lacks on long task sequences, without leaking into old tasks? We answer this from cached k/v inputs of the frozen ViT-B/16, before writing any training code.

## Run (on the cluster)

```bash
# edit the paths at the top first
sbatch --array=0-2 kernel_bt/phase0.sbatch   # ImageNet-R with 10, 20 and 50 tasks
sbatch --array=0-4 kernel_bt/phase0.sbatch   # + CIFAR-100 with 10 and 50 tasks
```

A single run looks like this:

```bash
python kernel_bt/phase0.py --config configs/mimg10_inflora.json --data_path /path/to/imagenet-r \
    --init_cls 4 --increment 4 --out kbt_runs/phase0_ImageNet_R_50
```

**Steps**
1. **Extract.** One forward pass over each task's train and test images, with adapters off and the same split as InfLoRA. This takes minutes.
2. **Analyse.** The tasks are replayed in order on the GPU, from about 15 minutes for 10 tasks to about 1.5 hours for 50.

Re-runs reuse `<out>/cache/`, so changing analysis flags such as `--sigma_mults` or `--pre_dims 0 128` skips extraction.

**Disk.** The cache takes about 160 MB per task, for all 12 blocks with 5000 + 2000 tokens. Use `--layers 7 8 9 10 11 12` to halve it.

**Outputs** (in `<out>`)
- `phase0_summary.txt`: the tables and the verdict.
- `phase0.json`: every number, per block, lift and task.
- `phase0.log`

## What is measured

Each block is run with linear InfLoRA (DualGPM on the exact `cur_matrix`, with the config's `lamb`/`lame`) and with each lift:
- RBF RFF features with D = 2048 at σ = c × median distance on task 1, for c ∈ {0.25, 0.5, 1, 2, 4};
- ReLU random features.

The kernel block gets its own DualGPM, with threshold 0.90→0.95 and at most 64 directions per task. Before each task t ≥ 2:

| Column | Meaning | Want |
|---|---|---|
| `rho` (table 1) | share of the new task's energy outside the protected M. The saturation curve, which nobody has published for InfLoRA | — |
| `nuLin / nuKer / nuHyb` | **usable signal**: energy of *held-out* new-task tokens inside B_t's row space | high |
| `ioLin / ioKer / ioHyb` | **leakage**: energy of *held-out old-task* tokens inside B_t's row space | low |
| `kshr` | share of the hybrid B_t that the joint SVD put in the kernel block | > 0 |
| `pOldL / pOldK` | energy of held-out old tokens outside M (for the kernel block, the GP posterior variance) | low |
| `ovrL / ovrK` | overlap of old and new principal subspaces ÷ overlap of two halves of the new task | lower for the kernel block |
| `s/dW`, `dB/s` | σ ÷ nearest-neighbour distance within the new task; distance to old tasks ÷ σ | both ≳ 1 |
| `linRs` | residual of fitting K ≈ a·11ᵀ + b·ZZᵀ + υI | near 0 means effectively linear |

All metrics use held-out tokens, so the sample-private part of the kernel, which inflates novelty on training tokens, cannot count as signal.

**Verdict** (design note). **GO** if, for some lift, the late-task (t > T/2) means satisfy ν_hyb ≥ ν_lin + 0.05 and ι_hyb ≤ ι_lin + 0.01 in at least half of the blocks.

Read the verdict next to `linRs` and `s/dW`. A pass at a bandwidth where the kernel is effectively linear, or where s/dW ≪ 1 (memorisation), is suspect.

## Limitations (deliberate, for a cheap first answer)

- **The backbone is the frozen pretrained ViT** (adapters off). In real InfLoRA, the inputs to later blocks drift as the A_t are trained. Block 1 is exact; deeper blocks are approximate.
- **Train tokens use the test transform** (no augmentation). InfLoRA builds `cur_matrix` from augmented images.
- **Kernel and hybrid statistics use 5000 sampled tokens per task.** The linear M uses the exact second moment over all tokens, as InfLoRA does.

If the verdict is GO, Phase 1 implements the lift in `models/vit_inflora.py`, reusing `kernel_bt/lifts.py` and `kernel_bt/dualgpm.py` unchanged.

Tests: `tests/test_kernel_bt.py` (synthetic tensors only). Among them, the torch DualGPM is checked against InfLoRA's own `update_DualGPM`, and the locality mechanism is checked on a toy example.
