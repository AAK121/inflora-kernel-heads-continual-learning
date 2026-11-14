# Kernel adapter between InfLoRA's factors: Δ_t(x) = A_t ψ_t(B_t x)

**Question.** Does a kernel adapter perform as well as InfLoRA's linear A_t B_t, or better?

**Background.** Phase 0 (`kernel_bt/`) showed that per-token inputs carry almost no task identity: ν ≈ ι for every input lift. A kernel therefore cannot be used to *separate* tasks in B_t.

**This experiment.** It keeps B_t exactly as in InfLoRA, the frozen projection onto the free subspace, and applies a kernel to the r coordinates u = B_t x. A_t is trained on ψ(u). Each task's adapter thus becomes a random-feature kernel machine on its own subspace.

**Why protection is kept.** ψ(0) = 0, so InfLoRA's interference argument carries over: an input with B_t x = 0 still gets no update, and leakage is at most Lip(ψ)·‖B_t x‖. The realistic goal is more expressiveness per task at InfLoRA's level of interference, not less forgetting.

## ψ variants (`models/kernel_adapter.py`)

| `psi` | ψ(u) | kernel | note |
|---|---|---|---|
| `none` | – | – | plain InfLoRA. Same code path and same log directory as before |
| `linear` | Ωu/√m | linear | **control:** same parameter count as the kernel variants, but still rank r overall |
| `relu` | √(2/m)·relu(Ωu) | arc-cosine | scale-free, no bandwidth |
| `rff` | s√(2/m)·[cos(Ωu/s) − 1, sin(Ωu/s)] | centred Gaussian | bandwidth s = `psi_sigma` · √E‖u‖², set per task |

**Bandwidth calibration.** E‖u‖² = tr(B_t C B_tᵀ) comes from InfLoRA's own `cur_matrix` when B_t is chosen. No extra pass is needed.

**Scale.** Every ψ is scaled so that E‖ψ(u)‖² ≈ ‖u‖², i.e. it starts at the linear adapter's scale.

**Step size (`psi_scale`).** Adam moves every entry of A_t by about lr. With m columns instead of r, the composed rank-r update therefore moves about √(m/r) times faster than InfLoRA's. In the first sweep (`Kernel_adapter_sweep_65429.out`), even the `linear128` control lost 8 points, so the nonlinearity was not the cause.
- `psi_scale` multiplies ψ's output, which under Adam scales the effective A_t step by the same factor.
- `psi_scale=auto` sets it to √(r/m), matching InfLoRA's step size.
- Runs that use it get `_xauto` in their log directory.
- `kernel_psi/run.sh` reruns the sweep with it.

**What else is unchanged.** Ω is frozen and drawn from its own generator (the run seed), so data order and LoRA initialisation match the baseline exactly. Ω and s are saved in the checkpoints. A_t is d×m instead of d×r, and is still zero-initialised. Only A_t is trained, as in InfLoRA.

## Run (cluster)

Edit `REPO` and `IMAGENET_R` at the top of `kernel_psi/train.sbatch` first.

```bash
sbatch --array=0-15 kernel_psi/train.sbatch                 # ImageNet-R/10 and /20 x 8 variants, seed 0
SEED=1 sbatch --array=0-15 kernel_psi/train.sbatch          # more seeds
EXTRA="epochs=10 init_epoch=10" sbatch --array=0-7 kernel_psi/train.sbatch   # quick pass, ImageNet-R/10 only
python kernel_psi/summarize.py                               # table of final / avg accuracy per variant
```

**Array indices** (index = 8·split + variant):
- split 0 is ImageNet-R/10; split 1 is ImageNet-R/20;
- variants 0–7 are InfLoRA, linear128, relu128, rff128 at s = 0.5/1/2, relu32 and rff32 at s = 1.

**Running one config by hand:**

```bash
python main.py --config configs/mimg10_inflora.json --device 0 \
    --set psi=rff psi_dim=128 psi_sigma=1 "seed=[0]" data_path=/path/to/imagenet-r
```

`--set key=value` overrides any config entry; values are parsed as JSON. Each run logs to its own directory, e.g. `logs/ImageNet_R/20_20_sip/InfLoRA/adam/10/0.98_1.0-0.0005_psi-rff128_s1/`. Baseline runs use the original path, so an existing InfLoRA log for the same seed is appended to and its checkpoints are overwritten.

**Cost.** The ψ branch is small: r = 10 → m ≤ 128 features per task. It is recomputed in backward (activation checkpointing), so memory stays close to InfLoRA's. Expect roughly the baseline's wall time.

## Reading the result

- **Kernel vs linear128 control.** Any gain of `relu`/`rff` over the `linear` control is due to the nonlinearity. The control isolates it from the larger A_t.
- **Kernel vs InfLoRA.**
  - Equal or better at both 10 and 20 tasks is the success criterion.
  - Also look at whether the gap grows from 10 to 20 tasks: late tasks get a smaller free subspace, which is where extra expressiveness per direction should matter.
- **rff bandwidth.** The trend over s = 0.5/1/2 should be smooth. A large s makes ψ ≈ linear; a small s makes it saturate and memorise.

**Limitations.**
- Only `InfLoRA` (`methods/inflora.py`) calibrates ψ. Other methods refuse `psi`.
- `get_matrix` / `get_pre_matrix` and per-task rank (`init_param_ada`) are not available with ψ.

**Tests:** `tests/test_kernel_adapter.py`. They cover:
- ψ(0) = 0, the scale, and the RFF kernel;
- no update for inputs orthogonal to the B_t;
- batched computation = sum over tasks;
- `psi=none` = the original forward;
- calibration, gradients only in the current A_t, and checkpointing;
- the global RNG left untouched.
