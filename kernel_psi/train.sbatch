#!/bin/bash
# InfLoRA with a kernel feature map between the factors: delta_t(x) = A_t psi_t(B_t x)  (models/kernel_adapter.py).
# One array task per (split, variant); index = 8 * split + variant.
#   split    0: ImageNet-R/10 (configs/mimg10_inflora.json)   1: ImageNet-R/20 (configs/mimg20_inflora.json)
#   variant  0: InfLoRA (baseline)      1: linear128 (control: same params, still rank r)
#            2: relu128                  3: rff128 s=0.5     4: rff128 s=1     5: rff128 s=2
#            6: relu32                   7: rff32 s=1
#
#   sbatch --array=0-15 kernel_psi/train.sbatch      # everything, seed 0
#   sbatch --array=0-7  kernel_psi/train.sbatch      # ImageNet-R/10 only
#   SEED=1 sbatch --array=0-15 kernel_psi/train.sbatch
#
# Edit the paths below first. Compute nodes are offline: the ViT-B/16 in21k .npz must already be in
# $(python -c "import torch; print(torch.hub.get_dir())")/checkpoints/.
#
#SBATCH --job-name=kpsi
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=kpsi_%A_%a.log

REPO=/path/to/gnr650_fproj
VENV=$REPO/.venv
IMAGENET_R=/path/to/imagenet-r
SEED=${SEED:-0}
EXTRA=${EXTRA:-}          # more --set overrides, e.g. EXTRA="epochs=10 init_epoch=10" for a quick pass

cd "$REPO" || exit 1
source "$VENV/bin/activate"

IDX=${SLURM_ARRAY_TASK_ID:-0}
case $((IDX / 8)) in
  0) CONFIG=configs/mimg10_inflora.json ;;
  1) CONFIG=configs/mimg20_inflora.json ;;
  *) echo "unknown array index $IDX"; exit 1 ;;
esac
case $((IDX % 8)) in
  0) PSI="psi=none" ;;
  1) PSI="psi=linear psi_dim=128" ;;
  2) PSI="psi=relu psi_dim=128" ;;
  3) PSI="psi=rff psi_dim=128 psi_sigma=0.5" ;;
  4) PSI="psi=rff psi_dim=128 psi_sigma=1" ;;
  5) PSI="psi=rff psi_dim=128 psi_sigma=2" ;;
  6) PSI="psi=relu psi_dim=32" ;;
  7) PSI="psi=rff psi_dim=32 psi_sigma=1" ;;
esac

echo "config=$CONFIG $PSI seed=$SEED $EXTRA"
python main.py --config "$CONFIG" --device 0 \
    --set $PSI "seed=[$SEED]" "data_path=$IMAGENET_R" "num_workers=${SLURM_CPUS_PER_TASK:-8}" $EXTRA
