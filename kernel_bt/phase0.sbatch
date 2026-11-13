#!/bin/bash
# Phase 0 of the kernelised B_t: offline go/no-go diagnostics (no training).
# One array task per sequence length, so the saturation trend over T is visible:
#   0: ImageNet-R/10   1: ImageNet-R/20   2: ImageNet-R/50   3: CIFAR-100/10   4: CIFAR-100/50
#
#   sbatch --array=0-2 kernel_bt/phase0.sbatch        # ImageNet-R only
#   sbatch --array=0-4 kernel_bt/phase0.sbatch        # + CIFAR-100
#
# Edit the four paths below first. Compute nodes are offline: the ViT-B/16 in21k .npz must already be
# in $(python -c "import torch; print(torch.hub.get_dir())")/checkpoints/ (as for InfLoRA training).
#
#SBATCH --job-name=kbt_phase0
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=kbt_phase0_%A_%a.log

REPO=/path/to/gnr650_fproj
VENV=$REPO/.venv
IMAGENET_R=/path/to/imagenet-r
CIFAR=/path/to/cifar100          # must already hold cifar-100-python/ (download on the login node:
                                  #   python -c "from torchvision import datasets as d; d.CIFAR100('/path/to/cifar100', download=True)")
OUT_ROOT=$REPO/kbt_runs           # caches take ~1.6 GB (10 tasks) to ~8 GB (50 tasks) per run

cd "$REPO" || exit 1
source "$VENV/bin/activate"

case ${SLURM_ARRAY_TASK_ID:-0} in
  0) ARGS="--config configs/mimg10_inflora.json --data_path $IMAGENET_R"; TAG=ImageNet_R_10 ;;
  1) ARGS="--config configs/mimg20_inflora.json --data_path $IMAGENET_R"; TAG=ImageNet_R_20 ;;
  2) ARGS="--config configs/mimg10_inflora.json --data_path $IMAGENET_R --init_cls 4 --increment 4"; TAG=ImageNet_R_50 ;;
  3) ARGS="--config configs/cifar100_inflora.json --data_path $CIFAR"; TAG=cifar100_10 ;;
  4) ARGS="--config configs/cifar100_inflora.json --data_path $CIFAR --init_cls 2 --increment 2"; TAG=cifar100_50 ;;
  *) echo "unknown array index ${SLURM_ARRAY_TASK_ID}"; exit 1 ;;
esac

python kernel_bt/phase0.py $ARGS --device 0 --num_workers "${SLURM_CPUS_PER_TASK:-4}" \
    --out "$OUT_ROOT/phase0_$TAG"
