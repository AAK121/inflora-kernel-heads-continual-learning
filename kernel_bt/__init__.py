"""Kernelised InfLoRA B_t (report/kernel_inflora.tex).

lifts.py       frozen feature map psi(x) = [x; alpha * phi(z(x))], calibrated on task 1
dualgpm.py     InfLoRA's DualGPM in torch, for any dimension, with the task-1 fix and a cap
diagnostics.py Phase 0 metrics (novelty, usable signal, leakage, protection, angles, linearization)
phase0.py      Phase 0 driver: cache block inputs of the frozen ViT, run the diagnostics, go/no-go
"""
