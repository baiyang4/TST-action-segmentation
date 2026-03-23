#!/bin/bash
# Create conda environment for TST experiments
# Run this AFTER: module load miniforge3
#
# Usage: bash scripts/setup_env.sh

set -e

ENV_NAME="tst_env"

echo "=== Creating conda env: $ENV_NAME ==="
conda create -n $ENV_NAME python=3.9 -y
conda activate $ENV_NAME

echo "=== Installing PyTorch 2.0 (CUDA 11.8) ==="
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

echo "=== Installing common dependencies ==="
pip install numpy pandas scipy tqdm tensorboard wandb
pip install einops opencv-python lmdb distinctipy yacs

# For ASFormer/TST (numba for sinusoidal encoding)
pip install numba

echo "=== Environment ready ==="
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
conda list | head -20
