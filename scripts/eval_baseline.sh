#!/bin/bash
# Evaluate DiffAct baseline (without TST) for comparison
# Usage: bash scripts/eval_baseline.sh <dataset> <split> <gpu_id>

set -e

DATASET=${1:?Usage: bash scripts/eval_baseline.sh <dataset> <split> <gpu_id>}
SPLIT=${2:?Specify split number}
GPU=${3:-0}

cd "$(dirname "$0")/.."

echo "Evaluating DiffAct baseline: $DATASET split $SPLIT"
CUDA_VISIBLE_DEVICES=$GPU python eval_diffact_baseline.py \
    --dataset "$DATASET" --split "$SPLIT" --gpu 0
