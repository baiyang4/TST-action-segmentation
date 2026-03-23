#!/bin/bash
# Train DiffAct+TST (non-cached mode)
# Usage: bash scripts/train_diffact_tst.sh <dataset> <split> <gpu_id>
# Example: bash scripts/train_diffact_tst.sh gtea 1 0

set -e

DATASET=${1:?Usage: bash scripts/train_diffact_tst.sh <dataset> <split> <gpu_id>}
SPLIT=${2:?Specify split number}
GPU=${3:-0}

cd "$(dirname "$0")/.."

# Dataset-specific configs
case $DATASET in
    gtea)
        LR=0.0005; LR_TF=0.00005; INNER=64; EPOCH=60
        ;;
    50salads)
        LR=0.0005; LR_TF=0.00005; INNER=64; EPOCH=60
        ;;
    breakfast)
        LR=0.00005; LR_TF=0.000005; INNER=128; EPOCH=60
        ;;
    *)
        echo "Unknown dataset: $DATASET (choose: gtea, 50salads, breakfast)"
        exit 1
        ;;
esac

# Map dataset name to DiffAct config naming
case $DATASET in
    gtea)      CFG_NAME="GTEA-Trained-S${SPLIT}" ;;
    50salads)  CFG_NAME="50salads-Trained-S${SPLIT}" ;;
    breakfast) CFG_NAME="Breakfast-Trained-S${SPLIT}" ;;
esac

CONFIG="backbones/DiffAct/configs/${CFG_NAME}.json"
CKPT="backbones/DiffAct/trained_models/${CFG_NAME}/release.model"

echo "=========================================="
echo "DiffAct+TST Training (non-cached)"
echo "Dataset: $DATASET | Split: $SPLIT | GPU: $GPU"
echo "LR: $LR | LR_TF: $LR_TF | inner_dim: $INNER | epochs: $EPOCH"
echo "=========================================="

CUDA_VISIBLE_DEVICES=$GPU python -m tst.train \
    --backbone diffact \
    --dataset "$DATASET" \
    --split "$SPLIT" \
    --stage 2 \
    --backbone_config "$CONFIG" \
    --backbone_checkpoint "$CKPT" \
    --csv_dir asformer_tst/csv \
    --lr "$LR" \
    --lr_transformer "$LR_TF" \
    --inner_dim "$INNER" \
    --epoch "$EPOCH" \
    --scheduler cosine \
    --eval_freq auto
