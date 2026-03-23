#!/bin/bash
# Train DiffAct+TST (cached mode — fast, recommended for 50Salads/Breakfast)
# Requires pre-computed backbone features. Run scripts/precompute_cache.sh first.
# Usage: bash scripts/train_diffact_tst_cached.sh <dataset> <split> <gpu_id>

set -e

DATASET=${1:?Usage: bash scripts/train_diffact_tst_cached.sh <dataset> <split> <gpu_id>}
SPLIT=${2:?Specify split number}
GPU=${3:-0}

cd "$(dirname "$0")/.."

case $DATASET in
    gtea)      LR=0.0005; LR_TF=0.00005; INNER=64; EPOCH=60 ;;
    50salads)  LR=0.0005; LR_TF=0.00005; INNER=64; EPOCH=60 ;;
    breakfast) LR=0.00005; LR_TF=0.000005; INNER=128; EPOCH=60 ;;
    *)         echo "Unknown dataset: $DATASET"; exit 1 ;;
esac

case $DATASET in
    gtea)      CFG_NAME="GTEA-Trained-S${SPLIT}" ;;
    50salads)  CFG_NAME="50salads-Trained-S${SPLIT}" ;;
    breakfast) CFG_NAME="Breakfast-Trained-S${SPLIT}" ;;
esac

CACHE_DIR="cache/diffact/${DATASET}/split_${SPLIT}"
if [ ! -d "$CACHE_DIR" ]; then
    echo "Error: Cache not found at $CACHE_DIR"
    echo "Run: bash scripts/precompute_cache.sh $DATASET $SPLIT $GPU"
    exit 1
fi

echo "=========================================="
echo "DiffAct+TST Training (cached)"
echo "Dataset: $DATASET | Split: $SPLIT | GPU: $GPU"
echo "Cache: $CACHE_DIR"
echo "=========================================="

CUDA_VISIBLE_DEVICES=$GPU python -m tst.train \
    --backbone diffact \
    --dataset "$DATASET" \
    --split "$SPLIT" \
    --stage 2 \
    --backbone_config "backbones/DiffAct/configs/${CFG_NAME}.json" \
    --backbone_checkpoint "backbones/DiffAct/trained_models/${CFG_NAME}/release.model" \
    --csv_dir asformer_tst/csv \
    --lr "$LR" \
    --lr_transformer "$LR_TF" \
    --inner_dim "$INNER" \
    --epoch "$EPOCH" \
    --scheduler cosine \
    --eval_freq auto \
    --cache_dir "$CACHE_DIR"
