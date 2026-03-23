#!/bin/bash
# Precompute backbone features for cached training
# Usage: bash scripts/precompute_cache.sh <dataset> <split> <gpu_id> [n_seeds]

set -e

DATASET=${1:?Usage: bash scripts/precompute_cache.sh <dataset> <split> <gpu_id> [n_seeds]}
SPLIT=${2:?Specify split number}
GPU=${3:-0}
N_SEEDS=${4:-90}

cd "$(dirname "$0")/.."

case $DATASET in
    gtea)      CFG_NAME="GTEA-Trained-S${SPLIT}" ;;
    50salads)  CFG_NAME="50salads-Trained-S${SPLIT}" ;;
    breakfast) CFG_NAME="Breakfast-Trained-S${SPLIT}" ;;
    *)         echo "Unknown dataset: $DATASET"; exit 1 ;;
esac

echo "=========================================="
echo "Precomputing backbone features"
echo "Dataset: $DATASET | Split: $SPLIT | Seeds: $N_SEEDS"
echo "=========================================="

CUDA_VISIBLE_DEVICES=$GPU python -m tst.precompute_backbone \
    --backbone diffact \
    --dataset "$DATASET" \
    --split "$SPLIT" \
    --backbone_config "backbones/DiffAct/configs/${CFG_NAME}.json" \
    --backbone_checkpoint "backbones/DiffAct/trained_models/${CFG_NAME}/release.model" \
    --csv_dir asformer_tst/csv \
    --output_dir "cache/diffact/${DATASET}/split_${SPLIT}" \
    --n_seeds "$N_SEEDS" \
    --train_mode
