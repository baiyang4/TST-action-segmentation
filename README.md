# TST: Temporal Segment Transformer for Action Segmentation

A plug-in segment-level refinement module that can be applied on top of any frame-level action segmentation backbone. TST uses a DETR-style set prediction approach with Hungarian matching to refine frame-level predictions at the segment level.

## Results

### DiffAct + TST

| Dataset | F1@10 | F1@25 | F1@50 | Edit | Acc |
|---------|-------|-------|-------|------|-----|
| GTEA | **94.2** | **93.0** | **87.1** | **90.9** | 81.4 |
| 50Salads | **92.3** | **91.8** | **87.4** | **87.4** | 89.7 |
| Breakfast | **81.2** | **77.1** | 65.9 | **79.0** | 76.9 |

### Comparison with SOTA

| Method | GTEA F1@50 | 50Salads F1@50 | Breakfast F1@50 |
|--------|-----------|----------------|-----------------|
| ASFormer (NeurIPS'21) | 79.2 | 76.0 | 57.4 |
| DiffAct (ICCV'23) | 84.7 | 83.7 | 64.6 |
| BaFormer (NeurIPS'24) | 83.5 | 83.9 | 63.2 |
| **Ours (DiffAct+TST)** | **87.1** | **87.4** | **65.9** |

## Installation

```bash
conda create -n tst python=3.10
conda activate tst
pip install torch torchvision  # PyTorch >= 2.0
pip install scipy numpy
```

## Data Preparation

Download the I3D features and ground truth annotations for each dataset:
- [GTEA, 50Salads, Breakfast](https://zenodo.org/records/3625992#.Xiv9jGhKhPY)

Place them under `dataset/`:
```
dataset/
├── gtea/
│   ├── features/       # .npy files (2048-dim I3D features)
│   ├── groundTruth/    # .txt files (per-frame action labels)
│   ├── mapping.txt     # action name → index mapping
│   └── splits/         # train/test split files
├── 50salads/
│   └── ...
└── breakfast/
    └── ...
```

## Pre-trained DiffAct Backbone

Download the pre-trained DiffAct models and place them under `backbones/DiffAct/trained_models/`:
```
backbones/DiffAct/
├── configs/
│   ├── GTEA-Trained-S{1-4}.json
│   ├── 50salads-Trained-S{1-5}.json
│   └── Breakfast-Trained-S{1-4}.json
└── trained_models/
    ├── GTEA-Trained-S{1-4}/release.model
    ├── 50salads-Trained-S{1-5}/release.model
    └── Breakfast-Trained-S{1-4}/release.model
```

## Pre-trained TST Weights

Download our DiffAct+TST checkpoints from [HuggingFace](https://huggingface.co/yangbai123/TST-action-segmentation):

```bash
# Download all checkpoints
huggingface-cli download yangbai123/TST-action-segmentation --local-dir model/release/diffact_tst
```

## Training

### Quick Start (Non-cached, recommended for GTEA)

```bash
# Train DiffAct+TST on GTEA split 1, GPU 0
bash scripts/train_diffact_tst.sh gtea 1 0

# Train all GTEA splits in parallel (4 GPUs)
for split in 1 2 3 4; do
    bash scripts/train_diffact_tst.sh gtea $split $((split-1)) &
done
```

### Cached Training (recommended for 50Salads / Breakfast)

For larger datasets, precomputing backbone features significantly speeds up training by avoiding repeated DiffAct inference.

```bash
# Step 1: Precompute backbone features (one-time cost)
bash scripts/precompute_cache.sh 50salads 1 0 60    # 60 seeds for diversity

# Step 2: Train with cached features
bash scripts/train_diffact_tst_cached.sh 50salads 1 0
```

The precomputed cache stores `(frame_features, frame_predictions)` for each video under multiple random seeds, providing training diversity through stochastic DDIM sampling.

### Dataset-specific Configurations

| Dataset | lr | lr_transformer | inner_dim | Epochs | Training mode |
|---------|-----|---------------|-----------|--------|---------------|
| GTEA | 5e-4 | 5e-5 | 64 | 60 | Non-cached |
| 50Salads | 5e-4 | 5e-5 | 64 | 60 | Cached |
| Breakfast | 5e-5 | 5e-6 | 128 | 60 | Cached |

### Evaluate DiffAct Baseline (without TST)

```bash
bash scripts/eval_baseline.sh gtea 1 0
```


## Visualization

Generate qualitative comparisons (GT vs DiffAct vs DiffAct+TST):

```bash
python scripts/visualize.py \
    --dataset gtea --split 1 \
    --checkpoint model/release/diffact_tst/gtea/split_1_best.pth \
    --gpu 0
```

This produces colored segment bar plots saved to `docs/qualitative/`.

## ASFormer + TST

TST with ASFormer+ASRF as the backbone is available in the `asformer_tst/` directory. This variant supports a full 3-stage training pipeline including end-to-end fine-tuning. See [asformer_tst/README.md](asformer_tst/README.md) for details:
1. Pre-train ASFormer+ASRF backbone
2. Freeze backbone, train TST head
3. Fine-tune backbone + TST end-to-end



## Acknowledgements

- [DiffAct](https://github.com/Finspire13/DiffAct) for the diffusion-based backbone
- [ASFormer](https://github.com/ChinaYi/ASFormer) for the temporal convolution backbone
