# ASFormer + TST

3-stage training pipeline for Temporal Segment Transformer with ASFormer+ASRF backbone.

## Pipeline

| Stage | Script | Description |
|-------|--------|-------------|
| 1 | `python train.py --dataset gtea --split 1` | Train ASFormer+ASRF backbone |
| 2 | `python train.py --dataset gtea --split 1 --stage2 --s1_model <path>` | Freeze backbone, train TST head |
| 3 | `python train.py --dataset gtea --split 1 --stage3 --s2_model <path>` | Fine-tune all end-to-end |

## TST Head Training (Alternative)

```bash
python train_tst.py --dataset gtea --split 1 --lr 0.0001 \
    --asformer_model <path_to_stage1_checkpoint>
```

## Structure

```
asformer_tst/
├── train.py              # 3-stage training (ASFormer+ASRF → TST → finetune)
├── train_tst.py          # TST head training with pre-trained ASFormer
├── models/               # ASFormer + TST model architecture
├── refiner/              # Hungarian matcher, predict, loss
├── libs/                 # Dataset, transforms, loss functions
├── configs/              # Training configs
├── csv/                  # Data split files
└── src/                  # Evaluation utilities
```

## Data

Uses the same dataset structure as the main project. See the [main README](../README.md) for data preparation.
