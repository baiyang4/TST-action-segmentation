"""Configuration for TST training.

Separates backbone config from TST config, making it easy to
swap backbones while keeping TST settings consistent.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class TSTConfig:
    """TST Refiner configuration."""
    # Architecture
    sd_dim: int = 256
    n_layers: int = 10          # pixel decoder TCN layers
    sa_rate: int = 4            # self-attention local window
    dropout: float = 0.1

    # Loss weights
    cls_weight: float = 1.0
    dice_weight: float = 1.0
    focal_weight: float = 1.0


@dataclass
class TrainConfig:
    """Training configuration (backbone-agnostic)."""
    # Dataset
    dataset: str = '50salads'
    split: int = 1
    dataset_root: str = './dataset'

    # Training
    batch_size: int = 1
    max_epoch: int = 50
    learning_rate: float = 0.0005
    weight_decay: float = 0.0001
    optimizer: str = 'Adam'
    seed: int = 666

    # Stage
    stage: int = 2              # 2 = frozen backbone, 3 = full finetune

    # Backbone
    backbone_type: str = 'asformer'  # 'asformer', 'diffact', 'ltcontext', 'fact', 'mstcn'
    backbone_checkpoint: str = ''     # path to pre-trained backbone weights
    stage2_checkpoint: str = ''       # path to stage2 weights (for stage3)

    # Features
    in_channel: int = 2048      # input feature dimension (I3D=2048, MAEv2=varies)
    n_features: int = 64        # backbone internal feature dimension

    # Backbone-specific (ASFormer/ASRF)
    n_stages: int = 4
    n_stages_asb: int = 4
    n_stages_brb: int = 4

    # Evaluation
    iou_thresholds: List[float] = field(default_factory=lambda: [0.1, 0.25, 0.5])
    boundary_th: float = 0.5
    tolerance: int = 5

    # Backbone loss weights
    lambda_b: float = 0.1      # boundary loss weight
    ce_weight: float = 1.0
    gstmse_weight: float = 1.0

    # Paths
    model_root: str = './model'
    result_root: str = './result'
    record_root: str = './record'
    csv_dir: str = './csv'


# Preset configs for different datasets
DATASET_CONFIGS = {
    '50salads': {'n_classes': 19, 'sample_rate': 8},
    'gtea': {'n_classes': 11, 'sample_rate': 1},
    'breakfast': {'n_classes': 48, 'sample_rate': 1},
    'assembly101': {'n_classes': 202, 'sample_rate': 1},
}

# Feature dimensions output by each backbone's encoder.
# Used to set feat_dim when constructing TSTRefiner.
# All backbones use I3D features (2048-dim .npy), same dataset as original TST paper.
#
#   ASFormer: 64 (n_features, last decoder layer output)
#   DiffAct:  num_f_maps * len(feature_layer_indices)
#             = 64*3=192 for 50salads/gtea default; 256*3=768 for breakfast default
#   LTContext: model_dim // dim_reduction = 64//2 = 32 for default config
#   BaFormer:  embed_dim = 64 (from configs/framed_en_de.yaml)
#   FACT:      f_dim from UpdateBlock config (varies, typically 256)
BACKBONE_FEAT_DIMS = {
    'asformer': {
        '50salads': 64, 'gtea': 64, 'breakfast': 64, 'assembly101': 64,
    },
    'diffact': {
        '50salads': 192,   # 64 * 3 layers
        'gtea': 192,       # 64 * 3 layers
        'breakfast': 768,  # 256 * 3 layers
        'assembly101': 192,
    },
    'ltcontext': {
        '50salads': 32,    # model_dim=64, dim_reduction=2 → reduced_dim=32
        'gtea': 32,
        'breakfast': 32,
        'assembly101': 32,
    },
    'baformer': {
        '50salads': 64,    # embed_dim from framed_en_de.yaml
        'gtea': 64,
        'breakfast': 64,
        'assembly101': 64,
    },
    'fact': {
        # Actual output dim = hid_dim from Bi block config (includes appended class probs)
        '50salads': 512, 'gtea': 512, 'breakfast': 512, 'assembly101': 512,
    },
}
