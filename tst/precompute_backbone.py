"""Pre-compute backbone outputs for fast TST training.

For each video × seed, saves (frame_features, frame_predictions) to disk.
During TST training, loads a random or epoch-specific cached sample — instant.

Usage:
    CUDA_VISIBLE_DEVICES=0 python -m tst.precompute_backbone \
        --backbone diffact \
        --dataset gtea --split 1 \
        --backbone_config backbones/DiffAct/configs/GTEA-Trained-S1.json \
        --backbone_checkpoint backbones/DiffAct/trained_models/GTEA-Trained-S1/release.model \
        --n_seeds 90 \
        --output_dir cache/diffact/gtea/split_1
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from tqdm import tqdm

from .config import DATASET_CONFIGS, BACKBONE_FEAT_DIMS


def parse_args():
    parser = argparse.ArgumentParser(description='Pre-compute backbone outputs')
    parser.add_argument('--backbone', required=True, choices=['diffact', 'fact'])
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--split', required=True, type=int)
    parser.add_argument('--backbone_config', required=True)
    parser.add_argument('--backbone_checkpoint', required=True)
    parser.add_argument('--n_seeds', type=int, default=90, help='Number of DDIM seeds to cache')
    parser.add_argument('--output_dir', required=True, help='Directory to save cached features')
    parser.add_argument('--in_channel', type=int, default=2048)
    parser.add_argument('--dataset_root', default='./dataset')
    return parser.parse_args()


def build_adapter(args, n_classes):
    """Build backbone adapter (same as train.py but minimal)."""
    TAS_ROOT = os.path.join(os.path.dirname(__file__), '..')

    if args.backbone == 'diffact':
        sys.path.insert(0, os.path.join(TAS_ROOT, 'backbones', 'DiffAct'))
        from model import ASDiffusionModel
        from utils import load_config_file

        config = load_config_file(args.backbone_config)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        backbone = ASDiffusionModel(
            config['encoder_params'],
            config['decoder_params'],
            config['diffusion_params'],
            n_classes, device,
        )
        from .wrapper import DiffActAdapter
        adapter = DiffActAdapter(backbone)

    elif args.backbone == 'fact':
        fact_parent = os.path.join(TAS_ROOT, 'FACT_actseg')
        if fact_parent not in sys.path:
            sys.path.insert(0, fact_parent)
        from src.models.blocks import FACT
        from src.configs.default import get_cfg_defaults

        cfg = get_cfg_defaults()
        cfg.merge_from_file(args.backbone_config)
        backbone = FACT(cfg, in_dim=args.in_channel, n_classes=n_classes)
        from .wrapper import FACTAdapter
        adapter = FACTAdapter(backbone)

    else:
        raise ValueError(f"Unsupported backbone: {args.backbone}")

    # Load checkpoint
    state_dict = torch.load(args.backbone_checkpoint, map_location='cpu')
    # Handle DiffAct key format
    if args.backbone == 'diffact':
        adapter.backbone.load_state_dict(state_dict)
    elif args.backbone == 'fact':
        if 'frame_pe.pe' in state_dict:
            del state_dict['frame_pe.pe']
        adapter.backbone.load_state_dict(state_dict, strict=False)

    return adapter


def main():
    args = parse_args()
    device = 'cuda'

    n_classes = DATASET_CONFIGS[args.dataset]['n_classes']
    sample_rate = DATASET_CONFIGS[args.dataset]['sample_rate']

    adapter = build_adapter(args, n_classes)
    adapter.to(device)
    adapter.train()  # train mode: encoder dropout active → diverse features per seed

    # Load video list (trainval)
    features_path = os.path.join(args.dataset_root, args.dataset, 'features')
    vid_list_file = os.path.join(
        args.dataset_root, args.dataset, 'splits',
        f'train.split{args.split}.bundle'
    )
    with open(vid_list_file, 'r') as f:
        vid_list = [line.strip() for line in f if line.strip()]

    os.makedirs(args.output_dir, exist_ok=True)

    # Save metadata
    meta = {
        'backbone': args.backbone,
        'dataset': args.dataset,
        'split': args.split,
        'n_seeds': args.n_seeds,
        'n_videos': len(vid_list),
        'sample_rate': sample_rate,
        'feat_dim': None,  # filled after first video
    }

    print(f"Pre-computing {args.n_seeds} seeds × {len(vid_list)} videos = {args.n_seeds * len(vid_list)} samples")
    print(f"Output: {args.output_dir}")

    with torch.no_grad():
        for seed_idx in tqdm(range(args.n_seeds), desc='Seeds'):
            seed_dir = os.path.join(args.output_dir, f'seed_{seed_idx:03d}')
            os.makedirs(seed_dir, exist_ok=True)

            # Set unique random state per seed: controls both dropout masks and DDIM noise
            torch.manual_seed(seed_idx * 111 + 1)
            torch.cuda.manual_seed_all(seed_idx * 111 + 1)

            for vid in vid_list:
                vid_name = vid.split('.')[0]
                out_file = os.path.join(seed_dir, f'{vid_name}.pt')

                if os.path.exists(out_file):
                    continue  # skip if already cached

                # Load features
                features = np.load(os.path.join(features_path, vid_name + '.npy'))
                features = features[:, ::sample_rate]
                input_x = torch.tensor(features, dtype=torch.float).unsqueeze(0).to(device)

                # Forward through backbone
                frame_features, frame_predictions, _ = adapter(input_x)

                # Save as compressed tensors (CPU, float16 to save space)
                torch.save({
                    'frame_features': frame_features[0].cpu().half(),    # [feat_dim, T]
                    'frame_predictions': frame_predictions[0].cpu().half(),  # [n_classes, T]
                }, out_file)

                if meta['feat_dim'] is None:
                    meta['feat_dim'] = frame_features.shape[1]

    # Save metadata
    with open(os.path.join(args.output_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    # Report size
    total_size = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, fns in os.walk(args.output_dir)
        for f in fns
    )
    print(f"\nDone. Total cache size: {total_size / 1e9:.2f} GB")
    print(f"Meta: {meta}")


if __name__ == '__main__':
    main()
