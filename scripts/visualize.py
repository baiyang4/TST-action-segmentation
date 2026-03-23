"""Qualitative visualization: GT vs Backbone vs TST predictions.

Generates colored segment bar plots comparing ground truth, backbone (DiffAct)
predictions, and TST-refined predictions for test videos.

Usage:
    python scripts/visualize.py --dataset gtea --split 1 \
        --checkpoint model/release/diffact_tst/gtea/split_1_best.pth --gpu 0
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_segment_idx(sequence):
    """Extract segment boundaries and labels from frame-level predictions."""
    seg_time = [0]
    seg_label = [sequence[0]]
    for i, s in enumerate(sequence):
        if s != seg_label[-1]:
            seg_time.append(i)
            seg_label.append(s)
    seg_time.append(len(sequence))
    return seg_time, seg_label


def color_map(N=256, normalized=True):
    """Generate distinct colors for N classes."""
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)
    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7 - j)
            g = g | (bitget(c, 1) << 7 - j)
            b = b | (bitget(c, 2) << 7 - j)
            c = c >> 3
        cmap[i] = np.array([r, g, b])
    if normalized:
        cmap = cmap / 255
    return cmap


def plot_segments(ax, seg_time, seg_label, colors, y=0, linewidth=30):
    """Plot colored segment bars on a matplotlib axis."""
    for i in range(len(seg_time) - 1):
        label = seg_label[i] if isinstance(seg_label[i], int) else int(seg_label[i])
        ax.plot([seg_time[i], seg_time[i + 1]], [y, y],
                linewidth=linewidth, color=colors[label], solid_capstyle='butt')


def main():
    parser = argparse.ArgumentParser(description='Qualitative visualization')
    parser.add_argument('--dataset', required=True, choices=['gtea', '50salads', 'breakfast'])
    parser.add_argument('--split', required=True, type=int)
    parser.add_argument('--checkpoint', required=True, help='TST checkpoint path')
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--output_dir', default='docs/qualitative')
    parser.add_argument('--max_videos', default=5, type=int, help='Max videos to visualize')
    args = parser.parse_args()

    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'

    # Load model
    from tst.train import build_backbone, parse_args as tst_parse_args
    from tst.config import DATASET_CONFIGS, BACKBONE_FEAT_DIMS
    from tst.wrapper import BackboneWithTST
    from tst.tst_refiner import TSTRefiner

    n_classes = DATASET_CONFIGS[args.dataset]['n_classes']
    sample_rate = DATASET_CONFIGS[args.dataset]['sample_rate']
    feat_dim = BACKBONE_FEAT_DIMS['diffact'][args.dataset]

    # Determine inner_dim from checkpoint
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    inner_dim = 64
    if 'refiner.feat_down.weight' in ckpt:
        inner_dim = ckpt['refiner.feat_down.weight'].shape[0]

    # Build backbone
    dataset_to_cfg = {
        'gtea': 'GTEA', '50salads': '50salads', 'breakfast': 'Breakfast'
    }
    cfg_name = f"{dataset_to_cfg[args.dataset]}-Trained-S{args.split}"
    backbone_config = f"backbones/DiffAct/configs/{cfg_name}.json"
    backbone_ckpt = f"backbones/DiffAct/trained_models/{cfg_name}/release.model"

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backbones', 'DiffAct'))
    from model import ASDiffusionModel
    from utils import load_config_file
    config = load_config_file(backbone_config)
    backbone = ASDiffusionModel(
        config['encoder_params'], config['decoder_params'],
        config['diffusion_params'], n_classes, device,
    )
    bk_weights = torch.load(backbone_ckpt, map_location='cpu')
    backbone.load_state_dict(bk_weights)

    from tst.wrapper import DiffActAdapter
    adapter = DiffActAdapter(backbone)

    refiner = TSTRefiner(
        n_classes=n_classes, feat_dim=feat_dim, inner_dim=inner_dim
    )
    model = BackboneWithTST(adapter, refiner, freeze_backbone=True)
    model.load_state_dict(ckpt, strict=False)
    model.to(device)
    model.eval()

    # Load data
    dataset_root = './dataset'
    gt_path = os.path.join(dataset_root, args.dataset, 'groundTruth')
    features_path = os.path.join(dataset_root, args.dataset, 'features')
    mapping_file = os.path.join(dataset_root, args.dataset, 'mapping.txt')
    vid_list_file = os.path.join(dataset_root, args.dataset, 'splits',
                                  f'test.split{args.split}.bundle')

    # Load action mapping
    actions_dict = {}
    with open(mapping_file, 'r') as f:
        for line in f:
            idx, name = line.strip().split()
            actions_dict[name] = int(idx)
    id2action = {v: k for k, v in actions_dict.items()}

    with open(vid_list_file, 'r') as f:
        vids = [l.strip() for l in f if l.strip()]

    colors = color_map(100)
    os.makedirs(args.output_dir, exist_ok=True)

    with torch.no_grad():
        for vid_idx, vid in enumerate(vids[:args.max_videos]):
            vid_name = vid.split('.')[0]

            # Load GT
            with open(os.path.join(gt_path, vid), 'r') as f:
                gt_labels = [actions_dict[l.strip()] for l in f]

            # Load features and run model
            features = np.load(os.path.join(features_path, vid_name + '.npy'))
            features = features[:, ::sample_rate]
            input_x = torch.tensor(features, dtype=torch.float).unsqueeze(0).to(device)

            output = model(input_x)

            # DiffAct backbone prediction
            action_idx = output['action_idx'].cpu().numpy()

            # TST refined prediction (hardmax)
            segment_cls = output['segment_cls']
            mask_cls = F.softmax(segment_cls[-1][0][:, :-1], dim=1)
            pred_cls = torch.max(mask_cls, 1)[1]
            T = len(action_idx)
            tst_pred = np.zeros(T, dtype=int)
            action_list = action_idx.tolist()
            prev = action_list[0]
            start_idx, cls_idx = 0, 0
            for idx in range(1, T):
                if action_list[idx] != prev:
                    tst_pred[start_idx:idx] = pred_cls[cls_idx].item()
                    cls_idx += 1
                    prev = action_list[idx]
                    start_idx = idx
            tst_pred[start_idx:T] = pred_cls[min(cls_idx, len(pred_cls) - 1)].item()

            # Downsample GT to match feature length
            gt_ds = gt_labels[::sample_rate][:T]

            # Get segments
            gt_time, gt_label = get_segment_idx(gt_ds)
            bk_time, bk_label = get_segment_idx(action_idx.tolist())
            tst_time, tst_label = get_segment_idx(tst_pred.tolist())

            # Plot
            fig, axes = plt.subplots(3, 1, figsize=(15, 2.5), sharex=True)
            fig.suptitle(f'{vid_name} ({args.dataset} split {args.split})', fontsize=12)

            for ax, title, times, labels in [
                (axes[0], 'Ground Truth', gt_time, gt_label),
                (axes[1], 'DiffAct', bk_time, bk_label),
                (axes[2], 'DiffAct + TST (Ours)', tst_time, tst_label),
            ]:
                plot_segments(ax, times, labels, colors)
                ax.set_ylabel(title, fontsize=9, rotation=0, labelpad=100, va='center')
                ax.set_yticks([])
                ax.set_xlim(0, T)

            plt.tight_layout()
            out_path = os.path.join(args.output_dir, f'{args.dataset}_s{args.split}_{vid_name}.png')
            plt.savefig(out_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f'Saved: {out_path}')


if __name__ == '__main__':
    main()
