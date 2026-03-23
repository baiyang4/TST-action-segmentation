"""Segment extraction utilities for TST.

These functions convert frame-level predictions/features into segment-level
representations, which is the core preprocessing step before TST refinement.
"""

import torch
import torch.nn.functional as F
import numpy as np


def extract_segments(action_idx, frame_features, n_classes):
    """Extract segment-level information from frame-level predictions and features.

    GT matching is handled externally via HungarianMatcher (DETR-style).

    Args:
        action_idx: [T] tensor of per-frame action class indices (argmax of predictions).
        frame_features: [bs, sd_dim, T] frame features (projected to sd_dim).
        n_classes: number of action classes.

    Returns:
        dict with keys:
            segment_features: [bs, num_seg, sd_dim] mean-pooled segment features
            segment_onehot: [bs, num_seg, n_classes] one-hot predicted class per segment
            pred_labels: [bs, num_seg] predicted label per segment
    """
    device = frame_features.device

    boundaries = [0]
    prev = action_idx[0]
    for i, idx in enumerate(action_idx):
        if idx != prev:
            boundaries.append(i)
        prev = idx
    boundaries.append(len(action_idx))

    segment_features = []
    segment_onehot = []
    pred_labels = []

    for s_i in range(len(boundaries) - 1):
        start = boundaries[s_i]
        end = boundaries[s_i + 1]

        seg_feat = frame_features[:, :, start:end].mean(-1, keepdim=True).permute(0, 2, 1)  # [bs, 1, sd_dim]
        segment_features.append(seg_feat)

        pred_label = torch.mean(action_idx[start:end].float()).long()
        pred_labels.append(pred_label)

        onehot = torch.zeros(n_classes, device=device)
        onehot[pred_label] = 1
        segment_onehot.append(onehot)

    segment_features = torch.cat(segment_features, dim=1)  # [bs, num_seg, sd_dim]
    segment_onehot = torch.stack(segment_onehot).unsqueeze(0).to(device)  # [bs, num_seg, n_classes]
    pred_labels = torch.LongTensor([p.item() for p in pred_labels]).view(1, -1).to(device)

    return {
        'segment_features': segment_features,
        'segment_onehot': segment_onehot,
        'pred_labels': pred_labels,
    }


def frame_preds_to_segment_preds(segment_cls, segment_mask, T):
    """Convert TST segment-level outputs back to frame-level predictions.

    Args:
        segment_cls: [bs, num_seg, n_classes] segment class logits
        segment_mask: [bs, num_seg, T] segment mask logits

    Returns:
        frame_preds: [bs, n_classes, T] frame-level predictions
    """
    # Soft mask voting: weight each segment's class prediction by its mask
    mask_weights = segment_mask.sigmoid()  # [bs, num_seg, T]
    cls_probs = F.softmax(segment_cls, dim=-1)  # [bs, num_seg, n_classes]

    # frame_preds[b, c, t] = sum_s mask_weights[b, s, t] * cls_probs[b, s, c]
    frame_preds = torch.bmm(cls_probs.permute(0, 2, 1), mask_weights)  # [bs, n_classes, T]

    return frame_preds


def get_sinusoid_encoding_table(n_position, d_hid):
    """Generate sinusoidal positional encoding table."""
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])

    return torch.FloatTensor(sinusoid_table)
