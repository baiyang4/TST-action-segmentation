"""Hungarian Matcher for TST segment-GT segment matching.

Ported from hasr/model_maskformer_CASAgCA_pre_newmatching_bk/matcher.py.
Matches predicted segments to GT segments via temporal IoU + linear_sum_assignment.
"""
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn


class HungarianMatcher(nn.Module):
    """Matches predicted segments (from backbone action_idx) to GT segments via temporal IoU.

    Returns indices: list of (pred_idx array, gt_idx array) for matched pairs.
    Unmatched predicted segments receive background class in the loss.
    """

    def __init__(self, cost_class=1, cost_mask=1, cost_dice=1, alpha=0.25, gamma=2):
        super().__init__()
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice
        self.alpha = alpha
        self.gamma = gamma

    @torch.no_grad()
    def forward(self, action_idx, seg_gt_t, batch_targets_segment, location_segment):
        """
        Args:
            action_idx: [T] per-frame predicted class indices
            seg_gt_t: [bs, T] GT frame labels (unused, kept for API compat)
            batch_targets_segment: [num_gt_seg, T] GT segment binary masks
            location_segment: list of [start, end] pairs for GT segments

        Returns:
            indices: list of one tuple (pred_idx_array, gt_idx_array)
            metric_all: [num_pred_seg, num_gt_seg] cost matrix
        """
        T = action_idx.shape[0]
        pred_l = action_idx.tolist()
        prev_action_class = pred_l[0]

        pred_segment = []
        start_id = 0
        for idx, action_class in enumerate(pred_l):
            if action_class != prev_action_class:
                tmp = torch.zeros(T)
                tmp[start_id:idx] = 1
                pred_segment.append(tmp)
                prev_action_class = action_class
                start_id = idx
        tmp = torch.zeros(T)
        tmp[start_id:T] = 1
        pred_segment.append(tmp)

        pred_segment = torch.stack(pred_segment, dim=0).to(batch_targets_segment.device)  # [num_pred_seg, T]

        num_pred_seg = pred_segment.shape[0]
        num_gt_seg = batch_targets_segment.shape[0]
        metric_all = []

        for i in range(num_pred_seg):
            ps = pred_segment[i]
            pred_start = torch.where(ps == 1)[0][0]
            pred_end = torch.where(ps == 1)[0][-1]

            row = []
            for j in range(num_gt_seg):
                gs = batch_targets_segment[j]
                gt_start = torch.where(gs == 1)[0][0]
                gt_end = torch.where(gs == 1)[0][-1]

                overlap = torch.where(ps + gs == 2)[0].shape[0]
                if overlap == 0:
                    sub_metric = 0
                elif pred_start <= gt_start and gt_end <= pred_end:
                    denom = max(int(pred_end - pred_start), 1)
                    sub_metric = overlap / denom
                elif gt_start < pred_start and pred_end < gt_end:
                    denom = max(int(gt_end - gt_start), 1)
                    sub_metric = overlap / denom
                elif gt_start < pred_start:
                    denom = max(int(pred_end - gt_start), 1)
                    sub_metric = overlap / denom
                else:
                    denom = max(int(gt_end - pred_start), 1)
                    sub_metric = overlap / denom

                row.append(-sub_metric)
            metric_all.append(row)

        metric_all = torch.tensor(metric_all).cpu()  # [num_pred_seg, num_gt_seg]
        raw_indices = linear_sum_assignment(metric_all)

        new_pred_idx, new_gt_idx = [], []
        for pred_i, gt_i in zip(raw_indices[0], raw_indices[1]):
            if metric_all[pred_i, gt_i] != 0:
                new_pred_idx.append(pred_i)
                new_gt_idx.append(gt_i)

        indices = [(
            torch.as_tensor(new_pred_idx, dtype=torch.int64),
            torch.as_tensor(new_gt_idx, dtype=torch.int64),
        )]
        return indices, metric_all
