"""TST loss functions — DETR-style Hungarian-matched set prediction loss.

Ported from hasr/model_maskformer_CASAgCA_pre_newmatching_bk/refiner_train.py.
Loss is computed only on matched (pred_seg, gt_seg) pairs; unmatched predictions
receive background class with downweighted CE loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
# Index helpers
# ─────────────────────────────────────────────

def _src_idx(indices):
    batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
    src_idx   = torch.cat([src for (src, _) in indices])
    return batch_idx, src_idx

def _tgt_idx(indices):
    batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
    tgt_idx   = torch.cat([tgt for (_, tgt) in indices])
    return batch_idx, tgt_idx


# ─────────────────────────────────────────────
# Per-decoder loss components
# ─────────────────────────────────────────────

def loss_labels(segment_cls, targets_cls, indices, n_classes, bg_weight=0.1):
    """CE loss with background class for unmatched predictions.

    Args:
        segment_cls: [bs, num_pred_seg, n_classes+1]
        targets_cls: [num_gt_seg] GT class labels
        indices: list of (pred_idx, gt_idx) matched pairs
        n_classes: number of action classes (background = n_classes)
        bg_weight: weight for background class in CE loss
    """
    src_idx = _src_idx(indices)

    # Gather GT labels for matched predictions
    target_classes_o = torch.cat([targets_cls[J] for (_, J) in indices]).to(segment_cls.device)

    # All unmatched predictions → background (index = n_classes)
    target_classes = torch.full(segment_cls.shape[:2], n_classes,
                                dtype=torch.int64, device=segment_cls.device)
    target_classes[src_idx] = target_classes_o.long()

    # Downweight background class
    empty_weight = torch.ones(n_classes + 1, device=segment_cls.device)
    empty_weight[-1] = bg_weight

    return F.cross_entropy(segment_cls.transpose(1, 2), target_classes, empty_weight)


def loss_mask_dice(segment_mask, seg_gt_target, indices, num_masks):
    """Focal + Dice loss computed only on matched (pred, gt) pairs.

    Args:
        segment_mask: [bs, num_pred_seg, T]
        seg_gt_target: [bs, num_gt_seg, T]
        indices: list of (pred_idx, gt_idx) matched pairs
        num_masks: normalization factor
    """
    src_masks = segment_mask[_src_idx(indices)]       # [matched, T]
    tgt_masks = seg_gt_target[_tgt_idx(indices)]      # [matched, T]

    focal = _sigmoid_focal_loss(src_masks, tgt_masks, num_masks)
    dice  = _dice_loss(src_masks, tgt_masks, num_masks)
    return focal, dice


def _sigmoid_focal_loss(inputs, targets, num_masks, alpha=0.25, gamma=2):
    prob    = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets.float(), reduction="none")
    p_t     = prob * targets + (1 - prob) * (1 - targets)
    loss    = ce_loss * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    return loss.mean(1).sum() / num_masks


def _dice_loss(inputs, targets, num_masks):
    inputs = inputs.sigmoid().flatten(1)
    numerator   = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


# ─────────────────────────────────────────────
# TSTLoss
# ─────────────────────────────────────────────

class TSTLoss(nn.Module):
    """Hungarian-matched set prediction loss for TST.

    Mirrors the loss in hasr/model_maskformer_CASAgCA_pre_newmatching_bk/refiner_train.py.
    Sums CE + focal + dice over all decoder outputs.

    Args:
        n_classes: number of action classes (background = n_classes)
    """

    def __init__(self, n_classes, bg_weight=0.1):
        super().__init__()
        self.n_classes = n_classes
        self.bg_weight = bg_weight

    def forward(self, tst_output, seg_gt_cls, seg_gt_target, indices):
        """
        Args:
            tst_output: dict from TSTRefiner with:
                segment_cls:  [num_decoders, bs, num_pred_seg, n_classes+1]
                segment_mask: [num_decoders, bs, num_pred_seg, T]
            seg_gt_cls:    [num_gt_seg] GT class labels per GT segment
            seg_gt_target: [num_gt_seg, T] GT binary segment masks
            indices:       list of (pred_idx, gt_idx) from HungarianMatcher

        Returns:
            total: scalar loss
            loss_dict: dict with individual components
        """
        segment_cls  = tst_output['segment_cls']   # [D, bs, num_pred, n_classes+1]
        segment_mask = tst_output['segment_mask']  # [D, bs, num_pred, T]

        num_decoders = segment_cls.shape[0]
        num_masks    = seg_gt_target.shape[0]

        total      = 0.0
        sum_ce     = 0.0
        sum_focal  = 0.0
        sum_dice   = 0.0

        seg_gt_target_bs = seg_gt_target.unsqueeze(0)  # [bs=1, num_gt_seg, T]

        for d in range(num_decoders):
            ce_d    = loss_labels(segment_cls[d], seg_gt_cls, indices, self.n_classes, self.bg_weight)
            f_d, dc_d = loss_mask_dice(segment_mask[d], seg_gt_target_bs, indices, num_masks)
            total   = total + ce_d + f_d + dc_d
            sum_ce    += ce_d.item()
            sum_focal += f_d.item()
            sum_dice  += dc_d.item()

        return total, {
            'cls_loss':   sum_ce    / num_decoders,
            'focal_loss': sum_focal / num_decoders,
            'dice_loss':  sum_dice  / num_decoders,
            'total_loss': total.item(),
        }
