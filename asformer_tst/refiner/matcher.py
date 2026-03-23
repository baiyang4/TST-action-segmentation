# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/models/matcher.py
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import pdb
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn
import numpy as np

def batch_dice_loss(inputs, targets):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss


def batch_sigmoid_focal_loss(inputs, targets, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    hw = inputs.shape[1]

    prob = inputs.sigmoid()
    focal_pos = ((1 - prob) ** gamma) * F.binary_cross_entropy_with_logits(
        inputs, torch.ones_like(inputs), reduction="none"
    )
    focal_neg = (prob ** gamma) * F.binary_cross_entropy_with_logits(
        inputs, torch.zeros_like(inputs), reduction="none"
    )
    if alpha >= 0:
        focal_pos = focal_pos * alpha
        focal_neg = focal_neg * (1 - alpha)

    loss = torch.einsum("nc,mc->nm", focal_pos, targets) + torch.einsum(
        "nc,mc->nm", focal_neg, (1 - targets)
    )

    return loss / hw


def batch_sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    hw = inputs.shape[1]

    pos = F.binary_cross_entropy_with_logits(
        inputs, torch.ones_like(inputs), reduction="none"
    )
    neg = F.binary_cross_entropy_with_logits(
        inputs, torch.zeros_like(inputs), reduction="none"
    )

    loss = torch.einsum("nc,mc->nm", pos, targets) + torch.einsum(
        "nc,mc->nm", neg, (1 - targets)
    )

    return loss / hw


batch_sigmoid_ce_loss_jit = torch.jit.script(
    batch_sigmoid_ce_loss
)  # type: torch.jit.ScriptModule

batch_dice_loss_jit = torch.jit.script(
    batch_dice_loss
)  # type: torch.jit.ScriptModule


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_mask: float = 1, cost_dice: float = 1, alpha: float = 0.25, gamma: float = 2):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_mask: This is the relative weight of the focal loss of the binary mask in the matching cost
            cost_dice: This is the relative weight of the dice loss of the binary mask in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice
        self.alpha = alpha
        self.gamma = gamma
        assert cost_class != 0 or cost_mask != 0 or cost_dice != 0, "all costs cant be 0"


    @torch.no_grad()
    def memory_efficient_forward(self, pred_t, seg_gt_t, batch_targets_segment, location_segment): 
        # pred_t: [T]
        # gt_t: [bs, T]
        # seg_gt_target: [num_gt_seg, T]
        # seg_gt_location: List

        # bk_pred: [T] => [num_pred_seg, T] = [num_gt_seg, T]
        T = pred_t.shape[0]
        pred_l = pred_t.tolist()
        prev_action_class = pred_l[0]

        pred_segment = []
        pred_segment_cls = []
        # pred_location_segment = []

        start_id = 0
        for idx, action_class in enumerate(pred_l):
            if action_class != prev_action_class:
                end_id = idx
                tmp = torch.zeros(T)
                tmp[start_id:end_id] = 1
                pred_segment.append(tmp)
                pred_segment_cls.append(prev_action_class)
                # pred_location_segment.append([start_id, end_id])
                prev_action_class = action_class
                start_id = idx
        tmp = torch.zeros(T)
        tmp[start_id:T] = 1
        pred_segment.append(tmp)
        pred_segment_cls.append(action_class) 
        # pred_location_segment.append([start_id, T])

        pred_segment = torch.stack(pred_segment, dim = 0).to(batch_targets_segment.device)   # [num_pred_seg, T]
        pred_segment_cls = torch.tensor(pred_segment_cls).to(batch_targets_segment.device)    # [num_pred_seg] 

        # 计算 pred_segment[num_pred_seg, T]  & batch_targets_segment [num_gt_seg, T]的tiou
        num_pred_seg = pred_segment.shape[0]
        num_gt_seg = batch_targets_segment.shape[0]
        metric_all = []

        for idx in range(num_pred_seg):
            pred_seg_sub = pred_segment[idx] # [T]
            pred_start = torch.where(pred_seg_sub==1)[0][0]
            pred_end = torch.where(pred_seg_sub==1)[0][-1]

            curr_metric = [] # [num_gt_pred]
            num_gt_seg = batch_targets_segment.shape[0]
            for gt_idx in range(num_gt_seg):
                gt_seg_sub = batch_targets_segment[gt_idx] # [T]
                gt_start = torch.where(gt_seg_sub==1)[0][0]
                gt_end = torch.where(gt_seg_sub==1)[0][-1]

                sub_sum = pred_seg_sub + gt_seg_sub
                overlap_num = torch.where(sub_sum==2)[0].shape[0]
                sub_metric = overlap_num
                if overlap_num == 0:
                    sub_metric = 0
                elif pred_start <= gt_start and gt_end <= pred_end:
                    sub_metric = overlap_num / (pred_end - pred_start)
                elif gt_start < pred_start and pred_end < gt_end:
                    sub_metric = overlap_num / (gt_end - gt_start)
                elif gt_start < pred_start:
                    sub_metric = overlap_num / (pred_end - gt_start)

                elif pred_end < gt_end:
                    sub_metric = overlap_num / (gt_end - pred_start)

                curr_metric.append(-sub_metric)
            metric_all.append(curr_metric)

        # if num_pred_seg >= num_gt_seg:
        metric_all = torch.tensor(metric_all).cpu() # [num_pred_seg, num_gt_seg]
        
        indices = [linear_sum_assignment(metric_all)]

        pred_idx = indices[0][0]
        gt_idx = indices[0][1]
        num_gt_idx = gt_idx.shape[0]

        indices_new = []

        new_pred_idx = []
        new_gt_idx = []

        for idx in range(num_gt_idx):
            pred_sub = pred_idx[idx]
            gt_sub = gt_idx[idx]
            if metric_all[pred_sub, gt_sub] !=0 :
                new_pred_idx.append(pred_sub)
                new_gt_idx.append(gt_sub)

        indices_new = [([np.array(new_pred_idx), np.array(new_gt_idx)])]
        return [ (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices_new ], metric_all


    @torch.no_grad()
    def forward(self, action_idx, seg_gt_t, seg_gt_target, seg_gt_location):
        """Performs the matching
        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_masks": Tensor of dim [batch_size, num_queries, H_pred, W_pred] with the predicted masks
            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "masks": Tensor of dim [num_target_boxes, H_gt, W_gt] containing the target masks
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        return self.memory_efficient_forward(action_idx, seg_gt_t, seg_gt_target, seg_gt_location)

    def __repr__(self):
        head = "Matcher " + self.__class__.__name__
        body = [
            "cost_class: {}".format(self.cost_class),
            "cost_mask: {}".format(self.cost_mask),
            "cost_dice: {}".format(self.cost_dice),
        ]
        _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)


