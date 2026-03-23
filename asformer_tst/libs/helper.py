import os
import time
import pdb
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

from libs.class_id_map import get_id2class_map
from libs.metric import AverageMeter, BoundaryScoreMeter, ScoreMeter
from libs.postprocess import PostProcessor


def dice_loss(inputs, targets, num_masks):
        inputs = inputs.sigmoid()
        inputs = inputs.flatten(1)
        numerator = 2 * (inputs * targets).sum(-1)
        denominator = inputs.sum(-1) + targets.sum(-1)
        loss = 1 - (numerator + 1) / (denominator + 1)
        return loss.sum() / num_masks


def sigmoid_focal_loss(inputs, targets, num_masks, alpha: float = 0.25, gamma: float = 2):
        prob = inputs.sigmoid()
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets.float(), reduction="none")
        p_t = prob * targets + (1 - prob) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** gamma)

        if alpha >= 0:
            alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
            loss = alpha_t * loss

        return loss.mean(1).sum() / num_masks


def train(
    train_loader: DataLoader,
    model: nn.Module,
    criterion_cls: nn.Module, # cls
    criterion_bound: nn.Module, # bound
    lambda_bound_loss: float, # lambda
    optimizer: optim.Optimizer,
    epoch: int,
    device: str,
) -> float:
    losses = AverageMeter("Loss", ":.4e")

    # switch training mode
    model.train()

    loss_all = 0
    bk_ce_loss_all = 0
    bk_boundary_loss = 0
    qclass_loss_all = 0
    qmask_dice_loss_all = 0
    qmask_ce_loss_all = 0
    
    start_time = time.time() 

    for i, sample in enumerate(train_loader):
        x = sample["feature"]
        t = sample["label"]
        b = sample["boundary"]
        mask = sample["mask"]
        seg_target = sample['targets_segment'] # [num_seg, T]
        seg_location = sample['location_segment'] # List[] len:num_seg


        x = x.to(device) # [bs, dim, T]
        t = t.to(device) # [bs, T]
        b = b.to(device) # [bs, 1, T]
        mask = mask.to(device)
        seg_target =seg_target.to(device)

        batch_size = x.shape[0]

        # compute output and loss
        # output_cls: [bs, num_class, T],  output_bound: List-4. [0]-[bs, 1, T]
        # segment_cls: [bs, num_seg, num_class],  segment_mask: [bs, num_seg, T]
        # GTlabel_list: [bs, num_seg].     GTmask_list: [num_seg, T]
        output_cls, output_bound, segment_cls, segment_mask, GTlabel_list, GTmask_list, stage3_flag = model(x, batch_target=t, batch_targets_segment=seg_target, location_segment=seg_location)

        loss = 0.0

        bk_ce_loss = 0
        bk_boundary_loss = 0

        if stage3_flag:

            if isinstance(output_cls, list):
                n = len(output_cls)
                for out in output_cls:
                    bk_ce_loss += criterion_cls(out, t, x) / n
            else:
                bk_ce_loss += criterion_cls(output_cls, t, x)
        
            if isinstance(output_bound, list):
                n = len(output_bound)
                for out in output_bound:
                    bk_boundary_loss += lambda_bound_loss * criterion_bound(out, b, mask) / n
            else:
                bk_boundary_loss += lambda_bound_loss * criterion_bound(output_bound, b, mask)

        # >>>>> segment classloss
        qclass_loss = F.cross_entropy(segment_cls.permute(0, 2, 1), GTlabel_list)

        # >>>>> segment mask loss
        num_mask = GTmask_list.shape[0]
        qmask_dice_loss = dice_loss(segment_mask[0], GTmask_list, num_mask)
        qmask_ce_loss = sigmoid_focal_loss(segment_mask[0], GTmask_list, num_mask)

        # record loss
        loss = bk_ce_loss + bk_boundary_loss + qclass_loss + qmask_dice_loss + qmask_ce_loss

        losses.update(loss.item(), batch_size)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_all += loss.item()/len(train_loader) 
        if stage3_flag:
            bk_ce_loss_all += bk_ce_loss.item()/len(train_loader)
            bk_boundary_loss += bk_boundary_loss.item() /len(train_loader)
        else:
            bk_ce_loss_all = bk_ce_loss/len(train_loader)
            bk_boundary_loss += bk_boundary_loss/len(train_loader)
            
        qclass_loss_all += qclass_loss.item() / len(train_loader)
        qmask_dice_loss_all += qmask_dice_loss.item() / len(train_loader)
        qmask_ce_loss_all += qmask_ce_loss.item() / len(train_loader)

    print('epoch time: {:.2f}s'.format(time.time()-start_time), flush=True)
    print("bk_ce_loss_all:{:.6f},  bk_boundary_loss:{:.6f},  qclass_loss:{:.6f},  qmask_dice_loss:{:.6f},  qmask_ce_loss:{:.6f}".format(bk_ce_loss_all, bk_boundary_loss, qclass_loss_all, qmask_dice_loss_all, qmask_ce_loss_all), flush=True)
    return losses.avg


def validate(
    val_loader: DataLoader,
    model: nn.Module,
    criterion_cls: nn.Module,
    criterion_bound: nn.Module,
    lambda_bound_loss: float,
    device: str,
    dataset: str,
    dataset_dir: str,
    iou_thresholds: Tuple[float],
    boundary_th: float,
    tolerance: int,
) -> Tuple[float, float, float, float, float, float, float, float]:
    losses = AverageMeter("Loss", ":.4e")
    scores_cls = ScoreMeter(
        id2class_map=get_id2class_map(dataset, dataset_dir=dataset_dir),
        iou_thresholds=iou_thresholds,
    )
    scores_bound = BoundaryScoreMeter(
        tolerance=tolerance, boundary_threshold=boundary_th
    )

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for sample in val_loader:
            x = sample["feature"]
            t = sample["label"]
            b = sample["boundary"]
            mask = sample["mask"]

            x = x.to(device)
            t = t.to(device)
            b = b.to(device)
            mask = mask.to(device)

            batch_size = x.shape[0]

            # compute output and loss
            output_cls, output_bound = model(x)

            loss = 0.0
            loss += criterion_cls(output_cls, t, x)
            loss += criterion_bound(output_bound, b, mask)

            # measure accuracy and record loss
            losses.update(loss.item(), batch_size)

            # calcualte accuracy and f1 score
            output_cls = output_cls.to("cpu").data.numpy()
            output_bound = output_bound.to("cpu").data.numpy()

            t = t.to("cpu").data.numpy()
            b = b.to("cpu").data.numpy()
            mask = mask.to("cpu").data.numpy()

            # update score
            scores_cls.update(output_cls, t, output_bound, mask)
            scores_bound.update(output_bound, b, mask)

    cls_acc, edit_score, segment_f1s = scores_cls.get_scores()
    bound_acc, precision, recall, bound_f1s = scores_bound.get_scores()

    return (
        losses.avg,
        cls_acc,
        edit_score,
        segment_f1s,
        bound_acc,
        precision,
        recall,
        bound_f1s,
    )


def evaluate(
    val_loader: DataLoader,
    model: nn.Module,
    device: str,
    boundary_th: float,
    dataset: str,
    dataset_dir: str,
    iou_thresholds: Tuple[float],
    tolerance: float,
    result_path: str,
    refinement_method: Optional[str] = None,
) -> None:
    postprocessor = PostProcessor(refinement_method, boundary_th)

    scores_before_refinement = ScoreMeter(
        id2class_map=get_id2class_map(dataset, dataset_dir=dataset_dir),
        iou_thresholds=iou_thresholds,
    )

    scores_bound = BoundaryScoreMeter(
        tolerance=tolerance, boundary_threshold=boundary_th
    )

    scores_after_refinement = ScoreMeter(
        id2class_map=get_id2class_map(dataset, dataset_dir=dataset_dir),
        iou_thresholds=iou_thresholds,
    )

    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        for sample in val_loader:
            x = sample["feature"]
            t = sample["label"]
            b = sample["boundary"]
            mask = sample["mask"]

            x = x.to(device)
            t = t.to(device)
            b = b.to(device)
            mask = mask.to(device)

            # compute output and loss
            output_cls, output_bound = model(x)

            # calcualte accuracy and f1 score
            output_cls = output_cls.to("cpu").data.numpy()
            output_bound = output_bound.to("cpu").data.numpy()

            x = x.to("cpu").data.numpy()
            t = t.to("cpu").data.numpy()
            b = b.to("cpu").data.numpy()
            mask = mask.to("cpu").data.numpy()

            refined_output_cls = postprocessor(
                output_cls, boundaries=output_bound, masks=mask
            )

            # update score
            scores_before_refinement.update(output_cls, t)
            scores_bound.update(output_bound, b, mask)
            scores_after_refinement.update(refined_output_cls, t)

    print("Before refinement:", scores_before_refinement.get_scores())
    print("Boundary scores:", scores_bound.get_scores())
    print("After refinement:", scores_after_refinement.get_scores())

    # save logs
    scores_before_refinement.save_scores(
        os.path.join(result_path, "test_as_before_refine.csv")
    )
    scores_before_refinement.save_confusion_matrix(
        os.path.join(result_path, "test_c_matrix_before_refinement.csv")
    )

    scores_bound.save_scores(os.path.join(result_path, "test_br.csv"))

    scores_after_refinement.save_scores(
        os.path.join(result_path, "test_as_after_majority_vote.csv")
    )
    scores_after_refinement.save_confusion_matrix(
        os.path.join(result_path, "test_c_matrix_after_majority_vote.csv")
    )
