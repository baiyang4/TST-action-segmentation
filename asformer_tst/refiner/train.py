import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import pdb
import sys
sys.path.append('./backbones/asrf')
from libs.postprocess import PostProcessor
from .matcher import HungarianMatcher

def refiner_train(cfg, dataset, train_loader, model, backbones, backbone_names, optimizer, epoch, split_dict, device, num_actions):

    normal_ce = nn.CrossEntropyLoss()

    matcher = HungarianMatcher()

    total_loss = 0.0
    loss_ce_all = 0.0
    loss_mask_all = 0.0
    loss_dice_all = 0.0

    for idx, sample in enumerate(train_loader):
        
        model.train()
        x = sample['feature']
        t = sample['label']
        mask = sample["mask"]
        seg_gt_target = sample['targets_segment'] # [num_seg, T]
        seg_gt_location = sample['location_segment'] # List[] len:num_seg
        seg_gt_cls = sample['targets_cls'] # [num_seg]
        seg_gt_target = seg_gt_target.to(device)
        seg_gt_cls = seg_gt_cls.to(device)

        ##########################################################################
        # >>>>>>>>>> step1: mstcn stack lerning inference for pred<<<<<<<<<<<<<<<<
        ##########################################################################
        split_idx = 0
        for i in range(eval('cfg.num_splits["{}"]'.format(dataset))):
            if sample['feature_path'][0].split('/')[-1].split('.')[0] in split_dict[i+1]:
                split_idx = i+1
                break
      
        bb_key = random.choice(backbone_names)
        curr_backbone = backbones[bb_key][split_idx]
        tmp = np.random.randint(10, 51)
        # print("mstcn_backbone index:{}".format(tmp))
        curr_backbone.load_state_dict(torch.load('{}/{}/{}/split_{}/epoch-{}.model'.format(cfg.model_root, bb_key, dataset, str(i+1), tmp)))
        curr_backbone.to(device)
        curr_backbone.eval()

        x, seg_gt_t = x.to(device), t.to(device)
        B, L, D = x.shape
        mask = torch.ones(x.size(), device=device)
        action_pred = curr_backbone(x, mask)
        action_idx = torch.argmax(action_pred[-1], dim=1).squeeze().detach() # [T]

        #####################################################################
        # >>>>>>>>>> matching: tIOU <<<<<<<<<<<<<<<<
        #####################################################################
        # seg_gt_target: [num_gt_seg, T]
        # seg_gt_cls: [num_gt_seg]
        indices, _ = matcher(action_idx, seg_gt_t, seg_gt_target, seg_gt_location)

        #####################################################################
        # >>>>>>>>>> step3: our model <<<<<<<<<<<<<<<<
        #####################################################################
        # segment_cls: [num_decoder, bs, num_seg, num_class+1]  # segment_mask: [num_decoder, bs, num_seg, T]
        # GTlabel_list: [bs, num_seg].     # GTmask_list: [num_seg, T]
        segment_cls, segment_mask, _, _ = model(action_idx.to(device), x) # [bs, num_seg]

        #####################################################################
        # >>>>>>>>>> step4: loss <<<<<<<<<<<<<<<<
        #####################################################################
        loss = 0 
        for idx, (seg_mask, seg_cls) in enumerate(zip(segment_mask, segment_cls)):
            # indices = matcher(seg_mask, seg_cls, seg_gt_target, seg_gt_cls)

            loss_ce = loss_labels(seg_cls, seg_gt_cls, indices, num_actions)
            
            num_masks = seg_gt_target.shape[0]
            loss_mask, loss_dice = loss_mask_dice(seg_mask, seg_gt_target.unsqueeze(0), indices, num_masks)
    
            loss += loss_ce + loss_mask + loss_dice

            loss_ce_all += loss_ce.item() / len(train_loader)
            loss_mask_all += loss_mask.item() / len(train_loader)
            loss_dice_all += loss_dice.item() / len(train_loader)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss / len(train_loader)
    print("total_loss:{:.3f}, loss_ce:{:.3f}, loss_mask:{:.3f}. loss_dice:{:.3f}".format(total_loss, loss_ce_all, loss_mask_all, loss_dice_all),flush=True)
    return total_loss.item()



#####################################################################
# >>>>>>>>>> step4: loss <<<<<<<<<<<<<<<<
#####################################################################

def get_src_permutation_idx(indices):
    # permute predictions following indices
    batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
    src_idx = torch.cat([src for (src, _) in indices])
    return batch_idx, src_idx

def get_tgt_permutation_idx(indices):
    # permute targets following indices
    batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
    tgt_idx = torch.cat([tgt for (_, tgt) in indices])
    return batch_idx, tgt_idx


def loss_labels(segment_cls, targets_segment_cls, indices, num_actions):
    # segment_cls: [bs, num_pred_seg, num_class+1]
    # targets_segment_cls: [num_seg_class]
    idx = get_src_permutation_idx(indices)
    
    target_classes_o = []
    for (_, J) in indices:
        target_classes_o.append(targets_segment_cls[J])
    target_classes_o = torch.cat(target_classes_o).to(segment_cls.device)

    num_classes = num_actions
    target_classes = torch.full(segment_cls.shape[:2], num_classes, dtype=torch.int64, device=segment_cls.device) # [bs, num_pred_seg]
    target_classes[idx] = target_classes_o.long()

    # background class
    empty_weight = torch.ones(num_classes + 1)
    empty_weight[-1] = 0.1

    loss_ce = F.cross_entropy(segment_cls.transpose(1, 2), target_classes, empty_weight.to(segment_cls.device))
    return loss_ce


def loss_mask_dice(pred_segment_mask, target_segment_mask, indices, num_masks):
        # pred_segment_mask: [bs, num_queries, T]
        # target_segment_mask: [bs, num_video_seg, T]

        src_idx = get_src_permutation_idx(indices)
        tgt_idx = get_tgt_permutation_idx(indices)

        src_masks = pred_segment_mask[src_idx] # [matched_queries, T]
        target_masks = target_segment_mask[tgt_idx] # [matched_queries, T]

        loss_mask = sigmoid_focal_loss(src_masks, target_masks, num_masks)
        loss_dice = dice_loss(src_masks, target_masks, num_masks)
        return loss_mask, loss_dice


def sigmoid_focal_loss(inputs, targets, num_masks, alpha: float = 0.25, gamma: float = 2):
        """
        Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
        Args:
            inputs: A float tensor of arbitrary shape. The predictions for each example.
            targets: A float tensor with the same shape as inputs. Stores the binary classification label for each element in inputs (0 for the negative class and 1 for the positive class).
            alpha: (optional) Weighting factor in range (0,1) to balance positive vs negative examples. Default = -1 (no weighting).
            gamma: Exponent of the modulating factor (1 - p_t) to balance easy vs hard examples.
        Returns:
            Loss tensor
        """
        prob = inputs.sigmoid()
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets.float(), reduction="none")
        p_t = prob * targets + (1 - prob) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** gamma)

        if alpha >= 0:
            alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
            loss = alpha_t * loss

        return loss.mean(1).sum() / num_masks

def dice_loss(inputs, targets, num_masks):
        """
        Compute the DICE loss, similar to generalized IOU for masks
        Args:
            inputs: A float tensor of arbitrary shape. The predictions for each example.
            targets: A float tensor with the same shape as inputs. Stores the binary
                    classification label for each element in inputs (0 for the negative class and 1 for the positive class).
        """
        inputs = inputs.sigmoid()
        inputs = inputs.flatten(1)
        numerator = 2 * (inputs * targets).sum(-1)
        denominator = inputs.sum(-1) + targets.sum(-1)
        loss = 1 - (numerator + 1) / (denominator + 1)
        return loss.sum() / num_masks
    

