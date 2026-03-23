"""Prediction / inference utilities for BackboneWithTST.

Converts segment-level TST outputs back to frame-level predictions,
then saves per-video prediction files for evaluation.
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import median_filter

from .utils import frame_preds_to_segment_preds
from .config import DATASET_CONFIGS

# Dataset-specific post-processing configs (matching DiffAct baseline)
POSTPROCESS_CONFIGS = {
    'gtea': {'type': 'purge', 'value': 3},
    '50salads': {'type': 'median', 'value': 30},
    'breakfast': {'type': 'median', 'value': 15},
}


def _get_segments(frame_labels):
    """Get segment labels, starts, ends from frame-level label sequence."""
    labels, starts, ends = [], [], []
    last = frame_labels[0]
    labels.append(last)
    starts.append(0)
    for i in range(1, len(frame_labels)):
        if frame_labels[i] != last:
            ends.append(i)
            last = frame_labels[i]
            labels.append(last)
            starts.append(i)
    ends.append(len(frame_labels))
    return labels, starts, ends


def purge_short_segments(predicted, value):
    """Remove segments <= value frames, merge into neighbors. Port from DiffAct."""
    labels, starts, ends = _get_segments(predicted)
    for e in range(len(labels)):
        duration = ends[e] - starts[e]
        if duration <= value:
            if e == 0:
                predicted[starts[e]:ends[e]] = labels[e + 1] if len(labels) > 1 else labels[0]
            elif e == len(labels) - 1:
                predicted[starts[e]:ends[e]] = labels[e - 1]
            else:
                mid = starts[e] + duration // 2
                predicted[starts[e]:mid] = labels[e - 1]
                predicted[mid:ends[e]] = labels[e + 1]
    return predicted


def _hardmax_assign(segment_cls, action_idx, device):
    """Hardmax inference: argmax per segment, assign to frames via action_idx boundaries."""
    mask_cls = F.softmax(segment_cls[-1][0][:, :-1], dim=1)  # [num_seg, n_classes] drop bg
    pred_cls = torch.max(mask_cls, 1)[1]  # [num_seg]

    T = action_idx.shape[0]
    predicted = torch.zeros(T, dtype=torch.long, device=device)
    action_list = action_idx.tolist()
    prev_action_class = action_list[0]
    start_idx, cls_idx = 0, 0
    for idx in range(1, T):
        if action_list[idx] != prev_action_class:
            predicted[start_idx:idx] = pred_cls[cls_idx]
            cls_idx += 1
            prev_action_class = action_list[idx]
            start_idx = idx
    predicted[start_idx:T] = pred_cls[min(cls_idx, len(pred_cls) - 1)]
    return predicted


def _soft_mask_voting(segment_cls, segment_mask):
    """Soft mask voting: weighted combination of segment class probs and mask activations."""
    cls_probs = F.softmax(segment_cls[-1][0][:, :-1], dim=1)  # [num_seg, n_classes] drop bg
    mask_weights = segment_mask[-1][0].sigmoid()  # [num_seg, T]
    frame_preds = torch.matmul(cls_probs.permute(1, 0), mask_weights)  # [n_classes, T]
    predicted = torch.argmax(frame_preds, dim=0)  # [T]
    return predicted


def _hybrid_inference(segment_cls, segment_mask, action_idx, backbone_preds, device, margin=5):
    """Hybrid: TST segment class in segment middle, backbone prediction near boundaries."""
    # Get hardmax TST prediction
    tst_pred = _hardmax_assign(segment_cls, action_idx, device)
    # Get backbone per-frame prediction
    bk_pred = torch.max(backbone_preds[0], 0)[1]  # [T]

    T = action_idx.shape[0]
    predicted = tst_pred.clone()

    # Find segment boundaries from action_idx
    action_list = action_idx.tolist()
    boundaries = [0]
    for idx in range(1, T):
        if action_list[idx] != action_list[idx - 1]:
            boundaries.append(idx)
    boundaries.append(T)

    # Near boundaries (within ±margin frames), use backbone prediction
    for b in boundaries[1:-1]:  # skip first and last
        start = max(0, b - margin)
        end = min(T, b + margin)
        predicted[start:end] = bk_pred[start:end]

    return predicted


def predict_with_tst(model, features_path, vid_list_file, epoch,
                     actions_dict, device, sample_rate, result_dir,
                     dataset=None, n_seeds=1, infer_mode='hardmax'):
    """Run inference with BackboneWithTST model and save predictions.

    Args:
        model: BackboneWithTST instance
        features_path: path to pre-extracted video features (.npy)
        vid_list_file: path to text file listing test video names
        epoch: current epoch (for naming output files)
        actions_dict: dict mapping action_name -> action_id
        device: 'cuda' or 'cpu'
        sample_rate: temporal downsampling rate
        result_dir: directory to save prediction files
        dataset: dataset name for post-processing (None = no postprocess)
        n_seeds: number of DDIM seeds for averaging (1 = single run, >1 = multi-seed)
        infer_mode: 'hardmax' | 'soft' | 'hybrid' — inference strategy
    """
    model.eval()
    id2action = {v: k for k, v in actions_dict.items()}

    with torch.no_grad():
        model.to(device)
        with open(vid_list_file, 'r') as f:
            list_of_vids = f.read().split('\n')[:-1]

        for vid in list_of_vids:
            features = np.load(os.path.join(features_path, vid.split('.')[0] + '.npy'))
            features = features[:, ::sample_rate]
            input_x = torch.tensor(features, dtype=torch.float).unsqueeze(0).to(device)

            if n_seeds > 1:
                # Multi-seed: run N times, convert each to frame-level predictions, majority vote.
                # Different DDIM noise → different segments (varying count), so we vote at frame level.
                T_feat = input_x.shape[2]
                vote_counts = torch.zeros(T_feat, dtype=torch.long, device=device)
                # Accumulate per-class votes at frame level
                n_cls = DATASET_CONFIGS[dataset]['n_classes'] if dataset else 11
                frame_votes = torch.zeros(n_cls, T_feat, device=device)
                for s in range(n_seeds):
                    model.adapter.eval_seed = s * 111 + 1
                    output = model(input_x)
                    run_pred = _hardmax_assign(output['segment_cls'], output['action_idx'], device)
                    for c in range(n_cls):
                        frame_votes[c] += (run_pred == c).float()
                model.adapter.eval_seed = 666  # restore
                predicted = torch.argmax(frame_votes, dim=0)
            else:
                # Single run inference
                output = model(input_x)
                if infer_mode == 'hardmax':
                    predicted = _hardmax_assign(output['segment_cls'], output['action_idx'], device)
                elif infer_mode == 'soft':
                    predicted = _soft_mask_voting(output['segment_cls'], output['segment_mask'])
                elif infer_mode == 'hybrid':
                    # Need backbone predictions for boundary fallback
                    frame_preds = output.get('backbone_predictions', None)
                    if frame_preds is None:
                        # Re-extract from backbone output
                        _, frame_preds, _ = model.adapter(input_x)
                    predicted = _hybrid_inference(
                        output['segment_cls'], output['segment_mask'],
                        output['action_idx'], frame_preds, device)
                else:
                    predicted = _hardmax_assign(output['segment_cls'], output['action_idx'], device)

            # Post-processing (optional, dataset-specific)
            predicted = predicted.cpu().numpy()
            if dataset is not None and dataset in POSTPROCESS_CONFIGS:
                pp = POSTPROCESS_CONFIGS[dataset]
                if pp['type'] == 'purge':
                    predicted = purge_short_segments(predicted, pp['value'])
                elif pp['type'] == 'median':
                    predicted = median_filter(predicted, size=pp['value'])

            # Save predictions (upsample by sample_rate to match original frame rate)
            recognition = []
            for pred in predicted:
                recognition.extend([id2action[int(pred)]] * sample_rate)

            os.makedirs(result_dir, exist_ok=True)
            output_file = os.path.join(result_dir, vid.split('.')[0])
            with open(output_file, 'w') as f:
                f.write('### Frame level recognition: ###\n')
                f.write(' '.join(recognition))


def predict_backbone_only(backbone_adapter, features_path, vid_list_file, epoch,
                          actions_dict, device, sample_rate, result_dir):
    """Run inference with backbone only (no TST), for comparison.

    Args:
        backbone_adapter: BackboneAdapter instance (or raw backbone)
        Other args same as predict_with_tst
    """
    backbone_adapter.eval()
    id2action = {v: k for k, v in actions_dict.items()}

    with torch.no_grad():
        backbone_adapter.to(device)
        with open(vid_list_file, 'r') as f:
            list_of_vids = f.read().split('\n')[:-1]

        for vid in list_of_vids:
            features = np.load(os.path.join(features_path, vid.split('.')[0] + '.npy'))
            features = features[:, ::sample_rate]
            input_x = torch.tensor(features, dtype=torch.float).unsqueeze(0).to(device)

            _, frame_predictions, _ = backbone_adapter(input_x)
            predicted = torch.max(frame_predictions, 1)[1].squeeze(0)  # [T]

            predicted = predicted.cpu().numpy()
            recognition = [id2action[pred] for pred in predicted]

            os.makedirs(result_dir, exist_ok=True)
            output_file = os.path.join(result_dir, vid.split('.')[0])
            with open(output_file, 'w') as f:
                f.write('### Frame level recognition: ###\n')
                f.write(' '.join(recognition))
