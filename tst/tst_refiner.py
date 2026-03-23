"""Temporal Segment Transformer (TST) - Standalone Plug-in Refiner Module.

TST is a segment-level refinement module that can be plugged into any
action segmentation backbone. It only requires:
    1. Frame features: [bs, feat_dim, T] from the backbone
    2. Initial predictions: [bs, n_classes, T] from the backbone

Architecture:
    - Frame Encoder (Pixel Decoder): TCN that refines backbone features
    - Segment Encoder: Extracts segment representations from frame features
    - Segment Decoder: Cross-attention (segment-frame) + Self-attention (inter-segment)
    - Output Head: Segment classification + Mask voting
"""

import math
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import extract_segments, get_sinusoid_encoding_table


# ============================================================
# Building blocks
# ============================================================

class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channel, out_channels):
        super().__init__()
        self.conv_dilated = nn.Conv1d(in_channel, out_channels, 3, padding=dilation, dilation=dilation)
        self.conv_in = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()

    def forward(self, x):
        out = F.relu(self.conv_dilated(x))
        out = self.conv_in(out)
        out = self.dropout(out)
        return x + out


class SingleStageTCN(nn.Module):
    """Frame-level temporal convolutional network (used as pixel decoder)."""
    def __init__(self, in_channel, n_features, n_classes, n_layers):
        super().__init__()
        self.conv_in = nn.Conv1d(in_channel, n_features, 1)
        self.layers = nn.ModuleList([
            DilatedResidualLayer(2 ** i, n_features, n_features)
            for i in range(n_layers)
        ])
        self.conv_out = nn.Conv1d(n_features, n_classes, 1)

    def forward(self, x):
        out = self.conv_in(x)
        feature_all = []
        for layer in self.layers:
            out = layer(out)
            feature_all.append(out)
        out = self.conv_out(out)
        return out, feature_all


class MLP(nn.Module):
    """Simple multi-layer perceptron."""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


# ============================================================
# Segment Decoder Attention Modules
# ============================================================

class SegmentFrameCrossAttention(nn.Module):
    """Cross-attention: each segment query attends to its local frame features.

    Attention is masked so that each segment only attends to frames in
    [prev_segment, current_segment, next_segment].
    """
    def __init__(self, d_model, nhead=1, dim_feedforward=512, dropout=0.1, activation="relu"):
        super().__init__()
        self.tgt2_linear = nn.Linear(d_model, d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)
        self.softmax = nn.Softmax(dim=-1)

    def _build_attention_mask(self, num_seg, T, action_idx):
        """Build local attention mask from action predictions."""
        action_idx = action_idx.tolist()
        prev_action_class = action_idx[0]
        seg_t = []

        start_id = 0
        for idx, action_class in enumerate(action_idx):
            if action_class != prev_action_class:
                end_id = idx
                tmp = torch.zeros(T)
                tmp[start_id:end_id] = 1
                seg_t.append(tmp)
                prev_action_class = action_class
                start_id = idx
        tmp = torch.zeros(T)
        tmp[start_id:T] = 1
        seg_t.append(tmp)
        seg_t = torch.stack(seg_t, dim=0)  # [num_seg, T]

        # Each segment attends to prev + current + next segment's frames
        attention_mask = torch.zeros((num_seg, T), device=seg_t.device)
        for idx in range(num_seg):
            if idx == 0:
                try:
                    attention_mask[idx] = seg_t[idx] + seg_t[idx + 1]
                except:
                    attention_mask[idx] = seg_t[idx]  # fallback when num_seg=1
            elif idx == num_seg - 1:
                attention_mask[idx] = seg_t[idx - 1] + seg_t[idx]
            else:
                attention_mask[idx] = seg_t[idx - 1] + seg_t[idx] + seg_t[idx + 1]
        return attention_mask.unsqueeze(0)  # [1, num_seg, T]

    def forward(self, tgt, memory, pos=None, query_pos=None, action_idx=None):
        """
        Args:
            tgt: [num_seg, bs, d_model] segment queries (initialized to zeros)
            memory: [T, bs, d_model] frame features from pixel decoder
            pos: [T, 1, d_model] sinusoidal positional encoding for frames
            query_pos: [num_seg, bs, d_model] segment embeddings
            action_idx: [T] per-frame action indices for mask construction
        """
        q = tgt + query_pos  # [num_seg, bs, dim]
        k = memory + pos     # [T, bs, dim]
        v = memory           # [T, bs, dim]
        num_seg, _, dim = q.shape
        T = k.shape[0]

        attention = torch.bmm(q.permute(1, 0, 2), k.permute(1, 2, 0)) / np.sqrt(dim)  # [bs, num_seg, T]
        attention_mask = self._build_attention_mask(num_seg, T, action_idx)  # [1, num_seg, T]
        attention[~attention_mask.bool()] = float('-inf')

        attention = self.softmax(attention)  # [bs, num_seg, T]
        tgt2 = torch.bmm(v.permute(1, 2, 0), attention.permute(0, 2, 1))  # [bs, dim, num_seg]
        tgt2 = self.tgt2_linear(F.relu(tgt2.permute(0, 2, 1))).permute(1, 0, 2)

        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt  # [num_seg, bs, d_model]


class InterSegmentSelfAttention(nn.Module):
    """Self-attention among segments with local windowed masking.

    Each segment attends to its +-rate neighboring segments.
    """
    def __init__(self, d_model, nhead=1, dim_feedforward=512, dropout=0.1, activation="relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)
        self.softmax = nn.Softmax(dim=-1)
        self.conv_out = nn.Conv1d(d_model, d_model, 1)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def _build_local_mask(self, num_seg, rate):
        mask = np.zeros((num_seg, num_seg))
        for idx in range(num_seg):
            start = max(0, idx - rate)
            end = min(num_seg, idx + rate + 1)
            mask[idx][start:end] = 1
        return mask

    def forward(self, src, pos=None, rate=4):
        """
        Args:
            src: [bs, num_seg, d_model]
            pos: [num_seg, d_model] sinusoidal positional encoding
            rate: local attention window size (+-rate neighbors)
        """
        q = k = self.with_pos_embed(src, pos)  # [bs, num_seg, dim]
        q = q.permute(0, 2, 1)  # [bs, dim, num_seg]
        k = k.permute(0, 2, 1)
        v = src.permute(0, 2, 1)

        _, dim, num_seg = q.shape
        attention = torch.bmm(q.permute(0, 2, 1), k) / np.sqrt(dim)  # [bs, num_seg, num_seg]
        # Global SA (local mask commented out, matching hasr implementation)
        # attn_mask = self._build_local_mask(num_seg, rate)
        # attn_mask = torch.tensor(attn_mask).unsqueeze(0).to(src.device)
        # attention[~attn_mask.bool()] = float('-inf')

        attention = self.softmax(attention).permute(0, 2, 1)
        src2 = torch.bmm(v, attention)
        src2 = self.conv_out(F.relu(src2)).permute(0, 2, 1)

        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src  # [bs, num_seg, d_model]


# ============================================================
# Main TST Refiner Module
# ============================================================

class TSTRefiner(nn.Module):
    """Temporal Segment Transformer - Standalone Plug-in Refiner.

    Can be attached to ANY action segmentation backbone. Only requires:
        - frame_features: [bs, feat_dim, T] from backbone encoder
        - frame_predictions: [bs, n_classes, T] from backbone output

    Architecture:
        1. Pixel Decoder (Frame Encoder): TCN refines backbone features
        2. Segment Encoder: Extract segment representations from predictions
        3. Segment Decoder: Cross-attn (segment<->frame) + Self-attn (segment<->segment)
        4. Output Head: Per-segment class prediction + mask prediction

    Args:
        n_classes: number of action classes
        feat_dim: dimension of backbone frame features (e.g., 64)
        sd_dim: segment decoder hidden dimension (default: 256)
        n_layers: number of TCN layers in pixel decoder (default: 10)
        sa_rate: local self-attention window rate (default: 4)
        dropout: dropout rate (default: 0.1)
    """

    def __init__(
        self,
        n_classes,
        feat_dim=64,
        inner_dim=64,
        sd_dim=256,
        n_layers=10,
        sa_rate=4,
        dropout=0.1,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.feat_dim = feat_dim
        self.inner_dim = inner_dim
        self.sd_dim = sd_dim
        self.sa_rate = sa_rate
        self.hidden_dim = sd_dim * 2  # concat of feat + label embedding

        # --- Feature Down-projection (e.g. 192→64 for DiffAct) ---
        self.feat_down = nn.Linear(feat_dim, inner_dim)

        # --- Pixel Decoder (Frame Encoder) ---
        self.pixel_decoder = SingleStageTCN(inner_dim, inner_dim, n_classes, n_layers)
        self.pd_proj = MLP(inner_dim, sd_dim, self.hidden_dim, 3)   # Round 1: pd_f[-3]
        self.pd_proj2 = MLP(inner_dim, sd_dim, self.hidden_dim, 3)  # Round 2: pd_f[-2]

        # --- Segment Encoder ---
        self.feat_proj = nn.Linear(inner_dim, sd_dim)  # project features to sd_dim
        self.onehot_proj = MLP(n_classes, 128, sd_dim, 3)  # project one-hot to sd_dim
        self.label_embedding = nn.Embedding(n_classes, sd_dim)  # learnable label embeddings

        # --- Segment Decoder (two rounds: CA→SA→CA) ---
        self.cross_attn = SegmentFrameCrossAttention(
            self.hidden_dim, nhead=1, dim_feedforward=self.hidden_dim, dropout=dropout
        )
        self.self_attn = InterSegmentSelfAttention(
            self.hidden_dim, nhead=1, dim_feedforward=self.hidden_dim, dropout=dropout
        )
        self.cross_attn2 = SegmentFrameCrossAttention(
            self.hidden_dim, nhead=1, dim_feedforward=self.hidden_dim, dropout=dropout
        )

        # --- Output Heads ---
        self.class_head = nn.Linear(self.hidden_dim, n_classes + 1)  # +1 for background
        self.mask_head = nn.Linear(self.hidden_dim, inner_dim)

    def forward(self, frame_features, frame_predictions):
        """
        Args:
            frame_features: [bs, feat_dim, T] backbone frame features
            frame_predictions: [bs, n_classes, T] backbone class predictions (logits)

        Returns:
            dict with keys:
                segment_cls: [2, bs, num_seg, n_classes+1] (2 decoders, +1 for background)
                segment_mask: [2, bs, num_seg, T] segment mask logits (2 decoders)
                action_idx: [T] per-frame action indices (used by matcher in training loop)
        """
        device = frame_features.device

        # --- Step 0: Down-project backbone features to inner_dim ---
        frame_features = self.feat_down(
            frame_features.permute(0, 2, 1)  # [bs, T, feat_dim]
        ).permute(0, 2, 1)  # [bs, inner_dim, T]

        # --- Step 1: Pixel Decoder ---
        _, pd_features = self.pixel_decoder(frame_features)  # list of [bs, inner_dim, T]
        pd_f_r1 = pd_features[-3]   # Round 1 cross-attention memory
        pd_f_r2 = pd_features[-2]   # Round 2 cross-attention memory
        pd_f_last = pd_features[-1]  # for mask prediction

        # Project to hidden dim for cross-attention
        pd_memory = self.pd_proj(pd_f_r1.permute(0, 2, 1)).permute(0, 2, 1)    # [bs, hidden_dim, T]
        pd_memory2 = self.pd_proj2(pd_f_r2.permute(0, 2, 1)).permute(0, 2, 1)  # [bs, hidden_dim, T]

        # --- Step 2: Determine action_idx (always from backbone predictions) ---
        action_idx = torch.max(frame_predictions[0], 0)[1]  # [T] argmax of predictions

        # --- Step 3: Segment Encoder ---
        bk_proj = self.feat_proj(frame_features.permute(0, 2, 1)).permute(0, 2, 1)  # [bs, sd_dim, T]

        seg_info = extract_segments(action_idx, bk_proj, self.n_classes)

        seg_feat = seg_info['segment_features']   # [bs, num_seg, sd_dim]
        seg_onehot = seg_info['segment_onehot']   # [bs, num_seg, n_classes]
        pred_labels = seg_info['pred_labels']      # [bs, num_seg]

        seg_onehot_proj = self.onehot_proj(seg_onehot)  # [bs, num_seg, sd_dim]
        seg_feat = seg_feat + seg_onehot_proj
        label_embed = self.label_embedding(pred_labels)  # [bs, num_seg, sd_dim]
        segment_queries = torch.cat([seg_feat, label_embed], dim=2)  # [bs, num_seg, hidden_dim]
        segment_queries = segment_queries.permute(1, 0, 2)  # [num_seg, bs, hidden_dim]

        T = pd_f_last.shape[2]
        frame_pos = get_sinusoid_encoding_table(T, self.hidden_dim).to(device).unsqueeze(1)  # [T, 1, hidden_dim]

        # --- Step 4: Round 1 — Cross-Attention (pd_f[-3]) → Self-Attention ---
        pd_memory_seq = pd_memory.permute(2, 0, 1)  # [T, bs, hidden_dim]
        tgt = torch.zeros_like(segment_queries)      # [num_seg, bs, hidden_dim]
        tgt = self.cross_attn(
            tgt=tgt, query_pos=segment_queries, memory=pd_memory_seq,
            pos=frame_pos, action_idx=action_idx
        ).permute(1, 0, 2)  # [bs, num_seg, hidden_dim]

        num_seg = tgt.shape[1]
        seg_pos = get_sinusoid_encoding_table(num_seg, self.hidden_dim).to(device)
        refined1 = self.self_attn(src=tgt, pos=seg_pos, rate=self.sa_rate)  # [bs, num_seg, hidden_dim]

        # --- Step 5: Round 2 — Cross-Attention (pd_f[-2]) ---
        pd_memory2_seq = pd_memory2.permute(2, 0, 1)  # [T, bs, hidden_dim]
        tgt2 = self.cross_attn2(
            tgt=refined1.permute(1, 0, 2), query_pos=segment_queries,
            memory=pd_memory2_seq, pos=frame_pos, action_idx=action_idx
        ).permute(1, 0, 2)  # [bs, num_seg, hidden_dim]

        # Stack both decoder outputs: [2, bs, num_seg, hidden_dim]
        refined = torch.stack([refined1, tgt2], dim=0)

        # --- Step 6: Output Heads ---
        segment_cls = self.class_head(refined)   # [2, bs, num_seg, n_classes]
        mask_embed = self.mask_head(refined)      # [2, bs, num_seg, feat_dim]
        segment_mask = torch.einsum('nbsc,bct->nbst', mask_embed, pd_f_last)  # [2, bs, num_seg, T]

        return {
            'segment_cls': segment_cls,    # [2, bs, num_seg, n_classes+1]
            'segment_mask': segment_mask,  # [2, bs, num_seg, T]
            'action_idx': action_idx,      # [T] for matcher in training loop
        }
