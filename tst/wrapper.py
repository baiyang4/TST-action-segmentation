"""BackboneWithTST - Universal wrapper to attach TST to any backbone.

Usage:
    # Define a backbone adapter (see examples below)
    adapter = ASFormerAdapter(backbone_model)

    # Create TST refiner
    refiner = TSTRefiner(n_classes=19, feat_dim=64)

    # Wrap them together
    model = BackboneWithTST(adapter, refiner)

    # Forward pass
    output = model(x, gt_labels=t, gt_segment_masks=seg_masks, gt_segment_locations=seg_locs)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .tst_refiner import TSTRefiner


class BackboneAdapter(nn.Module):
    """Abstract interface that all backbone adapters must implement.

    Any action segmentation backbone needs to be wrapped in an adapter
    that provides a unified interface for TST. The adapter must extract:
        1. frame_features: [bs, feat_dim, T] - encoder features
        2. frame_predictions: [bs, n_classes, T] - class prediction logits
        3. backbone_outputs: dict - any other backbone outputs (for backbone loss)
    """

    def forward(self, x, gt_labels=None):
        """
        Args:
            x: [bs, in_channel, T] input features (e.g., I3D features)
            gt_labels: [bs, T] ground truth frame labels (optional, used by some adapters)

        Returns:
            frame_features: [bs, feat_dim, T]
            frame_predictions: [bs, n_classes, T]
            backbone_outputs: dict with any backbone-specific outputs needed for loss
        """
        raise NotImplementedError

    def compute_loss(self, backbone_outputs, gt_labels, gt_boundary=None, mask=None, **kwargs):
        """Compute backbone-specific training loss.

        Override in subclasses that need backbone loss during stage 3 training.
        Default: returns 0 (backbone loss not needed or handled externally).

        Args:
            backbone_outputs: dict returned by forward()
            gt_labels: [bs, T] ground truth frame labels
            gt_boundary: [bs, 1, T] boundary labels (optional)
            mask: [bs, 1, T] valid frame mask (optional)

        Returns:
            scalar loss tensor
        """
        return torch.tensor(0.0, device=gt_labels.device)


class ASFormerAdapter(BackboneAdapter):
    """Adapter for ASFormer / ASRF backbone (used in original TST paper).

    The ASRF backbone returns:
        - outputs_cls: list of [bs, n_classes, T] multi-stage predictions
        - outputs_bound: list of [bs, 1, T] boundary predictions
        - as_f: [bs, feat_dim, T] last decoder features
        - br_f: boundary features

    feat_dim: n_features from ASRF config (default 64)
    """

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, x, gt_labels=None):
        outputs_cls, outputs_bound, as_f, br_f = self.backbone(x)
        return (
            as_f,              # frame_features: [bs, feat_dim, T]
            outputs_cls[-1],   # frame_predictions: [bs, n_classes, T] (last stage)
            {
                'outputs_cls': outputs_cls,
                'outputs_bound': outputs_bound,
            }
        )

    # ASFormer loss is handled externally in train.py (uses HASR loss functions)


class DiffActAdapter(BackboneAdapter):
    """Adapter for DiffAct backbone (ICCV 2023).

    DiffAct = ASFormer-style Encoder + Diffusion Decoder (25-step DDIM sampling).
    Full model baseline: ~83.7 F1@50 on 50Salads.

    Forward strategy:
      - Training (Stage 2, frozen backbone):
          frame_predictions = encoder_out  (ddim_sample is too slow per-batch and @no_grad)
          frame_features    = backbone_feats (concatenated intermediate encoder layers)
          TST is supervised by GT segments → learns to refine encoder-level predictions.
          At evaluation time, segment extraction uses ddim_sample output (see eval mode below).

      - Inference (eval mode):
          frame_predictions = ddim_sample(x)  (full 25-step diffusion, final output ~83.7)
          frame_features    = backbone_feats  (from encoder)
          TST refines the already high-quality diffusion predictions at segment level.

      - Stage 3 (unfrozen backbone): NOT supported for DiffAct because ddim_sample is
          @torch.no_grad() and not differentiable. Only Stage 2 is used.

    Architecture:
        encoder.forward(x, get_features=True) →
            encoder_out: [bs, n_classes, T]
            backbone_feats: [bs, feat_dim, T]
                where feat_dim = num_f_maps * len(feature_layer_indices)
                default: 64 * 3 = 192 (50salads/gtea), 256*3=768 (breakfast)

    feat_dim: 192 for 50salads/gtea default config, 768 for breakfast config
    """

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.eval_seed = 666  # settable from outside for multi-seed inference

    def forward(self, x, gt_labels=None):
        # x: [bs, input_dim, T]
        encoder_out, backbone_feats = self.backbone.encoder(x, get_features=True)

        # Training: no seed → varied DDIM noise → diverse segment boundaries (augmentation)
        # Eval: fixed seed → reproducible results (overridable via eval_seed for multi-seed)
        seed = self.eval_seed if not self.training else None
        frame_predictions = self.backbone.ddim_sample(x, seed=seed)

        return backbone_feats, frame_predictions, {'encoder_out': encoder_out}

    def compute_loss(self, backbone_outputs, gt_labels, gt_boundary=None, mask=None, **kwargs):
        # Stage 3 not supported for DiffAct; this is only called if someone forces stage 3
        encoder_out = backbone_outputs['encoder_out']  # [bs, n_classes, T]
        ce_loss = F.cross_entropy(
            encoder_out.transpose(2, 1).contiguous().view(-1, encoder_out.shape[1]),
            gt_labels.view(-1).long(),
            ignore_index=255,
        )
        mse_loss = torch.clamp(
            F.mse_loss(
                F.log_softmax(encoder_out[:, :, 1:], dim=1),
                F.log_softmax(encoder_out.detach()[:, :, :-1], dim=1),
            ), min=0, max=16,
        )
        return ce_loss + 0.15 * mse_loss


class LTContextAdapter(BackboneAdapter):
    """Adapter for LTContext backbone (LTC).

    LTC is a multi-stage model with windowed attention + long-term context attention.
    Requires a small modification to ltcontext.py to return the final stage features
    (already done: LTC.forward(..., return_features=True)).

    Architecture:
        forward(x, masks, return_features=True) →
            logits: [n_stages, bs, n_classes, T]
            feature: [bs, reduced_dim, T]
                where reduced_dim = model_dim // dim_reduction
                default: 64 // 2 = 32

    feat_dim: 32 for default config (model_dim=64, dim_reduction=2)

    Note: masks should be [bs, 1, T] (ones for no masking).
    """

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, x, gt_labels=None):
        # x: [bs, input_dim, T]
        # masks: [bs, 1, T] (broadcast-compatible with [bs, n_classes, T])
        masks = torch.ones(x.size(0), 1, x.size(2), device=x.device)
        logits, feature = self.backbone(x, masks, return_features=True)
        # logits: [n_stages, bs, n_classes, T]
        # feature: [bs, reduced_dim, T]
        return feature, logits[-1], {'all_logits': logits}

    def compute_loss(self, backbone_outputs, gt_labels, gt_boundary=None, mask=None, **kwargs):
        all_logits = backbone_outputs['all_logits']  # [n_stages, bs, n_classes, T]
        n_stages = all_logits.shape[0]
        total_ce = torch.tensor(0.0, device=gt_labels.device)
        total_mse = torch.tensor(0.0, device=gt_labels.device)
        for i in range(n_stages):
            out = all_logits[i]  # [bs, n_classes, T]
            ce = F.cross_entropy(
                out.transpose(2, 1).contiguous().view(-1, out.shape[1]),
                gt_labels.view(-1).long(),
                ignore_index=255,
            )
            mse = torch.clamp(
                F.mse_loss(
                    F.log_softmax(out[:, :, 1:], dim=1),
                    F.log_softmax(out.detach()[:, :, :-1], dim=1),
                ), min=0, max=16,
            )
            total_ce = total_ce + ce / n_stages
            total_mse = total_mse + mse / n_stages
        return total_ce + 0.15 * total_mse


class BaFormerAdapter(BackboneAdapter):
    """Adapter for BaFormer backbone (NeurIPS 2024).

    BaFormer = ASFormer-style frame_decoder (encoder) + Transformer Decoder with N=100 segment queries.
    Full model baseline: ~83.9 F1@50 on 50Salads.

    This adapter uses the FULL BaFormer output:
      - frame_features    = frame_decoder['feature']  [bs, embed_dim, T]  (encoder features)
      - frame_predictions = mask_voting(pred_logits, pred_masks)  [bs, n_classes, T]
          where pred_logits [bs, n_queries, n_classes] and pred_masks [bs, n_queries, T]
          come from the transformer_decoder (predictor).

    The mask voting converts BaFormer's segment-level predictions back to frame-level,
    giving TST the full-model quality predictions (~83.9) as its starting point.

    Architecture:
        Network.forward(x) →
          frame_decoder(x, mask) → frame_out (features + class_logits)
          predictor(multi_features, mask_features) →
              pred_logits: [bs, n_queries, n_classes]
              pred_masks:  [bs, n_queries, T]
          mask_voting → frame_predictions: [bs, n_classes, T]

    feat_dim: embed_dim from config (default 64)

    Note: Requires detectron2 for model construction.
    """

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, x, gt_labels=None):
        # x: [bs, input_dim, T]
        mask = torch.ones(x.size(0), 1, x.size(2), device=x.device)

        # Run frame_decoder (encoder) to get frame-level features
        frame_out = self.backbone.frame_decoder(x, mask)
        # frame_out['feature']:        [bs, embed_dim, T]
        # frame_out['mask_features']:  [bs, embed_dim, T]
        # frame_out['multi_features']: list of [bs, embed_dim, T] per attention layer

        # Run transformer_decoder (segment queries) on top
        seg_outputs = self.backbone.predictor(
            frame_out['multi_features'], frame_out['mask_features'], mask=None
        )
        # seg_outputs['pred_logits']: [bs, n_queries, n_classes]
        # seg_outputs['pred_masks']:  [bs, n_queries, T]

        # Convert segment predictions → frame-level via mask voting (same as BaFormer's eval)
        pred_logits  = seg_outputs['pred_logits']   # [bs, n_queries, n_classes]
        pred_masks   = seg_outputs['pred_masks']    # [bs, n_queries, T]
        mask_weights = pred_masks.sigmoid()          # [bs, n_queries, T]
        cls_probs    = F.softmax(pred_logits, dim=-1)  # [bs, n_queries, n_classes]
        frame_predictions = torch.bmm(
            cls_probs.permute(0, 2, 1), mask_weights
        )  # [bs, n_classes, T]

        return frame_out['feature'], frame_predictions, {**frame_out, **seg_outputs}

    def compute_loss(self, backbone_outputs, gt_labels, gt_boundary=None, mask=None, **kwargs):
        # Stage 3: use encoder-level class_logits for backbone loss (has gradients,
        # avoids the complex SetCriterion bipartite matching from BaFormer's full training)
        class_logits = backbone_outputs.get('class_logits')  # [bs, n_classes, T]
        if class_logits is None:
            return torch.tensor(0.0, device=gt_labels.device)
        ce_loss = F.cross_entropy(
            class_logits.transpose(2, 1).contiguous().view(-1, class_logits.shape[1]),
            gt_labels.view(-1).long(),
            ignore_index=255,
        )
        mse_loss = torch.clamp(
            F.mse_loss(
                F.log_softmax(class_logits[:, :, 1:], dim=1),
                F.log_softmax(class_logits.detach()[:, :, :-1], dim=1),
            ), min=0, max=16,
        )
        return ce_loss + 0.15 * mse_loss


class FACTAdapter(BackboneAdapter):
    """Adapter for FACT backbone (CVPR 2024).

    FACT uses a block-based frame-action cross-attention architecture.
    Each block updates both frame_feature [T, 1, f_dim] and action_feature [M, 1, a_dim],
    storing frame_clogit [T, 1, n_classes] as an attribute.

    Architecture:
        _forward_one_video(seq, trans) → block_output (list of [frame_feat, action_feat])
        block_list[-1].frame_clogit: [T, 1, n_classes]

    Input format: FACT expects per-video [T, input_dim] sequences (bs=1 only).

    feat_dim: f_dim - n_classes (feature dim after stripping appended class probs)
              In practice, use full f_dim as TST will project it internally.

    Note: batch_size must be 1 (FACT processes one video at a time).
    """

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, x, gt_labels=None):
        # x: [1, input_dim, T] — FACT only supports bs=1
        assert x.size(0) == 1, "FACTAdapter requires batch_size=1"
        T = x.size(2)

        # Reshape: [1, input_dim, T] → [T, input_dim] → unsqueeze batch dim → [T, 1, input_dim]
        seq = x[0].T.unsqueeze(1)  # [T, 1, input_dim]

        # Derive transcript from GT labels if available (used for positional encoding)
        trans = None
        if gt_labels is not None and self.backbone.cfg.FACT.trans:
            from .fact_utils import torch_class_label_to_segment_label
            trans = torch_class_label_to_segment_label(gt_labels[0])[0]

        # Forward through all blocks
        block_output = self.backbone._forward_one_video(seq, trans)

        # Extract features and predictions from the last block
        # frame_feature: [T, 1, f_dim]  (f_dim includes appended class probs)
        frame_feature = block_output[-1][0]  # [T, 1, f_dim]
        frame_feature_3d = frame_feature.squeeze(1).T.unsqueeze(0)  # [1, f_dim, T]

        # frame_clogit: [T, 1, n_classes]
        frame_clogit = self.backbone.block_list[-1].frame_clogit
        frame_predictions = frame_clogit.squeeze(1).T.unsqueeze(0)  # [1, n_classes, T]

        return frame_feature_3d, frame_predictions, {'block_output': block_output}

    def compute_loss(self, backbone_outputs, gt_labels, gt_boundary=None, mask=None, **kwargs):
        # Frame-level CE on the last block's frame_clogit
        frame_clogit = self.backbone.block_list[-1].frame_clogit  # [T, 1, n_classes]
        frame_preds = frame_clogit.squeeze(1)  # [T, n_classes]
        ce_loss = F.cross_entropy(
            frame_preds,
            gt_labels[0].long(),
            ignore_index=255,
        )
        return ce_loss


class BackboneWithTST(nn.Module):
    """Universal wrapper that combines any backbone with TST refiner.

    This is the main entry point for training and inference. It:
    1. Runs the backbone to get frame features + initial predictions
    2. Feeds them into TST for segment-level refinement
    3. Returns both backbone outputs (for backbone loss) and TST outputs (for TST loss)

    Args:
        adapter: BackboneAdapter wrapping the backbone
        refiner: TSTRefiner module
        freeze_backbone: if True, freeze backbone parameters (stage 2 training)
    """

    def __init__(self, adapter, refiner, freeze_backbone=False):
        super().__init__()
        self.adapter = adapter
        self.refiner = refiner
        self._backbone_frozen = freeze_backbone

        if freeze_backbone:
            self.freeze_backbone()

    def train(self, mode=True):
        super().train(mode)
        return self

    def freeze_backbone(self):
        for p in self.adapter.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.adapter.parameters():
            p.requires_grad = True
        self._backbone_frozen = False

    def forward(self, x):
        """
        Args:
            x: [bs, in_channel, T] input features

        Returns:
            dict with keys:
                segment_cls:  [2, bs, num_seg, n_classes+1]
                segment_mask: [2, bs, num_seg, T]
                action_idx:   [T] per-frame predicted class (for HungarianMatcher)
                backbone_outputs: dict with backbone-specific outputs
        """
        # Step 1: Run backbone
        frame_features, frame_predictions, backbone_outputs = self.adapter(x)

        # Step 2: Run TST refiner (GT matching is done externally via HungarianMatcher)
        tst_output = self.refiner(frame_features, frame_predictions)

        # Combine outputs
        tst_output['backbone_outputs'] = backbone_outputs
        tst_output['frame_predictions'] = frame_predictions
        return tst_output

    @staticmethod
    def load_backbone_weights(model, checkpoint_path, prefix='backbone.'):
        """Load pre-trained backbone weights into the adapter.

        Args:
            model: BackboneWithTST instance
            checkpoint_path: path to backbone checkpoint
            prefix: prefix to add to state dict keys (default: 'backbone.')
        """
        backbone_dict = torch.load(checkpoint_path, map_location='cpu')
        model_dict = model.state_dict()
        pretrained = {}
        for k, v in backbone_dict.items():
            full_key = f'adapter.{prefix}{k}'
            if full_key in model_dict:
                pretrained[full_key] = v
            elif f'adapter.backbone.{k}' in model_dict:
                pretrained[f'adapter.backbone.{k}'] = v
        model.load_state_dict(pretrained, strict=False)
        print(f"Loaded {len(pretrained)}/{len(backbone_dict)} backbone weights from {checkpoint_path}")
