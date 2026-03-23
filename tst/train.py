"""Unified training script for Backbone + TST.

This script is backbone-agnostic. It uses the BackboneWithTST wrapper
and BackboneAdapter pattern to support any backbone.

Usage:
    # Stage 2: Train TST with frozen ASFormer backbone
    python -m tst.train --backbone asformer --dataset 50salads --split 1 \
        --stage 2 --backbone_checkpoint path/to/asformer.pth --epoch 50

    # Stage 3: Full finetune
    python -m tst.train --backbone asformer --dataset 50salads --split 1 \
        --stage 3 --stage2_checkpoint path/to/stage2.pth --epoch 50
"""

import os
import sys
import time
import datetime
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .tst_refiner import TSTRefiner
from .wrapper import (BackboneWithTST, ASFormerAdapter, DiffActAdapter,
                      LTContextAdapter, BaFormerAdapter, FACTAdapter)
from .losses import TSTLoss
from .matcher import HungarianMatcher
from .config import TSTConfig, TrainConfig, DATASET_CONFIGS, BACKBONE_FEAT_DIMS
from .predict import predict_with_tst


def parse_args():
    parser = argparse.ArgumentParser(description='Train Backbone + TST')
    # Core
    parser.add_argument('--backbone', default='asformer',
                        choices=['asformer', 'diffact', 'ltcontext', 'baformer', 'fact', 'mstcn'])
    parser.add_argument('--dataset', default='50salads', choices=['50salads', 'gtea', 'breakfast', 'assembly101'])
    parser.add_argument('--split', default=1, type=int)
    parser.add_argument('--stage', default=2, type=int, choices=[2, 3])

    # Training
    parser.add_argument('--epoch', default=90, type=int)
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--seed', default=666, type=int)
    parser.add_argument('--weight_decay', default=0.0001, type=float)
    parser.add_argument('--n_seeds', default=1, type=int, help='Multi-seed inference averaging (final eval only)')
    parser.add_argument('--lr_transformer', default=0, type=float,
                        help='LR for CA/SA modules. 0 = use --lr')
    parser.add_argument('--dropout', default=0.1, type=float, help='Dropout rate for TST modules')
    parser.add_argument('--bg_weight', default=0.1, type=float, help='Background class CE weight')
    parser.add_argument('--scheduler', default='cosine', choices=['cosine', 'step', 'none'],
                        help='LR scheduler type')
    parser.add_argument('--eval_freq', default='auto',
                        help='Eval frequency: "auto" (every epoch ≤30, every 5 after), or integer')
    parser.add_argument('--warmup_epochs', default=0, type=float,
                        help='Linear warmup in epochs (0 to disable, supports fractional e.g. 0.2 = 1/5 epoch)')
    parser.add_argument('--grad_accum', default=1, type=int,
                        help='Gradient accumulation steps (effective batch = grad_accum × 1)')

    # Checkpoints
    parser.add_argument('--backbone_checkpoint', default='', help='Pre-trained backbone weights')
    parser.add_argument('--stage2_checkpoint', default='', help='Stage 2 weights (for stage 3)')

    # TST hyperparams
    parser.add_argument('--sd_dim', default=256, type=int, help='Segment decoder dimension')
    parser.add_argument('--inner_dim', default=64, type=int, help='Internal feature dim after down-projection')
    parser.add_argument('--sa_rate', default=4, type=int, help='Self-attention local window')
    parser.add_argument('--n_layers', default=10, type=int, help='Pixel decoder TCN layers')

    # Backbone hyperparams
    parser.add_argument('--n_features', default=0, type=int,
                        help='Backbone feature dim for TSTRefiner. 0 = auto from BACKBONE_FEAT_DIMS.')
    parser.add_argument('--in_channel', default=2048, type=int)
    parser.add_argument('--backbone_config', default='',
                        help='Path to backbone config file (JSON/YAML). Required for diffact/ltcontext/baformer/fact.')
    parser.add_argument('--n_stages', default=4, type=int)
    parser.add_argument('--n_stages_asb', default=4, type=int)
    parser.add_argument('--n_stages_brb', default=4, type=int)
    parser.add_argument('--lambda_b', default=0.1, type=float)

    # Paths
    parser.add_argument('--dataset_root', default='./dataset')
    parser.add_argument('--model_root', default='./model')
    parser.add_argument('--result_root', default='./result')
    parser.add_argument('--csv_dir', default='asformer_tst/csv')

    # Cached backbone features (from precompute_backbone.py)
    parser.add_argument('--cache_dir', default='', help='Dir with pre-computed backbone features. Skips live backbone forward.')

    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_backbone(args, n_classes):
    """Build backbone model and adapter based on backbone type.

    Returns:
        adapter: BackboneAdapter instance (backbone is already wrapped)
    """
    TAS_ROOT = os.path.join(os.path.dirname(__file__), '..')

    if args.backbone == 'asformer':
        sys.path.insert(0, os.path.join(TAS_ROOT, 'hasr', 'model_asformer_asrf_s23'))
        from libs.models.tcn import ActionSegmentRefinementFramework
        n_feat = args.n_features if args.n_features > 0 else 64
        backbone = ActionSegmentRefinementFramework(
            in_channel=args.in_channel,
            n_features=n_feat,
            n_classes=n_classes,
            n_stages=args.n_stages,
            n_layers=args.n_layers,
            n_stages_asb=args.n_stages_asb,
            n_stages_brb=args.n_stages_brb,
        )
        return ASFormerAdapter(backbone)

    elif args.backbone == 'diffact':
        # Requires: DiffAct repo at backbones/DiffAct/
        # backbone_config: path to a DiffAct JSON config (e.g. backbones/DiffAct/default_configs.py)
        # Usage: python -m tst.train --backbone diffact --backbone_config /path/to/config.json \
        #          --backbone_checkpoint /path/to/diffact.pt ...
        sys.path.insert(0, os.path.join(TAS_ROOT, 'backbones', 'DiffAct'))
        from model import ASDiffusionModel
        from utils import load_config_file

        if not args.backbone_config:
            raise ValueError("--backbone_config required for DiffAct (path to JSON config)")
        config = load_config_file(args.backbone_config)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        backbone = ASDiffusionModel(
            config['encoder_params'],
            config['decoder_params'],
            config['diffusion_params'],
            n_classes,
            device,
        )
        return DiffActAdapter(backbone)

    elif args.backbone == 'ltcontext':
        # Requires: LTContext repo at backbones/LTContext/
        # backbone_config: path to LTContext YAML config
        sys.path.insert(0, os.path.join(TAS_ROOT, 'backbones', 'LTContext'))
        from ltc.model.ltcontext import LTC
        from ltc.config.defaults import get_cfg

        if args.backbone_config:
            from ltc.config.defaults import get_cfg
            cfg = get_cfg()
            cfg.merge_from_file(args.backbone_config)
        else:
            # Use default config — works for 50salads/gtea/breakfast with I3D features
            from ltc.config.defaults import get_cfg
            cfg = get_cfg()
            cfg.LTC.NUM_CLASSES = n_classes   # type: ignore
            cfg.INPUT_DIM = args.in_channel

        backbone = LTC(cfg)
        return LTContextAdapter(backbone)

    elif args.backbone == 'baformer':
        # Requires: BaFormer repo at backbones/BaFormer/ and detectron2
        # backbone_config: path to BaFormer YAML config (default: configs/framed_en_de.yaml)
        sys.path.insert(0, os.path.join(TAS_ROOT, 'backbones', 'BaFormer'))
        from action_segmentation.config import get_cfg
        from action_segmentation.models.bk_fde_tde import Network

        config_file = args.backbone_config or os.path.join(
            TAS_ROOT, 'backbones', 'BaFormer', 'configs', 'framed_en_de.yaml')
        cfg = get_cfg()
        cfg.merge_from_file(config_file)
        cfg.dataset.n_classes = n_classes
        backbone = Network(cfg)
        return BaFormerAdapter(backbone)

    elif args.backbone == 'fact':
        # Requires: CVPR2024-FACT repo at backbones/CVPR2024-FACT/
        # backbone_config: path to FACT config file
        # FACT uses relative imports, so we add the parent dir and import as a package
        fact_parent = os.path.join(TAS_ROOT, 'FACT_actseg')
        if fact_parent not in sys.path:
            sys.path.insert(0, fact_parent)
        from src.models.blocks import FACT
        from src.configs.default import get_cfg_defaults

        if not args.backbone_config:
            raise ValueError("--backbone_config required for FACT (path to config file)")
        cfg = get_cfg_defaults()
        cfg.merge_from_file(args.backbone_config)
        backbone = FACT(cfg, in_dim=args.in_channel, n_classes=n_classes)
        return FACTAdapter(backbone)

    elif args.backbone == 'mstcn':
        raise NotImplementedError("MS-TCN adapter not yet implemented.")

    else:
        raise ValueError(f"Unknown backbone: {args.backbone}")


def build_dataset(args, n_classes):
    """Build train dataset and loader."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'hasr', 'model_asformer_asrf_s23'))
    from libs.dataset import ActionSegmentationDataset, collate_fn
    from libs.transformer import TempDownSamp, ToTensor
    from torchvision.transforms import Compose

    sample_rate = DATASET_CONFIGS[args.dataset]['sample_rate']

    train_data = ActionSegmentationDataset(
        args.dataset,
        transform=Compose([ToTensor(), TempDownSamp(sample_rate)]),
        mode="trainval",
        split=args.split,
        dataset_dir=args.dataset_root,
        csv_dir=args.csv_dir,
    )
    train_loader = DataLoader(
        train_data,
        batch_size=1,
        shuffle=True,
        drop_last=False,
        collate_fn=collate_fn,
    )
    return train_loader, sample_rate


def train_one_epoch(model, train_loader, tst_loss_fn, matcher, asformer_loss_fns,
                    optimizer, device, stage, lambda_b=0.1,
                    cache_dir='', epoch=0, scheduler=None, grad_accum=1):
    """Train for one epoch.

    Args:
        model: BackboneWithTST instance
        train_loader: DataLoader
        tst_loss_fn: TSTLoss instance
        asformer_loss_fns: dict with 'cls' and 'bound' (ASFormer-specific, None for other backbones)
        optimizer: optimizer
        device: device
        stage: 2 or 3
        lambda_b: boundary loss weight (ASFormer only)
        cache_dir: if set, load pre-computed backbone features from disk (fast mode)
        epoch: current epoch (selects which cached seed to use)
    """
    model.train()
    total_loss = 0
    loss_components = {'tst': 0, 'bk': 0}

    # Determine cache seed directory for this epoch
    use_cache = bool(cache_dir) and stage == 2
    if use_cache:
        import json
        meta_path = os.path.join(cache_dir, 'meta.json')
        with open(meta_path) as f:
            meta = json.load(f)
        n_seeds = meta['n_seeds']
        seed_idx = epoch % n_seeds
        seed_dir = os.path.join(cache_dir, f'seed_{seed_idx:03d}')

    start_time = time.time()
    optimizer.zero_grad()

    for i, sample in enumerate(train_loader):
        x = sample["feature"].to(device)
        t = sample["label"].to(device)
        b = sample["boundary"].to(device)
        mask = sample["mask"].to(device)
        seg_target = sample['targets_segment'].to(device)
        seg_gt_cls  = sample['targets_segment_cls'].to(device)
        seg_location = sample['location_segment']

        if use_cache:
            # Load cached backbone features (skip live DiffAct forward)
            fp = sample['feature_path'][0] if isinstance(sample['feature_path'], list) else sample['feature_path']
            vid_name = os.path.splitext(os.path.basename(fp))[0]
            cached = torch.load(os.path.join(seed_dir, f'{vid_name}.pt'), map_location=device)
            frame_features = cached['frame_features'].float().unsqueeze(0)  # [1, feat_dim, T]
            frame_predictions = cached['frame_predictions'].float().unsqueeze(0)  # [1, n_classes, T]
            output = model.refiner(frame_features, frame_predictions)
        else:
            # Live backbone forward
            output = model(x)

        # Hungarian matching: pred segments ↔ GT segments (temporal IoU)
        action_idx = output['action_idx']
        indices, _ = matcher(action_idx, t, seg_target, seg_location)

        # TST loss (Hungarian-matched CE + focal + dice)
        tst_total, tst_details = tst_loss_fn(output, seg_gt_cls, seg_target, indices)
        loss = tst_total

        # Backbone loss (only in stage 3 when backbone is unfrozen)
        if stage == 3:
            bk_out = output['backbone_outputs']

            if asformer_loss_fns is not None:
                # ASFormer-specific loss (multi-stage cls + boundary)
                bk_cls_loss = 0
                bk_bound_loss = 0
                if isinstance(bk_out.get('outputs_cls'), list):
                    n = len(bk_out['outputs_cls'])
                    for out in bk_out['outputs_cls']:
                        bk_cls_loss += asformer_loss_fns['cls'](out, t, x) / n
                if isinstance(bk_out.get('outputs_bound'), list):
                    n = len(bk_out['outputs_bound'])
                    for out in bk_out['outputs_bound']:
                        bk_bound_loss += lambda_b * asformer_loss_fns['bound'](out, b, mask) / n
                bk_loss = bk_cls_loss + bk_bound_loss
            else:
                # Generic: use adapter.compute_loss()
                bk_loss = model.adapter.compute_loss(
                    bk_out, gt_labels=t, gt_boundary=b, mask=mask)

            loss = loss + bk_loss
            loss_components['bk'] += (bk_loss.item() if isinstance(bk_loss, torch.Tensor) else 0) / len(train_loader)

        # Gradient accumulation: scale loss, accumulate, step every grad_accum iters
        loss = loss / grad_accum
        loss.backward()
        if (i + 1) % grad_accum == 0 or (i + 1) == len(train_loader):
            optimizer.step()
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()

        total_loss += loss.item() * grad_accum / len(train_loader)
        loss_components['tst'] += tst_details['total_loss'] / len(train_loader)

    elapsed = time.time() - start_time
    print(f'  epoch time: {elapsed:.1f}s | total_loss: {total_loss:.4f} | '
          f'tst: {loss_components["tst"]:.4f} | bk: {loss_components["bk"]:.4f}', flush=True)

    return total_loss


def main():
    args = parse_args()
    set_seed(args.seed)
    print(args, flush=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Dataset info
    dataset_info = DATASET_CONFIGS[args.dataset]
    n_classes = dataset_info['n_classes']

    # Build model
    adapter = build_backbone(args, n_classes)

    # Determine feat_dim for TSTRefiner
    if args.n_features > 0:
        feat_dim = args.n_features
    else:
        feat_dim = BACKBONE_FEAT_DIMS.get(args.backbone, {}).get(args.dataset, 64)
        print(f"Auto feat_dim={feat_dim} for {args.backbone}/{args.dataset}", flush=True)

    refiner = TSTRefiner(
        n_classes=n_classes,
        feat_dim=feat_dim,
        inner_dim=args.inner_dim,
        sd_dim=args.sd_dim,
        n_layers=args.n_layers,
        sa_rate=args.sa_rate,
        dropout=args.dropout,
    )
    freeze_backbone = (args.stage == 2)
    model = BackboneWithTST(adapter, refiner, freeze_backbone=freeze_backbone)

    # Load weights
    if args.stage == 2 and args.backbone_checkpoint:
        # DiffAct/LTContext/FACT state dicts have bare keys (encoder.xxx, not backbone.encoder.xxx)
        # ASFormer checkpoints may vary; prefix='' maps {k} → adapter.backbone.{k} directly
        backbone_prefix = '' if args.backbone != 'asformer' else 'backbone.'
        BackboneWithTST.load_backbone_weights(model, args.backbone_checkpoint, prefix=backbone_prefix)
    elif args.stage == 3 and args.stage2_checkpoint:
        state_dict = torch.load(args.stage2_checkpoint, map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
        model.unfreeze_backbone()
        print(f"Stage 3: Loaded stage2 checkpoint, backbone unfrozen", flush=True)

    model.to(device)
    print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}', flush=True)
    print(f'Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}', flush=True)

    # Dataset
    train_loader, sample_rate = build_dataset(args, n_classes)

    # Optimizer with per-module LR support
    lr_tf = args.lr_transformer if args.lr_transformer > 0 else args.lr
    if freeze_backbone:
        # Stage 2: only refiner params, with optional per-module LR
        r = model.refiner
        param_groups = [
            {"params": list(r.pixel_decoder.parameters()), "lr": args.lr},
            {"params": list(r.cross_attn.parameters()) +
                       list(r.self_attn.parameters()) +
                       list(r.cross_attn2.parameters()), "lr": lr_tf},
            {"params": list(r.class_head.parameters()) +
                       list(r.mask_head.parameters()) +
                       list(r.pd_proj.parameters()) +
                       list(r.pd_proj2.parameters()) +
                       list(r.feat_down.parameters()) +
                       list(r.feat_proj.parameters()) +
                       list(r.onehot_proj.parameters()) +
                       list(r.label_embedding.parameters()), "lr": args.lr},
        ]
        optimizer = torch.optim.Adam(param_groups, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr, weight_decay=args.weight_decay,
        )
    if lr_tf != args.lr:
        print(f'Per-module LR: pixel_decoder/heads={args.lr}, transformer={lr_tf}', flush=True)

    # LR scheduler with optional warmup (supports fractional epochs)
    # With grad_accum, scheduler steps only on optimizer.step() calls
    n_train = len(train_loader)
    steps_per_epoch = (n_train + args.grad_accum - 1) // args.grad_accum  # ceil division
    warmup_steps = int(args.warmup_epochs * steps_per_epoch)
    total_steps = args.epoch * steps_per_epoch

    if args.scheduler == 'cosine':
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps - warmup_steps, eta_min=1e-6
        )
        if warmup_steps > 0:
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_steps]
            )
        else:
            scheduler = cosine_scheduler
    elif args.scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    else:
        scheduler = None

    # Loss + Matcher
    tst_loss_fn = TSTLoss(n_classes=n_classes, bg_weight=args.bg_weight)
    matcher = HungarianMatcher()

    # Backbone loss for stage 3
    # ASFormer uses specialized HASR loss functions; other backbones use adapter.compute_loss()
    asformer_loss_fns = None
    if args.stage == 3 and args.backbone == 'asformer':
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'hasr', 'model_asformer_asrf_s23'))
        from libs.loss_fn import ActionSegmentationLoss, BoundaryRegressionLoss
        from libs.class_weight import get_class_weight, get_pos_weight

        class_weight = get_class_weight(
            dataset=args.dataset, split=args.split,
            dataset_dir=args.dataset_root, csv_dir=args.csv_dir, mode="trainval"
        ).to(device)

        criterion_cls = ActionSegmentationLoss(
            ce=True, focal=False, tmse=False, gstmse=True,
            weight=class_weight, ignore_index=255,
            ce_weight=1.0, focal_weight=1.0, tmse_weight=0.15, gstmse_weight=1.0,
        )

        pos_weight = get_pos_weight(
            dataset=args.dataset, split=args.split,
            csv_dir=args.csv_dir, mode="trainval"
        ).to(device)
        criterion_bound = BoundaryRegressionLoss(pos_weight=pos_weight)

        asformer_loss_fns = {'cls': criterion_cls, 'bound': criterion_bound}

    # Paths
    exp_name = f'{args.backbone}_tst_stage{args.stage}'
    model_dir = os.path.join(args.model_root, exp_name, args.dataset, f'split_{args.split}')
    result_dir = os.path.join(args.result_root, exp_name, args.dataset, f'split_{args.split}')
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    # Evaluation setup
    mapping_file = os.path.join(args.dataset_root, args.dataset, 'mapping.txt')
    with open(mapping_file, 'r') as f:
        actions = f.read().split('\n')[:-1]
    actions_dict = {a.split()[1]: int(a.split()[0]) for a in actions}

    features_path = os.path.join(args.dataset_root, args.dataset, 'features/')
    gt_path = os.path.join(args.dataset_root, args.dataset, 'groundTruth/')
    vid_list_file_tst = os.path.join(args.dataset_root, args.dataset, f'splits/test.split{args.split}.bundle')

    # Load eval function once (avoids DiffAct utils.py collision via importlib)
    import importlib.util
    _eval_spec = importlib.util.spec_from_file_location(
        '_eval_utils',
        os.path.join(os.path.dirname(__file__), '..',
            'hasr', 'model_maskformer_CASAgCA_pre_newmatching_bk', 'utils.py')
    )
    _eval_mod = importlib.util.module_from_spec(_eval_spec)
    _eval_spec.loader.exec_module(_eval_mod)
    eval_txts = _eval_mod.eval_txts

    # Train
    print(f"\nStart training: {exp_name} | {args.dataset} split {args.split}", flush=True)
    start_time = time.time()

    best_f1_50 = 0.0
    best_epoch = -1

    for epoch in range(args.epoch):
        print(f"\nEpoch {epoch}/{args.epoch-1} (lr={optimizer.param_groups[0]['lr']:.6f})", flush=True)

        train_loss = train_one_epoch(
            model, train_loader, tst_loss_fn, matcher, asformer_loss_fns,
            optimizer, device, args.stage, args.lambda_b,
            cache_dir=args.cache_dir, epoch=epoch, scheduler=scheduler,
            grad_accum=args.grad_accum,
        )

        # Evaluate with smart frequency
        if args.eval_freq == 'auto':
            do_eval = (epoch <= 30) or (epoch % 5 == 0) or (epoch == args.epoch - 1)
        else:
            do_eval = (epoch % int(args.eval_freq) == 0) or (epoch == args.epoch - 1)

        f1_50 = 0.0
        if do_eval:
            predict_with_tst(
                model, features_path, vid_list_file_tst, epoch,
                actions_dict, device, sample_rate, result_dir,
                dataset=args.dataset,
            )
            try:
                results = eval_txts(args.dataset_root, result_dir, args.dataset, args.split, 'asrf')
                f1_50 = float(results.get('F1@0.50', 0.0))
                print(f"  Results: {results}", flush=True)
            except Exception as e:
                print(f"  Eval error: {e}", flush=True)

        # Save best checkpoint only
        if f1_50 >= best_f1_50:
            best_f1_50 = f1_50
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(model_dir, 'best.pth'))

    total_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    print(f'\nTraining complete. Total time: {total_time}', flush=True)
    print(f'Best F1@50 during training: {best_f1_50:.3f} at epoch {best_epoch}', flush=True)

    # Final evaluation with best checkpoint (predictions may have been overwritten by last epoch)
    best_ckpt = os.path.join(model_dir, 'best.pth')
    if os.path.exists(best_ckpt):
        model.load_state_dict(torch.load(best_ckpt, map_location=device))
        predict_with_tst(model, features_path, vid_list_file_tst, 'best',
                         actions_dict, device, sample_rate, result_dir,
                         dataset=args.dataset, n_seeds=args.n_seeds)
        try:
            results = eval_txts(args.dataset_root, result_dir, args.dataset, args.split, 'asrf')
            print(f'Final (best ckpt) Results: {results}', flush=True)
        except Exception as e:
            print(f'  Final eval error: {e}', flush=True)


if __name__ == '__main__':
    main()
