import os
import sys
import csv
import time
import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
import pdb
import math
import copy
import random
import argparse

sys.path.append('./model_asformer_asrf_s23')
from libs import models
from libs.models.tcn import ourmodel
from libs.optimizer import get_optimizer
from libs.loss_fn import ActionSegmentationLoss, BoundaryRegressionLoss
from libs.class_weight import get_class_weight, get_pos_weight
from libs.dataset import ActionSegmentationDataset, collate_fn
from libs.transformer import TempDownSamp, ToTensor
from libs.helper import train, validate, evaluate
from libs.checkpoint import resume, save_checkpoint

from src.utils import eval_txts, load_meta
from src.predict import predict_backbone, predict_backbone_new
import configs.asrf_config as cfg

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> hy parameter
parser = argparse.ArgumentParser()
parser.add_argument('--split', default=5, type=int) 
parser.add_argument('--epoch', default=2, type=int) 
parser.add_argument('--seed', default=666, type=int)
parser.add_argument('--lr', default=0.000005, type=float) 
parser.add_argument('--model_name', default='tmp')
parser.add_argument('--dataset', default='50salads')
parser.add_argument('--stage2', action="store_true")
parser.add_argument('--stage3', action="store_true")
parser.add_argument('--s1_model', default='model/model_asformer_asrf/50salads/dataset-50salads_split-5/model_80.prm', help='backbone pretrained model')
parser.add_argument('--s2_model', default='model/model_asrf_s23_asformer/50salads/split_5/epoch-5.pth', help='refine model trained from stage2')

args = parser.parse_args()
print(args, flush=True)

random.seed(args.seed) # python
np.random.seed(args.seed) # numpy
torch.manual_seed(args.seed) # torch
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark = False

device = 'cuda'
dataset = args.dataset         # choose from gtea, 50salads, breakfast
split = args.split             # gtea : 1~4, 50salads : 1~5, breakfast : 1~4
model_name = args.model_name
cfg.max_epoch = args.epoch
cfg.learning_rate = args.lr

actions_dict, num_actions, gt_path, features_path, vid_list_file, vid_list_file_tst, sample_rate, model_dir, result_dir,  record_dir = load_meta(cfg.dataset_root, cfg.model_root, cfg.result_root, cfg.record_root, dataset, split, model_name)

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  dataset
train_data = ActionSegmentationDataset(
        dataset,
        transform=Compose([ToTensor(), TempDownSamp(sample_rate)]),
        mode="trainval" if not cfg.param_search else "training",
        split=split,
        dataset_dir=cfg.dataset_root,
        csv_dir=cfg.csv_dir,
    )
train_loader = DataLoader(
        train_data,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True if cfg.batch_size > 1 else False,
        collate_fn=collate_fn,
    )

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  model
backbone = models.ActionSegmentRefinementFramework(
    in_channel = cfg.in_channel,
    n_features = cfg.n_features,
    n_classes = num_actions,
    n_stages = 4,
    n_layers = cfg.n_layers,
    n_stages_asb = cfg.n_stages_asb,
    n_stages_brb = cfg.n_stages_brb
)
model = ourmodel(
    args=args,
    backbone = backbone,
    n_features = cfg.n_features,
    n_classes = num_actions,
    n_stages = cfg.n_stages,
    n_layers = cfg.n_layers,
    n_stages_asb = cfg.n_stages_asb,
    n_stages_brb = cfg.n_stages_brb
)
model.to(device)
print('Model Size: ', sum(p.numel() for p in model.parameters()), flush=True)

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  optimizer
optimizer = get_optimizer('Adam', model, cfg.learning_rate, momentum=cfg.momentum, dampening=cfg.dampening, weight_decay=cfg.weight_decay, nesterov=cfg.nesterov)

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  load model
if args.stage2:
    backbone_dict = torch.load(args.s1_model, map_location='cpu')
    model_dict = model.state_dict()
    pretrained_dict = {}
    for k ,v in backbone_dict.items():
        if 'backbone.'+k in model_dict:
            pretrained_dict['backbone.'+k] = v
    model.load_state_dict(pretrained_dict, strict=False)

    for p in model.backbone.parameters():
        p.requires_grad = False

    predict_backbone('asrf', model.backbone, model_dir, result_dir, features_path, vid_list_file_tst, 200, actions_dict, device, sample_rate) 
    results = eval_txts(cfg.dataset_root, result_dir, dataset, split, 'asrf')
    print(results, flush=True)
    print("stage2: Load backbone backbone pretrained model success and freeze!\n", flush=True)

elif args.stage3:
    model_dict = model.state_dict()
    refine_dict = torch.load(args.s2_model, map_location='cpu')
    pretrained_dict = {}
    for k ,v in refine_dict.items():
        if k in model_dict:
            pretrained_dict[k] = v
    model.load_state_dict(pretrained_dict, strict=False)

    predict_backbone('asrf', model.backbone, model_dir, result_dir, features_path, vid_list_file_tst, 200, actions_dict, device, sample_rate) 
    results = eval_txts(cfg.dataset_root, result_dir, dataset, split, 'asrf')
    print(results, flush=True)
    predict_backbone_new('asrf', model, model_dir, result_dir, features_path, vid_list_file_tst, 300, actions_dict, device, sample_rate) 
    results = eval_txts(cfg.dataset_root, result_dir, dataset, split, 'asrf')
    print(results, flush=True)
    print("stage3: Load backbone+refine model success and full prameter will be finetune!\n")

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  loss
if cfg.class_weight:
    class_weight = get_class_weight(dataset=dataset, split=split, dataset_dir=cfg.dataset_root, csv_dir=cfg.csv_dir, mode="training" if cfg.param_search else "trainval")
    class_weight = class_weight.to(device)
else:
    class_weight = None
# print(class_weight)

criterion_cls = ActionSegmentationLoss(
        ce=cfg.ce,
        focal=cfg.focal,
        tmse=cfg.tmse,
        gstmse=cfg.gstmse,
        weight=class_weight,
        ignore_index=255,
        ce_weight=cfg.ce_weight,
        focal_weight=cfg.focal_weight,
        tmse_weight=cfg.tmse_weight,
        gstmse_weight=cfg.gstmse,
    ) # cls

pos_weight = get_pos_weight(
        dataset=dataset,
        split=split,
        csv_dir=cfg.csv_dir,
        mode="training" if cfg.param_search else "trainval",
    ).to(device)

criterion_bound = BoundaryRegressionLoss(pos_weight=pos_weight) # bound

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  train ASRF
print("Start training", flush=True)
start_time = time.time()

for epoch in range(0, cfg.max_epoch):
    # training
    train_loss = train(train_loader, model, criterion_cls, criterion_bound, cfg.lambda_b, optimizer, epoch, device)
    torch.save(model.state_dict(), os.path.join(model_dir, "epoch-"+str(epoch)+".pth"))

    print("epoch: {}\tlr: {:.6f}\ttrain loss: {:.4f}".format(epoch, optimizer.param_groups[0]["lr"], train_loss), flush=True)
    
    predict_backbone_new('asrf', model, model_dir, result_dir, features_path, vid_list_file_tst, epoch, actions_dict, device, sample_rate) 
    results = eval_txts(cfg.dataset_root, result_dir, dataset, split, 'asrf')
    print(results, flush=True)
    print()

total_time = time.time() - start_time
total_time_str = str(datetime.timedelta(seconds=int(total_time)))
print('Training time {}'.format(total_time_str), flush=True)


