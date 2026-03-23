import os
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
import math
import copy
import random
import csv
import pdb
import argparse
import time
import datetime

import sys
sys.path.append('./model_asformer_CASAgCA_s23')
from libs import models
from libs.optimizer import get_optimizer
from libs.dataset import ActionSegmentationDataset, collate_fn
from libs.transformer import TempDownSamp, ToTensor

sys.path.append('./backbones/ms-tcn')
from model import MultiStageModel

from model_maskformer_CASAgCA_pre_newmatching_bk.utils import eval_txts, load_meta
from model_maskformer_CASAgCA_pre_newmatching_bk.predict import predict_refiner
from model_maskformer_CASAgCA_pre_newmatching_bk.refiner_train import refiner_train
from model_maskformer_CASAgCA_pre_newmatching_bk.tcn import ourmodel, Asformer

import configs.refiner_config as cfg
import configs.sstda_config as sstda_cfg


##########################################################################
# >>>>>>>>>> hy parameter <<<<<<<<<<<<<<<<
##########################################################################
parser = argparse.ArgumentParser()
parser.add_argument('--split', default=1, type=int) 
parser.add_argument('--epoch', default=2, type=int) 
parser.add_argument('--seed', default=666, type=int)
parser.add_argument('--lr', default=0.0001, type=float) 
parser.add_argument('--model_name', default='tmp')
parser.add_argument('--dataset', default='breakfast')
parser.add_argument('--asformer_model', default='model/model_maskformer_CASAgCA_pre_newmatching_bk/breakfast/split_2/epoch-12.model', help='backbone pretrained model')

args = parser.parse_args()
print(args, flush=True)

np.random.seed(0)
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark = False

device = 'cuda'
dataset = args.dataset     # choose from gtea, 50salads, breakfast
split = args.split            # gtea : 1~4, 50salads : 1~5, breakfast : 1~4
pool_backbone_name = ['mstcn'] # 'asrf', 'mstcn', 'sstda', 'mgru'
main_backbone_name = 'mstcn'
model_name = args.model_name
cfg.max_epoch = args.epoch
cfg.lr = args.lr
# model_name = 'refiner'+main_backbone_name.upper()+'-'+'-'.join(pool_backbone_name) 

actions_dict, num_actions, gt_path, features_path, vid_list_file, vid_list_file_tst, sample_rate, model_dir, result_dir,  record_dir = load_meta(cfg.dataset_root, cfg.model_root, cfg.result_root, cfg.record_root, dataset, split, model_name)

##########################################################################
# >>>>>>>>>> dataset <<<<<<<<<<<<<<<<
##########################################################################
train_data = ActionSegmentationDataset(
        dataset,
        transform=Compose([ToTensor(), TempDownSamp(sample_rate)]),
        mode="trainval",
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
        pin_memory=True
    )

curr_split_dir = os.path.join(cfg.dataset_root, dataset, 'splits')
split_dict = {k+1:[] for k in range(cfg.num_splits[dataset])}
for i in range(eval('cfg.num_splits["{}"]'.format(dataset))):
    curr_fp = os.path.join(curr_split_dir, 'test.split{}.bundle'.format(i+1))
    f = open(curr_fp, 'r')
    lines = f.readlines()
    for l in lines:
        curr_name = l.split('.')[0]
        split_dict[i+1].append(curr_name)
    f.close()

##########################################################################
# >>>>>>>>>> load mstcn model <<<<<<<<<<<<<<<<
##########################################################################
pool_backbones = {bn: {k+1:None for k in range(cfg.num_splits[dataset])} for bn in cfg.backbone_names}

for i in range(eval('cfg.num_splits["{}"]'.format(dataset))):
    if 'mstcn' in cfg.backbone_names:
        curr_mstcn = MultiStageModel(cfg.num_stages,
                                     num_layers = cfg.num_layers,
                                     num_f_maps = cfg.num_f_maps,
                                     dim = cfg.features_dim,
                                     num_classes = num_actions)
        curr_mstcn.load_state_dict(torch.load(os.path.join(cfg.model_root, 'mstcn', dataset,
                                                          'split_{}'.format(i+1),
                                                          'epoch-{}.model'.format(cfg.best['mstcn'][dataset][i]))))
        curr_mstcn.to(device)
        pool_backbones['mstcn'][i+1] = curr_mstcn
main_backbones = copy.deepcopy(pool_backbones[main_backbone_name])
print("load best mstcn metric sucess!", cfg.best['mstcn'], flush=True)

##########################################################################
# >>>>>>>>>> load our model <<<<<<<<<<<<<<<<
##########################################################################
backbone = Asformer(3, 10, 2, 2, cfg.n_features, cfg.in_channel, num_actions)
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

##########################################################################
# >>>>>>>>>> load Asformer model <<<<<<<<<<<<<<<<
##########################################################################
model_dict = model.state_dict()
refine_dict = torch.load(args.asformer_model, map_location='cpu')
pretrained_dict = {}
for k ,v in refine_dict.items():
    if k in model_dict:
        pretrained_dict[k] = v
model.load_state_dict(pretrained_dict, strict=False)

predict_refiner(model, main_backbone_name, main_backbones, 
                    split_dict, model_dir, result_dir, 
                    features_path, vid_list_file_tst,
                    200, actions_dict, device, sample_rate)    
results = eval_txts(cfg.dataset_root, result_dir, dataset, split, model_name)
print(results, flush=True)
 
print(args.asformer_model, flush=True)
print("Use split:{}".format(split), flush=True)
print("Load Asformer backbone pretrained model success and freeze!\n", flush=True)

pdb.set_trace()

##########################################################################
# >>>>>>>>>> optimizer <<<<<<<<<<<<<<<<
##########################################################################
optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

##########################################################################
# >>>>>>>>>> train <<<<<<<<<<<<<<<<
##########################################################################
print("Start training", flush=True)
start_time = time.time()

for epoch in range(cfg.max_epoch):
    train_loss = refiner_train(cfg, dataset, train_loader, model, pool_backbones, pool_backbone_name, optimizer, epoch, split_dict, device, num_actions)
    torch.save(model.state_dict(), os.path.join(model_dir, "epoch-"+str(epoch+1)+".model"))
    print("epoch: {}\tlr: {:.5f}\ttrain loss: {:.4f}".format(epoch+1, optimizer.param_groups[0]["lr"], train_loss), flush=True)

    predict_refiner(model, main_backbone_name, main_backbones, 
                    split_dict, model_dir, result_dir, 
                    features_path, vid_list_file_tst,
                    epoch, actions_dict, device, sample_rate)    
    results = eval_txts(cfg.dataset_root, result_dir, dataset, split, model_name)
    print(results, flush=True)
    print()

total_time = time.time() - start_time
total_time_str = str(datetime.timedelta(seconds=int(total_time)))
print('Training time {}'.format(total_time_str), flush=True)



