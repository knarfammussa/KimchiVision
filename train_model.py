from lstm.train_util import train_model
from lstm.lstm import MotionLSTM
from lstm.simple_lstm import SimpleMotionLSTM

#mtr modules
from mtr.datasets import build_dataloader
from mtr.config import cfg, cfg_from_yaml_file
from mtr.utils import common_utils

#common libs
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from torchinfo import summary
import math
from easydict import EasyDict as edict

import os

cfg_from_yaml_file("/code/jjiang23/csc587/KimchiVision/cfg/kimchiConfig.yaml", cfg)
logger = common_utils.create_logger("/files/waymo/damon_log.txt", rank=0)
args = edict({
    "batch_size": 1,
    "workers": 32,
    "merge_all_iters_to_one_epoch": False,
    "epochs": 5,
    "add_worker_init_fn": False,
    
})

#prepare data
train_set, train_loader, train_sampler = build_dataloader(
    dataset_cfg=cfg.DATA_CONFIG,
    batch_size=args.batch_size,
    dist=False, workers=args.workers,
    logger=logger,
    training=True,
    merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch,
    total_epochs=args.epochs,
    add_worker_init_fn=args.add_worker_init_fn,
)

test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        batch_size=args.batch_size,
        dist=False, workers=args.workers, logger=logger, training=False
)

# model = MotionLSTM()
model = SimpleMotionLSTM()

# Train the model
trained_model = train_model(model, train_loader, test_loader)