# Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
# Published at NeurIPS 2022
# Written by Shaoshuai Shi 
# All Rights Reserved

from lstm.lstm import MotionLSTM
from lstm.simple_lstm import SimpleMotionLSTM
from mtr_test import eval_single_ckpt
from pathlib import Path

from mtr.config import cfg, cfg_from_yaml_file
from mtr.utils import common_utils
from easydict import EasyDict as edict

from mtr.datasets import build_dataloader

import torch


device = torch.device('cuda')
model = SimpleMotionLSTM()
model.to(device)
model_path = '/files/waymo/saved_models/best_motion_lstm.pth20250609_111228'
state_dict = torch.load(model_path, map_location='cuda')
model.load_state_dict(torch.load(model_path, map_location='cuda'))

cfg_from_yaml_file("/code/jjiang23/csc587/KimchiVision/cfg/kimchiConfig.yaml", cfg)
logger = common_utils.create_logger("/files/waymo/damon_log_simple.txt", rank=0)
args = edict({
    "batch_size": 2,
    "workers": 32,
    "merge_all_iters_to_one_epoch": False,
    "epochs": 5,
    "add_worker_init_fn": False,
    "ckpt": None,
    "save_to_file": False,
})

test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        batch_size=args.batch_size,
        dist=False, workers=args.workers, logger=logger, training=False
)

output_dir = Path("/files/waymo/simple_output") / 'output' / 'eval'
output_dir.mkdir(parents=True, exist_ok=True)

eval_output_dir = output_dir / 'eval'

eval_single_ckpt(model, test_loader, args, eval_output_dir=eval_output_dir, logger=logger, epoch_id='no_number', dist_test=False)