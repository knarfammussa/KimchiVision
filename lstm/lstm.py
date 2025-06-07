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
import sys

from einops import rearrange, repeat

# get mtr's path
path = "/home/dlin42/KimchiVision"
print(f"home of the LSTM {path}, change when needed")
sys.path.append(path)

#mtr modules
from mtr.datasets import build_dataloader
from mtr.config import cfg, cfg_from_yaml_file
from mtr.utils import common_utils

# define model

class MotionLSTM(nn.Module):
    '''
    Input: 
    - obj_trajs (num_center_objects(batch_size), num_objects, num_timestamps, num_attrs)
    - obj_trajs_mask (num_center_objects(batch_size), num_objects, num_timestamps)
    - map_polylines (num_center_objects(batch_size),num_polylines, num_points_each_polyline, 7)
    - map_polylines_mask (num_center_objects(batch_size),num_polylines(4000), num_points_each_polyline(20))
    - track index (num_center_objects(batch_size), )

    '''
    def __init__(self, 
                # Each obj comes with 36 parameters
                input_dim=36,
                # Encoder and decoder parameters for object trajectories
                encoder_hidden_dim=256,
                encoder_output_dim=256,  # Output dimension of the encoder

                decoder_hidden_dim=256,
                decoder_output_dim=256,  # Output dimension of the decoder
                # LSTM parameters
                lstm_hidden_dim=256,
                lstm_num_layers=2,
                # attention parameters
                # mode _predictor parameters
                mode_predictor_hidden_dim=256,
                mode_predictor_output_dim=256,  # Output dimension of the mode predictor
                # trajectory decoder parameters
                trajectory_decoder_hidden_dim=256,
                trajectory_decoder_output_dim=256,  # Output dimension of the trajectory decoder
                num_modes=6,  # Number of prediction modes
                future_steps=80,  # Number of future timesteps to predict
                dropout=0.1):
        
        super().__init__()
        self.input_dim = input_dim
        self.encoder_hidden_dim = encoder_hidden_dim
        self.lstm_num_layers = lstm_num_layers
        self.num_modes = num_modes
        self.future_steps = future_steps
        self.dropout = dropout

        # Feature encoder for input trajectories
        self.feature_encoder = nn.Sequential(
            nn.Linear(input_dim, encoder_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(encoder_hidden_dim, encoder_output_dim)
        )
        
        # TODO: Add Map here
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=encoder_output_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout if lstm_num_layers > 1 else 0,
            bidirectional=False
        )
        
        # Multi-modal prediction heads
        self.mode_predictor = nn.Sequential(
            nn.Linear(lstm_hidden_dim, mode_predictor_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mode_predictor_output_dim, num_modes)
        )
        
        # Trajectory decoder for each mode
        self.traj_decoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(lstm_hidden_dim, trajectory_decoder_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(trajectory_decoder_hidden_dim, trajectory_decoder_hidden_dim),
                nn.ReLU(),
                nn.Linear(trajectory_decoder_output_dim, future_steps * 4)  # x, y, vx, vy for each timestep
            ) for _ in range(num_modes)
        ])
        
        # Attention mechanism for object interactions
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_hidden_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
    def forward(self, batch):
        """
        A batch contains:
        - batch_size
        - input_dict: a dictionary containing: {
            'scenario_id'
            'obj_trajs'
            'obj_trajs_mask'
            'track_index_to_predict'
            'map_polylines'
            'map_polylines_mask'
            'map_polylines_center'
        
        }
        - batch_sample_count
        """
        input = batch["input_dict"]
        obj_trajs = input["obj_trajs"].to("cuda")
        obj_pos = input["obj_trajs_pos"].to("cuda")
        obj_last_pos = input["obj_trajs_last_pos"].to("cuda")
        obj_type = self.convert_type(input["obj_types"]).to("cuda") # car, bicycycle, pedestrian
        obj_trajs_mask = input['obj_trajs_mask'].to("cuda")
        obj_of_interest = input['track_index_to_predict']

        # resize everything to the right shape, then concatenate
        num_center_objects, num_objects, num_timestamps, num_attrs = obj_trajs.shape
        obj_last_pos = repeat(obj_last_pos, "c o p -> c o timestamps p", timestamps=num_timestamps)
        obj_type = repeat(obj_type, "type -> centers type timestamps 1", centers=num_center_objects, timestamps=num_timestamps)
        objs = torch.cat([obj_trajs, obj_pos, obj_last_pos, obj_type], dim=-1)
        
        # flatten for the encoder, then expand the resulting features
        objs = rearrange(objs, "c o t p -> (c o t) p")
        objs_features = rearrange(self.feature_encoder(objs), "(center objs timestamps) hidden -> center objs timestamps hidden", center=num_center_objects, objs=num_objects, timestamps=num_timestamps)

        # apply mask
        obj_trajs_mask = repeat(obj_trajs_mask, "center objects timestamps -> center objects timestamps features", features=self.encoder_hidden_dim).float()
        objs_features = obj_trajs_mask * objs_features

        # apply the LSTM across all objects
        all_lstm = []
        for i in range(num_objects):
            obj_features = objs_features[:, i, :, :]  # (batch_size, num_timestamps, hidden_dim)
            lstm_out, _ = self.lstm(obj_features)  # (batch_size, num_timestamps, hidden_dim)
            all_lstm.append(lstm_out)

        post_lstm = torch.stack(all_lstm, dim=1)
        center_objs = []

        # get the center objects
        for i in range(num_center_objects):
            center_idx = obj_of_interest[i]
            center_objs.append(post_lstm[i, center_idx, :, :])

        center_objs = torch.stack(center_objs, dim=0)
        # predicted_trajs = []
        # for mode in range(self.num_modes):
        #     traj = self.traj_decoders[mode](center_objs)
        #     traj = rearrange(traj, "(center future dim) -> center future dim", center=num_center_objects, future=self.future_steps, dim=4)
        #     predicted_trajs.append(traj)
        
        
        confidence_scores = F.softmax(self.mode_predictor(center_objs), -1)
        return confidence_scores

    def _print_batch(self, batch):
        for key, val in batch["input_dict"].items():
            print(f"Key: {key}, Val: {val.shape}")

    def convert_type(self, obj_type):
        mapping = {
            "TYPE_VEHICLE": 2,
            "TYPE_PEDESTRIAN": 1,
            "TYPE_CYCLIST": 0
        }
        new_list = []
        for i in range(len(obj_type)):
            new_list.append(mapping[obj_type[i]])
        return torch.tensor(new_list)
    