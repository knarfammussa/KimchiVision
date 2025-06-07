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

# get mtr's path
path = "/home/dlin42/KimchiVision"
print("THIS IS THE PATH OF THE LSTM, change when needed")
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
                input_dim=29 + 4000*20*7,  # Based on MTR dataset obj_trajs feature dimension
                # Map polylines encoder parameters
                map_polyline_encoder_output_dim=256,  # Hidden dimension for the map polyline encoder
                map_polyline_encoder_input_dim= 4000*20*7,  # Input dimension for the map polyline encoder
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
        
        self.input_dim = input_dim
        self.encoder_hidden_dim = encoder_hidden_dim
        self.lstm_num_layers = lstm_num_layers
        self.num_modes = num_modes
        self.future_steps = future_steps
        self.dropout = dropout
        
        # # Map polylines encoder
        # self.map_polyline_encoder = nn.Sequential(
        #     nn.Linear(map_polyline_encoder_input_dim, 512),  
        #     nn.ReLU(),
        #     nn.Linear(encoder_hidden_dim, encoder_output_dim)
        # )

        # Feature encoder for input trajectories
        self.feature_encoder = nn.Sequential(
            nn.Linear(input_dim, encoder_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(encoder_hidden_dim, encoder_output_dim)
        )
        
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
        
    def forward(self,):
        pass
    