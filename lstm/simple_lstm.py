#common libs
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from torchinfo import summary
import math
from easydict import EasyDict as edict

from einops import rearrange, repeat

class SimpleMotionLSTM(nn.Module):
    '''
    Input: 
    - obj_trajs (num_center_objects(batch_size), num_objects, num_timestamps, num_attrs)
    - obj_trajs_mask (num_center_objects(batch_size), num_objects, num_timestamps)
    - map_polylines (num_center_objects(batch_size),num_polylines, num_points_each_polyline, 7)
    - map_polylines_mask (num_center_objects(batch_size),num_polylines(4000), num_points_each_polyline(20))
    - track index (num_center_objects(batch_size), )

    '''
    def __init__(self, 
                input_dim=129 * 11 * 29,  # max num agents, max timesteps, dim of attributes
                # Map polylines encoder parameters
                map_polyline_encoder_output_dim=256,  # Hidden dimension for the map polyline encoder
                map_polyline_encoder_hidden_dim=256,  # Hidden dimension for the map polyline encoder
                map_polyline_encoder_input_dim= 768*20*9,  # Input dimension for the map polyline encoder
                # Encoder and decoder parameters for object trajectories
                encoder_hidden_dim=256,
                encoder_output_dim=256,  # Output dimension of the encoder

                decoder_hidden_dim=256,
                decoder_output_dim=256,  # Output dimension of the decoder
                # LSTM parameters
                lstm_input_dim=512,
                lstm_hidden_dim=512,
                lstm_num_layers=2,
                # attention parameters
                # mode _predictor parameters
                mode_predictor_input_dim=80 * 4,
                mode_predictor_hidden_dim=256,
                mode_predictor_output_dim=256,  # Output dimension of the mode predictor
                # trajectory decoder parameters
                trajectory_decoder_hidden_dim=256,
                trajectory_decoder_output_dim=256,  # Output dimension of the trajectory decoder
                num_modes=6,  # Number of prediction modes
                future_steps=80,  # Number of future timesteps to predict
                state_dim=4, # x, y, vx, vy
                dropout=0.1):
        super().__init__()

        self.input_dim = input_dim
        self.encoder_hidden_dim = encoder_hidden_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_num_layers = lstm_num_layers
        self.num_modes = num_modes
        self.future_steps = future_steps
        self.dropout = dropout
        self.state_dim = state_dim
        self.map_polyline_encoder_input_dim = map_polyline_encoder_input_dim
        
        # Map polylines encoder
        self.map_polyline_encoder = nn.Sequential(
            nn.Linear(map_polyline_encoder_input_dim, map_polyline_encoder_hidden_dim),  
            nn.ReLU(),
            nn.Linear(map_polyline_encoder_hidden_dim, map_polyline_encoder_output_dim)
        )

        # Feature encoder for input trajectories
        self.feature_encoder = nn.Sequential(
            nn.Linear(input_dim, encoder_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(encoder_hidden_dim, encoder_output_dim)
        )

        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout if lstm_num_layers > 1 else 0,
            bidirectional=False
        )
        
        # Multi-modal prediction heads
        self.mode_predictor = nn.Sequential(
            nn.Linear(mode_predictor_input_dim, mode_predictor_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mode_predictor_output_dim, 1)
        )
        
        # Trajectory decoder for each mode
        self.traj_decoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(lstm_hidden_dim, trajectory_decoder_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(trajectory_decoder_hidden_dim, trajectory_decoder_hidden_dim),
                nn.ReLU(),
                nn.Linear(trajectory_decoder_output_dim, future_steps * self.state_dim)  # x, y, width, height, heading, vx, vy for each timestep
            ) for _ in range(num_modes)
        ])
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if len(param.shape) >= 2:
                    nn.init.xavier_uniform_(param)
                else:
                    nn.init.uniform_(param, -0.1, 0.1)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
        
    def forward(self, batch):
        """
        A batch contains:
        - batch_size
        - input_dict: a dictionary containing: {
            'obj_trajs'
            'obj_trajs_mask'
            'track_index_to_predict'
            'map_polylines'
            'map_polylines_mask'
            'map_polylines_center'
        
        }
        - batch_sample_count
        """
        input_dict=batch["input_dict"]
        obj_trajs = input_dict['obj_trajs'].to("cuda")  # (batch_size, num_objects, num_timestamps, input_dim)
        obj_trajs_mask = input_dict['obj_trajs_mask'].to("cuda")  # (batch_size, num_objects, num_timestamps)
        track_indices = input_dict['track_index_to_predict'].to("cuda")  # (batch_size,)
        
        map_polylines=input_dict["map_polylines"].to('cuda')  # (batch_size, num_polylines, num_points_each_polyline, 9)
        map_polylines_mask=input_dict["map_polylines_mask"].to('cuda') # (batch_size, num_polylines, num_points_each_polyline)
        

        # obj shape
        centers, objects, timesteps, trajs = obj_trajs.shape

        # map shape
        batch, polylines, line, point = map_polylines.shape

        # mask the objects
        obj_trajs_mask = rearrange(obj_trajs_mask, "centers objects timesteps -> centers objects timesteps 1")
        obj_trajs = obj_trajs * obj_trajs_mask
        obj_trajs = rearrange(obj_trajs, "centers objects timesteps trajs -> centers timesteps (objects trajs)")
        # pad if not long enough
        if obj_trajs.shape[2] < self.input_dim:
            padding_length = self.input_dim - obj_trajs.shape[2]
            obj_trajs = F.pad(obj_trajs, (padding_length, 0))

        trajs = []
        for center in range(centers):
            trajs.append(self.feature_encoder(obj_trajs[center, :, :]))
        obj_trajs = torch.stack(trajs)
        

        # encode the maps
        map_polylines_mask = repeat(map_polylines_mask, "centers objects timesteps -> centers objects timesteps 1")
        map_polylines = map_polylines * map_polylines_mask
        map_polylines = rearrange(map_polylines, "centers polylines line point -> centers (polylines line point)")
              
        # the max polylines retrieved is 768. Pad when necessary
        if map_polylines.shape[1] < self.map_polyline_encoder_input_dim:
            padding_length = self.map_polyline_encoder_input_dim - map_polylines.shape[1]
            map_polylines = F.pad(map_polylines, (padding_length, 0))
        
        # encode map
        map_polylines = self.map_polyline_encoder(map_polylines)
        map_polylines = repeat(map_polylines, "centers embedding -> centers timesteps embedding", timesteps=timesteps)
        
        # stitch the map and the objects
        obj_map_features = torch.cat([obj_trajs, map_polylines], dim=-1)
        
        # run the LSTM for each center
        obj_map_features, _ = self.lstm(obj_map_features)
        
        # get the last valid index
        valid_obj_map_features = []
        obj_trajs_mask = rearrange(obj_trajs_mask, "centers objects timesteps 1 -> centers objects timesteps")
        for center in range(centers):
            # Take the last valid output for each sequence
            center_idx = track_indices[center]
            last_valid_idx = torch.nonzero(obj_trajs_mask[center, center_idx, :])[-1].squeeze()
            valid_obj_map_features.append(obj_map_features[center, last_valid_idx, :])

        valid_obj_map_features = torch.stack(valid_obj_map_features)

        # Now predict the trajectories for each of the modes
        pred_trajs_list = []
        for mode_idx in range(self.num_modes):
            traj_flat = self.traj_decoders[mode_idx](valid_obj_map_features)  # (batch_size, future_steps * 4)
            traj = rearrange(traj_flat, "b (future traj) -> b future traj", future=self.future_steps, traj=self.state_dim)
            pred_trajs_list.append(traj)

        pred_trajs = torch.stack(pred_trajs_list, dim=1)  # (batch_size, num_modes, future_steps, 4)
        
        # predict a confidence score for each of the trajectories
        pred_scores_list = []
        for center in range(centers):
            traj_flat = rearrange(pred_trajs[center, :, :, :], "mode timestep state -> mode (timestep state)")
            mode_logits = self.mode_predictor(traj_flat).squeeze(dim=1)
            score = F.softmax(mode_logits, dim=-1)
            pred_scores_list.append(score)
        
        # Predict mode probabilities
        pred_scores = torch.stack(pred_scores_list)  # (batch_size, num_modes)

        return pred_scores, pred_trajs

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
    