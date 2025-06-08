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
                input_dim=29,  # Based on MTR dataset obj_trajs feature dimension
                # Map polylines encoder parameters
                map_polyline_encoder_output_dim=256,  # Hidden dimension for the map polyline encoder
                map_polyline_encoder_hidden_dim=512,  # Hidden dimension for the map polyline encoder
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
        super(MotionLSTM, self).__init__()

        self.input_dim = input_dim
        self.encoder_hidden_dim = encoder_hidden_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_num_layers = lstm_num_layers
        self.num_modes = num_modes
        self.future_steps = future_steps
        self.dropout = dropout
        
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

        #fusion layer for feature encoder with map polylines
        self.fusion_layer = nn.Sequential(
            nn.Linear(encoder_output_dim + map_polyline_encoder_output_dim, encoder_output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
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
        
        static_map_polylines=input_dict["static_map_polylines"].to('cuda')  # (batch_size, num_polylines, num_points_each_polyline, 7)
        static_map_polylines_mask=input_dict["static_map_polylines_mask"].to('cuda') # (batch_size, num_polylines, num_points_each_polyline)

        # obj shape
        batch, objects, timestamps, trajs = obj_trajs.shape

        # map shape
        batch, polylines, line, point = static_map_polylines.shape

        # apply mask than encode map 
        map_features = static_map_polylines * rearrange(static_map_polylines_mask, "b num_poly mask -> b num_poly mask 1").float()
        map_features = rearrange(map_features, "b num_poly num_points point -> b (num_poly num_points point)")
        map_features = self.map_polyline_encoder(map_features)

        # encode features than apply mask
        obj_features = rearrange(obj_trajs, "b o t p -> (b o t) p")
        obj_features = self.feature_encoder(obj_features)
        obj_features = rearrange(obj_features, "(b o t) e -> b o t e", b=batch, o=objects, t=timestamps, e=self.encoder_hidden_dim)

        obj_features = obj_features * rearrange(obj_trajs_mask, "b o t -> b o t 1").float()

        # concatentate map and objects, then encode them
        map_features = repeat(map_features, "b map_features -> b o t map_features", o=objects, t=timestamps)
        map_object_features = torch.cat([map_features, obj_features], dim=-1)
        map_object_features = self.fusion_layer(map_object_features)

        # for each object, pass it through the LSTM
        all_lstm_outputs = []
        
        for obj_idx in range(objects):
            object = map_object_features[:, obj_idx, :, :]  # (batch_size, num_timestamps, hidden_dim)
            lstm_out, _ = self.lstm(object)  # (batch_size, num_timestamps, hidden_dim)

            # Take the last valid output for each sequence
            last_outputs = []
            for b in range(batch):
                last_valid_idx = torch.nonzero(obj_trajs_mask[b, obj_idx, :])[-1].squeeze()
                last_outputs.append(lstm_out[b, last_valid_idx, :])
            
            last_output = torch.stack(last_outputs, dim=0)  # (batch_size, hidden_dim)
            all_lstm_outputs.append(last_output)
        
        all_lstm_outputs = torch.stack(all_lstm_outputs, dim=1)  # (batch_size, num_objects, hidden_dim)

        # Apply attention mechanism for object interactions
        attn_output, _ = self.attention(
            all_lstm_outputs, all_lstm_outputs, all_lstm_outputs  
        ) # (batch_size, num_objects, hidden_dim)

        # Extract the objcts we are trying to predict
        obj_interest_features = []
        for b in range(batch):
            obj_idx = track_indices[b]
            obj_interest_features.append(attn_output[b, obj_idx, :])
        obj_interest_features = torch.stack(obj_interest_features, dim=0)  # (batch_size, hidden_dim)
        
        # Predict mode probabilities
        mode_logits = self.mode_predictor(obj_interest_features)  # (batch_size, num_modes)
        pred_scores = F.softmax(mode_logits, dim=-1)

        # Now predict the trajectories for each of the modes
        pred_trajs_list = []
        for mode_idx in range(self.num_modes):
            traj_flat = self.traj_decoders[mode_idx](obj_interest_features)  # (batch_size, future_steps * 4)
            traj = rearrange(traj_flat, "b (future traj) -> b future traj", future=self.future_steps, traj=4)
            pred_trajs_list.append(traj)
        
        pred_trajs = torch.stack(pred_trajs_list, dim=1)  # (batch_size, num_modes, future_steps, 4)

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
    