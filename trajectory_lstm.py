import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
import math

class TrajectoryLSTM(nn.Module):
    """
    LSTM model for trajectory prediction using MTR dataset format.
    
    Input: obj_trajs from MTR dataset with shape (num_center_objects, num_objects, num_timestamps, num_features)
    Output: Future trajectory predictions
    """
    
    def __init__(self, 
                 input_dim=29,  # Based on MTR dataset obj_trajs feature dimension
                 hidden_dim=256,
                 num_layers=2,
                 num_modes=6,  # Number of prediction modes
                 future_steps=80,  # Number of future timesteps to predict
                 dropout=0.1):
        super(TrajectoryLSTM, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_modes = num_modes
        self.future_steps = future_steps
        self.dropout = dropout
        
        # Feature encoder for input trajectories
        self.feature_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        # Multi-modal prediction heads
        self.mode_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_modes)
        )
        
        # Trajectory decoder for each mode
        self.traj_decoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, future_steps * 4)  # x, y, vx, vy for each timestep
            ) for _ in range(num_modes)
        ])
        
        # Attention mechanism for object interactions
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        #loss function
        self.criterion = self.TrajectoryLoss()
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
    
    class TrajectoryLoss(nn.Module):
        """
        Loss function for trajectory prediction
        """
        
        def __init__(self, 
                    regression_loss_weight=1.0,
                    classification_loss_weight=1.0,
                    future_loss_weight=1.0):
            super().__init__()
            self.reg_weight = regression_loss_weight
            self.cls_weight = classification_loss_weight
            self.future_weight = future_loss_weight
        
        def forward(self, pred_scores, pred_trajs, batch_dict):
            """
            Compute loss
            
            Args:
                pred_scores: (batch_size, num_modes)
                pred_trajs: (batch_size, num_modes, future_steps, 4)
                batch_dict: Contains ground truth data
            
            Returns:
                loss_dict: Dictionary containing different loss components
            """
            center_gt_trajs = batch_dict['input_dict']['center_gt_trajs'].to('cuda')  # (batch_size, future_steps, 4)
            center_gt_trajs_mask = batch_dict['input_dict']['center_gt_trajs_mask'].to('cuda')  # (batch_size, future_steps)
            
            batch_size, num_modes, future_steps, _ = pred_trajs.shape
            
            # Compute trajectory regression loss for each mode
            gt_trajs_expanded = center_gt_trajs.unsqueeze(1).expand(-1, num_modes, -1, -1)
            gt_mask_expanded = center_gt_trajs_mask.unsqueeze(1).expand(-1, num_modes, -1)
            
            # L2 loss for position (x, y)
            pos_loss = F.mse_loss(
                pred_trajs[:, :, :, :2] * gt_mask_expanded.unsqueeze(-1),
                gt_trajs_expanded[:, :, :, :2] * gt_mask_expanded.unsqueeze(-1),
                reduction='none'
            ).sum(dim=-1)  # (batch_size, num_modes, future_steps)
            
            # L2 loss for velocity (vx, vy)
            vel_loss = F.mse_loss(
                pred_trajs[:, :, :, 2:4] * gt_mask_expanded.unsqueeze(-1),
                gt_trajs_expanded[:, :, :, 2:4] * gt_mask_expanded.unsqueeze(-1),
                reduction='none'
            ).sum(dim=-1)  # (batch_size, num_modes, future_steps)
            
            # Weighted loss over time (give more weight to near future)
            time_weights = torch.exp(-0.1 * torch.arange(future_steps, device=pred_trajs.device))
            time_weights = time_weights.view(1, 1, -1)
            
            pos_loss = (pos_loss * time_weights * gt_mask_expanded).sum(dim=-1)  # (batch_size, num_modes)
            vel_loss = (vel_loss * time_weights * gt_mask_expanded).sum(dim=-1)  # (batch_size, num_modes)
            
            # Find best mode for each sample
            total_traj_loss = pos_loss + vel_loss  # (batch_size, num_modes)
            best_mode_indices = torch.argmin(total_traj_loss, dim=1)  # (batch_size,)
            
            # Regression loss (best mode)
            best_pos_loss = pos_loss[torch.arange(batch_size), best_mode_indices].mean()
            best_vel_loss = vel_loss[torch.arange(batch_size), best_mode_indices].mean()
            regression_loss = best_pos_loss + best_vel_loss
            
            # Classification loss (encourage higher confidence for best mode)
            target_scores = torch.zeros_like(pred_scores)
            target_scores[torch.arange(batch_size), best_mode_indices] = 1.0
            classification_loss = F.cross_entropy(pred_scores, target_scores)
            
            # Total loss
            total_loss = (self.reg_weight * regression_loss + 
                        self.cls_weight * classification_loss)
            
            loss_dict = {
                'total_loss': total_loss,
                'regression_loss': regression_loss,
                'classification_loss': classification_loss,
                'pos_loss': best_pos_loss,
                'vel_loss': best_vel_loss
            }
            
            return loss_dict
    
    def forward(self, batch_dict):
        """
        Forward pass of the model
        
        Args:
            batch_dict: Dictionary containing:
                - obj_trajs: (batch_size, num_objects, num_timestamps, input_dim)
                - obj_trajs_mask: (batch_size, num_objects, num_timestamps)
                - track_index_to_predict: (batch_size,) indices of center objects
        
        Returns:
            pred_scores: (batch_size, num_modes) - confidence scores for each mode
            pred_trajs: (batch_size, num_modes, future_steps, 4) - predicted trajectories
        """
        input_dict=batch_dict["input_dict"]
        obj_trajs = input_dict['obj_trajs'].to("cuda")  # (batch_size, num_objects, num_timestamps, input_dim)
        obj_trajs_mask = input_dict['obj_trajs_mask'].to("cuda")  # (batch_size, num_objects, num_timestamps)
        track_indices = input_dict['track_index_to_predict'].to("cuda")  # (batch_size,)
        # map_polylines, map_polylines_mask = input_dict['map_polylines'].to("cuda"), input_dict['map_polylines_mask'].to("cuda") # (num_center_objects, num_topk_polylines, num_points_each_polyline, 9): [x, y, z, dir_x, dir_y, dir_z, global_type, pre_x, pre_y]

        

        
        batch_size, num_objects, num_timestamps, input_dim = obj_trajs.shape
        
        # Encode input features
        obj_features = self.feature_encoder(obj_trajs.view(-1, input_dim))
        obj_features = obj_features.view(batch_size, num_objects, num_timestamps, self.hidden_dim)
        
        # Apply mask to features
        mask_expanded = obj_trajs_mask.unsqueeze(-1).expand(-1, -1, -1, self.hidden_dim)
        obj_features = obj_features * mask_expanded.float()
        
        # Process each object's trajectory through LSTM
        all_lstm_outputs = []
        
        for obj_idx in range(num_objects):
            obj_seq = obj_features[:, obj_idx, :, :]  # (batch_size, num_timestamps, hidden_dim)
            lstm_out, _ = self.lstm(obj_seq)  # (batch_size, num_timestamps, hidden_dim)
            
            # Take the last valid output for each sequence
            seq_lengths = obj_trajs_mask[:, obj_idx, :].sum(dim=1)  # (batch_size,)
            last_outputs = []
            for b in range(batch_size):
                if seq_lengths[b] > 0:
                    last_idx = int(seq_lengths[b] - 1)
                    last_outputs.append(lstm_out[b, last_idx, :])
                else:
                    last_outputs.append(torch.zeros(self.hidden_dim, device=obj_seq.device))
            
            last_output = torch.stack(last_outputs, dim=0)  # (batch_size, hidden_dim)
            all_lstm_outputs.append(last_output)
        
        all_lstm_outputs = torch.stack(all_lstm_outputs, dim=1)  # (batch_size, num_objects, hidden_dim)
        
        # Apply attention mechanism for object interactions
        attn_output, _ = self.attention(
            all_lstm_outputs, all_lstm_outputs, all_lstm_outputs,
            key_padding_mask=~(obj_trajs_mask.sum(dim=2) > 0)  # (batch_size, num_objects)
        )
        
        # Extract center object features
        center_features = []
        for b in range(batch_size):
            center_idx = track_indices[b]
            center_features.append(attn_output[b, center_idx, :])
        center_features = torch.stack(center_features, dim=0)  # (batch_size, hidden_dim)
        
        # Predict mode probabilities
        mode_logits = self.mode_predictor(center_features)  # (batch_size, num_modes)
        pred_scores = F.softmax(mode_logits, dim=-1)
        
        # Predict trajectories for each mode
        pred_trajs_list = []
        for mode_idx in range(self.num_modes):
            traj_flat = self.traj_decoders[mode_idx](center_features)  # (batch_size, future_steps * 4)
            traj = traj_flat.view(batch_size, self.future_steps, 4)  # (batch_size, future_steps, 4)
            pred_trajs_list.append(traj)
        
        pred_trajs = torch.stack(pred_trajs_list, dim=1)  # (batch_size, num_modes, future_steps, 4)
        
        
        # Compute loss
        loss_dict = self.criterion(pred_scores, pred_trajs, batch_dict)

        if self.training:
            return loss_dict["total_loss"], {}, {}
        else:
            batch_dict['pred_scores'] = pred_scores
            batch_dict['pred_trajs'] = pred_trajs

            return batch_dict
        
