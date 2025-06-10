import torch
import torch.nn as nn
import torch.nn.functional as F
#define loss function
class MotionLoss(nn.Module):
    def __init__(self, 
                 regression_loss_weight=1.0,
                 classification_loss_weight=1.0,
                 future_loss_weight=1.0):
        super(MotionLoss, self).__init__()
        self.reg_weight = regression_loss_weight
        self.cls_weight = classification_loss_weight
        self.future_weight = future_loss_weight
    
    def forward(self, pred_scores, pred_trajs, batch_dict):
        # Get device from predictions instead of hardcoding 'cuda'
        device = pred_scores.device
        
        center_gt_trajs = batch_dict['input_dict']['center_gt_trajs'].to(device)
        center_gt_trajs_mask = batch_dict['input_dict']['center_gt_trajs_mask'].to(device)
        
        batch_size, num_modes, future_steps, _ = pred_trajs.shape
        
        # Compute trajectory regression loss for each mode
        gt_trajs_expanded = center_gt_trajs.unsqueeze(1).expand(-1, num_modes, -1, -1)
        gt_mask_expanded = center_gt_trajs_mask.unsqueeze(1).expand(-1, num_modes, -1)
        
        # L2 loss for position (x, y)
        pos_loss = F.mse_loss(
            pred_trajs[:, :, :, :2] * gt_mask_expanded.unsqueeze(-1),
            gt_trajs_expanded[:, :, :, :2] * gt_mask_expanded.unsqueeze(-1),
            reduction='none'
        ).sum(dim=-1)
        
        # L2 loss for velocity (vx, vy)
        vel_loss = F.mse_loss(
            pred_trajs[:, :, :, 2:4] * gt_mask_expanded.unsqueeze(-1),
            gt_trajs_expanded[:, :, :, 2:4] * gt_mask_expanded.unsqueeze(-1),
            reduction='none'
        ).sum(dim=-1)
        
        # Weighted loss over time
        time_weights = torch.exp(-0.025 * torch.arange(future_steps, device=device))
        time_weights = time_weights.view(1, 1, -1)
        
        pos_loss = (pos_loss * time_weights * gt_mask_expanded).sum(dim=-1)
        vel_loss = (vel_loss * time_weights * gt_mask_expanded).sum(dim=-1)
        
        # Find best mode for each sample
        total_traj_loss = pos_loss + vel_loss
        best_mode_indices = torch.argmin(total_traj_loss, dim=1)
        
        # Regression loss (best mode)
        best_pos_loss = pos_loss[torch.arange(batch_size), best_mode_indices].mean()
        best_vel_loss = vel_loss[torch.arange(batch_size), best_mode_indices].mean()
        regression_loss = best_pos_loss + best_vel_loss
        
        # FIXED: Classification loss - use class indices directly
        classification_loss = F.cross_entropy(pred_scores, best_mode_indices)
        
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
                # Encoder parameters for object trajectories
                encoder_hidden_dim=256,
                encoder_output_dim=256,  # Output dimension of the encoder
                # LSTM parameters
                lstm_hidden_dim=256,
                lstm_num_layers=2,
                # Mode predictor parameters
                mode_predictor_hidden_dim=256,
                # Trajectory decoder parameters
                trajectory_decoder_hidden_dim=256,
                num_modes=6,  # Number of prediction modes
                future_steps=80,  # Number of future timesteps to predict
                dropout=0.1):
        super(MotionLSTM, self).__init__()

        self.criterion = MotionLoss()

        self.input_dim = input_dim
        self.encoder_hidden_dim = encoder_hidden_dim
        self.encoder_output_dim = encoder_output_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_num_layers = lstm_num_layers
        self.num_modes = num_modes
        self.future_steps = future_steps
        self.dropout = dropout
        self.map_polyline_encoder_output_dim = map_polyline_encoder_output_dim
        
        # Map polylines encoder - will be initialized dynamically
        self.map_polyline_encoder = None
        self.map_polyline_encoder_hidden_dim = map_polyline_encoder_hidden_dim

        # Feature encoder for input trajectories
        self.feature_encoder = nn.Sequential(
            nn.Linear(input_dim, encoder_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(encoder_hidden_dim, encoder_output_dim)
        )

        # Fusion layer for feature encoder with map polylines
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
            nn.Linear(mode_predictor_hidden_dim, num_modes)
        )
        
        # Trajectory decoder for each mode
        self.traj_decoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(lstm_hidden_dim, trajectory_decoder_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(trajectory_decoder_hidden_dim, trajectory_decoder_hidden_dim),
                nn.ReLU(),
                nn.Linear(trajectory_decoder_hidden_dim, future_steps * 4)  # x, y, vx, vy for each timestep
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
    
    def _init_map_encoder(self, input_size):
        """Initialize map encoder with correct input size"""
        if self.map_polyline_encoder is None:
            self.map_polyline_encoder = nn.Sequential(
                nn.Linear(input_size, self.map_polyline_encoder_hidden_dim),  
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.map_polyline_encoder_hidden_dim, self.map_polyline_encoder_output_dim)
            )
            # Move to same device as other parameters
            device = next(self.parameters()).device
            self.map_polyline_encoder = self.map_polyline_encoder.to(device)
            # Initialize weights for the new layers
            for module in self.map_polyline_encoder:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.constant_(module.bias, 0)
    
    def _init_weights(self):
        """Initialize model weights"""
        for name, param in self.named_parameters():
            if 'weight' in name and len(param.shape) >= 2:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
        
    def forward(self, batch_dict):
        """
        Forward pass of the model
        
        Args:
            batch_dict: Dictionary containing:
                - obj_trajs: (batch_size, num_objects, num_timestamps, input_dim)
                - obj_trajs_mask: (batch_size, num_objects, num_timestamps)
                - track_index_to_predict: (batch_size,) indices of center objects
                - static_map_polylines: (batch_size, num_polylines, num_points_each_polyline, 7)
                - static_map_polylines_mask: (batch_size, num_polylines, num_points_each_polyline)
        
        Returns:
            pred_scores: (batch_size, num_modes) - confidence scores for each mode
            pred_trajs: (batch_size, num_modes, future_steps, 4) - predicted trajectories
        """
        input_dict = batch_dict["input_dict"]
        
        # Get device from model parameters
        device = next(self.parameters()).device
        
        obj_trajs = input_dict['obj_trajs'].to(device)  # (batch_size, num_objects, num_timestamps, input_dim)
        obj_trajs_mask = input_dict['obj_trajs_mask'].to(device)  # (batch_size, num_objects, num_timestamps)
        track_indices = input_dict['track_index_to_predict'].to(device)  # (batch_size,)
        
        static_map_polylines = input_dict["static_map_polylines"].to(device)  # (batch_size, num_polylines, num_points_each_polyline, 7)
        static_map_polylines_mask = input_dict["static_map_polylines_mask"].to(device)  # (batch_size, num_polylines, num_points_each_polyline)
        
        batch_size, num_objects, num_timestamps, input_dim = obj_trajs.shape
        
        # Encode map polylines
        map_polyline_features = static_map_polylines * static_map_polylines_mask.unsqueeze(-1).float()  # Apply mask to polylines
        map_flat_size = map_polyline_features.shape[1] * map_polyline_features.shape[2] * map_polyline_features.shape[3]
        map_polyline_features = map_polyline_features.view(batch_size, map_flat_size)  # Flatten
        
        # Initialize map encoder with correct input size
        self._init_map_encoder(map_flat_size)
        map_polyline_features = self.map_polyline_encoder(map_polyline_features)  # (batch_size, map_polyline_encoder_output_dim)

        # Encode input features
        obj_features = self.feature_encoder(obj_trajs.view(-1, input_dim))
        obj_features = obj_features.view(batch_size, num_objects, num_timestamps, self.encoder_output_dim)
        
        # Apply mask to features (only once)
        mask_expanded = obj_trajs_mask.unsqueeze(-1).expand(-1, -1, -1, self.encoder_output_dim)
        obj_features = obj_features * mask_expanded.float()  # (batch_size, num_objects, num_timestamps, encoder_output_dim)
        
        # Concatenate object features with map polyline features
        map_polyline_expanded = map_polyline_features[:, None, None, :]  # (batch_size, 1, 1, map_encoder_output_dim)
        map_polyline_expanded = map_polyline_expanded.expand(-1, num_objects, num_timestamps, -1)
        obj_map_features = torch.cat((obj_features, map_polyline_expanded), dim=-1)  
        obj_features = self.fusion_layer(obj_map_features)  # (batch_size, num_objects, num_timestamps, encoder_output_dim)
        
        # Process trajectories through LSTM more efficiently
        # Reshape to process all objects together
        obj_features_reshaped = obj_features.view(batch_size * num_objects, num_timestamps, self.encoder_output_dim)
        obj_mask_reshaped = obj_trajs_mask.view(batch_size * num_objects, num_timestamps)
        
        # Create packed sequence for efficient LSTM processing
        seq_lengths = obj_mask_reshaped.sum(dim=1).cpu()  # (batch_size * num_objects,)
        
        # Only process sequences with length > 0
        valid_indices = seq_lengths > 0
        if valid_indices.sum() > 0:
            valid_features = obj_features_reshaped[valid_indices]
            valid_lengths = seq_lengths[valid_indices]
            
            # Pack sequences
            packed_input = nn.utils.rnn.pack_padded_sequence(
                valid_features, valid_lengths, batch_first=True, enforce_sorted=False
            )
            
            # Process through LSTM
            packed_output, _ = self.lstm(packed_input)
            lstm_output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
            
            # Get last valid output for each sequence
            last_outputs = []
            for i, length in enumerate(output_lengths):
                if length > 0:
                    last_outputs.append(lstm_output[i, length-1, :])
                else:
                    last_outputs.append(torch.zeros(self.lstm_hidden_dim, device=device))
            
            valid_last_outputs = torch.stack(last_outputs, dim=0)
        
        # Reconstruct full output tensor
        all_lstm_outputs = torch.zeros(batch_size * num_objects, self.lstm_hidden_dim, device=device)
        if valid_indices.sum() > 0:
            all_lstm_outputs[valid_indices] = valid_last_outputs
        
        all_lstm_outputs = all_lstm_outputs.view(batch_size, num_objects, self.lstm_hidden_dim)
        
        # Apply attention mechanism for object interactions
        # Create attention mask: True for positions to ignore
        attn_mask = ~(obj_trajs_mask.sum(dim=2) > 0)  # (batch_size, num_objects)
        
        attn_output, _ = self.attention(
            all_lstm_outputs, all_lstm_outputs, all_lstm_outputs,
            key_padding_mask=attn_mask
        )
        
        # Extract center object features
        center_features = []
        for b in range(batch_size):
            center_idx = track_indices[b]
            center_features.append(attn_output[b, center_idx, :])
        center_features = torch.stack(center_features, dim=0)  # (batch_size, lstm_hidden_dim)
        
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
        
        loss_dict = self.criterion(pred_scores, pred_trajs, batch_dict)

        if self.training:
            return loss_dict["total_loss"], {}, {}
        else:
            batch_dict['pred_scores'] = pred_scores
            batch_dict['pred_trajs'] = pred_trajs

            return batch_dict