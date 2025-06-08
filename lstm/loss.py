import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat

#define loss function
class MotionLoss(nn.Module):
    """
    Loss function for trajectory prediction
    """
    
    def __init__(self, 
                 regression_loss_weight=1.0,
                 classification_loss_weight=1.0,
                 future_loss_weight=1.0,
                 gamma=0.025 # this gamma is chosen because at timestep 80, the time weight won't be zero
                ):
        super(MotionLoss, self).__init__()
        self.reg_weight = regression_loss_weight
        self.cls_weight = classification_loss_weight
        self.future_weight = future_loss_weight
        self.gamma = gamma

    def forward(self, pred_scores, pred_trajs, input):
        gt_trajs = input['input_dict']['center_gt_trajs'].to('cuda')  # (batch_size, future_steps, 4)
        gt_trajs_mask = input['input_dict']['center_gt_trajs_mask'].to('cuda')  # (batch_size, future_steps)

        batch_size, num_modes, future_steps, _ = pred_trajs.shape

        gt_trajs = repeat(gt_trajs, "batch future point -> batch mode future point", mode=num_modes)
        gt_trajs_mask = repeat(gt_trajs_mask, "batch future -> batch mode future 1", mode=num_modes)

        # L2 loss for position (x, y)
        pos_loss = F.mse_loss(
            pred_trajs[:, :, :, :2] * gt_trajs_mask,
            gt_trajs[:, :, :, :2] * gt_trajs_mask,
            reduction='none'
        ).sum(dim=-1)  # (batch_size, num_modes, future_steps)
        
        # L2 loss for velocity (vx, vy)
        vel_loss = F.mse_loss(
            pred_trajs[:, :, :, 2:4] * gt_trajs_mask,
            gt_trajs[:, :, :, 2:4] * gt_trajs_mask,
            reduction='none'
        ).sum(dim=-1)  # (batch_size, num_modes, future_steps)

        # Weighted loss over time (give more weight to near future)
        time_weights = torch.exp(-self.gamma * torch.arange(future_steps, device=pred_trajs.device))
        time_weights = repeat(time_weights, "weight -> batch modes weight", batch=batch_size, modes=num_modes)
        
        gt_trajs_mask = rearrange(gt_trajs_mask, "batch modes future single -> batch modes (future single)")
        pos_loss = (pos_loss * time_weights * gt_trajs_mask).sum(dim=-1)  # (batch_size, num_modes)
        vel_loss = (vel_loss * time_weights * gt_trajs_mask).sum(dim=-1)  # (batch_size, num_modes)

        total_loss = (pos_loss + vel_loss)
        # pick the mode that is the winner, based on total loss
        winner = torch.argmin(total_loss, dim=1)

        # fancy indexing: What I am saying is, pick the winner across each batch.
        regression_loss = total_loss[torch.arange(batch_size), winner]

        targets = torch.zeros([batch_size, 6]).to("cuda")
        targets[torch.arange(batch_size), winner] = 1
        classification_loss = F.cross_entropy(pred_scores, targets)

        loss = (self.reg_weight * regression_loss + self.cls_weight * classification_loss)

        return loss.mean()