from lstm.loss import MotionLoss
from datetime import datetime
import torch
import numpy as np
from torch import nn

#train loop
def train_model(model, train_dataloader, val_dataloader, num_epochs=5, lr=1e-3):
    """
    Training loop for the LSTM model
    """
    assert torch.cuda.is_available(), "CUDA is not available. Please check your PyTorch installation."
    device = torch.device('cuda')
    model.to(device)

    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = MotionLoss()
    
    best_val_loss = float('inf')
    now = datetime.now()
    print(now.strftime("%Y%m%d_%H%M%S"))
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_losses = []
        
        for batch_idx, batch_dict in enumerate(train_dataloader):
            # Move data to device
            for key, value in batch_dict.items():
                if isinstance(value, torch.Tensor):
                    batch_dict[key] = value.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            pred_scores, pred_trajs = model(batch_dict)
            
            # Compute loss
            loss = criterion(pred_scores, pred_trajs, batch_dict)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_losses.append(loss.item())
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')

        
        # Validation phase
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch_dict in val_dataloader:
                # Move data to device
                for key, value in batch_dict.items():
                    if isinstance(value, torch.Tensor):
                        batch_dict[key] = value.to(device)
                
                pred_scores, pred_trajs = model(batch_dict)
                loss = criterion(pred_scores, pred_trajs, batch_dict)
                val_losses.append(loss)
        
        scheduler.step()
        
        avg_train_loss = torch.Tensor(train_losses).mean()
        avg_val_loss = torch.Tensor(val_losses).mean()
        
        print(f'Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            now = datetime.now()
            torch.save(model.state_dict(), '/files/waymo/saved_models/best_motion_lstm.pth' + now.strftime("%Y%m%d_%H%M%S"))

    return model