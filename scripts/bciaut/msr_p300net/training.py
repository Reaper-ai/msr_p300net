import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure path to parent directory is added to import shared config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config

# Import model-specific components locally
from model import apply_max_norm

class SmoothBalancedFocalLoss(nn.Module):
    """
    Computes a balanced binary focal loss with label smoothing to handle class imbalance
    and output overconfidence.
    """
    def __init__(self, gamma=2.0, pos_weight=5.0, smoothing=0.05):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.smoothing = smoothing

    def forward(self, inputs, targets):
        # Apply label smoothing
        targets_smooth = targets * (1.0 - self.smoothing) + (1.0 - targets) * self.smoothing
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets_smooth,
            pos_weight=torch.tensor([self.pos_weight], device=inputs.device),
            reduction='none'
        )
        with torch.no_grad():
            probs = torch.sigmoid(inputs)
            pt = torch.where(targets == 1.0, probs, 1.0 - probs)
        return ((1 - pt) ** self.gamma * bce_loss).mean()

def train_subject_epoch(model, loader, optimizer, scheduler, criterion, amp_scaler):
    """
    Trains a model for a single epoch using automatic mixed precision (AMP) and max-norm constraint.
    """
    model.train()
    running_loss = 0.0
    for batch in loader:
        data, target = batch[0], batch[1]
        optimizer.zero_grad(set_to_none=True)
        
        # AMP for faster training
        with torch.amp.autocast(device_type=config.DEVICE.type):
            output = model(data)
            loss = criterion(output, target.float().unsqueeze(1))
            
        amp_scaler.scale(loss).backward()
        amp_scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        
        scale_before = amp_scaler.get_scale()
        amp_scaler.step(optimizer)
        amp_scaler.update()
        
        apply_max_norm(model)
        
        if not (scale_before > amp_scaler.get_scale()):
            scheduler.step()
            
        running_loss += loss.item()
    return running_loss / len(loader)

def evaluate_val_loss(model, loader, criterion):
    """
    Evaluates validation loss on the validation dataset.
    """
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            data, target = batch[0], batch[1]
            with torch.amp.autocast(device_type=config.DEVICE.type):
                output = model(data)
                loss = criterion(output, target.float().unsqueeze(1))
            running_loss += loss.item()
    return running_loss / len(loader)
