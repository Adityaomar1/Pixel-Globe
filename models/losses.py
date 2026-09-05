import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Binary Dice Loss for change detection.
    Directly optimizes the overlap metric (Dice / F1-Score).
    """
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        logits: (B, 1, H, W) or (B, H, W) raw model output
        targets: (B, 1, H, W) or (B, H, W) binary ground truth (0 or 1)
        """
        probs = torch.sigmoid(logits)
        probs = probs.view(-1)
        targets = targets.view(-1).float()

        intersection = (probs * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)
        return 1.0 - dice


class FocalLoss(nn.Module):
    """
    Binary Focal Loss to address extreme class imbalance between changed and unchanged pixels.
    """
    def __init__(self, alpha=0.75, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        targets = targets.float()
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_t * ((1 - p_t) ** self.gamma) * bce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class CombinedLoss(nn.Module):
    """
    Composite Change Detection Loss:
    Loss = w_bce * BCEWithLogits + w_dice * DiceLoss + w_focal * FocalLoss
    """
    def __init__(self, w_bce=1.0, w_dice=1.0, w_focal=0.5, pos_weight=None):
        super(CombinedLoss, self).__init__()
        self.w_bce = w_bce
        self.w_dice = w_dice
        self.w_focal = w_focal
        
        pos_weight_tensor = torch.tensor([pos_weight]) if pos_weight is not None else None
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        self.dice = DiceLoss()
        self.focal = FocalLoss(alpha=0.75, gamma=2.0)

    def forward(self, logits, targets):
        loss_bce = self.bce(logits, targets.float()) if self.w_bce > 0 else 0.0
        loss_dice = self.dice(logits, targets.float()) if self.w_dice > 0 else 0.0
        loss_focal = self.focal(logits, targets.float()) if self.w_focal > 0 else 0.0

        total_loss = (self.w_bce * loss_bce) + (self.w_dice * loss_dice) + (self.w_focal * loss_focal)
        return total_loss, {
            'bce': float(loss_bce) if isinstance(loss_bce, torch.Tensor) else 0.0,
            'dice': float(loss_dice) if isinstance(loss_dice, torch.Tensor) else 0.0,
            'focal': float(loss_focal) if isinstance(loss_focal, torch.Tensor) else 0.0,
            'total': float(total_loss)
        }
