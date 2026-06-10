
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Dice Loss for binary segmentation.

    Formula: 1 - (2 * intersection + smooth) / (pred + target + smooth)

    Args:
        smooth: Smoothing factor to avoid division by zero (default: 1.0)
        reduction: Reduction method ('mean', 'sum', or 'none')
    """

    def __init__(self, smooth: float = 1.0, reduction: str = "mean") -> None:
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute Dice loss.

        Args:
            pred: Predictions [B, 1, H, W] (logits or probabilities)
            target: Ground truth [B, 1, H, W] (values in [0, 1])

        Returns:
            Loss value
        """
        if pred.min() < 0 or pred.max() > 1:
            pred = torch.sigmoid(pred)

        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)

        intersection = (pred_flat * target_flat).sum(dim=1)
        union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

        # Dice loss = 1 - Dice coefficient
        loss = 1.0 - dice

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class TverskyLoss(nn.Module):
    """
    Tversky Loss - generalization of Dice Loss.

    Allows weighting false positives vs false negatives differently.
    Useful for handling class imbalance.

    Args:
        alpha: Weight for false positives (default: 0.5)
        beta: Weight for false negatives (default: 0.5)
        smooth: Smoothing factor (default: 1.0)

    Note:
        - alpha=beta=0.5: Equivalent to Dice Loss
        - alpha>beta: Penalize false positives more (reduce over-segmentation)
        - alpha<beta: Penalize false negatives more (reduce under-segmentation)
    """

    def __init__(self, alpha: float = 0.5, beta: float = 0.5, smooth: float = 1.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.min() < 0 or pred.max() > 1:
            pred = torch.sigmoid(pred)

        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)

        # True positives, false positives, false negatives
        tp = (pred_flat * target_flat).sum(dim=1)
        fp = (pred_flat * (1 - target_flat)).sum(dim=1)
        fn = ((1 - pred_flat) * target_flat).sum(dim=1)

        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)

        return (1.0 - tversky).mean()


class DiceFocalLoss(nn.Module):
    """
    Combined Dice + Focal Loss for segmentation.

    Combines the benefits of both:
    - Dice: Handles class imbalance, optimizes overlap directly
    - Focal: Focuses on hard examples, improves boundary quality

    Args:
        dice_weight: Weight for Dice loss (default: 0.5)
        focal_weight: Weight for Focal loss (default: 0.5)
        focal_alpha: Focal loss alpha parameter (default: 0.25)
        focal_gamma: Focal loss gamma parameter (default: 2.0)
        smooth: Dice loss smoothing (default: 1.0)
    """

    def __init__(
        self,
        dice_weight: float = 0.5,
        focal_weight: float = 0.5,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        smooth: float = 1.0,
    ) -> None:
        super().__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute combined loss.

        Args:
            pred: Predictions [B, 1, H, W] (logits)
            target: Ground truth [B, 1, H, W]

        Returns:
            Combined loss value
        """
        dice_loss = self._dice_loss(pred, target)

        focal_loss = self._focal_loss(pred, target)

        total_loss = self.dice_weight * dice_loss + self.focal_weight * focal_loss

        return total_loss

    def _dice_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_sigmoid = torch.sigmoid(pred)

        pred_flat = pred_sigmoid.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)

        intersection = (pred_flat * target_flat).sum(dim=1)
        union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return (1.0 - dice).mean()

    def _focal_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_sigmoid = torch.sigmoid(pred)

        ce_loss = F.binary_cross_entropy_with_logits(pred_sigmoid, target, reduction="none")
        p_t = pred_sigmoid * target + (1 - pred_sigmoid) * (1 - target)
        focal_term = (1 - p_t) ** self.focal_gamma

        loss = self.focal_alpha * focal_term * ce_loss

        return loss.mean()


class DiceBCELoss(nn.Module):
    """
    Combined Dice + Binary Cross Entropy Loss.

    Simpler alternative to DiceFocalLoss.

    Args:
        dice_weight: Weight for Dice loss (default: 0.5)
        bce_weight: Weight for BCE loss (default: 0.5)
        smooth: Dice loss smoothing (default: 1.0)
    """

    def __init__(
        self,
        dice_weight: float = 0.5,
        bce_weight: float = 0.5,
        smooth: float = 1.0,
    ) -> None:
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_sigmoid = torch.sigmoid(pred)
        pred_flat = pred_sigmoid.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)

        intersection = (pred_flat * target_flat).sum(dim=1)
        union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = (1.0 - dice).mean()

        bce_loss = F.binary_cross_entropy_with_logits(pred, target)

        return self.dice_weight * dice_loss + self.bce_weight * bce_loss


class SoftDiceLoss(nn.Module):
    """
    Soft Dice Loss with squared terms.

    More stable gradients than standard Dice loss.

    Args:
        smooth: Smoothing factor (default: 1.0)
    """

    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.min() < 0 or pred.max() > 1:
            pred = torch.sigmoid(pred)

        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)

        intersection = (pred_flat * target_flat).sum(dim=1)

        # Use squared terms for more stable gradients
        dice = (2.0 * intersection + self.smooth) / (
            (pred_flat**2).sum(dim=1) + (target_flat**2).sum(dim=1) + self.smooth
        )

        return (1.0 - dice).mean()


class ComboLoss(nn.Module):
    """
    Combination loss: Dice + BCE + Focal.

    Comprehensive loss for challenging segmentation tasks.

    Args:
        dice_weight: Weight for Dice loss (default: 0.4)
        bce_weight: Weight for BCE loss (default: 0.3)
        focal_weight: Weight for Focal loss (default: 0.3)
    """

    def __init__(
        self,
        dice_weight: float = 0.4,
        bce_weight: float = 0.3,
        focal_weight: float = 0.3,
    ) -> None:
        super().__init__()
        self.dice_loss = DiceLoss()
        self.focal_loss = DiceFocalLoss(dice_weight=0.0, focal_weight=1.0)
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.focal_weight = focal_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        dice_loss = self.dice_loss(pred, target)
        bce_loss = F.binary_cross_entropy_with_logits(pred, target)
        focal_loss = self.focal_loss(pred, target)

        return (
            self.dice_weight * dice_loss
            + self.bce_weight * bce_loss
            + self.focal_weight * focal_loss
        )

class WeightedHeatmapMSE(nn.Module):
    def __init__(self, pos_weight: float = 100.0) -> None:
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        w = 1.0 + self.pos_weight * target          # 1 in background, ~pos_weight at peaks
        return (w * (pred - target) ** 2).mean()

def get_loss_function(loss_type: str, **kwargs) -> nn.Module:
    """
    Get loss function by name.

    Args:
        loss_type: Loss function name
        **kwargs: Additional arguments for loss function

    Returns:
        Loss function instance

    Available losses:
        - "dice": Dice Loss
        - "dice_focal": Dice + Focal Loss (recommended)
        - "dice_bce": Dice + BCE Loss
        - "tversky": Tversky Loss
        - "soft_dice": Soft Dice Loss
        - "combo": Combo Loss (Dice + BCE + Focal)
    """
    loss_registry = {
        "dice": DiceLoss,
        "dice_focal": DiceFocalLoss,
        "dice_bce": DiceBCELoss,
        "tversky": TverskyLoss,
        "soft_dice": SoftDiceLoss,
        "combo": ComboLoss,
        "weighted_heatmap_mse": WeightedHeatmapMSE,

    }

    if loss_type not in loss_registry:
        raise ValueError(
            f"Unknown loss type: {loss_type}. "
            f"Available: {list(loss_registry.keys())}"
        )

    return loss_registry[loss_type](**kwargs)
