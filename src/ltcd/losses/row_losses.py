
import torch
import torch.nn as nn
import torch.nn.functional as F


class RowMSELoss(nn.Module):
    """
    Simple MSE loss for row keypoint prediction.

    Args:
        reduction: Reduction method ('mean', 'sum', 'none')
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Compute MSE loss.

        Args:
            pred: Predicted keypoints [B, num_cols, 2]
            target: Ground truth keypoints [B, num_cols, 2]
            mask: Optional validity mask [B, num_cols]

        Returns:
            Dictionary with loss value
        """
        mse = (pred - target) ** 2

        if mask is not None:
            # Apply mask: [B, num_cols] -> [B, num_cols, 1]
            mask = mask.unsqueeze(-1).float()
            mse = mse * mask
            loss = mse.sum() / (mask.sum() * 2 + 1e-8)
        else:
            loss = mse.mean()

        return {"loss": loss, "mse_loss": loss}


class LineDistanceLoss(nn.Module):
    """
    Loss that penalizes deviation from a line defined by first and last points.

    For each row, computes:
    1. Line equation from first point (p0) and last point (pN)
    2. Perpendicular distance from each intermediate point to this line

    This encourages the predicted row to be straight.

    Args:
        reduction: Reduction method ('mean', 'sum')
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        pred: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Compute line distance loss.

        Args:
            pred: Predicted keypoints [B, num_cols, 2]
            mask: Optional validity mask [B, num_cols]

        Returns:
            Dictionary with line distance loss
        """
        B, N, _ = pred.shape

        if N < 2:
            return {"line_loss": torch.tensor(0.0, device=pred.device)}

        p0 = pred[:, 0, :]  # [B, 2]
        pN = pred[:, -1, :]  # [B, 2]

        direction = pN - p0  # [B, 2]

        direction_norm = torch.norm(direction, dim=1, keepdim=True) + 1e-8
        direction = direction / direction_norm  # [B, 2]

        # Perpendicular direction (rotate 90 degrees)
        perp = torch.stack([-direction[:, 1], direction[:, 0]], dim=1)  # [B, 2]

        # Compute perpendicular distance for each intermediate point
        # Vector from p0 to each point: [B, N, 2]
        vectors = pred - p0.unsqueeze(1)

        # Perpendicular distance = |dot(vector, perp)|
        # perp: [B, 2] -> [B, 1, 2]
        perp_distances = torch.abs((vectors * perp.unsqueeze(1)).sum(dim=2))  # [B, N]

        if mask is not None:
            perp_distances = perp_distances * mask.float()
            line_loss = perp_distances.sum() / (mask.float().sum() + 1e-8)
        else:
            line_loss = perp_distances.mean()

        return {"line_loss": line_loss}


class RowPredictionLoss(nn.Module):
    """
    Combined loss for row prediction: MSE + Line Distance.

    Total loss = mse_weight * MSE + line_weight * LineDistance

    The line distance loss encourages predictions to lie on a straight line
    defined by the first and last predicted points.

    Args:
        mse_weight: Weight for MSE loss (default: 1.0)
        line_weight: Weight for line distance loss (default: 0.1)
        use_gt_line: If True, compute line from GT first/last points instead of predicted
    """

    def __init__(
        self,
        mse_weight: float = 1.0,
        line_weight: float = 0.1,
        use_gt_line: bool = False,
    ) -> None:
        super().__init__()
        self.mse_weight = mse_weight
        self.line_weight = line_weight
        self.use_gt_line = use_gt_line

    def _compute_line_distance(
        self,
        points: torch.Tensor,
        line_p0: torch.Tensor,
        line_pN: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute perpendicular distance from points to line defined by line_p0 and line_pN.

        Args:
            points: Points to measure [B, N, 2]
            line_p0: First point of line [B, 2]
            line_pN: Last point of line [B, 2]
            mask: Optional validity mask [B, N]

        Returns:
            Mean perpendicular distance
        """
        B, N, _ = points.shape

        direction = line_pN - line_p0  # [B, 2]
        direction_norm = torch.norm(direction, dim=1, keepdim=True) + 1e-8
        direction = direction / direction_norm

        perp = torch.stack([-direction[:, 1], direction[:, 0]], dim=1)  # [B, 2]

        vectors = points - line_p0.unsqueeze(1)  # [B, N, 2]

        perp_distances = torch.abs((vectors * perp.unsqueeze(1)).sum(dim=2))  # [B, N]

        if mask is not None:
            perp_distances = perp_distances * mask.float()
            return perp_distances.sum() / (mask.float().sum() + 1e-8)
        else:
            return perp_distances.mean()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Compute combined loss.

        Args:
            pred: Predicted keypoints [B, num_cols, 2]
            target: Ground truth keypoints [B, num_cols, 2]
            mask: Optional validity mask [B, num_cols]

        Returns:
            Dictionary with total loss and components
        """
        mse = (pred - target) ** 2

        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).float()
            mse = mse * mask_expanded
            mse_loss = mse.sum() / (mask_expanded.sum() + 1e-8)
        else:
            mse_loss = mse.mean()

        if self.use_gt_line:
            # Use ground truth first/last points to define the line
            line_p0 = target[:, 0, :]
            line_pN = target[:, -1, :]
        else:
            # Use predicted first/last points to define the line
            line_p0 = pred[:, 0, :]
            line_pN = pred[:, -1, :]

        line_loss = self._compute_line_distance(pred, line_p0, line_pN, mask)

        total_loss = self.mse_weight * mse_loss + self.line_weight * line_loss

        return {
            "loss": total_loss,
            "mse_loss": mse_loss,
            "line_loss": line_loss,
        }


class RowPredictionLossV2(nn.Module):
    """
    Enhanced row prediction loss with additional constraints.

    Loss components:
    1. MSE: Point-wise error
    2. Line distance: Distance from points to line (first -> last)
    3. Spacing regularity: Encourages uniform spacing between consecutive points

    Args:
        mse_weight: Weight for MSE loss (default: 1.0)
        line_weight: Weight for line distance loss (default: 0.1)
        spacing_weight: Weight for spacing regularity loss (default: 0.05)
        use_gt_line: If True, use GT points to define the line
    """

    def __init__(
        self,
        mse_weight: float = 1.0,
        line_weight: float = 0.1,
        spacing_weight: float = 0.05,
        use_gt_line: bool = False,
    ) -> None:
        super().__init__()
        self.mse_weight = mse_weight
        self.line_weight = line_weight
        self.spacing_weight = spacing_weight
        self.use_gt_line = use_gt_line

    def _compute_line_distance(
        self,
        points: torch.Tensor,
        line_p0: torch.Tensor,
        line_pN: torch.Tensor,
    ) -> torch.Tensor:
        """Compute perpendicular distance from points to line."""
        direction = line_pN - line_p0
        direction_norm = torch.norm(direction, dim=1, keepdim=True) + 1e-8
        direction = direction / direction_norm

        perp = torch.stack([-direction[:, 1], direction[:, 0]], dim=1)
        vectors = points - line_p0.unsqueeze(1)
        perp_distances = torch.abs((vectors * perp.unsqueeze(1)).sum(dim=2))

        return perp_distances.mean()

    def _compute_spacing_loss(self, points: torch.Tensor) -> torch.Tensor:
        B, N, _ = points.shape

        if N < 3:
            return torch.tensor(0.0, device=points.device)

        # Compute distances between consecutive points
        diffs = points[:, 1:, :] - points[:, :-1, :]  # [B, N-1, 2]
        distances = torch.norm(diffs, dim=2)  # [B, N-1]

        # Compute variance of distances (want low variance = uniform spacing)
        mean_dist = distances.mean(dim=1, keepdim=True)  # [B, 1]
        spacing_variance = ((distances - mean_dist) ** 2).mean()

        return spacing_variance

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Compute combined loss.

        Args:
            pred: Predicted keypoints [B, num_cols, 2]
            target: Ground truth keypoints [B, num_cols, 2]
            mask: Optional validity mask [B, num_cols]

        Returns:
            Dictionary with total loss and components
        """
        mse = (pred - target) ** 2

        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).float()
            mse = mse * mask_expanded
            mse_loss = mse.sum() / (mask_expanded.sum() + 1e-8)
        else:
            mse_loss = mse.mean()

        if self.use_gt_line:
            line_p0 = target[:, 0, :]
            line_pN = target[:, -1, :]
        else:
            line_p0 = pred[:, 0, :]
            line_pN = pred[:, -1, :]

        line_loss = self._compute_line_distance(pred, line_p0, line_pN)

        spacing_loss = self._compute_spacing_loss(pred)

        total_loss = (
            self.mse_weight * mse_loss
            + self.line_weight * line_loss
            + self.spacing_weight * spacing_loss
        )

        return {
            "loss": total_loss,
            "mse_loss": mse_loss,
            "line_loss": line_loss,
            "spacing_loss": spacing_loss,
        }


def get_row_loss_function(
    loss_type: str = "mse",
    **kwargs,
) -> nn.Module:
    """
    Get row prediction loss function by name.

    Args:
        loss_type: Loss type name
        **kwargs: Additional arguments for the loss function

    Returns:
        Loss function instance

    Available losses:
        - "mse": Simple MSE loss
        - "line": Line distance loss only
        - "combined": MSE + Line distance
        - "full": MSE + Line distance + Spacing regularity
    """
    loss_registry = {
        "mse": RowMSELoss,
        "line": LineDistanceLoss,
        "combined": RowPredictionLoss,
        "full": RowPredictionLossV2,
    }

    if loss_type not in loss_registry:
        raise ValueError(
            f"Unknown loss type: {loss_type}. "
            f"Available: {list(loss_registry.keys())}"
        )

    return loss_registry[loss_type](**kwargs)
