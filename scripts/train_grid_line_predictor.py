
import argparse
import shutil
from datetime import datetime
from pathlib import Path

import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ltcd.archs.grid_line_predictor import (    UnifiedGridLinePredictor,    points_to_normal_form,
)
from ltcd.archs.yolo_keypoint_predictor import (
    YOLOKeypointPredictor,
    YOLOKeypointPredictorSmall,
    YOLOKeypointPredictorMedium,
    YOLOKeypointLoss,
    assign_keypoints_to_grid,
)
from ltcd.datasets.keypoint_dataset import KeypointDatasetSimple
from ltcd.training import EMA, EarlyStopping, GridAugmentedDataset


class GridLineLoss(nn.Module):
    """
    Loss for grid line prediction.

    Components:
    1. Point MSE/Smooth L1: Direct coordinate error
    2. Line consistency: Points should lie on predicted line
    3. Normal form loss: Line parameters should match
    4. Coordinate loss: Supervise shared x/y coordinate predictions
    5. Row/Column alignment: Points should be aligned
    6. Ordering loss: Lines should be in order
    """

    def __init__(
        self,
        horizontal_weight: float = 1.0,
        vertical_weight: float = 1.0,
        line_weight: float = 0.1,
        normal_weight: float = 0.1,
        coord_weight: float = 1.0,
        row_align_weight: float = 0.0,
        col_align_weight: float = 0.0,
        ordering_weight: float = 0.0,
        use_smooth_l1: bool = False,
        smooth_l1_beta: float = 0.1,
    ) -> None:
        super().__init__()
        self.horizontal_weight = horizontal_weight
        self.vertical_weight = vertical_weight
        self.line_weight = line_weight
        self.normal_weight = normal_weight
        self.coord_weight = coord_weight
        self.row_align_weight = row_align_weight
        self.col_align_weight = col_align_weight
        self.ordering_weight = ordering_weight
        self.use_smooth_l1 = use_smooth_l1
        self.smooth_l1_beta = smooth_l1_beta

    def _compute_point_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.use_smooth_l1:
            return F.smooth_l1_loss(pred, target, beta=self.smooth_l1_beta)
        return F.mse_loss(pred, target)

    def _compute_line_residual_loss(self, lines: torch.Tensor) -> torch.Tensor:
        B, num_lines, num_points, _ = lines.shape

        if num_points < 2:
            return torch.tensor(0.0, device=lines.device)

        total_loss = torch.tensor(0.0, device=lines.device)

        for line_idx in range(num_lines):
            line = lines[:, line_idx, :, :]
            cos_t, sin_t, rho = points_to_normal_form(line)

            x, y = line[:, :, 0], line[:, :, 1]
            residuals = torch.abs(
                x * cos_t.unsqueeze(1) + y * sin_t.unsqueeze(1) - rho.unsqueeze(1)
            )
            total_loss = total_loss + residuals.mean()

        return total_loss / num_lines

    def _compute_normal_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compare line parameters."""
        pred_cos, pred_sin, pred_rho = points_to_normal_form(pred)
        target_cos, target_sin, target_rho = points_to_normal_form(target)

        cos_loss = F.mse_loss(pred_cos, target_cos)
        sin_loss = F.mse_loss(pred_sin, target_sin)
        rho_loss = F.mse_loss(pred_rho, target_rho)

        return cos_loss + sin_loss + rho_loss

    def forward(
        self,
        pred_horizontal: torch.Tensor | None,
        pred_vertical: torch.Tensor | None,
        target_horizontal: torch.Tensor,
        target_vertical: torch.Tensor,
        pred_x_coords: torch.Tensor | None = None,
        pred_y_coords: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Compute combined loss.

        Args:
            pred_horizontal: [B, num_rows, num_samples, 2] or None
            pred_vertical: [B, num_cols, num_samples, 2] or None
            target_horizontal: [B, num_rows, num_cols, 2]
            target_vertical: [B, num_cols, num_rows, 2]
            pred_x_coords: [B, num_cols] - predicted column x-positions
            pred_y_coords: [B, num_rows] - predicted row y-positions
        """
        total_loss = torch.tensor(0.0, device=target_horizontal.device)
        losses = {}

        if pred_x_coords is not None:
            target_x_coords = target_horizontal[:, 0, :, 0]
            x_coord_loss = F.mse_loss(pred_x_coords, target_x_coords)
            total_loss = total_loss + self.coord_weight * x_coord_loss
            losses["x_coord_loss"] = x_coord_loss

        if pred_y_coords is not None:
            target_y_coords = target_vertical[:, 0, :, 1]
            y_coord_loss = F.mse_loss(pred_y_coords, target_y_coords)
            total_loss = total_loss + self.coord_weight * y_coord_loss
            losses["y_coord_loss"] = y_coord_loss

        if pred_horizontal is not None:
            B, num_rows, num_samples, _ = pred_horizontal.shape
            _, _, target_samples, _ = target_horizontal.shape

            if num_samples == target_samples:
                target_h = target_horizontal
            else:
                target_h = self._interpolate_target(target_horizontal, num_samples)

            h_point_loss = self._compute_point_loss(pred_horizontal, target_h)
            h_line_loss = self._compute_line_residual_loss(pred_horizontal)
            h_normal_loss = self._compute_normal_loss(pred_horizontal, target_h)

            h_total = h_point_loss + self.line_weight * h_line_loss + self.normal_weight * h_normal_loss
            total_loss = total_loss + self.horizontal_weight * h_total

            losses["h_point_loss"] = h_point_loss
            losses["h_line_loss"] = h_line_loss
            losses["h_normal_loss"] = h_normal_loss

            # Row alignment (y variance)
            if self.row_align_weight > 0:
                pred_y = pred_horizontal[..., 1]
                row_y_var = pred_y.var(dim=2).mean()
                total_loss = total_loss + self.row_align_weight * row_y_var
                losses["row_align_loss"] = row_y_var

        if pred_vertical is not None:
            B, num_cols, num_samples, _ = pred_vertical.shape
            _, _, target_samples, _ = target_vertical.shape

            if num_samples == target_samples:
                target_v = target_vertical
            else:
                target_v = self._interpolate_target(target_vertical, num_samples)

            v_point_loss = self._compute_point_loss(pred_vertical, target_v)
            v_line_loss = self._compute_line_residual_loss(pred_vertical)
            v_normal_loss = self._compute_normal_loss(pred_vertical, target_v)

            v_total = v_point_loss + self.line_weight * v_line_loss + self.normal_weight * v_normal_loss
            total_loss = total_loss + self.vertical_weight * v_total

            losses["v_point_loss"] = v_point_loss
            losses["v_line_loss"] = v_line_loss
            losses["v_normal_loss"] = v_normal_loss

            # Column alignment (x variance)
            if self.col_align_weight > 0:
                pred_x = pred_vertical[..., 0]
                col_x_var = pred_x.var(dim=2).mean()
                total_loss = total_loss + self.col_align_weight * col_x_var
                losses["col_align_loss"] = col_x_var

        if self.ordering_weight > 0 and pred_horizontal is not None and pred_vertical is not None:
            row_y_mean = pred_horizontal[..., 1].mean(dim=2)
            row_order_violation = F.relu(row_y_mean[:, :-1] - row_y_mean[:, 1:])
            row_order_loss = row_order_violation.mean()

            col_x_mean = pred_vertical[..., 0].mean(dim=2)
            col_order_violation = F.relu(col_x_mean[:, :-1] - col_x_mean[:, 1:])
            col_order_loss = col_order_violation.mean()

            ordering_loss = row_order_loss + col_order_loss
            total_loss = total_loss + self.ordering_weight * ordering_loss
            losses["ordering_loss"] = ordering_loss

        losses["loss"] = total_loss
        return losses

    def _interpolate_target(
        self,
        target: torch.Tensor,
        num_samples: int,
    ) -> torch.Tensor:
        """Interpolate target points to match prediction sample count."""
        B, num_lines, num_points, _ = target.shape

        if num_points == num_samples:
            return target

        t_src = torch.linspace(0, 1, num_points, device=target.device)
        t_dst = torch.linspace(0, 1, num_samples, device=target.device)

        target_flat = target.view(B * num_lines, num_points, 2)
        x = target_flat[:, :, 0]
        y = target_flat[:, :, 1]

        indices = torch.searchsorted(t_src, t_dst).clamp(1, num_points - 1)
        t0 = t_src[indices - 1]
        t1 = t_src[indices]
        weight = ((t_dst - t0) / (t1 - t0 + 1e-8)).unsqueeze(0)

        x0 = x[:, indices - 1]
        x1 = x[:, indices]
        y0 = y[:, indices - 1]
        y1 = y[:, indices]

        x_interp = x0 + weight * (x1 - x0)
        y_interp = y0 + weight * (y1 - y0)

        result = torch.stack([x_interp, y_interp], dim=-1)
        return result.view(B, num_lines, num_samples, 2)


def compute_metrics(
    pred_horizontal: torch.Tensor,
    pred_vertical: torch.Tensor | None,
    target_horizontal: torch.Tensor,
    target_vertical: torch.Tensor | None,
    image_size: tuple[int, int] = (512, 512),
) -> dict[str, float]:
    """
    Compute evaluation metrics from line predictions.

    Uses horizontal line predictions for primary metrics.
    """
    with torch.no_grad():
        pred_h_px = pred_horizontal.clone()
        pred_h_px[..., 0] = (pred_h_px[..., 0] + 1) / 2 * image_size[1]
        pred_h_px[..., 1] = (pred_h_px[..., 1] + 1) / 2 * image_size[0]

        target_h_px = target_horizontal.clone()
        target_h_px[..., 0] = (target_h_px[..., 0] + 1) / 2 * image_size[1]
        target_h_px[..., 1] = (target_h_px[..., 1] + 1) / 2 * image_size[0]

        pixel_error_h = torch.sqrt(((pred_h_px - target_h_px) ** 2).sum(dim=-1))

        if pred_vertical is not None and target_vertical is not None:
            pred_v_px = pred_vertical.clone()
            pred_v_px[..., 0] = (pred_v_px[..., 0] + 1) / 2 * image_size[1]
            pred_v_px[..., 1] = (pred_v_px[..., 1] + 1) / 2 * image_size[0]

            target_v_px = target_vertical.clone()
            target_v_px[..., 0] = (target_v_px[..., 0] + 1) / 2 * image_size[1]
            target_v_px[..., 1] = (target_v_px[..., 1] + 1) / 2 * image_size[0]

            pixel_error_v = torch.sqrt(((pred_v_px - target_v_px) ** 2).sum(dim=-1))
            pixel_error = torch.cat([pixel_error_h.flatten(), pixel_error_v.flatten()])
        else:
            pixel_error = pixel_error_h

        mean_pixel_error = pixel_error.mean().item()
        max_pixel_error = pixel_error.max().item()

        diagonal = np.sqrt(image_size[0] ** 2 + image_size[1] ** 2)
        pck_5 = (pixel_error < 0.05 * diagonal).float().mean().item()
        pck_10 = (pixel_error < 0.10 * diagonal).float().mean().item()

        pck_5px = (pixel_error < 5).float().mean().item()
        pck_10px = (pixel_error < 10).float().mean().item()
        pck_20px = (pixel_error < 20).float().mean().item()

        return {
            "pixel_error": mean_pixel_error,
            "max_pixel_error": max_pixel_error,
            "pck_0.05": pck_5,
            "pck_0.10": pck_10,
            "pck_5px": pck_5px,
            "pck_10px": pck_10px,
            "pck_20px": pck_20px,
        }


def denormalize_points(
    points: np.ndarray,
    image_size: tuple[int, int],
) -> np.ndarray:
    """Convert points from [-1, 1] to pixel coordinates."""
    points = points.copy()
    points = (points + 1) / 2
    points[..., 0] *= image_size[1]
    points[..., 1] *= image_size[0]
    return points


def draw_lines_comparison(
    image: Image.Image,
    pred_horizontal: np.ndarray | None,
    pred_vertical: np.ndarray | None,
    target_horizontal: np.ndarray,
    target_vertical: np.ndarray,
    point_radius: int = 2,
    line_width: int = 1,
) -> Image.Image:
    """Draw both predicted and target lines on image for comparison."""
    image = image.copy()
    draw = ImageDraw.Draw(image)

    # Draw target horizontal lines (green)
    if target_horizontal is not None:
        num_rows = target_horizontal.shape[0]
        for row in range(num_rows):
            points = target_horizontal[row]
            for i in range(len(points) - 1):
                x1, y1 = points[i]
                x2, y2 = points[i + 1]
                draw.line([(x1, y1), (x2, y2)], fill=(0, 180, 0), width=line_width)

    # Draw target vertical lines (green)
    if target_vertical is not None:
        num_cols = target_vertical.shape[0]
        for col in range(num_cols):
            points = target_vertical[col]
            for i in range(len(points) - 1):
                x1, y1 = points[i]
                x2, y2 = points[i + 1]
                draw.line([(x1, y1), (x2, y2)], fill=(0, 180, 0), width=line_width)

    # Draw predicted horizontal lines (red)
    if pred_horizontal is not None:
        num_rows = pred_horizontal.shape[0]
        for row in range(num_rows):
            points = pred_horizontal[row]
            for i in range(len(points) - 1):
                x1, y1 = points[i]
                x2, y2 = points[i + 1]
                draw.line([(x1, y1), (x2, y2)], fill=(255, 50, 50), width=line_width)

    # Draw predicted vertical lines (red)
    if pred_vertical is not None:
        num_cols = pred_vertical.shape[0]
        for col in range(num_cols):
            points = pred_vertical[col]
            for i in range(len(points) - 1):
                x1, y1 = points[i]
                x2, y2 = points[i + 1]
                draw.line([(x1, y1), (x2, y2)], fill=(255, 50, 50), width=line_width)

    # Draw target points (green circles)
    if target_horizontal is not None:
        for row in range(target_horizontal.shape[0]):
            for point in target_horizontal[row]:
                x, y = point
                draw.ellipse(
                    [x - point_radius, y - point_radius, x + point_radius, y + point_radius],
                    fill=(0, 200, 0),
                    outline=(0, 100, 0),
                )

    # Draw predicted points (red circles)
    if pred_horizontal is not None:
        for row in range(pred_horizontal.shape[0]):
            for point in pred_horizontal[row]:
                x, y = point
                draw.ellipse(
                    [x - point_radius, y - point_radius, x + point_radius, y + point_radius],
                    fill=(255, 50, 50),
                    outline=(150, 0, 0),
                )

    return image


def draw_keypoints_comparison(
    image: Image.Image,
    pred_kps: np.ndarray | None,    # [K_pred, 2] in pixel coords
    gt_kps: np.ndarray,             # [K_gt, 2] in pixel coords
    point_radius: int = 3,
) -> Image.Image:
    """Plot GT (green) and predicted (red) keypoints as dots; no line connections."""
    image = image.copy()
    draw = ImageDraw.Draw(image)

    for x, y in gt_kps:
        draw.ellipse(
            [x - point_radius, y - point_radius, x + point_radius, y + point_radius],
            fill=(0, 200, 0), outline=(0, 100, 0),
        )

    if pred_kps is not None:
        for x, y in pred_kps:
            draw.ellipse(
                [x - point_radius, y - point_radius, x + point_radius, y + point_radius],
                fill=(255, 50, 50), outline=(150, 0, 0),
            )

    return image


@torch.no_grad()
def visualize_samples(
    model: nn.Module,
    dataset: Dataset,
    output_dir: Path,
    epoch: int,
    num_samples: int = 4,
    image_size: tuple[int, int] = (512, 512),
    device: str = "cuda",
    split: str = "train",
    conf_threshold: float = 0.5,
) -> None:
    """Visualize model predictions on sample images."""
    model.eval()

    vis_dir = output_dir / "images" / split
    vis_dir.mkdir(parents=True, exist_ok=True)

    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    is_yolo = isinstance(model, (YOLOKeypointPredictor, YOLOKeypointPredictorSmall, YOLOKeypointPredictorMedium))

    for i, idx in enumerate(indices):
        sample = dataset[idx]
        image_tensor = sample["image"].unsqueeze(0).to(device)
        target_h = sample["horizontal"].numpy()
        target_v = sample["vertical"].numpy()

        if is_yolo:
            kps_norm = model.decode(image_tensor, conf_threshold=conf_threshold)[0].cpu().numpy()
            pred_kps_px = denormalize_points(kps_norm, image_size) if kps_norm.size else None
            gt_kps_px = denormalize_points(target_h.reshape(-1, 2), image_size)

            image_denorm = sample["image"] * std + mean
            image_np = (image_denorm.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            vis_image = draw_keypoints_comparison(Image.fromarray(image_np), pred_kps_px, gt_kps_px)

            kept = 0 if pred_kps_px is None else len(pred_kps_px)
            vis_path = vis_dir / f"epoch_{epoch:04d}_sample_{i:02d}_k{kept}.png"
            vis_image.save(vis_path)
            continue

        output = model(image_tensor)
        pred_h = output.get("horizontal")
        pred_v = output.get("vertical")

        pred_h_np = pred_h[0].cpu().numpy() if pred_h is not None else None
        pred_v_np = pred_v[0].cpu().numpy() if pred_v is not None else None

        if pred_h_np is not None:
            pred_h_np = denormalize_points(pred_h_np, image_size)
        if pred_v_np is not None:
            pred_v_np = denormalize_points(pred_v_np, image_size)
        target_h_px = denormalize_points(target_h, image_size)
        target_v_px = denormalize_points(target_v, image_size)

        image_denorm = sample["image"] * std + mean
        image_denorm = torch.clamp(image_denorm, 0, 1)
        image_np = (image_denorm.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        image_pil = Image.fromarray(image_np)

        vis_image = draw_lines_comparison(
            image_pil, pred_h_np, pred_v_np, target_h_px, target_v_px
        )

        vis_path = vis_dir / f"epoch_{epoch:04d}_sample_{i:02d}.png"
        vis_image.save(vis_path)

    model.train()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    scaler: torch.amp.GradScaler | None,
    clip_grad: float = 1.0,
    ema: EMA | None = None,
    image_size: tuple[int, int] = (512, 512),
    teacher_ratio: float = 0.0,
) -> dict[str, float]:
    """Train for one epoch."""
    model.train()
    running_metrics = {}
    num_batches = 0

    for batch in tqdm(loader, desc="Training"):
        images = batch["image"].to(device)
        target_h = batch["grid"].to(device)
        target_v = target_h.permute(0, 2, 1, 3).contiguous()

        optimizer.zero_grad()

        with torch.amp.autocast(device_type="cuda", enabled=(scaler is not None)):
            if isinstance(model, (YOLOKeypointPredictor, YOLOKeypointPredictorSmall, YOLOKeypointPredictorMedium)):
                output = model(images)
                B, grid_h, grid_w = output["conf_logit"].shape
                flat_gt = target_h.view(B, -1, 2)
                coord_tgt, conf_tgt = assign_keypoints_to_grid(
                    flat_gt, grid_h, grid_w
                )
                loss_dict = criterion(
                    output["grid"], output["conf_logit"], coord_tgt, conf_tgt
                )
                # Skip per-cell pixel/PCK metrics: dense 64×64 preds can't be compared
                # to grid-shaped targets element-wise. Loss components track progress.
                pred_h = None
                pred_v = None
            elif isinstance(model) and teacher_ratio > 0:
                output = model(images, teacher_horizontal=target_h,
                               teacher_vertical=target_v, teacher_ratio=teacher_ratio)
                pred_h = output.get("horizontal")
                pred_v = output.get("vertical")
                loss_dict = criterion(pred_h, pred_v, target_h, target_v, None, None)
            else:
                output = model(images)
                pred_h = output.get("horizontal")
                pred_v = output.get("vertical")
                pred_x_coords = output.get("x_coords")
                pred_y_coords = output.get("y_coords")
                loss_dict = criterion(
                    pred_h, pred_v, target_h, target_v,
                    pred_x_coords, pred_y_coords,
                )
            loss = loss_dict["loss"]

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)
            optimizer.step()

        if ema is not None:
            ema.update()

        for k, v in loss_dict.items():
            if k not in running_metrics:
                running_metrics[k] = 0.0
            running_metrics[k] += v.item()

        if pred_h is not None:
            metrics = compute_metrics(
                pred_h.detach(), pred_v.detach() if pred_v is not None else pred_h.detach(),
                target_h, target_v, image_size
            )
            for k, v in metrics.items():
                if k not in running_metrics:
                    running_metrics[k] = 0.0
                running_metrics[k] += v

        num_batches += 1

    return {k: v / num_batches for k, v in running_metrics.items()}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
    image_size: tuple[int, int] = (512, 512),
) -> dict[str, float]:
    """Evaluate the model."""
    model.eval()
    running_metrics = {}
    num_batches = 0

    for batch in tqdm(loader, desc="Evaluating"):
        images = batch["image"].to(device)
        target_h = batch["grid"].to(device)
        target_v = target_h.permute(0, 2, 1, 3).contiguous()

        if isinstance(model, (YOLOKeypointPredictor, YOLOKeypointPredictorSmall, YOLOKeypointPredictorMedium)):
            output = model(images)
            B, grid_h, grid_w = output["conf_logit"].shape
            flat_gt = target_h.view(B, -1, 2)
            coord_tgt, conf_tgt = assign_keypoints_to_grid(
                flat_gt, grid_h, grid_w
            )
            loss_dict = criterion(
                output["grid"], output["conf_logit"], coord_tgt, conf_tgt
            )
            pred_h = None
            pred_v = None
        else:
            output = model(images)
            pred_h = output.get("horizontal")
            pred_v = output.get("vertical")
            pred_x_coords = output.get("x_coords")
            pred_y_coords = output.get("y_coords")
            loss_dict = criterion(
                pred_h, pred_v, target_h, target_v,
                pred_x_coords, pred_y_coords,
            )

        for k, v in loss_dict.items():
            if k not in running_metrics:
                running_metrics[k] = 0.0
            running_metrics[k] += v.item()

        if pred_h is not None:
            metrics = compute_metrics(
                pred_h, pred_v if pred_v is not None else pred_h,
                target_h, target_v, image_size
            )
            for k, v in metrics.items():
                if k not in running_metrics:
                    running_metrics[k] = 0.0
                running_metrics[k] += v

        num_batches += 1

    return {k: v / num_batches for k, v in running_metrics.items()}


def create_model(cfg) -> nn.Module:
    model_type = cfg.params.model_type
    p = cfg.params

    if model_type == "unified":
        return UnifiedGridLinePredictor(
            hidden_size=p.hidden_size,
            num_rows=p.num_rows,
            num_cols=p.num_cols,
            num_samples=p.get("num_samples", 10),
            num_lstm_layers=p.num_lstm_layers,
            dropout=p.dropout,
            use_pretrained=p.use_pretrained,
        )
    elif model_type == "yolo_keypoint":
        return YOLOKeypointPredictor(
            backbone_name=p.get("backbone_name", "resnet34"),
            hidden_size=p.hidden_size,
            use_pretrained=p.use_pretrained,
        )
    elif model_type == "yolo_keypoint_small":
        return YOLOKeypointPredictorSmall(
            hidden_size=p.hidden_size,
            use_pretrained=p.use_pretrained,
        )
    elif model_type == "yolo_keypoint_medium":
        return YOLOKeypointPredictorMedium(
            hidden_size=p.hidden_size,
            use_pretrained=p.use_pretrained,
        )
    else:  # two_head
        return TwoHeadGridLinePredictor(
            hidden_size=p.hidden_size,
            num_rows=p.num_rows,
            num_cols=p.num_cols,
            num_row_samples=p.num_row_samples,
            num_col_samples=p.num_col_samples,
            num_lstm_layers=p.num_lstm_layers,
            dropout=p.dropout,
            use_pretrained=p.use_pretrained,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    p = cfg.params

    mlflow.set_experiment(cfg.experiment_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    dataset_dir = Path(p.dataset_dir)
    train_dir = dataset_dir / "train"
    test_dir = dataset_dir / "test"

    print("\nLoading datasets...")
    image_size = tuple(p.image_size)

    train_base = KeypointDatasetSimple(
        train_dir,
        image_size=image_size,
        num_keypoints=p.num_keypoints,
    )
    test_base = KeypointDatasetSimple(
        test_dir,
        image_size=image_size,
        num_keypoints=p.num_keypoints,
    )

    aug_cfg = p.get("augmentation", {})

    train_dataset = GridAugmentedDataset(
        train_base,
        num_rows=p.num_rows,
        num_cols=p.num_cols,
        augment=p.get("augment", True),
        flip_prob=aug_cfg.get("flip_prob", 0.5),
        rotation_range=aug_cfg.get("rotation_range", 5.0),
        scale_range=tuple(aug_cfg.get("scale_range", [0.95, 1.05])),
    )
    test_dataset = GridAugmentedDataset(
        test_base,
        num_rows=p.num_rows,
        num_cols=p.num_cols,
        augment=False,
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    num_workers = p.num_workers
    train_loader = DataLoader(
        train_dataset,
        batch_size=p.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=p.batch_size,
        shuffle=False,
        num_workers=max(1, num_workers // 2),
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    model_type = p.model_type
    print(f"\nCreating model: {model_type}")
    model = create_model(cfg)
    model = model.to(device)

    num_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print(f"Model parameters: {num_params:,}")

    with mlflow.start_run(run_name=cfg.run_name):
        mlflow.log_params(dict(p))
        mlflow.log_param("num_parameters", num_params)

        if isinstance(model, (YOLOKeypointPredictor, YOLOKeypointPredictorSmall, YOLOKeypointPredictorMedium)):
            criterion = YOLOKeypointLoss(
                coord_weight=p.get("coord_weight_yolo", 5.0),
                conf_weight=p.get("conf_weight", 1.0),
                pos_weight=p.get("pos_weight", 10.0),
            )
        else:
            criterion = GridLineLoss(
                horizontal_weight=p.horizontal_weight,
                vertical_weight=p.vertical_weight,
                line_weight=p.line_weight,
                normal_weight=p.normal_weight,
                coord_weight=p.get("coord_weight", 1.0),
                row_align_weight=p.get("row_align_weight", 0.0),
                col_align_weight=p.get("col_align_weight", 0.0),
                ordering_weight=p.get("ordering_weight", 0.0),
                use_smooth_l1=p.get("use_smooth_l1", False),
                smooth_l1_beta=p.get("smooth_l1_beta", 0.1),
            )

        backbone_params = []
        other_params = []
        for name, param in model.named_parameters():
            if "backbone" in name:
                backbone_params.append(param)
            else:
                other_params.append(param)

        optimizer = torch.optim.AdamW([
            {"params": backbone_params, "lr": p.lr * 0.1},
            {"params": other_params, "lr": p.lr},
        ], weight_decay=p.weight_decay)

        warmup_epochs = p.get("warmup_epochs", 5)

        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            else:
                progress = (epoch - warmup_epochs) / (p.epochs - warmup_epochs)
                return 0.5 * (1 + np.cos(np.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        scaler = torch.amp.GradScaler() if device == "cuda" else None

        ema = None
        if p.get("use_ema", False):
            ema_decay = p.get("ema_decay", 0.999)
            ema = EMA(model, decay=ema_decay)
            print(f"Using EMA with decay={ema_decay}")

        early_stopping = None
        if p.get("early_stopping", False) and p.get("early_stopping_patience", 0) > 0:
            patience = p.early_stopping_patience
            min_delta = p.get("early_stopping_min_delta", 0.0)
            early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)
            print(f"Early stopping enabled with patience={patience}")

        date_str = datetime.now().strftime("%Y_%m_%d")
        output_dir = Path("checkpoints") / cfg.run_name / date_str
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {output_dir}")

        config_copy_path = output_dir / "config.yaml"
        shutil.copy(args.config, config_copy_path)
        print(f"Config saved to: {config_copy_path}")

        start_epoch = 0
        best_val_loss = float("inf")
        best_pixel_error = float("inf")

        if args.resume:
            print(f"Resuming from {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if "scheduler_state_dict" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            best_val_loss = checkpoint.get("val_loss", float("inf"))
            best_pixel_error = checkpoint.get("best_pixel_error", float("inf"))
            if ema is not None:
                ema._register()

        print("\n" + "=" * 60)
        print("Starting training")
        print("=" * 60)

        for epoch in range(start_epoch, p.epochs):
            print(f"\nEpoch [{epoch + 1}/{p.epochs}]")

            train_metrics = train_one_epoch(
                model, train_loader, criterion, optimizer, device, scaler,
                clip_grad=p.get("clip_grad", 1.0),
                ema=ema,
                image_size=image_size,
                teacher_ratio=p.get("teacher_forcing_ratio", 0.0),
            )

            if ema is not None:
                ema.apply_shadow()
            val_metrics = evaluate(model, test_loader, criterion, device, image_size)
            if ema is not None:
                ema.restore()

            scheduler.step()

            current_lr = optimizer.param_groups[1]["lr"]
            log_metrics = {"learning_rate": current_lr}
            for k, v in train_metrics.items():
                log_metrics[f"train_{k}"] = v
            for k, v in val_metrics.items():
                log_metrics[f"val_{k}"] = v
            mlflow.log_metrics(log_metrics, step=epoch)

            print(f"Train Loss: {train_metrics['loss']:.6f} | Val Loss: {val_metrics['loss']:.6f}")
            if "h_point_loss" in train_metrics:
                print(f"  H Point: {train_metrics['h_point_loss']:.6f} -> {val_metrics['h_point_loss']:.6f}")
            if "v_point_loss" in train_metrics:
                print(f"  V Point: {train_metrics['v_point_loss']:.6f} -> {val_metrics['v_point_loss']:.6f}")
            if "pixel_error" in train_metrics:
                print(f"  Pixel Error: {train_metrics['pixel_error']:.2f}px -> {val_metrics['pixel_error']:.2f}px")
            if "pck_10px" in train_metrics:
                print(f"  pck_10px: {train_metrics['pck_10px']:.3f} -> {val_metrics['pck_10px']:.3f}")
            print(f"  LR: {current_lr:.6f}")

            visualize_every = p.get("visualize_every_n_epochs", 0)
            num_vis_samples = p.get("num_vis_samples", 4)
            if visualize_every > 0 and (epoch + 1) % visualize_every == 0:
                print(f"  Saving visualizations...")
                if ema is not None:
                    ema.apply_shadow()
                visualize_samples(
                    model, train_dataset, output_dir, epoch + 1,
                    num_samples=num_vis_samples, image_size=image_size,
                    device=device, split="train"
                )
                visualize_samples(
                    model, test_dataset, output_dir, epoch + 1,
                    num_samples=num_vis_samples, image_size=image_size,
                    device=device, split="val"
                )
                if ema is not None:
                    ema.restore()

            if ema is not None:
                ema.apply_shadow()

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_loss": val_metrics["loss"],
                "best_pixel_error": best_pixel_error,
                "config": {
                    "model_type": model_type,
                    "hidden_size": p.hidden_size,
                    "num_rows": p.num_rows,
                    "num_cols": p.num_cols,
                    "num_lstm_layers": p.num_lstm_layers,
                    "dropout": p.dropout,
                    "num_samples": p.get("num_samples", p.num_row_samples),
                    "num_row_samples": p.get("num_row_samples", p.get("num_samples", 5)),
                    "num_col_samples": p.get("num_col_samples", p.get("num_samples", 18)),
                    "backbone_name": p.get("backbone_name", "resnet34"),
                    "use_attention": p.get("use_attention", True),
                    "image_size": list(image_size),
                },
            }

            if ema is not None:
                ema.restore()

            torch.save(checkpoint, output_dir / "last.pth")

            # Save best model (loss)
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                if ema is not None:
                    ema.apply_shadow()
                torch.save(checkpoint, output_dir / "best.pth")
                if ema is not None:
                    ema.restore()
                print(f"  New best model (loss)! Loss: {best_val_loss:.6f}")

            # Save best model (pixel error)
            if "pixel_error" in val_metrics and val_metrics["pixel_error"] < best_pixel_error:
                best_pixel_error = val_metrics["pixel_error"]
                checkpoint["best_pixel_error"] = best_pixel_error
                if ema is not None:
                    ema.apply_shadow()
                torch.save(checkpoint, output_dir / "best_pixel_error.pth")
                if ema is not None:
                    ema.restore()
                print(f"  New best model (pixel error)! Error: {best_pixel_error:.2f}px")

            if early_stopping is not None:
                if early_stopping(val_metrics["loss"]):
                    print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                    break

        mlflow.log_metric("best_val_loss", best_val_loss)
        mlflow.log_metric("best_val_pixel_error", best_pixel_error)

        print("\n" + "=" * 60)
        print(f"Training complete! Best val loss: {best_val_loss:.6f}")
        if best_pixel_error < float("inf"):
            print(f"Best val pixel error: {best_pixel_error:.2f}px")
        print(f"Models saved to: {output_dir}")
        print("=" * 60)


if __name__ == "__main__":
    main()
