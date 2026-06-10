
import argparse
import copy
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

from ltcd.datasets.keypoint_dataset import KeypointDatasetSimple
from ltcd.training import EMA, EarlyStopping, GridAugmentedDataset


class CrossAttentionLoss(nn.Module):
    """
    Loss function for cross-attention keypoint detection.

    Components:
    1. Grid loss: Direct coordinate error (MSE or Smooth L1)
    2. Row alignment: Points in same row should have similar y
    3. Column alignment: Points in same column should have similar x
    4. Ordering loss: Rows should be ordered top-to-bottom, columns left-to-right
    5. Auxiliary loss: Loss from intermediate decoder layers
    """

    def __init__(
        self,
        grid_weight: float = 1.0,
        row_align_weight: float = 0.1,
        col_align_weight: float = 0.1,
        ordering_weight: float = 0.05,
        offset_reg_weight: float = 0.0,
        use_smooth_l1: bool = True,
        smooth_l1_beta: float = 0.1,
        aux_loss_weight: float = 0.4,
    ) -> None:
        super().__init__()
        self.grid_weight = grid_weight
        self.row_align_weight = row_align_weight
        self.col_align_weight = col_align_weight
        self.ordering_weight = ordering_weight
        self.offset_reg_weight = offset_reg_weight
        self.use_smooth_l1 = use_smooth_l1
        self.smooth_l1_beta = smooth_l1_beta
        self.aux_loss_weight = aux_loss_weight

    def _compute_grid_loss(
        self, pred_grid: torch.Tensor, target_grid: torch.Tensor
    ) -> torch.Tensor:
        """Compute grid coordinate loss."""
        if self.use_smooth_l1:
            return F.smooth_l1_loss(pred_grid, target_grid, beta=self.smooth_l1_beta)
        return F.mse_loss(pred_grid, target_grid)

    def forward(
        self,
        pred_grid: torch.Tensor,
        target_grid: torch.Tensor,
        offsets: torch.Tensor | None = None,
        aux_outputs: list[torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Compute loss.

        Args:
            pred_grid: [B, num_rows, num_cols, 2]
            target_grid: [B, num_rows, num_cols, 2]
            offsets: [B, num_rows, num_cols, 2] (optional)
            aux_outputs: List of intermediate decoder outputs (optional)

        Returns:
            Dict with loss components
        """
        losses = {}
        total_loss = torch.tensor(0.0, device=pred_grid.device)

        grid_loss = self._compute_grid_loss(pred_grid, target_grid)
        total_loss = total_loss + self.grid_weight * grid_loss
        losses["grid_loss"] = grid_loss

        # Row alignment: variance of y within each row
        pred_y = pred_grid[..., 1]
        row_y_var = pred_y.var(dim=2).mean()
        total_loss = total_loss + self.row_align_weight * row_y_var
        losses["row_align_loss"] = row_y_var

        # Column alignment: variance of x within each column
        pred_x = pred_grid[..., 0]
        col_x_var = pred_x.var(dim=1).mean()
        total_loss = total_loss + self.col_align_weight * col_x_var
        losses["col_align_loss"] = col_x_var

        # Ordering loss: penalize out-of-order rows and columns
        if self.ordering_weight > 0:
            # Rows should be ordered by y (top to bottom = increasing y)
            row_y_mean = pred_y.mean(dim=2)  # [B, num_rows]
            row_order_violation = F.relu(row_y_mean[:, :-1] - row_y_mean[:, 1:])
            row_order_loss = row_order_violation.mean()

            # Columns should be ordered by x (left to right = increasing x)
            col_x_mean = pred_x.mean(dim=1)  # [B, num_cols]
            col_order_violation = F.relu(col_x_mean[:, :-1] - col_x_mean[:, 1:])
            col_order_loss = col_order_violation.mean()

            ordering_loss = row_order_loss + col_order_loss
            total_loss = total_loss + self.ordering_weight * ordering_loss
            losses["ordering_loss"] = ordering_loss

        if offsets is not None and self.offset_reg_weight > 0:
            offset_mag = (offsets ** 2).mean()
            total_loss = total_loss + self.offset_reg_weight * offset_mag
            losses["offset_reg_loss"] = offset_mag

        # Auxiliary loss from intermediate decoder layers
        if aux_outputs is not None and self.aux_loss_weight > 0:
            aux_loss = torch.tensor(0.0, device=pred_grid.device)
            for aux_grid in aux_outputs:
                aux_loss = aux_loss + self._compute_grid_loss(aux_grid, target_grid)
            aux_loss = aux_loss / len(aux_outputs)
            total_loss = total_loss + self.aux_loss_weight * aux_loss
            losses["aux_loss"] = aux_loss

        losses["loss"] = total_loss
        return losses


def compute_metrics(
    pred_grid: torch.Tensor,
    target_grid: torch.Tensor,
    image_size: tuple[int, int] = (512, 512),
) -> dict[str, float]:
    """
    Compute evaluation metrics.

    Args:
        pred_grid: [B, num_rows, num_cols, 2] in [-1, 1] range
        target_grid: [B, num_rows, num_cols, 2] in [-1, 1] range
        image_size: (H, W) for pixel error calculation

    Returns:
        Dict with metrics
    """
    with torch.no_grad():
        pred_pixels = pred_grid.clone()
        target_pixels = target_grid.clone()

        pred_pixels[..., 0] = (pred_pixels[..., 0] + 1) / 2 * image_size[1]  # x -> width
        pred_pixels[..., 1] = (pred_pixels[..., 1] + 1) / 2 * image_size[0]  # y -> height
        target_pixels[..., 0] = (target_pixels[..., 0] + 1) / 2 * image_size[1]
        target_pixels[..., 1] = (target_pixels[..., 1] + 1) / 2 * image_size[0]

        # Pixel error (L2 distance)
        pixel_error = torch.sqrt(((pred_pixels - target_pixels) ** 2).sum(dim=-1))
        mean_pixel_error = pixel_error.mean().item()
        max_pixel_error = pixel_error.max().item()

        # PCK (Percentage of Correct Keypoints) at different thresholds
        # Threshold as percentage of image diagonal
        diagonal = np.sqrt(image_size[0] ** 2 + image_size[1] ** 2)
        pck_5 = (pixel_error < 0.05 * diagonal).float().mean().item()
        pck_10 = (pixel_error < 0.10 * diagonal).float().mean().item()
        pck_20 = (pixel_error < 0.20 * diagonal).float().mean().item()

        pck_5px = (pixel_error < 5).float().mean().item()
        pck_10px = (pixel_error < 10).float().mean().item()
        pck_20px = (pixel_error < 20).float().mean().item()

        return {
            "pixel_error": mean_pixel_error,
            "max_pixel_error": max_pixel_error,
            "pck@0.05": pck_5,
            "pck@0.10": pck_10,
            "pck@0.20": pck_20,
            "pck@5px": pck_5px,
            "pck@10px": pck_10px,
            "pck@20px": pck_20px,
        }


def denormalize_grid(
    grid: np.ndarray,
    image_size: tuple[int, int],
) -> np.ndarray:
    """
    Convert grid from [-1, 1] to pixel coordinates.

    Args:
        grid: Grid in [-1, 1] range [num_rows, num_cols, 2]
        image_size: (H, W) image size

    Returns:
        Grid in pixel coordinates
    """
    grid = grid.copy()
    grid = (grid + 1) / 2
    grid[..., 0] *= image_size[1]  # x -> width
    grid[..., 1] *= image_size[0]  # y -> height
    return grid


def draw_grid_on_image(
    image: Image.Image,
    grid: np.ndarray,
    point_radius: int = 3,
    line_width: int = 1,
) -> Image.Image:
    """
    Draw predicted grid on image.

    Args:
        image: PIL Image
        grid: Grid of keypoints [num_rows, num_cols, 2] in pixel coordinates
        color: RGB color for drawing
        point_radius: Radius of keypoint circles
        line_width: Width of grid lines

    Returns:
        Image with grid drawn
    """
    image = image.copy()
    draw = ImageDraw.Draw(image)

    num_rows, num_cols = grid.shape[:2]

    def get_row_color(row_idx: int) -> tuple[int, int, int]:
        t = row_idx / (num_rows - 1) if num_rows > 1 else 0
        r = int(255 * (1 - t))
        g = int(255 * t * 0.7)
        b = int(255 * t)
        return (r, g, b)

    # Draw horizontal lines (rows)
    for row in range(num_rows):
        row_color = get_row_color(row)
        for col in range(num_cols - 1):
            x1, y1 = grid[row, col]
            x2, y2 = grid[row, col + 1]
            draw.line([(x1, y1), (x2, y2)], fill=row_color, width=line_width)

    # Draw vertical lines (columns)
    for col in range(num_cols):
        for row in range(num_rows - 1):
            x1, y1 = grid[row, col]
            x2, y2 = grid[row + 1, col]
            draw.line([(x1, y1), (x2, y2)], fill=(100, 100, 255), width=line_width)

    for row in range(num_rows):
        row_color = get_row_color(row)
        for col in range(num_cols):
            x, y = grid[row, col]
            draw.ellipse(
                [x - point_radius, y - point_radius, x + point_radius, y + point_radius],
                fill=row_color,
                outline=(0, 0, 0),
                width=1,
            )

    return image


def draw_comparison(
    image: Image.Image,
    pred_grid: np.ndarray,
    target_grid: np.ndarray,
    point_radius: int = 3,
    line_width: int = 1,
) -> Image.Image:
    """
    Draw both predicted and target grids on image for comparison.

    Args:
        image: PIL Image
        pred_grid: Predicted grid [num_rows, num_cols, 2] in pixel coordinates
        target_grid: Target grid [num_rows, num_cols, 2] in pixel coordinates

    Returns:
        Image with both grids drawn (red=pred, green=target)
    """
    image = image.copy()
    draw = ImageDraw.Draw(image)

    num_rows, num_cols = pred_grid.shape[:2]

    # Draw target grid in green (underneath)
    for row in range(num_rows):
        for col in range(num_cols - 1):
            x1, y1 = target_grid[row, col]
            x2, y2 = target_grid[row, col + 1]
            draw.line([(x1, y1), (x2, y2)], fill=(0, 200, 0), width=line_width)

    for col in range(num_cols):
        for row in range(num_rows - 1):
            x1, y1 = target_grid[row, col]
            x2, y2 = target_grid[row + 1, col]
            draw.line([(x1, y1), (x2, y2)], fill=(0, 200, 0), width=line_width)

    # Draw predicted grid in red (on top)
    for row in range(num_rows):
        for col in range(num_cols - 1):
            x1, y1 = pred_grid[row, col]
            x2, y2 = pred_grid[row, col + 1]
            draw.line([(x1, y1), (x2, y2)], fill=(255, 50, 50), width=line_width)

    for col in range(num_cols):
        for row in range(num_rows - 1):
            x1, y1 = pred_grid[row, col]
            x2, y2 = pred_grid[row + 1, col]
            draw.line([(x1, y1), (x2, y2)], fill=(255, 50, 50), width=line_width)

    # Draw target points (green circles)
    for row in range(num_rows):
        for col in range(num_cols):
            x, y = target_grid[row, col]
            draw.ellipse(
                [x - point_radius, y - point_radius, x + point_radius, y + point_radius],
                fill=(0, 200, 0),
                outline=(0, 100, 0),
            )

    # Draw predicted points (red circles)
    for row in range(num_rows):
        for col in range(num_cols):
            x, y = pred_grid[row, col]
            draw.ellipse(
                [x - point_radius, y - point_radius, x + point_radius, y + point_radius],
                fill=(255, 50, 50),
                outline=(150, 0, 0),
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
) -> None:
    """
    Visualize model predictions on sample images.

    Args:
        model: Trained model
        dataset: Dataset to sample from
        output_dir: Output directory for visualizations
        epoch: Current epoch number
        num_samples: Number of samples to visualize
        image_size: Image size (H, W)
        device: Device
        split: 'train' or 'val'
    """
    model.eval()

    vis_dir = output_dir / "images" / split
    vis_dir.mkdir(parents=True, exist_ok=True)

    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    for i, idx in enumerate(indices):
        sample = dataset[idx]
        image_tensor = sample["image"].unsqueeze(0).to(device)
        target_grid = sample["grid"].numpy()

        output = model(image_tensor)
        pred_grid = output["grid"][0].cpu().numpy()

        pred_grid_px = denormalize_grid(pred_grid, image_size)
        target_grid_px = denormalize_grid(target_grid, image_size)

        image_denorm = sample["image"] * std + mean
        image_denorm = torch.clamp(image_denorm, 0, 1)
        image_np = (image_denorm.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        image_pil = Image.fromarray(image_np)

        # Draw comparison (pred=red, target=green)
        vis_image = draw_comparison(image_pil, pred_grid_px, target_grid_px)

        vis_path = vis_dir / f"epoch_{epoch:04d}_sample_{i:02d}.png"
        vis_image.save(vis_path)

    model.train()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
    scaler: torch.cuda.amp.GradScaler | None = None,
    clip_grad: float = 1.0,
    ema: EMA | None = None,
    image_size: tuple[int, int] = (512, 512),
) -> dict[str, float]:
    """Train for one epoch."""
    model.train()

    total_metrics = {}
    num_batches = 0

    for batch in tqdm(loader, desc="Training"):
        images = batch["image"].to(device)
        target_grid = batch["grid"].to(device)

        optimizer.zero_grad()

        with torch.amp.autocast(device_type="cuda", enabled=(scaler is not None)):
            output = model(images)
            pred_grid = output["grid"]
            offsets = output.get("offsets")
            aux_outputs = output.get("aux_outputs")

            loss_dict = criterion(pred_grid, target_grid, offsets, aux_outputs)
            loss = loss_dict["loss"]

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()

        if ema is not None:
            ema.update()

        for key, value in loss_dict.items():
            if key not in total_metrics:
                total_metrics[key] = 0.0
            total_metrics[key] += value.item()

        metrics = compute_metrics(pred_grid.detach(), target_grid, image_size)
        for key, value in metrics.items():
            if key not in total_metrics:
                total_metrics[key] = 0.0
            total_metrics[key] += value

        num_batches += 1

    return {k: v / num_batches for k, v in total_metrics.items()}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
    image_size: tuple[int, int] = (512, 512),
) -> dict[str, float]:
    """Evaluate model."""
    model.eval()

    total_metrics = {}
    num_batches = 0

    for batch in tqdm(loader, desc="Evaluating"):
        images = batch["image"].to(device)
        target_grid = batch["grid"].to(device)

        output = model(images)
        pred_grid = output["grid"]
        offsets = output.get("offsets")
        aux_outputs = output.get("aux_outputs")

        loss_dict = criterion(pred_grid, target_grid, offsets, aux_outputs)

        for key, value in loss_dict.items():
            if key not in total_metrics:
                total_metrics[key] = 0.0
            total_metrics[key] += value.item()

        metrics = compute_metrics(pred_grid, target_grid, image_size)
        for key, value in metrics.items():
            if key not in total_metrics:
                total_metrics[key] = 0.0
            total_metrics[key] += value

        num_batches += 1

    return {k: v / num_batches for k, v in total_metrics.items()}


def create_model(cfg) -> nn.Module:
    model_type = cfg.model.type

    if model_type == "cross_attention":
        from ltcd.archs.cross_attention_keypoint import CrossAttentionKeypointDetector

        model = CrossAttentionKeypointDetector(
            num_rows=cfg.model.num_rows,
            num_cols=cfg.model.num_cols,
            hidden_size=cfg.model.hidden_size,
            num_heads=cfg.model.get("num_heads", 8),
            num_encoder_layers=cfg.model.get("num_encoder_layers", 2),
            num_decoder_layers=cfg.model.get("num_decoder_layers", 4),
            dropout=cfg.model.get("dropout", 0.1),
            use_pretrained=cfg.model.get("use_pretrained", True),
            backbone_name=cfg.model.get("backbone_name", "resnet34"),
        )

        model = HierarchicalCrossAttentionDetector(
            num_rows=cfg.model.num_rows,
            num_cols=cfg.model.num_cols,
            hidden_size=cfg.model.hidden_size,
            num_heads=cfg.model.get("num_heads", 8),
            num_layers=cfg.model.get("num_layers", 3),
            dropout=cfg.model.get("dropout", 0.1),
            use_pretrained=cfg.model.get("use_pretrained", True),
        )
    elif model_type == "lightweight":
        from ltcd.archs.cross_attention_keypoint import LightweightCrossAttentionDetector

        model = LightweightCrossAttentionDetector(
            num_rows=cfg.model.num_rows,
            num_cols=cfg.model.num_cols,
            hidden_size=cfg.model.get("hidden_size", 128),
            num_sample_points=cfg.model.get("num_sample_points", 4),
            dropout=cfg.model.get("dropout", 0.1),
            use_pretrained=cfg.model.get("use_pretrained", True),
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)

    experiment_name = cfg.get("experiment_name", "cross_attention_keypoint")
    run_name = cfg.get("run_name", f"cross_attn_{cfg.model.type}")
    mlflow.set_experiment(experiment_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    image_size = tuple(cfg.data.image_size)
    num_keypoints = cfg.model.num_rows * cfg.model.num_cols

    train_base = KeypointDatasetSimple(
        cfg.data.train_dir,
        image_size=image_size,
        num_keypoints=num_keypoints,
    )
    val_base = KeypointDatasetSimple(
        cfg.data.val_dir,
        image_size=image_size,
        num_keypoints=num_keypoints,
    )

    aug_cfg = cfg.data.get("augmentation", {})

    train_dataset = GridAugmentedDataset(
        train_base,
        num_rows=cfg.model.num_rows,
        num_cols=cfg.model.num_cols,
        augment=cfg.data.get("augment", True),
        flip_prob=aug_cfg.get("flip_prob", 0.5),
        rotation_range=aug_cfg.get("rotation_range", 5.0),
        scale_range=tuple(aug_cfg.get("scale_range", [0.95, 1.05])),
    )
    val_dataset = GridAugmentedDataset(
        val_base,
        num_rows=cfg.model.num_rows,
        num_cols=cfg.model.num_cols,
        augment=False,
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.data.get("num_workers", 4),
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.data.get("num_workers", 4),
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )

    model = create_model(cfg)
    model = model.to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model type: {cfg.model.type}")
    print(f"Grid size: {cfg.model.num_rows} rows x {cfg.model.num_cols} cols")
    print(f"Parameters: {num_params:,}")

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "model_type": cfg.model.type,
            "num_rows": cfg.model.num_rows,
            "num_cols": cfg.model.num_cols,
            "hidden_size": cfg.model.hidden_size,
            "num_heads": cfg.model.get("num_heads", 8),
            "batch_size": cfg.training.batch_size,
            "lr": cfg.training.lr,
            "epochs": cfg.training.epochs,
        })
        mlflow.log_param("num_parameters", num_params)

        criterion = CrossAttentionLoss(
            grid_weight=cfg.loss.get("grid_weight", 1.0),
            row_align_weight=cfg.loss.get("row_align_weight", 0.1),
            col_align_weight=cfg.loss.get("col_align_weight", 0.1),
            ordering_weight=cfg.loss.get("ordering_weight", 0.05),
            offset_reg_weight=cfg.loss.get("offset_reg_weight", 0.0),
            use_smooth_l1=cfg.loss.get("use_smooth_l1", False),
            smooth_l1_beta=cfg.loss.get("smooth_l1_beta", 0.1),
            aux_loss_weight=cfg.loss.get("aux_loss_weight", 0.4),
        )

        backbone_params = []
        other_params = []
        for name, param in model.named_parameters():
            if "backbone" in name:
                backbone_params.append(param)
            else:
                other_params.append(param)

        optimizer = torch.optim.AdamW([
            {"params": backbone_params, "lr": cfg.training.lr * 0.1},
            {"params": other_params, "lr": cfg.training.lr},
        ], weight_decay=cfg.training.get("weight_decay", 1e-4))

        def lr_lambda(epoch):
            warmup_epochs = cfg.training.get("warmup_epochs", 5)
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            else:
                progress = (epoch - warmup_epochs) / (cfg.training.epochs - warmup_epochs)
                return 0.5 * (1 + np.cos(np.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        scaler = torch.cuda.amp.GradScaler() if device == "cuda" else None

        # EMA (Exponential Moving Average)
        ema = None
        if cfg.training.get("use_ema", False):
            ema_decay = cfg.training.get("ema_decay", 0.999)
            ema = EMA(model, decay=ema_decay)
            print(f"Using EMA with decay={ema_decay}")

        early_stopping = None
        if cfg.training.get("early_stopping_patience", 0) > 0:
            patience = cfg.training.get("early_stopping_patience")
            min_delta = cfg.training.get("early_stopping_min_delta", 0.0)
            early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)
            print(f"Early stopping enabled with patience={patience}")

        date_str = datetime.now().strftime("%Y_%m_%d")
        output_dir = Path(cfg.output.dir) / date_str
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {output_dir}")

        start_epoch = 0
        best_loss = float("inf")
        best_pixel_error = float("inf")

        if args.resume:
            print(f"Resuming from {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            best_loss = checkpoint.get("best_loss", float("inf"))
            best_pixel_error = checkpoint.get("best_pixel_error", float("inf"))
            if ema is not None:
                ema._register()  # Re-register with loaded weights

        for epoch in range(start_epoch, cfg.training.epochs):
            print(f"\nEpoch {epoch + 1}/{cfg.training.epochs}")

            train_metrics = train_one_epoch(
                model, train_loader, optimizer, criterion, device, scaler,
                clip_grad=cfg.training.get("clip_grad", 1.0),
                ema=ema,
                image_size=image_size,
            )

            if ema is not None:
                ema.apply_shadow()
            val_metrics = evaluate(model, val_loader, criterion, device, image_size)
            if ema is not None:
                ema.restore()

            scheduler.step()
            current_lr = optimizer.param_groups[1]["lr"]  # Non-backbone LR

            log_metrics = {"learning_rate": current_lr}
            for k, v in train_metrics.items():
                log_metrics[f"train_{k}"] = v
            for k, v in val_metrics.items():
                log_metrics[f"val_{k}"] = v
            mlflow.log_metrics(log_metrics, step=epoch)

            print(f"Train Loss: {train_metrics['loss']:.6f} | Val Loss: {val_metrics['loss']:.6f}")
            print(f"  Grid: {train_metrics['grid_loss']:.6f} -> {val_metrics['grid_loss']:.6f}")
            print(f"  Pixel Error: {train_metrics['pixel_error']:.2f}px -> {val_metrics['pixel_error']:.2f}px")
            print(f"  PCK@10px: {train_metrics['pck@10px']:.3f} -> {val_metrics['pck@10px']:.3f}")
            if "row_align_loss" in train_metrics:
                print(f"  Row Align: {train_metrics['row_align_loss']:.6f}")
                print(f"  Col Align: {train_metrics['col_align_loss']:.6f}")
            if "ordering_loss" in train_metrics:
                print(f"  Ordering: {train_metrics['ordering_loss']:.6f}")
            if "aux_loss" in train_metrics:
                print(f"  Aux Loss: {train_metrics['aux_loss']:.6f}")
            print(f"  LR: {current_lr:.6f}")

            visualize_every = cfg.output.get("visualize_every_n_epochs", 0)
            num_vis_samples = cfg.output.get("num_vis_samples", 4)
            if visualize_every > 0 and (epoch + 1) % visualize_every == 0:
                print(f"  Saving visualizations...")
                # Use EMA weights if available for visualization
                if ema is not None:
                    ema.apply_shadow()
                visualize_samples(
                    model, train_dataset, output_dir, epoch + 1,
                    num_samples=num_vis_samples, image_size=image_size,
                    device=device, split="train"
                )
                visualize_samples(
                    model, val_dataset, output_dir, epoch + 1,
                    num_samples=num_vis_samples, image_size=image_size,
                    device=device, split="val"
                )
                if ema is not None:
                    ema.restore()

            # Save checkpoint (use EMA weights if available)
            if ema is not None:
                ema.apply_shadow()

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
                "best_loss": best_loss,
                "best_pixel_error": best_pixel_error,
                "config": {
                    "model_type": cfg.model.type,
                    "num_rows": cfg.model.num_rows,
                    "num_cols": cfg.model.num_cols,
                    "hidden_size": cfg.model.hidden_size,
                    "num_heads": cfg.model.get("num_heads", 8),
                    "num_encoder_layers": cfg.model.get("num_encoder_layers", 2),
                    "num_decoder_layers": cfg.model.get("num_decoder_layers", 4),
                    "backbone_name": cfg.model.get("backbone_name", "resnet34"),
                    "num_layers": cfg.model.get("num_layers", 3),
                    "num_sample_points": cfg.model.get("num_sample_points", 4),
                    "image_size": list(cfg.data.image_size),
                },
            }

            if ema is not None:
                ema.restore()

            torch.save(checkpoint, output_dir / "last.pth")

            if val_metrics["loss"] < best_loss:
                best_loss = val_metrics["loss"]
                if ema is not None:
                    ema.apply_shadow()
                torch.save(checkpoint, output_dir / "best.pth")
                if ema is not None:
                    ema.restore()
                print(f"  New best model (loss)! Loss: {best_loss:.6f}")

            if val_metrics["pixel_error"] < best_pixel_error:
                best_pixel_error = val_metrics["pixel_error"]
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

        mlflow.log_metric("best_val_loss", best_loss)
        mlflow.log_metric("best_val_pixel_error", best_pixel_error)

        print("\nTraining complete!")
        print(f"Best validation loss: {best_loss:.6f}")
        print(f"Best validation pixel error: {best_pixel_error:.2f}px")


if __name__ == "__main__":
    main()
