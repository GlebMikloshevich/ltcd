
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

from ltcd.archs.normal_line_predictor import (
    NormalLinePredictor,    points_to_normal_form,
)
from ltcd.datasets.keypoint_dataset import KeypointDatasetSimple
from ltcd.training import EMA, EarlyStopping, GridAugmentedDataset


class NormalFormLoss(nn.Module):

    def __init__(
        self,
        first_row_weight: float = 2.0,
        other_rows_weight: float = 1.0,
        normal_weight: float = 0.1,
        line_weight: float = 0.1,
        column_weight: float = 0.1,
        ordering_weight: float = 0.0,
        use_smooth_l1: bool = False,
        smooth_l1_beta: float = 0.1,
    ) -> None:
        super().__init__()
        self.first_row_weight = first_row_weight
        self.other_rows_weight = other_rows_weight
        self.normal_weight = normal_weight
        self.line_weight = line_weight
        self.column_weight = column_weight
        self.ordering_weight = ordering_weight
        self.use_smooth_l1 = use_smooth_l1
        self.smooth_l1_beta = smooth_l1_beta

    def _compute_point_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.use_smooth_l1:
            return F.smooth_l1_loss(pred, target, beta=self.smooth_l1_beta)
        return F.mse_loss(pred, target)

    def _compute_line_residual_loss(self, rows: torch.Tensor) -> torch.Tensor:
        B, num_rows, num_cols, _ = rows.shape

        if num_cols < 2:
            return torch.tensor(0.0, device=rows.device)

        total_loss = torch.tensor(0.0, device=rows.device)

        for row_idx in range(num_rows):
            row = rows[:, row_idx, :, :]
            cos_theta, sin_theta, rho = points_to_normal_form(row)

            x = row[:, :, 0]
            y = row[:, :, 1]

            residuals = torch.abs(
                x * cos_theta.unsqueeze(1) + y * sin_theta.unsqueeze(1) - rho.unsqueeze(1)
            )
            total_loss = total_loss + residuals.mean()

        return total_loss / num_rows

    def _compute_column_alignment_loss(self, rows: torch.Tensor) -> torch.Tensor:
        B, num_rows, num_cols, _ = rows.shape

        if num_rows < 2:
            return torch.tensor(0.0, device=rows.device)

        x_coords = rows[:, :, :, 0]  # [B, num_rows, num_cols]
        x_var = x_coords.var(dim=1)  # [B, num_cols]

        return x_var.mean()

    def _compute_normal_consistency_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compare predicted line parameters with GT line parameters."""
        pred_cos, pred_sin, pred_rho = points_to_normal_form(pred)
        target_cos, target_sin, target_rho = points_to_normal_form(target)

        cos_loss = F.mse_loss(pred_cos, target_cos)
        sin_loss = F.mse_loss(pred_sin, target_sin)
        rho_loss = F.mse_loss(pred_rho, target_rho)

        return cos_loss + sin_loss + rho_loss

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Compute combined loss.

        Args:
            pred: Predicted keypoints [B, num_rows, num_cols, 2]
            target: Ground truth keypoints [B, num_rows, num_cols, 2]
        """
        num_rows = pred.size(1)
        losses = {}

        first_row_pred = pred[:, 0, :, :]
        first_row_target = target[:, 0, :, :]
        first_row_loss = self._compute_point_loss(first_row_pred, first_row_target)
        losses["first_row_loss"] = first_row_loss

        if num_rows > 1:
            other_rows_pred = pred[:, 1:, :, :]
            other_rows_target = target[:, 1:, :, :]
            other_rows_loss = self._compute_point_loss(other_rows_pred, other_rows_target)
        else:
            other_rows_loss = torch.tensor(0.0, device=pred.device)
        losses["other_rows_loss"] = other_rows_loss

        normal_loss = self._compute_normal_consistency_loss(pred, target)
        losses["normal_loss"] = normal_loss

        line_loss = self._compute_line_residual_loss(pred)
        losses["line_loss"] = line_loss

        column_loss = self._compute_column_alignment_loss(pred)
        losses["column_loss"] = column_loss

        if self.ordering_weight > 0 and num_rows > 1:
            row_y_mean = pred[..., 1].mean(dim=2)  # [B, num_rows]
            row_order_violation = F.relu(row_y_mean[:, :-1] - row_y_mean[:, 1:])
            ordering_loss = row_order_violation.mean()
            losses["ordering_loss"] = ordering_loss
        else:
            ordering_loss = torch.tensor(0.0, device=pred.device)
            losses["ordering_loss"] = ordering_loss

        total_loss = (
            self.first_row_weight * first_row_loss
            + self.other_rows_weight * other_rows_loss
            + self.normal_weight * normal_loss
            + self.line_weight * line_loss
            + self.column_weight * column_loss
            + self.ordering_weight * ordering_loss
        )
        losses["loss"] = total_loss

        return losses


def compute_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    image_size: tuple[int, int] = (512, 512),
) -> dict[str, float]:
    """
    Compute evaluation metrics.

    Args:
        pred: [B, num_rows, num_cols, 2] in [-1, 1] range
        target: [B, num_rows, num_cols, 2] in [-1, 1] range
        image_size: (H, W) for pixel error calculation
    """
    with torch.no_grad():
        pred_px = pred.clone()
        target_px = target.clone()

        pred_px[..., 0] = (pred_px[..., 0] + 1) / 2 * image_size[1]
        pred_px[..., 1] = (pred_px[..., 1] + 1) / 2 * image_size[0]
        target_px[..., 0] = (target_px[..., 0] + 1) / 2 * image_size[1]
        target_px[..., 1] = (target_px[..., 1] + 1) / 2 * image_size[0]

        pixel_error = torch.sqrt(((pred_px - target_px) ** 2).sum(dim=-1))
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
            "pck@0.05": pck_5,
            "pck@0.10": pck_10,
            "pck@5px": pck_5px,
            "pck@10px": pck_10px,
            "pck@20px": pck_20px,
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


def draw_comparison(
    image: Image.Image,
    pred_rows: np.ndarray,
    target_rows: np.ndarray,
    point_radius: int = 2,
    line_width: int = 1,
) -> Image.Image:
    """
    Draw both predicted and target rows on image for comparison.

    Args:
        image: PIL Image
        pred_rows: [num_rows, num_cols, 2] in pixel coordinates
        target_rows: [num_rows, num_cols, 2] in pixel coordinates

    Returns:
        Image with rows drawn (red=pred, green=target)
    """
    image = image.copy()
    draw = ImageDraw.Draw(image)

    num_rows, num_cols = pred_rows.shape[:2]

    # Draw target rows (green)
    for row in range(num_rows):
        points = target_rows[row]
        for i in range(num_cols - 1):
            x1, y1 = points[i]
            x2, y2 = points[i + 1]
            draw.line([(x1, y1), (x2, y2)], fill=(0, 180, 0), width=line_width)

    # Draw predicted rows (red)
    for row in range(num_rows):
        points = pred_rows[row]
        for i in range(num_cols - 1):
            x1, y1 = points[i]
            x2, y2 = points[i + 1]
            draw.line([(x1, y1), (x2, y2)], fill=(255, 50, 50), width=line_width)

    # Draw target points (green circles)
    for row in range(num_rows):
        for col in range(num_cols):
            x, y = target_rows[row, col]
            draw.ellipse(
                [x - point_radius, y - point_radius, x + point_radius, y + point_radius],
                fill=(0, 200, 0),
                outline=(0, 100, 0),
            )

    # Draw predicted points (red circles)
    for row in range(num_rows):
        for col in range(num_cols):
            x, y = pred_rows[row, col]
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
    """Visualize model predictions on sample images."""
    model.eval()

    vis_dir = output_dir / "images" / split
    vis_dir.mkdir(parents=True, exist_ok=True)

    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    for i, idx in enumerate(indices):
        sample = dataset[idx]
        image_tensor = sample["image"].unsqueeze(0).to(device)
        target = sample["keypoints"].numpy()  # [num_rows, num_cols, 2]

        pred = model(image_tensor, teacher_forcing_rows=None)
        pred_np = pred[0].cpu().numpy()  # [num_rows, num_cols, 2]

        pred_px = denormalize_points(pred_np, image_size)
        target_px = denormalize_points(target, image_size)

        image_denorm = sample["image"] * std + mean
        image_denorm = torch.clamp(image_denorm, 0, 1)
        image_np = (image_denorm.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        image_pil = Image.fromarray(image_np)

        vis_image = draw_comparison(image_pil, pred_px, target_px)

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
    teacher_forcing_ratio: float = 0.5,
    predict_other_rows: bool = True,
    ema: EMA | None = None,
    image_size: tuple[int, int] = (512, 512),
) -> dict[str, float]:
    """Train for one epoch."""
    model.train()
    running_metrics = {}
    num_batches = 0

    for batch in tqdm(loader, desc="Training"):
        images = batch["image"].to(device)
        keypoints = batch["grid"].to(device)

        if not predict_other_rows:
            target = keypoints[:, :1, :, :]
        else:
            target = keypoints

        optimizer.zero_grad()

        with torch.amp.autocast(device_type="cuda", enabled=(scaler is not None)):
            pred = model(
                image=images,
                teacher_forcing_rows=keypoints if predict_other_rows else None,
                teacher_forcing_ratio=teacher_forcing_ratio,
            )

            loss_dict = criterion(pred, target)
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

        metrics = compute_metrics(pred.detach(), target, image_size)
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
    predict_other_rows: bool = True,
    image_size: tuple[int, int] = (512, 512),
) -> dict[str, float]:
    """Evaluate the model."""
    model.eval()
    running_metrics = {}
    num_batches = 0

    for batch in tqdm(loader, desc="Evaluating"):
        images = batch["image"].to(device)
        keypoints = batch["grid"].to(device)

        if not predict_other_rows:
            target = keypoints[:, :1, :, :]
        else:
            target = keypoints

        pred = model(image=images, teacher_forcing_rows=None)

        loss_dict = criterion(pred, target)

        for k, v in loss_dict.items():
            if k not in running_metrics:
                running_metrics[k] = 0.0
            running_metrics[k] += v.item()

        metrics = compute_metrics(pred, target, image_size)
        for k, v in metrics.items():
            if k not in running_metrics:
                running_metrics[k] = 0.0
            running_metrics[k] += v

        num_batches += 1

    return {k: v / num_batches for k, v in running_metrics.items()}


def create_model(cfg) -> nn.Module:
    p = cfg.params
    return NormalLinePredictor(
        hidden_size=p.hidden_size,
        num_rows=p.num_rows,
        num_cols=p.num_cols,
        num_lstm_layers=p.num_lstm_layers,
        dropout=p.dropout,
        use_pretrained=p.use_pretrained,
        predict_other_rows=p.get("predict_other_rows", True),
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

    model_type = p.get("model_type", "v1")
    predict_other_rows = p.get("predict_other_rows", True)

    print(f"\nCreating model: NormalLinePredictor {model_type}")
    print(f"Predict other rows: {predict_other_rows}")

    model = create_model(cfg)
    model = model.to(device)

    num_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print(f"Model parameters: {num_params:,}")

    with mlflow.start_run(run_name=cfg.run_name):
        mlflow.log_params(dict(p))
        mlflow.log_param("num_parameters", num_params)

        criterion = NormalFormLoss(
            first_row_weight=p.first_row_weight,
            other_rows_weight=p.get("other_rows_weight", 1.0),
            normal_weight=p.get("normal_weight", 0.1),
            line_weight=p.line_weight,
            column_weight=p.column_weight,
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

        teacher_forcing_ratio = p.get("teacher_forcing_ratio", 0.5)
        print(f"Teacher forcing ratio: {teacher_forcing_ratio}")

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
                teacher_forcing_ratio=teacher_forcing_ratio,
                predict_other_rows=predict_other_rows,
                ema=ema,
                image_size=image_size,
            )

            if ema is not None:
                ema.apply_shadow()
            val_metrics = evaluate(
                model, test_loader, criterion, device,
                predict_other_rows=predict_other_rows,
                image_size=image_size,
            )
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
            print(f"  First Row: {train_metrics['first_row_loss']:.6f} -> {val_metrics['first_row_loss']:.6f}")
            if predict_other_rows:
                print(f"  Other Rows: {train_metrics['other_rows_loss']:.6f} -> {val_metrics['other_rows_loss']:.6f}")
            if "pixel_error" in val_metrics:
                print(f"  Pixel Error: {train_metrics['pixel_error']:.2f}px -> {val_metrics['pixel_error']:.2f}px")
            if "pck@10px" in val_metrics:
                print(f"  PCK@10px: {train_metrics['pck@10px']:.3f} -> {val_metrics['pck@10px']:.3f}")
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
                    "predict_other_rows": predict_other_rows,
                    "line_representation": "normal_form",
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
