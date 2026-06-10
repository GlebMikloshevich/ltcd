
import argparse
from pathlib import Path

import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from ltcd.datasets.keypoint_dataset import KeypointDatasetSimple


class GridDataset(Dataset):

    def __init__(
        self,
        base_dataset: KeypointDatasetSimple,
        num_rows: int = 18,
        num_cols: int = 5,
        augment: bool = False,
    ) -> None:
        self.base_dataset = base_dataset
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.augment = augment

        # Color augmentation (applied after base dataset transforms)
        if augment:
            self.color_jitter = transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            )
        else:
            self.color_jitter = None

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> dict:
        image, keypoints = self.base_dataset[idx]

        if self.color_jitter is not None:
            image = self.color_jitter(image)

        # Reshape from [num_keypoints, 2] to grid
        # Column-major: [col0_row0, col0_row1, ..., col1_row0, ...]
        grid = keypoints.view(self.num_cols, self.num_rows, 2)
        grid = grid.permute(1, 0, 2)  # [num_rows, num_cols, 2]

        return {
            "image": image,
            "grid": grid,
            "keypoints_flat": keypoints,
        }


class DeformableGridLoss(nn.Module):

    def __init__(
        self,
        grid_weight: float = 1.0,
        row_align_weight: float = 0.1,
        col_align_weight: float = 0.1,
        smoothness_weight: float = 0.05,
        offset_reg_weight: float = 0.1,
    ) -> None:
        super().__init__()
        self.grid_weight = grid_weight
        self.row_align_weight = row_align_weight
        self.col_align_weight = col_align_weight
        self.smoothness_weight = smoothness_weight
        self.offset_reg_weight = offset_reg_weight

    def forward(
        self,
        pred_grid: torch.Tensor,
        target_grid: torch.Tensor,
        offsets: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Compute combined loss.

        Args:
            pred_grid: [B, num_rows, num_cols, 2]
            target_grid: [B, num_rows, num_cols, 2]
            offsets: [B, num_rows, num_cols, 2] (optional)

        Returns:
            Dict with loss components
        """
        losses = {}
        total_loss = torch.tensor(0.0, device=pred_grid.device)

        grid_loss = F.mse_loss(pred_grid, target_grid)
        total_loss = total_loss + self.grid_weight * grid_loss
        losses["grid_loss"] = grid_loss

        # Row alignment loss: variance of y-coordinates within each row
        pred_y = pred_grid[..., 1]  # [B, R, C]
        row_y_var = pred_y.var(dim=2).mean()  # Variance across columns
        total_loss = total_loss + self.row_align_weight * row_y_var
        losses["row_align_loss"] = row_y_var

        # Column alignment loss: variance of x-coordinates within each column
        pred_x = pred_grid[..., 0]  # [B, R, C]
        col_x_var = pred_x.var(dim=1).mean()  # Variance across rows
        total_loss = total_loss + self.col_align_weight * col_x_var
        losses["col_align_loss"] = col_x_var

        if offsets is not None:
            # Smoothness loss: penalize gradient of offset field
            dx = offsets[:, :, 1:, :] - offsets[:, :, :-1, :]  # Horizontal gradient
            dy = offsets[:, 1:, :, :] - offsets[:, :-1, :, :]  # Vertical gradient
            smoothness_loss = (dx ** 2).mean() + (dy ** 2).mean()
            total_loss = total_loss + self.smoothness_weight * smoothness_loss
            losses["smoothness_loss"] = smoothness_loss

            # Offset regularization: penalize large offsets
            offset_mag = (offsets ** 2).mean()
            total_loss = total_loss + self.offset_reg_weight * offset_mag
            losses["offset_reg_loss"] = offset_mag

        losses["loss"] = total_loss
        return losses


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
    scaler: torch.cuda.amp.GradScaler | None = None,
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

            loss_dict = criterion(pred_grid, target_grid, offsets)
            loss = loss_dict["loss"]

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        for key, value in loss_dict.items():
            if key not in total_metrics:
                total_metrics[key] = 0.0
            total_metrics[key] += value.item()
        num_batches += 1

    return {k: v / num_batches for k, v in total_metrics.items()}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
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

        loss_dict = criterion(pred_grid, target_grid, offsets)

        for key, value in loss_dict.items():
            if key not in total_metrics:
                total_metrics[key] = 0.0
            total_metrics[key] += value.item()
        num_batches += 1

    return {k: v / num_batches for k, v in total_metrics.items()}


def create_model(cfg) -> nn.Module:
    model_type = cfg.model.type

    if model_type == "affine":
        from ltcd.archs.deformable_grid_predictor import AffineGridPredictor

        model = AffineGridPredictor(
            num_rows=cfg.model.num_rows,
            num_cols=cfg.model.num_cols,
            hidden_size=cfg.model.hidden_size,
            offset_scale=cfg.model.get("offset_scale", 0.1),
            use_pretrained=cfg.model.get("use_pretrained", True),
            backbone_name=cfg.model.get("backbone_name", "resnet34"),
        )

        model = TPSDeformableGridPredictor(
            num_rows=cfg.model.num_rows,
            num_cols=cfg.model.num_cols,
            num_control_points=cfg.model.get("num_control_points", 16),
            hidden_size=cfg.model.hidden_size,
            offset_scale=cfg.model.get("offset_scale", 0.05),
            use_pretrained=cfg.model.get("use_pretrained", True),
            backbone_name=cfg.model.get("backbone_name", "resnet34"),
        )

        model = MultiScaleDeformableGridPredictor(
            num_rows=cfg.model.num_rows,
            num_cols=cfg.model.num_cols,
            hidden_size=cfg.model.hidden_size,
            offset_scale=cfg.model.get("offset_scale", 0.1),
            use_pretrained=cfg.model.get("use_pretrained", True),
        )

        model = CornerHomographyPredictor(
            num_rows=cfg.model.num_rows,
            num_cols=cfg.model.num_cols,
            hidden_size=cfg.model.hidden_size,
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

    experiment_name = cfg.get("experiment_name", "deformable_grid")
    run_name = cfg.get("run_name", f"deformable_grid_{cfg.model.type}")
    mlflow.set_experiment(experiment_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "model_type": cfg.model.type,
            "num_rows": cfg.model.num_rows,
            "num_cols": cfg.model.num_cols,
            "hidden_size": cfg.model.hidden_size,
            "offset_scale": cfg.model.get("offset_scale", 0.1),
            "backbone_name": cfg.model.get("backbone_name", "resnet34"),
            "batch_size": cfg.training.batch_size,
            "epochs": cfg.training.epochs,
            "lr": cfg.training.lr,
            "weight_decay": cfg.training.get("weight_decay", 1e-4),
            "image_size": str(cfg.data.image_size),
            "grid_weight": cfg.loss.get("grid_weight", 1.0),
            "row_align_weight": cfg.loss.get("row_align_weight", 0.1),
            "col_align_weight": cfg.loss.get("col_align_weight", 0.1),
            "smoothness_weight": cfg.loss.get("smoothness_weight", 0.05),
            "offset_reg_weight": cfg.loss.get("offset_reg_weight", 0.1),
        })

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

        train_dataset = GridDataset(
            train_base,
            num_rows=cfg.model.num_rows,
            num_cols=cfg.model.num_cols,
            augment=cfg.data.get("augment", True),
        )
        val_dataset = GridDataset(
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
        mlflow.log_param("num_parameters", num_params)

        criterion = DeformableGridLoss(
            grid_weight=cfg.loss.get("grid_weight", 1.0),
            row_align_weight=cfg.loss.get("row_align_weight", 0.1),
            col_align_weight=cfg.loss.get("col_align_weight", 0.1),
            smoothness_weight=cfg.loss.get("smoothness_weight", 0.05),
            offset_reg_weight=cfg.loss.get("offset_reg_weight", 0.1),
        )

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.training.lr,
            weight_decay=cfg.training.get("weight_decay", 1e-4),
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.training.epochs,
            eta_min=cfg.training.get("min_lr", 1e-6),
        )

        scaler = torch.cuda.amp.GradScaler() if device == "cuda" else None

        output_dir = Path(cfg.output.dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        start_epoch = 0
        best_loss = float("inf")

        if args.resume:
            print(f"Resuming from {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            best_loss = checkpoint.get("best_loss", float("inf"))

        for epoch in range(start_epoch, cfg.training.epochs):
            print(f"\nEpoch {epoch + 1}/{cfg.training.epochs}")

            train_metrics = train_one_epoch(
                model, train_loader, optimizer, criterion, device, scaler
            )
            val_metrics = evaluate(model, val_loader, criterion, device)

            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]

            print(f"Train Loss: {train_metrics['loss']:.6f} | Val Loss: {val_metrics['loss']:.6f}")
            print(f"  Grid: {train_metrics['grid_loss']:.6f} -> {val_metrics['grid_loss']:.6f}")
            if "row_align_loss" in train_metrics:
                print(f"  Row Align: {train_metrics['row_align_loss']:.6f}")
                print(f"  Col Align: {train_metrics['col_align_loss']:.6f}")
            if "offset_reg_loss" in train_metrics:
                print(f"  Offset Reg: {train_metrics['offset_reg_loss']:.6f}")
            print(f"  LR: {current_lr:.6f}")

            log_metrics = {"learning_rate": current_lr}
            for k, v in train_metrics.items():
                log_metrics[f"train_{k}"] = v
            for k, v in val_metrics.items():
                log_metrics[f"val_{k}"] = v
            mlflow.log_metrics(log_metrics, step=epoch)

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
                "best_loss": best_loss,
                "config": {
                    "model_type": cfg.model.type,
                    "num_rows": cfg.model.num_rows,
                    "num_cols": cfg.model.num_cols,
                    "hidden_size": cfg.model.hidden_size,
                    "offset_scale": cfg.model.get("offset_scale", 0.1),
                    "backbone_name": cfg.model.get("backbone_name", "resnet34"),
                    "num_control_points": cfg.model.get("num_control_points", 16),
                    "image_size": list(cfg.data.image_size),
                },
            }

            torch.save(checkpoint, output_dir / "last.pth")

            if val_metrics["loss"] < best_loss:
                best_loss = val_metrics["loss"]
                torch.save(checkpoint, output_dir / "best.pth")
                print(f"  New best model saved! Loss: {best_loss:.6f}")

        mlflow.log_metric("best_val_loss", best_loss)

        print("\nTraining complete!")
        print(f"Best validation loss: {best_loss:.6f}")


if __name__ == "__main__":
    main()
