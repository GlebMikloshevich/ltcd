
import os
from pathlib import Path

import hydra
import mlflow
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ltcd.archs.first_row_regressor import FirstRowRegressor
from ltcd.datasets.keypoint_dataset import KeypointDatasetSimple


class FirstRowDataset(Dataset):
    """
    Dataset wrapper that extracts only the first row of keypoints.

    Keypoints are stored column-major:
    [col0_row0, col0_row1, ..., col0_rowN, col1_row0, col1_row1, ...]

    Args:
        base_dataset: Underlying keypoint dataset
        num_rows: Number of rows in the table grid
        num_cols: Number of columns in the table grid
    """

    def __init__(
        self,
        base_dataset: KeypointDatasetSimple,
        num_rows: int = 18,
        num_cols: int = 5,
    ) -> None:
        self.base_dataset = base_dataset
        self.num_rows = num_rows
        self.num_cols = num_cols

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get sample with only first row keypoints.

        Returns:
            image: Image tensor [3, H, W]
            first_row_keypoints: First row keypoints [num_cols, 2]
        """
        image, all_keypoints = self.base_dataset[idx]

        # Column-major: for row 0, take indices 0, num_rows, 2*num_rows, ...
        indices = [col * self.num_rows for col in range(self.num_cols)]
        first_row = all_keypoints[indices]

        return image, first_row


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    scaler: torch.amp.GradScaler,
) -> dict[str, float]:
    """Train for one epoch."""
    model.train()
    running_loss = 0.0

    for images, keypoints in tqdm(loader, desc="Training"):
        images = images.to(device)
        keypoints = keypoints.to(device)

        optimizer.zero_grad()

        with torch.amp.autocast(device_type="cuda", enabled=(device == "cuda")):
            pred = model(images)
            loss = F.mse_loss(pred, keypoints)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

    return {"loss": running_loss / len(loader)}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: str,
) -> dict[str, float]:
    """Evaluate the model."""
    model.eval()
    running_loss = 0.0
    running_dist = 0.0

    for images, keypoints in tqdm(loader, desc="Evaluating"):
        images = images.to(device)
        keypoints = keypoints.to(device)

        pred = model(images)
        loss = F.mse_loss(pred, keypoints)

        running_loss += loss.item()

        dist = torch.norm(pred - keypoints, dim=-1).mean()
        running_dist += dist.item()

    n = len(loader)
    return {
        "loss": running_loss / n,
        "mean_dist": running_dist / n,
    }


@hydra.main(version_base=None, config_path="../confs", config_name="train_first_row")
def main(cfg: DictConfig) -> None:
    torch.multiprocessing.set_start_method("spawn", force=True)

    mlflow.set_experiment(cfg.experiment_name)

    with mlflow.start_run(run_name=cfg.run_name):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

        print("Configuration:")
        print(cfg)
        mlflow.log_params(dict(cfg.params))

        dataset_dir = Path(cfg.params.dataset_dir)
        train_dir = dataset_dir / "train"
        test_dir = dataset_dir / "test"

        print("\nLoading datasets...")
        image_size = tuple(cfg.params.image_size)

        train_base = KeypointDatasetSimple(
            train_dir,
            image_size=image_size,
            num_keypoints=cfg.params.num_keypoints,
        )
        test_base = KeypointDatasetSimple(
            test_dir,
            image_size=image_size,
            num_keypoints=cfg.params.num_keypoints,
        )

        train_dataset = FirstRowDataset(
            train_base,
            num_rows=cfg.params.num_rows,
            num_cols=cfg.params.num_cols,
        )
        test_dataset = FirstRowDataset(
            test_base,
            num_rows=cfg.params.num_rows,
            num_cols=cfg.params.num_cols,
        )

        print(f"Train samples: {len(train_dataset)}")
        print(f"Test samples: {len(test_dataset)}")

        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.params.batch_size,
            shuffle=True,
            num_workers=cfg.params.num_workers,
            pin_memory=True,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=cfg.params.batch_size,
            shuffle=False,
            num_workers=cfg.params.num_workers // 2,
            pin_memory=True,
        )

        model = FirstRowRegressor(
            hidden_size=cfg.params.hidden_size,
            num_cols=cfg.params.num_cols,
            dropout=cfg.params.dropout,
            use_pretrained=cfg.params.use_pretrained,
        )

        model = model.to(device)

        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model parameters: {num_params:,}")
        mlflow.log_param("num_parameters", num_params)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.params.lr,
            weight_decay=cfg.params.weight_decay,
        )

        scheduler_type = getattr(cfg.params, "scheduler", "cosine")
        print(f"Using scheduler: {scheduler_type}")

        if scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=cfg.params.epochs,
                eta_min=cfg.params.lr * 0.01,
            )
        elif scheduler_type == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=cfg.params.step_size,
                gamma=0.5,
            )
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.5,
                patience=5,
            )

        scaler = torch.amp.GradScaler()

        os.makedirs("checkpoints", exist_ok=True)
        best_val_loss = float("inf")

        print("\n" + "=" * 60)
        print("Starting training (First Row Regression with MSE)")
        print("=" * 60)

        for epoch in range(cfg.params.epochs):
            print(f"\nEpoch [{epoch + 1}/{cfg.params.epochs}]")

            train_metrics = train_one_epoch(model, train_loader, optimizer, device, scaler)
            val_metrics = evaluate(model, test_loader, device)

            if scheduler_type == "plateau":
                scheduler.step(val_metrics["loss"])
            else:
                scheduler.step()

            current_lr = optimizer.param_groups[0]["lr"]
            mlflow.log_metrics(
                {
                    "train_loss": train_metrics["loss"],
                    "val_loss": val_metrics["loss"],
                    "val_mean_dist": val_metrics["mean_dist"],
                    "learning_rate": current_lr,
                },
                step=epoch,
            )

            print(f"Train Loss: {train_metrics['loss']:.6f} | Val Loss: {val_metrics['loss']:.6f}")
            print(f"  Mean Dist: {val_metrics['mean_dist']:.6f} | LR: {current_lr:.6f}")

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                model_path = f"checkpoints/{cfg.model_name}_best.pth"
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_loss": val_metrics["loss"],
                        "config": {
                            "model_type": model_type,
                            "hidden_size": cfg.params.hidden_size,
                            "num_cols": cfg.params.num_cols,
                            "dropout": cfg.params.dropout,
                        },
                    },
                    model_path,
                )
                print(f"Saved best model with val_loss: {val_metrics['loss']:.6f}")
                mlflow.log_artifact(model_path)

        final_path = f"checkpoints/{cfg.model_name}_final.pth"
        torch.save(model.state_dict(), final_path)
        mlflow.log_artifact(final_path)
        mlflow.log_metric("best_val_loss", best_val_loss)

        print("\n" + "=" * 60)
        print(f"Training complete! Best val loss: {best_val_loss:.6f}")
        print("=" * 60)


if __name__ == "__main__":
    main()
