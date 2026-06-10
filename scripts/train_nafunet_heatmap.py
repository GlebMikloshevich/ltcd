
import math
import os
from pathlib import Path

import hydra
import mlflow
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from ltcd.archs.nafunet import NAFUNetBase, NAFUNetSmall
from ltcd.generators.utils import generate_gaussian
from ltcd.losses.segmentation_losses import get_loss_function


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

def find_corner_indices(kps: np.ndarray) -> tuple[int, int, int, int]:
    x, y = kps[:, 0], kps[:, 1]
    tl = int(np.argmin(x + y))
    tr = int(np.argmax(x - y))
    bl = int(np.argmin(x - y))
    br = int(np.argmax(x + y))
    return tl, tr, bl, br


def make_heatmap(
    kps_px: np.ndarray,   # [K, 2] absolute (x, y) in pixels
    height: int,
    width: int,
    sigma: float = 5.0,
) -> np.ndarray:
    """Render K Gaussian blobs into a [H, W] float32 heatmap in [0, 1]."""
    kfs = max(int(min(height, width) / 31), 3)
    gaussian = generate_gaussian(kfs, kfs, sigma=sigma)
    mask = np.zeros((height, width), dtype=np.float32)
    half = kfs // 2
    for kx, ky in kps_px:
        kx_i, ky_i = int(round(kx)), int(round(ky))
        gsy, gsx = ky_i - half, kx_i - half
        my1 = max(0, gsy);   my2 = min(height, gsy + kfs)
        mx1 = max(0, gsx);   mx2 = min(width,  gsx + kfs)
        gy1 = my1 - gsy;     gx1 = mx1 - gsx
        mask[my1:my2, mx1:mx2] = np.maximum(
            mask[my1:my2, mx1:mx2],
            gaussian[gy1:gy1 + (my2 - my1), gx1:gx1 + (mx2 - mx1)],
        )
    return mask


def init_output_bias_for_prior(model: nn.Module, prior: float = 0.01) -> None:
    final_conv = model.output[-1]
    assert isinstance(final_conv, nn.Conv2d), (
        f"Expected Conv2d as final layer, got {type(final_conv).__name__}"
    )
    bias_value = -math.log((1.0 - prior) / prior)
    nn.init.constant_(final_conv.bias, bias_value)
    nn.init.normal_(final_conv.weight, std=0.01)


def heatmap_to_rgb(hm: np.ndarray) -> np.ndarray:
    v = hm.clip(0, 1)
    r = np.clip(v * 3.0,       0, 1)
    g = np.clip(v * 3.0 - 1.0, 0, 1)
    b = np.clip(v * 3.0 - 2.0, 0, 1)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

class HeatmapDataset(Dataset):
    """
    Loads image + keypoint files and builds heatmap targets on-the-fly.

    Expected directory layout:
        <root>/images/*.png
        <root>/keypoints/*.npy   — [N, 2] absolute pixel coords (x, y)

    Returns:
        image   : [3, H, W]  ImageNet-normalised
        heatmap : [1, H, W]  single-channel merged heatmap (both modes)
    """

    def __init__(
        self,
        dataset_dir: str | Path,
        mode: str = "corners",
        image_size: tuple[int, int] = (512, 512),
        sigma: float = 5.0,
        use_precomputed_masks: bool = False,
    ) -> None:
        assert mode in ("corners", "all_points"), f"Unknown mode: {mode}"
        if use_precomputed_masks and mode != "all_points":
            raise ValueError(
                "use_precomputed_masks is only valid with mode='all_points' — "
                "precomputed masks contain all keypoints, not just corners."
            )
        dataset_dir = Path(dataset_dir)

        img_dir  = dataset_dir / "images"
        kp_dir   = dataset_dir / "keypoints"
        mask_dir = dataset_dir / "masks"

        stems = {p.stem for p in kp_dir.glob("*.npy")}
        if use_precomputed_masks:
            mask_stems = {p.stem for p in mask_dir.glob("*.npy")}
            stems &= mask_stems
            if not stems:
                raise ValueError(f"No image/keypoint/mask triples found in {dataset_dir}")

        self.image_paths    = sorted(p for p in img_dir.glob("*.png") if p.stem in stems)
        self.keypoint_paths = [kp_dir / f"{p.stem}.npy" for p in self.image_paths]
        self.mask_paths     = (
            [mask_dir / f"{p.stem}.npy" for p in self.image_paths]
            if use_precomputed_masks else None
        )

        if len(self.image_paths) == 0:
            raise ValueError(f"No matched image/keypoint pairs in {dataset_dir}")

        self.mode                  = mode
        self.image_size            = image_size  # (H, W)
        self.sigma                 = sigma
        self.use_precomputed_masks = use_precomputed_masks

        self.to_tensor = transforms.ToTensor()
        self.resize    = transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR)
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    @property
    def out_channels(self) -> int:
        return 1

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        H, W = self.image_size

        image = Image.open(self.image_paths[idx]).convert("RGB")
        orig_w, orig_h = image.size
        image = self.normalize(self.to_tensor(self.resize(image)))

        if self.use_precomputed_masks:
            mask = np.load(self.mask_paths[idx]).astype(np.float32)  # [H0, W0]
            if mask.shape != (H, W):
                mask_pil = Image.fromarray(mask).resize((W, H), resample=Image.BILINEAR)
                mask = np.array(mask_pil, dtype=np.float32)
            hm = mask
        else:
            kps = np.load(self.keypoint_paths[idx]).astype(np.float32)  # [N, 2] (x, y)
            kps[:, 0] *= W / orig_w
            kps[:, 1] *= H / orig_h

            if self.mode == "corners":
                tl, tr, bl, br = find_corner_indices(kps)
                corner_kps = kps[[tl, tr, bl, br]]
                hm = make_heatmap(corner_kps, H, W, self.sigma)
            else:
                hm = make_heatmap(kps, H, W, self.sigma)

        heatmap = torch.from_numpy(hm).unsqueeze(0)  # [1, H, W]
        return image, heatmap


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

def compute_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute RMSE, IoU and Dice across all channels."""
    pred_sig = torch.sigmoid(pred)
    rmse = torch.sqrt(nn.functional.mse_loss(pred_sig, target)).item()

    pred_bin   = (pred_sig > threshold).float()
    target_bin = (target   > threshold).float()
    intersection = (pred_bin * target_bin).sum()
    union = pred_bin.sum() + target_bin.sum() - intersection
    iou  = ((intersection + 1e-8) / (union + 1e-8)).item()
    dice = ((2 * intersection + 1e-8) / (pred_bin.sum() + target_bin.sum() + 1e-8)).item()

    return {"rmse": rmse, "iou": iou, "dice": dice}


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

def _apply_loss(
    criterion: nn.Module,
    pred: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Apply loss per channel and average (handles C=1 and C=4)."""
    B, C, H, W = pred.shape
    if C == 1:
        return criterion(pred, target)
    # reshape (not view) because model output may be non-contiguous
    return criterion(pred.reshape(B * C, 1, H, W), target.reshape(B * C, 1, H, W))


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

_DENORM_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_DENORM_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def overlay_heatmap(image_np: np.ndarray, hm: np.ndarray, alpha: float = 0.6) -> np.ndarray:
    color = heatmap_to_rgb(hm).astype(np.float32)          # [H,W,3]
    weight = hm[:, :, None]                                 # [H,W,1] — transparent where hm≈0
    out = image_np.astype(np.float32) * (1 - alpha * weight) + color * (alpha * weight)
    return out.clip(0, 255).astype(np.uint8)


@torch.no_grad()
def save_visualizations(
    model: nn.Module,
    dataset: Dataset,
    output_dir: Path,
    epoch: int,
    device: str,
    split: str = "train",
    num_samples: int = 3,
) -> None:
    """
    Save a grid of [image | image+gt overlay | image+pred overlay] for num_samples.
    """
    model.eval()
    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(exist_ok=True)

    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)

    rows = []
    for idx in indices:
        image, target = dataset[idx]                               # [3,H,W], [C,H,W]
        pred = torch.sigmoid(
            model(image.unsqueeze(0).to(device))[0].cpu()
        )                                                          # [C, H, W]

        # Denormalize image → uint8
        image_np = ((image * _DENORM_STD + _DENORM_MEAN).clamp(0, 1)
                    .permute(1, 2, 0).numpy() * 255).astype(np.uint8)

        # Max-project channels → [H, W]
        target_hm = target.max(dim=0).values.numpy()
        pred_hm   = pred.max(dim=0).values.numpy()

        row = np.concatenate(
            [image_np,
             overlay_heatmap(image_np, target_hm),
             overlay_heatmap(image_np, pred_hm)],
            axis=1,
        )
        rows.append(row)

    grid = np.concatenate(rows, axis=0)
    Image.fromarray(grid).save(vis_dir / f"epoch_{epoch:04d}_{split}.png")
    model.train()


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    scaler: torch.amp.GradScaler,
    use_amp: bool = False,
    clip_grad: float | None = None,
) -> dict[str, float]:
    model.train()
    totals: dict[str, float] = {"loss": 0.0, "rmse": 0.0, "iou": 0.0, "dice": 0.0}

    for imgs, masks in tqdm(loader, desc="Training"):
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()

        with torch.amp.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
            preds = model(imgs)
            loss  = _apply_loss(criterion, preds, masks)

        scaler.scale(loss).backward()
        if clip_grad is not None:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            m = compute_metrics(preds, masks)
        totals["loss"] += loss.item()
        for k in ("rmse", "iou", "dice"):
            totals[k] += m[k]

    n = len(loader)
    return {k: v / n for k, v in totals.items()}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {"loss": 0.0, "rmse": 0.0, "iou": 0.0, "dice": 0.0}

    for imgs, masks in tqdm(loader, desc="Evaluating"):
        imgs, masks = imgs.to(device), masks.to(device)
        preds = model(imgs)
        loss  = _apply_loss(criterion, preds, masks)
        m     = compute_metrics(preds, masks)
        totals["loss"] += loss.item()
        for k in ("rmse", "iou", "dice"):
            totals[k] += m[k]

    n = len(loader)
    return {k: v / n for k, v in totals.items()}


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="../confs", config_name="train_nafunet_heatmap")
def main(cfg: DictConfig) -> None:
    torch.multiprocessing.set_start_method("spawn", force=True)
    mlflow.set_experiment(cfg.experiment_name)

    with mlflow.start_run(run_name=cfg.run_name):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        print(f"Mode: {cfg.params.mode}")
        mlflow.log_params(dict(cfg.params))

        # Checkpoint directory: ./checkpoints/{model_name}/{run_name}/
        ckpt_dir = Path("checkpoints") / cfg.model_name / cfg.run_name
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, ckpt_dir / "config.yaml")

        dataset_dir = Path(cfg.params.dataset_dir)
        image_size  = tuple(cfg.params.image_size)
        mode        = cfg.params.mode

        ds_mode       = str(mode)
        ds_image_size = (int(image_size[0]), int(image_size[1]))
        ds_sigma      = float(cfg.params.get("sigma", 5.0))
        ds_use_masks  = bool(cfg.params.get("use_precomputed_masks", False))

        train_dataset = HeatmapDataset(
            dataset_dir / "train",
            mode=ds_mode, image_size=ds_image_size, sigma=ds_sigma,
            use_precomputed_masks=ds_use_masks,
        )
        val_dataset = HeatmapDataset(
            dataset_dir / "test",
            mode=ds_mode, image_size=ds_image_size, sigma=ds_sigma,
            use_precomputed_masks=ds_use_masks,
        )
        print(f"Use precomputed masks: {ds_use_masks}")
        print(f"Train: {len(train_dataset)}  Val: {len(val_dataset)}")
        print(f"Output channels: {train_dataset.out_channels}")

        num_workers = cfg.params.get("num_workers", 4)
        train_loader = DataLoader(
            train_dataset, batch_size=cfg.params.batch_size,
            shuffle=True, num_workers=num_workers, pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=cfg.params.batch_size,
            shuffle=False, num_workers=num_workers, pin_memory=True,
        )

        out_ch = train_dataset.out_channels
        model_name = cfg.get("model_name", "nafunet_small")
        model_registry = {
            "nafunet_small": NAFUNetSmall,
            "nafunet_base": NAFUNetBase,
        }
        if model_name not in model_registry:
            raise ValueError(
                f"Unknown model_name '{model_name}'. Options: {list(model_registry)}",
            )
        model = model_registry[model_name](out_channels=out_ch)

        # Bias the final layer toward "all background" so the model doesn't
        # park at sigmoid(0)=0.5 while gradients fight to escape.
        prior = float(cfg.params.get("output_prior", 0.01))
        init_output_bias_for_prior(model, prior=prior)
        print(f"Output bias initialised for prior={prior} → bias={-math.log((1 - prior) / prior):.3f}")

        model = model.to(device)

        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model: {model_name}  params: {num_params:,}")
        mlflow.log_param("num_parameters", num_params)

        loss_type = cfg.params.get("loss_type", "dice_focal")
        criterion = get_loss_function(loss_type)
        print(f"Loss: {loss_type}")

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg.params.lr,
            weight_decay=cfg.params.get("weight_decay", 0.0),
        )
        scheduler_type = cfg.params.get("scheduler", "cosine")
        warmup_epochs  = cfg.params.get("warmup_epochs", 0)

        if scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=cfg.params.epochs - warmup_epochs, eta_min=1e-6,
            )
        elif scheduler_type == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=cfg.params.get("step_size", 10), gamma=0.5,
            )
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=5,
            )

        use_amp   = cfg.params.get("use_amp", False)
        clip_grad = cfg.params.get("clip_grad", None)
        scaler    = torch.amp.GradScaler(enabled=use_amp)

        vis_every     = cfg.params.get("visualize_every", 5)
        best_val_loss = float("inf")
        best_val_rmse = float("inf")

        def _checkpoint(filename: str) -> None:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_m["loss"],
                "val_rmse": val_m["rmse"],
            }, ckpt_dir / filename)

        for epoch in range(cfg.params.epochs):
            print(f"\nEpoch [{epoch + 1}/{cfg.params.epochs}]")

            if epoch < warmup_epochs:
                warmup_lr = cfg.params.lr * (epoch + 1) / warmup_epochs
                for pg in optimizer.param_groups:
                    pg["lr"] = warmup_lr
                print(f"  Warmup LR: {warmup_lr:.6f}")

            train_m = train_one_epoch(model, train_loader, criterion, optimizer,
                                      device, scaler, use_amp, clip_grad)
            val_m   = evaluate(model, val_loader, criterion, device)

            if epoch >= warmup_epochs:
                if scheduler_type == "plateau":
                    scheduler.step(val_m["loss"])
                else:
                    scheduler.step()

            current_lr = optimizer.param_groups[0]["lr"]
            mlflow.log_metrics({
                "train_loss": train_m["loss"], "train_rmse": train_m["rmse"],
                "train_iou":  train_m["iou"],  "train_dice": train_m["dice"],
                "val_loss":   val_m["loss"],   "val_rmse":   val_m["rmse"],
                "val_iou":    val_m["iou"],    "val_dice":   val_m["dice"],
                "learning_rate": current_lr,
            }, step=epoch)

            print(f"  Loss  train={train_m['loss']:.4f}  val={val_m['loss']:.4f}  lr={current_lr:.2e}")
            print(f"  RMSE  train={train_m['rmse']:.4f}  val={val_m['rmse']:.4f}")
            print(f"  Dice  train={train_m['dice']:.4f}  val={val_m['dice']:.4f}")

            _checkpoint("last.pth")

            if val_m["loss"] < best_val_loss:
                best_val_loss = val_m["loss"]
                _checkpoint("best_loss.pth")
                print(f"  → best loss {best_val_loss:.4f}")

            if val_m["rmse"] < best_val_rmse:
                best_val_rmse = val_m["rmse"]
                _checkpoint("best_rmse.pth")
                print(f"  → best rmse {best_val_rmse:.4f}")

            if vis_every > 0 and (epoch + 1) % vis_every == 0:
                save_visualizations(model, train_dataset, ckpt_dir,
                                    epoch + 1, device, split="train")
                save_visualizations(model, val_dataset,   ckpt_dir,
                                    epoch + 1, device, split="val")

        print(f"\nDone.  best_val_loss={best_val_loss:.4f}  best_val_rmse={best_val_rmse:.4f}")
        mlflow.log_metrics({"best_val_loss": best_val_loss, "best_val_rmse": best_val_rmse})


if __name__ == "__main__":
    main()
