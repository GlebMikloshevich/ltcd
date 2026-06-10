
import os
from pathlib import Path
from typing import Iterator

import hydra
import mlflow
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from ltcd.archs.nafunet import NAFUNetBase, NAFUNetSmall
from ltcd.generators.doc_generator import DocumentGenerator
from ltcd.generators.string_generators import get_string_generator
from ltcd.losses.segmentation_losses import get_loss_function


def compute_iou(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    pred_binary = (pred > threshold).float()
    target_binary = (target > threshold).float()

    intersection = (pred_binary * target_binary).sum()
    union = pred_binary.sum() + target_binary.sum() - intersection

    iou = (intersection + 1e-8) / (union + 1e-8)
    return iou.item()


def compute_dice(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    pred_binary = (pred > threshold).float()
    target_binary = (target > threshold).float()

    intersection = (pred_binary * target_binary).sum()
    dice = (2.0 * intersection + 1e-8) / (pred_binary.sum() + target_binary.sum() + 1e-8)

    return dice.item()


class InfiniteDataGenerator:

    def __init__(
        self,
        doc_generator: DocumentGenerator,
        string_generator,
        batch_size: int,
        image_size: tuple[int, int] = (512, 512),
        device: str = "cpu",
        prefetch_batches: int = 2,  # Number of batches to prepare in parallel
        num_workers: int = 4,  # Workers for parallel generation
    ):
        self.doc_generator = doc_generator
        self.string_generator = string_generator
        self.batch_size = batch_size
        self.image_size = image_size
        self.device = device
        self.prefetch_batches = prefetch_batches
        self.num_workers = num_workers

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def generate_batch_data(self) -> list[dict[str, str]]:
        batch_data = []
        for _ in range(self.batch_size):
            sample_data = {}
            for field_name in self.doc_generator.config.fields.keys():
                sample_data[field_name] = self.string_generator.generate()
            batch_data.append(sample_data)
        return batch_data

    def _prepare_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        batch_data = self.generate_batch_data()

        results = self.doc_generator.generate(batch_data, max_workers=self.num_workers)

        images = []
        masks = []

        for scene_image, keypoints, mask in results:
            scene_image_rgb = scene_image.convert("RGB")
            image_tensor = self.transform(scene_image_rgb)
            images.append(image_tensor)

            mask_resized = np.array(
                Image.fromarray(mask).resize(self.image_size, Image.Resampling.BILINEAR)
            )
            mask_tensor = torch.from_numpy(mask_resized).float().unsqueeze(0)
            masks.append(mask_tensor)

        images_batch = torch.stack(images)
        masks_batch = torch.stack(masks)

        return images_batch, masks_batch

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        from concurrent.futures import ThreadPoolExecutor
        from queue import Queue
        import threading

        batch_queue = Queue(maxsize=self.prefetch_batches)

        def producer():
            with ThreadPoolExecutor(max_workers=self.prefetch_batches) as executor:
                futures = []
                while True:
                    while len(futures) < self.prefetch_batches:
                        future = executor.submit(self._prepare_batch)
                        futures.append(future)

                    completed_future = futures.pop(0)
                    batch = completed_future.result()

                    # Put batch in queue (will block if queue is full)
                    batch_queue.put(batch)

        producer_thread = threading.Thread(target=producer, daemon=True)
        producer_thread.start()

        # Consumer: yield batches from queue
        while True:
            images_batch, masks_batch = batch_queue.get()
            yield images_batch.to(self.device), masks_batch.to(self.device)


def train_one_iteration(
    model: nn.Module,
    data_generator: Iterator,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    num_iterations: int,
    device: str,
    accumulation_steps: int = 1,
    grad_clip: float = 0.0,
    use_amp: bool = True,
) -> dict[str, float]:
    """Train for a specified number of iterations."""
    model.train()
    running_loss = 0.0
    running_iou = 0.0
    running_dice = 0.0

    optimizer.zero_grad()

    pbar = tqdm(range(num_iterations), desc="Training")
    for i in pbar:
        imgs, masks = next(data_generator)

        if use_amp:
            with torch.amp.autocast(device_type='cuda' if device == 'cuda' else 'cpu'):
                outputs = model(imgs)
                loss = criterion(outputs, masks) / accumulation_steps
        else:
            outputs = model(imgs)
            loss = criterion(outputs, masks) / accumulation_steps

        scaler.scale(loss).backward()

        if (i + 1) % accumulation_steps == 0:
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        with torch.no_grad():
            iou = compute_iou(outputs, masks)
            dice = compute_dice(outputs, masks)

        running_loss += loss.item() * accumulation_steps
        running_iou += iou
        running_dice += dice

        if i % 10 == 0:
            pbar.set_postfix({
                'loss': f'{loss.item() * accumulation_steps:.4f}',
                'dice': f'{dice:.4f}',
            })

    return {
        "loss": running_loss / num_iterations,
        "iou": running_iou / num_iterations,
        "dice": running_dice / num_iterations,
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    data_generator: Iterator,
    criterion: nn.Module,
    num_iterations: int,
    device: str,
) -> dict[str, float]:
    """Validate the model on generated data."""
    model.eval()
    running_loss = 0.0
    running_iou = 0.0
    running_dice = 0.0

    for _ in tqdm(range(num_iterations), desc="Validating"):
        imgs, masks = next(data_generator)

        outputs = model(imgs)
        loss = criterion(outputs, masks)

        iou = compute_iou(outputs, masks)
        dice = compute_dice(outputs, masks)

        running_loss += loss.item()
        running_iou += iou
        running_dice += dice

    return {
        "loss": running_loss / num_iterations,
        "iou": running_iou / num_iterations,
        "dice": running_dice / num_iterations,
    }


@hydra.main(version_base=None, config_path="../confs", config_name="train_segmentation_infinite")
def main(cfg: DictConfig) -> None:
    torch.multiprocessing.set_start_method("spawn", force=True)
    mlflow.set_experiment(cfg.experiment_name)

    with mlflow.start_run(run_name=cfg.run_name):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

        print("Configuration:")
        print(cfg)
        mlflow.log_params(dict(cfg.params))

        print(f"Loading document generator from: {cfg.params.generator_dir}")
        doc_generator = DocumentGenerator.from_folder(cfg.params.generator_dir)

        string_gen_config = getattr(cfg.params, "string_generator", {})
        string_generator = get_string_generator(
            generator_type=string_gen_config.get("type", "random"),
            **string_gen_config.get("params", {}),
        )
        print(f"Using string generator: {string_gen_config.get('type', 'random')}")

        print("Creating data generators...")
        prefetch_batches = getattr(cfg.params, "prefetch_batches", 2)
        num_workers = getattr(cfg.params, "num_workers", 4)

        print(f"Prefetching {prefetch_batches} batches in parallel")
        print(f"Using {num_workers} workers per batch generation")

        train_generator = InfiniteDataGenerator(
            doc_generator=doc_generator,
            string_generator=string_generator,
            batch_size=cfg.params.batch_size,
            image_size=tuple(cfg.params.image_size),
            device=device,
            prefetch_batches=prefetch_batches,
            num_workers=num_workers,
        )

        val_generator = InfiniteDataGenerator(
            doc_generator=doc_generator,
            string_generator=string_generator,
            batch_size=cfg.params.batch_size,
            image_size=tuple(cfg.params.image_size),
            device=device,
            prefetch_batches=prefetch_batches,
            num_workers=num_workers,
        )

        train_iter = iter(train_generator)
        val_iter = iter(val_generator)

        model_variant = getattr(cfg.params, "model_variant", "base")
        print(f"Creating NAFUNet model: {model_variant}")

        if model_variant == "small":
            model = NAFUNetSmall(in_channels=3, out_channels=1)
        else:  # base
            model = NAFUNetBase(in_channels=3, out_channels=1)

        model = model.to(device)

        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model parameters: {num_params:,}")
        mlflow.log_param("num_parameters", num_params)

        loss_type = getattr(cfg.params, "loss_type", "dice_focal")
        print(f"Using loss: {loss_type}")
        criterion = get_loss_function(loss_type)

        optimizer_type = getattr(cfg.params, "optimizer", "adam")
        weight_decay = getattr(cfg.params, "weight_decay", 0.0)

        if optimizer_type == "adamw":
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=cfg.params.lr,
                weight_decay=weight_decay,
                betas=(0.9, 0.999),
            )
            print("Using AdamW optimizer")
        else:
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=cfg.params.lr,
                weight_decay=weight_decay,
            )
            print("Using Adam optimizer")

        scheduler_type = getattr(cfg.params, "scheduler", "plateau")
        total_steps = cfg.params.num_epochs * cfg.params.steps_per_epoch

        if scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=total_steps,
                eta_min=cfg.params.lr * 0.01,
            )
            print("Using CosineAnnealingLR scheduler")
        elif scheduler_type == "step":
            step_size = getattr(cfg.params, "step_size", 1000)
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=step_size,
                gamma=0.5,
            )
            print("Using StepLR scheduler")
        else:  # plateau
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.5,
                patience=5,
                verbose=True,
            )
            print("Using ReduceLROnPlateau scheduler")

        use_amp = getattr(cfg.params, "use_amp", True)
        scaler = torch.amp.GradScaler(enabled=use_amp)
        print(f"Mixed precision (AMP): {use_amp}")

        accumulation_steps = getattr(cfg.params, "accumulation_steps", 1)
        grad_clip = getattr(cfg.params, "grad_clip", 0.0)
        print(f"Gradient accumulation: {accumulation_steps}")
        print(f"Effective batch size: {cfg.params.batch_size * accumulation_steps}")
        if grad_clip > 0:
            print(f"Gradient clipping: {grad_clip}")

        best_val_loss = float("inf")
        best_val_dice = 0.0
        global_step = 0

        steps_per_epoch = cfg.params.steps_per_epoch
        val_steps = getattr(cfg.params, "val_steps", 100)

        print(f"\nStarting training for {cfg.params.num_epochs} epochs")
        print(f"Steps per epoch: {steps_per_epoch}")
        print(f"Validation steps: {val_steps}")

        for epoch in range(cfg.params.num_epochs):
            print(f"\n{'='*70}")
            print(f"Epoch [{epoch + 1}/{cfg.params.num_epochs}]")
            print(f"{'='*70}")

            train_metrics = train_one_iteration(
                model=model,
                data_generator=train_iter,
                criterion=criterion,
                optimizer=optimizer,
                scaler=scaler,
                num_iterations=steps_per_epoch,
                device=device,
                accumulation_steps=accumulation_steps,
                grad_clip=grad_clip,
                use_amp=use_amp,
            )

            val_metrics = validate(
                model=model,
                data_generator=val_iter,
                criterion=criterion,
                num_iterations=val_steps,
                device=device,
            )

            if scheduler_type == "plateau":
                scheduler.step(val_metrics["loss"])
            elif scheduler_type == "cosine" or scheduler_type == "step":
                for _ in range(steps_per_epoch):
                    scheduler.step()

            current_lr = optimizer.param_groups[0]["lr"]
            global_step += steps_per_epoch

            mlflow.log_metrics(
                {
                    "train_loss": train_metrics["loss"],
                    "train_iou": train_metrics["iou"],
                    "train_dice": train_metrics["dice"],
                    "val_loss": val_metrics["loss"],
                    "val_iou": val_metrics["iou"],
                    "val_dice": val_metrics["dice"],
                    "learning_rate": current_lr,
                },
                step=global_step,
            )

            print(f"\nTrain Loss: {train_metrics['loss']:.4f} | Val Loss: {val_metrics['loss']:.4f} | LR: {current_lr:.6f}")
            print(f"  IoU:  {train_metrics['iou']:.4f} -> {val_metrics['iou']:.4f}")
            print(f"  Dice: {train_metrics['dice']:.4f} -> {val_metrics['dice']:.4f}")

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                os.makedirs("checkpoints", exist_ok=True)
                model_path = f"checkpoints/{cfg.model_name}_best.pth"
                torch.save(
                    {
                        "epoch": epoch,
                        "global_step": global_step,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_loss": val_metrics["loss"],
                        "val_dice": val_metrics["dice"],
                        "config": dict(cfg.params),
                    },
                    model_path,
                )
                print(f"✓ Saved best model (loss) with val_loss: {val_metrics['loss']:.4f}")

            if val_metrics["dice"] > best_val_dice:
                best_val_dice = val_metrics["dice"]
                dice_model_path = f"checkpoints/{cfg.model_name}_best_dice.pth"
                torch.save(
                    {
                        "epoch": epoch,
                        "global_step": global_step,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_loss": val_metrics["loss"],
                        "val_dice": val_metrics["dice"],
                        "config": dict(cfg.params),
                    },
                    dice_model_path,
                )
                print(f"✓ Saved best model (dice) with val_dice: {val_metrics['dice']:.4f}")

            if (epoch + 1) % getattr(cfg.params, "save_interval", 10) == 0:
                checkpoint_path = f"checkpoints/{cfg.model_name}_epoch_{epoch+1}.pth"
                torch.save(
                    {
                        "epoch": epoch,
                        "global_step": global_step,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "config": dict(cfg.params),
                    },
                    checkpoint_path,
                )
                print(f"✓ Saved checkpoint at epoch {epoch+1}")

        final_model_path = f"checkpoints/{cfg.model_name}_final.pth"
        torch.save(model.state_dict(), final_model_path)

        print(f"\n{'='*70}")
        print("Training completed!")
        print(f"Best validation loss: {best_val_loss:.4f}")
        print(f"Best validation Dice: {best_val_dice:.4f}")
        print(f"Total steps: {global_step}")
        print(f"{'='*70}")


if __name__ == "__main__":
    main()
