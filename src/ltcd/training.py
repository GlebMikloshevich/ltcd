"""Shared training/inference infrastructure: EMA, EarlyStopping, the
grid-aware dataset wrapper, and the canonical inference preprocessor."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from ltcd.datasets.keypoint_dataset import KeypointDatasetSimple


# ImageNet stats, matching what KeypointDataset uses during training. Inference
# must use the same to avoid silent distribution shift between the two paths.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def preprocess_image(
    image_path: str,
    image_size: tuple[int, int] = (512, 512),
) -> tuple[torch.Tensor, tuple[int, int]]:
    """Load + resize + ImageNet-normalise an image for model inference.

    Returns a `[1, 3, H, W]` tensor and the *original* PIL size as `(W, H)` —
    the order PIL exposes — so callers can scale predictions back into the
    source pixel grid.
    """
    image = Image.open(image_path).convert("RGB")
    original_size = image.size

    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])
    tensor = transform(image).unsqueeze(0)
    return tensor, original_size


class EMA:
    """Exponential moving average of model parameters.

    Used for evaluation: training updates the live model, but val/test use
    the EMA shadow weights. `apply_shadow()` swaps them in, `restore()` puts
    the live weights back.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.model = model
        self.decay = decay
        self.shadow: dict[str, torch.Tensor] = {}
        self.backup: dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self) -> None:
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = self.decay * self.shadow[name] + (1.0 - self.decay) * param.data

    def apply_shadow(self) -> None:
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self) -> None:
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


class EarlyStopping:
    """Stop training when val loss stops improving by `min_delta` for `patience` epochs."""

    def __init__(self, patience: int = 10, min_delta: float = 0.0) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.should_stop = False

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


class GridAugmentedDataset(Dataset):
    """Wrap KeypointDatasetSimple with grid-aware geometric + colour augmentation.

    Returns a dict with the same shape every script downstream consumes:
      - image:          augmented image tensor [3, H, W]
      - grid:           keypoints reshaped to [num_rows, num_cols, 2]
      - keypoints_flat: keypoints as a flat list [num_rows * num_cols, 2]

    Column-major reshape (`view(num_cols, num_rows, 2).permute(1, 0, 2)`)
    matches the dataset annotation order.
    """

    def __init__(
        self,
        base_dataset: KeypointDatasetSimple,
        num_rows: int = 18,
        num_cols: int = 5,
        augment: bool = False,
        flip_prob: float = 0.5,
        rotation_range: float = 5.0,
        scale_range: tuple[float, float] = (0.95, 1.05),
    ) -> None:
        self.base_dataset = base_dataset
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.augment = augment
        self.flip_prob = flip_prob
        self.rotation_range = rotation_range
        self.scale_range = scale_range

        self.color_jitter = (
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
            if augment
            else None
        )

    def __len__(self) -> int:
        return len(self.base_dataset)

    def _apply_geometric_augmentation(
        self,
        image: torch.Tensor,
        keypoints: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if np.random.random() < self.flip_prob:
            image = TF.hflip(image)
            keypoints = keypoints.clone()
            keypoints[:, 0] = -keypoints[:, 0]

            # Horizontal flip also reverses column order — without this the
            # leftmost keypoint after the flip would still carry the rightmost
            # column's label, breaking row/column alignment losses.
            grid = keypoints.view(self.num_cols, self.num_rows, 2)
            grid = torch.flip(grid, dims=[0])
            keypoints = grid.view(-1, 2)

        if self.rotation_range > 0:
            angle = np.random.uniform(-self.rotation_range, self.rotation_range)
            image = TF.rotate(image, angle, fill=0)

            # Negate the angle: TF.rotate spins the image counter-clockwise,
            # so the keypoints must rotate clockwise to compensate.
            angle_rad = np.deg2rad(-angle)
            cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
            rot_matrix = torch.tensor(
                [[cos_a, -sin_a], [sin_a, cos_a]],
                dtype=keypoints.dtype,
            )
            keypoints = keypoints @ rot_matrix.T

        if self.scale_range[0] < self.scale_range[1]:
            scale = np.random.uniform(self.scale_range[0], self.scale_range[1])
            # Clamp to [-1, 1] so post-scale keypoints don't fall outside the
            # grid_sample valid range used by downstream samplers.
            keypoints = torch.clamp(keypoints * scale, -1, 1)

        return image, keypoints

    def __getitem__(self, idx: int) -> dict:
        image, keypoints = self.base_dataset[idx]

        if self.augment:
            image, keypoints = self._apply_geometric_augmentation(image, keypoints)

        if self.color_jitter is not None:
            image = self.color_jitter(image)

        grid = keypoints.view(self.num_cols, self.num_rows, 2).permute(1, 0, 2)

        return {
            "image": image,
            "grid": grid,
            "keypoints_flat": keypoints,
        }
