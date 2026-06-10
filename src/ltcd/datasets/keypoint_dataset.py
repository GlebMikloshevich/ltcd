from pathlib import Path, PurePath

import numpy as np
import torch

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class KeypointDataset(Dataset):
    """Keypoint + heatmap dataset for segmentation-based predictors.

    Expects a directory layout of `images/`, `masks/` and `keypoints/`, with
    matching filenames. Coordinates are normalised to [-1, 1] so they can be
    fed straight into grid_sample.
    """

    def __init__(
        self,
        dataset_dir: PurePath,
        image_size: tuple[int, int] = (512, 512),
        num_keypoints: int = 75,
    ) -> None:
        dataset_dir = Path(dataset_dir)
        image_dir = dataset_dir / "images"
        mask_dir = dataset_dir / "masks"
        keypoints_dir = dataset_dir / "keypoints"

        self.image_paths = sorted(image_dir.glob("*.png"))
        self.mask_paths = sorted(mask_dir.glob("*.npy"))
        self.keypoint_paths = sorted(keypoints_dir.glob("*.npy"))

        # Sanity-check at construction time: a missing or extra file in any of
        # the three folders silently desynchronises the lists, which would only
        # show up as bizarre training behaviour later.
        assert len(self.image_paths) == len(self.mask_paths) == len(self.keypoint_paths), (
            f"Mismatch: {len(self.image_paths)} images, "
            f"{len(self.mask_paths)} masks, {len(self.keypoint_paths)} keypoints"
        )

        self.image_size = image_size
        self.num_keypoints = num_keypoints

        self.to_tensor = transforms.ToTensor()
        self.resize = transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR)
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image = Image.open(self.image_paths[idx]).convert("RGB")
        original_size = image.size

        keypoints = np.load(self.keypoint_paths[idx]).astype(np.float32)
        heatmap = np.load(self.mask_paths[idx]).astype(np.float32)

        image = self.resize(image)

        # PIL resize: float arrays must be wrapped in an Image first; numpy's
        # resize doesn't interpolate.
        heatmap_pil = Image.fromarray(heatmap).resize(self.image_size, resample=Image.BILINEAR)
        heatmap = np.array(heatmap_pil, dtype=np.float32)

        # Keypoints live in original-image pixel space; rescale to the resized
        # frame before normalising to [-1, 1].
        scale_x = self.image_size[1] / original_size[0]
        scale_y = self.image_size[0] / original_size[1]
        keypoints[:, 0] *= scale_x
        keypoints[:, 1] *= scale_y

        keypoints_normalized = keypoints.copy()
        keypoints_normalized[:, 0] = (keypoints[:, 0] / self.image_size[1]) * 2 - 1
        keypoints_normalized[:, 1] = (keypoints[:, 1] / self.image_size[0]) * 2 - 1

        # Zero-padding masquerades as a valid keypoint at the centre of the
        # image after the [-1, 1] mapping. Models that ingest this dataset rely
        # on either a fixed num_keypoints or an explicit padding mask.
        if len(keypoints_normalized) < self.num_keypoints:
            padding = np.zeros((self.num_keypoints - len(keypoints_normalized), 2), dtype=np.float32)
            keypoints_normalized = np.vstack([keypoints_normalized, padding])
        elif len(keypoints_normalized) > self.num_keypoints:
            keypoints_normalized = keypoints_normalized[:self.num_keypoints]

        image = self.normalize(self.to_tensor(image))
        keypoints_tensor = torch.from_numpy(keypoints_normalized)
        heatmap_tensor = torch.from_numpy(heatmap).unsqueeze(0)

        return image, keypoints_tensor, heatmap_tensor


class KeypointDatasetSimple(Dataset):
    """Keypoints-only variant used by the coordinate-regression predictors.

    Matches images to keypoint files by filename stem instead of sorted index,
    so missing keypoint files just shrink the dataset rather than corrupting
    the alignment.
    """

    def __init__(
        self,
        dataset_dir: PurePath,
        image_size: tuple[int, int] = (512, 512),
        num_keypoints: int = 75,
    ) -> None:
        dataset_dir = Path(dataset_dir)
        image_dir = dataset_dir / "images"
        keypoints_dir = dataset_dir / "keypoints"

        image_paths = sorted(list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg")))

        self.image_paths: list[Path] = []
        self.keypoint_paths: list[Path] = []
        for img_path in image_paths:
            keypoint_path = keypoints_dir / f"{img_path.stem}.npy"
            if keypoint_path.exists():
                self.image_paths.append(img_path)
                self.keypoint_paths.append(keypoint_path)

        if not self.image_paths:
            raise ValueError(f"No matching image-keypoint pairs found in {dataset_dir}")

        print(f"Found {len(self.image_paths)} image-keypoint pairs")

        self.image_size = image_size
        self.num_keypoints = num_keypoints

        self.to_tensor = transforms.ToTensor()
        self.resize = transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR)
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = Image.open(self.image_paths[idx]).convert("RGB")
        original_size = image.size

        keypoints = np.load(self.keypoint_paths[idx]).astype(np.float32)

        image = self.resize(image)

        scale_x = self.image_size[1] / original_size[0]
        scale_y = self.image_size[0] / original_size[1]
        keypoints[:, 0] *= scale_x
        keypoints[:, 1] *= scale_y

        keypoints_normalized = keypoints.copy()
        keypoints_normalized[:, 0] = (keypoints[:, 0] / self.image_size[1]) * 2 - 1
        keypoints_normalized[:, 1] = (keypoints[:, 1] / self.image_size[0]) * 2 - 1

        # Same zero-pad caveat as KeypointDataset.
        if len(keypoints_normalized) < self.num_keypoints:
            padding = np.zeros((self.num_keypoints - len(keypoints_normalized), 2), dtype=np.float32)
            keypoints_normalized = np.vstack([keypoints_normalized, padding])
        elif len(keypoints_normalized) > self.num_keypoints:
            keypoints_normalized = keypoints_normalized[:self.num_keypoints]

        image = self.normalize(self.to_tensor(image))
        keypoints_tensor = torch.from_numpy(keypoints_normalized)

        return image, keypoints_tensor
