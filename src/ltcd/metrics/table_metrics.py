import numpy as np
import torch


def compute_pck(
    pred_keypoints: np.ndarray | torch.Tensor,
    gt_keypoints: np.ndarray | torch.Tensor,
    image_size: tuple[int, int],
    threshold: float = 0.05,
) -> float:
    """Percentage of keypoints with L2 error below `threshold` × image diagonal.

    Zero-padded GT entries are filtered out before the comparison —
    KeypointDataset pads short sequences with zeros, and those would otherwise
    be counted as "correct" predictions at the image centre.
    """
    if isinstance(pred_keypoints, torch.Tensor):
        pred_keypoints = pred_keypoints.cpu().numpy()
    if isinstance(gt_keypoints, torch.Tensor):
        gt_keypoints = gt_keypoints.cpu().numpy()

    if pred_keypoints.ndim == 3:
        pred_keypoints = pred_keypoints.reshape(-1, 2)
        gt_keypoints = gt_keypoints.reshape(-1, 2)

    valid_mask = (gt_keypoints != 0).any(axis=-1)
    if not valid_mask.any():
        return 0.0

    pred_keypoints = pred_keypoints[valid_mask]
    gt_keypoints = gt_keypoints[valid_mask]

    h, w = image_size
    image_diagonal = np.sqrt(h ** 2 + w ** 2)
    distances = np.linalg.norm(pred_keypoints - gt_keypoints, axis=-1)
    return float((distances < threshold * image_diagonal).mean())


def compute_pck_at_thresholds(
    pred_keypoints: np.ndarray | torch.Tensor,
    gt_keypoints: np.ndarray | torch.Tensor,
    image_size: tuple[int, int],
    thresholds: list[float] | None = None,
) -> dict[str, float]:
    """PCK sweep — Chapter 4's headline metric is PCK at {1, 5, 10} px / 512px."""
    if thresholds is None:
        thresholds = [0.01, 0.02, 0.05, 0.10]
    return {
        f"PCK@{int(t * 100)}": compute_pck(pred_keypoints, gt_keypoints, image_size, threshold=t)
        for t in thresholds
    }


def denormalize_keypoints(
    keypoints: np.ndarray | torch.Tensor,
    image_size: tuple[int, int],
) -> np.ndarray:
    """Map normalised [-1, 1] coords to pixel coords on an HxW canvas."""
    if isinstance(keypoints, torch.Tensor):
        keypoints = keypoints.cpu().numpy()

    h, w = image_size
    keypoints_pixel = keypoints.copy()
    keypoints_pixel[..., 0] = (keypoints[..., 0] + 1) * (w - 1) / 2
    keypoints_pixel[..., 1] = (keypoints[..., 1] + 1) * (h - 1) / 2
    return keypoints_pixel


def normalize_keypoints(
    keypoints: np.ndarray | torch.Tensor,
    image_size: tuple[int, int],
) -> np.ndarray:
    """Map pixel coords on an HxW canvas to normalised [-1, 1]."""
    if isinstance(keypoints, torch.Tensor):
        keypoints = keypoints.cpu().numpy()

    h, w = image_size
    keypoints_norm = keypoints.copy()
    keypoints_norm[..., 0] = (keypoints[..., 0] * 2 / (w - 1)) - 1
    keypoints_norm[..., 1] = (keypoints[..., 1] * 2 / (h - 1)) - 1
    return keypoints_norm
