import cv2
import numpy as np


def generate_gaussian(height: int, width: int, sigma: float = 5) -> np.ndarray:
    """Discrete 2D Gaussian kernel, peak normalised to 1.

    Used as the ground-truth heatmap target. Peak = 1 (not unit-volume) so the
    sigmoid output can saturate at the keypoint and the loss reads naturally
    as "how close are we to a definite positive".
    """
    mask = np.zeros((height, width), dtype=np.float32)
    mask[height // 2, width // 2] = 1
    # cv2.GaussianBlur scales sigma in pixels; we want the same visible spread
    # regardless of kernel size, hence the size-relative sigma.
    sigmas = np.array(mask.shape) / sigma
    mask = cv2.GaussianBlur(mask, ksize=(0, 0), sigmaY=sigmas[0], sigmaX=sigmas[1])
    return mask / mask.max()
