from __future__ import annotations

import numpy as np


def mse(original: np.ndarray, modified: np.ndarray) -> float:
    """Calculate Mean Squared Error between two images."""
    original = original.astype(np.float64)
    modified = modified.astype(np.float64)

    return float(np.mean((original - modified) ** 2))


def psnr(original: np.ndarray, modified: np.ndarray) -> float:
    """Calculate Peak Signal-to-Noise Ratio in dB."""
    error = mse(original, modified)

    if error == 0:
        return float("inf")

    max_pixel = 255.0

    return float(10 * np.log10((max_pixel**2) / error))


def changed_pixel_count(
    original: np.ndarray,
    modified: np.ndarray,
) -> int:
    """Count the number of pixels changed between two images."""
    return int(np.count_nonzero(original != modified))


def changed_pixel_percentage(
    original: np.ndarray,
    modified: np.ndarray,
) -> float:
    """Calculate percentage of pixels changed."""
    total_pixels = original.size

    if total_pixels == 0:
        return 0.0

    return float(changed_pixel_count(original, modified) / total_pixels * 100)
