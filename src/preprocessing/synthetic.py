"""Synthetic grayscale images for development and unit testing only."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter


def make_synthetic_image(
    seed: int,
    shape: tuple[int, int] = (512, 512),
) -> np.ndarray:
    """Create a deterministic textured uint8 test image.

    These images are for development/testing only and must not be
    presented as BOSSbase experimental results.
    """

    rng = np.random.default_rng(seed)

    smooth = gaussian_filter(
        rng.normal(size=shape),
        sigma=8.0,
    )

    fine = gaussian_filter(
        rng.normal(size=shape),
        sigma=1.0,
    )

    image = (
        smooth
        / (
            np.abs(smooth).max()
            + 1e-9
        )
        * 90.0
        + fine * 12.0
        + 128.0
    )

    return np.clip(
        np.rint(image),
        0,
        255,
    ).astype(np.uint8)
