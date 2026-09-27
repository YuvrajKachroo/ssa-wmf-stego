"""Phase 11: baseline steganalysis features.

This is a SPAM-style adjacent-pixel difference feature, not the
full Spatial Rich Model (SRM).

Features:
    - horizontal difference histogram
    - vertical difference histogram
    - global image moments
    - horizontal-difference moments

Default dimensionality:
    2 * (2T + 1) + 8
    = 30 for T=3
"""

from __future__ import annotations

import numpy as np
from scipy.stats import kurtosis, skew


def difference_histogram(
    image: np.ndarray,
    axis: int,
    T: int = 3,
) -> np.ndarray:
    """Return a normalized histogram of adjacent-pixel differences.

    Parameters
    ----------
    image:
        2-D grayscale image.

    axis:
        0 for vertical differences, 1 for horizontal differences.

    T:
        Difference clipping threshold.

    Returns
    -------
    np.ndarray
        Normalized histogram with 2*T+1 bins.
    """

    image = np.asarray(
        image,
        dtype=np.float64,
    )

    if image.ndim != 2:
        raise ValueError("image must be a 2D array.")

    if axis not in (0, 1):
        raise ValueError("axis must be 0 or 1.")

    if T < 0:
        raise ValueError("T must be non-negative.")

    diff = np.diff(
        image,
        axis=axis,
    )

    diff_clipped = np.clip(
        diff,
        -T,
        T,
    )

    # Integer-centered bins:
    # [-T, ..., T]
    bins = (
        np.arange(
            -T,
            T + 2,
        )
        - 0.5
    )

    histogram, _ = np.histogram(
        diff_clipped,
        bins=bins,
    )

    total = histogram.sum()

    if total == 0:
        return np.zeros(
            2 * T + 1,
            dtype=np.float64,
        )

    return histogram.astype(np.float64) / total


def global_moments(
    image: np.ndarray,
) -> np.ndarray:
    """Return eight global statistical features.

    Features:
        image mean
        image standard deviation
        image skewness
        image kurtosis
        horizontal-difference mean
        horizontal-difference standard deviation
        horizontal-difference skewness
        horizontal-difference kurtosis
    """

    image = np.asarray(
        image,
        dtype=np.float64,
    )

    if image.ndim != 2:
        raise ValueError("image must be a 2D array.")

    diff = np.diff(
        image,
        axis=1,
    ).ravel()

    flat = image.ravel()

    return np.array(
        [
            flat.mean(),
            flat.std(),
            skew(flat),
            kurtosis(flat),
            diff.mean(),
            diff.std(),
            skew(diff),
            kurtosis(diff),
        ],
        dtype=np.float64,
    )


def extract_features(
    image: np.ndarray,
    T: int = 3,
) -> np.ndarray:
    """Extract the complete SPAM-style baseline feature vector.

    Default T=3 gives:

        7 horizontal histogram features
        + 7 vertical histogram features
        + 8 statistical moments
        = 22 features

    """
    horizontal = difference_histogram(
        image,
        axis=1,
        T=T,
    )

    vertical = difference_histogram(
        image,
        axis=0,
        T=T,
    )

    moments = global_moments(image)

    return np.concatenate(
        [
            horizontal,
            vertical,
            moments,
        ]
    )
