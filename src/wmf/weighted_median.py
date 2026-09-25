"""Weighted Median Filter used by the SSA-WMF steganography pipeline."""

from __future__ import annotations

import numpy as np


def similarity_weight(
    center_value: float, neighbor_value: float, sigma: float
) -> float:
    """Calculate the Gaussian intensity-similarity weight.

    The WMF assigns larger weights to neighboring pixels whose intensity
    is closer to the center pixel.

    weight = exp(-(center - neighbor)^2 / (2 * sigma^2))
    """
    if sigma <= 0:
        raise ValueError("sigma must be greater than zero")

    difference = float(center_value) - float(neighbor_value)

    return float(np.exp(-(difference * difference) / (2.0 * sigma * sigma)))


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """Return the weighted median of values."""
    order = np.argsort(values)

    sorted_values = values[order]
    sorted_weights = weights[order]

    total_weight = np.sum(sorted_weights)
    target = total_weight / 2.0

    cumulative = 0.0

    for index in range(len(sorted_values)):
        cumulative += sorted_weights[index]

        if cumulative >= target:
            return float(sorted_values[index])

    return float(sorted_values[-1])


def _single_pass(
    image: np.ndarray,
    gamma: int,
    sigma: float,
) -> np.ndarray:
    """Apply one weighted-median-filter pass."""
    height, width = image.shape

    result = np.empty_like(image, dtype=np.float64)

    window_size = 2 * gamma + 1
    number_of_values = window_size * window_size

    for row in range(height):
        for column in range(width):
            center = image[row, column]

            values = np.empty(number_of_values, dtype=np.float64)
            weights = np.empty(number_of_values, dtype=np.float64)

            index = 0

            for offset_row in range(-gamma, gamma + 1):
                neighbor_row = min(
                    max(row + offset_row, 0),
                    height - 1,
                )

                for offset_column in range(-gamma, gamma + 1):
                    neighbor_column = min(
                        max(column + offset_column, 0),
                        width - 1,
                    )

                    neighbor = image[neighbor_row, neighbor_column]

                    values[index] = neighbor
                    weights[index] = similarity_weight(
                        center,
                        neighbor,
                        sigma,
                    )

                    index += 1

            result[row, column] = _weighted_median(values, weights)

    return result


def weighted_median_filter(
    image: np.ndarray,
    gamma: int = 5,
    sigma: float = 3.0,
    tau: int = 2,
) -> np.ndarray:
    """Apply the weighted median filter.

    Parameters
    ----------
    image:
        Two-dimensional image or suitability array.

    gamma:
        Radius of the local filtering window.

    sigma:
        Standard deviation controlling intensity similarity.

    tau:
        Number of sequential WMF iterations.

    Returns
    -------
    np.ndarray
        Filtered floating-point array.
    """
    image = np.asarray(image)

    if image.ndim != 2:
        raise ValueError("image must be a 2D array")

    if gamma < 0:
        raise ValueError("gamma must be non-negative")

    if sigma <= 0:
        raise ValueError("sigma must be greater than zero")

    if tau <= 0:
        raise ValueError("tau must be greater than zero")

    result = image.astype(np.float64, copy=True)

    for _ in range(tau):
        result = _single_pass(result, gamma, sigma)

    return result
