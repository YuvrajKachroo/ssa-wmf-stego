"""Phase 8: Cost map generation for SSA-WMF steganography."""

from __future__ import annotations

import numpy as np


PAPER_EPSILON = np.exp(-10.0)


def compute_cost_map(
    wmf_output: np.ndarray,
    epsilon: float = PAPER_EPSILON,
) -> np.ndarray:
    """Convert the WMF-smoothed suitability map into a cost map.

    The paper defines:

        rho = 1 / (O(zeta) + epsilon)

    where O(zeta) is the WMF output.
    """

    wmf_output = np.asarray(wmf_output, dtype=np.float64)

    if wmf_output.ndim != 2:
        raise ValueError("wmf_output must be a 2D array")

    if not np.isfinite(wmf_output).all():
        raise ValueError("wmf_output must contain only finite values")

    if epsilon <= 0:
        raise ValueError("epsilon must be greater than zero")

    # WMF output represents a non-negative suitability quantity.
    if np.any(wmf_output < 0):
        raise ValueError("wmf_output must be non-negative")

    cost = 1.0 / (wmf_output + epsilon)

    if not np.isfinite(cost).all():
        raise RuntimeError("cost map contains non-finite values")

    return cost
