"""2D-SSA component selection."""

from __future__ import annotations

import numpy as np


def select_by_rank_range(
    eigenvalues: np.ndarray,
    s: int,
    t: int,
) -> np.ndarray:
    """
    Select an inclusive 1-indexed eigenvalue rank range.

    Example:
        s=8, t=9 with 9 components
        selects components 8 and 9.

    Returns 0-based indices.
    """
    number_of_components = len(eigenvalues)

    if not (1 <= s <= t <= number_of_components):
        raise ValueError(f"Invalid range s={s}, t={t} for L={number_of_components}")

    return np.arange(
        s - 1,
        t,
    )


def select_top_k(
    eigenvalues: np.ndarray,
    k: int,
) -> np.ndarray:
    """
    Select the k largest-eigenvalue components.
    """
    if k < 0:
        raise ValueError("k must be non-negative.")

    number_of_components = len(eigenvalues)

    k = min(
        k,
        number_of_components,
    )

    return np.arange(k)


def select_bottom_k(
    eigenvalues: np.ndarray,
    k: int,
) -> np.ndarray:
    """
    Select the k smallest-eigenvalue components.
    """
    if k < 0:
        raise ValueError("k must be non-negative.")

    number_of_components = len(eigenvalues)

    k = min(
        k,
        number_of_components,
    )

    return np.arange(
        number_of_components - k,
        number_of_components,
    )


def select_explicit(
    indices,
) -> np.ndarray:
    """
    Select explicit 0-based component indices.

    Duplicate indices are removed and the result
    is sorted.
    """
    return np.asarray(
        sorted(set(int(index) for index in indices)),
        dtype=int,
    )


def group_and_sum(
    components: list[np.ndarray] | np.ndarray,
    indices: np.ndarray,
) -> np.ndarray:
    """
    Sum the selected component images.

    Parameters
    ----------
    components:
        Component images with shape
        (number_of_components, height, width).

    indices:
        0-based component indices.

    Returns
    -------
    np.ndarray
        Partial reconstruction from the
        selected components.
    """
    if len(indices) == 0:
        raise ValueError("No components selected.")

    components = np.asarray(components)

    if components.ndim != 3:
        raise ValueError(
            "Components must have shape (number_of_components, height, width)."
        )

    if np.any(indices < 0) or np.any(indices >= components.shape[0]):
        raise ValueError("Component index is out of range.")

    return components[indices].sum(axis=0)
