"""Phase 5: image reconstruction from selected 2D-SSA components."""

from __future__ import annotations

import numpy as np

from .selection import group_and_sum


def reconstruct_image(
    components: list[np.ndarray] | np.ndarray,
    indices: np.ndarray,
    image_shape: tuple[int, int],
) -> np.ndarray:
    """
    Reconstruct an image from selected 2D-SSA components.

    The reconstruction is the sum of the selected component images.

    The result is intentionally NOT clipped to [0, 255].
    When high-frequency components are selected, the result
    represents a low-energy residual/detail map.
    """
    reconstruction = group_and_sum(
        components,
        indices,
    )

    if reconstruction.shape != image_shape:
        raise RuntimeError(
            "Reconstruction shape does not match the supplied image shape."
        )

    return reconstruction.astype(
        np.float64,
        copy=False,
    )


def normalize_for_display(
    image: np.ndarray,
) -> np.ndarray:
    """
    Normalize an arbitrary-range image to uint8 [0, 255].

    This function is ONLY for visualization.

    The normalized image must NOT be passed to WMF
    or the cost-function pipeline.
    """
    image = np.asarray(image)

    minimum = image.min()
    maximum = image.max()

    if maximum - minimum < 1e-12:
        return np.zeros_like(
            image,
            dtype=np.uint8,
        )

    scaled = (image - minimum) / (maximum - minimum) * 255.0

    return np.clip(
        np.rint(scaled),
        0,
        255,
    ).astype(np.uint8)
