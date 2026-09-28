"""Streamlit-free helpers for the Phase 14 demonstration app."""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from src.cost_function.cost import compute_cost_map
from src.cost_function.suitability import compute_suitability_map
from src.embedding.extraction import (
    SecretEmbedResult,
    StegoKey,
    embed_secret,
    extract_secret,
)
from src.evaluation.runner import payload_to_w
from src.ssa.reconstruction import normalize_for_display
from src.ssa.ssa2d import (
    create_elementary_components,
    create_trajectory_matrix,
    decompose_trajectory_matrix,
    reconstruct_component,
)
from src.wmf.weighted_median import weighted_median_filter


def to_uint8_grayscale(
    image: Image.Image,
) -> np.ndarray:
    """Convert a PIL image only when it is already 8-bit grayscale.

    Phase 1 intentionally rejects RGB/RGBA/etc. rather than silently
    converting them.
    """

    if image.mode != "L":
        raise ValueError(
            "Expected an 8-bit grayscale image "
            f"(mode 'L'), got '{image.mode}'."
        )

    array = np.asarray(
        image,
        dtype=np.uint8,
    )

    if array.ndim != 2:
        raise ValueError(
            f"Expected a 2-D image, got {array.ndim}-D data."
        )

    return np.ascontiguousarray(
        array
    )


def run_analysis(
    cover: np.ndarray,
    *,
    window_height: int = 3,
    window_width: int = 3,
    start_component: int = 8,
    end_component: int = 9,
    gamma: int = 5,
    sigma: float = 3.0,
    tau: int = 2,
) -> dict[str, Any]:
    """Run the current paper-style SSA -> zeta -> WMF -> cost pipeline."""

    cover = np.asarray(
        cover,
        dtype=np.uint8,
    )

    if cover.ndim != 2:
        raise ValueError(
            "cover must be a 2-D grayscale image."
        )

    trajectory = create_trajectory_matrix(
        cover.astype(np.float64),
        window_height=window_height,
        window_width=window_width,
    )

    (
        eigenvalues,
        eigenvectors,
        singular_values,
    ) = decompose_trajectory_matrix(
        trajectory
    )

    components = create_elementary_components(
        trajectory,
        eigenvectors,
        singular_values,
    )

    reconstructed_components = np.stack(
        [
            reconstruct_component(
                component,
                cover.shape,
                window_shape=(
                    window_height,
                    window_width,
                ),
            )
            for component in components
        ],
        axis=0,
    )

    zeta = compute_suitability_map(
        reconstructed_components,
        start_component=start_component,
        end_component=end_component,
    )

    filtered = weighted_median_filter(
        zeta,
        gamma=gamma,
        sigma=sigma,
        tau=tau,
    )

    rho = compute_cost_map(
        filtered
    )

    return {
        "trajectory": trajectory,
        "eigenvalues": eigenvalues,
        "singular_values": singular_values,
        "components": components,
        "reconstructed_components": reconstructed_components,
        "selected_components": reconstructed_components[
            start_component - 1 : end_component
        ],
        "zeta": zeta,
        "filtered": filtered,
        "rho": rho,
    }


def symmetric_directional_costs(
    rho: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Create +1/-1 directional costs from the current symmetric cost map."""

    rho = np.asarray(
        rho,
        dtype=np.float64,
    )

    if rho.ndim != 2:
        raise ValueError(
            "rho must be a 2-D array."
        )

    if not np.isfinite(rho).all():
        raise ValueError(
            "rho must contain only finite values."
        )

    if np.any(rho < 0):
        raise ValueError(
            "rho must be non-negative."
        )

    return (
        rho.copy(),
        rho.copy(),
    )


def embed_message(
    cover: np.ndarray,
    rho: np.ndarray,
    secret: bytes,
    *,
    target_bpp: float,
    h: int,
    seed: int,
) -> tuple[np.ndarray, SecretEmbedResult, StegoKey]:
    """Embed a byte payload using the existing Phase 10 API."""

    cover = np.asarray(
        cover,
        dtype=np.uint8,
    )

    if cover.ndim != 2:
        raise ValueError(
            "cover must be a 2-D grayscale image."
        )

    w = payload_to_w(
        target_bpp
    )

    trimmed_width = (
        cover.shape[1] // w
    ) * w

    if trimmed_width <= 0:
        raise ValueError(
            f"Image width {cover.shape[1]} is too small for w={w}."
        )

    working_cover = cover[
        :,
        :trimmed_width,
    ].copy()

    working_rho = np.asarray(
        rho,
        dtype=np.float64,
    )[
        :,
        :trimmed_width,
    ].copy()

    n_trim = working_cover.size

    if n_trim % w != 0:
        raise RuntimeError(
            "Internal error: working image size "
            "is not divisible by w."
        )

    rho_plus, rho_minus = (
        symmetric_directional_costs(
            working_rho
        )
    )

    key = StegoKey(
        h=int(h),
        w=int(w),
        seed=int(seed),
    )

    cover_flat = (
        working_cover
        .flatten()
        .astype(np.int64)
    )

    result = embed_secret(
        cover_flat,
        rho_plus.flatten(),
        rho_minus.flatten(),
        secret,
        key,
    )

    # Put the cropped stego image back into the same 2-D region
    # from which it was created. Do not copy the flattened crop
    # into the flattened full image, because that shifts every
    # subsequent row when the width was trimmed.
    stego = cover.copy()

    stego[
        :,
        :trimmed_width,
    ] = (
        result.stego
        .reshape(working_cover.shape)
        .astype(np.uint8)
    )

    return (
        stego,
        result,
        key,
    )


def extract_message(
    stego: np.ndarray,
    key: StegoKey,
) -> bytes:
    """Extract a secret using only the stego image and key."""

    stego = np.asarray(
        stego,
        dtype=np.uint8,
    )

    if stego.ndim != 2:
        raise ValueError(
            "stego must be a 2-D grayscale image."
        )

    w = key.w

    trimmed_width = (
        stego.shape[1] // w
    ) * w

    if trimmed_width <= 0:
        raise ValueError(
            f"Image width {stego.shape[1]} is too small for w={w}."
        )

    working = stego[
        :,
        :trimmed_width,
    ]

    flat = (
        working
        .flatten()
        .astype(np.int64)
    )

    return extract_secret(
        flat,
        key,
    )


def normalized_display(
    image: np.ndarray,
) -> np.ndarray:
    """Return a uint8 display image from arbitrary numeric data."""

    return normalize_for_display(
        np.asarray(image)
    )


