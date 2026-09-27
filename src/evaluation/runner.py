"""Phase 13: automated experiment runner.

The current STC implementation provides discrete payload rates
approximately equal to 1 / w bits per pixel.

For every requested target payload:
    target_bpp -> nearest achievable w -> actual_bpp = 1 / w

The experiment reports both target and actual payload rates.

Implemented methods:
    adaptive : real SSA-WMF cost map
    uniform  : rho = 1 everywhere

Other methods such as HUGO, WOW, S-UNIWARD, HILL, MiPOD, and
true LSB are not implemented in this runner yet.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.cost_function.cost import compute_cost_map
from src.cost_function.suitability import compute_suitability_map
from src.embedding.stc import make_random_submatrix
from src.embedding.ternary import (
    embed_ternary_lsb,
    extract_ternary_lsb,
)
from src.evaluation.metrics import (
    mse,
    psnr,
    ssim,
)
from src.ssa.ssa2d import (
    create_elementary_components,
    create_trajectory_matrix,
    decompose_trajectory_matrix,
    reconstruct_component,
)
from src.wmf.weighted_median import weighted_median_filter


def payload_to_w(target_bpp: float) -> int:
    """Map a target payload to the nearest achievable STC width."""

    if target_bpp <= 0:
        raise ValueError(
            "target_bpp must be greater than zero."
        )

    if target_bpp > 1.0:
        raise ValueError(
            "target_bpp cannot exceed 1.0 bpp."
        )

    return max(
        1,
        round(1.0 / target_bpp),
    )


@dataclass(frozen=True)
class PrecomputedImage:
    """SSA/WMF/cost information reused across experiments."""

    cover: np.ndarray
    zeta: np.ndarray
    rho_adaptive: np.ndarray


def precompute(
    image: np.ndarray,
    window_height: int = 3,
    window_width: int = 3,
    start_component: int = 8,
    end_component: int = 9,
    gamma: int = 5,
    sigma: float = 3.0,
    tau: int = 2,
) -> PrecomputedImage:
    """Run the Phase 3-8 adaptive-cost pipeline once."""

    image = np.asarray(
        image,
        dtype=np.uint8,
    )

    if image.ndim != 2:
        raise ValueError(
            "image must be a 2D grayscale image."
        )

    trajectory = create_trajectory_matrix(
        image.astype(np.float64),
        window_height=window_height,
        window_width=window_width,
    )

    (
        _eigenvalues,
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
                image.shape,
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

    rho_adaptive = compute_cost_map(
        filtered
    )

    return PrecomputedImage(
        cover=image,
        zeta=zeta,
        rho_adaptive=rho_adaptive,
    )


@dataclass(frozen=True)
class ImageResult:
    """Metrics for one method/payload/image combination."""

    method: str
    target_bpp: float
    actual_bpp: float
    image_index: int
    n_changed: int
    n_trim: int
    psnr: float
    ssim: float
    mse: float
    embedding_time_s: float
    extraction_success: bool
    file_size_bytes: int


def run_one(
    pre: PrecomputedImage,
    method: str,
    target_bpp: float,
    h: int,
    image_index: int,
    rng: np.random.Generator,
    out_png_path: str | Path | None = None,
) -> tuple[ImageResult, np.ndarray]:
    """Run one embedding experiment.

    The embedding is performed on a top-left rectangular crop whose
    width is divisible by the selected STC w. The untouched remainder
    of the original image is copied back unchanged.
    """

    if method not in {
        "adaptive",
        "uniform",
    }:
        raise ValueError(
            f"unknown method '{method}'. "
            "Only 'adaptive' and 'uniform' are implemented."
        )

    if h <= 0:
        raise ValueError(
            "h must be greater than zero."
        )

    w = payload_to_w(
        target_bpp
    )

    actual_bpp = 1.0 / w

    image = pre.cover

    height, width = image.shape

    # ---------------------------------------------------------
    # Crop width so the number of embedded pixels is divisible
    # by the STC submatrix width w.
    # ---------------------------------------------------------
    trimmed_width = (
        width // w
    ) * w

    if trimmed_width <= 0:
        raise ValueError(
            f"image width {width} is too small for w={w}."
        )

    working_cover = image[
        :,
        :trimmed_width,
    ].copy()

    n_trim = working_cover.size

    if n_trim % w != 0:
        raise RuntimeError(
            "Internal error: trimmed image size is not divisible by w."
        )

    message_length = (
        n_trim // w
    )

    H_hat = make_random_submatrix(
        h,
        w,
        seed=123,
    )

    if method == "adaptive":
        rho = pre.rho_adaptive[
            :,
            :trimmed_width,
        ]
    else:
        rho = np.ones(
            working_cover.shape,
            dtype=np.float64,
        )

    message_bits = rng.integers(
        0,
        2,
        size=message_length,
        dtype=np.uint8,
    )

    # ---------------------------------------------------------
    # Embed
    # ---------------------------------------------------------
    start = time.perf_counter()

    stego_working, _stc_result = (
        embed_ternary_lsb(
            working_cover,
            rho,
            message_bits,
            H_hat,
        )
    )

    embedding_time = (
        time.perf_counter()
        - start
    )

    # ---------------------------------------------------------
    # Extract immediately to verify correctness.
    # ---------------------------------------------------------
    extracted_bits = extract_ternary_lsb(
        stego_working,
        H_hat,
        message_length=message_length,
    )

    extraction_success = bool(
        np.array_equal(
            extracted_bits,
            message_bits,
        )
    )

    # ---------------------------------------------------------
    # Restore full-size image.
    # ---------------------------------------------------------
    stego_full = image.copy()

    stego_full[
        :,
        :trimmed_width,
    ] = stego_working

    # ---------------------------------------------------------
    # Quality metrics
    # ---------------------------------------------------------
    mse_value = mse(
        image,
        stego_full,
    )

    psnr_value = psnr(
        image,
        stego_full,
    )

    ssim_value = ssim(
        image,
        stego_full,
    )

    n_changed = int(
        np.count_nonzero(
            image != stego_full
        )
    )

    # ---------------------------------------------------------
    # Optional PNG output
    # ---------------------------------------------------------
    file_size = 0

    if out_png_path is not None:

        out_png_path = Path(
            out_png_path
        )

        out_png_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        written = cv2.imwrite(
            str(out_png_path),
            stego_full,
        )

        if not written:
            raise IOError(
                f"Could not save {out_png_path}"
            )

        file_size = (
            out_png_path.stat().st_size
        )

    result = ImageResult(
        method=method,
        target_bpp=target_bpp,
        actual_bpp=actual_bpp,
        image_index=image_index,
        n_changed=n_changed,
        n_trim=n_trim,
        psnr=psnr_value,
        ssim=ssim_value,
        mse=mse_value,
        embedding_time_s=embedding_time,
        extraction_success=extraction_success,
        file_size_bytes=file_size,
    )

    return (
        result,
        stego_full,
    )
