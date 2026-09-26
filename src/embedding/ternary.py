"""Pixel-level modification layer for binary LSB STC.

The STC operates on the LSB plane. When STC requests a bit flip,
this module realizes that flip by changing the corresponding
grayscale pixel by +1 or -1.

For the scalar Phase 8 cost map, both directions have the same
cost. Boundary pixels are handled explicitly:

    0   -> +1 only
    255 -> -1 only
    otherwise -> either direction
"""

from __future__ import annotations

import numpy as np

from .stc import STCResult, stc_decode, stc_encode


def directional_flip_costs(
    image: np.ndarray,
    cost_map: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build feasible +1 and -1 costs for each pixel.

    Parameters
    ----------
    image:
        8-bit grayscale cover image.

    cost_map:
        Non-negative scalar modification cost for each pixel.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (rho_plus, rho_minus)

        Infeasible directions are represented by +inf.
    """
    image = np.asarray(image)

    cost_map = np.asarray(
        cost_map,
        dtype=np.float64,
    )

    if image.ndim != 2:
        raise ValueError("image must be a 2D grayscale array.")

    if image.dtype != np.uint8:
        raise ValueError("image must have dtype uint8.")

    if cost_map.shape != image.shape:
        raise ValueError("cost_map must have the same shape as image.")

    if not np.isfinite(cost_map).all():
        raise ValueError("cost_map must contain only finite values.")

    if np.any(cost_map < 0):
        raise ValueError("cost_map must be non-negative.")

    rho_plus = cost_map.copy()
    rho_minus = cost_map.copy()

    # +1 is impossible at 255.
    rho_plus[image == 255] = np.inf

    # -1 is impossible at 0.
    rho_minus[image == 0] = np.inf

    return rho_plus, rho_minus


def realize_lsb_flips(
    image: np.ndarray,
    original_bits: np.ndarray,
    stego_bits: np.ndarray,
    rho_plus: np.ndarray,
    rho_minus: np.ndarray,
) -> np.ndarray:
    """Convert binary LSB flips into +/-1 pixel modifications."""
    image = np.asarray(image)

    original_bits = np.asarray(
        original_bits,
        dtype=np.uint8,
    )

    stego_bits = np.asarray(
        stego_bits,
        dtype=np.uint8,
    )

    rho_plus = np.asarray(
        rho_plus,
        dtype=np.float64,
    )

    rho_minus = np.asarray(
        rho_minus,
        dtype=np.float64,
    )

    if image.ndim != 2:
        raise ValueError("image must be 2D.")

    if image.dtype != np.uint8:
        raise ValueError("image must have dtype uint8.")

    flat_image = image.reshape(-1)

    if len(original_bits) != len(flat_image):
        raise ValueError("original_bits must contain one bit per image pixel.")

    if len(stego_bits) != len(flat_image):
        raise ValueError("stego_bits must contain one bit per image pixel.")

    flat_plus = rho_plus.reshape(-1)
    flat_minus = rho_minus.reshape(-1)

    if len(flat_plus) != len(flat_image):
        raise ValueError("rho_plus has incorrect size.")

    if len(flat_minus) != len(flat_image):
        raise ValueError("rho_minus has incorrect size.")

    result = flat_image.copy()

    for index in range(len(flat_image)):
        if original_bits[index] == stego_bits[index]:
            continue

        plus_cost = flat_plus[index]
        minus_cost = flat_minus[index]

        if not np.isfinite(plus_cost) and not np.isfinite(minus_cost):
            raise RuntimeError(f"Cannot realize LSB flip at pixel index {index}.")

        # Choose the cheaper feasible direction.
        if plus_cost <= minus_cost:
            result[index] = np.uint8(int(result[index]) + 1)
        else:
            result[index] = np.uint8(int(result[index]) - 1)

    return result.reshape(image.shape)


def embed_ternary_lsb(
    image: np.ndarray,
    cost_map: np.ndarray,
    message_bits: np.ndarray,
    H_hat: np.ndarray,
) -> tuple[np.ndarray, STCResult]:
    """Embed message bits into an image using LSB-plane STC.

    This is the Phase 9 reduction:

        image -> LSB plane -> binary STC -> +/-1 realization
    """
    image = np.asarray(image)

    if image.ndim != 2:
        raise ValueError("image must be 2D.")

    if image.dtype != np.uint8:
        raise ValueError("image must have dtype uint8.")

    flat_image = image.reshape(-1)

    cover_bits = (flat_image & 1).astype(np.uint8)

    message_bits = np.asarray(
        message_bits,
        dtype=np.uint8,
    )

    rho_plus, rho_minus = directional_flip_costs(
        image,
        cost_map,
    )

    # Binary STC needs one cost for the act of flipping the LSB.
    #
    # This is the minimum FEASIBLE directional cost.
    flip_cost = np.minimum(
        rho_plus,
        rho_minus,
    )

    result = stc_encode(
        cover_bits,
        flip_cost.reshape(-1),
        message_bits,
        H_hat,
    )

    stego = realize_lsb_flips(
        image,
        cover_bits,
        result.y,
        rho_plus,
        rho_minus,
    )

    return stego, result


def extract_ternary_lsb(
    stego_image: np.ndarray,
    H_hat: np.ndarray,
    message_length: int,
) -> np.ndarray:
    """Extract message bits from an STC stego image."""
    stego_image = np.asarray(stego_image)

    if stego_image.ndim != 2:
        raise ValueError("stego_image must be 2D.")

    if stego_image.dtype != np.uint8:
        raise ValueError("stego_image must have dtype uint8.")

    stego_bits = (stego_image.reshape(-1) & 1).astype(np.uint8)

    return stc_decode(
        stego_bits,
        H_hat,
        message_length,
    )
