"""Phase 10: practical secret embedding/extraction API.

The encoder and decoder share a deterministic StegoKey
(h, w, seed), which is used to regenerate the same H_hat.

The extraction side requires only:
    - the stego pixels
    - the StegoKey

It does not require the cover image or the cost map.

The implementation follows the Phase 10 scope:
- deterministic StegoKey
- embed_secret()
- extract_secret()
- hardened message-header validation
- feasible +/-1 pixel modifications at intensity boundaries
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .message import decode_message, encode_message
from .stc import (
    STCResult,
    make_random_submatrix,
    stc_decode,
    stc_encode,
)


@dataclass(frozen=True)
class StegoKey:
    """Shared parameters used to regenerate the STC H_hat."""

    h: int
    w: int
    seed: int

    def __post_init__(self) -> None:
        if self.h <= 0:
            raise ValueError("h must be greater than zero.")

        if self.w <= 0:
            raise ValueError("w must be greater than zero.")

    def build_H_hat(self) -> np.ndarray:
        """Reconstruct the deterministic STC submatrix."""

        return make_random_submatrix(
            self.h,
            self.w,
            self.seed,
        )


@dataclass(frozen=True)
class SecretEmbedResult:
    """Result returned by embed_secret()."""

    stego: np.ndarray
    total_cost: float


def _validate_flat_arrays(
    cover_flat: np.ndarray,
    rho_plus_flat: np.ndarray,
    rho_minus_flat: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Validate and make directional costs physically feasible."""

    cover_flat = np.asarray(
        cover_flat,
    )

    rho_plus_flat = np.asarray(
        rho_plus_flat,
        dtype=np.float64,
    )

    rho_minus_flat = np.asarray(
        rho_minus_flat,
        dtype=np.float64,
    )

    if cover_flat.ndim != 1:
        raise ValueError("cover_flat must be one-dimensional.")

    # Convert integer-like cover values to uint8 while validating range.
    if cover_flat.dtype != np.uint8:
        if not np.issubdtype(
            cover_flat.dtype,
            np.integer,
        ):
            raise ValueError("cover_flat must contain integer pixel values.")

        if not np.all((cover_flat >= 0) & (cover_flat <= 255)):
            raise ValueError("cover_flat values must be in [0, 255].")

        cover_flat = cover_flat.astype(np.uint8)

    if rho_plus_flat.shape != cover_flat.shape:
        raise ValueError("rho_plus_flat must have the same shape as cover_flat.")

    if rho_minus_flat.shape != cover_flat.shape:
        raise ValueError("rho_minus_flat must have the same shape as cover_flat.")

    if np.any(rho_plus_flat < 0):
        raise ValueError("rho_plus_flat must be non-negative.")

    if np.any(rho_minus_flat < 0):
        raise ValueError("rho_minus_flat must be non-negative.")

    if not np.isfinite(rho_plus_flat[np.isfinite(rho_plus_flat)]).all():
        raise ValueError("rho_plus_flat contains invalid values.")

    if not np.isfinite(rho_minus_flat[np.isfinite(rho_minus_flat)]).all():
        raise ValueError("rho_minus_flat contains invalid values.")

    # Work on copies because we change boundary feasibility below.
    rho_plus_flat = rho_plus_flat.copy()
    rho_minus_flat = rho_minus_flat.copy()

    # ---------------------------------------------------------
    # Physical feasibility constraints
    # ---------------------------------------------------------
    #
    # Pixel 255 cannot be increased to 256.
    # Therefore +1 is impossible.
    #
    # Pixel 0 cannot be decreased to -1.
    # Therefore -1 is impossible.
    #
    # Use +inf to tell STC that these directions cannot be selected.
    #
    rho_plus_flat[cover_flat == 255] = np.inf

    rho_minus_flat[cover_flat == 0] = np.inf

    return (
        cover_flat,
        rho_plus_flat,
        rho_minus_flat,
    )


def _realize_lsb_flips(
    cover_flat: np.ndarray,
    stego_bits: np.ndarray,
    rho_plus_flat: np.ndarray,
    rho_minus_flat: np.ndarray,
) -> np.ndarray:
    """Convert requested LSB flips into feasible +/-1 pixel changes."""

    cover_flat = np.asarray(
        cover_flat,
        dtype=np.uint8,
    )

    stego_bits = np.asarray(
        stego_bits,
        dtype=np.uint8,
    )

    rho_plus_flat = np.asarray(
        rho_plus_flat,
        dtype=np.float64,
    )

    rho_minus_flat = np.asarray(
        rho_minus_flat,
        dtype=np.float64,
    )

    if len(stego_bits) != len(cover_flat):
        raise ValueError("stego_bits and cover_flat must have the same length.")

    result = cover_flat.copy()

    original_bits = (cover_flat & 1).astype(np.uint8)

    for index in range(len(result)):
        # No modification required.
        if original_bits[index] == stego_bits[index]:
            continue

        plus_cost = rho_plus_flat[index]
        minus_cost = rho_minus_flat[index]

        plus_feasible = np.isfinite(plus_cost) and int(result[index]) < 255

        minus_feasible = np.isfinite(minus_cost) and int(result[index]) > 0

        if not plus_feasible and not minus_feasible:
            raise RuntimeError(
                f"LSB flip at index {index} has no feasible pixel modification."
            )

        # Choose the cheaper feasible direction.
        if plus_feasible and (not minus_feasible or plus_cost <= minus_cost):
            result[index] = np.uint8(int(result[index]) + 1)

        elif minus_feasible:
            result[index] = np.uint8(int(result[index]) - 1)

        else:
            raise RuntimeError(f"Could not realize LSB flip at index {index}.")

    return result


def embed_secret(
    cover_flat: np.ndarray,
    rho_plus_flat: np.ndarray,
    rho_minus_flat: np.ndarray,
    secret: bytes,
    key: StegoKey,
) -> SecretEmbedResult:
    """Embed a byte secret using the shared StegoKey.

    Parameters
    ----------
    cover_flat:
        One-dimensional uint8 cover pixels.

    rho_plus_flat:
        Cost of increasing each pixel by +1.

    rho_minus_flat:
        Cost of decreasing each pixel by -1.

    secret:
        Arbitrary byte payload.

    key:
        Shared deterministic STC key.

    Returns
    -------
    SecretEmbedResult
        Stego pixels and total STC modification cost.
    """

    (
        cover_flat,
        rho_plus_flat,
        rho_minus_flat,
    ) = _validate_flat_arrays(
        cover_flat,
        rho_plus_flat,
        rho_minus_flat,
    )

    if len(cover_flat) % key.w != 0:
        raise ValueError(f"length {len(cover_flat)} is not divisible by w={key.w}")

    # Number of message/syndrome bits supported by the
    # current STC formulation.
    capacity_bits = len(cover_flat) // key.w

    H_hat = key.build_H_hat()

    cover_bits = (cover_flat & 1).astype(np.uint8)

    # Preserve the cover's existing syndrome in the unused
    # capacity positions. Only the header + actual payload
    # should be changed.
    cover_syndrome = stc_decode(
        cover_bits,
        H_hat,
        capacity_bits,
    )

    message_bits = encode_message(
        secret,
        capacity_bits=capacity_bits,
        fill_bits=cover_syndrome,
    )

    # Binary STC flip cost:
    # choose the cheaper FEASIBLE directional modification.
    flip_cost = np.minimum(
        rho_plus_flat,
        rho_minus_flat,
    )

    # A finite cover pixel should always have at least one
    # feasible direction. Still check explicitly.
    if not np.isfinite(flip_cost).all():
        # If a position is impossible in both directions,
        # it can never realize an LSB flip. The current
        # implementation has no additional mechanism for that case.
        impossible = np.flatnonzero(~np.isfinite(flip_cost))

        # If there are impossible positions, STC may still
        # succeed provided it never needs to flip them.
        # Therefore do not reject the entire embedding here.
        _ = impossible

    result: STCResult = stc_encode(
        cover_bits,
        flip_cost,
        message_bits,
        H_hat,
    )

    stego = _realize_lsb_flips(
        cover_flat,
        result.y,
        rho_plus_flat,
        rho_minus_flat,
    )

    return SecretEmbedResult(
        stego=stego,
        total_cost=result.total_cost,
    )


def extract_secret(
    stego_flat: np.ndarray,
    key: StegoKey,
) -> bytes:
    """Extract a secret using only the stego pixels and shared key.

    The extraction process does not require:
        - the original cover
        - the cost map
        - directional costs

    The recovered message bits are passed to decode_message(),
    which validates the embedded payload-length header.
    """

    stego_flat = np.asarray(
        stego_flat,
        dtype=np.uint8,
    )

    if stego_flat.ndim != 1:
        raise ValueError("stego_flat must be one-dimensional.")

    if len(stego_flat) % key.w != 0:
        raise ValueError(f"length {len(stego_flat)} is not divisible by w={key.w}")

    capacity_bits = len(stego_flat) // key.w

    H_hat = key.build_H_hat()

    stego_bits = (stego_flat & 1).astype(np.uint8)

    message_bits = stc_decode(
        stego_bits,
        H_hat,
        capacity_bits,
    )

    return decode_message(message_bits)
