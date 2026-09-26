"""Syndrome-Trellis Codes (STC) implemented from scratch via Viterbi.

The encoder finds a minimum-cost modified bit sequence y such that the
STC syndrome of y equals the supplied message bits.

This implementation uses a seeded binary H_hat submatrix. The particular
H_hat construction is an implementation choice and is not claimed to be
the paper's exact submatrix.

Ternary pixel modification is handled separately in ternary.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


INF = float("inf")


def _column_vectors(H_hat: np.ndarray) -> np.ndarray:
    """Convert each H_hat column into an integer bit-vector."""
    h, w = H_hat.shape

    columns = np.zeros(w, dtype=np.int64)

    for column in range(w):
        value = 0

        for row in range(h):
            if H_hat[row, column]:
                value |= 1 << row

        columns[column] = value

    return columns


def make_random_submatrix(
    h: int,
    w: int,
    seed: int,
) -> np.ndarray:
    """Create a reproducible random binary STC submatrix.

    Every row is guaranteed to contain at least one 1.
    """
    if h <= 0:
        raise ValueError("h must be greater than zero.")

    if w <= 0:
        raise ValueError("w must be greater than zero.")

    rng = np.random.default_rng(seed)

    H_hat = rng.integers(
        0,
        2,
        size=(h, w),
        dtype=np.uint8,
    )

    # Prevent degenerate all-zero rows.
    for row in range(h):
        if not H_hat[row].any():
            column = rng.integers(0, w)
            H_hat[row, column] = 1

    return H_hat


@dataclass
class STCResult:
    """Result returned by STC encoding."""

    y: np.ndarray
    total_cost: float


def stc_encode(
    x: np.ndarray,
    rho: np.ndarray,
    msg: np.ndarray,
    H_hat: np.ndarray,
) -> STCResult:
    """Encode message bits using minimum-cost Viterbi STC.

    Parameters
    ----------
    x:
        Cover bits, shape (n,).

    rho:
        Modification cost for each bit, shape (n,).

    msg:
        Message/syndrome bits, shape (m,).

    H_hat:
        Binary STC submatrix, shape (h, w).

    Notes
    -----
    This implementation currently requires:

        n = m * w

    where:
        n = number of cover bits
        m = number of message bits
        w = H_hat width
    """

    x = np.asarray(x, dtype=np.uint8)
    rho = np.asarray(rho, dtype=np.float64)
    msg = np.asarray(msg, dtype=np.uint8)
    H_hat = np.asarray(H_hat, dtype=np.uint8)

    if H_hat.ndim != 2:
        raise ValueError("H_hat must be a 2D array.")

    h, w = H_hat.shape

    if h <= 0 or w <= 0:
        raise ValueError("H_hat must have positive dimensions.")

    n = len(x)
    m = len(msg)

    if n != m * w:
        raise ValueError(f"n={n} must equal m*w={m * w} (m={m}, w={w})")

    if len(rho) != n:
        raise ValueError("rho must have the same length as x.")

    if not np.isfinite(rho).all():
        raise ValueError("rho must contain only finite values.")

    if np.any(rho < 0):
        raise ValueError("rho must be non-negative.")

    number_of_states = 1 << h

    column_vectors = _column_vectors(H_hat)

    state_indices = np.arange(number_of_states)

    # Minimum cost reaching each trellis state.
    cost = np.full(
        number_of_states,
        INF,
        dtype=np.float64,
    )

    cost[0] = 0.0

    # Backpointer stores the selected output bit for every state.
    backpointer = np.zeros(
        (n, number_of_states),
        dtype=np.uint8,
    )

    t = 0

    for block in range(m):
        remaining_rows = m - block

        for column in range(w):
            column_vector = int(column_vectors[column])

            # Mask contributions from rows that do not exist
            # near the end of the trellis.
            if h > remaining_rows:
                column_vector &= (1 << remaining_rows) - 1

            cover_bit = int(x[t])

            new_cost = np.full(
                number_of_states,
                INF,
                dtype=np.float64,
            )

            new_backpointer = np.zeros(
                number_of_states,
                dtype=np.uint8,
            )

            for output_bit in (0, 1):
                modification_cost = 0.0 if output_bit == cover_bit else rho[t]

                state_change = column_vector if output_bit == 1 else 0

                new_states = state_indices ^ state_change

                candidate_cost = cost + modification_cost

                better = candidate_cost < new_cost[new_states]

                predecessor_indices = state_indices[better]

                destination_indices = new_states[better]

                new_cost[destination_indices] = candidate_cost[predecessor_indices]

                new_backpointer[destination_indices] = output_bit

            cost = new_cost
            backpointer[t] = new_backpointer

            t += 1

        # Resolve the current message/syndrome bit.
        message_bit = int(msg[block])

        bit_zero = state_indices & 1

        cost = np.where(
            bit_zero == message_bit,
            cost,
            INF,
        )

        # Shift the trellis window.
        half_states = number_of_states >> 1

        indices = np.arange(half_states)

        shifted_cost = cost[(indices << 1) | message_bit]

        cost = np.full(
            number_of_states,
            INF,
            dtype=np.float64,
        )

        cost[:half_states] = shifted_cost

    total_cost = cost[0]

    if not np.isfinite(total_cost):
        raise RuntimeError(
            "STC encoding is infeasible. Check H_hat and message dimensions."
        )

    # -------------------------
    # Viterbi traceback
    # -------------------------

    y = np.zeros(
        n,
        dtype=np.uint8,
    )

    state = 0
    t = n - 1

    for block in range(m - 1, -1, -1):
        message_bit = int(msg[block])

        # Undo the block shift.
        state = (state << 1) | message_bit

        for column in range(w - 1, -1, -1):
            output_bit = int(backpointer[t, state])

            y[t] = output_bit

            column_vector = int(column_vectors[column])

            remaining_rows = m - block

            if h > remaining_rows:
                column_vector &= (1 << remaining_rows) - 1

            state_change = column_vector if output_bit == 1 else 0

            # Reverse the XOR transition.
            state ^= state_change

            t -= 1

    return STCResult(
        y=y,
        total_cost=float(total_cost),
    )


def stc_decode(
    y: np.ndarray,
    H_hat: np.ndarray,
    m: int,
) -> np.ndarray:
    """Decode the message/syndrome from STC stego bits."""

    y = np.asarray(y, dtype=np.uint8)
    H_hat = np.asarray(H_hat, dtype=np.uint8)

    if H_hat.ndim != 2:
        raise ValueError("H_hat must be a 2D array.")

    if m < 0:
        raise ValueError("m must be non-negative.")

    h, w = H_hat.shape

    n = len(y)

    if n != m * w:
        raise ValueError(f"n={n} must equal m*w={m * w}")

    column_vectors = _column_vectors(H_hat)

    state = 0

    message = np.zeros(
        m,
        dtype=np.uint8,
    )

    t = 0

    for block in range(m):
        remaining_rows = m - block

        for column in range(w):
            column_vector = int(column_vectors[column])

            if h > remaining_rows:
                column_vector &= (1 << remaining_rows) - 1

            if y[t]:
                state ^= column_vector

            t += 1

        message[block] = state & 1

        state >>= 1

    return message
