import itertools

import numpy as np
import pytest

from src.embedding.stc import (
    make_random_submatrix,
    stc_decode,
    stc_encode,
)


def _brute_force_optimal(
    x,
    rho,
    msg,
    H_hat,
):
    """Exhaustive search for tiny instances."""

    h, w = H_hat.shape
    n, m = len(x), len(msg)

    best_cost = None
    best_y = None

    for bits in itertools.product([0, 1], repeat=n):
        y = np.array(
            bits,
            dtype=np.uint8,
        )

        decoded = stc_decode(
            y,
            H_hat,
            m,
        )

        if np.array_equal(decoded, msg):
            cost = float(np.sum(rho[y != x]))

            if best_cost is None or cost < best_cost:
                best_cost = cost
                best_y = y

    return best_cost, best_y


@pytest.mark.parametrize(
    "seed",
    range(8),
)
def test_matches_brute_force_optimum_tiny_cases(seed):

    rng = np.random.default_rng(seed)

    h, w, m = 3, 2, 4

    H_hat = make_random_submatrix(
        h,
        w,
        seed,
    )

    x = rng.integers(
        0,
        2,
        size=m * w,
        dtype=np.uint8,
    )

    rho = rng.uniform(
        0.1,
        5.0,
        size=m * w,
    )

    msg = rng.integers(
        0,
        2,
        size=m,
        dtype=np.uint8,
    )

    result = stc_encode(
        x,
        rho,
        msg,
        H_hat,
    )

    brute_cost, _ = _brute_force_optimal(
        x,
        rho,
        msg,
        H_hat,
    )

    assert result.total_cost == pytest.approx(
        brute_cost,
        abs=1e-9,
    )

    decoded = stc_decode(
        result.y,
        H_hat,
        m,
    )

    assert np.array_equal(
        decoded,
        msg,
    )

    assert np.sum(rho[result.y != x]) == pytest.approx(
        result.total_cost,
        abs=1e-9,
    )


def test_zero_cost_when_cover_already_satisfies_message():

    h, w, m = 3, 2, 5

    H_hat = make_random_submatrix(
        h,
        w,
        seed=1,
    )

    rng = np.random.default_rng(2)

    x = rng.integers(
        0,
        2,
        size=m * w,
        dtype=np.uint8,
    )

    msg = stc_decode(
        x,
        H_hat,
        m,
    )

    rho = rng.uniform(
        0.1,
        5.0,
        size=m * w,
    )

    result = stc_encode(
        x,
        rho,
        msg,
        H_hat,
    )

    assert result.total_cost == pytest.approx(
        0.0,
        abs=1e-9,
    )

    assert np.array_equal(
        result.y,
        x,
    )


def test_extraction_recovers_message_on_larger_random_case():

    h, w, m = 6, 2, 200

    H_hat = make_random_submatrix(
        h,
        w,
        seed=42,
    )

    rng = np.random.default_rng(3)

    x = rng.integers(
        0,
        2,
        size=m * w,
        dtype=np.uint8,
    )

    rho = rng.uniform(
        0.1,
        10.0,
        size=m * w,
    )

    msg = rng.integers(
        0,
        2,
        size=m,
        dtype=np.uint8,
    )

    result = stc_encode(
        x,
        rho,
        msg,
        H_hat,
    )

    decoded = stc_decode(
        result.y,
        H_hat,
        m,
    )

    assert np.array_equal(
        decoded,
        msg,
    )


def test_higher_cost_positions_flipped_less_often_than_low_cost():

    h, w, m = 5, 3, 100

    H_hat = make_random_submatrix(
        h,
        w,
        seed=7,
    )

    rng = np.random.default_rng(4)

    n = m * w

    x = rng.integers(
        0,
        2,
        size=n,
        dtype=np.uint8,
    )

    msg = rng.integers(
        0,
        2,
        size=m,
        dtype=np.uint8,
    )

    cheap = np.full(
        n,
        0.1,
    )

    expensive = np.full(
        n,
        100.0,
    )

    rho = np.concatenate(
        [
            cheap[: n // 2],
            expensive[n // 2 :],
        ]
    )

    result = stc_encode(
        x,
        rho,
        msg,
        H_hat,
    )

    flips = result.y != x

    flips_first_half = flips[: n // 2].sum()

    flips_second_half = flips[n // 2 :].sum()

    assert flips_first_half >= flips_second_half


def test_rejects_wrong_length():

    H_hat = make_random_submatrix(
        4,
        2,
        seed=0,
    )

    with pytest.raises(ValueError):
        stc_encode(
            np.zeros(
                7,
                dtype=np.uint8,
            ),
            np.ones(7),
            np.zeros(
                3,
                dtype=np.uint8,
            ),
            H_hat,
        )


def test_make_random_submatrix_reproducible():

    a = make_random_submatrix(
        4,
        3,
        seed=5,
    )

    b = make_random_submatrix(
        4,
        3,
        seed=5,
    )

    assert np.array_equal(
        a,
        b,
    )


def test_make_random_submatrix_no_degenerate_rows():

    for seed in range(50):
        H_hat = make_random_submatrix(
            h=3,
            w=1,
            seed=seed,
        )

        assert np.all(H_hat.any(axis=1))
