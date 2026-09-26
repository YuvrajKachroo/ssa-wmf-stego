import numpy as np
import pytest

from src.embedding.stc import make_random_submatrix
from src.embedding.ternary import (
    directional_flip_costs,
    embed_ternary_lsb,
    extract_ternary_lsb,
)


def test_directional_costs_at_boundaries():

    image = np.array(
        [[0, 1, 254, 255]],
        dtype=np.uint8,
    )

    cost = np.ones_like(
        image,
        dtype=np.float64,
    )

    rho_plus, rho_minus = directional_flip_costs(
        image,
        cost,
    )

    assert np.isinf(rho_minus[0, 0])
    assert np.isfinite(rho_plus[0, 0])

    assert np.isfinite(rho_plus[0, 1])
    assert np.isfinite(rho_minus[0, 1])

    assert np.isfinite(rho_plus[0, 2])
    assert np.isfinite(rho_minus[0, 2])

    assert np.isfinite(rho_minus[0, 3])
    assert np.isinf(rho_plus[0, 3])


def test_directional_costs_preserve_regular_pixels():

    image = np.array(
        [[1, 100, 200]],
        dtype=np.uint8,
    )

    cost = np.array(
        [[2.0, 3.0, 4.0]],
        dtype=np.float64,
    )

    rho_plus, rho_minus = directional_flip_costs(
        image,
        cost,
    )

    assert np.array_equal(
        rho_plus,
        cost,
    )

    assert np.array_equal(
        rho_minus,
        cost,
    )


def test_embed_and_extract_message_bits():

    rng = np.random.default_rng(42)

    image = rng.integers(
        0,
        256,
        size=(32, 32),
        dtype=np.uint8,
    )

    cost_map = np.ones(
        image.shape,
        dtype=np.float64,
    )

    h, w, message_length = 5, 2, 512

    H_hat = make_random_submatrix(
        h,
        w,
        seed=42,
    )

    message_bits = rng.integers(
        0,
        2,
        size=message_length,
        dtype=np.uint8,
    )

    stego, result = embed_ternary_lsb(
        image,
        cost_map,
        message_bits,
        H_hat,
    )

    extracted = extract_ternary_lsb(
        stego,
        H_hat,
        message_length,
    )

    assert np.array_equal(
        extracted,
        message_bits,
    )

    assert result.total_cost >= 0


def test_pixel_changes_are_at_most_one():

    rng = np.random.default_rng(10)

    image = rng.integers(
        1,
        255,
        size=(32, 32),
        dtype=np.uint8,
    )

    cost_map = np.ones(
        image.shape,
        dtype=np.float64,
    )

    H_hat = make_random_submatrix(
        4,
        2,
        seed=4,
    )

    message_bits = rng.integers(
        0,
        2,
        size=512,
        dtype=np.uint8,
    )

    stego, _ = embed_ternary_lsb(
        image,
        cost_map,
        message_bits,
        H_hat,
    )

    difference = np.abs(image.astype(np.int16) - stego.astype(np.int16))

    assert np.all(difference <= 1)


def test_output_is_valid_uint8_image():

    rng = np.random.default_rng(20)

    image = rng.integers(
        0,
        256,
        size=(32, 32),
        dtype=np.uint8,
    )

    cost_map = np.ones(
        image.shape,
        dtype=np.float64,
    )

    H_hat = make_random_submatrix(
        5,
        2,
        seed=9,
    )

    message_bits = rng.integers(
        0,
        2,
        size=512,
        dtype=np.uint8,
    )

    stego, _ = embed_ternary_lsb(
        image,
        cost_map,
        message_bits,
        H_hat,
    )

    assert stego.dtype == np.uint8
    assert stego.shape == image.shape
    assert np.all(stego <= 255)


def test_rejects_wrong_cost_shape():

    image = np.zeros(
        (8, 8),
        dtype=np.uint8,
    )

    cost_map = np.ones(
        (8, 7),
        dtype=np.float64,
    )

    with pytest.raises(ValueError):
        directional_flip_costs(
            image,
            cost_map,
        )
