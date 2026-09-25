import numpy as np
import pytest

from src.wmf.weighted_median import (
    similarity_weight,
    weighted_median_filter,
)


def test_similarity_weight_identical_values():
    assert similarity_weight(10.0, 10.0, 3.0) == pytest.approx(1.0)


def test_similarity_weight_decreases_with_difference():
    near = similarity_weight(10.0, 11.0, 3.0)
    far = similarity_weight(10.0, 20.0, 3.0)

    assert near > far


def test_constant_image_is_unchanged():
    image = np.full((5, 5), 100.0)

    result = weighted_median_filter(
        image,
        gamma=1,
        sigma=3.0,
        tau=2,
    )

    np.testing.assert_array_equal(result, image)


def test_gamma_zero_is_identity():
    image = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
    )

    result = weighted_median_filter(
        image,
        gamma=0,
        sigma=3.0,
        tau=2,
    )

    np.testing.assert_array_equal(result, image)


def test_output_shape_is_preserved():
    image = np.random.default_rng(0).normal(size=(8, 11))

    result = weighted_median_filter(
        image,
        gamma=1,
        sigma=3.0,
        tau=1,
    )

    assert result.shape == image.shape


def test_output_is_float64():
    image = np.arange(25, dtype=np.uint8).reshape(5, 5)

    result = weighted_median_filter(
        image,
        gamma=1,
        sigma=3.0,
        tau=1,
    )

    assert result.dtype == np.float64


def test_multiple_iterations_match_repeated_single_iteration():
    image = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 10.0, 20.0, 0.0],
            [0.0, 30.0, 40.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )

    once = weighted_median_filter(
        image,
        gamma=1,
        sigma=3.0,
        tau=1,
    )

    twice_manually = weighted_median_filter(
        once,
        gamma=1,
        sigma=3.0,
        tau=1,
    )

    twice_direct = weighted_median_filter(
        image,
        gamma=1,
        sigma=3.0,
        tau=2,
    )

    np.testing.assert_allclose(twice_direct, twice_manually)


def test_invalid_parameters():
    image = np.ones((5, 5))

    with pytest.raises(ValueError):
        weighted_median_filter(image, gamma=-1)

    with pytest.raises(ValueError):
        weighted_median_filter(image, sigma=0)

    with pytest.raises(ValueError):
        weighted_median_filter(image, tau=0)


def test_invalid_image_dimension():
    image = np.ones((5, 5, 3))

    with pytest.raises(ValueError):
        weighted_median_filter(image)
