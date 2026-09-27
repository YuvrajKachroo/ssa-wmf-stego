import numpy as np
import pytest

from src.steganalysis.features import (
    difference_histogram,
    extract_features,
    global_moments,
)


def make_test_image(
    seed: int,
    shape: tuple[int, int] = (32, 32),
) -> np.ndarray:
    rng = np.random.default_rng(seed)

    return rng.integers(
        0,
        256,
        size=shape,
        dtype=np.uint8,
    )


def test_difference_histogram_sums_to_one():
    image = make_test_image(0)

    histogram = difference_histogram(
        image,
        axis=1,
        T=3,
    )

    assert histogram.shape == (7,)
    assert histogram.sum() == pytest.approx(1.0)


def test_difference_histogram_flat_image_is_delta_at_zero():

    image = np.full(
        (16, 16),
        100,
        dtype=np.float64,
    )

    histogram = difference_histogram(
        image,
        axis=1,
        T=3,
    )

    expected = np.zeros(7)
    expected[3] = 1.0

    assert np.allclose(
        histogram,
        expected,
    )


def test_global_moments_shape_and_known_values():

    image = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
    )

    moments = global_moments(image)

    assert moments.shape == (8,)
    assert moments[0] == pytest.approx(2.5)


def test_extract_features_deterministic_and_correct_dim():

    image = make_test_image(1)

    first = extract_features(
        image,
        T=3,
    )

    second = extract_features(
        image,
        T=3,
    )

    assert np.array_equal(
        first,
        second,
    )

    assert first.shape == (22,)


def test_features_differ_between_different_images():

    image_1 = make_test_image(0)
    image_2 = make_test_image(1)

    features_1 = extract_features(image_1)

    features_2 = extract_features(image_2)

    assert not np.allclose(
        features_1,
        features_2,
    )
