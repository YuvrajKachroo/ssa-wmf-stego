import numpy as np
import pytest

from src.cost_function.cost import (
    PAPER_EPSILON,
    compute_cost_map,
)


def test_paper_epsilon():
    assert PAPER_EPSILON == pytest.approx(np.exp(-10.0))


def test_inverse_cost_relationship():
    suitability = np.array(
        [
            [1.0, 2.0],
            [4.0, 10.0],
        ]
    )

    result = compute_cost_map(suitability)

    expected = 1.0 / (suitability + PAPER_EPSILON)

    np.testing.assert_allclose(
        result,
        expected,
    )


def test_higher_suitability_has_lower_cost():
    suitability = np.array([[1.0, 2.0, 4.0, 8.0]])

    cost = compute_cost_map(suitability)

    assert np.all(np.diff(cost[0]) < 0)


def test_cost_is_positive():
    suitability = np.array(
        [
            [0.0, 1.0],
            [10.0, 100.0],
        ]
    )

    cost = compute_cost_map(suitability)

    assert np.all(cost > 0)


def test_zero_suitability_is_finite():
    suitability = np.zeros(
        (4, 4),
        dtype=np.float64,
    )

    cost = compute_cost_map(suitability)

    assert np.isfinite(cost).all()
    assert np.allclose(
        cost,
        1.0 / PAPER_EPSILON,
    )


def test_output_shape_is_preserved():
    suitability = np.ones(
        (7, 9),
        dtype=np.float64,
    )

    cost = compute_cost_map(suitability)

    assert cost.shape == suitability.shape


def test_output_is_float64():
    suitability = np.ones(
        (4, 4),
        dtype=np.float32,
    )

    cost = compute_cost_map(suitability)

    assert cost.dtype == np.float64


def test_negative_wmf_output_is_rejected():
    suitability = np.array(
        [
            [1.0, -0.5],
            [2.0, 3.0],
        ]
    )

    with pytest.raises(ValueError):
        compute_cost_map(suitability)


def test_invalid_dimensions_are_rejected():
    suitability = np.ones(
        (4, 4, 1),
        dtype=np.float64,
    )

    with pytest.raises(ValueError):
        compute_cost_map(suitability)


def test_invalid_epsilon_is_rejected():
    suitability = np.ones(
        (4, 4),
        dtype=np.float64,
    )

    with pytest.raises(ValueError):
        compute_cost_map(
            suitability,
            epsilon=0.0,
        )

    with pytest.raises(ValueError):
        compute_cost_map(
            suitability,
            epsilon=-1.0,
        )
