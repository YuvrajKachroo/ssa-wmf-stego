import numpy as np
import pytest

from src.cost_function.suitability import compute_suitability_map


def test_suitability_sums_selected_components():
    components = np.array(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[10.0, 20.0], [30.0, 40.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ]
    )

    result = compute_suitability_map(
        components,
        1,
        2,
    )

    expected = np.abs(components[0] + components[1])

    np.testing.assert_array_equal(result, expected)


def test_absolute_value_is_applied_after_sum():
    components = np.array(
        [
            [[10.0, -5.0], [3.0, -8.0]],
            [[-15.0, 2.0], [-7.0, 10.0]],
        ]
    )

    result = compute_suitability_map(
        components,
        1,
        2,
    )

    expected = np.array(
        [
            [5.0, 3.0],
            [4.0, 2.0],
        ]
    )

    np.testing.assert_array_equal(result, expected)


def test_component_range_uses_one_based_numbering():
    components = np.array(
        [
            [[1.0, 1.0]],
            [[2.0, 2.0]],
            [[3.0, 3.0]],
            [[4.0, 4.0]],
        ]
    )

    result = compute_suitability_map(
        components,
        2,
        3,
    )

    expected = np.array([[5.0, 5.0]])

    np.testing.assert_array_equal(result, expected)


def test_paper_components_8_and_9():
    components = np.zeros(
        (9, 2, 2),
        dtype=np.float64,
    )

    components[7] = np.array(
        [
            [1.0, -2.0],
            [3.0, -4.0],
        ]
    )

    components[8] = np.array(
        [
            [-5.0, 6.0],
            [-7.0, 8.0],
        ]
    )

    result = compute_suitability_map(
        components,
        8,
        9,
    )

    expected = np.abs(components[7] + components[8])

    np.testing.assert_array_equal(result, expected)


def test_output_shape_is_preserved():
    components = np.ones(
        (9, 8, 10),
        dtype=np.float64,
    )

    result = compute_suitability_map(
        components,
        8,
        9,
    )

    assert result.shape == (8, 10)


def test_output_is_float64():
    components = np.ones(
        (9, 4, 4),
        dtype=np.float32,
    )

    result = compute_suitability_map(
        components,
        8,
        9,
    )

    assert result.dtype == np.float64


def test_invalid_component_dimensions():
    components = np.ones((4, 4))

    with pytest.raises(ValueError):
        compute_suitability_map(
            components,
            1,
            2,
        )


def test_invalid_component_range():
    components = np.ones(
        (9, 4, 4),
        dtype=np.float64,
    )

    with pytest.raises(ValueError):
        compute_suitability_map(
            components,
            0,
            2,
        )

    with pytest.raises(ValueError):
        compute_suitability_map(
            components,
            8,
            10,
        )

    with pytest.raises(ValueError):
        compute_suitability_map(
            components,
            5,
            4,
        )
