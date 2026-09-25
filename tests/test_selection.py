import numpy as np
import pytest

from src.ssa.selection import (
    select_by_rank_range,
    select_top_k,
    select_bottom_k,
    select_explicit,
    group_and_sum,
)


def test_select_by_rank_range():
    eigenvalues = np.arange(
        9,
        dtype=np.float64,
    )

    indices = select_by_rank_range(
        eigenvalues,
        s=8,
        t=9,
    )

    assert np.array_equal(
        indices,
        np.array([7, 8]),
    )


def test_invalid_rank_range():
    eigenvalues = np.arange(
        9,
        dtype=np.float64,
    )

    with pytest.raises(ValueError):
        select_by_rank_range(
            eigenvalues,
            s=0,
            t=9,
        )

    with pytest.raises(ValueError):
        select_by_rank_range(
            eigenvalues,
            s=8,
            t=10,
        )

    with pytest.raises(ValueError):
        select_by_rank_range(
            eigenvalues,
            s=9,
            t=8,
        )


def test_top_and_bottom_k():
    eigenvalues = np.array(
        [
            100,
            80,
            60,
            40,
            20,
        ],
        dtype=np.float64,
    )

    top = select_top_k(
        eigenvalues,
        k=2,
    )

    bottom = select_bottom_k(
        eigenvalues,
        k=2,
    )

    assert np.array_equal(
        top,
        np.array([0, 1]),
    )

    assert np.array_equal(
        bottom,
        np.array([3, 4]),
    )


def test_select_explicit():
    indices = select_explicit([4, 2, 4, 1, 2])

    assert np.array_equal(
        indices,
        np.array([1, 2, 4]),
    )


def test_group_and_sum():
    components = np.array(
        [
            np.ones((4, 4)),
            np.full((4, 4), 2.0),
            np.full((4, 4), 3.0),
        ]
    )

    indices = np.array([1, 2])

    result = group_and_sum(
        components,
        indices,
    )

    expected = np.full(
        (4, 4),
        5.0,
    )

    assert np.allclose(
        result,
        expected,
    )


def test_group_and_sum_empty():
    components = np.ones((3, 4, 4))

    with pytest.raises(ValueError):
        group_and_sum(
            components,
            np.array([], dtype=int),
        )


def test_group_and_sum_invalid_shape():
    components = np.ones((3, 4))

    with pytest.raises(ValueError):
        group_and_sum(
            components,
            np.array([0]),
        )


def test_group_and_sum_invalid_index():
    components = np.ones((3, 4, 4))

    with pytest.raises(ValueError):
        group_and_sum(
            components,
            np.array([5]),
        )
