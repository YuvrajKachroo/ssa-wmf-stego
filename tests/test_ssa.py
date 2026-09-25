import numpy as np
import pytest

from src.ssa.ssa2d import (
    create_trajectory_matrix,
    decompose_trajectory_matrix,
    create_elementary_components,
    diagonal_averaging,
    reconstruct_component,
)


def test_trajectory_matrix_shape():
    image = np.arange(
        25,
        dtype=np.uint8,
    ).reshape(5, 5)

    trajectory = create_trajectory_matrix(
        image,
        window_height=3,
        window_width=3,
    )

    assert trajectory.shape == (9, 9)
    assert trajectory.dtype == np.float64


def test_trajectory_matrix_values():
    image = np.arange(
        25,
        dtype=np.uint8,
    ).reshape(5, 5)

    trajectory = create_trajectory_matrix(
        image,
        window_height=3,
        window_width=3,
    )

    expected_first_window = image[
        0:3,
        0:3,
    ].reshape(-1)

    expected_second_window = image[
        0:3,
        1:4,
    ].reshape(-1)

    assert np.allclose(
        trajectory[:, 0],
        expected_first_window,
    )

    assert np.allclose(
        trajectory[:, 1],
        expected_second_window,
    )


def test_eigenvalue_decomposition():
    image = np.arange(
        25,
        dtype=np.uint8,
    ).reshape(5, 5)

    trajectory = create_trajectory_matrix(
        image,
        window_height=3,
        window_width=3,
    )

    (
        eigenvalues,
        eigenvectors,
        singular_values,
    ) = decompose_trajectory_matrix(trajectory)

    assert len(eigenvalues) == 9
    assert eigenvectors.shape == (9, 9)
    assert len(singular_values) == 9

    assert np.all(eigenvalues >= 0)

    assert np.all(eigenvalues[:-1] >= eigenvalues[1:])


def test_eigenvectors_are_orthonormal():
    image = np.arange(
        25,
        dtype=np.uint8,
    ).reshape(5, 5)

    trajectory = create_trajectory_matrix(
        image,
        window_height=3,
        window_width=3,
    )

    (
        eigenvalues,
        eigenvectors,
        singular_values,
    ) = decompose_trajectory_matrix(trajectory)

    identity = eigenvectors.T @ eigenvectors

    assert np.allclose(
        identity,
        np.eye(9),
        atol=1e-8,
    )


def test_elementary_component_shapes():
    image = np.arange(
        25,
        dtype=np.uint8,
    ).reshape(5, 5)

    trajectory = create_trajectory_matrix(
        image,
        window_height=3,
        window_width=3,
    )

    (
        eigenvalues,
        eigenvectors,
        singular_values,
    ) = decompose_trajectory_matrix(trajectory)

    components = create_elementary_components(
        trajectory,
        eigenvectors,
        singular_values,
    )

    assert len(components) == 9

    for component in components:
        assert component.shape == trajectory.shape
        assert component.dtype == np.float64


def test_elementary_components_sum_to_trajectory():
    image = np.arange(
        25,
        dtype=np.uint8,
    ).reshape(5, 5)

    trajectory = create_trajectory_matrix(
        image,
        window_height=3,
        window_width=3,
    )

    (
        eigenvalues,
        eigenvectors,
        singular_values,
    ) = decompose_trajectory_matrix(trajectory)

    components = create_elementary_components(
        trajectory,
        eigenvectors,
        singular_values,
    )

    reconstructed_trajectory = sum(components)

    assert np.allclose(
        reconstructed_trajectory,
        trajectory,
        atol=1e-8,
    )


def test_diagonal_averaging_shape():
    image = np.arange(
        25,
        dtype=np.uint8,
    ).reshape(5, 5)

    trajectory = create_trajectory_matrix(
        image,
        window_height=3,
        window_width=3,
    )

    (
        eigenvalues,
        eigenvectors,
        singular_values,
    ) = decompose_trajectory_matrix(trajectory)

    components = create_elementary_components(
        trajectory,
        eigenvectors,
        singular_values,
    )

    reconstructed = diagonal_averaging(
        components[0],
        image_shape=image.shape,
        window_shape=(3, 3),
    )

    assert reconstructed.shape == image.shape
    assert reconstructed.dtype == np.float64
    assert np.all(np.isfinite(reconstructed))


def test_reconstruct_component():
    image = np.arange(
        25,
        dtype=np.uint8,
    ).reshape(5, 5)

    trajectory = create_trajectory_matrix(
        image,
        window_height=3,
        window_width=3,
    )

    (
        eigenvalues,
        eigenvectors,
        singular_values,
    ) = decompose_trajectory_matrix(trajectory)

    components = create_elementary_components(
        trajectory,
        eigenvectors,
        singular_values,
    )

    reconstructed = reconstruct_component(
        components[0],
        image_shape=image.shape,
        window_shape=(3, 3),
    )

    assert reconstructed.shape == image.shape
    assert np.all(np.isfinite(reconstructed))


def test_all_components_reconstruct_original_image():
    image = np.arange(
        64 * 64,
        dtype=np.float64,
    ).reshape(64, 64)

    trajectory = create_trajectory_matrix(
        image,
        window_height=3,
        window_width=3,
    )

    (
        eigenvalues,
        eigenvectors,
        singular_values,
    ) = decompose_trajectory_matrix(trajectory)

    components = create_elementary_components(
        trajectory,
        eigenvectors,
        singular_values,
    )

    reconstructed_components = []

    for component in components:
        reconstructed = diagonal_averaging(
            component,
            image_shape=image.shape,
            window_shape=(3, 3),
        )

        reconstructed_components.append(reconstructed)

    reconstructed_image = np.sum(
        reconstructed_components,
        axis=0,
    )

    assert np.allclose(
        reconstructed_image,
        image,
        atol=1e-8,
    )


def test_invalid_image_dimension():
    image = np.zeros(
        (5, 5, 3),
        dtype=np.uint8,
    )

    with pytest.raises(ValueError):
        create_trajectory_matrix(
            image,
            window_height=3,
            window_width=3,
        )


def test_invalid_window_size():
    image = np.zeros(
        (5, 5),
        dtype=np.uint8,
    )

    with pytest.raises(ValueError):
        create_trajectory_matrix(
            image,
            window_height=0,
            window_width=3,
        )


def test_window_larger_than_image():
    image = np.zeros(
        (5, 5),
        dtype=np.uint8,
    )

    with pytest.raises(ValueError):
        create_trajectory_matrix(
            image,
            window_height=6,
            window_width=3,
        )


def test_non_square_window():
    image = np.arange(
        30,
        dtype=np.uint8,
    ).reshape(5, 6)

    trajectory = create_trajectory_matrix(
        image,
        window_height=2,
        window_width=3,
    )

    assert trajectory.shape == (6, 16)
