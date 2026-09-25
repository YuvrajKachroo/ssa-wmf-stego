import numpy as np
import pytest

from src.ssa.reconstruction import (
    normalize_for_display,
    reconstruct_image,
)
from src.ssa.selection import (
    select_bottom_k,
    select_top_k,
)


def test_reconstruct_all_components_equals_original():
    image = np.arange(
        48 * 48,
        dtype=np.float64,
    ).reshape(48, 48)

    from src.ssa.ssa2d import (
        create_trajectory_matrix,
        decompose_trajectory_matrix,
        create_elementary_components,
        diagonal_averaging,
    )

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

    trajectory_components = create_elementary_components(
        trajectory,
        eigenvectors,
        singular_values,
    )

    components = np.array(
        [
            diagonal_averaging(
                component,
                image_shape=image.shape,
                window_shape=(3, 3),
            )
            for component in trajectory_components
        ]
    )

    indices = np.arange(len(components))

    reconstruction = reconstruct_image(
        components,
        indices,
        image.shape,
    )

    assert np.allclose(
        reconstruction,
        image,
        atol=1e-8,
    )


def test_reconstruct_subset_matches_manual_sum():
    image = np.arange(
        48 * 48,
        dtype=np.float64,
    ).reshape(48, 48)

    from src.ssa.ssa2d import (
        create_trajectory_matrix,
        decompose_trajectory_matrix,
        create_elementary_components,
        diagonal_averaging,
    )

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

    trajectory_components = create_elementary_components(
        trajectory,
        eigenvectors,
        singular_values,
    )

    components = np.array(
        [
            diagonal_averaging(
                component,
                image_shape=image.shape,
                window_shape=(3, 3),
            )
            for component in trajectory_components
        ]
    )

    indices = select_bottom_k(
        eigenvalues,
        2,
    )

    reconstruction = reconstruct_image(
        components,
        indices,
        image.shape,
    )

    manual = components[indices[0]] + components[indices[1]]

    assert np.allclose(
        reconstruction,
        manual,
    )


def test_high_frequency_subset_has_lower_energy():
    image = np.arange(
        64 * 64,
        dtype=np.float64,
    ).reshape(64, 64)

    from src.ssa.ssa2d import (
        create_trajectory_matrix,
        decompose_trajectory_matrix,
        create_elementary_components,
        diagonal_averaging,
    )

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

    trajectory_components = create_elementary_components(
        trajectory,
        eigenvectors,
        singular_values,
    )

    components = np.array(
        [
            diagonal_averaging(
                component,
                image_shape=image.shape,
                window_shape=(3, 3),
            )
            for component in trajectory_components
        ]
    )

    low_frequency = reconstruct_image(
        components,
        select_top_k(eigenvalues, 2),
        image.shape,
    )

    high_frequency = reconstruct_image(
        components,
        select_bottom_k(eigenvalues, 2),
        image.shape,
    )

    assert high_frequency.std() < low_frequency.std()


def test_normalize_for_display_range():
    image = np.array(
        [
            [-3.0, 0.0],
            [1.0, 5.0],
        ]
    )

    result = normalize_for_display(image)

    assert result.dtype == np.uint8
    assert result.min() == 0
    assert result.max() == 255


def test_normalize_constant_image():
    image = np.full(
        (4, 4),
        7.0,
    )

    result = normalize_for_display(image)

    assert result.dtype == np.uint8
    assert np.all(result == 0)


def test_reconstruction_shape_mismatch():
    components = np.ones(
        (3, 8, 8),
        dtype=np.float64,
    )

    indices = np.array(
        [0, 1],
        dtype=int,
    )

    with pytest.raises(RuntimeError):
        reconstruct_image(
            components,
            indices,
            image_shape=(16, 16),
        )
