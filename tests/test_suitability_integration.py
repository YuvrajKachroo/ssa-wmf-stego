import numpy as np

from src.cost_function.suitability import compute_suitability_map
from src.ssa.ssa2d import (
    create_trajectory_matrix,
    decompose_trajectory_matrix,
    create_elementary_components,
    reconstruct_component,
)


def test_suitability_with_real_ssa_pipeline():
    """Verify ζ using the actual 2D-SSA implementation."""

    # Small deterministic image for a fast integration test.
    image = np.arange(64, dtype=np.float64).reshape(8, 8)

    # Paper uses a 3x3 SSA window.
    trajectory = create_trajectory_matrix(
        image,
        3,
        3,
    )

    eigenvalues, eigenvectors, singular_values = decompose_trajectory_matrix(trajectory)

    components = create_elementary_components(
        trajectory,
        eigenvectors,
        singular_values,
    )

    reconstructed_components = np.stack(
        [
            reconstruct_component(
                component,
                image.shape,
                (3, 3),
            )
            for component in components
        ]
    )

    # A 3x3 window produces 9 components.
    assert reconstructed_components.shape == (9, 8, 8)

    # Paper's selected components: 8th and 9th.
    suitability = compute_suitability_map(
        reconstructed_components,
        8,
        9,
    )

    expected = np.abs(reconstructed_components[7] + reconstructed_components[8])

    np.testing.assert_allclose(
        suitability,
        expected,
    )

    assert suitability.shape == image.shape
    assert np.all(suitability >= 0)
    assert np.isfinite(suitability).all()
