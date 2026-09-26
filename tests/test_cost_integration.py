import numpy as np

from src.ssa.ssa2d import (
    create_trajectory_matrix,
    decompose_trajectory_matrix,
    create_elementary_components,
    reconstruct_component,
)
from src.cost_function.suitability import compute_suitability_map
from src.wmf.weighted_median import weighted_median_filter
from src.cost_function.cost import compute_cost_map


def test_full_ssa_wmf_cost_pipeline():
    # Small deterministic synthetic image
    image = np.arange(64, dtype=np.float64).reshape(8, 8)

    # ---------------------------------------------------------
    # Step 1: 2D-SSA decomposition using a 3x3 window
    # ---------------------------------------------------------
    trajectory = create_trajectory_matrix(
        image,
        window_height=3,
        window_width=3,
    )

    eigenvalues, eigenvectors, singular_values = decompose_trajectory_matrix(trajectory)

    elementary_components = create_elementary_components(
        trajectory,
        eigenvectors,
        singular_values,
    )

    # Reconstruct all 9 SSA components into image space
    reconstructed_components = np.stack(
        [
            reconstruct_component(
                component,
                image.shape,
                window_shape=(3, 3),
            )
            for component in elementary_components
        ],
        axis=0,
    )

    assert reconstructed_components.shape == (9, 8, 8)

    # ---------------------------------------------------------
    # Step 2: Suitability map using components 8 and 9
    # ---------------------------------------------------------
    suitability = compute_suitability_map(
        reconstructed_components,
        start_component=8,
        end_component=9,
    )

    assert suitability.shape == image.shape
    assert np.all(suitability >= 0)
    assert np.isfinite(suitability).all()

    # ---------------------------------------------------------
    # Step 3: WMF smoothing
    # Paper parameters: gamma=5, sigma=3, tau=2
    # ---------------------------------------------------------
    wmf_output = weighted_median_filter(
        suitability,
        gamma=5,
        sigma=3.0,
        tau=2,
    )

    assert wmf_output.shape == image.shape
    assert np.isfinite(wmf_output).all()
    assert np.all(wmf_output >= 0)

    # ---------------------------------------------------------
    # Step 4: Cost map
    # ---------------------------------------------------------
    cost = compute_cost_map(wmf_output)

    assert cost.shape == image.shape
    assert cost.dtype == np.float64
    assert np.isfinite(cost).all()
    assert np.all(cost > 0)
