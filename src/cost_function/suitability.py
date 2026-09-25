"""Phase 7: Embedding suitability map construction.

The suitability map follows the formulation from the paper:

    ζ = |Σ G_m|

where G_m are reconstructed 2D-SSA components selected by their
1-based component numbers.
"""

from __future__ import annotations

import numpy as np


def compute_suitability_map(
    components: np.ndarray,
    start_component: int,
    end_component: int,
) -> np.ndarray:
    """Compute the embedding suitability map ζ.

    Parameters
    ----------
    components:
        Reconstructed 2D-SSA components with shape:

            (number_of_components, height, width)

    start_component:
        First selected component using 1-based numbering.

    end_component:
        Last selected component using 1-based numbering.

    Returns
    -------
    np.ndarray
        Floating-point suitability map ζ.

    Notes
    -----
    The paper defines:

        ζ = |Σ G_m|

    The selected components are summed first and the absolute value
    is applied to the resulting image.
    """

    components = np.asarray(components)

    if components.ndim != 3:
        raise ValueError(
            "components must have shape (number_of_components, height, width)"
        )

    number_of_components = components.shape[0]

    if not (1 <= start_component <= end_component <= number_of_components):
        raise ValueError(
            "component range must satisfy "
            "1 <= start_component <= end_component "
            "<= number_of_components"
        )

    # Convert paper's 1-based component numbering to Python's
    # zero-based indexing.
    start_index = start_component - 1
    end_index = end_component

    selected_components = components[start_index:end_index]

    # Paper definition:
    #
    #     ζ = |Σ G_m|
    #
    suitability = np.abs(np.sum(selected_components, axis=0))

    return suitability.astype(np.float64, copy=False)
