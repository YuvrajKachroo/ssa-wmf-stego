from __future__ import annotations

import numpy as np


def create_trajectory_matrix(
    image: np.ndarray,
    window_height: int,
    window_width: int,
) -> np.ndarray:
    """
    Create the 2D-SSA trajectory matrix.

    Parameters
    ----------
    image:
        2D grayscale image.
    window_height:
        Height of the SSA window.
    window_width:
        Width of the SSA window.

    Returns
    -------
    np.ndarray
        Trajectory matrix.
    """

    if image.ndim != 2:
        raise ValueError("Image must be a 2D grayscale array.")

    if window_height <= 0 or window_width <= 0:
        raise ValueError("Window dimensions must be positive.")

    height, width = image.shape

    if window_height > height or window_width > width:
        raise ValueError("SSA window cannot be larger than the image.")

    rows = height - window_height + 1
    cols = width - window_width + 1

    trajectory = np.empty(
        (
            window_height * window_width,
            rows * cols,
        ),
        dtype=np.float64,
    )

    column = 0

    for i in range(rows):
        for j in range(cols):
            window = image[
                i : i + window_height,
                j : j + window_width,
            ]

            trajectory[:, column] = window.reshape(-1)
            column += 1

    return trajectory


def decompose_trajectory_matrix(
    trajectory: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform eigenvalue decomposition of the trajectory matrix.

    Returns
    -------
    eigenvalues:
        Eigenvalues sorted in descending order.

    eigenvectors:
        Corresponding eigenvectors.

    singular_values:
        Square roots of the eigenvalues.
    """

    if trajectory.ndim != 2:
        raise ValueError("Trajectory matrix must be 2D.")

    covariance = trajectory @ trajectory.T

    eigenvalues, eigenvectors = np.linalg.eigh(covariance)

    order = np.argsort(eigenvalues)[::-1]

    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    eigenvalues = np.maximum(
        eigenvalues,
        0.0,
    )

    singular_values = np.sqrt(eigenvalues)

    return (
        eigenvalues,
        eigenvectors,
        singular_values,
    )


def create_elementary_components(
    trajectory: np.ndarray,
    eigenvectors: np.ndarray,
    singular_values: np.ndarray,
) -> list[np.ndarray]:
    """
    Create elementary SSA components.

    Each elementary matrix is:

        A_i = U_i (U_i^T Y)

    where U_i is the i-th eigenvector and Y
    is the trajectory matrix.

    The singular values are accepted for API compatibility,
    but are not used in the reconstruction formula.
    """

    if trajectory.ndim != 2:
        raise ValueError("Trajectory matrix must be 2D.")

    if eigenvectors.ndim != 2:
        raise ValueError("Eigenvectors must be a 2D array.")

    if singular_values.ndim != 1:
        raise ValueError("Singular values must be a 1D array.")

    number_of_components = min(
        eigenvectors.shape[1],
        len(singular_values),
    )

    components = []

    for i in range(number_of_components):
        eigenvector = eigenvectors[:, i]

        projection = eigenvector @ trajectory

        component = np.outer(
            eigenvector,
            projection,
        )

        components.append(component)

    return components


def diagonal_averaging(
    component: np.ndarray,
    image_shape: tuple[int, int],
    window_shape: tuple[int, int],
) -> np.ndarray:
    """
    Reconstruct a 2D image from an SSA trajectory component.

    The reconstruction is performed by averaging
    overlapping windows.
    """

    image_height, image_width = image_shape

    window_height, window_width = window_shape

    expected_rows = image_height - window_height + 1

    expected_cols = image_width - window_width + 1

    expected_shape = (
        window_height * window_width,
        expected_rows * expected_cols,
    )

    if component.shape != expected_shape:
        raise ValueError(
            "Component shape does not match the supplied image and window dimensions."
        )

    reconstructed = np.zeros(
        image_shape,
        dtype=np.float64,
    )

    weights = np.zeros(
        image_shape,
        dtype=np.float64,
    )

    column = 0

    for i in range(expected_rows):
        for j in range(expected_cols):
            window = component[:, column].reshape(
                window_height,
                window_width,
            )

            reconstructed[
                i : i + window_height,
                j : j + window_width,
            ] += window

            weights[
                i : i + window_height,
                j : j + window_width,
            ] += 1.0

            column += 1

    reconstructed /= np.maximum(
        weights,
        1.0,
    )

    return reconstructed


def reconstruct_component(
    component: np.ndarray,
    image_shape: tuple[int, int],
    window_shape: tuple[int, int],
) -> np.ndarray:
    """
    Convenience wrapper for reconstructing
    a single SSA component.
    """

    return diagonal_averaging(
        component=component,
        image_shape=image_shape,
        window_shape=window_shape,
    )
