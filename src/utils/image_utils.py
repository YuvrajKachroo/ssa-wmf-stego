from pathlib import Path

import cv2
import numpy as np


def load_grayscale_image(path: str | Path) -> np.ndarray:
    """Load an image as an 8-bit grayscale NumPy array."""
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")

    return image


def save_grayscale_image(
    image: np.ndarray,
    path: str | Path,
) -> None:
    """Save a grayscale NumPy array as an image."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    success = cv2.imwrite(str(path), image)

    if not success:
        raise IOError(f"Could not save image: {path}")
