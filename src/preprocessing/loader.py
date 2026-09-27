"""Strict image I/O for the steganography pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def load_grayscale(
    path: str | Path,
) -> np.ndarray:
    """Load an 8-bit grayscale image as a 2-D uint8 array.

    The loader intentionally does not silently convert RGB or other
    image modes. Such files are rejected so dataset problems are
    detected early.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Image file does not exist: {path}"
        )

    if not path.is_file():
        raise ValueError(
            f"Expected an image file, got: {path}"
        )

    try:
        with Image.open(path) as image:
            if image.mode != "L":
                raise ValueError(
                    f"{path.name}: expected 8-bit grayscale "
                    f"(mode 'L'), got '{image.mode}'"
                )

            array = np.asarray(
                image,
                dtype=np.uint8,
            )

    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(
            f"Could not read image '{path}': {exc}"
        ) from exc

    if array.ndim != 2:
        raise ValueError(
            f"{path.name}: expected a 2-D image, "
            f"got {array.ndim}-D data"
        )

    if array.dtype != np.uint8:
        raise ValueError(
            f"{path.name}: expected uint8 data, "
            f"got {array.dtype}"
        )

    return array


def save_grayscale(
    array: np.ndarray,
    path: str | Path,
) -> None:
    """Save a 2-D uint8 array losslessly."""

    array = np.asarray(array)

    if array.ndim != 2:
        raise ValueError(
            f"expected 2-D array, got {array.ndim}-D"
        )

    if array.dtype != np.uint8:
        raise ValueError(
            f"expected uint8 array, got {array.dtype}"
        )

    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    Image.fromarray(array).save(path)


def image_id_from_path(
    path: str | Path,
) -> int:
    """Extract a numeric image ID from a filename.

    Examples:
        1.pgm   -> 1
        10000.pgm -> 10000
    """

    path = Path(path)
    stem = path.stem

    if not stem.isdigit():
        raise ValueError(
            f"filename stem is not a numeric image ID: "
            f"{path.name}"
        )

    return int(stem)


def list_images(
    directory: str | Path,
    extensions: tuple[str, ...] = (".pgm",),
) -> list[Path]:
    """Return image files sorted by numeric image ID."""

    directory = Path(directory)

    if not directory.exists():
        raise FileNotFoundError(
            f"Image directory does not exist: {directory}"
        )

    if not directory.is_dir():
        raise ValueError(
            f"Expected a directory, got: {directory}"
        )

    normalized_extensions = {
        extension.lower()
        for extension in extensions
    }

    files = [
        path
        for path in directory.iterdir()
        if path.is_file()
        and path.suffix.lower()
        in normalized_extensions
    ]

    return sorted(
        files,
        key=lambda path: (
            0,
            int(path.stem),
        )
        if path.stem.isdigit()
        else (
            1,
            path.stem,
        ),
    )
