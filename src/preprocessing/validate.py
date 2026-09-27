"""Dataset and image validation utilities."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from .loader import (
    image_id_from_path,
    list_images,
    load_grayscale,
)


DEFAULT_EXPECTED_SHAPE = (512, 512)
DEFAULT_ID_RANGE = (1, 10000)


def validate_array(
    array: np.ndarray,
    expected_shape: tuple[int, int] = DEFAULT_EXPECTED_SHAPE,
) -> list[str]:
    """Return validation problems.

    An empty list means the array is valid.
    """

    problems: list[str] = []

    if array.ndim != 2:
        problems.append(
            f"ndim={array.ndim}, expected 2"
        )

    if array.dtype != np.uint8:
        problems.append(
            f"dtype={array.dtype}, expected uint8"
        )

    if (
        array.ndim == 2
        and tuple(array.shape)
        != tuple(expected_shape)
    ):
        problems.append(
            f"shape={tuple(array.shape)}, "
            f"expected {tuple(expected_shape)}"
        )

    if array.size > 0:
        if int(array.min()) < 0:
            problems.append(
                f"minimum pixel value {array.min()} is below 0"
            )

        if int(array.max()) > 255:
            problems.append(
                f"maximum pixel value {array.max()} is above 255"
            )

    return problems


def pixel_hash(
    array: np.ndarray,
) -> str:
    """Return SHA-256 hash of the image pixel bytes."""

    return hashlib.sha256(
        np.ascontiguousarray(array).tobytes()
    ).hexdigest()


def validate_directory(
    directory: str | Path,
    expected_shape: tuple[int, int] = DEFAULT_EXPECTED_SHAPE,
    id_range: tuple[int, int] = DEFAULT_ID_RANGE,
    extensions: tuple[str, ...] = (".pgm",),
) -> pd.DataFrame:
    """Validate every supported image in a directory.

    Returns columns:

        image_id
        path
        ok
        problems
        sha256
    """

    rows: list[dict] = []

    for path in list_images(
        directory,
        extensions=extensions,
    ):
        problems: list[str] = []
        sha = ""
        image_id = -1

        try:
            image_id = image_id_from_path(path)

            if not (
                id_range[0]
                <= image_id
                <= id_range[1]
            ):
                problems.append(
                    f"id {image_id} outside {id_range}"
                )

            array = load_grayscale(path)

            problems.extend(
                validate_array(
                    array,
                    expected_shape=expected_shape,
                )
            )

            sha = pixel_hash(array)

        except Exception as exc:
            problems.append(
                f"load_error: {exc}"
            )

        rows.append(
            {
                "image_id": image_id,
                "path": str(path),
                "ok": False,
                "problems": ";".join(problems),
                "sha256": sha,
            }
        )

    columns = [
        "image_id",
        "path",
        "ok",
        "problems",
        "sha256",
    ]

    dataframe = pd.DataFrame(
        rows,
        columns=columns,
    )

    if dataframe.empty:
        return dataframe

    duplicate_mask = (
        dataframe["sha256"].ne("")
        & dataframe.duplicated(
            "sha256",
            keep="first",
        )
    )

    dataframe.loc[
        duplicate_mask,
        "problems",
    ] = dataframe.loc[
        duplicate_mask,
        "problems",
    ].apply(
        lambda text: (
            f"{text};duplicate_pixels"
            if text
            else "duplicate_pixels"
        )
    )

    dataframe["ok"] = (
        dataframe["problems"] == ""
    )

    return dataframe
