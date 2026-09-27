"""Deterministic subset selection and dataset splitting."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


SPLITS = (
    "train",
    "val",
    "test",
)


def select_subset(
    image_ids,
    n: int,
    seed: int,
) -> list[int]:
    """Select n unique image IDs reproducibly."""

    ids = np.array(
        sorted(
            set(
                int(image_id)
                for image_id in image_ids
            )
        )
    )

    if n < 0:
        raise ValueError(
            "n must be non-negative"
        )

    if n > len(ids):
        raise ValueError(
            f"requested {n} images but only "
            f"{len(ids)} are available"
        )

    rng = np.random.default_rng(seed)

    selected = rng.choice(
        ids,
        size=n,
        replace=False,
    )

    return sorted(
        int(image_id)
        for image_id in selected
    )


def make_splits(
    image_ids,
    ratios: dict[str, float],
    seed: int,
) -> pd.DataFrame:
    """Split unique image IDs into train/val/test."""

    if set(ratios) != set(SPLITS):
        raise ValueError(
            f"ratios must have keys {SPLITS}"
        )

    if any(
        ratio < 0
        for ratio in ratios.values()
    ):
        raise ValueError(
            "split ratios cannot be negative"
        )

    total = sum(
        ratios.values()
    )

    if abs(total - 1.0) > 1e-9:
        raise ValueError(
            f"ratios must sum to 1, got {total}"
        )

    ids = np.array(
        sorted(
            set(
                int(image_id)
                for image_id in image_ids
            )
        )
    )

    rng = np.random.default_rng(seed)
    rng.shuffle(ids)

    n = len(ids)

    n_train = round(
        n * ratios["train"]
    )

    n_val = round(
        n * ratios["val"]
    )

    parts = {
        "train": ids[:n_train],
        "val": ids[
            n_train:
            n_train + n_val
        ],
        "test": ids[
            n_train + n_val:
        ],
    }

    frames = [
        pd.DataFrame(
            {
                "image_id": values,
                "split": split_name,
            }
        )
        for split_name, values
        in parts.items()
    ]

    dataframe = pd.concat(
        frames,
        ignore_index=True,
    )

    return dataframe.sort_values(
        "image_id"
    ).reset_index(
        drop=True
    )


def assert_no_leakage(
    dataframe: pd.DataFrame,
) -> None:
    """Verify each image ID belongs to exactly one split."""

    required_columns = {
        "image_id",
        "split",
    }

    missing = (
        required_columns
        - set(dataframe.columns)
    )

    if missing:
        raise AssertionError(
            f"manifest missing columns: {missing}"
        )

    if dataframe["image_id"].duplicated().any():
        duplicates = dataframe.loc[
            dataframe["image_id"].duplicated(
                keep=False
            ),
            "image_id",
        ].unique()[:5]

        raise AssertionError(
            "image IDs appear more than once "
            f"(e.g. {list(duplicates)})"
        )

    bad_splits = set(
        dataframe["split"]
    ) - set(SPLITS)

    if bad_splits:
        raise AssertionError(
            f"unknown split names: {bad_splits}"
        )


def save_manifest(
    dataframe: pd.DataFrame,
    path: str | Path,
) -> None:
    """Save a split manifest as CSV."""

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_csv(
        path,
        index=False,
    )


def load_manifest(
    path: str | Path,
) -> pd.DataFrame:
    """Load and validate a split manifest."""

    dataframe = pd.read_csv(path)

    assert_no_leakage(
        dataframe
    )

    return dataframe
