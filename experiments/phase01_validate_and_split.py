"""Phase 1 dataset validation and deterministic split experiment."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )

from src.preprocessing.loader import (
    save_grayscale,
)
from src.preprocessing.split import (
    assert_no_leakage,
    make_splits,
    save_manifest,
    select_subset,
)
from src.preprocessing.synthetic import (
    make_synthetic_image,
)
from src.preprocessing.validate import (
    DEFAULT_EXPECTED_SHAPE,
    DEFAULT_ID_RANGE,
    validate_directory,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a grayscale image dataset, "
            "select a deterministic subset, and "
            "create train/val/test manifests."
        )
    )

    parser.add_argument(
        "--directory",
        type=Path,
        default=Path(
            "data/raw/BOSSbase_1.01"
        ),
        help="Directory containing dataset images.",
    )

    parser.add_argument(
        "--synthetic",
        type=int,
        default=0,
        help=(
            "Generate N synthetic PGM images "
            "for development/testing."
        ),
    )

    parser.add_argument(
        "--n",
        type=int,
        default=200,
        help=(
            "Number of valid image IDs to select "
            "for the development subset."
        ),
    )

    parser.add_argument(
        "--subset-seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.4,
    )

    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path(
            "data/processed"
        ),
    )

    args = parser.parse_args()

    raw_directory = args.directory

    if args.synthetic > 0:
        raw_directory = Path(
            "data/raw/synthetic"
        )

        raw_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        for image_id in range(
            1,
            args.synthetic + 1,
        ):
            save_grayscale(
                make_synthetic_image(
                    seed=image_id
                ),
                raw_directory
                / f"{image_id}.pgm",
            )

        print(
            f"[synthetic] wrote "
            f"{args.synthetic} images to "
            f"{raw_directory}"
        )
        print(
            "[synthetic] TEST DATA ONLY"
        )

    if not raw_directory.exists():
        raise FileNotFoundError(
            f"Dataset directory does not exist: "
            f"{raw_directory}"
        )

    report = validate_directory(
        raw_directory,
        expected_shape=(
            DEFAULT_EXPECTED_SHAPE
        ),
        id_range=(
            DEFAULT_ID_RANGE
        ),
    )

    output_directory = (
        args.output_directory
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    validation_path = (
        output_directory
        / "validation_report.csv"
    )

    report.to_csv(
        validation_path,
        index=False,
    )

    valid_count = int(
        report["ok"].sum()
    )

    invalid_count = (
        len(report)
        - valid_count
    )

    print(
        f"Checked {len(report)} files: "
        f"{valid_count} valid, "
        f"{invalid_count} invalid"
    )

    if invalid_count:
        print(
            report.loc[
                ~report["ok"],
                [
                    "image_id",
                    "problems",
                ],
            ]
            .head(10)
            .to_string(
                index=False
            )
        )

    good_ids = (
        report.loc[
            report["ok"],
            "image_id",
        ]
        .astype(int)
        .tolist()
    )

    if not good_ids:
        raise RuntimeError(
            "No valid images were found."
        )

    subset_size = min(
        args.n,
        len(good_ids),
    )

    subset = select_subset(
        good_ids,
        subset_size,
        seed=args.subset_seed,
    )

    ratios = {
        "train": args.train_ratio,
        "val": args.val_ratio,
        "test": args.test_ratio,
    }

    splits = make_splits(
        subset,
        ratios=ratios,
        seed=args.split_seed,
    )

    assert_no_leakage(
        splits
    )

    manifest_path = (
        output_directory
        / "split_dev.csv"
    )

    save_manifest(
        splits,
        manifest_path,
    )

    counts = (
        splits[
            "split"
        ]
        .value_counts()
        .to_dict()
    )

    print(
        f"Dev subset: {subset_size} images"
    )

    print(
        f"Split counts: {counts}"
    )

    print(
        f"Validation report: "
        f"{validation_path}"
    )

    print(
        f"Split manifest: "
        f"{manifest_path}"
    )


if __name__ == "__main__":
    main()
