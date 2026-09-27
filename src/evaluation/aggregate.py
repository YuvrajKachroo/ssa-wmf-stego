"""Phase 13: aggregation and detection summary."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.steganalysis.classifier import (
    build_dataset,
    train_and_evaluate_grouped,
)
from src.evaluation.runner import ImageResult


@dataclass(frozen=True)
class SummaryRow:
    """Summary metrics for one method/payload combination."""

    method: str
    target_bpp: float
    actual_bpp: float
    n_images: int

    psnr_mean: float
    psnr_std: float

    ssim_mean: float
    ssim_std: float

    mse_mean: float

    changed_pixels_mean: float

    embedding_time_mean_s: float

    extraction_success_rate: float

    file_size_mean_bytes: float

    detection_accuracy: float
    detection_pe: float


def image_results_to_df(
    results: list[ImageResult],
) -> pd.DataFrame:
    """Convert ImageResult objects to a DataFrame."""

    if not results:
        return pd.DataFrame()

    return pd.DataFrame(
        [
            result.__dict__
            for result in results
        ]
    )


def summarize(
    results: list[ImageResult],
    covers,
    stegos,
    method: str,
    target_bpp: float,
) -> SummaryRow:
    """Aggregate per-image results and run grouped detection."""

    if not results:
        raise ValueError(
            "results must not be empty."
        )

    if len(covers) != len(stegos):
        raise ValueError(
            "covers and stegos must have equal length."
        )

    if len(results) != len(covers):
        raise ValueError(
            "results and covers must have equal length."
        )

    df = image_results_to_df(
        results
    )

    X, y, groups = build_dataset(
        covers,
        stegos,
    )

    detection = (
        train_and_evaluate_grouped(
            X,
            y,
            groups,
            test_size=0.3,
            seed=0,
        )
    )

    return SummaryRow(
        method=method,
        target_bpp=target_bpp,
        actual_bpp=float(
            df["actual_bpp"].iloc[0]
        ),
        n_images=len(df),

        psnr_mean=float(
            df["psnr"].mean()
        ),
        psnr_std=float(
            df["psnr"].std(
                ddof=1
            )
        ),

        ssim_mean=float(
            df["ssim"].mean()
        ),
        ssim_std=float(
            df["ssim"].std(
                ddof=1
            )
        ),

        mse_mean=float(
            df["mse"].mean()
        ),

        changed_pixels_mean=float(
            df["n_changed"].mean()
        ),

        embedding_time_mean_s=float(
            df[
                "embedding_time_s"
            ].mean()
        ),

        extraction_success_rate=float(
            df[
                "extraction_success"
            ].mean()
        ),

        file_size_mean_bytes=float(
            df[
                "file_size_bytes"
            ].mean()
        ),

        detection_accuracy=float(
            detection.accuracy
        ),

        detection_pe=float(
            detection.detection_error
        ),
    )


def summaries_to_df(
    summaries: list[SummaryRow],
) -> pd.DataFrame:
    """Convert SummaryRow objects to a DataFrame."""

    if not summaries:
        return pd.DataFrame()

    return pd.DataFrame(
        [
            summary.__dict__
            for summary in summaries
        ]
    )
