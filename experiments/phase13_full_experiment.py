"""Phase 13: full payload-sweep experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.evaluation.aggregate import (
    image_results_to_df,
    summarize,
    summaries_to_df,
)
from src.evaluation.runner import (
    precompute,
    run_one,
)


PAYLOADS = [
    0.05,
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
]

METHODS = [
    "adaptive",
    "uniform",
]


def make_synthetic_cover(
    seed: int,
    height: int,
    width: int,
) -> np.ndarray:
    """Create a deterministic synthetic grayscale cover."""

    rng = np.random.default_rng(seed)

    y, x = np.mgrid[0:height, 0:width]

    gradient = (
        65.0
        + 75.0 * x / max(width - 1, 1)
        + 55.0 * y / max(height - 1, 1)
    )

    sinusoid = (
        18.0
        * np.sin(x / 7.0)
        * np.cos(y / 9.0)
    )

    noise = rng.normal(
        loc=0.0,
        scale=10.0,
        size=(height, width),
    )

    image = (
        gradient
        + sinusoid
        + noise
    )

    return np.clip(
        image,
        0,
        255,
    ).astype(np.uint8)


def make_output_directories(
    root: Path,
) -> None:
    (root / "metrics").mkdir(
        parents=True,
        exist_ok=True,
    )

    (root / "figures").mkdir(
        parents=True,
        exist_ok=True,
    )

    (root / "tables").mkdir(
        parents=True,
        exist_ok=True,
    )

    (root / "phase13_png").mkdir(
        parents=True,
        exist_ok=True,
    )


def write_comparison_table(
    summary_df: pd.DataFrame,
    path: Path,
) -> None:
    columns = [
        "method",
        "target_bpp",
        "actual_bpp",
        "n_images",
        "psnr_mean",
        "psnr_std",
        "ssim_mean",
        "ssim_std",
        "mse_mean",
        "changed_pixels_mean",
        "embedding_time_mean_s",
        "extraction_success_rate",
        "detection_accuracy",
        "detection_pe",
    ]

    display_df = summary_df[
        columns
    ].copy()

    numeric_columns = [
        "target_bpp",
        "actual_bpp",
        "psnr_mean",
        "psnr_std",
        "ssim_mean",
        "ssim_std",
        "mse_mean",
        "changed_pixels_mean",
        "embedding_time_mean_s",
        "extraction_success_rate",
        "detection_accuracy",
        "detection_pe",
    ]

    for column in numeric_columns:
        display_df[column] = display_df[column].map(
            lambda value: f"{value:.6f}"
        )

    display_df["n_images"] = (
        display_df["n_images"]
        .astype(int)
    )

    markdown = display_df.to_markdown(
        index=False
    )

    path.write_text(
        markdown,
        encoding="utf-8",
    )


def plot_payload_sweep(
    summary_df: pd.DataFrame,
    path: Path,
) -> None:
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18, 5),
    )

    for method in METHODS:
        subset = summary_df[
            summary_df["method"] == method
        ].sort_values("actual_bpp")

        axes[0].plot(
            subset["actual_bpp"],
            subset["psnr_mean"],
            marker="o",
            label=method,
        )

        axes[1].plot(
            subset["actual_bpp"],
            subset["ssim_mean"],
            marker="o",
            label=method,
        )

        axes[2].plot(
            subset["actual_bpp"],
            subset["detection_pe"],
            marker="o",
            label=method,
        )

    axes[0].set_title("Mean PSNR vs Payload")
    axes[0].set_xlabel("Actual bpp")
    axes[0].set_ylabel("PSNR (dB)")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].set_title("Mean SSIM vs Payload")
    axes[1].set_xlabel("Actual bpp")
    axes[1].set_ylabel("SSIM")
    axes[1].grid(True)
    axes[1].legend()

    axes[2].set_title("Detection Error vs Payload")
    axes[2].set_xlabel("Actual bpp")
    axes[2].set_ylabel("Pe")
    axes[2].set_ylim(0.0, 0.5)
    axes[2].grid(True)
    axes[2].legend()

    fig.suptitle(
        "Phase 13: Adaptive vs Uniform Cost"
    )

    fig.tight_layout()

    fig.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)


def run_experiment(
    n_images: int,
    height: int,
    width: int,
    output_root: Path,
    seed: int,
    save_png: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    make_output_directories(
        output_root
    )

    rng = np.random.default_rng(seed)

    print(
        f"Generating {n_images} synthetic cover images..."
    )

    covers = [
        make_synthetic_cover(
            seed=seed + index,
            height=height,
            width=width,
        )
        for index in range(n_images)
    ]

    print(
        "Precomputing SSA -> suitability -> WMF -> cost..."
    )

    precomputed = []

    for index, cover in enumerate(covers):
        print(
            f"  Precompute {index + 1}/{n_images}"
        )

        precomputed.append(
            precompute(
                cover,
                window_height=3,
                window_width=3,
                start_component=8,
                end_component=9,
                gamma=5,
                sigma=3.0,
                tau=2,
            )
        )

    all_results = []
    summary_rows = []

    for target_bpp in PAYLOADS:
        print(
            f"\n=== Target payload: {target_bpp:.2f} bpp ==="
        )

        for method in METHODS:
            print(
                f"Method: {method}"
            )

            method_results = []
            method_stegos = []

            for image_index, pre in enumerate(
                precomputed
            ):
                print(
                    f"  Image "
                    f"{image_index + 1}/{n_images}",
                    end="\r",
                )

                # Same message seed across adaptive
                # and uniform for fair comparison.
                message_seed = (
                    seed
                    + 100000
                    + image_index * 1000
                    + int(round(target_bpp * 1000))
                )

                message_rng = np.random.default_rng(
                    message_seed
                )

                png_path = None

                if save_png:
                    actual_estimate = (
                        1.0
                        / max(
                            1,
                            round(1.0 / target_bpp),
                        )
                    )

                    png_path = (
                        output_root
                        / "phase13_png"
                        / method
                        / f"bpp_{actual_estimate:.3f}"
                        / f"image_{image_index:03d}.png"
                    )

                result, stego = run_one(
                    pre=pre,
                    method=method,
                    target_bpp=target_bpp,
                    h=5,
                    image_index=image_index,
                    rng=message_rng,
                    out_png_path=png_path,
                )

                method_results.append(
                    result
                )

                method_stegos.append(
                    stego
                )

            print()

            all_results.extend(
                method_results
            )

            summary = summarize(
                results=method_results,
                covers=covers,
                stegos=method_stegos,
                method=method,
                target_bpp=target_bpp,
            )

            summary_rows.append(
                summary
            )

            print(
                f"  actual_bpp={summary.actual_bpp:.6f} | "
                f"PSNR={summary.psnr_mean:.4f} dB | "
                f"SSIM={summary.ssim_mean:.6f} | "
                f"changed={summary.changed_pixels_mean:.2f} | "
                f"Pe={summary.detection_pe:.4f}"
            )

    per_image_df = image_results_to_df(
        all_results
    )

    summary_df = summaries_to_df(
        summary_rows
    )

    per_image_path = (
        output_root
        / "metrics"
        / "phase13_per_image.csv"
    )

    summary_path = (
        output_root
        / "metrics"
        / "phase13_summary.csv"
    )

    per_image_df.to_csv(
        per_image_path,
        index=False,
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    figure_path = (
        output_root
        / "figures"
        / "phase13_payload_sweep.png"
    )

    plot_payload_sweep(
        summary_df,
        figure_path,
    )

    table_path = (
        output_root
        / "tables"
        / "phase13_comparison.md"
    )

    write_comparison_table(
        summary_df,
        table_path,
    )

    return (
        per_image_df,
        summary_df,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the Phase 13 adaptive vs "
            "uniform payload sweep."
        )
    )

    parser.add_argument(
        "--n-images",
        type=int,
        default=25,
    )

    parser.add_argument(
        "--height",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--width",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results"),
    )

    parser.add_argument(
        "--no-png",
        action="store_true",
        help="Do not save individual stego PNG files.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.n_images <= 0:
        raise ValueError(
            "--n-images must be positive."
        )

    if args.height <= 0 or args.width <= 0:
        raise ValueError(
            "Image dimensions must be positive."
        )

    _, summary_df = run_experiment(
        n_images=args.n_images,
        height=args.height,
        width=args.width,
        output_root=args.output_root,
        seed=args.seed,
        save_png=not args.no_png,
    )

    print(
        "\n=== Phase 13 Summary ==="
    )

    print(
        summary_df.to_string(
            index=False
        )
    )

    print(
        "\nSaved:"
    )

    print(
        args.output_root
        / "metrics"
        / "phase13_per_image.csv"
    )

    print(
        args.output_root
        / "metrics"
        / "phase13_summary.csv"
    )

    print(
        args.output_root
        / "figures"
        / "phase13_payload_sweep.png"
    )

    print(
        args.output_root
        / "tables"
        / "phase13_comparison.md"
    )


if __name__ == "__main__":
    main()
