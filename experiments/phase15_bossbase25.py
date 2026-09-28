from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from src.app_utils import (
    embed_message,
    extract_message,
    run_analysis,
)
from src.evaluation.metrics import (
    mse,
    psnr,
    ssim,
)


PAYLOADS = [
    0.05,
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
]

N_IMAGES = 25

WINDOW_HEIGHT = 3
WINDOW_WIDTH = 3
START_COMPONENT = 8
END_COMPONENT = 9

GAMMA = 5
SIGMA = 3.0
TAU = 2

STC_H = 5
SEED = 123


def payload_to_w(target_bpp: float) -> int:
    if target_bpp <= 0:
        raise ValueError("target_bpp must be greater than zero.")

    return max(
        1,
        round(1.0 / target_bpp),
    )


def make_secret(
    image_shape: tuple[int, int],
    actual_bpp: float,
    seed: int,
) -> bytes:
    """
    Create a deterministic payload that fits the actual
    width-trimmed STC capacity used by embed_message().
    """

    height, width = image_shape

    w = payload_to_w(
        actual_bpp
    )

    trimmed_width = (
        width // w
    ) * w

    capacity_bits = (
        height * trimmed_width
    ) // w

    payload_bits = (
        capacity_bits - 32
    )

    payload_bytes = (
        payload_bits // 8
    )

    if payload_bytes <= 0:
        raise ValueError(
            "Payload capacity is too small for the 32-bit header."
        )

    rng = np.random.default_rng(
        seed
    )

    return rng.integers(
        0,
        256,
        size=payload_bytes,
        dtype=np.uint8,
    ).tobytes()


def main() -> None:
    manifest_path = Path(
        "data/processed/split_dev.csv"
    )

    dataset_dir = Path(
        "data/raw/BOSSbase_1.01"
    )

    manifest = pd.read_csv(
        manifest_path
    )

    manifest = (
        manifest
        .sort_values("image_id")
        .head(N_IMAGES)
        .reset_index(drop=True)
    )

    print(
        f"Running BOSSbase experiment on "
        f"{len(manifest)} images..."
    )
    print(
        f"Payloads: {PAYLOADS}"
    )
    print()

    rows = []

    total_start = time.perf_counter()

    for image_number, manifest_row in enumerate(
        manifest.itertuples(index=False),
        start=1,
    ):
        image_id = int(
            manifest_row.image_id
        )

        image_path = (
            dataset_dir
            / f"{image_id}.pgm"
        )

        cover = np.array(
            Image.open(
                image_path
            ).convert("L"),
            dtype=np.uint8,
        )

        analysis_start = time.perf_counter()

        analysis = run_analysis(
            cover,
            window_height=WINDOW_HEIGHT,
            window_width=WINDOW_WIDTH,
            start_component=START_COMPONENT,
            end_component=END_COMPONENT,
            gamma=GAMMA,
            sigma=SIGMA,
            tau=TAU,
        )

        analysis_time = (
            time.perf_counter()
            - analysis_start
        )

        adaptive_rho = analysis["rho"]

        print(
            f"[{image_number:02d}/{len(manifest)}] "
            f"Image {image_id} "
            f"analysis={analysis_time:.2f}s"
        )

        for target_bpp in PAYLOADS:
            w = payload_to_w(
                target_bpp
            )

            actual_bpp = 1.0 / w

            secret = make_secret(
                cover.shape,
                actual_bpp,
                seed=(
                    image_id
                    * 1000
                    + int(target_bpp * 100)
                ),
            )

            for method, rho in (
                (
                    "adaptive",
                    adaptive_rho,
                ),
                (
                    "uniform",
                    np.ones_like(
                        adaptive_rho,
                        dtype=np.float64,
                    ),
                ),
            ):
                start = time.perf_counter()

                stego, embed_result, key = (
                    embed_message(
                        cover,
                        rho,
                        secret,
                        target_bpp=target_bpp,
                        h=STC_H,
                        seed=SEED,
                    )
                )

                embedding_time = (
                    time.perf_counter()
                    - start
                )

                extraction_start = (
                    time.perf_counter()
                )

                extracted = extract_message(
                    stego,
                    key,
                )

                extraction_time = (
                    time.perf_counter()
                    - extraction_start
                )

                extraction_success = (
                    extracted == secret
                )

                rows.append(
                    {
                        "image_id": image_id,
                        "method": method,
                        "target_bpp": target_bpp,
                        "actual_bpp": actual_bpp,
                        "stc_w": w,
                        "n_secret_bytes": len(secret),
                        "psnr": psnr(
                            cover,
                            stego,
                        ),
                        "ssim": ssim(
                            cover,
                            stego,
                        ),
                        "mse": mse(
                            cover,
                            stego,
                        ),
                        "changed_pixels": int(
                            np.count_nonzero(
                                cover != stego
                            )
                        ),
                        "embedding_time_s": (
                            embedding_time
                        ),
                        "extraction_time_s": (
                            extraction_time
                        ),
                        "extraction_success": (
                            extraction_success
                        ),
                        "stc_cost": float(
                            embed_result.total_cost
                        ),
                    }
                )

    elapsed_total = (
        time.perf_counter()
        - total_start
    )

    results = pd.DataFrame(rows)

    results_dir = Path(
        "results/metrics"
    )

    figures_dir = Path(
        "results/figures"
    )

    tables_dir = Path(
        "results/tables"
    )

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    figures_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    tables_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    per_image_path = (
        results_dir
        / "phase15_bossbase25_per_image.csv"
    )

    summary = (
        results
        .groupby(
            [
                "method",
                "target_bpp",
                "actual_bpp",
                "stc_w",
            ],
            as_index=False,
        )
        .agg(
            n_images=(
                "image_id",
                "count",
            ),
            psnr_mean=(
                "psnr",
                "mean",
            ),
            psnr_std=(
                "psnr",
                "std",
            ),
            ssim_mean=(
                "ssim",
                "mean",
            ),
            ssim_std=(
                "ssim",
                "std",
            ),
            mse_mean=(
                "mse",
                "mean",
            ),
            changed_pixels_mean=(
                "changed_pixels",
                "mean",
            ),
            embedding_time_mean_s=(
                "embedding_time_s",
                "mean",
            ),
            extraction_success_rate=(
                "extraction_success",
                "mean",
            ),
            stc_cost_mean=(
                "stc_cost",
                "mean",
            ),
        )
    )

    summary_path = (
        results_dir
        / "phase15_bossbase25_summary.csv"
    )

    results.to_csv(
        per_image_path,
        index=False,
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    # -------------------------
    # Plot 1: PSNR
    # -------------------------

    plt.figure(
        figsize=(8, 5)
    )

    for method in (
        "adaptive",
        "uniform",
    ):
        subset = summary[
            summary["method"] == method
        ]

        plt.plot(
            subset["actual_bpp"],
            subset["psnr_mean"],
            marker="o",
            label=method,
        )

    plt.xlabel(
        "Actual payload (bpp)"
    )
    plt.ylabel(
        "Mean PSNR (dB)"
    )
    plt.title(
        "BOSSbase 25-image PSNR comparison"
    )
    plt.grid(
        True,
        alpha=0.3,
    )
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        figures_dir
        / "phase15_bossbase25_psnr.png",
        dpi=150,
    )

    plt.close()

    # -------------------------
    # Plot 2: changed pixels
    # -------------------------

    plt.figure(
        figsize=(8, 5)
    )

    for method in (
        "adaptive",
        "uniform",
    ):
        subset = summary[
            summary["method"] == method
        ]

        plt.plot(
            subset["actual_bpp"],
            subset["changed_pixels_mean"],
            marker="o",
            label=method,
        )

    plt.xlabel(
        "Actual payload (bpp)"
    )
    plt.ylabel(
        "Mean changed pixels"
    )
    plt.title(
        "BOSSbase 25-image modification count"
    )
    plt.grid(
        True,
        alpha=0.3,
    )
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        figures_dir
        / "phase15_bossbase25_changed_pixels.png",
        dpi=150,
    )

    plt.close()

    # -------------------------
    # Plot 3: SSIM
    # -------------------------

    plt.figure(
        figsize=(8, 5)
    )

    for method in (
        "adaptive",
        "uniform",
    ):
        subset = summary[
            summary["method"] == method
        ]

        plt.plot(
            subset["actual_bpp"],
            subset["ssim_mean"],
            marker="o",
            label=method,
        )

    plt.xlabel(
        "Actual payload (bpp)"
    )
    plt.ylabel(
        "Mean SSIM"
    )
    plt.title(
        "BOSSbase 25-image SSIM comparison"
    )
    plt.grid(
        True,
        alpha=0.3,
    )
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        figures_dir
        / "phase15_bossbase25_ssim.png",
        dpi=150,
    )

    plt.close()

    markdown_path = (
        tables_dir
        / "phase15_bossbase25_summary.md"
    )

    markdown_path.write_text(
        summary.to_markdown(
            index=False
        ),
        encoding="utf-8",
    )

    print()
    print(
        "=== Phase 15 BOSSbase Summary ==="
    )

    print(
        summary.to_string(
            index=False
        )
    )

    print()
    print(
        f"Total runtime: "
        f"{elapsed_total:.2f} seconds"
    )

    print()
    print("Saved:")
    print(per_image_path)
    print(summary_path)
    print(
        figures_dir
        / "phase15_bossbase25_psnr.png"
    )
    print(
        figures_dir
        / "phase15_bossbase25_changed_pixels.png"
    )
    print(
        figures_dir
        / "phase15_bossbase25_ssim.png"
    )
    print(markdown_path)


if __name__ == "__main__":
    main()

