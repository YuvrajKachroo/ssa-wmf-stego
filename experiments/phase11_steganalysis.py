"""Phase 11: classical steganalysis experiment.

Compares:

    1. Adaptive cost:
       real SSA -> suitability -> WMF -> cost map

    2. Uniform cost:
       rho = 1 everywhere

Both schemes use the SAME:
    - cover images
    - STC implementation
    - H_hat
    - payload
    - message bits

Steganalysis:
    - SPAM-style adjacent-pixel difference features
    - Random Forest classifier
    - grouped train/test split

Important:
This is a small-scale sanity-check experiment, not a security
proof. Full-scale evaluation belongs to Phase 13.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from src.cost_function.cost import compute_cost_map
from src.cost_function.suitability import compute_suitability_map
from src.embedding.stc import make_random_submatrix
from src.embedding.ternary import embed_ternary_lsb
from src.ssa.ssa2d import (
    create_elementary_components,
    create_trajectory_matrix,
    decompose_trajectory_matrix,
    reconstruct_component,
)
from src.steganalysis.classifier import (
    build_dataset,
    train_and_evaluate_grouped,
)
from src.wmf.weighted_median import weighted_median_filter


def build_adaptive_cost_map(
    image: np.ndarray,
) -> np.ndarray:
    """Run the actual Phase 3-8 pipeline to produce rho."""

    # ---------------------------------------------------------
    # Phase 3: 2D-SSA
    # ---------------------------------------------------------
    trajectory = create_trajectory_matrix(
        image.astype(np.float64),
        window_height=3,
        window_width=3,
    )

    (
        eigenvalues,
        eigenvectors,
        singular_values,
    ) = decompose_trajectory_matrix(trajectory)

    components = create_elementary_components(
        trajectory,
        eigenvectors,
        singular_values,
    )

    # ---------------------------------------------------------
    # Phase 5: reconstruct the 9 components
    # ---------------------------------------------------------
    reconstructed_components = np.stack(
        [
            reconstruct_component(
                component,
                image.shape,
                window_shape=(3, 3),
            )
            for component in components
        ],
        axis=0,
    )

    # ---------------------------------------------------------
    # Phase 7: suitability
    #
    # Paper-derived implementation used in Phases 7-8:
    #     zeta = |G8 + G9|
    # ---------------------------------------------------------
    suitability = compute_suitability_map(
        reconstructed_components,
        start_component=8,
        end_component=9,
    )

    # ---------------------------------------------------------
    # Phase 6: WMF
    # ---------------------------------------------------------
    filtered_suitability = weighted_median_filter(
        suitability,
        gamma=5,
        sigma=3.0,
        tau=2,
    )

    # ---------------------------------------------------------
    # Phase 8: cost
    # ---------------------------------------------------------
    rho = compute_cost_map(filtered_suitability)

    return rho


def generate_stego_pair(
    image: np.ndarray,
    adaptive_cost: np.ndarray,
    H_hat: np.ndarray,
    message_bits: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate adaptive-cost and uniform-cost stego images."""

    # ---------------------------------------------------------
    # Adaptive embedding
    # ---------------------------------------------------------
    adaptive_stego, _ = embed_ternary_lsb(
        image,
        adaptive_cost,
        message_bits,
        H_hat,
    )

    # ---------------------------------------------------------
    # Uniform-cost embedding
    # ---------------------------------------------------------
    uniform_cost = np.ones(
        image.shape,
        dtype=np.float64,
    )

    uniform_stego, _ = embed_ternary_lsb(
        image,
        uniform_cost,
        message_bits,
        H_hat,
    )

    return (
        adaptive_stego,
        uniform_stego,
    )


def main() -> None:

    parser = argparse.ArgumentParser(
        description=("Phase 11 adaptive-vs-uniform classical steganalysis experiment.")
    )

    parser.add_argument(
        "--n-images",
        type=int,
        default=60,
        help="Number of synthetic cover images.",
    )

    parser.add_argument(
        "--height",
        type=int,
        default=64,
        help="Image height.",
    )

    parser.add_argument(
        "--width",
        type=int,
        default=64,
        help="Image width.",
    )

    parser.add_argument(
        "--h",
        type=int,
        default=5,
        help="STC constraint height.",
    )

    parser.add_argument(
        "--w",
        type=int,
        default=2,
        help=("STC submatrix width. Current payload rate is approximately 1/w bpp."),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Seed used to construct H_hat.",
    )

    parser.add_argument(
        "--message-seed",
        type=int,
        default=999,
        help="Seed for random message bits.",
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.3,
        help="Fraction of base images assigned to test.",
    )

    args = parser.parse_args()

    if args.n_images < 4:
        raise ValueError("n-images must be at least 4.")

    if args.height <= 0 or args.width <= 0:
        raise ValueError("Image dimensions must be positive.")

    if args.h <= 0 or args.w <= 0:
        raise ValueError("STC h and w must be positive.")

    # ---------------------------------------------------------
    # Configuration
    # ---------------------------------------------------------
    shape = (
        args.height,
        args.width,
    )

    H_hat = make_random_submatrix(
        args.h,
        args.w,
        seed=args.seed,
    )

    message_rng = np.random.default_rng(args.message_seed)

    covers: list[np.ndarray] = []
    adaptive_stegos: list[np.ndarray] = []
    uniform_stegos: list[np.ndarray] = []

    start_time = time.perf_counter()

    # ---------------------------------------------------------
    # Generate cover / adaptive / uniform triples
    # ---------------------------------------------------------
    for image_index in range(args.n_images):
        # Deterministic synthetic cover.
        rng = np.random.default_rng(image_index)

        image = rng.integers(
            0,
            256,
            size=shape,
            dtype=np.uint8,
        )

        # -----------------------------------------------------
        # Build the real Phase 3-8 adaptive cost map.
        # -----------------------------------------------------
        adaptive_cost = build_adaptive_cost_map(image)

        # -----------------------------------------------------
        # Current STC implementation:
        #
        #     n = m * w
        #
        # Therefore m = number of available STC message bits.
        # -----------------------------------------------------
        n_pixels = image.size

        n_trim = (n_pixels // args.w) * args.w

        message_length = n_trim // args.w

        # Current ternary wrapper operates on all image pixels,
        # so trimming is only necessary when the image size isn't
        # divisible by w.
        if n_trim != n_pixels:
            image_for_embedding = image.flatten()[:n_trim].reshape(
                args.height,
                n_trim // args.height,
            )

            # This branch is deliberately rejected because
            # changing the 2-D geometry would make the comparison
            # confusing.
            raise ValueError(
                "Image size must be divisible by STC w for this "
                "experiment. Choose dimensions such that "
                "height*width is divisible by w."
            )

        # Random near-capacity message.
        message_bits = message_rng.integers(
            0,
            2,
            size=message_length,
            dtype=np.uint8,
        )

        adaptive_stego, uniform_stego = generate_stego_pair(
            image,
            adaptive_cost,
            H_hat,
            message_bits,
        )

        covers.append(image)

        adaptive_stegos.append(adaptive_stego)

        uniform_stegos.append(uniform_stego)

        if (image_index + 1) % 10 == 0:
            print(f"Generated {image_index + 1}/{args.n_images} images...")

    elapsed = time.perf_counter() - start_time

    # ---------------------------------------------------------
    # Experiment metadata
    # ---------------------------------------------------------
    actual_bpp = message_length / n_pixels

    print()
    print(f"Generated {args.n_images} cover/adaptive/uniform triples in {elapsed:.2f}s")

    print(f"Image shape       : {shape}")

    print(f"STC H_hat shape   : {H_hat.shape}")

    print(f"Payload bits      : {message_length}")

    print(f"Payload rate      : {actual_bpp:.4f} bpp")

    # ---------------------------------------------------------
    # Feature dataset: adaptive
    # ---------------------------------------------------------
    print()
    print("Extracting adaptive-cost features...")

    X_adaptive, y_adaptive, groups_adaptive = build_dataset(
        covers,
        adaptive_stegos,
    )

    # ---------------------------------------------------------
    # Feature dataset: uniform
    # ---------------------------------------------------------
    print("Extracting uniform-cost features...")

    X_uniform, y_uniform, groups_uniform = build_dataset(
        covers,
        uniform_stegos,
    )

    print(
        "Feature dimension:",
        X_adaptive.shape[1],
    )

    # ---------------------------------------------------------
    # Grouped evaluation
    #
    # IMPORTANT:
    # A cover and its corresponding stego must remain
    # together in either train or test.
    # ---------------------------------------------------------
    print()
    print("Training adaptive-cost detector...")

    adaptive_result = train_and_evaluate_grouped(
        X_adaptive,
        y_adaptive,
        groups_adaptive,
        test_size=args.test_size,
        seed=0,
    )

    print("Training uniform-cost detector...")

    uniform_result = train_and_evaluate_grouped(
        X_uniform,
        y_uniform,
        groups_uniform,
        test_size=args.test_size,
        seed=0,
    )

    # ---------------------------------------------------------
    # Results
    # ---------------------------------------------------------
    print()
    print("==================================================")
    print("PHASE 11: CLASSICAL STEGANALYSIS")
    print("==================================================")

    print()
    print("Adaptive cost (real rho):")

    print(f"  Accuracy            = {adaptive_result.accuracy:.3f}")

    print(f"  Detection error Pe  = {adaptive_result.detection_error:.3f}")

    print(f"  False alarm rate    = {adaptive_result.false_alarm_rate:.3f}")

    print(f"  Missed detection    = {adaptive_result.missed_detection_rate:.3f}")

    print(f"  Test samples        = {adaptive_result.n_test}")

    print()
    print("Uniform cost (rho=1):")

    print(f"  Accuracy            = {uniform_result.accuracy:.3f}")

    print(f"  Detection error Pe  = {uniform_result.detection_error:.3f}")

    print(f"  False alarm rate    = {uniform_result.false_alarm_rate:.3f}")

    print(f"  Missed detection    = {uniform_result.missed_detection_rate:.3f}")

    print(f"  Test samples        = {uniform_result.n_test}")

    print()

    if adaptive_result.detection_error >= uniform_result.detection_error:
        direction = "Adaptive cost has higher Pe than uniform cost in this run."
    else:
        direction = "Adaptive cost has lower Pe than uniform cost in this run."

    print("Relative comparison:")

    print(f"  {direction}")

    print()
    print("IMPORTANT:")

    print("This is a small-scale sanity-check experiment.")

    print(
        "SPAM-style features are substantially weaker "
        "than full SRM/maxSRMd2-style steganalysis."
    )

    print("These results should NOT be presented as a security proof.")

    print("Phase 13 should provide the larger-scale experimental evaluation.")

    print("==================================================")


if __name__ == "__main__":
    main()
