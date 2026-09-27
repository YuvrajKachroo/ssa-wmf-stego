"""Phase 12: CNN-based steganalysis experiment.

Compares:
    - SimpleBaselineCNN
    - XuNetStyle

The experiment uses the actual Phase 3-8 adaptive-cost pipeline
to generate cover/stego pairs.

The split is performed by base image identity so a cover and its
corresponding stego never appear in different splits.

Important:
Small synthetic datasets are expected to overfit CNNs. Therefore
the results are a pipeline/data-scale demonstration, not a
security claim.
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
from src.steganalysis.cnn_pipeline import (
    build_datasets,
    train_and_evaluate_cnn,
)
from src.steganalysis.xunet import (
    SimpleBaselineCNN,
    XuNetStyle,
)
from src.wmf.weighted_median import weighted_median_filter


def build_adaptive_cost_map(
    image: np.ndarray,
) -> np.ndarray:
    """Run the actual SSA -> suitability -> WMF -> cost pipeline."""

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

    suitability = compute_suitability_map(
        reconstructed_components,
        start_component=8,
        end_component=9,
    )

    filtered = weighted_median_filter(
        suitability,
        gamma=5,
        sigma=3.0,
        tau=2,
    )

    return compute_cost_map(filtered)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--n-images",
        type=int,
        default=60,
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
        "--h",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--w",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=123,
    )

    parser.add_argument(
        "--message-seed",
        type=int,
        default=999,
    )

    args = parser.parse_args()

    if args.n_images < 4:
        raise ValueError("n-images must be at least 4.")

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
    stegos: list[np.ndarray] = []

    start = time.perf_counter()

    for image_index in range(args.n_images):
        rng = np.random.default_rng(image_index)

        image = rng.integers(
            0,
            256,
            size=shape,
            dtype=np.uint8,
        )

        cost_map = build_adaptive_cost_map(image)

        n_pixels = image.size

        if n_pixels % args.w != 0:
            raise ValueError("Image size must be divisible by STC w.")

        message_length = n_pixels // args.w

        # Near-full-capacity random payload.
        message_bits = message_rng.integers(
            0,
            2,
            size=message_length,
            dtype=np.uint8,
        )

        stego, _ = embed_ternary_lsb(
            image,
            cost_map,
            message_bits,
            H_hat,
        )

        covers.append(image)

        stegos.append(stego)

        if (image_index + 1) % 10 == 0:
            print(f"Generated {image_index + 1}/{args.n_images} cover/stego pairs...")

    elapsed = time.perf_counter() - start

    train_ds, test_ds = build_datasets(
        covers,
        stegos,
        test_size=0.3,
        seed=0,
    )

    print()
    print(f"Generated {args.n_images} cover/stego pairs in {elapsed:.2f}s")

    print(f"Image shape        : {shape}")

    print(f"Train samples      : {len(train_ds)}")

    print(f"Test samples       : {len(test_ds)}")

    print(f"Payload rate       : {1.0 / args.w:.4f} bpp")

    models = [
        (
            "SimpleBaselineCNN",
            SimpleBaselineCNN(),
        ),
        (
            "XuNetStyle",
            XuNetStyle(),
        ),
    ]

    for name, model in models:
        print()
        print(f"========== {name} ==========")

        start = time.perf_counter()

        result = train_and_evaluate_cnn(
            model,
            train_ds,
            test_ds,
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=0,
        )

        elapsed_model = time.perf_counter() - start

        first_loss = result.train_losses[0]

        final_loss = result.train_losses[-1]

        print(f"Training time      : {elapsed_model:.2f}s")

        print(f"Initial train loss : {first_loss:.4f}")

        print(f"Final train loss   : {final_loss:.4f}")

        print(f"Test accuracy      : {result.test_accuracy:.3f}")

        print(f"Detection error Pe : {result.test_detection_error:.3f}")

        if final_loss < 0.05 and result.test_accuracy < 0.65:
            print(
                "Interpretation     : "
                "overfitting signature "
                "(near-zero training loss, "
                "chance-level test accuracy)."
            )

        elif abs(result.test_accuracy - 0.5) < 0.1:
            print("Interpretation     : test performance is near chance.")

        else:
            print(
                "Interpretation     : "
                "non-chance result observed; "
                "requires larger-scale validation."
            )

    print()
    print("==================================================")
    print("PHASE 12 COMPLETE")
    print("==================================================")
    print("The experiment is a small-scale CNN pipeline/data-scale demonstration.")
    print("Do not interpret these results as a security proof.")
    print("A real CNN evaluation should use large-scale real-image data.")
    print("==================================================")


if __name__ == "__main__":
    main()
