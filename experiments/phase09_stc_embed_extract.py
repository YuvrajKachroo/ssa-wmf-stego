"""Phase 9: complete SSA-WMF-STC embedding/extraction demonstration."""

from pathlib import Path

import cv2
import numpy as np

from src.cost_function.cost import compute_cost_map
from src.cost_function.suitability import compute_suitability_map
from src.embedding.message import bytes_to_bits, bits_to_bytes
from src.embedding.stc import make_random_submatrix
from src.embedding.ternary import (
    embed_ternary_lsb,
    extract_ternary_lsb,
)
from src.ssa.ssa2d import (
    create_elementary_components,
    create_trajectory_matrix,
    decompose_trajectory_matrix,
    reconstruct_component,
)
from src.wmf.weighted_median import weighted_median_filter


OUTPUT_DIR = Path("outputs") / "phase09"


def calculate_psnr(
    original: np.ndarray,
    modified: np.ndarray,
) -> float:
    """Calculate PSNR between two uint8 grayscale images."""

    original = original.astype(np.float64)
    modified = modified.astype(np.float64)

    mse = np.mean((original - modified) ** 2)

    if mse == 0:
        return float("inf")

    return float(10.0 * np.log10((255.0**2) / mse))


def main() -> None:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # 1. Create deterministic synthetic cover
    # ---------------------------------------------------------
    rng = np.random.default_rng(42)

    cover = rng.integers(
        0,
        256,
        size=(32, 32),
        dtype=np.uint8,
    )

    print("Cover shape:", cover.shape)

    cv2.imwrite(
        str(OUTPUT_DIR / "cover.png"),
        cover,
    )

    # ---------------------------------------------------------
    # 2. 2D-SSA
    # ---------------------------------------------------------
    trajectory = create_trajectory_matrix(
        cover.astype(np.float64),
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
                cover.shape,
                window_shape=(3, 3),
            )
            for component in components
        ],
        axis=0,
    )

    print(
        "SSA components:",
        reconstructed_components.shape[0],
    )

    # ---------------------------------------------------------
    # 3. Suitability map: |G8 + G9|
    # ---------------------------------------------------------
    suitability = compute_suitability_map(
        reconstructed_components,
        start_component=8,
        end_component=9,
    )

    print(
        "Suitability range:",
        float(suitability.min()),
        "to",
        float(suitability.max()),
    )

    # ---------------------------------------------------------
    # 4. WMF
    # ---------------------------------------------------------
    wmf_output = weighted_median_filter(
        suitability,
        gamma=5,
        sigma=3.0,
        tau=2,
    )

    print(
        "WMF range:",
        float(wmf_output.min()),
        "to",
        float(wmf_output.max()),
    )

    # ---------------------------------------------------------
    # 5. Cost map
    # ---------------------------------------------------------
    cost_map = compute_cost_map(wmf_output)

    print(
        "Cost range:",
        float(cost_map.min()),
        "to",
        float(cost_map.max()),
    )

    # ---------------------------------------------------------
    # 6. Message
    #
    # 32 x 32 = 1024 pixels
    # w = 2
    # 1024 / 2 = 512 message bits
    #
    # 512 bits = 64 bytes
    # 4-byte header + 60-byte payload = 64 bytes
    # ---------------------------------------------------------
    message = b"SSA-WMF Phase 9 STC integration test payload. 12345678901234"

    assert len(message) == 60

    message_bits = bytes_to_bits(message)

    assert len(message_bits) == 512

    print(
        "Payload bytes:",
        len(message),
    )

    print(
        "Encoded message bits:",
        len(message_bits),
    )

    # ---------------------------------------------------------
    # 7. STC submatrix
    # ---------------------------------------------------------
    H_hat = make_random_submatrix(
        h=5,
        w=2,
        seed=42,
    )

    print(
        "H_hat shape:",
        H_hat.shape,
    )

    # ---------------------------------------------------------
    # 8. Embed
    # ---------------------------------------------------------
    stego, stc_result = embed_ternary_lsb(
        cover,
        cost_map,
        message_bits,
        H_hat,
    )

    cv2.imwrite(
        str(OUTPUT_DIR / "stego.png"),
        stego,
    )

    # ---------------------------------------------------------
    # 9. Measure image changes
    # ---------------------------------------------------------
    difference = np.abs(cover.astype(np.int16) - stego.astype(np.int16))

    changed_pixels = int(np.count_nonzero(cover != stego))

    change_ratio = changed_pixels / cover.size

    psnr = calculate_psnr(
        cover,
        stego,
    )

    print(
        "Changed pixels:",
        changed_pixels,
    )

    print(
        "Change ratio:",
        change_ratio,
    )

    print(
        "PSNR:",
        psnr,
        "dB",
    )

    print(
        "STC total cost:",
        stc_result.total_cost,
    )

    # ---------------------------------------------------------
    # 10. Extract
    # ---------------------------------------------------------
    extracted_bits = extract_ternary_lsb(
        stego,
        H_hat,
        message_length=len(message_bits),
    )

    extracted_message = bits_to_bytes(extracted_bits)

    # ---------------------------------------------------------
    # 11. Exact verification
    # ---------------------------------------------------------
    assert extracted_message == message

    assert np.all(difference <= 1)

    print()
    print("====================================")
    print("PHASE 9 END-TO-END RESULT")
    print("====================================")
    print("Embedding : SUCCESS")
    print("Extraction: EXACT")
    print("Message   :", extracted_message)
    print("PSNR      :", psnr, "dB")
    print("Changed   :", changed_pixels, "pixels")
    print("====================================")

    print()
    print(
        "Cover saved:",
        OUTPUT_DIR / "cover.png",
    )

    print(
        "Stego saved:",
        OUTPUT_DIR / "stego.png",
    )


if __name__ == "__main__":
    main()
