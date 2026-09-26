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


def test_complete_phase9_pipeline():
    """Test the complete Phase 3-9 SSA-WMF-STC pipeline."""

    rng = np.random.default_rng(123)

    # ---------------------------------------------------------
    # 1. Create deterministic grayscale cover image
    # ---------------------------------------------------------
    image = rng.integers(
        0,
        256,
        size=(32, 32),
        dtype=np.uint8,
    )

    assert image.shape == (32, 32)
    assert image.dtype == np.uint8

    # ---------------------------------------------------------
    # 2. 2D-SSA decomposition
    # ---------------------------------------------------------
    trajectory = create_trajectory_matrix(
        image.astype(np.float64),
        window_height=3,
        window_width=3,
    )

    eigenvalues, eigenvectors, singular_values = decompose_trajectory_matrix(trajectory)

    components = create_elementary_components(
        trajectory,
        eigenvectors,
        singular_values,
    )

    # Reconstruct all 9 SSA components.
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

    assert reconstructed_components.shape == (9, 32, 32)
    assert np.isfinite(reconstructed_components).all()

    # ---------------------------------------------------------
    # 3. Paper-defined suitability map
    #
    # ζ = |G8 + G9|
    #
    # Component numbering in the paper is 1-based.
    # ---------------------------------------------------------
    suitability = compute_suitability_map(
        reconstructed_components,
        start_component=8,
        end_component=9,
    )

    assert suitability.shape == image.shape
    assert suitability.dtype == np.float64
    assert np.all(suitability >= 0)
    assert np.isfinite(suitability).all()

    # ---------------------------------------------------------
    # 4. WMF
    #
    # Paper parameters:
    # gamma = 5
    # sigma = 3
    # tau = 2
    # ---------------------------------------------------------
    wmf_output = weighted_median_filter(
        suitability,
        gamma=5,
        sigma=3.0,
        tau=2,
    )

    assert wmf_output.shape == image.shape
    assert wmf_output.dtype == np.float64
    assert np.all(wmf_output >= 0)
    assert np.isfinite(wmf_output).all()

    # ---------------------------------------------------------
    # 5. Cost map
    #
    # rho = 1 / (WMF(zeta) + epsilon)
    # ---------------------------------------------------------
    cost_map = compute_cost_map(wmf_output)

    assert cost_map.shape == image.shape
    assert cost_map.dtype == np.float64
    assert np.all(cost_map > 0)
    assert np.isfinite(cost_map).all()

    # ---------------------------------------------------------
    # 6. Exact-capacity message
    #
    # Image:
    #     32 x 32 = 1024 pixels
    #
    # STC:
    #     w = 2
    #
    # Therefore:
    #     message bits = 1024 / 2 = 512 bits
    #     total bytes represented by 512 bits = 64 bytes
    #
    # Message format:
    #     4-byte length header
    #     60-byte payload
    #
    # Total:
    #     4 + 60 = 64 bytes
    #     64 x 8 = 512 bits
    # ---------------------------------------------------------
    message = b"SSA-WMF Phase 9 STC integration test payload. 12345678901234"

    assert len(message) == 60

    message_bits = bytes_to_bits(message)

    assert message_bits.dtype == np.uint8
    assert len(message_bits) == 512

    # ---------------------------------------------------------
    # 7. STC submatrix
    #
    # h = 5 -> 32 trellis states
    # w = 2 -> 0.5 bit/pixel rate for this implementation
    # ---------------------------------------------------------
    H_hat = make_random_submatrix(
        h=5,
        w=2,
        seed=42,
    )

    assert H_hat.shape == (5, 2)

    # ---------------------------------------------------------
    # 8. Embed message
    # ---------------------------------------------------------
    stego_image, stc_result = embed_ternary_lsb(
        image,
        cost_map,
        message_bits,
        H_hat,
    )

    assert stego_image.shape == image.shape
    assert stego_image.dtype == np.uint8
    assert np.isfinite(stc_result.total_cost)
    assert stc_result.total_cost >= 0

    # ---------------------------------------------------------
    # 9. Verify pixel modifications
    # ---------------------------------------------------------
    difference = np.abs(image.astype(np.int16) - stego_image.astype(np.int16))

    # Pixel-level realization is restricted to +/-1.
    assert np.all(difference <= 1)

    # At least one pixel should normally change for a random message.
    changed_pixels = np.count_nonzero(image != stego_image)

    assert changed_pixels > 0

    # ---------------------------------------------------------
    # 10. Extract message bits
    # ---------------------------------------------------------
    extracted_bits = extract_ternary_lsb(
        stego_image,
        H_hat,
        message_length=len(message_bits),
    )

    assert extracted_bits.dtype == np.uint8
    assert len(extracted_bits) == len(message_bits)

    # ---------------------------------------------------------
    # 11. Decode exact original message
    # ---------------------------------------------------------
    extracted_message = bits_to_bytes(extracted_bits)

    assert extracted_message == message
