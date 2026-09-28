import numpy as np

from src.embedding.extraction import (
    StegoKey,
    embed_secret,
    extract_secret,
)


def test_short_secret_roundtrip_does_not_modify_most_pixels():
    rng = np.random.default_rng(2026)

    cover = rng.integers(
        1,
        255,
        size=(
            32,
            32,
        ),
        dtype=np.uint8,
    )

    rho_plus = np.ones(
        cover.size,
        dtype=np.float64,
    )

    rho_minus = np.ones(
        cover.size,
        dtype=np.float64,
    )

    key = StegoKey(
        h=5,
        w=2,
        seed=123,
    )

    secret = b"Short UI test"

    result = embed_secret(
        cover.flatten(),
        rho_plus,
        rho_minus,
        secret,
        key,
    )

    recovered = extract_secret(
        result.stego,
        key,
    )

    assert recovered == secret

    changed_pixels = int(
        np.count_nonzero(
            cover.flatten()
            != result.stego
        )
    )

    # The short message must not force nearly the
    # entire cover to change.
    assert changed_pixels < (
        cover.size // 2
    )


def test_encode_message_preserves_unused_syndrome():
    from src.embedding.message import (
        encode_message,
    )

    capacity = 64

    fill = np.ones(
        capacity,
        dtype=np.uint8,
    )

    encoded = encode_message(
        b"A",
        capacity_bits=capacity,
        fill_bits=fill,
    )

    # 32-bit header + 8 payload bits = 40.
    # The remaining 24 bits should remain unchanged.
    assert np.all(
        encoded[40:] == 1
    )


def test_encode_message_rejects_wrong_fill_length():
    from src.embedding.message import (
        encode_message,
    )

    try:
        encode_message(
            b"A",
            capacity_bits=64,
            fill_bits=np.zeros(
                32,
                dtype=np.uint8,
            ),
        )
    except ValueError as exc:
        assert "capacity_bits" in str(exc)
    else:
        raise AssertionError(
            "Expected ValueError for invalid fill_bits length."
        )
