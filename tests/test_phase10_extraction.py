import cv2
import numpy as np
import pytest

from src.embedding.extraction import (
    StegoKey,
    embed_secret,
    extract_secret,
)
from src.embedding.message import decode_message


def _setup(
    seed_img=0,
    shape=(48, 48),
    h=5,
    w=2,
    seed_key=99,
):
    """Create a deterministic Phase 10 test setup."""

    rng = np.random.default_rng(seed_img)

    image = rng.integers(
        0,
        256,
        size=shape,
        dtype=np.uint8,
    )

    rng_cost = np.random.default_rng(1)

    n = image.size

    n_trim = (n // w) * w

    cover_flat = image.flatten()[:n_trim]

    rho_plus = rng_cost.uniform(
        0.1,
        10.0,
        n_trim,
    )

    rho_minus = rng_cost.uniform(
        0.1,
        10.0,
        n_trim,
    )

    key = StegoKey(
        h=h,
        w=w,
        seed=seed_key,
    )

    return (
        image,
        cover_flat,
        rho_plus,
        rho_minus,
        key,
        n_trim,
    )


def test_correct_key_round_trip():
    """Correct key must recover the exact original secret."""

    (
        image,
        cover_flat,
        rho_plus,
        rho_minus,
        key,
        n_trim,
    ) = _setup()

    secret = b"correct key extraction test"

    result = embed_secret(
        cover_flat,
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


def test_wrong_key_fails_or_returns_wrong_data():
    """Wrong key must not recover the original secret.

    A ValueError is the preferred outcome when the decoded header
    is obviously inconsistent. A wrong-key extraction can, in
    principle, produce plausible-looking bits, so a non-equal
    byte string is also accepted.
    """

    (
        image,
        cover_flat,
        rho_plus,
        rho_minus,
        key,
        n_trim,
    ) = _setup()

    secret = b"this should not be recoverable with the wrong key"

    result = embed_secret(
        cover_flat,
        rho_plus,
        rho_minus,
        secret,
        key,
    )

    wrong_key = StegoKey(
        h=key.h,
        w=key.w,
        seed=key.seed + 1,
    )

    try:
        recovered = extract_secret(
            result.stego,
            wrong_key,
        )

        assert recovered != secret

    except ValueError:
        # Header sanity check rejected the wrong key.
        pass


def test_disk_round_trip_survives_lossless_save(tmp_path):
    """A lossless disk save/reload must preserve the extractable message."""

    (
        image,
        cover_flat,
        rho_plus,
        rho_minus,
        key,
        n_trim,
    ) = _setup(shape=(64, 64))

    secret = b"disk round trip works"

    result = embed_secret(
        cover_flat,
        rho_plus,
        rho_minus,
        secret,
        key,
    )

    # Put the modified flat region back into the full image.
    stego_flat = image.flatten().copy()

    stego_flat[:n_trim] = result.stego

    stego_image = stego_flat.reshape(image.shape)

    path = tmp_path / "stego.pgm"

    written = cv2.imwrite(
        str(path),
        stego_image,
    )

    assert written

    reloaded = cv2.imread(
        str(path),
        cv2.IMREAD_GRAYSCALE,
    )

    assert reloaded is not None

    assert np.array_equal(
        reloaded,
        stego_image,
    )

    reloaded_flat = reloaded.flatten()[:n_trim]

    recovered = extract_secret(
        reloaded_flat,
        key,
    )

    assert recovered == secret


def test_tampering_after_embedding_breaks_or_corrupts_extraction():
    """A post-embedding pixel modification can corrupt extraction.

    The current scheme has no error-correction layer. Depending on
    the modified pixel's trellis window, extraction may either raise
    a validation error or return data different from the original.
    """

    (
        image,
        cover_flat,
        rho_plus,
        rho_minus,
        key,
        n_trim,
    ) = _setup(shape=(64, 64))

    secret = b"tampering should break this message reliably enough to notice"

    result = embed_secret(
        cover_flat,
        rho_plus,
        rho_minus,
        secret,
        key,
    )

    tampered = result.stego.copy()

    # Modify one pixel by exactly one gray-level.
    if tampered[0] < 255:
        tampered[0] += 1
    else:
        tampered[0] -= 1

    try:
        recovered = extract_secret(
            tampered,
            key,
        )

        assert recovered != secret

    except ValueError:
        # Header sanity check caught the corruption.
        pass


def test_stego_key_deterministic_H_hat():
    """The same key must always regenerate the same H_hat."""

    key = StegoKey(
        h=4,
        w=2,
        seed=7,
    )

    first = key.build_H_hat()
    second = key.build_H_hat()

    assert np.array_equal(
        first,
        second,
    )


def test_rejects_length_not_divisible_by_w():
    """Embedding/extraction require a length compatible with w."""

    key = StegoKey(
        h=4,
        w=3,
        seed=0,
    )

    with pytest.raises(ValueError):
        embed_secret(
            np.zeros(10, dtype=np.uint8),
            np.ones(10),
            np.ones(10),
            b"x",
            key,
        )

    with pytest.raises(ValueError):
        extract_secret(
            np.zeros(
                10,
                dtype=np.uint8,
            ),
            key,
        )


def test_decode_message_rejects_impossible_header():
    """Impossible header length must produce a clear ValueError."""

    garbage = (
        np.random.default_rng(0)
        .integers(
            0,
            2,
            40,
        )
        .astype(np.uint8)
    )

    # All 32 header bits set -> an impossibly large payload length.
    garbage[:32] = 1

    with pytest.raises(ValueError):
        decode_message(garbage)
