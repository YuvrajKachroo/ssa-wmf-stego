import numpy as np
import pytest
from PIL import Image

from src.app_utils import (
    embed_message,
    extract_message,
    normalized_display,
    run_analysis,
    symmetric_directional_costs,
    to_uint8_grayscale,
)
from src.embedding.extraction import StegoKey
from src.preprocessing.synthetic import make_synthetic_image


def test_to_uint8_grayscale_accepts_l_mode():
    image = Image.fromarray(
        make_synthetic_image(
            0,
            shape=(32, 32),
        ),
        mode="L",
    )

    array = to_uint8_grayscale(image)

    assert array.shape == (32, 32)
    assert array.dtype == np.uint8


def test_to_uint8_grayscale_rejects_rgb():
    image = Image.new(
        "RGB",
        (32, 32),
    )

    with pytest.raises(
        ValueError,
        match="grayscale",
    ):
        to_uint8_grayscale(image)


def test_run_analysis_shapes():
    cover = make_synthetic_image(
        1,
        shape=(32, 32),
    )

    analysis = run_analysis(
        cover
    )

    assert analysis["zeta"].shape == cover.shape
    assert analysis["filtered"].shape == cover.shape
    assert analysis["rho"].shape == cover.shape
    assert analysis["reconstructed_components"].shape[1:] == cover.shape

    assert np.isfinite(
        analysis["rho"]
    ).all()

    assert (
        analysis["rho"] >= 0
    ).all()


def test_directional_costs_are_independent_copies():
    rho = np.ones(
        (4, 4),
        dtype=np.float64,
    )

    plus, minus = (
        symmetric_directional_costs(
            rho
        )
    )

    assert np.array_equal(
        plus,
        rho,
    )

    assert np.array_equal(
        minus,
        rho,
    )

    plus[0, 0] = 99.0

    assert minus[0, 0] == 1.0
    assert rho[0, 0] == 1.0


def test_embed_extract_roundtrip():
    cover = make_synthetic_image(
        2,
        shape=(32, 32),
    )

    analysis = run_analysis(
        cover
    )

    secret = (
        b"Phase 14 test message"
    )

    stego, result, key = embed_message(
        cover,
        analysis["rho"],
        secret,
        target_bpp=0.5,
        h=4,
        seed=123,
    )

    recovered = extract_message(
        stego,
        key,
    )

    assert recovered == secret
    assert stego.shape == cover.shape
    assert stego.dtype == np.uint8
    assert result.total_cost >= 0.0


def test_normalized_display():
    image = np.array(
        [[0.0, 1.0], [2.0, 3.0]],
        dtype=np.float64,
    )

    displayed = normalized_display(
        image
    )

    assert displayed.shape == image.shape
    assert displayed.dtype == np.uint8
    assert displayed.min() == 0
    assert displayed.max() == 255


def test_wrong_key_fails_or_changes_message():
    cover = make_synthetic_image(
        3,
        shape=(32, 32),
    )

    analysis = run_analysis(
        cover
    )

    secret = b"Wrong key test"

    stego, _, key = embed_message(
        cover,
        analysis["rho"],
        secret,
        target_bpp=0.5,
        h=4,
        seed=123,
    )

    wrong_key = StegoKey(
        h=4,
        w=key.w,
        seed=999,
    )

    try:
        recovered = extract_message(
            stego,
            wrong_key,
        )
    except ValueError:
        return

    assert recovered != secret
