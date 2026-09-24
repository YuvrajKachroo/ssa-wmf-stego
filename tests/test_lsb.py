import numpy as np
import pytest

from src.embedding.lsb import (
    bits_to_text,
    embed_lsb,
    extract_lsb,
    text_to_bits,
)


def test_text_to_bits_and_back():
    message = "Hello, steganography!"

    bits = text_to_bits(message)

    assert len(bits) == len(message.encode("utf-8")) * 8
    assert bits_to_text(bits) == message


def test_lsb_embed_and_extract():
    image = np.zeros((64, 64), dtype=np.uint8)

    message = "Hello, LSB!"

    stego = embed_lsb(image, message)

    extracted = extract_lsb(stego)

    assert extracted == message


def test_message_too_large():
    image = np.zeros((4, 4), dtype=np.uint8)

    with pytest.raises(ValueError):
        embed_lsb(image, "This message is too large.")


def test_only_lsb_changes():
    rng = np.random.default_rng(42)

    image = rng.integers(
        0,
        256,
        size=(64, 64),
        dtype=np.uint8,
    )

    message = "Test message"

    stego = embed_lsb(image, message)

    difference = np.abs(image.astype(np.int16) - stego.astype(np.int16))

    assert np.all(difference <= 1)
