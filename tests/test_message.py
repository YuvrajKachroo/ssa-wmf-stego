import numpy as np
import pytest

from src.embedding.message import (
    LENGTH_HEADER_BITS,
    bytes_to_bits,
    bits_to_bytes,
)


def test_empty_message_round_trip():

    message = b""

    bits = bytes_to_bits(message)

    assert bits.dtype == np.uint8
    assert bits.ndim == 1
    assert len(bits) == LENGTH_HEADER_BITS

    recovered = bits_to_bytes(bits)

    assert recovered == message


def test_simple_message_round_trip():

    message = b"Hello, steganography!"

    bits = bytes_to_bits(message)

    recovered = bits_to_bytes(bits)

    assert recovered == message


def test_binary_data_round_trip():

    message = bytes(range(256))

    bits = bytes_to_bits(message)

    recovered = bits_to_bytes(bits)

    assert recovered == message


def test_utf8_bytes_round_trip():

    message = "Hello, नमस्ते, chess!".encode("utf-8")

    bits = bytes_to_bits(message)

    recovered = bits_to_bytes(bits)

    assert recovered == message


def test_length_header_is_32_bits():

    message = b"abc"

    bits = bytes_to_bits(message)

    assert len(bits) == 32 + 24


def test_header_declares_payload_length():

    message = b"abcde"

    bits = bytes_to_bits(message)

    header = bits[:32]

    decoded_length = int(
        "".join(str(int(bit)) for bit in header),
        2,
    )

    assert decoded_length == len(message)


def test_bits_are_binary():

    message = b"test"

    bits = bytes_to_bits(message)

    assert np.all((bits == 0) | (bits == 1))


def test_rejects_non_binary_bits():

    bits = np.zeros(
        32,
        dtype=np.uint8,
    )

    bits[10] = 2

    with pytest.raises(ValueError):
        bits_to_bytes(bits)


def test_rejects_insufficient_header():

    bits = np.zeros(
        31,
        dtype=np.uint8,
    )

    with pytest.raises(ValueError):
        bits_to_bytes(bits)


def test_rejects_non_byte_aligned_bits():

    bits = np.zeros(
        33,
        dtype=np.uint8,
    )

    with pytest.raises(ValueError):
        bits_to_bytes(bits)


def test_rejects_length_mismatch():

    message = b"hello"

    bits = bytes_to_bits(message)

    # Remove one byte from the encoded payload.
    truncated = bits[:-8]

    with pytest.raises(ValueError):
        bits_to_bytes(truncated)


def test_rejects_multidimensional_bits():

    bits = np.zeros(
        (4, 8),
        dtype=np.uint8,
    )

    with pytest.raises(ValueError):
        bits_to_bytes(bits)


def test_requires_bytes_input():

    with pytest.raises(TypeError):
        bytes_to_bits("hello")
