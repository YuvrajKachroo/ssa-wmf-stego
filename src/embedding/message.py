from __future__ import annotations

import numpy as np


LENGTH_HEADER_BITS = 32
HEADER_BITS = LENGTH_HEADER_BITS


def bytes_to_bits(data: bytes) -> np.ndarray:
    """Encode bytes as a 32-bit big-endian length header + payload bits."""

    payload = bytes(data)

    header = np.array(
        [
            (len(payload) >> shift) & 1
            for shift in range(LENGTH_HEADER_BITS - 1, -1, -1)
        ],
        dtype=np.uint8,
    )

    if len(payload) == 0:
        return header

    payload_bits = np.unpackbits(
        np.frombuffer(payload, dtype=np.uint8)
    ).astype(np.uint8)

    return np.concatenate(
        [
            header,
            payload_bits,
        ]
    )


def bits_to_bytes(bits: np.ndarray) -> bytes:
    """Decode a complete encoded message."""

    bits = np.asarray(bits)

    if bits.ndim != 1:
        raise ValueError("bits must be one-dimensional")

    if not np.all((bits == 0) | (bits == 1)):
        raise ValueError("bits must contain only 0/1 values")

    bits = bits.astype(np.uint8, copy=False)

    if bits.size < LENGTH_HEADER_BITS:
        raise ValueError("not enough bits for length header")

    payload_length = 0

    for bit in bits[:LENGTH_HEADER_BITS]:
        payload_length = (payload_length << 1) | int(bit)

    expected_length = (
        LENGTH_HEADER_BITS
        + payload_length * 8
    )

    if bits.size != expected_length:
        raise ValueError(
            "bit length does not match encoded payload length"
        )

    if payload_length == 0:
        return b""

    payload_bits = bits[LENGTH_HEADER_BITS:]

    return np.packbits(payload_bits).tobytes()


def encode_message(
    data: bytes,
    capacity_bits: int,
    fill_bits: np.ndarray | None = None,
) -> np.ndarray:
    """Create a full STC syndrome vector.

    The message occupies the beginning of the syndrome.
    Unused capacity is preserved from fill_bits when supplied.
    """

    encoded_message = bytes_to_bits(data)

    if capacity_bits < encoded_message.size:
        raise ValueError(
            f"message requires {encoded_message.size} bits, "
            f"but capacity is {capacity_bits}"
        )

    if fill_bits is None:
        encoded = np.zeros(
            capacity_bits,
            dtype=np.uint8,
        )

    else:
        fill_bits = np.asarray(fill_bits)

        if fill_bits.ndim != 1:
            raise ValueError("fill_bits must be one-dimensional")

        if fill_bits.size != capacity_bits:
            raise ValueError(
                f"fill_bits must contain exactly "
                f"capacity_bits={capacity_bits} bits"
            )

        if not np.all(
            (fill_bits == 0) | (fill_bits == 1)
        ):
            raise ValueError(
                "fill_bits must contain only 0/1 values"
            )

        encoded = fill_bits.astype(
            np.uint8,
            copy=True,
        )

    encoded[: encoded_message.size] = encoded_message

    return encoded


def decode_message(bits: np.ndarray) -> bytes:
    """Decode a message from a syndrome vector.

    Extra trailing bits are allowed because the STC capacity
    is generally larger than the actual secret message.
    """

    bits = np.asarray(bits)

    if bits.ndim != 1:
        raise ValueError("bits must be one-dimensional")

    if not np.all((bits == 0) | (bits == 1)):
        raise ValueError("bits must contain only 0/1 values")

    bits = bits.astype(np.uint8, copy=False)

    if bits.size < LENGTH_HEADER_BITS:
        raise ValueError("not enough bits for length header")

    payload_length = 0

    for bit in bits[:LENGTH_HEADER_BITS]:
        payload_length = (payload_length << 1) | int(bit)

    end = (
        LENGTH_HEADER_BITS
        + payload_length * 8
    )

    if end > bits.size:
        raise ValueError(
            f"declared payload requires {end} bits, "
            f"but only {bits.size} bits are available"
        )

    encoded_message = bits[:end]

    return bits_to_bytes(encoded_message)
