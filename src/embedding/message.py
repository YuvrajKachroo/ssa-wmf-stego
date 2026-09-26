"""Message byte/bit codec for the Phase 9 STC pipeline.

Format:

    32-bit unsigned big-endian payload length in bytes
    followed by the payload bytes.

The resulting stream is converted to a NumPy uint8 bit array
containing only 0 and 1.
"""

from __future__ import annotations

import numpy as np


LENGTH_HEADER_BITS = 32
LENGTH_HEADER_BYTES = 4
MAX_PAYLOAD_BYTES = (2**32) - 1


def bytes_to_bits(message: bytes) -> np.ndarray:
    """Encode bytes as [32-bit length header | payload] bits.

    Parameters
    ----------
    message:
        Arbitrary bytes object.

    Returns
    -------
    np.ndarray
        One-dimensional uint8 array containing only 0 and 1.
    """
    if not isinstance(message, bytes):
        raise TypeError("message must be a bytes object.")

    payload_length = len(message)

    if payload_length > MAX_PAYLOAD_BYTES:
        raise ValueError("message is too large for the 32-bit length header.")

    header = payload_length.to_bytes(
        LENGTH_HEADER_BYTES,
        byteorder="big",
        signed=False,
    )

    encoded = header + message

    return np.unpackbits(
        np.frombuffer(
            encoded,
            dtype=np.uint8,
        )
    ).astype(
        np.uint8,
        copy=False,
    )


def bits_to_bytes(bits: np.ndarray) -> bytes:
    """Decode [32-bit length header | payload] bits back to bytes.

    The supplied bit array must contain exactly the number of bits
    declared by its 32-bit length header.
    """
    bits = np.asarray(bits)

    if bits.ndim != 1:
        raise ValueError("bits must be a one-dimensional array.")

    if bits.size < LENGTH_HEADER_BITS:
        raise ValueError("bits must contain at least the 32-bit length header.")

    if not np.all((bits == 0) | (bits == 1)):
        raise ValueError("bits must contain only 0 and 1.")

    if bits.size % 8 != 0:
        raise ValueError("number of bits must be divisible by 8.")

    bits = bits.astype(
        np.uint8,
        copy=False,
    )

    header_bytes = np.packbits(bits[:LENGTH_HEADER_BITS]).tobytes()

    payload_length = int.from_bytes(
        header_bytes,
        byteorder="big",
        signed=False,
    )

    expected_bits = LENGTH_HEADER_BITS + payload_length * 8

    if bits.size != expected_bits:
        raise ValueError(
            "bit array length does not match the payload length declared in the header."
        )

    payload_bits = bits[LENGTH_HEADER_BITS:]

    if payload_length == 0:
        return b""

    return np.packbits(payload_bits).tobytes()
