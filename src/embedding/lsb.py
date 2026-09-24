from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def text_to_bits(message: str) -> str:
    """Convert UTF-8 text into a binary string."""
    data = message.encode("utf-8")
    return "".join(f"{byte:08b}" for byte in data)


def bits_to_text(bits: str) -> str:
    """Convert a binary string back into UTF-8 text."""
    if len(bits) % 8 != 0:
        raise ValueError("Number of bits must be divisible by 8.")

    data = bytes(int(bits[i : i + 8], 2) for i in range(0, len(bits), 8))

    return data.decode("utf-8")


def embed_lsb(
    image: np.ndarray,
    message: str,
) -> np.ndarray:
    """
    Embed a text message into the least significant bits of an image.

    Parameters
    ----------
    image:
        8-bit grayscale image.
    message:
        Text message to embed.

    Returns
    -------
    np.ndarray
        Stego image.
    """
    if image.dtype != np.uint8:
        raise ValueError("Image must have dtype uint8.")

    if image.ndim != 2:
        raise ValueError("Image must be grayscale.")

    message_bits = text_to_bits(message)

    # Store the message length in the first 32 bits.
    length_bits = f"{len(message_bits):032b}"
    payload = length_bits + message_bits

    flat = image.flatten()

    if len(payload) > len(flat):
        raise ValueError("Message is too large for this image.")

    stego = flat.copy()

    for i, bit in enumerate(payload):
        stego[i] = (stego[i] & 0xFE) | int(bit)

    return stego.reshape(image.shape)


def extract_lsb(image: np.ndarray) -> str:
    """Extract a UTF-8 text message from an LSB stego image."""

    if image.dtype != np.uint8:
        raise ValueError("Image must have dtype uint8.")

    if image.ndim != 2:
        raise ValueError("Image must be grayscale.")

    flat = image.flatten()

    # Read the 32-bit message length.
    length_bits = "".join(str(pixel & 1) for pixel in flat[:32])
    message_length = int(length_bits, 2)

    if message_length > len(flat) - 32:
        raise ValueError("Invalid or corrupted embedded message.")

    message_bits = "".join(str(pixel & 1) for pixel in flat[32 : 32 + message_length])

    return bits_to_text(message_bits)


def embed_lsb_file(
    cover_path: str | Path,
    stego_path: str | Path,
    message: str,
) -> None:
    """Load a cover image, embed a message, and save the stego image."""

    cover = cv2.imread(str(cover_path), cv2.IMREAD_GRAYSCALE)

    if cover is None:
        raise FileNotFoundError(f"Could not read cover image: {cover_path}")

    stego = embed_lsb(cover, message)

    stego_path = Path(stego_path)
    stego_path.parent.mkdir(parents=True, exist_ok=True)

    if not cv2.imwrite(str(stego_path), stego):
        raise IOError(f"Could not save stego image: {stego_path}")
