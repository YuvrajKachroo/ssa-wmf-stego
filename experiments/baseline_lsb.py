from pathlib import Path

import cv2
import numpy as np

from src.embedding.lsb import embed_lsb, extract_lsb
from src.evaluation.metrics import (
    changed_pixel_count,
    changed_pixel_percentage,
    mse,
    psnr,
)


def main() -> None:
    # Create a reproducible 512 x 512 grayscale cover image.
    rng = np.random.default_rng(42)

    cover = rng.integers(
        0,
        256,
        size=(512, 512),
        dtype=np.uint8,
    )

    message = "SSA-WMF steganography baseline test."

    # Embed the message.
    stego = embed_lsb(cover, message)

    # Extract the message.
    extracted = extract_lsb(stego)

    # Calculate image-quality metrics.
    image_mse = mse(cover, stego)
    image_psnr = psnr(cover, stego)
    changed_pixels = changed_pixel_count(cover, stego)
    changed_percentage = changed_pixel_percentage(cover, stego)

    # Save the images.
    cover_path = Path("data/cover/test_cover.png")
    stego_path = Path("data/stego/test_stego.png")

    cover_path.parent.mkdir(parents=True, exist_ok=True)
    stego_path.parent.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(cover_path), cover)
    cv2.imwrite(str(stego_path), stego)

    print("=== LSB Baseline Experiment ===")
    print(f"Message: {message}")
    print(f"Extracted: {extracted}")
    print(f"Extraction correct: {message == extracted}")
    print()
    print(f"Image shape: {cover.shape}")
    print(f"MSE: {image_mse:.6f}")
    print(f"PSNR: {image_psnr:.4f} dB")
    print(f"Changed pixels: {changed_pixels}")
    print(f"Changed pixels: {changed_percentage:.4f}%")
    print()
    print(f"Cover saved to: {cover_path}")
    print(f"Stego saved to: {stego_path}")


if __name__ == "__main__":
    main()
