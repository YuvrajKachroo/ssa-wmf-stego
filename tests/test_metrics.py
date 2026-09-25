import numpy as np

from src.evaluation.metrics import (
    changed_pixel_count,
    changed_pixel_percentage,
    mse,
    psnr,
)


def test_identical_images():
    image = np.zeros((10, 10), dtype=np.uint8)

    assert mse(image, image) == 0.0
    assert psnr(image, image) == float("inf")
    assert changed_pixel_count(image, image) == 0
    assert changed_pixel_percentage(image, image) == 0.0


def test_changed_pixels():
    original = np.zeros((10, 10), dtype=np.uint8)
    modified = original.copy()

    modified[0, 0] = 1
    modified[1, 1] = 1

    assert changed_pixel_count(original, modified) == 2
    assert changed_pixel_percentage(original, modified) == 2.0


def test_mse_and_psnr():
    original = np.zeros((10, 10), dtype=np.uint8)
    modified = np.ones((10, 10), dtype=np.uint8)

    assert mse(original, modified) == 1.0
    assert np.isclose(psnr(original, modified), 48.1308036086791)
