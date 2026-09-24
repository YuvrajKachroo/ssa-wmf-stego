import numpy as np

from src.utils.image_utils import load_grayscale_image, save_grayscale_image


def test_image_save_and_load(tmp_path):
    image = np.zeros((64, 64), dtype=np.uint8)

    image_path = tmp_path / "test.png"

    save_grayscale_image(image, image_path)

    loaded = load_grayscale_image(image_path)

    assert loaded.shape == (64, 64)
    assert loaded.dtype == np.uint8
    assert np.array_equal(image, loaded)
