import numpy as np
import pytest
from PIL import Image

from src.preprocessing.loader import (
    image_id_from_path,
    list_images,
    load_grayscale,
    save_grayscale,
)
from src.preprocessing.synthetic import (
    make_synthetic_image,
)


def test_roundtrip_pgm(tmp_path):
    image = make_synthetic_image(0)

    save_grayscale(
        image,
        tmp_path / "1.pgm",
    )

    loaded = load_grayscale(
        tmp_path / "1.pgm"
    )

    assert loaded.dtype == np.uint8
    assert loaded.shape == (512, 512)
    assert np.array_equal(
        image,
        loaded,
    )


def test_pixel_range(tmp_path):
    save_grayscale(
        make_synthetic_image(1),
        tmp_path / "1.pgm",
    )

    image = load_grayscale(
        tmp_path / "1.pgm"
    )

    assert image.min() >= 0
    assert image.max() <= 255


def test_rejects_rgb(tmp_path):
    Image.new(
        "RGB",
        (512, 512),
    ).save(
        tmp_path / "1.png"
    )

    with pytest.raises(
        ValueError,
        match="grayscale",
    ):
        load_grayscale(
            tmp_path / "1.png"
        )


def test_save_rejects_wrong_dtype(tmp_path):
    with pytest.raises(ValueError):
        save_grayscale(
            np.zeros(
                (8, 8),
                dtype=np.float32,
            ),
            tmp_path / "x.pgm",
        )


def test_list_images_numeric_order(tmp_path):
    for image_id in (10, 2, 1):
        save_grayscale(
            np.zeros(
                (4, 4),
                dtype=np.uint8,
            ),
            tmp_path
            / f"{image_id}.pgm",
        )

    assert [
        path.name
        for path in list_images(tmp_path)
    ] == [
        "1.pgm",
        "2.pgm",
        "10.pgm",
    ]


def test_image_id_from_path():
    assert (
        image_id_from_path("data/7.pgm")
        == 7
    )

    with pytest.raises(ValueError):
        image_id_from_path(
            "data/cat.pgm"
        )
