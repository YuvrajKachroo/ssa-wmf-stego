import numpy as np

from src.preprocessing.loader import (
    save_grayscale,
)
from src.preprocessing.synthetic import (
    make_synthetic_image,
)
from src.preprocessing.validate import (
    validate_array,
    validate_directory,
)


def test_validate_array_ok():
    assert (
        validate_array(
            make_synthetic_image(0)
        )
        == []
    )


def test_validate_array_bad_shape_and_dtype():
    assert any(
        "shape" in problem
        for problem in validate_array(
            np.zeros(
                (256, 256),
                dtype=np.uint8,
            )
        )
    )

    assert any(
        "dtype" in problem
        for problem in validate_array(
            np.zeros(
                (512, 512),
                dtype=np.float32,
            )
        )
    )

    assert any(
        "ndim" in problem
        for problem in validate_array(
            np.zeros(
                (512, 512, 3),
                dtype=np.uint8,
            )
        )
    )


def test_directory_clean(tmp_path):
    for image_id in (1, 2, 3):
        save_grayscale(
            make_synthetic_image(image_id),
            tmp_path
            / f"{image_id}.pgm",
        )

    dataframe = validate_directory(
        tmp_path
    )

    assert len(dataframe) == 3
    assert dataframe["ok"].all()


def test_directory_flags_wrong_size(tmp_path):
    save_grayscale(
        make_synthetic_image(1),
        tmp_path / "1.pgm",
    )

    save_grayscale(
        np.zeros(
            (100, 100),
            dtype=np.uint8,
        ),
        tmp_path / "2.pgm",
    )

    dataframe = (
        validate_directory(tmp_path)
        .set_index("image_id")
    )

    assert dataframe.loc[
        1,
        "ok",
    ]

    assert not dataframe.loc[
        2,
        "ok",
    ]

    assert (
        "shape"
        in dataframe.loc[
            2,
            "problems",
        ]
    )


def test_directory_flags_duplicates(tmp_path):
    image = make_synthetic_image(5)

    save_grayscale(
        image,
        tmp_path / "1.pgm",
    )

    save_grayscale(
        image,
        tmp_path / "2.pgm",
    )

    dataframe = (
        validate_directory(tmp_path)
        .set_index("image_id")
    )

    assert dataframe.loc[
        1,
        "ok",
    ]

    assert (
        "duplicate_pixels"
        in dataframe.loc[
            2,
            "problems",
        ]
    )


def test_directory_flags_id_out_of_range(tmp_path):
    save_grayscale(
        make_synthetic_image(1),
        tmp_path / "20000.pgm",
    )

    dataframe = validate_directory(
        tmp_path
    )

    assert not dataframe[
        "ok"
    ].iloc[0]

    assert (
        "outside"
        in dataframe[
            "problems"
        ].iloc[0]
    )
