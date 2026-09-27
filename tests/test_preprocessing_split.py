import pytest

from src.preprocessing.split import (
    assert_no_leakage,
    load_manifest,
    make_splits,
    save_manifest,
    select_subset,
)


RATIOS = {
    "train": 0.4,
    "val": 0.1,
    "test": 0.5,
}


def test_sizes():
    dataframe = make_splits(
        range(1, 101),
        RATIOS,
        seed=0,
    )

    assert (
        dataframe[
            "split"
        ].value_counts().to_dict()
        == {
            "test": 50,
            "train": 40,
            "val": 10,
        }
    )


def test_complete_and_disjoint():
    dataframe = make_splits(
        range(1, 101),
        RATIOS,
        seed=0,
    )

    assert sorted(
        dataframe["image_id"]
    ) == list(
        range(1, 101)
    )

    assert_no_leakage(
        dataframe
    )


def test_deterministic_and_seed_sensitive():
    first = make_splits(
        range(1, 101),
        RATIOS,
        seed=1,
    )

    second = make_splits(
        range(1, 101),
        RATIOS,
        seed=1,
    )

    third = make_splits(
        range(1, 101),
        RATIOS,
        seed=2,
    )

    assert first.equals(
        second
    )

    assert not first.equals(
        third
    )


def test_bad_ratios():
    with pytest.raises(ValueError):
        make_splits(
            range(10),
            {
                "train": 0.5,
                "val": 0.5,
                "test": 0.5,
            },
            seed=0,
        )

    with pytest.raises(ValueError):
        make_splits(
            range(10),
            {
                "train": 0.5,
                "test": 0.5,
            },
            seed=0,
        )


def test_negative_ratio():
    with pytest.raises(ValueError):
        make_splits(
            range(10),
            {
                "train": -0.1,
                "val": 0.1,
                "test": 1.0,
            },
            seed=0,
        )


def test_leakage_detected():
    dataframe = make_splits(
        range(1, 21),
        RATIOS,
        seed=0,
    )

    dataframe.loc[
        0,
        "image_id",
    ] = dataframe.loc[
        1,
        "image_id",
    ]

    with pytest.raises(
        AssertionError
    ):
        assert_no_leakage(
            dataframe
        )


def test_subset():
    subset = select_subset(
        range(1, 1001),
        50,
        seed=3,
    )

    assert (
        len(subset)
        == len(set(subset))
        == 50
    )

    assert subset == sorted(
        subset
    )

    assert subset == (
        select_subset(
            range(1, 1001),
            50,
            seed=3,
        )
    )

    with pytest.raises(
        ValueError
    ):
        select_subset(
            range(5),
            10,
            seed=0,
        )


def test_manifest_roundtrip(tmp_path):
    dataframe = make_splits(
        range(1, 21),
        RATIOS,
        seed=0,
    )

    path = (
        tmp_path
        / "manifest.csv"
    )

    save_manifest(
        dataframe,
        path,
    )

    loaded = load_manifest(
        path
    )

    assert loaded.equals(
        dataframe
    )
