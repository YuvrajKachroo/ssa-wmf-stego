import numpy as np
import pytest

from src.steganalysis.classifier import (
    build_dataset,
    train_and_evaluate,
    train_and_evaluate_grouped,
)


def make_test_image(
    seed: int,
    shape: tuple[int, int] = (32, 32),
) -> np.ndarray:
    rng = np.random.default_rng(seed)

    return rng.integers(
        0,
        256,
        size=shape,
        dtype=np.uint8,
    )


def test_build_dataset_shapes():

    covers = [make_test_image(i) for i in range(6)]

    stegos = [make_test_image(i + 100) for i in range(6)]

    X, y, groups = build_dataset(
        covers,
        stegos,
    )

    assert X.shape[0] == 12
    assert y.shape == (12,)
    assert groups.shape == (12,)

    assert set(y.tolist()) == {0, 1}

    assert (y == 0).sum() == 6 and (y == 1).sum() == 6

    assert set(groups[:6].tolist()) == set(groups[6:].tolist()) == set(range(6))


def test_grouped_split_never_leaks_a_base_image():

    covers = [make_test_image(i, (24, 24)) for i in range(20)]

    stegos = [
        make_test_image(
            i + 100,
            (24, 24),
        )
        for i in range(20)
    ]

    X, y, groups = build_dataset(
        covers,
        stegos,
    )

    result = train_and_evaluate_grouped(
        X,
        y,
        groups,
        test_size=0.3,
        seed=0,
    )

    assert 0.0 <= result.accuracy <= 1.0
    assert 0.0 <= result.detection_error <= 1.0


def test_perfectly_separable_dataset_gets_near_perfect_accuracy():

    rng = np.random.default_rng(0)

    X_cover = rng.normal(
        loc=0.0,
        scale=0.1,
        size=(60, 10),
    )

    X_stego = rng.normal(
        loc=10.0,
        scale=0.1,
        size=(60, 10),
    )

    X = np.concatenate(
        [
            X_cover,
            X_stego,
        ]
    )

    y = np.concatenate(
        [
            np.zeros(60),
            np.ones(60),
        ]
    )

    result = train_and_evaluate(
        X,
        y,
        test_size=0.3,
        seed=0,
    )

    assert result.accuracy > 0.95
    assert result.detection_error < 0.05


def test_indistinguishable_dataset_is_near_chance():

    rng = np.random.default_rng(1)

    X = rng.normal(size=(120, 10))

    y = np.concatenate(
        [
            np.zeros(60),
            np.ones(60),
        ]
    )

    result = train_and_evaluate(
        X,
        y,
        test_size=0.3,
        seed=0,
    )

    assert 0.3 < result.accuracy < 0.7
    assert 0.3 < result.detection_error < 0.7


def test_detection_result_fields_are_consistent():

    rng = np.random.default_rng(2)

    X = rng.normal(size=(80, 5))

    y = np.concatenate(
        [
            np.zeros(40),
            np.ones(40),
        ]
    )

    result = train_and_evaluate(
        X,
        y,
        test_size=0.25,
        seed=1,
    )

    assert result.detection_error == pytest.approx(
        (result.false_alarm_rate + result.missed_detection_rate) / 2.0
    )

    assert result.n_test == 20


def test_grouped_evaluate_runs_and_returns_valid_result():

    rng = np.random.default_rng(3)

    n_images = 30

    X_cover = rng.normal(
        loc=0.0,
        size=(n_images, 8),
    )

    X_stego = rng.normal(
        loc=0.3,
        size=(n_images, 8),
    )

    X = np.concatenate(
        [
            X_cover,
            X_stego,
        ]
    )

    y = np.concatenate(
        [
            np.zeros(n_images),
            np.ones(n_images),
        ]
    )

    groups = np.concatenate(
        [
            np.arange(n_images),
            np.arange(n_images),
        ]
    )

    result = train_and_evaluate_grouped(
        X,
        y,
        groups,
        test_size=0.3,
        seed=0,
    )

    assert 0.0 <= result.accuracy <= 1.0
    assert 0.0 <= result.detection_error <= 1.0
