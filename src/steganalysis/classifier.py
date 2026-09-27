"""Phase 11: classical steganalysis classifier and metrics.

The evaluation uses:
    - Random Forest
    - grouped train/test splitting for cover/stego pairs

Detection error:

    Pe = (P_false_alarm + P_missed_detection) / 2

A value near 0.5 corresponds to chance-level detection.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from .features import extract_features


def build_dataset(
    cover_images,
    stego_images,
    T: int = 3,
):
    """Build feature matrix for paired cover/stego images.

    Labels:
        0 = cover
        1 = stego

    groups:
        base image identity

    The group IDs are essential because each cover/stego pair
    originates from the same underlying image.
    """

    if len(cover_images) != len(stego_images):
        raise ValueError("cover_images and stego_images must have equal length.")

    X_cover = np.array(
        [
            extract_features(
                image,
                T=T,
            )
            for image in cover_images
        ]
    )

    X_stego = np.array(
        [
            extract_features(
                image,
                T=T,
            )
            for image in stego_images
        ]
    )

    X = np.concatenate(
        [
            X_cover,
            X_stego,
        ],
        axis=0,
    )

    y = np.concatenate(
        [
            np.zeros(
                len(X_cover),
                dtype=np.uint8,
            ),
            np.ones(
                len(X_stego),
                dtype=np.uint8,
            ),
        ]
    )

    groups = np.concatenate(
        [
            np.arange(len(X_cover)),
            np.arange(len(X_stego)),
        ]
    )

    return X, y, groups


@dataclass(frozen=True)
class DetectionResult:
    """Detection metrics."""

    accuracy: float
    detection_error: float
    false_alarm_rate: float
    missed_detection_rate: float
    n_test: int


def train_and_evaluate(
    X,
    y,
    test_size: float = 0.3,
    seed: int = 0,
) -> DetectionResult:
    """Evaluate independent samples with a stratified split.

    Use train_and_evaluate_grouped() when X/y contain
    cover/stego pairs from the same base image.
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )

    return _fit_and_score(
        X_train,
        X_test,
        y_train,
        y_test,
        seed,
    )


def train_and_evaluate_grouped(
    X,
    y,
    groups,
    test_size: float = 0.3,
    seed: int = 0,
) -> DetectionResult:
    """Evaluate while keeping each cover/stego pair together."""

    groups = np.asarray(groups)

    if len(groups) != len(X):
        raise ValueError("groups must have one entry per sample.")

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=seed,
    )

    train_idx, test_idx = next(
        splitter.split(
            X,
            y,
            groups,
        )
    )

    train_groups = set(groups[train_idx])

    test_groups = set(groups[test_idx])

    if train_groups.intersection(test_groups):
        raise RuntimeError("Group leakage detected.")

    return _fit_and_score(
        X[train_idx],
        X[test_idx],
        y[train_idx],
        y[test_idx],
        seed,
    )


def _fit_and_score(
    X_train,
    X_test,
    y_train,
    y_test,
    seed: int,
) -> DetectionResult:
    """Train Random Forest and compute detection metrics."""

    classifier = RandomForestClassifier(
        n_estimators=200,
        random_state=seed,
    )

    classifier.fit(
        X_train,
        y_train,
    )

    y_pred = classifier.predict(X_test)

    accuracy = float(np.mean(y_pred == y_test))

    is_cover = y_test == 0

    is_stego = y_test == 1

    false_alarm_rate = (
        float(np.mean(y_pred[is_cover] == 1)) if np.any(is_cover) else 0.0
    )

    missed_detection_rate = (
        float(np.mean(y_pred[is_stego] == 0)) if np.any(is_stego) else 0.0
    )

    detection_error = (false_alarm_rate + missed_detection_rate) / 2.0

    return DetectionResult(
        accuracy=accuracy,
        detection_error=detection_error,
        false_alarm_rate=false_alarm_rate,
        missed_detection_rate=missed_detection_rate,
        n_test=len(y_test),
    )
