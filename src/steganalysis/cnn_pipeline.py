"""Phase 12 CNN training/evaluation pipeline.

The train/test split is performed by BASE IMAGE identity.

A cover image and its corresponding stego image must always
remain in the same split to prevent the leakage problem found
during Phase 11.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class StegoImageDataset(Dataset):
    """Dataset containing cover/stego grayscale images."""

    def __init__(
        self,
        images: list[np.ndarray],
        labels: list[int],
    ) -> None:

        if len(images) != len(labels):
            raise ValueError("images and labels must have equal length.")

        self.images = images
        self.labels = labels

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(
        self,
        index: int,
    ):
        image = np.asarray(
            self.images[index],
        )

        if image.ndim != 2:
            raise ValueError("Each image must be a 2D grayscale array.")

        if image.dtype != np.uint8:
            raise ValueError("Each image must have dtype uint8.")

        tensor = torch.from_numpy(image.astype(np.float32) / 255.0).unsqueeze(0)

        label = int(self.labels[index])

        return tensor, label


def group_split_indices(
    n_images: int,
    test_size: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Split base-image IDs into train/test groups."""

    if n_images < 2:
        raise ValueError("n_images must be at least 2.")

    if not (0.0 < test_size < 1.0):
        raise ValueError("test_size must be between 0 and 1.")

    rng = np.random.default_rng(seed)

    ids = np.arange(n_images)

    rng.shuffle(ids)

    n_test = max(
        1,
        int(round(n_images * test_size)),
    )

    n_test = min(
        n_test,
        n_images - 1,
    )

    test_ids = ids[:n_test]
    train_ids = ids[n_test:]

    return (
        train_ids,
        test_ids,
    )


def build_datasets(
    covers,
    stegos,
    test_size: float = 0.3,
    seed: int = 0,
) -> tuple[
    StegoImageDataset,
    StegoImageDataset,
]:
    """Build leakage-safe train/test datasets."""

    if len(covers) != len(stegos):
        raise ValueError("covers and stegos must have equal length.")

    n_images = len(covers)

    train_ids, test_ids = group_split_indices(
        n_images,
        test_size,
        seed,
    )

    def make_subset(
        ids: np.ndarray,
    ) -> StegoImageDataset:

        images = []
        labels = []

        for index in ids:
            images.append(covers[index])
            labels.append(0)

            images.append(stegos[index])
            labels.append(1)

        return StegoImageDataset(
            images,
            labels,
        )

    return (
        make_subset(train_ids),
        make_subset(test_ids),
    )


@dataclass(frozen=True)
class TrainResult:
    """CNN training/evaluation result."""

    train_losses: list[float]
    test_accuracy: float
    test_detection_error: float


def train_and_evaluate_cnn(
    model: nn.Module,
    train_ds: StegoImageDataset,
    test_ds: StegoImageDataset,
    epochs: int = 10,
    lr: float = 1e-3,
    batch_size: int = 8,
    seed: int = 0,
) -> TrainResult:
    """Train a CNN and evaluate detection performance."""

    if epochs <= 0:
        raise ValueError("epochs must be greater than zero.")

    if lr <= 0:
        raise ValueError("lr must be greater than zero.")

    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero.")

    torch.manual_seed(seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
    )

    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]

    optimizer = torch.optim.Adam(
        trainable_parameters,
        lr=lr,
    )

    criterion = nn.CrossEntropyLoss()

    train_losses: list[float] = []

    # ---------------------------------------------------------
    # Training
    # ---------------------------------------------------------
    model.train()

    for _epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0

        for x, y in train_loader:
            optimizer.zero_grad()

            output = model(x)

            loss = criterion(
                output,
                y,
            )

            loss.backward()

            optimizer.step()

            epoch_loss += float(loss.item())

            n_batches += 1

        train_losses.append(
            epoch_loss
            / max(
                n_batches,
                1,
            )
        )

    # ---------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------
    model.eval()

    correct = 0
    total = 0

    false_alarms = 0
    missed_detections = 0

    n_cover = 0
    n_stego = 0

    with torch.no_grad():
        for x, y in test_loader:
            output = model(x)

            predictions = output.argmax(dim=1)

            correct += int((predictions == y).sum().item())

            total += len(y)

            for prediction, target in zip(
                predictions.tolist(),
                y.tolist(),
            ):
                if target == 0:
                    n_cover += 1

                    if prediction == 1:
                        false_alarms += 1

                else:
                    n_stego += 1

                    if prediction == 0:
                        missed_detections += 1

    accuracy = correct / max(
        total,
        1,
    )

    false_alarm_rate = false_alarms / max(
        n_cover,
        1,
    )

    missed_detection_rate = missed_detections / max(
        n_stego,
        1,
    )

    detection_error = (false_alarm_rate + missed_detection_rate) / 2.0

    return TrainResult(
        train_losses=train_losses,
        test_accuracy=float(accuracy),
        test_detection_error=float(detection_error),
    )
