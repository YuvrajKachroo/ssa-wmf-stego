import numpy as np
import pytest
import torch

from src.steganalysis.cnn_pipeline import (
    StegoImageDataset,
    build_datasets,
    group_split_indices,
    train_and_evaluate_cnn,
)
from src.steganalysis.xunet import (
    SimpleBaselineCNN,
    XuNetStyle,
)


def test_simple_cnn_forward_shape():

    model = SimpleBaselineCNN()

    x = torch.randn(
        4,
        1,
        64,
        64,
    )

    output = model(x)

    assert output.shape == (
        4,
        2,
    )


def test_xunet_forward_shape():

    model = XuNetStyle()

    x = torch.randn(
        4,
        1,
        64,
        64,
    )

    output = model(x)

    assert output.shape == (
        4,
        2,
    )


def test_xunet_hpf_is_frozen_and_matches_kv_kernel():

    model = XuNetStyle()

    assert not model.hpf.weight.requires_grad

    weights = model.hpf.weight.detach().numpy().squeeze()

    assert weights.shape == (
        5,
        5,
    )

    assert weights[2, 2] == pytest.approx(-1.0)


def test_xunet_hpf_stays_frozen_after_training_step():

    model = XuNetStyle()

    weights_before = model.hpf.weight.detach().clone()

    x = torch.randn(
        2,
        1,
        32,
        32,
    )

    y = torch.tensor([0, 1])

    optimizer = torch.optim.Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=0.1,
    )

    output = model(x)

    loss = torch.nn.functional.cross_entropy(
        output,
        y,
    )

    loss.backward()

    optimizer.step()

    assert torch.allclose(
        model.hpf.weight,
        weights_before,
    )


def test_group_split_never_overlaps():

    train_ids, test_ids = group_split_indices(
        20,
        test_size=0.3,
        seed=0,
    )

    assert set(train_ids).isdisjoint(set(test_ids))

    assert len(train_ids) + len(test_ids) == 20


def test_stego_image_dataset_shapes_and_normalization():

    images = [
        np.full(
            (16, 16),
            255,
            dtype=np.uint8,
        ),
        np.zeros(
            (16, 16),
            dtype=np.uint8,
        ),
    ]

    dataset = StegoImageDataset(
        images,
        [1, 0],
    )

    x0, y0 = dataset[0]

    assert x0.shape == (
        1,
        16,
        16,
    )

    assert x0.max().item() == pytest.approx(1.0)

    assert y0 == 1


def test_build_datasets_no_group_leakage():

    covers = [
        np.random.default_rng(i).integers(
            0,
            256,
            (16, 16),
            dtype=np.uint8,
        )
        for i in range(10)
    ]

    stegos = [image.copy() for image in covers]

    train_ds, test_ds = build_datasets(
        covers,
        stegos,
        test_size=0.3,
        seed=0,
    )

    assert len(train_ds) + len(test_ds) == 20


def test_training_loop_runs_and_loss_decreases():

    rng = np.random.default_rng(0)

    covers = [
        rng.integers(
            0,
            40,
            (32, 32),
            dtype=np.uint8,
        )
        for _ in range(12)
    ]

    stegos = [
        rng.integers(
            200,
            256,
            (32, 32),
            dtype=np.uint8,
        )
        for _ in range(12)
    ]

    train_ds, test_ds = build_datasets(
        covers,
        stegos,
        test_size=0.3,
        seed=0,
    )

    model = SimpleBaselineCNN()

    result = train_and_evaluate_cnn(
        model,
        train_ds,
        test_ds,
        epochs=5,
        batch_size=4,
        seed=0,
    )

    assert all(np.isfinite(loss) for loss in result.train_losses)

    assert result.train_losses[-1] < result.train_losses[0]

    assert result.test_accuracy > 0.8
