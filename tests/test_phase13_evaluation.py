import numpy as np
import pandas as pd
import pytest

from src.evaluation.runner import (
    ImageResult,
    payload_to_w,
    precompute,
    run_one,
)
from src.evaluation.aggregate import (
    SummaryRow,
    image_results_to_df,
    summarize,
    summaries_to_df,
)


def make_cover(seed: int = 0, shape=(32, 32)) -> np.ndarray:
    rng = np.random.default_rng(seed)

    height, width = shape
    y, x = np.mgrid[0:height, 0:width]

    base = (
        70.0
        + 70.0 * x / max(width - 1, 1)
        + 50.0 * y / max(height - 1, 1)
    )

    noise = rng.normal(
        loc=0.0,
        scale=12.0,
        size=shape,
    )

    return np.clip(
        base + noise,
        0,
        255,
    ).astype(np.uint8)


def test_payload_to_w_known_values():
    assert payload_to_w(1.0) == 1
    assert payload_to_w(0.5) == 2
    assert payload_to_w(0.2) == 5


def test_payload_to_w_phase13_discrete_rates():
    assert payload_to_w(0.05) == 20
    assert payload_to_w(0.10) == 10
    assert payload_to_w(0.20) == 5
    assert payload_to_w(0.30) == 3
    assert payload_to_w(0.40) == 2
    assert payload_to_w(0.50) == 2


def test_payload_to_w_rejects_invalid_values():
    with pytest.raises(ValueError):
        payload_to_w(0.0)

    with pytest.raises(ValueError):
        payload_to_w(-0.1)

    with pytest.raises(ValueError):
        payload_to_w(1.1)


def test_precompute_shapes():
    cover = make_cover(10)

    pre = precompute(
        cover,
        window_height=3,
        window_width=3,
        start_component=8,
        end_component=9,
        gamma=5,
        sigma=3.0,
        tau=2,
    )

    assert pre.cover.shape == cover.shape
    assert pre.zeta.shape == cover.shape
    assert pre.rho_adaptive.shape == cover.shape

    assert np.isfinite(pre.zeta).all()
    assert np.isfinite(pre.rho_adaptive).all()
    assert (pre.rho_adaptive >= 0).all()


def test_run_one_adaptive_exact_extraction():
    cover = make_cover(20)

    pre = precompute(
        cover,
        window_height=3,
        window_width=3,
        start_component=8,
        end_component=9,
        gamma=5,
        sigma=3.0,
        tau=2,
    )

    result, stego = run_one(
        pre=pre,
        method="adaptive",
        target_bpp=0.5,
        h=4,
        image_index=0,
        rng=np.random.default_rng(1234),
    )

    assert result.method == "adaptive"
    assert result.target_bpp == pytest.approx(0.5)
    assert result.actual_bpp == pytest.approx(0.5)

    assert result.extraction_success is True
    assert result.n_changed > 0

    assert result.psnr > 0
    assert result.ssim > 0
    assert result.mse >= 0

    assert stego.shape == cover.shape
    assert stego.dtype == np.uint8


def test_run_one_uniform_exact_extraction():
    cover = make_cover(21)

    pre = precompute(
        cover,
        window_height=3,
        window_width=3,
        start_component=8,
        end_component=9,
        gamma=5,
        sigma=3.0,
        tau=2,
    )

    result, stego = run_one(
        pre=pre,
        method="uniform",
        target_bpp=0.5,
        h=4,
        image_index=1,
        rng=np.random.default_rng(5678),
    )

    assert result.method == "uniform"
    assert result.extraction_success is True
    assert result.actual_bpp == pytest.approx(0.5)
    assert result.n_changed > 0

    assert stego.shape == cover.shape
    assert stego.dtype == np.uint8


def test_run_one_rejects_unknown_method():
    cover = make_cover(30)

    pre = precompute(
        cover,
        window_height=3,
        window_width=3,
        start_component=8,
        end_component=9,
        gamma=5,
        sigma=3.0,
        tau=2,
    )

    with pytest.raises(ValueError):
        run_one(
            pre=pre,
            method="wow",
            target_bpp=0.5,
            h=4,
            image_index=0,
            rng=np.random.default_rng(1),
        )


def test_image_results_to_df():
    result = ImageResult(
        method="adaptive",
        target_bpp=0.5,
        actual_bpp=0.5,
        image_index=0,
        n_changed=100,
        n_trim=1024,
        psnr=45.0,
        ssim=0.98,
        mse=2.0,
        embedding_time_s=0.01,
        extraction_success=True,
        file_size_bytes=1500,
    )

    df = image_results_to_df([result])

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1

    expected_columns = {
        "method",
        "target_bpp",
        "actual_bpp",
        "image_index",
        "n_changed",
        "n_trim",
        "psnr",
        "ssim",
        "mse",
        "embedding_time_s",
        "extraction_success",
        "file_size_bytes",
    }

    assert expected_columns.issubset(
        set(df.columns)
    )


def test_summaries_to_df():
    summary = SummaryRow(
        method="adaptive",
        target_bpp=0.5,
        actual_bpp=0.5,
        n_images=10,
        psnr_mean=45.0,
        psnr_std=1.0,
        ssim_mean=0.98,
        ssim_std=0.01,
        mse_mean=2.0,
        changed_pixels_mean=100.0,
        embedding_time_mean_s=0.02,
        extraction_success_rate=1.0,
        file_size_mean_bytes=1500.0,
        detection_accuracy=0.5,
        detection_pe=0.5,
    )

    df = summaries_to_df([summary])

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert df.iloc[0]["method"] == "adaptive"
    assert df.iloc[0]["actual_bpp"] == pytest.approx(0.5)


def test_summarize_small_grouped_dataset():
    covers = [
        make_cover(seed)
        for seed in range(12)
    ]

    stegos = []

    for index, cover in enumerate(covers):
        stego = cover.copy()

        # Deterministic tiny perturbation.
        row = index % cover.shape[0]
        col = (index * 3) % cover.shape[1]

        stego[row, col] = min(
            int(stego[row, col]) + 1,
            255,
        )

        stegos.append(stego)

    results = []

    for index in range(12):
        results.append(
            ImageResult(
                method="adaptive",
                target_bpp=0.5,
                actual_bpp=0.5,
                image_index=index,
                n_changed=1,
                n_trim=1024,
                psnr=50.0,
                ssim=0.99,
                mse=1.0,
                embedding_time_s=0.01,
                extraction_success=True,
                file_size_bytes=0,
            )
        )

    summary = summarize(
        results=results,
        covers=covers,
        stegos=stegos,
        method="adaptive",
        target_bpp=0.5,
    )

    assert isinstance(summary, SummaryRow)
    assert summary.method == "adaptive"
    assert summary.n_images == 12
    assert summary.actual_bpp == pytest.approx(0.5)

    assert summary.extraction_success_rate == pytest.approx(1.0)

    assert 0.0 <= summary.detection_accuracy <= 1.0
    assert 0.0 <= summary.detection_pe <= 0.5
