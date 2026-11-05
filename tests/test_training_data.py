import numpy as np
import pandas as pd
import pytest

from lunar_crater.training.data import (
    MIN_SPREAD_M,
    augment,
    load_examples,
    normalize_dem,
    rim_target,
)


def test_normalize_dem():
    dem = np.arange(16, dtype=float).reshape(4, 4) * 100
    dem[0, 0] = np.nan
    out = normalize_dem(dem)

    assert out.dtype == np.float32 and out[0, 0] == 0.0
    assert np.nanmedian(out[1:]) == pytest.approx(0.0, abs=0.5)
    assert np.std(out.ravel()[1:]) == pytest.approx(1.0, rel=0.01)


def test_normalize_keeps_flat_tiles_flat():
    dem = np.full((8, 8), 1000.0)
    dem[0, 0] = 1010.0
    out = normalize_dem(dem)
    assert out.max() == pytest.approx(10.0 / MIN_SPREAD_M)
    assert not normalize_dem(np.full((4, 4), np.nan)).any()


def test_rim_target_skips_small_craters():
    craters = pd.DataFrame(
        {"x_px": [16.0, 8.0], "y_px": [16.0, 8.0], "r_px": [10.0, 3.0], "diam_km": [2.4, 0.7]}
    )
    target = rim_target(craters, 32, thickness_px=2.0)
    assert target.dtype == np.float32
    assert target[16, 26] == 1 and target[16, 16] == 0
    assert target[8, 5] == 0  # the 0.7 km crater is not drawn


def test_augment_moves_image_and_target_together():
    rng = np.random.default_rng(1)
    image = np.random.default_rng(0).normal(size=(16, 16)).astype(np.float32)
    for _ in range(8):
        a, b = augment(image, image.copy(), rng)
        assert np.array_equal(a, b)
        assert a.flags["C_CONTIGUOUS"]
        assert sorted(a.ravel()) == sorted(image.ravel())


def test_load_examples(tmp_path):
    (tmp_path / "tiles").mkdir()
    np.savez(tmp_path / "tiles" / "t0.npz", dem=np.ones((32, 32)), mask=np.zeros((32, 32)))
    np.savez(tmp_path / "tiles" / "t1.npz", dem=np.ones((32, 32)), mask=np.zeros((32, 32)))
    tiles = pd.DataFrame({"tile_id": ["t0", "t1"], "size_px": [32, 32]})
    craters = pd.DataFrame(
        {"tile_id": ["t0"], "x_px": [16.0], "y_px": [16.0], "r_px": [8.0], "diam_km": [2.0]}
    )
    images, targets = load_examples(tmp_path, tiles, craters)

    assert images.shape == targets.shape == (2, 32, 32)
    assert targets.dtype == np.uint8
    assert targets[0].sum() > 0 and targets[1].sum() == 0
