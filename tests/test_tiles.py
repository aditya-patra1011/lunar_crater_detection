import json

import numpy as np
import pandas as pd
import pytest

from lunar_crater.data import tiles
from lunar_crater.data.projection import MOON_RADIUS_M
from lunar_crater.data.tiles import CraterIndex, TileSpec, random_centres, rasterize_rims
from test_dem import write_dem

SPEC = TileSpec(lat=10.0, lon=20.0, size_px=64, scale_m=100.0)
KM_PER_DEG = np.pi * MOON_RADIUS_M / 180 / 1000


def catalog(*rows):
    return pd.DataFrame(rows, columns=["crater_id", "lat", "lon", "diam_km"])


def test_plane_coords_round_trip_to_pixels():
    x, y = SPEC.plane_coords()
    col, row = SPEC.to_pixels(x, y)

    np.testing.assert_allclose(col, np.tile(np.arange(64), (64, 1)), atol=1e-9)
    np.testing.assert_allclose(row, np.tile(np.arange(64)[:, None], (1, 64)), atol=1e-9)
    assert y[0, 0] > y[-1, 0]  # first row is the northern edge


def test_rim_is_drawn_at_the_radius():
    mask = rasterize_rims([32.0], [32.0], [10.0], 64)
    yy, xx = np.nonzero(mask)
    dist = np.hypot(xx - 32, yy - 32)

    assert mask.dtype == np.uint8
    assert np.all(np.abs(dist - 10) <= 0.5)
    # The ring has no gaps: every point on the true circle is within a pixel of the rim.
    theta = np.linspace(0, 2 * np.pi, 720)
    px, py = 32 + 10 * np.cos(theta), 32 + 10 * np.sin(theta)
    gap = np.min(np.hypot(px[:, None] - xx, py[:, None] - yy), axis=1)
    assert gap.max() <= 1.0


def test_rim_is_clipped_at_the_tile_edge():
    mask = rasterize_rims([0.0, 500.0], [0.0, 500.0], [10.0, 5.0], 64)

    assert mask[:11, :11].sum() > 0
    assert mask[11:, 11:].sum() == 0


def test_query_finds_crater_at_centre():
    craters = CraterIndex(catalog(("a", 10.0, 20.0, 2.0))).query(SPEC)

    assert craters["crater_id"].tolist() == ["a"]
    row = craters.iloc[0]
    assert (row.x_px, row.y_px) == pytest.approx((31.5, 31.5), abs=1e-6)
    assert row.r_px == pytest.approx(10.0, rel=1e-4)  # 1 km radius at 100 m/px
    assert row.centre_in_tile


def test_query_keeps_rims_crossing_from_outside():
    # Centre 5 km east of the tile centre (tile half-width 3.2 km), radius 3 km.
    lon = 20.0 + 5.0 / (KM_PER_DEG * np.cos(np.radians(10.0)))
    craters = CraterIndex(catalog(("edge", 10.0, lon, 6.0))).query(SPEC)

    assert craters["crater_id"].tolist() == ["edge"]
    assert not craters.iloc[0].centre_in_tile


def test_query_drops_far_craters_and_ones_that_swallow_the_tile():
    far = ("far", 12.0, 20.0, 2.0)
    basin = ("basin", 10.0, 20.0, 300.0)
    small = ("small", 10.0, 20.0, 0.1)

    craters = CraterIndex(catalog(far, basin, small)).query(SPEC, min_radius_px=1.0)

    assert craters.empty


def test_random_centres_are_reproducible_and_in_band():
    a = random_centres(500, max_abs_lat=30.0, seed=3)

    np.testing.assert_array_equal(a, random_centres(500, max_abs_lat=30.0, seed=3))
    assert np.all(np.abs(a[:, 0]) <= 30.0)
    assert np.all((a[:, 1] >= -180.0) & (a[:, 1] < 180.0))


def test_cli_writes_tiles_index_and_metadata(tmp_path):
    dem = write_dem(tmp_path / "dem.tif")
    csv = tmp_path / "craters.csv"
    csv.write_text(
        "CRATER_ID,LAT_CIRC_IMG,LON_CIRC_IMG,DIAM_CIRC_IMG\n"
        + "\n".join(
            f"c{i},{lat},{lon % 360},50" for i, (lat, lon) in enumerate(random_centres(200))
        )
    )
    out = tmp_path / "out"

    argv = [
        "--out",
        str(out),
        "--dem",
        str(dem),
        "--catalog",
        str(csv),
        "--count",
        "5",
        "--size",
        "32",
    ]
    assert tiles.main([*argv, "--scale", "5000"]) == 0

    index = pd.read_csv(out / "tiles.csv")
    assert index["tile_id"].tolist() == [f"t{i:06d}" for i in range(5)]
    assert (index["nan_fraction"] == 0).all()
    with np.load(out / "tiles" / "t000000.npz") as tile:
        assert tile["dem"].shape == tile["mask"].shape == (32, 32)
        assert tile["dem"].dtype == np.float32
    assert set(pd.read_csv(out / "craters.csv")["tile_id"]) <= set(index["tile_id"])
    assert json.loads((out / "meta.json").read_text())["args"]["count"] == 5

    assert tiles.main([*argv, "--scale", "5000"]) == 1  # refuses a non-empty folder
