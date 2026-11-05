import numpy as np
import pandas as pd
import pytest

from lunar_crater.data.tiles import CraterIndex, TileSpec
from lunar_crater.postprocess.georef import to_moon_coords


@pytest.mark.parametrize("lat0, lon0", [(0.0, 0.0), (42.0, -170.0), (-85.0, 179.5)])
def test_catalog_craters_map_back_to_their_position_and_size(lat0, lon0):
    spec = TileSpec(lat0, lon0)
    rng = np.random.default_rng(0)
    catalog = pd.DataFrame(
        {
            "crater_id": [f"c{i}" for i in range(20)],
            "lat": np.clip(lat0 + rng.uniform(-0.3, 0.3, 20), -90, 90),
            "lon": (lon0 + rng.uniform(-0.3, 0.3, 20) + 180) % 360 - 180,
            "diam_km": rng.uniform(1, 40, 20),
        }
    )
    placed = CraterIndex(catalog).query(spec)
    assert len(placed) == len(catalog)

    back = to_moon_coords(placed, spec).merge(catalog, on="crater_id", suffixes=("", "_true"))
    assert back["lat"].to_numpy() == pytest.approx(back["lat_true"].to_numpy(), abs=1e-8)
    gap = (back["lon"] - back["lon_true"] + 180) % 360 - 180
    assert np.abs(gap).max() < 1e-6
    assert back["diam_km"].to_numpy() == pytest.approx(back["diam_km_true"].to_numpy(), rel=1e-9)


def test_tile_centre_and_empty_detections():
    spec = TileSpec(10.0, 20.0, size_px=257)
    one = pd.DataFrame({"x_px": [128.0], "y_px": [128.0], "r_px": [10.0]})
    found = to_moon_coords(one, spec)
    assert found["lat"].iloc[0] == pytest.approx(10.0)
    assert found["lon"].iloc[0] == pytest.approx(20.0)
    assert found["diam_km"].iloc[0] == pytest.approx(2.369, abs=0.001)
    assert to_moon_coords(one.iloc[:0], spec).columns.tolist()[-3:] == ["lat", "lon", "diam_km"]
