import math

import numpy as np
import pandas as pd
import pytest

from lunar_crater.data import split as split_mod
from lunar_crater.data.projection import MOON_RADIUS_M, to_unit_vectors
from lunar_crater.data.split import (
    DROPPED,
    apply_buffer,
    assign_cells,
    cell_of,
    cells_per_band,
    split_tiles,
)
from lunar_crater.data.tiles import random_centres

KM_PER_DEG = math.pi * MOON_RADIUS_M / 180 / 1000


def tiles_table(n=3000, seed=0):
    centres = random_centres(n, seed=seed)
    return pd.DataFrame(
        {
            "tile_id": [f"t{i:06d}" for i in range(n)],
            "lat": centres[:, 0],
            "lon": centres[:, 1],
            "size_px": 256,
            "scale_m": 118.45,
            "n_craters": 10,
        }
    )


def test_cells_cover_the_sphere_with_similar_areas():
    per_band = cells_per_band(6, 12)
    centres = random_centres(200_000, seed=1)
    cells = cell_of(centres[:, 0], centres[:, 1], n_bands=6, cells_at_equator=12)
    counts = np.bincount(cells, minlength=per_band.sum())

    assert cells.min() == 0 and cells.max() == per_band.sum() - 1
    assert per_band.tolist() == [3, 9, 12, 12, 9, 3]
    # Uniform points: similar cell populations mean similar cell areas.
    assert counts.max() / counts.min() < 1.2


def test_cells_handle_poles_and_the_antimeridian():
    cells = cell_of([90.0, -90.0, 0.0, 0.0], [0.0, 0.0, 180.0, -180.0])

    per_band = cells_per_band(6, 12)
    assert cells[1] < per_band[0]  # south pole is in the first band
    assert cells[0] >= per_band.sum() - per_band[-1]  # north pole in the last
    assert cells[2] == cells[3]  # 180 E and 180 W are the same place


def test_assign_cells_respects_fractions_and_seed():
    names = assign_cells(40, (0.7, 0.15, 0.15), seed=2)

    assert (names == "test").sum() == 6 and (names == "val").sum() == 6
    np.testing.assert_array_equal(names, assign_cells(40, (0.7, 0.15, 0.15), seed=2))
    with pytest.raises(ValueError):
        assign_cells(10, (0.5, 0.5, 0.5))


def test_buffer_drops_lower_priority_tiles():
    step = 10 / KM_PER_DEG  # 10 km apart along the equator
    lat = np.zeros(4)
    lon = np.array([0.0, 1, 2, 10]) * step
    split = np.array(["test", "val", "train", "train"], dtype=object)

    result = apply_buffer(lat, lon, split, min_separation_km=15)

    # val is 10 km from test -> dropped; the first train tile is then 20 km from test
    # (no kept val beside it) -> kept; the far train tile is untouched.
    assert result.tolist() == ["test", DROPPED, "train", "train"]


def test_split_keeps_splits_apart_by_tile_diagonal_plus_buffer():
    tiles = tiles_table()
    result = split_tiles(tiles, buffer_km=30, seed=0)
    kept = result["split"] != DROPPED

    xyz = to_unit_vectors(tiles["lat"][kept], tiles["lon"][kept])
    names = result["split"][kept].to_numpy()
    sep_km = np.degrees(np.arccos(np.clip(xyz @ xyz.T, -1, 1))) * KM_PER_DEG
    min_cross = sep_km[names[:, None] != names[None, :]].min()

    diagonal_km = 256 * 118.45 * math.sqrt(2) / 1000
    assert min_cross >= diagonal_km + 30 - 1e-6
    assert set(names) == {"train", "val", "test"}


def test_split_is_reproducible(tmp_path):
    tiles = tiles_table(500)
    tiles.to_csv(tmp_path / "tiles.csv", index=False)

    assert split_mod.main([str(tmp_path), "--seed", "4"]) == 0
    first = pd.read_csv(tmp_path / "split.csv")
    assert split_mod.main([str(tmp_path), "--seed", "4"]) == 0

    pd.testing.assert_frame_equal(first, pd.read_csv(tmp_path / "split.csv"))
    summary = pd.read_csv(tmp_path / "split_summary.csv", index_col=0)
    assert summary["tiles"].sum() == 500


def test_seed_search_picks_the_least_imbalanced_seed(tmp_path):
    tiles = tiles_table(800)
    scores = [split_mod.imbalance(tiles, split_tiles(tiles, seed=s)) for s in range(8)]

    seed, score = split_mod.most_balanced_seed(tiles, 8)

    assert score == min(scores) and seed == scores.index(min(scores))
    tiles.to_csv(tmp_path / "tiles.csv", index=False)
    assert split_mod.main([str(tmp_path), "--search-seeds", "8"]) == 0
    meta = pd.read_json(tmp_path / "split_meta.json", typ="series")
    assert meta["seed_used"] == seed
