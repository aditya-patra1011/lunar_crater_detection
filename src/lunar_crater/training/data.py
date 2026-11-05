"""Model inputs and rim targets for one tile, in numpy (no torch needed).

Inputs are the tile heights with the tile median removed, divided by the tile's spread
(with a floor, so flat mare is not blown up into noise). Targets are crater rims drawn
from the tile's catalog rows, so the rim width can change without re-tiling.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from lunar_crater.data.tiles import rasterize_rims

#: Spread floor in metres. Tiles flatter than this keep their true relief.
MIN_SPREAD_M = 50.0


def normalize_dem(dem: np.ndarray) -> np.ndarray:
    """Heights relative to the tile median in units of the tile's spread; NaN becomes 0."""
    dem = np.asarray(dem, dtype=np.float64)
    if not np.isfinite(dem).any():
        return np.zeros(dem.shape, dtype=np.float32)
    centred = dem - np.nanmedian(dem)
    spread = max(float(np.nanstd(centred)), MIN_SPREAD_M)
    return np.nan_to_num(centred / spread).astype(np.float32)


def rim_target(
    craters: pd.DataFrame, size_px: int, *, thickness_px: float = 2.0, min_diam_km: float = 1.0
) -> np.ndarray:
    """A (size, size) float32 mask of the rims of craters at least ``min_diam_km`` across.

    ``craters`` is the tile's rows of craters.csv. Craters centred outside the tile are
    drawn too, so cut rims are labelled where they cross the tile.
    """
    keep = craters[craters["diam_km"] >= min_diam_km]
    mask = rasterize_rims(
        keep["x_px"].to_numpy(),
        keep["y_px"].to_numpy(),
        keep["r_px"].to_numpy(),
        size_px,
        thickness_px=thickness_px,
    )
    return mask.astype(np.float32)


def augment(image: np.ndarray, target: np.ndarray, rng: np.random.Generator):
    """The same random rotation by a multiple of 90 degrees and flip, applied to both.

    The target may have leading channels. Craters have no preferred direction, and these
    moves keep pixels on the grid, so the target stays exact.
    """
    k = int(rng.integers(4))
    image, target = np.rot90(image, k, axes=(-2, -1)), np.rot90(target, k, axes=(-2, -1))
    if rng.random() < 0.5:
        image, target = image[..., ::-1], target[..., ::-1]
    return np.ascontiguousarray(image), np.ascontiguousarray(target)


def load_examples(
    tiles_dir,
    tiles: pd.DataFrame,
    craters: pd.DataFrame,
    *,
    thickness_px: float = 2.0,
    large_min_diam_km: float = 0.0,
    large_thickness_px: float = 4.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalized float32 inputs (n, size, size) and uint8 targets (n, channels, size, size).

    Channel 0 holds all rims. If ``large_min_diam_km`` is set, channel 1 holds only rims
    of craters at least that wide, drawn ``large_thickness_px`` wide.
    """
    by_tile = dict(tuple(craters.groupby("tile_id")))
    empty = craters.iloc[:0]
    images, targets = [], []
    for t in tiles.itertuples():
        with np.load(tiles_dir / "tiles" / f"{t.tile_id}.npz") as z:
            images.append(normalize_dem(z["dem"]))
        rows = by_tile.get(t.tile_id, empty)
        size = int(t.size_px)
        channels = [rim_target(rows, size, thickness_px=thickness_px)]
        if large_min_diam_km:
            channels.append(
                rim_target(
                    rows, size, thickness_px=large_thickness_px, min_diam_km=large_min_diam_km
                )
            )
        targets.append(np.stack(channels).astype(np.uint8))
    return np.stack(images), np.stack(targets)
