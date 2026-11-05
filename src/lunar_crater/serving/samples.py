"""Example tiles for the dashboard, scored against the catalog.

The examples are test tiles, from regions the models never trained or tuned on.
"""

from __future__ import annotations

import base64
from pathlib import Path

import numpy as np
import pandas as pd

from lunar_crater.evaluation.matching import match_circles
from lunar_crater.serving.predictor import CraterPredictor


def encode(array: np.ndarray, dtype) -> str:
    """Base64 of the array's bytes, little-endian, row by row."""
    return base64.b64encode(np.ascontiguousarray(array, dtype=dtype).tobytes()).decode()


class SampleTiles:
    """A few tiles from one split of a tile set, if the tile set is present."""

    def __init__(self, tiles_dir: Path, size_px: int, *, split: str = "test", count: int = 8):
        self.tiles_dir = Path(tiles_dir)
        self.tiles = pd.DataFrame(columns=["tile_id", "lat", "lon", "n_craters"])
        if not (self.tiles_dir / "tiles.csv").exists():
            return
        tiles = pd.read_csv(self.tiles_dir / "tiles.csv")
        names = pd.read_csv(self.tiles_dir / "split.csv")
        tiles = tiles.merge(names[["tile_id", "split"]], on="tile_id")
        tiles = tiles[(tiles["split"] == split) & (tiles["size_px"] == size_px)]
        if "nan_fraction" in tiles:
            tiles = tiles[tiles["nan_fraction"] == 0]
        varied = tiles[tiles["n_craters"].between(10, 80)]
        tiles = varied if len(varied) else tiles
        tiles = tiles.sample(n=min(count, len(tiles)), random_state=0)
        self.tiles = tiles.sort_values("lat", ascending=False).reset_index(drop=True)
        craters = pd.read_csv(self.tiles_dir / "craters.csv")
        self.craters = craters[craters["tile_id"].isin(self.tiles["tile_id"])]

    def listing(self) -> list[dict]:
        return self.tiles[["tile_id", "lat", "lon", "n_craters"]].to_dict("records")

    def __contains__(self, tile_id: str) -> bool:
        return tile_id in set(self.tiles["tile_id"])

    def run(self, tile_id: str, predictor: CraterPredictor) -> dict:
        """Detections on one tile, each marked as matching a catalog crater or not."""
        tile = self.tiles.set_index("tile_id").loc[tile_id]
        with np.load(self.tiles_dir / "tiles" / f"{tile_id}.npz") as z:
            dem = z["dem"].astype(np.float32)
        rims = predictor.rim_maps(dem)
        found = predictor(dem, float(tile["lat"]), float(tile["lon"]))

        truth = self.craters[self.craters["tile_id"] == tile_id].reset_index(drop=True)
        cols = ["x_px", "y_px", "r_px"]
        p_idx, t_idx, _ = match_circles(found[cols].to_numpy(), truth[cols].to_numpy())
        matched = np.zeros(len(found), dtype=bool)
        matched[p_idx] = True
        caught = np.zeros(len(truth), dtype=bool)
        caught[t_idx] = True
        # Craters the protocol counts: centred in the tile and at least 1 km across.
        counted = truth["centre_in_tile"].to_numpy(dtype=bool) & (truth["diam_km"] >= 1.0)
        catalog = truth.assign(found=caught)[counted]

        return {
            "tile_id": tile_id,
            "lat": float(tile["lat"]),
            "lon": float(tile["lon"]),
            "size_px": predictor.size_px,
            "scale_m": predictor.scale_m,
            "heights": encode(np.nan_to_num(dem, nan=np.nanmedian(dem)), "<f4"),
            "rims": encode(np.round(np.clip(rims[0], 0, 1) * 255), "u1"),
            "detections": found.assign(matched=matched).to_dict("records"),
            "catalog": catalog[[*cols, "diam_km", "found"]].to_dict("records"),
        }
