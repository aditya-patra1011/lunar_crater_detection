"""Cut orthographic DEM tiles and rasterize crater rims as targets.

See docs/decisions/0001-orthographic-tiles.md for why tiles are reprojected.

Usage::

    python -m lunar_crater.data.tiles --count 1000 --out data/interim/tiles/v1
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from lunar_crater import __version__
from lunar_crater.data.catalog import load_robbins
from lunar_crater.data.dem import DemRaster
from lunar_crater.data.projection import (
    MOON_RADIUS_M,
    ortho_forward,
    ortho_inverse,
    to_unit_vectors,
)

DEFAULT_DEM = Path("data/raw/Lunar_LRO_LOLA_Global_LDEM_118m_Mar2014.tif")
DEFAULT_CATALOG = Path("data/raw/lunar_crater_database_robbins_2018.zip")


@dataclass(frozen=True)
class TileSpec:
    """A square tile in an orthographic projection centred on (lat, lon)."""

    lat: float
    lon: float
    size_px: int = 256
    scale_m: float = 118.4505876  # LOLA 256 px/deg at the equator

    @property
    def half_diagonal_m(self) -> float:
        return self.size_px * self.scale_m / math.sqrt(2)

    def plane_coords(self) -> tuple[np.ndarray, np.ndarray]:
        """Plane coordinates (m) of every pixel centre; rows run north to south."""
        offsets = (np.arange(self.size_px) - (self.size_px - 1) / 2) * self.scale_m
        x, y = np.meshgrid(offsets, -offsets)
        return x, y

    def to_pixels(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        """Plane coordinates (m) to fractional (col, row) pixel positions."""
        centre = (self.size_px - 1) / 2
        return np.asarray(x) / self.scale_m + centre, -np.asarray(y) / self.scale_m + centre

    def from_pixels(self, col, row) -> tuple[np.ndarray, np.ndarray]:
        """Fractional (col, row) pixel positions to plane coordinates (m)."""
        centre = (self.size_px - 1) / 2
        return (np.asarray(col) - centre) * self.scale_m, -(np.asarray(row) - centre) * self.scale_m


def sample_tile(dem: DemRaster, spec: TileSpec) -> np.ndarray:
    """Return the tile's heights in metres as a (size, size) float32 array."""
    x, y = spec.plane_coords()
    lat, lon = ortho_inverse(x, y, spec.lat, spec.lon, dem.radius)
    return dem.sample(lat, lon).reshape(spec.size_px, spec.size_px)


class CraterIndex:
    """Find the catalog craters whose rims cross a tile."""

    def __init__(self, catalog: pd.DataFrame, radius: float = MOON_RADIUS_M):
        self.catalog = catalog.reset_index(drop=True)
        self.radius = radius
        self._xyz = to_unit_vectors(self.catalog["lat"].to_numpy(), self.catalog["lon"].to_numpy())
        # Crater radius as an angle at the Moon's centre.
        self._r_ang = self.catalog["diam_km"].to_numpy() * 500.0 / radius
        self._r_ang_max = float(self._r_ang.max(initial=0.0))

    def query(self, spec: TileSpec, *, min_radius_px: float = 0.0) -> pd.DataFrame:
        """Return craters with columns crater_id, diam_km, x_px, y_px, r_px, centre_in_tile.

        ``x_px``/``y_px`` are column/row positions in the tile; the rim may extend past
        the tile edge.
        """
        centre = to_unit_vectors(spec.lat, spec.lon)
        reach = spec.half_diagonal_m / self.radius
        # Cheap pre-filter: no crater can be further away than this and still reach the tile.
        dot = self._xyz @ centre
        cand = np.flatnonzero(dot > np.cos(min(reach + self._r_ang_max, np.pi)))
        dist = np.arccos(np.clip(dot[cand], -1.0, 1.0))
        r_ang = self._r_ang[cand]
        # The rim can only cross the tile if the crater reaches it and does not swallow it.
        near = (dist < reach + r_ang) & (dist + reach > r_ang)
        hits = self.catalog.iloc[cand[near]]
        r_ang = r_ang[near]

        x, y, visible = ortho_forward(
            hits["lat"].to_numpy(), hits["lon"].to_numpy(), spec.lat, spec.lon, self.radius
        )
        col, row = spec.to_pixels(x, y)
        r_px = self.radius * np.sin(np.minimum(r_ang, np.pi / 2)) / spec.scale_m
        edge = spec.size_px - 0.5
        out = pd.DataFrame(
            {
                "crater_id": hits["crater_id"].to_numpy(),
                "diam_km": hits["diam_km"].to_numpy(),
                "x_px": col,
                "y_px": row,
                "r_px": r_px,
                "centre_in_tile": (col >= -0.5) & (col < edge) & (row >= -0.5) & (row < edge),
            }
        )
        return out[visible & (r_px >= min_radius_px)].reset_index(drop=True)


def rasterize_rims(x_px, y_px, r_px, size_px: int, *, thickness_px: float = 1.0) -> np.ndarray:
    """Draw crater rims as rings in a (size, size) uint8 mask (1 = rim)."""
    mask = np.zeros((size_px, size_px), dtype=bool)
    half = thickness_px / 2
    for cx, cy, r in zip(x_px, y_px, r_px, strict=True):
        x0, x1 = max(math.floor(cx - r - half), 0), min(math.ceil(cx + r + half), size_px - 1)
        y0, y1 = max(math.floor(cy - r - half), 0), min(math.ceil(cy + r + half), size_px - 1)
        if x0 > x1 or y0 > y1:
            continue
        yy, xx = np.ogrid[y0 : y1 + 1, x0 : x1 + 1]
        mask[y0 : y1 + 1, x0 : x1 + 1] |= np.abs(np.hypot(xx - cx, yy - cy) - r) <= half
    return mask.astype(np.uint8)


def random_centres(count: int, *, max_abs_lat: float = 90.0, seed: int = 0) -> np.ndarray:
    """Draw (lat, lon) pairs uniformly over the sphere within a latitude band."""
    rng = np.random.default_rng(seed)
    z_max = math.sin(math.radians(max_abs_lat))
    lat = np.degrees(np.arcsin(rng.uniform(-z_max, z_max, count)))
    lon = rng.uniform(-180.0, 180.0, count)
    return np.column_stack([lat, lon])


def _log(message: str) -> None:
    print(message, file=sys.stderr)


def build_tiles(args: argparse.Namespace) -> int:
    out: Path = args.out
    if out.exists() and any(out.iterdir()):
        _log(f"error: {out} is not empty")
        return 1
    (out / "tiles").mkdir(parents=True, exist_ok=True)

    _log(f"loading catalog {args.catalog}")
    index = CraterIndex(load_robbins(args.catalog))
    tile_rows, crater_tables = [], []
    with DemRaster(args.dem) as dem:
        scale = args.scale or dem.metres_per_pixel
        centres = random_centres(args.count, max_abs_lat=args.max_abs_lat, seed=args.seed)
        for i, (lat, lon) in enumerate(centres):
            tile_id = f"t{i:06d}"
            spec = TileSpec(float(lat), float(lon), args.size, scale)
            heights = sample_tile(dem, spec)
            craters = index.query(spec, min_radius_px=args.min_radius_px)
            mask = rasterize_rims(
                craters["x_px"],
                craters["y_px"],
                craters["r_px"],
                spec.size_px,
                thickness_px=args.rim_thickness,
            )
            np.savez_compressed(out / "tiles" / f"{tile_id}.npz", dem=heights, mask=mask)

            tile_rows.append(
                {
                    "tile_id": tile_id,
                    **asdict(spec),
                    "n_craters": int(craters["centre_in_tile"].sum()),
                    "nan_fraction": float(np.isnan(heights).mean()),
                }
            )
            crater_tables.append(craters.assign(tile_id=tile_id))
            if (i + 1) % max(args.count // 10, 1) == 0:
                _log(f"{i + 1}/{args.count} tiles")

    pd.DataFrame(tile_rows).to_csv(out / "tiles.csv", index=False)
    craters = pd.concat(crater_tables, ignore_index=True)
    craters[["tile_id", *craters.columns[:-1]]].to_csv(out / "craters.csv", index=False)
    meta = {
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "scale_m": scale,
        "lunar_crater_version": __version__,
        "python": platform.python_version(),
        "numpy": np.__version__,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    _log(f"wrote {args.count} tiles to {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True, help="output folder (must be empty)")
    parser.add_argument("--dem", type=Path, default=DEFAULT_DEM)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--count", type=int, default=1000, help="number of tiles")
    parser.add_argument("--size", type=int, default=256, help="tile size in pixels")
    parser.add_argument(
        "--scale", type=float, default=None, help="metres per pixel (default: the DEM's)"
    )
    parser.add_argument("--max-abs-lat", type=float, default=90.0)
    parser.add_argument("--min-radius-px", type=float, default=1.0, help="skip smaller craters")
    parser.add_argument("--rim-thickness", type=float, default=1.0, help="rim width in pixels")
    parser.add_argument("--seed", type=int, default=0)
    return build_tiles(parser.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
