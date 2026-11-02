"""Split tiles into train/val/test by region, with a buffer between splits.

The sphere is cut into near-equal-area cells, each cell goes wholly to one split, and
tiles that sit too close to a tile of another split are dropped. See
docs/decisions/0002-geographic-split.md.

Usage::

    python -m lunar_crater.data.split data/interim/tiles/v1
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from lunar_crater.data.projection import MOON_RADIUS_M, to_unit_vectors

SPLITS = ("train", "val", "test")
DROPPED = "buffer"


def cell_of(lat, lon, *, n_bands: int = 6, cells_at_equator: int = 12) -> np.ndarray:
    """Return the cell index of each point.

    Latitude bands are equally tall; each band is cut into longitude cells in
    proportion to its area, so cells have similar areas (within ~10% by default)
    without the thin slivers that equal-area bands give near the poles.
    """
    lat = np.asarray(lat, dtype=float)
    band = np.minimum(((lat + 90.0) / (180.0 / n_bands)).astype(int), n_bands - 1)
    per_band = cells_per_band(n_bands, cells_at_equator)
    first = np.concatenate([[0], np.cumsum(per_band)[:-1]])
    lon01 = (np.asarray(lon, dtype=float) + 180.0) % 360.0 / 360.0
    n_lon = per_band[band]
    return first[band] + np.minimum((lon01 * n_lon).astype(int), n_lon - 1)


def cells_per_band(n_bands: int, cells_at_equator: int) -> np.ndarray:
    """Cells in each band, proportional to band area, with ``cells_at_equator`` at the equator."""
    edges = np.sin(np.radians(np.linspace(-90.0, 90.0, n_bands + 1)))
    areas = np.diff(edges)
    return np.maximum(np.round(cells_at_equator * areas / areas.max()).astype(int), 1)


def assign_cells(
    n_cells: int, fractions: tuple[float, float, float] = (0.7, 0.15, 0.15), *, seed: int = 0
) -> np.ndarray:
    """Randomly give each cell a split name, in the given train/val/test proportions."""
    if not math.isclose(sum(fractions), 1.0) or min(fractions) < 0:
        raise ValueError(f"fractions must be non-negative and sum to 1, got {fractions}")
    order = np.random.default_rng(seed).permutation(n_cells)
    n_test = round(fractions[2] * n_cells)
    n_val = round(fractions[1] * n_cells)
    names = np.full(n_cells, "train", dtype=object)
    names[order[:n_test]] = "test"
    names[order[n_test : n_test + n_val]] = "val"
    return names


def apply_buffer(lat, lon, split, min_separation_km: float, *, chunk: int = 4096) -> np.ndarray:
    """Return split names with tiles too close to another split replaced by ``"buffer"``.

    Test tiles are always kept; a val tile is dropped near a test tile, and a train
    tile near any kept val or test tile.
    """
    xyz = to_unit_vectors(np.asarray(lat), np.asarray(lon))
    cos_limit = math.cos(min_separation_km * 1000.0 / MOON_RADIUS_M)
    result = np.asarray(split, dtype=object).copy()
    for name, blockers in (("val", ("test",)), ("train", ("val", "test"))):
        mine = np.flatnonzero(result == name)
        others = xyz[np.isin(result, blockers)]
        if len(mine) == 0 or len(others) == 0:
            continue
        for start in range(0, len(mine), chunk):
            idx = mine[start : start + chunk]
            too_close = (xyz[idx] @ others.T > cos_limit).any(axis=1)
            result[idx[too_close]] = DROPPED
    return result


def split_tiles(
    tiles: pd.DataFrame,
    *,
    n_bands: int = 6,
    cells_at_equator: int = 12,
    fractions: tuple[float, float, float] = (0.7, 0.15, 0.15),
    buffer_km: float = 30.0,
    seed: int = 0,
) -> pd.DataFrame:
    """Return ``tile_id``, ``cell`` and ``split`` for every tile in a tiles.csv table.

    Tiles of different splits end up at least ``buffer_km`` apart, edge to edge.
    """
    cells = cell_of(tiles["lat"], tiles["lon"], n_bands=n_bands, cells_at_equator=cells_at_equator)
    names = assign_cells(int(cells_per_band(n_bands, cells_at_equator).sum()), fractions, seed=seed)
    diagonal_km = (tiles["size_px"] * tiles["scale_m"]).max() * math.sqrt(2) / 1000.0
    split = apply_buffer(tiles["lat"], tiles["lon"], names[cells], diagonal_km + buffer_km)
    return pd.DataFrame({"tile_id": tiles["tile_id"], "cell": cells, "split": split})


def summarize(tiles: pd.DataFrame, split: pd.DataFrame) -> pd.DataFrame:
    """Per-split counts and simple terrain mix, to spot an unbalanced draw."""
    df = tiles.merge(split, on="tile_id")
    grouped = df.groupby("split")
    summary = pd.DataFrame(
        {
            "tiles": grouped.size(),
            "cells": grouped["cell"].nunique(),
            "craters": grouped["n_craters"].sum(),
            "craters_per_tile": grouped["n_craters"].mean().round(1),
            "mean_abs_lat": grouped["lat"].apply(lambda s: s.abs().mean()).round(1),
            "near_side_fraction": grouped["lon"].apply(lambda s: (s.abs() < 90).mean()).round(2),
        }
    )
    return summary.reindex([*SPLITS, DROPPED]).dropna(how="all")


def imbalance(tiles: pd.DataFrame, split: pd.DataFrame) -> float:
    """Worst relative gap between val/test and the whole tile set in terrain mix.

    Compares near-side fraction (as a share of its 0.5 ideal), craters per tile and
    mean |latitude|. Uses only tile metadata, never model results.
    """
    near = (tiles["lon"].abs() < 90).mean()
    density = tiles["n_craters"].mean()
    abs_lat = tiles["lat"].abs().mean()
    summary = summarize(tiles, split)
    worst = 0.0
    for name in ("val", "test"):
        if name not in summary.index:
            return math.inf
        row = summary.loc[name]
        worst = max(
            worst,
            abs(row["near_side_fraction"] - near) / 0.5,
            abs(row["craters_per_tile"] / density - 1),
            abs(row["mean_abs_lat"] / abs_lat - 1),
        )
    return float(worst)


def most_balanced_seed(tiles: pd.DataFrame, n_seeds: int, **kwargs) -> tuple[int, float]:
    """Try seeds 0..n_seeds-1 and return the one with the lowest :func:`imbalance`."""
    scores = [(imbalance(tiles, split_tiles(tiles, seed=s, **kwargs)), s) for s in range(n_seeds)]
    score, seed = min(scores)
    return seed, score


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("tiles_dir", type=Path, help="folder written by lunar_crater.data.tiles")
    parser.add_argument("--n-bands", type=int, default=6)
    parser.add_argument("--cells-at-equator", type=int, default=12)
    parser.add_argument("--fractions", type=float, nargs=3, default=(0.7, 0.15, 0.15))
    parser.add_argument("--buffer-km", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--search-seeds",
        type=int,
        default=0,
        metavar="N",
        help="ignore --seed and use the most balanced of seeds 0..N-1",
    )
    args = parser.parse_args(argv)

    tiles = pd.read_csv(args.tiles_dir / "tiles.csv")
    options = {
        "n_bands": args.n_bands,
        "cells_at_equator": args.cells_at_equator,
        "fractions": tuple(args.fractions),
        "buffer_km": args.buffer_km,
    }
    seed = args.seed
    if args.search_seeds:
        seed, _ = most_balanced_seed(tiles, args.search_seeds, **options)
        print(f"most balanced seed: {seed}", file=sys.stderr)
    split = split_tiles(tiles, seed=seed, **options)
    summary = summarize(tiles, split)
    split.to_csv(args.tiles_dir / "split.csv", index=False)
    summary.to_csv(args.tiles_dir / "split_summary.csv")
    meta = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    meta |= {"seed_used": seed, "imbalance": imbalance(tiles, split)}
    (args.tiles_dir / "split_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(summary.to_string(), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
