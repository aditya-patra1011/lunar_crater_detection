"""Compare two harness runs on one split, with bootstrap intervals on their F1 difference.

Resamples tiles, then resamples the split's regions (cells). Tiles in one region share
terrain, so the region intervals are the more honest, and wider, of the two. Each run is
scored at the threshold in its run.json.

Usage::

    python scripts/compare_runs.py data/interim/tiles/v1 outputs/unet_v1_test \\
        outputs/classical_v1_test
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from lunar_crater.evaluation.harness import load_split
from lunar_crater.evaluation.matching import evaluate_tile

GROUPS = {"all": (1.0, math.inf), "1-8 km": (1.0, 8.0), "over 8 km": (8.0, math.inf)}


def tile_counts(run_dir: Path, tiles: pd.DataFrame, craters: pd.DataFrame) -> np.ndarray:
    """(n_tiles, n_groups, 4) counts: matched and total catalog craters, then predictions."""
    run = json.loads((run_dir / "run.json").read_text())
    preds = pd.read_csv(run_dir / "predictions.csv")
    preds = preds[preds["score"] >= run["min_score"]]
    by_pred, by_true = (
        dict(tuple(preds.groupby("tile_id"))),
        dict(tuple(craters.groupby("tile_id"))),
    )
    counts = np.zeros((len(tiles), len(GROUPS), 4))
    for i, t in enumerate(tiles.itertuples()):
        truth, pred = evaluate_tile(
            by_pred.get(t.tile_id, preds.iloc[:0]),
            by_true.get(t.tile_id, craters.iloc[:0]),
            size_px=t.size_px,
            scale_m=t.scale_m,
        )
        for g, (lo, hi) in enumerate(GROUPS.values()):
            a = truth[(truth["diam_km"] >= lo) & (truth["diam_km"] < hi)]
            b = pred[(pred["diam_km"] >= lo) & (pred["diam_km"] < hi)]
            counts[i, g] = a["matched"].sum(), len(a), b["matched"].sum(), len(b)
    return counts


def f1(counts: np.ndarray) -> np.ndarray:
    """Pooled F1 per group from (n_tiles, n_groups, 4) counts; 0 where it is undefined."""
    tp_true, n_true, tp_pred, n_pred = counts.sum(axis=0).T
    recall = np.divide(tp_true, n_true, out=np.zeros_like(n_true), where=n_true > 0)
    precision = np.divide(tp_pred, n_pred, out=np.zeros_like(n_pred), where=n_pred > 0)
    total = precision + recall
    return np.divide(2 * precision * recall, total, out=np.zeros_like(total), where=total > 0)


def bootstrap(a, b, units: list[np.ndarray], n_boot: int, rng) -> pd.DataFrame:
    """95% intervals for F1 of a, F1 of b and their difference, resampling ``units``."""
    samples = []
    for _ in range(n_boot):
        idx = np.concatenate([units[i] for i in rng.integers(len(units), size=len(units))])
        samples.append([f1(a[idx]), f1(b[idx])])
    fa, fb = np.array(samples).transpose(1, 0, 2)
    rows = {}
    for g, name in enumerate(GROUPS):
        lo_a, hi_a = np.percentile(fa[:, g], [2.5, 97.5])
        lo_b, hi_b = np.percentile(fb[:, g], [2.5, 97.5])
        lo_d, hi_d = np.percentile(fa[:, g] - fb[:, g], [2.5, 97.5])
        rows[name] = {
            "a_low": lo_a, "a_high": hi_a, "b_low": lo_b, "b_high": hi_b,
            "diff_low": lo_d, "diff_high": hi_d,
        }  # fmt: skip
    return pd.DataFrame(rows).T


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("tiles_dir", type=Path)
    parser.add_argument("run_a", type=Path)
    parser.add_argument("run_b", type=Path)
    parser.add_argument("--split", default="test")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    tiles, craters = load_split(args.tiles_dir, args.split)
    cells = pd.read_csv(args.tiles_dir / "split.csv").set_index("tile_id")["cell"]
    cell = cells.loc[tiles["tile_id"]].to_numpy()
    a = tile_counts(args.run_a, tiles, craters)
    b = tile_counts(args.run_b, tiles, craters)
    rng = np.random.default_rng(args.seed)

    point = pd.DataFrame({"a": f1(a), "b": f1(b)}, index=list(GROUPS))
    print(f"F1 of a={args.run_a.name}, b={args.run_b.name}")
    print(point.round(3).to_string())
    print("\nResampling tiles:")
    units = [np.array([i]) for i in range(len(tiles))]
    print(bootstrap(a, b, units, args.n_boot, rng).round(3).to_string())
    print(f"\nResampling {len(np.unique(cell))} regions:")
    units = [np.nonzero(cell == c)[0] for c in np.unique(cell)]
    print(bootstrap(a, b, units, args.n_boot, rng).round(3).to_string())


if __name__ == "__main__":
    main()
