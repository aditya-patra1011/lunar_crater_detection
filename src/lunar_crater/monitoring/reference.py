"""Build the drift monitor's reference set from harness runs of the served model.

Thresholds are learned from how much regions in the reference differ from each other.

Usage::

    python -m lunar_crater.monitoring.reference data/interim/tiles/v1 \\
        models/crater_detector.reference.json --min-score 0.4362 \\
        --run train=outputs/served_v1_train --run val=outputs/served_v1_val
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from lunar_crater.evaluation.harness import load_split
from lunar_crater.monitoring.drift import Reference, learn_thresholds, tile_features


def features_for_run(tiles_dir: Path, run_dir: Path, split: str, min_score: float):
    """One row of monitor features per tile of ``split``, from a harness run."""
    tiles, _ = load_split(tiles_dir, split)
    cells = pd.read_csv(tiles_dir / "split.csv").set_index("tile_id")["cell"]
    preds = pd.read_csv(run_dir / "predictions.csv")
    preds = preds[preds["score"] >= min_score]
    scores = {k: g["score"].to_numpy() for k, g in preds.groupby("tile_id")}
    rows = []
    for t in tiles.itertuples():
        with np.load(tiles_dir / "tiles" / f"{t.tile_id}.npz") as z:
            features = tile_features(z["dem"], t.scale_m, scores.get(t.tile_id, []))
        rows.append({**features, "region": cells[t.tile_id]})
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("tiles_dir", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument(
        "--run", action="append", required=True, metavar="SPLIT=RUN_DIR",
        help="a split and the harness run of the served model on it; repeat for more",
    )  # fmt: skip
    parser.add_argument("--min-score", type=float, required=True, help="the served threshold")
    args = parser.parse_args(argv)

    parts, names = [], []
    for item in args.run:
        split, _, run_dir = item.partition("=")
        parts.append(features_for_run(args.tiles_dir, Path(run_dir), split, args.min_score))
        names.append(split)
    features = pd.concat(parts, ignore_index=True)
    watch, drift = learn_thresholds(features, features["region"].to_numpy())
    source = f"{' and '.join(names)} tiles from {features['region'].nunique()} regions"
    reference = Reference.build(features, source=source)
    reference.watch, reference.drift = watch, drift
    reference.save(args.out)
    print(f"reference from {len(features)} tiles written to {args.out}", file=sys.stderr)
    print(f"watch above {watch}", file=sys.stderr)
    print(f"drift above {drift}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
