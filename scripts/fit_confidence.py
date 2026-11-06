"""Fit confidence curves on val detections and check them on test detections.

Both runs must come from the same detector, run by the harness with a low threshold so
that low-scoring detections are kept. The curves are fitted on val only; test is used
to report how well they hold.

Usage::

    python scripts/fit_confidence.py data/interim/tiles/v1 outputs/served_v1_val \\
        outputs/served_v1_test models/crater_detector.confidence.json --min-score 0.4362
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from lunar_crater.evaluation.harness import load_split
from lunar_crater.postprocess.confidence import (
    Confidence,
    calibration_error,
    label_detections,
    reliability,
)


def labelled(tiles_dir: Path, run_dir: Path, split: str, min_score: float) -> pd.DataFrame:
    tiles, craters = load_split(tiles_dir, split)
    preds = pd.read_csv(run_dir / "predictions.csv")
    return label_detections(preds[preds["score"] >= min_score], tiles, craters)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("tiles_dir", type=Path)
    parser.add_argument("val_run", type=Path)
    parser.add_argument("test_run", type=Path)
    parser.add_argument("out", type=Path, help="JSON file for the curves")
    parser.add_argument("--min-score", type=float, required=True, help="the served threshold")
    args = parser.parse_args()

    val = labelled(args.tiles_dir, args.val_run, "val", args.min_score)
    confidence = Confidence.fit(val, fitted_on=f"val detections from {args.val_run.name}")
    confidence.save(args.out)

    test = labelled(args.tiles_dir, args.test_run, "test", args.min_score)
    for name, rows in (("val (fitted)", val), ("test", test)):
        p = confidence(rows["score"], rows["diam_km"])
        y = rows["matched"].to_numpy(float)
        brier, flat = np.mean((p - y) ** 2), np.mean((y.mean() - y) ** 2)
        print(f"\n{name}: {len(rows)} detections, {y.mean():.3f} match the catalog")
        print(f"  calibration error {calibration_error(p, y):.3f}")
        print(f"  Brier score {brier:.4f} (one rate for all: {flat:.4f})")
        for group, mask in (("small", rows["diam_km"] < confidence.large_from_km),
                            ("large", rows["diam_km"] >= confidence.large_from_km)):  # fmt: skip
            if mask.any():
                print(f"  {group}: {int(mask.sum())} detections, predicted {p[mask].mean():.3f}, "
                      f"observed {y[mask].mean():.3f}")  # fmt: skip
        print(reliability(p, y).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
