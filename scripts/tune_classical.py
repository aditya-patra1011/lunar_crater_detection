"""Tune the classical detector: fit its radius scale on train, pick its shape on val.

For each annulus inner fraction, the radius scale is fitted on train tiles. Each
(inner, nms_iou) pair is then scored on val tiles at its best score threshold. Test
tiles are never touched.

Usage::

    python scripts/tune_classical.py data/interim/tiles/v1 out.csv --train-tiles 200 \\
        --val-tiles 150 --workers 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from lunar_crater.baselines.classical import RadialSlopeDetector, fit_radius_scale
from lunar_crater.evaluation.harness import (
    best_threshold,
    default_thresholds,
    detect_tiles,
    load_split,
    sweep,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("tiles_dir", type=Path)
    parser.add_argument("out", type=Path, help="CSV of val results per setting")
    parser.add_argument("--train-tiles", type=int, default=200)
    parser.add_argument("--val-tiles", type=int, default=150)
    parser.add_argument("--inner", type=float, nargs="+", default=[0.3, 0.5, 0.7])
    parser.add_argument("--nms-iou", type=float, nargs="+", default=[0.2, 0.3, 0.5])
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    train, train_craters = load_split(args.tiles_dir, "train")
    train = train.sample(n=min(args.train_tiles, len(train)), random_state=0)
    train_craters = train_craters[
        train_craters["tile_id"].isin(train["tile_id"])
        & train_craters["centre_in_tile"]
        & (train_craters["diam_km"] >= 1.0)
    ]
    val, val_craters = load_split(args.tiles_dir, "val")
    val = val.sample(n=min(args.val_tiles, len(val)), random_state=0)

    rows = []
    for inner in args.inner:
        preds = detect_tiles(
            args.tiles_dir, train, RadialSlopeDetector(inner=inner), workers=args.workers
        )
        scale = fit_radius_scale(preds, train_craters)
        print(f"inner {inner}: radius scale {scale:.3f}", file=sys.stderr)
        for nms_iou in args.nms_iou:
            detector = RadialSlopeDetector(inner=inner, nms_iou=nms_iou, radius_scale=scale)
            preds = detect_tiles(args.tiles_dir, val, detector, workers=args.workers)
            curve = sweep(preds, val, val_craters, default_thresholds(preds["score"]))
            best = curve[curve["min_score"] == best_threshold(curve)].iloc[0]
            rows.append(
                {"inner": inner, "nms_iou": nms_iou, "radius_scale": scale, **best.to_dict()}
            )
            print(f"  nms_iou {nms_iou}: F1 {best['f1']:.3f}", file=sys.stderr)

    table = pd.DataFrame(rows).sort_values("f1", ascending=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    print(table.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
