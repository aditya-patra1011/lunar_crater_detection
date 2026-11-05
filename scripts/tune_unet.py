"""Tune the U-Net's ring matching: fit the radius scale on train, the rest on val.

The network runs once per tile; only the ring matching repeats. Each setting is scored
on val tiles at its best threshold. Test tiles are never touched.

Usage::

    python scripts/tune_unet.py data/interim/tiles/v1 models/unet_v1/best.pt out.csv
"""

from __future__ import annotations

import argparse
import itertools
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from lunar_crater.baselines.classical import fit_radius_scale
from lunar_crater.evaluation.harness import best_threshold, default_thresholds, load_split, sweep
from lunar_crater.models.detector import load_model, predict_rims
from lunar_crater.postprocess.rings import RingMatcher


def rim_maps(tiles_dir: Path, tiles: pd.DataFrame, checkpoint: Path) -> list[np.ndarray]:
    model = load_model(str(checkpoint))
    maps = []
    for t in tiles.itertuples():
        with np.load(tiles_dir / "tiles" / f"{t.tile_id}.npz") as z:
            maps.append(predict_rims(model, z["dem"]).astype(np.float32))
    return maps


def _match(job) -> pd.DataFrame:
    tile_id, prob, matcher = job
    return matcher(prob).assign(tile_id=tile_id)


def match_all(tiles, maps, matcher, pool) -> pd.DataFrame:
    jobs = [(t, p, matcher) for t, p in zip(tiles["tile_id"], maps, strict=True)]
    return pd.concat(pool.map(_match, jobs, chunksize=4), ignore_index=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("tiles_dir", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("out", type=Path, help="CSV of val results per setting")
    parser.add_argument("--train-tiles", type=int, default=200)
    parser.add_argument("--val-tiles", type=int, default=150)
    parser.add_argument("--thickness-frac", type=float, nargs="+", default=[0.0, 0.1, 0.2])
    parser.add_argument("--nms-iou", type=float, nargs="+", default=[0.1, 0.3])
    parser.add_argument("--size-exponent", type=float, nargs="+", default=[0.0, 0.5, 1.0])
    parser.add_argument("--workers", type=int, default=6)
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
    train_maps = rim_maps(args.tiles_dir, train, args.checkpoint)
    val_maps = rim_maps(args.tiles_dir, val, args.checkpoint)
    print("rim maps ready", file=sys.stderr, flush=True)

    rows = []
    with ProcessPoolExecutor(args.workers) as pool:
        for frac, nms_iou in itertools.product(args.thickness_frac, args.nms_iou):
            base = RingMatcher(thickness_frac=frac, nms_iou=nms_iou, min_score=0.1)
            scale = fit_radius_scale(match_all(train, train_maps, base, pool), train_craters)
            for exponent in args.size_exponent:
                matcher = RingMatcher(
                    thickness_frac=frac,
                    nms_iou=nms_iou,
                    min_score=0.1,
                    radius_scale=scale,
                    size_exponent=exponent,
                )
                preds = match_all(val, val_maps, matcher, pool)
                curve = sweep(preds, val, val_craters, default_thresholds(preds["score"]))
                best = curve[curve["min_score"] == best_threshold(curve)].iloc[0]
                setting = {
                    "thickness_frac": frac,
                    "nms_iou": nms_iou,
                    "radius_scale": scale,
                    "size_exponent": exponent,
                }
                rows.append({**setting, **best.to_dict()})
                print(f"{setting}: F1 {best['f1']:.3f}", file=sys.stderr, flush=True)

    table = pd.DataFrame(rows).sort_values("f1", ascending=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    print(table.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
