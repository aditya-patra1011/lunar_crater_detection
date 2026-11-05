"""Run a crater detector over one split of a tile set and score it.

A detector is any callable ``detector(dem, scale_m) -> DataFrame`` with columns x_px,
y_px, r_px and score. The harness runs it on every tile of the split, scores the
detections with the protocol in docs/EVALUATION.md, and sweeps the score threshold so
detectors can be compared at their best operating point (chosen on val).

Usage::

    python -m lunar_crater.evaluation.harness data/interim/tiles/v1 --split val \\
        --out outputs/classical_val
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from lunar_crater.evaluation.matching import evaluate_tile, summarize

#: Detector classes by name, as "module:Class" so optional dependencies (torch for the
#: U-Net) are imported only when that detector is used.
DETECTORS = {
    "classical": "lunar_crater.baselines.classical:RadialSlopeDetector",
    "unet": "lunar_crater.models.detector:UNetDetector",
}


def detector_class(name: str):
    """The detector class registered under ``name``."""
    target = DETECTORS[name]
    if not isinstance(target, str):
        return target
    module, _, attr = target.partition(":")
    return getattr(importlib.import_module(module), attr)


PRED_COLUMNS = ["tile_id", "x_px", "y_px", "r_px", "score"]


def load_split(tiles_dir: Path, split: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The tiles of one split and the catalog craters on them."""
    tiles = pd.read_csv(tiles_dir / "tiles.csv")
    names = pd.read_csv(tiles_dir / "split.csv")
    tiles = tiles.merge(names[["tile_id", "split"]], on="tile_id")
    tiles = tiles[tiles["split"] == split].reset_index(drop=True)
    craters = pd.read_csv(tiles_dir / "craters.csv")
    return tiles, craters[craters["tile_id"].isin(tiles["tile_id"])].reset_index(drop=True)


def _detect_one(args) -> pd.DataFrame:
    path, tile_id, scale_m, detector = args
    with np.load(path) as z:
        found = detector(z["dem"], scale_m)
    return found.assign(tile_id=tile_id)[PRED_COLUMNS]


def detect_tiles(tiles_dir: Path, tiles: pd.DataFrame, detector, *, workers: int = 1):
    """Run ``detector`` on every tile; returns one row per detection."""
    jobs = [
        (tiles_dir / "tiles" / f"{t.tile_id}.npz", t.tile_id, t.scale_m, detector)
        for t in tiles.itertuples()
    ]
    if workers > 1:
        with ProcessPoolExecutor(workers) as pool:
            parts = list(pool.map(_detect_one, jobs, chunksize=4))
    else:
        parts = [_detect_one(job) for job in jobs]
    if not parts:
        return pd.DataFrame(columns=PRED_COLUMNS)
    return pd.concat(parts, ignore_index=True)


def score(
    preds: pd.DataFrame,
    tiles: pd.DataFrame,
    craters: pd.DataFrame,
    *,
    min_score: float = -np.inf,
    iou_threshold: float = 0.5,
) -> pd.DataFrame:
    """Metrics by diameter bin for detections scoring at least ``min_score``."""
    preds = preds[preds["score"] >= min_score]
    by_tile_pred = dict(tuple(preds.groupby("tile_id")))
    by_tile_true = dict(tuple(craters.groupby("tile_id")))
    empty_pred = preds.iloc[:0]
    empty_true = craters.iloc[:0]
    truth_parts, pred_parts = [], []
    for t in tiles.itertuples():
        truth_rows, pred_rows = evaluate_tile(
            by_tile_pred.get(t.tile_id, empty_pred),
            by_tile_true.get(t.tile_id, empty_true),
            size_px=t.size_px,
            scale_m=t.scale_m,
            iou_threshold=iou_threshold,
        )
        truth_parts.append(truth_rows)
        pred_parts.append(pred_rows)
    return summarize(pd.concat(truth_parts), pd.concat(pred_parts))


def sweep(preds, tiles, craters, thresholds, **kwargs) -> pd.DataFrame:
    """Overall recall, precision and F1 at each score threshold."""
    rows = []
    for th in thresholds:
        overall = score(preds, tiles, craters, min_score=th, **kwargs).loc["all"]
        rows.append({"min_score": th, **overall.to_dict()})
    return pd.DataFrame(rows)


def default_thresholds(scores: pd.Series, n: int = 20) -> list[float]:
    """``n`` thresholds from keeping every detection to the top 5%, then finer to the top 0.5%."""
    if scores.empty:
        return [-np.inf]
    steps = np.concatenate([np.linspace(0, 0.95, n), [0.96, 0.97, 0.98, 0.99, 0.995]])
    return np.unique(np.quantile(scores, steps)).tolist()


def best_threshold(curve: pd.DataFrame) -> float:
    """The threshold with the highest overall F1 (the lowest one if F1 is undefined)."""
    if curve["f1"].notna().any():
        return float(curve.loc[curve["f1"].idxmax(), "min_score"])
    return float(curve["min_score"].min())


def parse_params(detector_cls, pairs: list[str]) -> dict:
    """Turn ``name=value`` strings into keyword arguments typed like the detector's fields."""
    fields = {f.name: f for f in dataclasses.fields(detector_cls)}
    params = {}
    for pair in pairs:
        name, sep, value = pair.partition("=")
        if not sep or name not in fields:
            raise ValueError(f"expected name=value with name in {sorted(fields)}, got {pair!r}")
        params[name] = type(fields[name].default)(value)
    return params


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("tiles_dir", type=Path, help="folder with tiles.csv and split.csv")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--detector", default="classical", choices=sorted(DETECTORS))
    parser.add_argument(
        "--set",
        dest="params",
        nargs="*",
        default=[],
        metavar="NAME=VALUE",
        help="detector parameters, e.g. min_score=0.08",
    )
    parser.add_argument("--out", type=Path, required=True, help="folder for the results")
    parser.add_argument("--limit", type=int, help="score only the first N tiles")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="*",
        help="score thresholds to sweep (default: 20 steps over the scores)",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        help="score threshold for metrics.csv, chosen on val (default: best F1 on this split)",
    )
    args = parser.parse_args(argv)
    if args.split == "test" and args.min_score is None:
        parser.error("--split test needs --min-score, chosen on val")

    detector_cls = detector_class(args.detector)
    detector = detector_cls(**parse_params(detector_cls, args.params))
    tiles, craters = load_split(args.tiles_dir, args.split)
    if args.limit:
        tiles = tiles.head(args.limit)
        craters = craters[craters["tile_id"].isin(tiles["tile_id"])]

    start = time.perf_counter()
    preds = detect_tiles(args.tiles_dir, tiles, detector, workers=args.workers)
    seconds = time.perf_counter() - start

    thresholds = args.thresholds or default_thresholds(preds["score"])
    curve = sweep(preds, tiles, craters, thresholds)
    best = best_threshold(curve) if args.min_score is None else args.min_score
    metrics = score(preds, tiles, craters, min_score=best)

    args.out.mkdir(parents=True, exist_ok=True)
    preds.to_csv(args.out / "predictions.csv", index=False)
    curve.to_csv(args.out / "sweep.csv", index=False)
    metrics.to_csv(args.out / "metrics.csv")
    run = {
        "tiles_dir": str(args.tiles_dir),
        "split": args.split,
        "n_tiles": len(tiles),
        "detector": args.detector,
        "params": dataclasses.asdict(detector),
        "min_score": best,
        "min_score_from": "this split" if args.min_score is None else "given",
        "seconds_per_tile": seconds / max(len(tiles), 1),
    }
    (args.out / "run.json").write_text(json.dumps(run, indent=2) + "\n")
    print(f"min_score {best:.4g} on {len(tiles)} {args.split} tiles", file=sys.stderr)
    print(metrics.round(3).to_string(), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
