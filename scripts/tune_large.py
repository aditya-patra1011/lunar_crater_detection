"""Choose what finds large craters, scoring each choice on every val tile.

Compares each model alone, the models averaged, and the averaged small craters joined
with large craters from the large-rim output or the classical filter, over a grid of
size cuts and score gains. Radius scales are fitted on train tiles. Test tiles are never
touched. The ring matching settings come from ``scripts/tune_unet.py``.

With ``--main-checkpoint`` the small craters come from that model's rim map instead of
the averaged one, and the given checkpoints supply only the large-rim output.

Usage::

    python scripts/tune_large.py data/interim/tiles/v1 a/best.pt,b/best.pt out.csv \\
        --thickness-frac 0.2 --nms-iou 0.1 --size-exponent 0.5
"""

from __future__ import annotations

import argparse
import itertools
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from lunar_crater.baselines.classical import fit_radius_scale
from lunar_crater.evaluation.harness import (
    best_threshold,
    default_thresholds,
    load_split,
    score,
    sweep,
)
from lunar_crater.models.detector import UNetDetector, load_model, predict_rims


def rim_maps(tiles_dir: Path, tiles: pd.DataFrame, checkpoint: str) -> np.ndarray:
    """Rim maps of one model, (n_tiles, channels, h, w)."""
    model = load_model(checkpoint)
    maps = []
    for t in tiles.itertuples():
        with np.load(tiles_dir / "tiles" / f"{t.tile_id}.npz") as z:
            maps.append(predict_rims([model], z["dem"]).astype(np.float32))
    return np.stack(maps)


def _parts(job):
    path, scale_m, prob, detector = job
    with np.load(path) as z:
        return detector.parts(prob, z["dem"], scale_m)


def _matched(job) -> pd.DataFrame:
    prob, detector = job
    return detector.matcher()(prob[0])


def tagged(frames, tiles: pd.DataFrame) -> pd.DataFrame:
    """One frame of detections from one frame per tile."""
    parts = [f.assign(tile_id=t) for f, t in zip(frames, tiles["tile_id"], strict=True)]
    return pd.concat(parts, ignore_index=True)


def tile_parts(tiles_dir, tiles, maps, detector, pool):
    jobs = [
        (tiles_dir / "tiles" / f"{t.tile_id}.npz", t.scale_m, prob, detector)
        for t, prob in zip(tiles.itertuples(), maps, strict=True)
    ]
    return list(pool.map(_parts, jobs, chunksize=4))


def matched_only(tiles, maps, detector, pool) -> pd.DataFrame:
    jobs = [(prob, detector) for prob in maps]
    return tagged(pool.map(_matched, jobs, chunksize=4), tiles)


def result(preds, tiles, craters, setting: dict) -> dict:
    """The setting with its best-threshold val metrics, overall and by size."""
    curve = sweep(preds, tiles, craters, default_thresholds(preds["score"]))
    threshold = best_threshold(curve)
    metrics = score(preds, tiles, craters, min_score=threshold)
    row = {**setting, "min_score": threshold}
    row.update(
        {"recall": metrics.loc["all", "recall"], "precision": metrics.loc["all", "precision"]}
    )
    row.update({f"f1 {name}": metrics.loc[name, "f1"] for name in metrics.index})
    print(f"{setting}: F1 {row['f1 all']:.3f}", file=sys.stderr, flush=True)
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("tiles_dir", type=Path)
    parser.add_argument("checkpoints", help="best.pt files joined by commas")
    parser.add_argument("out", type=Path, help="CSV of val results per setting")
    parser.add_argument("--main-checkpoint", help="take the main rim map from this model")
    parser.add_argument("--thickness-frac", type=float, default=0.2)
    parser.add_argument("--nms-iou", type=float, default=0.1)
    parser.add_argument("--size-exponent", type=float, default=0.5)
    parser.add_argument("--sources", nargs="+", default=["head", "classical"])
    parser.add_argument("--cuts-km", type=float, nargs="+", default=[6.0, 8.0])
    parser.add_argument("--gains", type=float, nargs="+", default=[1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    parser.add_argument("--train-tiles", type=int, default=200)
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

    paths = args.checkpoints.split(",")
    train_each = [rim_maps(args.tiles_dir, train, path) for path in paths]
    val_each = [rim_maps(args.tiles_dir, val, path) for path in paths]
    train_maps, val_maps = np.mean(train_each, axis=0), np.mean(val_each, axis=0)
    label = "averaged"
    if args.main_checkpoint:
        label = Path(args.main_checkpoint).parent.name
        train_maps[:, 0] = rim_maps(args.tiles_dir, train, args.main_checkpoint)[:, 0]
        val_maps[:, 0] = rim_maps(args.tiles_dir, val, args.main_checkpoint)[:, 0]
    print("rim maps ready", file=sys.stderr, flush=True)

    base = UNetDetector(
        thickness_frac=args.thickness_frac,
        nms_iou=args.nms_iou,
        size_exponent=args.size_exponent,
        min_score=0.1,
    )
    rows = []
    with ProcessPoolExecutor(args.workers) as pool:
        if args.main_checkpoint:
            named = [(label, train_maps, val_maps)]
        else:
            named = [
                (Path(p).parent.name, tr, va)
                for p, tr, va in zip(paths, train_each, val_each, strict=True)
            ]
            if len(paths) > 1:
                named.append((label, train_maps, val_maps))
        scale = None
        for name, train_m, val_m in named:
            scale = fit_radius_scale(matched_only(train, train_m, base, pool), train_craters)
            detector = replace(base, radius_scale=scale)
            preds = matched_only(val, val_m, detector, pool)
            setting = {"maps": name, "source": "none", "radius_scale": scale}
            rows.append(result(preds, val, val_craters, setting))
        base = replace(base, radius_scale=scale)

        for source, cut in itertools.product(args.sources, args.cuts_km):
            detector = replace(base, large_source=source, large_from_km=cut, large_min_score=0.03)
            large = tagged(
                (p[1] for p in tile_parts(args.tiles_dir, train, train_maps, detector, pool)), train
            )
            wide = train_craters[train_craters["diam_km"] >= cut]
            large_scale = fit_radius_scale(large, wide)
            detector = replace(detector, large_radius_scale=large_scale)
            parts = tile_parts(args.tiles_dir, val, val_maps, detector, pool)
            for gain in args.gains:
                gained = replace(detector, large_gain=gain)
                preds = tagged((gained.merge(*p) for p in parts), val)
                setting = {
                    "maps": label,
                    "source": source,
                    "large_from_km": cut,
                    "large_gain": gain,
                    "radius_scale": base.radius_scale,
                    "large_radius_scale": large_scale,
                }
                rows.append(result(preds, val, val_craters, setting))

    table = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    print(table.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
