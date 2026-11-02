"""Plot detections against catalog rims on a few tiles, coloured by match.

Usage::

    python scripts/plot_detections.py data/interim/tiles/v1 outputs/classical_v1_val \\
        docs/figures/classical_val.png --min-score 0.11
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402

from lunar_crater.evaluation.matching import match_circles  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("tiles_dir", type=Path)
    parser.add_argument("run_dir", type=Path, help="harness output folder")
    parser.add_argument("out", type=Path)
    parser.add_argument("--min-score", type=float, help="default: the run's threshold")
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    run = json.loads((args.run_dir / "run.json").read_text())
    min_score = run["min_score"] if args.min_score is None else args.min_score
    preds = pd.read_csv(args.run_dir / "predictions.csv")
    preds = preds[preds["score"] >= min_score]
    craters = pd.read_csv(args.tiles_dir / "craters.csv")
    tile_ids = pd.Series(preds["tile_id"].unique())
    picks = tile_ids.sample(min(args.count, len(tile_ids)), random_state=args.seed)

    fig, axes = plt.subplots(1, len(picks), figsize=(4 * len(picks), 4.4), squeeze=False)
    cols = ["x_px", "y_px", "r_px"]
    for ax, tile_id in zip(axes[0], picks, strict=True):
        with np.load(args.tiles_dir / "tiles" / f"{tile_id}.npz") as data:
            dem = data["dem"]
        p = preds[preds["tile_id"] == tile_id]
        t = craters[(craters["tile_id"] == tile_id) & (craters["diam_km"] >= 1.0)]
        pi, ti, _ = match_circles(p[cols].to_numpy(), t[cols].to_numpy())
        ax.imshow(dem, cmap="gray")
        for k, (x, y, r) in enumerate(t[cols].to_numpy()):
            colour = "deepskyblue" if k in ti else "gold"
            ax.add_patch(Circle((x, y), r, fill=False, lw=1.0, ec=colour))
        for k, (x, y, r) in enumerate(p[cols].to_numpy()):
            if k not in pi:
                ax.add_patch(Circle((x, y), r, fill=False, lw=1.0, ec="red", ls="--"))
        ax.set_xlim(-0.5, dem.shape[1] - 0.5)
        ax.set_ylim(dem.shape[0] - 0.5, -0.5)
        ax.set_title(f"{tile_id}: {len(pi)} matched", fontsize=10)
        ax.set_axis_off()

    fig.suptitle(
        "Catalog craters found (blue) and missed (yellow); unmatched detections (red, dashed)"
    )
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=80)


if __name__ == "__main__":
    main()
