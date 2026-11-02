"""Plot a few tiles with their crater rims overlaid, as a visual check of the labels.

Usage::

    python scripts/plot_tiles.py data/interim/tiles/v1 docs/figures/sample_tiles.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("tiles_dir", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    index = pd.read_csv(args.tiles_dir / "tiles.csv")
    picks = index.sample(args.count, random_state=args.seed)
    fig, axes = plt.subplots(2, args.count, figsize=(3.2 * args.count, 6.6), squeeze=False)
    for k, tile in enumerate(picks.itertuples()):
        with np.load(args.tiles_dir / "tiles" / f"{tile.tile_id}.npz") as data:
            dem, mask = data["dem"], data["mask"]
        shade = (dem - np.nanmin(dem)) / max(np.nanmax(dem) - np.nanmin(dem), 1e-6)
        overlay = plt.cm.gray(shade)[..., :3]
        overlay[mask == 1] = (0.9, 0.2, 0.1)

        axes[0, k].imshow(dem, cmap="gray")
        axes[0, k].set_title(f"{tile.lat:.1f}°, {tile.lon:.1f}°", fontsize=10)
        axes[1, k].imshow(overlay)
        axes[1, k].set_title(f"{tile.n_craters} craters", fontsize=10)
        for ax in axes[:, k]:
            ax.set_axis_off()

    fig.suptitle("LOLA 118 m tiles (top) with Robbins crater rims (bottom)")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=80)


if __name__ == "__main__":
    main()
