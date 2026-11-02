"""Map tile centres coloured by split, as a visual check of the geographic split.

Usage::

    python scripts/plot_split.py data/interim/tiles/v1 docs/figures/split_map.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

COLOURS = {"train": "#4C78A8", "val": "#F58518", "test": "#E45756", "buffer": "#BBBBBB"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("tiles_dir", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()

    tiles = pd.read_csv(args.tiles_dir / "tiles.csv").merge(
        pd.read_csv(args.tiles_dir / "split.csv"), on="tile_id"
    )
    fig, ax = plt.subplots(figsize=(10, 5.2))
    for name, colour in COLOURS.items():
        part = tiles[tiles["split"] == name]
        ax.scatter(part["lon"], part["lat"], s=6, c=colour, label=f"{name} ({len(part)})")
    ax.set(xlim=(-180, 180), ylim=(-90, 90), xlabel="longitude (°E)", ylabel="latitude (°)")
    ax.set_xticks(range(-180, 181, 60))
    ax.set_yticks(range(-90, 91, 30))
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", fontsize=9, markerscale=2, framealpha=0.9)
    ax.set_title("Tile centres by split (equirectangular view)")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=80)


if __name__ == "__main__":
    main()
