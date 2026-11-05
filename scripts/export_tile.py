"""Save one tile's heights as a .npy file to send to the API, and print its centre.

Usage::

    python scripts/export_tile.py data/interim/tiles/v1 t000843 demo_tile.npy
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("tiles_dir", type=Path)
    parser.add_argument("tile_id")
    parser.add_argument("out", type=Path)
    args = parser.parse_args()

    tile = pd.read_csv(args.tiles_dir / "tiles.csv").set_index("tile_id").loc[args.tile_id]
    with np.load(args.tiles_dir / "tiles" / f"{args.tile_id}.npz") as z:
        np.save(args.out, z["dem"])
    print(f"lat={tile['lat']:.6f} lon={tile['lon']:.6f}")


if __name__ == "__main__":
    main()
