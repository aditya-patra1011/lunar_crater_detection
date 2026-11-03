"""Pack a tile set into one zip for training elsewhere (for example on Colab).

The zip holds the tile arrays, tiles.csv, craters.csv and the split files, under the
tile set's folder name. Tiles are already compressed, so they are stored as they are.

Usage::

    python scripts/pack_tiles.py data/interim/tiles/v1 lunar_tiles_v1.zip
"""

from __future__ import annotations

import argparse
import hashlib
import zipfile
from pathlib import Path

FILES = ("tiles.csv", "craters.csv", "split.csv", "meta.json", "split_meta.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("tiles_dir", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()

    name = args.tiles_dir.name
    missing = [f for f in FILES if not (args.tiles_dir / f).exists()]
    if missing:
        raise SystemExit(f"missing from {args.tiles_dir}: {', '.join(missing)}")
    tiles = sorted((args.tiles_dir / "tiles").glob("*.npz"))
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in FILES:
            zf.write(args.tiles_dir / f, f"{name}/{f}")
        for t in tiles:
            zf.write(t, f"{name}/tiles/{t.name}", compress_type=zipfile.ZIP_STORED)

    digest = hashlib.sha256(args.out.read_bytes()).hexdigest()
    size_mb = args.out.stat().st_size / 1e6
    print(f"{args.out}: {len(tiles)} tiles, {size_mb:.0f} MB, sha256 {digest}")


if __name__ == "__main__":
    main()
