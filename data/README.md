# Data

Nothing in this folder is committed except this file, [DATA_CARD.md](DATA_CARD.md) and the `.gitkeep` placeholders. Raw data and processed outputs stay local.

## Sources

- LRO LOLA elevation data (USGS Astrogeology and the PDS Geosciences Node)
- Robbins lunar crater database (USGS Astrogeology)

[DATA_CARD.md](DATA_CARD.md) describes each dataset, with links, formats and known issues.

## Download

From the repository root:

```bash
python -m lunar_crater.data.download --list
python -m lunar_crater.data.download robbins_2018 lola_ldem_64
python -m lunar_crater.data.download lola_ldem_118m   # 8.5 GB
```

Files go to `raw/` and are checked against their recorded size and SHA-256. An interrupted download resumes when you run the same command again.

## Layout

- `raw/`: files exactly as downloaded
- `interim/`: tiles and intermediate rasters
- `processed/`: model-ready datasets and splits

## Tiles

```bash
python -m lunar_crater.data.tiles --count 1000 --out data/interim/tiles/v1
```

This cuts 256 x 256 tiles at 118 m/px from the 118 m DEM. Tile centres are drawn uniformly over the sphere with a fixed seed, and each tile is reprojected to a local orthographic grid ([decision 0001](../docs/decisions/0001-orthographic-tiles.md)). Generation takes about 0.4 s per tile. Each output folder contains:

- `tiles/<tile_id>.npz`: `dem` (float32 heights in metres, NaN where no data) and `mask` (uint8, 1 on crater rims)
- `tiles.csv`: one row per tile with its centre, size, scale, number of craters centred in it, and fraction of missing pixels
- `craters.csv`: every catalog crater whose rim crosses a tile, with its pixel centre and radius and whether its centre lies inside the tile
- `meta.json`: the arguments and package versions used

To check labels by eye, run `python scripts/plot_tiles.py data/interim/tiles/v1 out.png`, which needs `matplotlib` from `requirements-dev.txt`.
