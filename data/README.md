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
