# Data

Nothing in this folder is committed except this file and the `.gitkeep` placeholders. Raw data and processed outputs stay local.

## Sources

- LRO LOLA elevation data (PDS Geosciences Node): https://pds-geosciences.wustl.edu/missions/lro/lola.htm
- Robbins Moon Crater Database v1 (USGS Astrogeology): https://astrogeology.usgs.gov/search/map/moon_crater_database_v1_robbins

## Layout

- `raw/`: files exactly as downloaded
- `interim/`: tiles and intermediate rasters
- `processed/`: model-ready datasets and splits

The download script arrives in a later PR. Until then, place downloads in `raw/` by hand.
