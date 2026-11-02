# Data card

The raw datasets this project uses: what they contain, how they are stored, and what to watch out for. Download them with:

```bash
python -m lunar_crater.data.download --list
python -m lunar_crater.data.download robbins_2018 lola_ldem_64
```

Sizes and SHA-256 checksums are recorded in `src/lunar_crater/data/download.py`. All three datasets are U.S. government data without copyright restrictions. Cite the sources below.

## Robbins lunar crater database (`robbins_2018`)

| | |
| --- | --- |
| Role | Labels: crater positions and diameters |
| Source | [USGS Astrogeology](https://astrogeology.usgs.gov/search/map/moon_crater_database_v1_robbins), PDS4 bundle (zip, 96 MB) |
| File used | `lunar_crater_database_robbins_2018_bundle/data/lunar_crater_database_robbins_2018.csv` (238 MB) |
| Rows | 1,296,796 craters |
| Key columns | `CRATER_ID`, `LAT_CIRC_IMG`, `LON_CIRC_IMG` (degrees, longitude 0-360 E), `DIAM_CIRC_IMG` (km), plus ellipse fits and their uncertainties |
| Citation | Robbins, S. J. (2019). A new global database of lunar impact craters >1-2 km. *JGR Planets*, 124(4), 871-892. [doi:10.1029/2018JE005592](https://doi.org/10.1029/2018JE005592) |

The `_IMG` columns come from rims traced on imagery. The bundle's `document/` folder holds the archive description PDF, which defines every column.

## LOLA global DEM, 118 m (`lola_ldem_118m`)

| | |
| --- | --- |
| Role | Main model input |
| Source | [USGS Astrogeology](https://astrogeology.usgs.gov/search/map/moon_lro_lola_dem_118m), LOLA Science Team, 2014 |
| Format | BigTIFF, 8.5 GB, global simple cylindrical |
| Resolution | 256 px/deg, about 118 m/px at the equator |
| Values | Scaled integers. Read the scale and offset from the file's metadata, or convert with `gdal_translate -ot Float32 -unscale` |

## LOLA global DEM, 474 m (`lola_ldem_64`)

| | |
| --- | --- |
| Role | Small stand-in for building and testing the pipeline |
| Source | [PDS Geosciences Node](https://pds-geosciences.wustl.edu/missions/lro/lola.htm), LOLA GDR `ldem_64.img` + `ldem_64.lbl` (531 MB) |
| Format | Raw 16-bit little-endian signed integers, 11,520 lines x 23,040 samples, PDS3 label |
| Projection | Simple cylindrical, 64 px/deg (473.8 m/px), longitude 0-360 E, latitude 90 to -90 |
| Values | Height in metres = DN x 0.5, relative to a 1,737,400 m sphere |

At 474 m/px a 1 km crater spans about 2 pixels, so this DEM can't be used to evaluate small craters.

## Known issues

- **Incomplete labels.** The catalog is roughly complete above 1-2 km only. A detection with no catalog match may be a real crater, so measured precision is a lower bound.
- **Labels from imagery.** Rims were traced on images, not on topography. Some degraded craters are clear in images but faint in the DEM, so a few labels mark craters the model can barely see.
- **Projection stretch.** Simple cylindrical grids stretch east-west by 1/cos(latitude): 2x at 60 degrees. The tiler reprojects every tile to remove this (see [decision 0001](../docs/decisions/0001-orthographic-tiles.md)).
- **Track gaps.** LOLA samples along orbit tracks that are widely spaced near the equator, so low-latitude DEM pixels between tracks are interpolated and can show track artefacts.
- **Resolution floor.** At 118 m/px a 1 km crater is about 8.5 pixels across. Smaller craters are not reliably resolved, so the evaluation should set a minimum diameter.
