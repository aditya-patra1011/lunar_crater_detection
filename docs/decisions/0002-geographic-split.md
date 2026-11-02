# 0002: Split tiles by geographic region, with a buffer

- Status: accepted
- Date: 2026-11-02

## Context

Tiles are drawn at random over the whole Moon (decision 0001). A random tile-level split would put neighbouring and even overlapping tiles into train and test. Those tiles share terrain and sometimes the same craters, so test scores would measure memorisation of places rather than detection on terrain the model has not seen. The project's aim is the second.

## Decision

1. **Cells.** The sphere is cut into 6 latitude bands of 30 degrees. Each band is divided into longitude cells in proportion to its area (3, 9, 12, 12, 9, 3), giving 48 cells whose areas agree to within about 10%. A cell spans roughly 900 km. Cells are near-equal in area, unlike a plain latitude-longitude grid, and avoid the thin polar slivers of equal-area bands.
2. **Assignment.** Each cell goes wholly to train, val or test at random (70/15/15 by cell count, seed 0). A tile belongs to the cell that contains its centre.
3. **Buffer.** A tile is dropped (split `buffer`) if its centre is closer than the tile diagonal plus 30 km to a tile of a higher-priority split. Priority is test, then val, then train, so evaluation regions are never trimmed. Tiles of different splits therefore never overlap and are at least 30 km apart edge to edge. Only 0.28% of catalogued craters (3,674 of 1.3M) are 30 km or larger, so very few craters can appear in tiles of two splits.

`python -m lunar_crater.data.split <tiles dir>` writes `split.csv`, `split_summary.csv` (tiles, craters and terrain mix per split) and `split_meta.json`.

## Consequences

- With 48 cells, test gets 7 cells (about 15% of the surface) spread over different latitudes and both hemispheres, but a single draw can still be unbalanced. `split_summary.csv` reports mean |latitude|, near-side fraction and crater density per split so an unbalanced draw is visible. Changing the seed is the fix, and must happen before any model is evaluated on test.
- Border tiles are lost to the buffer, mostly from train. The share depends on cell size and is reported in the summary.
- Large basins (over 30 km) can still reach tiles of two splits. They are rare, and a later evaluation can exclude craters wider than the buffer if needed.
- The split is a function of tile centres only, so regenerating tiles with the same seed and parameters reproduces it.
