# 0002: Split tiles by geographic region, with a buffer

- Status: accepted
- Date: 2026-11-02

## Context

Tiles are drawn at random over the whole Moon (decision 0001). A random tile-level split would put neighbouring and even overlapping tiles into train and test. Those tiles share terrain and sometimes the same craters, so test scores would measure memorisation of places rather than detection on terrain the model has not seen. The project's aim is the second.

## Decision

1. **Cells.** The sphere is cut into 6 latitude bands of 30 degrees. Each band is divided into longitude cells in proportion to its area (3, 9, 12, 12, 9, 3), giving 48 cells whose areas agree to within about 10%. A cell spans roughly 900 km. Cells are near-equal in area, unlike a plain latitude-longitude grid, and avoid the thin polar slivers of equal-area bands.
2. **Assignment.** Each cell goes wholly to train, val or test at random (70/15/15 by cell count, seed 0). A tile belongs to the cell that contains its centre.
3. **Buffer.** A tile is dropped (split `buffer`) if its centre is closer than the tile diagonal plus 30 km to a tile of a higher-priority split. Priority is test, then val, then train, so evaluation regions are never trimmed. Tiles of different splits therefore never overlap and are at least 30 km apart edge to edge. Only 0.28% of catalogued craters (3,674 of 1.3M) are 30 km or larger, so very few craters can appear in tiles of two splits.

4. **Seed choice.** One random draw can leave test unrepresentative. Seed 0 on the 3,000-tile set gave test a near-side fraction of 0.33 against 0.49 overall, and 25% more craters per tile. `--search-seeds N` tries seeds 0..N-1 and keeps the one whose val and test are closest to the whole set. The score is the largest relative gap in near-side fraction, craters per tile and mean |latitude|. It uses tile metadata only, never model results, and is run once before any model exists. On the v1 tiles, seed 122 of 200 gives a worst gap of 11%, against 33% for seed 0:

   | split | tiles | cells | craters per tile | mean \|lat\| | near side |
   | --- | --- | --- | --- | --- | --- |
   | train | 2,071 | 34 | 31.8 | 33.3 | 0.49 |
   | val | 450 | 7 | 34.2 | 31.1 | 0.54 |
   | test | 429 | 7 | 30.5 | 31.6 | 0.44 |
   | buffer | 50 | – | 31.7 | 35.8 | 0.50 |

`python -m lunar_crater.data.split <tiles dir> --search-seeds 200` writes `split.csv`, `split_summary.csv` (tiles, craters and terrain mix per split) and `split_meta.json` (parameters, the seed used and its imbalance score).

## Consequences

- With 48 cells, test gets 7 cells (about 15% of the surface). Even the most balanced seed is only approximately representative: val has no northern high-latitude cell, for example. The summary file keeps this visible.
- Border tiles are lost to the buffer, mostly from train: 50 of 3,000 (1.7%) for v1, whose tiles average about 110 km apart. Denser tile sets will lose more, and the summary reports the share.
- The seed search is tied to a tile set. Regenerating tiles with different parameters means re-running the search, and results from different splits are not comparable.
- Large basins (over 30 km) can still reach tiles of two splits. They are rare, and a later evaluation can exclude craters wider than the buffer if needed.
- The split is a function of tile centres only, so regenerating tiles with the same seed and parameters reproduces it.
