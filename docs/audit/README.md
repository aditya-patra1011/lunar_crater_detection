# False positive audit

Precision counts every detection with no catalog match as wrong, but the catalog misses some real craters, so the true precision is higher. This audit estimates how much higher by checking a random sample of those detections by eye.

## Result

Judged on 2026-11-06: 12 crater, 31 not crater and 17 unsure. Of the 43 judged detections, 28% are real craters (95% interval 17% to 43%). Applied to all 2,066 unmatched detections, precision rises from 0.735 against the catalog to about **0.81** (0.78 to 0.85).

| diameter | crater | not crater | unsure |
| --- | --- | --- | --- |
| 1-2 km | 10 | 16 | 11 |
| 2-6 km | 2 | 6 | 3 |
| 6 km and over | 0 | 9 | 3 |

The real craters the catalog lacks are all small, as expected for a catalog that is less complete below 2 km. The judging was cautious: a 1-2 km crater is only 8 to 17 pixels across, and several of the small unsure detections show the shading of a bowl. So 28% is more likely too low than too high.

**Detections at the tile edge.** 22 of the 23 sampled detections of 2 km and wider run off the edge of their tile, and none of them was judged a crater. The model fits whole circles to rims cut off by the edge. This holds across all 7,797 test detections, so it is not an artefact of the sample. Catalog craters centred outside the tile take part in matching ([protocol](../EVALUATION.md)), so these are real errors:

| diameter | share matching the catalog, circle inside the tile | circle crosses the edge |
| --- | --- | --- |
| 1-2 km | 0.77 (4,145) | 0.59 (555) |
| 2-6 km | 0.85 (1,744) | 0.51 (823) |
| 6 km and over | 0.86 (152) | 0.50 (378) |

## The sample

60 detections drawn at random (seed 0) from the 2,066 that the served model made on test tiles with no catalog match, at its normal threshold. They are sorted by size, so most are small: unmatched detections are mostly 1-2 km. Each one is in `audit.csv`, and drawn on the terrain in `audit_page_1.png` to `audit_page_5.png`, as a dashed circle labelled with its `audit_id` and diameter.

## How to judge

For each `audit_id`, look at its picture and fill in the `verdict` column of `audit.csv` with one of:

- `crater`: there is a bowl or a rim under the circle, roughly the same size. It does not have to be perfectly centred.
- `not crater`: the circle sits on a slope, a ridge, a wall of a larger crater, or flat ground.
- `unsure`: you cannot tell. Unsure verdicts are left out of the estimate.

The terrain is lit from the upper left, so a bowl shows a dark upper-left inner wall and a bright lower-right inner wall. Use `notes` for anything worth recording. To see a detection in context, its `lat` and `lon` can be looked up in a lunar map viewer such as the LROC QuickMap.

## Turning verdicts into a result

```bash
python scripts/audit_sheet.py summarize data/interim/tiles/v1 outputs/served_v1_test docs/audit --min-score 0.4362
```

It reports the share of judged detections that are real craters, with a 95% interval, and the corrected precision: matched detections plus that share of the unmatched ones.

To make a fresh sample:

```bash
python scripts/audit_sheet.py prepare data/interim/tiles/v1 outputs/served_v1_test docs/audit --min-score 0.4362
```
