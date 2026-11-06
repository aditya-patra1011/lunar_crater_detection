# False positive audit

Precision counts every detection with no catalog match as wrong, but the catalog misses some real craters, so the true precision is higher. This audit estimates how much higher by checking a random sample of those detections by eye.

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
