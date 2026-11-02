# Baselines

Scores are on tile set v1 with split seed 122, under the protocol in [EVALUATION.md](EVALUATION.md). Settings were tuned on train and val only, and each final setting was run on test once.

## Classical: radial-slope matched filter

`lunar_crater.baselines.classical.RadialSlopeDetector`, with no learning and numpy only.

**Method.** Inside a crater the ground rises towards the rim. For every candidate centre and radius (4 to 100 px, in steps of 8%), the filter averages the slope component pointing outward over an annulus from `inner × r` to `r`. A crater wall scores its slope, flat ground and planar tilt score zero, and a radius past the rim picks up the downhill outer flank and scores lower. Where the annulus is cut by the tile edge or missing data, the local plane is removed first, so edges do not create detections. Local maxima over position and radius are kept and thinned greedily by circle IoU. All scoring is done with FFT correlations, at about 2-3 s per tile on one core (0.47 s per tile with 6 workers).

**Tuning.**

1. **Radius scale, on train.** The slope peaks inside the rim, so raw radii are about 23% too small and most detections miss IoU 0.5. On 200 train tiles the median true/detected radius ratio is 1.30, for every annulus shape tried. Detections are reported at 1.3 × the filter radius. Before this, a 40-tile val check scored F1 0.20; with it, the val grid below scores 0.30 to 0.385.
2. **Shape, on 150 val tiles.** Grid over `inner` and `nms_iou`, each at its best score threshold (`scripts/tune_classical.py`):

   | inner | nms_iou 0.1 | 0.2 | 0.3 | 0.5 |
   | --- | --- | --- | --- | --- |
   | 0.15 | **0.385** | 0.374 | | |
   | 0.3 | 0.381 | 0.370 | 0.367 | 0.357 |
   | 0.5 | | 0.360 | 0.354 | 0.338 |
   | 0.7 | | 0.340 | 0.327 | 0.303 |

   The best setting sits at the edge of the grid, but the last step gained only 0.015 F1, so it was not pushed further.
3. **Threshold, on all val.** On the 450 val tiles, overall F1 peaks at a score of 0.11 (a mean wall slope of about 6 degrees).

**Results** (inner 0.15, nms_iou 0.1, radius_scale 1.3, min_score 0.11):

| diameter (km) | val recall | val precision | val F1 | test recall | test precision | test F1 | test craters |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1-2 | 0.371 | 0.375 | 0.373 | 0.375 | 0.336 | 0.355 | 9,153 |
| 2-4 | 0.383 | 0.482 | 0.427 | 0.395 | 0.371 | 0.382 | 2,787 |
| 4-8 | 0.529 | 0.447 | 0.485 | 0.531 | 0.388 | 0.448 | 830 |
| 8-16 | 0.532 | 0.594 | 0.561 | 0.534 | 0.398 | 0.456 | 223 |
| 16-32 | 0.316 | 0.561 | 0.405 | 0.426 | 0.537 | 0.476 | 68 |
| 32 and over | 0.040 | – | – | 0.000 | – | – | 28 |
| **all** | **0.388** | **0.411** | **0.399** | **0.391** | **0.350** | **0.369** | 13,089 |

![Detections on four val tiles](figures/classical_val.png)

**What it gets wrong.**

- **Degraded craters.** Old craters have shallow walls and soft rims, and many fall under the 6 degree threshold. Lowering the threshold to 0.05 finds 58% of val craters but at 16% precision.
- **The smallest sizes.** 1-2 km craters are 8-17 px across, where DEM noise and the 4 px minimum radius limit the filter. They are 70% of the test craters, so they dominate the overall score.
- **Large craters.** Craters over 32 km are mostly wider than the 100 px largest radius or cut by the tile edge, so the filter cannot see them. The tile size, not the method, sets this limit.
- **Basin floors and slopes.** Curved terrain inside large basins can look like a wall over part of an annulus, which shows up as red dashed circles in the figure.
- **Precision is a lower bound.** Some unmatched detections are real craters missing from the catalog, as noted in the protocol.

Test precision is lower than on val (0.35 against 0.41) while recall is the same. Val was used to pick the threshold, so a small drop is expected. Any learned model has to beat **0.369 test F1** to justify itself.

## Reproduce

```bash
python scripts/tune_classical.py data/interim/tiles/v1 outputs/tune_classical.csv --workers 6
python -m lunar_crater.evaluation.harness data/interim/tiles/v1 --split val --workers 6 \
    --set inner=0.15 nms_iou=0.1 radius_scale=1.3 --out outputs/classical_v1_val
python -m lunar_crater.evaluation.harness data/interim/tiles/v1 --split test --workers 6 \
    --set inner=0.15 nms_iou=0.1 radius_scale=1.3 --min-score 0.11 --out outputs/classical_v1_test
python scripts/plot_detections.py data/interim/tiles/v1 outputs/classical_v1_val \
    docs/figures/classical_val.png
```

The harness refuses to run on test without `--min-score`, so a threshold cannot be tuned there by accident. Each run folder holds `predictions.csv`, `sweep.csv` (precision and recall at each threshold), `metrics.csv` and `run.json`.
