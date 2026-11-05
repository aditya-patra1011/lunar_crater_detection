# Model card: U-Net plus large-rim model, with ring matching

This covers the best detector so far. `unet_v1` finds craters under 6 km, and `ms_unet_v1`, a two-output U-Net averaged over three seeds, finds those of 6 km and wider. Later models will update this card. Full results and method are in [docs/BASELINES.md](../docs/BASELINES.md).

## Intended use

Research and portfolio use: finding lunar craters of about 1-32 km in LOLA elevation data at 118 m/px, as a starting point for human review. It is not validated for landing-site or navigation decisions.

## Inputs and outputs

- **Input:** a 256 x 256 tile of heights at 118.45 m/px, reprojected to a local orthographic grid ([decision 0001](../docs/decisions/0001-orthographic-tiles.md)).
- **Output:** circles in tile pixels (x, y, radius), each with a score. Use with min_score 0.435, the large_from_km 6 cut and large_gain 1.25.

## Training data

2,071 tiles from the Robbins catalog and the LOLA 118 m DEM ([data card](../data/DATA_CARD.md)), from 34 regions of the Moon. 450 val tiles from 7 other regions chose the epoch and post-processing. Labels are catalog rims of at least 1 km, and for the large-rim output, at least 6 km.

## Evaluation

On 429 test tiles from 7 further regions, under [the protocol](../docs/EVALUATION.md) (circle IoU of at least 0.5, tile set v1, split seed 122):

| diameter (km) | recall | precision | F1 |
| --- | --- | --- | --- |
| 1-2 | 0.382 | 0.746 | 0.505 |
| 2-4 | 0.534 | 0.748 | 0.623 |
| 4-8 | 0.530 | 0.648 | 0.583 |
| 8-16 | 0.605 | 0.667 | 0.635 |
| 16-32 | 0.279 | 1.000 | 0.437 |
| 32 and over | 0.000 | – | – |
| **all** | **0.426** | **0.735** | **0.540** |

The U-Net alone scores F1 0.533 and the classical baseline 0.369. Over 8 km this detector scores 0.569, against 0.351 and 0.443. The 16-32 km row rests on 20 detections and 68 craters, so it is uncertain.

## Known limitations

- It finds fewer than half of catalogued craters at this threshold. Recall on 1-2 km craters, 70% of test craters, is 0.38.
- Nothing over 32 km was found.
- Precision is a lower bound, because the catalog misses real craters. The false positive audit has not been done yet.
- The small-crater model comes from one training run. Three seeds of the two-output network agree closely on val, but that is a different network. There are only 7 test regions, and results by terrain type have not been measured.
- It runs four networks per tile. Whether the three-seed average beats a single seed for large craters was not measured.
- Tuning assumed 118 m/px; other resolutions are untested.

## Files

Four `best.pt` files (31 MB each) are not in this repository. They can be rebuilt with the [training notebook](../notebooks/train_unet_colab.ipynb).
