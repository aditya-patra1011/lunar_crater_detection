# Model card: U-Net rim segmenter with ring matching (unet_v1)

This covers the first trained model, a baseline. Later models will update it. Full results and method are in [docs/BASELINES.md](../docs/BASELINES.md).

## Intended use

Research and portfolio use: finding lunar craters of about 1-16 km in LOLA elevation data at 118 m/px, as a starting point for human review. It is not validated for landing-site or navigation decisions.

## Inputs and outputs

- **Input:** a 256 x 256 tile of heights at 118.45 m/px, reprojected to a local orthographic grid ([decision 0001](../docs/decisions/0001-orthographic-tiles.md)).
- **Output:** circles in tile pixels (x, y, radius), each with a score. Use with min_score 0.43.

## Training data

2,071 tiles from the Robbins catalog and the LOLA 118 m DEM ([data card](../data/DATA_CARD.md)), from 34 regions of the Moon. 450 val tiles from 7 other regions chose the epoch and post-processing. Labels are catalog rims of at least 1 km.

## Evaluation

On 429 test tiles from 7 further regions, under [the protocol](../docs/EVALUATION.md) (circle IoU of at least 0.5, tile set v1, split seed 122):

| diameter (km) | recall | precision | F1 |
| --- | --- | --- | --- |
| 1-2 | 0.384 | 0.742 | 0.506 |
| 2-4 | 0.537 | 0.738 | 0.621 |
| 4-8 | 0.510 | 0.645 | 0.569 |
| 8-16 | 0.368 | 0.411 | 0.388 |
| 16-32 | 0.294 | 0.337 | 0.314 |
| 32 and over | 0.000 | – | – |
| **all** | **0.423** | **0.718** | **0.533** |

The classical baseline scores F1 0.369 overall, but is better on craters over 8 km.

## Known limitations

- It finds fewer than half of catalogued craters at this threshold.
- Large craters are weak: false circles can form from arcs of smaller rims, and nothing over 32 km was found.
- Precision is a lower bound, because the catalog misses real craters. The false positive audit has not been done yet.
- The results come from one training run with one seed, and from only 7 test regions. Results by terrain type have not been measured.
- Tuning assumed 118 m/px; other resolutions are untested.

## Files

`best.pt` (31 MB) is not in this repository. It can be rebuilt with the [training notebook](../notebooks/train_unet_colab.ipynb).
