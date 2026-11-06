# Model card: U-Net plus large-rim model, with ring matching

This covers the served detector. `unet_v1` finds craters under 6 km, and seed 0 of `ms_unet_v1`, a two-output U-Net, finds those of 6 km and wider. Both run from one ONNX file ([docs/SERVING.md](../docs/SERVING.md)). Averaging three seeds for large craters scores almost the same, at twice the cost. Full results and method are in [docs/BASELINES.md](../docs/BASELINES.md).

## Intended use

Research and portfolio use: finding lunar craters of about 1-32 km in LOLA elevation data at 118 m/px, as a starting point for human review. It is not validated for landing-site or navigation decisions.

## Inputs and outputs

- **Input:** a 256 x 256 tile of heights at 118.45 m/px, reprojected to a local orthographic grid ([decision 0001](../docs/decisions/0001-orthographic-tiles.md)).
- **Output:** circles in tile pixels (x, y, radius) and on the Moon (latitude, longitude, diameter), each with a score and a confidence: the estimated chance that it matches a catalog crater. The default threshold, 0.436, was chosen on val; the other settings are in `crater_detector.json`.

## Training data

2,071 tiles from the Robbins catalog and the LOLA 118 m DEM ([data card](../data/DATA_CARD.md)), from 34 regions of the Moon. 450 val tiles from 7 other regions chose the epoch and post-processing. Labels are catalog rims of at least 1 km, and for the large-rim output, at least 6 km.

## Evaluation

On 429 test tiles from 7 further regions, under [the protocol](../docs/EVALUATION.md) (circle IoU of at least 0.5, tile set v1, split seed 122):

| diameter (km) | recall | precision | F1 |
| --- | --- | --- | --- |
| 1-2 | 0.381 | 0.747 | 0.505 |
| 2-4 | 0.533 | 0.752 | 0.624 |
| 4-8 | 0.530 | 0.640 | 0.580 |
| 8-16 | 0.605 | 0.637 | 0.621 |
| 16-32 | 0.294 | 1.000 | 0.455 |
| 32 and over | 0.000 | – | – |
| **all** | **0.426** | **0.735** | **0.539** |

Confidence was fitted on val detections. On test detections its calibration error is 0.014 and its Brier score 0.146, against 0.195 for one rate for all detections ([details](../docs/SERVING.md#confidence)).

The U-Net alone scores F1 0.533 and the classical baseline 0.369. Over 8 km this detector scores 0.562, against 0.351 and 0.443. The 16-32 km row rests on 21 detections and 68 craters, so it is uncertain.

## Known limitations

- It finds fewer than half of catalogued craters at this threshold. Recall on 1-2 km craters, 70% of test craters, is 0.38.
- Nothing over 32 km was found.
- Precision and confidence are lower bounds, because the catalog misses real craters. A sample of 60 unmatched test detections has been prepared for checking by eye ([docs/audit](../docs/audit/README.md)); it has not been judged yet.
- The small-crater model comes from one training run. Three seeds of the two-output network agree closely on val, but that is a different network. There are only 7 test regions, and results by terrain type have not been measured.
- It takes about 1.6 s per tile on a laptop CPU, mostly in ring matching.
- The service checks a tile's shape but not its scale or projection, so tiles must be prepared like the training tiles. Its drift monitor flags tiles whose heights, slopes or scores differ from the reference tiles ([details](../docs/SERVING.md#monitoring)), but it needs 100 tiles to judge and can miss smaller changes.
- Tuning assumed 118 m/px; other resolutions are untested.

## Files

`crater_detector.onnx` (63 MB), `crater_detector.confidence.json`, `crater_detector.reference.json` and the `best.pt` files are not in this repository. They can be rebuilt with the [training notebook](../notebooks/train_unet_colab.ipynb) and the commands in [docs/SERVING.md](../docs/SERVING.md).
