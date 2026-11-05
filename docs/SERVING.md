# Serving

The API takes one elevation tile and returns its craters, with positions in tile pixels and on the Moon. It runs the networks with ONNX Runtime, so it does not need PyTorch. A dashboard at its home page explains the project and runs the model live.

## The served model

`unet_v1` for craters under 6 km, and the large-rim output of `ms_unet_v1` seed 0 for craters of 6 km and wider. Both networks are in one ONNX file.

This uses one seed for large craters, not the average of three in [BASELINES.md](BASELINES.md). On val each seed scored within 0.0005 F1 of the average overall, so one seed halves the work for no measurable loss. Seed 0 was taken as the first seed, though it also scored best of the three. The 16-32 km score varied between seeds (0.30 to 0.41 on val), since that bin has under 100 craters. Its threshold of 0.436 was chosen on val, and the model was run on test once:

| test | served (2 networks) | three-seed version (4 networks) |
| --- | --- | --- |
| F1, all | 0.539 | 0.540 |
| F1, over 8 km | 0.562 | 0.569 |

Resampling the 7 test regions, the difference is -0.001 to 0.000 overall and -0.024 to +0.006 over 8 km.

**Checks on the export.** ONNX and PyTorch rim probabilities differ by at most 1.5e-6. On 12 val tiles, the ONNX model reproduced all 266 PyTorch detections. Where a detection matches a catalog crater, its centre is a median 1.2 px (143 m) from the catalog's and its diameter a median 1.4% larger. Converting a catalog crater to tile pixels and back returns its latitude, longitude and diameter to within 1e-6 degrees and 1e-9 relative.

## Run it

Install the export and serving extras (or `requirements-dev.txt`), then export the model. Export needs the trained weights, described in [BASELINES.md](BASELINES.md):

```bash
pip install -e ".[export,serve]"
python -m lunar_crater.serving.export --main models/unet_v1/best.pt --large models/ms_unet_v1/seed0/best.pt --out models/crater_detector --min-score 0.4362
```

This writes `models/crater_detector.onnx` (63 MB) and `models/crater_detector.json` (settings and the ONNX file's SHA-256). Export refuses to finish if ONNX and PyTorch disagree by more than 0.001.

Two optional files sit beside the model. Both are built from saved runs of the served model on the tile set, so they need no network inference (see [Confidence](#confidence) and [Monitoring](#monitoring)):

```bash
python scripts/fit_confidence.py data/interim/tiles/v1 outputs/served_v1_val outputs/served_v1_test models/crater_detector.confidence.json --min-score 0.4362
python -m lunar_crater.monitoring.reference data/interim/tiles/v1 models/crater_detector.reference.json --min-score 0.4362 --run train=outputs/served_v1_train --run val=outputs/served_v1_val
```

Then start the API:

```bash
uvicorn --factory lunar_crater.serving.app:create_app
```

It loads `models/crater_detector.onnx`, or the path in the `LUNAR_MODEL` environment variable, and refuses to start if the file does not match its checksum. Startup takes a few seconds, because it runs one tile in advance so the first request is not slow.

## The dashboard

Open http://localhost:8000/ for a page that explains the problem, the approach and the results in plain language, with a chart of scores by crater size. Its "Try it" panel runs the model live on eight test tiles and draws three kinds of circle on the terrain: detections that match a catalog crater, detections not in the catalog, and catalog craters the model missed. A "Rim map" view shows what the networks see before circles are fitted. Visitors can also upload their own `.npy` tile.

The example tiles come from the tile set in `data/interim/tiles/v1`, or the folder in the `LUNAR_TILES` environment variable. Without it, the dashboard still works but offers only uploads. Interactive API docs are at http://localhost:8000/docs.

## Send a tile

A tile is a 256 x 256 `.npy` array of heights in metres, at 118.45 m/px, in an orthographic projection centred on the tile ([decision 0001](decisions/0001-orthographic-tiles.md)). To take one from the tile set:

```bash
python scripts/export_tile.py data/interim/tiles/v1 t000843 demo_tile.npy
```

It prints the tile centre (lat=-4.982013 lon=59.023744). Send the file with the centre:

```bash
curl -F tile=@demo_tile.npy -F lat=-4.982013 -F lon=59.023744 http://localhost:8000/detect
```

In PowerShell, write `curl.exe` instead of `curl`. Add `-F min_score=0.6` to keep only more confident craters. The response starts like this:

```json
{
  "model_id": "a137489e5b07",
  "min_score": 0.4362,
  "seconds": 1.64,
  "n_craters": 23,
  "craters": [
    {"lat": -5.1948, "lon": 58.7119, "diam_km": 6.615, "x_px": 48.0, "y_px": 182.0, "r_px": 27.92, "score": 1.0775, "confidence": 1.0}
  ]
}
```

`model_id` is the start of the ONNX file's checksum, so results can be traced to a model. `GET /health` returns it with the expected tile size and scale, and whether confidence and monitoring are loaded. `GET /samples` lists the dashboard's example tiles, and `GET /samples/{tile_id}` runs one and marks each detection against the catalog.

## Confidence

A score ranks detections but is not a probability: a score of 1.08 says nothing directly about how often such a detection is right. Each detection's `confidence` is the estimated chance that it matches a catalog crater under [the protocol](EVALUATION.md). It comes from two curves, one for craters under 6 km and one for 6 km and over, because the two come from different network outputs. Each curve was fitted with isotonic regression, which only assumes that a higher score is never less likely to be right, on the 9,889 val detections at the served threshold.

On the 7,797 test detections, not used for fitting:

| | val (fitted) | test |
| --- | --- | --- |
| calibration error (mean gap between predicted and observed rate, over 10 bins) | 0.006 | 0.014 |
| Brier score | 0.147 | 0.146 |
| Brier score with one rate for all detections | 0.191 | 0.195 |
| predicted / observed share, under 6 km | 0.749 / 0.749 | 0.751 / 0.744 |
| predicted / observed share, 6 km and over | 0.654 / 0.659 | 0.620 / 0.606 |

The worst test bin predicted 0.478 and observed 0.441. Like precision, confidence counts a real crater that the catalog lacks as wrong, so it understates the chance that a detection is a real crater. Without `crater_detector.confidence.json` the field is `null`.

## Monitoring

The model is only reliable on tiles like those it was tuned on. `GET /monitoring` reports whether the last 200 tiles sent to `/detect` still look like the reference tiles: 2,521 train and val tiles from 41 regions, run through the served model. The dashboard shows the same report.

It compares five measures of each tile:

| measure | what a shift suggests |
| --- | --- |
| `relief_m`: height range, 1st to 99th percentile | wrong units, or a different kind of terrain |
| `slope_deg`: median slope | wrong pixel scale, or smoothed data |
| `missing`: share of missing heights | gaps in the source data |
| `n_craters`: detections per tile | a change in the model's behaviour |
| `median_score`: median detection score | the model is less sure than usual |

For each measure, the population stability index (PSI) compares how the recent tiles spread across the reference's deciles. Regions of the Moon differ, so the usual rule of thumb (0.1 and 0.25) would flag normal terrain. Instead, each measure's alarm levels were learned by holding out 7 regions of the reference at a time and comparing 100 of their tiles with a reference built from the rest, 300 times: the 95th percentile of PSI is "watch" and the 99th is "drift". The thresholds are saved in the reference file. Half a tile is added to every decile of the recent tiles before comparing, so a decile left empty by chance in a small sample does not look like a large shift. With fewer than 100 recent tiles the status is "waiting".

Checks on held-out data:

- **All 429 test tiles:** every measure stable (PSI 0.003 to 0.056).
- **500 random windows of 100 test tiles:** 454 stable, 42 watch and 4 drift, all on `median_score`. This is close to the designed rates of 5% and 1%.
- **100 test tiles with heights multiplied by 3, as with wrong units:** drift on relief (PSI 1.24) and slope (2.37).
- **The same tiles smoothed over 9 x 9 pixels, like coarser data:** drift on `median_score` (0.34). The terrain measures did not flag it, but the model's scores did.

Only requests at the default threshold are recorded, since a custom `min_score` changes `n_craters` and `median_score`. The report is held in memory, so it restarts with the server. Without `crater_detector.reference.json` the status is "off".

## Limits

- **Tiles must match training.** Only 256 x 256 tiles at 118.45 m/px in this projection are valid. The API checks the shape, but it cannot check the scale or projection.
- **Bad input.** Files that are not numeric `.npy` arrays, have the wrong shape, are over about 0.5 MB, or have more than half their heights missing are refused with a 4xx error.
- **Speed.** About 1.6 s per tile on one laptop CPU (i7-1355U) on mains power. Most of that is ring matching, not the networks (about 0.25 s). On battery the same laptop took 10 to 40 s per tile. Monitoring adds about 0.003 s.
- **No authentication or rate limit.** It is meant to run locally.
