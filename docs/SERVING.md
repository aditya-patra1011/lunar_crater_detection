# Serving

The API takes one elevation tile and returns its craters, with positions in tile pixels and on the Moon. It runs the networks with ONNX Runtime, so it does not need PyTorch.

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

This writes `models/crater_detector.onnx` (63 MB) and `models/crater_detector.json` (settings and the ONNX file's SHA-256). Export refuses to finish if ONNX and PyTorch disagree by more than 0.001. Then start the API:

```bash
uvicorn --factory lunar_crater.serving.app:create_app
```

It loads `models/crater_detector.onnx`, or the path in the `LUNAR_MODEL` environment variable, and refuses to start if the file does not match its checksum. Startup takes a few seconds, because it runs one tile in advance so the first request is not slow. Interactive docs are at http://localhost:8000/docs.

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
    {"lat": -5.1948, "lon": 58.7119, "diam_km": 6.615, "x_px": 48.0, "y_px": 182.0, "r_px": 27.92, "score": 1.0775}
  ]
}
```

`model_id` is the start of the ONNX file's checksum, so results can be traced to a model. `GET /health` returns it with the expected tile size and scale.

## Limits

- **Tiles must match training.** Only 256 x 256 tiles at 118.45 m/px in this projection are valid. The API checks the shape, but it cannot check the scale or projection.
- **Bad input.** Files that are not numeric `.npy` arrays, have the wrong shape, are over about 0.5 MB, or have more than half their heights missing are refused with a 4xx error.
- **Speed.** About 1.6 s per tile on one laptop CPU (i7-1355U). Most of that is ring matching, not the networks (about 0.25 s).
- **No authentication or rate limit.** It is meant to run locally.
