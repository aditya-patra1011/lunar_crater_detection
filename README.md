# Lunar Crater Detection

[![CI](https://github.com/aditya-patra1011/lunar_crater_detection/actions/workflows/ci.yml/badge.svg)](https://github.com/aditya-patra1011/lunar_crater_detection/actions/workflows/ci.yml)

Detect impact craters on the Moon from LRO/LOLA elevation data, evaluate the detectors on terrain they were not trained on, and serve the best model behind a small API.

## Status

Complete. On held-out test terrain the served detector scores F1 0.539, against 0.369 for a classical baseline. It finds 43% of catalogued craters, and 74% of its detections match the catalog; checking a sample of the rest by eye puts the share that are real craters at about 81%. It runs from ONNX behind a FastAPI app that returns each crater's latitude, longitude, diameter and a calibrated confidence, watches incoming tiles for drift, and has a dashboard that explains the project and runs the model live.

## Results

On 429 test tiles from 7 regions not used for training or tuning, under [the evaluation protocol](docs/EVALUATION.md) (circle IoU of at least 0.5):

| detector | F1, all | F1, over 8 km |
| --- | --- | --- |
| classical radial-slope filter | 0.369 | 0.443 |
| U-Net | 0.533 | 0.351 |
| **served: U-Net plus a large-crater network** | **0.539** | **0.562** |

- **Learning beats hand-written rules** overall, but the U-Net alone stitched arcs of small craters into false large ones. A second output trained only on craters of 6 km and wider fixed that.
- **Small craters are the main gap.** Craters of 1-2 km are 70% of test craters, and 38% of them are found.
- **Precision is higher than the catalog says.** Of 60 unmatched detections checked by eye, 28% were real craters, all under 6 km, which puts precision near 0.81 instead of 0.735 ([audit](docs/audit/README.md)).
- **The tile edge causes most large false alarms.** Circles that run off the edge match the catalog about half as often as circles inside the tile.
- **Confidence is calibrated** on unseen data: on test detections it is off by 0.014 on average ([confidence](docs/SERVING.md#confidence)).
- **Drift monitoring** stays quiet on normal test tiles and flags tiles with heights in the wrong units or smoothed data ([monitoring](docs/SERVING.md#monitoring)).

## Documents

- [Data card](data/DATA_CARD.md) and [design decisions](docs/decisions)
- [Evaluation protocol](docs/EVALUATION.md)
- [Baselines and improved models](docs/BASELINES.md)
- [Serving, confidence and monitoring](docs/SERVING.md)
- [Model card](models/MODEL_CARD.md)
- [False positive audit](docs/audit/README.md)
- [Progress log](docs/PROGRESS.md)

## Problem

Landing-site assessment, terrain-relative navigation and surface dating all need crater maps, and manual cataloguing does not scale to new regions or resolutions. Convolutional crater detectors already exist (DeepMoon is one example), so this project focuses on a reproducible pipeline, evaluation on held-out terrain, confidence estimates, and a deployable service with drift monitoring.

## Approach

1. Tile LOLA elevation data and rasterize crater labels from the Robbins catalog.
2. Split by geography (blocked regions with a buffer), not by random tiles.
3. Train a classical baseline, a U-Net and a multi-scale U-Net.
4. Convert predictions to georeferenced craters (latitude, longitude, diameter).
5. Evaluate by crater size against the catalog, with a manual audit of false positives.
6. Export to ONNX, serve with FastAPI, and monitor input and prediction drift.

## Known limitations of the labels

The Robbins catalog is roughly complete above about 1-2 km, and its completeness cannot be checked without an independent database. A detection with no catalog match may be a real but unlabeled crater, so reported precision is a lower bound. A check of 60 such detections by eye found 28% were real craters ([audit](docs/audit/README.md)).

## Data

Data is not stored in this repository. See [data/README.md](data/README.md).

## Quickstart

```bash
python -m venv .venv
source .venv/Scripts/activate   # Git Bash on Windows; macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
pre-commit install
pytest
```

## Roadmap

- [x] Repository skeleton, CI and pre-commit
- [x] Data download and data card
- [x] Tiling and label rasterization
- [x] Geographic split and evaluation protocol
- [x] Classical baseline and evaluation harness
- [x] U-Net baseline training
- [x] Improved models and ensemble
- [x] Post-processing, ONNX export and FastAPI service
- [x] Drift monitoring, model card and final write-up

## Later development

Not needed to complete the project:

- Overlapping tiles, or dropping circles that run off the tile edge, to cut false large craters
- Object detector
- Retraining so the large-crater output does not weaken small craters
- Better recall on 1-2 km craters
- Finding craters over 32 km, which needs bigger tiles
