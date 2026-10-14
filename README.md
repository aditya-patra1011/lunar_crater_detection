# Lunar Crater Detection

[![CI](https://github.com/aditya-patra1011/lunar_crater_detection/actions/workflows/ci.yml/badge.svg)](https://github.com/aditya-patra1011/lunar_crater_detection/actions/workflows/ci.yml)

Detect impact craters on the Moon from LRO/LOLA elevation data, evaluate the detectors on terrain they were not trained on, and serve the best model behind a small API.

## Status

Work in progress. The repository skeleton and tooling are in place; no models have been trained yet.

## Problem

Landing-site assessment, terrain-relative navigation and surface dating all need crater maps, and manual cataloguing does not scale to new regions or resolutions. Convolutional crater detectors already exist (DeepMoon is one example), so this project focuses on a reproducible pipeline, evaluation on held-out terrain, confidence estimates, and a deployable service with drift monitoring.

## Approach

1. Tile LOLA elevation data and rasterize crater labels from the Robbins catalog.
2. Split by geography (blocked regions with a buffer), not by random tiles.
3. Train a classical baseline, a U-Net, a multi-scale U-Net and an object detector.
4. Convert predictions to georeferenced craters (latitude, longitude, diameter).
5. Evaluate by crater size against the catalog, with a manual audit of false positives.
6. Export to ONNX, serve with FastAPI, and monitor input and prediction drift.

## Known limitations of the labels

The Robbins catalog is roughly complete above about 1-2 km, and its completeness cannot be checked without an independent database. A detection with no catalog match may be a real but unlabeled crater, so reported precision is a lower bound.

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

- [ ] Repository skeleton, CI and pre-commit
- [ ] Data download and data card
- [ ] Tiling and label rasterization
- [ ] Geographic split and evaluation protocol
- [ ] Classical baseline and evaluation harness
- [ ] U-Net baseline training
- [ ] Improved models and ensemble
- [ ] Post-processing, ONNX export and FastAPI service
- [ ] Drift monitoring, model card and final write-up
