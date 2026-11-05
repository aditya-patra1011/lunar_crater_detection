# Progress log

One row per working day. Commit counts come from:

`git log --no-merges --date=short --pretty=format:'%ad' | sort | uniq -c`

| Date | Commits | What shipped |
| ---- | ------- | ------------ |
| 2026-10-14 | 7 | Package skeleton, packaging config, smoke tests, CI workflow, pre-commit, data README and model card stub |
| 2026-10-16 | 1 | File clean-up |
| 2026-11-01 | 11 | Skeleton rebuild: CI on `master` with a Python 3.11/3.14 matrix, coverage reporting, stray file removed. Resumable, checksum-verified downloader for the Robbins catalog and LOLA DEMs, plus the data card |
| 2026-11-02 | 21 | Orthographic DEM tiling with rasterized crater rims (decision 0001), Robbins catalog loader, DEM sampler for both LOLA rasters, sample tiles figure. 3,000-tile v1 set, geographic split with buffer and terrain-balanced seed (decision 0002), circle-IoU matching and the evaluation protocol. Classical radial-slope crater detector, evaluation harness with threshold sweep, baseline at test F1 0.369 |
| 2026-11-03 | 7 | U-Net, training script, rim targets and ring matching, Colab notebook and tile packer, U-Net in the harness |
| 2026-11-05 | 19 | U-Net trained on a Colab GPU and tuned on val: test F1 0.533 against 0.369 for the classical baseline. Size-aware ring matching, bootstrap comparison of runs, first model card. Two-output U-Net trained on three seeds; `unet_v1` plus its large-rim output reaches test F1 0.540, and 0.569 over 8 km against 0.351. Torch-free circle finding, detections converted to latitude, longitude and diameter, ONNX export and FastAPI app |
| 2026-11-06 | 2 | Served model checked against PyTorch and the catalog, startup warm-up, serving docs: two networks, test F1 0.539 |
