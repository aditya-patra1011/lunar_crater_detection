# Progress log

One row per working day. Commit counts come from:

`git log --no-merges --date=short --pretty=format:'%ad' | sort | uniq -c`

| Date | Commits | What shipped |
| ---- | ------- | ------------ |
| 2026-10-14 | 7 | Package skeleton, packaging config, smoke tests, CI workflow, pre-commit, data README and model card stub |
| 2026-10-16 | 1 | File clean-up |
| 2026-11-01 | 11 | Skeleton rebuild: CI on `master` with a Python 3.11/3.14 matrix, coverage reporting, stray file removed. Resumable, checksum-verified downloader for the Robbins catalog and LOLA DEMs, plus the data card |
| 2026-11-02 | 8 | Orthographic DEM tiling with rasterized crater rims (decision 0001), Robbins catalog loader, DEM sampler for both LOLA rasters, sample tiles figure |
