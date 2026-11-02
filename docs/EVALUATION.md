# Evaluation protocol

This is fixed before any model is trained, so baselines and models are scored the same way. The code is `lunar_crater.evaluation.matching`.

## Data

- **Test set:** tiles with `split == "test"` in `split.csv` ([decision 0002](decisions/0002-geographic-split.md)). They are used once per final model, never for tuning. Tuning uses `val`.
- **Ground truth:** the Robbins catalog craters in `craters.csv` for each tile.

## Matching

A predicted crater and a catalog crater are circles in the tile's pixel grid. They **match** when their intersection-over-union (IoU) is at least **0.5**.

- IoU is scale-free, so the same rule works for a 1 km and a 100 km crater. For two circles of equal size, IoU 0.5 allows a centre offset of about 0.53 radii; for two concentric circles it allows a radius ratio of about 0.7.
- Matching is one-to-one and greedy, taking the highest IoU first. Two predictions on the same crater give one true positive and one false positive.

## What is counted

- **Catalog craters** count if their centre is inside the tile and their diameter is at least 1 km.
- **Predictions** count if their centre is inside the tile and their diameter is at least 1 km. Predicted radii in pixels convert to surface diameters by inverting the tiling formula.
- Every catalog crater whose rim crosses the tile, including those centred outside it, can take part in matching. A prediction that matches a crater centred just outside the tile is therefore neither a true positive nor a false positive. This means craters cut by the tile edge are not penalised either way.

## Metrics

All metrics are computed over the pooled crater counts from all test tiles, and reported by diameter bin: 1-2, 2-4, 4-8, 8-16, 16-32 and over 32 km, plus all sizes.

- **Recall**: matched catalog craters / catalog craters, binned by catalog diameter.
- **Precision**: matched predictions / predictions, binned by predicted diameter.
- **F1**: the harmonic mean of the two, per bin.

## Caveats

- **Precision is a lower bound.** The catalog is complete only above about 1-2 km, and rims were traced on imagery, so some unmatched predictions are real craters. A manual audit of a sample of false positives will estimate how many.
- **The smallest bins are the hardest.** At 118 m/px a 1-2 km crater spans 8-17 pixels, and the catalog is least complete there.
- **Results depend on the test regions.** Report the split seed with every result, and compare models only on the same split.
