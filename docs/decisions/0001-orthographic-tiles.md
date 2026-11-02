# 0001: Resample DEM tiles into a local orthographic projection

- Status: accepted
- Date: 2026-11-02

## Context

Both LOLA DEMs are global simple cylindrical (equirectangular) grids. That projection stretches the east-west axis by 1/cos(latitude): by 2x at 60 degrees and without limit towards the poles. Cutting tiles straight from the grid would make the same crater look like a circle at the equator and a wide ellipse at high latitude. The model would then have to learn the distortion as well as the craters, and the labels would need to be ellipses.

Options considered:

1. **Cut tiles from the equirectangular grid, restricted to a latitude band** such as ±60 degrees. This is simple, but the stretch still reaches 2x inside the band, and the polar regions are lost.
2. **Cut tiles from the equirectangular grid and draw the labels as stretched ellipses.** The labels are correct, but the input distortion remains and depends on latitude.
3. **Resample every tile into an orthographic projection centred on the tile** (the DeepMoon approach). Every tile has the same metres per pixel, and craters are circles at any latitude, poles included.

## Decision

Option 3. Each tile is an N x N grid with a fixed scale in metres per pixel, in an orthographic projection on a 1,737.4 km sphere centred on the tile. Each output pixel is inverse-projected to latitude and longitude, and the DEM is sampled there with bilinear interpolation. Crater centres from the Robbins catalog are forward-projected into the same grid. Their rims are drawn as rings of radius R·sin(r/R) / scale pixels, where r is the crater radius along the surface.

At the default size (256 px at 118.45 m/px, about 30 km across), orthographic distortion at the tile edge is under 0.01%, so craters are circles to within a fraction of a pixel.

## Consequences

- Tiles can be placed anywhere, so tile centres are drawn uniformly on the sphere and a later geographic split can use any region.
- Bilinear resampling slightly smooths the DEM. At the native scale this is under one source pixel.
- The pipeline does its own projection maths (a few lines of numpy) instead of a full warp per tile. Tests check the maths against known points.
- Labels and inputs share one coordinate system, so converting predictions back to latitude, longitude and diameter is the inverse of the same functions.
