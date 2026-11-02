"""Read heights from a global simple cylindrical DEM at arbitrary points."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

from lunar_crater.data.projection import wrap_longitude


class DemRaster:
    """A global equirectangular DEM that can be sampled at any latitude and longitude.

    Heights are returned in metres relative to the reference sphere, with NaN where the
    raster has no data. Works with both the USGS GeoTIFF (centred on 0 E) and the PDS
    LDEM images (centred on 180 E).
    """

    def __init__(self, path: Path):
        self._src = rasterio.open(path)
        params = self._src.crs.to_dict() if self._src.crs else {}
        if params.get("proj") != "eqc" or params.get("lat_ts", 0) != 0:
            raise ValueError(f"{path}: expected an equirectangular raster, got {params}")
        transform = self._src.transform
        if transform.b or transform.d:
            raise ValueError(f"{path}: rotated rasters are not supported")

        self.radius = float(params.get("R") or params["a"])
        self.lon0 = float(params.get("lon_0", 0.0))
        self._x0, self._dx = transform.c, transform.a
        self._y0, self._dy = transform.f, transform.e
        self.width, self.height = self._src.width, self._src.height
        if not math.isclose(self._dx * self.width, 2 * math.pi * self.radius, rel_tol=1e-6):
            raise ValueError(f"{path}: expected a raster covering all longitudes")

        self._scale = self._src.scales[0]
        offset = self._src.offsets[0]
        # PDS labels give the reference radius as the offset; we want heights, not radii.
        self._offset = 0.0 if math.isclose(offset, self.radius) else offset
        self._nodata = self._src.nodata

    @property
    def metres_per_pixel(self) -> float:
        return abs(self._dx)

    def close(self) -> None:
        self._src.close()

    def __enter__(self) -> DemRaster:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def pixel_coords(self, lat, lon) -> tuple[np.ndarray, np.ndarray]:
        """Return fractional (row, col) positions, with pixel centres at whole numbers."""
        x = self.radius * np.radians(wrap_longitude(np.asarray(lon) - self.lon0))
        y = self.radius * np.radians(np.asarray(lat, dtype=float))
        col = (x - self._x0) / self._dx - 0.5
        row = (y - self._y0) / self._dy - 0.5
        return row, col

    def sample(self, lat, lon) -> np.ndarray:
        """Bilinearly interpolate heights (float32, metres) at the given points."""
        row, col = self.pixel_coords(lat, lon)
        r0 = max(math.floor(np.min(row)), 0)
        r1 = min(math.floor(np.max(row)) + 1, self.height - 1)

        # Unwrap columns around the first point so a window may straddle the antimeridian.
        ref = np.ravel(col)[0]
        col = ref + (col - ref + self.width / 2) % self.width - self.width / 2
        c_lo = math.floor(np.min(col))
        c_hi = math.floor(np.max(col)) + 1
        if c_hi - c_lo + 1 >= self.width:  # near a pole: every longitude is needed
            c_lo, c_hi = 0, self.width - 1
            col = col % self.width
        band = self._read(r0, r1, c_lo, c_hi)

        rr = np.clip(row - r0, 0, band.shape[0] - 1)
        i0 = np.floor(rr).astype(np.intp)
        i1 = np.minimum(i0 + 1, band.shape[0] - 1)
        fr = (rr - i0).astype(np.float32)

        cc = col - c_lo
        c0 = np.floor(cc).astype(np.intp)
        fc = (cc - c0).astype(np.float32)
        c1 = (c0 + 1) % band.shape[1]

        # Convert only the four neighbours of each point, not the whole (wide) band.
        def corner(i, c):
            values = band[i, c].astype(np.float32)
            if self._nodata is not None:
                values[band[i, c] == self._nodata] = np.nan
            return values

        top = corner(i0, c0) * (1 - fc) + corner(i0, c1) * fc
        bottom = corner(i1, c0) * (1 - fc) + corner(i1, c1) * fc
        heights = top * (1 - fr) + bottom * fr
        return (heights * self._scale + self._offset).astype(np.float32)

    def _read(self, r0: int, r1: int, c0: int, c1: int) -> np.ndarray:
        """Read rows r0..r1 and columns c0..c1 (inclusive); columns may run past either edge."""

        def window(a: int, b: int) -> np.ndarray:
            return self._src.read(1, window=Window(a, r0, b - a + 1, r1 - r0 + 1))

        if c0 < 0:
            return np.hstack([window(c0 + self.width, self.width - 1), window(0, c1)])
        if c1 >= self.width:
            return np.hstack([window(c0, self.width - 1), window(0, c1 - self.width)])
        return window(c0, c1)
