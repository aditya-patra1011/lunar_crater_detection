"""Convert detections from tile pixels to positions and sizes on the Moon."""

from __future__ import annotations

import numpy as np
import pandas as pd

from lunar_crater.data.projection import MOON_RADIUS_M, ortho_inverse
from lunar_crater.data.tiles import TileSpec


def to_moon_coords(found: pd.DataFrame, spec: TileSpec) -> pd.DataFrame:
    """Add lat and lon (degrees) and diam_km to detections with x_px, y_px and r_px.

    This inverts the projection used to cut the tile and the radius used to place catalog
    craters on it, so a catalog crater maps back to its own position and size.
    """
    x, y = spec.from_pixels(found["x_px"].to_numpy(), found["y_px"].to_numpy())
    lat, lon = ortho_inverse(x, y, spec.lat, spec.lon)
    sine = np.minimum(found["r_px"].to_numpy() * spec.scale_m / MOON_RADIUS_M, 1.0)
    return found.assign(lat=lat, lon=lon, diam_km=np.arcsin(sine) * MOON_RADIUS_M * 2 / 1000)
