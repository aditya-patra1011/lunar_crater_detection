"""Orthographic projection on a spherical Moon.

Angles are in degrees and distances in metres. ``x`` points east and ``y`` points
north in the tangent plane at the projection centre (see docs/decisions/0001).
"""

from __future__ import annotations

import numpy as np

MOON_RADIUS_M = 1_737_400.0


def wrap_longitude(lon):
    """Wrap longitudes to [-180, 180)."""
    return (np.asarray(lon, dtype=float) + 180.0) % 360.0 - 180.0


def to_unit_vectors(lat, lon) -> np.ndarray:
    """Return an (..., 3) array of unit vectors for the given latitudes and longitudes."""
    lat_r, lon_r = np.radians(lat), np.radians(lon)
    cos_lat = np.cos(lat_r)
    return np.stack([cos_lat * np.cos(lon_r), cos_lat * np.sin(lon_r), np.sin(lat_r)], axis=-1)


def ortho_forward(lat, lon, lat0: float, lon0: float, radius: float = MOON_RADIUS_M):
    """Project points onto the plane tangent at (lat0, lon0).

    Returns ``(x, y, visible)``; ``visible`` is False on the far hemisphere, where the
    projection folds back onto the near side.
    """
    phi, dlam = np.radians(lat), np.radians(np.asarray(lon, dtype=float) - lon0)
    phi0 = np.radians(lat0)
    x = radius * np.cos(phi) * np.sin(dlam)
    y = radius * (np.cos(phi0) * np.sin(phi) - np.sin(phi0) * np.cos(phi) * np.cos(dlam))
    cos_c = np.sin(phi0) * np.sin(phi) + np.cos(phi0) * np.cos(phi) * np.cos(dlam)
    return x, y, cos_c > 0


def ortho_inverse(x, y, lat0: float, lon0: float, radius: float = MOON_RADIUS_M):
    """Map plane coordinates back to ``(lat, lon)``; lon is wrapped to [-180, 180)."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    phi0 = np.radians(lat0)
    rho = np.hypot(x, y)
    c = np.arcsin(np.clip(rho / radius, -1.0, 1.0))
    sin_c, cos_c = np.sin(c), np.cos(c)
    with np.errstate(invalid="ignore", divide="ignore"):
        lat = np.where(
            rho == 0,
            phi0,
            np.arcsin(cos_c * np.sin(phi0) + y * sin_c * np.cos(phi0) / rho),
        )
    lon = np.radians(lon0) + np.arctan2(
        x * sin_c, rho * cos_c * np.cos(phi0) - y * sin_c * np.sin(phi0)
    )
    return np.degrees(lat), wrap_longitude(np.degrees(lon))
