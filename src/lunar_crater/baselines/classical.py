"""A classical crater detector for DEM tiles: a radial-slope matched filter.

Inside a crater the ground rises towards the rim, so on the inner wall the downhill
direction points at the centre. For a candidate centre c and radius r the detector
takes the mean of the slope component pointing away from c (uphill outward) over the
annulus between ``inner * r`` and ``r``. A bowl of radius r scores its wall slope, and
a radius larger than the crater picks up the downhill outer flank and scores lower. The
score is a slope (metres per metre), so the same threshold works at every size.

Over a whole annulus the outward directions cancel, so a tilted plane scores zero. Where
the annulus is cut by the tile edge or missing data they do not, so the mean gradient
over the visible annulus, projected on the mean outward direction, is subtracted. That
removes any planar tilt exactly and changes nothing when the annulus is whole.

Every candidate (c, r) is scored with FFT correlations, local maxima over position and
radius are kept, and overlapping detections are thinned greedily by circle IoU.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd

from lunar_crater.evaluation.matching import circle_iou


@dataclass(frozen=True)
class RadialSlopeDetector:
    """Detect craters as peaks of mean outward wall slope.

    ``min_radius_px`` and ``max_radius_px`` bound the radii tried, spaced by
    ``radius_step`` (a ratio). ``inner`` sets the annulus' inner edge as a fraction of
    the radius. Peaks need a score of at least ``min_score`` (a slope; 0.05 is about 3
    degrees) and at least ``min_coverage`` of the annulus inside the tile and free of
    missing data. Detections overlapping a stronger one by more than ``nms_iou`` are
    dropped. Reported radii are the filter radius times ``radius_scale``, which corrects
    for the wall slope peaking inside the rim; it is fitted on train tiles
    (``fit_radius_scale``).
    """

    min_radius_px: float = 4.0
    max_radius_px: float = 100.0
    radius_step: float = 1.08
    inner: float = 0.5
    min_score: float = 0.05
    min_coverage: float = 0.6
    nms_iou: float = 0.3
    radius_scale: float = 1.0

    def radii(self) -> np.ndarray:
        n = int(
            np.floor(np.log(self.max_radius_px / self.min_radius_px) / np.log(self.radius_step))
        )
        return self.min_radius_px * self.radius_step ** np.arange(n + 1)

    def score_volume(self, dem: np.ndarray, scale_m: float) -> tuple[np.ndarray, np.ndarray]:
        """Scores and annulus coverage for every radius, each shaped (n_radii, h, w)."""
        valid = np.isfinite(dem)
        filled = np.where(valid, dem, np.nanmean(dem) if valid.any() else 0.0)
        gy, gx = np.gradient(filled, scale_m)
        # A gradient touching a missing pixel is unreliable, so drop it.
        ok = valid & np.roll(valid, 1, 0) & np.roll(valid, -1, 0)
        ok &= np.roll(valid, 1, 1) & np.roll(valid, -1, 1)
        gx, gy = np.where(ok, gx, 0.0), np.where(ok, gy, 0.0)

        radii = self.radii()
        pad = int(np.ceil(radii[-1])) + 1
        h, w = dem.shape
        shape = (_fast_size(h + pad), _fast_size(w + pad))
        fx, fy, fm = (np.fft.rfft2(a, shape) for a in (gx, gy, ok.astype(float)))

        scores = np.empty((len(radii), h, w))
        coverage = np.empty_like(scores)

        def corr(f, k, sign=1.0):
            return sign * np.fft.irfft2(f * k, shape)[:h, :w]

        for i, r in enumerate(radii):
            kx, ky, km, total = _kernels(float(r), self.inner, pad, shape)
            # Correlation is convolution with the flipped kernel; the outward unit
            # vector changes sign under the flip (sign=-1), the annulus does not.
            n = np.maximum(corr(fm, km), 1e-9)
            g_dot_u = corr(fx * kx + fy * ky, 1.0, -1.0)
            plane = corr(fx, km) * corr(fm, kx, -1.0) + corr(fy, km) * corr(fm, ky, -1.0)
            coverage[i] = n / total
            scores[i] = (g_dot_u - plane / n) / n
        return scores, coverage

    def __call__(self, dem: np.ndarray, scale_m: float) -> pd.DataFrame:
        """Detect craters in one tile; returns x_px, y_px, r_px and score."""
        radii = self.radii()
        scores, coverage = self.score_volume(dem, scale_m)
        scores = np.where(coverage >= self.min_coverage, scores, -np.inf)
        k, y, x = np.nonzero(_local_maxima(scores) & (scores >= self.min_score))
        found = pd.DataFrame(
            {
                "x_px": x.astype(float),
                "y_px": y.astype(float),
                "r_px": radii[k] * self.radius_scale,
                "score": scores[k, y, x],
            }
        )
        found = found.sort_values("score", ascending=False, ignore_index=True)
        return found.iloc[_greedy_nms(found, self.nms_iou)].reset_index(drop=True)


def fit_radius_scale(pred: pd.DataFrame, truth: pd.DataFrame, min_iou: float = 0.3) -> float:
    """Median ratio of true to detected radius over loosely matched pairs.

    ``pred`` and ``truth`` hold x_px, y_px, r_px for the same tile (or for many tiles,
    with a ``tile_id`` column in both). Each true crater is paired with the detection of
    highest IoU, and pairs with IoU of at least ``min_iou`` vote.
    """
    if "tile_id" not in truth:
        pred, truth = pred.assign(tile_id=0), truth.assign(tile_id=0)
    ratios = []
    for tile_id, t in truth.groupby("tile_id"):
        q = pred[pred["tile_id"] == tile_id]
        if q.empty:
            continue
        cols = ["x_px", "y_px", "r_px"]
        iou = circle_iou(*t[cols].to_numpy().T[:, :, None], *q[cols].to_numpy().T[:, None, :])
        best = iou.argmax(axis=1)
        ok = iou.max(axis=1) >= min_iou
        ratios.append(t["r_px"].to_numpy()[ok] / q["r_px"].to_numpy()[best[ok]])
    ratios = np.concatenate(ratios) if ratios else np.array([])
    if ratios.size == 0:
        raise ValueError("no detection overlaps a true crater; cannot fit a radius scale")
    return float(np.median(ratios))


@lru_cache(maxsize=256)
def _kernels(r: float, inner: float, pad: int, shape: tuple[int, int]):
    """FFTs of the annulus kernels, centred at the origin of a ``shape`` grid."""
    d = np.arange(-pad, pad + 1)
    dy, dx = np.meshgrid(d, d, indexing="ij")
    dist = np.hypot(dx, dy)
    ring = (dist >= inner * r) & (dist <= r) & (dist > 0)
    ux = np.where(ring, dx / np.where(dist > 0, dist, 1), 0.0)
    uy = np.where(ring, dy / np.where(dist > 0, dist, 1), 0.0)

    def fft(k: np.ndarray) -> np.ndarray:
        grid = np.zeros(shape)
        grid[: k.shape[0], : k.shape[1]] = k
        grid = np.roll(grid, (-pad, -pad), axis=(0, 1))
        return np.fft.rfft2(grid)

    return fft(ux), fft(uy), fft(ring.astype(float)), float(ring.sum())


def _fast_size(n: int) -> int:
    """The smallest size >= n with no prime factor above 5, which FFTs handle fast."""
    while True:
        m = n
        for p in (2, 3, 5):
            while m % p == 0:
                m //= p
        if m == 1:
            return n
        n += 1


def _local_maxima(volume: np.ndarray) -> np.ndarray:
    """True where a value is the maximum of its 3x3x3 neighbourhood."""
    peak = volume
    for axis in range(volume.ndim):
        padded = np.pad(
            peak,
            [(1, 1) if a == axis else (0, 0) for a in range(volume.ndim)],
            constant_values=-np.inf,
        )
        n = volume.shape[axis]
        peak = np.maximum(
            np.maximum(padded.take(range(0, n), axis), padded.take(range(1, n + 1), axis)),
            padded.take(range(2, n + 2), axis),
        )
    return (volume == peak) & np.isfinite(volume)


def _greedy_nms(found: pd.DataFrame, max_iou: float) -> list[int]:
    """Indices of rows (sorted best first) that overlap no better row by more than max_iou."""
    x, y, r = (found[c].to_numpy() for c in ("x_px", "y_px", "r_px"))
    keep: list[int] = []
    for i in range(len(found)):
        if not keep or circle_iou(x[i], y[i], r[i], x[keep], y[keep], r[keep]).max() <= max_iou:
            keep.append(i)
    return keep
