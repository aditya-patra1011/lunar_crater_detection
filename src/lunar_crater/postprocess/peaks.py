"""Shared helpers for detectors that score every (radius, y, x) and keep the peaks."""

from __future__ import annotations

import numpy as np
import pandas as pd

from lunar_crater.evaluation.matching import circle_iou


def find_circles(
    scores: np.ndarray, radii: np.ndarray, min_score: float, nms_iou: float
) -> pd.DataFrame:
    """Circles at the local maxima of a (n_radii, h, w) score volume.

    Keeps maxima of their 3x3x3 neighbourhood scoring at least ``min_score``, then drops
    any that overlap a better one by more than ``nms_iou``. Returns x_px, y_px, r_px and
    score, best first. ``radii`` gives the radius reported for each slice.
    """
    k, y, x = np.nonzero(local_maxima(scores) & (scores >= min_score))
    found = pd.DataFrame(
        {
            "x_px": x.astype(float),
            "y_px": y.astype(float),
            "r_px": np.asarray(radii, dtype=float)[k],
            "score": scores[k, y, x],
        }
    )
    found = found.sort_values("score", ascending=False, ignore_index=True)
    return found.iloc[greedy_nms(found, nms_iou)].reset_index(drop=True)


def fast_size(n: int) -> int:
    """The smallest size >= n with no prime factor above 5, which FFTs handle fast."""
    while True:
        m = n
        for p in (2, 3, 5):
            while m % p == 0:
                m //= p
        if m == 1:
            return n
        n += 1


def local_maxima(volume: np.ndarray) -> np.ndarray:
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


def greedy_nms(found: pd.DataFrame, max_iou: float) -> list[int]:
    """Indices of rows (sorted best first) that overlap no better row by more than max_iou."""
    x, y, r = (found[c].to_numpy() for c in ("x_px", "y_px", "r_px"))
    keep: list[int] = []
    for i in range(len(found)):
        if not keep or circle_iou(x[i], y[i], r[i], x[keep], y[keep], r[keep]).max() <= max_iou:
            keep.append(i)
    return keep
