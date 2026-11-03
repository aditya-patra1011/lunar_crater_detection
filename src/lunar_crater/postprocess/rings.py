"""Turn a rim probability map into circles by ring template matching.

For every candidate centre and radius the score is the mean rim probability on a thin
ring minus the mean inside it, each over the part inside the tile. A crater's interior
is mostly rim-free, while a small ring laid across another crater's rim has that rim
running through its middle, so the subtraction removes most such false circles. Peaks
over position and radius are kept and thinned greedily by circle IoU, as in the
classical detector.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd

from lunar_crater.postprocess.peaks import fast_size, find_circles


@dataclass(frozen=True)
class RingMatcher:
    """Find circles in a (h, w) rim probability map.

    Radii run from ``min_radius_px`` to ``max_radius_px`` in steps of ``radius_step``
    (a ratio). A ring is ``thickness_px`` wide, and the interior is everything closer
    to the centre than ``r - thickness_px``. Peaks need a score (ring mean minus interior
    mean) of at least ``min_score`` and at least ``min_coverage`` of the ring inside the
    tile.
    """

    min_radius_px: float = 4.0
    max_radius_px: float = 100.0
    radius_step: float = 1.05
    thickness_px: float = 2.0
    min_score: float = 0.2
    min_coverage: float = 0.5
    nms_iou: float = 0.3

    def radii(self) -> np.ndarray:
        n = int(
            np.floor(np.log(self.max_radius_px / self.min_radius_px) / np.log(self.radius_step))
        )
        return self.min_radius_px * self.radius_step ** np.arange(n + 1)

    def score_volume(self, prob: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Ring-minus-interior score and ring coverage for every radius, (n_radii, h, w)."""
        radii = self.radii()
        h, w = prob.shape
        pad = int(np.ceil(radii[-1] + self.thickness_px)) + 1
        shape = (fast_size(h + pad), fast_size(w + pad))
        fp = np.fft.rfft2(np.nan_to_num(prob), shape)
        fm = np.fft.rfft2(np.ones_like(prob), shape)

        scores = np.empty((len(radii), h, w))
        coverage = np.empty_like(scores)
        for i, r in enumerate(radii):
            ring, disc, total = _kernels(float(r), self.thickness_px, pad, shape)
            n_ring = np.maximum(np.fft.irfft2(fm * ring, shape)[:h, :w], 1e-9)
            n_disc = np.maximum(np.fft.irfft2(fm * disc, shape)[:h, :w], 1e-9)
            on_ring = np.fft.irfft2(fp * ring, shape)[:h, :w] / n_ring
            inside = np.fft.irfft2(fp * disc, shape)[:h, :w] / n_disc
            scores[i] = on_ring - inside
            coverage[i] = n_ring / total
        return scores, coverage

    def __call__(self, prob: np.ndarray) -> pd.DataFrame:
        """Circles in one map; returns x_px, y_px, r_px and score, best first."""
        radii = self.radii()
        scores, coverage = self.score_volume(prob)
        scores = np.where(coverage >= self.min_coverage, scores, -np.inf)
        return find_circles(scores, radii, self.min_score, self.nms_iou)


@lru_cache(maxsize=256)
def _kernels(r: float, thickness: float, pad: int, shape: tuple[int, int]):
    """FFTs of the ring and interior disc (symmetric, so correlation is convolution)."""
    d = np.arange(-pad, pad + 1)
    dy, dx = np.meshgrid(d, d, indexing="ij")
    dist = np.hypot(dx, dy)
    ring = (np.abs(dist - r) <= thickness / 2).astype(float)
    disc = (dist < max(r - thickness, 1.0)).astype(float)

    def fft(k: np.ndarray) -> np.ndarray:
        grid = np.zeros(shape)
        grid[: k.shape[0], : k.shape[1]] = k
        return np.fft.rfft2(np.roll(grid, (-pad, -pad), axis=(0, 1)))

    return fft(ring), fft(disc), float(ring.sum())
