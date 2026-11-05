"""Turn rim probability maps into circles, in numpy (no torch needed)."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace

import numpy as np
import pandas as pd

from lunar_crater.baselines.classical import RadialSlopeDetector
from lunar_crater.postprocess.peaks import greedy_nms
from lunar_crater.postprocess.rings import RingMatcher


@dataclass(frozen=True)
class RimCircleFinder:
    """Ring matching on rim maps, with large craters optionally handled apart.

    Most fields are passed to :class:`RingMatcher`. ``large_source`` picks what finds
    craters of ``large_from_km`` and wider. ``"none"`` leaves them to the main rim map.
    ``"head"`` matches rings on the large-rim output, and ``"classical"`` runs the
    radial-slope filter. Their scores are multiplied by ``large_gain``, to share one
    threshold with the main map, and ``large_radius_scale`` corrects their radii.
    """

    min_radius_px: float = 4.0
    max_radius_px: float = 100.0
    radius_step: float = 1.05
    thickness_px: float = 2.0
    thickness_frac: float = 0.0
    min_score: float = 0.1
    min_coverage: float = 0.5
    nms_iou: float = 0.3
    radius_scale: float = 1.0
    size_exponent: float = 0.0
    large_source: str = "none"
    large_from_km: float = 6.0
    large_gain: float = 1.0
    large_radius_scale: float = 1.0
    large_min_score: float = 0.1

    def matcher(self) -> RingMatcher:
        names = {f.name for f in fields(RingMatcher)}
        return RingMatcher(**{n: getattr(self, n) for n in names})

    def circles(self, prob: np.ndarray, dem: np.ndarray, scale_m: float) -> pd.DataFrame:
        """Circles from (channels, h, w) rim maps; ``dem`` is used by the classical source."""
        if self.large_source == "none":
            return self.matcher()(prob[0])
        return self.merge(*self.parts(prob, dem, scale_m))

    def parts(self, prob: np.ndarray, dem: np.ndarray, scale_m: float):
        """Circles below and above ``large_from_km``, before they are merged."""
        matcher = self.matcher()
        cut_px = self.large_from_km * 1000 / 2 / scale_m
        small = replace(matcher, max_radius_px=cut_px / self.radius_scale)(prob[0])
        if self.large_source == "head":
            if len(prob) < 2:
                raise ValueError("large_source=head needs a model with a large-rim output")
            large = replace(
                matcher,
                min_radius_px=cut_px / self.large_radius_scale,
                min_score=self.large_min_score,
                radius_scale=self.large_radius_scale,
                size_exponent=0.0,
            )(prob[1])
        elif self.large_source == "classical":
            large = RadialSlopeDetector(
                min_radius_px=cut_px / self.large_radius_scale,
                max_radius_px=self.max_radius_px,
                inner=0.15,
                nms_iou=0.1,
                min_score=self.large_min_score,
                radius_scale=self.large_radius_scale,
            )(dem, scale_m)
        else:
            raise ValueError(f"unknown large_source {self.large_source!r}")
        return small, large

    def merge(self, small: pd.DataFrame, large: pd.DataFrame) -> pd.DataFrame:
        """Put both sizes on one score scale, using ``large_gain``, and thin overlaps."""
        large = large.assign(score=large["score"] * self.large_gain)
        found = pd.concat([small, large], ignore_index=True)
        found = found.sort_values("score", ascending=False, ignore_index=True)
        return found.iloc[greedy_nms(found, self.nms_iou)].reset_index(drop=True)
