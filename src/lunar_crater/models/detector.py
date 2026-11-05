"""Crater detection with trained U-Nets: predict rim probabilities, then match rings."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, fields, replace
from functools import lru_cache

import numpy as np
import pandas as pd
import torch

from lunar_crater.baselines.classical import RadialSlopeDetector
from lunar_crater.models.unet import UNet
from lunar_crater.postprocess.peaks import greedy_nms
from lunar_crater.postprocess.rings import RingMatcher
from lunar_crater.training.data import normalize_dem


@lru_cache(maxsize=4)
def load_model(checkpoint: str, device: str = "cpu") -> UNet:
    """Load a ``best.pt`` written by ``lunar_crater.training.train_unet``."""
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model = UNet(**state["config"]["model"])
    model.load_state_dict(state["state_dict"])
    return model.to(device).eval()


def predict_rims(models: Sequence[UNet], dem: np.ndarray, device: str = "cpu") -> np.ndarray:
    """Rim probabilities for one tile, averaged over ``models``; shaped (channels, h, w)."""
    x = torch.from_numpy(normalize_dem(dem))[None, None].to(device)
    with torch.no_grad():
        probs = [torch.sigmoid(model(x))[0].cpu().numpy() for model in models]
    return np.mean(probs, axis=0).astype(np.float64)


@dataclass(frozen=True)
class UNetDetector:
    """Harness detector: U-Net rim maps followed by ring matching.

    ``checkpoint`` is the path to ``best.pt``, or several joined by commas to average
    their rim maps. Most other fields are passed to :class:`RingMatcher`. Models are
    loaded once per process.

    ``large_source`` picks what finds craters of ``large_from_km`` and wider. ``"none"``
    leaves them to the main rim map. ``"head"`` matches rings on the network's large-rim
    output, and ``"classical"`` runs the radial-slope filter. Their scores are multiplied
    by ``large_gain``, to share one threshold with the main map, and ``large_radius_scale``
    corrects their radii.
    """

    checkpoint: str = ""
    device: str = "cpu"
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

    def models(self) -> list[UNet]:
        if not self.checkpoint:
            raise ValueError("set checkpoint=<path to best.pt>")
        return [load_model(path, self.device) for path in self.checkpoint.split(",")]

    def __call__(self, dem: np.ndarray, scale_m: float) -> pd.DataFrame:
        return self.circles(predict_rims(self.models(), dem, self.device), dem, scale_m)

    def circles(self, prob: np.ndarray, dem: np.ndarray, scale_m: float) -> pd.DataFrame:
        """Circles from (channels, h, w) rim maps; ``dem`` is used by the classical source."""
        matcher = self.matcher()
        if self.large_source == "none":
            return matcher(prob[0])

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

        large = large.assign(score=large["score"] * self.large_gain)
        found = pd.concat([small, large], ignore_index=True)
        found = found.sort_values("score", ascending=False, ignore_index=True)
        return found.iloc[greedy_nms(found, self.nms_iou)].reset_index(drop=True)
