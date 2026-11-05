"""Crater detection with a trained U-Net: predict rim probabilities, then match rings."""

from __future__ import annotations

from dataclasses import dataclass, fields
from functools import lru_cache

import numpy as np
import pandas as pd
import torch

from lunar_crater.models.unet import UNet
from lunar_crater.postprocess.rings import RingMatcher
from lunar_crater.training.data import normalize_dem


@lru_cache(maxsize=4)
def load_model(checkpoint: str, device: str = "cpu") -> UNet:
    """Load a ``best.pt`` written by ``lunar_crater.training.train_unet``."""
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model = UNet(**state["config"]["model"])
    model.load_state_dict(state["state_dict"])
    return model.to(device).eval()


def predict_rims(model: UNet, dem: np.ndarray, device: str = "cpu") -> np.ndarray:
    """Rim probability for each pixel of one tile."""
    x = torch.from_numpy(normalize_dem(dem))[None, None].to(device)
    with torch.no_grad():
        return torch.sigmoid(model(x))[0, 0].cpu().numpy().astype(np.float64)


@dataclass(frozen=True)
class UNetDetector:
    """Harness detector: U-Net rim map followed by ring matching.

    ``checkpoint`` is the path to ``best.pt``. The other fields are passed to
    :class:`RingMatcher`. The model is loaded once per process.
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

    def matcher(self) -> RingMatcher:
        names = {f.name for f in fields(RingMatcher)}
        return RingMatcher(**{n: getattr(self, n) for n in names})

    def __call__(self, dem: np.ndarray, scale_m: float) -> pd.DataFrame:
        if not self.checkpoint:
            raise ValueError("set checkpoint=<path to best.pt>")
        prob = predict_rims(load_model(self.checkpoint, self.device), dem, self.device)
        return self.matcher()(prob)
