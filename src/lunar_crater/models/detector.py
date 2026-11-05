"""Crater detection with trained U-Nets: predict rim probabilities, then match rings."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd
import torch

from lunar_crater.models.unet import UNet
from lunar_crater.postprocess.finder import RimCircleFinder
from lunar_crater.training.data import normalize_dem


@lru_cache(maxsize=8)
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
class UNetDetector(RimCircleFinder):
    """Harness detector: U-Net rim maps followed by :class:`RimCircleFinder`.

    ``checkpoint`` is the path to ``best.pt``, or several joined by commas to average
    their rim maps. ``large_checkpoint`` takes the large-rim output from other models,
    again joined by commas, leaving ``checkpoint`` to supply only the main map. Models are
    loaded once per process.
    """

    checkpoint: str = ""
    large_checkpoint: str = ""
    device: str = "cpu"

    def models(self) -> list[UNet]:
        if not self.checkpoint:
            raise ValueError("set checkpoint=<path to best.pt>")
        return [load_model(path, self.device) for path in self.checkpoint.split(",")]

    def rim_maps(self, dem: np.ndarray) -> np.ndarray:
        """Rim maps for one tile, (channels, h, w), from the configured models."""
        prob = predict_rims(self.models(), dem, self.device)
        if not self.large_checkpoint:
            return prob
        models = [load_model(path, self.device) for path in self.large_checkpoint.split(",")]
        large = predict_rims(models, dem, self.device)
        if len(large) < 2:
            raise ValueError("large_checkpoint needs models with a large-rim output")
        return np.stack([prob[0], large[1]])

    def __call__(self, dem: np.ndarray, scale_m: float) -> pd.DataFrame:
        return self.circles(self.rim_maps(dem), dem, scale_m)
