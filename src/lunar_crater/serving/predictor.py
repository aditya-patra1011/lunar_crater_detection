"""Find craters in a tile with the exported model, without PyTorch."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import onnxruntime
import pandas as pd

from lunar_crater.data.tiles import TileSpec
from lunar_crater.postprocess.finder import RimCircleFinder
from lunar_crater.postprocess.georef import to_moon_coords
from lunar_crater.training.data import normalize_dem


class CraterPredictor:
    """The ONNX network, its settings and the steps from heights to craters."""

    def __init__(self, onnx_path: Path):
        onnx_path = Path(onnx_path)
        settings = json.loads(onnx_path.with_suffix(".json").read_text())
        digest = hashlib.sha256(onnx_path.read_bytes()).hexdigest()
        if digest != settings["onnx_sha256"]:
            raise ValueError(f"{onnx_path} does not match the checksum in its settings")
        self.model_id = digest[:12]
        self.size_px = int(settings["size_px"])
        self.scale_m = float(settings["scale_m"])
        self.min_score = float(settings["min_score"])
        self.finder = RimCircleFinder(**settings["finder"])
        self.session = onnxruntime.InferenceSession(
            str(onnx_path), providers=["CPUExecutionProvider"]
        )

    def rim_maps(self, dem: np.ndarray) -> np.ndarray:
        """Rim probabilities for one tile, (channels, h, w)."""
        x = normalize_dem(dem)[None, None]
        return self.session.run(None, {"tile": x})[0][0].astype(np.float64)

    def __call__(
        self, dem: np.ndarray, lat: float, lon: float, *, min_score: float | None = None
    ) -> pd.DataFrame:
        """Craters scoring at least ``min_score``, best first, in pixels and on the Moon."""
        if dem.shape != (self.size_px, self.size_px):
            raise ValueError(f"expected a {self.size_px} x {self.size_px} tile, got {dem.shape}")
        threshold = self.min_score if min_score is None else min_score
        found = self.finder.circles(self.rim_maps(dem), dem, self.scale_m)
        found = found[found["score"] >= threshold].reset_index(drop=True)
        return to_moon_coords(found, TileSpec(lat, lon, self.size_px, self.scale_m))
