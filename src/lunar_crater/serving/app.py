"""An HTTP API that finds craters in an elevation tile.

Usage::

    uvicorn --factory lunar_crater.serving.app:create_app

The model path comes from the ``LUNAR_MODEL`` environment variable. Interactive docs are
at ``/docs``.
"""

from __future__ import annotations

import io
import os
import time
from pathlib import Path
from typing import Annotated

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from lunar_crater import __version__
from lunar_crater.data.projection import wrap_longitude
from lunar_crater.serving.predictor import CraterPredictor

DEFAULT_MODEL = "models/crater_detector.onnx"
#: Tiles above this share of missing heights are refused, since the model did not see them.
MAX_MISSING = 0.5


class Crater(BaseModel):
    lat: float
    lon: float
    diam_km: float
    x_px: float
    y_px: float
    r_px: float
    score: float


class Detections(BaseModel):
    model_id: str
    min_score: float
    seconds: float
    n_craters: int
    craters: list[Crater]


class Health(BaseModel):
    status: str
    version: str
    model_id: str
    size_px: int
    scale_m: float


def read_tile(data: bytes, size_px: int) -> np.ndarray:
    """Heights from the bytes of a .npy file; raises HTTP 422 if it is not a usable tile."""
    try:
        dem = np.load(io.BytesIO(data), allow_pickle=False)
    except (ValueError, OSError, EOFError) as error:
        raise HTTPException(422, "tile must be a .npy file holding numbers") from error
    if not isinstance(dem, np.ndarray) or dem.dtype.kind not in "fiu":
        raise HTTPException(422, "tile must be a .npy file holding numbers")
    if dem.shape != (size_px, size_px):
        raise HTTPException(422, f"tile must be {size_px} x {size_px}, got {dem.shape}")
    dem = dem.astype(np.float32)
    if np.mean(~np.isfinite(dem)) > MAX_MISSING:
        raise HTTPException(422, f"more than {MAX_MISSING:.0%} of the tile's heights are missing")
    return dem


def create_app(predictor: CraterPredictor | None = None) -> FastAPI:
    """Build the API around ``predictor``, or the model at ``LUNAR_MODEL``."""
    if predictor is None:
        predictor = CraterPredictor(Path(os.environ.get("LUNAR_MODEL", DEFAULT_MODEL)))
    # The first tile builds the ring kernels, which takes far longer than later tiles.
    predictor(np.zeros((predictor.size_px, predictor.size_px), np.float32), 0.0, 0.0)
    # Room for the largest numeric .npy of this shape (float64) and its header.
    max_bytes = predictor.size_px**2 * 8 + 1024
    app = FastAPI(title="Lunar crater detection", version=__version__)

    @app.get("/health")
    def health() -> Health:
        return Health(
            status="ok",
            version=__version__,
            model_id=predictor.model_id,
            size_px=predictor.size_px,
            scale_m=predictor.scale_m,
        )

    @app.post("/detect")
    def detect(
        tile: Annotated[UploadFile, File(description="heights in metres, as a .npy file")],
        lat: Annotated[float, Form(ge=-90, le=90, description="tile centre latitude, degrees")],
        lon: Annotated[float, Form(ge=-360, le=360, description="tile centre longitude, degrees")],
        min_score: Annotated[
            float | None, Form(description="default: the threshold chosen on val")
        ] = None,
    ) -> Detections:
        data = tile.file.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise HTTPException(413, "tile file is too large")
        dem = read_tile(data, predictor.size_px)
        start = time.perf_counter()
        found = predictor(dem, lat, float(wrap_longitude(lon)), min_score=min_score)
        return Detections(
            model_id=predictor.model_id,
            min_score=predictor.min_score if min_score is None else min_score,
            seconds=time.perf_counter() - start,
            n_craters=len(found),
            craters=found.to_dict("records"),
        )

    return app
