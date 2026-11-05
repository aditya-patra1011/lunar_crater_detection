"""An HTTP API that finds craters in an elevation tile.

Usage::

    uvicorn --factory lunar_crater.serving.app:create_app

The model path comes from the ``LUNAR_MODEL`` environment variable, and example tiles
for the dashboard from ``LUNAR_TILES``. The dashboard is at ``/`` and interactive API docs
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
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

from lunar_crater import __version__
from lunar_crater.data.projection import wrap_longitude
from lunar_crater.monitoring.drift import MIN_TILES, DriftMonitor, Reference, tile_features
from lunar_crater.serving.predictor import CraterPredictor
from lunar_crater.serving.samples import SampleTiles

DEFAULT_MODEL = "models/crater_detector.onnx"
DEFAULT_TILES = "data/interim/tiles/v1"
STATIC = Path(__file__).parent / "static"
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
    confidence: float | None = None


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
    confidence: bool
    monitoring: bool


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


def create_app(
    predictor: CraterPredictor | None = None,
    tiles_dir: Path | None = None,
    reference: Reference | None = None,
    monitor_min_tiles: int = MIN_TILES,
) -> FastAPI:
    """Build the API around ``predictor``, or the model at ``LUNAR_MODEL``.

    Example tiles come from ``tiles_dir``, or ``LUNAR_TILES``; without a tile set the
    dashboard still works, but only with uploaded tiles. Tiles sent to ``/detect`` are
    compared with ``reference``, or ``<model>.reference.json`` beside the model.
    """
    if predictor is None:
        model_path = Path(os.environ.get("LUNAR_MODEL", DEFAULT_MODEL))
        predictor = CraterPredictor(model_path)
        reference_path = model_path.with_suffix(".reference.json")
        if reference is None and reference_path.exists():
            reference = Reference.load(reference_path)
    if tiles_dir is None:
        tiles_dir = Path(os.environ.get("LUNAR_TILES", DEFAULT_TILES))
    samples = SampleTiles(tiles_dir, predictor.size_px)
    monitor = DriftMonitor(reference, min_tiles=monitor_min_tiles)
    page = (STATIC / "index.html").read_text(encoding="utf-8")
    # The first tile builds the ring kernels, which takes far longer than later tiles.
    predictor(np.zeros((predictor.size_px, predictor.size_px), np.float32), 0.0, 0.0)
    # Room for the largest numeric .npy of this shape (float64) and its header.
    max_bytes = predictor.size_px**2 * 8 + 1024
    app = FastAPI(title="Lunar crater detection", version=__version__)

    @app.get("/", include_in_schema=False)
    def dashboard() -> HTMLResponse:
        return HTMLResponse(page)

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon() -> FileResponse:
        return FileResponse(STATIC / "favicon.svg", media_type="image/svg+xml")

    @app.get("/samples")
    def sample_list() -> list[dict]:
        """Example test tiles available to the dashboard."""
        return samples.listing()

    @app.get("/samples/{tile_id}")
    def sample(tile_id: str) -> dict:
        """Detections on one example tile, marked against the catalog."""
        if tile_id not in samples:
            raise HTTPException(404, f"no example tile {tile_id!r}")
        start = time.perf_counter()
        result = samples.run(tile_id, predictor)
        return {**result, "model_id": predictor.model_id, "min_score": predictor.min_score,
                "seconds": time.perf_counter() - start}  # fmt: skip

    @app.get("/monitoring")
    def monitoring() -> dict:
        """Whether recent tiles sent to /detect still look like the reference tiles."""
        return monitor.report()

    @app.get("/health")
    def health() -> Health:
        return Health(
            status="ok",
            version=__version__,
            model_id=predictor.model_id,
            size_px=predictor.size_px,
            scale_m=predictor.scale_m,
            confidence=predictor.confidence is not None,
            monitoring=monitor.reference is not None,
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
        if min_score is None:
            # Only the default threshold is comparable with the reference.
            monitor.record(tile_features(dem, predictor.scale_m, found["score"]))
        return Detections(
            model_id=predictor.model_id,
            min_score=predictor.min_score if min_score is None else min_score,
            seconds=time.perf_counter() - start,
            n_craters=len(found),
            craters=found.to_dict("records"),
        )

    return app
