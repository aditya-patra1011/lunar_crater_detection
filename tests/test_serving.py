import io
import json

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnxruntime")
pytest.importorskip("onnxscript")
pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from lunar_crater.data.tiles import TileSpec  # noqa: E402
from lunar_crater.models.detector import UNetDetector  # noqa: E402
from lunar_crater.models.unet import UNet  # noqa: E402
from lunar_crater.postprocess.finder import RimCircleFinder  # noqa: E402
from lunar_crater.serving.app import create_app  # noqa: E402
from lunar_crater.serving.export import export  # noqa: E402
from lunar_crater.serving.predictor import CraterPredictor  # noqa: E402

SIZE = 64
SCALE_M = 118.45
FINDER = RimCircleFinder(
    thickness_frac=0.2,
    nms_iou=0.1,
    max_radius_px=30,
    large_source="head",
    large_from_km=3.0,
    large_min_score=0.0,
    min_score=0.0,
)


def save_model(path, seed, **config):
    torch.manual_seed(seed)
    config = {"base": 4, "depth": 2, **config}
    torch.save({"state_dict": UNet(**config).state_dict(), "config": {"model": config}}, path)
    return str(path)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    folder = tmp_path_factory.mktemp("model")
    main = save_model(folder / "main.pt", 0)
    large = [save_model(folder / f"large{i}.pt", i, large_level=1) for i in (1, 2)]
    out = folder / "crater_detector"
    spec = TileSpec(0.0, 0.0, SIZE, SCALE_M)
    settings = export(main, large, out, FINDER, min_score=-1.0, spec=spec)
    return {"main": main, "large": large, "onnx": out.with_suffix(".onnx"), "settings": settings}


@pytest.fixture(scope="module")
def predictor(built):
    return CraterPredictor(built["onnx"])


@pytest.fixture(scope="module")
def client(predictor):
    return TestClient(create_app(predictor))


@pytest.fixture
def dem():
    return np.random.default_rng(3).normal(scale=300, size=(SIZE, SIZE)).astype(np.float32)


def as_npy(array) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, array)
    return buffer.getvalue()


def post(client, data: bytes, lat=10.0, lon=20.0, **form):
    return client.post(
        "/detect", files={"tile": ("t.npy", data)}, data={"lat": lat, "lon": lon, **form}
    )


def test_onnx_matches_the_pytorch_detector(built, predictor, dem):
    assert built["settings"]["max_difference_from_pytorch"] < 1e-3
    detector = UNetDetector(
        checkpoint=built["main"], large_checkpoint=",".join(built["large"]), **vars(FINDER)
    )
    assert predictor.rim_maps(dem) == pytest.approx(detector.rim_maps(dem), abs=1e-3)

    onnx_found = predictor(dem, 0.0, 0.0)
    torch_found = detector(dem, SCALE_M)
    assert len(onnx_found) == len(torch_found) > 0
    assert onnx_found["r_px"].to_numpy() == pytest.approx(torch_found["r_px"].to_numpy())


def test_predictor_applies_the_threshold_and_checks_the_shape(predictor, dem):
    every = predictor(dem, 0.0, 0.0)
    cut = float(every["score"].median())
    some = predictor(dem, 0.0, 0.0, min_score=cut)
    assert 0 < len(some) < len(every) and (some["score"] >= cut).all()
    with pytest.raises(ValueError, match="64 x 64"):
        predictor(np.zeros((32, 32)), 0.0, 0.0)


def test_predictor_refuses_a_changed_model(built, tmp_path):
    onnx_path = tmp_path / "copy.onnx"
    onnx_path.write_bytes(built["onnx"].read_bytes() + b"x")
    onnx_path.with_suffix(".json").write_text(json.dumps(built["settings"]))
    with pytest.raises(ValueError, match="checksum"):
        CraterPredictor(onnx_path)


def test_export_needs_large_outputs(built, tmp_path):
    spec = TileSpec(0.0, 0.0, SIZE, SCALE_M)
    with pytest.raises(ValueError, match="large-rim output"):
        export(built["main"], [built["main"]], tmp_path / "x", FINDER, min_score=0.0, spec=spec)


def test_health(client, predictor):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["model_id"] == predictor.model_id
    assert body["size_px"] == SIZE


def test_detect_returns_craters_on_the_moon(client, predictor, dem):
    response = post(client, as_npy(dem))
    assert response.status_code == 200
    body = response.json()
    assert body["n_craters"] == len(body["craters"]) > 0 and body["min_score"] == -1.0
    expected = predictor(dem, 10.0, 20.0)
    got = pd.DataFrame(body["craters"])
    assert got["lat"].to_numpy() == pytest.approx(expected["lat"].to_numpy())
    assert np.abs(got["lat"] - 10).max() < 1 and np.abs(got["lon"] - 20).max() < 1


def test_detect_wraps_longitude_and_takes_a_threshold(client, dem):
    response = post(client, as_npy(dem), lon=350.0, min_score=99.0)
    assert response.status_code == 200
    assert response.json()["n_craters"] == 0 and response.json()["min_score"] == 99.0
    assert post(client, as_npy(dem), lon=350.0).json()["craters"][0]["lon"] < 0


@pytest.mark.parametrize(
    "data, status",
    [
        (b"not a numpy file", 422),
        (as_npy(np.zeros((32, 32))), 422),
        (as_npy(np.full((SIZE, SIZE), np.nan)), 422),
        (as_npy(np.array([["a"] * SIZE] * SIZE)), 422),
        (as_npy(np.array([{"x": 1}], dtype=object)), 422),
        (b"\0" * (SIZE**2 * 8 + 2000), 413),
    ],
    ids=["not_numpy", "wrong_shape", "all_missing", "text", "objects", "too_large"],
)
def test_detect_refuses_bad_tiles(client, data, status):
    assert post(client, data).status_code == status


def test_detect_refuses_an_impossible_latitude(client, dem):
    assert post(client, as_npy(dem), lat=91.0).status_code == 422
