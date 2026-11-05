import json

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from lunar_crater.evaluation import harness  # noqa: E402
from lunar_crater.models.detector import UNetDetector, load_model, predict_rims  # noqa: E402
from lunar_crater.models.unet import UNet, rim_loss  # noqa: E402
from lunar_crater.training import train_unet  # noqa: E402

SCALE_M = 118.45


def test_unet_shapes():
    model = UNet(base=4, depth=2)
    assert model(torch.zeros(2, 1, 32, 32)).shape == (2, 1, 32, 32)
    assert model(torch.zeros(1, 1, 36, 44)).shape == (1, 1, 36, 44)


def test_rim_loss():
    target = torch.zeros(1, 1, 8, 8)
    target[..., 2, :] = 1
    good = torch.where(target > 0, 10.0, -10.0)
    assert float(rim_loss(good, target)) < 0.01
    assert float(rim_loss(-good, target)) > 5
    assert float(rim_loss(torch.full((1, 1, 8, 8), -10.0), torch.zeros(1, 1, 8, 8))) < 0.01


@pytest.fixture
def tiles_dir(tmp_path):
    """Four tiles with one bowl each, two in train and two in val."""
    (tmp_path / "tiles").mkdir()
    yy, xx = np.mgrid[0:32, 0:32]
    tiles, craters = [], []
    for i, split in enumerate(["train", "train", "val", "val"]):
        cx, cy, r = 14 + i, 16, 8
        d = np.hypot(xx - cx, yy - cy) / r
        dem = np.where(d < 1, 400 * (d**2 - 1), 0.0).astype(np.float32)
        np.savez(tmp_path / "tiles" / f"t{i}.npz", dem=dem, mask=np.zeros_like(dem))
        tiles.append({"tile_id": f"t{i}", "size_px": 32, "scale_m": SCALE_M, "split": split})
        craters.append({"tile_id": f"t{i}", "crater_id": f"c{i}", "diam_km": 2 * r * 0.118,
                        "x_px": cx, "y_px": cy, "r_px": r, "centre_in_tile": True})  # fmt: skip
    tiles = pd.DataFrame(tiles)
    tiles.drop(columns="split").to_csv(tmp_path / "tiles.csv", index=False)
    tiles.assign(cell=0)[["tile_id", "cell", "split"]].to_csv(tmp_path / "split.csv", index=False)
    pd.DataFrame(craters).to_csv(tmp_path / "craters.csv", index=False)
    return tmp_path


def test_train_writes_checkpoint_and_detector_runs(tiles_dir, tmp_path):
    out = tmp_path / "run"
    code = train_unet.main(
        [str(tiles_dir), "--out", str(out), "--epochs", "2", "--batch-size", "2",
         "--base", "4", "--depth", "2", "--device", "cpu"]
    )  # fmt: skip
    assert code == 0
    history = pd.read_csv(out / "history.csv")
    assert history["epoch"].tolist() == [1, 2]
    assert np.isfinite(history[["train_loss", "val_loss"]].to_numpy()).all()
    config = json.loads((out / "config.json").read_text())
    assert config["model"] == {"base": 4, "depth": 2}

    model = load_model(str(out / "best.pt"))
    with np.load(tiles_dir / "tiles" / "t0.npz") as z:
        prob = predict_rims(model, z["dem"])
    assert prob.shape == (32, 32) and ((prob >= 0) & (prob <= 1)).all()

    detector = UNetDetector(checkpoint=str(out / "best.pt"), max_radius_px=12, min_score=0.0)
    found = detector(np.zeros((32, 32)), SCALE_M)
    assert found.columns.tolist() == ["x_px", "y_px", "r_px", "score"]

    params = harness.parse_params(harness.detector_class("unet"), ["checkpoint=x.pt"])
    assert params == {"checkpoint": "x.pt"}


def test_detector_needs_a_checkpoint():
    with pytest.raises(ValueError):
        UNetDetector()(np.zeros((32, 32)), SCALE_M)


def test_pick_device():
    assert train_unet.pick_device("cpu").type == "cpu"
    assert train_unet.pick_device("auto").type in {"cpu", "cuda"}
