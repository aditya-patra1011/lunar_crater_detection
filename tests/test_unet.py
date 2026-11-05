import json
from dataclasses import replace

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
    assert config["model"] == {"base": 4, "depth": 2, "large_level": 0}

    model = load_model(str(out / "best.pt"))
    with np.load(tiles_dir / "tiles" / "t0.npz") as z:
        prob = predict_rims([model], z["dem"])
    assert prob.shape == (1, 32, 32) and ((prob >= 0) & (prob <= 1)).all()

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


def test_large_output_shape_and_old_checkpoints_still_load(tmp_path):
    model = UNet(base=4, depth=3, large_level=2)
    assert model(torch.zeros(2, 1, 32, 32)).shape == (2, 2, 32, 32)
    with pytest.raises(ValueError):
        UNet(base=4, depth=2, large_level=2)

    old = UNet(base=4, depth=2)
    path = tmp_path / "old.pt"
    torch.save({"state_dict": old.state_dict(), "config": {"model": {"base": 4, "depth": 2}}}, path)
    assert load_model(str(path)).large_level == 0


def test_two_output_training_ensemble_and_large_sources(tiles_dir, tmp_path):
    paths = []
    for seed in (0, 1):
        out = tmp_path / f"seed{seed}"
        train_unet.main(
            [str(tiles_dir), "--out", str(out), "--epochs", "1", "--batch-size", "2",
             "--base", "4", "--depth", "3", "--large-min-diam", "1.5", "--large-level", "2",
             "--device", "cpu", "--seed", str(seed)]
        )  # fmt: skip
        paths.append(str(out / "best.pt"))
    assert json.loads((tmp_path / "seed0" / "config.json").read_text())["model"]["large_level"] == 2

    one, two = load_model(paths[0]), load_model(paths[1])
    dem = np.random.default_rng(0).normal(size=(32, 32)).astype(np.float32)
    assert predict_rims([one, two], dem).shape == (2, 32, 32)
    assert predict_rims([one, two], dem) == pytest.approx(
        (predict_rims([one], dem) + predict_rims([two], dem)) / 2
    )
    detector = UNetDetector(
        checkpoint=",".join(paths),
        large_source="head",
        large_from_km=2.0,
        max_radius_px=20,
        min_score=0.0,
        large_min_score=0.0,
    )
    assert detector(dem, SCALE_M)["score"].is_monotonic_decreasing


def test_large_checkpoint_supplies_only_the_large_output(tiles_dir, tmp_path):
    def trained(name, *extra):
        out = tmp_path / name
        train_unet.main(
            [str(tiles_dir), "--out", str(out), "--epochs", "1", "--batch-size", "2",
             "--base", "4", "--depth", "3", "--device", "cpu", *extra]
        )  # fmt: skip
        return str(out / "best.pt")

    one = trained("one")
    two = trained("two", "--large-min-diam", "1.5", "--large-level", "2")
    dem = np.random.default_rng(0).normal(size=(32, 32)).astype(np.float32)

    prob = UNetDetector(checkpoint=one, large_checkpoint=two).rim_maps(dem)
    assert prob.shape == (2, 32, 32)
    assert prob[0] == pytest.approx(predict_rims([load_model(one)], dem)[0])
    assert prob[1] == pytest.approx(predict_rims([load_model(two)], dem)[1])

    with pytest.raises(ValueError, match="large-rim output"):
        UNetDetector(checkpoint=two, large_checkpoint=one).rim_maps(dem)


def test_size_gated_circles_use_each_source_on_its_side():
    size = 128
    yy, xx = np.mgrid[0:size, 0:size].astype(float)
    small_ring = (np.abs(np.hypot(xx - 30, yy - 30) - 8) <= 1).astype(float)
    large_ring = (np.abs(np.hypot(xx - 70, yy - 70) - 40) <= 2).astype(float)
    prob = np.stack([small_ring, large_ring])
    detector = UNetDetector(large_source="head", large_from_km=4.0, large_gain=0.5, min_score=0.5,
                            large_min_score=0.5, max_radius_px=60)  # fmt: skip
    found = detector.circles(prob, np.zeros((size, size)), SCALE_M)

    best = found.iloc[0]
    assert (best["x_px"], best["y_px"]) == (30, 30)
    big = found[found["r_px"] > 30].iloc[0]
    assert abs(big["r_px"] - 40) < 2 and big["score"] <= 0.5

    with pytest.raises(ValueError, match="large-rim output"):
        detector.circles(prob[:1], np.zeros((size, size)), SCALE_M)
    with pytest.raises(ValueError, match="unknown"):
        replace(detector, large_source="bogus").circles(prob, np.zeros((size, size)), SCALE_M)


def test_classical_large_source_finds_a_bowl():
    size = 128
    yy, xx = np.mgrid[0:size, 0:size].astype(float)
    d = np.hypot(xx - 64, yy - 64) / 30
    dem = np.where(d < 1, 1400 * (d**2 - 1), 0.0)
    detector = UNetDetector(large_source="classical", large_from_km=4.0, large_radius_scale=1.0,
                            min_score=0.5, large_min_score=0.05, max_radius_px=60)  # fmt: skip
    found = detector.circles(np.zeros((1, size, size)), dem, SCALE_M)
    assert len(found) and abs(found.iloc[0]["r_px"] - 30) < 3
