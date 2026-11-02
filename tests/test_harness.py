import json

import numpy as np
import pandas as pd
import pytest

from lunar_crater.baselines.classical import RadialSlopeDetector
from lunar_crater.evaluation import harness

SCALE_M = 118.45


def oracle(dem, scale_m):
    """A fake detector that reads crater circles hidden in the tile's first row."""
    x, y, r = dem[0, :3]
    return pd.DataFrame({"x_px": [x, 5.0], "y_px": [y, 5.0], "r_px": [r, 6.0], "score": [0.9, 0.1]})


@pytest.fixture
def tiles_dir(tmp_path):
    (tmp_path / "tiles").mkdir()
    rows, craters = [], []
    for i, split in enumerate(["val", "val", "test"]):
        tid = f"t{i}"
        dem = np.zeros((64, 64), dtype=np.float32)
        dem[0, :3] = (20 + i, 30, 10)
        np.savez(tmp_path / "tiles" / f"{tid}.npz", dem=dem, mask=np.zeros_like(dem))
        rows.append(
            {
                "tile_id": tid,
                "lat": 0,
                "lon": i,
                "size_px": 64,
                "scale_m": SCALE_M,
                "n_craters": 1,
                "nan_fraction": 0.0,
                "split": split,
            }
        )
        craters.append(
            {
                "tile_id": tid,
                "crater_id": f"c{i}",
                "diam_km": 2.4,
                "x_px": 20 + i,
                "y_px": 30,
                "r_px": 10,
                "centre_in_tile": True,
            }
        )
    tiles = pd.DataFrame(rows)
    tiles.drop(columns="split").to_csv(tmp_path / "tiles.csv", index=False)
    tiles.assign(cell=0)[["tile_id", "cell", "split"]].to_csv(tmp_path / "split.csv", index=False)
    pd.DataFrame(craters).to_csv(tmp_path / "craters.csv", index=False)
    return tmp_path


def test_load_split(tiles_dir):
    tiles, craters = harness.load_split(tiles_dir, "val")
    assert tiles["tile_id"].tolist() == ["t0", "t1"]
    assert craters["tile_id"].tolist() == ["t0", "t1"]


def test_detect_score_and_sweep(tiles_dir):
    tiles, craters = harness.load_split(tiles_dir, "val")
    preds = harness.detect_tiles(tiles_dir, tiles, oracle)
    assert preds.columns.tolist() == harness.PRED_COLUMNS
    assert len(preds) == 4

    everything = harness.score(preds, tiles, craters)
    assert everything.loc["all", "recall"] == 1.0
    assert everything.loc["all", "precision"] == 0.5
    confident = harness.score(preds, tiles, craters, min_score=0.5)
    assert confident.loc["all", "precision"] == 1.0

    curve = harness.sweep(preds, tiles, craters, [0.0, 0.5, 0.95])
    assert curve["f1"].tolist()[:2] == pytest.approx([2 / 3, 1.0])
    assert np.isnan(curve["f1"].iloc[2])
    assert harness.best_threshold(curve) == 0.5


def test_detect_in_parallel_matches_serial(tiles_dir):
    tiles, _ = harness.load_split(tiles_dir, "val")
    serial = harness.detect_tiles(tiles_dir, tiles, RadialSlopeDetector(max_radius_px=10))
    parallel = harness.detect_tiles(
        tiles_dir, tiles, RadialSlopeDetector(max_radius_px=10), workers=2
    )
    pd.testing.assert_frame_equal(serial, parallel)


def test_no_tiles_and_no_detections(tiles_dir):
    tiles, craters = harness.load_split(tiles_dir, "train")
    preds = harness.detect_tiles(tiles_dir, tiles, oracle)
    assert preds.empty and preds.columns.tolist() == harness.PRED_COLUMNS
    assert harness.default_thresholds(preds["score"]) == [-np.inf]
    curve = pd.DataFrame({"min_score": [0.2, 0.1], "f1": [np.nan, np.nan]})
    assert harness.best_threshold(curve) == 0.1


def test_parse_params():
    params = harness.parse_params(RadialSlopeDetector, ["inner=0.4", "min_radius_px=5"])
    assert params == {"inner": 0.4, "min_radius_px": 5.0}
    with pytest.raises(ValueError):
        harness.parse_params(RadialSlopeDetector, ["bogus=1"])
    with pytest.raises(ValueError):
        harness.parse_params(RadialSlopeDetector, ["inner"])


def test_main_writes_results(tiles_dir, tmp_path, monkeypatch):
    monkeypatch.setitem(harness.DETECTORS, "classical", RadialSlopeDetector)
    out = tmp_path / "out"
    code = harness.main(
        [
            str(tiles_dir),
            "--split",
            "test",
            "--out",
            str(out),
            "--set",
            "max_radius_px=20",
            "--limit",
            "1",
            "--min-score",
            "0.1",
        ]
    )
    assert code == 0
    run = json.loads((out / "run.json").read_text())
    assert run["n_tiles"] == 1 and run["params"]["max_radius_px"] == 20.0
    assert run["min_score"] == 0.1 and run["min_score_from"] == "given"
    for name in ("predictions.csv", "sweep.csv", "metrics.csv"):
        assert (out / name).exists()


def test_main_refuses_to_tune_on_test(tiles_dir, tmp_path):
    with pytest.raises(SystemExit):
        harness.main([str(tiles_dir), "--split", "test", "--out", str(tmp_path / "out")])
