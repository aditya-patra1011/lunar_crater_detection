import numpy as np
import pandas as pd
import pytest

from lunar_crater.postprocess.confidence import (
    Confidence,
    Curve,
    calibration_error,
    isotonic,
    label_detections,
    reliability,
)


def test_isotonic_pools_violators_and_rises():
    x, y = isotonic(np.array([1, 2, 3, 4, 5.0]), np.array([0, 1, 0, 1, 1.0]))
    assert y.tolist() == [0.0, 0.5, 1.0]
    assert np.all(np.diff(x) > 0)
    x, y = isotonic(np.array([3, 1, 2.0]), np.array([1, 0, 0.0]))
    assert y.tolist() == [0.0, 1.0] and x.tolist() == [1.5, 3.0]


def labelled_rows(rng, n=4000):
    score = rng.uniform(0, 1, n)
    diam = np.where(rng.uniform(size=n) < 0.2, 10.0, 2.0)
    # Small craters match with chance = score; large ones always at half that.
    chance = np.where(diam >= 6, score / 2, score)
    return pd.DataFrame({"score": score, "diam_km": diam, "matched": rng.uniform(size=n) < chance})


def test_fit_recovers_each_curve_and_round_trips(tmp_path):
    rng = np.random.default_rng(0)
    confidence = Confidence.fit(labelled_rows(rng), fitted_on="synthetic")
    scores = np.array([0.2, 0.5, 0.8])
    assert confidence(scores, [2, 2, 2]) == pytest.approx(scores, abs=0.08)
    assert confidence(scores, [10, 10, 10]) == pytest.approx(scores / 2, abs=0.1)
    assert np.all(np.diff(confidence(np.linspace(0, 1, 50), np.full(50, 2.0))) >= 0)
    assert len(confidence.small.score) <= 40

    confidence.save(tmp_path / "c.json")
    again = Confidence.load(tmp_path / "c.json")
    assert again(scores, [2, 10, 2]) == pytest.approx(confidence(scores, [2, 10, 2]))
    assert again.fitted_on == "synthetic" and again.n_detections == 4000

    test = labelled_rows(np.random.default_rng(1))
    p = confidence(test["score"], test["diam_km"])
    assert calibration_error(p, test["matched"].to_numpy(float)) < 0.05


def test_an_empty_group_gets_a_flat_zero_curve():
    rows = pd.DataFrame({"score": [0.1, 0.9], "diam_km": [2.0, 2.0], "matched": [False, True]})
    confidence = Confidence.fit(rows)
    assert confidence([0.5], [20.0]).tolist() == [0.0]
    assert Curve([0, 1], [0, 1])([0.25]).tolist() == [0.25]


def test_reliability_bins_by_count():
    p = np.linspace(0, 1, 100)
    table = reliability(p, (p > 0.5).astype(float), bins=4)
    assert table["n"].tolist() == [25, 25, 25, 25]
    assert table["observed"].tolist() == [0.0, 0.0, 1.0, 1.0]


def test_label_detections_follows_the_protocol():
    tiles = pd.DataFrame({"tile_id": ["t0"], "size_px": [100], "scale_m": [118.45]})
    craters = pd.DataFrame(
        {"tile_id": ["t0"], "x_px": [50.0], "y_px": [50.0], "r_px": [10.0], "diam_km": [2.4],
         "centre_in_tile": [True]}
    )  # fmt: skip
    preds = pd.DataFrame(
        {"tile_id": ["t0"] * 4, "x_px": [50.0, 20.0, 50.0, 120.0], "y_px": [50.0, 20.0, 51.0, 50.0],
         "r_px": [10.0, 8.0, 2.0, 10.0], "score": [0.9, 0.8, 0.7, 0.6]}
    )  # fmt: skip
    rows = label_detections(preds, tiles, craters)
    # The 2 px one is under 1 km and the last is centred off the tile, so both are dropped.
    assert rows["score"].tolist() == [0.9, 0.8]
    assert rows["matched"].tolist() == [True, False]
    assert rows["diam_km"].iloc[0] == pytest.approx(2.369, abs=0.001)
    assert label_detections(preds.iloc[:0], tiles, craters).empty
