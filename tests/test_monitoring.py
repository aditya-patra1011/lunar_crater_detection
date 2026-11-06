import numpy as np
import pandas as pd
import pytest

from lunar_crater.monitoring.drift import (
    MIN_TILES,
    DriftMonitor,
    Reference,
    bin_shares,
    learn_thresholds,
    psi,
    status,
    tile_features,
)


def test_tile_features_on_a_tilted_plane_with_a_hole():
    yy, xx = np.mgrid[0:64, 0:64].astype(float)
    dem = xx * 100.0  # 100 m rise per 100 m pixel: a 45 degree slope
    dem[0, :16] = np.nan
    f = tile_features(dem, 100.0, [0.5, 0.7, 0.9])
    assert f["slope_deg"] == pytest.approx(45.0, abs=0.5)
    assert f["missing"] == pytest.approx(16 / 64**2)
    assert f["n_craters"] == 3 and f["median_score"] == pytest.approx(0.7)
    assert tile_features(np.full((8, 8), np.nan), 100.0, [])["missing"] == 1.0


def test_psi_and_status():
    assert psi([0.5, 0.5], [0.5, 0.5]) == pytest.approx(0)
    assert psi([0.5, 0.5], [0.9, 0.1]) > 0.25
    assert (status(0.05), status(0.2), status(0.3)) == ("stable", "watch", "drift")
    assert bin_shares(np.array([0, 1, 2, 3.0]), [1.5]).tolist() == [0.5, 0.5]
    assert bin_shares(np.array([0, 1.0]), [1.5], prior=1).tolist() == [0.75, 0.25]


def test_small_samples_of_the_reference_do_not_look_shifted():
    rng = np.random.default_rng(3)
    reference = Reference.build(features(rng, 2000))
    worst = [max(f["psi"] for f in reference.compare(features(rng, MIN_TILES)).values())
             for _ in range(50)]  # fmt: skip
    assert np.percentile(worst, 95) < 0.25


def features(rng, n, relief_scale=1.0):
    return pd.DataFrame(
        {
            "relief_m": rng.lognormal(6, 0.5, n) * relief_scale,
            "slope_deg": rng.normal(5, 1, n),
            "missing": np.zeros(n),
            "n_craters": rng.poisson(20, n).astype(float),
            "median_score": rng.uniform(0.5, 0.8, n),
        }
    )


def test_reference_flags_shifted_terrain_and_round_trips(tmp_path):
    rng = np.random.default_rng(0)
    reference = Reference.build(features(rng, 450), source="synthetic")
    same = reference.compare(features(rng, 200))
    assert all(f["status"] == "stable" for f in same.values())
    shifted = reference.compare(features(rng, 200, relief_scale=3.0))
    assert shifted["relief_m"]["status"] == "drift"
    assert shifted["slope_deg"]["status"] == "stable"

    reference.save(tmp_path / "r.json")
    again = Reference.load(tmp_path / "r.json")
    assert again.n_tiles == 450 and again.edges == reference.edges


def test_monitor_waits_then_reports():
    rng = np.random.default_rng(1)
    monitor = DriftMonitor(Reference.build(features(rng, 300)), window=150)
    assert DriftMonitor(None).report()["status"] == "off"
    for row in features(rng, MIN_TILES - 1).to_dict("records"):
        monitor.record(row)
    assert monitor.report()["status"] == "waiting"
    for row in features(rng, 160, relief_scale=4.0).to_dict("records"):
        monitor.record(row)
    report = monitor.report()
    assert report["tiles_seen"] == MIN_TILES - 1 + 160 and report["in_window"] == 150
    assert report["status"] == "drift" and report["inputs_status"] == "drift"
    assert report["features"]["relief_m"]["psi"] > 0.25


def test_learned_thresholds_allow_for_normal_differences_between_regions(tmp_path):
    rng = np.random.default_rng(2)
    parts = []
    for region in range(20):  # each region has its own typical relief
        part = features(rng, 60, relief_scale=rng.uniform(0.6, 1.6))
        parts.append(part.assign(region=region))
    data = pd.concat(parts, ignore_index=True)
    watch, drift = learn_thresholds(data, data["region"].to_numpy(), held_out=4, window=100,
                                    repeats=60)  # fmt: skip
    assert watch["relief_m"] > 0.1 and drift["relief_m"] >= watch["relief_m"]
    assert watch["missing"] == 0.1 and drift["missing"] == 0.25

    reference = Reference.build(data)
    reference.watch, reference.drift = watch, drift
    reference.save(tmp_path / "r.json")
    again = Reference.load(tmp_path / "r.json")
    assert again.thresholds("relief_m") == (watch["relief_m"], drift["relief_m"])
    normal = features(rng, 100, relief_scale=1.2)
    assert again.compare(normal)["relief_m"]["status"] != "drift"
    assert again.compare(features(rng, 100, relief_scale=6.0))["relief_m"]["status"] == "drift"
