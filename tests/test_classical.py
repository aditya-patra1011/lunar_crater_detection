import numpy as np
import pandas as pd
import pytest

from lunar_crater.baselines.classical import (
    RadialSlopeDetector,
    fit_radius_scale,
)
from lunar_crater.evaluation.matching import circle_iou
from lunar_crater.postprocess.peaks import fast_size, local_maxima

SCALE_M = 118.45


def bowl_dem(craters, size=128, tilt=0.0):
    """A plane of slope ``tilt`` with parabolic bowls of depth/diameter 0.2."""
    yy, xx = np.mgrid[0:size, 0:size].astype(float)
    dem = tilt * xx * SCALE_M
    for cx, cy, r in craters:
        d = np.hypot(xx - cx, yy - cy) / r
        depth = 0.2 * 2 * r * SCALE_M
        dem += np.where(d < 1, depth * (d**2 - 1), 0.0)
    return dem


def best_iou(found, x, y, r):
    return circle_iou(found["x_px"], found["y_px"], found["r_px"], x, y, r).max()


@pytest.mark.parametrize("tilt", [0.0, 0.2])
def test_finds_bowls_on_a_tilted_plane(tilt):
    craters = [(40, 45, 12), (90, 80, 6)]
    found = RadialSlopeDetector(max_radius_px=40)(bowl_dem(craters, tilt=tilt), SCALE_M)

    for x, y, r in craters:
        assert best_iou(found, x, y, r) > 0.8
    assert found["score"].is_monotonic_decreasing


def test_flat_and_tilted_planes_give_nothing():
    det = RadialSlopeDetector(max_radius_px=30)
    assert det(bowl_dem([]), SCALE_M).empty
    assert det(bowl_dem([], tilt=0.3), SCALE_M).empty


def test_scores_are_wall_slopes():
    dem = bowl_dem([(64, 64, 20)])
    det = RadialSlopeDetector(min_radius_px=20, max_radius_px=21, radius_step=1.2, inner=0.0)
    scores, coverage = det.score_volume(dem, SCALE_M)

    # A parabola of depth 8r over radius r has slope 16 d/r: mean over the disc is 0.4*2/3.
    assert scores[0, 64, 64] == pytest.approx(0.8 * 2 / 3, rel=0.05)
    assert coverage[0, 64, 64] == pytest.approx(1.0)
    assert coverage[0, 0, 0] == pytest.approx(0.25, abs=0.03)


def test_missing_data_and_edges_are_tolerated():
    dem = bowl_dem([(64, 64, 15)])
    dem[:, 110:] = np.nan
    found = RadialSlopeDetector(max_radius_px=40)(dem, SCALE_M)

    assert best_iou(found, 64, 64, 15) > 0.8
    assert np.isfinite(found[["x_px", "y_px", "r_px", "score"]].to_numpy()).all()


def test_all_missing_tile_gives_nothing():
    assert RadialSlopeDetector(max_radius_px=20)(np.full((64, 64), np.nan), SCALE_M).empty


def test_nms_drops_near_duplicates_only():
    craters = [(40, 64, 12), (88, 64, 12)]
    found = RadialSlopeDetector(max_radius_px=30, nms_iou=0.3)(bowl_dem(craters), SCALE_M)
    x, y, r = (found[c].to_numpy() for c in ("x_px", "y_px", "r_px"))
    iou = circle_iou(x[:, None], y[:, None], r[:, None], x[None], y[None], r[None])

    assert (iou[np.triu_indices(len(found), 1)] <= 0.3).all()
    assert all(best_iou(found, *c) > 0.8 for c in craters)


def test_radius_scale_multiplies_radii():
    dem = bowl_dem([(64, 64, 15)])
    plain = RadialSlopeDetector(max_radius_px=30)(dem, SCALE_M)
    scaled = RadialSlopeDetector(max_radius_px=30, radius_scale=1.25)(dem, SCALE_M)
    assert scaled["r_px"].iloc[0] == pytest.approx(1.25 * plain["r_px"].iloc[0])


def test_fit_radius_scale():
    truth = pd.DataFrame(
        {
            "tile_id": ["a", "a", "b"],
            "x_px": [10, 50, 30],
            "y_px": [10, 50, 30],
            "r_px": [10.0, 20.0, 8.0],
        }
    )
    pred = truth.assign(r_px=truth["r_px"] / 1.25)
    pred.loc[len(pred)] = ["b", 90, 90, 5.0]

    assert fit_radius_scale(pred, truth) == pytest.approx(1.25)
    assert fit_radius_scale(
        pred.drop(columns="tile_id").iloc[:1], truth.drop(columns="tile_id").iloc[:1]
    ) == pytest.approx(1.25)
    with pytest.raises(ValueError):
        fit_radius_scale(pred.assign(x_px=500), truth)


def test_helpers():
    assert fast_size(229) == 240
    assert fast_size(256) == 256
    v = np.zeros((3, 4, 4))
    v[1, 2, 2] = 1
    v[0, 0, 0] = 0.5
    assert np.argwhere(local_maxima(v) & (v > 0)).tolist() == [[0, 0, 0], [1, 2, 2]]
