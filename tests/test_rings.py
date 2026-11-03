import numpy as np
import pytest

from lunar_crater.data.tiles import rasterize_rims
from lunar_crater.evaluation.matching import circle_iou
from lunar_crater.postprocess.peaks import find_circles, greedy_nms
from lunar_crater.postprocess.rings import RingMatcher


def ring_map(circles, size=128, thickness=2.0):
    x, y, r = (np.array(c, dtype=float) for c in zip(*circles, strict=True))
    return rasterize_rims(x, y, r, size, thickness_px=thickness).astype(float)


def best_iou(found, x, y, r):
    return circle_iou(found["x_px"], found["y_px"], found["r_px"], x, y, r).max()


def test_recovers_drawn_rings():
    circles = [(40, 40, 15), (90, 85, 7), (30, 100, 25)]
    found = RingMatcher(max_radius_px=40, min_score=0.2)(ring_map(circles))

    for c in circles:
        assert best_iou(found, *c) > 0.85
    # Small rings along a rim at the tile edge score at most about 0.5; the drawn
    # rings score 0.75 (cut by the edge) to 1, so a threshold separates them.
    top = found[found["score"] >= 0.6]
    assert len(top) == len(circles)
    assert found["score"].iloc[len(circles)] < 0.5


def test_filled_blob_is_not_a_crater():
    yy, xx = np.mgrid[0:128, 0:128]
    blob = 0.6 * (np.hypot(xx - 64, yy - 64) < 25)
    scores, _ = RingMatcher(min_radius_px=10, max_radius_px=10.5).score_volume(blob)
    assert scores[0, 64, 64] == pytest.approx(0.0, abs=1e-9)
    assert RingMatcher(max_radius_px=40, min_score=0.3)(blob).empty


def test_ring_cut_by_the_edge_is_found():
    found = RingMatcher(max_radius_px=40)(ring_map([(5, 64, 20)]))
    assert best_iou(found, 5, 64, 20) > 0.85


def test_empty_and_noisy_maps():
    matcher = RingMatcher(max_radius_px=30, min_score=0.3)
    assert matcher(np.zeros((64, 64))).empty
    noise = np.random.default_rng(0).uniform(0, 0.2, (64, 64))
    assert matcher(noise).empty


def test_score_is_mean_ring_probability():
    prob = 0.5 * ring_map([(64, 64, 20)])
    scores, coverage = RingMatcher(min_radius_px=20, max_radius_px=20.5).score_volume(prob)
    assert scores[0, 64, 64] == pytest.approx(0.5)
    assert coverage[0, 64, 64] == pytest.approx(1.0)


def test_find_circles_and_nms():
    scores = np.zeros((2, 32, 32))
    scores[0, 10, 10] = 0.9
    scores[1, 10, 11] = 0.8  # same crater one radius step larger
    scores[0, 25, 25] = 0.5
    found = find_circles(scores, np.array([5.0, 5.5]), min_score=0.4, nms_iou=0.3)

    assert found[["x_px", "y_px", "r_px"]].values.tolist() == [[10, 10, 5], [25, 25, 5]]
    assert greedy_nms(found.iloc[:0], 0.3) == []
