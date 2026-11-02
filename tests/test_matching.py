import math

import numpy as np
import pandas as pd
import pytest

from lunar_crater.evaluation.matching import (
    circle_iou,
    evaluate_tile,
    match_circles,
    radius_px_to_diam_km,
    summarize,
)


def test_iou_special_cases():
    assert circle_iou(0, 0, 5, 0, 0, 5) == pytest.approx(1.0)
    assert circle_iou(0, 0, 5, 20, 0, 5) == 0.0
    assert circle_iou(0, 0, 5, 10, 0, 5) == 0.0  # touching
    assert circle_iou(0, 0, 10, 1, 0, 5) == pytest.approx(0.25)  # contained


def test_iou_matches_a_pixel_count():
    g = np.mgrid[-20:20:0.02, -20:20:0.02]
    a = np.hypot(g[0], g[1]) <= 10
    b = np.hypot(g[0] - 7, g[1] - 3) <= 6
    expected = (a & b).sum() / (a | b).sum()

    assert circle_iou(0, 0, 10, 7, 3, 6) == pytest.approx(expected, abs=1e-3)
    assert circle_iou(7, 3, 6, 0, 0, 10) == pytest.approx(expected, abs=1e-3)


def test_matching_is_one_to_one_and_prefers_best_iou():
    true = [[10, 10, 5]]
    pred = [[10.5, 10, 5], [10, 10, 5]]  # the second is the better match

    p, t, iou = match_circles(pred, true)

    assert p.tolist() == [1] and t.tolist() == [0] and iou[0] == pytest.approx(1.0)


def test_matching_respects_threshold_and_empty_inputs():
    assert len(match_circles([[0, 0, 5]], [[3, 0, 5]], 0.5)[0]) == 0
    assert len(match_circles(np.empty((0, 3)), [[0, 0, 1]])[0]) == 0


def test_radius_conversion_round_trips_the_tile_formula():
    diam_km = np.array([1.0, 10.0, 500.0])
    r_px = 1_737_400 * np.sin(diam_km * 500 / 1_737_400) / 100.0
    np.testing.assert_allclose(radius_px_to_diam_km(r_px, 100.0), diam_km)


def truth_frame(rows):
    return pd.DataFrame(rows, columns=["x_px", "y_px", "r_px", "diam_km", "centre_in_tile"])


def pred_frame(rows):
    return pd.DataFrame(rows, columns=["x_px", "y_px", "r_px"])


def test_evaluate_tile_counting_rules():
    scale = 100.0  # m/px, so r_px 10 -> 2 km diameter
    truth = truth_frame(
        [
            (20, 20, 10, 2.0, True),  # found
            (60, 60, 10, 2.0, True),  # missed
            (-5, 30, 10, 2.0, False),  # centre outside: not counted
            (40, 40, 2, 0.4, True),  # below 1 km: not counted
        ]
    )
    pred = pred_frame(
        [
            (20, 20, 10),  # true positive
            (80, 20, 10),  # false positive
            (-4, 30, 10),  # matches the outside crater: neither TP nor FP
            (5, 80, 2),  # under 1 km: ignored
        ]
    )

    t, p = evaluate_tile(pred, truth, size_px=100, scale_m=scale)

    assert t["matched"].tolist() == [True, False]
    assert p["matched"].tolist() == [True, False]
    assert p["diam_km"].tolist() == pytest.approx([2.0, 2.0], rel=1e-6)


def test_summarize_bins_and_pools():
    truth = pd.DataFrame({"diam_km": [1.5, 1.5, 3.0, 40.0], "matched": [True, False, True, True]})
    pred = pd.DataFrame({"diam_km": [1.5, 3.0, 3.0, 40.0], "matched": [True, True, False, True]})

    s = summarize(truth, pred)

    assert s.loc["1-2", "recall"] == 0.5 and s.loc["1-2", "precision"] == 1.0
    assert s.loc["2-4", "recall"] == 1.0 and s.loc["2-4", "precision"] == 0.5
    assert s.loc[">=32", "n_true"] == 1
    assert s.loc["all", "recall"] == 0.75 and s.loc["all", "precision"] == 0.75
    assert s.loc["all", "f1"] == pytest.approx(0.75)
    assert math.isnan(s.loc["4-8", "recall"])  # no craters in the bin
