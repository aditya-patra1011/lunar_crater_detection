import numpy as np
import pytest

from lunar_crater.data.projection import (
    MOON_RADIUS_M,
    ortho_forward,
    ortho_inverse,
    to_unit_vectors,
    wrap_longitude,
)

CENTRES = [(0.0, 0.0), (45.0, 120.0), (-70.0, -179.5), (89.9, 10.0), (-90.0, 0.0)]


def test_wrap_longitude():
    np.testing.assert_allclose(
        wrap_longitude([0, 180, 190, -180, 359, -190]), [0, -180, -170, -180, -1, 170]
    )


@pytest.mark.parametrize(("lat0", "lon0"), CENTRES)
def test_centre_maps_to_origin(lat0, lon0):
    x, y, visible = ortho_forward(lat0, lon0, lat0, lon0)
    assert abs(x) < 1e-6 and abs(y) < 1e-6 and visible


@pytest.mark.parametrize(("lat0", "lon0"), CENTRES)
def test_round_trip(lat0, lon0):
    xs, ys = np.meshgrid(np.linspace(-50e3, 50e3, 7), np.linspace(-50e3, 50e3, 7))
    lat, lon = ortho_inverse(xs, ys, lat0, lon0)
    x2, y2, visible = ortho_forward(lat, lon, lat0, lon0)

    assert visible.all()
    np.testing.assert_allclose(x2, xs, atol=1e-4)
    np.testing.assert_allclose(y2, ys, atol=1e-4)


def test_axes_point_east_and_north():
    lat, lon = ortho_inverse(1000.0, 0.0, 10.0, 20.0)
    assert lon > 20 and abs(lat - 10) < 1e-3
    lat, lon = ortho_inverse(0.0, 1000.0, 10.0, 20.0)
    assert lat > 10 and abs(lon - 20) < 1e-6


def test_distance_near_centre_matches_arc_length():
    # 0.1 degree of latitude along a meridian is 0.1 * pi/180 * R of arc.
    _, y, _ = ortho_forward(0.1, 0.0, 0.0, 0.0)
    expected = MOON_RADIUS_M * np.sin(np.radians(0.1))
    assert y == pytest.approx(expected)


def test_far_side_is_not_visible():
    *_, visible = ortho_forward(0.0, 180.0, 0.0, 0.0)
    assert not visible


def test_unit_vectors():
    # (0 N, 0 E) -> x, (0 N, 90 E) -> y, north pole -> z
    v = to_unit_vectors([0.0, 0.0, 90.0], [0.0, 90.0, 0.0])
    np.testing.assert_allclose(v, np.eye(3), atol=1e-12)
