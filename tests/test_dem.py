import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin

from lunar_crater.data.dem import DemRaster
from lunar_crater.data.projection import MOON_RADIUS_M

PPD = 2  # pixels per degree
WIDTH, HEIGHT = 360 * PPD, 180 * PPD
PIXEL_M = 2 * np.pi * MOON_RADIUS_M / WIDTH


def pixel_centres(lon0):
    lat = 90 - (np.arange(HEIGHT) + 0.5) / PPD
    lon = lon0 - 180 + (np.arange(WIDTH) + 0.5) / PPD
    return np.meshgrid(lat, lon, indexing="ij")


def heights(lat, lon):
    return 100.0 * lat + 50.0 * np.cos(np.radians(lon))


def write_dem(path, *, lon0=0.0, scale=1.0, offset=0.0, nodata=None, crs=None):
    lat, lon = pixel_centres(lon0)
    data = np.round((heights(lat, lon) - offset) / scale).astype(np.int32)
    profile = {
        "driver": "GTiff",
        "width": WIDTH,
        "height": HEIGHT,
        "count": 1,
        "dtype": "int32",
        "crs": crs
        or CRS.from_proj4(f"+proj=eqc +lat_ts=0 +lon_0={lon0} +R={MOON_RADIUS_M} +units=m"),
        "transform": from_origin(
            -np.pi * MOON_RADIUS_M, np.pi * MOON_RADIUS_M / 2, PIXEL_M, PIXEL_M
        ),
        "nodata": nodata,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)
        dst.scales, dst.offsets = (scale,), (offset,)
    return path


@pytest.mark.parametrize("lon0", [0.0, 180.0])
def test_samples_pixel_centres_exactly(tmp_path, lon0):
    lat = np.array([0.25, 45.25, -60.75, 89.75])
    lon = np.array([0.25, -120.75, 179.75, 10.25])
    with DemRaster(write_dem(tmp_path / "dem.tif", lon0=lon0)) as dem:
        np.testing.assert_allclose(dem.sample(lat, lon), np.round(heights(lat, lon)), atol=1e-3)


def test_interpolates_between_pixels(tmp_path):
    with DemRaster(write_dem(tmp_path / "dem.tif")) as dem:
        # Halfway between the centres at 10.25 and 10.75 degrees latitude.
        value = dem.sample(np.array([10.5]), np.array([0.25]))

    expected = (np.round(heights(10.25, 0.25)) + np.round(heights(10.75, 0.25))) / 2
    assert value[0] == pytest.approx(expected, abs=1e-3)


def test_wraps_across_the_antimeridian(tmp_path):
    with DemRaster(write_dem(tmp_path / "dem.tif")) as dem:
        across = dem.sample(np.array([0.25]), np.array([180.0]))
        west, east = dem.sample(np.array([0.25, 0.25]), np.array([179.75, -179.75]))

    assert across[0] == pytest.approx((west + east) / 2, abs=1e-3)


@pytest.mark.parametrize("lon0", [0.0, 180.0])
@pytest.mark.parametrize("seam", [0.0, 180.0])
def test_narrow_window_matches_pointwise_sampling(tmp_path, lon0, seam):
    # A small patch straddling a raster edge (lon0 + 180) or not, read as one window.
    lat, lon = np.meshgrid(np.linspace(-1, 1, 9), seam + np.linspace(-1.3, 1.3, 11))
    with DemRaster(write_dem(tmp_path / "dem.tif", lon0=lon0)) as dem:
        together = dem.sample(lat, lon)
        one_by_one = np.array(
            [dem.sample(a, b) for a, b in zip(lat.ravel(), lon.ravel(), strict=True)]
        )

    np.testing.assert_allclose(together.ravel(), one_by_one.ravel(), atol=1e-3)
    np.testing.assert_allclose(together, heights(lat, lon), atol=1.0)


def test_polar_patch_uses_every_longitude(tmp_path):
    lat, lon = np.full(8, 89.9), np.linspace(-180, 135, 8)
    with DemRaster(write_dem(tmp_path / "dem.tif")) as dem:
        values = dem.sample(lat, lon)

    np.testing.assert_allclose(values, heights(lat, lon), atol=30.0)


def test_applies_scale_and_offset(tmp_path):
    path = write_dem(tmp_path / "dem.tif", scale=0.5, offset=20.0)
    with DemRaster(path) as dem:
        value = dem.sample(np.array([30.25]), np.array([40.25]))

    assert value[0] == pytest.approx(heights(30.25, 40.25), abs=0.5)


def test_treats_radius_offset_as_reference_sphere(tmp_path):
    # PDS LDEM labels give OFFSET = 1737400 m; the heights are relative to that sphere.
    path = write_dem(tmp_path / "dem.tif", scale=0.5, offset=0.0)
    with rasterio.open(path, "r+") as dst:
        dst.offsets = (MOON_RADIUS_M,)
    with DemRaster(path) as dem:
        value = dem.sample(np.array([30.25]), np.array([40.25]))

    assert value[0] == pytest.approx(heights(30.25, 40.25), abs=0.5)


def test_nodata_becomes_nan(tmp_path):
    path = write_dem(tmp_path / "dem.tif", nodata=-999)
    with rasterio.open(path, "r+") as dst:
        data = dst.read(1)
        data[0, :] = -999
        dst.write(data, 1)
    with DemRaster(path) as dem:
        values = dem.sample(np.array([89.75, 0.25]), np.array([0.25, 0.25]))

    assert np.isnan(values[0]) and not np.isnan(values[1])


def test_rejects_other_projections(tmp_path):
    path = write_dem(tmp_path / "dem.tif", crs=CRS.from_epsg(3857))
    with pytest.raises(ValueError, match="equirectangular"):
        DemRaster(path)
