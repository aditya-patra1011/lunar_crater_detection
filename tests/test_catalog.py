import zipfile

import pytest

from lunar_crater.data.catalog import ROBBINS_ZIP_MEMBER, load_robbins

CSV = (
    "CRATER_ID,LAT_CIRC_IMG,LON_CIRC_IMG,DIAM_CIRC_IMG,ARC_IMG\n"
    "00-1-000000,-19.83,264.757,940.96,0.57\n"
    "00-1-000001,10.0,10.0,0.8,0.9\n"
    "00-1-000002,-5.0,180.0,2.5,0.9\n"
)


@pytest.fixture(params=["csv", "zip"])
def catalog_path(request, tmp_path):
    if request.param == "csv":
        path = tmp_path / "craters.csv"
        path.write_text(CSV)
    else:
        path = tmp_path / "craters.zip"
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr(ROBBINS_ZIP_MEMBER, CSV)
    return path


def test_loads_and_renames_columns(catalog_path):
    df = load_robbins(catalog_path)

    assert list(df.columns) == ["crater_id", "lat", "lon", "diam_km"]
    assert df["crater_id"].tolist() == ["00-1-000000", "00-1-000001", "00-1-000002"]


def test_wraps_longitude(catalog_path):
    df = load_robbins(catalog_path)

    assert df["lon"].tolist() == pytest.approx([-95.243, 10.0, -180.0])


def test_filters_small_craters(catalog_path):
    df = load_robbins(catalog_path, min_diam_km=1.0)

    assert df["crater_id"].tolist() == ["00-1-000000", "00-1-000002"]
