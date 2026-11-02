"""Load the Robbins lunar crater database as a tidy table."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd

ROBBINS_ZIP_MEMBER = (
    "lunar_crater_database_robbins_2018_bundle/data/lunar_crater_database_robbins_2018.csv"
)

# Source column -> our name. The *_CIRC_IMG columns are the circle fits to rims traced
# on imagery; diameters are in km and longitudes run 0-360 E.
_COLUMNS = {
    "CRATER_ID": "crater_id",
    "LAT_CIRC_IMG": "lat",
    "LON_CIRC_IMG": "lon",
    "DIAM_CIRC_IMG": "diam_km",
}


def load_robbins(path: Path, *, min_diam_km: float = 0.0) -> pd.DataFrame:
    """Read the catalog from the downloaded zip (or the extracted CSV).

    Returns columns ``crater_id``, ``lat``, ``lon`` (wrapped to [-180, 180)) and
    ``diam_km``, keeping craters with ``diam_km >= min_diam_km``.
    """
    path = Path(path)
    read = {"usecols": list(_COLUMNS), "dtype": {"CRATER_ID": "string"}}
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as zf, zf.open(ROBBINS_ZIP_MEMBER) as fh:
            df = pd.read_csv(fh, **read)
    else:
        df = pd.read_csv(path, **read)

    df = df.rename(columns=_COLUMNS)[list(_COLUMNS.values())]
    df["lon"] = (df["lon"] + 180.0) % 360.0 - 180.0
    df = df[df["diam_km"] >= min_diam_km]
    return df.reset_index(drop=True)
