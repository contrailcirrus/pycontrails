"""Stage ERA5 met and rad for the CoCiP benchmark.

The benchmark day is 2020-01-30, the day with the largest PCR fraction in
``tests/benchmark/day-selection.csv`` (written by ``select_day.py``).

Copy 2020-01-30 and 2020-01-31 from the daily ERA5 zarr stores over the selection box
(25W-45E, 30N-72N) with a 10 degree buffer on each side for contrail advection.

    python tests/benchmark/stage_met.py

Write ``met.nc`` and ``rad.nc`` to ``OUTPUT``.
"""

from __future__ import annotations

import tempfile

import gcsfs
import pandas as pd
import xarray as xr

from pycontrails.datalib import ecmwf

STORE = "gs://contrails-301217-ecmwf-era5-zarr-v2"
OUTPUT = "gs://contrails-301217-benchmark-data/inputs"
DAYS = pd.date_range("2020-01-30", "2020-01-31", freq="D")
BUFFER_DEG = 10.0
LONGITUDE = (-25.0 - BUFFER_DEG, 45.0 + BUFFER_DEG)
LATITUDE = (30.0 - BUFFER_DEG, 72.0 + BUFFER_DEG)
RENAME = {v.short_name: v.standard_name for v in ecmwf.ECMWF_VARIABLES}


def open_kind(kind: str) -> xr.Dataset:
    """Open both days of the ``kind`` ("pl" or "sl") store, subset to the domain."""
    ds = xr.concat([xr.open_zarr(f"{STORE}/{day:%Y%m%d}_{kind}.zarr") for day in DAYS], dim="time")
    ds = ds.sel(longitude=slice(*LONGITUDE), latitude=slice(*LATITUDE))
    return ds.rename({k: v for k, v in RENAME.items() if k in ds.data_vars})


def main() -> None:
    """Stage met and rad."""
    fs = gcsfs.GCSFileSystem()
    for kind, name in (("pl", "met.nc"), ("sl", "rad.nc")):
        ds = open_kind(kind)
        encoding = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars}
        with tempfile.NamedTemporaryFile(suffix=".nc") as tmp:
            ds.load().to_netcdf(tmp.name, encoding=encoding)  # much faster if .load is used
            fs.put(tmp.name, f"{OUTPUT}/{name}")


if __name__ == "__main__":
    main()
