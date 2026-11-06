"""Select a day from January 2020 (pre-Covid) to use for CoCiP benchmarking.

For each day in January 2020, evaluate the PCR model on ERA5 over European airspace
25W-45E, 30N-72N, at flight levels FL300 to FL400 every hour. Select the day with
the largest fraction of PCR grid cells.

Run from any directory:

    python tests/benchmark/select_day.py

Write ``day-selection.csv`` next to this file.
"""

from __future__ import annotations

import pathlib
import warnings

import numpy as np
import pandas as pd
import tqdm
import xarray as xr

import pycontrails
from pycontrails.models import pcr
from pycontrails.physics import units

STORE = "gs://contrails-301217-ecmwf-era5-zarr-v2"
LONGITUDE = (-25.0, 45.0)
LATITUDE = (30.0, 72.0)
FLIGHT_LEVELS = np.arange(300, 410, 10)
PRESSURE_LEVELS = units.ft_to_pl(100.0 * FLIGHT_LEVELS)
DAYS = pd.date_range("2020-01-01", "2020-01-31", freq="D")
OUTPUT_DIR = pathlib.Path(__file__).parent


def open_day(day: pd.Timestamp) -> pycontrails.MetDataset:
    """Open ERA5 temperature and specific humidity over the box for ``day``."""
    path = f"{STORE}/{day:%Y%m%d}_pl.zarr"
    ds = xr.open_zarr(path)
    ds = ds[["t", "q"]].rename(t="air_temperature", q="specific_humidity")
    ds = ds.sel(longitude=slice(*LONGITUDE), latitude=slice(*LATITUDE))
    return pycontrails.MetDataset(ds.load(), copy=False)


def pcr_fraction(met: pycontrails.MetDataset) -> float:
    """Return the fraction of grid cells at ``FLIGHT_LEVELS`` flagged as PCR."""
    source = pycontrails.MetDataset.from_coords(
        longitude=met.data["longitude"],
        latitude=met.data["latitude"],
        level=PRESSURE_LEVELS,
        time=met.data["time"],
    )
    with warnings.catch_warnings():
        # The selection rule uses unscaled ERA5 humidity
        warnings.filterwarnings("ignore", message=r"\s*Met data appears to have originated")
        out = pcr.PCR(met).eval(source)
    return out.data["pcr"].mean().item()


def main() -> None:
    """Compute daily PCR fractions, write them and print the selected day."""
    rows = []
    for day in tqdm.tqdm(DAYS):
        met = open_day(day)
        fraction = pcr_fraction(met)
        rows.append({"date": day, "pcr": fraction})
        tqdm.tqdm.write(f"{day:%Y-%m-%d} pcr={fraction:.4f}")

    df = pd.DataFrame(rows).set_index("date")
    df.to_csv(OUTPUT_DIR / "day-selection.csv", float_format="%.6f")
    print(f"selected: {df['pcr'].idxmax():%Y-%m-%d}")


if __name__ == "__main__":
    main()
