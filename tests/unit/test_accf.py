"""Test ACCF model"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pycontrails import Flight, MetDataset
from pycontrails.models.accf import ACCF
from tests.unit import get_static_path

pytest.importorskip("climaccf", reason="climaccf package not available")


@pytest.fixture()
def met_accf_pl() -> MetDataset:
    """Met data (pressure levels) for ACCF algorithm testing.

    Returns
    -------
    MetDataset
    """
    path = get_static_path("met-accf-pl.nc")
    ds = xr.open_dataset(path)
    return MetDataset(ds, provider="ECMWF", dataset="ERA5", product="reanalysis")


@pytest.fixture()
def met_accf_sl() -> MetDataset:
    """Met data (single level) for ACCF algorithm testing.

    Returns
    -------
    MetDataset
    """
    path = get_static_path("met-accf-sl.nc")
    ds = xr.open_dataset(path)
    ds = ds.expand_dims("level").assign_coords(level=("level", [-1]))
    return MetDataset(ds, provider="ECMWF", dataset="ERA5", product="reanalysis")


@pytest.fixture()
def fl() -> Flight:
    """Create a flight for testing."""

    n = 10000
    longitude = np.linspace(45, 75, n) + np.linspace(0, 1, n)
    latitude = np.linspace(50, 60, n) - np.linspace(0, 1, n)
    level = np.full_like(longitude, 225)

    start = np.datetime64("2022-11-11")
    time = pd.date_range(start, start + np.timedelta64(90, "m"), periods=n)
    return Flight(
        longitude=longitude,
        latitude=latitude,
        level=level,
        time=time,
        aircraft_type="B737",
        flight_id=17,
    )


@pytest.mark.parametrize("use_watts", [True, False])
def test_accf_default(
    met_accf_pl: MetDataset,
    met_accf_sl: MetDataset,
    fl: Flight,
    use_watts: bool,
) -> None:
    """Test Default accf algorithm."""

    for name, da in met_accf_sl.data.items():
        assert da.attrs["units"] == "J m**-2"
        if use_watts:
            da = da / 3600.0
            da.attrs["units"] = "W m**-2"
            met_accf_sl.data[name] = da

    accf = ACCF(met=met_accf_pl, surface=met_accf_sl)
    out = accf.eval(fl, forecast_step=6.0)  # data was pinned with forecast_step=6.0

    # Pin some values
    rel = 0.01
    assert np.mean(out["olr"]) == pytest.approx(-35.23, rel=rel)
    assert np.mean(out["aCCF_CO2"]) == pytest.approx(6.94e-16, rel=rel)
    assert np.mean(out["aCCF_NOx"]) == pytest.approx(1.19e-12, rel=rel)
    assert np.mean(out["aCCF_Cont"]) == pytest.approx(9.91e-15, rel=rel)


@pytest.mark.parametrize("hres", [0.25, 0.5, 1.0, 2.0])
def test_accf_grid_horizontal_resolution(
    met_accf_pl: MetDataset,
    met_accf_sl: MetDataset,
    hres: float,
) -> None:
    """Confirm that a custom horizontal resolution is applied to the ACCF weather data."""
    accf = ACCF(met_accf_pl, met_accf_sl, params={"horizontal_resolution": hres})
    accf.eval()

    assert np.all(np.diff(accf.ds["longitude"]) == hres)
    assert np.all(np.diff(accf.ds["latitude"]) == hres)
