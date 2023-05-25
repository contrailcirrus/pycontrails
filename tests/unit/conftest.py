"""Load fixtures and paths corresponding to files in static directory."""

from __future__ import annotations

import json
import pathlib
from datetime import datetime

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pycontrails import DiskCacheStore, Flight, MetDataArray, MetDataset
from pycontrails.core import cache, met, met_var
from pycontrails.datalib.ecmwf import ERA5
from pycontrails.models.aircraft_performance import AircraftPerformance
from tests import BADA3_PATH, BADA4_PATH, BADA_AVAILABLE

# find default cache dir for testing
DISK_CACHE_DIR = cache._get_user_cache_dir()

################################
# Fixtures in "static" directory
################################


def get_static_path(filename: str | pathlib.Path) -> pathlib.Path:
    """Return a path to file in ``/tests/static/`` directory.

    Parameters
    ----------
    filename : str | pathlib.Path
        Filename to prefix

    Returns
    -------
    pathlib.Path
    """
    parent = pathlib.Path(__file__).parent
    return parent / "static" / filename


@pytest.fixture(scope="session")
def flight_data() -> pd.DataFrame:
    """Load test flight data from csv.

    Scoped for the session.

    Returns
    -------
    pd.DataFrame
    """
    _path = get_static_path("flight.csv")
    parse_dates = ["time"]
    return pd.read_csv(_path, parse_dates=parse_dates)


@pytest.fixture(scope="session")
def flight_attrs() -> dict:
    """Load flight metadata from json.

    Scoped for the session.

    Returns
    -------
    dict
    """
    _path = get_static_path("flight-metadata.json")
    with open(_path) as f:
        return json.load(f)


@pytest.fixture(scope="session")
def override_cache() -> DiskCacheStore:
    """Override cache store while testing so cached met is not found.

    Returns
    -------
    DiskCacheStore
    """
    cache_dir = pathlib.Path(DISK_CACHE_DIR) / "test"
    return DiskCacheStore(cache_dir=cache_dir, allow_clear=True)


@pytest.fixture(scope="session")
def met_pcc_pl(met_ecmwf_pl_path: str, override_cache: DiskCacheStore) -> MetDataset:
    """Met data (pressure levels) for PCC algorithm testing.

    Returns
    -------
    MetDataset
    """

    times = (datetime(2019, 5, 31, 5, 0, 0), datetime(2019, 5, 31, 6, 0, 0))
    pl_variables = ["air_temperature", "specific_humidity", "specific_cloud_ice_water_content"]
    pressure_levels = [300, 250, 225]

    # load
    era5 = ERA5(
        time=times,
        variables=pl_variables,
        pressure_levels=pressure_levels,
        paths=met_ecmwf_pl_path,
        cachestore=override_cache,
    )
    return era5.open_metdataset()


@pytest.fixture(scope="session")
def met_pcc_sl(met_ecmwf_sl_path: str, override_cache: DiskCacheStore) -> MetDataset:
    """Met data (single level) for PCC algorithm testing.

    Returns
    -------
    MetDataset
    """
    times = (datetime(2019, 5, 31, 5, 0, 0), datetime(2019, 5, 31, 6, 0, 0))
    sl_variables = ["surface_air_pressure"]

    # load
    era5 = ERA5(
        time=times,
        variables=sl_variables,
        paths=met_ecmwf_sl_path,
        cachestore=override_cache,
    )
    return era5.open_metdataset()


@pytest.fixture(scope="session")
def met_accf_pl() -> MetDataset:
    """Met data (pressure levels) for PCC algorithm testing.

    Returns
    -------
    MetDataset
    """
    _path = get_static_path("met-accf-pl.nc")
    return MetDataset(xr.open_dataset(_path))


@pytest.fixture(scope="session")
def met_accf_sl() -> MetDataset:
    """Met data (single level) for PCC algorithm testing.

    Returns
    -------
    MetDataset
    """
    _path = get_static_path("met-accf-sl.nc")
    ds = xr.open_dataset(_path)
    ds = ds.expand_dims("level").assign_coords(level=("level", [-1]))
    return MetDataset(ds)


@pytest.fixture(scope="session")
def met_ecmwf_pl_path() -> str:
    """Path to ERA5 data at pressure levels.

    Returns
    -------
    str
    """
    _path = get_static_path("met-ecmwf-pl.nc")
    return str(_path)


@pytest.fixture(scope="session")
def met_ecmwf_sl_path() -> str:
    """Path to ERA5 data at single levels.

    Returns
    -------
    str
    """
    _path = get_static_path("met-ecmwf-sl.nc")
    return str(_path)


@pytest.fixture(scope="session")
def met_issr(met_ecmwf_pl_path: str) -> MetDataset:
    """ISSR output as MetDataset.

    As of pycontrails v0.42.0, there is an issue with dask threading causing
    the test `tests/unit/test_dtypes::test_issr_sac_grid_output[linear-ISSR]`
    to hang. As a workaround, we open without dask parallelism.

    Returns
    -------
    MetDataset
    """
    ds = xr.open_dataset(met_ecmwf_pl_path)
    ds = met.standardize_variables(ds, (met_var.AirTemperature, met_var.SpecificHumidity))
    ds.attrs["met_source"] = "ERA5"
    return MetDataset(ds)


@pytest.fixture
def met_cocip1() -> MetDataset:
    """ERA5 Meteorology to run cocip with ``flight_cocip1``.

    See ``met_cocip1_module_scope`` for a module-scoped version.

    See tests/fixtures/cocip-met.py for domain and creation of the source data file.
    """
    _path = get_static_path("met-era5-cocip1.nc")
    return MetDataset(xr.open_dataset(_path))


@pytest.fixture
def rad_cocip1() -> MetDataset:
    """ERA5 radiation data to run cocip with ``flight_cocip1``.

    See ``rad_cocip1_module_scope`` for a module-scoped version.

    See tests/fixtures/cocip-met.py for domain and creation of the source data file.
    """
    _path = get_static_path("rad-era5-cocip1.nc")
    return MetDataset(xr.open_dataset(_path))


@pytest.fixture(scope="module")
def met_cocip1_module_scope() -> MetDataset:
    """Create met available at module scope."""
    _path = get_static_path("met-era5-cocip1.nc")
    return MetDataset(xr.open_dataset(_path))


@pytest.fixture(scope="module")
def rad_cocip1_module_scope() -> MetDataset:
    """Create rad available at module scope."""
    _path = get_static_path("rad-era5-cocip1.nc")
    return MetDataset(xr.open_dataset(_path))


@pytest.fixture
def met_cocip2() -> MetDataset:
    """ERA5 meteorology to run cocip with ``flight-cocip2.csv``."""
    _path = get_static_path("met-era5-cocip2.nc")
    return MetDataset(xr.open_dataset(_path))


@pytest.fixture
def rad_cocip2() -> MetDataset:
    """ERA5 radiation data to run cocip with ``flight-cocip2.csv``."""
    _path = get_static_path("rad-era5-cocip2.nc")
    return MetDataset(xr.open_dataset(_path))


@pytest.fixture
def met_gfs() -> MetDataset:
    """
    Load GFS met example.

    See tests/fixtures/gfs-met.py for domain and creation of the source data file.
    """
    _path = get_static_path("met-gfs.nc")
    return MetDataset(xr.open_dataset(_path))


@pytest.fixture
def rad_gfs() -> MetDataset:
    """
    Load GFS rad example.

    See tests/fixtures/gfs-met.py for domain and creation of the source data file.
    """
    _path = get_static_path("rad-gfs.nc")
    return MetDataset(xr.open_dataset(_path))


@pytest.fixture(scope="function")  # keep function scoped
def met_era5_fake() -> MetDataset:
    """Create a fake ERA5 MetDataset."""
    shape = (360, 181, 7, 4)

    temp = 230 * np.ones(shape)
    sh = np.zeros(shape)
    for i in range(0, shape[0], 5):
        sh[i, :, :, :] = 1e-3

    ds = xr.Dataset(
        data_vars={
            "air_temperature": (["longitude", "latitude", "level", "time"], temp),
            "specific_humidity": (["longitude", "latitude", "level", "time"], sh),
        },
        coords={
            "longitude": np.linspace(-180, 179, shape[0]),
            "latitude": np.linspace(-90, 90, shape[1]),
            "level": [150, 175, 200, 225, 250, 300, 350],
            "time": pd.date_range("2020 Jan 1 00:00", "2020 Jan 1 03:00", shape[3]),
        },
    )
    return MetDataset(ds)


@pytest.fixture
def flight_cocip1() -> Flight:
    """Keep at function scope (default)."""
    # demo synthetic flight
    attrs = {
        "aircraft_type": "A380",
        "wingspan": 48,
        "n_engine": 2,
        "flight_id": "test",
        "thrust": 0.22,  # thrust
        "nvpm_ei_n": 1.897462e15,
    }

    # Example flight
    df = pd.DataFrame()
    df["longitude"] = np.linspace(-21, -23, 20)
    df["latitude"] = np.linspace(55, 57, 20)
    df["altitude"] = np.linspace(11000, 11500, 20)
    df["engine_efficiency"] = np.linspace(0.34, 0.35, 20)  # ope
    df["fuel_flow"] = np.linspace(2.1, 2.4, 20)  # kg/s
    df["aircraft_mass"] = np.linspace(154445, 154345, 20)  # kg
    df["time"] = pd.date_range("2019-01-01T00:15:00", "2019-01-01T02:30:00", periods=20)
    return Flight(df, attrs=attrs)


@pytest.fixture
def flight_cocip2() -> Flight:
    """Test flight for cocip outputs.

    Compatible with ``met_cocip2`` and ``rad_cocip2``.
    """
    _path = get_static_path("flight-cocip2.csv")
    parse_dates = ["time"]
    return Flight(pd.read_csv(_path, parse_dates=parse_dates))


@pytest.fixture(scope="session")
def flight_meridian() -> Flight:
    """Test flight that crosses the meridian.

    Returns
    -------
    pd.DataFrame
    """
    _path = get_static_path("flight-meridian.csv")
    parse_dates = ["time"]
    df = pd.read_csv(_path, parse_dates=parse_dates)
    return Flight(df)


@pytest.fixture(scope="function")  # keep function scoped
def flight_fake() -> Flight:
    """Fake Flight fixture."""
    n_waypoints = 500

    df = pd.DataFrame(
        {
            "longitude": np.linspace(-45, 143, n_waypoints),
            "latitude": np.linspace(-20, 80, n_waypoints),
            "altitude": np.linspace(4000, 12000, n_waypoints),
            "time": pd.date_range("2020 Jan 1 00:17", "2020 Jan 1 02:36", n_waypoints),
        }
    )

    return Flight(df, attrs=dict(destination="SLC", flight_id="abcde"))


@pytest.fixture
def bada_model() -> AircraftPerformance:
    """Construct generic BADAFlight model."""

    BADAFlight = pytest.importorskip("pycontrails.ext.bada").BADAFlight

    if not BADA_AVAILABLE:
        pytest.skip("BADA data not available")

    params = {
        "bada3_path": BADA3_PATH,
        "bada4_path": BADA4_PATH,
    }
    return BADAFlight(params=params)


@pytest.fixture
def polygon_bug() -> MetDataArray:
    """Read the polygon bug example."""
    _path = get_static_path("polygon-bug.nc")
    da = xr.open_dataarray(_path)
    return MetDataArray(da)
