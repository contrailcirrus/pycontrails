"""Load fixtures and paths corresponding to files in static directory."""

from __future__ import annotations

import json
import pathlib
from datetime import datetime
from typing import Any

import dask
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pycontrails import DiskCacheStore, Flight, MetDataArray, MetDataset
from pycontrails.core import cache, met, met_var
from pycontrails.core.aircraft_performance import AircraftPerformance, AircraftPerformanceGrid
from pycontrails.datalib.ecmwf import ERA5
from tests import BADA3_PATH, BADA4_PATH, BADA_AVAILABLE
from tests.unit import get_static_path

# find default cache dir for testing
DISK_CACHE_DIR = cache._get_user_cache_dir()


# Add command line option for re-generating static output data
def pytest_addoption(parser: pytest.Parser) -> None:
    """Add command line option to regenerate static test results."""
    parser.addoption(
        "--regenerate-results",
        action="store",
        default=False,
        help="Regenerate static test results",
    )


@pytest.fixture()
def regenerate_results(request: pytest.FixtureRequest) -> Any:
    """Regenerate static test results in tests that use them."""
    return request.config.getoption("--regenerate-results")


@pytest.fixture(scope="session")
def flight_data() -> pd.DataFrame:
    """Load test flight data from csv.

    Scoped for the session.

    Returns
    -------
    pd.DataFrame
    """
    path = get_static_path("flight.csv")
    return pd.read_csv(path, parse_dates=["time"])


@pytest.fixture(scope="session")
def flight_attrs() -> dict[str, Any]:
    """Load flight metadata from json.

    Scoped for the session.

    Returns
    -------
    dict
    """
    path = get_static_path("flight-metadata.json")
    with path.open() as f:
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

    The v2024.03.0 release of xarray changes the datatype of decoded variables
    from float32 to float64. This breaks tests that expect variables as float32.
    As a workaround, we manually convert variables to float32 after decoding.

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
    met = era5.open_metdataset()
    met.data = met.data.astype("float32")
    return met


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
def met_ecmwf_pl_path() -> str:
    """Path to ERA5 data at pressure levels.

    Returns
    -------
    str
    """
    path = get_static_path("met-ecmwf-pl.nc")
    return str(path)


@pytest.fixture(scope="session")
def met_ecmwf_sl_path() -> str:
    """Path to ERA5 data at single levels.

    Returns
    -------
    str
    """
    path = get_static_path("met-ecmwf-sl.nc")
    return str(path)


@pytest.fixture(scope="session")
def met_issr(met_ecmwf_pl_path: str) -> MetDataset:
    """ISSR output as MetDataset.

    As of pycontrails v0.42.0, there is an issue with dask threading causing
    the test `tests/unit/test_dtypes::test_issr_sac_grid_output[linear-ISSR]`
    to hang. As a workaround, we open without dask parallelism.

    The v2024.03.0 release of xarray changes the datatype of decoded variables
    from float32 to float64. This breaks tests that expect variables as float32.
    As a workaround, we manually convert variables to float32 after decoding.

    Returns
    -------
    MetDataset
    """
    ds = xr.open_dataset(met_ecmwf_pl_path).astype("float32")
    ds = met.standardize_variables(ds, (met_var.AirTemperature, met_var.SpecificHumidity))
    ds.attrs.update(provider="ECMWF", dataset="ERA5", product="reanalysis")
    return MetDataset(ds)


@pytest.fixture()
def met_cocip1() -> MetDataset:
    """ERA5 Meteorology to run cocip with ``flight_cocip1``.

    See ``met_cocip1_module_scope`` for a module-scoped version.

    See tests/fixtures/cocip-met.py for domain and creation of the source data file.

    The v2024.03.0 release of xarray changes the datatype of decoded variables
    from float32 to float64. This breaks tests that expect variables as float32.
    As a workaround, we manually convert variables to float32 after decoding.
    """
    path = get_static_path("met-era5-cocip1.nc")
    ds = xr.open_dataset(path).astype("float32")
    ds["air_pressure"] = ds["air_pressure"].astype("float32")
    ds["altitude"] = ds["altitude"].astype("float32")
    return MetDataset(ds, provider="ECMWF", dataset="ERA5", product="reanalysis")


@pytest.fixture()
def rad_cocip1() -> MetDataset:
    """ERA5 radiation data to run cocip with ``flight_cocip1``.

    See ``rad_cocip1_module_scope`` for a module-scoped version.

    See tests/fixtures/cocip-met.py for domain and creation of the source data file.
    """
    path = get_static_path("rad-era5-cocip1.nc")
    ds = xr.open_dataset(path)
    return MetDataset(ds, provider="ECMWF", dataset="ERA5", product="reanalysis")


@pytest.fixture()
def met_generic_cocip1() -> MetDataset:
    """Generic meteorology to run cocip with ``flight-cocip1``."""
    path = get_static_path("met-era5-cocip1.nc")
    ds = xr.open_dataset(path).astype("float32")
    ds["air_pressure"] = ds["air_pressure"].astype("float32")
    ds["altitude"] = ds["altitude"].astype("float32")
    ds = ds.rename(
        {
            "specific_cloud_ice_water_content": "mass_fraction_of_cloud_ice_in_air",
            "fraction_of_cloud_cover": "cloud_area_fraction_in_atmosphere_layer",
        }
    )
    return MetDataset(ds, provider="Generic")


@pytest.fixture()
def rad_generic_cocip1() -> MetDataset:
    """Generic radiation data to run cocip with ``flight-cocip1``."""
    path = get_static_path("rad-era5-cocip1.nc")
    ds = xr.open_dataset(path)
    ds = ds.rename(
        {
            "top_net_solar_radiation": "toa_net_downward_shortwave_flux",
            "top_net_thermal_radiation": "toa_outgoing_longwave_flux",
        }
    )
    ds["toa_outgoing_longwave_flux"] *= -1
    ds = ds.assign_coords({"time": ds["time"] - np.timedelta64(30, "m")})
    return MetDataset(ds, provider="Generic")


@pytest.fixture()
def met_cocip_nonuniform_time(met_cocip1: MetDataset) -> MetDataset:
    """Return a MetDataset with nonuniform time."""
    ds = met_cocip1.data
    ds = ds.isel(time=slice(0, 3))
    time = ds["time"].values
    time[2] += np.timedelta64(5, "h")
    ds["time"] = time
    return MetDataset(ds)


@pytest.fixture(scope="module")
def met_cocip1_module_scope() -> MetDataset:
    """Create met available at module scope."""
    path = get_static_path("met-era5-cocip1.nc")
    ds = xr.open_dataset(path)
    return MetDataset(ds, provider="ECMWF", dataset="ERA5", product="reanalysis")


@pytest.fixture(scope="module")
def rad_cocip1_module_scope() -> MetDataset:
    """Create rad available at module scope."""
    path = get_static_path("rad-era5-cocip1.nc")
    ds = xr.open_dataset(path)
    return MetDataset(ds, provider="ECMWF", dataset="ERA5", product="reanalysis")


@pytest.fixture()
def met_cocip2() -> MetDataset:
    """ERA5 meteorology to run cocip with ``flight-cocip2.csv``."""
    path = get_static_path("met-era5-cocip2.nc")
    ds = xr.open_dataset(path)
    return MetDataset(ds, provider="ECMWF", dataset="ERA5", product="reanalysis")


@pytest.fixture()
def rad_cocip2() -> MetDataset:
    """ERA5 radiation data to run cocip with ``flight-cocip2.csv``."""
    path = get_static_path("rad-era5-cocip2.nc")
    ds = xr.open_dataset(path)
    return MetDataset(ds, provider="ECMWF", dataset="ERA5", product="reanalysis")


@pytest.fixture()
def met_generic_cocip2() -> MetDataset:
    """Generic meteorology to run cocip with ``flight-cocip2.csv``."""
    path = get_static_path("met-era5-cocip2.nc")
    ds = xr.open_dataset(path)
    ds = ds.rename(
        {
            "specific_cloud_ice_water_content": "mass_fraction_of_cloud_ice_in_air",
        }
    )
    return MetDataset(ds, provider="Generic")


@pytest.fixture()
def rad_generic_cocip2() -> MetDataset:
    """Generic radiation data to run cocip with ``flight-cocip2.csv``."""
    path = get_static_path("rad-era5-cocip2.nc")
    ds = xr.open_dataset(path)
    ds = ds.rename(
        {
            "top_net_solar_radiation": "toa_net_downward_shortwave_flux",
            "top_net_thermal_radiation": "toa_outgoing_longwave_flux",
        }
    )
    ds["toa_outgoing_longwave_flux"] *= -1
    ds["time"].attrs = {}  # don't mark time as already shifted
    return MetDataset(ds, provider="Generic")


@pytest.fixture()
def met_gfs() -> MetDataset:
    """
    Load GFS met example.

    See tests/fixtures/gfs-met.py for domain and creation of the source data file.
    """
    path = get_static_path("met-gfs.nc")
    ds = xr.open_dataset(path)
    return MetDataset(ds, provider="NCEP", dataset="GFS", product="forecast")


@pytest.fixture()
def rad_gfs() -> MetDataset:
    """
    Load GFS rad example.

    See tests/fixtures/gfs-met.py for domain and creation of the source data file.
    """
    path = get_static_path("rad-gfs.nc")
    ds = xr.open_dataset(path)
    return MetDataset(ds, provider="NCEP", dataset="GFS", product="forecast")


@pytest.fixture()  # keep function scoped
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


@pytest.fixture()
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


@pytest.fixture()
def flight_cocip2() -> Flight:
    """Test flight for cocip outputs.

    Compatible with ``met_cocip2`` and ``rad_cocip2``.
    """
    path = get_static_path("flight-cocip2.csv")
    df = pd.read_csv(path, parse_dates=["time"])
    return Flight(df)


@pytest.fixture(scope="session")
def flight_meridian() -> Flight:
    """Test flight that crosses the meridian.

    Returns
    -------
    pd.DataFrame
    """
    path = get_static_path("flight-meridian.csv")
    df = pd.read_csv(path, parse_dates=["time"])
    return Flight(df)


@pytest.fixture()  # keep function scoped
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


@pytest.fixture()
def bada_model() -> AircraftPerformance:
    """Construct generic ``BADAFlight`` trajectory AP model."""

    BADAFlight = pytest.importorskip("pycontrails.ext.bada", exc_type=ImportError).BADAFlight

    if not BADA_AVAILABLE:
        pytest.skip("BADA data not available")

    params = {"bada3_path": BADA3_PATH, "bada4_path": BADA4_PATH, "engine_deterioration_factor": 0}
    return BADAFlight(params=params)


@pytest.fixture()
def bada_grid_model() -> AircraftPerformanceGrid:
    """Construct generic ``BADAGrid`` gridded AP model."""

    BADAGrid = pytest.importorskip("pycontrails.ext.bada", exc_type=ImportError).BADAGrid

    if not BADA_AVAILABLE:
        pytest.skip("BADA data not available")

    params = {"bada3_path": BADA3_PATH, "bada4_path": BADA4_PATH, "engine_deterioration_factor": 0}
    return BADAGrid(params=params)


@pytest.fixture()
def polygon_bug() -> MetDataArray:
    """Read the polygon bug example."""
    path = get_static_path("polygon-bug.nc")
    da = xr.open_dataarray(path)
    return MetDataArray(da)


@pytest.fixture()
def _dask_single_threaded():
    """Run test using single-threaded dask scheduler.

    As of v0.52.1, using the default multi-threaded scheduler can cause
    some tests to hang while waiting to acquire a lock that is never released.
    This fixture can be used to run those tests using a single-threaded scheduler.
    """
    with dask.config.set(scheduler="single-threaded"):
        yield


@pytest.fixture()
def lnsp() -> xr.DataArray:
    """Load lnsp data for testing."""
    path = get_static_path("met-ecmwf-lnsp.nc")
    return xr.open_dataarray(path)


@pytest.fixture()
def era5_ml() -> xr.Dataset:
    """Load ERA5 data at model levels."""
    path = get_static_path("met-ecmwf-ml.nc")
    return xr.open_dataset(path)
