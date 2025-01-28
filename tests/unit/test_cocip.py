"""Test `Cocip`."""

from __future__ import annotations

import pathlib
import time as pythontime

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pycontrails import Fleet, Flight, MetDataset
from pycontrails.core import met_var
from pycontrails.core.aircraft_performance import AircraftPerformance
from pycontrails.core.met_var import MetVariable
from pycontrails.core.vector import GeoVectorDataset
from pycontrails.datalib.ecmwf import ERA5
from pycontrails.datalib.ecmwf import variables as ecmwf_var
from pycontrails.datalib.gfs import variables as gfs_var
from pycontrails.models import humidity_scaling as hs
from pycontrails.models.cocip import (
    Cocip,
    CocipFlightParams,
    CocipParams,
    contrail_properties,
    radiative_heating,
)
from pycontrails.models.cocip.output_formats import (
    contrail_flight_summary_statistics,
    flight_waypoint_summary_statistics,
    longitude_latitude_grid,
    time_slice_statistics,
)
from pycontrails.models.cocip.radiative_forcing import contrail_contrail_overlap_radiative_effects
from pycontrails.models.humidity_scaling import (
    ConstantHumidityScaling,
    ExponentialBoostHumidityScaling,
    ExponentialBoostLatitudeCorrectionHumidityScaling,
)
from tests.unit import get_static_path


@pytest.fixture()
def met(met_cocip1: MetDataset) -> MetDataset:
    """Rename fixture `met_cocip1` from conftest."""
    return met_cocip1


@pytest.fixture()
def rad(rad_cocip1: MetDataset) -> MetDataset:
    """Rename fixture `rad_cocip1` from conftest."""
    return rad_cocip1


@pytest.fixture()
def hres_dummy_met() -> MetDataset:
    """Generate dummy HRES met data with values equal to the forecast step"""

    time = pd.date_range("2024-01-01T00:00", freq="1h", periods=7)
    level = np.linspace(100, 400, 4)
    latitude = np.linspace(-80, 80, 17)
    longitude = np.linspace(-180, 170, 36)

    shape = (longitude.size, latitude.size, level.size, time.size)
    var = [
        "air_temperature",
        "specific_humidity",
        "eastward_wind",
        "northward_wind",
        "lagrangian_tendency_of_air_pressure",
        "tau_cirrus",
    ]
    ds = xr.Dataset(
        data_vars={
            key: (
                ("longitude", "latitude", "level", "time"),
                np.broadcast_to(
                    np.arange(time.size, dtype="float64").reshape((1, 1, 1, -1)), shape
                ),
            )
            for key in var
        },
        coords={
            "longitude": longitude,
            "latitude": latitude,
            "level": level,
            "time": time,
        },
    )
    attrs = {"provider": "ECMWF", "dataset": "HRES", "product": "forecast"}
    return MetDataset(ds, attrs=attrs)


@pytest.fixture()
def hres_dummy_rad() -> MetDataset:
    """Generate dummy HRES rad data with values equal to the forecast step"""

    time = pd.date_range("2024-01-01T00:00", freq="1h", periods=7)
    latitude = np.linspace(-80, 80, 17)
    longitude = np.linspace(-180, 170, 36)

    shape = (longitude.size, latitude.size, 1, time.size)
    var = ["top_net_solar_radiation", "top_net_thermal_radiation"]
    ds = xr.Dataset(
        data_vars={
            key: (
                ("longitude", "latitude", "level", "time"),
                np.broadcast_to(
                    np.arange(time.size, dtype="float64").reshape((1, 1, 1, -1)), shape
                ),
            )
            for key in var
        },
        coords={
            "longitude": longitude,
            "latitude": latitude,
            "level": [1],
            "time": time,
        },
        attrs={"radiation_accumulated": True},
    )

    attrs = {"provider": "ECMWF", "dataset": "HRES", "product": "forecast"}
    return MetDataset(ds, attrs=attrs)


@pytest.fixture()
def fl(flight_cocip1: Flight) -> Flight:
    """Rename fixture `cocip_fl` from conftest."""
    return flight_cocip1


@pytest.fixture()
def cocip_no_ef(fl: Flight, met: MetDataset, rad: MetDataset) -> Cocip:
    """Return `Cocip` instance evaluated on modified `fl`."""
    fl.update(longitude=np.linspace(-29, -32, 20, dtype=float))
    fl.update(latitude=np.linspace(54, 55, 20, dtype=float))
    fl.update(altitude=np.full(20, 10000.0, dtype=float))

    # set all radiative forcing to 0
    rad2 = rad.copy()
    rad2.data["top_net_solar_radiation"] = xr.zeros_like(rad2.data["top_net_solar_radiation"])
    rad2.data["top_net_thermal_radiation"] = xr.zeros_like(rad2.data["top_net_thermal_radiation"])

    # run - will not find any persistent contrails
    params = {
        "max_age": np.timedelta64(3, "h"),
        "process_emissions": False,
        "humidity_scaling": ExponentialBoostHumidityScaling(),
    }
    cocip = Cocip(met.copy(), rad=rad2, params=params)
    cocip.eval(source=fl)

    return cocip


@pytest.fixture()
def cocip_no_ef_lowmem(fl: Flight, met: MetDataset, rad: MetDataset) -> Cocip:
    """Return `Cocip` instance evaluated on modified `fl` using low-memory interpolation."""
    fl.update(longitude=np.linspace(-29, -32, 20, dtype=float))
    fl.update(latitude=np.linspace(54, 55, 20, dtype=float))
    fl.update(altitude=np.full(20, 10000.0, dtype=float))

    rad2 = rad.copy()
    rad2.data["top_net_solar_radiation"] = xr.zeros_like(rad2.data["top_net_solar_radiation"])
    rad2.data["top_net_thermal_radiation"] = xr.zeros_like(rad2.data["top_net_thermal_radiation"])
    params = {
        "max_age": np.timedelta64(3, "h"),
        "process_emissions": False,
        "humidity_scaling": ExponentialBoostHumidityScaling(),
        "preprocess_lowmem": True,
    }
    cocip = Cocip(met.copy(), rad=rad2, params=params)
    cocip.eval(source=fl)
    return cocip


@pytest.fixture()
def cocip_no_ef_lowmem_indices(fl: Flight, met: MetDataset, rad: MetDataset) -> Cocip:
    """Return `Cocip` instance evaluated on modified `fl` using low-memory interp + indices."""
    fl.update(longitude=np.linspace(-29, -32, 20, dtype=float))
    fl.update(latitude=np.linspace(54, 55, 20, dtype=float))
    fl.update(altitude=np.full(20, 10000.0, dtype=float))
    rad2 = rad.copy()
    rad2.data["top_net_solar_radiation"] = xr.zeros_like(rad2.data["top_net_solar_radiation"])
    rad2.data["top_net_thermal_radiation"] = xr.zeros_like(rad2.data["top_net_thermal_radiation"])
    params = {
        "max_age": np.timedelta64(3, "h"),
        "process_emissions": False,
        "humidity_scaling": ExponentialBoostHumidityScaling(),
        "preprocess_lowmem": True,
        "interpolation_use_indices": True,
    }
    cocip = Cocip(met.copy(), rad=rad2, params=params)
    cocip.eval(source=fl)
    return cocip


@pytest.fixture()
def cocip_no_ef_generic(
    fl: Flight, met_generic_cocip1: MetDataset, rad_generic_cocip1: MetDataset
) -> Cocip:
    """Return `Cocip` instance evaluated on modified `fl` using generic met data."""
    fl.update(longitude=np.linspace(-29, -32, 20, dtype=float))
    fl.update(latitude=np.linspace(54, 55, 20, dtype=float))
    fl.update(altitude=np.full(20, 10000.0, dtype=float))

    # set all radiative forcing to 0
    rad2 = rad_generic_cocip1.copy()
    rad2.data["toa_net_downward_shortwave_flux"] = xr.zeros_like(
        rad2.data["toa_net_downward_shortwave_flux"]
    )
    rad2.data["toa_outgoing_longwave_flux"] = xr.zeros_like(rad2.data["toa_outgoing_longwave_flux"])

    # run - will not find any persistent contrails
    params = {
        "max_age": np.timedelta64(3, "h"),
        "process_emissions": False,
        "humidity_scaling": ExponentialBoostHumidityScaling(),
    }
    with pytest.warns(UserWarning, match="Unknown provider 'Generic'"):
        cocip = Cocip(met_generic_cocip1.copy(), rad=rad2, params=params)
    cocip.eval(source=fl)

    return cocip


@pytest.fixture()
def cocip_persistent(fl: Flight, met: MetDataset, rad: MetDataset) -> Cocip:
    """Return `Cocip` instance evaluated on modified `fl`."""
    fl.update(longitude=np.linspace(-29, -32, 20))
    fl.update(latitude=np.linspace(56, 57, 20))
    fl.update(altitude=np.linspace(10900, 10900, 20))

    # Use a non-default met_time_buffer for backwards compatibility with original
    # pinned test data. This only affects the test_grid_cirrus test below. Flight
    # waypoint data remains unchanged.
    params = {
        "max_age": np.timedelta64(3, "h"),
        "process_emissions": False,
        "verbose_outputs": True,
        "met_time_buffer": (np.timedelta64(0, "h"), np.timedelta64(1, "h")),
        "humidity_scaling": ExponentialBoostHumidityScaling(),
        "compute_atr20": True,
    }
    cocip = Cocip(met.copy(), rad=rad.copy(), params=params)

    # Eventually the advected waypoints will blow out of bounds
    # Specifically, there is a contrail at 4:30 with latitude larger than 59
    # We acknowledge this here
    with pytest.warns(UserWarning, match="At time .* contrail has no intersection with the met"):
        cocip.eval(source=fl)

    return cocip


@pytest.fixture()
def cocip_persistent_lowmem(fl: Flight, met: MetDataset, rad: MetDataset) -> Cocip:
    """Return `Cocip` instance evaluated on modified `fl` using low-memory interpolation."""
    fl.update(longitude=np.linspace(-29, -32, 20))
    fl.update(latitude=np.linspace(56, 57, 20))
    fl.update(altitude=np.linspace(10900, 10900, 20))
    params = {
        "max_age": np.timedelta64(3, "h"),
        "process_emissions": False,
        "verbose_outputs": True,
        "met_time_buffer": (np.timedelta64(0, "h"), np.timedelta64(1, "h")),
        "humidity_scaling": ExponentialBoostHumidityScaling(),
        "compute_atr20": True,
        "preprocess_lowmem": True,
    }
    cocip = Cocip(met.copy(), rad=rad.copy(), params=params)
    with pytest.warns(UserWarning, match="At time .* contrail has no intersection with the met"):
        cocip.eval(source=fl)
    return cocip


@pytest.fixture()
def cocip_persistent_lowmem_indices(fl: Flight, met: MetDataset, rad: MetDataset) -> Cocip:
    """Return `Cocip` instance evaluated on modified `fl` using low-mem interpolation + indices."""
    fl.update(longitude=np.linspace(-29, -32, 20))
    fl.update(latitude=np.linspace(56, 57, 20))
    fl.update(altitude=np.linspace(10900, 10900, 20))
    params = {
        "max_age": np.timedelta64(3, "h"),
        "process_emissions": False,
        "verbose_outputs": True,
        "met_time_buffer": (np.timedelta64(0, "h"), np.timedelta64(1, "h")),
        "humidity_scaling": ExponentialBoostHumidityScaling(),
        "compute_atr20": True,
        "preprocess_lowmem": True,
        "interpolation_use_indices": True,
    }
    cocip = Cocip(met.copy(), rad=rad.copy(), params=params)
    with pytest.warns(UserWarning, match="At time .* contrail has no intersection with the met"):
        cocip.eval(source=fl)
    return cocip


@pytest.fixture()
def cocip_persistent_generic(
    fl: Flight, met_generic_cocip1: MetDataset, rad_generic_cocip1: MetDataset
) -> Cocip:
    """Return `Cocip` instance evaluated on modified `fl` with generic met data."""
    fl.update(longitude=np.linspace(-29, -32, 20))
    fl.update(latitude=np.linspace(56, 57, 20))
    fl.update(altitude=np.linspace(10900, 10900, 20))

    # Use a non-default met_time_buffer for backwards compatibility with original
    # pinned test data. This only affects the test_grid_cirrus test below. Flight
    # waypoint data remains unchanged.
    params = {
        "max_age": np.timedelta64(3, "h"),
        "process_emissions": False,
        "verbose_outputs": True,
        "met_time_buffer": (np.timedelta64(0, "h"), np.timedelta64(1, "h")),
        "humidity_scaling": ExponentialBoostHumidityScaling(),
        "compute_atr20": True,
    }
    with pytest.warns(UserWarning, match="Unknown provider 'Generic'"):
        cocip = Cocip(met_generic_cocip1.copy(), rad=rad_generic_cocip1.copy(), params=params)

    # Eventually the advected waypoints will blow out of bounds
    # Specifically, there is a contrail at 4:30 with latitude larger than 59
    # We acknowledge this here
    with pytest.warns(UserWarning, match="At time .* contrail has no intersection with the met"):
        cocip.eval(source=fl)

    return cocip


@pytest.fixture()
def cocip_persistent2(
    flight_cocip2: Flight, met_cocip2: MetDataset, rad_cocip2: MetDataset
) -> Cocip:
    """Return ``Cocip`` instance evaluated on modified ``flight_cocip2``."""

    # Use a non-default met_time_buffer for backwards compatibility with original
    # pinned test data. This only affects the test_grid_cirrus test below. Flight
    # waypoint data remains unchanged.
    params = {
        "max_age": np.timedelta64(5, "h"),
        "process_emissions": False,
        "verbose_outputs": True,
        "interpolation_bounds_error": True,
        "humidity_scaling": ExponentialBoostHumidityScaling(),
    }
    cocip = Cocip(met=met_cocip2.copy(), rad=rad_cocip2.copy(), params=params)
    cocip.eval(source=flight_cocip2)

    return cocip


@pytest.fixture()
def cocip_persistent2_lowmem(
    flight_cocip2: Flight, met_cocip2: MetDataset, rad_cocip2: MetDataset
) -> Cocip:
    """Return ``Cocip`` instance evaluated on ``flight_cocip2`` using low-memory interpolation."""
    params = {
        "max_age": np.timedelta64(5, "h"),
        "process_emissions": False,
        "verbose_outputs": True,
        "interpolation_bounds_error": True,
        "humidity_scaling": ExponentialBoostHumidityScaling(),
        "preprocess_lowmem": True,
    }
    cocip = Cocip(met=met_cocip2.copy(), rad=rad_cocip2.copy(), params=params)
    cocip.eval(source=flight_cocip2)
    return cocip


@pytest.fixture()
def cocip_persistent2_lowmem_indices(
    flight_cocip2: Flight, met_cocip2: MetDataset, rad_cocip2: MetDataset
) -> Cocip:
    """Return ``Cocip`` instance evaluated on ``flight_cocip2`` using low-mem interp + indices."""
    params = {
        "max_age": np.timedelta64(5, "h"),
        "process_emissions": False,
        "verbose_outputs": True,
        "interpolation_bounds_error": True,
        "humidity_scaling": ExponentialBoostHumidityScaling(),
        "preprocess_lowmem": True,
        "interpolation_use_indices": True,
    }
    cocip = Cocip(met=met_cocip2.copy(), rad=rad_cocip2.copy(), params=params)
    cocip.eval(source=flight_cocip2)
    return cocip


@pytest.fixture()
def cocip_persistent2_generic(
    flight_cocip2: Flight, met_generic_cocip2: MetDataset, rad_generic_cocip2: MetDataset
) -> Cocip:
    """Return ``Cocip`` instance evaluated on modified ``flight_cocip2`` with generic met data."""

    # Use a non-default met_time_buffer for backwards compatibility with original
    # pinned test data. This only affects the test_grid_cirrus test below. Flight
    # waypoint data remains unchanged.
    params = {
        "max_age": np.timedelta64(5, "h"),
        "process_emissions": False,
        "verbose_outputs": True,
        "interpolation_bounds_error": True,
        "humidity_scaling": ExponentialBoostHumidityScaling(),
    }
    with pytest.warns(UserWarning, match="Unknown provider 'Generic'"):
        cocip = Cocip(met=met_generic_cocip2.copy(), rad=rad_generic_cocip2.copy(), params=params)
    cocip.eval(source=flight_cocip2)

    return cocip


def test_init_model(met: MetDataset, rad: MetDataset) -> None:
    """Test `Cocip` initialization."""
    # require rad
    with pytest.raises(TypeError, match="positional argument: 'rad'"):
        Cocip(met=met)

    # warn if humidity scaling not specified
    with pytest.warns(UserWarning, match="originated from ECMWF"):
        Cocip(met=met, rad=rad)

    # init
    cocip = Cocip(met=met, rad=rad, humidity_scaling=ExponentialBoostHumidityScaling())

    # default params
    assert isinstance(cocip.params, dict)
    cp = CocipFlightParams().as_dict()
    for key in cp:
        if key == "humidity_scaling":
            continue
        v1 = cocip.params[key]
        v2 = cp[key]
        try:
            assert v1 == v2 or (np.isnan(v1) and np.isnan(v2))
        except ValueError:  # np.ndarray
            np.testing.assert_array_equal(v1, v2)

    # model hash
    assert isinstance(cocip.hash, str)

    # inputs
    assert cocip.met is met
    assert cocip.rad is rad

    # outputs
    assert cocip.contrail is None
    assert cocip.contrail_dataset is None


def test_cocip_processes_met(met: MetDataset, rad: MetDataset) -> None:
    """Check that Cocip seamlessly processes met data on init."""

    # met starts without tau_cirrus
    assert "tau_cirrus" not in met
    met_copy = met.copy()
    met_chunked = met.copy()
    met_chunked.data = met_chunked.data.chunk(2)
    met_chunked_copy = met_chunked.copy()
    q = met["specific_humidity"].values.copy()

    cocip = Cocip(met=met, rad=rad, humidity_scaling=ExponentialBoostHumidityScaling())

    # tau_cirrus not added to met in __init__ since it isn't dask-backed
    assert cocip.met is met
    assert "tau_cirrus" not in met

    # specific humidity no longer modified on met
    np.testing.assert_array_equal(q, met["specific_humidity"].values)

    # There is not longer any issue in instantiating another model
    another_cocip = Cocip(met=met, rad=rad, humidity_scaling=ExponentialBoostHumidityScaling())
    assert another_cocip.met is met

    cocip_3 = Cocip(met=met_chunked, rad=rad, humidity_scaling=ExponentialBoostHumidityScaling())

    # tau_cirrus is added to met when it is dask-backed
    assert cocip_3.met is met_chunked
    assert "tau_cirrus" in met_chunked

    cocip_4 = Cocip(
        met=met_chunked_copy,
        rad=rad,
        humidity_scaling=ExponentialBoostHumidityScaling(),
        compute_tau_cirrus_in_model_init=False,
    )

    # tau_cirrus is not added to met when it is dask-backed but the param is False
    assert cocip_4.met is met_chunked_copy
    assert "tau_cirrus" not in met_chunked_copy

    cocip_5 = Cocip(
        met=met_copy,
        rad=rad,
        humidity_scaling=ExponentialBoostHumidityScaling(),
        compute_tau_cirrus_in_model_init=True,
    )

    # tau_cirrus is added to met when it is not dask-backed but the param is True
    assert cocip_5.met is met_copy
    assert "tau_cirrus" in met_copy


@pytest.mark.parametrize("drop_geopotential", [True, False])
def test_cocip_processes_met_geopotential(
    met: MetDataset, rad: MetDataset, drop_geopotential: bool
) -> None:
    """Check that Cocip does not require geopotential data.

    This test was added in version 0.48.0 after geopotential data was made optional.
    """
    if drop_geopotential:
        met.data = met.data.drop_vars("geopotential")

    assert "tau_cirrus" not in met

    # Instantiating Cocip mutates met in place by adding tau_cirrus
    with pytest.warns(UserWarning, match="humidity scaling"):
        Cocip(met=met, rad=rad, compute_tau_cirrus_in_model_init=True)

    # Pin values for tau_cirrus as a way to check that the geopotential
    # calculation was done correctly
    assert "tau_cirrus" in met

    sum_tau_cirrus = float(met.data["tau_cirrus"].sum())
    if drop_geopotential:
        assert sum_tau_cirrus == pytest.approx(98.3, rel=0.1)
    else:
        assert sum_tau_cirrus == pytest.approx(96.1, rel=0.1)


def test_cocip_processes_rad(met: MetDataset, rad: MetDataset) -> None:
    """Check that Cocip seamlessly processes rad data on init."""

    # rad starts without time shifting
    t0 = pd.date_range("2019", freq="1h", periods=13)
    np.testing.assert_array_equal(rad["time"].values, t0)

    t1 = t0 - pd.Timedelta(30, "m")

    assert "shift_radiation_time" not in rad["time"].attrs
    with pytest.warns(UserWarning, match="humidity scaling"):
        cocip = Cocip(met=met, rad=rad, compute_tau_cirrus_in_model_init=True)

    # rad time shifted in __init__
    assert cocip.rad is rad
    assert "shift_radiation_time" in rad["time"].attrs
    assert rad["time"].attrs["shift_radiation_time"] == "-30 minutes"
    np.testing.assert_array_equal(rad["time"].values, t1)

    # should run again without issue
    with pytest.warns(UserWarning, match="humidity scaling"):
        another_cocip = Cocip(met=met, rad=rad)

    assert another_cocip.rad is rad

    # radiation not shifted a second time
    np.testing.assert_array_equal(rad["time"].values, t1)

    # drop a required rad variable, get an error
    rad.data = rad.data.drop_vars(["top_net_thermal_radiation"])

    # Delete some attributes to avoid the humidity scaling warning
    del met.attrs["provider"]
    del met.attrs["history"]

    with pytest.raises(KeyError, match="Dataset does not contain variable"):
        Cocip(met=met, rad=rad)


def test_cocip_processes_rad_with_warnings(met: MetDataset, rad: MetDataset) -> None:
    """Check that Cocip issues warnings for when rad data is different from expected."""

    # Delete some attributes to avoid the humidity scaling warning
    del met.attrs["provider"]
    del met.attrs["history"]

    # Downselect rad data to a subset of the expected time range
    rad.data = rad.data.isel(time=slice(0, 10, 2))

    with pytest.warns(UserWarning, match="Shifting radiation time dimension by unexpected"):
        Cocip(met, rad=rad)


def test_cocip_preserves_hres_attrs(hres_dummy_met: MetDataset, hres_dummy_rad: MetDataset) -> None:
    """Test that Cocip preserves dataset-level attributes after preprocessing met and rad data"""

    met = hres_dummy_met
    rad = hres_dummy_rad
    rad_attrs = rad.data.attrs
    met_attrs = met.data.attrs

    with pytest.warns(UserWarning, match="humidity scaling"):
        Cocip(met=met, rad=rad)
    assert met_attrs == met.data.attrs
    assert rad_attrs == rad.data.attrs


def test_cocip_processes_hres_rad_even_time(
    hres_dummy_met: MetDataset, hres_dummy_rad: MetDataset
) -> None:
    """Test that Cocip correctly processes HRES radiation data with even time steps"""

    met = hres_dummy_met
    rad = hres_dummy_rad

    with pytest.warns(UserWarning, match="humidity scaling"):
        Cocip(met=met, rad=rad)
    t = pd.date_range("2024-01-01T00:30", freq="1h", periods=6)
    shift = str(-np.timedelta64(30, "m").astype("timedelta64[ns]"))
    np.testing.assert_array_equal(rad["time"].values, t)
    assert rad["time"].attrs["shift_radiation_time"] == shift
    np.testing.assert_allclose(rad["top_net_solar_radiation"].values, 1)


def test_cocip_processes_hres_rad_uneven_time(
    hres_dummy_met: MetDataset, hres_dummy_rad: MetDataset
) -> None:
    """Test that Cocip correctly processes HRES radiation data with uneven time steps"""

    met = hres_dummy_met
    rad = hres_dummy_rad

    steps = [0, 1, 3, 6]
    met.data = met.data.isel(time=steps)
    rad.data = rad.data.isel(time=steps)

    t = [np.datetime64("2024-01-01T00") + np.timedelta64(step, "h") for step in steps]
    np.testing.assert_array_equal(rad["time"].values, t)

    with pytest.warns(UserWarning, match="humidity scaling"):
        Cocip(met=met, rad=rad)
    t = [
        np.datetime64("2024-01-01T00:30"),
        np.datetime64("2024-01-01T02:00"),
        np.datetime64("2024-01-01T04:30"),
    ]
    np.testing.assert_array_equal(rad["time"].values, t)
    assert rad["time"].attrs["shift_radiation_time"] == "variable"
    np.testing.assert_allclose(rad["top_net_solar_radiation"].values, 1)


def test_cocip_time_handling(fl: Flight, met: MetDataset, rad: MetDataset) -> None:
    """Check a few aspects of Cocip time handling."""
    params = {
        "max_age": np.timedelta64(4, "h"),
        "process_emissions": False,
        "humidity_scaling": ExponentialBoostHumidityScaling(),
    }
    cocip = Cocip(met, rad=rad.copy(), params=params)

    # Here, the timesteps attribute is not created until eval is called
    cocip.eval(source=fl)
    assert cocip.timesteps.size == 13

    # In general, we won't get equality here. But we do for this particular flight.
    assert cocip.timesteps[-1] == fl["time"][-1] + params["max_age"]
    assert fl["time"][0] < cocip.timesteps[0] < fl["time"][0] + cocip.params["dt_integration"]

    params["dt_integration"] = np.timedelta64(15, "m")
    cocip = Cocip(met, rad=rad.copy(), params=params)
    cocip.eval(source=fl)
    assert len(cocip.timesteps) == 26


def test_eval_setup(fl: Flight, met: MetDataset, rad: MetDataset) -> None:
    """Check `Cocip` logic appearing at start of `eval` method."""
    # required flight parameters
    del fl.attrs["flight_id"]
    cocip = Cocip(
        met=met,
        rad=rad,
        process_emissions=False,
        humidity_scaling=ExponentialBoostHumidityScaling(),
    )
    match = "Source flight does not contain `flight_id` data or attr. Adding `flight_id` of 0"
    with pytest.warns(UserWarning, match=match):
        cocip.eval(source=fl)


def test_flight_overrides(fl: Flight, met: MetDataset, rad: MetDataset) -> None:
    """Ensure flight keys not overwritten by `Cocip`."""
    fl.update(longitude=np.linspace(-29, -32, 20, dtype=float))
    fl.update(latitude=np.linspace(51, 58, 20, dtype=float))
    fl.update(altitude=np.full(20, 10000.0, dtype=float))

    # Add flight / met properties that override default inputs
    fl2 = fl.copy()
    fl2["true_airspeed"] = np.full(fl2.size, 40.0)
    fl2["segment_length"] = np.full(fl2.size, 10000.0)
    fl2["air_temperature"] = np.full(fl2.size, 220.0)

    # Here, fl1 has gaps > 40000 (default) between segments
    cocip = Cocip(
        met=met.copy(),
        rad=rad.copy(),
        process_emissions=False,
        max_seg_length_m=50000,
        humidity_scaling=ExponentialBoostHumidityScaling(),
    )

    # Eventually the advected waypoints will blow out of bounds
    # Specifically, there is a contrail at 4:30 with latitude larger than 59
    # We acknowledge this here
    match = "At time 2019-01-01T04:30:00.000000, the contrail has no intersection with the met"
    with pytest.warns(UserWarning, match=match):
        out1 = cocip.eval(fl)
    with pytest.warns(UserWarning, match=match):
        out2 = cocip.eval(fl2)

    # Because Cocip copies fl, the original isn't modified
    assert "segment_length" not in fl
    assert "true_airspeed" not in fl

    # We see the segment properties on the output
    assert np.all(out1["segment_length"][:-1] > 42000)
    assert np.all(out1["segment_length"][:-1] < 43000)
    assert np.all(out1["true_airspeed"][:-1] > 62)
    assert np.all(out1["true_airspeed"][:-1] < 69)

    # And when segment_length and true_airspeed are already on the flight,
    # the values are unchanged
    assert np.all(out2["segment_length"] == 10000)
    assert np.all(out2["true_airspeed"] == 40)
    assert np.all(out2["air_temperature"] == 220)

    # AND, these variables do impact the model output!
    assert out1["cocip"][18] == 0
    assert out2["cocip"][18] == 1

    # The rest are equal
    filt = np.ones(len(out1), dtype=bool)
    filt[18] = False
    np.testing.assert_array_equal(out1["cocip"][filt], out2["cocip"][filt])

    # Test "waypoint" data provided with flight
    fl3 = fl.copy()
    fl3["waypoint"] = np.arange(len(fl))
    # Using a short max age to avoid advecting out of bounds
    cocip.params["max_age"] = np.timedelta64(1, "h")
    out = cocip.eval(fl3)

    # No nans at positive predictions
    assert out.dataframe[out.dataframe.ef > 0].notna().all().all()

    fl3["waypoint"][14:20] += 1
    with pytest.raises(ValueError, match="non-sequential waypoints"):
        cocip.eval(fl3)


def test_flight_overrides_emissions(
    fl: Flight, met: MetDataset, rad: MetDataset, bada_model: AircraftPerformance
) -> None:
    """Ensure flight keys not overwritten by `Cocip` with emissions modeling."""

    fl4 = fl.copy()
    del fl4.data["aircraft_mass"]  # this will throw a warning, which causes an error
    del fl4.attrs["nvpm_ei_n"]
    nvpm_ei_n = np.linspace(1e15, 2e15, fl.size)
    fl4["nvpm_ei_n"] = nvpm_ei_n

    params = {
        "process_emissions": True,
        "aircraft_performance": bada_model,
        "humidity_scaling": ExponentialBoostHumidityScaling(),
    }
    cocip = Cocip(met=met.copy(), rad=rad.copy(), params=params)
    out4 = cocip.eval(fl4)

    assert np.all(out4["aircraft_mass"] != fl.data["aircraft_mass"])
    assert np.all(out4["nvpm_ei_n"] == nvpm_ei_n)
    assert np.all(out4["engine_efficiency"] == fl4.data["engine_efficiency"])
    assert np.all(out4["fuel_flow"] == fl4.data["fuel_flow"])
    assert np.all(out4["thrust"] == fl4.attrs["thrust"])
    assert out4.attrs["wingspan"] == fl4.attrs["wingspan"]


@pytest.mark.filterwarnings("ignore:distutils Version classes are deprecated")
def test_flight_output(fl: Flight, met: MetDataset, rad: MetDataset) -> None:
    """Check `Cocip` outputs against data in static directory."""
    cocip = Cocip(
        met,
        rad=rad,
        process_emissions=False,
        humidity_scaling=ExponentialBoostHumidityScaling(),
    )
    out = cocip.eval(source=fl)

    assert isinstance(out, Flight)
    assert out is not fl
    assert np.all(out["latitude"] == fl["latitude"])
    assert np.all(out["longitude"] == fl["longitude"])
    assert np.all(out["time"] == fl["time"])
    assert np.all(out.level == out.level)

    assert "rf_net_mean" in cocip.source
    assert "ef" in cocip.source
    assert "cocip" in cocip.source

    assert out.attrs["flight_id"] == fl.attrs["flight_id"]

    # make sure we can output flight dataframe
    # this only works with fastparquet because `age` which is not handled by pyarrow correctly
    out.dataframe.to_parquet(".test.pq", engine="fastparquet")
    assert pathlib.Path(".test.pq").exists()

    # note that fastparquet truncastes to [us],
    # so when we read this back in the last 3 digits are off
    # pyarrow doesn't support timedelta64 past "us", but pandas won't
    # allow casting to "us" in a dataframe - bit of a mess
    df = pd.read_parquet(".test.pq", engine="fastparquet")
    np.testing.assert_allclose(df["contrail_age"].values, out.dataframe["contrail_age"].values)

    # clean up
    pathlib.Path(".test.pq").unlink()

    # make sure we can output flight dataframe using pyarrow with age in seconds
    out.update(contrail_age=out["contrail_age"] / np.timedelta64(1, "s"))
    out.update(time=out["time"].astype("datetime64[us]"))
    out.dataframe.to_parquet(".test2.pq", engine="pyarrow")
    assert pathlib.Path(".test2.pq").exists()

    df = pd.read_parquet(".test2.pq", engine="pyarrow")
    np.testing.assert_allclose(df["contrail_age"].values, out.dataframe["contrail_age"].values)

    # clean up
    pathlib.Path(".test2.pq").unlink()


def test_eval_no_ef(cocip_no_ef: Cocip) -> None:
    """Confirm pinned values of `cocip_no_eval` fixture."""
    # Confirm no nan in output besides the terminal waypoint that is always nan
    assert cocip_no_ef._sac_flight.dataframe.iloc[:-1].notna().all().all()
    assert cocip_no_ef._downwash_flight.dataframe.iloc[:-1].notna().all().all()

    # Pin sizes
    assert cocip_no_ef._sac_flight.size == 20
    assert cocip_no_ef._downwash_flight.size == 19

    assert "rf_net_mean" in cocip_no_ef.source
    assert "ef" in cocip_no_ef.source
    assert "persistent_1" in cocip_no_ef.source

    # Flight doesn't produce any persistent contrails
    assert np.all(cocip_no_ef.source["ef"] == 0)
    assert np.all(cocip_no_ef.source["cocip"] == 0)
    assert np.all(cocip_no_ef.source["contrail_age"] == np.timedelta64(0, "ns"))

    # BUT it does satisfy the SAC
    assert np.all(cocip_no_ef.source["sac"] == 1)

    # And initial persistent
    assert np.all(cocip_no_ef.source["persistent_1"][:-1] == 1)
    assert cocip_no_ef.source["persistent_1"][-1] == 0


def test_eval_persistent(cocip_persistent: Cocip, regenerate_results: bool) -> None:
    """Confirm pinned values of `cocip_persistent` fixture."""
    assert cocip_persistent.timesteps.size == 11

    assert "rf_net_mean" in cocip_persistent.source
    assert "ef" in cocip_persistent.source
    assert "cocip" in cocip_persistent.source
    assert np.nansum(cocip_persistent.source["cocip"]) > 0
    assert cocip_persistent.contrail is not None

    # output json when algorithm has been adjusted
    # controlled by command line option --regenerate-results
    if regenerate_results:
        cocip_persistent.source.dataframe.to_json(
            get_static_path("cocip-flight-output.json"),
            indent=2,
            orient="records",
            date_unit="ns",
            double_precision=15,
        )
        cocip_persistent.contrail.to_json(
            get_static_path("cocip-contrail-output.json"),
            indent=2,
            orient="records",
            date_unit="ns",
            double_precision=15,
        )

    flight_output = pd.read_json(get_static_path("cocip-flight-output.json"), orient="records")
    contrail_output = pd.read_json(get_static_path("cocip-contrail-output.json"), orient="records")

    flight_output["time"] = pd.to_datetime(flight_output["time"])
    flight_output["contrail_age"] = pd.to_timedelta(flight_output["contrail_age"])
    contrail_output["time"] = pd.to_datetime(contrail_output["time"])
    contrail_output["formation_time"] = pd.to_datetime(contrail_output["formation_time"])
    contrail_output["age"] = pd.to_timedelta(contrail_output["age"])
    contrail_output["dt_integration"] = pd.to_timedelta(contrail_output["dt_integration"])

    rtol = 1e-4
    for key in flight_output:
        if key in ["time", "flight_id"]:
            np.testing.assert_array_equal(cocip_persistent.source[key], flight_output[key])
            continue
        if key == "atr20":
            np.testing.assert_allclose(
                cocip_persistent.source.get_data_or_attr(key),
                flight_output[key],
                err_msg=key,
                rtol=1e-2,
            )
            continue
        if key in ["tau_cirrus", "specific_cloud_ice_water_content"]:
            np.testing.assert_allclose(
                cocip_persistent.source.get_data_or_attr(key),
                flight_output[key],
                err_msg=key,
                rtol=rtol,
                atol=1e-14,  # for trace cloud ice
            )
            continue
        np.testing.assert_allclose(
            cocip_persistent.source.get_data_or_attr(key),
            flight_output[key],
            err_msg=key,
            rtol=rtol,
        )

    pd.testing.assert_frame_equal(
        cocip_persistent.contrail.reset_index(drop=True),
        contrail_output.reset_index(drop=True),
        check_dtype=False,
        check_like=True,  # ignore column order
        check_exact=False,
        rtol=rtol,
    )


def test_eval_persistent2(cocip_persistent2: Cocip, regenerate_results: bool) -> None:
    """Confirm pinned values of ``cocip_persistent2`` fixture."""
    assert cocip_persistent2.timesteps.size == 16

    assert "rf_net_mean" in cocip_persistent2.source
    assert "ef" in cocip_persistent2.source
    assert "cocip" in cocip_persistent2.source
    assert np.all(np.isfinite(cocip_persistent2.source["ef"]))
    assert np.all(np.isfinite(cocip_persistent2.source["cocip"]))

    assert np.sum(cocip_persistent2.source["cocip"]) > 0
    assert cocip_persistent2.contrail is not None

    # output json when algorithm has been adjusted
    # controlled by command line option --regenerate-results
    if regenerate_results:
        cocip_persistent2.source.dataframe.to_json(
            get_static_path("cocip-flight-output2.json"),
            indent=2,
            orient="records",
            date_unit="ns",
            double_precision=15,
        )
        cocip_persistent2.contrail.to_json(
            get_static_path("cocip-contrail-output2.json"),
            indent=2,
            orient="records",
            date_unit="ns",
            double_precision=15,
        )

    # confirm all discontinuous waypoints have an "ef" of 0 and an age of 0
    continuous = cocip_persistent2.contrail["continuous"].to_numpy()
    assert np.all(cocip_persistent2.contrail["ef"][~continuous] == 0)
    assert np.all(cocip_persistent2.contrail["age"][~continuous] == np.timedelta64(0, "ns"))

    flight_output = pd.read_json(get_static_path("cocip-flight-output2.json"), orient="records")
    contrail_output = pd.read_json(get_static_path("cocip-contrail-output2.json"), orient="records")

    flight_output["time"] = pd.to_datetime(flight_output["time"])
    flight_output["contrail_age"] = pd.to_timedelta(flight_output["contrail_age"])
    contrail_output["time"] = pd.to_datetime(contrail_output["time"])
    contrail_output["formation_time"] = pd.to_datetime(contrail_output["formation_time"])
    contrail_output["age"] = pd.to_timedelta(contrail_output["age"])
    contrail_output["dt_integration"] = pd.to_timedelta(contrail_output["dt_integration"])

    rtol = 1e-4
    for key in flight_output:
        if key in ["time", "flight_id"]:
            assert np.all(cocip_persistent2.source[key] == flight_output[key])
            continue
        if key == "level":
            np.testing.assert_allclose(
                cocip_persistent2.source.level, flight_output[key], err_msg=key
            )
            continue
        if key in ["tau_cirrus", "specific_cloud_ice_water_content"]:
            np.testing.assert_allclose(
                cocip_persistent2.source[key],
                flight_output[key],
                err_msg=key,
                rtol=rtol,
                atol=1e-14,  # for trace cloud ice
            )
            continue
        np.testing.assert_allclose(
            cocip_persistent2.source[key], flight_output[key], err_msg=key, rtol=rtol
        )

    pd.testing.assert_frame_equal(
        cocip_persistent2.contrail.reset_index(drop=True),
        contrail_output.reset_index(drop=True),
        check_dtype=False,
        check_like=True,  # ignore column order
        check_exact=False,
        rtol=rtol,
    )


@pytest.mark.parametrize(
    ("reference", "lowmem"),
    [
        ("cocip_no_ef", "cocip_no_ef_lowmem"),
        ("cocip_no_ef", "cocip_no_ef_lowmem_indices"),
        ("cocip_persistent", "cocip_persistent_lowmem"),
        ("cocip_persistent", "cocip_persistent_lowmem_indices"),
        ("cocip_persistent2", "cocip_persistent2_lowmem"),
        ("cocip_persistent2", "cocip_persistent2_lowmem_indices"),
    ],
)
def test_eval_lowmem(reference: str, lowmem: str, request: pytest.FixtureRequest) -> None:
    """Check that CoCiP output does not change when low-memory interpolation is used."""
    cocip = request.getfixturevalue(reference)
    cocip_lowmem = request.getfixturevalue(lowmem)
    pd.testing.assert_frame_equal(cocip.source.dataframe, cocip_lowmem.source.dataframe)
    pd.testing.assert_frame_equal(cocip.contrail, cocip_lowmem.contrail)


@pytest.mark.parametrize(
    ("reference", "generic"),
    [
        ("cocip_no_ef", "cocip_no_ef_generic"),
        ("cocip_persistent", "cocip_persistent_generic"),
        ("cocip_persistent2", "cocip_persistent2_generic"),
    ],
)
def test_eval_generic(reference: str, generic: str, request: pytest.FixtureRequest) -> None:
    """Check that CoCiP output does not change when equivalent generic met data is used."""
    cocip = request.getfixturevalue(reference)
    cocip_generic = request.getfixturevalue(generic)
    # pd.testing.assert_frame_equal(
    #     cocip.source.dataframe.rename(
    #         {"specific_cloud_ice_water_content": "mass_fraction_of_cloud_ice_in_air"}, axis=1
    #     ),
    #     cocip_generic.source.dataframe,
    # )
    pd.testing.assert_frame_equal(
        cocip.contrail.drop(["top_net_thermal_radiation", "top_net_solar_radiation"], axis=1),
        cocip_generic.contrail.drop(["toa_net_downward_shortwave_flux"], axis=1),
    )


def test_xarray_contrail(cocip_persistent: Cocip) -> None:
    """Confirm expected contrail output on attribute `contrail_dataset`."""
    assert cocip_persistent.contrail_dataset["longitude"].shape == (8, 12)

    # The contrail_dataset['timestep'] does not necessarily coincide with cocip.timesteps
    assert cocip_persistent.contrail_dataset["timestep"].size == 8
    assert cocip_persistent.contrail_dataset["waypoint"].size == 12


def test_emissions(met: MetDataset, rad: MetDataset, bada_model: AircraftPerformance) -> None:
    """Test `Cocip` use of `Emissions` model."""
    # demo synthetic flight for BADA
    attrs = {"flight_id": "test BADA/EDB", "aircraft_type": "A320", "load_factor": 0.7}

    # Example flight
    df = pd.DataFrame()
    df["longitude"] = np.linspace(-21, -40, 50, dtype=float)
    df["latitude"] = np.linspace(51, 59, 50, dtype=float)
    df["altitude"] = np.linspace(10900, 11000, 50, dtype=float)
    df["time"] = pd.date_range("2019-01-01T00:15:00", "2019-01-01T02:30:00", periods=50)

    fl = Flight(data=df, attrs=attrs)
    # Confirm all initial segment lengths are below the cocip
    # max_seg_length_m default param
    assert Cocip.default_params.max_seg_length_m == 40000
    assert np.all(fl.segment_length()[:-1] < 40000)

    params = {
        "process_emissions": True,
        "max_age": np.timedelta64(3, "h"),
        "aircraft_performance": bada_model,
        "humidity_scaling": ExponentialBoostHumidityScaling(),
    }
    cocip = Cocip(met, rad=rad, params=params)
    cocip.eval(source=fl)

    # NOTE: The final two values are both nan
    # In the hydrostatic ROCD formula, air_temperature is used
    # However, the flight is out-of-bounds at the final waypoint in the longitude
    # dimension, so we accumulate an extra nan
    sl = slice(None, -2)
    assert "engine_efficiency" in cocip.source
    assert np.all(cocip.source["engine_efficiency"][sl] < 0.28)
    assert np.all(cocip.source["engine_efficiency"][sl] > 0.21)

    assert "fuel_flow" in cocip.source
    assert np.all(cocip.source["fuel_flow"][sl] < 0.73)
    assert np.all(cocip.source["fuel_flow"][sl] > 0.60)

    assert "nvpm_ei_n" in cocip.source
    assert np.all(cocip.source["nvpm_ei_n"][sl] < 1.32e15)
    assert np.all(cocip.source["nvpm_ei_n"][sl] > 1.08e15)

    assert "thrust" in cocip.source
    assert np.all(cocip.source["thrust"][sl] < 51000)
    assert np.all(cocip.source["thrust"][sl] > 38000)

    assert "thrust_setting" in cocip.source
    assert np.all(cocip.source["thrust_setting"][sl] < 0.51)
    assert np.all(cocip.source["thrust_setting"][sl] > 0.40)


def test_intermediate_columns_in_flight_output(cocip_persistent: Cocip) -> None:
    """Ensure returned `Flight` instance has expected columns."""
    for col in ["rhi", "width", "T_critical_sac", "rh_critical_sac", "sac"]:
        assert col in cocip_persistent.source
        assert np.all(np.isfinite(cocip_persistent.source[col]))


def test_contrail_contrail_overlapping_effects() -> None:
    """Test `cocip.radiative_forcing.contrail_contrail_overlap_radiative_effects`."""
    # Load one time slice of contrail outputs over Europe
    contrails = GeoVectorDataset(
        pd.read_parquet(get_static_path("cocip-output-flts-20190101-eu.pq"))
    )
    is_time = contrails["time"] == pd.to_datetime("2019-01-01 13:00:00")
    contrails = contrails.filter(is_time)

    # Account for contrail-contrail overlapping
    cocip_params = CocipParams()
    contrails_overlap = contrail_contrail_overlap_radiative_effects(
        contrails, cocip_params.habit_distributions, cocip_params.radius_threshold_um
    )

    # Check outputs
    assert np.all(contrails_overlap["tau_cirrus_overlap"] >= contrails_overlap["tau_cirrus"])
    assert np.all(contrails_overlap["rsr_overlap"] >= contrails_overlap["rsr"])
    assert np.all(contrails_overlap["olr_overlap"] <= contrails_overlap["olr"])
    assert np.all(contrails_overlap["rf_lw_overlap"] <= contrails_overlap["rf_lw"])
    # Do not check for the change in `rf_net` because it's sign depends on the change in
    # `rf_sw` and `rf_lw`


@pytest.fixture()
def fleet(met: MetDataset) -> Fleet:
    """Create a ``Fleet`` within the bounds of ``met``."""
    ds = met.data
    lon0 = ds["longitude"].min().item()
    lon1 = ds["longitude"].max().item()
    lat0 = ds["latitude"].min().item()
    lat1 = ds["latitude"].max().item()
    time0 = ds["time"].min().item()

    n_flights = 100
    fls = []

    rng = np.random.default_rng(32123)
    for i in range(n_flights):
        n_waypoints = rng.integers(300, 500)
        longitude = rng.uniform(lon0, lon1, n_waypoints)
        latitude = rng.uniform(lat0, lat1, n_waypoints)
        altitude = rng.uniform(9000, 11000, n_waypoints)
        time = pd.date_range(time0, periods=n_waypoints, freq="1min") + 5 * pd.Timedelta(i, "min")
        fl = Flight(
            longitude=longitude,
            latitude=latitude,
            altitude=altitude,
            time=time,
            flight_id=str(i),
            true_airspeed=250.0,
            engine_efficiency=0.3,
            fuel_flow=2.0,
            aircraft_mass=150000,
            nvpm_ei_n=1.8e15,
            wingspan=48.0,
        )
        fls.append(fl)

    return Fleet.from_seq(fls, broadcast_numeric=False)


@pytest.mark.filterwarnings("ignore:.*the contrail has no intersection with the met")
@pytest.mark.parametrize("contrail_contrail_overlapping", [True, False])
def test_cocip_contrail_contrail_overlapping(
    met: MetDataset,
    rad: MetDataset,
    fleet: Fleet,
    contrail_contrail_overlapping: bool,
) -> None:
    """Test ``Cocip`` with and without contrail-contrail overlapping.

    Confirm the results are different.
    """
    params = {
        "dt_integration": np.timedelta64(5, "m"),
        "max_age": np.timedelta64(4, "h"),
        "contrail_contrail_overlapping": contrail_contrail_overlapping,
        "humidity_scaling": ExponentialBoostHumidityScaling(),
        "max_seg_length_m": 2e6,
    }
    cocip = Cocip(met, rad, params)
    out = cocip.eval(fleet)

    if contrail_contrail_overlapping:
        assert out["ef"].sum() == pytest.approx(621498.2e8, abs=8e7)
    else:
        assert out["ef"].sum() == pytest.approx(621500.2e8, abs=8e7)


# ------
# Output
# ------


def test_flight_waypoint_and_flight_summary_statistics() -> None:
    """Ensure flight waypoint and flight summary outputs are consistent with CoCiP outputs."""
    # Load flight waypoints
    flight_waypoints_opened = pd.read_parquet(get_static_path("flt-wypts-20190101-eu.pq"))

    cols_req = [
        "flight_id",
        "waypoint",
        "longitude",
        "latitude",
        "altitude",
        "time",
        "segment_length",
        "sac",
    ]
    flight_waypoints_in = GeoVectorDataset(flight_waypoints_opened[cols_req], copy=True)

    # Load CoCiP outputs
    contrails_opened = GeoVectorDataset(
        pd.read_parquet(get_static_path("cocip-output-flts-20190101-eu.pq")),
    )

    # -------------------------------
    # Test flight-waypoint statistics
    # -------------------------------
    flight_waypoints_out = flight_waypoint_summary_statistics(flight_waypoints_in, contrails_opened)

    n_contrails_unique = len(
        np.unique(contrails_opened["flight_id"] + "-" + contrails_opened["waypoint"].astype(str))
    )
    assert n_contrails_unique == (~np.isnan(flight_waypoints_out["ef"])).sum()
    np.testing.assert_allclose(
        np.nansum(flight_waypoints_out["ef"]), np.nansum(contrails_opened["ef"]), rtol=1
    )

    # ------------------------------
    # Test flight-summary statistics
    # ------------------------------
    flight_summary = contrail_flight_summary_statistics(flight_waypoints_out)
    assert len(flight_summary) == len(np.unique(flight_waypoints_out["flight_id"]))
    np.testing.assert_allclose(
        np.nansum(flight_summary["total_flight_distance_flown"]),
        np.nansum(flight_waypoints_out["segment_length"]),
        rtol=1,
    )
    assert np.all(
        flight_summary["total_flight_distance_flown"] >= flight_summary["total_contrails_formed"]
    )
    assert np.all(
        flight_summary["total_contrails_formed"]
        >= flight_summary["total_persistent_contrails_formed"]
    )
    np.testing.assert_allclose(
        np.nansum(flight_waypoints_out["persistent_contrail_length"]),
        np.nansum(flight_summary["total_persistent_contrails_formed"]),
        rtol=1,
    )
    assert (flight_summary["total_persistent_contrails_formed"] == 0.0).sum() == (
        np.nan_to_num(flight_summary["total_energy_forcing"]) == 0.0
    ).sum()
    assert (flight_summary["total_persistent_contrails_formed"] == 0.0).sum() >= (
        flight_summary["mean_lifetime_rf_net"].isna()
    ).sum()
    np.testing.assert_allclose(
        np.nansum(flight_summary["total_energy_forcing"]),
        np.nansum(flight_waypoints_out["ef"]),
        rtol=1,
    )


def test_gridded_and_time_slice_outputs() -> None:
    """Ensure gridded and time-slice outputs are consistent with CoCiP outputs."""
    t_start = pd.to_datetime("2019-01-01 12:00:00")
    t_end = pd.to_datetime("2019-01-01 13:00:00")

    # Load flight waypoints
    flight_waypoints_t = GeoVectorDataset(
        pd.read_parquet(get_static_path("flt-wypts-20190101-eu.pq"))
    )

    # Load CoCiP outputs
    contrails = GeoVectorDataset(
        pd.read_parquet(get_static_path("cocip-output-flts-20190101-eu.pq")),
    )
    is_time = contrails.dataframe["time"].between(t_start, t_end, inclusive="right")
    contrails_t = contrails.filter(is_time, copy=True)

    # Load meteorology
    era5_met = ERA5(
        time=("2019-01-01 12:00:00", "2019-01-01 13:00:00"),
        variables=[
            "air_temperature",
            "specific_humidity",
            "specific_cloud_ice_water_content",
            "geopotential",
        ],
        pressure_levels=[
            100,
            125,
            150,
            175,
            200,
            225,
            250,
            300,
            350,
            400,
            450,
            500,
            550,
            600,
            650,
            700,
            750,
            775,
            800,
            825,
            850,
            875,
            900,
            925,
            950,
            975,
            1000,
        ],
        paths=get_static_path("met-20190101-eu.nc"),
        cachestore=None,
    )
    met = era5_met.open_metdataset(wrap_longitude=False)

    # Load radiation
    ds_rad = xr.open_mfdataset(get_static_path("rad-20190101-eu.nc"))
    ds_rad["sdr"] = np.maximum((ds_rad["tisr"] / (1 * 60 * 60)), 0)
    ds_rad["rsr"] = np.maximum(((ds_rad["tisr"] - ds_rad["tsr"]) / (1 * 60 * 60)), 0)
    ds_rad["olr"] = np.maximum(-(ds_rad["ttr"] / (1 * 60 * 60)), 0)
    ds_rad = ds_rad.drop_vars(["tisr", "tsr", "ttr"])
    ds_rad = ds_rad.expand_dims({"level": np.array([-1])})
    rad = MetDataset(ds_rad, wrap_longitude=False)

    # --------------------
    # Test gridded outputs
    # --------------------
    ds = longitude_latitude_grid(
        t_start, t_end, flight_waypoints_t, contrails_t, met=met, spatial_bbox=(-12, 35, 20, 60)
    )

    np.testing.assert_allclose(
        ds["flight_distance_flown"].sum().values,
        np.nansum(flight_waypoints_t["segment_length"]),
        rtol=1,
    )
    np.testing.assert_allclose(ds["ef"].sum().values, contrails_t["ef"].sum(), rtol=1)

    # The number of grid cells at `t_end` with contrails must be equal
    assert np.count_nonzero(ds["persistent_contrails"]) == np.count_nonzero(ds["tau_contrail"])
    assert np.count_nonzero(ds["ef"]) == 227

    # --------------------------
    # Test time-slice statistics
    # --------------------------
    t_slice_stats = time_slice_statistics(
        t_start,
        t_end,
        flight_waypoints_t,
        contrails_t,
        humidity_scaling=ExponentialBoostLatitudeCorrectionHumidityScaling(),
        met=met,
        rad=rad,
        spatial_bbox=(-12, 35, 20, 60),
    )

    assert t_slice_stats["n_waypoints"] == len(flight_waypoints_t)
    np.testing.assert_allclose(
        t_slice_stats["total_flight_distance"], ds["flight_distance_flown"].sum(), rtol=1
    )
    assert (
        t_slice_stats["n_waypoints_forming_persistent_contrails"]
        > t_slice_stats["n_waypoints_with_persistent_contrails_at_t_end"]
    )
    np.testing.assert_allclose(
        t_slice_stats["total_persistent_contrails_formed"] * 1000,
        np.nansum(flight_waypoints_t["segment_length"][flight_waypoints_t["persistent_1"] == 1.0]),
        rtol=1,
    )
    np.testing.assert_allclose(t_slice_stats["total_contrail_ef"], ds["ef"].sum(), rtol=1)


def test_contrail_edges(cocip_persistent: Cocip, regenerate_results: bool) -> None:
    """Test contrail edges methods."""
    df_contrail = cocip_persistent.contrail
    assert df_contrail is not None

    (
        df_contrail["lon_edge_l"],
        df_contrail["lat_edge_l"],
        df_contrail["lon_edge_r"],
        df_contrail["lat_edge_r"],
    ) = contrail_properties.contrail_edges(
        df_contrail["longitude"].to_numpy(),
        df_contrail["latitude"].to_numpy(),
        df_contrail["sin_a"].to_numpy(),
        df_contrail["cos_a"].to_numpy(),
        df_contrail["width"].to_numpy(),
    )

    # summary json output for comparison
    # controlled by command line option --regenerate-results
    if regenerate_results:
        df_contrail[["lon_edge_l", "lat_edge_l", "lon_edge_r", "lat_edge_r"]].to_json(
            get_static_path("cocip-output-contrail-edges.json"), indent=2, orient="records"
        )

    # load summary json output for comparison
    df_contrail_previous = pd.read_json(get_static_path("cocip-output-contrail-edges.json"))

    keys = ["lon_edge_l", "lat_edge_l", "lon_edge_r", "lat_edge_r"]
    np.testing.assert_allclose(df_contrail[keys].to_numpy(), df_contrail_previous[keys].to_numpy())


def test_cocip_fleet(fl: Flight, met: MetDataset, rad: MetDataset):
    """Confirm that Cocip runs in "fleet" mode.

    Simple smoke test achieved by concatenating several copies of fl together.
    """
    initial_keys = set(fl)
    fls = [fl.copy() for _ in range(10)]
    del fl  # we don't want to accidentally use this below
    for i, fl in enumerate(fls):
        fl.attrs.update(flight_id=f"test_{i}")

    cocip = Cocip(
        met=met,
        rad=rad,
        process_emissions=False,
        humidity_scaling=ExponentialBoostHumidityScaling(),
    )
    cocip.eval(fls)

    # confirm nothing has been mutated (using default copy=True)
    for fl in fls:
        assert set(fl) == initial_keys

    for i, fl in enumerate(fls):
        fl.attrs.update(flight_id=f"flight_id_{i}")

    out = cocip.eval(source=fls)
    assert isinstance(out, list)
    for fl in out:
        assert isinstance(fl, Flight)

        # Removing anything that is different between flights
        del fl.data["flight_id"]
        pd.testing.assert_frame_equal(fl.dataframe, out[0].dataframe)


# This is not explicitly necessary, but I've added for benchmarking
@pytest.mark.skipif(True, reason="Profiling test off")
def test_cocip_performance(met: MetDataset, rad: MetDataset) -> None:
    """Test the performance of the CoCip algorithm over a wide domain."""
    # Demo flight
    n = 10000
    flight_parameters = {
        "flight_id": "test",
        "aircraft_type": "A380",
        "wingspan": 48,
        "bc_ei_n": 1.897462e15,
    }
    df = pd.DataFrame()
    df["longitude"] = np.linspace(-39, -21, n)
    df["latitude"] = np.linspace(50, 59, n)
    df["altitude"] = np.linspace(10900, 10900, n)
    df["time"] = pd.date_range("2019-01-01T00:15:00", "2019-01-01T02:05:00", periods=n)
    df["engine_efficiency"] = np.linspace(0.32, 0.45, n)  # ope
    df["fuel_flow"] = np.linspace(2.1, 2.4, n)  # kg/s
    df["aircraft_mass"] = np.linspace(154445, 154345, n)  # kg
    df["bc_ei_n"] = np.linspace(1.8e15, 1.9e15, n)  # kg
    fl = Flight(df, **flight_parameters)

    # run CoCip
    params = {
        "max_age": np.timedelta64(5, "h"),
        "process_emissions": False,
        "verbose_outputs": True,
    }

    # time 10 cocip calls
    start = pythontime.perf_counter()

    for _ in range(10):
        cocip = Cocip(met.copy(), rad=rad.copy(), params=params)
        cocip.eval(source=fl)

    duration = pythontime.perf_counter() - start

    # most recently 5.0 seconds on macbook air M1 2020 with 16GB ram
    print(f"10 Cocip runs took {duration:.2f} seconds")


def test_cocip_gfs(fl: Flight, met_gfs: MetDataset, rad_gfs: MetDataset) -> None:
    """Confirm pinned values of `cocip_persistent` fixture."""

    # met data is on 2022-01-01 - need to scale time to here
    dt = np.datetime64("2022-01-01 00:15:00") - fl["time"][0]
    fl_gfs = fl.copy()
    fl_gfs.update(time=fl_gfs["time"] + dt)
    fl_gfs.update(longitude=np.linspace(-29, -32, 20))
    fl_gfs.update(latitude=np.linspace(56, 58, 20))
    fl_gfs.update(altitude=np.linspace(10900, 10900, 20))

    # run CoCip
    # If process_emissions=True, BADA will run and overwrite values for
    # engine_efficiency, ...
    params = {
        "max_age": np.timedelta64(3, "h"),
        "process_emissions": False,
        "verbose_outputs": True,
    }
    cocip = Cocip(met=met_gfs, rad=rad_gfs, params=params)
    out = cocip.eval(source=fl_gfs)

    assert isinstance(out, Flight)

    assert "sac" in out
    assert "rhi" in out
    assert "ef" in out
    assert np.nansum(out["sac"]) > 0

    # TODO: find a test case where EF > 0 with GFS. Unfortunately seems
    # like GFS really doesn't do ISSR regions
    # assert np.nansum(out["ef"]) > 0


@pytest.mark.filterwarnings("ignore:.*the contrail has no intersection with the met")
def test_cocip_filtering(fl: Flight, met: MetDataset, rad: MetDataset):
    """Confirm Cocip runs with non-default filtering flags.

    Check model parameters:
    - filter_sac
    - filter_initially_persistent
    - persistent_buffer
    """
    # Boost air temperature for more interesting SAC dynamics
    met.data["air_temperature"] += 10

    scaler = hs.ExponentialBoostHumidityScaling(rhi_adj=0.8)
    params = {"met": met, "rad": rad, "process_emissions": False, "humidity_scaling": scaler}
    cocip = Cocip(**params)
    cocip.eval(fl)
    assert cocip._sac_flight.size == 18
    assert cocip._downwash_flight.size == 11
    assert len(cocip.contrail) == 0

    cocip = Cocip(**params, filter_sac=False)
    with pytest.warns(UserWarning, match="Manually overriding SAC filter"):
        cocip.eval(fl)
    assert cocip._sac_flight.size == 20
    assert cocip._downwash_flight.size == 11
    assert len(cocip.contrail) == 0

    cocip = Cocip(**params, filter_initially_persistent=False)
    with pytest.warns(UserWarning, match="Manually overriding initially persistent filter"):
        cocip.eval(fl)
    assert cocip._sac_flight.size == 18
    assert cocip._downwash_flight.size == 18
    assert len(cocip.contrail) == 0

    cocip = Cocip(
        **params,
        filter_initially_persistent=False,
        persistent_buffer=np.timedelta64(1, "h"),
    )
    with pytest.warns(UserWarning, match="Manually overriding initially persistent filter"):
        cocip.eval(fl)
    assert len(cocip.contrail) == 18 * 2
    assert "end_of_life" in cocip.contrail

    cocip = Cocip(
        **params,
        filter_initially_persistent=False,
        persistent_buffer=np.timedelta64(2, "h"),
    )
    with pytest.warns(UserWarning, match="Manually overriding initially persistent filter"):
        cocip.eval(fl)
    assert len(cocip.contrail) == 18 * 4
    assert "end_of_life" in cocip.contrail


def test_cocip_no_persistence_ef_fill_value(fl: Flight, met: MetDataset, rad: MetDataset):
    """Confirm that EF is filled with 0 for in-domain and nan for out-of-domain waypoints."""
    # Cherrypick some trajectory that is half in-domain and half out-of-domain
    # And make sure SAC is not satisfied for in-domain waypoints
    fl.update(altitude=np.linspace(11000, 13000, 20))
    fl.update(longitude=np.linspace(-39, -37, 20))
    fl.update(latitude=np.full(20, 51))

    # The first 8 waypoints are in-domain, the last 12 are out-of-domain
    np.testing.assert_array_equal(fl.coords_intersect_met(met), [True] * 8 + [False] * 12)

    params = {
        "met": met,
        "rad": rad,
        "process_emissions": False,
        "humidity_scaling": ExponentialBoostHumidityScaling(),
    }
    cocip = Cocip(**params)
    out = cocip.eval(fl)

    # Check that ef and contrail_age are 0 for in-domain and nan for out-of-domain waypoints
    np.testing.assert_array_equal(out["sac"], [0] * 8 + [np.nan] * 12)
    np.testing.assert_array_equal(out["ef"], [0] * 8 + [np.nan] * 12)
    np.testing.assert_array_equal(
        out["contrail_age"], [np.timedelta64(0, "s")] * 8 + [np.timedelta64("nat")] * 12
    )

    # This key was removed
    assert "_met_intersection" not in out


@pytest.mark.filterwarnings("ignore:.*the contrail has no intersection with the met")
@pytest.mark.parametrize("q_method", [None, "cubic-spline"])
def test_exponential_boost_coefficients(
    fl: Flight, met: MetDataset, rad: MetDataset, q_method: str | None
) -> None:
    """Ensure the correct coefficients are chosen depending on the interpolation q method."""
    params = {
        "max_age": np.timedelta64(30, "m"),
        "dt_integration": np.timedelta64(10, "m"),
        "process_emissions": False,
        "humidity_scaling": ExponentialBoostLatitudeCorrectionHumidityScaling(),
        "interpolation_q_method": q_method,
        "verbose_outputs": True,
    }

    if q_method is None:
        cocip = Cocip(met, rad=rad, params=params)
        fl = cocip.eval(source=fl)
        assert fl["rhi"].mean() == pytest.approx(0.68793, rel=1e-5)
        return

    assert q_method == "cubic-spline"
    with pytest.warns(UserWarning, match="Model Cocip uses interpolation_q_method=cubic-spline"):
        cocip = Cocip(met, rad=rad, params=params)
    fl = cocip.eval(source=fl)
    assert fl["rhi"].mean() == pytest.approx(0.69541, rel=1e-5)


def test_radiative_heating_effects_param(fl: Flight, met: MetDataset, rad: MetDataset):
    """Run Cocip with the radiative_heating_effects parameter.

    The purpose of this test is to show that real differences accrue when using the
    radiative_heating_effects parameter.
    """

    fl.update(longitude=np.linspace(-29.0, -32.0, 20))
    fl.update(latitude=np.linspace(56.0, 57.0, 20))
    fl.update(altitude=np.full(20, 10900.0))

    params = {
        "max_age": np.timedelta64(90, "m"),  # keep short to avoid blowing out of bounds
        "dt_integration": np.timedelta64(10, "m"),
        "process_emissions": False,
        "humidity_scaling": ExponentialBoostLatitudeCorrectionHumidityScaling(),
    }

    # Artificially shift time to get some SDR
    met.data = met.data.assign_coords(time=met.data["time"] + np.timedelta64(12, "h"))
    rad.data = rad.data.assign_coords(time=rad.data["time"] + np.timedelta64(12, "h"))
    fl.update(time=fl["time"] + np.timedelta64(12, "h"))

    cocip = Cocip(met, rad=rad, params=params)
    fl1 = cocip.eval(source=fl)
    contrail1 = cocip.contrail
    assert not cocip.params["radiative_heating_effects"]
    fl2 = cocip.eval(source=fl, radiative_heating_effects=True)
    assert cocip.params["radiative_heating_effects"]
    contrail2 = cocip.contrail

    # Compare output between two model runs
    expected = {"d_heat_rate", "cumul_differential_heat", "heat_rate", "cumul_heat"}
    assert set(contrail2.columns).difference(contrail1.columns) == expected

    # Pretty massive difference in EF
    assert fl1["ef"].sum() == pytest.approx(7.3e12, rel=0.1)
    assert fl2["ef"].sum() == pytest.approx(10.1e12, rel=0.1)

    # Not nonzero in the same places!
    filt1 = fl1["ef"] != 0
    filt2 = fl2["ef"] != 0
    assert np.all(filt1 >= filt2)
    assert filt1.sum() == 10
    assert filt2.sum() == 7


def test_radiative_heating_effects():
    """Test radiative heating implementation against precomputed values from Schumann.

    We're somewhat loose with typing in this test. In CoCiP, all of the variables
    defined below are numpy ndarrays. Here, for convenience, we take some as floats
    where it doesn't introduce TypeErrors.
    """
    air_temperature = 217.542
    rhi = 1.1024
    rho_air = 0.4730
    r_ice_vol = 0.000016687

    width = 22731
    depth = np.array([3381.4])
    sigma_yz = 7217929.7
    area_eff = contrail_properties.plume_effective_cross_sectional_area(width, depth, sigma_yz)
    depth_eff = contrail_properties.plume_effective_depth(width, area_eff)
    dT_dz = np.array([0.0023096])
    age_s = 46260

    tau_cirrus = 0.1486182
    sd0 = 1362.215
    sdr = 339.031
    rsr = np.array([145.401])
    olr = 217.848

    # Differential heating rate will always be zero if tau_contrail is zero
    tau_contrail = 0.0
    d_heat_rate = radiative_heating.differential_heating_rate(
        air_temperature,
        rhi,
        rho_air,
        r_ice_vol,
        depth_eff,
        tau_contrail,
        tau_cirrus,
        sd0,
        sdr,
        rsr,
        olr,
    )
    assert d_heat_rate.item() == 0.0

    # Test case -- heating rate - agreement with Fortran values
    tau_contrail = 0.150
    heat_rate = radiative_heating.heating_rate(
        air_temperature,
        rhi,
        rho_air,
        r_ice_vol,
        depth_eff,
        tau_contrail,
        tau_cirrus,
        sd0,
        sdr,
        rsr,
        olr,
    )
    np.testing.assert_allclose(heat_rate, 1.7841781452750816e-06)

    # Test case -- differential heating rate - agreement with Fortran values
    tau_contrail = 0.150
    d_heat_rate = radiative_heating.differential_heating_rate(
        air_temperature,
        rhi,
        rho_air,
        r_ice_vol,
        depth_eff,
        tau_contrail,
        tau_cirrus,
        sd0,
        sdr,
        rsr,
        olr,
    )
    np.testing.assert_allclose(d_heat_rate, -3.853027307117556e-06)

    cumul_rad_heat = -d_heat_rate * age_s  # Cumulative heat should be a positive value
    eff_heat_rate = radiative_heating.effective_heating_rate(
        d_heat_rate, cumul_rad_heat, dT_dz, depth
    )
    np.testing.assert_allclose(eff_heat_rate, -1.6819834787264378e-07)

    ratio = eff_heat_rate / d_heat_rate
    assert 0 < ratio < 1


@pytest.mark.parametrize("max_altitude", [11200, None])
def test_max_altitude_param(fl: Flight, met: MetDataset, rad: MetDataset, max_altitude):
    """Confirm that the max_altitude parameter can be disabled."""
    params = {
        "max_age": np.timedelta64(10, "m"),
        "dt_integration": np.timedelta64(5, "m"),
        "process_emissions": False,
        "humidity_scaling": ExponentialBoostHumidityScaling(rhi_adj=0.3),
        "max_altitude_m": max_altitude,
    }

    fl.update(longitude=np.linspace(-29, -32, 20))
    assert fl.altitude.min() == 11000
    assert fl.altitude.max() == 11500

    cocip = Cocip(met, rad=rad, params=params)
    ef = cocip.eval(source=fl)["ef"]

    # They're not all persistent because of continuity conventions
    if max_altitude is None:
        assert np.count_nonzero(ef) == 11
    else:
        assert np.count_nonzero(ef) == 6


@pytest.mark.parametrize("lon0", [-21, -20, -19.5])
def test_cocip_bounds_check(fl: Flight, met: MetDataset, rad: MetDataset, lon0: float) -> None:
    """Confirm that a warning is emitted when the advected contrail blows out of bounds."""
    cocip = Cocip(met, rad=rad, humidity_scaling=ConstantHumidityScaling(rhi_adj=0.6))

    fl["longitude"] += lon0 - fl["longitude"][0]
    assert fl["longitude"][0] == lon0

    if lon0 == -21:
        time = "2019-01-01T05:00:00.000000"
    elif lon0 == -20:
        time = "2019-01-01T04:00:00.000000"
    elif lon0 == -19.5:
        time = "2019-01-01T03:00:00.000000"
    else:
        raise ValueError(lon0)

    match = f"At time {time}, the contrail has no intersection with the met"
    with pytest.warns(UserWarning, match=match):
        cocip.eval(source=fl)


def test_cocip_met_nonuniform(
    fl: Flight,
    met_cocip_nonuniform_time: MetDataset,
    rad: MetDataset,
) -> None:
    """Confirm that a warning is emitted when the advected contrail blows out of bounds."""

    # Confirm the fixture has nonuniform time
    met_time = met_cocip_nonuniform_time.data["time"]
    assert len(met_time) == 3
    assert len(np.unique(np.diff(met_time))) == 2
    t0, t1, t2 = met_time.values

    cocip = Cocip(
        met_cocip_nonuniform_time,
        rad=rad,
        humidity_scaling=ConstantHumidityScaling(rhi_adj=0.6),
        dt_integration="30min",
        max_age="2hours",
    )
    cocip.eval(source=fl)

    assert cocip.contrail is not None
    contrail_time = cocip.contrail["time"]

    # Confirm the contrail has waypoints in both (t0, t1) and (t1, t2)
    assert np.all(contrail_time < t2)
    assert np.all(contrail_time > t0)
    assert np.any(contrail_time < t1)
    assert np.any(contrail_time > t1)


@pytest.mark.filterwarnings("ignore:Manually overriding SAC filter")
@pytest.mark.filterwarnings("ignore:Manually overriding initially persistent filter")
def test_cocip_survival_fraction(fl: Flight, met: MetDataset, rad: MetDataset):
    """Confirm Cocip runs with all survival fraction parameterizations.

    Set filter_sac and filter_initially_persistent to False to survival fraction
    parameterizations run on as many segments as possible
    """
    scaling = ConstantHumidityScaling()
    params = dict(
        met=met,
        rad=rad,
        process_emissions=False,
        humidity_scaling=scaling,
        filter_sac=False,
        filter_initially_persistent=False,
    )

    cocip = Cocip(**params, unterstrasser_ice_survival_fraction=False)
    assert not hasattr(cocip, "_sac_flight")
    cocip.eval(fl)
    assert len(cocip._sac_flight) == len(fl)
    assert "n_ice_per_m_1" in cocip._sac_flight

    cocip = Cocip(**params, unterstrasser_ice_survival_fraction=True)
    assert not hasattr(cocip, "_sac_flight")
    cocip.eval(fl)
    assert len(cocip._sac_flight) == len(fl)
    assert "n_ice_per_m_1" in cocip._sac_flight


@pytest.mark.parametrize(
    ("mvs", "target"),
    [
        (
            Cocip.generic_met_variables(),
            (
                met_var.AirTemperature,
                met_var.SpecificHumidity,
                met_var.EastwardWind,
                met_var.NorthwardWind,
                met_var.VerticalVelocity,
                met_var.MassFractionOfCloudIceInAir,
            ),
        ),
        (
            Cocip.generic_rad_variables(),
            (met_var.TOANetDownwardShortwaveFlux, met_var.TOAOutgoingLongwaveFlux),
        ),
        (
            Cocip.ecmwf_met_variables(),
            (
                met_var.AirTemperature,
                met_var.SpecificHumidity,
                met_var.EastwardWind,
                met_var.NorthwardWind,
                met_var.VerticalVelocity,
                ecmwf_var.SpecificCloudIceWaterContent,
            ),
        ),
        (
            Cocip.ecmwf_rad_variables(),
            (ecmwf_var.TopNetSolarRadiation, ecmwf_var.TopNetThermalRadiation),
        ),
        (
            Cocip.gfs_met_variables(),
            (
                met_var.AirTemperature,
                met_var.SpecificHumidity,
                met_var.EastwardWind,
                met_var.NorthwardWind,
                met_var.VerticalVelocity,
                gfs_var.CloudIceWaterMixingRatio,
            ),
        ),
        (
            Cocip.gfs_rad_variables(),
            (gfs_var.TOAUpwardShortwaveRadiation, gfs_var.TOAUpwardLongwaveRadiation),
        ),
    ],
)
def test_cocip_met_rad_variables_helper(
    mvs: tuple[MetVariable, ...], target: tuple[MetVariable, ...]
) -> None:
    """Test met and rad variable helper methods."""
    assert mvs == target
