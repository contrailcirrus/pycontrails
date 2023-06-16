"""Test `Cocip`."""

from __future__ import annotations

import json
import pathlib
import time as pythontime

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pycontrails import Flight, MetDataset
from pycontrails.core.met import originates_from_ecmwf
from pycontrails.models import humidity_scaling as hs
from pycontrails.models.aircraft_performance import AircraftPerformance
from pycontrails.models.cocip import (
    Cocip,
    CocipFlightParams,
    contrail_properties,
    radiative_heating,
)
from pycontrails.models.cocip.output import grid_cirrus
from pycontrails.models.humidity_scaling import (
    ExponentialBoostHumidityScaling,
    ExponentialBoostLatitudeCorrectionHumidityScaling,
)

from .conftest import get_static_path


@pytest.fixture
def met(met_cocip1: MetDataset) -> MetDataset:
    """Rename fixture `met_cocip1` from conftest."""
    return met_cocip1


@pytest.fixture
def rad(rad_cocip1: MetDataset) -> MetDataset:
    """Rename fixture `rad_cocip1` from conftest."""
    return rad_cocip1


@pytest.fixture
def fl(flight_cocip1: Flight) -> Flight:
    """Rename fixture `cocip_fl` from conftest."""
    return flight_cocip1


@pytest.fixture
def cocip_no_ef(fl: Flight, met: MetDataset, rad: MetDataset) -> Cocip:
    """Return `Cocip` instance evaluated on modified `fl`."""
    fl.update(longitude=np.linspace(-29, -32, 20))
    fl.update(latitude=np.linspace(54, 55, 20))
    fl.update(altitude=np.full(20, 10000))

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


@pytest.fixture
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
    }
    cocip = Cocip(met.copy(), rad=rad.copy(), params=params)

    # Eventually the advected waypoints will blow out of bounds
    # Specifically, there is a contrail at 4:30 with latitude larger than 59
    # We acknowledge this here
    with pytest.warns(UserWarning, match="outside of the met interpolation grid"):
        cocip.eval(source=fl)

    return cocip


@pytest.fixture
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
    q = met["specific_humidity"].values.copy()

    cocip = Cocip(met=met, rad=rad, humidity_scaling=ExponentialBoostHumidityScaling())

    # tau_cirrus added to met in __init__
    assert cocip.met is met
    assert "tau_cirrus" in met

    # specific humidity no longer modified on met
    np.testing.assert_array_equal(q, met["specific_humidity"].values)

    # There is not longer any issue in instantiating another model
    another_cocip = Cocip(met=met, rad=rad, humidity_scaling=ExponentialBoostHumidityScaling())
    assert another_cocip.met is met


def test_cocip_processes_rad(met: MetDataset, rad: MetDataset) -> None:
    """Check that Cocip seamlessly processes rad data on init."""

    # A little HACK to avoid the warning
    assert originates_from_ecmwf(met)
    del met.attrs["history"]
    assert not originates_from_ecmwf(met)

    # rad starts without time shifting
    t0 = pd.date_range("2019", freq="1H", periods=13)
    t1 = t0 + Cocip.default_params.shift_radiation_time
    np.testing.assert_array_equal(rad["time"].values, t0)
    assert "_pycontrails_modified" not in rad["time"].attrs
    cocip = Cocip(met=met, rad=rad)

    # time shifted in __init__
    assert cocip.rad is rad
    assert rad["time"].attrs["_pycontrails_modified"]
    assert rad["time"].attrs["shift_radiation_time"] == "-30 minutes"
    np.testing.assert_array_equal(rad["time"].values, t1)

    # should run again no problem
    another_cocip = Cocip(met=met, rad=rad)
    assert another_cocip.rad is rad

    # radiation not shifted a second time
    np.testing.assert_array_equal(rad["time"].values, t1)

    # drop a required rad variable, get an error
    rad.data = rad.data.drop_vars(["top_net_thermal_radiation"])
    with pytest.raises(KeyError, match="Dataset does not contain variable"):
        Cocip(met=met, rad=rad)


def test_cocip_processes_rad_with_warnings(met: MetDataset, rad: MetDataset) -> None:
    """Check that Cocip issues warnings for when rad data is different from expected."""

    del met.attrs["history"]  # see test_cocip_processes_rad

    original_time = rad["time"].values.copy()

    with pytest.warns(UserWarning, match="Shifting radiation time dimension by unexpected"):
        Cocip(met, rad=rad, shift_radiation_time=np.timedelta64(20, "m"))

    assert "shift_radiation_time" in rad["time"].attrs
    assert rad["time"].attrs["shift_radiation_time"] == "20 minutes"
    np.testing.assert_array_equal(original_time + np.timedelta64(20, "m"), rad["time"].values)

    # can't run again using a different time shift (the default)
    with pytest.raises(ValueError, match="has already been scaled"):
        Cocip(met, rad)


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
    fl.update(longitude=np.linspace(-29, -32, 20))
    fl.update(latitude=np.linspace(51, 58, 20))
    fl.update(altitude=np.full(20, 10000))

    # Add flight / met properties that override default inputs
    fl2 = fl.copy()
    fl2["true_airspeed"] = np.full(fl2.size, 40)
    fl2["segment_length"] = np.full(fl2.size, 10000)
    fl2["air_temperature"] = np.full(fl2.size, 220)

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
    with pytest.warns(UserWarning, match="outside of the met interpolation grid"):
        out1 = cocip.eval(fl)
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


def test_eval_persistent(cocip_persistent: Cocip) -> None:
    """Confirm pinned values of `cocip_persistent` fixture."""
    assert cocip_persistent.timesteps.size == 11

    assert "rf_net_mean" in cocip_persistent.source
    assert "ef" in cocip_persistent.source
    assert "cocip" in cocip_persistent.source
    assert np.nansum(cocip_persistent.source["cocip"]) > 0
    assert cocip_persistent.contrail is not None

    # # output json when algorithm has been adjusted
    # cocip_persistent.source.dataframe.to_json(
    #     get_static_path("cocip-flight-output.json"),
    #     indent=2,
    #     orient="records",
    #     date_unit="ns",
    #     double_precision=15,
    # )
    # cocip_persistent.contrail.to_json(
    #     get_static_path("cocip-contrail-output.json"),
    #     indent=2,
    #     orient="records",
    #     date_unit="ns",
    #     double_precision=15,
    # )

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


def test_eval_persistent2(cocip_persistent2: Cocip) -> None:
    """Confirm pinned values of ``cocip_persistent2`` fixture."""
    assert cocip_persistent2.timesteps.size == 16

    assert "rf_net_mean" in cocip_persistent2.source
    assert "ef" in cocip_persistent2.source
    assert "cocip" in cocip_persistent2.source
    assert np.all(np.isfinite(cocip_persistent2.source["ef"]))
    assert np.all(np.isfinite(cocip_persistent2.source["cocip"]))

    assert np.sum(cocip_persistent2.source["cocip"]) > 0
    assert cocip_persistent2.contrail is not None

    # # output json when algorithm has been adjusted
    # cocip_persistent2.source.dataframe.to_json(
    #     get_static_path("cocip-flight-output2.json"),
    #     indent=2,
    #     orient="records",
    #     date_unit="ns",
    #     double_precision=15,
    # )
    # cocip_persistent2.contrail.to_json(
    #     get_static_path("cocip-contrail-output2.json"),
    #     indent=2,
    #     orient="records",
    #     date_unit="ns",
    #     double_precision=15,
    # )

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

    rtol = 1e-3
    for key in flight_output:
        if key in ["time", "flight_id"]:
            assert np.all(cocip_persistent2.source[key] == flight_output[key])
            continue
        if key == "level":
            np.testing.assert_allclose(
                cocip_persistent2.source.level, flight_output[key], err_msg=key
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


def test_xarray_contrail(cocip_persistent: Cocip) -> None:
    """Confirm expected contrail output on attribute `contrail_dataset`."""
    assert cocip_persistent.contrail_dataset["longitude"].shape == (8, 12)

    # The contrail_dataset['timestep'] does not necessarily coincide with cocip.timesteps
    assert cocip_persistent.contrail_dataset["timestep"].size == 8
    assert cocip_persistent.contrail_dataset["waypoint"].size == 12


def test_emissions(met: MetDataset, rad: MetDataset, bada_model: AircraftPerformance) -> None:
    """Test `Cocip` use of `Emissions` model."""
    # demo synthetic flight for BADA
    attrs = {"flight_id": "test BADA/EDB", "aircraft_type": "A320"}

    # Example flight
    df = pd.DataFrame()
    df["longitude"] = np.linspace(-21, -40, 50)
    df["latitude"] = np.linspace(51, 59, 50)
    df["altitude"] = np.linspace(10900, 11000, 50)
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

    assert "engine_efficiency" in cocip.source
    assert np.all(cocip.source["engine_efficiency"][:-1] < 0.28)
    assert np.all(cocip.source["engine_efficiency"][:-1] > 0.21)

    assert "fuel_flow" in cocip.source
    assert np.all(cocip.source["fuel_flow"][:-1] < 0.73)
    assert np.all(cocip.source["fuel_flow"][:-1] > 0.60)

    assert "nvpm_ei_n" in cocip.source
    assert np.all(cocip.source["nvpm_ei_n"][:-1] < 1.32e15)
    assert np.all(cocip.source["nvpm_ei_n"][:-1] > 1.08e15)

    assert "thrust" in cocip.source
    assert np.all(cocip.source["thrust"][:-1] < 51000)
    assert np.all(cocip.source["thrust"][:-1] > 38000)

    assert "thrust_setting" in cocip.source
    assert np.all(cocip.source["thrust_setting"][:-1] < 0.51)
    assert np.all(cocip.source["thrust_setting"][:-1] > 0.40)


def test_intermediate_columns_in_flight_output(cocip_persistent: Cocip) -> None:
    """Ensure returned `Flight` instance has expected columns."""
    for col in ["rhi", "width", "T_critical_sac", "rh_critical_sac", "sac"]:
        assert col in cocip_persistent.source
        assert np.all(np.isfinite(cocip_persistent.source[col]))


# ------
# Output
# ------


def test_flight_statistics(cocip_persistent: Cocip) -> None:
    """Test `Cocip.output_flight_statistics`."""
    stats = cocip_persistent.output_flight_statistics()
    assert isinstance(stats, pd.Series)

    # update output stats when algorithm is adjusted
    # stats.to_json(get_static_path("cocip-flight-statistics.json"), indent=2)

    # compare each value after being processed through json
    output_stats = json.loads(stats.to_json())

    _path = get_static_path("cocip-flight-statistics.json")
    with open(_path, "r") as f:
        opened_stats = json.load(f)

    for k in output_stats:
        if opened_stats[k] is None:
            continue
        if isinstance(opened_stats[k], str):
            assert output_stats[k] == opened_stats[k]
            continue
        np.testing.assert_allclose(output_stats[k], opened_stats[k], err_msg=k, rtol=1e-5)


def test_grid_cirrus(cocip_persistent: Cocip) -> None:
    """Test `grid_cirrus.cirrus_summary_statistics` function."""
    df_contrails = cocip_persistent.contrail
    assert isinstance(df_contrails, pd.DataFrame)
    df_contrails["flight_id"] = cocip_persistent.source.attrs["flight_id"]

    grid_summary = grid_cirrus.cirrus_summary_statistics(
        df_contrails,
        cocip_persistent.met.data["fraction_of_cloud_cover"].isel(time=slice(0, 8)),
        cocip_persistent.met.data["tau_cirrus"].isel(time=slice(0, 8)),
    )

    # summary json output for comparison
    # grid_summary.to_json(
    #     get_static_path("cocip-output-grid-cirrus-summary.json"), indent=2, orient="records"
    # )

    # load summary json output for comparison
    grid_summary_previous = pd.read_json(get_static_path("cocip-output-grid-cirrus-summary.json"))

    for key in grid_summary:
        assert np.allclose(grid_summary[key], grid_summary_previous[key])


def test_contrail_edges(cocip_persistent: Cocip) -> None:
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
    # df_contrail[["lon_edge_l", "lat_edge_l", "lon_edge_r", "lat_edge_r"]].to_json(
    #     get_static_path("cocip-output-contrail-edges.json"), indent=2, orient="records"
    # )

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
    cocip = Cocip(
        met=met,
        rad=rad,
        process_emissions=False,
        humidity_scaling=ExponentialBoostHumidityScaling(),
    )
    with pytest.raises(ValueError, match="Duplicate `flight_id`"):
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

    for i in range(10):
        cocip = Cocip(met.copy(), rad=rad.copy(), params=params)
        _ = cocip.eval(source=fl)

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
    assert cocip._downwash_flight.size == 10
    assert len(cocip.contrail) == 0

    cocip = Cocip(**params, filter_sac=False)
    with pytest.warns(UserWarning, match="Manually overriding SAC filter"):
        cocip.eval(fl)
    assert cocip._sac_flight.size == 20
    assert cocip._downwash_flight.size == 10
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


def test_radiative_heating_effects_param(fl: Flight, met: MetDataset, rad: MetDataset):
    """Run Cocip with the radiative_heating_effects parameter.

    The purpose of this test is to show that real differences accrue when using the
    radiative_heating_effects parameter.
    """

    fl.update(longitude=np.linspace(-29, -32, 20))
    fl.update(latitude=np.linspace(56, 57, 20))
    fl.update(altitude=np.linspace(10900, 10900, 20))

    params = {
        "max_age": np.timedelta64(90, "m"),  # keep short to avoid blowing out of bounds
        "dt_integration": np.timedelta64(10, "m"),
        "process_emissions": False,
        "humidity_scaling": ExponentialBoostLatitudeCorrectionHumidityScaling(),
    }

    # Artifically shift time to get some SDR
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
