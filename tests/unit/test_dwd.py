from __future__ import annotations

import functools
import os
from datetime import datetime, timedelta
from typing import TypeVar

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pycontrails.datalib.dwd import ICON, flight_level_pressure, icon, ods
from tests import OFFLINE


class ICONGlobal(ICON):
    __init__ = functools.partialmethod(ICON.__init__, domain="global")


class ICONEurope(ICON):
    __init__ = functools.partialmethod(ICON.__init__, domain="europe")


class ICONGermany(ICON):
    __init__ = functools.partialmethod(ICON.__init__, domain="germany")


AnyICONDatalibClass = TypeVar(
    "AnyICONDatalibClass", type[ICON], type[ICONGlobal], type[ICONEurope], type[ICONGermany]
)

#############################
# Open Data Server utilities
#############################


@pytest.mark.unreliable
@pytest.mark.skipif(OFFLINE, reason="offline")
@pytest.mark.parametrize(("domain", "count"), [("global", 4), ("europe", 8), ("germany", 8)])
def test_list_forecasts(domain: str, count: int) -> None:
    """Test forecast cycle listing."""
    forecasts = ods.list_forecasts(domain)
    assert len(forecasts) == count


def test_list_forecasts_error() -> None:
    """Test errors with invalid domains."""
    with pytest.raises(ValueError, match="Unknown domain"):
        _ = ods.list_forecasts("foo")


@pytest.mark.unreliable
@pytest.mark.skipif(OFFLINE, reason="offline")
@pytest.mark.parametrize("domain", ["global", "europe", "germany"])
def test_list_forecast_steps(domain: str) -> None:
    """Test forecast step listing."""
    forecast = ods.list_forecasts(domain)[1]
    steps = ods.list_forecast_steps(domain, forecast)
    assert steps[0] == forecast
    assert steps[-1] >= forecast + timedelta(hours=48)  # all forecasts have >= 48 hour horizon


@pytest.mark.unreliable
@pytest.mark.skipif(OFFLINE, reason="offline")
@pytest.mark.parametrize("domain", ["global", "europe", "germany"])
def test_list_forecast_steps_warning(domain: str) -> None:
    """Test warning when no data found for forecast."""
    with pytest.warns(UserWarning, match="No data available for forecast"):
        steps = ods.list_forecast_steps(domain, datetime(1900, 1, 1))
    assert len(steps) == 0


def test_list_forecast_steps_invalid_domain() -> None:
    """Test forecast step listing with invalid domain."""
    with pytest.raises(ValueError, match="Unknown domain"):
        _ = ods.list_forecast_steps("foo", datetime(1900, 1, 1))


@pytest.mark.unreliable
@pytest.mark.skipif(OFFLINE, reason="offline")
@pytest.mark.parametrize("domain", ["global", "europe", "germany"])
def test_list_forecast_steps_icon_datalib_consistency(domain: str) -> None:
    """Test consistency between available forecast steps and ICON datalib expectations."""
    forecasts = ods.list_forecasts(domain)
    assert np.all(np.diff(np.array(forecasts)) == icon.forecast_frequency(domain))

    for forecast in forecasts[1:3]:
        assert forecast.hour in icon.valid_forecast_hours(domain)

        steps = ods.list_forecast_steps(domain, forecast)
        assert steps[0] == forecast
        assert steps[-1] == icon.last_step(domain, forecast)

        last_hourly = icon.last_hourly_step(domain, forecast)
        hourly = [t for t in steps if t <= last_hourly]
        assert np.all(np.diff(np.array(hourly)) == timedelta(hours=1))

        extended = [t for t in steps if t >= last_hourly]
        dt = pd.to_timedelta(icon.extended_forecast_timestep(domain, forecast))
        assert np.all(np.diff(np.array(extended)) == dt)


def test_global_latitude_rpath() -> None:
    """Test global latitude remote path generation."""
    forecast = datetime(2024, 12, 31, 18)
    path = ods.global_latitude_rpath(forecast)
    assert path == (
        "opendata.dwd.de/weather/nwp/icon/grib/18/clat/"
        "icon_global_icosahedral_time-invariant_2024123118_CLAT.grib2.bz2"
    )

    forecast = datetime(2025, 1, 1, 0)
    path = ods.global_latitude_rpath(forecast)
    assert path == (
        "opendata.dwd.de/weather/nwp/icon/grib/00/clat/"
        "icon_global_icosahedral_time-invariant_2025010100_CLAT.grib2.bz2"
    )


def test_global_longitude_rpath() -> None:
    """Test global longitude remote path generation."""
    forecast = datetime(2024, 12, 31, 18)
    path = ods.global_longitude_rpath(forecast)
    assert path == (
        "opendata.dwd.de/weather/nwp/icon/grib/18/clon/"
        "icon_global_icosahedral_time-invariant_2024123118_CLON.grib2.bz2"
    )

    forecast = datetime(2025, 1, 1, 0)
    path = ods.global_longitude_rpath(forecast)
    assert path == (
        "opendata.dwd.de/weather/nwp/icon/grib/00/clon/"
        "icon_global_icosahedral_time-invariant_2025010100_CLON.grib2.bz2"
    )


@pytest.mark.parametrize(
    ("domain", "forecast", "variable", "step", "level", "expected"),
    [
        (
            "global",
            datetime(2024, 12, 31, 18),
            "t",
            0,
            1,
            "opendata.dwd.de/weather/nwp/icon/grib/18/t/icon_global_icosahedral_model-level_2024123118_000_1_T.grib2.bz2",
        ),
        (
            "global",
            datetime(2025, 1, 1, 0),
            "q",
            6,
            30,
            "opendata.dwd.de/weather/nwp/icon/grib/00/q/icon_global_icosahedral_model-level_2025010100_006_30_Q.grib2.bz2",
        ),
        (
            "global",
            datetime(2025, 1, 1, 6),
            "athb_t",
            12,
            None,
            "opendata.dwd.de/weather/nwp/icon/grib/06/athb_t/icon_global_icosahedral_single-level_2025010106_012_ATHB_T.grib2.bz2",
        ),
        (
            "europe",
            datetime(2024, 12, 31, 18),
            "t",
            0,
            1,
            "opendata.dwd.de/weather/nwp/icon-eu/grib/18/t/icon-eu_europe_regular-lat-lon_model-level_2024123118_000_1_T.grib2.bz2",
        ),
        (
            "europe",
            datetime(2025, 1, 1, 0),
            "q",
            6,
            30,
            "opendata.dwd.de/weather/nwp/icon-eu/grib/00/q/icon-eu_europe_regular-lat-lon_model-level_2025010100_006_30_Q.grib2.bz2",
        ),
        (
            "europe",
            datetime(2025, 1, 1, 6),
            "athb_t",
            12,
            None,
            "opendata.dwd.de/weather/nwp/icon-eu/grib/06/athb_t/icon-eu_europe_regular-lat-lon_single-level_2025010106_012_ATHB_T.grib2.bz2",
        ),
        (
            "germany",
            datetime(2024, 12, 31, 18),
            "t",
            0,
            1,
            "opendata.dwd.de/weather/nwp/icon-d2/grib/18/t/icon-d2_germany_regular-lat-lon_model-level_2024123118_000_1_t.grib2.bz2",
        ),
        (
            "germany",
            datetime(2025, 1, 1, 0),
            "q",
            6,
            30,
            "opendata.dwd.de/weather/nwp/icon-d2/grib/00/q/icon-d2_germany_regular-lat-lon_model-level_2025010100_006_30_q.grib2.bz2",
        ),
        (
            "germany",
            datetime(2025, 1, 1, 6),
            "athb_t",
            12,
            None,
            "opendata.dwd.de/weather/nwp/icon-d2/grib/06/athb_t/icon-d2_germany_regular-lat-lon_single-level_2025010106_012_2d_athb_t.grib2.bz2",
        ),
    ],
)
def test_rpaths(
    domain: str, forecast: datetime, variable: str, step: int, level: int | None, expected: str
) -> None:
    """Test rpath generation."""
    assert ods.rpath(domain, forecast, variable, step, level) == expected


@pytest.mark.unreliable
@pytest.mark.skipif(OFFLINE, reason="offline")
@pytest.mark.parametrize("domain", ["global", "europe", "germany"])
def test_get(domain: str) -> None:
    """Test downloads"""
    forecast = ods.list_forecasts(domain)[1]

    ods.get(ods.rpath(domain, forecast, "t", 0, 1), os.devnull)
    ods.get(ods.rpath(domain, forecast, "athb_t", 0, None), os.devnull)

    if domain == "global":
        ods.get(ods.global_latitude_rpath(forecast), os.devnull)
        ods.get(ods.global_longitude_rpath(forecast), os.devnull)


########################
# Domain input handling
########################


def test_domain_input() -> None:
    """Test domain input."""
    dl = ICON(time=datetime(2000, 1, 1), variables=["t", "q"], pressure_levels=[200])
    assert dl.domain == "global"

    dl = ICON(
        domain="global", time=datetime(2000, 1, 1), variables=["t", "q"], pressure_levels=[200]
    )
    assert dl.domain == "global"

    dl = ICON(
        domain="europe", time=datetime(2000, 1, 1), variables=["t", "q"], pressure_levels=[200]
    )
    assert dl.domain == "europe"

    dl = ICON(
        domain="germany", time=datetime(2000, 1, 1), variables=["t", "q"], pressure_levels=[200]
    )
    assert dl.domain == "germany"

    with pytest.raises(ValueError, match="Unknown domain"):
        _ = ICON(
            domain="foo", time=datetime(2000, 1, 1), variables=["t", "q"], pressure_levels=[200]
        )


######################
# Time input handling
######################


def test_single_time_input() -> None:
    """Test TimeInput parsing."""
    # accept single time
    dl = ICON(time=datetime(2019, 5, 31, 0), variables=["t", "q"], pressure_levels=[200])
    assert dl.timesteps == [datetime(2019, 5, 31, 0)]
    dl = ICON(time=[datetime(2019, 5, 31, 0)], variables=["t", "q"], pressure_levels=[200])
    assert dl.timesteps == [datetime(2019, 5, 31, 0)]

    # accept single time with minutes defined
    dl = ICON(time=datetime(2019, 5, 31, 0, 29), variables=["t", "q"], pressure_levels=[200])
    assert dl.timesteps == [datetime(2019, 5, 31, 0), datetime(2019, 5, 31, 1)]
    dl = ICON(time=[datetime(2019, 5, 31, 0, 29)], variables=["t", "q"], pressure_levels=[200])
    assert dl.timesteps == [datetime(2019, 5, 31, 0), datetime(2019, 5, 31, 1)]


def test_time_input_wrong_length() -> None:
    """Test TimeInput parsing."""

    # throw ValueError for length == 0
    with pytest.raises(ValueError, match="Input time bounds must have length"):
        ICON(time=[], variables=["t", "q"], pressure_levels=[200])

    # throw ValueError for length > 2
    with pytest.raises(ValueError, match="Input time bounds must have length"):
        ICON(
            time=[datetime(2019, 5, 31, 0), datetime(2019, 5, 31, 0), datetime(2019, 5, 31, 0)],
            variables=["t", "q"],
            pressure_levels=[200],
        )


def test_time_input_two_times_nominal() -> None:
    """Test TimeInput parsing with 1 hour default timestep."""

    # accept pair (start, end)
    dl = ICON(
        time=(datetime(2019, 5, 31, 0), datetime(2019, 5, 31, 3)),
        variables=["t", "q"],
        pressure_levels=[200],
    )
    assert dl.timesteps == [
        datetime(2019, 5, 31, 0),
        datetime(2019, 5, 31, 1),
        datetime(2019, 5, 31, 2),
        datetime(2019, 5, 31, 3),
    ]

    dl = ICON(
        time=(datetime(2019, 5, 31, 0, 29), datetime(2019, 5, 31, 2, 40)),
        variables=["t", "q"],
        pressure_levels=[200],
    )
    assert dl.timesteps == [
        datetime(2019, 5, 31, 0),
        datetime(2019, 5, 31, 1),
        datetime(2019, 5, 31, 2),
        datetime(2019, 5, 31, 3),
    ]


def test_time_input_specified_freq() -> None:
    """Test TimeInput parsing with specified frequency."""

    dl = ICON(
        time=(datetime(2019, 5, 31, 0), datetime(2019, 6, 1, 0)),
        variables=["t", "q"],
        pressure_levels=[200],
        timestep_freq="12h",
    )
    assert dl.timesteps == [
        datetime(2019, 5, 31, 0),
        datetime(2019, 5, 31, 12),
        datetime(2019, 6, 1, 0),
    ]

    # non-zero shift modulo 12h frequency
    dl = ICON(
        time=(datetime(2019, 5, 31, 6), datetime(2019, 6, 1, 6)),
        variables=["t", "q"],
        pressure_levels=[200],
        timestep_freq="12h",
    )
    assert dl.timesteps == [
        datetime(2019, 5, 31, 6),
        datetime(2019, 5, 31, 18),
        datetime(2019, 6, 1, 6),
    ]


def test_time_input_numpy_pandas() -> None:
    """Test TimeInput parsing for alternate time input formats."""

    # support alternate types for input
    dl = ICON(
        time=pd.to_datetime(datetime(2019, 5, 31, 0, 29)),
        variables=["t", "q"],
        pressure_levels=[200],
    )
    assert dl.timesteps == [datetime(2019, 5, 31, 0), datetime(2019, 5, 31, 1)]
    dl = ICON(
        time=np.datetime64("2019-05-31T00:29:00"), variables=["t", "q"], pressure_levels=[200]
    )
    assert dl.timesteps == [datetime(2019, 5, 31, 0), datetime(2019, 5, 31, 1)]
    dl = ICON(
        time=(np.datetime64("2019-05-31T00:29:00"), np.datetime64("2019-05-31T02:40:00")),
        variables=["t", "q"],
        pressure_levels=[200],
    )
    assert dl.timesteps == [
        datetime(2019, 5, 31, 0),
        datetime(2019, 5, 31, 1),
        datetime(2019, 5, 31, 2),
        datetime(2019, 5, 31, 3),
    ]


def test_inputs() -> None:
    """Test ICON __init__ with different but equivalent inputs."""
    with pytest.raises(ValueError, match="Input time bounds must have length"):
        ICON(time=[], variables=[], pressure_levels=[17, 18, 19])

    d1 = ICON(time=datetime(2000, 1, 1), variables=["t", "q"], pressure_levels=200)
    d2 = ICON(time=[datetime(2000, 1, 1)], variables=["t", "q"], pressure_levels=[200])
    d3 = ICON(
        time=np.datetime64("2000-01-01 00:00:00"),
        variables=["t", "q"],
        pressure_levels=[200],
    )
    d4 = ICON(
        time=np.array([np.datetime64("2000-01-01 00:00:00")]),
        variables=["t", "q"],
        pressure_levels=[200],
    )
    assert d1.variables == d2.variables == d3.variables == d4.variables
    assert d1.pressure_levels == d2.pressure_levels == d3.pressure_levels == d4.pressure_levels
    assert d1.timesteps == d2.timesteps == d3.timesteps == d4.timesteps


##########################
# Variable input handling
##########################


@pytest.mark.parametrize(
    ("variables", "pressure_levels", "raises"),
    [
        (["t", "q"], None, False),
        (["air_temperature", "specific_humidity"], None, False),
        (["t", "q"], [200], False),
        (["t", "q"], -1, True),
    ],
)
def test_model_level_variables(
    variables: list[str], pressure_levels: int | list[int] | None, raises: bool
) -> None:
    """Test model-level variable parsing."""
    if raises:
        with pytest.raises(ValueError, match=f"{variables[0]} is not in supported parameters"):
            dl = ICON(
                time=datetime(2000, 1, 1), variables=variables, pressure_levels=pressure_levels
            )
        return

    dl = ICON(time=datetime(2000, 1, 1), variables=variables, pressure_levels=pressure_levels)
    assert dl.supported_variables == dl.pressure_level_variables
    assert dl.variable_shortnames == ["t", "q"]


@pytest.mark.parametrize(
    ("variables", "pressure_levels", "raises"),
    [
        ("rlut", -1, False),
        ("toa_outgoing_longwave_flux", -1, False),
        ("rlut", [200], True),
        ("rlut", None, True),
    ],
)
def test_single_level_variables(
    variables: str, pressure_levels: int | list[int] | None, raises: bool
) -> None:
    """Test model-level variable parsing."""
    if raises:
        with pytest.raises(ValueError, match=f"{variables} is not in supported parameters"):
            dl = ICON(
                time=datetime(2000, 1, 1), variables=variables, pressure_levels=pressure_levels
            )
        return

    dl = ICON(time=datetime(2000, 1, 1), variables=variables, pressure_levels=pressure_levels)
    assert dl.supported_variables == dl.single_level_variables
    assert dl.variable_shortnames == ["rlut"]


######################
# Vertical resolution
######################


@pytest.mark.parametrize(
    ("datalib", "nlev"),
    [
        (ICONGlobal, 120),
        (ICONEurope, 74),
        (ICONGermany, 65),
    ],
)
def test_retrieved_levels(datalib: AnyICONDatalibClass, nlev: int) -> None:
    """Test model levels included in download."""
    with pytest.raises(ValueError, match="Requested model_levels must be between"):
        dl = datalib(time=datetime(2000, 1, 1), variables=["t", "q"], model_levels=[0, 1, 2])
    with pytest.raises(ValueError, match="Requested model_levels must be between"):
        dl = datalib(
            time=datetime(2000, 1, 1), variables=["t", "q"], model_levels=[nlev - 1, nlev, nlev + 1]
        )

    dl = datalib(time=datetime(2000, 1, 1), variables=["t", "q"])
    assert dl.model_levels == list(range(1, nlev + 1))

    dl = datalib(time=datetime(2000, 1, 1), variables=["t", "q"], model_levels=[3, 4, 5])
    assert dl.model_levels == [3, 4, 5]


def test_pressure_levels() -> None:
    """Test pressure level inputs."""
    dl = ICON(time=datetime(2000, 1, 1), variables=["t", "q"])
    assert dl.pressure_levels == list(reversed(flight_level_pressure(200, 500)))

    dl = ICON(time=datetime(2000, 1, 1), variables=["t", "q"], pressure_levels=[200, 300])
    assert dl.pressure_levels == [200, 300]

    dl = ICON(time=datetime(2000, 1, 1), variables=["t", "q"], pressure_levels=200)
    assert dl.pressure_levels == [200]

    dl = ICON(time=datetime(2000, 1, 1), variables=["rlut"], pressure_levels=[-1])
    assert dl.pressure_levels == [-1]

    dl = ICON(time=datetime(2000, 1, 1), variables=["rlut"], pressure_levels=-1)
    assert dl.pressure_levels == [-1]


def test_open_metdataset_errors(met_ecmwf_pl_path: str) -> None:
    """Test open_metdataset error handing."""
    dl = ICON(time=(datetime(2000, 1, 1), datetime(2000, 1, 2)), variables=["t", "q"])
    ds = xr.open_dataset(met_ecmwf_pl_path)
    with pytest.raises(ValueError, match="Parameter 'dataset' is not supported"):
        dl.open_metdataset(dataset=ds)

    dl = ICON(
        time=(datetime(2000, 1, 1), datetime(2000, 1, 2)),
        variables=["t", "q"],
    )
    dl.cachestore = None
    with pytest.raises(ValueError, match="Cachestore is required"):
        dl.open_metdataset()


########################
# Horizontal resolution
########################


@pytest.mark.parametrize(
    ("datalib", "grid", "expected", "warn"),
    [
        (ICONGlobal, None, 0.25, False),
        (ICONGlobal, 0.1, 0.1, False),
        (ICONEurope, None, None, False),
        (ICONEurope, 0.1, None, True),
        (ICONGermany, None, None, False),
        (ICONGermany, 0.1, None, True),
    ],
)
def test_grid(
    datalib: AnyICONDatalibClass, grid: float | None, expected: float | None, warn: bool
) -> None:
    """Test horizontal resolution."""
    if warn:
        with pytest.warns(UserWarning, match="ICON-EU Europe and ICON-D2 Germany"):
            dl = datalib(
                time=datetime(2000, 1, 1),
                variables=["t", "q"],
                pressure_levels=[200],
                grid=grid,
            )
    else:
        dl = datalib(
            time=datetime(2000, 1, 1),
            variables=["t", "q"],
            pressure_levels=[200],
            grid=grid,
        )
    assert dl.grid == expected


###############################################
# Time resolution and forecast cycle selection
###############################################


@pytest.mark.parametrize(
    ("datalib", "forecast_time", "horizon"),
    [
        (ICONGlobal, datetime(2000, 1, 1), timedelta(hours=78)),
        (ICONGlobal, datetime(2000, 1, 1, 6), timedelta(hours=78)),
        (ICONEurope, datetime(2000, 1, 1), timedelta(hours=78)),
        (ICONEurope, datetime(2000, 1, 1, 3), timedelta(hours=30)),
    ],
)
def test_hourly_forecast_timestep_freq(
    datalib: AnyICONDatalibClass, forecast_time: datetime, horizon: timedelta
) -> None:
    """Test timestep frequency selection for hourly forecasts."""
    time = [forecast_time + horizon - timedelta(hours=1), forecast_time + horizon]
    dl = datalib(time=time, variables=["t", "q"], forecast_time=forecast_time)
    assert dl.timesteps == time

    time = [forecast_time + horizon, forecast_time + horizon + timedelta(hours=1)]
    with pytest.raises(ValueError, match="Forecast out to time"):
        dl = datalib(
            time=time, variables=["t", "q"], forecast_time=forecast_time, timestep_freq="1h"
        )


@pytest.mark.parametrize(
    ("datalib", "forecast_time", "horizon", "freq"),
    [
        (ICONGlobal, datetime(2000, 1, 1), timedelta(hours=180), timedelta(hours=3)),
        (ICONGlobal, datetime(2000, 1, 1, 6), timedelta(hours=120), timedelta(hours=3)),
        (ICONEurope, datetime(2000, 1, 1), timedelta(hours=120), timedelta(hours=3)),
        (ICONEurope, datetime(2000, 1, 1, 3), timedelta(hours=48), timedelta(hours=6)),
        (ICONGermany, datetime(2000, 1, 1), timedelta(hours=48), timedelta(hours=1)),
        (ICONGermany, datetime(2000, 1, 1, 6), timedelta(hours=48), timedelta(hours=1)),
    ],
)
def test_extended_forecast_timestep_freq(
    datalib: AnyICONDatalibClass, forecast_time: datetime, horizon: timedelta, freq: timedelta
) -> None:
    """Test timestep frequency selection for extended forecasts."""
    time = [forecast_time + horizon - freq, forecast_time + horizon]
    dl = datalib(time=time, variables=["t", "q"], forecast_time=forecast_time)
    assert dl.timesteps == time

    time = [forecast_time + horizon, forecast_time + horizon + freq]
    with pytest.raises(ValueError, match="Requested times extend to"):
        dl = datalib(
            time=time, variables=["t", "q"], forecast_time=forecast_time, timestep_freq="1h"
        )


def test_forecast_time_parsing() -> None:
    """Test forecast time validation."""
    dl = ICON(time=datetime(2000, 1, 1), variables=["t", "q"], forecast_time="2000-01-01 00:00:00")
    assert dl.forecast_time == datetime(2000, 1, 1, 0)

    dl = ICON(time=datetime(2000, 1, 1), variables=["t", "q"], forecast_time="1999-12-31 12:00:00")
    assert dl.forecast_time == datetime(1999, 12, 31, 12)

    with pytest.raises(ValueError, match="Forecast hour must be one of"):
        _ = ICON(
            time=datetime(2000, 1, 1), variables=["t", "q"], forecast_time="2000-01-01 01:00:00"
        )

    with pytest.raises(ValueError, match="Selected forecast time"):
        _ = ICON(
            time=datetime(2000, 1, 1), variables=["t", "q"], forecast_time="2000-01-01 12:00:00"
        )


def test_get_forecast_step() -> None:
    """Test forecast step calculation."""
    dl = ICON(time=datetime(2000, 1, 1), variables=["t", "q"], forecast_time="2000-01-01 00:00:00")
    assert dl.get_forecast_step(datetime(2000, 1, 1, 0)) == 0
    assert dl.get_forecast_step(datetime(2000, 1, 1, 3)) == 3
    assert dl.get_forecast_step(datetime(1999, 12, 31, 22)) == -2
    with pytest.raises(ValueError, match="Time-to-step conversion returned fractional"):
        dl.get_forecast_step(datetime(2000, 1, 1, 0, 30))


################
# Internals
################


def test_repr() -> None:
    """Test ICON __repr__."""
    era5 = ICON(time=datetime(2000, 1, 1), variables=["t", "q"], pressure_levels=200)
    out = repr(era5)
    assert "Domain:" in out
    assert "Forecast time:" in out
    assert "Steps:" in out


def test_model_level_cachepath() -> None:
    """Test cachepath creation for model-level variables."""
    dl = ICON(time=(datetime(2000, 1, 1), datetime(2000, 1, 2)), variables=["t", "q"])
    p = dl.create_cachepath(datetime(2000, 1, 1))
    assert "icon-pl-58bd6e70162ce032d7572a810ec91e30.nc" in p

    p1 = dl.create_cachepath(datetime(2000, 1, 1, 1))
    assert p1 != p

    dl = ICON(
        time=(datetime(2000, 1, 1), datetime(2000, 1, 2)),
        variables=["t", "q"],
        forecast_time="1999-12-31 12:00:00",
    )
    p1 = dl.create_cachepath(datetime(2000, 1, 1))
    assert p1 != p

    dl = ICON(
        time=(datetime(2000, 1, 1), datetime(2000, 1, 2)),
        variables=["t", "q"],
        pressure_levels=[150, 200, 250],
    )
    p1 = dl.create_cachepath(datetime(2000, 1, 1))
    assert p1 != p

    dl = ICON(
        time=(datetime(2000, 1, 1), datetime(2000, 1, 2)),
        variables=["cli"],
    )
    p1 = dl.create_cachepath(datetime(2000, 1, 1))
    assert p1 != p

    dl = ICON(
        time=(datetime(2000, 1, 1), datetime(2000, 1, 2)),
        variables=["t", "q"],
        grid=1.0,
    )
    p1 = dl.create_cachepath(datetime(2000, 1, 1))
    assert p1 != p

    dl = ICON(
        domain="europe",
        time=(datetime(2000, 1, 1), datetime(2000, 1, 2)),
        variables=["t", "q"],
    )
    p1 = dl.create_cachepath(datetime(2000, 1, 1))
    assert p1 != p
    assert "icon-eu-pl" in p1

    dl = ICON(
        domain="germany",
        time=(datetime(2000, 1, 1), datetime(2000, 1, 2)),
        variables=["t", "q"],
    )
    p1 = dl.create_cachepath(datetime(2000, 1, 1))
    assert p1 != p
    assert "icon-d2-pl" in p1

    dl.cachestore = None
    with pytest.raises(ValueError, match="Cachestore is required"):
        dl.create_cachepath(datetime(2000, 1, 1))


def test_single_level_cachepath() -> None:
    """Test cachepath creation for single-level variables."""
    dl = ICON(
        time=(datetime(2000, 1, 1), datetime(2000, 1, 2)), variables=["rlut"], pressure_levels=-1
    )
    p = dl.create_cachepath(datetime(2000, 1, 1))
    assert "icon-sl-e80c92af2db3e09a6ed36d8bf43348a4.nc" in p

    p1 = dl.create_cachepath(datetime(2000, 1, 1, 1))
    assert p1 != p

    dl = ICON(
        time=(datetime(2000, 1, 1), datetime(2000, 1, 2)),
        variables=["rlut"],
        forecast_time="1999-12-31 12:00:00",
        pressure_levels=-1,
    )
    p1 = dl.create_cachepath(datetime(2000, 1, 1))
    assert p1 != p

    dl = ICON(
        time=(datetime(2000, 1, 1), datetime(2000, 1, 2)), variables=["rst"], pressure_levels=-1
    )
    p1 = dl.create_cachepath(datetime(2000, 1, 1))
    assert p1 != p

    dl = ICON(
        time=(datetime(2000, 1, 1), datetime(2000, 1, 2)),
        variables=["rlut"],
        pressure_levels=-1,
        grid=1.0,
    )
    p1 = dl.create_cachepath(datetime(2000, 1, 1))
    assert p1 != p

    dl = ICON(
        domain="europe",
        time=(datetime(2000, 1, 1), datetime(2000, 1, 2)),
        variables=["rlut"],
        pressure_levels=-1,
    )
    p1 = dl.create_cachepath(datetime(2020, 1, 1))
    assert p1 != p
    assert "icon-eu-sl" in p1

    dl = ICON(
        domain="germany",
        time=(datetime(2000, 1, 1), datetime(2000, 1, 2)),
        variables=["rlut"],
        pressure_levels=-1,
    )
    p1 = dl.create_cachepath(datetime(2020, 1, 1))
    assert p1 != p
    assert "icon-d2-sl" in p1

    dl.cachestore = None
    with pytest.raises(ValueError, match="Cachestore is required"):
        dl.create_cachepath(datetime(2000, 1, 1))


@pytest.mark.parametrize(
    ("datalib", "prefix"),
    [
        (ICONGlobal, "opendata.dwd.de/weather/nwp/icon/grib/00/"),
        (ICONEurope, "opendata.dwd.de/weather/nwp/icon-eu/grib/00/"),
        (ICONGermany, "opendata.dwd.de/weather/nwp/icon-d2/grib/00/"),
    ],
)
def test_icon_rpaths(datalib: AnyICONDatalibClass, prefix: str) -> None:
    """Test rpath generation variables."""
    dl = datalib(
        time=(datetime(2000, 1, 1), datetime(2000, 1, 2, 6)),
        forecast_time="2000-01-01 00:00:00",
        variables=["t", "q"],
        model_levels=[1, 2, 3],
        timestep_freq="6h",
    )
    for time in dl.timesteps:
        rpaths = dl.rpaths(time)
        assert len(rpaths) == 9  # 2 variables + pressure on 3 levels
        assert all(rpath.startswith(prefix) for rpath in rpaths)

    dl = datalib(
        time=(datetime(2000, 1, 1), datetime(2000, 1, 2, 6)),
        forecast_time="2000-01-01 00:00:00",
        variables=["rlut"],
        pressure_levels=-1,
        timestep_freq="6h",
    )
    for time in dl.timesteps:
        rpaths = dl.rpaths(time)
        assert len(rpaths) == 1
        assert all(rpath.startswith(prefix) for rpath in rpaths)


@pytest.mark.parametrize(
    ("datalib", "dataset"),
    [
        (ICONGlobal, "ICON"),
        (ICONEurope, "ICON-EU"),
        (ICONGermany, "ICON-D2"),
    ],
)
def test_icon_set_metadata(datalib: AnyICONDatalibClass, dataset: str) -> None:
    """Test metadata setting."""
    dl = datalib(time=(datetime(2000, 1, 1), datetime(2000, 1, 2)), variables=["t", "q"])
    ds = xr.Dataset()
    dl.set_metadata(ds)
    assert ds.attrs["provider"] == "DWD"
    assert ds.attrs["dataset"] == dataset
    assert ds.attrs["product"] == "forecast"
