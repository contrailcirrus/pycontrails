"""Test `pycontrails.core.datalib` module."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from pycontrails.core import datalib
from pycontrails.core.met_var import AirTemperature, EastwardWind, NorthwardWind, RelativeHumidity


def test_parse_timesteps_single_time() -> None:
    """Test timestep parsing."""

    ts = datalib.parse_timesteps(datetime(2019, 5, 31, 0))
    assert ts == [datetime(2019, 5, 31, 0)]

    ts = datalib.parse_timesteps([datetime(2019, 5, 31, 0)])
    assert ts == [datetime(2019, 5, 31, 0)]


def test_parse_timesteps_frequency() -> None:
    """Test timestep parsing with floor and ceiling hour."""
    ts = datalib.parse_timesteps(datetime(2019, 5, 31, 5, 10))
    assert ts == [datetime(2019, 5, 31, 5), datetime(2019, 5, 31, 6)]


def test_parse_timesteps_multiple_times() -> None:
    """Test timestep parsing with zero and multiple times."""
    # throw ValueError for length == 0
    with pytest.raises(ValueError, match="Input time bounds must have length"):
        datalib.parse_timesteps([])

    # throw ValueError for length > 2
    with pytest.raises(ValueError, match="Input time bounds must have length"):
        datalib.parse_timesteps(
            [datetime(2019, 5, 31, 0), datetime(2019, 5, 31, 0), datetime(2019, 5, 31, 0)]
        )


def test_parse_timesteps_tuple_times() -> None:
    """Test timestep parsing with ``(start, end)`` input."""
    ts = datalib.parse_timesteps((datetime(2019, 5, 31, 0), datetime(2019, 5, 31, 3)))
    assert ts == [
        datetime(2019, 5, 31, 0),
        datetime(2019, 5, 31, 1),
        datetime(2019, 5, 31, 2),
        datetime(2019, 5, 31, 3),
    ]

    ts = datalib.parse_timesteps((datetime(2019, 5, 31, 0, 29), datetime(2019, 5, 31, 2, 40)))
    assert ts == [
        datetime(2019, 5, 31, 0),
        datetime(2019, 5, 31, 1),
        datetime(2019, 5, 31, 2),
        datetime(2019, 5, 31, 3),
    ]


def test_parse_timesteps_alternate_input() -> None:
    """Test timestep parsing with alternate input types."""
    ts = datalib.parse_timesteps(pd.to_datetime(datetime(2019, 5, 31, 0, 29)))
    assert ts == [datetime(2019, 5, 31, 0), datetime(2019, 5, 31, 1)]

    ts = datalib.parse_timesteps("2019-05-31T00:29:00")
    assert ts == [datetime(2019, 5, 31, 0), datetime(2019, 5, 31, 1)]
    ts = datalib.parse_timesteps(("2019-05-31T00:29:00", "2019-05-31T01:40:00"))
    assert ts == [datetime(2019, 5, 31, 0), datetime(2019, 5, 31, 1), datetime(2019, 5, 31, 2)]

    ts = datalib.parse_timesteps(np.datetime64("2019-05-31T00:29:00"))
    assert ts == [datetime(2019, 5, 31, 0), datetime(2019, 5, 31, 1)]
    ts = datalib.parse_timesteps(
        (np.datetime64("2019-05-31T00:29:00"), np.datetime64("2019-05-31T02:40:00"))
    )
    assert ts == [
        datetime(2019, 5, 31, 0),
        datetime(2019, 5, 31, 1),
        datetime(2019, 5, 31, 2),
        datetime(2019, 5, 31, 3),
    ]


def test_parse_timesteps_frequency_input() -> None:
    """Test timestep parsing with frequency input."""
    ts = datalib.parse_timesteps(("2019-05-31T00:29:00", "2019-05-31T08:40:00"), freq="3h")
    assert ts == [
        datetime(2019, 5, 31, 0),
        datetime(2019, 5, 31, 3),
        datetime(2019, 5, 31, 6),
        datetime(2019, 5, 31, 9),
    ]

    # set freq to None to get back input as a list
    ts = datalib.parse_timesteps("2019-05-31T00:29:00", freq=None)
    assert ts == [datetime(2019, 5, 31, 0, 29), datetime(2019, 5, 31, 0, 29)]
    ts = datalib.parse_timesteps(("2019-05-31T00:29:00", "2019-05-31T01:40:00"), freq=None)
    assert ts == [datetime(2019, 5, 31, 0, 29), datetime(2019, 5, 31, 1, 40)]


def test_parse_pressure_levels() -> None:
    """Test pressure level parsing."""

    # if input is -1, return [-1]
    pl = datalib.parse_pressure_levels([-1])
    assert pl == [-1]

    # support single values
    pl = datalib.parse_pressure_levels(100)
    assert pl == [100]

    # cast floats to int
    pl = datalib.parse_pressure_levels(100.0)
    assert pl == [100]

    pl = datalib.parse_pressure_levels([100.0, 200])
    assert pl == [100, 200]

    # support ndarrays
    pl = datalib.parse_pressure_levels(np.array([100, 200]))
    assert pl == [100, 200]

    # raise if mixed signs
    with pytest.raises(ValueError, match="Pressure levels must be all positive or all -1"):
        datalib.parse_pressure_levels(np.array([-1, 200]))

    # throw error if not supported
    with pytest.raises(ValueError, match=r"Pressure levels \[300\] are not supported."):
        datalib.parse_pressure_levels([100, 200, 300], supported=[100, 200])


def test_parse_variables() -> None:
    """Test variable parsing."""

    supported = [AirTemperature, RelativeHumidity]

    # by MetVariable
    v = datalib.parse_variables(AirTemperature, supported=supported)
    assert v == [AirTemperature]

    # raise value error if unmatched
    with pytest.raises(ValueError, match="not in supported parameters"):
        datalib.parse_variables(EastwardWind, supported=supported)

    # by list[MetVariable]
    v = datalib.parse_variables(
        [[EastwardWind, AirTemperature, RelativeHumidity], RelativeHumidity], supported=supported
    )
    assert v == [AirTemperature, RelativeHumidity]

    # raise value error if unmatched
    with pytest.raises(ValueError, match="not in supported parameters"):
        datalib.parse_variables([[EastwardWind, NorthwardWind]], supported=supported)

    # raise value error if a str
    with pytest.raises(TypeError, match="must be of type MetVariable"):
        datalib.parse_variables([["eastward_wind", "northward_wind"]], supported=supported)

    # by short name
    v = datalib.parse_variables("t", supported=supported)
    assert v == [AirTemperature]
    v = datalib.parse_variables(["t"], supported=supported)
    assert v == [AirTemperature]
    v = datalib.parse_variables(["t", "r"], supported=supported)
    assert v == [AirTemperature, RelativeHumidity]

    # by standard name
    v = datalib.parse_variables("air_temperature", supported=supported)
    assert v == [AirTemperature]
    v = datalib.parse_variables(["air_temperature"], supported=supported)
    assert v == [AirTemperature]

    # by grib id
    v = datalib.parse_variables(11, supported=supported)
    assert v == [AirTemperature]
    v = datalib.parse_variables([11], supported=supported)
    assert v == [AirTemperature]

    # by ecmwf id
    v = datalib.parse_variables(130, supported=supported)
    assert v == [AirTemperature]
    v = datalib.parse_variables([130], supported=supported)
    assert v == [AirTemperature]

    # support ndarray input
    v = datalib.parse_variables(np.array(["t", "r"]), supported=supported)
    assert v == [AirTemperature, RelativeHumidity]

    # The original dataclass is not copied
    v = datalib.parse_variables("t", supported=supported)
    assert v[0] is AirTemperature

    # raise value error if unmatched
    with pytest.raises(ValueError, match="not in supported parameters"):
        datalib.parse_variables("vorticity", supported=supported)


def test_parse_grid() -> None:
    """Test grid parsing."""

    # raise value error if not supported
    with pytest.raises(ValueError, match="Grid input 0.3 must be one of"):
        datalib.parse_grid(0.3, supported=[0.25, 0.5])


def test_round_hour() -> None:
    """Test round hour helper."""

    time = datetime(2022, 12, 7, 14)
    rounded = datalib.round_hour(time, 2)
    assert rounded == time

    time = datetime(2022, 12, 7, 15)
    rounded = datalib.round_hour(time, 2)
    assert rounded == datetime(2022, 12, 7, 14)

    time = datetime(2022, 12, 7, 17)
    rounded = datalib.round_hour(time, 6)
    assert rounded == datetime(2022, 12, 7, 12)

    time = datetime(2022, 12, 7, 17)
    rounded = datalib.round_hour(time, 1)
    assert rounded == datetime(2022, 12, 7, 17)

    with pytest.raises(ValueError, match="must be between"):
        datalib.round_hour(time, 25)
