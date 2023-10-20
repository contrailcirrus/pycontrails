"""GFS unit tests."""

from __future__ import annotations

from datetime import datetime

import pytest

from pycontrails import MetVariable
from pycontrails.core.met_var import AirTemperature
from pycontrails.datalib.gfs import GFS_VARIABLES, GFSForecast


def test_GFS_parameters() -> None:
    """Test GFS."""
    assert all([isinstance(p, MetVariable) for p in GFS_VARIABLES])


def test_GFSForecast_init() -> None:
    """Test GFSForecast init."""

    times = ("2022-03-01T03:00:00", "2022-03-01T06:30:00")
    gfs = GFSForecast(times, variables=["air_temperature"], pressure_levels=[300, 250])
    assert gfs.forecast_time == datetime(2022, 3, 1, 0)
    assert gfs.timesteps == [datetime(2022, 3, 1, h) for h in range(3, 8)]
    assert gfs.variables == [AirTemperature]
    assert gfs.grid == 0.25

    # forecast time jumps every 6 hours
    times = ("2022-03-01T08:00:00", "2022-03-01T010:30:00")
    gfs = GFSForecast(times, variables=["air_temperature"], pressure_levels=[300, 250])
    assert gfs.forecast_time == datetime(2022, 3, 1, 6)

    times = ("2022-03-01T13:00:00", "2022-03-01T014:30:00")
    gfs = GFSForecast(times, variables=["air_temperature"], pressure_levels=[300, 250])
    assert gfs.forecast_time == datetime(2022, 3, 1, 12)

    times = ("2022-03-01T19:00:00", "2022-03-01T020:30:00")
    gfs = GFSForecast(times, variables=["air_temperature"], pressure_levels=[300, 250])
    assert gfs.forecast_time == datetime(2022, 3, 1, 18)

    # unsupported pressure levels
    with pytest.raises(ValueError, match="Input time bounds must have length"):
        GFSForecast(time=[], variables=[], pressure_levels=[17, 18, 19])

    # grid strings
    times = ("2022-03-01T03:00:00", "2022-03-01T06:30:00")
    gfs = GFSForecast(times, variables=["air_temperature"], pressure_levels=[300, 250], grid=0.5)
    assert gfs.forecast_time == datetime(2022, 3, 1, 0)
    assert gfs.grid == 0.5
    assert gfs.timesteps == [datetime(2022, 3, 1, h) for h in range(3, 10, 3)]
    assert gfs.variables == [AirTemperature]
    assert gfs.grid == 0.5


def test_GFSForecast_repr() -> None:
    """Test GFSForecast __repr__."""
    gfs = GFSForecast(
        time=datetime(2000, 1, 1), variables="ice_water_mixing_ratio", pressure_levels=200
    )
    out = repr(gfs)
    assert "GFS" in out
    assert "Forecast time:" in out


def test_GFSForecast_grid_string() -> None:
    """Test GFSForecast _grid_string property."""

    # grid strings
    times = ("2022-03-01T03:00:00", "2022-03-01T06:30:00")
    gfs = GFSForecast(times, variables=["air_temperature"], pressure_levels=[300, 250], grid=0.5)
    assert gfs.grid == 0.5
    assert gfs._grid_string == "0p50"

    gfs = GFSForecast(times, variables=["air_temperature"], pressure_levels=[300, 250], grid=1.0)
    assert gfs.grid == 1
    assert gfs._grid_string == "1p00"


def test_GFSForecast_timesteps() -> None:
    """Test GFSForecast timesteps property."""

    times = ("2022-03-01T00:00:00", "2022-03-01T03:00:00")
    gfs = GFSForecast(times, variables="t", pressure_levels=[300, 250])
    assert gfs.timesteps == [datetime(2022, 3, 1, h) for h in range(0, 4)]

    times = ("2022-03-01T02:00:00", "2022-03-01T07:00:00")
    gfs = GFSForecast(times, variables="t", pressure_levels=[300, 250], grid=1.0)
    assert gfs.timesteps == [datetime(2022, 3, 1, h) for h in range(0, 10, 3)]

    times = ("2022-03-01T02:00:00", "2022-03-01T07:00:00")
    gfs = GFSForecast(times, variables="t", pressure_levels=[300, 250], grid=0.25)
    assert gfs.timesteps == [datetime(2022, 3, 1, h) for h in range(2, 8)]


def test_GFSForecast_forecast_path() -> None:
    """Test GFSForecast forecast path construction."""

    gfs = GFSForecast(time="2022-03-01T014:00:00", variables="t", pressure_levels=200)
    assert gfs.forecast_path == "gfs.20220301/12/atmos"


def test_GFSForecast_filenames_0p25() -> None:
    """Test GFSForecast filename construction."""

    times = ("2022-03-01T06:00:00", "2022-03-01T08:00:00")
    gfs = GFSForecast(time=times, variables="t", pressure_levels=200)
    assert len(gfs.timesteps) == 3

    # check filenames
    filename = gfs.filename(gfs.timesteps[0])
    assert filename == "gfs.t06z.pgrb2.0p25.f000"

    filename = gfs.filename(gfs.timesteps[1])
    assert filename == "gfs.t06z.pgrb2.0p25.f001"

    filename = gfs.filename(gfs.timesteps[2])
    assert filename == "gfs.t06z.pgrb2.0p25.f002"


def test_GFSForecast_filenames_1p00() -> None:
    """Test GFSForecast filename construction."""

    times = ("2022-03-01T06:00:00", "2022-03-01T08:00:00")
    gfs = GFSForecast(time=times, variables="t", pressure_levels=200, grid=1)
    assert len(gfs.timesteps) == 2

    # check filenames
    filename = gfs.filename(gfs.timesteps[0])
    assert filename == "gfs.t06z.pgrb2.1p00.f000"

    filename = gfs.filename(gfs.timesteps[1])
    assert filename == "gfs.t06z.pgrb2.1p00.f003"


def test_GFSForecast_hash() -> None:
    """Test GFSForecast hash string."""

    times = ("2022-03-01T06:00:00", "2022-03-01T08:00:00")
    gfs = GFSForecast(time=times, variables="t", pressure_levels=200)
    gfs2 = GFSForecast(time=times, variables=["t", "r"], pressure_levels=200)
    gfs3 = GFSForecast(time=times, variables=["t", "r"], pressure_levels=200, grid=0.5)

    assert gfs.hash == "2e7699eb113d903633823b858058c37a74264a98"
    assert gfs.hash != gfs2.hash
    assert gfs2.hash != gfs3.hash


def test_GFSForecast_cachepaths() -> None:
    """Test GFSForecast cachepath construction."""

    times = ("2022-03-01T06:00:00", "2022-03-01T08:00:00")
    gfs = GFSForecast(time=times, variables="t", pressure_levels=200)

    # get cachepaths for all timestaps
    cachepaths = [gfs.create_cachepath(t) for t in gfs.timesteps]

    assert "20220301-06-0-gfspl0.25.nc" in cachepaths[0]
    assert "20220301-06-1-gfspl0.25.nc" in cachepaths[1]
