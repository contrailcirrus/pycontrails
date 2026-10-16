"""Tests for the Met Office datalib (``s3``, ``ukmo``)."""

from __future__ import annotations

import datetime
import itertools
import pathlib
import shutil
import warnings
from unittest import mock

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pycontrails.core import met_var
from pycontrails.core.cache import DiskCacheStore, GCPCacheStore
from pycontrails.datalib.ukmo import s3
from pycontrails.datalib.ukmo.ukmo import (
    DATASET,
    PROVIDER,
    UM10,
    UM10DataNotFoundError,
)
from pycontrails.models.sac import SAC
from pycontrails.physics import thermo

#: Pressure levels (hPa) used by default
CRUISE_LEVELS_HPA = (300, 275, 250, 225, 200, 175, 150)

##############
# s3 utilities
##############


def dt(*args) -> datetime.datetime:
    return datetime.datetime(*args)


def hours(n: float) -> datetime.timedelta:
    return datetime.timedelta(hours=n)


def test_latest_forecast_within_cycle():
    assert s3.latest_forecast(dt(2026, 3, 15, 3)) == dt(2026, 3, 15, 0)
    assert s3.latest_forecast(dt(2026, 3, 15, 17, 59)) == dt(2026, 3, 15, 12)


def test_latest_forecast_on_cycle_boundary():
    assert s3.latest_forecast(dt(2026, 3, 15, 18)) == dt(2026, 3, 15, 18)


@pytest.mark.parametrize("hour", [1, 3, 5, 7, 23])
def test_check_forecast_time_rejects_hours_that_are_not_cycles(hour):
    with pytest.raises(ValueError, match="must be one of"):
        s3.check_forecast_time(dt(2026, 3, 15, hour))


def test_check_forecast_time_rejects_non_hourly():
    with pytest.raises(ValueError, match="must be one of"):
        s3.check_forecast_time(dt(2026, 3, 15, 6, 30))


@pytest.mark.parametrize("hour", [0, 12])
def test_last_step_long_cycles(hour):
    forecast_time = dt(2026, 3, 15, hour)
    assert s3.last_step(forecast_time) == forecast_time + hours(168)
    assert s3.last_3hourly_step(forecast_time) == forecast_time + hours(144)
    assert s3.last_hourly_step(forecast_time) == forecast_time + hours(54)


@pytest.mark.parametrize("hour", [6, 18])
def test_last_step_short_cycles(hour):
    forecast_time = dt(2026, 3, 15, hour)
    assert s3.last_step(forecast_time) == forecast_time + hours(69)
    assert s3.last_3hourly_step(forecast_time) == forecast_time + hours(69)
    assert s3.last_hourly_step(forecast_time) == forecast_time + hours(54)


def test_last_step_of_short_cycles_changed_on_2026_01_21():
    """06/18Z forecasts ended at +66h until the 2026-01-21 18Z forecast, which ends at +69h.
    Checked against the archive."""
    assert s3.last_step(dt(2026, 1, 21, 6)) == dt(2026, 1, 21, 6) + hours(66)
    assert s3.last_step(dt(2026, 1, 21, 18)) == dt(2026, 1, 21, 18) + hours(69)
    assert s3.last_step(dt(2026, 1, 22, 6)) == dt(2026, 1, 22, 6) + hours(69)
    assert s3.last_step(dt(2025, 6, 15, 18)) == dt(2025, 6, 15, 18) + hours(66)


def test_last_step_unaffected_by_2026_01_21_change_for_long_cycles():
    assert s3.last_step(dt(2025, 6, 15, 0)) == dt(2025, 6, 15, 0) + hours(168)


@pytest.mark.parametrize(
    ("forecast_time", "last_lead", "expected"),
    [
        (dt(2026, 3, 15, 0), 0, "1h"),
        (dt(2026, 3, 15, 0), 54, "1h"),
        (dt(2026, 3, 15, 0), 55, "3h"),
        (dt(2026, 3, 15, 0), 144, "3h"),
        (dt(2026, 3, 15, 0), 145, "6h"),
        (dt(2026, 3, 15, 0), 168, "6h"),
        (dt(2026, 3, 15, 6), 54, "1h"),
        (dt(2026, 3, 15, 6), 55, "3h"),
        (dt(2026, 3, 15, 6), 69, "3h"),
    ],
)
def test_forecast_timestep_freq(forecast_time, last_lead, expected):
    assert s3.forecast_timestep_freq(forecast_time, forecast_time + hours(last_lead)) == expected


def test_forecast_for_time_range_single_time_is_latest_forecast():
    assert s3.forecast_for_time_range(dt(2026, 3, 15, 7), dt(2026, 3, 15, 7)) == dt(2026, 3, 15, 6)


def test_forecast_for_time_range_within_latest_forecast():
    assert s3.forecast_for_time_range(dt(2026, 3, 15, 7), dt(2026, 3, 17, 7)) == dt(2026, 3, 15, 6)


def test_forecast_for_time_range_uses_older_forecast_when_latest_is_too_short():
    # The 06Z forecast ends at +69h and can't reach 70h after 07Z. 00Z reaches +168h.
    first, last = dt(2026, 3, 15, 7), dt(2026, 3, 15, 7) + hours(70)
    assert s3.forecast_for_time_range(first, last) == dt(2026, 3, 15, 0)


def test_forecast_for_time_range_picks_most_recent_covering_forecast():
    # 12Z is the latest forecast at 13Z. It is a long cycle, so it reaches 100h out.
    first = dt(2026, 3, 15, 13)
    assert s3.forecast_for_time_range(first, first + hours(100)) == dt(2026, 3, 15, 12)


def test_forecast_for_time_range_raises_when_no_forecast_is_long_enough():
    first = dt(2026, 3, 15, 0)
    with pytest.raises(ValueError, match="not contained in any single forecast"):
        s3.forecast_for_time_range(first, first + hours(169))


def test_lead_for_valid_time_within_hourly_period():
    assert s3.lead_for_valid_time(dt(2026, 3, 15, 6), dt(2026, 3, 15, 9)) == 3
    assert s3.lead_for_valid_time(dt(2026, 3, 15, 6), dt(2026, 3, 15, 6)) == 0


@pytest.mark.parametrize(("lead", "valid"), [(54, True), (55, False), (57, True), (144, True)])
def test_lead_for_valid_time_steps_for_long_cycle(lead, valid):
    forecast_time = dt(2026, 3, 15, 0)
    valid_time = forecast_time + hours(lead)
    if valid:
        assert s3.lead_for_valid_time(forecast_time, valid_time) == lead
    else:
        with pytest.raises(ValueError, match="only publishes every"):
            s3.lead_for_valid_time(forecast_time, valid_time)


@pytest.mark.parametrize(("lead", "valid"), [(147, False), (150, True), (153, False), (168, True)])
def test_lead_for_valid_time_six_hourly_tail_of_long_cycle(lead, valid):
    forecast_time = dt(2026, 3, 15, 12)
    valid_time = forecast_time + hours(lead)
    if valid:
        assert s3.lead_for_valid_time(forecast_time, valid_time) == lead
    else:
        with pytest.raises(ValueError, match="only publishes every 6h"):
            s3.lead_for_valid_time(forecast_time, valid_time)


def test_lead_for_valid_time_short_cycle_ends_at_69():
    forecast_time = dt(2026, 3, 15, 6)
    assert s3.lead_for_valid_time(forecast_time, forecast_time + hours(69)) == 69
    with pytest.raises(ValueError, match="after the end"):
        s3.lead_for_valid_time(forecast_time, forecast_time + hours(72))


def test_lead_for_valid_time_short_cycle_before_2026_01_21_ends_at_66():
    forecast_time = dt(2025, 9, 15, 6)
    assert s3.lead_for_valid_time(forecast_time, forecast_time + hours(66)) == 66
    with pytest.raises(ValueError, match="after the end"):
        s3.lead_for_valid_time(forecast_time, forecast_time + hours(69))


def test_lead_for_valid_time_rejects_valid_time_before_forecast():
    with pytest.raises(ValueError, match="before the forecast time"):
        s3.lead_for_valid_time(dt(2026, 3, 15, 6), dt(2026, 3, 15, 5))


def test_lead_for_valid_time_rejects_non_hourly():
    with pytest.raises(ValueError, match="only publishes every"):
        s3.lead_for_valid_time(dt(2026, 3, 15, 6), dt(2026, 3, 15, 7, 30))


def test_lead_for_valid_time_rejects_forecast_time_that_is_not_a_cycle():
    with pytest.raises(ValueError, match="must be one of"):
        s3.lead_for_valid_time(dt(2026, 3, 15, 7), dt(2026, 3, 15, 9))


def test_object_key_format_for_pressure_level_variable():
    run = datetime.datetime(2026, 8, 3, 0, 0)
    valid_time = datetime.datetime(2026, 8, 3, 0, 0)
    key = s3.object_key(run, valid_time, 0, "temperature_on_pressure_levels")
    assert key == (
        "global-deterministic-10km/20260803T0000Z/"
        "20260803T0000Z-PT0000H00M-temperature_on_pressure_levels.nc"
    )


def test_object_key_lead_and_minutes_padding():
    run = datetime.datetime(2026, 8, 3, 0, 0)
    valid_time = datetime.datetime(2026, 8, 4, 0, 0)
    key = s3.object_key(run, valid_time, 24, "temperature_on_pressure_levels")
    assert key == (
        "global-deterministic-10km/20260803T0000Z/"
        "20260804T0000Z-PT0024H00M-temperature_on_pressure_levels.nc"
    )


def test_select_pressure_levels_crops_to_passed_in_extent():
    pressure_pa = np.asarray(CRUISE_LEVELS_HPA, dtype=np.float64) * 100.0
    latitude = np.linspace(30.0, 70.0, 41)  # ascending, 1 degree steps
    longitude = np.linspace(-40.0, 0.0, 41)  # 1 degree steps
    data = np.zeros((len(pressure_pa), len(latitude), len(longitude)), dtype=np.float32)
    ds = xr.Dataset(
        {"air_temperature": (("pressure", "latitude", "longitude"), data)},
        coords={"pressure": pressure_pa, "latitude": latitude, "longitude": longitude},
    )

    shanwick_extent = (-30.0, -10.0, 45.0, 61.0)
    da = s3.select_pressure_levels(
        ds, "temperature_on_pressure_levels", CRUISE_LEVELS_HPA, extent=shanwick_extent
    )

    assert da["longitude"].values.min() >= -30.0
    assert da["longitude"].values.max() <= -10.0
    assert da["latitude"].values.min() >= 45.0
    assert da["latitude"].values.max() <= 61.0


def test_select_pressure_levels_no_extent_returns_full_domain():
    pressure_pa = np.asarray(CRUISE_LEVELS_HPA, dtype=np.float64) * 100.0
    latitude = np.linspace(10.0, 60.0, 51)
    longitude = np.linspace(-140.0, -50.0, 91)
    data = np.zeros((len(pressure_pa), len(latitude), len(longitude)), dtype=np.float32)
    ds = xr.Dataset(
        {"air_temperature": (("pressure", "latitude", "longitude"), data)},
        coords={"pressure": pressure_pa, "latitude": latitude, "longitude": longitude},
    )

    da = s3.select_pressure_levels(ds, "temperature_on_pressure_levels", CRUISE_LEVELS_HPA)

    np.testing.assert_array_equal(da["longitude"].values, longitude)
    np.testing.assert_array_equal(da["latitude"].values, latitude)


def test_select_pressure_levels_rejects_descending_longitude():
    pressure_pa = np.asarray(CRUISE_LEVELS_HPA, dtype=np.float64) * 100.0
    latitude = np.linspace(30.0, 70.0, 5)
    longitude = np.array([10.0, 5.0, 0.0])  # descending
    data = np.zeros((len(pressure_pa), len(latitude), len(longitude)), dtype=np.float32)
    ds = xr.Dataset(
        {"air_temperature": (("pressure", "latitude", "longitude"), data)},
        coords={"pressure": pressure_pa, "latitude": latitude, "longitude": longitude},
    )

    with pytest.raises(ValueError, match="longitude"):
        s3.select_pressure_levels(ds, "temperature_on_pressure_levels", CRUISE_LEVELS_HPA)


def test_select_pressure_levels_rejects_extent_producing_empty_crop():
    pressure_pa = np.asarray(CRUISE_LEVELS_HPA, dtype=np.float64) * 100.0
    latitude = np.linspace(30.0, 70.0, 41)
    longitude = np.linspace(-40.0, 0.0, 41)
    data = np.zeros((len(pressure_pa), len(latitude), len(longitude)), dtype=np.float32)
    ds = xr.Dataset(
        {"air_temperature": (("pressure", "latitude", "longitude"), data)},
        coords={"pressure": pressure_pa, "latitude": latitude, "longitude": longitude},
    )

    outside_extent = (100.0, 110.0, 80.0, 85.0)  # entirely outside the dataset's domain
    with pytest.raises(ValueError, match="empty result"):
        s3.select_pressure_levels(
            ds, "temperature_on_pressure_levels", CRUISE_LEVELS_HPA, extent=outside_extent
        )


class _FakeGetFilesystem:
    """Stand-in for an s3fs filesystem whose ``get`` copies a local file."""

    def __init__(self, source: pathlib.Path):
        self.source = source
        self.get_calls: list[tuple[str, str]] = []

    def get(self, rpath: str, lpath: str) -> None:
        self.get_calls.append((rpath, lpath))
        shutil.copy(self.source, lpath)


FILE_PRESSURE_PA = np.array([40000.0, 30000.0, 25000.0, 20000.0, 15000.0])
FILE_RUN = dt(2026, 3, 10, 0)
FILE_LEAD_HOURS = 15
FILE_VALID_TIME = FILE_RUN + hours(FILE_LEAD_HOURS)


def _write_archive_like_file(path: pathlib.Path, variable: str) -> xr.DataArray:
    """Write a small file laid out like an archive object, returning the data written."""
    shape = (len(FILE_PRESSURE_PA), 3, 4)
    data = xr.DataArray(
        np.arange(np.prod(shape), dtype=np.float32).reshape(shape),
        dims=("pressure", "latitude", "longitude"),
        coords={
            "pressure": FILE_PRESSURE_PA.astype(np.float32),
            "latitude": np.array([30.0, 31.0, 32.0], dtype=np.float32),
            "longitude": np.array([-100.0, -99.0, -98.0, -97.0], dtype=np.float32),
        },
    )
    ds = xr.Dataset(
        {variable: data},
        coords={
            "forecast_reference_time": np.datetime64(FILE_RUN, "ns"),
            "time": np.datetime64(FILE_VALID_TIME, "ns"),
            "forecast_period": np.timedelta64(FILE_LEAD_HOURS, "h"),
        },
    )
    ds.attrs["um_version"] = "13.8"
    ds.to_netcdf(path)
    return data


@pytest.mark.parametrize("parameter", list(s3.PARAMETER_VARIABLE))
def test_fetch_pressure_level_field_downloads_object_once_and_loads_requested_levels(
    tmp_path, parameter
):
    """Each object is fetched with a single whole-object ``fs.get``, then only the requested
    levels are loaded, and the temp file is removed.

    The single download is the point of this test. Reading the object remotely in chunks
    makes one S3 request per chunk that overlaps the data requested, which was very slow
    (around 10 minutes per file) for these objects. Chunked reads wouldn't call ``fs.get``,
    so this also fails if the implementation goes back to them.
    """
    variable = s3.PARAMETER_VARIABLE[parameter]
    written = _write_archive_like_file(tmp_path / "source.nc", variable)
    fs = _FakeGetFilesystem(tmp_path / "source.nc")
    key = s3.object_key(FILE_RUN, FILE_VALID_TIME, FILE_LEAD_HOURS, parameter)

    actual = s3.fetch_pressure_level_field(
        fs,
        key,
        parameter,
        levels_hpa=[300, 200],
        run=FILE_RUN,
        valid_time=FILE_VALID_TIME,
        lead_hours=FILE_LEAD_HOURS,
    )

    assert len(fs.get_calls) == 1
    rpath, lpath = fs.get_calls[0]
    assert rpath == f"s3://{s3.BUCKET}/{key}"
    assert not pathlib.Path(lpath).exists()

    assert list(actual.dims) == ["pressure", "latitude", "longitude"]
    np.testing.assert_array_equal(actual["pressure"].values, [30000.0, 20000.0])
    np.testing.assert_array_equal(actual.values, written.sel(pressure=[30000.0, 20000.0]).values)
    assert actual.attrs["um_version"] == "13.8"


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"run": FILE_RUN + hours(6)}, "does not match in-file forecast_reference_time"),
        ({"valid_time": FILE_VALID_TIME + hours(1)}, "does not match in-file time"),
        ({"lead_hours": FILE_LEAD_HOURS + 1}, "does not match in-file forecast_period"),
    ],
)
def test_fetch_pressure_level_field_rejects_file_that_does_not_match_its_key(
    tmp_path, kwargs, match
):
    parameter = "temperature_on_pressure_levels"
    _write_archive_like_file(tmp_path / "source.nc", s3.PARAMETER_VARIABLE[parameter])
    fs = _FakeGetFilesystem(tmp_path / "source.nc")
    expected = {"run": FILE_RUN, "valid_time": FILE_VALID_TIME, "lead_hours": FILE_LEAD_HOURS}

    with pytest.raises(ValueError, match=match):
        s3.fetch_pressure_level_field(fs, "key", parameter, levels_hpa=[300], **(expected | kwargs))


def test_fetch_pressure_level_field_rejects_level_missing_from_file(tmp_path):
    parameter = "temperature_on_pressure_levels"
    _write_archive_like_file(tmp_path / "source.nc", s3.PARAMETER_VARIABLE[parameter])
    fs = _FakeGetFilesystem(tmp_path / "source.nc")

    with pytest.raises(ValueError, match="expected exactly one 500"):
        s3.fetch_pressure_level_field(fs, "key", parameter, levels_hpa=[300, 500])


######
# ukmo
######

UKMO_LATITUDE = np.array([30.0, 31.0])
UKMO_LONGITUDE = np.array([-100.0, -99.0])
UKMO_PRESSURE_PA = np.asarray(CRUISE_LEVELS_HPA, dtype=np.float64) * 100.0


class _FakeClient:
    """Stand-in for an s3fs filesystem -- never actually called since the fetch
    functions themselves are replaced below."""


class _Listing:
    """Stand-in for the set returned by :func:`s3.list_run_keys`: contains every key
    except those ``absent`` reports as missing."""

    def __init__(self, absent=lambda key: False):
        self._absent = absent

    def __contains__(self, key: str) -> bool:
        return not self._absent(key)


def _patch_listing(monkeypatch, *, absent=lambda key: False) -> None:
    """Patch :func:`s3.list_run_keys` so the pre-download existence check sees every
    object, except those for which ``absent(key)`` is true."""
    monkeypatch.setattr(s3, "list_run_keys", lambda fs, run: _Listing(absent))


def _patch_fetch_with_values(
    monkeypatch,
    *,
    t_value: float = 220.0,
    rh_value: float = 0.8,
    latitude: np.ndarray = UKMO_LATITUDE,
    longitude: np.ndarray = UKMO_LONGITUDE,
    pressure_pa: np.ndarray = UKMO_PRESSURE_PA,
) -> mock.MagicMock:
    """Patch s3.py's live-fetch functions to return fixed per-parameter fields.

    Returns the mock so callers can assert on ``call_count``/``assert_not_called``
    -- the tests below care about *whether* a fetch happened, not just what data
    it would return.
    """
    monkeypatch.setattr(s3, "filesystem", _FakeClient)
    _patch_listing(monkeypatch)

    def fake_fetch(fs, key, parameter, **kwargs):
        variable = s3.PARAMETER_VARIABLE[parameter]
        value = t_value if variable == "air_temperature" else rh_value
        data = np.full((len(pressure_pa), len(latitude), len(longitude)), value, dtype=np.float64)
        return xr.DataArray(
            data,
            dims=("pressure", "latitude", "longitude"),
            coords={"pressure": pressure_pa, "latitude": latitude, "longitude": longitude},
        )

    fetch_mock = mock.MagicMock(side_effect=fake_fetch)
    monkeypatch.setattr(s3, "fetch_pressure_level_field", fetch_mock)
    return fetch_mock


def _write_cache_hit(
    dlib: UM10,
    t: datetime.datetime,
    *,
    t_value: float = 220.0,
    rh_value: float = 0.8,
) -> None:
    """Pre-populate ``dlib.cachestore`` with a correctly-shaped hit for hour ``t``.

    Matches exactly what :meth:`UM10._process_hour` would itself write --
    short-named variables, ``level`` in hPa -- so the inherited completeness
    check reports a hit.
    """
    level = np.asarray(dlib.pressure_levels, dtype=np.float64)
    shape = (len(level), len(UKMO_LATITUDE), len(UKMO_LONGITUDE), 1)
    values = {"t": t_value, "q": rh_value, "r": rh_value}
    data_vars = {
        name: (
            ("level", "latitude", "longitude", "time"),
            np.full(shape, values[name], dtype=np.float64),
        )
        for name in dlib.variable_shortnames
    }
    ds = xr.Dataset(
        data_vars,
        coords={
            "level": level,
            "latitude": UKMO_LATITUDE,
            "longitude": UKMO_LONGITUDE,
            "time": [pd.Timestamp(t)],
        },
    )
    ds.to_netcdf(dlib.create_cachepath(t), mode="w")


def test_download_dataset_fetches_on_miss(tmp_path, monkeypatch):
    fetch_mock = _patch_fetch_with_values(monkeypatch)
    hour = datetime.datetime(2024, 9, 1, 0)

    mds = UM10(hour, cachestore=DiskCacheStore(tmp_path)).open_metdataset()

    assert fetch_mock.call_count == 2  # one per required parameter
    assert mds.data.sizes["time"] == 1


def test_open_metdataset_reads_archive_hit_without_fetching(tmp_path, monkeypatch):
    fetch_mock = _patch_fetch_with_values(monkeypatch)
    hour = datetime.datetime(2024, 9, 1, 0)
    dlib = UM10(hour, cachestore=DiskCacheStore(tmp_path))
    _write_cache_hit(dlib, hour, t_value=999.0)

    mds = dlib.open_metdataset()

    fetch_mock.assert_not_called()
    np.testing.assert_allclose(mds.data["air_temperature"].values, 999.0)


def test_open_metdataset_partial_hit_only_fetches_miss(tmp_path, monkeypatch):
    fetch_mock = _patch_fetch_with_values(monkeypatch, t_value=210.0)
    hours = [datetime.datetime(2024, 9, 1, 0), datetime.datetime(2024, 9, 1, 1)]
    dlib = UM10(hours, cachestore=DiskCacheStore(tmp_path))
    _write_cache_hit(dlib, hours[0], t_value=999.0)

    mds = dlib.open_metdataset()

    assert fetch_mock.call_count == 2  # only hours[1]'s 2 parameters
    assert mds.data.sizes["time"] == 2
    hit_val = (
        mds.data.sel(time=hours[0])["air_temperature"]
        .isel(level=0, latitude=0, longitude=0)
        .compute()
        .item()
    )
    miss_val = (
        mds.data.sel(time=hours[1])["air_temperature"]
        .isel(level=0, latitude=0, longitude=0)
        .compute()
        .item()
    )
    assert np.isclose(hit_val, 999.0)
    assert np.isclose(miss_val, 210.0)


def test_download_dataset_rejects_non_disk_cache_store(tmp_path, monkeypatch):
    """Fetched hours can only be written to a ``DiskCacheStore``, so any other store must
    fail before a download rather than download data that can't be stored."""
    fetch_mock = _patch_fetch_with_values(monkeypatch)
    hour = datetime.datetime(2024, 9, 1, 0)
    cachestore = GCPCacheStore(
        bucket="fake-bucket", disk_cache=DiskCacheStore(tmp_path / "mirror"), read_only=True
    )

    with pytest.raises(NotImplementedError, match="DiskCacheStore"):
        UM10(hour, cachestore=cachestore).download_dataset([hour])

    fetch_mock.assert_not_called()


def test_download_dataset_raises_when_live_fetch_fails(tmp_path, monkeypatch):
    """An object listed in the archive that is gone by the time it's fetched."""
    monkeypatch.setattr(s3, "filesystem", _FakeClient)
    _patch_listing(monkeypatch)

    def failing_fetch(fs, key, parameter, **kwargs):
        raise FileNotFoundError("simulated 404")

    monkeypatch.setattr(s3, "fetch_pressure_level_field", failing_fetch)

    with pytest.raises(UM10DataNotFoundError, match="could not be fetched"):
        UM10(
            datetime.datetime(2024, 9, 1, 0), cachestore=DiskCacheStore(tmp_path)
        ).open_metdataset()


def test_download_dataset_propagates_programming_errors_unwrapped(tmp_path, monkeypatch):
    """A non-``FileNotFoundError`` failure (a bug, not a missing S3 object) must
    propagate as itself, not be relabeled as :class:`UM10DataNotFoundError`."""
    monkeypatch.setattr(s3, "filesystem", _FakeClient)
    _patch_listing(monkeypatch)

    def buggy_fetch(fs, key, parameter, **kwargs):
        raise TypeError("simulated bug")

    monkeypatch.setattr(s3, "fetch_pressure_level_field", buggy_fetch)

    with pytest.raises(TypeError, match="simulated bug"):
        UM10(
            datetime.datetime(2024, 9, 1, 0), cachestore=DiskCacheStore(tmp_path)
        ).open_metdataset()


def test_download_dataset_raises_before_downloading_when_hour_missing(tmp_path, monkeypatch):
    """A missing hour among several requested is reported before any hour is downloaded,
    and the error names every missing hour."""
    time_range = (datetime.datetime(2024, 9, 1, 0), datetime.datetime(2024, 9, 1, 2))
    hours = pd.date_range(*time_range, freq="1h").to_pydatetime().tolist()
    missing_hour = hours[2]
    fetch_mock = _patch_fetch_with_values(monkeypatch)
    _patch_listing(monkeypatch, absent=lambda key: missing_hour.strftime("%Y%m%dT%H%MZ-") in key)

    dlib = UM10(time_range, cachestore=DiskCacheStore(tmp_path))
    with pytest.raises(UM10DataNotFoundError, match="1 of 3 hour") as excinfo:
        dlib.download_dataset(hours)

    assert missing_hour.isoformat() in str(excinfo.value)
    fetch_mock.assert_not_called()
    for hour in hours:
        assert not dlib.cachestore.exists(dlib.create_cachepath(hour))


def test_download_dataset_stops_at_first_hour_vanishing_during_download(tmp_path, monkeypatch):
    """An object that vanishes after the existence check raises at that hour, without
    fetching the hours after it."""
    time_range = (datetime.datetime(2024, 9, 1, 0), datetime.datetime(2024, 9, 1, 2))
    hours = pd.date_range(*time_range, freq="1h").to_pydatetime().tolist()
    fetch_mock = _patch_fetch_with_values(monkeypatch)
    fetched_valid_times = []

    def fetch(fs, key, parameter, *, valid_time, **kwargs):
        fetched_valid_times.append(valid_time)
        if valid_time == hours[1]:
            raise FileNotFoundError("simulated 404")
        return fetch_mock.side_effect(fs, key, parameter)

    monkeypatch.setattr(s3, "fetch_pressure_level_field", fetch)

    dlib = UM10(time_range, cachestore=DiskCacheStore(tmp_path))
    with pytest.raises(UM10DataNotFoundError, match=hours[1].isoformat()):
        dlib.download_dataset(hours)

    assert hours[2] not in fetched_valid_times
    assert dlib.cachestore.exists(dlib.create_cachepath(hours[0]))


class _FakeLsFilesystem:
    """Stand-in for an s3fs filesystem supporting only ``ls``."""

    def __init__(self, paths: list[str] | None):
        self.paths = paths
        self.ls_kwargs: dict = {}

    def ls(self, path, **kwargs):
        self.ls_kwargs = {"path": path, **kwargs}
        if self.paths is None:
            raise FileNotFoundError(path)
        return self.paths


def test_list_run_keys_returns_keys_relative_to_bucket():
    run = datetime.datetime(2024, 9, 1, 0)
    key = s3.object_key(run, run, 0, "temperature_on_pressure_levels")
    fs = _FakeLsFilesystem([f"{s3.BUCKET}/{key}"])

    assert s3.list_run_keys(fs, run) == {key}
    assert fs.ls_kwargs["path"] == f"{s3.BUCKET}/{s3.PRODUCT_PREFIX}/20240901T0000Z"
    assert fs.ls_kwargs["refresh"] is True


def test_list_run_keys_empty_for_unpublished_run():
    assert s3.list_run_keys(_FakeLsFilesystem(None), datetime.datetime(2024, 9, 1, 0)) == set()


def test_construction_rejects_none_cachestore():
    """Without a cache store fetched data would be downloaded and then discarded, so
    this must fail before any download."""
    with pytest.raises(ValueError, match="cachestore is required"):
        UM10(datetime.datetime(2024, 9, 1, 0), cachestore=None)


def test_open_metdataset_rejects_descending_latitude(tmp_path, monkeypatch):
    _patch_fetch_with_values(monkeypatch, latitude=np.array([31.0, 30.0]))
    hour = datetime.datetime(2024, 9, 1, 0)

    with pytest.raises(ValueError, match="latitude"):
        UM10(hour, cachestore=DiskCacheStore(tmp_path)).open_metdataset()


def test_specific_humidity_round_trips_to_relative_humidity(tmp_path, monkeypatch):
    t_value, rh_value = 215.0, 0.7
    _patch_fetch_with_values(monkeypatch, t_value=t_value, rh_value=rh_value)
    hour = datetime.datetime(2024, 9, 1, 0)

    mds = UM10(hour, cachestore=DiskCacheStore(tmp_path)).open_metdataset()

    # MetDataset sorts `level` ascending, so index 0 is 150 hPa, not
    # CRUISE_LEVELS_HPA[0] (300 hPa) -- read the actual level back rather than
    # assuming which physical level ends up at a given index.
    level_pa = mds.data["level"].isel(level=0).compute().item() * 100.0
    expected_q = rh_value * thermo.q_sat_liquid(np.float64(t_value), np.float64(level_pa))
    q = (
        mds.data["specific_humidity"]
        .isel(level=0, latitude=0, longitude=0, time=0)
        .compute()
        .item()
    )
    assert np.isclose(q, expected_q, rtol=1e-4)

    recovered_rh = q / thermo.q_sat_liquid(np.float64(t_value), np.float64(level_pa))
    assert np.isclose(recovered_rh, rh_value, rtol=1e-4)


@pytest.mark.parametrize(
    ("variable", "short_name", "parameters"),
    [
        (met_var.AirTemperature, "air_temperature", {"temperature_on_pressure_levels"}),
        (met_var.RelativeHumidity, "relative_humidity", {"relative_humidity_on_pressure_levels"}),
        (
            met_var.SpecificHumidity,
            "specific_humidity",
            {"temperature_on_pressure_levels", "relative_humidity_on_pressure_levels"},
        ),
    ],
)
def test_only_requested_variable_is_fetched_and_cached(
    tmp_path, monkeypatch, variable, short_name, parameters
):
    """Only the requested variable is cached, and only the S3 parameters it needs are
    downloaded: no derived or companion fields are added."""
    fetch_mock = _patch_fetch_with_values(monkeypatch)
    hour = datetime.datetime(2024, 9, 1, 0)

    mds = UM10(hour, variables=[variable], cachestore=DiskCacheStore(tmp_path)).open_metdataset()

    assert list(mds.data.data_vars) == [short_name]
    assert {call.args[2] for call in fetch_mock.call_args_list} == parameters


def test_parameters_are_unique_and_follow_requested_variables(tmp_path):
    hour = datetime.datetime(2024, 9, 1, 0)
    variables = [met_var.AirTemperature, met_var.SpecificHumidity, met_var.RelativeHumidity]

    dlib = UM10(hour, variables=variables, cachestore=DiskCacheStore(tmp_path))

    assert dlib._parameters == [
        "temperature_on_pressure_levels",
        "relative_humidity_on_pressure_levels",
    ]


def test_missing_unrequested_parameter_does_not_fail_existence_check(tmp_path, monkeypatch):
    """Relative humidity absent from the archive doesn't matter if only temperature is
    requested."""
    _patch_fetch_with_values(monkeypatch)
    _patch_listing(monkeypatch, absent=lambda key: "relative_humidity" in key)
    hour = datetime.datetime(2024, 9, 1, 0)

    mds = UM10(
        hour, variables=[met_var.AirTemperature], cachestore=DiskCacheStore(tmp_path)
    ).open_metdataset()

    assert list(mds.data.data_vars) == ["air_temperature"]


def test_open_metdataset_output_can_be_used_by_models(tmp_path, monkeypatch):
    """The returned variables use standard names, as the models expect."""
    _patch_fetch_with_values(monkeypatch)
    hour = datetime.datetime(2024, 9, 1, 0)

    mds = UM10(hour, cachestore=DiskCacheStore(tmp_path)).open_metdataset()

    assert "sac" in SAC(met=mds).eval().data


def test_derived_specific_humidity_has_its_own_attrs(tmp_path, monkeypatch):
    """Not the attributes of the pressure coordinate it is computed with."""
    _patch_fetch_with_values(monkeypatch)
    hour = datetime.datetime(2024, 9, 1, 0)

    mds = UM10(hour, cachestore=DiskCacheStore(tmp_path)).open_metdataset()

    assert mds.data["specific_humidity"].attrs == {
        "standard_name": "specific_humidity",
        "units": "kg kg**-1",
    }


def test_hand_computed_rhi_matches_thermo_rhi_via_q(tmp_path, monkeypatch):
    t_value, rh_value = 210.0, 0.9
    _patch_fetch_with_values(monkeypatch, t_value=t_value, rh_value=rh_value)
    hour = datetime.datetime(2024, 9, 1, 0)

    mds = UM10(hour, cachestore=DiskCacheStore(tmp_path)).open_metdataset()

    level_pa = mds.data["level"].isel(level=0).compute().item() * 100.0
    q = (
        mds.data["specific_humidity"]
        .isel(level=0, latitude=0, longitude=0, time=0)
        .compute()
        .item()
    )

    rhi_expected = (
        rh_value * thermo.e_sat_liquid(np.float64(t_value)) / thermo.e_sat_ice(np.float64(t_value))
    )
    rhi_actual = thermo.rhi(np.float64(q), np.float64(t_value), np.float64(level_pa))
    assert np.isclose(rhi_actual, rhi_expected, rtol=1e-4)


def test_open_metdataset_rejects_missing_pressure_level_in_fetched_data(tmp_path, monkeypatch):
    """A fetched field missing one of the *requested but otherwise-supported* cruise
    levels (as opposed to a level outside ``supported_pressure_levels`` entirely, which
    ``parse_pressure_levels`` already rejects at construction time) must fail loudly
    in ``_process_hour``, not silently select the wrong level."""
    levels_missing_300 = UKMO_PRESSURE_PA[1:]  # drops 300 hPa, keeps the other 6
    _patch_fetch_with_values(monkeypatch, pressure_pa=levels_missing_300)
    hour = datetime.datetime(2024, 9, 1, 0)

    dlib = UM10(hour, pressure_levels=[300], cachestore=DiskCacheStore(tmp_path))
    with pytest.raises(ValueError, match="expected exactly one 300"):
        dlib.open_metdataset()


@pytest.mark.parametrize("level", [5, 1013, -1])
def test_construction_rejects_pressure_level_outside_supported_set(tmp_path, level):
    """5 hPa exists in the temperature field only, not relative humidity, so isn't supported."""
    with pytest.raises(ValueError, match="not supported"):
        UM10(
            datetime.datetime(2024, 9, 1, 0),
            pressure_levels=[level],
            cachestore=DiskCacheStore(tmp_path),
        )


def test_construction_accepts_every_supported_pressure_level(tmp_path):
    hour = datetime.datetime(2024, 9, 1, 0)
    supported = UM10(hour, cachestore=DiskCacheStore(tmp_path)).supported_pressure_levels

    dlib = UM10(hour, pressure_levels=supported, cachestore=DiskCacheStore(tmp_path))

    assert dlib.pressure_levels == sorted(supported)


def test_default_pressure_levels_are_the_cruise_levels(tmp_path):
    dlib = UM10(datetime.datetime(2024, 9, 1, 0), cachestore=DiskCacheStore(tmp_path))

    assert dlib.pressure_levels == sorted(CRUISE_LEVELS_HPA)


def test_provider_dataset_attrs_no_warning(tmp_path, monkeypatch):
    _patch_fetch_with_values(monkeypatch)
    hour = datetime.datetime(2024, 9, 1, 0)

    mds = UM10(hour, cachestore=DiskCacheStore(tmp_path)).open_metdataset()

    assert mds.attrs["provider"] == PROVIDER
    assert mds.attrs["dataset"] == DATASET
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert mds.provider_attr == PROVIDER
        assert mds.dataset_attr == DATASET


# -- cachestore/lead_hours interaction ----------------------------------------


def test_create_cachepath_differs_by_lead_hours(tmp_path):
    t = dt(2024, 9, 1, 0)
    cachestore = DiskCacheStore(tmp_path)
    default_um = UM10(t, cachestore=cachestore)
    fixed_um = UM10(t, cachestore=cachestore, lead_hours=24)

    assert default_um.create_cachepath(t) != fixed_um.create_cachepath(t)


def test_create_cachepath_differs_by_forecast_time(tmp_path):
    t = dt(2024, 9, 1, 12)
    cachestore = DiskCacheStore(tmp_path)
    latest = UM10(t, cachestore=cachestore)
    earlier = UM10(t, cachestore=cachestore, forecast_time=dt(2024, 9, 1, 6))

    assert latest.forecast_time == dt(2024, 9, 1, 12)
    assert latest.create_cachepath(t) != earlier.create_cachepath(t)


def test_create_cachepath_same_for_same_forecast_file_however_requested(tmp_path):
    """A fixed lead of 24h at ``t`` reads the same file as the forecast one day earlier."""
    t = dt(2024, 9, 1, 0)
    cachestore = DiskCacheStore(tmp_path)
    at_lead = UM10(t, cachestore=cachestore, lead_hours=24)
    from_forecast = UM10(t, cachestore=cachestore, forecast_time=dt(2024, 8, 31, 0))

    assert at_lead.create_cachepath(t) == from_forecast.create_cachepath(t)


def test_lead_hours_threaded_through_to_live_fetch(tmp_path, monkeypatch):
    """A fixed ``lead_hours`` must be used for the S3 object key, not the
    latest forecast."""
    hour = dt(2024, 9, 1, 12)
    _patch_fetch_with_values(monkeypatch)

    captured = {}
    original_object_key = s3.object_key

    def capturing_object_key(run, valid_time, lead_hours, parameter):
        captured["run"] = run
        captured["lead_hours"] = lead_hours
        return original_object_key(run, valid_time, lead_hours, parameter)

    monkeypatch.setattr(s3, "object_key", capturing_object_key)

    UM10(hour, cachestore=DiskCacheStore(tmp_path), lead_hours=24).open_metdataset()

    assert captured["lead_hours"] == 24
    assert captured["run"] == hour - hours(24)


def test_forecast_time_threaded_through_to_live_fetch(tmp_path, monkeypatch):
    """Every hour is read from the one forecast, at its own lead."""
    time_range = (dt(2024, 9, 1, 12), dt(2024, 9, 1, 14))
    _patch_fetch_with_values(monkeypatch)

    captured = []
    original_object_key = s3.object_key

    def capturing_object_key(run, valid_time, lead_hours, parameter):
        captured.append((run, lead_hours))
        return original_object_key(run, valid_time, lead_hours, parameter)

    monkeypatch.setattr(s3, "object_key", capturing_object_key)

    UM10(
        time_range, forecast_time=dt(2024, 9, 1, 6), cachestore=DiskCacheStore(tmp_path)
    ).open_metdataset()

    assert {run for run, _ in captured} == {dt(2024, 9, 1, 6)}
    assert {lead for _, lead in captured} == {6, 7, 8}


##########################
# time and forecast cycles
##########################


def make(time, tmp_path, **kwargs) -> UM10:
    return UM10(time, cachestore=DiskCacheStore(tmp_path), **kwargs)


def test_default_forecast_is_latest_at_or_before_start(tmp_path):
    dlib = make(dt(2026, 3, 15, 7), tmp_path)

    assert dlib.forecast_time == dt(2026, 3, 15, 6)
    assert dlib.timesteps == [dt(2026, 3, 15, 7)]


def test_default_forecast_for_time_range_is_one_forecast(tmp_path):
    dlib = make((dt(2026, 3, 15, 7), dt(2026, 3, 16, 7)), tmp_path)

    assert dlib.forecast_time == dt(2026, 3, 15, 6)
    assert dlib.timesteps[0] == dt(2026, 3, 15, 7)
    assert dlib.timesteps[-1] == dt(2026, 3, 16, 7)
    assert len(dlib.timesteps) == 25


def test_default_forecast_is_older_when_latest_is_too_short(tmp_path):
    """70 hours from 07Z is beyond the 06Z forecast (+69h), so the 00Z forecast is used."""
    start = dt(2026, 3, 15, 7)
    dlib = make((start, start + hours(70)), tmp_path)

    assert dlib.forecast_time == dt(2026, 3, 15, 0)


def test_default_forecast_range_too_long_for_any_forecast(tmp_path):
    start = dt(2026, 3, 15, 0)
    with pytest.raises(ValueError, match="not contained in any single forecast"):
        make((start, start + hours(170)), tmp_path)


def test_forecast_time_explicit(tmp_path):
    dlib = make(dt(2026, 3, 15, 9), tmp_path, forecast_time="2026-03-15 00:00")

    assert dlib.forecast_time == dt(2026, 3, 15, 0)


@pytest.mark.parametrize("forecast_time", ["2026-03-15 03:00", "2026-03-15 06:30"])
def test_forecast_time_must_be_a_forecast_cycle(tmp_path, forecast_time):
    with pytest.raises(ValueError, match="must be one of"):
        make(dt(2026, 3, 15, 9), tmp_path, forecast_time=forecast_time)


def test_forecast_time_must_be_parseable(tmp_path):
    with pytest.raises(ValueError, match="Failed to parse forecast time"):
        make(dt(2026, 3, 15, 9), tmp_path, forecast_time="not a time")


def test_forecast_time_after_first_timestep_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="after first timestep"):
        make(dt(2026, 3, 15, 5), tmp_path, forecast_time="2026-03-15 06:00")


def test_forecast_time_beyond_end_of_forecast_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="beyond end of forecast"):
        make(dt(2026, 3, 18, 7), tmp_path, forecast_time="2026-03-15 06:00")  # +73h > +69h


def test_timestep_freq_hourly_within_first_54_hours(tmp_path):
    dlib = make((dt(2026, 3, 15, 0), dt(2026, 3, 17, 6)), tmp_path)  # +54h

    assert dlib.timesteps[1] - dlib.timesteps[0] == hours(1)
    assert len(dlib.timesteps) == 55


def test_timestep_freq_three_hourly_beyond_54_hours(tmp_path):
    dlib = make((dt(2026, 3, 15, 0), dt(2026, 3, 17, 12)), tmp_path)  # +60h

    assert {b - a for a, b in itertools.pairwise(dlib.timesteps)} == {hours(3)}
    assert len(dlib.timesteps) == 21


def test_timestep_freq_six_hourly_beyond_144_hours(tmp_path):
    dlib = make((dt(2026, 3, 15, 0), dt(2026, 3, 21, 6)), tmp_path)  # +150h

    assert {b - a for a, b in itertools.pairwise(dlib.timesteps)} == {hours(6)}
    assert dlib.timesteps[-1] == dt(2026, 3, 21, 6)


def test_timesteps_stay_on_the_forecast_grid(tmp_path):
    """3-hourly timesteps are 3-hourly from the forecast time, not from midnight."""
    dlib = make((dt(2026, 3, 15, 13, 30), dt(2026, 3, 18, 13)), tmp_path)  # 12Z, to +73h

    assert dlib.forecast_time == dt(2026, 3, 15, 12)
    assert all((t - dlib.forecast_time) % hours(3) == hours(0) for t in dlib.timesteps)
    assert dlib.timesteps[0] >= dlib.forecast_time


def test_explicit_timestep_freq_coarser_than_available(tmp_path):
    dlib = make((dt(2026, 3, 15, 0), dt(2026, 3, 15, 12)), tmp_path, timestep_freq="6h")

    assert dlib.timesteps == [dt(2026, 3, 15, h) for h in (0, 6, 12)]


def test_explicit_timestep_freq_finer_than_available_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="cannot support requested timestep frequency"):
        make((dt(2026, 3, 15, 0), dt(2026, 3, 17, 12)), tmp_path, timestep_freq="1h")


def test_explicit_timestep_freq_ceiling_past_end_of_forecast_is_rejected(tmp_path):
    """6-hourly timesteps ending at +68h round up to +72h, past the end of a 06Z forecast."""
    with pytest.raises(ValueError, match="beyond end of forecast"):
        make(
            (dt(2026, 3, 15, 6), dt(2026, 3, 18, 2)),  # +68h
            tmp_path,
            forecast_time="2026-03-15 06:00",
            timestep_freq="6h",
        )


def test_forecast_time_and_lead_hours_are_exclusive(tmp_path):
    with pytest.raises(ValueError, match="Only one of forecast_time and lead_hours"):
        make(dt(2026, 3, 15, 0), tmp_path, forecast_time="2026-03-14 00:00", lead_hours=24)


def test_lead_hours_defaults_to_six_hourly_timesteps_on_the_lead_grid(tmp_path):
    dlib = make((dt(2026, 3, 15, 0), dt(2026, 3, 16, 0)), tmp_path, lead_hours=3)

    assert dlib.forecast_time is None
    assert {b - a for a, b in itertools.pairwise(dlib.timesteps)} == {hours(6)}
    assert all((t - hours(3)).hour in (0, 6, 12, 18) for t in dlib.timesteps)


def test_lead_hours_has_no_single_forecast_length_limit(tmp_path):
    dlib = make((dt(2026, 3, 1, 0), dt(2026, 3, 31, 0)), tmp_path, lead_hours=24)

    assert len(dlib.timesteps) == 121


@pytest.mark.parametrize("timestep_freq", ["1h", "3h", "9h"])
def test_lead_hours_requires_timestep_freq_multiple_of_six_hours(tmp_path, timestep_freq):
    with pytest.raises(ValueError, match="multiple of 6h"):
        make(dt(2026, 3, 15, 0), tmp_path, lead_hours=24, timestep_freq=timestep_freq)


def test_lead_hours_beyond_short_forecasts_defaults_to_twelve_hourly_timesteps(tmp_path):
    """Only 00/12Z forecasts reach a lead of 150h, so timesteps are 12 hours apart."""
    dlib = make((dt(2026, 3, 15, 0), dt(2026, 3, 17, 0)), tmp_path, lead_hours=150)

    assert {b - a for a, b in itertools.pairwise(dlib.timesteps)} == {hours(12)}
    assert all((t - hours(150)).hour in (0, 12) for t in dlib.timesteps)
    for t in dlib.timesteps:
        assert dlib._run_and_lead(t)[1] == 150


def test_lead_hours_beyond_short_forecasts_rejects_six_hourly_timesteps(tmp_path):
    with pytest.raises(ValueError, match="multiple of 12h"):
        make(dt(2026, 3, 15, 0), tmp_path, lead_hours=150, timestep_freq="6h")


@pytest.mark.parametrize(("lead_hours", "cadence"), [(0, 6), (69, 6), (70, 12), (168, 12)])
def test_run_cadence_hours_for_lead(lead_hours, cadence):
    assert s3.run_cadence_hours_for_lead(lead_hours) == cadence


def test_lead_hours_accepts_twelve_hourly_timesteps(tmp_path):
    dlib = make(
        (dt(2026, 3, 15, 0), dt(2026, 3, 16, 0)), tmp_path, lead_hours=24, timestep_freq="12h"
    )

    assert dlib.timesteps == [dt(2026, 3, 15, 0), dt(2026, 3, 15, 12), dt(2026, 3, 16, 0)]


@pytest.mark.parametrize(
    ("lead_hours", "match"),
    [
        (-6, "before the forecast time"),
        (169, "after the end"),
        (153, "only publishes every 6h"),
    ],
)
def test_lead_hours_the_archive_does_not_publish_is_rejected(tmp_path, lead_hours, match):
    with pytest.raises(ValueError, match=match):
        make(dt(2026, 3, 15, 0), tmp_path, lead_hours=lead_hours)
