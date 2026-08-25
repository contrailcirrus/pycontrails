"""Tests for the Met Office datalib (``s3``, ``ukmo``)."""

from __future__ import annotations

import datetime
import io
import os
import pathlib
import warnings
from unittest import mock

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pycontrails.core.cache import DiskCacheStore, GCPCacheStore
from pycontrails.datalib.metoffice import s3
from pycontrails.datalib.metoffice.ukmo import (
    DATASET,
    PROVIDER,
    MetOfficeDataNotFoundError,
    MetOfficeUM,
)
from pycontrails.physics import thermo
from tests import OFFLINE

##############
# s3 utilities
##############

FIXTURE_DIR = pathlib.Path(__file__).parents[1] / "fixtures" / "cache" / "metoffice"
CACHE_DIR = FIXTURE_DIR / ".cache"

#: The run/validity/lead fetched as the reference object for the bit-identity check
REFERENCE_RUN = datetime.datetime(2026, 8, 3, 0, 0)
REFERENCE_VALIDITY = datetime.datetime(2026, 8, 3, 0, 0)
REFERENCE_LEAD_HOURS = 0


def _cached_reference_file(parameter: str) -> pathlib.Path:
    """Download (and locally cache) the full reference object for ``parameter``."""
    key = s3.object_key(REFERENCE_RUN, REFERENCE_VALIDITY, REFERENCE_LEAD_HOURS, parameter)
    cache_path = CACHE_DIR / pathlib.PurePosixPath(key).name
    if cache_path.exists():
        return cache_path

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    body = s3.filesystem().get_object(Bucket=s3.BUCKET, Key=key)["Body"].read()
    cache_path.write_bytes(body)
    return cache_path


class _CountingClient:
    """Wraps a boto3 S3 client, counting bytes actually returned by ``get_object``."""

    def __init__(self, client) -> None:
        self._client = client
        self.bytes_read = 0

    def get_object(self, **kwargs):
        response = self._client.get_object(**kwargs)
        body = response["Body"].read()
        self.bytes_read += len(body)
        response["Body"] = io.BytesIO(body)
        return response

    def __getattr__(self, name):
        return getattr(self._client, name)


def test_run_and_lead_for_validity_within_cycle():
    run, lead_hours = s3.run_and_lead_for_validity(datetime.datetime(2024, 9, 15, 3))
    assert run == datetime.datetime(2024, 9, 15, 0)
    assert lead_hours == 3


def test_run_and_lead_for_validity_on_cycle_boundary():
    run, lead_hours = s3.run_and_lead_for_validity(datetime.datetime(2024, 9, 15, 18))
    assert run == datetime.datetime(2024, 9, 15, 18)
    assert lead_hours == 0


def test_run_and_lead_for_validity_rejects_non_hourly():
    with pytest.raises(ValueError, match="must fall on the hour"):
        s3.run_and_lead_for_validity(datetime.datetime(2024, 9, 15, 3, 30))


def test_run_hours_for_lead_all_cycles_at_and_below_boundary():
    assert s3.run_hours_for_lead(54) == (0, 6, 12, 18)


def test_run_hours_for_lead_long_cycles_only_above_boundary():
    assert s3.run_hours_for_lead(55) == (0, 12)


def test_run_for_validity_at_lead_matches_shortest_lead_at_lead_zero():
    validity = datetime.datetime(2024, 9, 15, 18)
    expected_run, _ = s3.run_and_lead_for_validity(validity)
    assert s3.run_for_validity_at_lead(validity, 0) == expected_run


def test_run_for_validity_at_lead_solves_run_minus_lead():
    run = s3.run_for_validity_at_lead(datetime.datetime(2024, 9, 15, 12), 24)
    assert run == datetime.datetime(2024, 9, 14, 12)


def test_run_for_validity_at_lead_rejects_unreachable_run_hour():
    # validity 2024-09-15T06 at lead 72 implies run 2024-09-12T06, but only 00/12Z
    # runs reach lead 72.
    with pytest.raises(ValueError, match="does not reach that lead"):
        s3.run_for_validity_at_lead(datetime.datetime(2024, 9, 15, 6), 72)


def test_run_for_validity_at_lead_rejects_non_hourly():
    with pytest.raises(ValueError, match="must fall on the hour"):
        s3.run_for_validity_at_lead(datetime.datetime(2024, 9, 15, 3, 30), 24)


def test_available_validity_times_at_lead_counts_per_day():
    start = datetime.datetime(2024, 9, 1, 0)
    end = datetime.datetime(2024, 9, 7, 23)

    at_24 = s3.available_validity_times_at_lead(start, end, 24)
    at_72 = s3.available_validity_times_at_lead(start, end, 72)

    assert len(at_24) == 7 * 4
    assert len(at_72) == 7 * 2
    assert all(v.hour in (0, 6, 12, 18) for v in at_24)
    assert all(v.hour in (0, 12) for v in at_72)


def test_matched_validity_times_reduces_to_hour_0_and_12():
    start = datetime.datetime(2024, 9, 1, 0)
    end = datetime.datetime(2024, 9, 7, 23)

    matched = s3.matched_validity_times(start, end, (0, 24, 48, 72))

    assert len(matched) == 7 * 2
    assert all(v.hour in (0, 12) for v in matched)


def test_matched_validity_times_empty_leads():
    start = datetime.datetime(2024, 9, 1)
    end = datetime.datetime(2024, 9, 2)
    assert s3.matched_validity_times(start, end, ()) == []


def test_object_key_format_for_pressure_level_variable():
    run = datetime.datetime(2026, 8, 3, 0, 0)
    validity = datetime.datetime(2026, 8, 3, 0, 0)
    key = s3.object_key(run, validity, 0, "temperature_on_pressure_levels")
    assert key == (
        "global-deterministic-10km/20260803T0000Z/"
        "20260803T0000Z-PT0000H00M-temperature_on_pressure_levels.nc"
    )


def test_object_key_lead_and_minutes_padding():
    run = datetime.datetime(2026, 8, 3, 0, 0)
    validity = datetime.datetime(2026, 8, 4, 0, 0)
    key = s3.object_key(run, validity, 24, "temperature_on_pressure_levels")
    assert key == (
        "global-deterministic-10km/20260803T0000Z/"
        "20260804T0000Z-PT0024H00M-temperature_on_pressure_levels.nc"
    )


def test_select_cruise_subset_crops_to_passed_in_extent():
    pressure_pa = np.asarray(s3.CRUISE_LEVELS_HPA, dtype=np.float64) * 100.0
    latitude = np.linspace(30.0, 70.0, 41)  # ascending, 1 degree steps
    longitude = np.linspace(-40.0, 0.0, 41)  # 1 degree steps
    data = np.zeros((len(pressure_pa), len(latitude), len(longitude)), dtype=np.float32)
    ds = xr.Dataset(
        {"air_temperature": (("pressure", "latitude", "longitude"), data)},
        coords={"pressure": pressure_pa, "latitude": latitude, "longitude": longitude},
    )

    shanwick_extent = (-30.0, -10.0, 45.0, 61.0)
    da = s3.select_cruise_subset(ds, "temperature_on_pressure_levels", extent=shanwick_extent)

    assert da["longitude"].values.min() >= -30.0
    assert da["longitude"].values.max() <= -10.0
    assert da["latitude"].values.min() >= 45.0
    assert da["latitude"].values.max() <= 61.0


def test_select_cruise_subset_no_extent_returns_full_domain():
    pressure_pa = np.asarray(s3.CRUISE_LEVELS_HPA, dtype=np.float64) * 100.0
    latitude = np.linspace(10.0, 60.0, 51)
    longitude = np.linspace(-140.0, -50.0, 91)
    data = np.zeros((len(pressure_pa), len(latitude), len(longitude)), dtype=np.float32)
    ds = xr.Dataset(
        {"air_temperature": (("pressure", "latitude", "longitude"), data)},
        coords={"pressure": pressure_pa, "latitude": latitude, "longitude": longitude},
    )

    da = s3.select_cruise_subset(ds, "temperature_on_pressure_levels")

    np.testing.assert_array_equal(da["longitude"].values, longitude)
    np.testing.assert_array_equal(da["latitude"].values, latitude)


@pytest.mark.skipif(OFFLINE, reason="offline")
@pytest.mark.parametrize(
    "parameter", ["temperature_on_pressure_levels", "relative_humidity_on_pressure_levels"]
)
def test_range_read_bit_identical_to_reference(parameter):
    fixture_path = _cached_reference_file(parameter)

    local_ds = xr.open_dataset(
        fixture_path, engine="h5netcdf", decode_times=True, decode_timedelta=True
    )
    expected = s3.select_cruise_subset(
        local_ds, parameter, key=str(fixture_path), extent=s3.CONUS_EXTENT
    ).load()

    key = s3.object_key(REFERENCE_RUN, REFERENCE_VALIDITY, REFERENCE_LEAD_HOURS, parameter)
    counting_client = _CountingClient(s3.filesystem())
    actual = s3.fetch_pressure_level_field(
        counting_client,
        key,
        parameter,
        run=REFERENCE_RUN,
        validity=REFERENCE_VALIDITY,
        lead_hours=REFERENCE_LEAD_HOURS,
        extent=s3.CONUS_EXTENT,
    )

    np.testing.assert_array_equal(actual.values, expected.values)
    assert list(actual.dims) == ["pressure", "latitude", "longitude"]
    assert actual.sizes["pressure"] == len(s3.CRUISE_LEVELS_HPA)

    whole_file_size = os.path.getsize(fixture_path)
    ratio = counting_client.bytes_read / whole_file_size
    print(
        f"\n[{parameter}] range-read bytes={counting_client.bytes_read} "
        f"whole-file bytes={whole_file_size} ratio={ratio:.4f}"
    )
    assert counting_client.bytes_read < whole_file_size


######
# ukmo
######

UKMO_LATITUDE = np.array([30.0, 31.0])
UKMO_LONGITUDE = np.array([-100.0, -99.0])
UKMO_PRESSURE_PA = np.asarray(s3.CRUISE_LEVELS_HPA, dtype=np.float64) * 100.0


class _FakeClient:
    """Stand-in for a boto3 client -- never actually called since the fetch
    functions themselves are replaced below."""


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
    dlib: MetOfficeUM,
    t: datetime.datetime,
    *,
    t_value: float = 220.0,
    rh_value: float = 0.8,
) -> None:
    """Pre-populate ``dlib.cachestore`` with a correctly-shaped hit for hour ``t``.

    Matches exactly what :meth:`MetOfficeUM._process_hour` would itself write --
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

    mds = MetOfficeUM(hour, cachestore=DiskCacheStore(tmp_path)).open_metdataset()

    assert fetch_mock.call_count == 2  # one per required parameter
    assert mds.data.sizes["time"] == 1


def test_open_metdataset_reads_archive_hit_without_fetching(tmp_path, monkeypatch):
    fetch_mock = _patch_fetch_with_values(monkeypatch)
    hour = datetime.datetime(2024, 9, 1, 0)
    dlib = MetOfficeUM(hour, cachestore=DiskCacheStore(tmp_path))
    _write_cache_hit(dlib, hour, t_value=999.0)

    mds = dlib.open_metdataset()

    fetch_mock.assert_not_called()
    np.testing.assert_allclose(mds.data["t"].values, 999.0)


def test_open_metdataset_partial_hit_only_fetches_miss(tmp_path, monkeypatch):
    fetch_mock = _patch_fetch_with_values(monkeypatch, t_value=210.0)
    hours = [datetime.datetime(2024, 9, 1, 0), datetime.datetime(2024, 9, 1, 1)]
    dlib = MetOfficeUM(hours, cachestore=DiskCacheStore(tmp_path))
    _write_cache_hit(dlib, hours[0], t_value=999.0)

    mds = dlib.open_metdataset()

    assert fetch_mock.call_count == 2  # only hours[1]'s 2 parameters
    assert mds.data.sizes["time"] == 2
    hit_val = (
        mds.data.sel(time=hours[0])["t"].isel(level=0, latitude=0, longitude=0).compute().item()
    )
    miss_val = (
        mds.data.sel(time=hours[1])["t"].isel(level=0, latitude=0, longitude=0).compute().item()
    )
    assert np.isclose(hit_val, 999.0)
    assert np.isclose(miss_val, 210.0)


def test_cache_dataset_writes_correctly_to_gcp_cache_store(tmp_path, monkeypatch):
    """``cache_dataset`` must stage through a real local file, since
    ``GCPCacheStore.path()`` returns a bucket-relative key, not a local
    filesystem path -- writing straight to it (as a plain ``CacheStore.path()``
    would suggest) is only safe for ``DiskCacheStore``.
    """
    _patch_fetch_with_values(monkeypatch)
    hour = datetime.datetime(2024, 9, 1, 0)

    class _FakeBlob:
        def exists(self):
            return False

    class _FakeBucket:
        def blob(self, key):
            return _FakeBlob()

    disk_mirror = DiskCacheStore(tmp_path / "gcp-mirror")
    cachestore = GCPCacheStore(bucket="fake-bucket", disk_cache=disk_mirror, read_only=True)
    monkeypatch.setattr(cachestore, "_bucket", _FakeBucket())

    dlib = MetOfficeUM(hour, cachestore=cachestore)
    mds = dlib.open_metdataset()

    assert mds.data.sizes["time"] == 1
    assert cachestore.exists(dlib.create_cachepath(hour))


def test_download_dataset_raises_when_live_fetch_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(s3, "filesystem", _FakeClient)

    def failing_fetch(fs, key, parameter, **kwargs):
        raise RuntimeError("simulated 404")

    monkeypatch.setattr(s3, "fetch_pressure_level_field", failing_fetch)

    with pytest.raises(MetOfficeDataNotFoundError, match="could not be fetched"):
        MetOfficeUM(
            datetime.datetime(2024, 9, 1, 0), cachestore=DiskCacheStore(tmp_path)
        ).open_metdataset()


def test_open_metdataset_rejects_descending_latitude(tmp_path, monkeypatch):
    _patch_fetch_with_values(monkeypatch, latitude=np.array([31.0, 30.0]))
    hour = datetime.datetime(2024, 9, 1, 0)

    with pytest.raises(AssertionError, match="latitude"):
        MetOfficeUM(hour, cachestore=DiskCacheStore(tmp_path)).open_metdataset()


def test_specific_humidity_round_trips_to_relative_humidity(tmp_path, monkeypatch):
    t_value, rh_value = 215.0, 0.7
    _patch_fetch_with_values(monkeypatch, t_value=t_value, rh_value=rh_value)
    hour = datetime.datetime(2024, 9, 1, 0)

    mds = MetOfficeUM(hour, cachestore=DiskCacheStore(tmp_path)).open_metdataset()

    # MetDataset sorts `level` ascending, so index 0 is 150 hPa, not
    # CRUISE_LEVELS_HPA[0] (300 hPa) -- read the actual level back rather than
    # assuming which physical level ends up at a given index.
    level_pa = mds.data["level"].isel(level=0).compute().item() * 100.0
    expected_q = rh_value * thermo.q_sat_liquid(np.float64(t_value), np.float64(level_pa))
    q = mds.data["q"].isel(level=0, latitude=0, longitude=0, time=0).compute().item()
    assert np.isclose(q, expected_q, rtol=1e-4)

    recovered_rh = q / thermo.q_sat_liquid(np.float64(t_value), np.float64(level_pa))
    assert np.isclose(recovered_rh, rh_value, rtol=1e-4)


def test_hand_computed_rhi_matches_thermo_rhi_via_q(tmp_path, monkeypatch):
    t_value, rh_value = 210.0, 0.9
    _patch_fetch_with_values(monkeypatch, t_value=t_value, rh_value=rh_value)
    hour = datetime.datetime(2024, 9, 1, 0)

    mds = MetOfficeUM(hour, cachestore=DiskCacheStore(tmp_path)).open_metdataset()

    level_pa = mds.data["level"].isel(level=0).compute().item() * 100.0
    q = mds.data["q"].isel(level=0, latitude=0, longitude=0, time=0).compute().item()

    rhi_expected = (
        rh_value * thermo.e_sat_liquid(np.float64(t_value)) / thermo.e_sat_ice(np.float64(t_value))
    )
    rhi_actual = thermo.rhi(np.float64(q), np.float64(t_value), np.float64(level_pa))
    assert np.isclose(rhi_actual, rhi_expected, rtol=1e-4)


def test_open_metdataset_rejects_missing_pressure_level_in_fetched_data(tmp_path, monkeypatch):
    """A fetched field missing one of the *requested but otherwise-supported* cruise
    levels (as opposed to a level outside ``s3.CRUISE_LEVELS_HPA`` entirely, which
    ``parse_pressure_levels`` already rejects at construction time) must fail loudly
    in ``_process_hour``, not silently select the wrong level."""
    levels_missing_300 = UKMO_PRESSURE_PA[1:]  # drops 300 hPa, keeps the other 6
    _patch_fetch_with_values(monkeypatch, pressure_pa=levels_missing_300)
    hour = datetime.datetime(2024, 9, 1, 0)

    dlib = MetOfficeUM(hour, pressure_levels=[300], cachestore=DiskCacheStore(tmp_path))
    with pytest.raises(ValueError, match="expected exactly one 300"):
        dlib.open_metdataset()


def test_construction_rejects_pressure_level_outside_supported_set(tmp_path):
    with pytest.raises(ValueError, match="not supported"):
        MetOfficeUM(
            datetime.datetime(2024, 9, 1, 0),
            pressure_levels=[100],
            cachestore=DiskCacheStore(tmp_path),
        )


def test_provider_dataset_attrs_no_warning(tmp_path, monkeypatch):
    _patch_fetch_with_values(monkeypatch)
    hour = datetime.datetime(2024, 9, 1, 0)

    mds = MetOfficeUM(hour, cachestore=DiskCacheStore(tmp_path)).open_metdataset()

    assert mds.attrs["provider"] == PROVIDER
    assert mds.attrs["dataset"] == DATASET
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert mds.provider_attr == PROVIDER
        assert mds.dataset_attr == DATASET


# -- cachestore/lead_hours interaction ----------------------------------------


def test_create_cachepath_differs_by_lead_hours(tmp_path):
    t = datetime.datetime(2024, 9, 1, 0)
    cachestore = DiskCacheStore(tmp_path)
    default_um = MetOfficeUM(t, cachestore=cachestore)
    fixed_um = MetOfficeUM(t, cachestore=cachestore, lead_hours=24)

    assert default_um.create_cachepath(t) != fixed_um.create_cachepath(t)


def test_fixed_lead_zero_cachepath_differs_from_shortest_lead(tmp_path):
    """lead_hours=0 must never be conflated with lead_hours=None (truthiness bug)."""
    t = datetime.datetime(2024, 9, 1, 0)
    cachestore = DiskCacheStore(tmp_path)
    default_um = MetOfficeUM(t, cachestore=cachestore)
    lead_zero_um = MetOfficeUM(t, cachestore=cachestore, lead_hours=0)

    assert default_um.create_cachepath(t) != lead_zero_um.create_cachepath(t)


def test_lead_hours_threaded_through_to_live_fetch(tmp_path, monkeypatch):
    """A fixed ``lead_hours`` must be used for the S3 object key, not the
    shortest-available lead."""
    hour = datetime.datetime(2024, 9, 1, 12)
    _patch_fetch_with_values(monkeypatch)

    captured = {}
    original_object_key = s3.object_key

    def capturing_object_key(run, validity, lead_hours, parameter):
        captured["run"] = run
        captured["lead_hours"] = lead_hours
        return original_object_key(run, validity, lead_hours, parameter)

    monkeypatch.setattr(s3, "object_key", capturing_object_key)

    MetOfficeUM(hour, cachestore=DiskCacheStore(tmp_path), lead_hours=24).open_metdataset()

    assert captured["lead_hours"] == 24
    assert captured["run"] == s3.run_for_validity_at_lead(hour, 24)
