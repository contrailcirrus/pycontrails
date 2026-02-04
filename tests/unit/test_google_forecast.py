"""Unit tests for Google Forecast datalib.

You can also run optional integration tests, by setting the
GOOGLE_API_KEY environment variable before running this test file.
"""

from __future__ import annotations

import os
from unittest import mock

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pycontrails import MetDataset
from pycontrails.core import cache
from pycontrails.datalib.google_forecast import (
    EffectiveEnergyForcing,
    GoogleForecast,
    Severity,
)
from pycontrails.physics import units


@pytest.fixture()
def mock_requests():
    with mock.patch("requests.get") as m:
        yield m


@pytest.fixture()
def local_cache(tmp_path):
    return cache.DiskCacheStore(cache_dir=tmp_path)


def test_google_forecast_init(local_cache):
    """Test initialization."""
    # Test with API key
    gf = GoogleForecast(
        time="2022-01-01 12:00:00",
        key="test-key",
        cachestore=local_cache,
    )
    assert gf.url == "https://contrails.googleapis.com/v2/grids"
    assert gf._credentials == "test-key"
    assert gf.timesteps == [pd.Timestamp("2022-01-01 12:00:00")]
    assert gf._request_headers == {"x-goog-api-key": "test-key"}

    # Test with default credentials (mocked)
    with (
        mock.patch("google.auth.default", return_value=("default-creds", "project")),
        mock.patch.dict(os.environ),
    ):
        if "GOOGLE_API_KEY" in os.environ:
            del os.environ["GOOGLE_API_KEY"]

        gf = GoogleForecast(time="2022-01-01 12:00:00", cachestore=local_cache)
        assert gf._credentials is None
        assert gf._request_headers == {"x-goog-api-key": "default-creds"}


def test_google_forecast_download(mock_requests, local_cache):
    """Test download_dataset."""

    time = pd.Timestamp("2022-01-01 12:00:00")
    gf = GoogleForecast(
        time=time,
        variables=[Severity, EffectiveEnergyForcing],
        key="test-key",
        cachestore=local_cache,
    )

    # Mock response content
    ds = xr.Dataset(
        {
            "contrails": (
                ("time", "level", "latitude", "longitude"),
                np.full((1, 1, 2, 2), 0.5),
            )
        },
        coords={
            "time": [time],
            "level": [200],
            "latitude": [10, 20],
            "longitude": [10, 20],
        },
    )
    mock_resp = mock.Mock()
    mock_resp.content = ds.to_netcdf()
    mock_resp.status_code = 200
    mock_resp.headers = {}
    mock_requests.return_value = mock_resp

    gf.download_dataset(gf.timesteps)

    # Check request
    mock_requests.assert_called_once()
    _, kwargs = mock_requests.call_args
    assert kwargs["params"]["time"] == "2022-01-01T12:00:00"
    assert "contrails" in kwargs["params"]["data"]
    assert "expected_effective_energy_forcing" in kwargs["params"]["data"]

    # Check cache
    cache_path = gf.create_cachepath(time)
    assert local_cache.exists(cache_path)

    # Verify content in cache
    ds_cache = xr.open_dataset(cache_path)
    assert np.allclose(ds_cache["contrails"], 0.5)


def test_google_forecast_flight_level_conversion(mock_requests, local_cache):
    """Test that flight_level is converted to level (pressure)."""

    time = pd.Timestamp("2022-01-01 12:00:00")
    gf = GoogleForecast(
        time=time,
        key="test-key",
        cachestore=local_cache,
    )

    # Create dataset with flight_level
    ds = xr.Dataset(
        {
            "contrails": (
                ("time", "flight_level", "latitude", "longitude"),
                np.zeros((1, 1, 2, 2)),
            )
        },
        coords={
            "time": [time],
            "flight_level": [300],  # FL300
            "latitude": [10, 20],
            "longitude": [10, 20],
        },
    )
    mock_resp = mock.Mock()
    mock_resp.content = ds.to_netcdf()
    mock_resp.status_code = 200
    mock_requests.return_value = mock_resp

    gf.download_dataset(gf.timesteps)

    # Load from cache and check conversion
    cache_path = gf.create_cachepath(time)
    ds_out = xr.open_dataset(cache_path)

    assert "level" in ds_out.dims
    assert "flight_level" not in ds_out.dims

    # FL300 -> ~300hPa (actually around 300.5 hPa or so depending on units)
    # units.ft_to_pl(30000)
    expected_pl = units.ft_to_pl(30000)
    assert np.allclose(ds_out["level"], expected_pl)


def test_google_forecast_open_metdataset(mock_requests, local_cache):
    """Test open_metdataset loads from cache/download."""

    time = pd.Timestamp("2022-01-01 12:00:00")
    gf = GoogleForecast(
        time=time,
        key="test-key",
        cachestore=local_cache,
    )

    # Mock download (simplest empty dataset with right coords)
    ds = xr.Dataset(
        {
            "contrails": (
                ("time", "level", "latitude", "longitude"),
                np.zeros((1, 1, 2, 2)),
            )
        },
        coords={
            "time": [time],
            "level": [200],
            "latitude": [10, 20],
            "longitude": [10, 20],
        },
    )
    mock_resp = mock.Mock()
    mock_resp.content = ds.to_netcdf()
    mock_resp.status_code = 200
    mock_requests.return_value = mock_resp

    mds = gf.open_metdataset()

    assert isinstance(mds, MetDataset)
    assert mds.data.attrs["provider"] == "Google"
    assert mds.data.attrs["dataset"] == "Contrails Forecast"

    # Check that it downloaded (since cache was empty)
    mock_requests.assert_called()

    # Check actual cache file existence
    cache_path = gf.create_cachepath(time)
    assert local_cache.exists(cache_path)

    # Close dataset to ensure no locks (though Linux generally allows this)
    mds.data.close()

    # Call again - should use cache and NOT download
    mock_requests.reset_mock()
    mds2 = gf.open_metdataset()
    mock_requests.assert_not_called()
    assert mds2.data.equals(mds.data)


def _check_mds_structure(mds: MetDataset, time: pd.Timestamp) -> None:
    assert isinstance(mds, MetDataset)

    # High level checks
    assert mds.data.sizes["time"] == 1
    assert mds.data["time"].values[0] == time.to_datetime64()
    assert mds.data.attrs["provider"] == "Google"

    # Global coverage
    assert mds.data["latitude"].min() == -90
    assert mds.data["latitude"].max() == 90
    assert mds.data["longitude"].min() == -180
    assert mds.data["longitude"].max() >= 179.75

    # Vertical coverage: at least FL270 to FL440
    # Pressure decreases with altitude, so max pressure >= FL270 equivalent
    # and min pressure <= FL440 equivalent.
    assert mds.data["level"].max() >= units.ft_to_pl(27000)
    assert mds.data["level"].min() <= units.ft_to_pl(44000)


def _check_mds_values(mds: MetDataset, severity: bool = False, eeef: bool = False) -> None:
    if severity:
        assert "contrails" in mds.data
        assert (mds.data["contrails"] > 0).any()
    else:
        assert "contrails" not in mds.data

    if eeef:
        assert "expected_effective_energy_forcing" in mds.data
        assert (mds.data["expected_effective_energy_forcing"] > 2e7).any()
    else:
        assert "expected_effective_energy_forcing" not in mds.data


@pytest.mark.skipif("GOOGLE_API_KEY" not in os.environ, reason="GOOGLE_API_KEY not set")
def test_integration_severity(local_cache):
    """Integration test for Severity variable."""
    # Forecast for 24h from now, rounded to hour
    # Use utcnow() to get naive UTC timestamp, consistent with other tests
    time = pd.Timestamp.utcnow().ceil("h") + pd.Timedelta("24h")

    gf = GoogleForecast(
        time=time,
        variables=[Severity],
        cachestore=local_cache,
    )
    mds = gf.open_metdataset()

    mds = gf.open_metdataset()

    _check_mds_structure(mds, time)
    _check_mds_values(mds, severity=True, eeef=False)


@pytest.mark.skipif("GOOGLE_API_KEY" not in os.environ, reason="GOOGLE_API_KEY not set")
def test_integration_eeef(local_cache):
    """Integration test for EffectiveEnergyForcing variable."""
    # Forecast for 24h from now, rounded to hour
    time = pd.Timestamp.utcnow().ceil("h") + pd.Timedelta("24h")

    gf = GoogleForecast(
        time=time,
        variables=[EffectiveEnergyForcing],
        cachestore=local_cache,
    )
    mds = gf.open_metdataset()

    mds = gf.open_metdataset()

    _check_mds_structure(mds, time)
    _check_mds_values(mds, severity=False, eeef=True)


@pytest.mark.skipif("GOOGLE_API_KEY" not in os.environ, reason="GOOGLE_API_KEY not set")
def test_integration_both(local_cache):
    """Integration test for both Severity and EffectiveEnergyForcing variables."""
    # Forecast for 24h from now, rounded to hour
    time = pd.Timestamp.utcnow().ceil("h") + pd.Timedelta("24h")

    gf = GoogleForecast(
        time=time,
        variables=[Severity, EffectiveEnergyForcing],
        cachestore=local_cache,
    )
    mds = gf.open_metdataset()

    mds = gf.open_metdataset()

    _check_mds_structure(mds, time)
    _check_mds_values(mds, severity=True, eeef=True)


def test_google_forecast_default_no_cache():
    """Test that default cachestore is None."""
    gf = GoogleForecast(time="2022-01-01 12:00:00")
    assert gf.cachestore is None


def test_google_forecast_no_cache_store(mock_requests):
    """Test behavior when cachestore is None."""
    time = pd.Timestamp("2022-01-01 12:00:00")

    # Initialize without cachestore (default is None now)
    gf = GoogleForecast(
        time=time,
        key="test-key",
        cachestore=None,  # Explicitly None
    )

    # Mock response
    ds = xr.Dataset(
        {
            "contrails": (
                ("time", "level", "latitude", "longitude"),
                np.full((1, 1, 2, 2), 0.5),
            )
        },
        coords={
            "time": [time],
            "level": [200],
            "latitude": [10, 20],
            "longitude": [10, 20],
        },
    )
    mock_resp = mock.Mock()
    mock_resp.content = ds.to_netcdf()
    mock_resp.status_code = 200
    mock_requests.return_value = mock_resp

    # Run open_metdataset
    mds = gf.open_metdataset()

    # Verify we got data
    assert isinstance(mds, MetDataset)
    assert mds.data.attrs["provider"] == "Google"

    # Verify request was made
    mock_requests.assert_called_once()

    # Check that download_dataset returns a list
    mock_requests.reset_mock()
    datasets = gf.download_dataset(gf.timesteps)
    assert isinstance(datasets, list)
    assert len(datasets) == 1
    assert isinstance(datasets[0], xr.Dataset)


@pytest.mark.skipif("GOOGLE_API_KEY" not in os.environ, reason="GOOGLE_API_KEY not set")
def test_integration_cache_consistency(tmp_path):
    """Integration test to verify consistency between cached and non-cached results."""
    # Forecast for 24h from now, rounded to hour
    time = pd.Timestamp.utcnow().ceil("h") + pd.Timedelta("24h")

    # 1. No Cache
    gf_no_cache = GoogleForecast(
        time=time,
        variables=[Severity],
        cachestore=None,
    )
    mds_no_cache = gf_no_cache.open_metdataset()

    # 2. With Cache
    local_cache = cache.DiskCacheStore(cache_dir=tmp_path)
    gf_with_cache = GoogleForecast(
        time=time,
        variables=[Severity],
        cachestore=local_cache,
    )
    mds_with_cache = gf_with_cache.open_metdataset()

    # Verify structure and content
    _check_mds_structure(mds_no_cache, time)
    _check_mds_structure(mds_with_cache, time)

    # Allow for floating point differences if serialization/deserialization introduces them
    # But typically they should be identical if downloaded from same source
    xr.testing.assert_allclose(mds_no_cache.data, mds_with_cache.data)

    # Verify cache file was created
    cache_path = gf_with_cache.create_cachepath(time)
    assert local_cache.exists(cache_path)
