"""Unit tests for Google Forecast datalib."""

from __future__ import annotations

import datetime
from unittest import mock

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pycontrails import MetDataset, MetVariable
from pycontrails.core import cache
from pycontrails.datalib.google_forecast import (
    EffectiveEnergyForcing,
    GoogleForecast,
    Severity,
)
from pycontrails.physics import units


@pytest.fixture
def mock_requests():
    with mock.patch("requests.get") as m:
        yield m


@pytest.fixture
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
    assert gf.request_headers == {"x-goog-api-key": "test-key"}

    # Test with default credentials (mocked)
    with mock.patch("google.auth.default", return_value=("default-creds", "project")):
        gf = GoogleForecast(time="2022-01-01 12:00:00", cachestore=local_cache)
        assert gf._credentials is None
        assert gf.request_headers == {"x-goog-api-key": "default-creds"}


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
    args, kwargs = mock_requests.call_args
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


def test_google_forecast_partial_cache(mock_requests, local_cache):
    """Test that only missing variables are downloaded and merged."""
    time = pd.Timestamp("2022-01-01 12:00:00")
    
    # Setup: Cache exists with only 'contrails'
    ds_existing = xr.Dataset(
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
    # Create cache file manually
    gf = GoogleForecast(time=time, cachestore=local_cache)
    cache_path = gf.create_cachepath(time)
    ds_existing.to_netcdf(cache_path)
    
    # Request both 'contrails' (Severity) and 'expected_effective_energy_forcing' (EffectiveEnergyForcing)
    gf = GoogleForecast(
        time=time,
        variables=[Severity, EffectiveEnergyForcing],
        key="test-key",
        cachestore=local_cache,
    )
    
    # Mock response for ONLY the missing variable
    ds_new = xr.Dataset(
        {
            "expected_effective_energy_forcing": (
                ("time", "level", "latitude", "longitude"),
                np.full((1, 1, 2, 2), 10.0),
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
    mock_resp.content = ds_new.to_netcdf()
    mock_resp.status_code = 200
    mock_resp.headers = {}
    mock_requests.return_value = mock_resp

    # Execute
    gf.download_dataset(gf.timesteps)
    
    # Verify Request
    mock_requests.assert_called_once()
    args, kwargs = mock_requests.call_args
    # "contrails" should NOT be in data params
    requested_params = kwargs["params"]["data"]
    assert "contrails" not in requested_params
    assert "expected_effective_energy_forcing" in requested_params
    
    # Verify Cache Merged
    ds_final = xr.open_dataset(cache_path)
    assert "contrails" in ds_final
    assert "expected_effective_energy_forcing" in ds_final
    assert np.allclose(ds_final["contrails"], 0.5)
    assert np.allclose(ds_final["expected_effective_energy_forcing"], 10.0)
