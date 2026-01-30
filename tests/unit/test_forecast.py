"""Unit tests for Google Forecast datalib."""

from __future__ import annotations

from unittest import mock

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pycontrails import MetDataset, VectorDataset
from pycontrails.datalib.forecast import ForecastApi, GoogleForecastParams


@pytest.fixture()
def mock_requests():
    with mock.patch("requests.get") as m:
        yield m


def test_google_forecast_init():
    """Test initialization."""
    params = GoogleForecastParams(credentials="test-token")
    gf = ForecastApi(params)
    assert gf._params.url == "https://contrails.googleapis.com/v2/grids"
    assert gf._params.credentials == "test-token"


def test_google_forecast_eval_interpolation(mock_requests):
    """Test eval with interpolation."""

    # 1. Setup Data
    # Source at 12:30 UTC (naive representation)
    time_source = pd.Timestamp("2022-01-01 12:30:00")
    source = VectorDataset(dict(time=[time_source]))

    # 2. Setup Mock Responses
    # We expect calls for 12:00 and 13:00 UTC
    time_12 = pd.Timestamp("2022-01-01 12:00:00").tz_localize("UTC")
    time_13 = pd.Timestamp("2022-01-01 13:00:00").tz_localize("UTC")

    # Helper to create NetCDF content
    def create_nc_content(time_val, value):
        ds = xr.Dataset(
            {
                "contrails": (
                    ("time", "level", "latitude", "longitude"),
                    np.full((1, 1, 2, 2), value),
                )
            },
            coords={
                "time": [time_val.tz_localize(None)],
                "level": [200],  # hPa
                "latitude": [10, 20],
                "longitude": [10, 20],
            },
        )
        return ds.to_netcdf()

    # Define side_effect for requests.get
    def get_side_effect(url, params, **kwargs):
        # params["time"] is isoformat
        req_time = pd.Timestamp(params["time"])
        if req_time == time_12:
            content = create_nc_content(time_12, 0.0)
        elif req_time == time_13:
            content = create_nc_content(time_13, 10.0)
        else:
            raise ValueError(f"Unexpected time requested: {req_time}")

        mock_resp = mock.Mock()
        mock_resp.content = content
        mock_resp.status_code = 200
        mock_resp.headers = {}
        return mock_resp

    mock_requests.side_effect = get_side_effect

    # 3. Run Code
    params = GoogleForecastParams(credentials="test-token")
    gf = ForecastApi(params)

    # Using 'contrails' (Severity) as variable for this test
    # The default checks for Severity
    out = gf.eval(source)

    # 4. Assertions
    assert isinstance(out, MetDataset)

    # Check that we requested 12:00 and 13:00
    assert mock_requests.call_count == 2

    # Check interpolation
    # At 12:30, value should be avg of 0.0 and 10.0 => 5.0
    assert "contrails" in out.data
    vals = out.data["contrails"].values
    assert np.allclose(vals, 5.0)

    # Check that output only includes the requested source time (12:30)
    assert out.data["time"].size == 1
    # Compare timestamps
    assert pd.Timestamp(out.data["time"].values[0]) == time_source


def test_google_forecast_eval_flight_level_conversion(mock_requests):
    """Test that flight_level is converted to level (pressure)."""

    # Use 12:30 to force 2 requests and avoid single-point interpolation issues if any
    time_source = pd.Timestamp("2022-01-01 12:30:00")
    source = VectorDataset(dict(time=[time_source]))

    time_12 = pd.Timestamp("2022-01-01 12:00:00").tz_localize("UTC")
    time_13 = pd.Timestamp("2022-01-01 13:00:00").tz_localize("UTC")

    # Helper to create NetCDF content with flight_level
    def create_nc_content_fl(time_val):
        ds = xr.Dataset(
            {
                "contrails": (
                    ("time", "flight_level", "latitude", "longitude"),
                    np.zeros((1, 1, 2, 2)),
                )
            },
            coords={
                "time": [time_val.tz_localize(None)],
                "flight_level": [300],  # FL300
                "latitude": [10, 20],
                "longitude": [10, 20],
            },
        )
        return ds.to_netcdf()

    def get_side_effect(url, params, **kwargs):
        req_time = pd.Timestamp(params["time"])
        if req_time in [time_12, time_13]:
            content = create_nc_content_fl(req_time)
        else:
            raise ValueError(f"Unexpected time requested: {req_time}")

        mock_resp = mock.Mock()
        mock_resp.content = content
        mock_resp.status_code = 200
        mock_resp.headers = {}
        return mock_resp

    mock_requests.side_effect = get_side_effect

    params = GoogleForecastParams(credentials="test-token")
    gf = ForecastApi(params)
    out = gf.eval(source)

    # Check that 'flight_level' is converted to 'level'
    assert "level" in out.data.coords
    assert "flight_level" not in out.data.dims

    # Check conversion: FL300 is approx 300 hPa
    level_val = out.data["level"].values[0]
    assert 200 < level_val < 400
