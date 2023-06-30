"""Test tau_cirrus module."""

from __future__ import annotations

from typing import Callable

import numpy as np
import pytest
import xarray as xr

from pycontrails import MetDataset
from pycontrails.models.tau_cirrus import tau_cirrus
from tests._deprecated import tau_cirrus_original


@pytest.mark.parametrize("formula", [tau_cirrus, tau_cirrus_original])
def test_shape_and_coords(met_cocip1: MetDataset, formula: Callable) -> None:
    """Ensure shape and coords are consistent."""
    met_cocip1.data["air_pressure"] = met_cocip1.data["air_pressure"].astype(np.float32)

    da = formula(met_cocip1)
    assert isinstance(da, xr.DataArray)

    # As long as the variables and air_pressure are float32, the output should also be float32
    assert da.dtype == np.float32

    assert da.sizes == met_cocip1.data.sizes
    assert list(da.dims) == met_cocip1.dim_order

    xr.testing.assert_equal(da.level, met_cocip1.data["level"])
    xr.testing.assert_equal(da.time, met_cocip1.data["time"])
    xr.testing.assert_equal(da.longitude, met_cocip1.data["longitude"])
    xr.testing.assert_equal(da.latitude, met_cocip1.data["latitude"])

    assert da.name == "tau_cirrus"


@pytest.mark.parametrize("formula", [tau_cirrus, tau_cirrus_original])
def test_nonnegative_values(met_cocip1: MetDataset, formula: Callable) -> None:
    """Ensure tau cirrus values are nonnegative."""
    da = formula(met_cocip1)
    assert (da >= 0).all()


def test_implementations_close(met_cocip1: MetDataset) -> None:
    """Ensure current implementation is close to original implementation."""
    met_cocip1.data["air_pressure"] = met_cocip1.data["air_pressure"].astype(np.float32)

    # Both are nonzero in same places
    da1 = tau_cirrus(met_cocip1)
    da2 = tau_cirrus_original(met_cocip1)
    xr.testing.assert_equal(da1 > 0, da2 > 0)

    # New implementation is larger
    assert (da1 >= da2).all()

    # But not twice as large
    assert (da1 <= 2 * da2).all()

    # Pin the mean values of each
    assert da1.mean() == 0.01444357167929411
    assert da2.mean().item() == 0.008129739202558994


def test_geopotential_approximation(met_cocip1: MetDataset):
    """Compare tau cirrus outputs when geopotential height is replaced by geometric height."""
    tc1 = tau_cirrus(met_cocip1)
    del met_cocip1.data["geopotential"]

    # Replace geopotential height with geometric height
    # This only has a "level" dimension, so this test also ensures that
    # the tau_cirrus function can handle this case.
    met_cocip1["geopotential_height"] = met_cocip1["altitude"]
    assert met_cocip1.data["geopotential_height"].dims == ("level",)
    tc2 = tau_cirrus(met_cocip1)

    assert tc1.dims == tc2.dims
    assert tc1.sizes == tc2.sizes

    # The important assertion is that the values are close enough from the perspective
    # of CoCiP.
    xr.testing.assert_allclose(tc1, tc2, rtol=0.04)
