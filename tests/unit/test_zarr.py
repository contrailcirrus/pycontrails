"""Test `MetDataset` opened from a Zarr store.

These tests require a ERA5_PL_ZARR_STORE environment variable and an internet
connection. They re slow to run.
"""

import os

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pycontrails import GeoVectorDataset, MetDataArray, MetDataset

ZARR_STORE = os.getenv("ERA5_PL_ZARR_STORE")

if ZARR_STORE is None:
    pytest.skip("Zarr store not available from environment variable", allow_module_level=True)


@pytest.fixture
def ds() -> xr.Dataset:
    """Open the Zarr store as an xarray Dataset."""
    return xr.open_zarr(ZARR_STORE)  # using default chunks="auto" for dask interop


def test_open_zarr_as_dataset(ds: xr.Dataset) -> None:
    """Confirm interaction with a zarr-based Dataset.

    This test downloads a single chunks from the Zarr store.
    """
    assert isinstance(ds, xr.Dataset)
    da = ds["eastward_wind"]
    assert isinstance(da, xr.DataArray)
    point = da.sel(longitude=55, latitude=33, level=350, time="2020-09-02T13")
    assert point.size == 1

    # This will download the right chunk from the cloud then send it through the
    # the zarr -> dask -> xarray pipeline
    assert point.values.item() == 13.631146430969238


def test_open_zarr_as_metdataset(ds: xr.Dataset) -> None:
    """Confirm a zarr-based Dataset can be opened as a MetDataset.

    No chunks (apart from xarray dimensions) are actually downloaded here.
    """
    mds = MetDataset(ds)
    assert isinstance(mds, MetDataset)
    assert isinstance(mds.shape, tuple)
    assert mds.is_wrapped
    assert mds.is_zarr

    mda = mds["air_temperature"]
    assert isinstance(mda, MetDataArray)

    assert mda.is_zarr
    assert mda.is_wrapped
    assert not mda.in_memory


def test_zarr_interpolation() -> None:
    """Confirm a zarr-based Dataset can interpolate a GeoVectorDataset.

    Two time chunks are downloaded here.
    """
    mds = MetDataset.from_zarr(ZARR_STORE)
    mda = mds["eastward_wind"]

    rng = np.random.default_rng(13579)
    n = 10000
    longitude = rng.uniform(-180, 180, n)
    latitude = rng.uniform(-90, 90, n)
    level = rng.uniform(200, 500, n)
    time = pd.date_range("2020-03-14T15", "2020-03-14T16", n)

    with pytest.raises(RuntimeError, match=r"loading at least \d+ GB of data into memory"):
        mda.interpolate(longitude, latitude, level, time)

    vector = GeoVectorDataset(longitude=longitude, latitude=latitude, level=level, time=time)
    mds = vector.downselect_met(mds)
    assert len(mds.variables["time"]) == 2
    assert np.all(vector.coords_intersect_met(mds))

    # Run interpolation on downselected data
    out = vector.intersect_met(mds["air_temperature"])
    assert isinstance(out, np.ndarray)
    assert out.size == n
    assert np.all(np.isfinite(out))
    assert np.all(out < 273)
    assert np.all(out > 195)
