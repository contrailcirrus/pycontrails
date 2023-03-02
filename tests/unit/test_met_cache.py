"""Test MetDataset and MetDataArrays are cached correctly when calling `__getitem__`."""

import pytest
import xarray as xr

from pycontrails import MetDataArray, MetDataset


def test_met_copy_data_arg(met_ecmwf_pl_path: str):
    """Test the `copy` argument in the MetDataset constructor."""
    ds = xr.open_dataset(met_ecmwf_pl_path)

    # The file on disk is not sorted correctly
    with pytest.raises(ValueError, match="not sorted"):
        MetDataset(ds, copy=False)

    # Sort it, then grab the data
    ds = MetDataset(ds).data

    # And test the `copy` parameter
    met = MetDataset(ds, copy=False)
    assert met.data is ds

    # Different objects if copied
    met = MetDataset(ds)
    assert met.data is not ds


def test_met_no_copy_same_view(met_ecmwf_pl_path: str):
    """Check that the `copy` flag provides view of same underlying data."""
    ds = xr.open_dataset(met_ecmwf_pl_path)

    # xarray __getitem__ counterintuitive
    assert ds["t"] is not ds["t"]
    assert ds["t"].values is not ds["t"].values

    # But they share the same view (so no memory leak)
    assert ds["t"].values.base is not None
    assert ds["t"].values.base is ds["t"].values.base

    # Old implementation is not memory efficient
    # Data is copied in the _preprocess_dims method
    mda1 = MetDataArray(ds["t"])
    mda2 = MetDataArray(ds["t"])
    assert mda1.data.values.base is not None
    assert mda2.data.values.base is not None
    assert mda1.data.values.base is not mda2.data.values.base

    # New implementation avoids this
    mda3 = MetDataArray(mda1.data, copy=False)
    mda4 = MetDataArray(mda1.data, copy=False)
    assert mda3.data is mda4.data
    assert mda3.data.values is mda4.data.values
    assert mda3.data.values.base is not None
    assert mda4.data.values.base is not None
    assert mda3.data.values.base is mda4.data.values.base
    assert mda1.data.values.base is mda3.data.values.base

    # And the culprit is _preprocess_dims
    mda3._preprocess_dims(False)
    mda4._preprocess_dims(False)
    assert mda3.data is not mda4.data
    assert mda3.data.values is not mda4.data.values
