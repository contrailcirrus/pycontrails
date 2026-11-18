"""Test when MetDataset and MetDataArray share memory with their input."""

import numpy as np
import pytest
import xarray as xr

from pycontrails import MetDataArray, MetDataset


@pytest.mark.parametrize("copy", [True, False])
def test_copy_deprecated(met_ecmwf_pl_path: str, copy: bool) -> None:
    """Check that passing the `copy` parameter warns."""
    ds = xr.open_dataset(met_ecmwf_pl_path)

    with pytest.warns(DeprecationWarning, match="'copy' parameter has no effect"):
        MetDataset(ds, copy=copy)

    with pytest.warns(DeprecationWarning, match="'copy' parameter has no effect"):
        MetDataArray(ds["t"], copy=copy)


def test_copy_only_if_unsorted(met_ecmwf_pl_path: str) -> None:
    """Check that the constructors copy the data only if it needs sorting."""
    ds = xr.load_dataset(met_ecmwf_pl_path)

    # The file on disk is not sorted, so the constructors sort a copy
    met = MetDataset(ds)
    assert not np.shares_memory(met.data["t"].values, ds["t"].values)

    mda = MetDataArray(ds["t"])
    assert not np.shares_memory(mda.data.values, ds["t"].values)

    # Sorted input is wrapped in a new object that shares memory with it
    sorted_ds = met.data
    met2 = MetDataset(sorted_ds)
    assert met2.data is not sorted_ds
    assert np.shares_memory(met2.data["t"].values, sorted_ds["t"].values)

    mda2 = MetDataArray(sorted_ds["t"])
    assert mda2.data is not sorted_ds["t"]
    assert np.shares_memory(mda2.data.values, sorted_ds["t"].values)
