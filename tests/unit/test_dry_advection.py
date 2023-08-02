"""Test the :mod:`dry_advection` module."""

import numpy as np
import pytest

from pycontrails import GeoVectorDataset, MetDataset
from pycontrails.models.dry_advection import DryAdvection


@pytest.fixture
def source(met_cocip1: MetDataset) -> GeoVectorDataset:
    """Return a GeoVectorDataset."""

    ds = met_cocip1.data.isel(time=[0])
    mds = MetDataset(ds.drop_vars(ds.data_vars))
    return mds.to_vector()


def test_dry_advection(met_cocip1: MetDataset, source: GeoVectorDataset) -> None:
    """Test the :class:`DryAdvection` model."""
    model = DryAdvection(
        met=met_cocip1,
        max_age=np.timedelta64(1, "h"),
        dt_integration=np.timedelta64(5, "m"),
    )
    out = model.eval(source)
    assert isinstance(out, GeoVectorDataset)
