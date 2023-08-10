"""Test the :mod:`dry_advection` module."""

from __future__ import annotations

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


@pytest.mark.parametrize("azimuth", [0.0, 90.0, 180.0, 270.0, None])
def test_dry_advection(
    met_cocip1: MetDataset, source: GeoVectorDataset, azimuth: float | None
) -> None:
    """Test the :class:`DryAdvection` model."""
    params = {
        "max_age": np.timedelta64(1, "h"),
        "dt_integration": np.timedelta64(5, "m"),
        "azimuth": azimuth,
    }
    if azimuth is None:
        params["width"] = None
        params["depth"] = None

    model = DryAdvection(met_cocip1, params)
    out = model.eval(source)
    assert isinstance(out, GeoVectorDataset)
    assert len(out) == 6144

    if azimuth is None:
        assert len(out.data) == 9
    else:
        assert len(out.data) == 25

    # Pin some values to ensure that the model is working as expected
    abs = 0.1
    if azimuth == 0.0:
        assert np.nanmean(out["width"]) == pytest.approx(470.9, abs=abs)
    elif azimuth == 90.0:
        assert np.nanmean(out["width"]) == pytest.approx(464.8, abs=abs)
    elif azimuth == 180.0:
        assert np.nanmean(out["width"]) == pytest.approx(470.9, abs=abs)
    elif azimuth == 270.0:
        assert np.nanmean(out["width"]) == pytest.approx(464.8, abs=abs)
    else:
        assert "width" not in out
