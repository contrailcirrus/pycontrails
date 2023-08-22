"""Test the :mod:`dry_advection` module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pycontrails import Flight, GeoVectorDataset, MetDataset
from pycontrails.models.cocip import Cocip
from pycontrails.models.dry_advection import DryAdvection
from pycontrails.models.humidity_scaling import ConstantHumidityScaling


@pytest.fixture()
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
    assert len(out) == 3532

    assert out["age"].max() == np.timedelta64(1, "h")

    if azimuth is None:
        assert len(out.data) == 11
    else:
        assert len(out.data) == 16

    # Pin some values to ensure that the model is working as expected
    abs = 0.1
    if azimuth in (0.0, 180.0):
        assert np.nanmean(out["width"]) == pytest.approx(1018.5, abs=abs)
    elif azimuth in (90.0, 270.0):
        assert np.nanmean(out["width"]) == pytest.approx(763.9, abs=abs)
    else:
        assert "width" not in out


@pytest.mark.filterwarnings("ignore")
def test_compare_dry_advection_to_cocip(
    flight_cocip1: Flight,
    met_cocip1: MetDataset,
    rad_cocip1: MetDataset,
) -> None:
    """Compare the dry advection model to Cocip predictions."""

    params = {"max_age": np.timedelta64(1, "h"), "dt_integration": np.timedelta64(5, "m")}

    model = DryAdvection(met_cocip1, params)
    out = model.eval(flight_cocip1)
    df1 = out.dataframe
    assert df1["longitude"].notna().all()
    assert df1["latitude"].notna().all()
    assert df1["level"].notna().all()
    assert df1["time"].notna().all()

    assert df1.shape == (208, 16)
    df1_sl = df1.query("time == '2019-01-01T01:25'")

    model = Cocip(
        met_cocip1,
        rad_cocip1,
        params,
        filter_sac=False,
        filter_initially_persistent=False,
        humidity_scaling=ConstantHumidityScaling(rhi_adj=0.6),
    )
    model.eval(flight_cocip1)
    df2 = model.contrail
    assert isinstance(df2, pd.DataFrame)
    assert df2.shape == (196, 58)
    df2_sl = df2.query("time == '2019-01-01T01:25'")

    # Pin some mean values to demonstrate the difference in vertical advection
    assert df1_sl["level"].mean() == pytest.approx(219.56, abs=0.01)
    assert df2_sl["level"].mean() == pytest.approx(222.07, abs=0.1)
