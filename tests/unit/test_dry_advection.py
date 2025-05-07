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
        assert len(out.data) == 10
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

    assert df1.shape == (208, 17)
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


@pytest.mark.parametrize(
    ("verbose_outputs", "include_source_in_output"),
    [(True, True), (True, False), (False, True), (False, False)],
)
def test_dry_advection_verbose_outputs(
    met_cocip1: MetDataset,
    source: GeoVectorDataset,
    verbose_outputs: bool,
    include_source_in_output: bool,
) -> None:
    """Test the :class:`DryAdvection` model."""
    params = {
        "max_age": np.timedelta64(1, "h"),
        "dt_integration": np.timedelta64(5, "m"),
        "verbose_outputs": verbose_outputs,
        "include_source_in_output": include_source_in_output,
    }

    model = DryAdvection(met_cocip1, params)
    out = model.eval(source)
    assert isinstance(out, GeoVectorDataset)
    n_variables = len(out.data)
    n_rows = len(out)

    extra_keys = ("ds_dz", "dsn_dz", "dT_dz")
    if verbose_outputs:
        assert all(key in out for key in extra_keys)
        assert n_variables == 19
    else:
        assert not any(key in out for key in extra_keys)
        assert n_variables == 16

    if include_source_in_output:
        assert n_rows == 3532 + len(source)
    else:
        assert n_rows == 3532


@pytest.mark.parametrize("include_source_in_output", [True, False])
def test_dry_advection_flight_id_in_output(
    met_cocip1: MetDataset, source: GeoVectorDataset, include_source_in_output: bool
) -> None:
    """Test the inclusion of a ``flight_id`` column in :class:`DryAdvection` output."""
    params = {
        "max_age": np.timedelta64(1, "h"),
        "dt_integration": np.timedelta64(5, "m"),
        "include_source_in_output": include_source_in_output,
    }

    model = DryAdvection(met_cocip1, params)
    out = model.eval(source)

    assert "flight_id" not in source
    assert "flight_id" not in out

    source1 = source.filter(source["latitude"] < 55.0, copy=True)
    source2 = source.filter(source["latitude"] >= 55, copy=True)
    source1["flight_id"] = np.full(len(source1), "flight1")
    source2["flight_id"] = np.full(len(source2), "flight2")
    source = source1 + source2
    model = DryAdvection(met_cocip1, params)
    out = model.eval(source)

    assert "flight_id" in source
    assert "flight_id" in out
    if include_source_in_output:
        assert (out["flight_id"] == "flight1").sum() == 2059 + len(source1)
        assert (out["flight_id"] == "flight2").sum() == 1473 + len(source2)
    else:
        assert (out["flight_id"] == "flight1").sum() == 2059
        assert (out["flight_id"] == "flight2").sum() == 1473
