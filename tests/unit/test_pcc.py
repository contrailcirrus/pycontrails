"""Test PCC algorithm."""

from __future__ import annotations

import pytest
import xarray as xr

from pycontrails import MetDataArray, MetDataset
from pycontrails.models.humidity_scaling import ExponentialBoostLatitudeCorrectionHumidityScaling
from pycontrails.models.pcc import PCC


def test_pcc_default(met_pcc_pl: MetDataset, met_pcc_sl: MetDataset) -> None:
    """Test Default pcc algorithm."""

    with pytest.warns(UserWarning, match="originated from ECMWF"):
        model = PCC(met=met_pcc_pl, surface=met_pcc_sl)

    pcc = model.eval()

    assert isinstance(pcc, MetDataArray)
    assert isinstance(pcc.data, xr.DataArray)
    assert isinstance(model.met, MetDataset)
    assert pcc.name == "pcc"


@pytest.fixture(params=["Smith1990", "Sundqvist1989", "Slingo1980"])
def fancy_pcc(
    met_pcc_pl: MetDataset,
    met_pcc_sl: MetDataset,
    request: pytest.FixtureRequest,
) -> PCC:
    """Generate PCC models for each cloud model type."""
    params = {
        "cloud_model": request.param,
        "humidity_scaling": ExponentialBoostLatitudeCorrectionHumidityScaling(),
    }
    return PCC(met_pcc_pl, surface=met_pcc_sl, params=params)


def test_pcc_cloud_model(fancy_pcc: PCC) -> None:
    """Test PCC algorithm with alternate cloud models."""
    out = fancy_pcc.eval()
    assert isinstance(out, MetDataArray)
    assert isinstance(out.data, xr.DataArray)
    assert isinstance(fancy_pcc.met, MetDataset)
    assert out.name == "pcc"
