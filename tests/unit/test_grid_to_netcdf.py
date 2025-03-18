"""Ensure gridded output can be written to disk."""

import pathlib

import numpy as np
import pytest
import xarray as xr

from pycontrails import MetDataset
from pycontrails.models.cocipgrid import CocipGrid
from pycontrails.models.humidity_scaling import ConstantHumidityScaling
from pycontrails.models.issr import ISSR
from pycontrails.models.pcr import PCR
from pycontrails.models.ps_model import PSGrid
from pycontrails.models.sac import SAC
from pycontrails.utils import temp


@pytest.mark.parametrize("model_klass", [SAC, ISSR, CocipGrid, PCR])
def test_model_output_to_netcdf(
    met_cocip1: MetDataset,
    rad_cocip1: MetDataset,
    model_klass: type,
) -> None:
    """Ensure gridded output can be written to netCDF.

    This was a problem in 0.28.0 with humidity scaling metadata.
    """
    kwargs = {"met": met_cocip1, "humidity_scaling": ConstantHumidityScaling()}
    if model_klass is CocipGrid:
        kwargs["rad"] = rad_cocip1
        kwargs["max_age"] = np.timedelta64(4, "h")
        kwargs["interpolation_bounds_error"] = True
        kwargs["aircraft_performance"] = PSGrid()

    model = model_klass(**kwargs)

    coords = met_cocip1.coords

    # Do some slicing to keep everything in bounds
    coords["time"] = coords["time"][:1]
    coords["longitude"] = coords["longitude"][:-1]
    coords["latitude"] = coords["latitude"][1:-4]
    coords["level"] = [220, 230]

    source = MetDataset.from_coords(**coords)
    out = model.eval(source)

    with temp.temp_file() as temp_file:
        path = pathlib.Path(temp_file)

        ds = out.data
        assert isinstance(ds, xr.Dataset)

        for da in ds.data_vars.values():
            assert np.all(np.isfinite(da))

        assert "pycontrails_version" in ds.attrs
        assert ds.attrs["met_source_provider"] == "ECMWF"
        assert ds.attrs["met_source_dataset"] == "ERA5"
        assert ds.attrs["met_source_product"] == "reanalysis"
        assert ds.attrs["humidity_scaling_name"] == "constant_scale"
        assert ds.attrs["humidity_scaling_formula"] == "rhi -> rhi / rhi_adj"
        assert ds.attrs["humidity_scaling_rhi_adj"] == 0.97

        assert not path.is_file()
        ds.to_netcdf(temp_file)
        assert path.is_file()

        assert path.stat().st_size > 10000

        # Specifying the decode_timedelta silences an xarray warning
        assert ds.attrs == xr.open_dataset(temp_file, decode_timedelta=True).attrs

    assert not path.is_file()
