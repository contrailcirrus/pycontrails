"""Ensure gridded output can be written to disk."""

import pathlib
import tempfile

import numpy as np
import pytest
import xarray as xr

try:
    from pycontrails.models.cocipgrid import CocipGrid
except ImportError:
    pytest.skip("CocipGrid not available", allow_module_level=True)

from pycontrails import MetDataset
from pycontrails.models.humidity_scaling import ConstantHumidityScaling
from pycontrails.models.issr import ISSR
from pycontrails.models.pcr import PCR
from pycontrails.models.sac import SAC
from tests import BADA3_PATH


@pytest.mark.skipif(not BADA3_PATH.is_dir(), reason="BADA3 not available")
@pytest.mark.parametrize("model", [SAC, ISSR, CocipGrid, PCR])
def test_model_output_to_netcdf(met_cocip1: MetDataset, rad_cocip1: MetDataset, model: type):
    """Ensure gridded output can be written to netCDF.

    This was a problem in 0.28.0 with humidity scaling metadata.
    """
    kwargs = {"met": met_cocip1, "humidity_scaling": ConstantHumidityScaling()}
    if issubclass(model, CocipGrid):
        kwargs["rad"] = rad_cocip1
        kwargs["max_age"] = np.timedelta64(4, "h")
        kwargs["interpolation_bounds_error"] = True
        kwargs["bada3_path"] = BADA3_PATH

    instance = model(**kwargs)

    coords = met_cocip1.coords

    # Do some slicing to keep everything in bounds
    coords["time"] = coords["time"][:1]
    coords["longitude"] = coords["longitude"][:-1]
    coords["latitude"] = coords["latitude"][1:-4]
    coords["level"] = [220, 230]

    source = MetDataset.from_coords(**coords)
    out = instance.eval(source)

    with tempfile.NamedTemporaryFile() as tmp_obj:
        da = out.data
        if isinstance(instance, CocipGrid):
            da = da["ef_per_m"]
            da.attrs.update(out.attrs)

        assert np.all(np.isfinite(da))

        assert "pycontrails_version" in da.attrs
        assert "humidity_scaling_name" in da.attrs

        assert isinstance(da, xr.DataArray)
        assert isinstance(da, xr.DataArray)
        assert pathlib.Path(tmp_obj.name).stat().st_size == 0

        da.to_netcdf(tmp_obj.name)
        assert pathlib.Path(tmp_obj.name).stat().st_size > 10000

        assert da.attrs == xr.open_dataarray(tmp_obj.name).attrs
