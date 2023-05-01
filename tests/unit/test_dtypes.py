"""Ensure model output dtypes are consistent with model inputs."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from pycontrails import Flight, GeoVectorDataset, MetDataset
from pycontrails.core.met import MetDataArray
from pycontrails.models.aircraft_performance import AircraftPerformance
from pycontrails.models.cocip import Cocip
from pycontrails.models.humidity_scaling import ConstantHumidityScaling
from pycontrails.models.issr import ISSR
from pycontrails.models.sac import SAC
from pycontrails.physics import units
from tests import BADA3_PATH, BADA4_PATH, BADA_AVAILABLE


@pytest.fixture
def met(met_ecmwf_pl_path: str) -> MetDataset:
    """Create a MetDataset for testing."""
    ds1 = xr.open_dataset(met_ecmwf_pl_path)

    # shift time and concatenate to create a new dataset
    ds2 = ds1.copy()
    ds2["time"] = ds2["time"] + np.timedelta64(1, "h")
    ds = xr.concat([ds1, ds2], dim="time")
    for v in ds.data_vars:
        assert ds[v].dtype == "float32"

    met = MetDataset(ds)
    assert met.shape == (15, 8, 3, 4)
    return met


@pytest.fixture(params=["float32", "float64"])
def mda(met: MetDataset, request) -> MetDataArray:
    """Create a MetDataArray for testing."""
    mda = met["t"]
    mda.data = mda.data.astype(request.param)
    return mda


@pytest.mark.parametrize("vector_dtype", ["float32", "float64", "int32", "int64"])
@pytest.mark.parametrize("method", ["linear", "nearest"])
def test_interpolation_out_dtype(mda: MetDataArray, method: str, vector_dtype: str):
    """Ensure interpolation output dtype is consistent with grid dtype."""

    t0 = mda.data.time.values[0] + np.timedelta64(30, "m")
    vector = GeoVectorDataset(
        longitude=np.array([1, 2, 3], dtype=vector_dtype),
        latitude=np.array([1, 2, 3], dtype=vector_dtype),
        level=np.array([251, 252, 253], dtype=vector_dtype),
        time=[t0, t0, t0],
    )
    out = vector.intersect_met(mda, bounds_error=True, method=method)
    assert out.size == 3
    assert out.dtype == mda.data.dtype


def test_unit_conversion():
    """Confirm basic unit conversion maintains the array dtype."""
    level = np.array([1, 2, 3], dtype="float32")
    assert level.dtype == "float32"
    altitude = units.pl_to_ft(level)
    assert altitude.dtype == "float32"
    altitude_ft = units.m_to_ft(altitude)
    assert altitude_ft.dtype == "float32"
    level_ft = units.ft_to_pl(altitude_ft)
    assert level_ft.dtype == "float32"


@pytest.mark.parametrize("model_class", [ISSR, SAC])
@pytest.mark.parametrize("method", ["linear", "nearest"])
def test_issr_sac_grid_output(met_issr: MetDataset, model_class: type, method: str):
    """Confirm ISSR and SAC gridded output is float32 when met input is float32."""
    assert all(v == "float32" for v in met_issr.data.dtypes.values())

    model = model_class(
        met_issr,
        interpolation_method=method,
        humidity_scaling=ConstantHumidityScaling(),
    )
    out = model.eval()
    assert isinstance(out, MetDataArray)
    assert out.data.dtype == "float32"
    assert out.shape == met_issr.shape


@pytest.fixture(scope="module")
def random_vector(met_issr: MetDataset):
    """Create a random vector with coordinates mostly matching `met_issr`."""
    # Build random vector
    # Purposely going out of bounds from met
    rng = np.random.default_rng(22777)
    longitude = rng.uniform(-180, 180, 1000).astype("float32")
    latitude = rng.uniform(-90, 90, 1000).astype("float32")
    level = rng.uniform(200, 300, 1000).astype("float32")

    t0 = met_issr.data.time.values[0]
    time = t0 + rng.integers(0, 60, 1000) * np.timedelta64(1, "m")

    return GeoVectorDataset(
        longitude=longitude,
        latitude=latitude,
        level=level,
        time=time,
    )


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.parametrize("model_class", [ISSR, SAC])
@pytest.mark.parametrize("method", ["linear", "nearest"])
def test_issr_sac_vector_output(
    met_issr: MetDataset,
    model_class: type,
    method: str,
    random_vector: GeoVectorDataset,
):
    """Confirm ISSR and SAC vector output is float32 when source and met input is float32."""
    # Run the model
    model = model_class(
        met_issr,
        interpolation_method=method,
        humidity_scaling=ConstantHumidityScaling(),
    )
    out = model.eval(random_vector)
    assert isinstance(out, GeoVectorDataset)
    assert out.size == 1000
    assert out[model.name].dtype == "float32"
    assert out["air_temperature"].dtype == "float32"
    assert out["specific_humidity"].dtype == "float32"

    # Confirm interpolation behaves as expected
    in_bounds = random_vector.coords_intersect_met(met_issr)
    interp_non_nan = np.isfinite(out["air_temperature"])
    np.testing.assert_array_equal(in_bounds, interp_non_nan)


@pytest.mark.skipif(not BADA4_PATH.is_dir(), reason="BADA4 not available")
@pytest.mark.parametrize("drop_aircraft_performance", [False, True])
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_cocip(
    met_cocip1: MetDataset,
    rad_cocip1: MetDataset,
    flight_cocip1: Flight,
    drop_aircraft_performance: bool,
    bada_model: AircraftPerformance,
):
    """Confirm CoCiP maintains float32 precision."""
    # Convert flight_cocip1 dtypes to float32
    # Grab all the keys up front to avoid mutating while iterating
    keys = [k for k in flight_cocip1 if k != "time"]
    for k in keys:
        flight_cocip1.update({k: flight_cocip1[k].astype("float32")})

    if drop_aircraft_performance:
        for k in set(keys).difference(["longitude", "latitude", "altitude"]):
            del flight_cocip1[k]
        flight_cocip1.attrs.update(aircraft_type="B737")

    # Run the model
    cocip = Cocip(
        met=met_cocip1,
        rad=rad_cocip1,
        process_emissions=drop_aircraft_performance,
        humidity_scaling=ConstantHumidityScaling(),
        aircraft_performance=bada_model,
    )
    out = cocip.eval(flight_cocip1)

    # Check that the output is float32 or less with a few exceptions
    exclude = {"time", "flight_id", "waypoint", "formation_time", "age", "contrail_age"}
    vectors = [
        cocip._sac_flight,
        cocip._downwash_flight,
        cocip._downwash_contrail,
        out,
        *cocip.contrail_list,
    ]
    for vector in vectors:
        for k in vector:
            if k in exclude:
                continue
            arr = vector[k]
            assert np.can_cast(arr.dtype, "float32"), k


@pytest.mark.skipif(not BADA_AVAILABLE, reason="BADA not available")
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.parametrize("bada_priority", [3, 4])
def test_cocip_grid(met_cocip1: MetDataset, rad_cocip1: MetDataset, bada_priority: int):
    """Confirm `CocipGrid` maintains float32 precision."""
    CocipGrid = pytest.importorskip("pycontrails.models.cocipgrid").CocipGrid

    # Keep max_age small to avoid blowing out of bounds
    # And keep dt_integration small to make the evolution interesting
    params = {
        "max_age": np.timedelta64(2, "h"),
        "bada_priority": bada_priority,
        "dt_integration": np.timedelta64(10, "m"),
        "humidity_scaling": ConstantHumidityScaling(),
    }
    if bada_priority == 3:
        params["bada3_path"] = BADA3_PATH
    elif bada_priority == 4:
        params["bada4_path"] = BADA4_PATH
    cg = CocipGrid(met=met_cocip1, rad=rad_cocip1, params=params)

    # Create float32 source
    source = cg.create_source(
        level=250,
        time=cg.met.data.time.values[0],
        longitude=cg.met.data.longitude.values,
        latitude=cg.met.data.latitude.values,
    )
    assert source.shape == (16, 8, 1, 1)

    out = cg.eval(source)
    assert out["ef_per_m"].data.dtype == "float32"
    assert out.attrs["bada_model"] == f"BADA{bada_priority}"
    assert out.attrs["max_age"] == "2 hours"
    assert out.attrs["dt_integration"] == "10 minutes"
    assert out.attrs["pycontrails_version"].startswith("0.")

    # Not much persistence, but we can pin them
    # And importantly, they are the same with BADA3 or BADA4
    nonzero = np.flatnonzero(out["ef_per_m"].data.values.ravel())
    np.testing.assert_array_equal(nonzero, [38, 46, 52, 53, 54, 61, 62, 70])
