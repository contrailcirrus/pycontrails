"""Ensure model output dtypes are consistent with model inputs."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from pycontrails import Flight, GeoVectorDataset, MetDataset
from pycontrails.core.met import MetDataArray
from pycontrails.models.aircraft_performance import AircraftPerformance
from pycontrails.models.cocip import Cocip
from pycontrails.models.cocipgrid import CocipGrid
from pycontrails.models.humidity_scaling import ConstantHumidityScaling
from pycontrails.models.issr import ISSR
from pycontrails.models.sac import SAC
from pycontrails.physics import units
from tests import BADA3_PATH, BADA4_PATH, BADA_AVAILABLE


def ds_to_float32(ds: xr.Dataset):
    """Convert spatial dimensions to float32."""
    ds["longitude"] = ds["longitude"].astype("float32")
    ds["latitude"] = ds["latitude"].astype("float32")
    ds["level"] = ds["level"].astype("float32")
    return ds.reset_coords(drop=True)


def confirm_float32(met: MetDataset):
    """Confirm all data arrays are float32."""
    assert met.data["longitude"].dtype == "float32"
    assert met.data["latitude"].dtype == "float32"
    assert met.data["level"].dtype == "float32"
    if met.data["level"].size > 1:
        assert met.data["air_pressure"].dtype == "float32"
        assert met.data["altitude"].dtype == "float32"
    assert met.data["time"].dtype == "datetime64[ns]"
    for v in met:
        assert met[v].data.dtype == "float32"


@pytest.fixture
def met32(met_ecmwf_pl_path: str):
    """Create a MetDataset with float32 precision spatial coordinates."""
    ds1 = xr.open_dataset(met_ecmwf_pl_path)
    ds1 = ds_to_float32(ds1)

    # shift time and concatenate to create a new dataset
    ds2 = ds1.copy()
    ds2["time"] = ds2["time"] + np.timedelta64(1, "h")
    ds = xr.concat([ds1, ds2], dim="time")

    met = MetDataset(ds)
    confirm_float32(met)
    assert met.shape == (15, 8, 3, 4)
    return met


@pytest.mark.parametrize("method", ["linear", "nearest"])
def test_interpolation_float32_out(method: str, met32: MetDataset):
    """Ensure interpolation output is float32 if all inputs have float32 precision."""
    mda = met32["t"]

    t0 = met32.data.time.values[0] + np.timedelta64(30, "m")
    vector = GeoVectorDataset(
        longitude=np.array([1, 2, 3], dtype="float32"),
        latitude=np.array([1, 2, 3], dtype="float32"),
        level=np.array([251, 252, 253], dtype="float32"),
        time=[t0, t0, t0],
    )
    out = vector.intersect_met(mda, bounds_error=True, method=method)
    assert out.size == 3
    assert out.dtype == "float32"

    # Linear interpolation is somewhat special in that it issues explicit distances
    # between coordinates. Nearest interpolation does not.
    mda.data["longitude"] = mda.data["longitude"].astype("float64")
    out = vector.intersect_met(mda, bounds_error=True, method=method)
    if method == "linear":
        # If any met coordinate is float64 and method=linear, the output will be float64
        assert out.dtype == "float64"
    else:
        assert out.dtype == "float32"
    mda.data["longitude"] = mda.data["longitude"].astype("float32")

    vector.update(longitude=vector["longitude"].astype("float64"))
    out = vector.intersect_met(mda, bounds_error=True, method=method)
    if method == "linear":
        # If any vector coordinate is float64 and method=linear, output will be float64
        assert out.dtype == "float64"
    else:
        assert out.dtype == "float32"

    # In either case, if the met variable has float64 coordinates, the output will be float64
    mda.data = mda.data.astype("float64")
    out = vector.intersect_met(mda, bounds_error=True, method=method)
    assert out.dtype == "float64"


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


@pytest.fixture(scope="module")
def issr_sac_met32(met_issr: MetDataset):
    """Create a MetDataset with float32 precision for ISSR."""
    ds = met_issr.data.copy()
    ds = ds_to_float32(ds)

    # Check it
    met = MetDataset(ds)
    confirm_float32(met)
    return met


@pytest.mark.parametrize("model_class", [ISSR, SAC])
@pytest.mark.parametrize("method", ["linear", "nearest"])
def test_issr_sac_grid_output(issr_sac_met32: MetDataset, model_class: type, method: str):
    """Confirm ISSR and SAC gridded output is float32 when met input is float32."""
    model = model_class(
        issr_sac_met32,
        interpolation_method=method,
        humidity_scaling=ConstantHumidityScaling(),
    )
    out = model.eval()
    assert isinstance(out, MetDataArray)
    assert out.data.dtype == "float32"
    assert out.shape == issr_sac_met32.shape


@pytest.fixture(scope="module")
def random_vector(issr_sac_met32: MetDataset):
    """Create a random vector with coordinates mostly matching `issr_sac_met32`."""
    # Build random vector
    # Purposely going out of bounds from met
    rng = np.random.default_rng(22777)
    longitude = rng.uniform(-180, 180, 1000).astype("float32")
    latitude = rng.uniform(-90, 90, 1000).astype("float32")
    level = rng.uniform(200, 300, 1000).astype("float32")

    t0 = issr_sac_met32.data.time.values[0]
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
    issr_sac_met32: MetDataset,
    model_class: type,
    method: str,
    random_vector: GeoVectorDataset,
):
    """Confirm ISSR and SAC vector output is float32 when source and met input is float32."""
    # Run the model
    model = model_class(
        issr_sac_met32,
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
    in_bounds = random_vector.coords_intersect_met(issr_sac_met32)
    interp_non_nan = np.isfinite(out["air_temperature"])
    np.testing.assert_array_equal(in_bounds, interp_non_nan)


@pytest.fixture
def cocip_met_rad(met_cocip1: MetDataset, rad_cocip1: MetDataset) -> tuple[MetDataset, MetDataset]:
    """Convert cocip met and rad input to float32."""
    ds_met = met_cocip1.data
    ds_met = ds_to_float32(ds_met)

    ds_rad = rad_cocip1.data
    ds_rad = ds_to_float32(ds_rad)

    met = MetDataset(ds_met)
    rad = MetDataset(ds_rad)

    confirm_float32(met)
    confirm_float32(rad)

    return met, rad


@pytest.mark.skipif(not BADA4_PATH.is_dir(), reason="BADA4 not available")
@pytest.mark.parametrize("drop_aircraft_performance", [False, True])
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_cocip(
    cocip_met_rad,
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
        *cocip_met_rad,
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
            if k not in exclude:
                arr = vector[k]
                assert np.can_cast(arr.dtype, "float32"), k


@pytest.mark.skipif(not BADA_AVAILABLE, reason="BADA not available")
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.parametrize("bada_priority", [3, 4])
def test_contrail_grid(cocip_met_rad: tuple[MetDataset, MetDataset], bada_priority: int):
    """Confirm `CocipGrid` maintains float32 precision."""
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
    cg = CocipGrid(*cocip_met_rad, params=params)

    # Create float32 source
    source = cg.create_source(
        level=250,
        time=cg.met.data.time.values[0],
        longitude=cg.met.data.longitude.values,
        latitude=cg.met.data.latitude.values,
    )
    ds = source.data
    ds = ds_to_float32(ds)
    source = MetDataset(ds)
    confirm_float32(source)
    assert source.shape == (16, 8, 1, 1)

    out = cg.eval(source)
    assert out["ef_per_m"].data.dtype == "float32"
    assert out.attrs["bada_model"] == f"BADA{bada_priority}"
    assert out.attrs["max_age"] == "2 hours"
    assert out.attrs["dt_integration"] == "10 minutes"
    assert out.attrs["pycontrails_version"].startswith("0.")
    assert out.data["longitude"].dtype == "float32"
    assert out.data["latitude"].dtype == "float32"
    assert out.data["level"].dtype == "float32"

    # Not much persistence, but we can pin them
    # And importantly, they are the same with BADA3 or BADA4
    (nonzero,) = out["ef_per_m"].data.values.ravel().nonzero()
    np.testing.assert_array_equal(nonzero, [38, 46, 52, 53, 54, 61, 62, 70])
