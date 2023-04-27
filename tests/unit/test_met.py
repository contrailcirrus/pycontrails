"""Test `MetDatset` and `MetDataArray` data structures."""

from __future__ import annotations

import json
import pathlib
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pycontrails import DiskCacheStore, MetDataArray, MetDataset, MetVariable
from pycontrails.core import cache as cache_module
from pycontrails.core.met import originates_from_ecmwf, shift_longitude
from pycontrails.datalib.ecmwf import ERA5
from tests import OPEN3D_AVAILABLE

DISK_CACHE_DIR = cache_module._get_user_cache_dir()


def test_sl_path_to_met(met_ecmwf_sl_path: str) -> None:
    """Confirm `MetDataset` constructor converts `Dataset` with correct conventions."""
    ds = xr.open_dataset(met_ecmwf_sl_path)
    assert set(ds.variables).issubset({"longitude", "latitude", "level", "time", "sp"})
    assert list(ds.data_vars) == ["sp"]

    mds = MetDataset(ds)
    assert mds.dim_order == ["longitude", "latitude", "level", "time"]

    # longitude and latitude correctly translated
    assert np.all(mds.data["longitude"].values >= -180)
    assert np.all(mds.data["longitude"].values < 180)
    assert np.all(mds.data["latitude"].values <= 90)
    assert np.all(mds.data["latitude"].values >= -90)

    # consistent shapes between coordinates and underlying np.array of values
    shapes: tuple[int, ...] = tuple()
    for coord in mds.dim_order:
        # every coordinate is ascending
        assert np.all(np.diff(mds.data[coord]).astype(float) > 0)
        shapes = shapes + mds.data[coord].shape

    # operators commute
    assert isinstance(mds["sp"], MetDataArray)
    assert isinstance(mds["sp"].data, xr.DataArray)
    assert isinstance(mds.data["sp"], xr.DataArray)
    xr.testing.assert_equal(mds["sp"].data, mds.data["sp"])

    assert shapes == mds["sp"].data.values.shape

    # wrap longitude
    mds = MetDataset(ds, wrap_longitude=True)

    # longitude and latitude correctly translated, but not wrapped
    assert np.all(mds.data["longitude"].values <= 200)
    assert np.all(mds.data["longitude"].values >= -185)

    assert np.all(mds.data["sp"][dict(longitude=0)] == mds.data["sp"][dict(longitude=-2)])
    assert np.all(mds.data["sp"][dict(longitude=1)] == mds.data["sp"][dict(longitude=-1)])

    assert not mds.is_zarr


def test_pl_path_to_met(met_ecmwf_pl_path: str) -> None:
    """Confirm `MetDataset` constructor converts `Dataset` with correct conventions."""
    ds = xr.open_dataset(met_ecmwf_pl_path)
    assert set(ds.variables).issubset(
        {"longitude", "latitude", "level", "time", "t", "r", "q", "ciwc"}
    )
    assert list(ds.data_vars) == ["t", "r", "q", "ciwc"]

    mds = MetDataset(ds)
    assert mds.dim_order == ["longitude", "latitude", "level", "time"]

    # longitude and latitude correctly translated
    assert np.all(mds.data["longitude"].values < 180)
    assert np.all(mds.data["longitude"].values >= -180)
    assert np.all(mds.data["latitude"].values <= 90)
    assert np.all(mds.data["latitude"].values >= -90)

    # consistent shapes between coordinates and underlying np.array of values
    shapes: tuple[int, ...] = tuple()
    for coord in mds.dim_order:
        # every coordinate is ascending
        assert np.all(np.diff(mds.data[coord]).astype(float) > 0)
        shapes = shapes + mds.data[coord].shape
    assert shapes == mds.data["t"].values.shape

    with pytest.raises(ValueError, match="Set 'copy=True' when using 'wrap_longitude=True'."):
        MetDataset(ds, wrap_longitude=True, copy=False)

    with pytest.raises(ValueError, match="Coordinate 'latitude' not sorted."):
        MetDataset(ds, copy=False)

    assert not mds.is_zarr


@pytest.mark.parametrize("variable", ["t", "q", "r", "sp"])
def test_metdataarray_constructor(
    met_ecmwf_sl_path: str, met_ecmwf_pl_path: str, variable: str
) -> None:
    """Check `MetDataArray` constructor applies conventions."""
    if variable == "sp":
        ds = xr.open_dataset(met_ecmwf_sl_path)
    else:
        ds = xr.open_dataset(met_ecmwf_pl_path)

    da = ds[variable]
    mda = MetDataArray(da, name="orcas")

    assert mda.name == "orcas"
    assert "MetDataArray" in repr(mda)
    assert isinstance(mda.hash, str)
    assert isinstance(mda.shape, tuple)
    assert not mda.is_wrapped
    assert mda.shape[0] == 15

    mda2 = MetDataArray(da, wrap_longitude=True)
    assert mda2.is_wrapped
    assert mda2.shape[0] == 17

    with pytest.raises(ValueError, match="Set 'copy=True' when using 'wrap_longitude=True'."):
        MetDataArray(mda.data, wrap_longitude=True, copy=False)

    # Cannot instantiate MetDataArray without time or level coord
    # This hacked DataArray only contains latitude and longitude
    insufficient_da = da["latitude"] + da["longitude"]
    with pytest.raises(ValueError, match="Meteorology data must contain dimension 'level'"):
        MetDataArray(insufficient_da)


@pytest.mark.parametrize("wrap_longitude", [True, False])
def test_pl_path_to_met_dataset(met_ecmwf_pl_path: str, wrap_longitude: bool):
    """Confirm that DataArray selection is consistent with attr MetDataArray.data."""
    ds = xr.open_dataset(met_ecmwf_pl_path)
    mds = MetDataset(ds, wrap_longitude=wrap_longitude)

    # TODO: I have no idea why this `sel` doesn't work all at once. Likely an xarray bug
    sel = dict(latitude=15.0, longitude=25.0, level=225)
    time_sel = dict(time="2019-05-31 05:00:00")
    val2 = mds["t"].data.loc[sel].loc[time_sel].values
    val1 = ds["t"].loc[sel].loc[time_sel].values
    assert val1 == val2

    sel = dict(latitude=15.0, longitude=25.0, level=225)
    time_sel = dict(time="2019-05-31 05:00:00")
    val2 = mds["q"].data.loc[sel].loc[time_sel].values
    val1 = ds["q"].loc[sel].loc[time_sel].values
    np.testing.assert_approx_equal(val1, val2)

    assert "MetDataset" in repr(mds)

    # hash
    assert isinstance(mds.hash, str)


@pytest.mark.parametrize(
    "lon,lat,match",
    [
        (np.linspace(-200, 200, 10), np.linspace(-100, 100, 10), "Latitude values must"),
        (np.linspace(-400, 200, 10), np.linspace(-80, 80, 10), "Shift to WGS84"),
    ],
)
def test_preprocess_dims_error(lon: np.ndarray, lat: np.ndarray, match: str):
    """Confirm ValueErrors if coordinates are not WGS84."""
    bad_da = xr.DataArray(
        data=1,
        dims=["longitude", "latitude", "level", "time"],
        coords={
            "longitude": lon,
            "latitude": lat,
            "time": [datetime(2000, 1, 1, h) for h in range(5)],
            "level": np.arange(100, 500, 100),
        },
    )

    with pytest.raises(ValueError, match=match):
        MetDataArray(bad_da)


def test_wrap_longitude(zero_like_da: xr.DataArray, met_ecmwf_pl_path: str) -> None:
    # wrap datasets
    ds = xr.open_dataset(met_ecmwf_pl_path)
    mds = MetDataset(ds, wrap_longitude=True)

    # longitude and latitude correctly translated, but not wrapped
    assert np.all(mds.data["longitude"].values <= 200)
    assert np.all(mds.data["longitude"].values >= -185)

    assert np.all(mds["t"].data[dict(longitude=0)] == mds["t"].data[dict(longitude=-2)])
    assert np.all(mds["t"].data[dict(longitude=1)] == mds["t"].data[dict(longitude=-1)])

    # wrap dataarrays
    mda = MetDataArray(zero_like_da, wrap_longitude=True)

    # longitude and latitude correctly translated, but not wrapped
    assert np.all(mda.data["longitude"].values <= 200)
    assert np.all(mda.data["longitude"].values >= -185)

    assert np.all(mda.data[dict(longitude=0)] == mda.data[dict(longitude=-2)])
    assert np.all(mda.data[dict(longitude=1)] == mda.data[dict(longitude=-1)])

    da = zero_like_da.copy()
    da2 = da[dict(longitude=-1)].assign_coords(longitude=180)
    da = xr.concat([da, da2], dim="longitude")

    # automatic wrapping inspection
    mda = MetDataArray(da)
    assert mda.is_wrapped


def test_shift_longitude(zero_like_da: xr.DataArray) -> None:
    da = zero_like_da.copy()

    lons = da["longitude"].values
    lons[lons < 0] = lons[lons < 0] + 360

    da = da.assign_coords(longitude=lons)
    assert np.all(da["longitude"].values >= 0)

    da2 = shift_longitude(da)

    # longitude correctly translated
    assert np.all(da2["longitude"].values <= 180)
    assert np.all(da2["longitude"].values >= -180)

    # longitude sorted
    assert np.all(np.diff(da2["longitude"].values) > 0)


@pytest.fixture
def zero_like_da() -> xr.DataArray:
    return xr.DataArray(
        data=0.0,
        dims=["longitude", "latitude", "level", "time"],
        coords={
            "longitude": np.arange(-180, 180, 1.0),
            "latitude": np.arange(-90, 90.1, 1.0),
            "time": [datetime(2000, 1, 1, h) for h in range(5)],
            "level": np.arange(100, 500, 100),
        },
    )


# a few arbitrary points with which to build sparse binary
@pytest.fixture(
    params=[
        {"lon": 13, "lat": 5, "level": 100},
        {"lon": 21, "lat": -8, "level": 200},
        {"lon": -34, "lat": 3, "level": 300},
        {"lon": -144, "lat": 42, "level": 400},
        {"lon": 27, "lat": 42, "level": 400},
        {"lon": 18, "lat": 0, "level": 400},
    ]
)
def sparse_binary(zero_like_da: xr.DataArray, request: Any) -> tuple[MetDataArray, dict]:
    """Return a MetDataArray with exactly one nonzero value at each time slice.

    Data has integer longitude and latitude values, 4 level coordinates, and 5 time coordinates.
    """
    # a single arbitrary positive point
    da = zero_like_da
    lon = request.param["lon"]
    lat = request.param["lat"]
    level = request.param["level"]
    point = (da["latitude"] == lat) & (da["longitude"] == lon) & (da["level"] == level)
    da = xr.where(point, 1.0, da)
    return MetDataArray(da), request.param


@pytest.fixture
def island_binary(zero_like_da: xr.DataArray) -> MetDataArray:
    # create 9 x 9 island

    da = zero_like_da
    island = (np.abs(da["latitude"]) < 5) & (np.abs(da["longitude"]) < 5)
    da = xr.where(island, 1.0, da)
    return MetDataArray(da)


@pytest.fixture
def antimeridian_binary(zero_like_da: xr.DataArray) -> MetDataArray:
    # create 9 x 9 island spanning antimeridian

    da = zero_like_da
    island = (np.abs(da["latitude"]) < 5) & ((da["longitude"] < -175) | (da["longitude"] > 175))
    da = xr.where(island, 1, da)
    return MetDataArray(da, wrap_longitude=True)


@pytest.fixture(params=["t", "r", "q"])
def median_binary(met_ecmwf_pl_path: str, request: Any) -> MetDataArray:
    # points with values above the median get 1, else 0
    ds = xr.open_dataset(met_ecmwf_pl_path)
    da = ds[request.param]
    da = (da > da.median()).astype(int)
    return MetDataArray(da)


def _generate_waypoints(
    x0: float, x1: float, y0: float, y1: float, z0: float, z1: float, t0: str, t1: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate waypoints spanning various diagonals of some 4d box."""
    n_points = 100

    x = np.linspace(x0, x1, n_points)
    y = np.linspace(y0, y1, n_points)
    z = np.linspace(z0, z1, n_points)
    t = pd.date_range(t0, t1, n_points)

    _x = np.concatenate([x, np.flip(x), x, x, x])
    _y = np.concatenate([y, y, np.flip(y), y, y])
    _z = np.concatenate([z, z, z, np.flip(z), z])
    _t = np.concatenate([t, t, t, t, np.flip(t)])

    return _x, _y, _z, _t


@pytest.fixture
def pl_path_to_doubled_time_mds(met_ecmwf_pl_path: str) -> MetDataset:
    return MetDataset(xr.open_dataset(met_ecmwf_pl_path))


@pytest.fixture
def sl_path_to_tripled_time_mds(met_ecmwf_sl_path: str) -> MetDataset:
    """Triple the number of time variables in the Dataset sitting at `met_ecmwf_sl_path`."""
    ds1 = xr.open_dataset(met_ecmwf_sl_path)

    ds2 = ds1[dict(time=0)]
    ds2["time"] = np.datetime64("2019-05-31T06", "ns")
    ds2 = ds2.expand_dims("time")

    ds3 = ds1[dict(time=0)]
    ds3["time"] = np.datetime64("2019-05-31T04", "ns")
    ds3 = ds3.expand_dims("time")

    ds = xr.concat([ds3, ds1[dict(time=[0])], ds2], "time")
    return MetDataset(ds)


def test_met_hashable(zero_like_da: xr.DataArray) -> None:
    mda = MetDataArray(zero_like_da)
    assert isinstance(mda.hash, str)


def test_met_values_property(zero_like_da: xr.DataArray, met_ecmwf_pl_path: str) -> None:
    """Test that met dataarray values are correctly lazy-loaded."""

    # this dataarray is already loaded
    mda = MetDataArray(zero_like_da)
    assert mda.in_memory
    assert mda.values is mda.data.values

    # this dataarray is not already loaded
    ds = xr.open_mfdataset(met_ecmwf_pl_path)
    mda = MetDataArray(ds["t"])
    assert not mda.in_memory
    assert mda.values is mda.data.values
    assert mda.in_memory


@pytest.mark.parametrize(
    "values, dimension",
    [
        ((-211, 53, 250, np.datetime64("2019-05-31T05:30")), 0),
        ((26, -112, 250, np.datetime64("2019-05-31T05:30")), 1),
        ((26, 53, 100, np.datetime64("2019-05-31T05:30")), 2),
        ((26, 53, 250, np.datetime64("2019-05-31T06:30")), 3),
    ],
)
def test_doubled_time_mds_interp_outside_grid(
    pl_path_to_doubled_time_mds: MetDataset,
    values: tuple[int, int, int, np.datetime64],
    dimension: int,
) -> None:
    mds = pl_path_to_doubled_time_mds
    variable = "q"
    mda = mds[variable]

    with pytest.raises(ValueError, match=f"out of bounds in dimension {dimension}"):
        mda.interpolate(*values, bounds_error=True)


# we get a scipy warning caused by division by zero
# this is an artifact of passing in waypoints agreeing with mda index
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_interpolate_wrap_antimeridian(sl_path_to_tripled_time_mds: MetDataset) -> None:
    ds = sl_path_to_tripled_time_mds
    mda = MetDataArray(ds["sp"].data, wrap_longitude=True)
    waypoints = pd.DataFrame(
        {
            "longitude": np.concatenate([np.linspace(-180, -160, 100), np.linspace(170, 180, 100)]),
            "latitude": 10,  # should be close to 15
            "level": -1,
            "time": pd.date_range("2019-05-31T04", "2019-05-31T05:59:59", 200),
        }
    )

    interpolated = mda.interpolate(
        waypoints["longitude"].values,
        waypoints["latitude"].values,
        waypoints["level"].values,
        waypoints["time"].values,
        method="nearest",
        bounds_error=True,
    )
    waypoints["interpolated"] = interpolated

    # mda['longitude'] = -160, -135, ..., 150, 175
    # waypoint longitudes < -172.5 are closest to mda longitude 175
    mda_value_175 = mda.data.sel(longitude=175, latitude=15).isel(time=0).values.item()
    mda_value_m160 = mda.data.sel(longitude=-160, latitude=15).isel(time=0).values.item()

    east_waypoints = waypoints["longitude"].between(-172.5, -160)
    mid_waypoints = waypoints["longitude"] < -172.5
    west_waypoints = waypoints["longitude"] > 170

    assert (waypoints.loc[east_waypoints, "interpolated"] == mda_value_m160).all()
    assert (waypoints.loc[mid_waypoints, "interpolated"] == mda_value_175).all()
    assert (waypoints.loc[west_waypoints, "interpolated"] == mda_value_175).all()


def test_interpolate_island_binary(island_binary: MetDataArray) -> None:
    # values in the island are 1
    t0, t1 = island_binary.data["time"].values[[0, -1]]
    waypoints = _generate_waypoints(-4, 4, -4, 4, 150, 350, t0, t1)
    interpolated = island_binary.interpolate(*waypoints)
    np.testing.assert_almost_equal(interpolated, np.ones((len(waypoints[0]),)))

    # and values outside are 0
    waypoints = _generate_waypoints(20, 30, 40, 50, 150, 350, t0, t1)
    interpolated = island_binary.interpolate(*waypoints)
    np.testing.assert_array_equal(interpolated, np.zeros((len(waypoints[0]),)))


def test_interpolate_antimeridian_binary(antimeridian_binary: MetDataArray) -> None:
    # values in the island are 1
    t0, t1 = antimeridian_binary.data["time"].values[[0, -1]]
    waypoints = _generate_waypoints(-180, -176, -4, 4, 150, 350, t0, t1)
    interpolated = antimeridian_binary.interpolate(*waypoints)
    np.testing.assert_almost_equal(interpolated, np.ones((len(waypoints[0]),)))

    # and values outside are 0
    waypoints = _generate_waypoints(-150, -140, -4, 4, 150, 350, t0, t1)
    interpolated = antimeridian_binary.interpolate(*waypoints)
    np.testing.assert_array_equal(interpolated, np.zeros((len(waypoints[0]),)))


def test_bad_interpolate(island_binary: MetDataArray) -> None:
    assert len(island_binary.interpolate(5, 5, 200, np.datetime64("2019-05-31T05:00:00"))) == 1

    with pytest.raises(ValueError):
        island_binary.interpolate(
            200, 5, 200, np.datetime64("2019-05-31T05:00:00"), bounds_error=True
        )
    with pytest.raises(ValueError):
        island_binary.interpolate(
            5, 100, 200, np.datetime64("2019-05-31T05:00:00"), bounds_error=True
        )
    with pytest.raises(ValueError):
        island_binary.interpolate(
            5, 5, 600, np.datetime64("2019-05-31T05:00:00"), bounds_error=True
        )
    with pytest.raises(ValueError):
        island_binary.interpolate(
            5, 5, 200, np.datetime64("2021-05-31T05:00:00"), bounds_error=True
        )


def test_binary(median_binary: MetDataArray) -> None:
    assert isinstance(median_binary, MetDataArray)
    assert isinstance(median_binary.proportion, float)
    assert 0.45 < median_binary.proportion < 0.55
    assert median_binary.binary


def test_edges_on_sparse_binary(sparse_binary: tuple[MetDataArray, dict]) -> None:
    mda, _ = sparse_binary
    edges = mda.find_edges()
    # edges is either all 0, or equals the original mda
    if edges.data.sum().item() > 0:
        assert (edges.data == mda.data).all().item()
    else:
        assert edges.data.sum().item() == 0


def test_edges_on_island_binary(island_binary: MetDataArray) -> None:
    edges = island_binary.find_edges()
    # 4 levels, 5 times, 4 sides to the island, 8 points to side
    assert edges.data.sum().item() == 4 * 5 * 4 * 8
    for i in range(-3, 4):
        for j in range(-3, 4):
            assert np.all(edges.data.sel(latitude=i, longitude=j) == 0)
    for i in range(-4, 5):
        for j in [-4, 4]:
            assert np.all(edges.data.sel(latitude=i, longitude=j) == 1)
            assert np.all(edges.data.sel(latitude=j, longitude=i) == 1)


def test_edges_on_antimeridian_binary(antimeridian_binary: MetDataArray) -> None:
    assert antimeridian_binary.shape == (361, 181, 4, 5)
    edges = antimeridian_binary.find_edges()
    assert edges.data.sum().item() == 4 * 5 * (4 * 9 + 6 * 2)  # additional boundary points added


def test_edges_on_median_binary(median_binary: MetDataArray) -> None:
    edges = median_binary.find_edges()

    # edges only exists on points with value 1
    assert ((median_binary.data - edges.data) >= 0).all().item()
    assert np.all(edges.data["longitude"] == median_binary.data["longitude"])  # consistent index
    assert np.all(edges.data["latitude"] == median_binary.data["latitude"])
    assert np.all(edges.data["level"] == median_binary.data["level"])
    assert np.all(edges.data["time"] == median_binary.data["time"])


def test_polygons_sparse_binary_specify_time_level(
    sparse_binary: tuple[MetDataArray, dict]
) -> None:
    mda, _ = sparse_binary
    # must specify level and time
    with pytest.raises(ValueError, match="coordinates is not 1"):
        mda.to_polygon_feature(iso_value=0.5)

    with pytest.raises(ValueError, match="coordinates is not 1"):
        mda.to_polygon_feature(level=mda.data["level"][0], iso_value=0.5)

    with pytest.raises(ValueError, match="coordinates is not 1"):
        mda.to_polygon_feature(time=mda.data["time"][0], iso_value=0.5)

    # if both are specified, we're okay
    geojson1 = mda.to_polygon_feature(
        level=mda.data["level"][0],
        time=mda.data["time"][0],
        iso_value=0.5,
    )

    # or if only a single level / time slice exists, we're okay
    # a single time coord
    sliced = MetDataArray(mda.data.isel(time=[0]))
    geojson2 = sliced.to_polygon_feature(level=mda.data["level"][0], iso_value=0.5)

    # a single level coord
    sliced = MetDataArray(mda.data.isel(level=[0]))
    geojson3 = sliced.to_polygon_feature(time=mda.data["time"][0], iso_value=0.5)

    # single time and level coords
    sliced = MetDataArray(mda.data.isel(time=[0], level=[0]))
    geojson4 = sliced.to_polygon_feature(iso_value=0.5)

    assert geojson1 == geojson2 == geojson3 == geojson4


@pytest.mark.parametrize("time", range(5))
def test_polygon_sparse_binary(
    sparse_binary: tuple[MetDataArray, dict], time: np.datetime64
) -> None:
    mda, nonzero_coord = sparse_binary
    mda = MetDataArray(mda.data.isel(time=[time]))
    level = nonzero_coord["level"]

    geojson = mda.to_polygon_feature(
        level=level,
        min_area=0.0,
        iso_value=0.5,
        epsilon=0.0,
        precision=6,
    )
    assert geojson["type"] == "Feature"
    assert geojson["geometry"]["type"] == "MultiPolygon"
    coords = geojson["geometry"]["coordinates"]
    assert isinstance(coords, list)

    # multipolygons are arrays of Polygons
    assert len(coords) == 1 and len(coords[0])
    coords = coords[0][0]

    # polygon closes in on itself
    assert coords[0] == coords[-1]

    # and has 4 distinct vertices
    assert len(coords) == 5
    coords = set(tuple(coord) for coord in coords)  # casting to set (an making each coord hashable)
    assert len(coords) == 4

    # and polygon successfully encloses nonzero point
    lon = nonzero_coord["lon"]
    lat = nonzero_coord["lat"]
    assert set(coord[0] for coord in coords) == {lon - 0.5, lon, lon + 0.5}
    assert set(coord[1] for coord in coords) == {lat - 0.5, lat, lat + 0.5}

    # polygon is JSON serializable
    assert json.dumps(geojson)


@pytest.fixture
def island_slice(island_binary: MetDataArray) -> MetDataArray:
    # every (level, time) slice of island_binary is identical -- just grab one
    return MetDataArray(island_binary.data.isel(level=[0], time=[0]))


def test_polygon_island_binary(island_slice: MetDataArray) -> None:
    geojson = island_slice.to_polygon_feature(iso_value=0.5, epsilon=0.01, precision=2)

    # contains a single polygon
    coords = geojson["geometry"]["coordinates"]
    assert isinstance(coords, list)
    assert len(coords) == 1 and len(coords[0]) == 1
    coords = coords[0][0]

    # polygon closes in on itself
    assert coords[0] == coords[-1]

    # polygon is square-shaped with the corners cut off
    # we have exactly two vertices at each island corner
    assert len(coords) == 9

    # all vertices have lon and lat value \pm 4 or \pm 4.5
    for coord in coords:
        for component in coord:
            assert component in [-4.5, -4, 4, 4.5]


@pytest.mark.parametrize("interiors", (True, False))
def test_nested_polygons(zero_like_da: xr.DataArray, interiors: bool) -> None:
    """This test shows that we *dont* currently support deeply nested polygons.

    To update once we support this feature.
    """
    da = zero_like_da.copy()

    island = (np.abs(da["latitude"]) < 50) & (np.abs(da["longitude"]) < 50)
    island2 = (np.abs(da["latitude"]) < 30) & (np.abs(da["longitude"]) < 30)
    island3 = (np.abs(da["latitude"]) < 20) & (np.abs(da["longitude"]) < 20)
    island4 = (np.abs(da["latitude"]) < 10) & (np.abs(da["longitude"]) < 10)
    island5 = (np.abs(da["latitude"]) < 5) & (np.abs(da["longitude"]) < 5)

    da = xr.where((island), 1, da)
    da = xr.where((island2), 0, da)
    da = xr.where((island3), 1, da)
    da = xr.where((island4), 0, da)
    da = xr.where((island5), 1, da)

    mda = MetDataArray(da.isel(level=[0], time=[0]))
    geojson = mda.to_polygon_feature(iso_value=0.5, interiors=interiors)

    # contains 1 Polygon with 1 nested polygon
    coords = geojson["geometry"]["coordinates"]
    assert isinstance(coords, list)

    if interiors:
        assert len(coords) == 3
    else:
        assert len(coords) == 1
        assert len(coords[0]) == 1


def test_polygons_with_holes():
    """Test ``to_polygon_feature`` method on handcrafted example.

    To get some intuition for this test, print or plot ``a`` in a notebook.
    """
    # Build up some data with interesting topology
    a = np.zeros((20, 10))

    # polygon 0: no holes
    a[1:11, 8] = 1

    # polygon 1: no holes
    a[2, 2] = 1
    a[2, 3] = 1
    a[3, 2] = 1

    # polygon 2: one hole
    a[8:12, 2:5] = 1
    a[9:11, 3] = 0

    # polygon 3: two holes
    a[13:17, 2:8] = 1
    a[14:16, 3:5] = 0
    a[15, 6] = 0
    a[13, 6:8] = 0

    # Pin the sum
    assert a.sum() == 40

    # Run through to_polygon_feature
    da = xr.DataArray(a, {"longitude": np.arange(20), "latitude": np.arange(10)})
    da = da.expand_dims(level=[-1], time=[0])
    mda = MetDataArray(da)
    polys = mda.to_polygon_feature(iso_value=0.5)["geometry"]["coordinates"]
    assert len(polys) == 4

    lens = [len(p) for p in polys]
    assert lens == [3, 2, 1, 1]

    # Pin some values from polygon 0
    outer, inner1, inner2 = polys[0]
    assert len(outer) == 12
    assert len(inner1) == 5  # octagon
    assert len(inner2) == 9  # diamond


@pytest.mark.parametrize("iso_value", [0, 0.001, 0.5, 0.9, 1])
def test_polygon_iso_value(island_slice: MetDataArray, iso_value: float) -> None:
    geojson = island_slice.to_polygon_feature(iso_value=iso_value, epsilon=0.1)
    coords = geojson["geometry"]["coordinates"]
    assert len(coords[0][0]) == 9


@pytest.mark.skipif(not OPEN3D_AVAILABLE, reason="Open3D not available")
@pytest.mark.parametrize("return_type", ["geojson", "mesh"])
def test_polyhedra_save(median_binary: MetDataArray, return_type: str) -> None:
    assert len(median_binary.data["time"]) == 1
    path: str | pathlib.Path
    if return_type == "geojson":
        path = "test_mesh.json"
    else:
        path = "test_mesh.ply"
    median_binary.to_polyhedra(return_type=return_type, path=path)

    path = pathlib.Path(path)
    assert path.is_file()
    path.unlink()


@pytest.mark.skipif(not OPEN3D_AVAILABLE, reason="Open3D not available")
def test_polyhedra_specify_time(sparse_binary: tuple[MetDataArray, dict]) -> None:
    mda, _ = sparse_binary
    with pytest.raises(ValueError, match="time input must be defined when the length"):
        mda.to_polyhedra()
    time = mda.data["time"][0]
    assert mda.to_polyhedra(time=time)


@pytest.mark.skipif(not OPEN3D_AVAILABLE, reason="Open3D not available")
def test_polyhedra_mesh(sparse_binary: tuple[MetDataArray, dict]) -> None:
    """Test the `to_polyhedra` method with return_type = "mesh"."""
    mda, nonzero = sparse_binary
    mda = MetDataArray(mda.data.isel(time=[0]))

    mesh = mda.to_polyhedra(return_type="mesh")
    assert len(mesh.triangles) in [4, 8]
    if len(mesh.triangles) == 4:
        assert len(mesh.vertices) == 5
        assert not mesh.is_watertight()
    else:
        assert len(mesh.vertices) == 6
        assert mesh.is_watertight()

    lon, lat, _ = mesh.get_center()
    assert lon == nonzero["lon"]
    assert lat == nonzero["lat"]

    mesh2 = mda.to_polyhedra(return_type="mesh", closed=True)
    assert len(mesh2.triangles) == 8
    assert len(mesh2.vertices) == 6
    assert mesh2.is_watertight()


@pytest.mark.skipif(not OPEN3D_AVAILABLE, reason="Open3D not available")
def test_polyhedra_geojson(sparse_binary: tuple[MetDataArray, dict]) -> None:
    """Test the `to_polyhedra` method."""

    mda, _ = sparse_binary
    mda = MetDataArray(mda.data.isel(time=[0]))

    # TODO: this test was written where closed=False, should add a test for closed
    geojson = mda.to_polyhedra(closed=False)
    assert isinstance(geojson, dict)
    assert geojson["type"] == "Feature"
    assert geojson["geometry"]["type"] == "MultiPolygon"
    coords = geojson["geometry"]["coordinates"]
    assert len(coords) in [4, 8]

    # comprised of triangles
    assert all(len(face) == 3 for poly in coords for face in poly)

    # and each point has three components
    assert all(len(point) == 3 for poly in coords for face in poly for point in face)

    # and altitude values are reasonable
    mda_altitude = set(mda.data["altitude"].values.astype(int))
    geojson_altitude = set([int(point[2]) for poly in coords for face in poly for point in face])
    assert geojson_altitude.issubset(mda_altitude)


def test_save_load(met_ecmwf_pl_path: str, met_era5_fake: MetDataset) -> None:
    """Test the `save` and `load` methods."""
    _cache = DiskCacheStore(cache_dir=f"{DISK_CACHE_DIR}/test", allow_clear=True)
    _cache.clear()
    mds = MetDataset(xr.open_dataset(met_ecmwf_pl_path), cachestore=_cache)

    # add new field
    mds.data["new"] = xr.ones_like(mds.data["t"])

    # save to cache
    assert not _cache.exists(f"{mds.hash}-0.nc")
    mds.save()
    assert _cache.exists(f"{mds.hash}-0.nc")

    mds2 = MetDataset.load(mds.hash, cachestore=_cache)
    assert "new" in mds2 and np.all(mds2.data["new"].values == 1)

    # met data array version
    mda = MetDataArray(mds.data["t"], cachestore=_cache)

    # save to cache
    assert not _cache.exists(f"{mda.hash}-0.nc")
    mda.save()
    assert _cache.exists(f"{mda.hash}-0.nc")

    mda2 = MetDataArray.load(mda.hash, cachestore=_cache)
    assert mda2.name == mda.name
    xr.testing.assert_equal(mda.data, mda2.data)

    # save multiple time slices separately - dataset
    mds = met_era5_fake.copy()
    mds.cachestore = _cache
    assert not _cache.exists(f"{mds.hash}-0.nc") and not _cache.exists(f"{mds.hash}-1.nc")
    mds.save()

    assert _cache.exists(f"{mds.hash}-0.nc")
    assert _cache.exists(f"{mds.hash}-1.nc")
    assert _cache.exists(f"{mds.hash}-2.nc")

    mds2 = MetDataset.load(mds.hash, cachestore=_cache)
    assert np.all(mds2.data["air_temperature"].values == 230)
    assert len(mds2.data["time"]) == 4

    # save multiple time slices separately - dataarray
    mda = MetDataArray(met_era5_fake.data["air_temperature"])
    mda.cachestore = _cache
    assert mda.hash != mds.hash
    assert not _cache.exists(f"{mda.hash}-0.nc") and not _cache.exists(f"{mda.hash}-1.nc")
    mda.save()

    assert _cache.exists(f"{mda.hash}-0.nc")
    assert _cache.exists(f"{mda.hash}-1.nc")
    assert _cache.exists(f"{mda.hash}-2.nc")

    mda2 = MetDataArray.load(mda.hash, cachestore=_cache)
    assert np.all(mda2.data.values == 230)
    assert len(mda2.data["time"]) == 4

    # cleanup
    _cache.clear()


def test_met_size(zero_like_da: xr.DataArray, met_ecmwf_pl_path: str) -> None:
    """Test `MetBase.size` property."""

    mda = MetDataArray(zero_like_da)
    assert mda.size == 360 * 181 * 4 * 5

    mds = MetDataset(xr.open_dataset(met_ecmwf_pl_path))
    assert mds.size == 8 * 15 * 3 * 2


def test_met_copy(met_ecmwf_pl_path: str) -> None:
    """Check that the method `copy` works as expected with a silly numerical experiment.."""
    ds = xr.open_dataset(met_ecmwf_pl_path)
    mds = MetDataset(ds)
    mds2 = mds.copy()

    assert mds.data is not mds2.data

    assert round(mds["t"].data[5][5][2][0].item()) == 235
    mds["t"].data *= 2
    assert round(mds["t"].data[5][5][2][0].item()) == 470
    assert round(mds2["t"].data[5][5][2][0].item()) == 235


def test_met_wrap_longitude_chunks(met_ecmwf_pl_path: str, override_cache: DiskCacheStore) -> None:
    """Check that the wrap_longitude method increments longitudinal chunks."""
    # Using the xr.open_dataset method
    ds = xr.open_dataset(met_ecmwf_pl_path)
    mds = MetDataset(ds)

    # NOTE: mds doesn't have chunks
    assert mds.data.chunks == {}
    wrapped = mds.wrap_longitude()
    assert wrapped.data.chunks == {}
    assert len(wrapped.data["longitude"]) == len(mds.data["longitude"]) + 2

    # Using our open_metdataset method
    era5 = ERA5(
        time=ds["time"].values,
        variables=list(ds.data_vars),
        pressure_levels=ds["level"].values,
        paths=met_ecmwf_pl_path,
        cachestore=override_cache,
    )
    mds2 = era5.open_metdataset()

    # This one's got chunks
    chunks = mds2.data.chunks
    assert "longitude" in chunks
    assert chunks["longitude"] == (15,)

    wrapped2 = mds2.wrap_longitude()
    assert len(wrapped.data["longitude"]) == len(mds.data["longitude"]) + 2

    # successfully incremented chunks
    assert wrapped2.data.chunks["longitude"] == (17,)


def test_broadcast_coords(zero_like_da: xr.DataArray, met_ecmwf_pl_path: str) -> None:
    """Test the `broadcast_coords` method."""

    mda = MetDataArray(zero_like_da)
    da1 = mda.broadcast_coords("level")
    assert da1.name == "level"
    assert da1.shape == (360, 181, 4, 5)
    assert da1[0, 1, 1, 1] == 200

    # non-dimension coords
    da2 = mda.broadcast_coords("air_pressure")
    assert da2.name == "air_pressure"
    assert da2.shape == (360, 181, 4, 5)
    assert da2[0, 1, 1, 1] == 20000

    # met dataset
    mds = MetDataset(xr.open_dataset(met_ecmwf_pl_path))
    da3 = mds.broadcast_coords("longitude")
    assert da3.name == "longitude"
    assert da3.shape == (15, 8, 3, 2)
    assert da3[0, 1, 1, 0] == -160


def test_downselect(zero_like_da: xr.DataArray, met_ecmwf_pl_path: str) -> None:
    """Test the `downselect` method."""

    mda = MetDataArray(zero_like_da)
    mda2 = mda.downselect([0, 0, 10, 20])
    assert mda is not mda2  # returns a copy
    assert mda2.name == mda.name
    assert (mda2.data["longitude"].min(), mda2.data["longitude"].max()) == (0, 10)
    assert (mda2.data["latitude"].min(), mda2.data["latitude"].max()) == (0, 20)
    assert (mda2.data["level"].min(), mda2.data["level"].max()) == (
        mda.data["level"].min(),
        mda.data["level"].max(),
    )

    # level selection
    mda2 = mda.downselect([0, 0, 200, 10, 20, 300])
    assert (mda2.data["longitude"].min(), mda2.data["longitude"].max()) == (0, 10)
    assert (mda2.data["latitude"].min(), mda2.data["latitude"].max()) == (0, 20)
    assert (mda2.data["level"].min(), mda2.data["level"].max()) == (200, 300)

    with pytest.raises(ValueError, match="is not length 4"):
        mda.downselect([0, 0, 200, 10, 20, 300, 5000])

    # wrap data
    mda2 = mda.downselect([179, 0, -179, 20])
    assert (mda2.data["longitude"].min(), mda2.data["longitude"].max()) == (-180, 179)
    assert np.all(mda2.data["longitude"] == np.array([-180, -179, 179]))
    assert (mda2.data["latitude"].min(), mda2.data["latitude"].max()) == (0, 20)

    # met dataset
    mds = MetDataset(xr.open_dataset(met_ecmwf_pl_path))
    mds2 = mds.downselect([0, 0, 125, 40])
    assert mds is not mds2  # does not copy
    assert (mds2.data["longitude"].min(), mds2.data["longitude"].max()) == (0, 125)
    assert (mds2.data["latitude"].min(), mds2.data["latitude"].max()) == (15, 40)
    assert (mds2.data["level"].min(), mds2.data["level"].max()) == (
        mds.data["level"].min(),
        mds.data["level"].max(),
    )

    mds2 = mds.downselect([0, 0, 200, 125, 40, 300])
    assert (mds2.data["longitude"].min(), mds2.data["longitude"].max()) == (0, 125)
    assert (mds2.data["latitude"].min(), mds2.data["latitude"].max()) == (15, 40)
    assert (mds2.data["level"].min(), mds2.data["level"].max()) == (225, 300)


def test_met_dataset_iter(met_ecmwf_pl_path: str) -> None:
    """Test implementation of MetDataset.__iter__."""
    met = MetDataset(xr.open_dataset(met_ecmwf_pl_path))
    assert list(met) == ["t", "r", "q", "ciwc"]


def test_met_dataset_setitem(met_ecmwf_pl_path: str) -> None:
    """Test implementation of MetDataset.__setitem__."""
    met = MetDataset(xr.open_dataset(met_ecmwf_pl_path))

    # both of the following should work
    met.data["r2"] = met.data["t"].copy()
    met["r3"] = met["t"].copy()
    np.testing.assert_array_equal(met["r2"].values, met["t"].values)
    np.testing.assert_array_equal(met["r3"].values, met["t"].values)

    met.data["r2"] = met["t"].copy().data
    np.testing.assert_array_equal(met["r2"].values, met["t"].values)

    with pytest.warns(UserWarning, match=r"Overwriting data in keys `\['r'\]`"):
        met["r"] = met["t"].copy()

    # update should work without warnings
    met.data.update({"r2": met.data["t"].copy()})
    np.testing.assert_array_equal(met["r2"].values, met["t"].values)

    met.update({"r3": met["t"].copy()})
    np.testing.assert_array_equal(met["r3"].values, met["t"].values)
    met.update({"r4": met.data["t"].copy()})
    np.testing.assert_array_equal(met["r4"].values, met["t"].values)

    met.data.update({"r5": met["t"].copy().data})
    np.testing.assert_array_equal(met["r5"].values, met["t"].values)


def test_met_variable() -> None:
    """Test MetVariable dataclass."""

    # new parameters
    NewParam = MetVariable(ecmwf_id=829, standard_name="new-param", short_name="wt", units="n")
    assert isinstance(NewParam, MetVariable)
    assert NewParam.short_name == "wt"
    assert NewParam.ecmwf_link is not None and "grib/param-db?id=829" in NewParam.ecmwf_link

    NewParam2 = MetVariable(standard_name="new-param2", short_name="wt", units="n")
    assert NewParam2.ecmwf_link is None

    # standard name and short name are required
    with pytest.raises(TypeError, match="short_name"):
        MetVariable(standard_name="new-param")

    with pytest.raises(TypeError, match="standard_name"):
        MetVariable(short_name="new-param")

    # should have attrs property
    NewParam = MetVariable(ecmwf_id=829, standard_name="new-param", short_name="wt", units="n")
    attrs = NewParam.attrs
    assert "standard_name" in attrs and attrs["standard_name"] == "new-param"
    assert "short_name" in attrs and attrs["short_name"] == "wt"
    assert "long_name" not in attrs
    assert "ecmwf_id" not in attrs


def test_met_ensure_vars(met_ecmwf_pl_path: str) -> None:
    """Test `met.ensure_vars()."""
    ds = xr.open_dataset(met_ecmwf_pl_path)
    met = MetDataset(ds)

    # should work with strings and return string names
    variables = met.ensure_vars(["t", "r", "q"])
    assert variables == ["t", "r", "q"]

    with pytest.raises(KeyError, match="Dataset does not contain variable `w`"):
        met.ensure_vars(["t", "r", "w"])

    # should work with MetVariables and return standard_name
    t = MetVariable(short_name="t", standard_name="t")
    r = MetVariable(short_name="r", standard_name="r")
    w = MetVariable(short_name="w", standard_name="w")
    variables = met.ensure_vars([t, r])
    assert variables == ["t", "r"]

    with pytest.raises(KeyError, match="Dataset does not contain variable `w`"):
        met.ensure_vars([t, r, w])

    # should work with a list of MetVariables and return standard_name
    variables = met.ensure_vars([[t, w], r])
    assert variables == ["t", "r"]

    variables = met.ensure_vars(["r", [w, t]])
    assert variables == ["r", "t"]


@pytest.mark.filterwarnings("ignore:Longitude is not evenly spaced")
def test_include_altitude(met_ecmwf_pl_path: str):
    """Check `to_polygon_feature` with `include_altitude=True`."""
    ds = xr.open_dataset(met_ecmwf_pl_path)
    mds = MetDataset(ds[dict(time=[0])])
    mda = mds["t"]

    geojson = mda.to_polygon_feature(level=225, include_altitude=True, iso_value=100)
    for poly in geojson["geometry"]["coordinates"]:
        for ring in poly:
            for coord in ring:
                assert len(coord) == 3
                assert coord[2] == 11037.0


@pytest.mark.filterwarnings("ignore:Longitude is not evenly spaced")
def test_include_altitude_raises(met_ecmwf_sl_path: str):
    """Check `to_polygon_feature` raises a ValueError when `include_altitude=True` and SL data."""
    ds = xr.open_dataset(met_ecmwf_sl_path)
    mds = MetDataset(ds[dict(time=[0])])
    mda = mds["sp"]
    with pytest.raises(ValueError, match=" but altitude is not found"):
        mda.to_polygon_feature(include_altitude=True, iso_value=20000)

    geojson = mda.to_polygon_feature(include_altitude=False, iso_value=20000)
    for poly in geojson["geometry"]["coordinates"]:
        for ring in poly:
            for coord in ring:
                assert len(coord) == 2


@pytest.mark.filterwarnings("ignore:Longitude is not evenly spaced")
def test_to_polygon_feature_collection(met_ecmwf_pl_path: str):
    """Confirm that `to_polygon_feature_collection` works."""
    ds = xr.open_dataset(met_ecmwf_pl_path)
    mds = MetDataset(ds[dict(time=[0])])
    mda = mds["q"]

    fc = mda.to_polygon_feature_collection(iso_value=0.0001)
    assert fc.keys() == {"type", "features"}
    assert fc["type"] == "FeatureCollection"
    features = fc["features"]
    assert len(features) == len(ds.level)
    for feature in features:
        assert feature["properties"]


def test_wrap_longitude_new_patterns():
    """Test new wrap_longitude patterns (pycontails 0.26.0)."""
    da1 = xr.DataArray(
        data=np.random.random((360, 5, 5, 1)),
        dims=["longitude", "latitude", "level", "time"],
        coords={
            "longitude": np.arange(-180, 180),
            "latitude": np.arange(0, 50, 10),
            "level": np.arange(50, 500, 100),
            "time": [datetime(2000, 1, 1, 0)],
        },
    )
    # Includes new longitude at 180
    mda1 = MetDataArray(da1, wrap_longitude=True)
    assert mda1.shape == (361, 5, 5, 1)
    assert mda1.is_wrapped
    np.testing.assert_array_equal(mda1.data.longitude, np.arange(-180, 181))
    np.testing.assert_array_equal(mda1.data.sel(longitude=180), mda1.data.sel(longitude=-180))

    # Drop longitude at -180
    da2 = da1.isel(longitude=slice(1, None))
    assert da2.shape == (359, 5, 5, 1)
    np.testing.assert_array_equal(da2.longitude, np.arange(-179, 180, 1))

    # New mda includes value at -181 and value at 181
    mda2 = MetDataArray(da2, wrap_longitude=True)
    assert mda2.shape == (361, 5, 5, 1)
    assert mda2.is_wrapped
    np.testing.assert_array_equal(mda2.data.longitude, np.r_[-181, np.arange(-179, 180), 181])

    # Drop longitude at 179
    da3 = da1.isel(longitude=slice(None, -1))
    assert da3.shape == (359, 5, 5, 1)
    np.testing.assert_array_equal(da3.longitude, np.arange(-180, 179, 1))
    mda3 = MetDataArray(da3, wrap_longitude=True)
    assert mda3.shape == (360, 5, 5, 1)
    assert mda3.is_wrapped
    np.testing.assert_array_equal(mda3.data.longitude, np.r_[np.arange(-180, 179), 180])


def test_originates_from_ecmwf_met_ecmwf_pl_path(met_ecmwf_pl_path: str):
    """Test the `originates_from_ecmwf` property."""
    ds = xr.open_dataset(met_ecmwf_pl_path)
    mds = MetDataset(ds)
    assert originates_from_ecmwf(mds)


def test_originates_from_ecmwf_met_ecmwf_sl_path(met_ecmwf_sl_path: str):
    """Test the `originates_from_ecmwf` property."""
    ds = xr.open_dataset(met_ecmwf_sl_path)
    mds = MetDataset(ds)
    assert originates_from_ecmwf(mds)


def test_originates_from_ecmwf_met_pcc_pl(met_pcc_pl: MetDataset):
    """Test the `originates_from_ecmwf` property."""
    assert originates_from_ecmwf(met_pcc_pl)


def test_originates_from_ecmwf_sparse_binary(sparse_binary: tuple[MetDataArray, Any]):
    """Test the `originates_from_ecmwf` property."""
    da = sparse_binary[0].data
    ds = da.to_dataset()
    mds = MetDataset(ds)
    assert not originates_from_ecmwf(mds)
