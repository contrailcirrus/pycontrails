"""Test interpolation edge cases, etc."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.interpolate

from pycontrails import GeoVectorDataset, MetDataArray, MetDataset
from pycontrails.core import interpolation as interp_mod
from pycontrails.core import models as models_mod


@pytest.fixture
def mda(met_pcc_pl: MetDataset) -> MetDataArray:
    """Return a MetDataArray for interpolation."""
    assert isinstance(met_pcc_pl, MetDataset)
    return met_pcc_pl["air_temperature"]


def test_basic_interpolation(mda: MetDataArray, caplog: pytest.LogCaptureFixture) -> None:
    """Test basic interpolation patterns."""
    assert mda.shape == (15, 8, 3, 2)

    longitude = np.array([1, 2, 3])
    latitude = np.array([1, 2, 3])
    level = np.array([230, 240, 250])
    t0 = np.datetime64("2019-05-31T05:30")
    time = np.array([t0, t0, t0])

    with caplog.at_level("DEBUG", logger="pycontrails.core.interpolation"):
        out1 = mda.interpolate(longitude, latitude, level, time, bounds_error=True, localize=False)
        assert len(caplog.records) == 0
        out2 = mda.interpolate(longitude, latitude, level, time, bounds_error=True, localize=True)
        # Confirm that the logger in _localize emitted messages
        assert len(caplog.records) == 4

    assert out1.dtype == mda.data.dtype
    assert out2.dtype == mda.data.dtype

    out3 = mda.interpolate(
        longitude.astype("float32"),
        latitude.astype("float32"),
        level.astype("float32"),
        time,
        localize=False,
    )
    out4 = mda.interpolate(
        longitude.astype("float32"),
        latitude.astype("float32"),
        level.astype("float32"),
        time,
        localize=True,
    )
    assert out3.dtype == mda.data.dtype
    assert out4.dtype == mda.data.dtype

    np.testing.assert_array_equal(out1, out2)
    np.testing.assert_array_equal(out3, out4)
    np.testing.assert_allclose(out1, out3, rtol=2e-7)

    for out in [out1, out2, out3, out4]:
        assert np.all(np.isfinite(out))


@pytest.mark.parametrize("localize", [True, False])
@pytest.mark.parametrize("dim", ["longitude", "latitude", "level", "time"])
def test_interpolation_singleton_dim_not_nan(dim: str, mda: MetDataArray, localize: bool) -> None:
    """Confirm that interpolation output is not nan when singleton dimension encountered."""
    out_of_bounds_da = mda.data.isel(**{dim: [-1]})
    mda.data = mda.data.isel(**{dim: [0]})

    # Exactly 1 singleton dimension
    assert 1 in mda.shape
    assert mda.shape.count(1) == 1

    coords = {
        "longitude": np.array([1]),
        "latitude": np.array([1]),
        "level": np.array([230]),
        "time": np.array([np.datetime64("2019-05-31T05:30")]),
    }

    # Overwrite coords to include the singleton dimension
    coords[dim] = mda.data[dim].values
    out = mda.interpolate(**coords, bounds_error=True, localize=localize)
    assert np.all(np.isfinite(out))

    # Overwrite coords to include an out of bounds value at the singleton dimension
    coords[dim] = out_of_bounds_da[dim].values
    axis = mda.data.get_axis_num(dim)
    match = f"One of the requested xi is out of bounds in dimension {axis}"
    with pytest.raises(ValueError, match=match):
        mda.interpolate(**coords, bounds_error=True, localize=localize)

    out = mda.interpolate(**coords, bounds_error=False, localize=localize)
    np.testing.assert_array_equal(out, [np.nan])

    out = mda.interpolate(**coords, bounds_error=False, localize=localize, fill_value=999)
    np.testing.assert_array_equal(out, [999])


@pytest.mark.parametrize("localize", [True, False])
def test_interpolation_single_level(mda: MetDataArray, localize: bool) -> None:
    """Confirm interpolation works with single level (level = -1) DataArray."""
    # Convert mda to a DataArray with a single level
    mda.data = mda.data.isel(level=[0]).assign_coords(level=[-1.0])
    np.testing.assert_array_equal(mda.data.level, [-1])
    assert mda.shape == (15, 8, 1, 2)

    longitude = [30, 40, 200]
    latitude = [30, 40, 60]
    level = [300, 400, 500]
    time = [
        np.datetime64("2019-05-31T05:30"),
        np.datetime64("2019-05-31T05:33"),
        np.datetime64("2019-05-31T05:36"),
    ]

    # Longitude is out of bounds
    with pytest.raises(ValueError, match="One of the requested xi is out of bounds in dimension 0"):
        mda.interpolate(longitude, latitude, level, time, bounds_error=True, localize=localize)

    # Only the third coordinate is out of bounds .... level interpolation works as intended
    out = mda.interpolate(longitude, latitude, level, time, bounds_error=False, localize=localize)
    np.testing.assert_array_equal(np.isnan(out), [False, False, True])


def test_localize(mda: MetDataArray) -> None:
    """Confirm _localize implementation."""
    coords = {}
    coords["longitude"] = np.array([1, 2, 3])
    coords["latitude"] = np.array([1, 2, 3])
    coords["level"] = np.array([250, 250, 250])
    t0 = np.datetime64("2019-05-31T05:30")
    coords["time"] = np.array([t0, t0, t0])

    da = mda.data
    out = interp_mod._localize(da, coords)
    assert out.ndim == 4
    np.testing.assert_array_equal(out["longitude"], [0, 25])
    np.testing.assert_array_equal(out["latitude"], [-10, 15])
    np.testing.assert_array_equal(out["level"], [250])
    np.testing.assert_array_equal(out["time"], da["time"])

    # Try one more situation
    coords["longitude"] = np.array([-180, -180, -190])
    coords["latitude"] = np.array([-100, 85, 86])
    coords["level"] = np.array([230, 240, 220])

    out = interp_mod._localize(da, coords)
    assert out.ndim == 4
    np.testing.assert_array_equal(out["longitude"], [-160])
    np.testing.assert_array_equal(out["latitude"], da["latitude"])
    np.testing.assert_array_equal(out["level"], [225.0, 250.0])
    np.testing.assert_array_equal(out["time"], da["time"])

    # And another situation
    t0 = np.datetime64("2019-05-31T06:00")
    coords["time"] = np.array([t0, t0, t0])
    out = interp_mod._localize(da, coords)
    assert out.ndim == 4
    np.testing.assert_array_equal(out["longitude"], [-160])
    np.testing.assert_array_equal(out["latitude"], da["latitude"])
    np.testing.assert_array_equal(out["level"], [225.0, 250.0])
    np.testing.assert_array_equal(out["time"], [t0])


@pytest.mark.parametrize("localize", [True, False])
def test_interpolation_time_resolution(mda: MetDataArray, localize: bool) -> None:
    """Confirm interpolation works as expected with time to float conversions."""
    longitude = 30
    latitude = 31
    level = 234
    time = np.datetime64("2019-05-31T06:00:01")

    # Time is out of bounds
    match = "One of the requested xi is out of bounds in dimension 3"
    with pytest.raises(ValueError, match=match):
        mda.interpolate(longitude, latitude, level, time, bounds_error=True, localize=localize)

    # If time is identical with the endpoint of the time dimension,
    # it should be in bounds.
    time = np.datetime64("2019-05-31T06:00:00")

    out = mda.interpolate(longitude, latitude, level, time, bounds_error=False, localize=localize)
    assert np.isfinite(out)

    # Increasing it by 1 us makes it out of bounds
    time += np.timedelta64(1, "us")
    out = mda.interpolate(longitude, latitude, level, time, bounds_error=False, localize=localize)
    assert np.isnan(out)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_regular_4d_grid_interpolator(mda: MetDataArray, dtype: str) -> None:
    """Confirm ``PycontrailsRegularGridInterpolator`` agrees with scipy parent class."""
    x = mda.data["longitude"].values
    y = mda.data["latitude"].values
    z = mda.data["level"].values
    t = mda.data["time"].values
    t = interp_mod._floatize_time(t, t[0])
    assert x.dtype == np.float64
    assert y.dtype == np.float64
    assert z.dtype == np.float64
    assert t.dtype == np.float64

    mda.data = mda.data.astype(dtype)

    points = x, y, z, t
    values = mda.values

    kwargs = {"method": "linear", "bounds_error": True, "fill_value": np.nan}
    rgi1 = interp_mod.PycontrailsRegularGridInterpolator(points, values, **kwargs)
    rgi2 = scipy.interpolate.RegularGridInterpolator(points, values, **kwargs)

    rng = np.random.default_rng(6567)
    n = 100_000
    x0 = rng.uniform(x.min(), x.max(), size=n)
    y0 = rng.uniform(y.min(), y.max(), size=n)
    z0 = rng.uniform(z.min(), z.max(), size=n)
    t0 = rng.uniform(t.min(), t.max(), size=n)
    xi = np.stack([x0, y0, z0, t0], axis=1)

    out1 = rgi1(xi)
    out2 = rgi2(xi)

    # The pycontrails version always uses the same dtype as the underlying values array
    assert out1.dtype == mda.data.dtype

    # The scipy version essentially always promotes to float64
    assert out2.dtype == np.float64

    np.testing.assert_allclose(out1, out2, rtol=1e-7)


@pytest.mark.parametrize("localize", [True, False])
@pytest.mark.parametrize("dim", ["longitude", "latitude", "level", "time"])
def test_regular_3d_grid_interpolator(mda: MetDataArray, dim: str, localize: bool) -> None:
    """Confirm ``PycontrailsRegularGridInterpolator`` can handle singleton dimensions.

    Previously, scipy 1.8 could not handle singleton dimensions. This was fixed in
    scipy 1.9 and the pycontrails homegrown implementation was removed in
    pycontrails 0.25.6.

    Implementations changed again in scipy 1.10 and pycontrails ~0.35.
    """
    da = mda.data.isel(**{dim: [0]})
    x = da["longitude"].values
    y = da["latitude"].values
    z = da["level"].values
    t = da["time"].values

    rng = np.random.default_rng(6567)
    n = 100_000
    x0 = rng.uniform(x.min(), x.max(), size=n)
    y0 = rng.uniform(y.min(), y.max(), size=n)
    z0 = rng.uniform(z.min(), z.max(), size=n)
    t0 = rng.uniform(t.min(), t.max(), size=n).astype("datetime64[ns]")

    # Run interpolation through pycontrails interface
    kwargs = {"method": "linear", "bounds_error": True, "fill_value": np.nan}
    out1 = interp_mod.interp(x0, y0, z0, t0, da=da, localize=localize, **kwargs)
    assert np.all(np.isfinite(out1))

    # Run interpolation through scipy RGI
    # First "floatize" time (this is done automatically in interp_mod.interp)
    t_float = interp_mod._floatize_time(t, t[0])
    t0 = interp_mod._floatize_time(t0, t[0])

    points = x, y, z, t_float
    values = da.values
    rgi = scipy.interpolate.RegularGridInterpolator(points, values, **kwargs)
    xi = np.stack([x0, y0, z0, t0], axis=1)

    # We no longer get nan values with single dimension interpolation
    # This was fixed in scipy 1.9.0. All values are finite now.
    out2 = rgi(xi)
    assert np.all(np.isfinite(out2))

    np.testing.assert_allclose(out1, out2, rtol=1e-7)


@pytest.mark.parametrize("method", ["nearest", "slinear", "cubic", "quintic"])
def test_scipy19_interpolation_methods(mda: MetDataArray, method: str) -> None:
    """Confirm ``PycontrailsRegularGridInterpolator`` can handle methods introduced in scipy 1.9."""
    da = mda.data
    assert da.shape == (15, 8, 3, 2)
    assert da.size == 720

    x = da["longitude"].values
    y = da["latitude"].values
    z = da["level"].values
    t = da["time"].values

    rng = np.random.default_rng(4345)
    x0 = rng.uniform(x.min(), x.max(), size=1000)
    y0 = rng.uniform(y.min(), y.max(), size=1000)
    z0 = rng.uniform(z.min(), z.max(), size=1000)
    t0 = rng.uniform(t.min(), t.max(), size=1000).astype("datetime64[ns]")

    # Run interpolation through pycontrails interface
    kwargs = {"method": method, "bounds_error": True, "fill_value": np.nan, "localize": True}
    if method in ["cubic", "quintic"]:
        with pytest.raises(ValueError, match="The number of derivatives at boundaries does not"):
            interp_mod.interp(x0, y0, z0, t0, da=da, **kwargs)
        return

    out1 = interp_mod.interp(x0, y0, z0, t0, da=da, **kwargs)
    assert np.all(np.isfinite(out1))
    assert out1.dtype == "float32"

    kwargs = {"method": "linear", "bounds_error": True, "fill_value": np.nan, "localize": True}
    out2 = interp_mod.interp(x0, y0, z0, t0, da=da, **kwargs)
    assert np.all(np.isfinite(out2))

    if method == "slinear":
        np.testing.assert_array_equal(out1, out2)
        return

    # Pin the RMSE (out of curiosity)
    rmse = (np.mean((out1 - out2) ** 2)) ** 0.5
    assert rmse == pytest.approx(3.3319390877860893, rel=1e-10)


@pytest.fixture
def arbitrary_coords() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    longitude = np.array([-55, -43, -55, -21, -17, -18], dtype=np.float32)
    latitude = np.array([12, 17, -8, -44, 22, 23], dtype=np.float32)
    level = np.array([227, 228, 231, 233, 231, 230], dtype=np.float32)
    time = np.array(
        [
            np.datetime64("2019-05-31T05:30"),
            np.datetime64("2019-05-31T05:33"),
            np.datetime64("2019-05-31T05:36"),
            np.datetime64("2019-05-31T05:40"),
            np.datetime64("2019-05-31T05:43"),
            np.datetime64("2019-05-31T05:46"),
        ]
    )
    return longitude, latitude, level, time


def test_indices_distinct_vars(
    met_pcc_pl: MetDataset,
    arbitrary_coords: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> None:
    """Check that indices are consistent when interpolating over distinct variables."""

    assert list(met_pcc_pl) == [
        "air_temperature",
        "specific_humidity",
        "specific_cloud_ice_water_content",
    ]

    out1, indices1 = interp_mod.interp(
        *arbitrary_coords,
        met_pcc_pl["air_temperature"].data,
        method="linear",
        bounds_error=True,
        fill_value=np.nan,
        localize=False,
        return_indices=True,
    )
    assert np.all(np.isfinite(out1))

    out2, indices2 = interp_mod.interp(
        *arbitrary_coords,
        met_pcc_pl["specific_humidity"].data,
        method="linear",
        bounds_error=True,
        fill_value=np.nan,
        localize=False,
        return_indices=True,
    )
    assert np.all(np.isfinite(out2))

    np.testing.assert_array_equal(indices1.xi_indices, indices2.xi_indices)
    np.testing.assert_array_equal(indices1.norm_distances, indices2.norm_distances)
    assert not np.any(indices1.out_of_bounds)
    assert not np.any(indices2.out_of_bounds)


@pytest.mark.parametrize(
    "var", ["air_temperature", "specific_humidity", "specific_cloud_ice_water_content"]
)
def test_indices_same_var(
    met_pcc_pl: MetDataset,
    arbitrary_coords: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    var: str,
) -> None:
    """Check that interpolation with and without indices gives the same result."""

    da = met_pcc_pl[var].data

    out1, indices = interp_mod.interp(
        *arbitrary_coords,
        da,
        method="linear",
        bounds_error=True,
        fill_value=np.nan,
        localize=False,
        return_indices=True,
    )

    # Expect this to be faster because indices are already computed
    out2 = interp_mod.interp(
        *arbitrary_coords,
        da,
        method="linear",
        bounds_error=True,
        fill_value=np.nan,
        localize=False,
        indices=indices,
        return_indices=False,
    )
    np.testing.assert_array_equal(out1, out2)


@pytest.mark.parametrize("bounds_error", [False, True])
@pytest.mark.parametrize("idx", range(5))
@pytest.mark.parametrize("coord", ["longitude", "latitude", "level", "time"])
def test_interpolation_propagate_nan(
    mda: MetDataArray, bounds_error: bool, idx: int, coord: str
) -> None:
    """Ensure nan values propagate through interpolation."""

    longitude = np.arange(5, dtype=float)
    latitude = np.arange(5, 10, dtype=float)
    level = np.linspace(225, 250, 5)
    time = np.r_[[np.datetime64("2019-05-31T05:20")] * 3, [np.datetime64("2019-05-31T05:36")] * 2]

    if coord == "longitude":
        longitude[idx] = np.nan
        dim = 0
    elif coord == "latitude":
        latitude[idx] = np.nan
        dim = 1
    elif coord == "level":
        level[idx] = np.nan
        dim = 2
    elif coord == "time":
        time[idx] = np.datetime64("NaT")
        dim = 3

    if bounds_error:
        match = f"One of the requested xi is out of bounds in dimension {dim}"
        with pytest.raises(ValueError, match=match):
            mda.interpolate(longitude, latitude, level, time, bounds_error=bounds_error)
        return

    out = mda.interpolate(longitude, latitude, level, time, bounds_error=bounds_error)
    assert np.flatnonzero(np.isnan(out)).item() == idx


@pytest.mark.parametrize("replace_value", [np.nan, 224])
@pytest.mark.parametrize("fill_value", [np.nan, 0.0, None, 55.0])
def test_fill_value(
    mda: MetDataArray,
    arbitrary_coords: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    replace_value: float | np.float64,
    fill_value: float | np.float64 | None,
) -> None:
    """Check implementation with nonstandard fill_value."""

    da = mda.data
    assert np.isnan(replace_value) or replace_value < da.level.min()
    arbitrary_coords[2][3] = replace_value

    with pytest.raises(ValueError, match="One of the requested xi is out of bounds in dimension 2"):
        interp_mod.interp(
            *arbitrary_coords,
            da,
            method="linear",
            bounds_error=True,
            fill_value=fill_value,
            localize=False,
        )

    out1, indices = interp_mod.interp(
        *arbitrary_coords,
        da,
        method="linear",
        bounds_error=False,
        fill_value=fill_value,
        localize=False,
        return_indices=True,
    )

    out2 = interp_mod.interp(
        *arbitrary_coords,
        da,
        method="linear",
        bounds_error=False,
        fill_value=fill_value,
        localize=False,
        indices=indices,
    )

    np.testing.assert_array_equal(out1, out2)
    if np.isnan(replace_value):
        assert np.isnan(out1[3])
        return
    if fill_value is None:
        assert out1[3] == pytest.approx(214.0025, abs=1e-4)
        return
    if np.isnan(fill_value):
        assert np.isnan(out1[3])
        return
    assert np.isfinite(out1[3])
    assert out1[3] == fill_value


@pytest.mark.parametrize("q_method", [None, "log-q-log-p", "cubic-spline", "log-q"])
def test_interpolation_q_method(
    met_pcc_pl: MetDataset,
    arbitrary_coords: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    q_method: str,
) -> None:
    """Pin values for experimental interpolation q methods."""

    longitude, latitude, level, time = arbitrary_coords
    vector = GeoVectorDataset(longitude=longitude, latitude=latitude, level=level, time=time)
    assert np.all(vector.coords_intersect_met(met_pcc_pl))

    if q_method == "log-q":
        with pytest.raises(ValueError, match="Invalid 'q_method' value 'log-q'"):
            models_mod.interpolate_met(met_pcc_pl, vector, "specific_humidity", q_method=q_method)
        return

    # The q values are different
    if q_method == "log-q-log-p":
        with pytest.warns(UserWarning, match="log_specific_humidity"):
            q = models_mod.interpolate_met(
                met_pcc_pl, vector, "specific_humidity", q_method=q_method
            )
    else:
        q = models_mod.interpolate_met(met_pcc_pl, vector, "specific_humidity", q_method=q_method)

    mean = q.mean()

    if q_method is None:
        assert mean == pytest.approx(6.557144e-05, abs=1e-8)
    elif q_method == "log-q-log-p":
        assert mean == pytest.approx(5.8988215e-05, abs=1e-8)
    elif q_method == "cubic-spline":
        assert mean == pytest.approx(6.476205e-05, abs=1e-8)

    # The T values are the same
    T = models_mod.interpolate_met(met_pcc_pl, vector, "air_temperature", q_method=q_method)
    assert T.mean() == pytest.approx(223.37915, abs=1e-3)
