"""Test datalib met utilities."""

import numpy as np
import numpy.typing as npt
import pytest
import xarray as xr

from pycontrails.datalib import met_utils

# Reference air temperature for synthetic model-level data (K)
T0 = 300.0

# Reference pressure for synthetic model-level data (hPa)
P0 = 1000.0

# Approximate value for R/cp
KAPPA = 2.0 / 7.0


@pytest.fixture()
def ds_ml() -> xr.Dataset:
    """Generate synthetic model-level data for testing interpolation to pressure levels.

    This test case takes advantage of the fact that log(T) is linear
    in log(p) for dry adiabatic temperature profiles; i.e., if
    `T(p) = T0 * (p/p0)^(R/cp)`,
    then
    `log(T) = log(T0) - R/cp * (log(p0) - log(p))`.

    Because the model-to-pressure-level conversion interpolates
    linearly in log(p), results will match `T(p)` exactly if
    interpolation is applied to `log(T)`.

    The synthetic data contains five model levels centered at
    100, 200, 300, 400, and 500 hPa. Pressure on each model level
    varies sinusoidally with an amplitude of 10 hPa in two spatial dimensions.

    """
    x = np.linspace(0, np.pi, 4)
    y = np.linspace(0, np.pi, 6)
    model_level = np.arange(1, 6)

    ds = xr.Dataset(coords={"x": x, "y": y, "model_level": model_level})
    ds["pressure_level"] = 100.0 * ds["model_level"] + 10.0 * np.sin(ds["x"]) * np.sin(ds["y"])
    ds["air_temperature"] = T0 * (ds["pressure_level"] / P0) ** KAPPA

    return ds


@pytest.mark.parametrize(
    "target_pl",
    [
        150,
        150.0,
        [150, 250, 350, 450],
        [150.0, 250.0, 350.0, 450.0],
        [150, 450, 350, 250],
    ],
)
def test_ml_to_pl(ds_ml: xr.Dataset, target_pl: npt.ArrayLike) -> None:
    """Test interpolation output."""

    ds_ml["log_air_temperature"] = np.log(ds_ml["air_temperature"])
    ds_ml = ds_ml.drop_vars("air_temperature")
    ds_pl = met_utils.ml_to_pl(ds_ml, target_pl)

    assert len(ds_pl.data_vars) == 1
    assert "log_air_temperature" in ds_pl.data_vars

    assert "level" in ds_pl.coords
    assert ds_pl.sizes["level"] == len(target_pl) if isinstance(target_pl, list) else 1
    assert ds_pl["level"].dtype == ds_ml["pressure_level"].dtype

    xr.testing.assert_equal(ds_pl["x"], ds_ml["x"])
    xr.testing.assert_equal(ds_pl["y"], ds_ml["y"])

    for p in target_pl if isinstance(target_pl, list) else [target_pl]:
        assert p in ds_pl["level"]
        np.testing.assert_allclose(
            np.exp(ds_pl["log_air_temperature"].sel(level=p).values), T0 * (p / P0) ** KAPPA
        )


@pytest.mark.parametrize(
    "target_pl",
    [
        150,
        150.0,
        [150, 250, 350, 450],
        [150.0, 250.0, 350.0, 450.0],
        [150, 450, 350, 250],
    ],
)
def test_ml_to_pl_lazy(ds_ml: xr.Dataset, target_pl: npt.ArrayLike) -> None:
    """Test lazy computation with dask-backed dataset."""

    ds_ml_lazy = ds_ml.chunk({"x": 2, "y": 3, "model_level": -1})
    ds_pl_lazy = met_utils.ml_to_pl(ds_ml_lazy, target_pl)
    ds_pl = met_utils.ml_to_pl(ds_ml, target_pl)

    assert not ds_pl_lazy["air_temperature"]._in_memory
    assert ds_pl_lazy["air_temperature"].chunks == ((2, 2), (3, 3), (ds_pl.sizes["level"],))
    xr.testing.assert_equal(ds_pl_lazy.compute(), ds_pl)


def test_ml_to_pl_out_of_bounds(ds_ml: xr.Dataset) -> None:
    """Test behavior with out-of-bounds interpolation."""

    ds_pl = met_utils.ml_to_pl(ds_ml, [50, 150, 250, 350, 450, 550])
    assert np.isnan(ds_pl.sel(level=slice(100, 500))).sum() == 0
    assert np.isnan(ds_pl.sel(level=50)).all()
    assert np.isnan(ds_pl.sel(level=550)).all()


def test_ml_to_pl_nan_in_data(ds_ml: xr.Dataset) -> None:
    """Test behavior with nans in non-pressure data."""

    ds_ml["air_temperature"].loc[dict(x=0, y=0, model_level=3)] = np.nan
    ds_pl = met_utils.ml_to_pl(ds_ml, [150, 250, 350, 450])

    assert np.isnan(ds_pl["air_temperature"]).sum() == 2
    assert np.isnan(ds_pl["air_temperature"].loc[dict(x=0, y=0, level=250)])
    assert np.isnan(ds_pl["air_temperature"].loc[dict(x=0, y=0, level=350)])


def test_ml_to_pl_nan_in_pressure(ds_ml: xr.Dataset) -> None:
    """Test behavior with nans in pressure data."""

    ds_ml["pressure_level"].loc[dict(x=0, y=0, model_level=3)] = np.nan
    ds_pl = met_utils.ml_to_pl(ds_ml, [150, 250, 350, 450])

    assert np.isnan(ds_pl["air_temperature"]).sum() == 4
    assert np.isnan(ds_pl["air_temperature"].sel(x=0, y=0)).all()


def test_ml_to_pl_nan_in_target_pl(ds_ml: xr.Dataset) -> None:
    """Test behavior with nans in target pressure levels."""

    with pytest.raises(ValueError, match="Target pressure levels must not"):
        _ = met_utils.ml_to_pl(ds_ml, np.nan)

    with pytest.raises(ValueError, match="Target pressure levels must not"):
        _ = met_utils.ml_to_pl(ds_ml, [150, 250, np.nan, 450])


def test_ml_to_pl_no_pressure(ds_ml: xr.Dataset) -> None:
    """Test missing pressure error handing."""

    ds_ml = ds_ml.rename_vars(pressure_level="foo")
    with pytest.raises(ValueError, match="The dataset must contain"):
        met_utils.ml_to_pl(ds_ml, 150)


def test_ml_to_pl_no_model_levels(ds_ml: xr.Dataset) -> None:
    """Test missing model_level coordinate warnings and errors."""

    ds_ml["foo"] = xr.zeros_like(ds_ml["x"])
    with pytest.warns(UserWarning, match="Variable 'foo' does not have"):
        ds_pl = met_utils.ml_to_pl(ds_ml, 150)

    assert len(ds_pl.data_vars) == 1
    assert "air_temperature" in ds_pl.data_vars

    ds_ml = ds_ml.rename_dims(model_level="bar")
    with (
        pytest.warns(UserWarning, match=r"Variable '.*' does not have"),
        pytest.raises(ValueError, match="Dataset has no variables with"),
    ):
        _ = met_utils.ml_to_pl(ds_ml, 150)


def test_ml_to_pl_invalid_chunks(ds_ml: xr.Dataset) -> None:
    """Test error handling with invalid chunking."""

    ds_ml_lazy = ds_ml.chunk({"model_level": 1})
    with pytest.raises(ValueError, match="The 'model_level' dimension must not"):
        _ = met_utils.ml_to_pl(ds_ml_lazy, 150)
