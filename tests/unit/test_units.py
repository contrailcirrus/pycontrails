"""Test pycontrails.physics.units module."""

from __future__ import annotations

from inspect import getmembers, isfunction

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pycontrails.physics import constants, units


@pytest.fixture(scope="module")
def rng():
    """Get random number generator."""
    return np.random.default_rng(12345)


def test_ft_m(rng):
    """Check `ft_to_m` and `m_to_ft` are bijective."""
    ft1 = rng.uniform(0, 10000, 10000)

    m1 = units.ft_to_m(ft1)
    ft2 = units.m_to_ft(m1)
    np.testing.assert_array_almost_equal(ft1, ft2)

    m2 = units.ft_to_m(ft2)
    np.testing.assert_array_almost_equal(m1, m2)


def test_knots_mps(rng):
    """Check that `m_per_s_to_knots` and `knots_to_m_per_s` are bijective."""
    mps1 = rng.uniform(0, 10000, 10000)

    knots1 = units.m_per_s_to_knots(mps1)
    mps2 = units.knots_to_m_per_s(knots1)
    np.testing.assert_array_almost_equal(mps1, mps2)

    knots2 = units.m_per_s_to_knots(mps2)
    np.testing.assert_array_almost_equal(knots1, knots2)


def test_rad_deg():
    """Check `degrees_to_radians` conversion on a few values."""
    d = 0
    r = units.degrees_to_radians(d)
    assert r == 0

    d = 180
    r = units.degrees_to_radians(d)
    assert r == pytest.approx(3.14159265)


def test_rad_deg_agree_np(rng):
    """Check that pycontrails implementation of agrees with numpy implementation."""
    x = rng.uniform(0, 1000, 10000)

    rad = units.radians_to_degrees(x)
    np_rad = np.rad2deg(x)
    np.testing.assert_array_equal(rad, np_rad)

    deg = units.degrees_to_radians(x)
    np_deg = np.deg2rad(x)
    np.testing.assert_array_equal(deg, np_deg)


def test_m_pl_bijective():
    """Check that the functions `pl_to_m` and `m_to_pl` are bijective."""
    m1 = np.arange(0, 50000)
    m2 = units.pl_to_m(units.m_to_pl(m1))
    np.testing.assert_allclose(m1, m2, atol=1e-11)

    pl1 = np.arange(1, 2000)
    pl2 = units.m_to_pl(units.pl_to_m(pl1))
    np.testing.assert_allclose(pl1, pl2, atol=1e-11)


def classical_pl_to_m(pl):
    """Classical implementation of pl_to_m."""
    pl_pa = pl * 100
    return (constants.T_msl / 0.0065) * (1 - (pl_pa / constants.p_surface) ** (1 / 5.255))


def test_pl_to_m_close_to_classical(rng):
    """Check that the classical conversion agrees for low altitudes."""
    p = rng.uniform(200, 1000, 10000)
    m1 = classical_pl_to_m(p)
    m2 = units.pl_to_m(p)
    np.testing.assert_allclose(m1, m2, rtol=1e-3)


def test_m_to_pl_int_and_array():
    """Check vectorized call agrees with naive loop.

    Calling two functions that use `np.piecewise` pattern.
    """
    arr = np.arange(15000)
    y1 = units.m_to_pl(arr)
    y2 = [units.m_to_pl(x) for x in arr]
    np.testing.assert_array_equal(y1, y2)

    arr = np.arange(100, 1000)
    y1 = units.pl_to_m(arr)
    y2 = [units.pl_to_m(x) for x in arr]
    np.testing.assert_array_equal(y1, y2)


def test_mach_tas(rng: np.random.Generator):
    """Check that the functions `tas_to_mach_number` and `mach_number_to_tas` are bijective."""
    T = rng.uniform(200, 300, 10000)
    tas1 = rng.uniform(200, 300, 10000)
    ma1 = units.tas_to_mach_number(tas1, T)

    # Ensure somewhat realistic values
    assert np.all(np.isfinite(ma1))
    assert np.all(ma1 < 1.1)
    assert np.all(ma1 > 0.5)
    assert ma1.mean() == pytest.approx(0.79, abs=0.01)

    tas2 = units.mach_number_to_tas(ma1, T)
    np.testing.assert_array_almost_equal(tas1, tas2, decimal=10)

    ma2 = units.tas_to_mach_number(tas2, T)
    assert np.all(np.isfinite(ma2))
    np.testing.assert_array_almost_equal(ma1, ma2, decimal=10)


@pytest.mark.parametrize(
    "func",
    [
        func
        for name, func in getmembers(units, isfunction)
        if name
        not in [
            "support_arraylike",
            "longitude_distance_to_m",
            "m_to_longitude_distance",
            "tas_to_mach_number",
            "mach_number_to_tas",
        ]
        and not name.startswith("_")
    ],
)
def test_arraylike_support(func):
    """Check that `unit` module functions support ArrayLike parameters."""
    x = 123
    y = 123.456789
    assert isinstance(func(x), float)
    assert isinstance(func(y), float)

    z = np.array([x, y])
    assert isinstance(func(z), np.ndarray)

    # Some functions can handle Series or DataArrays
    # Others convert to numpy
    da = xr.DataArray(z)
    assert isinstance(func(da), (xr.DataArray, np.ndarray))

    # Calling xr.apply_ufunc gives same result
    xr.testing.assert_equal(func(da), xr.apply_ufunc(func, da))

    s = pd.Series(z)
    assert isinstance(func(s), (pd.Series, np.ndarray))


@pytest.mark.parametrize("func", [units.pl_to_m, units.m_to_pl])
def test_handle_nan(func):
    """Check that `unit` module functions using `np.piecewise` pass NaN values through."""
    x = np.array([100, 200, np.nan, 300], dtype=np.float64)
    y = func(x)
    assert isinstance(y, np.ndarray)
    np.testing.assert_array_equal(np.isfinite(y), [True, True, False, True])
    assert np.isnan(x[2])
    assert np.isnan(y[2])
