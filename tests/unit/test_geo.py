"""Test geo module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyproj
import pytest

from pycontrails.physics import constants, geo

from .conftest import get_static_path


@pytest.fixture
def rand_geo_data():
    rng = np.random.default_rng(1234)
    n_pts = 100000
    lons = -180 + 360 * rng.random(n_pts)
    lats = -90 + 180 * rng.random(n_pts)
    az = 360 * rng.random(n_pts)
    dist = 100000 * (rng.random(n_pts) - 0.5)  # can be negative
    return {"lons": lons, "lats": lats, "az": az, "dist": dist}


@pytest.fixture
def geod() -> pyproj.Geod:
    return pyproj.Geod(a=constants.radius_earth)


def test_haversine(geod: pyproj.Geod, rand_geo_data: dict[str, np.ndarray]):
    """Check that `geo.segment_haversine` agrees with `pyproj` implementation."""
    lons = rand_geo_data["lons"]
    lats = rand_geo_data["lats"]

    pyproj_lengths = geod.line_lengths(lons, lats)

    # our version of haversine tacks an extra nan onto the end
    pycont_lengths = geo.segment_haversine(lons, lats)
    assert np.isnan(pycont_lengths[-1])
    np.testing.assert_allclose(pyproj_lengths, pycont_lengths[:-1])

    # test that haversine wraps longitude
    d_meridian = geo.haversine(-180, 5, 180, 5)
    assert np.round(d_meridian, 6) == 0

    d_meridian = geo.haversine(-180, 5, 179.9, 5)
    assert np.round(d_meridian, 6) == 11077.577787

    d_meridian = geo.haversine(-179.9, 5, 180, 5)
    assert np.round(d_meridian, 6) == 11077.577787


def test_forward_azimuth(geod: pyproj.Geod, rand_geo_data: dict[str, np.ndarray]):
    """Check that `geo.forward_azimuth` agrees with `pyproj` implementation."""
    # pyproj also returns "back azimuth", which is ignored
    pyproj_lons, pyproj_lats, _ = geod.fwd(**rand_geo_data)

    pycont_lons, pycont_lats = geo.forward_azimuth(**rand_geo_data)

    np.testing.assert_allclose(pyproj_lons, pycont_lons)
    np.testing.assert_allclose(pyproj_lats, pycont_lats)


def test_azimuth_to_direction(geod: pyproj.Geod, rand_geo_data: dict[str, np.ndarray]):
    """Check that `geo.azimuth_to_direction` agrees with `pyproj` implementation."""
    longitude = rand_geo_data["lons"]
    latitude = rand_geo_data["lats"]
    azimuth = rand_geo_data["az"]
    pycont_sin_a, pycont_cos_a = geo.azimuth_to_direction(azimuth, latitude)

    dist = np.ones_like(longitude)
    fwd_lon, fwd_lat, _ = geod.fwd(longitude, latitude, azimuth, dist)
    adj = fwd_lon - longitude
    opp = fwd_lat - latitude
    hyp = np.sqrt(adj * adj + opp * opp)
    pyproj_sin_a = opp / hyp
    pyproj_cos_a = adj / hyp

    # If we use a smaller dist above, the error here would be smaller
    np.testing.assert_allclose(pycont_sin_a, pyproj_sin_a, atol=1e-4)
    np.testing.assert_allclose(pycont_cos_a, pyproj_cos_a, atol=1e-4)


def test_segment_angle(geod: pyproj.Geod):
    rng = np.random.default_rng(123)  # seed chosen to get decent coverage in random walk
    longitude = 0.4 * rng.random(100000) - 0.2
    latitude = 0.4 * rng.random(100000) - 0.2

    # longitude-latitude random walk
    longitude = np.cumsum(longitude)
    latitude = np.cumsum(latitude)

    # stays within bounds
    assert latitude.max() < 90 and latitude.min() > -90

    sin, cos = geo.segment_angle(longitude, latitude)
    assert np.isnan(cos[-1])
    assert np.isnan(sin[-1])

    angle, _, _ = geod.inv(
        lons1=longitude[:-1],
        lats1=latitude[:-1],
        lons2=longitude[1:],
        lats2=latitude[1:],
    )
    angle = np.deg2rad(90 - angle)
    sin2 = np.sin(angle)
    cos2 = np.cos(angle)

    np.testing.assert_array_almost_equal(sin[:-1], sin2, decimal=10)
    np.testing.assert_array_almost_equal(cos[:-1], cos2, decimal=10)

    sin3, cos3 = deprecated_segment_angle(longitude, latitude)

    # Depreciated implemention is not mathematically, but still agrees to
    # 2 decimal places of accuracy. Depreciated version does not use the
    # append nan convention
    np.testing.assert_array_almost_equal(sin[:-1], sin3[:-1], decimal=2)
    np.testing.assert_array_almost_equal(cos[:-1], cos3[:-1], decimal=2)


def test_segment_angle_deprecated_diverge():
    # deprecated version differs from current implementation for large segments

    coords = np.array([20, 50])
    sin, cos = geo.segment_angle(coords, coords)
    sin2, cos2 = deprecated_segment_angle(coords, coords)

    # final coordinate (convention) is different
    assert np.isnan(cos[1])
    assert np.isnan(sin[1])
    assert sin2[1] == 0
    assert cos2[1] == 1

    # first coordinate differs
    with pytest.raises(AssertionError):
        np.testing.assert_array_almost_equal(sin, sin2, decimal=2)

    with pytest.raises(AssertionError):
        np.testing.assert_array_almost_equal(cos, cos2, decimal=2)


def deprecated_segment_angle(
    longitude: np.ndarray, latitude: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    OLD VERSION OF `geo.segment_angle` PORTED FROM FORTRAN
    THIS FUNCTION NEARLY AGREES WITH CURRENT IMPLEMENTATION FOR SMALL SEGMENTS
    CURRENT IMPLEMENTATION IS MATHEMATICALLY CORRECT AND TWICE AS FAST

    Calculates the sin(a) and cos(a) for the
    angle between each segment and the longitudinal axis


                (lon_2, lat_2)  X
                               /|
                              / |
                             /  |
                            /   |
                           /    |
                          /     |
                         /      |
        (lon_1, lat_1)  X -------> longitude (x-axis)

    Returns
    -------
    np.ndarray, np.ndarray
        sin(a), cos(a), where a is the angle between the segment and the longitudinal axis

    See Also
    --------
    :func:`haversine`
    """
    from pycontrails.physics import constants, units

    lats_next = np.roll(latitude, -1)
    lats_avg = 0.5 * (latitude + lats_next)

    # append nan to ensure length is the same as the original waypoints
    d_lon = np.append(np.diff(longitude), np.nan)
    d_lat = np.append(np.diff(latitude), np.nan)

    dist = geo.segment_haversine(longitude, latitude)
    sin_a = (constants.radius_earth * units.degrees_to_radians(d_lat)) / dist
    sin_a[-1] = 0  # last element is a discontinuous segment
    assert isinstance(sin_a, np.ndarray)

    cos_a = (
        constants.radius_earth
        * units.degrees_to_radians(d_lon)
        * np.cos(units.degrees_to_radians(lats_avg))
        / dist
    )
    cos_a[-1] = 1  # last element is a discontinuous segment

    return sin_a, cos_a


def test_latitude_advection_divergence():
    """Show that it is possible for latitude values to diverge near poles.

    This test highlights polar divergence arising in a `CocipGrid` simulation.
    """
    rng = np.random.default_rng(2468)
    latitude = rng.uniform(89.3, 89.7, 1000)  # start with latitude near 89.5
    v_wind = rng.uniform(-20, 20, 1000)  # with typical wind values
    dt = np.timedelta64(30, "m")
    latitude_t2 = geo.advect_latitude(latitude=latitude, v_wind=v_wind, dt=dt)
    assert np.any(latitude_t2 > 90)  # conclude some latitude values exceed 90


def test_solar_calculations() -> None:
    """Test solar calculations against NOAA Solar Calculator.

    See NOAA_Solar_Calculations_day.xls from https://gml.noaa.gov/grad/solcalc/calcdetails.html.
    """

    noaa_solar_csv = pd.read_csv(get_static_path("NOAA_Solar_Calculations_day.csv"))

    time = (
        pd.to_datetime(noaa_solar_csv["Date"])
        + pd.to_timedelta(noaa_solar_csv["Time (past local midnight)"])
    ).to_numpy()
    latitude = np.full(time.shape, fill_value=40.0)
    longitude = np.full(time.shape, fill_value=-105.0)

    # orbital position
    theta_rad = geo.orbital_position(time)
    # TODO: comparison column?

    # orbital correction
    # orbital_correction = geo.orbital_correction_for_solar_hour_angle(theta_rad)
    # TODO: comparison column?

    # solar zenith angle
    noaa_sza = noaa_solar_csv["Solar Zenith Angle (deg)"]
    cos_sza = geo.cosine_solar_zenith_angle(longitude, latitude, time, theta_rad)
    sza = np.arccos(cos_sza) * 180 / np.pi
    np.testing.assert_allclose(sza, noaa_sza, atol=1e-1)

    # solar declination
    noaa_sda = noaa_solar_csv["Sun Declin (deg)"]
    sda = geo.solar_declination_angle(theta_rad)
    np.testing.assert_allclose(sda, noaa_sda, atol=1e-1)

    # solar hour angle
    # TODO: the value diverge near midnight
    noaa_sha = noaa_solar_csv["Hour Angle (deg)"]
    sha = geo.solar_hour_angle(longitude, time, theta_rad)
    sha[sha < -180] = sha[sha < -180] + 360
    np.testing.assert_allclose(sha, noaa_sha, atol=3e-1)


def test_azimuth(geod: pyproj.Geod, rand_geo_data: dict[str, np.ndarray]):
    """Check that `geo.azimuth` agrees with `pyproj` implementation."""

    longitude = rand_geo_data["lons"]
    latitude = rand_geo_data["lats"]

    azimuth = geo.segment_azimuth(longitude, latitude)
    assert np.isnan(azimuth[-1])

    angle, _, _ = geod.inv(
        lons1=longitude[:-1],
        lats1=latitude[:-1],
        lons2=longitude[1:],
        lats2=latitude[1:],
    )

    # reset output [0, 360)
    angle = angle % 360
    np.testing.assert_array_almost_equal(azimuth[:-1], angle, decimal=10)

    # this should be 90 degrees off from the logitudinal angle
    # so sin(azimuth) == cos(a), cos(azimuth) == sin(a)
    sin_azimuth = np.sin(np.deg2rad(azimuth))
    cos_azimuth = np.cos(np.deg2rad(azimuth))
    sin_a, cos_a = geo.segment_angle(longitude, latitude)

    np.testing.assert_array_almost_equal(sin_azimuth, cos_a, decimal=10)
    np.testing.assert_array_almost_equal(cos_azimuth, sin_a, decimal=10)
