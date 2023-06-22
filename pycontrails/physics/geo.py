"""Tools for spherical geometry, solar radiation, and wind advection."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import xarray as xr

from pycontrails.physics import constants, units
from pycontrails.utils.types import ArrayLike

# ------------------
# Spherical Geometry
# ------------------


def haversine(lons0: ArrayLike, lats0: ArrayLike, lons1: ArrayLike, lats1: ArrayLike) -> ArrayLike:
    r"""Calculate haversine distance between points in (lons0, lats0) and (lons1, lats1).

    Handles coordinates crossing the antimeridian line (-180, 180).

    Parameters
    ----------
    lons0, lats0 : ArrayLike
        Coordinates of initial points, [:math:`\deg`]
    lons1, lats1 : ArrayLike
        Coordinates of terminal points, [:math:`\deg`]

    Returns
    -------
    ArrayLike
        Distances between corresponding points. [:math:`m`]

    Notes
    -----
    This formula does not take into account the non-spheroidal (ellipsoidal) shape of the Earth.
    Originally referenced from https://andrew.hedges.name/experiments/haversine/.

    References
    ----------
    - :cite:`CalculateDistanceBearing`

    See Also
    --------
    :func:`sklearn.metrics.pairwise.haversine_distances`:
        Compute the Haversine distance
    :class:`pyproj.Geod`:
        Performs forward and inverse geodetic, or Great Circle, computations
    """
    lats0_rad = units.degrees_to_radians(lats0)
    lats1_rad = units.degrees_to_radians(lats1)

    cos_lats0 = np.cos(lats0_rad)
    cos_lats1 = np.cos(lats1_rad)

    d_lons = units.degrees_to_radians(lons1) - units.degrees_to_radians(lons0)
    d_lats = lats1_rad - lats0_rad

    a = (np.sin(d_lats / 2.0)) ** 2 + cos_lats0 * cos_lats1 * ((np.sin(d_lons / 2.0)) ** 2)
    cc = 2.0 * np.arctan2(a**0.5, (1.0 - a) ** 0.5)
    return constants.radius_earth * cc


def segment_haversine(longitude: np.ndarray, latitude: np.ndarray) -> np.ndarray:
    r"""Calculate haversine distance between consecutive points along path.

    Parameters
    ----------
    longitude : np.ndarray
        1D Longitude values with index corresponding to latitude inputs, [:math:`\deg`]
    latitude : np.ndarray
        1D Latitude values with index corresponding to longitude inputs, [:math:`\deg`]

    Returns
    -------
    np.ndarray
        Haversine distance between (lat_i, lon_i) and (lat_i+1, lon_i+1), [:math:`m`]
        The final entry of the output is set to nan.

    See Also
    --------
    :meth:`pyproj.Geod.line_lengths`
    """
    dtype = np.result_type(longitude, latitude, np.float32)
    dist = np.empty(longitude.size, dtype=dtype)

    lons0 = longitude[:-1]
    lons1 = longitude[1:]
    lats0 = latitude[:-1]
    lats1 = latitude[1:]

    dist[:-1] = haversine(lons0, lats0, lons1, lats1)
    dist[-1] = np.nan
    return dist


def azimuth_to_direction(
    azimuth_: np.ndarray, latitude: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    r"""Calculate rectangular direction from spherical azimuth.

    This implementation uses the equation

    ``cos(latitude) / tan(azimuth) = sin_a / cos_a``

    to solve for `sin_a` and `cos_a`.

    Parameters
    ----------
    azimuth_ : np.ndarray
        Angle measured clockwise from true north, [:math:`\deg`]
    latitude : np.ndarray
        Latitude value of the point, [:math:`\deg`]

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple of sine and cosine values.
    """
    cos_lat = np.cos(units.degrees_to_radians(latitude))
    tan_az = np.tan(units.degrees_to_radians(azimuth_))

    num = cos_lat
    denom = tan_az
    mag = np.sqrt(num**2 + denom**2)

    # For azimuth in [0, 90) and (270, 360], sin_a positive
    sign_sin_a = np.where((azimuth_ - 90.0) % 360.0 - 180.0 >= 0.0, 1.0, -1.0)

    # For azimuth in [0, 180), cos_a positive
    sign_cos_a = np.where(azimuth_ % 360.0 - 180.0 <= 0.0, 1.0, -1.0)

    sin_a = sign_sin_a * np.abs(num) / mag
    cos_a = sign_cos_a * np.abs(denom) / mag
    return sin_a, cos_a


def azimuth(
    lons0: np.ndarray,
    lats0: np.ndarray,
    lons1: np.ndarray,
    lats1: np.ndarray,
) -> np.ndarray:
    r"""Calculate angle relative to true north for set of coordinates.

    Parameters
    ----------
    lons0 : np.ndarray
        Longitude values of initial endpoints, [:math:`\deg`].
    lats0 : np.ndarray
        Latitude values of initial endpoints, [:math:`\deg`].
    lons1 : np.ndarray
        Longitude values of terminal endpoints, [:math:`\deg`].
    lats1 : np.ndarray
        Latitude values of terminal endpoints, [:math:`\deg`].

    References
    ----------
    - :cite:`wikipediacontributorsAzimuth2023`

    Returns
    -------
    np.ndarray
        Azimuth relative to true north (:math:`0\deg`), [:math:`\deg`]

    See Also
    --------
    :func:`longitudinal_angle`
    """
    lons0 = units.degrees_to_radians(lons0)
    lons1 = units.degrees_to_radians(lons1)
    lats0 = units.degrees_to_radians(lats0)
    lats1 = units.degrees_to_radians(lats1)
    d_lon = lons1 - lons0

    num = np.sin(d_lon)
    denom = np.cos(lats0) * np.tan(lats1) - np.sin(lats0) * np.cos(d_lon)

    # outputs on [-180, 180] range
    alpha = units.radians_to_degrees(np.arctan2(num, denom))

    # return on [0, 360)
    return alpha % 360.0


def segment_azimuth(longitude: np.ndarray, latitude: np.ndarray) -> np.ndarray:
    r"""Calculate the angle between coordinate segments and true north.

    `np.nan` is added to the final value so the length of the output is the same as the inputs.

    Parameters
    ----------
    longitude : np.ndarray
        Longitude values, [:math:`\deg`]
    latitude : np.ndarray
        Latitude values, [:math:`\deg`]

    Returns
    -------
    np.ndarray
        Azimuth relative to true north (:math:`0\deg`), [:math:`\deg`]
        Final entry of each array is set to `np.nan`.

    References
    ----------
    - :cite:`wikipediacontributorsAzimuth2023`

    See Also
    --------
    :func:`azimuth`
    """
    dtype = np.result_type(longitude, latitude, np.float32)
    az = np.empty(longitude.size, dtype=dtype)

    lons0 = longitude[:-1]
    lons1 = longitude[1:]
    lats0 = latitude[:-1]
    lats1 = latitude[1:]

    az[:-1] = azimuth(lons0, lats0, lons1, lats1)
    az[-1] = np.nan
    return az


def longitudinal_angle(
    lons0: np.ndarray,
    lats0: np.ndarray,
    lons1: np.ndarray,
    lats1: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Calculate angle with longitudinal axis for sequence of segments.

    Parameters
    ----------
    lons0 : np.ndarray
        Longitude values of initial endpoints, [:math:`\deg`].
    lats0 : np.ndarray
        Latitude values of initial endpoints, [:math:`\deg`].
    lons1 : np.ndarray
        Longitude values of terminal endpoints, [:math:`\deg`].
    lats1 : np.ndarray
        Latitude values of terminal endpoints, [:math:`\deg`].

    References
    ----------
    - :cite:`wikipediacontributorsAzimuth2023`

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Sine, cosine values.
    """
    lons0 = units.degrees_to_radians(lons0)
    lons1 = units.degrees_to_radians(lons1)
    lats0 = units.degrees_to_radians(lats0)
    lats1 = units.degrees_to_radians(lats1)
    d_lon = lons1 - lons0

    num = np.sin(d_lon)
    denom = np.cos(lats0) * np.tan(lats1) - np.sin(lats0) * np.cos(d_lon)
    mag = np.sqrt(num**2 + denom**2)

    where = mag > 0.0
    out = np.full_like(mag, np.nan)

    sin_a = np.divide(denom, mag, out=out.copy(), where=where)
    cos_a = np.divide(num, mag, out=out.copy(), where=where)
    return sin_a, cos_a


def segment_angle(longitude: np.ndarray, latitude: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r"""Calculate the angle between coordinate segments and the longitudinal axis.

    `np.nan` is added to the final value so the length of the output is the same as the inputs.

    Parameters
    ----------
    longitude : np.ndarray
        Longitude values, [:math:`\deg`]
    latitude : np.ndarray
        Latitude values, [:math:`\deg`]

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        sin(a), cos(a), where ``a`` is the angle between the segment and the longitudinal axis.
        Final entry of each array is set to `np.nan`.

    References
    ----------
    - :cite:`wikipediacontributorsAzimuth2023`

    Notes
    -----
    ::

                (lon_2, lat_2)  X
                               /|
                              / |
                             /  |
                            /   |
                           /    |
                          /     |
                         /      |
        (lon_1, lat_1)  X -------> longitude (x-axis)

    See Also
    --------
    :func:`longitudinal_angle`
    """
    dtype = np.result_type(longitude, latitude, np.float32)
    sin_a = np.empty(longitude.size, dtype=dtype)
    cos_a = np.empty(longitude.size, dtype=dtype)

    lons0 = longitude[:-1]
    lons1 = longitude[1:]
    lats0 = latitude[:-1]
    lats1 = latitude[1:]

    sin_a[:-1], cos_a[:-1] = longitudinal_angle(lons0, lats0, lons1, lats1)
    sin_a[-1] = np.nan
    cos_a[-1] = np.nan
    return sin_a, cos_a


def segment_length(longitude: np.ndarray, latitude: np.ndarray, altitude: np.ndarray) -> np.ndarray:
    r"""Calculate the segment length between coordinates by assuming a great circle distance.

    Requires coordinates to be in EPSG:4326.
    Lengths are calculated using both horizontal and vertical displacement of segments.

    `np.nan` is added to the final value so the length of the output is the same as the inputs.

    Parameters
    ----------
    longitude : np.ndarray
        Longitude values, [:math:`\deg`]
    latitude : np.ndarray
        Latitude values, [:math:`\deg`]
    altitude : np.ndarray
        Altitude values, [:math:`m`]

    Returns
    -------
    np.ndarray
        Array of distances in [:math:`m`] between coordinates.
        Final entry of each array is set to `np.nan`.

    See Also
    --------
    :func:`haversine`
    :func:`segment_haversine`
    """
    dist_horizontal = segment_haversine(longitude, latitude)
    dist_vertical = np.empty_like(altitude)
    dist_vertical[:-1] = np.diff(altitude)
    dist_vertical[-1] = np.nan  # last segment is set to nan
    return (dist_horizontal**2 + dist_vertical**2) ** 0.5


def forward_azimuth(
    lons: np.ndarray, lats: np.ndarray, az: np.ndarray | float, dist: np.ndarray | float
) -> tuple[np.ndarray, np.ndarray]:
    r"""Calculate coordinates along forward azimuth.

    This function is identical to the `pyproj.Geod.fwd` method when working on
    a spherical earth. Both signatures are also identical. This implementation
    is generally more performant.

    Parameters
    ----------
    lons : np.ndarray
        Array of longitude values.
    lats : np.ndarray
        Array of latitude values.
    az : np.ndarray | float
        Azimuth, measured in [:math:`\deg`].
    dist : np.ndarray | float
        Distance [:math:`m`] between initial longitude latitude values and
        point to be computed.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Tuple of longitude latitude arrays.

    See Also
    --------
    :meth:pyproj.Geod.fwd
    """
    az_rad = units.degrees_to_radians(az)
    sin_az = np.sin(az_rad)
    cos_az = np.cos(az_rad)

    lats_rad = units.degrees_to_radians(lats)
    sin_lats = np.sin(lats_rad)
    cos_lats = np.cos(lats_rad)

    dist_ratio = dist / constants.radius_earth
    cos_dist_ratio = np.cos(dist_ratio)
    sin_dist_ratio = np.sin(dist_ratio)

    dest_lats_rad = np.arcsin(sin_lats * cos_dist_ratio + cos_lats * sin_dist_ratio * cos_az)
    dest_lats = units.radians_to_degrees(dest_lats_rad)

    delta_lons_rad = np.arctan2(
        sin_az * sin_dist_ratio * cos_lats,
        cos_dist_ratio - sin_lats * np.sin(dest_lats_rad),
    )
    dest_lons = lons + units.radians_to_degrees(delta_lons_rad)
    dest_lons = (dest_lons + 180.0) % 360.0 - 180.0

    return dest_lons, dest_lats


# ---------------
# Solar Radiation
# ---------------


def solar_direct_radiation(
    longitude: ArrayLike, latitude: ArrayLike, time: ArrayLike, threshold_cos_sza: float = 0.0
) -> np.ndarray:
    r"""Calculate the instantaneous theoretical solar direct radiation (SDR).

    Parameters
    ----------
    longitude : ArrayLike
        Longitude, [:math:`\deg`]
    latitude : ArrayLike
        Latitude, [:math:`\deg`]
    time : ArrayLike
        Time, formatted as :class:`np.datetime64`
    threshold_cos_sza : float, optional
        Set the SDR to 0 when the :func:`cosine_solar_zenith_angle` is below a certain value.
        By default, set to 0.

    Returns
    -------
    ArrayLike
        Solar direct radiation of incoming radiation, [:math:`W m^{-2}`]

    References
    ----------
    - :cite:`UOSRMLSolar`
    """
    theta_rad = orbital_position(time)

    # Use longitude and latitude to determine the dtype
    dtype = np.result_type(longitude, latitude)
    theta_rad = theta_rad.astype(dtype, copy=False)

    _solar_constant = solar_constant(theta_rad)
    cos_sza = cosine_solar_zenith_angle(longitude, latitude, time, theta_rad)

    # Note that np.where is more performant than xr.where, even for large arrays
    # (and especially for small arrays).
    # BUT xr.where is "safer" in the sense that it will pass numpy arrays through as if
    # they were pumped directly through np.where.
    # For now, explicitly check if we're work with xarray instances or numpy arrays
    # This will likely not work for native python numeric types
    if isinstance(cos_sza, xr.DataArray):
        return xr.where(cos_sza < threshold_cos_sza, 0.0, cos_sza * _solar_constant)
    return np.where(cos_sza < threshold_cos_sza, 0.0, cos_sza * _solar_constant)


def solar_constant(theta_rad: ArrayLike) -> ArrayLike:
    """Calculate the solar electromagnetic radiation per unit area from orbital position.

    On average, the extraterrestrial irradiance is 1367 W/m**2
    and varies by +- 3% as the Earth orbits the sun.

    Parameters
    ----------
    theta_rad : ArrayLike
        Orbital position, [:math:`rad`]. Use :func:`orbital_position` to calculate
        the orbital position from time input.

    Returns
    -------
    ArrayLike
        Solar constant, [:math:`W m^{-2}`]

    References
    ----------
    - :cite:`UOSRMLSolar`
    - :cite:`paltridgeRadiativeProcessesMeteorology1976`
    - :cite:`duffieSolarEngineeringThermal1991`

    Notes
    -----
    :math:`orbital_effect = (R_{av} / R)^{2}`
    where :math:`R` is the separation of Earth from the sun
    and :math:`R_{av}`is the mean separation.
    """
    orbital_effect = (
        1.00011
        + (0.034221 * np.cos(theta_rad))
        + (0.001280 * np.sin(theta_rad))
        + (0.000719 * np.cos(theta_rad * 2))
        + (0.000077 * np.sin(theta_rad * 2))
    )

    return constants.solar_constant * orbital_effect


def cosine_solar_zenith_angle(
    longitude: ArrayLike,
    latitude: ArrayLike,
    time: ArrayLike,
    theta_rad: ArrayLike,
) -> ArrayLike:
    r"""Calculate the cosine of the solar zenith angle.

    Return (:math:`\cos(\theta)`), where :math:`\theta` is the angle between the sun and the
    vertical direction.

    Parameters
    ----------
    longitude : ArrayLike
        Longitude, [:math:`\deg`]
    latitude : ArrayLike
        Latitude, [:math:`\deg`]
    time : ArrayLike
        Time, formatted as :class:`np.datetime64`
    theta_rad : ArrayLike
        Orbital position, [:math:`rad`]. Output of :func:`orbital_position`.

    Returns
    -------
    ArrayLike
        Cosine of the solar zenith angle

    References
    ----------
    - :cite:`wikipediacontributorsSolarZenithAngle2022`

    See Also
    --------
    :func:`orbital_position`
    :func:`solar_declination_angle`
    :func:`solar_hour_angle`
    """
    lat_rad = units.degrees_to_radians(latitude)
    sdec_rad = units.degrees_to_radians(solar_declination_angle(theta_rad))
    sha_rad = units.degrees_to_radians(solar_hour_angle(longitude, time, theta_rad))

    return np.sin(lat_rad) * np.sin(sdec_rad) + (
        np.cos(lat_rad) * np.cos(sdec_rad) * np.cos(sha_rad)
    )


def orbital_position(time: ArrayLike) -> ArrayLike:
    """Calculate the orbital position of Earth to a reference point set at the start of year.

    Parameters
    ----------
    time : ArrayLike
        ArrayLike of :class:`np.datetime64` times

    Returns
    -------
    ArrayLike
        Orbital position of Earth, [:math:`rad`]
    """
    dt_day = days_since_reference_year(time)
    theta = 360.0 * (dt_day / 365.25)
    return units.degrees_to_radians(theta)


def days_since_reference_year(time: ArrayLike, ref_year: int = 2000) -> ArrayLike:
    """Calculate the days elapsed since the start of the reference year.

    Parameters
    ----------
    time : ArrayLike
        ArrayLike of :class:`np.datetime64` times
    ref_year : int, optional
        Year of reference

    Returns
    -------
    ArrayLike
        Days elapsed since the reference year. Output ``dtype`` is ``np.float64``.

    Raises
    ------
    RuntimeError
        Raises when reference year is greater than the time of `time` element
    """
    date_start = np.datetime64(ref_year - 1970, "Y")
    dt_day = (time - date_start) / np.timedelta64(1, "D")

    if np.any(dt_day < 0.0):
        raise RuntimeError(
            f"Reference year {ref_year} is greater than the time of one or more waypoints."
        )

    return dt_day


def hours_since_start_of_day(time: ArrayLike) -> ArrayLike:
    """Calculate the hours elapsed since the start of day (00:00:00 UTC).

    Parameters
    ----------
    time : ArrayLike
        ArrayLike of :class:`np.datetime64` times

    Returns
    -------
    ArrayLike
        Hours elapsed since the start of today day. Output ``dtype`` is ``np.float64``.
    """
    return (time - time.astype("datetime64[D]")) / np.timedelta64(1, "h")


def solar_declination_angle(theta_rad: ArrayLike) -> ArrayLike:
    r"""Calculate the solar declination angle from the orbital position in radians (theta_rad).

    The solar declination angle is the angle between the rays of the Sun and the plane of the
    Earth's equator.

    It has a range of between -23.5 (winter solstice) and +23.5 (summer solstice) degrees.

    Parameters
    ----------
    theta_rad : ArrayLike
        Orbital position, [:math:`rad`]. Output of :func:`orbital_position`.

    Returns
    -------
    ArrayLike
        Solar declination angle, [:math:`\deg`]

    References
    ----------
    - :cite:`paltridgeRadiativeProcessesMeteorology1976`

    Notes
    -----
    Tested against :cite:`noaaSolarCalculationDetails`

    See Also
    --------
    :func:`orbital_position`
    :func:`cosine_solar_zenith_angle`
    """
    return (
        0.396372
        - (22.91327 * np.cos(theta_rad))
        + (4.02543 * np.sin(theta_rad))
        - (0.387205 * np.cos(2 * theta_rad))
        + (0.051967 * np.sin(2 * theta_rad))
        - (0.154527 * np.cos(3 * theta_rad))
        + (0.084798 * np.sin(3 * theta_rad))
    )


def solar_hour_angle(longitude: ArrayLike, time: ArrayLike, theta_rad: ArrayLike) -> ArrayLike:
    r"""Calculate the sun's East to West angular displacement around the polar axis.

    The solar hour angle is an expression of time in angular measurements:
    the value of the hour angle is zero at noon,
    negative in the morning, and positive in the afternoon, increasing by 15 degrees per hour.

    Parameters
    ----------
    longitude : ArrayLike
        Longitude, [:math:`\deg`]
    time : ArrayLike
        ArrayLike of :class:`np.datetime64` times
    theta_rad : ArrayLike
        Orbital position, [:math:`rad`]. Output of :func:`orbital_position`.

    Returns
    -------
    ArrayLike
        Solar hour angle, [:math:`\deg`]

    See Also
    --------
    :func:`orbital_position`
    :func:`cosine_solar_zenith_angle`
    :func:`orbital_correction_for_solar_hour_angle`
    """
    # Let the two float-like arrays dictate the dtype of the time conversion
    dtype = np.result_type(longitude, theta_rad)
    dt_hour = hours_since_start_of_day(time).astype(dtype)

    orbital_correction = orbital_correction_for_solar_hour_angle(theta_rad)
    return ((dt_hour - 12) * 15) + longitude + orbital_correction


def orbital_correction_for_solar_hour_angle(theta_rad: ArrayLike) -> ArrayLike:
    r"""Calculate correction to the solar hour angle due to Earth's orbital location.

    Parameters
    ----------
    theta_rad : ArrayLike
        Orbital position, [:math:`rad`]

    Returns
    -------
    ArrayLike
        Correction to the solar hour angle as a result of Earth's orbital location, [:math:`\deg`]

    References
    ----------
    - :cite:`paltridgeRadiativeProcessesMeteorology1976`

    Notes
    -----
    Tested against :cite:`noaaSolarCalculationDetails`
    """
    return (
        0.004297
        + (0.107029 * np.cos(theta_rad))
        - (1.837877 * np.sin(theta_rad))
        - (0.837378 * np.cos(2 * theta_rad))
        - (2.340475 * np.sin(2 * theta_rad))
    )


# ---------
# Advection
# ---------


def advect_longitude(
    longitude: ArrayLike, latitude: ArrayLike, u_wind: ArrayLike, dt: np.ndarray | np.timedelta64
) -> ArrayLike:
    r"""Calculate the longitude of a particle after time `dt` caused by advection due to wind.

    Automatically wrap over the antimeridian if necessary.

    Parameters
    ----------
    longitude : ArrayLike
        Original longitude, [:math:`\deg`]
    latitude : ArrayLike
        Original latitude, [:math:`\deg`]
    u_wind : ArrayLike
        Wind speed in the longitudinal direction, [:math:`m s^{-1}`]
    dt : np.ndarray
        Advection timestep

    Returns
    -------
    ArrayLike
        New longitude value, [:math:`\deg`]
    """
    # Use the same dtype as longitude, latitude, and u_wind
    dtype = np.result_type(longitude, latitude, u_wind)
    dt_s = _dt_to_float_seconds(dt, dtype)

    distance_m = u_wind * dt_s

    new_longitude = longitude + units.m_to_longitude_distance(distance_m, latitude)
    return (new_longitude + 180.0) % 360.0 - 180.0  # wrap antimeridian


def advect_latitude(
    latitude: ArrayLike, v_wind: ArrayLike, dt: np.ndarray | np.timedelta64
) -> ArrayLike:
    r"""Calculate the latitude of a particle after time ``dt`` caused by advection due to wind.

    .. note::

        It is possible for advected latitude values to lie outside of the WGS84 domain
        ``[-90, 90]``. In :class:`Cocip` models, latitude values close to the poles
        create an end of life condition, thereby avoiding this issue. In practice,
        such situations are very rare.

        These polar divergence issues could also be addressed by reflecting the
        longitude values 180 degrees via a spherical equivalence such as
        ``(lon, lat) ~ (lon + 180, 180 - lat)``. This approach is not currently taken.

    Parameters
    ----------
    latitude : ArrayLike
        Original latitude, [:math:`\deg`]
    v_wind : ArrayLike
        Wind speed in the latitudinal direction, [:math:`m s^{-1}`]
    dt : np.ndarray
        Advection time delta

    Returns
    -------
    ArrayLike
        New latitude value, [:math:`\deg`]
    """
    # Use the same dtype as latitude and v_wind
    dtype = np.result_type(latitude, v_wind)
    dt_s = _dt_to_float_seconds(dt, dtype)

    distance_m = v_wind * dt_s

    return latitude + units.m_to_latitude_distance(distance_m)


def advect_level(
    level: ArrayLike,
    vertical_velocity: ArrayLike,
    rho_air: ArrayLike,
    terminal_fall_speed: ArrayLike,
    dt: np.ndarray | np.timedelta64,
) -> ArrayLike:
    r"""Calculate the pressure level of a particle after time ``dt``.

    This function calculates the new pressure level of a particle as a result of
    vertical advection caused by the vertical velocity and terminal fall speed.

    Parameters
    ----------
    level : ArrayLike
        Pressure level, [:math:`hPa`]
    vertical_velocity : ArrayLike
        Vertical velocity, [:math:`m s^{-1}`]
    rho_air : ArrayLike
        Air density, [:math:`kg m^{-3}`]
    terminal_fall_speed : ArrayLike
        Terminal fall speed of the particle, [:math:`m s^{-1}`]
    dt : np.ndarray
        Time delta for each waypoint

    Returns
    -------
    ArrayLike
        New pressure level, [:math:`hPa`]
    """
    dt_s = _dt_to_float_seconds(dt, level.dtype)
    velocity = vertical_velocity + rho_air * terminal_fall_speed * constants.g

    return (level * 100.0 + (dt_s * velocity)) / 100.0


def _dt_to_float_seconds(dt: np.ndarray | np.timedelta64, dtype: npt.DTypeLike) -> np.ndarray:
    """Convert a time delta to seconds as a float with specified ``dtype`` precision.

    Parameters
    ----------
    dt : np.ndarray
        Time delta for each waypoint
    dtype : np.dtype
        Data type of the output array

    Returns
    -------
    np.ndarray
        Time delta in seconds as a float
    """
    out = np.empty(dt.shape, dtype=dtype)
    np.divide(dt, np.timedelta64(1, "s"), out=out)
    return out
