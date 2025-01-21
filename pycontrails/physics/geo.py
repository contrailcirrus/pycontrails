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


def segment_haversine(
    longitude: npt.NDArray[np.floating], latitude: npt.NDArray[np.floating]
) -> npt.NDArray[np.floating]:
    r"""Calculate haversine distance between consecutive points along path.

    Parameters
    ----------
    longitude : npt.NDArray[np.floating]
        1D Longitude values with index corresponding to latitude inputs, [:math:`\deg`]
    latitude : npt.NDArray[np.floating]
        1D Latitude values with index corresponding to longitude inputs, [:math:`\deg`]

    Returns
    -------
    npt.NDArray[np.floating]
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
    azimuth_: npt.NDArray[np.floating], latitude: npt.NDArray[np.floating]
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    r"""Calculate rectangular direction from spherical azimuth.

    This implementation uses the equation

    ``cos(latitude) / tan(azimuth) = sin_a / cos_a``

    to solve for `sin_a` and `cos_a`.

    Parameters
    ----------
    azimuth_ : npt.NDArray[np.floating]
        Angle measured clockwise from true north, [:math:`\deg`]
    latitude : npt.NDArray[np.floating]
        Latitude value of the point, [:math:`\deg`]

    Returns
    -------
    tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]
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
    lons0: npt.NDArray[np.floating],
    lats0: npt.NDArray[np.floating],
    lons1: npt.NDArray[np.floating],
    lats1: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    r"""Calculate angle relative to true north for set of coordinates.

    Parameters
    ----------
    lons0 : npt.NDArray[np.floating]
        Longitude values of initial endpoints, [:math:`\deg`].
    lats0 : npt.NDArray[np.floating]
        Latitude values of initial endpoints, [:math:`\deg`].
    lons1 : npt.NDArray[np.floating]
        Longitude values of terminal endpoints, [:math:`\deg`].
    lats1 : npt.NDArray[np.floating]
        Latitude values of terminal endpoints, [:math:`\deg`].

    References
    ----------
    - :cite:`wikipediacontributorsAzimuth2023`

    Returns
    -------
    npt.NDArray[np.floating]
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


def segment_azimuth(
    longitude: npt.NDArray[np.floating], latitude: npt.NDArray[np.floating]
) -> npt.NDArray[np.floating]:
    r"""Calculate the angle between coordinate segments and true north.

    `np.nan` is added to the final value so the length of the output is the same as the inputs.

    Parameters
    ----------
    longitude : npt.NDArray[np.floating]
        Longitude values, [:math:`\deg`]
    latitude : npt.NDArray[np.floating]
        Latitude values, [:math:`\deg`]

    Returns
    -------
    npt.NDArray[np.floating]
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
    lons0: npt.NDArray[np.floating],
    lats0: npt.NDArray[np.floating],
    lons1: npt.NDArray[np.floating],
    lats1: npt.NDArray[np.floating],
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    r"""Calculate angle with longitudinal axis for sequence of segments.

    Parameters
    ----------
    lons0 : npt.NDArray[np.floating]
        Longitude values of initial endpoints, [:math:`\deg`].
    lats0 : npt.NDArray[np.floating]
        Latitude values of initial endpoints, [:math:`\deg`].
    lons1 : npt.NDArray[np.floating]
        Longitude values of terminal endpoints, [:math:`\deg`].
    lats1 : npt.NDArray[np.floating]
        Latitude values of terminal endpoints, [:math:`\deg`].

    References
    ----------
    - :cite:`wikipediacontributorsAzimuth2023`

    Returns
    -------
    sin_a : npt.NDArray[np.floating]
        Sine values.
    cos_a : npt.NDArray[np.floating]
        Cosine values.
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


def segment_angle(
    longitude: npt.NDArray[np.floating], latitude: npt.NDArray[np.floating]
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    r"""Calculate the angle between coordinate segments and the longitudinal axis.

    `np.nan` is added to the final value so the length of the output is the same as the inputs.

    Parameters
    ----------
    longitude : npt.NDArray[np.floating]
        Longitude values, [:math:`\deg`]
    latitude : npt.NDArray[np.floating]
        Latitude values, [:math:`\deg`]

    Returns
    -------
    tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]
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


def segment_length(
    longitude: npt.NDArray[np.floating],
    latitude: npt.NDArray[np.floating],
    altitude: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    r"""Calculate the segment length between coordinates by assuming a great circle distance.

    Requires coordinates to be in EPSG:4326.
    Lengths are calculated using both horizontal and vertical displacement of segments.

    `np.nan` is added to the final value so the length of the output is the same as the inputs.

    Parameters
    ----------
    longitude : npt.NDArray[np.floating]
        Longitude values, [:math:`\deg`]
    latitude : npt.NDArray[np.floating]
        Latitude values, [:math:`\deg`]
    altitude : npt.NDArray[np.floating]
        Altitude values, [:math:`m`]

    Returns
    -------
    npt.NDArray[np.floating]
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
    lons: npt.NDArray[np.floating],
    lats: npt.NDArray[np.floating],
    az: npt.NDArray[np.floating] | float,
    dist: npt.NDArray[np.floating] | float,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    r"""Calculate coordinates along forward azimuth.

    This function is identical to the `pyproj.Geod.fwd` method when working on
    a spherical earth. Both signatures are also identical. This implementation
    is generally more performant.

    Parameters
    ----------
    lons : npt.NDArray[np.floating]
        Array of longitude values.
    lats : npt.NDArray[np.floating]
        Array of latitude values.
    az : npt.NDArray[np.floating] | float
        Azimuth, measured in [:math:`\deg`].
    dist : npt.NDArray[np.floating] | float
        Distance [:math:`m`] between initial longitude latitude values and
        point to be computed.

    Returns
    -------
    tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]
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
    - :cite:`uosolarradiationmonitoringlaboratoryUOSRMLSolar2022`
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
    - :cite:`uosolarradiationmonitoringlaboratoryUOSRMLSolar2022`
    - :cite:`paltridgeRadiativeProcessesMeteorology1976`
    - :cite:`duffieSolarEngineeringThermal1991`

    Notes
    -----
    :math:`orbital_effect = (R_{av} / R)^{2}`
    where :math:`R` is the separation of Earth from the sun
    and :math:`R_{av}` is the mean separation.
    """
    orbital_effect = (
        1.00011
        + (0.034221 * np.cos(theta_rad))
        + (0.001280 * np.sin(theta_rad))
        + (0.000719 * np.cos(theta_rad * 2))
        + (0.000077 * np.sin(theta_rad * 2))
    )

    return constants.solar_constant * orbital_effect  # type: ignore[return-value]


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
    - :cite:`wikipediacontributorsSolarZenithAngle2023`

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
        0.396372  # type: ignore[return-value]
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
        0.004297  # type: ignore[return-value]
        + (0.107029 * np.cos(theta_rad))
        - (1.837877 * np.sin(theta_rad))
        - (0.837378 * np.cos(2 * theta_rad))
        - (2.340475 * np.sin(2 * theta_rad))
    )


# ---------
# Advection
# ---------


def advect_longitude(
    longitude: ArrayLike,
    latitude: ArrayLike,
    u_wind: ArrayLike,
    dt: npt.NDArray[np.timedelta64] | np.timedelta64,
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
    dt_s = units.dt_to_seconds(dt, dtype)

    distance_m = u_wind * dt_s

    new_longitude = longitude + units.m_to_longitude_distance(distance_m, latitude)
    return (new_longitude + 180.0) % 360.0 - 180.0  # wrap antimeridian


def advect_latitude(
    latitude: ArrayLike,
    v_wind: ArrayLike,
    dt: npt.NDArray[np.timedelta64] | np.timedelta64,
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
    dt_s = units.dt_to_seconds(dt, dtype)

    distance_m = v_wind * dt_s

    return latitude + units.m_to_latitude_distance(distance_m)


def advect_level(
    level: ArrayLike,
    vertical_velocity: ArrayLike,
    rho_air: ArrayLike | float,
    terminal_fall_speed: ArrayLike | float,
    dt: npt.NDArray[np.timedelta64] | np.timedelta64,
) -> ArrayLike:
    r"""Calculate the pressure level of a particle after time ``dt``.

    This function calculates the new pressure level of a particle as a result of
    vertical advection caused by the vertical velocity and terminal fall speed.

    Parameters
    ----------
    level : ArrayLike
        Pressure level, [:math:`hPa`]
    vertical_velocity : ArrayLike
        Vertical velocity, [:math:`Pa s^{-1}`]
    rho_air : ArrayLike | float
        Air density, [:math:`kg m^{-3}`]
    terminal_fall_speed : ArrayLike | float
        Terminal fall speed of the particle, [:math:`m s^{-1}`]
    dt : npt.NDArray[np.timedelta64] | np.timedelta64
        Time delta for each waypoint

    Returns
    -------
    ArrayLike
        New pressure level, [:math:`hPa`]
    """
    dt_s = units.dt_to_seconds(dt, level.dtype)
    dp_dt = vertical_velocity + rho_air * terminal_fall_speed * constants.g

    return (level * 100.0 + (dt_s * dp_dt)) / 100.0


def advect_longitude_and_latitude_near_poles(
    longitude: npt.NDArray[np.floating],
    latitude: npt.NDArray[np.floating],
    u_wind: npt.NDArray[np.floating],
    v_wind: npt.NDArray[np.floating],
    dt: npt.NDArray[np.timedelta64] | np.timedelta64,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    r"""Advect a particle near the poles.

    This function calculates the longitude and latitude of a particle after time ``dt``
    caused by advection due to wind near the poles (above 80 degrees North and South).

    Automatically wrap over the antimeridian if necessary.

    Parameters
    ----------
    longitude : npt.NDArray[np.floating]
        Original longitude, [:math:`\deg`]
    latitude : npt.NDArray[np.floating]
        Original latitude, [:math:`\deg`]
    u_wind : npt.NDArray[np.floating]
        Wind speed in the longitudinal direction, [:math:`m s^{-1}`]
    v_wind : npt.NDArray[np.floating]
        Wind speed in the latitudinal direction, [:math:`m s^{-1}`]
    dt : npt.NDArray[np.timedelta64] | np.timedelta64
        Advection timestep

    Returns
    -------
    tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]
        New longitude and latitude values, [:math:`\deg`]

    Notes
    -----
    Near the poles, the longitude and latitude is converted to a 2-D Cartesian-like coordinate
    system to avoid numerical instabilities and singularities caused by convergence of meridians.

    See Also
    --------
    advect_longitude
    advect_latitude
    advect_horizontal
    """
    # Determine hemisphere sign (1 for Northern Hemisphere, -1 for Southern Hemisphere)
    hemisphere_sign = np.where(latitude > 0.0, 1.0, -1.0)

    # Convert longitude and latitude to radians
    sin_lon_rad = np.sin(units.degrees_to_radians(longitude))
    cos_lon_rad = np.cos(units.degrees_to_radians(longitude))

    # Convert longitude and latitude to 2-D Cartesian-like coordinate system, [:math:`\deg`]
    polar_radius = 90.0 - np.abs(latitude)
    x_cartesian = sin_lon_rad * polar_radius
    y_cartesian = -cos_lon_rad * polar_radius * hemisphere_sign

    # Convert winds from eastward and northward direction (u, v) to (X, Y), [:math:`\deg s^{-1}`]
    x_wind = units.radians_to_degrees(
        (u_wind * cos_lon_rad - v_wind * sin_lon_rad * hemisphere_sign) / constants.radius_earth
    )
    y_wind = units.radians_to_degrees(
        (u_wind * sin_lon_rad * hemisphere_sign + v_wind * cos_lon_rad) / constants.radius_earth
    )

    # Advect contrails in 2-D Cartesian-like plane, [:math:`\deg`]
    dtype = np.result_type(latitude, v_wind)
    dt_s = units.dt_to_seconds(dt, dtype)
    x_cartesian_new = x_cartesian + dt_s * x_wind
    y_cartesian_new = y_cartesian + dt_s * y_wind

    # Convert `y_cartesian_new` back to `latitude`, [:math:`\deg`]
    dist_squared = x_cartesian_new**2 + y_cartesian_new**2
    new_latitude = (90.0 - np.sqrt(dist_squared)) * hemisphere_sign

    # Convert `x_cartesian_new` back to `longitude`, [:math:`\deg`]
    new_lon_rad = np.arctan2(y_cartesian_new, x_cartesian_new)

    new_longitude = np.where(
        (x_wind == 0.0) & (y_wind == 0.0),
        longitude,
        90.0 + units.radians_to_degrees(new_lon_rad) * hemisphere_sign,
    )
    # new_longitude = 90.0 + units.radians_to_degrees(new_lon_rad) * hemisphere_sign
    new_longitude = (new_longitude + 180.0) % 360.0 - 180.0  # wrap antimeridian
    return new_longitude, new_latitude


def advect_horizontal(
    longitude: npt.NDArray[np.floating],
    latitude: npt.NDArray[np.floating],
    u_wind: npt.NDArray[np.floating],
    v_wind: npt.NDArray[np.floating],
    dt: npt.NDArray[np.timedelta64] | np.timedelta64,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    r"""Advect a particle in the horizontal plane.

    This function calls :func:`advect_longitude` and :func:`advect_latitude` when
    the position is far from the poles (<= 80.0 degrees). When the position is near
    the poles (> 80.0 degrees), :func:`advect_longitude_and_latitude_near_poles`
    is used instead.

    Parameters
    ----------
    longitude : npt.NDArray[np.floating]
        Original longitude, [:math:`\deg`]
    latitude : npt.NDArray[np.floating]
        Original latitude, [:math:`\deg`]
    u_wind : npt.NDArray[np.floating]
        Wind speed in the longitudinal direction, [:math:`m s^{-1}`]
    v_wind : npt.NDArray[np.floating]
        Wind speed in the latitudinal direction, [:math:`m s^{-1}`]
    dt : npt.NDArray[np.timedelta64] | np.timedelta64
        Advection timestep

    Returns
    -------
    tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]
        New longitude and latitude values, [:math:`\deg`]
    """
    near_poles = np.abs(latitude) > 80.0

    longitude_out = np.empty_like(longitude)
    latitude_out = np.empty_like(latitude)

    # Use simple spherical advection if position is far from the poles (<= 80.0 degrees)
    cond = ~near_poles
    lon_cond = longitude[cond]
    lat_cond = latitude[cond]
    u_wind_cond = u_wind[cond]
    v_wind_cond = v_wind[cond]
    dt_cond = dt if isinstance(dt, np.timedelta64) else dt[cond]
    longitude_out[cond] = advect_longitude(lon_cond, lat_cond, u_wind_cond, dt_cond)
    latitude_out[cond] = advect_latitude(lat_cond, v_wind_cond, dt_cond)

    # And use Cartesian-like advection if position is near the poles (> 80.0 degrees)
    cond = near_poles
    lon_cond = longitude[cond]
    lat_cond = latitude[cond]
    u_wind_cond = u_wind[cond]
    v_wind_cond = v_wind[cond]
    dt_cond = dt if isinstance(dt, np.timedelta64) else dt[cond]
    lon_out_cond, lat_out_cond = advect_longitude_and_latitude_near_poles(
        lon_cond, lat_cond, u_wind_cond, v_wind_cond, dt_cond
    )
    longitude_out[cond] = lon_out_cond
    latitude_out[cond] = lat_out_cond

    return longitude_out, latitude_out


# ---------------
# Grid properties
# ---------------


def spatial_bounding_box(
    longitude: npt.NDArray[np.floating], latitude: npt.NDArray[np.floating], buffer: float = 1.0
) -> tuple[float, float, float, float]:
    r"""
    Construct rectangular spatial bounding box from a set of waypoints.

    Parameters
    ----------
    longitude : np.ndarray
        1D Longitude values with index corresponding to longitude inputs, [:math:`\deg`]
    latitude : np.ndarray
        1D Latitude values with index corresponding to latitude inputs, [:math:`\deg`]
    buffer: float
        Add buffer to rectangular spatial bounding box, [:math:`\deg`]

    Returns
    -------
    tuple[float, float, float, float]
        Spatial bounding box, ``(lon_min, lat_min, lon_max, lat_max)``, [:math:`\deg`]

    Examples
    --------
    >>> rng = np.random.default_rng(654321)
    >>> lon = rng.uniform(-180, 180, size=30)
    >>> lat = rng.uniform(-90, 90, size=30)
    >>> spatial_bounding_box(lon, lat)
    (np.float64(-168.0), np.float64(-77.0), np.float64(155.0), np.float64(82.0))
    """
    lon_min = max(np.floor(np.min(longitude) - buffer), -180.0)
    lon_max = min(np.ceil(np.max(longitude) + buffer), 179.99)
    lat_min = max(np.floor(np.min(latitude) - buffer), -90.0)
    lat_max = min(np.ceil(np.max(latitude) + buffer), 90.0)
    return lon_min, lat_min, lon_max, lat_max


def domain_surface_area(
    spatial_bbox: tuple[float, float, float, float] = (-180.0, -90.0, 180.0, 90.0),
    spatial_grid_res: float = 0.5,
) -> float:
    r"""
    Calculate surface area in the provided spatial bounding box.

    Parameters
    ----------
    spatial_bbox : tuple[float, float, float, float]
        Spatial bounding box, ``(lon_min, lat_min, lon_max, lat_max)``, [:math:`\deg`]
    spatial_grid_res : float
        Spatial grid resolution, [:math:`\deg`]

    Returns
    -------
    float
        Domain surface area, [:math:`m^{2}`]
    """
    assert spatial_grid_res > 0.01
    west, south, east, north = spatial_bbox
    longitude = np.arange(west, east + 0.01, spatial_grid_res)
    latitude = np.arange(south, north + 0.01, spatial_grid_res)

    da_surface_area = grid_surface_area(longitude, latitude)
    return np.nansum(da_surface_area)


def grid_surface_area(
    longitude: npt.NDArray[np.floating], latitude: npt.NDArray[np.floating]
) -> xr.DataArray:
    r"""
    Calculate surface area that is covered by each pixel in a longitude-latitude grid.

    Parameters
    ----------
    longitude: npt.NDArray[np.floating]
        Longitude coordinates in a longitude-latitude grid, [:math:`\deg`].
        Must be in ascending order.
    latitude: npt.NDArray[np.floating]
        Latitude coordinates in a longitude-latitude grid, [:math:`\deg`].
        Must be in ascending order.

    Returns
    -------
    xr.DataArray
        Surface area of each pixel in a longitude-latitude grid, [:math:`m^{2}`]

    References
    ----------
    - https://www.pmel.noaa.gov/maillists/tmap/ferret_users/fu_2004/msg00023.html
    """
    # Ensure that grid spacing is uniform
    d_lon = np.diff(longitude)
    d_lon0 = d_lon[0]
    if np.any(d_lon != d_lon0):
        raise ValueError("Longitude grid spacing is not uniform.")

    d_lat = np.diff(latitude)
    d_lat0 = d_lat[0]
    if np.all(d_lat != d_lat[0]):
        raise ValueError("Latitude grid spacing is not uniform.")

    _, lat_2d = np.meshgrid(longitude, latitude)

    area_lat_btm = _area_between_latitude_and_north_pole(lat_2d - d_lat0)
    area_lat_top = _area_between_latitude_and_north_pole(lat_2d)

    area = (d_lon0 / 360.0) * (area_lat_btm - area_lat_top)
    area[area < 0.0] = np.nan  # Prevent negative values at -90 degree latitude slice

    return xr.DataArray(area.T, coords={"longitude": longitude, "latitude": latitude})


def _area_between_latitude_and_north_pole(
    latitude: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    r"""
    Calculate surface area from the provided latitude to the North Pole.

    Parameters
    ----------
    latitude: npt.NDArray[np.floating]
        1D Latitude values with index corresponding to latitude inputs, [:math:`\deg`]

    Returns
    -------
    npt.NDArray[np.floating]
        Surface area from latitude to North Pole, [:math:`m^{2}`]
    """
    lat_radians = units.degrees_to_radians(latitude)
    return 2.0 * np.pi * constants.radius_earth**2 * (1.0 - np.sin(lat_radians))
