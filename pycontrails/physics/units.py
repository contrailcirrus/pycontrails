"""Unit conversion support."""

from __future__ import annotations

import warnings

import numpy as np
import numpy.typing as npt
import xarray as xr

from pycontrails.physics import constants


def pl_to_ft[A: (np.ndarray, xr.DataArray, float)](pl: A) -> A:
    r"""Convert from pressure level (hPa) to altitude (ft).

    Assumes the ICAO standard atmosphere.

    Parameters
    ----------
    pl : A
        pressure level, [:math:`hPa`], [:math:`mbar`]

    Returns
    -------
    A
        altitude, [:math:`ft`]

    See Also
    --------
    pl_to_m
    ft_to_pl
    m_to_T_isa
    """
    return m_to_ft(pl_to_m(pl))


def ft_to_pl[A: (np.ndarray, xr.DataArray, float)](h: A) -> A:
    r"""Convert from altitude (ft) to pressure level (hPa).

    Assumes the ICAO standard atmosphere.

    Parameters
    ----------
    h : A
        altitude, [:math:`ft`]

    Returns
    -------
    A
        pressure level, [:math:`hPa`], [:math:`mbar`]

    See Also
    --------
    m_to_pl
    pl_to_ft
    m_to_T_isa
    """
    return m_to_pl(ft_to_m(h))


def kelvin_to_celsius[A: (np.ndarray, xr.DataArray, float)](kelvin: A) -> A:
    """Convert temperature from Kelvin to Celsius.

    Parameters
    ----------
    kelvin : A
        temperature [:math:`K`]

    Returns
    -------
    A
        temperature [:math:`C`]
    """
    return kelvin + constants.absolute_zero


def m_to_T_isa[A: (np.ndarray, xr.DataArray, float)](h: A) -> A:
    """Calculate the ambient temperature (K) for a given altitude (m).

    Assumes the ICAO standard atmosphere.

    Parameters
    ----------
    h : A
        altitude, [:math:`m`]

    Returns
    -------
    A
        ICAO standard atmosphere ambient temperature, [:math:`K`]


    References
    ----------
    - :cite:`wikipediacontributorsInternationalStandardAtmosphere2023`

    Notes
    -----
    See https://en.wikipedia.org/wiki/International_Standard_Atmosphere

    This implementation agrees with the ISA only up to 20000 m. A warning is emitted for
    higher altitudes.

    See Also
    --------
    m_to_pl
    ft_to_pl
    """
    if np.any(h > 20000.0):
        msg = "Altitude exceeds 20000 m, above which this implementation disagrees with the ISA."
        warnings.warn(msg, skip_file_prefixes=(__file__,))

    h_min = np.minimum(h, constants.h_tropopause)
    return constants.T_msl + h_min * constants.T_lapse_rate  # type: ignore[return-value]


_POWER_TERM = -constants.g / (constants.T_lapse_rate * constants.R_d)
_DECAY = (-constants.g / (constants.R_d * m_to_T_isa(constants.h_tropopause))).item()  # type: ignore[attr-defined]


def m_to_pl[A: (np.ndarray, xr.DataArray, float)](h: A) -> A:
    r"""Convert from altitude (m) to pressure level (hPa).

    Parameters
    ----------
    h : A
        altitude, [:math:`m`]

    Returns
    -------
    A
        pressure level, [:math:`hPa`], [:math:`mbar`]

    References
    ----------
    - :cite:`wikipediacontributorsBarometricFormula2023`

    Notes
    -----
    See https://en.wikipedia.org/wiki/Barometric_formula

    This implementation agrees with the ISA only up to 20000 m. A warning is emitted for
    higher altitudes.

    See Also
    --------
    m_to_T_isa
    ft_to_pl
    """
    T_isa = m_to_T_isa(h)
    T_ratio = T_isa / constants.T_msl
    p_isa = constants.p_surface * T_ratio**_POWER_TERM

    # Apply exponential decay term for altitudes above the tropopause, which is 1 = exp(0) below it
    excess_altitude = np.maximum(h - constants.h_tropopause, 0.0)
    decay_factor = np.exp(_DECAY * excess_altitude)

    return p_isa * decay_factor / 100.0


_PL_20KM = m_to_pl(20000.0).item()  # type: ignore[attr-defined]


def pl_to_m[A: (np.ndarray, xr.DataArray, float)](pl: A) -> A:
    r"""Convert from pressure level (hPa) to altitude (m).

    Function is slightly different from the classical formula:
    ``constants.T_msl / 0.0065) * (1 - (pl_pa / constants.p_surface) ** (1 / 5.255)``
    in order to provide a mathematical inverse to :func:`m_to_pl`.

    For low altitudes (below the tropopause), this implementation closely agrees to classical
    formula.

    Parameters
    ----------
    pl : A
        pressure level, [:math:`hPa`], [:math:`mbar`]

    Returns
    -------
    A
        altitude, [:math:`m`]

    References
    ----------
    - :cite:`wikipediacontributorsBarometricFormula2023`

    Notes
    -----
    See https://en.wikipedia.org/wiki/Barometric_formula

    This implementation agrees with the ISA only for pressure levels above the ISA pressure at
    20000 m. A warning is emitted for lower pressure levels.

    See Also
    --------
    pl_to_ft
    m_to_pl
    m_to_T_isa
    """
    if np.any(pl < _PL_20KM):
        msg = (
            f"Pressure level is below {_PL_20KM:.2f} hPa, the ISA pressure at "
            "20000 m, above which this implementation disagrees with the ISA."
        )
        warnings.warn(msg, skip_file_prefixes=(__file__,))

    pl_tropopause = m_to_pl(constants.h_tropopause).item()  # type: ignore[attr-defined]

    p_ratio = 100.0 * np.maximum(pl, pl_tropopause) / constants.p_surface
    T_isa = constants.T_msl * p_ratio ** (1.0 / _POWER_TERM)
    h_isa = (T_isa - constants.T_msl) / constants.T_lapse_rate

    # Add the altitude above the tropopause, which is 0 = log(1) below it
    excess_altitude = np.log(np.minimum(pl, pl_tropopause) / pl_tropopause) / _DECAY

    return h_isa + excess_altitude


def degrees_to_radians[A: (np.ndarray, xr.DataArray, float)](degrees: A) -> A:
    r"""Convert from degrees to radians.

    Parameters
    ----------
    degrees : A
        Degrees values, [:math:`\deg`]

    Returns
    -------
    A
        Radians values
    """
    return degrees * (np.pi / 180.0)


def radians_to_degrees[A: (np.ndarray, xr.DataArray, float)](radians: A) -> A:
    r"""Convert from radians to degrees.

    Parameters
    ----------
    radians : A
        degrees values, [:math:`\rad`]

    Returns
    -------
    A
        Radian values
    """
    return radians * (180.0 / np.pi)


def ft_to_m[A: (np.ndarray, xr.DataArray, float)](ft: A) -> A:
    """Convert length from feet to meter.

    Parameters
    ----------
    ft : A
        length, [:math:`ft`]

    Returns
    -------
    A
        length, [:math:`m`]
    """
    return ft * 0.3048


def m_to_ft[A: (np.ndarray, xr.DataArray, float)](m: A) -> A:
    """Convert length from meters to feet.

    Parameters
    ----------
    m : A
        length, [:math:`m`]

    Returns
    -------
    A
        length, [:math:`ft`]
    """
    return m / 0.3048


def m_per_s_to_knots[A: (np.ndarray, xr.DataArray, float)](m_per_s: A) -> A:
    r"""Convert speed from meters per second (m/s) to knots.

    Parameters
    ----------
    m_per_s : A
        Speed, [:math:`m \ s^{-1}`]

    Returns
    -------
    A
        Speed, [:math:`knots`]
    """
    return m_per_s / 0.514444


def knots_to_m_per_s[A: (np.ndarray, xr.DataArray, float)](knots: A) -> A:
    r"""Convert speed from knots to meters per second (m/s).

    Parameters
    ----------
    knots : A
        Speed, [:math:`knots`]

    Returns
    -------
    A
        Speed, [:math:`m \ s^{-1}`]
    """
    return knots * 0.514444


def longitude_distance_to_m[A: (np.ndarray, xr.DataArray, float)](
    distance_degrees: A, latitude_mean: A
) -> A:
    r"""
    Convert longitude degrees distance between two points to cartesian distances in meters.

    Parameters
    ----------
    distance_degrees : A
        longitude distance, [:math:`\deg`]
    latitude_mean : A, optional
        mean latitude between ``longitude_1`` and ``longitude_2``, [:math:`\deg`]

    Returns
    -------
    A
        cartesian distance along the longitude axis, [:math:`m`]
    """
    latitude_mean_rad = degrees_to_radians(latitude_mean)
    return (distance_degrees / 180.0) * np.pi * constants.radius_earth * np.cos(latitude_mean_rad)


def latitude_distance_to_m[A: (np.ndarray, xr.DataArray, float)](distance_degrees: A) -> A:
    r"""
    Convert latitude degrees distance between two points to cartesian distances in meters.

    Parameters
    ----------
    distance_degrees : A
        latitude distance, [:math:`\deg`]

    Returns
    -------
    A
        Cartesian distance along the latitude axis, [:math:`m`]
    """
    return (distance_degrees / 180.0) * np.pi * constants.radius_earth


def m_to_longitude_distance[A: (np.ndarray, xr.DataArray, float)](
    distance_m: A, latitude_mean: A
) -> A:
    r"""
    Convert cartesian distance (meters) to differences in longitude degrees.

    Small angle approximation for ``distance_m`` <<
    :data:`~pycontrails.physics.constants.radius_earth`

    Parameters
    ----------
    distance_m : A
        cartesian distance along longitude axis, [:math:`m`]
    latitude_mean : A
        mean latitude between ``longitude_1`` and ``longitude_2``, [:math:`\deg`]

    Returns
    -------
    A
        longitude distance, [:math:`\deg`]
    """
    return radians_to_degrees(
        distance_m / (constants.radius_earth * np.cos(degrees_to_radians(latitude_mean)))
    )


def m_to_latitude_distance[A: (np.ndarray, xr.DataArray, float)](distance_m: A) -> A:
    r"""
    Convert cartesian distance (meters) to differences in latitude degrees.

    Small angle approximation for ``distance_m`` <<
    :data:`~pycontrails.physics.constants.radius_earth`

    Parameters
    ----------
    distance_m : A
        cartesian distance along latitude axis, [:math:`m`]

    Returns
    -------
    A
        latitude distance, [:math:`\deg`]
    """
    return radians_to_degrees(distance_m / constants.radius_earth)


def tas_to_mach_number[A: (np.ndarray, xr.DataArray, float)](true_airspeed: A, T: A) -> A:
    r"""Calculate Mach number from true airspeed at a specified ambient temperature.

    Parameters
    ----------
    true_airspeed : A
        True airspeed, [:math:`m \ s^{-1}`]
    T : A
        Ambient temperature, [:math:`K`]

    Returns
    -------
    A
        Mach number, [:math: `Ma`]

    References
    ----------
    - :cite:`cumpstyJetPropulsion2015`
    """
    return true_airspeed / np.sqrt((constants.kappa * constants.R_d) * T)


def mach_number_to_tas(
    mach_number: float | npt.NDArray[np.floating], T: float | npt.NDArray[np.floating]
) -> float | npt.NDArray[np.floating]:
    r"""Calculate true airspeed from the Mach number at a specified ambient temperature.

    Parameters
    ----------
    mach_number : float | npt.NDArray[np.floating]
        Mach number, [:math: `Ma`]
    T : npt.NDArray[np.floating]
        Ambient temperature, [:math:`K`]

    Returns
    -------
    npt.NDArray[np.floating]
        True airspeed, [:math:`m \ s^{-1}`]

    References
    ----------
    - :cite:`cumpstyJetPropulsion2015`
    """
    return mach_number * np.sqrt((constants.kappa * constants.R_d) * T)


def lbs_to_kg[A: (np.ndarray, xr.DataArray, float)](lbs: A) -> A:
    r"""Convert mass from pounds (lbs) to kilograms (kg).

    Parameters
    ----------
    lbs : A
        mass, pounds [:math:`lbs`]

    Returns
    -------
    A
        mass, kilograms [:math:`kg`]
    """
    return lbs * 0.45359


def dt_to_seconds(
    dt: npt.NDArray[np.timedelta64] | np.timedelta64,
    dtype: npt.DTypeLike = np.float64,
) -> npt.NDArray[np.floating]:
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
