"""Thermodynamic relationships."""

from __future__ import annotations

import numpy as np
import xarray as xr

from pycontrails.physics import constants

# -------------------
# Material Properties
# -------------------


def rho_d[A: (np.ndarray, xr.DataArray, float)](T: A, p: A) -> A:
    r"""Calculate air density for (T, p) assuming dry air.

    Parameters
    ----------
    T : A
        Temperature, [:math:`K`]
    p : A
        Pressure, [:math:`Pa`]

    Returns
    -------
    A
        Air density of dry air, [:math:`kg \ m^{-3}`]
    """
    return p / (constants.R_d * T)


def rho_v[A: (np.ndarray, xr.DataArray, float)](T: A, p: A) -> A:
    r"""Calculate the air density for (T, p) assuming all water vapor.

    Parameters
    ----------
    T : A
        Temperature, [:math:`K`]
    p : A
        Pressure, [:math:`Pa`]

    Returns
    -------
    A
        Air density of water vapor, [:math:`kg \ m^{-3}`]
    """
    return p / (constants.R_v * T)


def c_pm[A: (np.ndarray, xr.DataArray, float)](q: A) -> A:
    r"""Calculate isobaric heat capacity of moist air.

    Parameters
    ----------
    q : A
        Specific humidity, [:math:`kg \ kg^{-1}`]

    Returns
    -------
    A
        Isobaric heat capacity of moist air, [:math:`J \ kg^{-1} \ K^{-1}`]

    Notes
    -----
    Some models (including CoCiP) use a constant value here (1004 :math:`J \ kg^{-1} \ K^{-1}`)

    """
    return constants.c_pd * (1.0 - q) + constants.c_pv * q


def p_vapor[A: (np.ndarray, xr.DataArray, float)](q: A, p: A) -> A:
    r"""Calculate the vapor pressure.

    Parameters
    ----------
    q : A
        Specific humidity, [:math:`kg \ kg^{-1}`]
    p : A
        Pressure, [:math:`Pa`]

    Returns
    -------
    A
        Vapor pressure, [:math:`Pa`]
    """
    return q * p / constants.epsilon


def water_vapor_partial_pressure_along_mixing_line[A: (np.ndarray, xr.DataArray, float)](
    specific_humidity: A,
    air_pressure: A,
    T_plume: A,
    T_ambient: A,
    G: A,
) -> A:
    """
    Calculate water vapor partial pressure along mixing line.

    Parameters
    ----------
    specific_humidity : A
        Specific humidity at each waypoint, [:math:`kg_{H_{2}O} / kg_{air}`]
    air_pressure : A
        Pressure altitude at each waypoint, [:math:`Pa`]
    T_plume : A
        Plume temperature evolution along mixing line, [:math:`K`]
    T_ambient : A
        Ambient temperature for each waypoint, [:math:`K`]
    G : A
        Slope of the mixing line in a temperature-humidity diagram.

    Returns
    -------
    A
        Water vapor partial pressure along mixing line (p_mw), [:math:`Pa`]

    References
    ----------
    Eq. (2) of Karcher et al. (2015).
    """
    p_wa = p_vapor(specific_humidity, air_pressure)
    return p_wa + G * (T_plume - T_ambient)


def diffusivity_water_vapor[A: (np.ndarray, xr.DataArray, float)](T: A, p: A) -> A:
    """
    Calculate molecular diffusivity of water vapor.

    Parameters
    ----------
    T: A
        Air temperature, [:math:`K`]

    p: A
        Air pressure, [:math:`Pa`]

    Returns
    -------
    A
        Molecular diffusivity of water vapor, [:math:`m^2 s^{-1}`]

    References
    ----------
    - :cite:`hallSurvivalIceParticles1976`
    - :cite:`pruppacherMicrophysicsCloudsPrecipitation2010`

    Notes
    -----
    The parameterization used by this function is valid for temperatures between -40 and 40 C,
    and input temperatures are clipped to this range.
    """
    # FIXME: Presently, mypy is not aware that numpy ufuncs will return `xr.DataArray``
    # when xr.DataArray is passed in. This will get fixed at some point in the future
    # as `numpy` their typing patterns, after which the "type: ignore" comment can
    # get ripped out.
    # We could explicitly check for `xr.DataArray` then use `xr.apply_ufunc`, but
    # this only renders our code more boilerplate and less performant.
    # This comment is pasted several places in `pycontrails` -- they should all be
    # addressed at the same time.
    T = np.clip(T, -40.0 - constants.absolute_zero, 40.0 - constants.absolute_zero)  # type: ignore[assignment]
    T0 = 273.15
    p0 = 101325.0
    return 0.0000211 * (T / T0) ** 1.94 * (p0 / p)


# -------------------
# Saturation Pressure
# -------------------


def e_sat_ice[A: (np.ndarray, xr.DataArray, float)](T: A) -> A:
    r"""Calculate saturation pressure of water vapor over ice.

    Parameters
    ----------
    T : A
        Temperature, [:math:`K`]

    Returns
    -------
    A
        Saturation pressure of water vapor over ice, [:math:`Pa`]

    References
    ----------
    - :cite:`sonntagAdvancementsFieldHygrometry1994`

    """
    # Goff Gratch equation (Smithsonian Tables, 1984)
    # return np.log10(-9.09718 * (273.16/T - 1) - 3.56654 * np.log10(273.16/T) + \
    #                  0.87679 * (1 - T/273.16) + np.log10(6.1071))

    # Magnus Teten (Murray, 1967)
    # return 6.1078 * np.exp(21.8745 * (T - 273.16) / (T - 7.66))

    # Zhang 2017 - incorrect implementation of Magnus Teten
    # return 6.1808 * np.exp(21.875 * (T - 276.16) / (T - 7.66))

    # Guide to Meteorological Instruments and Methods of Observation (CIMO Guide) (WMO, 2008)
    # return 6.112 * np.exp(22.46 * (T - 273.16) / (272.62 + T - 273.16))

    # Sonntag (1994) is used in CoCiP

    # FIXME: Presently, mypy is not aware that numpy ufuncs will return `xr.DataArray``
    # when xr.DataArray is passed in. This will get fixed at some point in the future
    # as `numpy` their typing patterns, after which the "type: ignore" comment can
    # get ripped out.
    # We could explicitly check for `xr.DataArray` then use `xr.apply_ufunc`, but
    # this only renders our code more boilerplate and less performant.
    # This comment is pasted several places in `pycontrails` -- they should all be
    # addressed at the same time.
    return 100.0 * np.exp(  # type: ignore[return-value]
        (-6024.5282 / T)
        + 24.7219
        + (0.010613868 * T)
        - (1.3198825e-5 * (T**2))
        - 0.49382577 * np.log(T)
    )


def sonntag_e_sat_liquid[A: (np.ndarray, xr.DataArray, float)](T: A) -> A:
    """Calculate saturation pressure of water vapor over liquid water using Sonntag (1994).

    Parameters
    ----------
    T : A
        Temperature, [:math:`K`]

    Returns
    -------
    A
        Saturation pressure of water vapor over liquid water, [:math:`Pa`]
    """
    return 100.0 * np.exp(  # type: ignore[return-value]
        -6096.9385 / T + 16.635794 - 0.02711193 * T + 1.673952 * 1e-5 * T**2 + 2.433502 * np.log(T)
    )


def mk05_e_sat_liquid[A: (np.ndarray, xr.DataArray, float)](T: A) -> A:
    """Calculate saturation pressure of water vapor over liquid water using Murphy and Koop (2005).

    Parameters
    ----------
    T : A
        Temperature, [:math:`K`]

    Returns
    -------
    A
        Saturation pressure of water vapor over liquid water, [:math:`Pa`]

    Notes
    -----
    Several formulations exist for the saturation vapor pressure over liquid water.

    Buck (Buck Research Manual 1996)..

        6.1121 * np.exp((18.678 * (T - 273.15) / 234.5) * (T - 273.15) / (257.14 + (T - 273.15)))

    Magnus Tetens (Murray, 1967)..

        6.1078 * np.exp(17.269388 * (T - 273.16) / (T - 35.86))

    Guide to Meteorological Instruments and Methods of Observation (CIMO Guide) (WMO, 2008)..

        6.112 * np.exp(17.62 * (T - 273.15) / (243.12 + T - 273.15))

    Sonntag (1994) (see :func:`sonntag_e_sat_liquid`) is used in older versions of CoCiP.
    """

    return np.exp(  # type: ignore[return-value]
        54.842763
        - 6763.22 / T
        - 4.21 * np.log(T)
        + 0.000367 * T
        + np.tanh(0.0415 * (T - 218.8))
        * (53.878 - 1331.22 / T - 9.44523 * np.log(T) + 0.014025 * T)
    )


def sonntag_e_sat_liquid_prime[A: (np.ndarray, xr.DataArray, float)](T: A) -> A:
    """Calculate the derivative of :func:`sonntag_e_sat_liquid`.

    Parameters
    ----------
    T : A
        Temperature, [:math:`K`].

    Returns
    -------
    A
        Derivative of :func:`sonntag_e_sat_liquid`
    """
    d_inside = 6096.9385 / (T**2) - 0.02711193 + 1.673952 * 1e-5 * 2 * T + 2.433502 / T
    return sonntag_e_sat_liquid(T) * d_inside


def mk05_e_sat_liquid_prime[A: (np.ndarray, xr.DataArray, float)](T: A) -> A:
    """Calculate the derivative of :func:`mk05_e_sat_liquid`.

    Parameters
    ----------
    T : A
        Temperature, [:math:`K`].

    Returns
    -------
    A
        Derivative of :func:`mk05_e_sat_liquid`
    """
    tanh_term = np.tanh(0.0415 * (T - 218.8))
    return mk05_e_sat_liquid(T) * (
        6763.22 / T**2
        - 4.21 / T
        + 0.000367
        + 0.0415 * (1 - tanh_term**2) * (53.878 - 1331.22 / T - 9.44523 * np.log(T) + 0.014025 * T)
        + tanh_term * (1331.22 / T**2 - 9.44523 / T + 0.014025)
    )


# Set aliases. These could be swapped out or made configurable.
e_sat_liquid = mk05_e_sat_liquid
e_sat_liquid_prime = mk05_e_sat_liquid_prime


def _e_sat_piecewise[A: (np.ndarray, xr.DataArray, float)](T: A) -> A:
    """Calculate `e_sat_liquid` when T is above freezing otherwise `e_sat_ice`.

    Parameters
    ----------
    T : A
        Temperature, [:math:`K`]

    Returns
    -------
    A
        Piecewise array of e_sat_liquid and e_sat_ice values.
    """
    ice = e_sat_ice(T)
    liquid = e_sat_liquid(T)
    is_liquid = T >= -constants.absolute_zero  # noqa: SIM300
    return ice + is_liquid * (liquid - ice)


# ----------------------------
# Saturation Specific Humidity
# ----------------------------


def q_sat[A: (np.ndarray, xr.DataArray, float)](T: A, p: A) -> A:
    r"""Calculate saturation specific humidity over liquid or ice.

    When T is above 0 C, liquid saturation is computed. Otherwise, ice saturation
    is computed.

    Parameters
    ----------
    T : A
        Temperature, [:math:`K`]
    p : A
        Pressure, [:math:`Pa`]

    Returns
    -------
    A
        Saturation specific humidity, [:math:`kg \ kg^{-1}`]

    Notes
    -----
    Smith et al. (1999)
    """
    e_sat = _e_sat_piecewise(T)
    return constants.epsilon * e_sat / p


def q_sat_ice[A: (np.ndarray, xr.DataArray, float)](T: A, p: A) -> A:
    r"""Calculate saturation specific humidity over ice.

    Parameters
    ----------
    T : A
        Temperature, [:math:`K`]
    p : A
        Pressure, [:math:`Pa`]

    Returns
    -------
    A
        Saturation specific humidity, [:math:`kg \ kg^{-1}`]

    Notes
    -----
    Smith et al. (1999)
    """
    return constants.epsilon * e_sat_ice(T) / p


def q_sat_liquid[A: (np.ndarray, xr.DataArray, float)](T: A, p: A) -> A:
    r"""Calculate saturation specific humidity over liquid.

    Parameters
    ----------
    T : A
        Temperature, [:math:`K`]
    p : A
        Pressure, [:math:`Pa`]

    Returns
    -------
    A
        Saturation specific humidity, [:math:`kg \ kg^{-1}`]

    Notes
    -----
    Smith et al. (1999)
    """
    return constants.epsilon * e_sat_liquid(T) / p


# -----------------
# Relative Humidity
# -----------------


def rh[A: (np.ndarray, xr.DataArray, float)](q: A, T: A, p: A) -> A:
    r"""Calculate the relative humidity with respect to to liquid water.

    Parameters
    ----------
    q : A
        Specific humidity, [:math:`kg \ kg^{-1}`]
    T : A
        Temperature, [:math:`K`]
    p : A
        Pressure, [:math:`Pa`]

    Returns
    -------
    A
        Relative Humidity, :math:`[0 - 1]`
    """
    return (q * p) / (constants.epsilon * e_sat_liquid(T))


def rhi[A: (np.ndarray, xr.DataArray, float)](q: A, T: A, p: A) -> A:
    r"""Calculate the relative humidity with respect to ice (RHi).

    Parameters
    ----------
    q : A
        Specific humidity, [:math:`kg \ kg^{-1}`]
    T : A
        Temperature, [:math:`K`]
    p : A
        Pressure, [:math:`Pa`]

    Returns
    -------
    A
        Relative Humidity over ice, :math:`[0 - 1]`
    """
    return (q * p) / (constants.epsilon * e_sat_ice(T))


# --------------
# Met Properties
# --------------


def pressure_dz[A: (np.ndarray, xr.DataArray, float)](T: A, p: A, dz: float) -> A:
    r"""Calculate the pressure altitude ``dz`` meters below input pressure.

    Returns surface pressure if the calculated pressure altitude is greater
    than :const:`~pycontrails.physics.constants.p_surface`.

    Parameters
    ----------
    T : A
        Temperature, [:math:`K`]
    p : A
        Pressure, [:math:`Pa`]
    dz : float
        Difference in altitude between measurements, [:math:`m`]

    Returns
    -------
    A
        Pressure at altitude, [:math:`Pa`]

    Notes
    -----
    This is used to calculate the temperature gradient and wind shear.
    """
    dp = rho_d(T, p) * constants.g * dz

    # FIXME: Presently, mypy is not aware that numpy ufuncs will return `xr.DataArray``
    # when xr.DataArray is passed in. This will get fixed at some point in the future
    # as `numpy` their typing patterns, after which the "type: ignore" comment can
    # get ripped out.
    # We could explicitly check for `xr.DataArray` then use `xr.apply_ufunc`, but
    # this only renders our code more boilerplate and less performant.
    # This comment is pasted several places in `pycontrails` -- they should all be
    # addressed at the same time.
    return np.minimum(p + dp, constants.p_surface)  # type: ignore[return-value]


def T_potential_gradient[A: (np.ndarray, xr.DataArray, float)](
    T_top: A,
    p_top: A,
    T_btm: A,
    p_btm: A,
    dz: float,
) -> A:
    r"""Calculate the potential temperature gradient between two altitudes.

    Parameters
    ----------
    T_top : A
        Temperature at original altitude, [:math:`K`]
    p_top : A
        Pressure at original altitude, [:math:`Pa`]
    T_btm : A
        Temperature at lower altitude, [:math:`K`]
    p_btm : A
        Pressure at lower altitude, [:math:`Pa`]
    dz : float
        Difference in altitude between measurements, [:math:`m`]

    Returns
    -------
    A
        Potential Temperature gradient, [:math:`K \ m^{-1}`]
    """
    T_potential_top = T_potential(T_top, p_top)
    T_potential_btm = T_potential(T_btm, p_btm)
    return (T_potential_top - T_potential_btm) / dz


def T_potential[A: (np.ndarray, xr.DataArray, float)](T: A, p: A) -> A:
    r"""Calculate potential temperature.

    The potential temperature is the temperature that
    an air parcel would attain if adiabatically
    brought to a standard reference pressure, :const:`~pycontrails.physics.constants.p_surface`.

    Parameters
    ----------
    T : A
        Temperature , [:math:`K`]
    p : A
        Pressure, [:math:`Pa`]

    Returns
    -------
    A
        Potential Temperature, [:math:`K`]

    References
    ----------
    - https://en.wikipedia.org/wiki/Potential_temperature
    """
    return T * (constants.p_surface / p) ** (constants.R_d / constants.c_pd)


def brunt_vaisala_frequency(p: np.ndarray, T: np.ndarray, T_grad: np.ndarray) -> np.ndarray:
    r"""Calculate the Brunt-Vaisaila frequency.

    The Brunt-Vaisaila frequency is the frequency at which a vertically
    displaced parcel will oscillate within a statically stable environment.

    Parameters
    ----------
    p : np.ndarray
        Pressure, [:math:`Pa`]
    T : np.ndarray
        Temperature , [:math:`K`]
    T_grad : np.ndarray
        Potential Temperature gradient (see :func:`T_potential_gradient`), [:math:`K \ m^{-1}`]

    Returns
    -------
    np.ndarray
        Brunt-Vaisaila frequency, [:math:`s^{-1}`]

    References
    ----------
    - https://en.wikipedia.org/wiki/Brunt%E2%80%93V%C3%A4is%C3%A4l%C3%A4_frequency
    """
    theta = T_potential(T, p)
    T_grad.clip(min=1e-6, out=T_grad)
    return (T_grad * constants.g / theta) ** 0.5
