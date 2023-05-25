"""Thermodynamic relationships."""

from __future__ import annotations

import numpy as np

from pycontrails.physics import constants
from pycontrails.utils.types import ArrayScalarLike, support_arraylike

# -------------------
# Material Properties
# -------------------


def rho_d(T: ArrayScalarLike, p: ArrayScalarLike) -> ArrayScalarLike:
    r"""Calculate air density for (T, p) assuming dry air.

    Parameters
    ----------
    T : ArrayScalarLike
        Temperature, [:math:`K`]
    p : ArrayScalarLike
        Pressure, [:math:`Pa`]

    Returns
    -------
    ArrayScalarLike
        Air density of dry air, [:math:`kg \ m^{-3}`]
    """
    return p / (constants.R_d * T)


def rho_v(T: ArrayScalarLike, p: ArrayScalarLike) -> ArrayScalarLike:
    r"""Calculate the air density for (T, p) assuming all water vapor.

    Parameters
    ----------
    T : ArrayScalarLike
        Temperature, [:math:`K`]
    p : ArrayScalarLike
        Pressure, [:math:`Pa`]

    Returns
    -------
    ArrayScalarLike
        Air density of water vapor, [:math:`kg \ m^{-3}`]
    """
    return p / (constants.R_v * T)


def c_pm(q: ArrayScalarLike) -> ArrayScalarLike:
    r"""Calculate isobaric heat capacity of moist air.

    Parameters
    ----------
    q : ArrayScalarLike
        Specific humidity, [:math:`kg \ kg^{-1}`]

    Returns
    -------
    ArrayScalarLike
        Isobaric heat capacity of moist air, [:math:`J \ kg^{-1} \ K^{-1}`]

    Notes
    -----
    Some models (including CoCiP) use a constant value here (1004 :math:`J \ kg^{-1} \ K^{-1}`)

    """
    return constants.c_pd * (1.0 + (constants.c_pv / constants.c_pd + 1.0) * q)


def p_vapor(q: ArrayScalarLike, p: ArrayScalarLike) -> ArrayScalarLike:
    r"""Calculate the vapor pressure.

    Parameters
    ----------
    q : ArrayScalarTypeVar
        Specific humidity, [:math:`kg \ kg^{-1}`]
    p : ArrayScalarTypeVar
        Pressure, [:math:`Pa`]

    Returns
    -------
    ArrayScalarTypeVar
        Vapor pressure, [:math:`Pa`]
    """
    return q * p * (constants.R_v / constants.R_d)


# -------------------
# Saturation Pressure
# -------------------


def e_sat_ice(T: ArrayScalarLike) -> ArrayScalarLike:
    r"""Calculate saturation pressure of water vapor over ice.

    Parameters
    ----------
    T : ArrayScalarLike
        Temperature, [:math:`K`]

    Returns
    -------
    ArrayScalarLike
        Saturation pressure of water vapor over ice, [:math:`Pa`]

    References
    ----------
    - :cite:`sonntag1994`

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


def e_sat_liquid(T: ArrayScalarLike) -> ArrayScalarLike:
    r"""Calculate saturation pressure of water vapor over liquid water.

    Parameters
    ----------
    T : ArrayScalarLike
        Temperature, [:math:`K`]

    Returns
    -------
    ArrayScalarLike
        Saturation pressure of water vapor over liquid water, [:math:`Pa`]

    References
    ----------
    - :cite:`sonntag1994`
    """
    # Buck (Buck Research Manual 1996)
    # 6.1121 * np.exp((18.678 * (T - 273.15) / 234.5) * (T - 273.15) / (257.14 + (T - 273.15)))

    # Magnus Tetens (Murray, 1967)
    # 6.1078 * np.exp(17.269388 * (T - 273.16) / (T – 35.86))

    # Guide to Meteorological Instruments and Methods of Observation (CIMO Guide) (WMO, 2008)
    # 6.112 * np.exp(17.62 * (T - 273.15) / (243.12 + T - 273.15))

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
        -6096.9385 / T
        + 16.635794
        - 0.02711193 * T
        + 1.673952 * 1e-5 * T**2
        + 2.433502 * np.log(T)
    )


@support_arraylike
def _e_sat_piecewise(T: np.ndarray) -> np.ndarray:
    """Calculate `e_sat_liquid` when T is above freezing otherwise `e_sat_ice`.

    Parameters
    ----------
    T : np.ndarray
        Temperature, [:math:`K`]

    Returns
    -------
    np.ndarray
        Piecewise array of e_sat_liquid and e_sat_ice values.
    """
    condlist = [T >= -constants.absolute_zero, T < constants.absolute_zero]
    funclist = [e_sat_liquid, e_sat_ice, np.nan]  # nan passed through
    return np.piecewise(T, condlist, funclist)


# ----------------------------
# Saturation Specific Humidity
# ----------------------------


def q_sat(T: ArrayScalarLike, p: ArrayScalarLike) -> ArrayScalarLike:
    r"""Calculate saturation specific humidity over liquid or ice.

    When T is above 0 C, liquid saturation is computed. Otherwise, ice saturation
    is computed.

    Parameters
    ----------
    T : ArrayScalarLike
        Temperature, [:math:`K`]
    p : ArrayScalarLike
        Pressure, [:math:`Pa`]

    Returns
    -------
    ArrayScalarLike
        Saturation specific humidity, [:math:`kg \ kg^{-1}`]

    Notes
    -----
    Smith et al. (1999)
    """
    e_sat = _e_sat_piecewise(T)
    return constants.epsilon * e_sat / p


def q_sat_ice(T: ArrayScalarLike, p: ArrayScalarLike) -> ArrayScalarLike:
    r"""Calculate saturation specific humidity over ice.

    Parameters
    ----------
    T : ArrayScalarLike
        Temperature, [:math:`K`]
    p : ArrayScalarLike
        Pressure, [:math:`Pa`]

    Returns
    -------
    ArrayScalarLike
        Saturation specific humidity, [:math:`kg \ kg^{-1}`]

    Notes
    -----
    Smith et al. (1999)
    """
    return constants.epsilon * e_sat_ice(T) / p


def q_sat_liquid(T: ArrayScalarLike, p: ArrayScalarLike) -> ArrayScalarLike:
    r"""Calculate saturation specific humidity over liquid.

    Parameters
    ----------
    T : ArrayScalarLike
        Temperature, [:math:`K`]
    p : ArrayScalarLike
        Pressure, [:math:`Pa`]

    Returns
    -------
    ArrayScalarLike
        Saturation specific humidity, [:math:`kg \ kg^{-1}`]

    Notes
    -----
    Smith et al. (1999)
    """
    return constants.epsilon * e_sat_liquid(T) / p


# -----------------
# Relative Humidity
# -----------------


def rh(q: ArrayScalarLike, T: ArrayScalarLike, p: ArrayScalarLike) -> ArrayScalarLike:
    r"""Calculate the relative humidity with respect to to liquid water.

    Parameters
    ----------
    q : ArrayScalarLike
        Specific humidity, [:math:`kg \ kg^{-1}`]
    T : ArrayScalarLike
        Temperature, [:math:`K`]
    p : ArrayScalarLike
        Pressure, [:math:`Pa`]

    Returns
    -------
    ArrayScalarLike
        Relative Humidity, :math:`[0 - 1]`
    """
    return (q * p * (constants.R_v / constants.R_d)) / e_sat_liquid(T)


def rhi(q: ArrayScalarLike, T: ArrayScalarLike, p: ArrayScalarLike) -> ArrayScalarLike:
    r"""Calculate the relative humidity with respect to ice (RHi).

    Parameters
    ----------
    q : ArrayScalarLike
        Specific humidity, [:math:`kg \ kg^{-1}`]
    T : ArrayScalarLike
        Temperature, [:math:`K`]
    p : ArrayScalarLike
        Pressure, [:math:`Pa`]

    Returns
    -------
    ArrayScalarLike
        Relative Humidity over ice, :math:`[0 - 1]`
    """
    return (q * p * (constants.R_v / constants.R_d)) / e_sat_ice(T)


# --------------
# Met Properties
# --------------


def p_dz(T: ArrayScalarLike, p: ArrayScalarLike, dz: float) -> ArrayScalarLike:
    r"""Calculate the pressure altitude ``dz`` meters below input pressure.

    Returns surface pressure if the calculated pressure altitude is greater
    than :const:`constants.p_surface`.

    Parameters
    ----------
    T : ArrayScalarLike
        Temperature, [:math:`K`]
    p : ArrayScalarLike
        Pressure, [:math:`Pa`]
    dz : float
        Difference in altitude between measurements, [:math:`m`]

    Returns
    -------
    ArrayScalarLike
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


def T_potential_gradient(
    T_top: ArrayScalarLike,
    p_top: ArrayScalarLike,
    T_btm: ArrayScalarLike,
    p_btm: ArrayScalarLike,
    dz: float,
) -> ArrayScalarLike:
    r"""Calculate the potential temperature gradient between two altitudes.

    Parameters
    ----------
    T_top : ArrayScalarLike
        Temperature at original altitude, [:math:`K`]
    p_top : ArrayScalarLike
        Pressure at original altitude, [:math:`Pa`]
    T_btm : ArrayScalarLike
        Temperature at lower altitude, [:math:`K`]
    p_btm : ArrayScalarLike
        Pressure at lower altitude, [:math:`Pa`]
    dz : float
        Difference in altitude between measurements, [:math:`m`]

    Returns
    -------
    ArrayScalarLike
        Potential Temperature gradient, [:math:`K \ m^{-1}`]
    """
    T_potential_top = T_potential(T_top, p_top)
    T_potential_btm = T_potential(T_btm, p_btm)
    return (T_potential_top - T_potential_btm) / dz


def T_potential(T: ArrayScalarLike, p: ArrayScalarLike) -> ArrayScalarLike:
    r"""Calculate potential temperature.

    The potential temperature is the temperature that
    an air parcel would attain if adiabatically
    brought to a standard reference pressure, :const:`constants.p_surface`.

    Parameters
    ----------
    T : ArrayScalarLike
        Temperature , [:math:`K`]
    p : ArrayScalarLike
        Pressure, [:math:`Pa`]

    Returns
    -------
    ArrayScalarLike
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
