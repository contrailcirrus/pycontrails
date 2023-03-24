"""
Monkey patch deprecated functions for validation script
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from pycontrails import JetA, MetDataset
from pycontrails.models import sac
from pycontrails.models.tau_cirrus import _assign_attrs, cirrus_effective_extinction_coef
from pycontrails.physics import constants, geo, thermo, units
from pycontrails.utils.types import ArrayLike

# ----------
# geo module
# ----------


def segment_angle(longitude: np.ndarray, latitude: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
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


# --------------------------
# contrail_properties module
# --------------------------


def segment_length_ratio(
    seg_length: np.ndarray, seg_length_t: np.ndarray, continuous: np.ndarray
) -> np.ndarray:
    """
    Calculates the ratio of contrail segment length before
    and after it is advected to the new location.

    TODO: @roger - I don't quite understand this function
    TODO: currently only works for 1D np.ndarray

    Parameters
    ----------
    seg_length : np.ndarray
        Original segment length of contrail waypoint, [:math:`m`]
    seg_length_t : np.ndarray
        Segment length of contrail waypoint after time step and advection, [:math:`m`]
    continuous : np.ndarray
        Description

    Returns
    -------
    np.ndarray
        Segment length ratio before and after it is advected to the new location.
    """

    # init new output
    seg_ratio = np.ones(seg_length_t.shape)

    s = (seg_length / np.maximum(seg_length_t, 1)) ** 0.5
    s_next = np.roll(s, 1)
    s_next[0] = 0

    # TODO: @roger what is going on here? Seems like something is duplicated
    bool_ops = continuous & (s > 0)
    seg_ratio[bool_ops] = seg_ratio[bool_ops] * s[bool_ops]
    bool_ops = s_next > 0
    seg_ratio[bool_ops] = seg_ratio[bool_ops] * s_next[bool_ops]

    return seg_ratio


def mean_energy_flux_per_m(
    rad_flux_per_m: np.ndarray,
    dt: np.ndarray | np.timedelta64,
    *,
    continuous: np.ndarray | None = None,
) -> np.ndarray:
    """
    Calculates the mean energy flux per length of contrail between two waypoints.

    Parameters
    ----------
    rad_flux_per_m : np.ndarray
        Mean radiative flux between time steps for waypoint, [:math:`W m^{-1}`].
        See :func:`mean_radiative_flux_per_m`.
    dt : np.ndarray
        timedelta of timestep for each waypoint

    Returns
    -------
    np.ndarray
        Mean energy flux per length of contrail between two waypoints, [:math:`J m^{-1}`]
    """
    # convert dt to seconds value
    dt_s = dt / np.timedelta64(1, "s")

    # TODO: Average between waypoints? Output energy_flux_0 in the future?
    energy_flux = rad_flux_per_m * dt_s
    energy_flux_t = np.roll(energy_flux, -1)
    energy_flux_per_m = (energy_flux + energy_flux_t) * 0.5
    if continuous is not None:
        energy_flux_per_m[~continuous] = 0

    return energy_flux_per_m


# ----------
# tau cirrus
# ----------


def tau_cirrus_original(met: MetDataset) -> xr.DataArray:
    """Calculate the optical depth of NWP cirrus around each pressure level.

    This implementation of `tau_cirrus` more closely aligned with Schumann's Fortran implementation.
    See `tau_cirrus` and check the `test_tau_cirrus` unit test module for agreement.
    """

    da = xr.zeros_like(met.data["specific_cloud_ice_water_content"])

    levels = met.data["level"]
    geopotential_height = met.data["geopotential"] / constants.g  # m

    for z in np.arange(len(levels)):
        z_top = max(z - 1, 0)  # Top Layer Index
        z_btm = min(z + 1, len(levels) - 1)  # Bottom Layer Index
        ds = met.data[dict(level=z)]

        beta_e = cirrus_effective_extinction_coef(
            ds["specific_cloud_ice_water_content"],
            ds["air_temperature"],
            ds["air_pressure"],
        )

        dz = 0.5 * (geopotential_height[dict(level=z_top)] - geopotential_height[dict(level=z_btm)])

        da[dict(level=z)] = da[dict(level=z_top)] + (beta_e * dz)

    return _assign_attrs(da)


# ------
# Units
# ------


def pl_to_m(pl: np.ndarray) -> np.ndarray:
    r"""Convert from pressure level (hPa) to altitude (m).

    Parameters
    ----------
    p : ArrayLike
        pressure level, [:math:`hPa`], [:math:`mbar`]

    Returns
    -------
    ArrayLike
        altitude, [:math:`m`]

    Notes
    -----
    Source: https://ozreport.com/docs/MarkGrahamarticle.pdf
    """

    pl_pa = pl * 100
    return (constants.T_msl / 0.0065) * (1 - (pl_pa / constants.p_surface) ** (1 / 5.255))


def _m_to_T_isa(h: np.ndarray) -> np.ndarray:
    """Calculate the ambient temperature (K) for a given altitude (m).

    Assumes the ICAO standard atmosphere.


    Parameters
    ----------
    h : ArrayLike
        altitude, [:math:`m`]

    Returns
    -------
    ArrayLike
        ICAO standard atmosphere ambient temperature, [:math:`K`]

    Notes
    -----
    Source: https://en.wikipedia.org/wiki/International_Standard_Atmosphere#ICAO_Standard_Atmosphere
    """
    T_isa = constants.T_msl + (h * constants.T_lapse_rate)
    T_stratosphere_isa = constants.T_msl + (constants.T_lapse_rate * constants.h_tropopause)

    # array
    if isinstance(T_isa, (np.ndarray, xr.DataArray)):
        # If h is ArrayLike, it supports indexing
        T_isa[h > constants.h_tropopause] = T_stratosphere_isa
        return T_isa

    # scalar
    return T_isa if h <= constants.h_tropopause else T_stratosphere_isa


def _low_altitude_m_to_pl(h: np.ndarray) -> np.ndarray:
    T_isa = _m_to_T_isa(h)
    power_term = -constants.g / (constants.T_lapse_rate * constants.R_d)
    return (constants.p_surface * (T_isa / constants.T_msl) ** power_term) / 100


def _high_altitude_m_to_pl(h: np.ndarray) -> np.ndarray:
    T_tropopause_isa = _m_to_T_isa(constants.h_tropopause)  # type: ignore[arg-type]
    power_term = -constants.g / (constants.T_lapse_rate * constants.R_d)
    p_tropopause_isa = constants.p_surface * (T_tropopause_isa / constants.T_msl) ** power_term
    inside_exp = (-constants.g / (constants.R_d * T_tropopause_isa)) * (h - constants.h_tropopause)
    return p_tropopause_isa * np.exp(inside_exp) / 100


def m_to_pl(h: np.ndarray) -> np.ndarray:
    r"""Convert from altitude (m) to pressure level (hPa).

    Parameters
    ----------
    h : ArrayLike
        altitude, [:math:`m`]

    Returns
    -------
    ArrayLike
        pressure level, [:math:`hPa`], [:math:`mbar`]

    Notes
    -----
    TODO: reference
    """
    # array
    if isinstance(h, (np.ndarray, xr.DataArray)):
        pl = _low_altitude_m_to_pl(h)
        pl[h > constants.h_tropopause] = _high_altitude_m_to_pl(h[h > constants.h_tropopause])
        return pl

    # scalar
    return _low_altitude_m_to_pl(h) if h <= constants.h_tropopause else _high_altitude_m_to_pl(h)


# ------
# SAC
# ------

jetA = JetA()


def T_critical_sac(
    air_temperature: ArrayLike,
    specific_humidity: ArrayLike,
    air_pressure: ArrayLike,
    engine_efficiency: float = 0.3,
    ei_h2o: float = jetA.ei_h2o,
    q_fuel: float = jetA.q_fuel,
    *,
    n_iter: int = 10,
    dx_threshold: float = 1e-3,
) -> ArrayLike:
    r"""Estimate actual Temperature threshold for persistent contrail formation iteratively.

    Applies Newton iteration and the Schmidt-Appleman criterion.

    This is required to estimate the activation rate of the aircraft emitted
    black carbon to contrail ice particles.

    Parameters
    ----------
    air_temperature : ArrayLike
        A sequence or array of temperature values, [:math:`K`]
    specific_humidity : ArrayLike
        A sequence or array of specific humidity values, [:math:`kg_{H_{2}O} \ kg_{air}`]
    air_pressure : ArrayLike
        A sequence or array of atmospheric pressure values, [:math:`Pa`].
    engine_efficiency: float, optional
        Engine efficiency, [:math:`0 - 1`]
    ei_h2o : float, optional
        Emission index of water vapor, [:math:`kg \ kg^{-1}`]
    q_fuel : float, optional
        Specific combustion heat of fuel combustion, [:math:`J \ kg^{-1} \ K^{-1}`]
    n_iter : int, optional
        Max number of iterations
    dx_threshold : float, optional
        Stop iteration once all elements of `dx` are smaller than `dx_threshold` in absolute value

    Returns
    -------
    ArrayLike
        Estimated Temperature threshold for contrail formation, [:math:`K`]

    Notes
    -----
    Source: Schumann, U., 1996. On conditions for contrail formation from aircraft exhausts.
            Meteorologische Zeitschrift, pp.4-23.
    """
    relative_humidity = thermo.rh(specific_humidity, air_temperature, air_pressure)
    G = sac.slope_mixing_line(specific_humidity, air_pressure, engine_efficiency, ei_h2o, q_fuel)
    T_sat_liquid_ = sac.T_sat_liquid(G)
    e_sat_T_sat_liquid = thermo.e_sat_liquid(T_sat_liquid_)

    # guess initial temperature
    relative_humidity_bounded = xr.where(relative_humidity < 1, relative_humidity, 0.9999)
    T0 = T_sat_liquid_ - (e_sat_T_sat_liquid / G)
    e_sat_T0 = thermo.e_sat_liquid(T0)
    c_1 = e_sat_T_sat_liquid * (T_sat_liquid_ - T0) / (2 * relative_humidity_bounded * e_sat_T0)
    c_2 = (
        1
        - (
            1
            + (4 * relative_humidity_bounded * e_sat_T0)
            / ((1 - relative_humidity_bounded) * e_sat_T_sat_liquid)
        )
        ** 0.5
    )
    T_crit_est = T_sat_liquid_ + (1 - relative_humidity) * c_1 * c_2

    # iterate up to n_iter until all elements of |dx| are smaller than dx_threshold
    for _ in np.arange(n_iter):
        e_sat_T_est = thermo.e_sat_liquid(T_crit_est)
        de_dt = (thermo.e_sat_liquid(T_crit_est + 1) - thermo.e_sat_liquid(T_crit_est - 1)) / 2
        dx = (
            e_sat_T_sat_liquid - G * (T_sat_liquid_ - T_crit_est) - relative_humidity * e_sat_T_est
        ) / (G - relative_humidity * de_dt)
        T_crit_est = T_crit_est - dx

        if (np.abs(dx) < dx_threshold).all():
            break

    # mask values where relative humidity is low
    is_dry = relative_humidity < 1e-6
    T_crit_est[is_dry] = T_sat_liquid_[is_dry] - (e_sat_T_sat_liquid[is_dry] / G[is_dry])

    # mask values where relative humidity is high
    is_liq_sat = relative_humidity > 0.9999
    T_crit_est[is_liq_sat] = T_sat_liquid_[is_liq_sat]

    return T_crit_est
