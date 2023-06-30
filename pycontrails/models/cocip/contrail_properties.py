"""Contrail Property Calculations."""

from __future__ import annotations

import logging
import warnings
from typing import Any, TypeVar

import numpy as np
import numpy.typing as npt

from pycontrails.models.cocip import radiative_heating
from pycontrails.physics import constants, thermo, units

logger = logging.getLogger(__name__)

####################
# Initial Contrail Properties
####################


def initial_iwc(
    air_temperature: npt.NDArray[np.float_],
    specific_humidity: npt.NDArray[np.float_],
    air_pressure: npt.NDArray[np.float_],
    fuel_dist: npt.NDArray[np.float_],
    width: npt.NDArray[np.float_],
    depth: npt.NDArray[np.float_],
    ei_h2o: float,
) -> npt.NDArray[np.float_]:
    r"""
    Estimate the initial contrail ice water content (iwc) before the wake vortex phase.

    Note that the ice water content is replaced by zero if it is negative (dry air),
    and this will end the contrail life-cycle in subsequent steps.

    Parameters
    ----------
    air_temperature : npt.NDArray[np.float_]
        ambient temperature for each waypoint, [:math:`K`]
    specific_humidity : npt.NDArray[np.float_]
        ambient specific humidity for each waypoint, [:math:`kg_{H_{2}O}/kg_{air}`]
    air_pressure : npt.NDArray[np.float_]
        initial pressure altitude at each waypoint, before the wake vortex phase, [:math:`Pa`]
    fuel_dist : npt.NDArray[np.float_]
        fuel consumption of the flight segment per distance traveled, [:math:`kg m^{-1}`]
    width : npt.NDArray[np.float_]
        initial contrail width, [:math:`m`]
    depth : npt.NDArray[np.float_]
        initial contrail depth, [:math:`m`]
    ei_h2o : float
        water vapor emissions index of fuel, [:math:`kg_{H_{2}O} \ kg_{fuel}^{-1}`]

    Returns
    -------
    npt.NDArray[np.float_]
        Initial contrail ice water content (iwc) at the original waypoint
        before the wake vortex phase, [:math:`kg_{H_{2}O}/kg_{air}`].
        Returns zero if iwc is is negative (dry air).
    """
    q_sat = thermo.q_sat_ice(air_temperature, air_pressure)
    q_exhaust_ = q_exhaust(air_temperature, air_pressure, fuel_dist, width, depth, ei_h2o)
    return np.maximum(q_exhaust_ + specific_humidity - q_sat, 0.0)


def q_exhaust(
    air_temperature: npt.NDArray[np.float_],
    air_pressure: npt.NDArray[np.float_],
    fuel_dist: npt.NDArray[np.float_],
    width: npt.NDArray[np.float_],
    depth: npt.NDArray[np.float_],
    ei_h2o: float,
) -> npt.NDArray[np.float_]:
    r"""
    Calculate the specific humidity released by water vapor from aircraft emissions.

    Parameters
    ----------
    air_temperature : npt.NDArray[np.float_]
        ambient temperature for each waypoint, [:math:`K`]
    air_pressure : npt.NDArray[np.float_]
        initial pressure altitude at each waypoint, before the wake vortex phase, [:math:`Pa`]
    fuel_dist : npt.NDArray[np.float_]
        fuel consumption of the flight segment per distance travelled, [:math:`kg m^{-1}`]
    width : npt.NDArray[np.float_]
        initial contrail width, [:math:`m`]
    depth : npt.NDArray[np.float_]
        initial contrail depth, [:math:`m`]
    ei_h2o : float
        water vapor emissions index of fuel, [:math:`kg_{H_{2}O} \ kg_{fuel}^{-1}`]

    Returns
    -------
    npt.NDArray[np.float_]
        Humidity released by water vapour from aircraft emissions, [:math:`kg_{H_{2}O}/kg_{air}`]
    """
    rho_air = thermo.rho_d(air_temperature, air_pressure)
    return (ei_h2o * fuel_dist) / ((np.pi / 4.0) * width * depth * rho_air)


def iwc_adiabatic_heating(
    air_temperature: npt.NDArray[np.float_],
    air_pressure: npt.NDArray[np.float_],
    air_pressure_1: npt.NDArray[np.float_],
) -> npt.NDArray[np.float_]:
    """
    Calculate the change in ice water content due to adiabatic heating from the wake vortex phase.

    Parameters
    ----------
    air_temperature : npt.NDArray[np.float_]
        ambient temperature for each waypoint, [:math:`K`]
    air_pressure : npt.NDArray[np.float_]
        initial pressure altitude at each waypoint, before the wake vortex phase, [:math:`Pa`]
    air_pressure_1 : npt.NDArray[np.float_]
        pressure altitude at each waypoint, after the wake vortex phase, [:math:`Pa`]

    Returns
    -------
    npt.NDArray[np.float_]
        Change in ice water content due to adiabatic heating from the wake
        vortex phase, [:math:`kg_{H_{2}O}/kg_{air}`]
    """
    p_ice = thermo.e_sat_ice(air_temperature)
    air_temperature_1 = temperature_adiabatic_heating(air_temperature, air_pressure, air_pressure_1)
    p_ice_1 = thermo.e_sat_ice(air_temperature_1)

    out = (constants.R_d / constants.R_v) * ((p_ice_1 / air_pressure_1) - (p_ice / air_pressure))
    out.clip(min=0.0, out=out)
    return out


def temperature_adiabatic_heating(
    air_temperature: npt.NDArray[np.float_],
    air_pressure: npt.NDArray[np.float_],
    air_pressure_1: npt.NDArray[np.float_],
) -> npt.NDArray[np.float_]:
    """Calculate the ambient air temperature for each waypoint after the wake vortex phase.

    This calculation accounts for adiabatic heating.

    Parameters
    ----------
    air_temperature : npt.NDArray[np.float_]
        ambient temperature for each waypoint, [:math:`K`]
    air_pressure : npt.NDArray[np.float_]
        initial pressure altitude at each waypoint, before the wake vortex phase, [:math:`Pa`]
    air_pressure_1 : npt.NDArray[np.float_]
        pressure altitude at each waypoint, after the wake vortex phase, [:math:`Pa`]

    Returns
    -------
    npt.NDArray[np.float_]
        ambient air temperature after the wake vortex phase, [:math:`K`]

    Notes
    -----
    Level 1, see Figure 1 of :cite:`schumannContrailCirrusPrediction2012`

    References
    ----------
    - :cite:`schumannContrailCirrusPrediction2012`
    """
    exponent = (constants.gamma - 1.0) / constants.gamma
    return air_temperature * (air_pressure_1 / air_pressure) ** exponent


def iwc_post_wake_vortex(
    iwc: npt.NDArray[np.float_], iwc_ad: npt.NDArray[np.float_]
) -> npt.NDArray[np.float_]:
    """
    Calculate the ice water content after the wake vortex phase (``iwc_1``).

    ``iwc_1`` is calculated by subtracting the initial iwc before the wake vortex phase (``iwc``)
    by the change in iwc from adiabatic heating experienced during the wake vortex phase.

    Note that the iwc is replaced by zero if it is negative (dry air),
    and this will end the contrail lifecycle in subsequent steps.

    Parameters
    ----------
    iwc : npt.NDArray[np.float_]
        initial ice water content at each waypoint before the wake vortex
        phase, [:math:`kg_{H_{2}O}/kg_{air}`]
    iwc_ad : npt.NDArray[np.float_]
        change in iwc from adiabatic heating during the wake vortex
        phase, [:math:`kg_{H_{2}O}/kg_{air}`]

    Returns
    -------
    npt.NDArray[np.float_]
        ice water content after the wake vortex phase, ``iwc_1``, [:math:`kg_{H_{2}O}/kg_{air}`]

    Notes
    -----
    Level 1, see Figure 1 of :cite:`schumannContrailCirrusPrediction2012`

    References
    ----------
    - :cite:`schumannContrailCirrusPrediction2012`
    """
    return np.maximum(iwc - iwc_ad, 0.0)


def ice_particle_number(
    nvpm_ei_n: npt.NDArray[np.float_],
    fuel_dist: npt.NDArray[np.float_],
    iwc: npt.NDArray[np.float_],
    iwc_1: npt.NDArray[np.float_],
    air_temperature: npt.NDArray[np.float_],
    T_crit_sac: npt.NDArray[np.float_],
    min_ice_particle_number_nvpm_ei_n: float,
) -> npt.NDArray[np.float_]:
    """Calculate the initial number of ice particles per distance after the wake vortex phase.

    The initial number of ice particle per distance is calculated from the black
    carbon number emissions index ``nvpm_ei_n`` and fuel burn per distance ``fuel_dist``.
    Note that a lower bound for ``nvpm_ei_n`` is set at ``1e13`` :math:`kg^{-1}` to account
    for the activation of ambient aerosol particles and organic volatile particles.

    Parameters
    ----------
    nvpm_ei_n : npt.NDArray[np.float_]
        black carbon number emissions index, [:math:`kg^{-1}`]
    fuel_dist : npt.NDArray[np.float_]
        fuel consumption of the flight segment per distance traveled, [:math:`kg m^{-1}`]
    iwc : npt.NDArray[np.float_]
        initial ice water content at each flight waypoint before the wake vortex
        phase, [:math:`kg_{H_{2}O}/kg_{air}`]
    iwc_1 : npt.NDArray[np.float_]
        ice water content after the wake vortex phase, [:math:`kg_{H_{2}O}/kg_{air}`]
    air_temperature : npt.NDArray[np.float_]
        ambient temperature for each waypoint, [:math:`K`]
    T_crit_sac : npt.NDArray[np.float_]
        estimated Schmidt-Appleman temperature threshold for contrail formation, [:math:`K`]
    min_ice_particle_number_nvpm_ei_n : float
        lower bound for nvpm_ei_n to account for ambient aerosol particles for
        newer engines [:math:`kg^{-1}`]

    Returns
    -------
    npt.NDArray[np.float_]
        initial number of ice particles per distance after the wake vortex phase, [:math:`# m^{-1}`]
    """
    f_surv = ice_particle_survival_factor(iwc, iwc_1)
    f_activation = ice_particle_activation_rate(air_temperature, T_crit_sac)
    nvpm_ei_n_activated = nvpm_ei_n * f_activation
    return fuel_dist * np.maximum(nvpm_ei_n_activated, min_ice_particle_number_nvpm_ei_n) * f_surv


def ice_particle_activation_rate(
    air_temperature: npt.NDArray[np.float_], T_crit_sac: npt.NDArray[np.float_]
) -> npt.NDArray[np.float_]:
    """
    Calculate the activation rate of black carbon particles to contrail ice crystals.

    The activation rate is calculated as a function of the difference between
    the ambient temperature and the Schmidt-Appleman threshold temperature ``T_crit_sac``.

    Parameters
    ----------
    air_temperature : npt.NDArray[np.float_]
        ambient temperature at each waypoint before wake_wortex, [:math:`K`]
    T_crit_sac : npt.NDArray[np.float_]
        estimated Schmidt-Appleman temperature threshold for contrail formation, [:math:`K`]

    Returns
    -------
    npt.NDArray[np.float_]
        Proportion of black carbon particles that activates to contrail ice parties.

    Notes
    -----
    The equation is not published but based on the raw data
    from :cite:`brauerAirborneMeasurementsContrail2021`.

    References
    ----------
    - :cite:`brauerAirborneMeasurementsContrail2021`
    """
    d_temp = air_temperature - T_crit_sac
    d_temp.clip(None, 0.0, out=d_temp)

    # NOTE: It seems somewhat unnecessary to do this additional "rounding down"
    # of d_temp for values below -5. As d_temp near -5, the activation rate approaches
    # 1. This additional rounding injects a small jump discontinuity into the activation rate that
    # likely does not match reality. I suggest removing the line below. This will change
    # model outputs roughly at 0.001 - 0.1%.
    d_temp[d_temp < -5.0] = -np.inf
    return -0.661 * np.exp(d_temp) + 1.0


def ice_particle_survival_factor(
    iwc: npt.NDArray[np.float_], iwc_1: npt.NDArray[np.float_]
) -> npt.NDArray[np.float_]:
    """
    Estimate the fraction of contrail ice particle number that survive the wake vortex phase.

    CoCiP assumes that this fraction is proportional to the change in ice water content
    (``iwc_1 - iwc``) before and after the wake vortex phase.

    Parameters
    ----------
    iwc : npt.NDArray[np.float_]
        initial ice water content at each waypoint before the wake vortex
        phase, [:math:`kg_{H_{2}O}/kg_{air}`]
    iwc_1 : npt.NDArray[np.float_]
        ice water content after the wake vortex phase, [:math:`kg_{H_{2}O}/kg_{air}`]

    Returns
    -------
    npt.NDArray[np.float_]
        Fraction of contrail ice particle number that survive the wake vortex phase.
    """
    f_surv = np.empty_like(iwc)
    is_positive = (iwc > 0.0) & (iwc_1 > 0.0)

    ratio = iwc_1[is_positive] / iwc[is_positive]
    ratio.clip(None, 1.0, out=ratio)

    f_surv[is_positive] = ratio
    f_surv[~is_positive] = 0.5

    return f_surv


def initial_persistent(
    iwc_1: npt.NDArray[np.float_], rhi_1: npt.NDArray[np.float_]
) -> npt.NDArray[np.float_]:
    """
    Determine if waypoints have persistent contrails.

    Conditions for persistent initial_contrails:

    1. ice water content at level 1: ``1e-12 < iwc < 1e10``
    2. rhi at level 1: ``0 < rhi < 1e10``

    .. versionchanged:: 0.25.1
        Returned array now has floating dtype. This is consistent with other filtering
        steps in the CoCiP model (ie, ``sac``).

    Parameters
    ----------
    iwc_1 : npt.NDArray[np.float_]
        ice water content after the wake vortex phase, [:math:`kg_{H_{2}O}/kg_{air}`]
    rhi_1 : npt.NDArray[np.float_]
        relative humidity with respect to ice after the wake vortex phase

    Returns
    -------
    npt.NDArray[np.float_]
        Mask of waypoints with persistent contrails. Waypoints with persistent contrails
        will have value 1.

    Notes
    -----
    The RHi at level 1 does not need to be above 100% if the iwc > 0 kg/kg. If iwc > 0
    and RHi < 100%, the contrail lifetime will not end immediately as the ice particles
    will gradually evaporate with the rate depending on the background RHi.
    """
    out = (iwc_1 > 1e-12) & (iwc_1 < 1e10) & (rhi_1 > 0.0) & (rhi_1 < 1e10)
    dtype = np.result_type(iwc_1, rhi_1)
    return out.astype(dtype)


def contrail_persistent(
    latitude: npt.NDArray[np.float_],
    altitude: npt.NDArray[np.float_],
    segment_length: npt.NDArray[np.float_],
    age: npt.NDArray[np.timedelta64],
    tau_contrail: npt.NDArray[np.float_],
    n_ice_per_m3: npt.NDArray[np.float_],
    params: dict[str, Any],
) -> npt.NDArray[np.bool_]:
    r"""
    Determine surviving contrail segments after time integration step.

    A contrail waypoint reaches its end of life if any of the following conditions hold:

    1. Contrail age exceeds ``max_age``
    2. Contrail optical depth lies outside of interval ``[min_tau, max_tau]``
    3. Ice number density lies outside of interval ``[min_n_ice_per_m3, max_n_ice_per_m3]``
    4. Altitude lies outside of the interval ``[min_altitude_m, max_altitude_m]``
    5. Segment length exceeds ``max_seg_length_m``
    6. Latitude values are within 1 degree of the north or south pole

    This function warns if all values in the ``tau_contrail`` array are nan.

    .. versionchanged:: 0.25.10

        Extreme values of ``latitude`` (ie, close to the north or south pole) now
        create an end of life condition. This check helps address issues related to
        divergence in the polar regions. (With large enough integration time delta,
        it is still possible for post-advection latitude values to lie outside of
        [-90, 90], but this is no longer possible with typical parameters and wind
        speeds.)

    Parameters
    ----------
    latitude : npt.NDArray[np.float_]
        Contrail latitude, [:math:`\deg`]
    altitude : npt.NDArray[np.float_]
        Contrail altitude, [:math:`m`]
    segment_length : npt.NDArray[np.float_]
        Contrail segment length, [:math:`m`]
    age : npt.NDArray[np.timedelta64]
        Contrail age
    tau_contrail : npt.NDArray[np.float_]
        Contrail optical depth
    n_ice_per_m3 : npt.NDArray[np.float_]
        Contrail ice particle number per volume of air, [:math:`# m^{-3}`]
    params : dict[str, Any]
        Dictionary of :class:`CocipParams` parameters determining the
        conditions for end of contrail life.

    Returns
    -------
    npt.NDArray[np.bool_]
        Boolean array indicating surviving contrails. Persisting contrails
        will be marked as True.
    """
    status_1 = _within_range(age, max=params["max_age"])
    status_2 = _within_range(tau_contrail, max=params["max_tau"], min=params["min_tau"])
    status_3 = _within_range(
        n_ice_per_m3, max=params["max_n_ice_per_m3"], min=params["min_n_ice_per_m3"]
    )
    status_4 = _within_range(altitude, max=params["max_altitude_m"], min=params["min_altitude_m"])
    status_5 = _within_range(segment_length, max=params["max_seg_length_m"])
    status_6 = _within_range(latitude, max=89.0, min=-89.0)  # type: ignore[type-var]

    logger.debug(
        "Survival stats. age: %s, tau: %s, ice: %s, altitude: %s, segment: %s, latitude: %s",
        np.sum(status_1),
        np.sum(status_2),
        np.sum(status_3),
        np.sum(status_4),
        np.sum(status_5),
        np.sum(status_6),
    )
    tau_contrail_nan = np.isnan(tau_contrail)
    logger.debug(
        "Fraction of nan in tau_contrail: %s / %s",
        tau_contrail_nan.sum(),
        tau_contrail_nan.size,
    )
    if np.all(tau_contrail_nan):
        warnings.warn(
            "All tau_contrail values are nan. This may be due to waypoints "
            "all lying outside of the met interpolation grid. It could "
            "indicate an issue with interpolation, or an insufficient "
            "met domain."
        )
    return status_1 & status_2 & status_3 & status_4 & status_5 & status_6


T = TypeVar("T", np.float_, np.timedelta64)


def _within_range(
    val: npt.NDArray[T],
    max: npt.NDArray[T] | T | None = None,
    min: npt.NDArray[T] | T | None = None,
) -> npt.NDArray[np.bool_]:
    """
    Check if the input values (val) are each within the specified range.

    If both ``max`` and ``min`` are None, a literal constant True is returned.

    Parameters
    ----------
    val : np.ndarray
        value of selected contrail property
    max : np.ndarray | float | np.timedelta64 | None, optional
        Upper bound. If None, no upper bound is imposed. None by default.
    min : np.ndarray | float | np.timedelta64 | None, optional
        Lower bound. If None, no lower bound is imposed. None by default.

    Returns
    -------
    npt.NDArray[np.bool_]
        Mask of waypoints. Waypoints with values within the specified range will be marked as true.
    """
    cond: npt.NDArray[np.bool_] = True  # type: ignore[assignment]
    if min is not None:
        cond &= val >= min
    if max is not None:
        cond &= val <= max
    return cond


####################
# Contrail Properties
####################


def contrail_edges(
    lon: npt.NDArray[np.float_],
    lat: npt.NDArray[np.float_],
    sin_a: npt.NDArray[np.float_],
    cos_a: npt.NDArray[np.float_],
    width: npt.NDArray[np.float_],
) -> tuple[
    npt.NDArray[np.float_], npt.NDArray[np.float_], npt.NDArray[np.float_], npt.NDArray[np.float_]
]:
    """
    Calculate the longitude and latitude of the contrail edges to account for contrail spreading.

    (lon_edge_l, lat_edge_l)        x---------------------

    (Contrail midpoint: lon, lat)   X===================== ->

    (lon_edge_r, lat_edge_r)        x---------------------


    Parameters
    ----------
    lon : npt.NDArray[np.float_]
        longitude of contrail waypoint, degrees
    lat : npt.NDArray[np.float_]
        latitude of contrail waypoint, degrees
    sin_a : npt.NDArray[np.float_]
        sin(a), where a is the angle between the plume and the longitudinal axis
    cos_a : npt.NDArray[np.float_]
        cos(a), where a is the angle between the plume and the longitudinal axis
    width : npt.NDArray[np.float_]
        contrail width at each waypoint, [:math:`m`]

    Returns
    -------
    tuple[npt.NDArray[np.float_], npt.NDArray[np.float_], npt.NDArray[np.float_], npt.NDArray[np.float_]]
        (lon_edge_l, lat_edge_l, lon_edge_r, lat_edge_r), longitudes and latitudes
        at the left and right edges of the contrail, degrees
    """  # noqa: E501
    dlon = units.m_to_longitude_distance(width * sin_a * 0.5, lat)
    dlat = units.m_to_latitude_distance(width * cos_a * 0.5)

    lon_edge_l = lon - dlon
    lat_edge_l = lat + dlat
    lon_edge_r = lon + dlon
    lat_edge_r = lat - dlat

    return lon_edge_l, lat_edge_l, lon_edge_r, lat_edge_r


def contrail_vertices(
    lon: npt.NDArray[np.float_],
    lat: npt.NDArray[np.float_],
    sin_a: npt.NDArray[np.float_],
    cos_a: npt.NDArray[np.float_],
    width: npt.NDArray[np.float_],
    segment_length: npt.NDArray[np.float_],
) -> tuple[
    npt.NDArray[np.float_],
    npt.NDArray[np.float_],
    npt.NDArray[np.float_],
    npt.NDArray[np.float_],
    npt.NDArray[np.float_],
    npt.NDArray[np.float_],
    npt.NDArray[np.float_],
    npt.NDArray[np.float_],
]:
    """
    Calculate the longitude and latitude of the contrail vertices.

    This is equivalent to running :meth:`contrail_edges` at each contrail waypoint
    and associating the next continuous waypoint with the previous.
    This method is helpful when you want to treat each contrail waypoint independently.

    (lon_1, lat_1)                  x--------------------x   (lon_4, lat_4)

    (Contrail waypoint: lon, lat)   X==================== ->

    (lon_2, lat_2)                  x--------------------x   (lon_3, lat_3)


    Parameters
    ----------
    lon : npt.NDArray[np.float_]
        longitude of contrail waypoint, degrees
    lat : npt.NDArray[np.float_]
        latitude of contrail waypoint, degrees
    sin_a : npt.NDArray[np.float_]
        sin(a), where a is the angle between the plume and the longitudinal axis
    cos_a : npt.NDArray[np.float_]
        cos(a), where a is the angle between the plume and the longitudinal axis
    width : npt.NDArray[np.float_]
        contrail width at each waypoint, [:math:`m`]
    segment_length : npt.NDArray[np.float_]
        contrail length at each waypoint, [:math:`m`]

    Returns
    -------
    tuple[npt.NDArray[np.float_], npt.NDArray[np.float_], npt.NDArray[np.float_], npt.NDArray[np.float_]]
        (lon_1, lat_1, lon_2, lat_2, lon_3, lat_3, lon_4, lat_4) degrees
    """  # noqa: E501
    dlon_width = units.m_to_longitude_distance(width * sin_a * 0.5, lat)
    dlat_width = units.m_to_latitude_distance(width * cos_a * 0.5)

    # using "lat" as mean here is a little inaccurate, but its a close approx
    dlon_length = units.m_to_longitude_distance(segment_length * cos_a, lat)
    dlat_length = units.m_to_latitude_distance(segment_length * sin_a)

    lon_1 = lon - dlon_width
    lon_2 = lon + dlon_width
    lon_3 = lon + dlon_width + dlon_length
    lon_4 = lon - dlon_width + dlon_length

    lat_1 = lat + dlat_width
    lat_2 = lat - dlat_width
    lat_3 = lat - dlat_width + dlat_length
    lat_4 = lat + dlat_width + dlat_length

    return lon_1, lat_1, lon_2, lat_2, lon_3, lat_3, lon_4, lat_4


def plume_effective_cross_sectional_area(
    width: npt.NDArray[np.float_], depth: npt.NDArray[np.float_], sigma_yz: npt.NDArray[np.float_]
) -> npt.NDArray[np.float_]:
    """
    Calculate the effective cross-sectional area of the contrail plume (``area_eff``).

    ``sigma_yy``, ``sigma_zz`` and ``sigma_yz`` are the parameters governing the
    contrail plume's temporal evolution.

    Parameters
    ----------
    width : npt.NDArray[np.float_]
        contrail width at each waypoint, [:math:`m`]
    depth : npt.NDArray[np.float_]
        contrail depth at each waypoint, [:math:`m`]
    sigma_yz : npt.NDArray[np.float_]
        temporal evolution of the contrail plume parameters

    Returns
    -------
    npt.NDArray[np.float_]
        effective cross-sectional area of the contrail plume, [:math:`m^{2}`]
    """
    sigma_yy = 0.125 * (width**2)
    sigma_zz = 0.125 * (depth**2)
    return new_effective_area_from_sigma(sigma_yy, sigma_zz, sigma_yz)


def plume_effective_depth(
    width: npt.NDArray[np.float_], area_eff: npt.NDArray[np.float_]
) -> npt.NDArray[np.float_]:
    """
    Calculate the effective depth of the contrail plume (``depth_eff``).

    ``depth_eff`` is calculated from the effective cross-sectional area (``area_eff``)
    and the contrail width.

    Parameters
    ----------
    width : npt.NDArray[np.float_]
        contrail width at each waypoint, [:math:`m`]
    area_eff : npt.NDArray[np.float_]
        effective cross-sectional area of the contrail plume, [:math:`m^{2}`]

    Returns
    -------
    npt.NDArray[np.float_]
        effective depth of the contrail plume, [:math:`m`]
    """
    return area_eff / width


def plume_mass_per_distance(
    area_eff: npt.NDArray[np.float_], rho_air: npt.NDArray[np.float_]
) -> npt.NDArray[np.float_]:
    """
    Calculate the contrail plume mass per unit length.

    Parameters
    ----------
    area_eff : npt.NDArray[np.float_]
        effective cross-sectional area of the contrail plume, [:math:`m^{2}`]
    rho_air : npt.NDArray[np.float_]
        density of air for each waypoint, [:math:`kg m^{-3}`]

    Returns
    -------
    npt.NDArray[np.float_]
        contrail plume mass per unit length, [:math:`kg m^{-1}`]
    """
    return area_eff * rho_air


def ice_particle_number_per_volume_of_plume(
    n_ice_per_m: npt.NDArray[np.float_], area_eff: npt.NDArray[np.float_]
) -> npt.NDArray[np.float_]:
    """
    Calculate the number of contrail ice particles per volume of plume (``n_ice_per_vol``).

    Parameters
    ----------
    n_ice_per_m : npt.NDArray[np.float_]
        number of ice particles per distance at time t, [:math:`m^{-1}`]
    area_eff : npt.NDArray[np.float_]
        effective cross-sectional area of the contrail plume, [:math:`m^{2}`]

    Returns
    -------
    npt.NDArray[np.float_]
        number of ice particles per volume of contrail plume at time t, [:math:`# m^{-3}`]
    """
    return n_ice_per_m / area_eff


def ice_particle_number_per_mass_of_air(
    n_ice_per_vol: npt.NDArray[np.float_], rho_air: npt.NDArray[np.float_]
) -> npt.NDArray[np.float_]:
    """
    Calculate the number of contrail ice particles per mass of air.

    Parameters
    ----------
    n_ice_per_vol : npt.NDArray[np.float_]
        number of ice particles per volume of contrail plume at time t, [:math:`# m^{-3}`]
    rho_air : npt.NDArray[np.float_]
        density of air for each waypoint, [:math:`kg m^{-3}`]

    Returns
    -------
    npt.NDArray[np.float_]
        number of ice particles per mass of air at time t, [:math:`# kg^{-1}`]
    """
    return n_ice_per_vol / rho_air


def ice_particle_volume_mean_radius(
    iwc: npt.NDArray[np.float_], n_ice_per_kg_air: npt.NDArray[np.float_]
) -> npt.NDArray[np.float_]:
    """
    Calculate the ice particle volume mean radius.

    Parameters
    ----------
    iwc : npt.NDArray[np.float_]
        contrail ice water content, i.e., contrail ice mass per
        kg of air, [:math:`kg_{H_{2}O}/kg_{air}`]

    n_ice_per_kg_air : npt.NDArray[np.float_]
        number of ice particles per mass of air, [:math:`# kg^{-1}`]

    Returns
    -------
    npt.NDArray[np.float_]
        ice particle volume mean radius, [:math:`m`]

    Notes
    -----
    ``r_ice_vol`` is the mean radius of a sphere that has the same volume as the
    contrail ice particle.

    ``r_ice_vol`` calculated by dividing the total volume of contrail
    ice particle per kg of air (``total_ice_volume``, :math:`m**3/kg-air`) with the
    number of contrail ice particles per kg of air (``n_ice_per_kg_air``, :math:`#/kg-air`).
    """
    total_ice_volume = iwc / constants.rho_ice
    r_ice_vol = ((3 / (4.0 * np.pi)) * (total_ice_volume / n_ice_per_kg_air)) ** (1 / 3)
    zero_negative_values = iwc <= 0.0
    r_ice_vol[zero_negative_values] = iwc[zero_negative_values]
    r_ice_vol.clip(min=1e-10, out=r_ice_vol)
    return r_ice_vol


def ice_particle_terminal_fall_speed(
    air_pressure: npt.NDArray[np.float_],
    air_temperature: npt.NDArray[np.float_],
    r_ice_vol: npt.NDArray[np.float_],
) -> npt.NDArray[np.float_]:
    """
    Calculate the terminal fall speed of contrail ice particles.

    ``v_t`` is calculated based on a parametric model
    from :cite:`spichtingerModellingCirrusClouds2009`, using inputs of pressure
    level, ambient temperature and the ice particle volume mean radius.

    Parameters
    ----------
    air_pressure : npt.NDArray[np.float_]
        Pressure altitude at each waypoint, [:math:`Pa`]
    air_temperature : npt.NDArray[np.float_]
        Ambient temperature for each waypoint, [:math:`K`]
    r_ice_vol : npt.NDArray[np.float_]
        Ice particle volume mean radius, [:math:`m`]

    Returns
    -------
    npt.NDArray[np.float_]
        Terminal fall speed of contrail ice particles, [:math:`m s^{-1}`]

    References
    ----------
    - :cite:`spichtingerModellingCirrusClouds2009`
    """
    ipm = ice_particle_mass(r_ice_vol)

    alpha = np.full_like(r_ice_vol, np.nan)

    # For ice particle mass >= 4.264e-8 kg
    particle_mass = ipm >= 4.264e-8
    alpha[particle_mass] = 8.80 * ipm[particle_mass] ** 0.096

    # For ice particle mass in [2.166e-9 kg, 4.264e-8 kg)
    particle_mass = (ipm < 4.264e-8) & (ipm >= 2.166e-9)
    alpha[particle_mass] = 329.8 * ipm[particle_mass] ** 0.31

    # For ice particle mass in [2.146e-13 kg, 2.166e-9 kg)
    particle_mass = (ipm < 2.166e-9) & (ipm >= 2.146e-13)
    alpha[particle_mass] = 63292.4 * ipm[particle_mass] ** 0.57

    # For ice particle mass < 2.146e-13 kg
    particle_mass = ipm < 2.146e-13
    alpha[particle_mass] = 735.4 * ipm[particle_mass] ** 0.42

    return alpha * (30000.0 / air_pressure) ** 0.178 * (233.0 / air_temperature) ** 0.394


def ice_particle_mass(r_ice_vol: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
    """
    Calculate the contrail ice particle mass.

    It is calculated by multiplying the mean ice particle volume with the density of ice

    Parameters
    ----------
    r_ice_vol : npt.NDArray[np.float_]
        Ice particle volume mean radius, [:math:`m`]

    Returns
    -------
    npt.NDArray[np.float_]
        Mean contrail ice particle mass, [:math:`kg`]
    """
    return ((4 / 3) * np.pi * r_ice_vol**3) * constants.rho_ice


def horizontal_diffusivity(
    ds_dz: npt.NDArray[np.float_], depth: npt.NDArray[np.float_]
) -> npt.NDArray[np.float_]:
    """
    Calculate contrail horizontal diffusivity.

    Parameters
    ----------
    ds_dz : npt.NDArray[np.float_]
        Total wind shear (eastward and northward winds) with respect
        to altitude (``dz``), [:math:`m s^{-1} / Pa`]
    depth : npt.NDArray[np.float_]
        Contrail depth at each waypoint, [:math:`m`]

    Returns
    -------
    npt.NDArray[np.float_]
        horizontal diffusivity, [:math:`m^{2} s^{-1}`]

    References
    ----------
    - :cite:`schumannContrailCirrusPrediction2012`

    Notes
    -----
    Accounts for the turbulence-induced diffusive contrail spreading in
    the horizontal direction.
    """
    return 0.1 * ds_dz * depth**2


def vertical_diffusivity(
    air_pressure: npt.NDArray[np.float_],
    air_temperature: npt.NDArray[np.float_],
    dT_dz: npt.NDArray[np.float_],
    depth_eff: npt.NDArray[np.float_],
    terminal_fall_speed: npt.NDArray[np.float_],
    sedimentation_impact_factor: npt.NDArray[np.float_] | float,
    eff_heat_rate: npt.NDArray[np.float_] | None,
) -> npt.NDArray[np.float_]:
    """
    Calculate contrail vertical diffusivity.

    Parameters
    ----------
    air_pressure : npt.NDArray[np.float_]
        Pressure altitude at each waypoint, [:math:`Pa`]
    air_temperature : npt.NDArray[np.float_]
        Ambient temperature for each waypoint, [:math:`K`]
    dT_dz : npt.NDArray[np.float_]
        Temperature gradient with respect to altitude (dz), [:math:`K m^{-1}`]
    depth_eff : npt.NDArray[np.float_]
        Effective depth of the contrail plume, [:math:`m`]
    terminal_fall_speed : npt.NDArray[np.float_]
        Terminal fall speed of contrail ice particles, [:math:`m s^{-1}`]
    sedimentation_impact_factor : float
        Enhancement parameter denoted by `f_T` in eq. (35) Schumann (2012).
    eff_heat_rate: npt.NDArray[np.float_] | None
        Effective heating rate, i.e., rate of which the contrail plume
        is heated, [:math:`K s^{-1}`]. If None is passed, the radiative
        heating effects on contrail cirrus properties are not included.

    Returns
    -------
    npt.NDArray[np.float_]
        vertical diffusivity, [:math:`m^{2} s^{-1}`]

    References
    ----------
    - :cite:`schumannContrailCirrusPrediction2012`
    - :cite:`schumannAviationinducedCirrusRadiation2013`

    Notes
    -----
    Accounts for the turbulence-induced diffusive contrail spreading in the vertical direction.
    See eq. (35) of :cite:`schumannContrailCirrusPrediction2012`.

    The first term in Eq. (35) of Schumann (2012)
    (c_V * w'_N^2 / N_BV, where c_V = 0.2 and w'_N^2 = 0.1) is different
    than outlined below. Here, a constant of 0.01 is used when radiative
    heating effects are not activated. This update comes from Schumann and
    Graf (2013), which found that the original formulation estimated thinner
    contrails relative to satellite observations. The vertical diffusivity
    was enlarged so that the simulated contrails are more consistent with observations.
    """
    n_bv = thermo.brunt_vaisala_frequency(air_pressure, air_temperature, dT_dz)
    n_bv.clip(min=0.001, out=n_bv)

    cvs: npt.NDArray[np.float_] | float
    if eff_heat_rate is not None:
        cvs = radiative_heating.convective_velocity_scale(depth_eff, eff_heat_rate, air_temperature)
        cvs.clip(min=0.01, out=cvs)
    else:
        cvs = 0.01

    return cvs / n_bv + sedimentation_impact_factor * terminal_fall_speed * depth_eff


####################
# Ice particle losses
####################


def particle_losses_aggregation(
    r_ice_vol: npt.NDArray[np.float_],
    terminal_fall_speed: npt.NDArray[np.float_],
    area_eff: npt.NDArray[np.float_],
    agg_efficiency: float = 1.0,
) -> npt.NDArray[np.float_]:
    """
    Calculate the rate of contrail ice particle losses due to sedimentation-induced aggregation.

    Parameters
    ----------
    r_ice_vol : npt.NDArray[np.float_]
        Ice particle volume mean radius, [:math:`m`]
    terminal_fall_speed : npt.NDArray[np.float_]
        Terminal fall speed of contrail ice particles, [:math:`m s^{-1}`]
    area_eff : npt.NDArray[np.float_]
        Effective cross-sectional area of the contrail plume, [:math:`m^{2}`]
    agg_efficiency : float, optional
        Aggregation efficiency

    Returns
    -------
    npt.NDArray[np.float_]
        Rate of contrail ice particle losses due to sedimentation-induced
        aggregation, [:math:`# s^{-1}`]

    Notes
    -----
    The aggregation efficiency (``agg_efficiency = 1``) was calibrated based on
    the observed lifetime and optical properties from the Contrail Library (COLI)
    database (:cite:`schumannPropertiesIndividualContrails2017`).

    References
    ----------
    - :cite:`schumannPropertiesIndividualContrails2017`
    """
    return (8.0 * agg_efficiency * np.pi * r_ice_vol**2 * terminal_fall_speed) / area_eff


def particle_losses_turbulence(
    width: npt.NDArray[np.float_],
    depth: npt.NDArray[np.float_],
    depth_eff: npt.NDArray[np.float_],
    diffuse_h: npt.NDArray[np.float_],
    diffuse_v: npt.NDArray[np.float_],
    turb_efficiency: float = 0.1,
) -> npt.NDArray[np.float_]:
    """
    Calculate the rate of contrail ice particle losses due to plume-internal turbulence.

    Parameters
    ----------
    width : npt.NDArray[np.float_]
        Contrail width at each waypoint, [:math:`m`]
    depth : npt.NDArray[np.float_]
        Contrail depth at each waypoint, [:math:`m`]
    depth_eff : npt.NDArray[np.float_]
        Effective depth of the contrail plume, [:math:`m`]
    diffuse_h : npt.NDArray[np.float_]
        Horizontal diffusivity, [:math:`m^{2} s^{-1}`]
    diffuse_v : npt.NDArray[np.float_]
        Vertical diffusivity, [:math:`m^{2} s^{-1}`]
    turb_efficiency : float, optional
        Turbulence sublimation efficiency

    Returns
    -------
    npt.NDArray[np.float_]
        Rate of contrail ice particle losses due to plume-internal turbulence, [:math:`# s^{-1}`]

    Notes
    -----
    The turbulence sublimation efficiency (``turb_efficiency = 0.1``) was calibrated
    based on the observed lifetime and optical properties from the Contrail Library (COLI)
    database (:cite:`schumannPropertiesIndividualContrails2017`).

    References
    ----------
    - :cite:`schumannPropertiesIndividualContrails2017`
    """
    inner_term = (diffuse_h / (np.maximum(width, depth)) ** 2) + (diffuse_v / depth_eff**2)
    return turb_efficiency * np.abs(inner_term)


####################
# Optical properties
####################


def contrail_optical_depth(
    r_ice_vol: npt.NDArray[np.float_],
    n_ice_per_m: npt.NDArray[np.float_],
    width: npt.NDArray[np.float_],
) -> npt.NDArray[np.float_]:
    """
    Calculate the contrail optical depth for each waypoint.

    Parameters
    ----------
    r_ice_vol : npt.NDArray[np.float_]
        ice particle volume mean radius, [:math:`m`]
    n_ice_per_m : npt.NDArray[np.float_]
        Number of contrail ice particles per distance, [:math:`m^{-1}`]
    width : npt.NDArray[np.float_]
        Contrail width, [:math:`m`]

    Returns
    -------
    npt.NDArray[np.float_]
        Contrail optical depth
    """
    q_ext = scattering_extinction_efficiency(r_ice_vol)
    tau_contrail = constants.c_r * np.pi * r_ice_vol**2 * (n_ice_per_m / width) * q_ext

    bool_small = r_ice_vol <= 1e-9
    tau_contrail[bool_small] = 0.0
    tau_contrail.clip(min=0.0, out=tau_contrail)
    return tau_contrail


def scattering_extinction_efficiency(r_ice_vol: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
    """
    Calculate the scattering extinction efficiency (``q_ext``) based on Mie-theory.

    Parameters
    ----------
    r_ice_vol : npt.NDArray[np.float_]
        ice particle volume mean radius, [:math:`m`]

    Returns
    -------
    npt.NDArray[np.float_]
        scattering extinction efficiency

    References
    ----------
    - https://en.wikipedia.org/wiki/Mie_scattering
    """
    phase_delay = light_wave_phase_delay(r_ice_vol)
    return 2.0 - (4.0 / phase_delay) * (
        np.sin(phase_delay) - ((1.0 - np.cos(phase_delay)) / phase_delay)
    )


def light_wave_phase_delay(r_ice_vol: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
    """
    Calculate the phase delay of the light wave passing through the contrail ice particle.

    Parameters
    ----------
    r_ice_vol : npt.NDArray[np.float_]
        ice particle volume mean radius, [:math:`m`]

    Returns
    -------
    npt.NDArray[np.float_]
        phase delay of the light wave passing through the contrail ice particle

    References
    ----------
    - https://en.wikipedia.org/wiki/Mie_scattering
    """
    phase_delay = (4.0 * np.pi * (constants.mu_ice - 1.0) / constants.lambda_light) * r_ice_vol
    phase_delay.clip(min=None, max=100.0, out=phase_delay)
    return phase_delay


#######################################################
# Contrail evolution: second-order Runge Kutta scheme
#######################################################
# Notation "t1" implies properties at the start of the time step (before the time integration step)
# Notation "t2" implies properties at the end of the time step (after the time integration step)


def segment_length_ratio(
    seg_length_t1: npt.NDArray[np.float_],
    seg_length_t2: npt.NDArray[np.float_],
) -> npt.NDArray[np.float_]:
    """Calculate the ratio of contrail segment length pre-advection to post-advection.

    Parameters
    ----------
    seg_length_t1 : npt.NDArray[np.float_]
        Segment length of contrail waypoint at the start of the time step, [:math:`m`]
    seg_length_t2 : npt.NDArray[np.float_]
        Segment length of contrail waypoint after time step and advection, [:math:`m`]

    Returns
    -------
    npt.NDArray[np.float_]
        Ratio of segment length before advection to segment length after advection.

    Notes
    -----
    This implementation differs from the original fortran implementation.
    Instead of taking a geometric mean between
    the previous and following segments, a simple ratio is computed.

    For terminal waypoints along a flight trajectory, the associated segment length is 0. In this
    case, the segment ratio is set to 1 (the naive ratio 0 / 0 is undefined). According to CoCiP
    conventions, terminus waypoints are "discontinuous" within the flight trajectory, and will not
    contribute to contrail calculations.

    More broadly, any undefined (nan values, or division by 0) segment ratio is set to 1.
    This convention ensures that the contrail calculations are not affected by undefined
    segment-based properties.

    Presently, the output of this function is only used by :func:`plume_temporal_evolution`
    and :func:`new_ice_particle_number` as a scaling term.

    A `seg_ratio` value of 1 is the same as not applying any scaling in these two functions.
    """
    is_defined = (seg_length_t2 > 0.0) & np.isfinite(seg_length_t1)
    default_value = np.ones_like(seg_length_t1)
    return np.divide(seg_length_t1, seg_length_t2, out=default_value, where=is_defined)


def plume_temporal_evolution(
    width_t1: npt.NDArray[np.float_],
    depth_t1: npt.NDArray[np.float_],
    sigma_yz_t1: npt.NDArray[np.float_],
    dsn_dz_t1: npt.NDArray[np.float_],
    diffuse_h_t1: npt.NDArray[np.float_],
    diffuse_v_t1: npt.NDArray[np.float_],
    seg_ratio: npt.NDArray[np.float_] | float,
    dt: npt.NDArray[np.timedelta64] | np.timedelta64,
    max_contrail_depth: float,
) -> tuple[npt.NDArray[np.float_], npt.NDArray[np.float_], npt.NDArray[np.float_]]:
    """
    Calculate the temporal evolution of the contrail plume parameters.

    Refer to equation (6) of Schumann (2012). See also equations (29), (30), and (31).

    Parameters
    ----------
    width_t1 : npt.NDArray[np.float_]
        contrail width at the start of the time step, [:math:`m`]
    depth_t1 : npt.NDArray[np.float_]
        contrail depth at the start of the time step, [:math:`m`]
    sigma_yz_t1 : npt.NDArray[np.float_]
        sigma_yz governs the contrail plume's temporal evolution at the start of the time step
    dsn_dz_t1 : npt.NDArray[np.float_]
        vertical gradient of the horizontal velocity (wind shear) normal to the contrail axis
        at the start of the time step, [:math:`m s^{-1} / Pa`]::

                           X-----------------------X               X
                                        ^                           |
                                        | (dsn_dz)                  |  <-- (dsn_dz)
                                        |                           |
                                                                    X
    diffuse_h_t1 : npt.NDArray[np.float_]
        horizontal diffusivity at the start of the time step, [:math:`m^{2} s^{-1}`]
    diffuse_v_t1 : npt.NDArray[np.float_]
        vertical diffusivity at the start of the time step, [:math:`m^{2} s^{-1}`]
    seg_ratio : npt.NDArray[np.float_] | float
        Segment length ratio before and after it is advected to the new location.
        See :func:`segment_length_ratio`.
    dt : npt.NDArray[np.timedelta64] | np.timedelta64
        integrate contrails with time steps of dt, [:math:`s`]
    max_contrail_depth: float
        Constrain maximum contrail depth to prevent unrealistic values, [:math:`m`]

    Returns
    -------
    sigma_yy_t2 : npt.NDArray[np.float_]
        The ``yy`` component of convariance matrix, [:math:`m^{2}`]
    sigma_zz_t2 : npt.NDArray[np.float_]
        The ``zz`` component of convariance matrix, [:math:`m^{2}`]
    sigma_yz_t2 : npt.NDArray[np.float_]
        The ``yz`` component of convariance matrix, [:math:`m^{2}`]
    """
    # Convert dt to seconds value and use dtype of other variables
    dtype = np.result_type(width_t1, depth_t1, sigma_yz_t1, dsn_dz_t1, diffuse_h_t1, diffuse_v_t1)
    dt_s = dt / np.timedelta64(1, "s")
    dt_s = dt_s.astype(dtype, copy=False)

    sigma_yy = 0.125 * (width_t1**2)
    sigma_zz = 0.125 * (depth_t1**2)

    # Convert from max_contrail_depth to an upper bound for diffuse_v_t1
    # All three terms involve the diffuse_v_t1 variable, so we need to
    # calculate the max value for diffuse_v_t1 and apply it to all three terms.
    # If we don't do this, we violate the some mathematical constraints of the
    # covariance matrix (positive definite). In particular, for downstream
    # calculations, we required that
    #   sigma_yy_t2 * sigma_zz_t2 - sigma_yz_t2**2 >= 0
    max_sigma_zz = 0.125 * max_contrail_depth**2
    max_diffuse_v = (max_sigma_zz - sigma_zz) / (2.0 * dt_s)
    diffuse_v_t1 = np.minimum(diffuse_v_t1, max_diffuse_v)

    # Avoid some redundant calculations
    dsn_dz_t1_2 = dsn_dz_t1**2
    dt_s_2 = dt_s**2
    dt_s_3 = dt_s * dt_s_2

    # Calculate the return arrays
    sigma_yy_t2 = (
        ((2 / 3) * dsn_dz_t1_2 * diffuse_v_t1 * dt_s_3)
        + (dsn_dz_t1_2 * sigma_zz * dt_s_2)
        + (2.0 * (diffuse_h_t1 + dsn_dz_t1 * sigma_yz_t1) * dt_s)
        + sigma_yy
    ) * (seg_ratio**2)

    sigma_zz_t2 = (2.0 * diffuse_v_t1 * dt_s) + sigma_zz

    sigma_yz_t2 = (
        (dsn_dz_t1 * diffuse_v_t1 * dt_s_2) + (dsn_dz_t1 * sigma_zz * dt_s) + sigma_yz_t1
    ) * seg_ratio

    return sigma_yy_t2, sigma_zz_t2, sigma_yz_t2


def new_contrail_dimensions(
    sigma_yy_t2: npt.NDArray[np.float_],
    sigma_zz_t2: npt.NDArray[np.float_],
) -> tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]]:
    """
    Calculate the new contrail width and depth.

    Parameters
    ----------
    sigma_yy_t2 : npt.NDArray[np.float_]
        element yy, covariance matrix of the Gaussian concentration
        field, Eq. (6) of Schumann (2012)
    sigma_zz_t2 : npt.NDArray[np.float_]
        element zz, covariance matrix of the Gaussian concentration
        field, Eq. (6) of Schumann (2012)

    Returns
    -------
    width_t2 : npt.NDArray[np.float_]
        Contrail width at the end of the time step, [:math:`m`]
    depth_t2 : npt.NDArray[np.float_]
        Contrail depth at the end of the time step, [:math:`m`]
    """
    width_t2 = (8 * sigma_yy_t2) ** 0.5
    depth_t2 = (8 * sigma_zz_t2) ** 0.5
    return width_t2, depth_t2


def new_effective_area_from_sigma(
    sigma_yy: npt.NDArray[np.float_],
    sigma_zz: npt.NDArray[np.float_],
    sigma_yz: npt.NDArray[np.float_],
) -> npt.NDArray[np.float_]:
    """
    Calculate effective cross-sectional area of contrail plume (``area_eff``) from sigma parameters.

    This method calculates the same output as :func`plume_effective_cross_sectional_area`, but
    calculated with different input parameters.

    Parameters
    ----------
    sigma_yy : npt.NDArray[np.float_]
        element yy, covariance matrix of the Gaussian concentration
        field, Eq. (6) of Schumann (2012)
    sigma_zz : npt.NDArray[np.float_]
        element zz, covariance matrix of the Gaussian concentration
        field, Eq. (6) of Schumann (2012)
    sigma_yz : npt.NDArray[np.float_]
        element yz, covariance matrix of the Gaussian concentration
        field, Eq. (6) of Schumann (2012)

    Returns
    -------
    npt.NDArray[np.float_]
        Effective cross-sectional area of the contrail plume (area_eff)
    """
    det_sigma = sigma_yy * sigma_zz - sigma_yz**2
    return 2.0 * np.pi * det_sigma**0.5


def new_ice_water_content(
    iwc_t1: npt.NDArray[np.float_],
    q_t1: npt.NDArray[np.float_],
    q_t2: npt.NDArray[np.float_],
    q_sat_t1: npt.NDArray[np.float_],
    q_sat_t2: npt.NDArray[np.float_],
    mass_plume_t1: npt.NDArray[np.float_],
    mass_plume_t2: npt.NDArray[np.float_],
) -> npt.NDArray[np.float_]:
    """
    Calculate the new contrail ice water content after the time integration step (``iwc_t2``).

    Parameters
    ----------
    iwc_t1 : npt.NDArray[np.float_]
        contrail ice water content, i.e., contrail ice mass per kg of air,
        at the start of the time step, [:math:`kg_{H_{2}O}/kg_{air}`]
    q_t1 : npt.NDArray[np.float_]
        specific humidity for each waypoint at the start of the
        time step, [:math:`kg_{H_{2}O}/kg_{air}`]
    q_t2 : npt.NDArray[np.float_]
        specific humidity for each waypoint at the end of the
        time step, [:math:`kg_{H_{2}O}/kg_{air}`]
    q_sat_t1 : npt.NDArray[np.float_]
        saturation humidity for each waypoint at the start of the
        time step, [:math:`kg_{H_{2}O}/kg_{air}`]
    q_sat_t2 : npt.NDArray[np.float_]
        saturation humidity for each waypoint at the end of the
        time step, [:math:`kg_{H_{2}O}/kg_{air}`]
    mass_plume_t1 : npt.NDArray[np.float_]
        contrail plume mass per unit length at the start of the
        time step, [:math:`kg_{air} m^{-1}`]
    mass_plume_t2 : npt.NDArray[np.float_]
        contrail plume mass per unit length at the end of the
        time step, [:math:`kg_{air} m^{-1}`]

    Returns
    -------
    npt.NDArray[np.float_]
        Contrail ice water content at the end of the time step, [:math:`kg_{ice} kg_{air}^{-1}`]

    Notes
    -----
    (1) The ice water content is fully conservative.
    (2) ``mass_h2o_t2``: the total H2O mass (ice + vapour) per unit of
        contrail plume [Units of kg-H2O/m]
    (3) ``q_sat`` is used to calculate mass_h2o because air inside the
        contrail is assumed to be ice saturated.
    (4) ``(mass_plume_t2 - mass_plume) * q_mean``: contrail absorbs
        (releases) H2O from (to) surrounding air.
    (5) ``iwc_t2 = mass_h2o_t2 / mass_plume_t2 - q_sat_t2``: H2O in the
        gas phase is removed (``- q_sat_t2``).
    """
    q_mean = 0.5 * (q_t1 + q_t2)
    mass_h2o_t1 = mass_plume_t1 * (iwc_t1 + q_sat_t1)
    mass_h2o_t2 = mass_h2o_t1 + (mass_plume_t2 - mass_plume_t1) * q_mean
    iwc_t2 = (mass_h2o_t2 / mass_plume_t2) - q_sat_t2
    iwc_t2.clip(min=0.0, out=iwc_t2)
    return iwc_t2


def new_ice_particle_number(
    n_ice_per_m_t1: npt.NDArray[np.float_],
    dn_dt_agg: npt.NDArray[np.float_],
    dn_dt_turb: npt.NDArray[np.float_],
    seg_ratio: npt.NDArray[np.float_] | float,
    dt: npt.NDArray[np.timedelta64] | np.timedelta64,
) -> npt.NDArray[np.float_]:
    """Calculate the number of ice particles per distance at the end of the time step.

    Parameters
    ----------
    n_ice_per_m_t1 : npt.NDArray[np.float_]
        number of contrail ice particles per distance at the start of
        the time step, [:math:`m^{-1}`]
    dn_dt_agg : npt.NDArray[np.float_]
        rate of ice particle losses due to sedimentation-induced aggregation, [:math:`# s^{-1}`]
    dn_dt_turb : npt.NDArray[np.float_]
        rate of contrail ice particle losses due to plume-internal turbulence, [:math:`# s^{-1}`]
    seg_ratio : npt.NDArray[np.float_] | float
        Segment length ratio before and after it is advected to the new location.
    dt : npt.NDArray[np.timedelta64] | np.timedelta64
        integrate contrails with time steps of dt, [:math:`s`]

    Returns
    -------
    npt.NDArray[np.float_]
        number of ice particles per distance at the end of the time step, [:math:`m^{-1}`]
    """
    # Convert dt to seconds value and use dtype of other variables
    dtype = np.result_type(n_ice_per_m_t1, dn_dt_agg, dn_dt_turb, seg_ratio)
    dt_s = dt / np.timedelta64(1, "s")
    dt_s = dt_s.astype(dtype)

    n_ice_per_m_t1 = np.maximum(n_ice_per_m_t1, 0.0)

    exp_term = np.where(dn_dt_turb * dt_s < 80.0, np.exp(-dn_dt_turb * dt_s), 0.0)

    numerator = dn_dt_turb * n_ice_per_m_t1 * exp_term
    denominator = dn_dt_turb + (dn_dt_agg * n_ice_per_m_t1 * (1 - exp_term))
    n_ice_per_m_t2 = (numerator / denominator) * seg_ratio

    small_loss = (dn_dt_turb * dt_s) < 1e-5  # For small ice particle losses
    denom = 1 + (dn_dt_agg * dt_s * n_ice_per_m_t1)
    n_ice_per_m_t2[small_loss] = n_ice_per_m_t1[small_loss] / denom[small_loss]
    n_ice_per_m_t2.clip(min=0.0, out=n_ice_per_m_t2)
    return n_ice_per_m_t2


########
# Energy Forcing
########
# TODO: This should be moved closer to the radiative forcing calculations


def energy_forcing(
    rf_net_t1: npt.NDArray[np.float_],
    rf_net_t2: npt.NDArray[np.float_],
    width_t1: npt.NDArray[np.float_],
    width_t2: npt.NDArray[np.float_],
    seg_length_t2: npt.NDArray[np.float_] | float,
    dt: npt.NDArray[np.timedelta64] | np.timedelta64,
) -> npt.NDArray[np.float_]:
    """Calculate the contrail energy forcing over time step.

    The contrail energy forcing is calculated as the local contrail net
    radiative forcing (RF', change in energy flux per contrail area) multiplied
    by its width and integrated over its length and lifetime.

    Parameters
    ----------
    rf_net_t1 : npt.NDArray[np.float_]
        local contrail net radiative forcing at the start of the time step, [:math:`W m^{-2}`]
    rf_net_t2 : npt.NDArray[np.float_]
        local contrail net radiative forcing at the end of the time step, [:math:`W m^{-2}`]
    width_t1 : npt.NDArray[np.float_]
        contrail width at the start of the time step, [:math:`m`]
    width_t2 : npt.NDArray[np.float_]
        contrail width at the end of the time step, [:math:`m`]
    seg_length_t2 : npt.NDArray[np.float_] | float
        Segment length of contrail waypoint at the end of the time step, [:math:`m`]
    dt : npt.NDArray[np.timedelta64] | np.timedelta64
        integrate contrails with time steps of dt, [:math:`s`]

    Returns
    -------
    npt.NDArray[np.float_]
        Contrail energy forcing over time step dt, [:math:`J`].
    """
    rad_flux_per_m = mean_radiative_flux_per_m(rf_net_t1, rf_net_t2, width_t1, width_t2)
    energy_flux_per_m = mean_energy_flux_per_m(rad_flux_per_m, dt)
    return energy_flux_per_m * seg_length_t2


def mean_radiative_flux_per_m(
    rf_net_t1: npt.NDArray[np.float_],
    rf_net_t2: npt.NDArray[np.float_],
    width_t1: npt.NDArray[np.float_],
    width_t2: npt.NDArray[np.float_],
) -> npt.NDArray[np.float_]:
    """Calculate the mean radiative flux per length of contrail between two time steps.

    Parameters
    ----------
    rf_net_t1 : npt.NDArray[np.float_]
        local contrail net radiative forcing at the start of the time step, [:math:`W m^{-2}`]
    rf_net_t2 : npt.NDArray[np.float_]
        local contrail net radiative forcing at the end of the time step, [:math:`W m^{-2}`]
    width_t1 : npt.NDArray[np.float_]
        contrail width at the start of the time step, [:math:`m`]
    width_t2 : npt.NDArray[np.float_]
        contrail width at the end of the time step, [:math:`m`]

    Returns
    -------
    npt.NDArray[np.float_]
        Mean radiative flux between time steps, [:math:`W m^{-1}`]
    """
    rad_flux_per_m_t1 = width_t1 * rf_net_t1
    rad_flux_per_m_t2 = width_t2 * rf_net_t2
    return (rad_flux_per_m_t1 + rad_flux_per_m_t2) * 0.5


def mean_energy_flux_per_m(
    rad_flux_per_m: npt.NDArray[np.float_], dt: npt.NDArray[np.timedelta64] | np.timedelta64
) -> npt.NDArray[np.float_]:
    """Calculate the mean energy flux per length of contrail on segment following waypoint.

    Parameters
    ----------
    rad_flux_per_m : npt.NDArray[np.float_]
        Mean radiative flux between time steps for waypoint, [:math:`W m^{-1}`].
        See :func:`mean_radiative_flux_per_m`.
    dt : npt.NDArray[np.timedelta64]
        timedelta of integration timestep for each waypoint.

    Returns
    -------
    npt.NDArray[np.float_]
        Mean energy flux per length of contrail after waypoint, [:math:`J m^{-1}`]

    Notes
    -----
    Implementation differs from original fortran in two ways:

    - Discontinuity is no longer set to 0 (this occurs directly in model :class:`Cocip`)
    - Instead of taking an average of the previous and following segments,
      energy flux is only calculated for the following segment.

    See Also
    --------
    :func:`mean_radiative_flux_per_m`
    """
    dt_s = dt / np.timedelta64(1, "s")
    dt_s = dt_s.astype(rad_flux_per_m.dtype, copy=False)
    return rad_flux_per_m * dt_s
