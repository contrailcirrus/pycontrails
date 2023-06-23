"""Jet aircraft trajectory and performance parameters.

This module includes common functions to calculate jet aircraft trajectory
and performance parameters, including fuel quantities, mass, thrust setting
and propulsion efficiency.
"""

from __future__ import annotations

import logging

import numpy as np
import numpy.typing as npt

from pycontrails.core import flight
from pycontrails.physics import constants, units
from pycontrails.utils.types import ArrayScalarLike

logger = logging.getLogger(__name__)


# -------------------
# Aircraft performance
# -------------------


def acceleration(
    true_airspeed: npt.NDArray[np.float_], segment_duration: npt.NDArray[np.float_]
) -> npt.NDArray[np.float_]:
    r"""Calculate the acceleration/deceleration at each waypoint.

    Parameters
    ----------
    true_airspeed : npt.NDArray[np.float_]
        True airspeed, [:math:`m \ s^{-1}`]
    segment_duration : npt.NDArray[np.float_]
        Time difference between waypoints, [:math:`s`]

    Returns
    -------
    npt.NDArray[np.float_]
        Acceleration/deceleration, [:math:`m \ s^{-2}`]

    See Also
    --------
    :func:`flight.segment_duration`
    """
    dv_dt = np.empty_like(true_airspeed)
    dv_dt[:-1] = np.diff(true_airspeed) / segment_duration[:-1]
    dv_dt[-1] = 0.0
    np.nan_to_num(dv_dt, copy=False)
    return dv_dt


def climb_descent_angle(
    true_airspeed: npt.NDArray[np.float_], rocd: npt.NDArray[np.float_]
) -> npt.NDArray[np.float_]:
    r"""Calculate angle between the horizontal plane and the actual flight path.

    Parameters
    ----------
    true_airspeed : npt.NDArray[np.float_]
        True airspeed, [:math:`m \ s^{-1}`]
    rocd : npt.NDArray[np.float_]
        Rate of climb/descent, [:math:`ft min^{-1}`]

    Returns
    -------
    npt.NDArray[np.float_]
        Climb (positive value) or descent (negative value) angle, [:math:`\deg`]

    See Also
    --------
    :func:`flight.segment_rocd`
    :func:`flight.segment_true_airspeed`
    """
    rocd_ms = units.ft_to_m(rocd) / 60.0
    sin_theta = rocd_ms / true_airspeed
    return units.radians_to_degrees(np.arcsin(sin_theta))


def clip_mach_number(
    true_airspeed: npt.NDArray[np.float_],
    air_temperature: npt.NDArray[np.float_],
    max_mach_number: float,
) -> tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]]:
    r"""Compute the Mach number from the true airspeed and ambient temperature.

    This method clips the computed Mach number to the value of ``max_mach_number``.

    If no mach number exceeds ``max_mach_number``, the original array ``true_airspeed``
    and the computed Mach number are returned.

    Parameters
    ----------
    true_airspeed : npt.NDArray[np.float_]
        Array of true airspeed, [:math:`m \ s^{-1}`]
    air_temperature : npt.NDArray[np.float_]
        Array of ambient temperature, [:math: `K`]
    max_mach_number : float
        Maximum mach number associated to aircraft, [:math: `Ma`]. If no clipping
        is desired, this can be set tp `np.inf`.

    Returns
    -------
    true_airspeed : npt.NDArray[np.float_]
        Array of true airspeed, [:math:`m \ s^{-1}`]. All values are clipped at
        ``max_mach_number``.
    mach_num : npt.NDArray[np.float_]
        Array of Mach numbers, [:math:`Ma`]. All values are clipped at
        ``max_mach_number``.
    """
    mach_num = units.tas_to_mach_number(true_airspeed, air_temperature)

    is_unrealistic = mach_num > max_mach_number
    if not np.any(is_unrealistic):
        return true_airspeed, mach_num

    msg = (
        f"Unrealistic Mach numbers found. Discovered {np.sum(is_unrealistic)} / "
        f"{is_unrealistic.size} values exceeding this, the largest of which "
        f"is {np.nanmax(mach_num):.4f}. These are all clipped at {max_mach_number}."
    )
    logger.debug(msg)

    max_tas = units.mach_number_to_tas(max_mach_number, air_temperature)
    adjusted_mach_num = np.where(is_unrealistic, max_mach_number, mach_num)
    adjusted_true_airspeed = np.where(is_unrealistic, max_tas, true_airspeed)

    return adjusted_true_airspeed, adjusted_mach_num


def overall_propulsion_efficiency(
    true_airspeed: npt.NDArray[np.float_],
    F_thrust: npt.NDArray[np.float_],
    fuel_flow: npt.NDArray[np.float_],
    q_fuel: float,
    is_descent: npt.NDArray[np.bool_] | None,
    threshold: float = 0.5,
) -> npt.NDArray[np.float_]:
    r"""Calculate the overall propulsion efficiency (OPE).

    Negative OPE values can occur during the descent phase and is clipped to a
    lower bound of 0, while an upper bound of ``threshold`` is also applied.
    The most efficient engines today do not exceed this value.

    Parameters
    ----------
    true_airspeed: npt.NDArray[np.float_]
        True airspeed for each waypoint, [:math:`m s^{-1}`].
    F_thrust: npt.NDArray[np.float_]
        Thrust force provided by the engine, [:math:`N`].
    fuel_flow: npt.NDArray[np.float_]
        Fuel mass flow rate, [:math:`kg s^{-1}`].
    q_fuel : float
        Lower calorific value (LCV) of fuel, [:math:`J \ kg_{fuel}^{-1}`].
    is_descent : npt.NDArray[np.float_] | None
        Boolean array that indicates if a waypoint is in a descent phase.
    threshold : float
        Upper bound for realistic engine efficiency.

    Returns
    -------
    npt.NDArray[np.float_]
        Overall propulsion efficiency (OPE)

    References
    ----------
    - :cite:`schumannConditionsContrailFormation1996`
    - :cite:`cumpstyJetPropulsion2015`
    """
    ope = (F_thrust * true_airspeed) / (fuel_flow * q_fuel)
    if is_descent is not None:
        ope[is_descent] = 0.0

    n_unrealistic = np.sum(ope > threshold)
    if n_unrealistic:
        logger.debug(
            "Found %s engine efficiency values exceeding %s. These are clipped.",
            n_unrealistic,
            threshold,
        )
    ope.clip(0.0, threshold, out=ope)  # clip in place
    return ope


# -------------------
# Aircraft fuel quantities
# -------------------


def fuel_burn(
    fuel_flow: npt.NDArray[np.float_], segment_duration: npt.NDArray[np.float_]
) -> npt.NDArray[np.float_]:
    """Calculate the fuel consumption at each waypoint.

    Parameters
    ----------
    fuel_flow: npt.NDArray[np.float_]
        Fuel mass flow rate, [:math:`kg s^{-1}`]
    segment_duration: npt.NDArray[np.float_]
        Time difference between waypoints, [:math:`s`]

    Returns
    -------
    npt.NDArray[np.float_]
        Fuel consumption at each waypoint, [:math:`kg`]
    """
    return fuel_flow * segment_duration


def equivalent_fuel_flow_rate_at_sea_level(
    fuel_flow_cruise: npt.NDArray[np.float_],
    theta_amb: npt.NDArray[np.float_],
    delta_amb: npt.NDArray[np.float_],
    mach_num: npt.NDArray[np.float_],
) -> npt.NDArray[np.float_]:
    r"""Convert fuel mass flow rate at cruise conditions to equivalent flow rate at sea level.

    Refer to Eq. (40) in :cite:`duboisFuelFlowMethod22006`.

    Parameters
    ----------
    fuel_flow_cruise : npt.NDArray[np.float_]
        Fuel mass flow rate per engine, [:math:`kg s^{-1}`]
    theta_amb : npt.NDArray[np.float_]
        Ratio of the ambient temperature to the temperature at mean sea-level.
    delta_amb : npt.NDArray[np.float_]
        Ratio of the pressure altitude to the surface pressure.
    mach_num : npt.NDArray[np.float_]
        Mach number, [:math: `Ma`]

    Returns
    -------
    npt.NDArray[np.float_]
        Estimate of fuel flow per engine at sea level, [:math:`kg \ s^{-1}`].

    References
    ----------
    - :cite:`duboisFuelFlowMethod22006`
    """
    return fuel_flow_cruise * (theta_amb**3.8 / delta_amb) * np.exp(0.2 * mach_num**2)


def equivalent_fuel_flow_rate_at_cruise(
    fuel_flow_sls: npt.NDArray[np.float_] | float,
    theta_amb: npt.NDArray[np.float_],
    delta_amb: npt.NDArray[np.float_],
    mach_num: npt.NDArray[np.float_],
) -> npt.NDArray[np.float_]:
    r"""Convert fuel mass flow rate at sea level to equivalent fuel flow rate at cruise conditions.

    Refer to Eq. (40) in :cite:`duboisFuelFlowMethod22006`.

    Parameters
    ----------
    fuel_flow_sls : npt.NDArray[np.float_] | float
        Fuel mass flow rate, [:math:`kg s^{-1}`]
    theta_amb : npt.NDArray[np.float_]
        Ratio of the ambient temperature to the temperature at mean sea-level.
    delta_amb : npt.NDArray[np.float_]
        Ratio of the pressure altitude to the surface pressure.
    mach_num : npt.NDArray[np.float_]
        Mach number

    Returns
    -------
    npt.NDArray[np.float_]
        Estimate of fuel mass flow rate at sea level, [:math:`kg \ s^{-1}`]

    References
    ----------
    - :cite:`duboisFuelFlowMethod22006`
    """
    return fuel_flow_sls / ((theta_amb**3.8 / delta_amb) * np.exp(0.2 * mach_num**2))


def minimum_fuel_flow_rate_at_cruise(
    fuel_flow_idle_sls: float, altitude_ft: npt.NDArray[np.float_]
) -> npt.NDArray[np.float_]:
    r"""Calculate minimum fuel mass flow rate at cruise conditions.

    Parameters
    ----------
    fuel_flow_idle_sls : float
        Fuel mass flow rate under engine idle and sea level static conditions, [:math:`kg \ s^{-1}`]
    altitude_ft : npt.NDArray[np.float_]
        Waypoint altitude, [:math: `ft`]

    Returns
    -------
    npt.NDArray[np.float_]
        Minimum fuel mass flow rate at cruise conditions, [:math:`kg \ s^{-1}`]
    """
    x = altitude_ft / 10000.0
    return fuel_flow_idle_sls * (1.0 - 0.178 * x + 0.0085 * x**2)


def reserve_fuel_requirements(
    rocd: npt.NDArray[np.float_],
    altitude_ft: npt.NDArray[np.float_],
    fuel_flow: npt.NDArray[np.float_],
    fuel_burn: npt.NDArray[np.float_],
) -> float:
    r"""
    Estimate reserve fuel requirements.

    Parameters
    ----------
    rocd: npt.NDArray[np.float_]
        Rate of climb and descent, [:math:`ft \ min^{-1}`]
    altitude_ft: npt.NDArray[np.float_]
        Altitude, [:math:`ft`]
    fuel_flow: npt.NDArray[np.float_]
        Fuel mass flow rate, [:math:`kg \ s^{-1}`].
    fuel_burn: npt.NDArray[np.float_]
        Fuel consumption for each waypoint, [:math:`kg`]

    Returns
    -------
    npt.NDArray[np.float_]
        Reserve fuel requirements, [:math:`kg`]

    References
    ----------
    - :cite:`wasiukAircraftPerformanceModel2015`

    Notes
    -----
    The real-world calculation of the reserve fuel requirements is highly complex
    (refer to Section 2.3.3 of :cite:`wasiukAircraftPerformanceModel2015`).
    This implementation is simplified by taking the maximum between the following two conditions:

    1. Fuel required to fly +90 minutes at the main cruise altitude at the end of the
       cruise aircraft weight.
    2. Uplift the total fuel consumption for the flight by +15%

    See Also
    --------
    :func:`flight.segment_phase`
    :func:`fuel_burn`
    """
    segment_phase = flight.segment_phase(rocd, altitude_ft)

    is_cruise = (segment_phase == flight.FlightPhase.CRUISE) & np.isfinite(fuel_flow)

    # If there is no cruise phase, take the mean over the whole flight
    if not np.any(is_cruise):
        ff_end_of_cruise = np.nanmean(fuel_flow).item()

    # Otherwise, take the average of the final 10 waypoints
    else:
        ff_end_of_cruise = np.mean(fuel_flow[is_cruise][-10:]).item()

    reserve_fuel_1 = (90.0 * 60.0) * ff_end_of_cruise  # 90 minutes at cruise fuel flow
    reserve_fuel_2 = 0.15 * np.nansum(fuel_burn).item()  # 15% uplift on total fuel burn

    return max(reserve_fuel_1, reserve_fuel_2)


# -------------
# Aircraft mass
# -------------


def aircraft_weight(aircraft_mass: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
    """Calculate the aircraft weight at each waypoint.

    Parameters
    ----------
    aircraft_mass : npt.NDArray[np.float_]
        Aircraft mass, [:math:`kg`]

    Returns
    -------
    npt.NDArray[np.float_]
        Aircraft weight, [:math:`N`]
    """
    return aircraft_mass * constants.g


def initial_aircraft_mass(
    *,
    operating_empty_weight: float,
    max_takeoff_weight: float,
    max_payload: float,
    total_fuel_burn: float,
    total_reserve_fuel: float,
    load_factor: float,
) -> float:
    """Estimate initial aircraft mass as a function of load factor and fuel requirements.

    This function uses the following equation::

        TOM = OEM + PM + FM_nc + TFM
            = OEM + LF * MPM + FM_nc + TFM

    where:
    - TOM is the aircraft take-off mass
    - OEM is the aircraft operating empty weight
    - PM is the payload mass
    - FM_nc is the mass of the fuel not consumed
    - TFM is the trip fuel mass
    - LF is the load factor
    - MPM is the maximum payload mass

    Parameters
    ----------
    operating_empty_weight: float
        Aircraft operating empty weight, i.e. the basic weight of an aircraft including
        the crew and necessary equipment, but excluding usable fuel and payload, [:math:`kg`]
    max_takeoff_weight: float
        Aircraft maximum take-off weight, [:math:`kg`]
    max_payload: float
        Aircraft maximum payload, [:math:`kg`]
    total_fuel_burn: float
        Total fuel consumption for the flight, obtained from prior iterations, [:math:`kg`]
    total_reserve_fuel: float
        Total reserve fuel requirements, [:math:`kg`]
    load_factor: float
        Aircraft load factor assumption (between 0 and 1)

    Returns
    -------
    float
        Aircraft mass at the initial waypoint, [:math:`kg`]

    References
    ----------
    - :cite:`wasiukAircraftPerformanceModel2015`

    See Also
    --------
    :func:`reserve_fuel_requirements`
    """
    tom = operating_empty_weight + load_factor * max_payload + total_fuel_burn + total_reserve_fuel
    return min(tom, max_takeoff_weight)


def update_aircraft_mass(
    *,
    operating_empty_weight: float,
    max_takeoff_weight: float,
    max_payload: float,
    fuel_burn: npt.NDArray[np.float_],
    total_reserve_fuel: float,
    load_factor: float,
    takeoff_mass: float | None,
) -> npt.NDArray[np.float_]:
    """Update aircraft mass based on the simulated total fuel consumption.

    Used internally for finding aircraft mass iteratively.

    Parameters
    ----------
    operating_empty_weight: float
        Aircraft operating empty weight, i.e. the basic weight of an aircraft including
        the crew and necessary equipment, but excluding usable fuel and payload, [:math:`kg`].
    ref_mass: float
        Aircraft reference mass, [:math:`kg`].
    max_takeoff_weight: float
        Aircraft maximum take-off weight, [:math:`kg`].
    max_payload: float
        Aircraft maximum payload, [:math:`kg`]
    fuel_burn: npt.NDArray[np.float_]
        Fuel consumption for each waypoint, [:math:`kg`]
    total_reserve_fuel: float
        Total reserve fuel requirements, [:math:`kg`]
    load_factor: float
        Aircraft load factor assumption (between 0 and 1). This is the ratio of the
        actual payload weight to the maximum payload weight.
    takeoff_mass: float | None
        Initial aircraft mass, [:math:`kg`]. If None, the initial mass is calculated
        using :func:`initial_aircraft_mass`. If supplied, all other parameters except
        ``fuel_burn`` are ignored.

    Returns
    -------
    npt.NDArray[np.float_]
        Updated aircraft mass, [:math:`kg`]

    See Also
    --------
    :func:`fuel_burn`
    :func:`reserve_fuel_requirements`
    :func:`initial_aircraft_mass`
    """
    if takeoff_mass is None:
        takeoff_mass = initial_aircraft_mass(
            operating_empty_weight=operating_empty_weight,
            max_takeoff_weight=max_takeoff_weight,
            max_payload=max_payload,
            total_fuel_burn=np.nansum(fuel_burn).item(),
            total_reserve_fuel=total_reserve_fuel,
            load_factor=load_factor,
        )

    # Calculate updated aircraft mass for each waypoint
    amass = np.empty_like(fuel_burn)
    amass[0] = takeoff_mass
    amass[1:] = takeoff_mass - np.nancumsum(fuel_burn)[:-1]

    return amass


# ------------------------------------------------------------------
# Temperature and pressure at different sections of the jet engine
# ------------------------------------------------------------------


def compressor_inlet_temperature(T: ArrayScalarLike, mach_num: ArrayScalarLike) -> ArrayScalarLike:
    """Calculate compressor inlet temperature for Jet engine, :math:`T_{2}`.

    Parameters
    ----------
    T : ArrayScalarLike
        Ambient temperature, [:math:`K`]
    mach_num : ArrayScalarLike
        Mach number

    Returns
    -------
    ArrayScalarLike
        Compressor inlet temperature, [:math:`K`]

    References
    ----------
    - :cite:`stettlerGlobalCivilAviation2013`
    - :cite:`cumpstyJetPropulsion2015`
    """
    return T * (1.0 + ((constants.kappa - 1.0) / 2.0) * mach_num**2)


def compressor_inlet_pressure(p: ArrayScalarLike, mach_num: ArrayScalarLike) -> ArrayScalarLike:
    """Calculate compressor inlet pressure for Jet engine, :math:`P_{2}`.

    Parameters
    ----------
    p : ArrayScalarLike
        Ambient pressure, [:math:`Pa`]
    mach_num : ArrayScalarLike
        Mach number

    Returns
    -------
    ArrayScalarLike
        Compressor inlet pressure, [:math:`Pa`]

    References
    ----------
    - :cite:`stettlerGlobalCivilAviation2013`
    - :cite:`cumpstyJetPropulsion2015`
    """
    power_term = constants.kappa / (constants.kappa - 1.0)
    return p * (1.0 + ((constants.kappa - 1.0) / 2.0) * mach_num**2) ** power_term


def combustor_inlet_pressure(
    pressure_ratio: float,
    p_comp_inlet: ArrayScalarLike,
    thrust_setting: ArrayScalarLike,
) -> ArrayScalarLike:
    """Calculate combustor inlet pressure, :math:`P_{3}`.

    Parameters
    ----------
    pressure_ratio : float
        Engine pressure ratio, unitless
    p_comp_inlet : ArrayScalarLike
        Compressor inlet pressure, [:math:`Pa`]
    thrust_setting : ArrayScalarLike
        Engine thrust setting, unitless

    Returns
    -------
    ArrayScalarLike
        Combustor inlet pressure, [:math:`Pa`]

    References
    ----------
    - :cite:`stettlerGlobalCivilAviation2013`
    - :cite:`cumpstyJetPropulsion2015`
    """
    return (p_comp_inlet * (pressure_ratio - 1.0) * thrust_setting) + p_comp_inlet


def combustor_inlet_temperature(
    comp_efficiency: float,
    T_comp_inlet: ArrayScalarLike,
    p_comp_inlet: ArrayScalarLike,
    p_comb_inlet: ArrayScalarLike,
) -> ArrayScalarLike:
    """Calculate combustor inlet temperature, :math:`T_{3}`.

    Parameters
    ----------
    comp_efficiency : float
        Engine compressor efficiency, [:math:`0 - 1`]
    T_comp_inlet : ArrayScalarLike
        Compressor inlet temperature, [:math:`K`]
    p_comp_inlet : ArrayScalarLike
        Compressor inlet pressure, [:math:`Pa`]
    p_comb_inlet : ArrayScalarLike
        Compressor inlet pressure, [:math:`Pa`]

    Returns
    -------
    ArrayScalarLike
        Combustor inlet temperature, [:math:`K`]

    References
    ----------
    - :cite:`stettlerGlobalCivilAviation2013`
    - :cite:`cumpstyJetPropulsion2015`
    """
    power_term = (constants.kappa - 1.0) / (constants.kappa * comp_efficiency)
    return T_comp_inlet * (p_comb_inlet / p_comp_inlet) ** power_term


def turbine_inlet_temperature(
    afr: ArrayScalarLike, T_comb_inlet: ArrayScalarLike, q_fuel: float
) -> ArrayScalarLike:
    r"""Calculate turbine inlet temperature, :math:`T_{4}`.

    Parameters
    ----------
    afr : ArrayScalarLike
        Air-to-fuel ratio, unitless
    T_comb_inlet : ArrayScalarLike
        Combustor inlet temperature, [:math:`K`]
    q_fuel : float
        Lower calorific value (LCV) of fuel, :math:`[J \ kg_{fuel}^{-1}]`

    Returns
    -------
    ArrayScalarLike
        Tubrine inlet temperature, [:math:`K`]

    References
    ----------
    - :cite:`cumpstyJetPropulsion2015`
    """
    return (afr * constants.c_pd * T_comb_inlet + q_fuel) / (constants.c_p_combustion * (1.0 + afr))


# --------------------------------------
# Engine thrust force and thrust settings
# --------------------------------------


def thrust_force(
    altitude: npt.NDArray[np.float_],
    true_airspeed: npt.NDArray[np.float_],
    segment_duration: npt.NDArray[np.float_],
    aircraft_mass: npt.NDArray[np.float_],
    F_drag: npt.NDArray[np.float_],
) -> npt.NDArray[np.float_]:
    r"""Calculate the thrust force at each waypoint.

    Parameters
    ----------
    altitude : npt.NDArray[np.float_]
        Waypoint altitude, [:math:`m`]
    true_airspeed : npt.NDArray[np.float_]
        True airspeed, [:math:`m \ s^{-1}`]
    segment_duration : npt.NDArray[np.float_]
        Time difference between waypoints, [:math:`s`]
    aircraft_mass : npt.NDArray[np.float_]
        Aircraft mass, [:math:`kg`]
    F_drag : npt.NDArray[np.float_]
        Draft force, [:math:`N`]

    Returns
    -------
    npt.NDArray[np.float_]
        Thrust force, [:math:`N`]

    References
    ----------
    - :cite:`eurocontrolUSERMANUALBASE2010`

    Notes
    -----
    The model balances the rate of forces acting on the aircraft
    with the rate in increase in energy.

    This estimate of thrust force is used in the BADA Total-Energy Model (Eq. 3.2-1).

    Negative thrust must be corrected.
    """
    dh_dt = np.empty_like(altitude)
    dh_dt[:-1] = np.diff(altitude) / segment_duration[:-1]
    dh_dt[-1] = 0.0
    np.nan_to_num(dh_dt, copy=False)

    dv_dt = acceleration(true_airspeed, segment_duration)

    return (
        F_drag
        + (aircraft_mass * constants.g * dh_dt + aircraft_mass * true_airspeed * dv_dt)
        / true_airspeed
    )


def thrust_setting_nd(
    true_airspeed: ArrayScalarLike,
    thrust_setting: ArrayScalarLike,
    T: ArrayScalarLike,
    p: ArrayScalarLike,
    pressure_ratio: float,
    q_fuel: float,
    *,
    comp_efficiency: float = 0.9,
    cruise: bool = False,
) -> ArrayScalarLike:
    r"""Calculate the non-dimensionalized thrust setting of a Jet engine.

    Result is in terms of the ratio of turbine inlet to the
    compressor inlet temperature (t4_t2)

    Parameters
    ----------
    true_airspeed : ArrayScalarLike
        True airspeed, [:math:`m \ s^{-1}`]
    thrust_setting : ArrayScalarLike
        Engine thrust setting, unitless
    T : ArrayScalarLike
        Ambient temperature, [:math:`K`]
    p : ArrayScalarLike
        Ambient pressure, [:math:`Pa`]
    pressure_ratio : float
        Engine pressure ratio, unitless
    q_fuel : float
        Lower calorific value (LCV) of fuel, :math:`[J \ kg_{fuel}^{-1}]`
    comp_efficiency : float, optional
        Engine compressor efficiency, [:math:`0 - 1`].
        Defaults to 0.9
    cruise : bool, optional
        Defaults to False

    Returns
    -------
    ArrayScalarLike
        Ratio of turbine inlet to the compressor inlet temperature, unitless

    References
    ----------
    - :cite:`cumpstyJetPropulsion2015`
    - :cite:`teohAviationContrailClimate2022`
    """
    mach_num = units.tas_to_mach_number(true_airspeed, T)
    T_compressor_inlet = compressor_inlet_temperature(T, mach_num)
    p_compressor_inlet = compressor_inlet_pressure(p, mach_num)
    p_combustor_inlet = combustor_inlet_pressure(pressure_ratio, p_compressor_inlet, thrust_setting)
    T_combustor_inlet = combustor_inlet_temperature(
        comp_efficiency, T_compressor_inlet, p_compressor_inlet, p_combustor_inlet
    )
    afr = air_to_fuel_ratio(thrust_setting, cruise=cruise, T_compressor_inlet=T_compressor_inlet)
    T_turbine_inlet = turbine_inlet_temperature(afr, T_combustor_inlet, q_fuel)
    return T_turbine_inlet / T_compressor_inlet


def air_to_fuel_ratio(
    thrust_setting: ArrayScalarLike,
    *,
    cruise: bool = False,
    T_compressor_inlet: None | ArrayScalarLike = None,
) -> ArrayScalarLike:
    """Calculate air-to-fuel ratio from thrust setting.

    Parameters
    ----------
    thrust_setting : ArrayScalarLike
        Engine thrust setting, unitless
    cruise : bool
        Estimate thrust setting for cruise conditions. Defaults to False.
    T_compressor_inlet : None | ArrayScalarLike
        Compressor inlet temperature, [:math:`K`]
        Required if ``cruise`` is True.
        Defaults to None

    Returns
    -------
    ArrayScalarLike
        Air-to-fuel ratio, unitless

    References
    ----------
    - :cite:`cumpstyJetPropulsion2015`
    - AFR equation from :cite:`stettlerGlobalCivilAviation2013`
    - Scaling factor to cruise from Eq. (30) of :cite:`duboisFuelFlowMethod22006`

    """
    afr = (0.0121 * thrust_setting + 0.008) ** (-1)

    if not cruise:
        return afr

    if T_compressor_inlet is None:
        raise ValueError("`T_compressor_inlet` is required when `cruise` is True")

    return afr * (T_compressor_inlet / constants.T_msl)


# -------------------
# Atmospheric ratios
# -------------------


def temperature_ratio(T: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
    """Calculate the ratio of ambient temperature relative to the temperature at mean sea level.

    Parameters
    ----------
    T : npt.NDArray[np.float_]
        Air temperature, [:math:`K`]

    Returns
    -------
    npt.NDArray[np.float_]
        Ratio of the temperature to the temperature at mean sea-level (MSL).
    """
    return T / constants.T_msl


def pressure_ratio(p: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
    """Calculate the ratio of ambient pressure relative to the surface pressure.

    Parameters
    ----------
    p : npt.NDArray[np.float_]
        Air pressure, [:math:`Pa`]

    Returns
    -------
    npt.NDArray[np.float_]
        Ratio of the pressure altitude to the surface pressure.
    """
    return p / constants.p_surface


def density_ratio(rho: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
    r"""Calculate the ratio of air density relative to the air density at mean-sea-level.

    Parameters
    ----------
    rho : npt.NDArray[np.float_]
        Air density, [:math:`kg \ m^{3}`]

    Returns
    -------
    npt.NDArray[np.float_]
        Ratio of the density to the air density at mean sea-level (MSL).
    """
    return rho / constants.rho_msl
