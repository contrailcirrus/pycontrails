"""Jet aircraft trajectory and performance parameters.

This module includes common functions to calculate jet aircraft trajectory
and performance parameters, including fuel quantities, mass, thrust setting
and propulsion efficiency.
"""

from __future__ import annotations

import logging

import numpy as np
import numpy.typing as npt

from pycontrails.core.flight import FlightPhase
from pycontrails.physics import constants, units
from pycontrails.utils.types import ArrayScalarLike

logger = logging.getLogger(__name__)


# -------------------
# Aircraft performance
# -------------------


def identify_phase_of_flight(rocd: np.ndarray, *, threshold_rocd: float = 100.0) -> FlightPhase:
    """Identify the phase of flight (climb, cruise, descent) for each waypoint.

    Parameters
    ----------
    rocd: np.ndarray
        Rate of climb and descent, [:math:`ft min^{-1}`]
    threshold_rocd: float
        ROCD threshold to identify climb and descent, [:math:`ft min^{-1}`].
        Currently set to 100 ft/min.

    Returns
    -------
    FlightPhase
        Booleans marking if the waypoints are at cruise, climb, or descent

    Notes
    -----
    Flight data derived from ADS-B and radar sources could contain noise leading
    to small changes in altitude and ROCD. Hence, an arbitrary ``threshold_rocd``
    is specified to identify the different phases of flight.
    """
    nan = np.isnan(rocd)
    climb = rocd > threshold_rocd
    descent = rocd < -threshold_rocd
    cruise = ~(nan | climb | descent)
    return FlightPhase(cruise=cruise, climb=climb, descent=descent, nan=nan)


def rate_of_climb_descent(
    dt: npt.NDArray[np.float_], altitude_ft: npt.NDArray[np.float_]
) -> npt.NDArray[np.float_]:
    """Calculate the rate of climb and descent (ROCD).

    Parameters
    ----------
    dt: npt.NDArray[np.float_]
        Time difference between waypoints, [:math:`s`].
        Expected to have numeric `dtype`, not `"timedelta64".
    altitude_ft: npt.NDArray[np.float_]
        Altitude of each waypoint, [:math:`ft`]

    Returns
    -------
    npt.NDArray[np.float_]
        Rate of climb and descent, [:math:`ft min^{-1}`]
    """
    dt_min = dt / 60.0

    out = np.empty_like(altitude_ft)
    out[:-1] = np.diff(altitude_ft) / dt_min[:-1]
    out[-1] = np.nan

    return out


def clip_mach_number(
    true_airspeed: np.ndarray,
    air_temperature: np.ndarray,
    max_mach_number: float,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Compute the Mach number from the true airspeed and ambient temperature.

    This method clips the computed Mach number to the value of `max_mach_number`.

    Parameters
    ----------
    true_airspeed : np.ndarray
        Array of true airspeed, [:math:`m \ s^{-1}`]
    air_temperature : np.ndarray
        Array of ambient temperature, [:math: `K`]
    max_mach_number : float
        Maximum mach number associated to aircraft, [:math: `Ma`]. If no clipping
        is desired, this can be set tp `np.inf`.

    Returns
    -------
    tuple[np.ndarray, np.ndarray] :
        Pair of true airspeed and Mach number arrays. Both are corrected so that
        the Mach numbers are clipped at `max_mach_number`.
    """
    mach_num = units.tas_to_mach_number(true_airspeed, air_temperature)

    is_unrealistic = mach_num > max_mach_number
    if np.any(is_unrealistic):
        msg = (
            f"Unrealistic Mach numbers found. Discovered {np.sum(is_unrealistic)} "
            f"/ {is_unrealistic.size} values exceeding this, the largest of which "
            f"is {np.nanmax(mach_num)}. These are all clipped at {max_mach_number}."
        )
        logger.debug(msg)

    max_tas = units.mach_number_to_tas(max_mach_number, air_temperature)
    adjusted_mach_num = np.where(is_unrealistic, max_mach_number, mach_num)
    adjusted_true_airspeed = np.where(is_unrealistic, max_tas, true_airspeed)

    return adjusted_true_airspeed, adjusted_mach_num


def overall_propulsion_efficiency(
    true_airspeed: np.ndarray,
    F_thrust: np.ndarray,
    fuel_flow: np.ndarray,
    q_fuel: float,
    is_descent: np.ndarray | None,
    threshold: float = 0.5,
) -> np.ndarray:
    r"""Calculate the overall propulsion efficiency (OPE).

    Negative OPE values can occur during the descent phase and is clipped to a
    lower bound of 0, while an upper bound of ``threshold`` is also applied.
    The most efficient engines today do not exceed this value.

    Parameters
    ----------
    true_airspeed: np.ndarray
        True airspeed for each waypoint, [:math:`m s^{-1}`].
    F_thrust: np.ndarray
        Thrust force provided by the engine, [:math:`N`].
    fuel_flow: np.ndarray
        Fuel mass flow rate, [:math:`kg s^{-1}`].
    q_fuel : float
        Lower calorific value (LCV) of fuel, [:math:`J \ kg_{fuel}^{-1}`].
    is_descent : np.ndarray | None
        Boolean array that indicates if a waypoint is in a descent phase.
    threshold : float
        Upper bound for realistic engine efficiency.

    Returns
    -------
    np.ndarray
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


def fuel_burn(fuel_flow: np.ndarray, dt: np.ndarray) -> np.ndarray:
    """Calculate the fuel consumption at each waypoint.

    Parameters
    ----------
    fuel_flow: np.ndarray
        Fuel mass flow rate, [:math:`kg s^{-1}`]
    dt: np.ndarray
        Time difference between waypoints, [:math:`s`]

    Returns
    -------
    np.ndarray
        Fuel consumption at each waypoint, [:math:`kg`]
    """
    return fuel_flow * dt


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


def reserve_fuel_requirements(
    rocd: np.ndarray, fuel_flow: np.ndarray, fuel_burn: np.ndarray
) -> float:
    r"""
    Estimate reserve fuel requirements.

    Parameters
    ----------
    rocd: np.ndarray
        Rate of climb and descent, [:math:`ft \ min^{-1}`]
    fuel_flow: np.ndarray
        Fuel mass flow rate, [:math:`kg \ s^{-1}`].
    fuel_burn: np.ndarray
        Fuel consumption for each waypoint, [:math:`kg`]

    Returns
    -------
    np.ndarray
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
    :func:`identify_phase_of_flight`
    :func:`fuel_burn`
    """
    phase_of_flight = identify_phase_of_flight(rocd)

    # In case flight does not have cruise phase
    is_climb_cruise = phase_of_flight.climb | phase_of_flight.cruise

    # If there are no climb and cruise phase, take the mean
    if not np.all(is_climb_cruise):
        ff_end_of_cruise = np.nanmean(fuel_flow)

    # Take average of final three waypoints
    else:
        ff_end_of_cruise = np.nanmean(fuel_flow[is_climb_cruise][-3:])

    reserve_fuel_1 = ff_end_of_cruise * (90 * 60)
    reserve_fuel_2 = 0.15 * float(np.nansum(fuel_burn))
    return np.maximum(reserve_fuel_1, reserve_fuel_2)


# -------------------
# Aircraft mass
# -------------------


def aircraft_weight(aircraft_mass: np.ndarray) -> np.ndarray:
    """Calculate the aircraft weight at each waypoint.

    Parameters
    ----------
    aircraft_mass : np.ndarray
        Aircraft mass, [:math:`kg`]

    Returns
    -------
    np.ndarray
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
    initial_amass = (
        operating_empty_weight
        + (load_factor * max_payload)
        + (total_fuel_burn + total_reserve_fuel)
    )
    return np.minimum(initial_amass, max_takeoff_weight)


def update_aircraft_mass(
    *,
    operating_empty_weight: float,
    ref_mass: float,
    max_takeoff_weight: float,
    max_payload: float,
    fuel_burn: npt.NDArray[np.float_],
    total_reserve_fuel: float,
    load_factor: None | float = None,
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
    fuel_burn: np.ndarray
        Fuel consumption for each waypoint, [:math:`kg`]
    total_reserve_fuel: float
        Total reserve fuel requirements, [:math:`kg`]
    load_factor: None | float
        Aircraft load factor assumption (between 0 and 1). If None is given, the reference mass
        from the BADA 3 database will be used to get the mass of the specific aircraft type.

    Returns
    -------
    npt.NDArray[np.float_]
        Updated aircraft mass, [:math:`kg`]

    See Also
    --------
    :func:`fuel_burn`
    :func:`reserve_fuel_requirements`
    """
    if load_factor is not None:
        initial_amass = initial_aircraft_mass(
            operating_empty_weight=operating_empty_weight,
            max_takeoff_weight=max_takeoff_weight,
            max_payload=max_payload,
            total_fuel_burn=float(np.nansum(fuel_burn)),
            total_reserve_fuel=total_reserve_fuel,
            load_factor=load_factor,
        )
    else:
        initial_amass = ref_mass

    # Calculate updated aircraft mass for each waypoint
    amass = np.empty_like(fuel_burn)
    amass[0] = initial_amass
    amass[1:] = initial_amass - np.nancumsum(fuel_burn)[:-1]
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
    return T * (1 + ((constants.kappa - 1) / 2) * mach_num**2)


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
    power_term = constants.kappa / (constants.kappa - 1)
    return p * (1 + ((constants.kappa - 1) / 2) * mach_num**2) ** power_term


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
    return (p_comp_inlet * (pressure_ratio - 1) * thrust_setting) + p_comp_inlet


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
    power_term = (constants.kappa - 1) / (constants.kappa * comp_efficiency)
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
    return (afr * constants.c_pd * T_comb_inlet + q_fuel) / (constants.c_p_combustion * (1 + afr))


# --------------------------------------
# Engine thrust force and thrust settings
# --------------------------------------


def thrust_force(
    altitude: np.ndarray,
    true_airspeed: np.ndarray,
    dt: np.ndarray,
    aircraft_mass: np.ndarray,
    F_drag: np.ndarray,
) -> np.ndarray:
    r"""Calculate the thrust force at each waypoint.

    Parameters
    ----------
    altitude : np.ndarray
        Waypoint altitude, [:math:`m`]
    true_airspeed : np.ndarray
        True airspeed, [:math:`m \ s^{-1}`]
    dt : np.ndarray
        Time between waypoints, [:math:`s`]
    aircraft_mass : np.ndarray
        Aircraft mass, [:math:`kg`]
    F_drag : np.ndarray
        Draft force, [:math:`N`]

    Returns
    -------
    np.ndarray
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
    dh_dt[:-1] = np.diff(altitude) / dt[:-1]
    dh_dt[-1] = 0.0
    np.nan_to_num(dh_dt, copy=False)

    dv_dt = np.empty_like(true_airspeed)
    dv_dt[:-1] = np.diff(true_airspeed) / dt[:-1]
    dv_dt[-1] = 0.0
    np.nan_to_num(dv_dt, copy=False)

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


def temperature_ratio(T: np.ndarray) -> np.ndarray:
    """Calculate the ratio of ambient temperature relative to the temperature at mean sea level.

    Parameters
    ----------
    T : np.ndarray
        Air temperature, [:math:`K`]

    Returns
    -------
    np.ndarray
        Ratio of the temperature to the temperature at mean sea-level (MSL).
    """
    return T / constants.T_msl


def pressure_ratio(p: np.ndarray) -> np.ndarray:
    """Calculate the ratio of ambient pressure relative to the surface pressure.

    Parameters
    ----------
    p : np.ndarray
        Air pressure, [:math:`Pa`]

    Returns
    -------
    np.ndarray
        Ratio of the pressure altitude to the surface pressure.
    """
    return p / constants.p_surface


def density_ratio(rho: np.ndarray) -> np.ndarray:
    r"""Calculate the ratio of air density relative to the air density at mean-sea-level.

    Parameters
    ----------
    rho : np.ndarray
        Air density, [:math:`kg \ m^{3}`]

    Returns
    -------
    np.ndarray
        Ratio of the density to the air density at mean sea-level (MSL).
    """
    return rho / constants.rho_msl
