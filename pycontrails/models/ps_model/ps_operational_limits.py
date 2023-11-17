from __future__ import annotations

import numpy as np
import numpy.typing as npt

from pycontrails.physics import constants, jet, units
from pycontrails.utils.types import ArrayOrFloat
from pycontrails.core import flight
from pycontrails.models.ps_model.ps_aircraft_params import PSAircraftEngineParams


# ------------------------
# Operational speed limits
# ------------------------

def maximum_permitted_mach_number_by_altitude(
    altitude_ft: ArrayOrFloat,
    air_pressure: ArrayOrFloat,
    max_mach_num: float,
    p_i_max: float,
    p_inf_co: float, *,
    atm_speed_limit: bool = True,
    buffer: float = 0.02
) -> ArrayOrFloat:
    """
    Calculate maximum permitted Mach number at a given altitude.

    Parameters
    ----------
    altitude_ft : ArrayOrFloat
        Waypoint altitude, [:math: `ft`]
    air_pressure: ArrayOrFloat
        Pressure altitude at each waypoint, [:math:`Pa`]
    max_mach_num: float
        Maximum permitted operational Mach number of the aircraft type.
    p_i_max : float
        Maximum permitted operational impact pressure of the aircraft type, [:math:`Pa`]
    p_inf_co: float
        Crossover pressure altitude, [:math:`Pa`]
    atm_speed_limit: bool
        Apply air traffic management speed limit of 250 knots below 10,000 feet
    buffer: float
        Additional buffer for maximum permitted Mach number.

    Returns
    -------
    ArrayOrFloat
        Maximum permitted Mach number at a given altitude for each waypoint.

    Notes
    -----
    Below 10,000 ft, the ATM speed limit of 250 knots (p_i = 10510 Pa) is imposed. Below the
    crossover altitude, the maximum operational speed is limited by the maximum permitted impact
    pressure. Above the crossover altitude, the maximum operational speed is determined by the
    maximum permitted Mach number of the aircraft type.
    """
    if atm_speed_limit:
        p_i_max = np.where(altitude_ft < 10000.0, 10510.0, p_i_max)

    return np.where(
        air_pressure > p_inf_co,
        2**0.5 * ((1 + (2 / constants.kappa) * (p_i_max / air_pressure))**0.5 - 1)**0.5,
        max_mach_num
    ) + buffer


# ------------------------------
# Thrust requirements and limits
# ------------------------------

def required_thrust_coefficient(
    c_lift: ArrayOrFloat,
    c_drag: ArrayOrFloat,
    true_airspeed: ArrayOrFloat, *,
    rocd: ArrayOrFloat = 300,
) -> ArrayOrFloat:
    """
    Calculate required thrust coefficient to fly the stated true airspeed and climb rate.

    Parameters
    ----------
    c_lift : ArrayOrFloat
        Lift coefficient
    c_drag : ArrayOrFloat
        Total airframe drag coefficient
    true_airspeed : ArrayOrFloat
        True airspeed, [:math:`m \ s^{-1}`]
    rocd : ArrayOrFloat
        Rate of climb and descent over segment, [:math:`ft min^{-1}`]

    Returns
    -------
    ArrayOrFloat
        Required thrust coefficient to fly the stated true airspeed and climb rate

    Notes
    -----
    The maximum "useful" operational altitude is being deemed to have been reached when the
    achievable climb rate drops to about 300 ft/min.
    """
    dh_dt = units.ft_to_m(rocd) / 60
    return c_lift * ((c_drag / c_lift) + (dh_dt / true_airspeed))


def max_available_thrust_coefficient(
    air_temperature: ArrayOrFloat,
    mach_number: ArrayOrFloat,
    c_t_eta_b: ArrayOrFloat,
    atyp_param: PSAircraftEngineParams,
) -> ArrayOrFloat:
    """
    Calculate maximum available thrust coefficient.

    Parameters
    ----------
    air_temperature : ArrayOrFloat
        Ambient temperature at each waypoint, [:math:`K`]
    mach_number : ArrayOrFloat
        Mach number at each waypoint.
    c_t_eta_b : ArrayOrFloat
        Thrust coefficient at maximum overall propulsion efficiency for a given Mach Number.
    atyp_param : AircraftEngineParams
        Extracted aircraft and engine parameters.

    Returns
    -------
    ArrayOrFloat
        Maximum available thrust coefficient that can be supplied by the engines.
    """
    tr_max = _normalised_max_throttle_parameter(
        air_temperature, mach_number, atyp_param.tet_mcc, atyp_param.mec, atyp_param.tec
    )
    c_t_max_over_c_t_eta_b = 1 + 2.5 * (tr_max - 1)
    return c_t_max_over_c_t_eta_b * c_t_eta_b


def _normalised_max_throttle_parameter(
    air_temperature: ArrayOrFloat,
    mach_number: ArrayOrFloat,
    tet_mcc: float,
    mec: float,
    tec: float,
) -> ArrayOrFloat:
    """
    Calculate normalised maximum throttle parameter.

    Parameters
    ----------
    air_temperature : ArrayOrFloat
        Ambient temperature at each waypoint, [:math:`K`]
    mach_number : ArrayOrFloat
        Mach number at each waypoint
    tet_mcc : float
        Turbine entry temperature at maximum continuous climb rating, [:math:`K`]
    mec : float
        Engine constant used to calculate the throttle parameter
    tec : float
        Engine constant used to calculate the throttle parameter

    Returns
    -------
    ArrayOrFloat
        Normalised maximum throttle parameter, `tr_max`.

    Notes
    -----
    The normalised throttle parameter is the ratio of the total temperature of the gas at turbine
    entry to the freestream total temperature, `tet_over_t_inf`, divided by 'tet_over_t_inf' at
    maximum overall propulsion efficiency for a given Mach Number.
    """
    return (tet_mcc / air_temperature) / (
        tec * (1 - 0.53 * (mach_number - mec)**2) * (1 + 0.2 * mach_number**2)
    )


# --------------------
# Aircraft mass limits
# --------------------

def maximum_allowable_aircraft_mass(
    air_pressure: ArrayOrFloat,
    mach_number: ArrayOrFloat,
    mach_num_des: float,
    c_l_do: float,
    wing_surface_area: float,
) -> ArrayOrFloat:
    """
    Calculate maximum allowable aircraft mass for a given altitude and mach number.

    Parameters
    ----------
    air_pressure : ArrayOrFloat
        Ambient pressure, [:math:`Pa`]
    mach_number : ArrayOrFloat
        Mach number at each waypoint
    mach_num_des : float
        Design optimum Mach number where the fuel mass flow rate is at a minimum.
    c_l_do : float
        Design optimum lift coefficient.
    wing_surface_area : float
        Aircraft wing surface area, [:math:`m^2`]

    Returns
    -------
    ArrayOrFloat
        Maximum allowable aircraft mass, [:math:`kg`]
    """
    c_l_maxu = maximum_usable_lift_coefficient(mach_number, mach_num_des, c_l_do)
    return (1 / constants.g) * (
        c_l_maxu * 0.5 * constants.kappa * air_pressure * mach_number ** 2 * wing_surface_area
    )


def maximum_usable_lift_coefficient(
    mach_number: ArrayOrFloat,
    mach_num_des: float,
    c_l_do: float
) -> ArrayOrFloat:
    """
    Calculate maximum usable lift coefficient.

    Parameters
    ----------
    mach_number : ArrayOrFloat
        Mach number at each waypoint
    mach_num_des : float
        Design optimum Mach number where the fuel mass flow rate is at a minimum.
    c_l_do : float
        Design optimum lift coefficient.

    Returns
    -------
    ArrayOrFloat
        Maximum usable lift coefficient.
    """
    m_over_m_des = mach_number / mach_num_des
    c_l_maxu_over_c_l_do = np.where(
        m_over_m_des < 0.70,
        1.8 - 0.024 * m_over_m_des - 0.824 * m_over_m_des ** 2,
        13.272 - 42.262 * m_over_m_des + 49.883 * m_over_m_des ** 2 - 19.683 * m_over_m_des ** 3
    )
    return c_l_maxu_over_c_l_do * c_l_do


# ----------------
# Fuel flow limits
# ----------------
def correct_fuel_flow(
    fuel_flow: ArrayOrFloat,
    altitude_ft: ArrayOrFloat,
    air_temperature: ArrayOrFloat,
    air_pressure: ArrayOrFloat,
    mach_number: ArrayOrFloat,
    fuel_flow_idle_sls: float,
    fuel_flow_max_sls: float,
    flight_phase: npt.NDArray[np.uint8] | flight.FlightPhase,
) -> ArrayOrFloat:
    r"""Correct fuel mass flow rate to ensure that they are within operational limits.

    Parameters
    ----------
    fuel_flow : ArrayOrFloat
        Fuel mass flow rate, [:math:`kg s^{-1}`]
    altitude_ft : ArrayOrFloat
        Waypoint altitude, [:math: `ft`]
    air_temperature : ArrayOrFloat
        Ambient temperature at each waypoint, [:math:`K`]
    air_pressure : ArrayOrFloat
        Ambient pressure, [:math:`Pa`]
    mach_number : ArrayOrFloat
        Mach number at each waypoint
    fuel_flow_idle_sls : float
        Fuel mass flow rate under engine idle and sea level static conditions, [:math:`kg \ s^{-1}`]
    fuel_flow_max_sls : float
        Fuel mass flow rate at take-off and sea level static conditions, [:math:`kg \ s^{-1}`]
    flight_phase : npt.NDArray[np.uint8] | flight.FlightPhase
        Phase state of each waypoint.

    Returns
    -------
    ArrayOrFloat
        Corrected fuel mass flow rate, [:math:`kg \ s^{-1}`]
    """
    min_fuel_flow = jet.minimum_fuel_flow_rate_at_cruise(fuel_flow_idle_sls, altitude_ft)
    max_fuel_flow = jet.equivalent_fuel_flow_rate_at_cruise(
        fuel_flow_max_sls,
        (air_temperature / constants.T_msl),
        (air_pressure / constants.p_surface),
        mach_number,
    )

    # Account for descent conditions
    # Assume max_fuel_flow at descent is not more than 30% of fuel_flow_max_sls
    # We need this assumption because PTF files are not available in the PS model.
    descent = flight_phase == flight.FlightPhase.DESCENT
    max_fuel_flow[descent] = 0.3 * fuel_flow_max_sls
    return np.clip(fuel_flow, min_fuel_flow, max_fuel_flow)

