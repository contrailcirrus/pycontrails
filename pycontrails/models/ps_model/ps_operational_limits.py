"""Support for calculating operational limits of the PS model."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import scipy.optimize

from pycontrails.core import flight
from pycontrails.models.ps_model.ps_aircraft_params import PSAircraftEngineParams
from pycontrails.physics import constants, jet, units
from pycontrails.utils.types import ArrayOrFloat

# ------------------------
# Operational speed limits
# ------------------------


def max_mach_number_by_altitude(
    altitude_ft: ArrayOrFloat,
    air_pressure: ArrayOrFloat,
    max_mach_num: float,
    p_i_max: float,
    p_inf_co: float,
    *,
    atm_speed_limit: bool = True,
    buffer: float = 0.02,
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
    Below 10,000 ft, the ATM speed limit of 250 knots (``p_i = 10510 Pa``) is imposed. Below the
    crossover altitude, the maximum operational speed is limited by the maximum permitted impact
    pressure. Above the crossover altitude, the maximum operational speed is determined by the
    maximum permitted Mach number of the aircraft type.
    """
    if atm_speed_limit:
        p_i_max = np.where(altitude_ft < 10000.0, 10510.0, p_i_max)  # type: ignore[assignment]

    return (
        np.where(  # type: ignore[return-value]
            air_pressure > p_inf_co,
            2.0**0.5
            * ((1.0 + (2.0 / constants.kappa) * (p_i_max / air_pressure)) ** 0.5 - 1.0) ** 0.5,
            max_mach_num,
        )
        + buffer
    )


# ------------------------------
# Thrust requirements and limits
# ------------------------------


def required_thrust_coefficient(
    c_lift: ArrayOrFloat,
    c_drag: ArrayOrFloat,
    true_airspeed: ArrayOrFloat,
    *,
    rocd: ArrayOrFloat = 300.0,  # type: ignore[assignment]
) -> ArrayOrFloat:
    r"""
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
    dh_dt = units.ft_to_m(rocd) / 60.0
    return c_lift * ((c_drag / c_lift) + (dh_dt / true_airspeed))


def max_available_thrust_coefficient(
    air_temperature: ArrayOrFloat,
    mach_number: ArrayOrFloat,
    c_t_eta_b: ArrayOrFloat,
    atyp_param: PSAircraftEngineParams,
    *,
    buffer: float = 0.20,
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
    atyp_param : PSAircraftEngineParams
        Extracted aircraft and engine parameters.
    buffer : float, optional
        Additional buffer for maximum throttle parameter `tr_max`. The default value recommended by
        Ian Poll is 0.2, which increases the maximum throttle parameter by 20%.

    Returns
    -------
    ArrayOrFloat
        Maximum available thrust coefficient that can be supplied by the engines.
    """
    tr_max = _normalised_max_throttle_parameter(
        air_temperature,
        mach_number,
        atyp_param.tet_mcc,
        atyp_param.tr_ec,
        atyp_param.m_ec,
        buffer=buffer,
    )
    c_t_max_over_c_t_eta_b = 1.0 + 2.5 * (tr_max - 1.0)
    return c_t_max_over_c_t_eta_b * c_t_eta_b


def get_excess_thrust_available(
    mach_number: ArrayOrFloat,
    air_temperature: ArrayOrFloat,
    air_pressure: ArrayOrFloat,
    aircraft_mass: ArrayOrFloat,
    theta: ArrayOrFloat,
    atyp_param: PSAircraftEngineParams,
) -> ArrayOrFloat:
    r"""
    Calculate the excess thrust coefficient available at specified operation condition.

    Parameters
    ----------
    mach_number : ArrayOrFloat
        Mach number at each waypoint
    air_temperature : ArrayOrFloat
        Ambient temperature at each waypoint, [:math:`K`]
    air_pressure : ArrayOrFloat
        Ambient pressure, [:math:`Pa`]
    aircraft_mass : ArrayOrFloat
        Aircraft mass at each waypoint, [:math:`kg`]
    theta : ArrayOrFloat
        Climb (positive value) or descent (negative value) angle, [:math:`\deg`]
    atyp_param : PSAircraftEngineParams
        Extracted aircraft and engine parameters.

    Returns
    -------
    ArrayOrFloat
        The difference between the maximum rated thrust coefficient and the thrust coefficient
        required to maintain the current mach_number.
    """
    from pycontrails.models.ps_model.ps_model import (
        airframe_drag_coefficient,
        lift_coefficient,
        oswald_efficiency_factor,
        reynolds_number,
        skin_friction_coefficient,
        thrust_coefficient_at_max_efficiency,
        wave_drag_coefficient,
        zero_lift_drag_coefficient,
    )

    rn = reynolds_number(atyp_param.wing_surface_area, mach_number, air_temperature, air_pressure)
    if isinstance(rn, float):
        if rn <= 0.0:
            return np.nan
    else:
        rn[rn <= 0.0] = np.nan

    c_lift = lift_coefficient(
        atyp_param.wing_surface_area, aircraft_mass, air_pressure, mach_number, theta
    )

    c_f = skin_friction_coefficient(rn)
    c_drag_0 = zero_lift_drag_coefficient(c_f, atyp_param.psi_0)
    e_ls = oswald_efficiency_factor(c_drag_0, atyp_param)
    c_drag_w = wave_drag_coefficient(mach_number, c_lift, atyp_param)
    c_drag = airframe_drag_coefficient(
        c_drag_0, c_drag_w, c_lift, e_ls, atyp_param.wing_aspect_ratio
    )

    tas = units.mach_number_to_tas(mach_number, air_temperature)
    req_thrust_coeff = required_thrust_coefficient(c_lift, c_drag, tas)  # type: ignore[type-var]

    c_t_eta_b = thrust_coefficient_at_max_efficiency(
        mach_number, atyp_param.m_des, atyp_param.c_t_des
    )
    max_thrust_coeff = max_available_thrust_coefficient(
        air_temperature, mach_number, c_t_eta_b, atyp_param
    )

    return max_thrust_coeff - req_thrust_coeff  # type: ignore[return-value]


def _normalised_max_throttle_parameter(
    air_temperature: ArrayOrFloat,
    mach_number: ArrayOrFloat,
    tet_mcc: float,
    tr_ec: float,
    m_ec: float,
    *,
    buffer: float = 0.20,
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
    tr_ec : float
        Engine characteristic ratio of total turbine-entry-temperature to the total freestream
        temperature for maximum overall efficiency.
    m_ec : float
        Engine characteristic Mach number associated with `tr_ec`.
    buffer : float, optional
        Additional buffer for maximum throttle parameter. The default value recommended by Ian Poll
        is 0.2, which increases the maximum throttle parameter by 20%. This affects the maximum
        available thrust coefficient calculated downstream.

    Returns
    -------
    ArrayOrFloat
        Normalised maximum throttle parameter, `tr_max`.

    Notes
    -----
    The normalised throttle parameter is the ratio of the total temperature of the gas at turbine
    entry to the freestream total temperature, normalised with its value for maximum engine
    overall efficiency at the same freestream Mach number.
    """
    return (
        (tet_mcc / air_temperature)
        / (tr_ec * (1.0 - 0.53 * (mach_number - m_ec) ** 2) * (1.0 + 0.2 * mach_number**2))
        * (1.0 + buffer)
    )


# --------------------
# Aircraft mass limits
# --------------------


def max_allowable_aircraft_mass(
    air_pressure: ArrayOrFloat,
    mach_number: ArrayOrFloat,
    mach_num_des: float,
    c_l_do: float,
    wing_surface_area: float,
    amass_mtow: float,
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
    amass_mtow: float
        Aircraft maximum take-off weight, [:math:`kg`]

    Returns
    -------
    ArrayOrFloat
        Maximum allowable aircraft mass, [:math:`kg`]
    """
    c_l_maxu = max_usable_lift_coefficient(mach_number, mach_num_des, c_l_do)
    amass_max = (1.0 / constants.g) * (
        c_l_maxu * 0.5 * constants.kappa * air_pressure * (mach_number**2) * wing_surface_area
    )
    return np.minimum(amass_max, amass_mtow)


def max_usable_lift_coefficient(
    mach_number: ArrayOrFloat, mach_num_des: float, c_l_do: float
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
        1.8 + 0.160 * m_over_m_des - 1.085 * m_over_m_des**2,
        13.272 - 42.262 * m_over_m_des + 49.883 * m_over_m_des**2 - 19.683 * m_over_m_des**3,
    )
    return c_l_maxu_over_c_l_do * c_l_do  # type: ignore[return-value]


def minimum_mach_num(
    air_pressure: ArrayOrFloat,
    aircraft_mass: ArrayOrFloat,
    atyp_param: PSAircraftEngineParams,
) -> ArrayOrFloat:
    """
    Calculate minimum mach number to avoid stall.

    Parameters
    ----------
    air_pressure : ArrayOrFloat
        Ambient pressure, [:math:`Pa`]
    aircraft_mass : ArrayOrFloat
        Aircraft mass at each waypoint, [:math:`kg`]
    atyp_param : PSAircraftEngineParams
        Extracted aircraft and engine parameters.

    Returns
    -------
    ArrayOrFloat
        Minimum mach number to avoid stall.
    """

    def excess_mass(
        mach_number: ArrayOrFloat,
        air_pressure: ArrayOrFloat,
        aircraft_mass: ArrayOrFloat,
        mach_num_des: float,
        c_l_do: float,
        wing_surface_area: float,
    ) -> ArrayOrFloat:
        amass_max = max_allowable_aircraft_mass(
            air_pressure,
            mach_number,
            mach_num_des,
            c_l_do,
            wing_surface_area,
            1e10,  # clipped to this value which we want to ignore
        )
        return amass_max - aircraft_mass

    return scipy.optimize.newton(
        excess_mass,
        args=(
            air_pressure,
            aircraft_mass,
            atyp_param.m_des,
            atyp_param.c_l_do,
            atyp_param.wing_surface_area,
        ),
        x0=np.full_like(air_pressure, 0.4),
        x1=np.full_like(air_pressure, 0.5),
        tol=1e-4,
    )


def maximum_mach_num(
    altitude_ft: ArrayOrFloat,
    air_pressure: ArrayOrFloat,
    aircraft_mass: ArrayOrFloat,
    air_temperature: ArrayOrFloat,
    theta: ArrayOrFloat,
    atyp_param: PSAircraftEngineParams,
) -> ArrayOrFloat:
    r"""
    Return the maximum mach number at the current operating conditions.

    The value returned  will be the lesser of the maximum operational mach
    number of the aircraft or the mach number obtainable at maximum thrust

    Parameters
    ----------
    altitude_ft  : ArrayOrFloat
        Altitude, [:math:`ft`]
    air_pressure : ArrayOrFloat
        Ambient pressure, [:math:`Pa`]
    aircraft_mass : ArrayOrFloat
        Aircraft mass at each waypoint, [:math:`kg`]
    air_temperature : ArrayOrFloat
        Array of ambient temperature, [:math: `K`]
    theta : ArrayOrFloat
        Climb (positive value) or descent (negative value) angle, [:math:`\deg`]
    atyp_param : PSAircraftEngineParams
        Extracted aircraft and engine parameters.

    Returns
    -------
    ArrayOrFloat
        Maximum mach number given thrust limiations.
    """
    # Max speed ignoring thrust limits
    mach_num_op_lim = max_mach_number_by_altitude(
        altitude_ft,
        air_pressure,
        atyp_param.max_mach_num,
        atyp_param.p_i_max,
        atyp_param.p_inf_co,
    )

    return scipy.optimize.newton(
        func=get_excess_thrust_available,
        args=(air_temperature, air_pressure, aircraft_mass, theta, atyp_param),
        x0=mach_num_op_lim,
        x1=mach_num_op_lim - 0.01,
        tol=1e-4,
    ).clip(max=mach_num_op_lim)


# ----------------
# Fuel flow limits
# ----------------


def fuel_flow_idle(
    fuel_flow_idle_sls: float, altitude_ft: ArrayOrFloat
) -> npt.NDArray[np.floating]:
    r"""Calculate minimum fuel mass flow rate at flight idle conditions.

    Parameters
    ----------
    fuel_flow_idle_sls : float
        Fuel mass flow rate under engine idle and sea level static conditions, [:math:`kg \ s^{-1}`]
    altitude_ft : ArrayOrFloat
        Waypoint altitude, [:math: `ft`]

    Returns
    -------
    npt.NDArray[np.floating]
        Fuel mass flow rate at flight idle conditions, [:math:`kg \ s^{-1}`]
    """
    x = altitude_ft / 10000.0
    return fuel_flow_idle_sls * (1.0 - 0.178 * x + 0.0085 * x**2)  # type: ignore[return-value]


def max_fuel_flow(
    air_temperature: ArrayOrFloat,
    air_pressure: ArrayOrFloat,
    mach_number: ArrayOrFloat,
    fuel_flow_max_sls: float,
    flight_phase: npt.NDArray[np.uint8] | flight.FlightPhase,
) -> npt.NDArray[np.floating]:
    r"""Correct maximum fuel mass flow rate that can be supplied by the engine.

    Parameters
    ----------
    air_temperature : ArrayOrFloat
        Ambient temperature at each waypoint, [:math:`K`]
    air_pressure : ArrayOrFloat
        Ambient pressure, [:math:`Pa`]
    mach_number : ArrayOrFloat
        Mach number at each waypoint
    fuel_flow_max_sls : float
        Fuel mass flow rate at take-off and sea level static conditions, [:math:`kg \ s^{-1}`]
    flight_phase : npt.NDArray[np.uint8] | flight.FlightPhase
        Phase state of each waypoint.

    Returns
    -------
    npt.NDArray[np.floating]
        Maximum allowable fuel mass flow rate, [:math:`kg \ s^{-1}`]
    """
    ff_max = jet.equivalent_fuel_flow_rate_at_cruise(
        fuel_flow_max_sls,
        air_temperature / constants.T_msl,
        air_pressure / constants.p_surface,
        mach_number,
    )

    # Account for descent conditions
    # Assume `max_fuel_flow` at descent is not more than 30% of fuel_flow_max_sls
    # We need this assumption because PTF files are not available in the PS model.
    descent = flight_phase == flight.FlightPhase.DESCENT
    ff_max[descent] = 0.3 * fuel_flow_max_sls
    return ff_max
