"""Calculate nitrogen oxide (NOx), carbon monoxide (CO) and hydrocarbon (HC) emissions.

This modules applies the Fuel Flow Method 2 (FFM2) from DuBois & Paynter (2006) for a given
aircraft-engine pair.

References
----------
- :cite:`duboisFuelFlowMethod22006`
"""

from __future__ import annotations

import functools

import numpy as np
import numpy.typing as npt

from pycontrails.core.interpolation import EmissionsProfileInterpolator
from pycontrails.physics import jet, units


@functools.cache
def nitrogen_oxide_emissions_index_profile(
    ff_idle: float,
    ff_approach: float,
    ff_climb: float,
    ff_take_off: float,
    ei_nox_idle: float,
    ei_nox_approach: float,
    ei_nox_climb: float,
    ei_nox_take_off: float,
) -> EmissionsProfileInterpolator:
    """
    Create the nitrogen oxide (NOx) emissions index (EI) profile for the given engine type.

    Parameters
    ----------
    ff_idle: float
        ICAO EDB fuel mass flow rate at idle conditions (7% power), [:math:`kg s^{-1}`]
    ff_approach: float
        ICAO EDB fuel mass flow rate at approach (30% power), [:math:`kg s^{-1}`]
    ff_climb: float
        ICAO EDB fuel mass flow rate at climb out (85% power), [:math:`kg s^{-1}`]
    ff_take_off: float
        ICAO EDB fuel mass flow rate at take-off (100% power), [:math:`kg s^{-1}`]
    ei_nox_idle: float
        ICAO EDB NOx emissions index at idle conditions (7% power), [:math:`g_{NO_{X}}/kg_{fuel}`]
    ei_nox_approach: float
        ICAO EDB NOx emissions index at approach (30% power), [:math:`g_{NO_{X}}/kg_{fuel}`]
    ei_nox_climb: float
        ICAO EDB NOx emissions index at climb out (85% power), [:math:`g_{NO_{X}}/kg_{fuel}`]
    ei_nox_take_off: float
        ICAO EDB NOx emissions index at take-off (100% power), [:math:`g_{NO_{X}}/kg_{fuel}`]

    Returns
    -------
    EmissionsProfileInterpolator
        log of NOx emissions index versus the log of fuel mass flow rate for a given engine type

    Raises
    ------
    ValueError
        If any EI nox values are non-positive.
    """
    fuel_flow = np.array([ff_idle, ff_approach, ff_climb, ff_take_off], dtype=float)
    installation_correction_factor = np.array([1.100, 1.020, 1.013, 1.010])
    fuel_flow *= installation_correction_factor

    ei_nox = np.array([ei_nox_idle, ei_nox_approach, ei_nox_climb, ei_nox_take_off], dtype=float)

    if np.any(ei_nox <= 0):
        raise ValueError("Zero value(s) encountered in the EI NOx.")

    return EmissionsProfileInterpolator(xp=np.log(fuel_flow), fp=np.log(ei_nox))


@functools.cache
def co_hc_emissions_index_profile(
    ff_idle: float,
    ff_approach: float,
    ff_climb: float,
    ff_take_off: float,
    ei_idle: float,
    ei_approach: float,
    ei_climb: float,
    ei_take_off: float,
) -> EmissionsProfileInterpolator:
    """Create carbon monoxide (CO) and hydrocarbon (HC) emissions index (EI) profile.

    Parameters
    ----------
    ff_idle: float
        ICAO EDB fuel mass flow rate at idle conditions
        (7% power), [:math:`kg s^{-1}`]
    ff_approach: float
        ICAO EDB fuel mass flow rate at approach
        (30% power), [:math:`kg s^{-1}`]
    ff_climb: float
        ICAO EDB fuel mass flow rate at climb out
        (85% power), [:math:`kg s^{-1}`]
    ff_take_off: float
        ICAO EDB fuel mass flow rate at take-off
        (100% power), [:math:`kg s^{-1}`]
    ei_idle: float
        ICAO EDB CO or HC emissions index at idle conditions
        (7% power), [:math:`g_{pollutant}/kg_{fuel}`]
    ei_approach: float
        ICAO EDB CO or HC emissions index at approach
        (30% power), [:math:`g_{pollutant}/kg_{fuel}`]
    ei_climb: float
        ICAO EDB CO or HC emissions index at climb out
        (85% power), [:math:`g_{pollutant}/kg_{fuel}`]
    ei_take_off: float
        ICAO EDB CO or HC emissions index at take-off
        (100% power), [:math:`g_{pollutant}/kg_{fuel}`]

    Returns
    -------
    EmissionsProfileInterpolator
        log of CO or HC emissions index versus the log of fuel mass
        flow rate for a given engine type
    """
    fuel_flow_edb = np.array([ff_idle, ff_approach, ff_climb, ff_take_off], dtype=float)
    installation_correction_factor = np.array([1.100, 1.020, 1.013, 1.010])
    fuel_flow_edb *= installation_correction_factor

    ei_edb = np.array([ei_idle, ei_approach, ei_climb, ei_take_off], dtype=float)
    min_vals_ = np.array([1e-3, 1e-3, 1e-4, 1e-4])
    ei_edb = np.maximum(ei_edb, min_vals_)

    # Get straight-line equation between idle and approach
    m, c = np.polyfit(fuel_flow_edb[:2], ei_edb[:2], deg=1)
    ei_climb_extrapolate = m * fuel_flow_edb[2] + c
    ei_hi = np.mean(ei_edb[2:])

    ff_low_power = fuel_flow_edb[3] * 0.03
    ei_co_low_power = min((m * ff_low_power + c), (2 * ei_edb[0]))
    ei_co_low_power = max(
        ei_co_low_power, 1e-3
    )  # Prevent zero/negative values, similar to line 115

    # Permutation 1: Emissions profile when the bi-linear fit does not work
    # (Figure 14 of DuBois & Paynter, 2006)
    if ei_edb[1] < ei_edb[2]:
        ff_profile = np.insert(fuel_flow_edb, 0, ff_low_power)
        ei_profile = np.array([ei_co_low_power, ei_edb[0], ei_edb[1], ei_hi, ei_hi])

    # Permutation 2: Emissions profile using a bi-linear fit (Figure 8 of DuBois & Paynter, 2006)
    elif ei_climb_extrapolate < ei_edb[2] and (m != 0):
        ff_intersect = (ei_hi - c) / m
        # Ensure intersection is between 30% and 85% fuel mass flow rate
        ff_intersect = np.clip(ff_intersect, fuel_flow_edb[1] + 0.01, fuel_flow_edb[2] - 0.01)
        ff_profile = np.array(
            [
                ff_low_power,
                fuel_flow_edb[0],
                fuel_flow_edb[1],
                ff_intersect,
                fuel_flow_edb[2],
                fuel_flow_edb[3],
            ]
        )
        ei_profile = np.array([ei_co_low_power, ei_edb[0], ei_edb[1], ei_hi, ei_hi, ei_hi])

    # Permutation 3: Point-to-point fit (Figure 13 of DuBois & Paynter, 2006)
    else:
        ff_profile = np.insert(fuel_flow_edb, 0, ff_low_power)
        ei_profile = np.insert(ei_edb, 0, ei_co_low_power)

    return EmissionsProfileInterpolator(xp=np.log(ff_profile), fp=np.log(ei_profile))


def estimate_nox(
    log_ei_nox_profile: EmissionsProfileInterpolator,
    fuel_flow_per_engine: npt.NDArray[np.float_],
    true_airspeed: npt.NDArray[np.float_],
    air_pressure: npt.NDArray[np.float_],
    air_temperature: npt.NDArray[np.float_],
    specific_humidity: None | npt.NDArray[np.float_] = None,
) -> npt.NDArray[np.float_]:
    """Estimate the nitrogen oxide (NOx) emissions index (EI) at cruise conditions.

    Parameters
    ----------
    log_ei_nox_profile: EmissionsProfileInterpolator
        emissions profile containing the log of EI NOx versus log of fuel flow.
    fuel_flow_per_engine: npt.NDArray[np.float_]
        fuel mass flow rate per engine, [:math:`kg s^{-1}`]
    true_airspeed: npt.NDArray[np.float_]
        true airspeed for each waypoint, [:math:`m s^{-1}`]
    air_pressure : npt.NDArray[np.float_]
        pressure altitude at each waypoint, [:math:`Pa`]
    air_temperature : npt.NDArray[np.float_]
        ambient temperature for each waypoint, [:math:`K`]
    specific_humidity: npt.NDArray[np.float_] | None
        specific humidity for each waypoint, [:math:`kg_{H_{2}O}/kg_{air}`]
    """

    if specific_humidity is None:
        specific_humidity = _estimate_specific_humidity(air_temperature, air_pressure, rh=0.6)

    # Derived quantities
    mach_num = units.tas_to_mach_number(true_airspeed, air_temperature)
    theta_amb = jet.temperature_ratio(air_temperature)
    delta_amb = jet.pressure_ratio(air_pressure)
    fuel_flow_sl = jet.equivalent_fuel_flow_rate_at_sea_level(
        fuel_flow_per_engine, theta_amb, delta_amb, mach_num
    )
    ei_nox_sl = log_ei_nox_profile.log_interp(fuel_flow_sl)

    q_correction = _get_humidity_correction_factor(specific_humidity)
    ei_cruise = ei_at_cruise(ei_nox_sl, theta_amb, delta_amb, "NOX")
    return ei_cruise * q_correction


def estimate_ei(
    log_ei_profile: EmissionsProfileInterpolator,
    fuel_flow_per_engine: npt.NDArray[np.float_],
    true_airspeed: npt.NDArray[np.float_],
    air_pressure: npt.NDArray[np.float_],
    air_temperature: npt.NDArray[np.float_],
) -> npt.NDArray[np.float_]:
    """Estimate carbon monoxide (CO) or hydrocarbon (HC) emissions index (EI).

    Parameters
    ----------
    log_ei_profile: EmissionsProfileInterpolator
        emissions profile containing the log of EI CO or EI HC versus log of fuel flow.
    fuel_flow_per_engine: npt.NDArray[np.float_]
        fuel mass flow rate per engine, [:math:`kg s^{-1}`]
    true_airspeed: npt.NDArray[np.float_]
        true airspeed for each waypoint, [:math:`m s^{-1}`]
    air_pressure : npt.NDArray[np.float_]
        pressure altitude at each waypoint, [:math:`Pa`]
    air_temperature : npt.NDArray[np.float_]
        ambient temperature for each waypoint, [:math:`K`]
    """
    mach_num = units.tas_to_mach_number(true_airspeed, air_temperature)
    theta_amb = jet.temperature_ratio(air_temperature)
    delta_amb = jet.pressure_ratio(air_pressure)
    fuel_flow_sl = jet.equivalent_fuel_flow_rate_at_sea_level(
        fuel_flow_per_engine, theta_amb, delta_amb, mach_num
    )

    ei_sl = log_ei_profile.log_interp(fuel_flow_sl)
    return ei_at_cruise(ei_sl, theta_amb, delta_amb, "HC")


# -----------------------
# Common helper functions
# -----------------------


def ei_at_cruise(
    ei_sl: npt.NDArray[np.float_],
    theta_amb: npt.NDArray[np.float_],
    delta_amb: npt.NDArray[np.float_],
    ei_type: str,
) -> npt.NDArray[np.float_]:
    """Convert the estimated EI at sea level to cruise conditions.

    Refer to Eqs. (15) and (16) in DuBois & Paynter (2006).

    Parameters
    ----------
    ei_sl : npt.NDArray[np.float_]
        Sea level EI values.
    theta_amb : npt.NDArray[np.float_]
        Ratio of the ambient temperature to the temperature at mean sea-level.
    delta_amb : npt.NDArray[np.float_]
        Ratio of the pressure altitude to the surface pressure.
    ei_type : str
        One of {"HC", "CO", "NOX"}

    Returns
    -------
    npt.NDArray[np.float_]
        Estimated cruise EI values.

    References
    ----------
    - :cite:`duboisFuelFlowMethod22006`
    """
    if ei_type in ["HC", "CO"]:
        # bottom of page 3, x = 1
        return ei_sl * (theta_amb**3.3 / delta_amb**1.02)
    if ei_type == "NOX":
        # bottom of page 3, y = 0.5
        y = 0.5
        return ei_sl * (delta_amb**1.02 / theta_amb**3.3) ** y
    raise ValueError("Expect ei_type to be one of 'HC', 'CO', or 'NOX'")


def _get_humidity_correction_factor(
    specific_humidity: npt.NDArray[np.float_],
) -> npt.NDArray[np.float_]:
    return np.exp(-19 * (specific_humidity - 0.00634))


def _estimate_specific_humidity(
    air_temperature: npt.NDArray[np.float_], air_pressure: npt.NDArray[np.float_], rh: float
) -> npt.NDArray[np.float_]:
    """Estimate the specific humidity by assuming a fixed relative humidity.

    Refer to Eqs. (43), (44) and (45) in DuBois & Paynter (2006).

    Parameters
    ----------
    air_temperature : npt.NDArray[np.float_]
        Air temperature, [:math:`K`]
    air_pressure : npt.NDArray[np.float_]
        Air pressure, [:math:`Pa`]
    rh : float
        Relative humidity, [:math:`0 - 1`]

    Returns
    -------
    npt.NDArray[np.float_]
        Estimated specific humidity, [:math:`kg kg^{-1}`]

    References
    ----------
    - :cite:`duboisFuelFlowMethod22006`
    """
    # Equation (43)
    air_temperature_celsius = units.kelvin_to_celsius(air_temperature)

    # Equation (44)
    exponent = (7.5 * air_temperature_celsius) / (237.3 + air_temperature_celsius)
    P_sat = 6.107 * 10**exponent

    # Equation (45)
    air_pressure_hpa = air_pressure / 100
    numer = 0.62197058 * rh * P_sat
    denom = air_pressure_hpa - rh * P_sat
    return numer / denom
