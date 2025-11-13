"""Calculate nitrogen oxide (NOx), carbon monoxide (CO) and hydrocarbon (HC) emissions.

This modules applies the Fuel Flow Method 2 (FFM2) from DuBois & Paynter (2006) for a given
aircraft-engine pair.

References
----------
- :cite:`duboisFuelFlowMethod22006`
"""

from __future__ import annotations

import dataclasses
import functools

import numpy as np
import numpy.typing as npt

from pycontrails.core.interpolation import EmissionsProfileInterpolator
from pycontrails.physics import jet, units

# ------------------------------------------------------
# Data structure for ICAO EDB: Gaseous (NOx, CO, and HC)
# ------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class EDBGaseous:
    """Gaseous emissions data.

    -------------------------------------
    ENGINE IDENTIFICATION AND TYPE:
    -------------------------------------
    manufacturer: str
        engine manufacturer
    engine_name: str
        name of engine
    combustor: str
        description of engine combustor

    -------------------------------------
    ENGINE CHARACTERISTICS:
    -------------------------------------
    bypass_ratio: float
        engine bypass ratio
    pressure_ratio: float
        engine pressure ratio
    rated_thrust: float
        rated thrust of engine, [:math:`kN`]

    -------------------------------------
    FUEL CONSUMPTION:
    -------------------------------------
    ff_7: float
        fuel mass flow rate at 7% thrust setting, [:math:`kg s^{-1}`]
    ff_30: float
        fuel mass flow rate at 30% thrust setting, [:math:`kg s^{-1}`]
    ff_85: float
        fuel mass flow rate at 85% thrust setting, [:math:`kg s^{-1}`]
    ff_100: float
        fuel mass flow rate at 100% thrust setting, [:math:`kg s^{-1}`]

    -------------------------------------
    EMISSIONS:
    -------------------------------------
    ei_nox_7: float
        NOx emissions index at 7% thrust setting, [:math:`g_{NO_{X}}/kg_{fuel}`]
    ei_nox_30: float
        NOx emissions index at 30% thrust setting, [:math:`g_{NO_{X}}/kg_{fuel}`]
    ei_nox_85: float
        NOx emissions index at 85% thrust setting, [:math:`g_{NO_{X}}/kg_{fuel}`]
    ei_nox_100: float
        NOx emissions index at 100% thrust setting, [:math:`g_{NO_{X}}/kg_{fuel}`]

    ei_co_7: float
        CO emissions index at 7% thrust setting, [:math:`g_{CO}/kg_{fuel}`]
    ei_co_30: float
        CO emissions index at 30% thrust setting, [:math:`g_{CO}/kg_{fuel}`]
    ei_co_85: float
        CO emissions index at 85% thrust setting, [:math:`g_{CO}/kg_{fuel}`]
    ei_co_100: float
        CO emissions index at 100% thrust setting, [:math:`g_{CO}/kg_{fuel}`]

    ei_hc_7: float
        HC emissions index at 7% thrust setting, [:math:`g_{HC}/kg_{fuel}`]
    ei_hc_30: float
        HC emissions index at 30% thrust setting, [:math:`g_{HC}/kg_{fuel}`]
    ei_hc_85: float
        HC emissions index at 85% thrust setting, [:math:`g_{HC}/kg_{fuel}`]
    ei_hc_100: float
        HC emissions index at 100% thrust setting, [:math:`g_{HC}/kg_{fuel}`]

    sn_7: float
        smoke number at 7% thrust setting
    sn_30: float
        smoke number at 30% thrust setting
    sn_85: float
        smoke number at 85% thrust setting
    sn_100: float
        smoke number at 100% thrust setting
    sn_max: float
        maximum smoke number value across the range of thrust setting
    """

    # Engine identification and type
    manufacturer: str
    engine_name: str
    combustor: str

    # Engine characteristics
    bypass_ratio: float
    pressure_ratio: float
    rated_thrust: float
    temp_min: float
    temp_max: float
    pressure_min: float
    pressure_max: float

    # Fuel consumption
    ff_7: float
    ff_30: float
    ff_85: float
    ff_100: float

    # Emissions
    ei_nox_7: float
    ei_nox_30: float
    ei_nox_85: float
    ei_nox_100: float

    ei_co_7: float
    ei_co_30: float
    ei_co_85: float
    ei_co_100: float

    ei_hc_7: float
    ei_hc_30: float
    ei_hc_85: float
    ei_hc_100: float

    sn_7: float
    sn_30: float
    sn_85: float
    sn_100: float
    sn_max: float

    @property
    def log_ei_nox_profile(self) -> EmissionsProfileInterpolator:
        """Get the logarithmic emissions index profile for NOx emissions."""
        return nitrogen_oxide_emissions_index_profile_ffm2(
            ff_idle=self.ff_7,
            ff_approach=self.ff_30,
            ff_climb=self.ff_85,
            ff_take_off=self.ff_100,
            ei_nox_idle=self.ei_nox_7,
            ei_nox_approach=self.ei_nox_30,
            ei_nox_climb=self.ei_nox_85,
            ei_nox_take_off=self.ei_nox_100,
        )

    @property
    def log_ei_co_profile(self) -> EmissionsProfileInterpolator:
        """Get the logarithmic emissions index profile for CO emissions."""
        return co_hc_emissions_index_profile_ffm2(
            ff_idle=self.ff_7,
            ff_approach=self.ff_30,
            ff_climb=self.ff_85,
            ff_take_off=self.ff_100,
            ei_idle=self.ei_co_7,
            ei_approach=self.ei_co_30,
            ei_climb=self.ei_co_85,
            ei_take_off=self.ei_co_100,
        )

    @property
    def log_ei_hc_profile(self) -> EmissionsProfileInterpolator:
        """Get the logarithmic emissions index profile for HC emissions."""
        return co_hc_emissions_index_profile_ffm2(
            ff_idle=self.ff_7,
            ff_approach=self.ff_30,
            ff_climb=self.ff_85,
            ff_take_off=self.ff_100,
            ei_idle=self.ei_hc_7,
            ei_approach=self.ei_hc_30,
            ei_climb=self.ei_hc_85,
            ei_take_off=self.ei_hc_100,
        )


# -------------------------------------------------------
#  Fuel Flow Method 2 (FFM2) from DuBois & Paynter (2006)
# -------------------------------------------------------


@functools.cache
def nitrogen_oxide_emissions_index_profile_ffm2(
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
def co_hc_emissions_index_profile_ffm2(
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


def estimate_nox_ffm2(
    log_ei_nox_profile: EmissionsProfileInterpolator,
    fuel_flow_per_engine: npt.NDArray[np.floating],
    true_airspeed: npt.NDArray[np.floating],
    air_pressure: npt.NDArray[np.floating],
    air_temperature: npt.NDArray[np.floating],
    specific_humidity: None | npt.NDArray[np.floating] = None,
) -> npt.NDArray[np.floating]:
    """Estimate the nitrogen oxide (NOx) emissions index (EI) at cruise conditions.

    Parameters
    ----------
    log_ei_nox_profile: EmissionsProfileInterpolator
        emissions profile containing the log of EI NOx versus log of fuel flow.
    fuel_flow_per_engine: npt.NDArray[np.floating]
        fuel mass flow rate per engine, [:math:`kg s^{-1}`]
    true_airspeed: npt.NDArray[np.floating]
        true airspeed for each waypoint, [:math:`m s^{-1}`]
    air_pressure : npt.NDArray[np.floating]
        pressure altitude at each waypoint, [:math:`Pa`]
    air_temperature : npt.NDArray[np.floating]
        ambient temperature for each waypoint, [:math:`K`]
    specific_humidity: npt.NDArray[np.floating] | None
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


def estimate_ei_co_hc_ffm2(
    log_ei_profile: EmissionsProfileInterpolator,
    fuel_flow_per_engine: npt.NDArray[np.floating],
    true_airspeed: npt.NDArray[np.floating],
    air_pressure: npt.NDArray[np.floating],
    air_temperature: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    """Estimate carbon monoxide (CO) or hydrocarbon (HC) emissions index (EI).

    Parameters
    ----------
    log_ei_profile: EmissionsProfileInterpolator
        emissions profile containing the log of EI CO or EI HC versus log of fuel flow.
    fuel_flow_per_engine: npt.NDArray[np.floating]
        fuel mass flow rate per engine, [:math:`kg s^{-1}`]
    true_airspeed: npt.NDArray[np.floating]
        true airspeed for each waypoint, [:math:`m s^{-1}`]
    air_pressure : npt.NDArray[np.floating]
        pressure altitude at each waypoint, [:math:`Pa`]
    air_temperature : npt.NDArray[np.floating]
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
    ei_sl: npt.NDArray[np.floating],
    theta_amb: npt.NDArray[np.floating],
    delta_amb: npt.NDArray[np.floating],
    ei_type: str,
) -> npt.NDArray[np.floating]:
    """Convert the estimated EI at sea level to cruise conditions.

    Refer to Eqs. (15) and (16) in DuBois & Paynter (2006).

    Parameters
    ----------
    ei_sl : npt.NDArray[np.floating]
        Sea level EI values.
    theta_amb : npt.NDArray[np.floating]
        Ratio of the ambient temperature to the temperature at mean sea-level.
    delta_amb : npt.NDArray[np.floating]
        Ratio of the pressure altitude to the surface pressure.
    ei_type : str
        One of {"HC", "CO", "NOX"}

    Returns
    -------
    npt.NDArray[np.floating]
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
    specific_humidity: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    return np.exp(-19 * (specific_humidity - 0.00634))


def _estimate_specific_humidity(
    air_temperature: npt.NDArray[np.floating], air_pressure: npt.NDArray[np.floating], rh: float
) -> npt.NDArray[np.floating]:
    """Estimate the specific humidity by assuming a fixed relative humidity.

    Refer to Eqs. (43), (44) and (45) in DuBois & Paynter (2006).

    Parameters
    ----------
    air_temperature : npt.NDArray[np.floating]
        Air temperature, [:math:`K`]
    air_pressure : npt.NDArray[np.floating]
        Air pressure, [:math:`Pa`]
    rh : float
        Relative humidity, [:math:`0 - 1`]

    Returns
    -------
    npt.NDArray[np.floating]
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
