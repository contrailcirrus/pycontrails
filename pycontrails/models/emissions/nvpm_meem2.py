from __future__ import annotations

import functools
import warnings
import numpy as np
import numpy.typing as npt

from pycontrails.core.interpolation import EmissionsProfileInterpolator
from pycontrails.physics import jet, units
from pycontrails.models.emissions import EDBnvpm


@functools.cache
def nvpm_emissions_index_profile_meem(
    ff_7: float,
    ff_30: float,
    ff_85: float,
    ff_100: float,
    nvpm_ei_n_7: float,
    nvpm_ei_n_30: float,
    nvpm_ei_n_85: float,
    nvpm_ei_n_100: float,
    nvpm_ei_m_7: float,
    nvpm_ei_m_30: float,
    nvpm_ei_m_85: float,
    nvpm_ei_m_100: float,
    fuel_hydrogen_content: float = 13.8,
) -> tuple[EmissionsProfileInterpolator, EmissionsProfileInterpolator]:
    """
    Create the nvPM number emissions index (EI) profile for the given engine type.

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
    nvpm_ei_n_7: float
        ICAO EDB nvPM number emissions index at idle conditions (7% power), [:math:`kg_{fuel}^{-1}`]
    nvpm_ei_n_30: float
        ICAO EDB nvPM number emissions index at approach (30% power), [:math:`kg_{fuel}^{-1}`]
    nvpm_ei_n_85: float
        ICAO EDB nvPM number emissions index at climb out (85% power), [:math:`kg_{fuel}^{-1}`]
    nvpm_ei_n_100: float
        ICAO EDB nvPM number emissions index at take-off (100% power), [:math:`kg_{fuel}^{-1}`]
    nvpm_ei_m_7: float
        ICAO EDB nvPM mass emissions index at idle conditions (7% power), [:math:`kg/kg_{fuel}`]
    nvpm_ei_m_30: float
        ICAO EDB nvPM mass emissions index at approach (30% power), [:math:`kg/kg_{fuel}`]
    nvpm_ei_m_85: float
        ICAO EDB nvPM mass emissions index at climb out (85% power), [:math:`kg/kg_{fuel}`]
    nvpm_ei_m_100: float
        ICAO EDB nvPM mass emissions index at take-off (100% power), [:math:`kg/kg_{fuel}`]

    Returns
    -------
    tuple[EmissionsProfileInterpolator, EmissionsProfileInterpolator]
        nvPM mass and number emissions index versus the fuel mass flow rate for a given engine type
    """
    fuel_flow = np.array([ff_7, ff_30, ff_85, ff_100], dtype=float)
    installation_correction_factor = np.array([1.100, 1.020, 1.013, 1.010])
    fuel_flow *= installation_correction_factor

    # Extract nvPM emissions arrays
    nvpm_ei_m = np.array([nvpm_ei_m_7, nvpm_ei_m_30, nvpm_ei_m_85, nvpm_ei_m_100], dtype=float)
    nvpm_ei_n = np.array([nvpm_ei_n_7, nvpm_ei_n_30, nvpm_ei_n_85, nvpm_ei_n_100], dtype=float)

    # TODO: SAF adjustments
    if not (13.4 <= fuel_hydrogen_content <= 15.4):
        warnings.warn(
            f"Fuel hydrogen content {fuel_hydrogen_content} % is outside the valid range" 
            "(13.4 - 15.4 %), and may lead to inaccuracies."
        )

    # Adjust nvPM emissions index due to fuel hydrogen content differences
    thrust_setting = np.array([0.07, 0.30, 0.85, 1.00])
    k_mass = nvpm_mass_ei_fuel_adjustment_meem(fuel_hydrogen_content, thrust_setting)
    k_num = nvpm_number_ei_fuel_adjustment_meem(fuel_hydrogen_content, thrust_setting)
    nvpm_ei_m *= k_mass
    nvpm_ei_n *= k_num

    nvpm_ei_m_interp = EmissionsProfileInterpolator(xp=fuel_flow, fp=nvpm_ei_m)
    nvpm_ei_n_interp = EmissionsProfileInterpolator(xp=fuel_flow, fp=nvpm_ei_n)
    return nvpm_ei_m_interp, nvpm_ei_n_interp


def nvpm_mass_ei_fuel_adjustment_meem(
    hydrogen_content: float | npt.NDArray[np.floating],
    thrust_setting: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    # TODO: Documentation (Unit test?)
    return np.exp(
        (1.08 * thrust_setting - 1.31) * (hydrogen_content - 13.8)
    )


def nvpm_number_ei_fuel_adjustment_meem(
    hydrogen_content: float | npt.NDArray[np.floating],
    thrust_setting: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    # TODO: Documentation (Unit test?)
    return np.exp(
        (0.99 * thrust_setting - 1.05) * (hydrogen_content - 13.8)
    )


def estimate_nvpm(
    edb_nvpm: EDBnvpm,
    fuel_flow_per_engine: npt.NDArray[np.floating],
    true_airspeed: npt.NDArray[np.floating],
    air_pressure: npt.NDArray[np.floating],
    air_temperature: npt.NDArray[np.floating],
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    # TODO: Documentation
    # TODO: Step 0 (SCOPE11 implementation)
    # TODO: Step 1 (Add fifth point and lean-burn)

    # Temporary notes: STEP 2
    mach_num = units.tas_to_mach_number(true_airspeed, air_temperature)
    theta_amb = jet.temperature_ratio(air_temperature)
    delta_amb = jet.pressure_ratio(air_pressure)
    fuel_flow_per_engine = jet.equivalent_fuel_flow_rate_at_sea_level(
        fuel_flow_per_engine, theta_amb, delta_amb, mach_num
    )
    fuel_flow_per_engine.clip(edb_nvpm.ff_7, edb_nvpm.ff_100, out=fuel_flow_per_engine)  # clip in place

    # TODO: Step 3.1 (Interpolation)
    # Interpolate nvPM EI_m and EI_n (Ground)
    nvpm_ei_m_sl = edb_nvpm.nvpm_ei_m_meem.interp(fuel_flow_per_engine)
    nvpm_ei_m_sl = nvpm_ei_m_sl * 1e-6  # mg-nvPM/kg-fuel to kg-nvPM/kg-fuel
    nvpm_ei_n_sl = edb_nvpm.nvpm_ei_n_meem.interp(fuel_flow_per_engine)

    # TODO: Step 3.2 (Account for SAF)

    # TODO: Step 4 (Ground to flight transportation - Mass and number)
    nvpm_ei_m = nvpm_ei_m_sl * ((delta_amb ** 1.377) / (theta_amb ** 4.455)) * (1.1 ** 2.5)
    nvpm_ei_n = (nvpm_ei_m / nvpm_ei_m_sl) * nvpm_ei_n_sl
    return nvpm_ei_m, nvpm_ei_n



