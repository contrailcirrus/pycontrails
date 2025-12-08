"""Support for nvPM emissions modeling.

This module includes models related to estimating the non-volatile particulate matter (nvPM) mass
and number emissions index. Here, the terms "non-volatile particulate matter" (nvPM) and
"black carbon" (BC) are assumed with the same definition and used interchangeably.
"""

from __future__ import annotations

import dataclasses
import functools
import warnings

import numpy as np
import numpy.typing as npt

from pycontrails.core.interpolation import EmissionsProfileInterpolator
from pycontrails.physics import constants, jet, units
from pycontrails.utils.types import ArrayScalarLike

# ---------------------------------
# Data structure for ICAO EDB: nvPM
# ---------------------------------


@dataclasses.dataclass
class EDBnvpm:
    """A data class for EDB nvPM data.

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
    pressure_ratio: float
        engine pressure ratio

    -------------------------------------
    nvPM EMISSIONS:
    -------------------------------------
    nvpm_ei_m: EmissionsProfileInterpolator
         non-volatile PM mass emissions index profile (mg/kg) vs.
         non-dimensionalized thrust setting (t4_t2)
    nvpm_ei_n: EmissionsProfileInterpolator
        non-volatile PM number emissions index profile (1/kg) vs.
        non-dimensionalized thrust setting (t4_t2)
    """

    # Engine identification and type
    manufacturer: str
    engine_name: str
    combustor: str

    # Engine characteristics
    pressure_ratio: float
    temp_min: float
    temp_max: float
    fuel_heat: float

    # Fuel consumption
    ff_7: float
    ff_30: float
    ff_85: float
    ff_100: float

    # Emissions (System loss corrected)
    nvpm_ei_m_7: float
    nvpm_ei_m_30: float
    nvpm_ei_m_85: float
    nvpm_ei_m_100: float

    nvpm_ei_n_7: float
    nvpm_ei_n_30: float
    nvpm_ei_n_85: float
    nvpm_ei_n_100: float

    # Fifth data point for MEEM2
    nvpm_ei_m_use_max: bool
    nvpm_ei_m_no_sl_30: float
    nvpm_ei_m_no_sl_85: float
    nvpm_ei_m_no_sl_max: float

    nvpm_ei_n_use_max: bool
    nvpm_ei_n_no_sl_30: float
    nvpm_ei_n_no_sl_85: float
    nvpm_ei_n_no_sl_max: float

    @property
    def nvpm_ei_m_t4_t2(self) -> EmissionsProfileInterpolator:
        """Get the nvPM emissions index mass profile."""
        return nvpm_emission_profiles_t4_t2(
            pressure_ratio=self.pressure_ratio,
            combustor=self.combustor,
            temp_min=self.temp_min,
            temp_max=self.temp_max,
            q_fuel=self.fuel_heat * 1e6,
            ff_7=self.ff_7,
            ff_30=self.ff_30,
            ff_85=self.ff_85,
            ff_100=self.ff_100,
            nvpm_ei_m_7=self.nvpm_ei_m_7,
            nvpm_ei_m_30=self.nvpm_ei_m_30,
            nvpm_ei_m_85=self.nvpm_ei_m_85,
            nvpm_ei_m_100=self.nvpm_ei_m_100,
            nvpm_ei_n_7=self.nvpm_ei_n_7,
            nvpm_ei_n_30=self.nvpm_ei_n_30,
            nvpm_ei_n_85=self.nvpm_ei_n_85,
            nvpm_ei_n_100=self.nvpm_ei_n_100,
        )[0]

    @property
    def nvpm_ei_n_t4_t2(self) -> EmissionsProfileInterpolator:
        """Get the nvPM emissions index number profile."""
        return nvpm_emission_profiles_t4_t2(
            pressure_ratio=self.pressure_ratio,
            combustor=self.combustor,
            temp_min=self.temp_min,
            temp_max=self.temp_max,
            q_fuel=self.fuel_heat * 1e6,
            ff_7=self.ff_7,
            ff_30=self.ff_30,
            ff_85=self.ff_85,
            ff_100=self.ff_100,
            nvpm_ei_m_7=self.nvpm_ei_m_7,
            nvpm_ei_m_30=self.nvpm_ei_m_30,
            nvpm_ei_m_85=self.nvpm_ei_m_85,
            nvpm_ei_m_100=self.nvpm_ei_m_100,
            nvpm_ei_n_7=self.nvpm_ei_n_7,
            nvpm_ei_n_30=self.nvpm_ei_n_30,
            nvpm_ei_n_85=self.nvpm_ei_n_85,
            nvpm_ei_n_100=self.nvpm_ei_n_100,
        )[1]


# ---------------------------------
# nvPM emissions: T4/T2 methodology
# ---------------------------------


@functools.cache
def nvpm_emission_profiles_t4_t2(
    pressure_ratio: float,
    combustor: str,
    temp_min: float,
    temp_max: float,
    q_fuel: float,
    ff_7: float,
    ff_30: float,
    ff_85: float,
    ff_100: float,
    nvpm_ei_m_7: float,
    nvpm_ei_m_30: float,
    nvpm_ei_m_85: float,
    nvpm_ei_m_100: float,
    nvpm_ei_n_7: float,
    nvpm_ei_n_30: float,
    nvpm_ei_n_85: float,
    nvpm_ei_n_100: float,
) -> tuple[EmissionsProfileInterpolator, EmissionsProfileInterpolator]:
    """
    Create the nvPM number emissions index (EI) profile for the given engine type.

    Parameters
    ----------
    pressure_ratio : float
        Engine pressure ratio, unitless
    combustor : str
        Engine combustor type provided by the ICAO EDB column `Combustor Description`.
    temp_min : float
        Minimum temperature, provided by the ICAO EDB column `Ambient Temp Min (K)`, [:math:`K`]
    temp_max : float
        Maximum temperature, provided by the ICAO EDB column `Ambient Temp Max (K)`, [:math:`K`]
    q_fuel : float
        Lower calorific value (LCV) of fuel, :math:`[J kg_{fuel}^{-1}]`
    ff_7: float
        ICAO EDB fuel mass flow rate at idle conditions (7% power), [:math:`kg s^{-1}`]
    ff_30: float
        ICAO EDB fuel mass flow rate at approach (30% power), [:math:`kg s^{-1}`]
    ff_85: float
        ICAO EDB fuel mass flow rate at climb out (85% power), [:math:`kg s^{-1}`]
    ff_100: float
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
    # Extract fuel flow
    fuel_flow = np.array([ff_7, ff_30, ff_85, ff_100])
    fuel_flow_max = fuel_flow[-1]

    # Extract nvPM emissions arrays
    nvpm_ei_m = np.array([nvpm_ei_m_7, nvpm_ei_m_30, nvpm_ei_m_85, nvpm_ei_m_100])
    nvpm_ei_n = np.array([nvpm_ei_n_7, nvpm_ei_n_30, nvpm_ei_n_85, nvpm_ei_n_100])

    is_staged_combustor = combustor in ("DAC", "TAPS", "TAPS II")
    if is_staged_combustor:
        # In this case, all of our interpolators will have size 5
        fuel_flow = np.insert(fuel_flow, 2, (fuel_flow[1] * 1.001))
        nvpm_ei_n_lean_burn = np.mean(nvpm_ei_n[2:])
        nvpm_ei_n = np.r_[nvpm_ei_n[:2], [nvpm_ei_n_lean_burn] * 3]
        nvpm_ei_m_lean_burn = np.mean(nvpm_ei_m[2:])
        nvpm_ei_m = np.r_[nvpm_ei_m[:2], [nvpm_ei_m_lean_burn] * 3]

    thrust_setting = fuel_flow / fuel_flow_max
    avg_temp = (temp_min + temp_max) / 2.0

    t4_t2 = jet.thrust_setting_nd(
        true_airspeed=0.0,
        thrust_setting=thrust_setting,
        T=avg_temp,
        p=constants.p_surface,
        pressure_ratio=pressure_ratio,
        q_fuel=q_fuel,
        cruise=False,
    )

    nvpm_ei_m_interp = EmissionsProfileInterpolator(t4_t2, nvpm_ei_m)
    nvpm_ei_n_interp = EmissionsProfileInterpolator(t4_t2, nvpm_ei_n)
    return nvpm_ei_m_interp, nvpm_ei_n_interp


def estimate_nvpm_t4_t2(
    edb_nvpm: EDBnvpm,
    true_airspeed: npt.NDArray[np.floating],
    air_temperature: npt.NDArray[np.floating],
    air_pressure: npt.NDArray[np.floating],
    thrust_setting: npt.NDArray[np.floating],
    q_fuel: float,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    r"""Calculate nvPM mass and number emissions index using the T4/T2 methodology.

    Interpolate the non-volatile particulate matter (nvPM) mass and number emissions index from
    the emissions profile of a given engine type that is provided by the ICAO EDB.

    The non-dimensional thrust setting (t4_t2) is clipped to the minimum and maximum t4_t2 values
    that is estimated from the four ICAO EDB datapoints to prevent extrapolating the nvPM values.

    Parameters
    ----------
    edb_nvpm : EDBnvpm
        EDB nvPM data
    true_airspeed: npt.NDArray[np.floating]
        true airspeed for each waypoint, [:math:`m s^{-1}`]
    air_temperature: npt.NDArray[np.floating]
        ambient temperature for each waypoint, [:math:`K`]
    air_pressure: npt.NDArray[np.floating]
        pressure altitude at each waypoint, [:math:`Pa`]
    thrust_setting : npt.NDArray[np.floating]
        thrust setting
    q_fuel : float
        Lower calorific value (LCV) of fuel, [:math:`J \ kg_{fuel}^{-1}`].

    Returns
    -------
    nvpm_ei_m : npt.NDArray[np.floating]
        nvPM mass emissions index, [:math:`kg/kg_{fuel}`]
    nvpm_ei_n : npt.NDArray[np.floating]
        nvPM number emissions index, [:math:`kg_{fuel}^{-1}`]
    """
    # Non-dimensionalized thrust setting
    t4_t2 = jet.thrust_setting_nd(
        true_airspeed,
        thrust_setting,
        air_temperature,
        air_pressure,
        edb_nvpm.pressure_ratio,
        q_fuel,
        cruise=True,
    )

    # Interpolate nvPM EI_m and EI_n
    nvpm_ei_m = edb_nvpm.nvpm_ei_m_t4_t2.interp(t4_t2)
    nvpm_ei_m = nvpm_ei_m * 1e-6  # mg-nvPM/kg-fuel to kg-nvPM/kg-fuel
    nvpm_ei_n = edb_nvpm.nvpm_ei_n_t4_t2.interp(t4_t2)
    return nvpm_ei_m, nvpm_ei_n


# ---------------------
# nvPM emissions: MEEM2
# ---------------------


def nvpm_mass_emission_profiles_meem(
    combustor: str,
    hydrogen_content: float,
    ff_7: float,
    ff_30: float,
    ff_85: float,
    ff_100: float,
    nvpm_ei_m_7: float,
    nvpm_ei_m_30: float,
    nvpm_ei_m_85: float,
    nvpm_ei_m_100: float,
    fifth_data_point_mass: bool = False,
    nvpm_ei_m_30_no_sl: float | None = None,
    nvpm_ei_m_85_no_sl: float | None = None,
    nvpm_ei_m_max_no_sl: float | None = None,
) -> EmissionsProfileInterpolator:
    """
    Create the nvPM mass emissions index (EI) profile for the given engine type using MEEM2.

    Parameters
    ----------
    combustor : str
        Engine combustor type provided by the ICAO EDB column `Combustor Description`.
    hydrogen_content : float
        The percentage of hydrogen mass content in the fuel.
    ff_7: float
        ICAO EDB fuel mass flow rate at idle conditions (7% power), [:math:`kg s^{-1}`]
    ff_30: float
        ICAO EDB fuel mass flow rate at approach (30% power), [:math:`kg s^{-1}`]
    ff_85: float
        ICAO EDB fuel mass flow rate at climb out (85% power), [:math:`kg s^{-1}`]
    ff_100: float
        ICAO EDB fuel mass flow rate at take-off (100% power), [:math:`kg s^{-1}`]
    nvpm_ei_m_7: float
        ICAO EDB loss-corrected nvPM mass EI at idle (7% power), [:math:`kg/kg_{fuel}`]
    nvpm_ei_m_30: float
        ICAO EDB loss-corrected nvPM mass EI at approach (30% power), [:math:`kg/kg_{fuel}`]
    nvpm_ei_m_85: float
        ICAO EDB loss-corrected nvPM mass EI at climb out (85% power), [:math:`kg/kg_{fuel}`]
    nvpm_ei_m_100: float
        ICAO EDB loss-corrected nvPM mass EI at take-off (100% power), [:math:`kg/kg_{fuel}`]
    fifth_data_point_mass : bool,
        Does the maximum nvPM EI mass occur between 30% and 85% of fuel flow?
    nvpm_ei_m_30_no_sl : float | None,
        ICAO EDB nvPM mass EI at approach (30% power), [:math:`kg/kg_{fuel}`]
    nvpm_ei_m_85_no_sl : float | None,
        ICAO EDB nvPM mass EI at climb out (85% power), [:math:`kg/kg_{fuel}`]
    nvpm_ei_m_max_no_sl : float | None,
        ICAO EDB maximum nvPM mass EI, [:math:`kg/kg_{fuel}`]

    Returns
    -------
    EmissionsProfileInterpolator
        nvPM mass emissions index versus the fuel mass flow rate for a given engine type

    References
    ----------
    # TODO: Add to bibliography
    - (Ahrens et al., 2025) https://doi.org/10.4271/2025-01-6000
    """
    fuel_flow = np.array([ff_7, ff_30, ff_85, ff_100], dtype=float)
    thrust_setting = np.array([0.07, 0.30, 0.85, 1.00])

    # Adjustment of EEDB fuel flows for installation effects
    installation_correction_factor = np.array([1.100, 1.020, 1.013, 1.010])
    fuel_flow *= installation_correction_factor

    # nvPM mass emissions profile
    nvpm_ei_m = np.array([nvpm_ei_m_7, nvpm_ei_m_30, nvpm_ei_m_85, nvpm_ei_m_100], dtype=float)

    # Deal with lean-burn combustors the same way as T4/T2 methodology
    # Temporary solution as no details are provided in paper. Awaiting authors recommendation.
    is_staged_combustor = combustor in ("DAC", "TAPS", "TAPS II")
    if is_staged_combustor:
        fuel_flow = np.insert(fuel_flow, 2, fuel_flow[1] * 1.001)
        thrust_setting = np.insert(thrust_setting, 2, thrust_setting[1] * 1.001)
        nvpm_ei_m_lean_burn = np.mean(nvpm_ei_m[2:])
        nvpm_ei_m = np.r_[nvpm_ei_m[:2], [nvpm_ei_m_lean_burn] * 3]

    # Add fifth data point if the maximum nvPM EI number occurs between 30% and 85% of fuel flow
    elif fifth_data_point_mass:
        if nvpm_ei_m_30_no_sl is None or nvpm_ei_m_85_no_sl is None or nvpm_ei_m_max_no_sl is None:
            raise ValueError(
                "nvpm_ei_m_30_no_sl, nvpm_ei_m_85_no_sl, and nvpm_ei_m_max_no_sl "
                "must be provided when fifth_data_point_mass is True."
            )

        # Calculate fuel flow (5th point)
        ff_fifth = 0.5 * (fuel_flow[1] + fuel_flow[2])
        fuel_flow = np.insert(fuel_flow, 2, ff_fifth)
        thrust_setting = np.insert(thrust_setting, 2, 0.575)

        # Calculate nvPM number emissions index (5th point)
        k_loss_correction = 0.5 * (
            (nvpm_ei_m_30 / nvpm_ei_m_30_no_sl) + (nvpm_ei_m_85 / nvpm_ei_m_85_no_sl)
        )
        nvpm_ei_m_fifth = nvpm_ei_m_max_no_sl * k_loss_correction
        nvpm_ei_m = np.insert(nvpm_ei_m, 2, nvpm_ei_m_fifth)

    # Adjust nvPM emissions index due to fuel hydrogen content differences
    if not (13.4 <= hydrogen_content <= 15.4):
        warnings.warn(
            f"Fuel hydrogen content {hydrogen_content} % is outside the valid range"
            "(13.4 - 15.4 %), and may lead to inaccuracies."
        )

    k_mass = mass_fuel_composition_correction_meem(hydrogen_content, thrust_setting)
    return EmissionsProfileInterpolator(xp=fuel_flow, fp=(nvpm_ei_m * k_mass))


def nvpm_number_emission_profiles_meem(
    combustor: str,
    hydrogen_content: float,
    ff_7: float,
    ff_30: float,
    ff_85: float,
    ff_100: float,
    nvpm_ei_n_7: float,
    nvpm_ei_n_30: float,
    nvpm_ei_n_85: float,
    nvpm_ei_n_100: float,
    fifth_data_point_number: bool = False,
    nvpm_ei_n_30_no_sl: float | None = None,
    nvpm_ei_n_85_no_sl: float | None = None,
    nvpm_ei_n_max_no_sl: float | None = None,
) -> EmissionsProfileInterpolator:
    """
    Create the nvPM number emissions index (EI) profile for the given engine type using MEEM2.

    Parameters
    ----------
    combustor : str
        Engine combustor type provided by the ICAO EDB column `Combustor Description`.
    hydrogen_content : float
        The percentage of hydrogen mass content in the fuel.
    ff_7: float
        ICAO EDB fuel mass flow rate at idle conditions (7% power), [:math:`kg s^{-1}`]
    ff_30: float
        ICAO EDB fuel mass flow rate at approach (30% power), [:math:`kg s^{-1}`]
    ff_85: float
        ICAO EDB fuel mass flow rate at climb out (85% power), [:math:`kg s^{-1}`]
    ff_100: float
        ICAO EDB fuel mass flow rate at take-off (100% power), [:math:`kg s^{-1}`]
    nvpm_ei_n_7: float
        ICAO EDB loss-corrected nvPM number EI at idle (7% power), [:math:`kg/kg_{fuel}`]
    nvpm_ei_n_30: float
        ICAO EDB loss-corrected nvPM number EI at approach (30% power), [:math:`kg/kg_{fuel}`]
    nvpm_ei_n_85: float
        ICAO EDB loss-corrected nvPM number EI at climb out (85% power), [:math:`kg/kg_{fuel}`]
    nvpm_ei_n_100: float
        ICAO EDB loss-corrected nvPM number EI at take-off (100% power), [:math:`kg/kg_{fuel}`]
    fifth_data_point_number : bool,
        Does the maximum nvPM EI number occur between 30% and 85% of fuel flow?
    nvpm_ei_n_30_no_sl : float | None,
        ICAO EDB nvPM number EI at approach (30% power), [:math:`kg/kg_{fuel}`]
    nvpm_ei_n_85_no_sl : float | None,
        ICAO EDB nvPM number EI at climb out (85% power), [:math:`kg/kg_{fuel}`]
    nvpm_ei_n_max_no_sl : float | None,
        ICAO EDB maximum nvPM number EI, [:math:`kg/kg_{fuel}`]

    Returns
    -------
    EmissionsProfileInterpolator
        nvPM number emissions index versus the fuel mass flow rate for a given engine type

    References
    ----------
    # TODO: Add to bibliography
    - (Ahrens et al., 2025) https://doi.org/10.4271/2025-01-6000
    """
    fuel_flow = np.array([ff_7, ff_30, ff_85, ff_100], dtype=float)
    thrust_setting = np.array([0.07, 0.30, 0.85, 1.00])

    # Adjustment of EEDB fuel flows for installation effects
    installation_correction_factor = np.array([1.100, 1.020, 1.013, 1.010])
    fuel_flow *= installation_correction_factor

    # nvPM number emissions profile
    nvpm_ei_n = np.array([nvpm_ei_n_7, nvpm_ei_n_30, nvpm_ei_n_85, nvpm_ei_n_100], dtype=float)

    # Deal with lean-burn combustors the same way as T4/T2 methodology
    # Temporary solution as no details are provided in paper. Awaiting authors recommendation.
    is_staged_combustor = combustor in ("DAC", "TAPS", "TAPS II")
    if is_staged_combustor:
        fuel_flow = np.insert(fuel_flow, 2, fuel_flow[1] * 1.001)
        thrust_setting = np.insert(thrust_setting, 2, thrust_setting[1] * 1.001)
        nvpm_ei_n_lean_burn = np.mean(nvpm_ei_n[2:])
        nvpm_ei_n = np.r_[nvpm_ei_n[:2], [nvpm_ei_n_lean_burn] * 3]

    # Add fifth data point if the maximum nvPM EI number occurs between 30% and 85% of fuel flow
    elif fifth_data_point_number:
        if nvpm_ei_n_30_no_sl is None or nvpm_ei_n_85_no_sl is None or nvpm_ei_n_max_no_sl is None:
            raise ValueError(
                "nvpm_ei_n_30_no_sl, nvpm_ei_n_85_no_sl, and nvpm_ei_n_max_no_sl "
                "must be provided when fifth_data_point_number is True."
            )

        # Calculate fuel flow (5th point)
        ff_fifth = 0.5 * (fuel_flow[1] + fuel_flow[2])
        fuel_flow = np.insert(fuel_flow, 2, ff_fifth)
        thrust_setting = np.insert(thrust_setting, 2, 0.575)

        # Calculate nvPM number emissions index (5th point)
        k_loss_correction = 0.5 * (
            (nvpm_ei_n_30 / nvpm_ei_n_30_no_sl) + (nvpm_ei_n_85 / nvpm_ei_n_85_no_sl)
        )
        nvpm_ei_n_fifth = nvpm_ei_n_max_no_sl * k_loss_correction
        nvpm_ei_n = np.insert(nvpm_ei_n, 2, nvpm_ei_n_fifth)

    # Adjust nvPM emissions index due to fuel hydrogen content differences
    if hydrogen_content < 13.4 or hydrogen_content > 15.4:
        warnings.warn(
            f"Fuel hydrogen content {hydrogen_content} % is outside the valid range"
            "(13.4 - 15.4 %), and may lead to inaccuracies."
        )

    k_num = number_fuel_composition_correction_meem(hydrogen_content, thrust_setting)
    return EmissionsProfileInterpolator(xp=fuel_flow, fp=nvpm_ei_n * k_num)


def mass_fuel_composition_correction_meem(
    hydrogen_content: float | npt.NDArray[np.floating],
    thrust_setting: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    r"""
    Calculate fuel composition correction factor for nvPM mass emissions index.

    Parameters
    ----------
    hydrogen_content: float
        The percentage of hydrogen mass content in the fuel.
    thrust_setting : ArrayScalarLike
        Engine thrust setting, unitless

    Returns
    -------
    npt.NDArray[np.floating]
        nvPM mass fuel composition correction factor
    """
    return np.exp((1.08 * thrust_setting - 1.31) * (hydrogen_content - 13.8))


def number_fuel_composition_correction_meem(
    hydrogen_content: float | npt.NDArray[np.floating],
    thrust_setting: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    r"""
    Calculate fuel composition correction factor for nvPM number emissions index.

    Parameters
    ----------
    hydrogen_content: float
        The percentage of hydrogen mass content in the fuel.
    thrust_setting : ArrayScalarLike
        Engine thrust setting, unitless

    Returns
    -------
    npt.NDArray[np.floating]
        nvPM number fuel composition correction factor
    """
    return np.exp((0.99 * thrust_setting - 1.05) * (hydrogen_content - 13.8))


def estimate_nvpm_meem(
    nvpm_ei_m_profile: EmissionsProfileInterpolator,
    nvpm_ei_n_profile: EmissionsProfileInterpolator,
    fuel_flow_per_engine: npt.NDArray[np.floating],
    true_airspeed: npt.NDArray[np.floating],
    air_pressure: npt.NDArray[np.floating],
    air_temperature: npt.NDArray[np.floating],
    ff_7: float,
    ff_100: float,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    r"""Calculate nvPM mass and number emissions index using the MEEM2 methodology.

    Interpolate the non-volatile particulate matter (nvPM) mass and number emissions index from
    the emissions profile of a given engine type that is provided by the ICAO EDB.

    Parameters
    ----------
    nvpm_ei_m_profile : EmissionsProfileInterpolator
        MEEM2-derived nvPM mass emissions index versus the fuel flow for the selected engine
        See :func:`nvpm_mass_emission_profiles_meem`.
    nvpm_ei_n_profile : EmissionsProfileInterpolator
        MEEM2-derived nvPM number emissions index versus the fuel flow for the engine
        See :func:`nvpm_number_emission_profiles_meem`.
    fuel_flow_per_engine: npt.NDArray[np.floating]
        fuel mass flow rate per engine, [:math:`kg s^{-1}`]
    true_airspeed: npt.NDArray[np.floating]
        true airspeed for each waypoint, [:math:`m s^{-1}`]
    air_pressure: npt.NDArray[np.floating]
        pressure altitude at each waypoint, [:math:`Pa`]
    air_temperature: npt.NDArray[np.floating]
        ambient temperature for each waypoint, [:math:`K`]
    ff_7: float
        ICAO EDB fuel flow at idle (7% power) for the selected engine, [:math:`kg s^{-1}`]
    ff_100: float
        ICAO EDB fuel flow at take-off (100% power) for the selected engine, [:math:`kg s^{-1}`]

    Returns
    -------
    nvpm_ei_m : npt.NDArray[np.floating]
        nvPM mass emissions index, [:math:`kg/kg_{fuel}`]
    nvpm_ei_n : npt.NDArray[np.floating]
        nvPM number emissions index, [:math:`kg_{fuel}^{-1}`]

    References
    ----------
    # TODO: Add to bibliography
    - (Ahrens et al., 2025) https://doi.org/10.4271/2025-01-6000
    """
    # Fuel flow correction from altitude to ground
    mach_num = units.tas_to_mach_number(true_airspeed, air_temperature)
    theta_amb = jet.temperature_ratio(air_temperature)
    delta_amb = jet.pressure_ratio(air_pressure)
    fuel_flow_per_engine = jet.equivalent_fuel_flow_rate_at_sea_level(
        fuel_flow_per_engine, theta_amb, delta_amb, mach_num
    )
    fuel_flow_per_engine.clip(ff_7, ff_100, out=fuel_flow_per_engine)  # clip in place

    # Interpolate nvPM EI_m for ground conditions
    nvpm_ei_m_sl = nvpm_ei_m_profile.interp(fuel_flow_per_engine)
    nvpm_ei_m_sl = nvpm_ei_m_sl * 1e-6  # mg-nvPM/kg-fuel to kg-nvPM/kg-fuel

    # Interpolate nvPM EI_n for ground conditions
    nvpm_ei_n_sl = nvpm_ei_n_profile.interp(fuel_flow_per_engine)

    # Convert nvPM EI_m and EI_n from ground to cruise conditions
    nvpm_ei_m = nvpm_ei_m_sl * ((delta_amb**1.377) / (theta_amb**4.455)) * (1.1**2.5)
    nvpm_ei_n = (nvpm_ei_m / nvpm_ei_m_sl) * nvpm_ei_n_sl
    return nvpm_ei_m, nvpm_ei_n


# ---------------------------------------------------------------------------
# nvPM emissions: Smoke Correlation for Particle Emissions - CAEP11 (SCOPE11)
# ---------------------------------------------------------------------------


def number_ei_scope11(
    nvpm_ei_m_e: npt.NDArray[np.floating],
    sn: npt.NDArray[np.floating],
    air_temperature: npt.NDArray[np.floating] | float,
    air_pressure: npt.NDArray[np.floating] | float,
    thrust_setting: npt.NDArray[np.floating],
    afr: npt.NDArray[np.floating],
    q_fuel: float,
    bypass_ratio: float,
    pressure_ratio: float,
    comp_efficiency: float = 0.9,
) -> npt.NDArray[np.floating]:
    r"""
    Estimate nvPM number emissions index at the four ICAO certification test points using SCOPE11.

    Parameters
    ----------
    nvpm_ei_m_e : npt.NDArray[np.floating]
        nvPM mass emissions index at the engine exit, [:math:`kg \ kg_{fuel}^{-1}`]
        See :func:`estimate_nvpm_mass_ei_scope11`
    sn : npt.NDArray[np.floating]
        Smoke number, unitless
    air_temperature: npt.NDArray[np.floating]
        Ambient temperature for each waypoint, [:math:`K`]
    air_pressure: npt.NDArray[np.floating]
        Pressure altitude at each waypoint, [:math:`Pa`]
    thrust_setting : ArrayScalarLike
        Engine thrust setting, unitless
    afr : npt.NDArray[np.floating]
        Air-to-fuel ratio, unitless
        (106 at idle, 83 at approach, 51 at climb-out, and 45 at take-off)
    q_fuel : float
        Lower calorific value (LCV) of fuel, [:math:`J \ kg_{fuel}^{-1}`].
    bypass_ratio : float
        Engine bypass ratio from the ICAO EDB
    pressure_ratio : float
        Engine pressure ratio from the ICAO EDB
    comp_efficiency : float
        Engine compressor efficiency, assumed to be 0.9

    Returns
    -------
    npt.NDArray[np.floating]
        nvPM number emissions index, [:math:`kg_{fuel}^{-1}`]
    """
    c_bc_i = mass_concentration_instrument_sampling_point(sn)
    k_slm = mass_system_loss_correction_factor(c_bc_i, bypass_ratio)
    c_bc_e = mass_concentration_engine_exit(c_bc_i, k_slm)
    c_bc_c = mass_concentration_combustor_exit(
        c_bc_e,
        air_temperature,  # type: ignore[arg-type]
        air_pressure,  # type: ignore[arg-type]
        thrust_setting,
        afr,
        q_fuel,
        bypass_ratio,
        pressure_ratio,
        comp_efficiency,
    )
    nvpm_gmd = geometric_mean_diameter_scope11(c_bc_c) * 1e-9  # nm to m
    return nvpm_ei_m_e / ((np.pi / 6) * 1000.0 * (nvpm_gmd**3) * np.exp(4.5 * (np.log(1.8) ** 2)))


def mass_ei_scope11(
    sn: npt.NDArray[np.floating],
    afr: npt.NDArray[np.floating],
    bypass_ratio: float,
) -> npt.NDArray[np.floating]:
    r"""
    Estimate nvPM mass emissions index at the four ICAO certification test points using SCOPE11.

    Parameters
    ----------
    sn : npt.NDArray[np.floating]
        Smoke number, unitless
    afr : npt.NDArray[np.floating]
        Air-to-fuel ratio, unitless
        (106 at idle, 83 at approach, 51 at climb-out, and 45 at take-off)
    bypass_ratio : float
        Engine bypass ratio from the ICAO EDB

    Returns
    -------
    npt.NDArray[np.floating]
        nvPM mass emissions index at the engine exit, [:math:`kg \ kg_{fuel}^{-1}`]

    References
    ----------
    # TODO: Add to bibliography
    - (Agarwal et al., 2019) https://doi.org/10.1021/acs.est.8b04060
    """
    c_bc_i = mass_concentration_instrument_sampling_point(sn)
    q_mixed = exhaust_gas_volume_per_kg_fuel(afr, bypass_ratio=bypass_ratio)
    nvpm_ei_m_i = convert_nvpm_mass_concentration_to_ei(c_bc_i, q_mixed)
    k_slm = mass_system_loss_correction_factor(c_bc_i, bypass_ratio)
    return nvpm_ei_m_i * k_slm * 1e-9  # Convert from micro-g to kg


def mass_concentration_instrument_sampling_point(
    sn: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    r"""
    Estimate nvPM mass concentration at the instrument sampling point.

    Parameters
    ----------
    sn : npt.NDArray[np.floating]
        Smoke number, unitless

    Returns
    -------
    npt.NDArray[np.floating]
        nvPM mass concentration at the instrument sampling point, [:math:`\mu g m^{-3}]
    """
    return (648.4 * np.exp(0.0766 * sn)) / (1 + np.exp(-1.098 * (sn - 3.064)))


def mass_system_loss_correction_factor(
    c_bc_i: npt.NDArray[np.floating], bypass_ratio: float
) -> npt.NDArray[np.floating]:
    r"""
    Estimate nvPM mass concentration/EI system loss correction factors.

    Parameters
    ----------
    c_bc_i : npt.NDArray[np.floating]
        nvPM mass concentration at the instrument sampling point, [:math:`\mu g m^{-3}]
        See :func:`mass_concentration_instrument_sampling_point`
    bypass_ratio : float
        Engine bypass ratio from the ICAO EDB

    Returns
    -------
    npt.NDArray[np.floating]
        nvPM mass concentration/EI system loss correction factor.
    """
    numer = 3.219 * c_bc_i * (1 + bypass_ratio) + 312.5
    denom = c_bc_i * (1 + bypass_ratio) + 42.6
    return np.log(numer / denom)


def mass_concentration_engine_exit(
    c_bc_i: npt.NDArray[np.floating],
    k_slm: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    r"""
    Estimate nvPM mass concentration at the engine exit.

    Parameters
    ----------
    c_bc_i : npt.NDArray[np.floating]
        nvPM mass concentration at the instrument sampling point, [:math:`\mu g m^{-3}]
        See :func:`mass_concentration_instrument_sampling_point`
    k_slm : npt.NDArray[np.floating]
        nvPM mass concentration/EI system loss correction factor
        See :func:`mass_system_loss_correction_factor`

    Returns
    -------
    npt.NDArray[np.floating]
        nvPM mass concentration at the engine exit, [:math:`\mu g m^{-3}]
    """
    return c_bc_i * k_slm


def mass_concentration_combustor_exit(
    c_bc_e: npt.NDArray[np.floating],
    air_temperature: npt.NDArray[np.floating],
    air_pressure: npt.NDArray[np.floating],
    thrust_setting: npt.NDArray[np.floating],
    afr: npt.NDArray[np.floating],
    q_fuel: float,
    bypass_ratio: float,
    pressure_ratio: float,
    comp_efficiency: float = 0.9,
) -> npt.NDArray[np.floating]:
    r"""
    Estimate nvPM mass concentration at the combustor exit.

    Parameters
    ----------
    c_bc_e : npt.NDArray[np.floating]
        nvPM mass concentration at the engine exit plane, [:math:`\mu g m^{-3}]
        See :func:`mass_concentration_engine_exit`
    air_temperature: npt.NDArray[np.floating]
        Ambient temperature for each waypoint, [:math:`K`]
    air_pressure: npt.NDArray[np.floating]
        Pressure altitude at each waypoint, [:math:`Pa`]
    thrust_setting : ArrayScalarLike
        Engine thrust setting, unitless
    afr : npt.NDArray[np.floating]
        Air-to-fuel ratio, unitless
    q_fuel : float
        Lower calorific value (LCV) of fuel, [:math:`J \ kg_{fuel}^{-1}`].
    bypass_ratio : float
        Engine bypass ratio from the ICAO EDB
    pressure_ratio : float
        Engine pressure ratio from the ICAO EDB
    comp_efficiency : float
        Engine compressor efficiency, assumed to be 0.9

    Returns
    -------
    npt.NDArray[np.floating]
        nvPM mass concentration at the combustor exit, [:math:`\mu g m^{-3}]
    """
    rho_air_4 = air_density_combustor_exit(
        air_temperature, air_pressure, thrust_setting, afr, q_fuel, pressure_ratio, comp_efficiency
    )
    return c_bc_e * (1 + bypass_ratio) * (rho_air_4 / constants.rho_msl)


def geometric_mean_diameter_scope11(c_bc_c: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    r"""
    Estimate nvPM geometric mean diameter for SCOPE11 applications.

    Parameters
    ----------
    c_bc_c : npt.NDArray[np.floating]
        nvPM mass concentration at the combustor exit, [:math:`\mu g m^{-3}]
        See :func:`mass_concentration_combustor_exit`

    Returns
    -------
    npt.NDArray[np.floating]
        nvPM geometric mean diameter, [:math:`nm`]
    """
    return 5.08 * c_bc_c**0.185


def air_density_combustor_exit(
    air_temperature: npt.NDArray[np.floating],
    air_pressure: npt.NDArray[np.floating],
    thrust_setting: npt.NDArray[np.floating],
    afr: npt.NDArray[np.floating],
    q_fuel: float,
    pressure_ratio: float,
    comp_efficiency: float = 0.9,
) -> npt.NDArray[np.floating]:
    r"""
    Estimate air density at the combustor exit.

    Parameters
    ----------
    air_temperature : npt.NDArray[np.floating]
        Ambient temperature for each waypoint, [:math:`K`]
    air_pressure : npt.NDArray[np.floating]
        Pressure altitude at each waypoint, [:math:`Pa`]
    thrust_setting : ArrayScalarLike
        Engine thrust setting, unitless
    afr : npt.NDArray[np.floating]
        Air-to-fuel ratio, unitless
    q_fuel : float
        Lower calorific value (LCV) of fuel, [:math:`J \ kg_{fuel}^{-1}`].
    pressure_ratio : float
        Engine pressure ratio from the ICAO EDB
    comp_efficiency : float
        Engine compressor efficiency, assumed to be 0.9

    Returns
    -------
    npt.NDArray[np.floating]
        Air density at the combustor exit, [:math:`kg \ m^{3}`]
    """
    p_combustor_inlet = jet.combustor_inlet_pressure(pressure_ratio, air_pressure, thrust_setting)
    T_combustor_inlet = jet.combustor_inlet_temperature(
        comp_efficiency, air_temperature, air_pressure, p_combustor_inlet
    )
    T_turbine_inlet = jet.turbine_inlet_temperature(afr, T_combustor_inlet, q_fuel)
    return p_combustor_inlet / (constants.R_d * T_turbine_inlet)


# ---------------------------------------------------------
# nvPM Mass Emissions Index: Formation and Oxidation Method
# ---------------------------------------------------------


def mass_emissions_index_fox(
    air_pressure: npt.NDArray[np.floating],
    air_temperature: npt.NDArray[np.floating],
    true_airspeed: npt.NDArray[np.floating],
    fuel_flow_per_engine: npt.NDArray[np.floating],
    thrust_setting: npt.NDArray[np.floating],
    pressure_ratio: float,
    *,
    comp_efficiency: float = 0.9,
) -> npt.NDArray[np.floating]:
    r"""
    Calculate the nvPM mass emissions index using the Formation and Oxidation Method (FOX).

    Parameters
    ----------
    air_pressure: npt.NDArray[np.floating]
        Pressure altitude at each waypoint, [:math:`Pa`]
    air_temperature: npt.NDArray[np.floating]
        Ambient temperature for each waypoint, [:math:`K`]
    true_airspeed: npt.NDArray[np.floating]
        True airspeed for each waypoint, [:math:`m s^{-1}`]
    fuel_flow_per_engine: npt.NDArray[np.floating]
        Fuel mass flow rate per engine, [:math:`kg s^{-1}`]
    thrust_setting: npt.NDArray[np.floating]
        Engine thrust setting, which is the fuel mass flow rate divided by
        the maximum fuel mass flow rate
    pressure_ratio: float
        Engine pressure ratio from the ICAO EDB
    comp_efficiency: float
        Engine compressor efficiency, assumed to be 0.9

    Returns
    -------
    npt.NDArray[np.floating]
        nvPM mass emissions index, [:math:`mg \ kg_{fuel}^{-1}`]

    References
    ----------
    - :cite:`stettlerGlobalCivilAviation2013`
    """
    # Reference conditions: 100% thrust setting
    p_3_ref = jet.combustor_inlet_pressure(pressure_ratio, constants.p_surface, 1.0)
    t_3_ref = jet.combustor_inlet_temperature(
        comp_efficiency, constants.T_msl, constants.p_surface, p_3_ref
    )
    t_fl_ref = flame_temperature(t_3_ref)
    afr_ref = jet.air_to_fuel_ratio(1.0, cruise=False)
    fuel_flow_max = fuel_flow_per_engine / thrust_setting
    c_bc_ref = mass_concentration_fox(fuel_flow_max, t_fl_ref, afr_ref)

    # Cruise conditions
    mach_num = units.tas_to_mach_number(true_airspeed, air_temperature)
    t_2_cru = jet.compressor_inlet_temperature(air_temperature, mach_num)
    p_2_cru = jet.compressor_inlet_pressure(air_pressure, mach_num)
    p_3_cru = jet.combustor_inlet_pressure(pressure_ratio, p_2_cru, thrust_setting)
    t_3_cru = jet.combustor_inlet_temperature(comp_efficiency, t_2_cru, p_2_cru, p_3_cru)
    t_fl_cru = flame_temperature(t_3_cru)
    afr_cru = jet.air_to_fuel_ratio(thrust_setting, cruise=True, T_compressor_inlet=t_2_cru)
    c_bc_cru = mass_concentration_cruise_fox(
        c_bc_ref=c_bc_ref,
        t_fl_cru=t_fl_cru,
        t_fl_ref=t_fl_ref,
        p_3_cru=p_3_cru,
        p_3_ref=p_3_ref,
        afr_cru=afr_cru,
        afr_ref=afr_ref,
    )
    q_exhaust_cru = exhaust_gas_volume_per_kg_fuel(afr_cru)
    return convert_nvpm_mass_concentration_to_ei(c_bc_cru, q_exhaust_cru)


def flame_temperature(t_3: ArrayScalarLike) -> ArrayScalarLike:
    """
    Calculate the flame temperature at the combustion chamber (t_fl).

    Parameters
    ----------
    t_3: ArrayScalarLike
        Combustor inlet temperature, [:math:`K`]

    Returns
    -------
    ArrayScalarLike
        Flame temperature at the combustion chamber, [:math:`K`]
    """
    return 0.9 * t_3 + 2120.0


def mass_concentration_fox(
    fuel_flow: npt.NDArray[np.floating],
    t_fl: npt.NDArray[np.floating] | float,
    afr: npt.NDArray[np.floating] | float,
) -> npt.NDArray[np.floating]:
    """Calculate the nvPM mass concentration for ground conditions (``c_bc_ref``).

    This quantity is computed at the instrument sampling point without correcting
    for particle line losses.

    Parameters
    ----------
    fuel_flow: npt.NDArray[np.floating]
        Fuel mass flow rate, [:math:`kg s^{-1}`]
    t_fl: npt.NDArray[np.floating] | float
        Flame temperature at the combustion chamber, [:math:`K`]
    afr: npt.NDArray[np.floating] | float
        Air-to-fuel ratio

    Returns
    -------
    npt.NDArray[np.floating]:
        nvPM mass concentration for ground conditions, [:math:`mg m^{-3}`]
    """
    # avoid float32 -> float64 promotion
    coeff = 356.0 * np.exp(np.float32(-6390.0) / t_fl) - 608.0 * afr * np.exp(
        np.float32(-19778.0) / t_fl
    )
    return fuel_flow * coeff


def mass_concentration_cruise_fox(
    c_bc_ref: npt.NDArray[np.floating],
    t_fl_cru: npt.NDArray[np.floating],
    t_fl_ref: npt.NDArray[np.floating] | float,
    p_3_cru: npt.NDArray[np.floating],
    p_3_ref: npt.NDArray[np.floating] | float,
    afr_cru: npt.NDArray[np.floating],
    afr_ref: npt.NDArray[np.floating] | float,
) -> npt.NDArray[np.floating]:
    """Calculate the nvPM mass concentration for cruise conditions (``c_bc_cru``).

    This quantity is computed at the instrument sampling point without correcting
    for particle line losses.

    Parameters
    ----------
    c_bc_ref: npt.NDArray[np.floating]
        nvPM mass concentration at reference conditions, [:math:`mg m^{-3}`]
    t_fl_cru: npt.NDArray[np.floating]
        Flame temperature at cruise conditions, [:math:`K`]
    t_fl_ref: npt.NDArray[np.floating] | float
        Flame temperature at reference conditions, [:math:`K`]
    p_3_cru: npt.NDArray[np.floating]
        Combustor inlet pressure at cruise conditions, [:math:`Pa`]
    p_3_ref: npt.NDArray[np.floating] | float
        Combustor inlet pressure at reference conditions, [:math:`Pa`]
    afr_cru: npt.NDArray[np.floating]
        Air-to-fuel ratio at cruise conditions
    afr_ref: npt.NDArray[np.floating] | float
        Air-to-fuel ratio at reference conditions

    Returns
    -------
    npt.NDArray[np.floating]:
        nvPM mass concentration for cruise conditions, [:math:`mg m^{-3}`]
    """
    scaling_factor = dopelheuer_lecht_scaling_factor(
        t_fl_cru=t_fl_cru,
        t_fl_ref=t_fl_ref,
        p_3_cru=p_3_cru,
        p_3_ref=p_3_ref,
        afr_cru=afr_cru,
        afr_ref=afr_ref,
    )
    return c_bc_ref * scaling_factor


def dopelheuer_lecht_scaling_factor(
    t_fl_cru: npt.NDArray[np.floating],
    t_fl_ref: npt.NDArray[np.floating] | float,
    p_3_cru: npt.NDArray[np.floating],
    p_3_ref: npt.NDArray[np.floating] | float,
    afr_cru: npt.NDArray[np.floating],
    afr_ref: npt.NDArray[np.floating] | float,
) -> npt.NDArray[np.floating]:
    """Estimate scaling factor to convert reference nvPM mass concentration from ground to cruise.

    Parameters
    ----------
    t_fl_cru: npt.NDArray[np.floating]
        Flame temperature at cruise conditions, [:math:`K`]
    t_fl_ref: npt.NDArray[np.floating] | float
        Flame temperature at reference conditions, [:math:`K`]
    p_3_cru: npt.NDArray[np.floating]
        Combustor inlet pressure at cruise conditions, [:math:`Pa`]
    p_3_ref: npt.NDArray[np.floating] | float
        Combustor inlet pressure at reference conditions, [:math:`Pa`]
    afr_cru: npt.NDArray[np.floating]
        Air-to-fuel ratio at cruise conditions
    afr_ref: npt.NDArray[np.floating] | float
        Air-to-fuel ratio at reference conditions

    Returns
    -------
    npt.NDArray[np.floating]
        Dopelheuer & Lecht scaling factor

    References
    ----------
    - :cite:`dopelheuerInfluenceEnginePerformance1998`
    """
    exp_term = np.exp(20000.0 / t_fl_cru - 20000.0 / t_fl_ref)
    return (afr_ref / afr_cru) ** 2.5 * (p_3_cru / p_3_ref) ** 1.35 * exp_term


# ----------------------------------------------------------------------------
# nvPM Mass Emissions Index: "Improved" Formation and Oxidation Method (ImFOX)
# ----------------------------------------------------------------------------


def mass_emissions_index_imfox(
    fuel_flow_per_engine: npt.NDArray[np.floating],
    thrust_setting: npt.NDArray[np.floating],
    fuel_hydrogen: float,
) -> npt.NDArray[np.floating]:
    r"""Calculate the nvPM mass EI using the "Improved" Formation and Oxidation Method (ImFOX).

    Parameters
    ----------
    fuel_flow_per_engine: npt.NDArray[np.floating]
        Fuel mass flow rate per engine, [:math:`kg s^{-1}`]
    thrust_setting: npt.NDArray[np.floating]
        Engine thrust setting, which is the fuel mass flow rate divided by the
        maximum fuel mass flow rate
    fuel_hydrogen: float
        Percentage of hydrogen mass content in the fuel (13.8% for conventional Jet A-1 fuel)

    Returns
    -------
    npt.NDArray[np.floating]
        nvPM mass emissions index, [:math:`mg \ kg_{fuel}^{-1}`]

    References
    ----------
    - :cite:`abrahamsonPredictiveModelDevelopment2016`
    """
    afr_cru = air_to_fuel_ratio_imfox(thrust_setting)
    t_4_cru = turbine_inlet_temperature_imfox(afr_cru)
    c_bc_cru = mass_concentration_imfox(
        fuel_flow_per_engine, afr_cru, t_4_cru, fuel_hydrogen=fuel_hydrogen
    )
    q_exhaust_cru = exhaust_gas_volume_per_kg_fuel(afr_cru)
    return convert_nvpm_mass_concentration_to_ei(c_bc_cru, q_exhaust_cru)


def air_to_fuel_ratio_imfox(thrust_setting: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    """Calculate the air-to-fuel ratio at cruise conditions via Abrahamson's method.

    See Eq. (11) in :cite:`abrahamsonPredictiveModelDevelopment2016`.

    Parameters
    ----------
    thrust_setting: npt.NDArray[np.floating]
        Engine thrust setting, which is the fuel mass flow rate divided by
        the maximum fuel mass flow rate

    Returns
    -------
    npt.NDArray[np.floating]
        Air-to-fuel ratio at cruise conditions

    References
    ----------
    - :cite:`abrahamsonPredictiveModelDevelopment2016`
    """
    return 55.4 - 30.8 * thrust_setting


def turbine_inlet_temperature_imfox(afr: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    """Calculate the turbine inlet temperature using Abrahamson's method.

    See Eq. (13) in :cite:`abrahamsonPredictiveModelDevelopment2016`.

    Parameters
    ----------
    afr: npt.NDArray[np.floating]
        air-to-fuel ratio at cruise conditions

    Returns
    -------
    npt.NDArray[np.floating]
        turbine inlet temperature, [:math:`K`]

    References
    ----------
    - :cite:`abrahamsonPredictiveModelDevelopment2016`
    """
    return 490.0 + 42266.0 / afr


def mass_concentration_imfox(
    fuel_flow_per_engine: npt.NDArray[np.floating],
    afr: npt.NDArray[np.floating],
    t_4: npt.NDArray[np.floating],
    fuel_hydrogen: float,
) -> npt.NDArray[np.floating]:
    """Calculate nvPM mass concentration for ground and cruise conditions with ImFOX methodology.

    This quantity is computed at the instrument sampling point without
    correcting for particle line losses.

    Parameters
    ----------
    fuel_flow_per_engine: npt.NDArray[np.floating]
        fuel mass flow rate per engine, [:math:`kg s^{-1}`]
    afr: npt.NDArray[np.floating]
        air-to-fuel ratio
    t_4: npt.NDArray[np.floating]
        turbine inlet temperature, [:math:`K`]
    fuel_hydrogen: float
        percentage of hydrogen mass content in the fuel (13.8% for conventional Jet A-1 fuel)

    Returns
    -------
    npt.NDArray[np.floating]
        nvPM mass concentration, [:math:`mg m^{-3}`]
    """
    # avoid float32 -> float64 promotion
    exp_term = np.exp(np.float32(13.6) - fuel_hydrogen)
    formation_term = 295.0 * np.exp(np.float32(-6390.0) / t_4)
    oxidation_term = 608.0 * afr * np.exp(np.float32(-19778.0) / t_4)
    return fuel_flow_per_engine * exp_term * (formation_term - oxidation_term)


# ----------------------------------------------------------
# nvPM Number Emissions Index: Fractal Aggregates (FA) model
# ----------------------------------------------------------


def geometric_mean_diameter_sac(
    air_pressure: npt.NDArray[np.floating],
    air_temperature: npt.NDArray[np.floating],
    true_airspeed: npt.NDArray[np.floating],
    thrust_setting: npt.NDArray[np.floating],
    pressure_ratio: float,
    q_fuel: float,
    *,
    comp_efficiency: float = 0.9,
    delta_loss: float = 5.75,
    cruise: bool = True,
) -> npt.NDArray[np.floating]:
    r"""Calculate the nvPM GMD for singular annular combustor (SAC) engines.

    The nvPM GMD (geometric mean diameter) is estimated using the non-dimensionalized engine thrust
    setting, the ratio of turbine inlet to the compressor inlet temperature (``t4_t2``).

    Parameters
    ----------
    air_pressure: npt.NDArray[np.floating]
        Pressure altitude at each waypoint, [:math:`Pa`]
    air_temperature: npt.NDArray[np.floating]
        Ambient temperature for each waypoint, [:math:`K`]
    true_airspeed: npt.NDArray[np.floating]
        True airspeed for each waypoint, [:math:`m s^{-1}`]
    thrust_setting: npt.NDArray[np.floating]
        Engine thrust setting, which is the fuel mass flow rate divided by the
        maximum fuel mass flow rate
    pressure_ratio: float
        Engine pressure ratio from the ICAO EDB
    q_fuel : float
        Lower calorific value (LCV) of fuel, [:math:`J \ kg_{fuel}^{-1}`]
    comp_efficiency: float
        Engine compressor efficiency (assumed to be 0.9)
    delta_loss: float
        Correction factor accounting for particle line losses (assumed to be 5.75 nm), [:math:`nm`]
    cruise: bool
        Set to true when the aircraft is not on the ground.

    Returns
    -------
    npt.NDArray[np.floating]
        nvPM geometric mean diameter, [:math:`nm`]

    References
    ----------
    - :cite:`teohMitigatingClimateForcing2020`
    """
    t4_t2 = jet.thrust_setting_nd(
        true_airspeed,
        thrust_setting,
        air_temperature,
        air_pressure,
        pressure_ratio,
        q_fuel,
        comp_efficiency=comp_efficiency,
        cruise=cruise,
    )
    return (2.5883 * t4_t2**2) - (5.3723 * t4_t2) + 16.721 - delta_loss


def number_emissions_index_fractal_aggregates(
    nvpm_ei_m: npt.NDArray[np.floating],
    gmd: npt.NDArray[np.floating],
    *,
    gsd: float | np.floating | npt.NDArray[np.floating] = np.float32(1.80),  # avoid promotion
    rho_bc: float | np.floating = np.float32(1770.0),
    k_tem: float | np.floating = np.float32(1.621e-5),
    d_tem: float | np.floating = np.float32(0.39),
    d_fm: float | np.floating = np.float32(2.76),
) -> npt.NDArray[np.floating]:
    """
    Estimate the nvPM number emission index using the fractal aggregates (FA) model.

    The FA model estimates the number emissions index from the mass emissions index,
    particle size distribution, and morphology.

    Parameters
    ----------
    nvpm_ei_m: npt.NDArray[np.floating]
        nvPM mass emissions index, [:math:`kg/kg_{fuel}`]
    gmd: npt.NDArray[np.floating]
        nvPM geometric mean diameter, [:math:`m`]
    gsd: float
        nvPM geometric standard deviation (assumed to be 1.80)
    rho_bc: float
        nvPM material density (1770 kg/m**3), [:math:`kg m^{-3}`]
    k_tem: float
        Transmission electron microscopy prefactor coefficient (assumed to be 1.621e-5)
    d_tem: float
        Transmission electron microscopy exponent coefficient (assumed to be 0.39)
    d_fm: float
        Mass-mobility exponent (assumed to be 2.76)

    Returns
    -------
    npt.NDArray[np.floating]
        nvPM number emissions index, [:math:`kg_{fuel}^{-1}`]

    References
    ----------
    - FA model: :cite:`teohMethodologyRelateBlack2019`
    - ``gmd``, ``gsd``, ``d_fm``: :cite:`teohMitigatingClimateForcing2020`
    - ``rho_bc``: :cite:`parkMeasurementInherentMaterial2004`
    - ``k_tem``, ``d_tem``: :cite:`dastanpourObservationsCorrelationPrimary2014`
    """
    phi = 3.0 * d_tem + (1.0 - d_tem) * d_fm
    exponential_term = np.exp(0.5 * phi**2 * np.log(gsd) ** 2)
    denom = rho_bc * (np.pi / 6.0) * k_tem ** (3.0 - d_fm) * gmd**phi * exponential_term
    return nvpm_ei_m / denom


# -------------------------------------------------------------
# Scale nvPM emissions indices due to sustainable aviation fuel
# -------------------------------------------------------------


def nvpm_number_ei_pct_reduction_due_to_saf(
    hydrogen_content: float | npt.NDArray[np.floating],
    thrust_setting: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    """
    Adjust nvPM number emissions index to account for the effects of sustainable aviation fuels.

    Parameters
    ----------
    hydrogen_content: float
        The percentage of hydrogen mass content in the fuel.
    thrust_setting: npt.NDArray[np.floating]
        Engine thrust setting, where the equivalent fuel mass flow rate per engine at
        sea level, :math:`[0 - 1]`.

    Returns
    -------
    npt.NDArray[np.floating]
        Percentage reduction in nvPM number emissions index

    References
    ----------
    - :cite:`teohTargetedUseSustainable2022`
    - :cite:`bremEffectsFuelAromatic2015`
    """
    a0 = -114.21
    a1 = 1.06
    a2 = 0.5
    return _template_saf_reduction(hydrogen_content, thrust_setting, a0, a1, a2)


def nvpm_mass_ei_pct_reduction_due_to_saf(
    hydrogen_content: float | npt.NDArray[np.floating],
    thrust_setting: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    """
    Adjust nvPM mass emissions index to account for the effects of sustainable aviation fuels.

    For fuel with hydrogen mass content > 14.3, the adjustment factor is adopted from
    Teoh et al. (2022), which was used to calculate the change in nvPM EIn.

    Parameters
    ----------
    hydrogen_content: float
        The percentage of hydrogen mass content in the fuel.
    thrust_setting: npt.NDArray[np.floating]
        Engine thrust setting, where the equivalent fuel mass flow rate per engine at
        sea level, :math:`[0 - 1]`.

    Returns
    -------
    npt.NDArray[np.floating]
        Percentage reduction in nvPM number emissions index

    References
    ----------
    - :cite:`teohTargetedUseSustainable2022`
    - :cite:`bremEffectsFuelAromatic2015`
    """
    a0 = -124.05
    a1 = 1.02
    a2 = 0.6
    return _template_saf_reduction(hydrogen_content, thrust_setting, a0, a1, a2)


def _template_saf_reduction(
    hydrogen_content: float | npt.NDArray[np.floating],
    thrust_setting: npt.NDArray[np.floating],
    a0: float,
    a1: float,
    a2: float,
) -> npt.NDArray[np.floating]:
    # Thrust setting cannot be computed when engine data is not provided in
    # the ICAO EDB, so set default to 45% thrust.
    thrust_setting = np.nan_to_num(thrust_setting, nan=0.45)
    delta_h = hydrogen_content - 13.80
    d_nvpm_ein_pct = (a0 + a1 * (thrust_setting * 100.0)) * delta_h

    # Adjust when delta_h is large
    if isinstance(delta_h, np.ndarray):
        filt = delta_h > 0.5
        d_nvpm_ein_pct[filt] *= np.exp(0.5 * (a2 - delta_h[filt]))
    elif delta_h > 0.5:
        d_nvpm_ein_pct *= np.exp(0.5 * (a2 - delta_h))

    d_nvpm_ein_pct.clip(min=-90.0, max=0.0, out=d_nvpm_ein_pct)
    return d_nvpm_ein_pct


# -----------------------
# Commonly used functions
# -----------------------


def exhaust_gas_volume_per_kg_fuel(
    afr: npt.NDArray[np.floating],
    *,
    bypass_ratio: float = 0.0,
) -> npt.NDArray[np.floating]:
    """
    Calculate the volume of exhaust gas per mass of fuel burnt.

    Parameters
    ----------
    afr: npt.NDArray[np.floating]
        Air-to-fuel ratio
    bypass_ratio: float
        Engine bypass ratio

    Returns
    -------
    npt.NDArray[np.floating]
        Volume of exhaust gas per mass of fuel burnt, [:math:`m^{3}/kg_{fuel}`]

    References
    ----------
    - :cite:`stettlerGlobalCivilAviation2013`
    # TODO: Add to bibliography
    - (Agarwal et al., 2019) https://doi.org/10.1021/acs.est.8b04060
    """
    return 0.776 * afr * (1 + bypass_ratio) + 0.877


def convert_nvpm_mass_concentration_to_ei(
    c_bc: npt.NDArray[np.floating], q_exhaust: npt.NDArray[np.floating]
) -> npt.NDArray[np.floating]:
    """
    Convert the nvPM mass concentration to an emissions index.

    Parameters
    ----------
    c_bc: npt.NDArray[np.floating]
        nvPM mass concentration, [:math:`mg m^{-3}`]
    q_exhaust: npt.NDArray[np.floating]
        Volume of exhaust gas per mass of fuel burnt, [:math:`m^{3}/kg_{fuel}`]

    Returns
    -------
    npt.NDArray[np.floating]
        nvPM mass emissions index, [:math:`mg/kg_{fuel}`]

    References
    ----------
    - :cite:`stettlerGlobalCivilAviation2013`
    """
    return c_bc * q_exhaust
