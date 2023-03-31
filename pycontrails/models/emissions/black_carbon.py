"""Non-volatile particulate matter (nvPM) calculations."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from pycontrails.physics import constants, jet, units
from pycontrails.utils.types import ArrayScalarLike

# ---------------------------------------------------------
# nvPM Mass Emissions Index: Formation and Oxidation Method
# ---------------------------------------------------------


def mass_emissions_index_fox(
    air_pressure: npt.NDArray[np.float_],
    air_temperature: npt.NDArray[np.float_],
    true_airspeed: npt.NDArray[np.float_],
    fuel_flow_per_engine: npt.NDArray[np.float_],
    thrust_setting: npt.NDArray[np.float_],
    pressure_ratio: float,
    *,
    comp_efficiency: float = 0.9,
) -> npt.NDArray[np.float_]:
    r"""
    Calculate the black carbon mass emissions index using the Formation and Oxidation Method (FOX).

    Parameters
    ----------
    air_pressure: npt.NDArray[np.float_]
        Pressure altitude at each waypoint, [:math:`Pa`]
    air_temperature: npt.NDArray[np.float_]
        Ambient temperature for each waypoint, [:math:`K`]
    true_airspeed: npt.NDArray[np.float_]
        True airspeed for each waypoint, [:math:`m s^{-1}`]
    fuel_flow_per_engine: npt.NDArray[np.float_]
        Fuel mass flow rate per engine, [:math:`kg s^{-1}`]
    thrust_setting: npt.NDArray[np.float_]
        Engine thrust setting, which is the fuel mass flow rate divided by
        the maximum fuel mass flow rate
    pressure_ratio: float
        Engine pressure ratio from the ICAO EDB
    comp_efficiency: float
        Engine compressor efficiency, assumed to be 0.9

    Returns
    -------
    npt.NDArray[np.float_]
        Black carbon mass emissions index, [:math:`mg \ kg_{fuel}^{-1}`]

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
    c_bc_ref = bc_mass_concentration_fox(fuel_flow_max, t_fl_ref, afr_ref)

    # Cruise conditions
    mach_num = units.tas_to_mach_number(true_airspeed, air_temperature)
    t_2_cru = jet.compressor_inlet_temperature(air_temperature, mach_num)
    p_2_cru = jet.compressor_inlet_pressure(air_pressure, mach_num)
    p_3_cru = jet.combustor_inlet_pressure(pressure_ratio, p_2_cru, thrust_setting)
    t_3_cru = jet.combustor_inlet_temperature(comp_efficiency, t_2_cru, p_2_cru, p_3_cru)
    t_fl_cru = flame_temperature(t_3_cru)
    afr_cru = jet.air_to_fuel_ratio(thrust_setting, cruise=True, T_compressor_inlet=t_2_cru)
    c_bc_cru = bc_mass_concentration_cruise_fox(
        c_bc_ref=c_bc_ref,
        t_fl_cru=t_fl_cru,
        t_fl_ref=t_fl_ref,
        p_3_cru=p_3_cru,
        p_3_ref=p_3_ref,
        afr_cru=afr_cru,
        afr_ref=afr_ref,
    )
    q_exhaust_cru = exhaust_gas_volume_per_kg_fuel(afr_cru)
    return bc_mass_emissions_index(c_bc_cru, q_exhaust_cru)


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
    return 0.9 * t_3 + 2120


def bc_mass_concentration_fox(
    fuel_flow: npt.NDArray[np.float_],
    t_fl: npt.NDArray[np.float_] | float,
    afr: npt.NDArray[np.float_] | float,
) -> npt.NDArray[np.float_]:
    """Calculate the black carbon mass concentration for ground conditions (``c_bc_ref``).

    This quantity is computed at the instrument sampling point without correcting
    for particle line losses.

    Parameters
    ----------
    fuel_flow: npt.NDArray[np.float_]
        Fuel mass flow rate, [:math:`kg s^{-1}`]
    t_fl: npt.NDArray[np.float_] | float
        Flame temperature at the combustion chamber, [:math:`K`]
    afr: npt.NDArray[np.float_] | float
        Air-to-fuel ratio

    Returns
    -------
    npt.NDArray[np.float_]:
        Black carbon mass concentration for ground conditions, [:math:`mg m^{-3}`]
    """
    return fuel_flow * (356 * np.exp(-6390 / t_fl) - 608 * afr * np.exp(-19778 / t_fl))


def bc_mass_concentration_cruise_fox(
    c_bc_ref: npt.NDArray[np.float_],
    t_fl_cru: npt.NDArray[np.float_],
    t_fl_ref: npt.NDArray[np.float_] | float,
    p_3_cru: npt.NDArray[np.float_],
    p_3_ref: npt.NDArray[np.float_] | float,
    afr_cru: npt.NDArray[np.float_],
    afr_ref: npt.NDArray[np.float_] | float,
) -> npt.NDArray[np.float_]:
    """Calculate the black carbon mass concentration for cruise conditions (``c_bc_cru``).

    This quantity is computed at the instrument sampling point without correcting
    for particle line losses.

    Parameters
    ----------
    c_bc_ref: npt.NDArray[np.float_]
        Black carbon mass concentration at reference conditions, [:math:`mg m^{-3}`]
    t_fl_cru: npt.NDArray[np.float_]
        Flame temperature at cruise conditions, [:math:`K`]
    t_fl_ref: npt.NDArray[np.float_] | float
        Flame temperature at reference conditions, [:math:`K`]
    p_3_cru: npt.NDArray[np.float_]
        Combustor inlet pressure at cruise conditions, [:math:`Pa`]
    p_3_ref: npt.NDArray[np.float_] | float
        Combustor inlet pressure at reference conditions, [:math:`Pa`]
    afr_cru: npt.NDArray[np.float_]
        Air-to-fuel ratio at cruise conditions
    afr_ref: npt.NDArray[np.float_] | float
        Air-to-fuel ratio at reference conditions

    Returns
    -------
    npt.NDArray[np.float_]:
        Black carbon mass concentration for cruise conditions, [:math:`mg m^{-3}`]
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
    t_fl_cru: npt.NDArray[np.float_],
    t_fl_ref: npt.NDArray[np.float_] | float,
    p_3_cru: npt.NDArray[np.float_],
    p_3_ref: npt.NDArray[np.float_] | float,
    afr_cru: npt.NDArray[np.float_],
    afr_ref: npt.NDArray[np.float_] | float,
) -> npt.NDArray[np.float_]:
    """Estimate scaling factor to convert the reference BC mass concentration from ground to cruise.

    Parameters
    ----------
    t_fl_cru: npt.NDArray[np.float_]
        Flame temperature at cruise conditions, [:math:`K`]
    t_fl_ref: npt.NDArray[np.float_] | float
        Flame temperature at reference conditions, [:math:`K`]
    p_3_cru: npt.NDArray[np.float_]
        Combustor inlet pressure at cruise conditions, [:math:`Pa`]
    p_3_ref: npt.NDArray[np.float_] | float
        Combustor inlet pressure at reference conditions, [:math:`Pa`]
    afr_cru: npt.NDArray[np.float_]
        Air-to-fuel ratio at cruise conditions
    afr_ref: npt.NDArray[np.float_] | float
        Air-to-fuel ratio at reference conditions

    Returns
    -------
    npt.NDArray[np.float_]
        Dopelheuer & Lecht scaling factor

    References
    ----------
    - :cite:`dopelheuerInfluenceEnginePerformance1998`
    """
    exp_term = np.exp(20000 / t_fl_cru) / np.exp(20000 / t_fl_ref)
    return (afr_ref / afr_cru) ** 2.5 * (p_3_cru / p_3_ref) ** 1.35 * exp_term


# ----------------------------------------------------------------------------
# nvPM Mass Emissions Index: "Improved" Formation and Oxidation Method (ImFOX)
# ----------------------------------------------------------------------------


def mass_emissions_index_imfox(
    fuel_flow_per_engine: npt.NDArray[np.float_],
    thrust_setting: npt.NDArray[np.float_],
    fuel_hydrogen: float,
) -> npt.NDArray[np.float_]:
    r"""Calculate the BC mass EI using the "Improved" Formation and Oxidation Method (ImFOX).

    Parameters
    ----------
    fuel_flow_per_engine: npt.NDArray[np.float_]
        Fuel mass flow rate per engine, [:math:`kg s^{-1}`]
    thrust_setting: npt.NDArray[np.float_]
        Engine thrust setting, which is the fuel mass flow rate divided by the
        maximum fuel mass flow rate
    fuel_hydrogen: float
        Percentage of hydrogen mass content in the fuel (13.8% for conventional Jet A-1 fuel)

    Returns
    -------
    npt.NDArray[np.float_]
        Black carbon mass emissions index, [:math:`mg \ kg_{fuel}^{-1}`]

    References
    ----------
    - :cite:`abrahamsonPredictiveModelDevelopment2016`
    """
    afr_cru = air_to_fuel_ratio_imfox(thrust_setting)
    t_4_cru = turbine_inlet_temperature_imfox(afr_cru)
    c_bc_cru = bc_mass_concentration_imfox(
        fuel_flow_per_engine, afr_cru, t_4_cru, fuel_hydrogen=fuel_hydrogen
    )
    q_exhaust_cru = exhaust_gas_volume_per_kg_fuel(afr_cru)
    return bc_mass_emissions_index(c_bc_cru, q_exhaust_cru)


def air_to_fuel_ratio_imfox(thrust_setting: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
    """Calculate the air-to-fuel ratio at cruise conditions via Abrahamson's method.

    See Eq. (11) in :cite:`abrahamsonPredictiveModelDevelopment2016`.

    Parameters
    ----------
    thrust_setting: npt.NDArray[np.float_]
        Engine thrust setting, which is the fuel mass flow rate divided by
        the maximum fuel mass flow rate

    Returns
    -------
    npt.NDArray[np.float_]
        Air-to-fuel ratio at cruise conditions

    References
    ----------
    - :cite:`abrahamsonPredictiveModelDevelopment2016`
    """
    return 55.4 - 30.8 * thrust_setting


def turbine_inlet_temperature_imfox(afr: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
    """Calculate the turbine inlet temperature using Abrahamson's method.

    See Eq. (13) in :cite:`abrahamsonPredictiveModelDevelopment2016`.

    Parameters
    ----------
    afr: npt.NDArray[np.float_]
        air-to-fuel ratio at cruise conditions

    Returns
    -------
    npt.NDArray[np.float_]
        turbine inlet temperature, [:math:`K`]

    References
    ----------
    - :cite:`abrahamsonPredictiveModelDevelopment2016`
    """
    return 490 + 42266 / afr


def bc_mass_concentration_imfox(
    fuel_flow_per_engine: npt.NDArray[np.float_],
    afr: npt.NDArray[np.float_],
    t_4: npt.NDArray[np.float_],
    fuel_hydrogen: float,
) -> npt.NDArray[np.float_]:
    """Calculate the BC mass concentration for ground and cruise conditions with ImFOX methodology.

    This quantity is computed at the instrument sampling point without
    correcting for particle line losses.

    Parameters
    ----------
    fuel_flow_per_engine: npt.NDArray[np.float_]
        fuel mass flow rate per engine, [:math:`kg s^{-1}`]
    afr: npt.NDArray[np.float_]
        air-to-fuel ratio
    t_4: npt.NDArray[np.float_]
        turbine inlet temperature, [:math:`K`]
    fuel_hydrogen: float
        percentage of hydrogen mass content in the fuel (13.8% for conventional Jet A-1 fuel)

    Returns
    -------
    npt.NDArray[np.float_]
        Black carbon mass concentration, [:math:`mg m^{-3}`]
    """
    exp_term = np.exp(13.6 - fuel_hydrogen)
    formation_term = 295 * np.exp(-6390 / t_4)
    oxidation_term = 608 * afr * np.exp(-19778 / t_4)
    return fuel_flow_per_engine * exp_term * (formation_term - oxidation_term)


# ---------------------------------------------------------
# Commonly Used Black Carbon Mass Emissions Index Functions
# ---------------------------------------------------------


def exhaust_gas_volume_per_kg_fuel(afr: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
    """
    Calculate the volume of exhaust gas per mass of fuel burnt.

    Parameters
    ----------
    afr: npt.NDArray[np.float_]
        Air-to-fuel ratio

    Returns
    -------
    npt.NDArray[np.float_]
        Volume of exhaust gas per mass of fuel burnt, [:math:`m^{3}/kg_{fuel}`]

    References
    ----------
    - :cite:`stettlerGlobalCivilAviation2013`
    """
    return 0.776 * afr + 0.877


def bc_mass_emissions_index(
    c_bc: npt.NDArray[np.float_], q_exhaust: npt.NDArray[np.float_]
) -> npt.NDArray[np.float_]:
    """
    Calculate the black carbon mass emissions index.

    Parameters
    ----------
    c_bc: npt.NDArray[np.float_]
        Black carbon mass concentration, [:math:`mg m^{-3}`]
    q_exhaust: npt.NDArray[np.float_]
        Volume of exhaust gas per mass of fuel burnt, [:math:`m^{3}/kg_{fuel}`]

    Returns
    -------
    npt.NDArray[np.float_]
        Black carbon mass emissions index, [:math:`mg/kg_{fuel}`]

    References
    ----------
    - :cite:`stettlerGlobalCivilAviation2013`
    """
    return c_bc * q_exhaust


# ----------------------------------------------------------
# nvPM Number Emissions Index: Fractal Aggregates (FA) model
# ----------------------------------------------------------


def geometric_mean_diameter_sac(
    air_pressure: npt.NDArray[np.float_],
    air_temperature: npt.NDArray[np.float_],
    true_airspeed: npt.NDArray[np.float_],
    thrust_setting: npt.NDArray[np.float_],
    pressure_ratio: float,
    q_fuel: float,
    *,
    comp_efficiency: float = 0.9,
    delta_loss: float = 5.75,
    cruise: bool = True,
) -> npt.NDArray[np.float_]:
    r"""Calculate the BC GMD for singular annular combustor (SAC) engines.

    The BC (black carbon) GMD (geometric mean diameter) is estimated using
    the non-dimensionalized engine thrust setting, the ratio of turbine inlet
    to the compressor inlet temperature (``t4_t2``).

    Parameters
    ----------
    air_pressure: npt.NDArray[np.float_]
        Pressure altitude at each waypoint, [:math:`Pa`]
    air_temperature: npt.NDArray[np.float_]
        Ambient temperature for each waypoint, [:math:`K`]
    true_airspeed: npt.NDArray[np.float_]
        True airspeed for each waypoint, [:math:`m s^{-1}`]
    thrust_setting: npt.NDArray[np.float_]
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
    npt.NDArray[np.float_]
        black carbon geometric mean diameter, [:math:`nm`]

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
    nvpm_ei_m: npt.NDArray[np.float_],
    gmd: npt.NDArray[np.float_],
    *,
    gsd: float | npt.NDArray[np.float_] = 1.80,
    rho_bc: float = 1770,
    k_tem: float = 1.621e-5,
    d_tem: float = 0.39,
    d_fm: float = 2.76,
) -> npt.NDArray[np.float_]:
    """
    Estimate the black carbon number emission index using the fractal aggregates (FA) model.

    The FA model estimates the number emissions index from the mass emissions index,
    particle size distribution, and morphology.

    Parameters
    ----------
    nvpm_ei_m: npt.NDArray[np.float_]
        Black carbon mass emissions index, [:math:`kg/kg_{fuel}`]
    gmd: npt.NDArray[np.float_]
        Black carbon geometric mean diameter, [:math:`m`]
    gsd: float
        Black carbon geometric standard deviation (assumed to be 1.80)
    rho_bc: float
        Black carbon material density (1770 kg/m**3), [:math:`kg m^{-3}`]
    k_tem: float
        Transmission electron microscopy prefactor coefficient (assumed to be 1.621e-5)
    d_tem: float
        Transmission electron microscopy exponent coefficient (assumed to be 0.39)
    d_fm: float
        Mass-mobility exponent (assumed to be 2.76)

    Returns
    -------
    npt.NDArray[np.float_]
        Black carbon number emissions index, [:math:`kg_{fuel}^{-1}`]

    References
    ----------
    - FA model: :cite:`teohMethodologyRelateBlack2019`
    - ``gmd``, ``gsd``, ``d_fm``: :cite:`teohMitigatingClimateForcing2020`
    - ``rho_bc``: :cite:`parkMeasurementInherentMaterial2004`
    - ``k_tem``, ``d_tem``: :cite:`dastanpourObservationsCorrelationPrimary2014`
    """
    phi = 3 * d_tem + (1 - d_tem) * d_fm
    exponential_term = np.exp(0.5 * phi**2 * np.log(gsd) ** 2)
    denom = rho_bc * (np.pi / 6) * k_tem ** (3 - d_fm) * gmd**phi * exponential_term
    return nvpm_ei_m / denom


# -------------------------------------------------------------
# Scale nvPM emissions indices due to sustainable aviation fuel
# -------------------------------------------------------------


def nvpm_number_ei_pct_reduction_due_to_saf(
    hydrogen_content: float | npt.NDArray[np.float_],
    thrust_setting: npt.NDArray[np.float_],
) -> npt.NDArray[np.float_]:
    """
    Adjust nvPM number emissions index to account for the effects of sustainable aviation fuels.

    Parameters
    ----------
    hydrogen_content: float
        The percentage of hydrogen mass content in the fuel.
    thrust_setting: npt.NDArray[np.float_]
        Engine thrust setting, where the equivalent fuel mass flow rate per engine at
        sea level, :math:`[0 - 1]`.

    Returns
    -------
    npt.NDArray[np.float_]
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
    hydrogen_content: float | npt.NDArray[np.float_],
    thrust_setting: npt.NDArray[np.float_],
) -> npt.NDArray[np.float_]:
    """
    Adjust nvPM mass emissions index to account for the effects of sustainable aviation fuels.

    For fuel with hydrogen mass content > 14.3, the adjustment factor is adopted from
    Teoh et al. (2022), which was used to calculate the change in nvPM EIn.

    Parameters
    ----------
    hydrogen_content: float
        The percentage of hydrogen mass content in the fuel.
    thrust_setting: npt.NDArray[np.float_]
        Engine thrust setting, where the equivalent fuel mass flow rate per engine at
        sea level, :math:`[0 - 1]`.

    Returns
    -------
    npt.NDArray[np.float_]
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
    hydrogen_content: float | npt.NDArray[np.float_],
    thrust_setting: npt.NDArray[np.float_],
    a0: float,
    a1: float,
    a2: float,
) -> npt.NDArray[np.float_]:
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
