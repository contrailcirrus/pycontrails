"""Wave-vortex downwash functions.

This module includes both equations from the original CoCiP model
:cite:`schumannContrailCirrusPrediction2012` as well as improvements from
:cite:`unterstrasserPropertiesYoungContrails2016`.

Unterstrasser Notes
-------------------

Improved estimation of the survival fraction of the contrail ice crystal number ``f_surv``
during the  wake-vortex phase. This is a parameterised model that is developed based on
outputs provided by large eddy simulations.

For comparison, CoCiP assumes that ``f_surv`` is equal to the change in the contrail ice water
content (by mass) before and after the wake vortex phase. However, for larger (smaller) ice
particles, their survival fraction by number could be smaller (larger) than their survival fraction
by mass. This is particularly important in the "soot-poor" scenario, for example, in cleaner
lean-burn engines where their soot emissions can be 3-4 orders of magnitude lower than conventional
RQL engines.

"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from pycontrails.models.cocip import wind_shear
from pycontrails.physics import constants, thermo

# ---------------------------------------
# Original equations from Schumann (2012)
# ---------------------------------------


def max_downward_displacement(
    wingspan: npt.NDArray[np.float64] | float,
    true_airspeed: npt.NDArray[np.float64],
    aircraft_mass: npt.NDArray[np.float64] | float,
    air_temperature: npt.NDArray[np.float64],
    dT_dz: npt.NDArray[np.float64],
    ds_dz: npt.NDArray[np.float64],
    air_pressure: npt.NDArray[np.float64],
    effective_vertical_resolution: float,
    wind_shear_enhancement_exponent: float | npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Calculate the maximum contrail downward displacement after the wake vortex phase.

    Parameters
    ----------
    wingspan : npt.NDArray[np.float64] | float
        aircraft wingspan, [:math:`m`]
    true_airspeed : npt.NDArray[np.float64]
        true airspeed for each waypoint, [:math:`m s^{-1}`]
    aircraft_mass : npt.NDArray[np.float64] | float
        aircraft mass for each waypoint, [:math:`kg`]
    air_temperature : npt.NDArray[np.float64]
        ambient temperature for each waypoint, [:math:`K`]
    dT_dz : npt.NDArray[np.float64]
        potential temperature gradient, [:math:`K m^{-1}`]
    ds_dz : npt.NDArray[np.float64]
        Difference in wind speed over dz in the atmosphere, [:math:`m s^{-1} / m`]
    air_pressure : npt.NDArray[np.float64]
        pressure altitude at each waypoint, [:math:`Pa`]
    effective_vertical_resolution: float
        Passed through to :func:`wind_shear.wind_shear_enhancement_factor`, [:math:`m`]
    wind_shear_enhancement_exponent: npt.NDArray[np.float64] | float
        Passed through to :func:`wind_shear.wind_shear_enhancement_factor`

    Returns
    -------
    npt.NDArray[np.float64]
        Max contrail downward displacement after the wake vortex phase, [:math:`m`]

    References
    ----------
    - :cite:`holzapfelProbabilisticTwoPhaseWake2003`
    - :cite:`schumannContrailCirrusPrediction2012`
    """
    rho_air = thermo.rho_d(air_temperature, air_pressure)
    n_bv = thermo.brunt_vaisala_frequency(air_pressure, air_temperature, dT_dz)
    t_0 = effective_time_scale(wingspan, true_airspeed, aircraft_mass, rho_air)

    dz_max_strong = downward_displacement_strongly_stratified(
        wingspan, true_airspeed, aircraft_mass, rho_air, n_bv
    )

    is_weakly_stratified = n_bv * t_0 < 0.8
    if isinstance(wingspan, np.ndarray):
        wingspan = wingspan[is_weakly_stratified]
    if isinstance(aircraft_mass, np.ndarray):
        aircraft_mass = aircraft_mass[is_weakly_stratified]

    dz_max_weak = downward_displacement_weakly_stratified(
        wingspan=wingspan,
        true_airspeed=true_airspeed[is_weakly_stratified],
        aircraft_mass=aircraft_mass,
        rho_air=rho_air[is_weakly_stratified],
        n_bv=n_bv[is_weakly_stratified],
        dz_max_strong=dz_max_strong[is_weakly_stratified],
        ds_dz=ds_dz[is_weakly_stratified],
        t_0=t_0[is_weakly_stratified],
        effective_vertical_resolution=effective_vertical_resolution,
        wind_shear_enhancement_exponent=wind_shear_enhancement_exponent,
    )

    dz_max_strong[is_weakly_stratified] = dz_max_weak
    return dz_max_strong


def effective_time_scale(
    wingspan: npt.NDArray[np.float64] | float,
    true_airspeed: npt.NDArray[np.float64],
    aircraft_mass: npt.NDArray[np.float64] | float,
    rho_air: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    r"""
    Calculate the effective time scale of the wake vortex.

    Parameters
    ----------
    wingspan : npt.NDArray[np.float64]
        aircraft wingspan, [:math:`m`]
    true_airspeed : npt.NDArray[np.float64]
        true airspeed for each waypoint, [:math:`m \ s^{-1}`]
    aircraft_mass : npt.NDArray[np.float64]
        aircraft mass for each waypoint, [:math:`kg`]
    rho_air : npt.NDArray[np.float64]
        density of air for each waypoint, [:math:`kg \ m^{-3}`]

    Returns
    -------
    npt.NDArray[np.float64]
        Wake vortex effective time scale, [:math:`s`]

    Notes
    -----
    See section 2.5 (pg 547) of :cite:`schumannContrailCirrusPrediction2012`.

    References
    ----------
    - :cite:`schumannContrailCirrusPrediction2012`
    """
    c = np.pi**4 / 32
    return c * wingspan**3 * rho_air * true_airspeed / (aircraft_mass * constants.g)


def downward_displacement_strongly_stratified(
    wingspan: npt.NDArray[np.float64] | float,
    true_airspeed: npt.NDArray[np.float64],
    aircraft_mass: npt.NDArray[np.float64] | float,
    rho_air: npt.NDArray[np.float64],
    n_bv: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Calculate the maximum contrail downward displacement under strongly stratified conditions.

    Parameters
    ----------
    wingspan : npt.NDArray[np.float64] | float
        aircraft wingspan, [:math:`m`]
    true_airspeed : npt.NDArray[np.float64]
        true airspeed for each waypoint, [:math:`m s^{-1}`]
    aircraft_mass : npt.NDArray[np.float64] | float
        aircraft mass for each waypoint, [:math:`kg`]
    rho_air : npt.NDArray[np.float64]
        density of air for each waypoint, [:math:`kg m^{-3}`]
    n_bv : npt.NDArray[np.float64]
        Brunt-Vaisaila frequency, [:math:`s^{-1}`]

    Returns
    -------
    npt.NDArray[np.float64]
        Maximum contrail downward displacement, strongly stratified conditions, [:math:`m`]

    Notes
    -----
    See section 2.5 (pg 547 - 548) of :cite:`schumannContrailCirrusPrediction2012`.

    References
    ----------
    - :cite:`schumannContrailCirrusPrediction2012`
    """
    c = (1.49 * 16) / (2 * np.pi**3)  # This is W2 in Schumann's Fortran code
    return (c * aircraft_mass * constants.g) / (wingspan**2 * rho_air * true_airspeed * n_bv)


def downward_displacement_weakly_stratified(
    wingspan: npt.NDArray[np.float64] | float,
    true_airspeed: npt.NDArray[np.float64],
    aircraft_mass: npt.NDArray[np.float64] | float,
    rho_air: npt.NDArray[np.float64],
    n_bv: npt.NDArray[np.float64],
    dz_max_strong: npt.NDArray[np.float64],
    ds_dz: npt.NDArray[np.float64],
    t_0: npt.NDArray[np.float64],
    effective_vertical_resolution: float,
    wind_shear_enhancement_exponent: npt.NDArray[np.float64] | float,
) -> npt.NDArray[np.float64]:
    """
    Calculate the maximum contrail downward displacement under weakly/stably stratified conditions.

    Parameters
    ----------
    wingspan : npt.NDArray[np.float64] | float
        aircraft wingspan, [:math:`m`]
    true_airspeed : npt.NDArray[np.float64]
        true airspeed for each waypoint, [:math:`m s^{-1}`]
    aircraft_mass : npt.NDArray[np.float64] | float
        aircraft mass for each waypoint, [:math:`kg`]
    rho_air : npt.NDArray[np.float64]
        density of air for each waypoint, [:math:`kg m^{-3}`]
    n_bv : npt.NDArray[np.float64]
        Brunt-Vaisaila frequency, [:math:`s^{-1}`]
    dz_max_strong : npt.NDArray[np.float64]
        Max contrail downward displacement under strongly stratified conditions, [:math:`m`]
    ds_dz : npt.NDArray[np.float64]
        Difference in wind speed over dz in the atmosphere, [:math:`m s^{-1} / m`]
    t_0 : npt.NDArray[np.float64]
        Wake vortex effective time scale, [:math:`s`]
    effective_vertical_resolution: float
        Passed through to :func:`wind_shear.wind_shear_enhancement_factor`, [:math:`m`]
    wind_shear_enhancement_exponent: npt.NDArray[np.float64] | float
        Passed through to :func:`wind_shear.wind_shear_enhancement_factor`

    Returns
    -------
    npt.NDArray[np.float64]
        Maximum contrail downward displacement, weakly/stably stratified conditions, [:math:`m`]

    Notes
    -----
    See section 2.5 (pg 548) of :cite:`schumannContrailCirrusPrediction2012`.

    References
    ----------
    - :cite:`schumannContrailCirrusPrediction2012`
    """
    b_0 = wake_vortex_separation(wingspan)
    dz_max = np.maximum(dz_max_strong, 10.0)
    shear_enhancement_factor = wind_shear.wind_shear_enhancement_factor(
        dz_max, effective_vertical_resolution, wind_shear_enhancement_exponent
    )

    # Calculate epsilon and epsilon star
    # In Schumann's Fortran code, epsn = EDR and epsn_st = EPSN
    epsn = turbulent_kinetic_energy_dissipation_rate(ds_dz, shear_enhancement_factor)
    epsn_st = normalized_dissipation_rate(epsn, wingspan, true_airspeed, aircraft_mass, rho_air)
    return b_0 * (7.68 * (1 - 4.07 * epsn_st + 5.67 * epsn_st**2) * (0.79 - n_bv * t_0) + 1.88)


def wake_vortex_separation(
    wingspan: npt.NDArray[np.float64] | float,
) -> npt.NDArray[np.float64] | float:
    """
    Calculate the wake vortex separation.

    Parameters
    ----------
    wingspan : npt.NDArray[np.float64] | float
        aircraft wingspan, [:math:`m`]

    Returns
    -------
    npt.NDArray[np.float64]
        wake vortex separation, [:math:`m`]
    """
    return (np.pi * wingspan) / 4.0


def turbulent_kinetic_energy_dissipation_rate(
    ds_dz: npt.NDArray[np.float64],
    shear_enhancement_factor: npt.NDArray[np.float64] | float = 1.0,
) -> npt.NDArray[np.float64]:
    """
    Calculate the turbulent kinetic energy dissipation rate (epsilon).

    The shear enhancement factor is used to account for any sub-grid scale turbulence.

    Parameters
    ----------
    ds_dz : npt.NDArray[np.float64]
        Difference in wind speed over dz in the atmosphere, [:math:`m s^{-1} / m`]
    shear_enhancement_factor : npt.NDArray[np.float64] | float
        Multiplication factor to enhance the wind shear

    Returns
    -------
    npt.NDArray[np.float64]
        turbulent kinetic energy dissipation rate, [:math:`m^{2} s^{-3}`]

    Notes
    -----
    - See eq. (37) in :cite:`schumannContrailCirrusPrediction2012`.
    - In a personal correspondence, Dr. Schumann identified a print error in Eq. (37)
      of the 2012 paper where the shear term should not be squared.
      The correct equation is listed in Eq. (13) :cite:`schumannTurbulentMixingStably1995`.

    References
    ----------
    - :cite:`schumannContrailCirrusPrediction2012`
    - :cite:`schumannTurbulentMixingStably1995`
    """
    return 0.5 * 0.1**2 * (ds_dz * shear_enhancement_factor**2)


def normalized_dissipation_rate(
    epsilon: npt.NDArray[np.float64],
    wingspan: npt.NDArray[np.float64] | float,
    true_airspeed: npt.NDArray[np.float64],
    aircraft_mass: npt.NDArray[np.float64] | float,
    rho_air: npt.NDArray[np.float64] | float,
) -> npt.NDArray[np.float64]:
    """
    Calculate the normalized dissipation rate of the sinking wake vortex.

    Parameters
    ----------
    epsilon: npt.NDArray[np.float64]
        turbulent kinetic energy dissipation rate, [:math:`m^{2} s^{-3}`]
    wingspan : npt.NDArray[np.float64] | float
        aircraft wingspan, [:math:`m`]
    true_airspeed : npt.NDArray[np.float64]
        true airspeed for each waypoint, [:math:`m s^{-1}`]
    aircraft_mass : npt.NDArray[np.float64]
        aircraft mass for each waypoint, [:math:`kg`]
    rho_air : npt.NDArray[np.float64]
        density of air for each waypoint, [:math:`kg m^{-3}`]

    Returns
    -------
    npt.NDArray[np.float64]
        Normalized dissipation rate of the sinking wake vortex

    Notes
    -----
    See page 548 of :cite:`schumannContrailCirrusPrediction2012`.

    References
    ----------
    - :cite:`schumannContrailCirrusPrediction2012`
    """
    c = (np.pi / 4) ** (1 / 3) * np.pi**3 / 8  # This is W6 in Schumann's Fortran code
    numer = c * (epsilon * wingspan) ** (1 / 3) * wingspan**2 * rho_air * true_airspeed

    # epsn_st = epsilon star
    epsn_st = numer / (constants.g * aircraft_mass)

    # In a personal correspondence, Schumann gives the precise value
    # of 0.358906526 here
    # In the 2012 paper, Schumann gives 0.36
    # The precise value is likely insignificant because we don't expect epsn_st
    # to be larger than 0.36
    return np.minimum(epsn_st, 0.36)


def initial_contrail_width(
    wingspan: npt.NDArray[np.float64] | float, dz_max: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """
    Calculate the initial contrail width.

    Parameters
    ----------
    wingspan : npt.NDArray[np.float64] | float
        aircraft wingspan, [:math:`m`]
    dz_max : npt.NDArray[np.float64]
        Max contrail downward displacement after the wake vortex phase, [:math:`m`]
        Only the size of this array is used; the values are ignored.

    Returns
    -------
    npt.NDArray[np.float64]
        Initial contrail width, [:math:`m`]
    """
    return np.full_like(dz_max, np.pi / 4) * wingspan


def initial_contrail_depth(
    dz_max: npt.NDArray[np.float64], initial_wake_vortex_depth: float | npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """
    Calculate the initial contrail depth.

    Parameters
    ----------
    dz_max : npt.NDArray[np.float64]
        Max contrail downward displacement after the wake vortex phase, [:math:`m`]
    initial_wake_vortex_depth : float | npt.NDArray[np.float64]
        Initial wake vortex depth scaling factor.
        Denoted `C_D0` in eq (14) in :cite:`schumannContrailCirrusPrediction2012`.

    Returns
    -------
    npt.NDArray[np.float64]
        Initial contrail depth, [:math:`m`]
    """
    return dz_max * initial_wake_vortex_depth


# --------------------
# Unterstrasser (2016)
# --------------------


def ice_particle_number_survival_fraction(
    air_temperature: npt.NDArray[np.float64],
    rhi_0: npt.NDArray[np.float64],
    ei_h2o: npt.NDArray[np.float64] | float,
    wingspan: npt.NDArray[np.float64] | float,
    true_airspeed: npt.NDArray[np.float64],
    fuel_flow: npt.NDArray[np.float64],
    aei_n: npt.NDArray[np.float64],
    z_desc: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    r"""
    Calculate fraction of ice particle number surviving the wake vortex phase and required inputs.

    This implementation is based on the work of :cite:`unterstrasserPropertiesYoungContrails2016`
    and is an improved estimation compared with
    :func:`contrail_properties.ice_particle_survival_fraction`.

    Parameters
    ----------
    air_temperature : npt.NDArray[np.float64]
        ambient temperature for each waypoint, [:math:`K`]
    rhi_0: npt.NDArray[np.float64]
        Relative humidity with respect to ice at the flight waypoint
    ei_h2o : npt.NDArray[np.float64] | float
        Emission index of water vapor, [:math:`kg \ kg^{-1}`]
    wingspan : npt.NDArray[np.float64] | float
        aircraft wingspan, [:math:`m`]
    true_airspeed : npt.NDArray[np.float64]
        true airspeed for each waypoint, [:math:`m s^{-1}`]
    fuel_flow : npt.NDArray[np.float64]
        Fuel mass flow rate, [:math:`kg s^{-1}`]
    aei_n : npt.NDArray[np.float64]
        Apparent ice crystal number emissions index at contrail formation, [:math:`kg^{-1}`]
    z_desc : npt.NDArray[np.float64]
        Final vertical displacement of the wake vortex, ``dz_max`` in :mod:`wake_vortex.py`,
        [:math:`m`].

    Returns
    -------
    npt.NDArray[np.float64]
        Fraction of contrail ice particle number that survive the wake vortex phase.

    Notes
    -----
    - See eq. (3), (9), and (10) in :cite:`unterstrasserPropertiesYoungContrails2016`.
    - For consistency in CoCiP, ``z_desc`` should be calculated using :func:`dz_max` instead of
      using :func:`z_desc_length_scale`.
    """
    # Length scales
    z_atm = z_atm_length_scale(air_temperature, rhi_0)
    rho_emit = emitted_water_vapour_concentration(ei_h2o, wingspan, true_airspeed, fuel_flow)
    z_emit = z_emit_length_scale(rho_emit, air_temperature)
    z_total = z_total_length_scale(aei_n, z_atm, z_emit, z_desc)
    return _ice_number_survival_fraction(z_total)


def z_total_length_scale(
    aei_n: npt.NDArray[np.float64],
    z_atm: npt.NDArray[np.float64],
    z_emit: npt.NDArray[np.float64],
    z_desc: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Calculate the total length-scale effect of the wake vortex downwash.

    Parameters
    ----------
    aei_n : npt.NDArray[np.float64]
        Apparent ice crystal number emissions index at contrail formation, [:math:`kg^{-1}`]
    z_atm : npt.NDArray[np.float64]
        Length-scale effect of ambient supersaturation on the ice crystal mass budget, [:math:`m`]
    z_emit : npt.NDArray[np.float64]
        Length-scale effect of water vapour emissions on the ice crystal mass budget, [:math:`m`]
    z_desc : npt.NDArray[np.float64]
        Final vertical displacement of the wake vortex, `dz_max` in `wake_vortex.py`, [:math:`m`]

    Returns
    -------
    npt.NDArray[np.float64]
        Total length-scale effect of the wake vortex downwash, [:math:`m`]
    """
    alpha_base = (aei_n / 2.8e14) ** (-0.18)
    alpha_atm = 1.7 * alpha_base
    alpha_emit = 1.15 * alpha_base

    z_total = alpha_atm * z_atm + alpha_emit * z_emit - 0.6 * z_desc

    z_total.clip(min=0.0, out=z_total)
    return z_total


def z_atm_length_scale(
    air_temperature: npt.NDArray[np.float64],
    rhi_0: npt.NDArray[np.float64],
    *,
    n_iter: int = 10,
) -> npt.NDArray[np.float64]:
    """Calculate the length-scale effect of ambient supersaturation on the ice crystal mass budget.

    Parameters
    ----------
    air_temperature : npt.NDArray[np.float64]
        Ambient temperature for each waypoint, [:math:`K`].
    rhi_0 : npt.NDArray[np.float64]
        Relative humidity with respect to ice at the flight waypoint.
    n_iter : int
        Number of iterations, set to 10 as default where ``z_atm`` is accurate to within +-1 m.

    Returns
    -------
    npt.NDArray[np.float64]
        The effect of the ambient supersaturation on the ice crystal mass budget,
        provided as a length scale equivalent, [:math:`m`].

    Notes
    -----
    - See eq. (5) in :cite:`unterstrasserPropertiesYoungContrails2016`.
    """
    # Only perform operation when the ambient condition is supersaturated w.r.t. ice
    issr = rhi_0 > 1.0

    rhi_issr = rhi_0[issr]
    air_temperature_issr = air_temperature[issr]

    # Solve non-linear equation numerically using the bisection method
    # Did not use scipy functions because it is unstable when dealing with np.arrays
    z_1 = np.zeros_like(rhi_issr)
    z_2 = np.full_like(rhi_issr, 1000.0)
    lhs = rhi_issr * thermo.e_sat_ice(air_temperature_issr) / air_temperature_issr

    for _ in range(n_iter):
        z_est = 0.5 * (z_1 + z_2)
        rhs = (thermo.e_sat_ice(air_temperature_issr + 9.8e-3 * z_est)) / (
            air_temperature_issr + 9.8e-3 * z_est
        )
        z_1[lhs > rhs] = z_est[lhs > rhs]
        z_2[lhs < rhs] = z_est[lhs < rhs]

    out = np.zeros_like(rhi_0)
    out[issr] = 0.5 * (z_1 + z_2)
    return out


def emitted_water_vapour_concentration(
    ei_h2o: npt.NDArray[np.float64] | float,
    wingspan: npt.NDArray[np.float64] | float,
    true_airspeed: npt.NDArray[np.float64],
    fuel_flow: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    r"""
    Calculate aircraft-emitted water vapour concentration in the plume.

    Parameters
    ----------
    ei_h2o : npt.NDArray[np.float64] | float
        Emission index of water vapor, [:math:`kg \ kg^{-1}`]
    wingspan : npt.NDArray[np.float64] | float
        aircraft wingspan, [:math:`m`]
    true_airspeed : npt.NDArray[np.float64]
        true airspeed for each waypoint, [:math:`m s^{-1}`]
    fuel_flow : npt.NDArray[np.float64]
        Fuel mass flow rate, [:math:`kg s^{-1}`]

    Returns
    -------
    npt.NDArray[np.float64]
        Aircraft-emitted water vapour concentration in the plume, [:math:`kg m^{-3}`]

    Notes
    -----
    - See eq. (6) and (A8) in :cite:`unterstrasserPropertiesYoungContrails2016`.
    """
    h2o_per_dist = (ei_h2o * fuel_flow) / true_airspeed
    area_p = plume_area(wingspan)
    return h2o_per_dist / area_p


def z_emit_length_scale(
    rho_emit: npt.NDArray[np.float64], air_temperature: npt.NDArray[np.float64], *, n_iter: int = 10
) -> npt.NDArray[np.float64]:
    """Calculate the length-scale effect of water vapour emissions on the ice crystal mass budget.

    Parameters
    ----------
    rho_emit : npt.NDArray[np.float64] | float
        Aircraft-emitted water vapour concentration in the plume, [:math:`kg m^{-3}`]
    air_temperature : npt.NDArray[np.float64]
        ambient temperature for each waypoint, [:math:`K`]
    n_iter : int
        Number of iterations, set to 10 as default where ``z_emit`` is accurate to within +-1 m.

    Returns
    -------
    npt.NDArray[np.float64]
        The effect of the aircraft water vapour emission on the ice crystal mass budget,
        provided as a length scale equivalent, [:math:`m`]

    Notes
    -----
    - See eq. (7) in :cite:`unterstrasserPropertiesYoungContrails2016`.
    """
    # Solve non-linear equation numerically using the bisection method
    # Did not use scipy functions because it is unstable when dealing with np.arrays
    z_1 = np.zeros_like(rho_emit)
    z_2 = np.full_like(rho_emit, 1000.0)

    lhs = (thermo.e_sat_ice(air_temperature) / (constants.R_v * air_temperature)) + rho_emit

    for _ in range(n_iter):
        z_est = 0.5 * (z_1 + z_2)
        rhs = thermo.e_sat_ice(air_temperature + 9.8e-3 * z_est) / (
            constants.R_v * (air_temperature + 9.8e-3 * z_est)
        )
        z_1[lhs > rhs] = z_est[lhs > rhs]
        z_2[lhs < rhs] = z_est[lhs < rhs]

    return 0.5 * (z_1 + z_2)


def plume_area(wingspan: npt.NDArray[np.float64] | float) -> npt.NDArray[np.float64] | float:
    """Calculate area of the wake-vortex plume.

    Parameters
    ----------
    wingspan : npt.NDArray[np.float64] | float
        aircraft wingspan, [:math:`m`]

    Returns
    -------
    float
        Area of two wake-vortex plumes, [:math:`m^{2}`]

    Notes
    -----
    - See eq. (A6) and (A7) in :cite:`unterstrasserPropertiesYoungContrails2016`.
    """
    r_plume = 1.5 + 0.314 * wingspan
    return 2.0 * 2.0 * np.pi * r_plume**2


def z_desc_length_scale(
    wingspan: npt.NDArray[np.float64] | float,
    air_temperature: npt.NDArray[np.float64],
    air_pressure: npt.NDArray[np.float64],
    true_airspeed: npt.NDArray[np.float64],
    aircraft_mass: npt.NDArray[np.float64],
    dT_dz: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Calculate the final vertical displacement of the wake vortex.

    Parameters
    ----------
    wingspan : npt.NDArray[np.float64] | float
        aircraft wingspan, [:math:`m`]
    air_temperature : npt.NDArray[np.float64]
        ambient temperature for each waypoint, [:math:`K`]
    air_pressure : npt.NDArray[np.float64]
        pressure altitude at each waypoint, [:math:`Pa`]
    true_airspeed : npt.NDArray[np.float64]
        true airspeed for each waypoint, [:math:`m s^{-1}`]
    aircraft_mass : npt.NDArray[np.float64] | float
        aircraft mass for each waypoint, [:math:`kg`]
    dT_dz : npt.NDArray[np.float64]
        potential temperature gradient, [:math:`K m^{-1}`]

    Returns
    -------
    npt.NDArray[np.float64]
        Final vertical displacement of the wake vortex, [:math:`m`]

    Notes
    -----
    - See eq. (4) in :cite:`unterstrasserPropertiesYoungContrails2016`.
    """
    gamma_0 = _initial_wake_vortex_circulation(
        wingspan, air_temperature, air_pressure, true_airspeed, aircraft_mass
    )
    n_bv = thermo.brunt_vaisala_frequency(air_pressure, air_temperature, dT_dz)
    return ((8.0 * gamma_0) / (np.pi * n_bv)) ** 0.5


def _initial_wake_vortex_circulation(
    wingspan: npt.NDArray[np.float64] | float,
    air_temperature: npt.NDArray[np.float64],
    air_pressure: npt.NDArray[np.float64],
    true_airspeed: npt.NDArray[np.float64],
    aircraft_mass: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Calculate initial wake vortex circulation.

    Parameters
    ----------
    wingspan : npt.NDArray[np.float64] | float
        aircraft wingspan, [:math:`m`]
    air_temperature : npt.NDArray[np.float64]
        ambient temperature for each waypoint, [:math:`K`]
    air_pressure : npt.NDArray[np.float64]
        pressure altitude at each waypoint, [:math:`Pa`]
    true_airspeed : npt.NDArray[np.float64]
        true airspeed for each waypoint, [:math:`m s^{-1}`]
    aircraft_mass : npt.NDArray[np.float64] | float
        aircraft mass for each waypoint, [:math:`kg`]

    Returns
    -------
    npt.NDArray[np.float64]
        Initial wake vortex circulation, [:math:`m^{2} s^{-1}`]

    Notes
    -----
    - This is a measure of the strength/intensity of the wake vortex circulation.
    - See eq. (A1) in :cite:`unterstrasserPropertiesYoungContrails2016`.
    """
    b_0 = wake_vortex_separation(wingspan)
    rho_air = thermo.rho_d(air_temperature, air_pressure)
    return (constants.g * aircraft_mass) / (rho_air * b_0 * true_airspeed)


def _ice_number_survival_fraction(z_total: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Calculate fraction of ice particle number surviving the wake vortex phase.

    Parameters
    ----------
    z_total : npt.NDArray[np.float64]
        Total length-scale effect of the wake vortex downwash, [:math:`m`]

    Returns
    -------
    npt.NDArray[np.float64]
        Fraction of ice particle number surviving the wake vortex phase
    """
    f_surv = 0.45 + (1.19 / np.pi) * np.arctan(-1.35 + (z_total / 100.0))
    np.clip(f_surv, 0.0, 1.0, out=f_surv)
    return f_surv


def initial_contrail_depth_u2016(
    z_desc: npt.NDArray[np.float64],
    f_surv: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Calculate initial contrail depth using :cite:`unterstrasserPropertiesYoungContrails2016`.

    Parameters
    ----------
    z_desc : npt.NDArray[np.float64]
        Final vertical displacement of the wake vortex, ``dz_max`` in :mod:`wake_vortex.py`,
        [:math:`m`].
    f_surv : npt.NDArray[np.float64]
        Fraction of contrail ice particle number that survive the wake vortex phase.
        See :func:`ice_particle_survival_fraction`.

    Returns
    -------
    npt.NDArray[np.float64]
        Initial contrail depth, [:math:`m`]

    Notes
    -----
    - See eq. (12), and (13) in :cite:`unterstrasserPropertiesYoungContrails2016`.
    - For consistency in CoCiP, `z_desc` should be calculated using :func:`dz_max` instead of
      using :func:`z_desc_length_scale`.
    """
    return z_desc * np.where(
        f_surv <= 0.2,
        6.0 * f_surv,
        0.15 * f_surv + (6.0 - 0.15) * 0.2,
    )
