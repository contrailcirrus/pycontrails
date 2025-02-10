"""Wave-vortex downwash functions from Lottermoser & Unterstrasser (2025).

Notes
-----
:cite:`unterstrasserPropertiesYoungContrails2016` provides a parameterized model of the
survival fraction of the contrail ice crystal number ``f_surv`` during the  wake-vortex phase.
The model has since been updated in :cite:`lottermoserHighResolutionEarlyContrails2025`. This update
improves the goodness-of-fit between the parameterised model and LES, and expands the parameter
space and can now be used for very low and very high soot inputs, different fuel types (where the
EI H2Os are different), and higher ambient temperatures (up to 235 K) to accomodate for contrails
formed by liquid hydrogen aircraft. The model was developed based on output from large eddy
simulations, and improves agreement with LES outputs relative to the default survival fraction
parameterization used in CoCiP.

For comparison, CoCiP assumes that ``f_surv`` is equal to the change in the contrail ice water
content (by mass) before and after the wake vortex phase. However, for larger (smaller) ice
particles, their survival fraction by number could be smaller (larger) than their survival fraction
by mass. This is particularly important in the "soot-poor" scenario, for example, in cleaner
lean-burn engines where their soot emissions can be 3-4 orders of magnitude lower than conventional
RQL engines.

ADD CITATION TO BIBTEX: :cite:`lottermoserHighResolutionEarlyContrails2025`
Lottermoser, A. and Unterstraßer, S.: High-resolution modelling of early contrail evolution from
hydrogen-powered aircraft, EGUsphere [preprint], https://doi.org/10.5194/egusphere-2024-3859, 2025.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from pycontrails.models.cocip.wake_vortex import wake_vortex_separation
from pycontrails.physics import constants, thermo


def ice_particle_number_survival_fraction(
    air_temperature: npt.NDArray[np.floating],
    rhi_0: npt.NDArray[np.floating],
    ei_h2o: npt.NDArray[np.floating] | float,
    wingspan: npt.NDArray[np.floating] | float,
    true_airspeed: npt.NDArray[np.floating],
    fuel_flow: npt.NDArray[np.floating],
    aei_n: npt.NDArray[np.floating],
    z_desc: npt.NDArray[np.floating],
    *,
    analytical_solution: bool = True,
) -> npt.NDArray[np.floating]:
    r"""
    Calculate fraction of ice particle number surviving the wake vortex phase and required inputs.

    This implementation is based on the work of :cite:`unterstrasserPropertiesYoungContrails2016`
    and is an improved estimation compared with
    :func:`contrail_properties.ice_particle_survival_fraction`.

    Parameters
    ----------
    air_temperature : npt.NDArray[np.floating]
        ambient temperature for each waypoint, [:math:`K`]
    rhi_0: npt.NDArray[np.floating]
        Relative humidity with respect to ice at the flight waypoint
    ei_h2o : npt.NDArray[np.floating] | float
        Emission index of water vapor, [:math:`kg \ kg^{-1}`]
    wingspan : npt.NDArray[np.floating] | float
        aircraft wingspan, [:math:`m`]
    true_airspeed : npt.NDArray[np.floating]
        true airspeed for each waypoint, [:math:`m s^{-1}`]
    fuel_flow : npt.NDArray[np.floating]
        Fuel mass flow rate, [:math:`kg s^{-1}`]
    aei_n : npt.NDArray[np.floating]
        Apparent ice crystal number emissions index at contrail formation, [:math:`kg^{-1}`]
    z_desc : npt.NDArray[np.floating]
        Final vertical displacement of the wake vortex, ``dz_max`` in :mod:`wake_vortex.py`,
        [:math:`m`].
    analytical_solution : bool
        Use analytical solution to calculate ``z_atm`` and ``z_emit`` instead of numerical solution.

    Returns
    -------
    npt.NDArray[np.floating]
        Fraction of contrail ice particle number that survive the wake vortex phase.

    References
    ----------
    - :cite:`unterstrasserPropertiesYoungContrails2016`
    - :cite:`lottermoserHighResolutionEarlyContrails2025`

    Notes
    -----
    - For consistency in CoCiP, ``z_desc`` should be calculated using :func:`dz_max` instead of
      using :func:`z_desc_length_scale`.
    """
    rho_emit = emitted_water_vapour_concentration(ei_h2o, wingspan, true_airspeed, fuel_flow)

    # Length scales
    if analytical_solution:
        z_atm = z_atm_length_scale_analytical(air_temperature, rhi_0)
        z_emit = z_emit_length_scale_analytical(rho_emit, air_temperature)

    else:
        z_atm = z_atm_length_scale_numerical(air_temperature, rhi_0)
        z_emit = z_emit_length_scale_numerical(rho_emit, air_temperature)

    z_total = z_total_length_scale(z_atm, z_emit, z_desc, true_airspeed, fuel_flow, aei_n, wingspan)
    return _survival_fraction_from_length_scale(z_total)


def z_total_length_scale(
    z_atm: npt.NDArray[np.floating],
    z_emit: npt.NDArray[np.floating],
    z_desc: npt.NDArray[np.floating],
    true_airspeed: npt.NDArray[np.floating],
    fuel_flow: npt.NDArray[np.floating],
    aei_n: npt.NDArray[np.floating],
    wingspan: npt.NDArray[np.floating] | float,
) -> npt.NDArray[np.floating]:
    """
    Calculate the total length-scale effect of the wake vortex downwash.

    Parameters
    ----------
    z_atm : npt.NDArray[np.floating]
        Length-scale effect of ambient supersaturation on the ice crystal mass budget, [:math:`m`]
    z_emit : npt.NDArray[np.floating]
        Length-scale effect of water vapour emissions on the ice crystal mass budget, [:math:`m`]
    z_desc : npt.NDArray[np.floating]
        Final vertical displacement of the wake vortex, `dz_max` in `wake_vortex.py`, [:math:`m`]
    true_airspeed : npt.NDArray[np.floating]
        true airspeed for each waypoint, [:math:`m s^{-1}`]
    fuel_flow : npt.NDArray[np.floating]
        Fuel mass flow rate, [:math:`kg s^{-1}`]
    aei_n : npt.NDArray[np.floating]
        Apparent ice crystal number emissions index at contrail formation, [:math:`kg^{-1}`]
    wingspan : npt.NDArray[np.floating] | float
        aircraft wingspan, [:math:`m`]

    Returns
    -------
    npt.NDArray[np.floating]
        Total length-scale effect of the wake vortex downwash, [:math:`m`]

    Notes
    -----
    - For `psi`, see Appendix A1 in :cite:`lottermoserHighResolutionEarlyContrails2025`.
    - For `z_total`, see Eq. (9) and (10) in :cite:`lottermoserHighResolutionEarlyContrails2025`.
    """
    # Calculate psi term
    fuel_dist = fuel_flow / true_airspeed  # Units: [:math:`kg m^{-1}`]
    n_ice_dist = fuel_dist * aei_n  # Units: [:math:`m^{-1}`]

    n_ice_per_vol = n_ice_dist / plume_area(wingspan)  # Units: [:math:`m^{-3}`]
    n_ice_per_vol_ref = 3.38e12 / plume_area(60.3)

    psi = (n_ice_per_vol_ref / n_ice_per_vol) ** 0.16

    # Calculate total length-scale effect
    return psi * (1.27 * z_atm + 0.42 * z_emit) - 0.49 * z_desc


def z_atm_length_scale_analytical(
    air_temperature: npt.NDArray[np.floating],
    rhi_0: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    """Calculate the length-scale effect of ambient supersaturation on the ice crystal mass budget.

    Parameters
    ----------
    air_temperature : npt.NDArray[np.floating]
        Ambient temperature for each waypoint, [:math:`K`].
    rhi_0 : npt.NDArray[np.floating]
        Relative humidity with respect to ice at the flight waypoint.

    Returns
    -------
    npt.NDArray[np.floating]
        The effect of the ambient supersaturation on the ice crystal mass budget,
        provided as a length scale equivalent, estimated with analytical fit [:math:`m`].

    Notes
    -----
    - See Eq. (A2) in :cite:`lottermoserHighResolutionEarlyContrails2025`.
    """
    z_atm = np.zeros_like(rhi_0)

    # Only perform operation when the ambient condition is supersaturated w.r.t. ice
    issr = rhi_0 > 1.0

    s_i = rhi_0 - 1.0
    z_atm[issr] = 607.46 * s_i[issr] ** 0.897 * (air_temperature[issr] / 205.0) ** 2.225
    return z_atm


def z_atm_length_scale_numerical(
    air_temperature: npt.NDArray[np.floating],
    rhi_0: npt.NDArray[np.floating],
    *,
    n_iter: int = 10,
) -> npt.NDArray[np.floating]:
    """Calculate the length-scale effect of ambient supersaturation on the ice crystal mass budget.

    Parameters
    ----------
    air_temperature : npt.NDArray[np.floating]
        Ambient temperature for each waypoint, [:math:`K`].
    rhi_0 : npt.NDArray[np.floating]
        Relative humidity with respect to ice at the flight waypoint.
    n_iter : int
        Number of iterations, set to 10 as default where ``z_atm`` is accurate to within +-1 m.

    Returns
    -------
    npt.NDArray[np.floating]
        The effect of the ambient supersaturation on the ice crystal mass budget,
        provided as a length scale equivalent, estimated with numerical methods [:math:`m`].

    Notes
    -----
    - See Eq. (6) in :cite:`lottermoserHighResolutionEarlyContrails2025`.
    """
    # Only perform operation when the ambient condition is supersaturated w.r.t. ice
    issr = rhi_0 > 1.0

    rhi_issr = rhi_0[issr]
    air_temperature_issr = air_temperature[issr]

    # Solve non-linear equation numerically using the bisection method
    # Did not use scipy functions because it is unstable when dealing with np.arrays
    z_1 = np.zeros_like(rhi_issr)
    z_2 = np.full_like(rhi_issr, 1000.0)
    lhs = rhi_issr * thermo.e_sat_ice(air_temperature_issr) / (air_temperature_issr**3.5)

    dry_adiabatic_lapse_rate = constants.g / constants.c_pd
    for _ in range(n_iter):
        z_est = 0.5 * (z_1 + z_2)
        rhs = (thermo.e_sat_ice(air_temperature_issr + dry_adiabatic_lapse_rate * z_est)) / (
            air_temperature_issr + dry_adiabatic_lapse_rate * z_est
        ) ** 3.5
        z_1[lhs > rhs] = z_est[lhs > rhs]
        z_2[lhs < rhs] = z_est[lhs < rhs]

    out = np.zeros_like(rhi_0)
    out[issr] = 0.5 * (z_1 + z_2)
    return out


def emitted_water_vapour_concentration(
    ei_h2o: npt.NDArray[np.floating] | float,
    wingspan: npt.NDArray[np.floating] | float,
    true_airspeed: npt.NDArray[np.floating],
    fuel_flow: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    r"""
    Calculate aircraft-emitted water vapour concentration in the plume.

    Parameters
    ----------
    ei_h2o : npt.NDArray[np.floating] | float
        Emission index of water vapor, [:math:`kg \ kg^{-1}`]
    wingspan : npt.NDArray[np.floating] | float
        aircraft wingspan, [:math:`m`]
    true_airspeed : npt.NDArray[np.floating]
        true airspeed for each waypoint, [:math:`m s^{-1}`]
    fuel_flow : npt.NDArray[np.floating]
        Fuel mass flow rate, [:math:`kg s^{-1}`]

    Returns
    -------
    npt.NDArray[np.floating]
        Aircraft-emitted water vapour concentration in the plume, [:math:`kg m^{-3}`]

    Notes
    -----
    - See eq. (6) and (A8) in :cite:`unterstrasserPropertiesYoungContrails2016`.
    """
    h2o_per_dist = (ei_h2o * fuel_flow) / true_airspeed
    area_p = plume_area(wingspan)
    return h2o_per_dist / area_p


def z_emit_length_scale_analytical(
    rho_emit: npt.NDArray[np.floating],
    air_temperature: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    """Calculate the length-scale effect of water vapour emissions on the ice crystal mass budget.

    Parameters
    ----------
    rho_emit : npt.NDArray[np.floating] | float
        Aircraft-emitted water vapour concentration in the plume, [:math:`kg m^{-3}`]
    air_temperature : npt.NDArray[np.floating]
        ambient temperature for each waypoint, [:math:`K`]

    Returns
    -------
    npt.NDArray[np.floating]
        The effect of the aircraft water vapour emission on the ice crystal mass budget,
        provided as a length scale equivalent, estimated with analytical fit [:math:`m`]

    Notes
    -----
    - See Eq. (A3) in :cite:`lottermoserHighResolutionEarlyContrails2025`.
    """
    t_205 = air_temperature - 205.0
    return (
        1106.6
        * ((rho_emit * 1e5) ** (0.678 + 0.0116 * t_205))
        * np.exp(-(0.0807 + 0.000428 * t_205) * t_205)
    )


def z_emit_length_scale_numerical(
    rho_emit: npt.NDArray[np.floating],
    air_temperature: npt.NDArray[np.floating],
    *,
    n_iter: int = 10,
) -> npt.NDArray[np.floating]:
    """Calculate the length-scale effect of water vapour emissions on the ice crystal mass budget.

    Parameters
    ----------
    rho_emit : npt.NDArray[np.floating] | float
        Aircraft-emitted water vapour concentration in the plume, [:math:`kg m^{-3}`]
    air_temperature : npt.NDArray[np.floating]
        ambient temperature for each waypoint, [:math:`K`]
    n_iter : int
        Number of iterations, set to 10 as default where ``z_emit`` is accurate to within +-1 m.

    Returns
    -------
    npt.NDArray[np.floating]
        The effect of the aircraft water vapour emission on the ice crystal mass budget,
        provided as a length scale equivalent, estimated with numerical methods [:math:`m`]

    Notes
    -----
    - See Eq. (7) in :cite:`lottermoserHighResolutionEarlyContrails2025`.
    """
    # Solve non-linear equation numerically using the bisection method
    # Did not use scipy functions because it is unstable when dealing with np.arrays
    z_1 = np.zeros_like(rho_emit)
    z_2 = np.full_like(rho_emit, 1000.0)

    lhs = (thermo.e_sat_ice(air_temperature) / (constants.R_v * air_temperature**3.5)) + (
        rho_emit / (air_temperature**2.5)
    )

    dry_adiabatic_lapse_rate = constants.g / constants.c_pd
    for _ in range(n_iter):
        z_est = 0.5 * (z_1 + z_2)
        rhs = thermo.e_sat_ice(air_temperature + dry_adiabatic_lapse_rate * z_est) / (
            constants.R_v * (air_temperature + dry_adiabatic_lapse_rate * z_est) ** 3.5
        )
        z_1[lhs > rhs] = z_est[lhs > rhs]
        z_2[lhs < rhs] = z_est[lhs < rhs]

    return 0.5 * (z_1 + z_2)


def plume_area(wingspan: npt.NDArray[np.floating] | float) -> npt.NDArray[np.floating] | float:
    """Calculate area of the wake-vortex plume.

    Parameters
    ----------
    wingspan : npt.NDArray[np.floating] | float
        aircraft wingspan, [:math:`m`]

    Returns
    -------
    float
        Area of two wake-vortex plumes, [:math:`m^{2}`]

    Notes
    -----
    - See Appendix A2 in eq. (A6) and (A7) in :cite:`lottermoserHighResolutionEarlyContrails2025`.
    """
    r_plume = 1.5 + 0.314 * wingspan
    return 2.0 * np.pi * r_plume**2


def z_desc_length_scale(
    wingspan: npt.NDArray[np.floating] | float,
    air_temperature: npt.NDArray[np.floating],
    air_pressure: npt.NDArray[np.floating],
    true_airspeed: npt.NDArray[np.floating],
    aircraft_mass: npt.NDArray[np.floating],
    dT_dz: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    """Calculate the final vertical displacement of the wake vortex.

    Parameters
    ----------
    wingspan : npt.NDArray[np.floating] | float
        aircraft wingspan, [:math:`m`]
    air_temperature : npt.NDArray[np.floating]
        ambient temperature for each waypoint, [:math:`K`]
    air_pressure : npt.NDArray[np.floating]
        pressure altitude at each waypoint, [:math:`Pa`]
    true_airspeed : npt.NDArray[np.floating]
        true airspeed for each waypoint, [:math:`m s^{-1}`]
    aircraft_mass : npt.NDArray[np.floating] | float
        aircraft mass for each waypoint, [:math:`kg`]
    dT_dz : npt.NDArray[np.floating]
        potential temperature gradient, [:math:`K m^{-1}`]

    Returns
    -------
    npt.NDArray[np.floating]
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
    wingspan: npt.NDArray[np.floating] | float,
    air_temperature: npt.NDArray[np.floating],
    air_pressure: npt.NDArray[np.floating],
    true_airspeed: npt.NDArray[np.floating],
    aircraft_mass: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    """Calculate initial wake vortex circulation.

    Parameters
    ----------
    wingspan : npt.NDArray[np.floating] | float
        aircraft wingspan, [:math:`m`]
    air_temperature : npt.NDArray[np.floating]
        ambient temperature for each waypoint, [:math:`K`]
    air_pressure : npt.NDArray[np.floating]
        pressure altitude at each waypoint, [:math:`Pa`]
    true_airspeed : npt.NDArray[np.floating]
        true airspeed for each waypoint, [:math:`m s^{-1}`]
    aircraft_mass : npt.NDArray[np.floating] | float
        aircraft mass for each waypoint, [:math:`kg`]

    Returns
    -------
    npt.NDArray[np.floating]
        Initial wake vortex circulation, [:math:`m^{2} s^{-1}`]

    Notes
    -----
    - This is a measure of the strength/intensity of the wake vortex circulation.
    - See eq. (A1) in :cite:`unterstrasserPropertiesYoungContrails2016`.
    """
    b_0 = wake_vortex_separation(wingspan)
    rho_air = thermo.rho_d(air_temperature, air_pressure)
    return (constants.g * aircraft_mass) / (rho_air * b_0 * true_airspeed)


def _survival_fraction_from_length_scale(
    z_total: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    """
    Calculate fraction of ice particle number surviving the wake vortex phase.

    Parameters
    ----------
    z_total : npt.NDArray[np.floating]
        Total length-scale effect of the wake vortex downwash, [:math:`m`]

    Returns
    -------
    npt.NDArray[np.floating]
        Fraction of ice particle number surviving the wake vortex phase
    """
    f_surv = 0.42 + (1.31 / np.pi) * np.arctan(-1.00 + (z_total / 100.0))
    np.clip(f_surv, 0.0, 1.0, out=f_surv)
    return f_surv


def initial_contrail_depth(
    z_desc: npt.NDArray[np.floating],
    f_surv: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    """Calculate initial contrail depth using :cite:`unterstrasserPropertiesYoungContrails2016`.

    Parameters
    ----------
    z_desc : npt.NDArray[np.floating]
        Final vertical displacement of the wake vortex, ``dz_max`` in :mod:`wake_vortex.py`,
        [:math:`m`].
    f_surv : npt.NDArray[np.floating]
        Fraction of contrail ice particle number that survive the wake vortex phase.
        See :func:`ice_particle_survival_fraction`.

    Returns
    -------
    npt.NDArray[np.floating]
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
