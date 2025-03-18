"""
Module for calculating radiative forcing of contrail cirrus.

References
----------
- :cite:`schumannEffectiveRadiusIce2011`
- :cite:`schumannParametricRadiativeForcing2012`
"""

from __future__ import annotations

import dataclasses
import itertools

import numpy as np
import numpy.typing as npt
import xarray as xr

from pycontrails.core.met import MetDataArray, MetDataset
from pycontrails.core.vector import GeoVectorDataset
from pycontrails.physics import geo


@dataclasses.dataclass(frozen=True)
class RFConstants:
    """
    Constants that are used to calculate the local contrail radiative forcing.

    See Table 1 of :cite:`schumannParametricRadiativeForcing2012`.

    Each coefficient has 8 elements, one corresponding to each contrail ice particle habit (shape)::

        [
            Sphere,
            Solid column,
            Hollow column,
            Rough aggregate,
            Rosette-6,
            Plate,
            Droxtal,
            Myhre,
        ]

    For each waypoint, the distinct mix of ice particle habits are approximated using the mean
    contrail ice particle radius (``r_vol_um``) relative to ``radius_threshold_um``.

    For example:

    - if ``r_vol_um`` for a waypoint < 5 um, the mix of ice particle habits will be 100% droxtals.
    - if ``r_vol_um`` for a waypoint between 5 and 9.5 um, the mix of ice particle habits will
      be 30% solid columns, 70% droxtals.

    See Table 2 from :cite:`schumannEffectiveRadiusIce2011`.


    References
    ----------
    - :cite:`schumannEffectiveRadiusIce2011`
    - :cite:`schumannParametricRadiativeForcing2012`
    """

    # -----
    # Variables/coefficients used to calculate the local contrail longwave radiative forcing.
    # -----

    #: Linear approximation of Stefan-Boltzmann Law
    #: :math:`k_t` in Eq. (2) in :cite:`schumannParametricRadiativeForcing2012`
    k_t = np.array([1.93466, 1.95456, 1.95994, 1.95906, 1.94397, 1.95123, 2.30363, 1.94611])

    #: Approximates the temperature of the atmosphere without contrails
    #: :math:`T_{0}` in Eq. (2) in :cite:`schumannParametricRadiativeForcing2012`
    T_0 = np.array([152.237, 152.724, 152.923, 152.360, 151.879, 152.318, 165.692, 153.073])

    #: Approximate the effective emmissivity factor
    #: :math:`\delta_{\tau} in :cite:`schumannParametricRadiativeForcing2012`
    delta_t = np.array(
        [0.940846, 0.808397, 0.736222, 0.675591, 0.748757, 0.708515, 0.927592, 0.795527]
    )

    #: Effective radius scaling factor for optical properties (extinction relative to scattering)
    #: :math:`\delta_{lr} in Eq. (3) in :cite:`schumannParametricRadiativeForcing2012`
    delta_lr = np.array([0.211276, 0.341194, 0.325496, 0.255921, 0.170265, 1.65441, 0.201949, 0])

    #: Optical depth scaling factor for reduction of the OLR at the contrail level due
    #: to existing cirrus above the contrail
    #: :math:`\delta_{lc} in Eq. (4) in :cite:`schumannParametricRadiativeForcing2012`
    delta_lc = np.array(
        [0.159942, 0.0958129, 0.0924850, 0.0462023, 0.132925, 0.0870067, 0.0626339, 0.0665289]
    )

    # -----
    # Variables/coefficients used to calculate the local contrail shortwave radiative forcing.
    # -----

    #: Approximates the dependence on the effective albedo
    #: :math:`t_a`: Eq. (5) in :cite:`schumannParametricRadiativeForcing2012`
    t_a = np.array([0.879119, 0.901701, 0.881812, 0.899144, 0.879896, 0.883212, 0.899096, 1.00744])

    # Approximates the albedo of the contrail
    #: :math:`A_{\mu}` in Eq. (6) in :cite:`schumannParametricRadiativeForcing2012`
    A_mu = np.array(
        [0.361226, 0.294072, 0.343894, 0.317866, 0.337227, 0.310978, 0.342593, 0.269179]
    )

    # Approximates the albedo of the contrail
    #: :math:`C_{\mu}` in Eq. (6) in :cite:`schumannParametricRadiativeForcing2012`
    C_mu = np.array(
        [0.709300, 0.678016, 0.687546, 0.675315, 0.712041, 0.713317, 0.660267, 0.545716]
    )

    #: Approximates the effective contrail optical depth
    #: :math:`delta_sr` in Eq. (7) and (8) in :cite:`schumannParametricRadiativeForcing2012`
    delta_sr = np.array(
        [0.149851, 0.0254270, 0.0238836, 0.0463724, 0.0478892, 0.0700234, 0.0517942, 0]
    )

    #: Approximates the effective contrail optical depth
    #: :math:`F_r` in Eq. (7) and (8) in :cite:`schumannParametricRadiativeForcing2012`
    F_r = np.array([0.511852, 0.576911, 0.597351, 0.225750, 0.550734, 0.817858, 0.249004, 0])

    #: Approximates the contrail reflectances
    #: :math:`\gamma` in Eq. (9) in :cite:`schumannParametricRadiativeForcing2012`
    gamma_lower = np.array(
        [0.323166, 0.392598, 0.356189, 0.345040, 0.407515, 0.523604, 0.310853, 0.274741]
    )

    #: Approximates the contrail reflectances
    #: :math:`\Gamma` in Eq. (9) in :cite:`schumannParametricRadiativeForcing2012`
    gamma_upper = np.array(
        [0.241507, 0.347023, 0.288452, 0.296813, 0.327857, 0.437560, 0.274710, 0.208154]
    )

    #: Approximate the SZA-dependent contrail sideward scattering
    #: :math:`B_{\mu}` in Eq. (10) in :cite:`schumannParametricRadiativeForcing2012`
    B_mu = np.array([1.67592, 1.55687, 1.71065, 1.55843, 1.70782, 1.71789, 1.56399, 1.59015])

    #: Account for the optical depth of natural cirrus above the contrail
    #: :math:`\delta_{sc}` in Eq. (11) in :cite:`schumannParametricRadiativeForcing2012`
    delta_sc = np.array(
        [0.157017, 0.143274, 0.167995, 0.148547, 0.173036, 0.162442, 0.171855, 0.213488]
    )

    #: Account for the optical depth of natural cirrus above the contrail
    # :math:`\delta'_{sc}` in Eq. (11) in :cite:`schumannParametricRadiativeForcing2012`
    delta_sc_aps = np.array(
        [0.229574, 0.197611, 0.245036, 0.204875, 0.248328, 0.254029, 0.244051, 0.302246]
    )


# create a new constants class to use within module
RF_CONST = RFConstants()


# ----------
# Ice Habits
# ----------


def habit_weights(
    r_vol_um: npt.NDArray[np.floating],
    habit_distributions: npt.NDArray[np.floating],
    radius_threshold_um: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    r"""Assign weights to different ice particle habits for each waypoint.

    For each waypoint, the distinct mix of ice particle habits are approximated
    using the mean contrail ice particle radius (``r_vol_um``) binned by ``radius_threshold_um``.

    For example:

    - For waypoints with r_vol_um < 5 um, the mix of ice particle habits will
      be from Group 1 (100% Droxtals, refer to :attr:`CocipParams().habit_distributions`).
    - For waypoints with 5 um <= ``r_vol_um`` < 9.5 um, the mix of ice particle
      habits will be from Group 2 (30% solid columns, 70% droxtals)

    Parameters
    ----------
    r_vol_um : npt.NDArray[np.floating]
        Contrail ice particle volume mean radius, [:math:`\mu m`]
    habit_distributions : npt.NDArray[np.floating]
        Habit weight distributions.
        See :attr:`CocipParams().habit_distributions`
    radius_threshold_um : npt.NDArray[np.floating]
        Radius thresholds for habit distributions.
        See :attr:`CocipParams.radius_threshold_um`

    Returns
    -------
    npt.NDArray[np.floating]
        Array with shape ``n_waypoints x 8 columns``, where each column is the weights to the ice
        particle habits, [:math:`[0 - 1]`], and the sum of each column should be equal to 1.

    Raises
    ------
    ValueError
        Raises when ``habit_distributions`` do not sum to 1 across columns or
        if there is a size mismatch with ``radius_threshold_um``.
    """
    # all rows of the habit weights should sum to 1
    if not np.allclose(np.sum(habit_distributions, axis=1), 1.0, atol=1e-3):
        raise ValueError("Habit weight distributions must sum to 1 across columns")

    if habit_distributions.shape[0] != (radius_threshold_um.size + 1):
        raise ValueError(
            "The number of rows in `habit_distributions` must equal 1 + the "
            "size of `radius_threshold_um`"
        )

    # assign ice particle habits for each waypoint
    idx = habit_weight_regime_idx(r_vol_um, radius_threshold_um)
    return habit_distributions[idx]


def habit_weight_regime_idx(
    r_vol_um: npt.NDArray[np.floating], radius_threshold_um: npt.NDArray[np.floating]
) -> npt.NDArray[np.intp]:
    r"""
    Determine regime of ice particle habits based on contrail ice particle volume mean radius.

    Parameters
    ----------
    r_vol_um : npt.NDArray[np.floating]
        Contrail ice particle volume mean radius, [:math:`\mu m`]
    radius_threshold_um : npt.NDArray[np.floating]
        Radius thresholds for habit distributions.
        See :attr:`CocipParams.radius_threshold_um`

    Returns
    -------
    npt.NDArray[np.intp]
        Row index of the habit distribution in array :attr:`CocipParams().habit_distributions`
    """
    # find the regime for each waypoint using thresholds
    idx = np.digitize(r_vol_um, radius_threshold_um)

    # set any nan values to the "0" type
    idx[np.isnan(r_vol_um)] = 0

    return idx


def effective_radius_by_habit(
    r_vol_um: npt.NDArray[np.floating], habit_idx: npt.NDArray[np.intp]
) -> np.ndarray:
    r"""Calculate the effective radius ``r_eff_um`` via the mean ice particle radius and habit type.

    The ``habit_idx`` corresponds to the habit types in ``rf_const.habits``.
    Each habit type has a specific parameterization to calculate ``r_eff_um`` based on ``r_vol_um``.
    derived from :cite:`schumannEffectiveRadiusIce2011`.

    Parameters
    ----------
    r_vol_um : npt.NDArray[np.floating]
        Contrail ice particle volume mean radius, [:math:`\mu m`]
    habit_idx : npt.NDArray[np.intp]
        Habit type index for the contrail ice particle, corresponding to the
        habits in ``rf_const.habits``.

    Returns
    -------
    npt.NDArray[np.floating]
        Effective radius of ice particles for each combination of ``r_vol_um``
        and ``habit_idx``, [:math:`\mu m`]

    References
    ----------
    - :cite:`schumannEffectiveRadiusIce2011`
    """
    cond_list = [
        habit_idx == 0,
        habit_idx == 1,
        habit_idx == 2,
        habit_idx == 3,
        habit_idx == 4,
        habit_idx == 5,
        habit_idx == 6,
        habit_idx == 7,
    ]
    func_list = [
        effective_radius_sphere,
        effective_radius_solid_column,
        effective_radius_hollow_column,
        effective_radius_rough_aggregate,
        effective_radius_rosette,
        effective_radius_plate,
        effective_radius_droxtal,
        effective_radius_myhre,
        0.0,
    ]
    return np.piecewise(r_vol_um, cond_list, func_list)


def effective_radius_sphere(r_vol_um: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    r"""
    Calculate the effective radius of contrail ice particles assuming a sphere particle habit.

    Parameters
    ----------
    r_vol_um : npt.NDArray[np.floating]
        Contrail ice particle volume mean radius, [:math:`\mu m`]

    Returns
    -------
    npt.NDArray[np.floating]
        Effective radius, [:math:`\mu m`]
    """
    return np.minimum(r_vol_um, 25.0)


def effective_radius_solid_column(r_vol_um: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    r"""
    Calculate the effective radius of contrail ice particles assuming a solid column particle habit.

    Parameters
    ----------
    r_vol_um : npt.NDArray[np.floating]
        Contrail ice particle volume mean radius, [:math:`\mu m`]

    Returns
    -------
    npt.NDArray[np.floating]
        Effective radius, [:math:`\mu m`]
    """
    r_eff_um = (
        0.2588 * np.exp(-(6.912e-3 * r_vol_um)) + 0.6372 * np.exp(-(3.142e-4 * r_vol_um))
    ) * r_vol_um
    is_small = r_vol_um <= 42.2
    r_eff_um[is_small] = 0.824 * r_vol_um[is_small]
    return np.minimum(r_eff_um, 45.0)


def effective_radius_hollow_column(r_vol_um: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    r"""Calculate the effective radius of ice particles assuming a hollow column particle habit.

    Parameters
    ----------
    r_vol_um : npt.NDArray[np.floating]
        Contrail ice particle volume mean radius, [:math:`\mu m`]

    Returns
    -------
    npt.NDArray[np.floating]
        Effective radius, [:math:`\mu m`]
    """
    r_eff_um = (
        0.2281 * np.exp(-(7.359e-3 * r_vol_um)) + 0.5651 * np.exp(-(3.350e-4 * r_vol_um))
    ) * r_vol_um
    is_small = r_vol_um <= 39.7
    r_eff_um[is_small] = 0.729 * r_vol_um[is_small]
    return np.minimum(r_eff_um, 45.0)


def effective_radius_rough_aggregate(
    r_vol_um: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    r"""Calculate the effective radius of ice particles assuming a rough aggregate particle habit.

    Parameters
    ----------
    r_vol_um : npt.NDArray[np.floating]
        Contrail ice particle volume mean radius, [:math:`\mu m`]

    Returns
    -------
    npt.NDArray[np.floating]
        Effective radius, [:math:`\mu m`]
    """
    r_eff_um = 0.574 * r_vol_um
    return np.minimum(r_eff_um, 45.0)


def effective_radius_rosette(r_vol_um: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    r"""
    Calculate the effective radius of contrail ice particles assuming a rosette particle habit.

    Parameters
    ----------
    r_vol_um : npt.NDArray[np.floating]
        Contrail ice particle volume mean radius, [:math:`\mu m`]

    Returns
    -------
    npt.NDArray[np.floating]
        Effective radius, [:math:`\mu m`]
    """
    r_eff_um = r_vol_um * (
        0.1770 * np.exp(-(2.144e-2 * r_vol_um)) + 0.4267 * np.exp(-(3.562e-4 * r_vol_um))
    )
    return np.minimum(r_eff_um, 45.0)


def effective_radius_plate(r_vol_um: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    r"""
    Calculate the effective radius of contrail ice particles assuming a plate particle habit.

    Parameters
    ----------
    r_vol_um : npt.NDArray[np.floating]
        Contrail ice particle volume mean radius, [:math:`\mu m`]

    Returns
    -------
    npt.NDArray[np.floating]
        Effective radius, [:math:`\mu m`]
    """
    r_eff_um = r_vol_um * (
        0.1663 + 0.3713 * np.exp(-(0.0336 * r_vol_um)) + 0.3309 * np.exp(-(0.0035 * r_vol_um))
    )
    return np.minimum(r_eff_um, 45.0)


def effective_radius_droxtal(r_vol_um: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    r"""
    Calculate the effective radius of contrail ice particles assuming a droxtal particle habit.

    Parameters
    ----------
    r_vol_um : npt.NDArray[np.floating]
        Contrail ice particle volume mean radius, [:math:`\mu m`]

    Returns
    -------
    npt.NDArray[np.floating]
        Effective radius, [:math:`\mu m`]
    """
    r_eff_um = 0.94 * r_vol_um
    return np.minimum(r_eff_um, 45.0)


def effective_radius_myhre(r_vol_um: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    r"""
    Calculate the effective radius of contrail ice particles assuming a sphere particle habit.

    Parameters
    ----------
    r_vol_um : npt.NDArray[np.floating]
        Contrail ice particle volume mean radius, [:math:`\mu m`]

    Returns
    -------
    npt.NDArray[np.floating]
        Effective radius, [:math:`\mu m`]
    """
    return np.minimum(r_vol_um, 45.0)


# -----------------
# Radiative Forcing
# -----------------


def longwave_radiative_forcing(
    r_vol_um: npt.NDArray[np.floating],
    olr: npt.NDArray[np.floating],
    air_temperature: npt.NDArray[np.floating],
    tau_contrail: npt.NDArray[np.floating],
    tau_cirrus: npt.NDArray[np.floating],
    habit_weights_: npt.NDArray[np.floating],
    r_eff_um: npt.NDArray[np.floating] | None = None,
) -> npt.NDArray[np.floating]:
    r"""
    Calculate the local contrail longwave radiative forcing (:math:`RF_{LW}`).

    All returned values are positive.

    Parameters
    ----------
    r_vol_um : npt.NDArray[np.floating]
        Contrail ice particle volume mean radius, [:math:`\mu m`]
    olr : npt.NDArray[np.floating]
        Outgoing longwave radiation at each waypoint, [:math:`W m^{-2}`]
    air_temperature : npt.NDArray[np.floating]
        Ambient temperature at each waypoint, [:math:`K`]
    tau_contrail : npt.NDArray[np.floating]
        Contrail optical depth at each waypoint
    tau_cirrus : npt.NDArray[np.floating]
        Optical depth of numerical weather prediction (NWP) cirrus above the
        contrail at each waypoint
    habit_weights_ : npt.NDArray[np.floating]
        Weights to different ice particle habits for each waypoint,
        ``n_waypoints x 8`` (habit) columns, [:math:`[0 - 1]`]
    r_eff_um : npt.NDArray[np.floating], optional
        Provide effective radius corresponding to elements in ``r_vol_um``, [:math:`\mu m`].
        Defaults to None, which means the effective radius will be calculated using ``r_vol_um``
        and habit types in :func:`effective_radius_by_habit`.

    Returns
    -------
    npt.NDArray[np.floating]
        Local contrail longwave radiative forcing (positive), [:math:`W m^{-2}`]

    Raises
    ------
    ValueError
        If `r_eff_um` and `olr` have different shapes.

    References
    ----------
    - :cite:`schumannParametricRadiativeForcing2012`
    """
    # get list of habit weight indexs where the weights > 0
    # this is a tuple of (np.array[waypoint index], np.array[habit type index])
    habit_weight_mask = habit_weights_ > 0.0
    idx0, idx1 = np.nonzero(habit_weight_mask)

    # Convert parametric coefficients for vectorized operations
    delta_t = RF_CONST.delta_t[idx1]
    delta_lc = RF_CONST.delta_lc[idx1]
    delta_lr = RF_CONST.delta_lr[idx1]
    k_t = RF_CONST.k_t[idx1]
    T_0 = RF_CONST.T_0[idx1]

    olr_h = olr[idx0]
    tau_cirrus_h = tau_cirrus[idx0]
    tau_contrail_h = tau_contrail[idx0]
    air_temperature_h = air_temperature[idx0]

    # effective radius
    if r_eff_um is None:
        r_vol_um_h = r_vol_um[idx0]
        r_eff_um_h = effective_radius_by_habit(r_vol_um_h, idx1)
    else:
        if r_eff_um.shape != olr.shape:
            raise ValueError(
                "User provided effective radius (`r_eff_um`) must have the same shape as `olr`"
                f" {olr.shape}"
            )

        r_eff_um_h = r_eff_um[idx0]

    # Longwave radiation calculations
    e_lw = olr_reduction_natural_cirrus(tau_cirrus_h, delta_lc)
    f_lw = contrail_effective_emissivity(r_eff_um_h, delta_lr)

    # calculate the RF LW per habit type
    # see eqn (2) in :cite:`schumannParametricRadiativeForcing2012`
    rf_lw_per_habit = (
        (olr_h - k_t * (air_temperature_h - T_0))
        * e_lw
        * (1.0 - np.exp(-delta_t * f_lw * tau_contrail_h))
    )
    rf_lw_per_habit.clip(min=0.0, out=rf_lw_per_habit)

    # Weight and sum the RF contributions of each habit type according the habit weight
    # regime at the waypoint
    # see eqn (12) in :cite:`schumannParametricRadiativeForcing2012`
    # use fancy indexing to re-assign values to 2d array of waypoint x habit type
    rf_lw_weighted = np.zeros_like(habit_weights_)
    rf_lw_weighted[idx0, idx1] = rf_lw_per_habit * habit_weights_[habit_weight_mask]
    return np.sum(rf_lw_weighted, axis=1)


def shortwave_radiative_forcing(
    r_vol_um: npt.NDArray[np.floating],
    sdr: npt.NDArray[np.floating],
    rsr: npt.NDArray[np.floating],
    sd0: npt.NDArray[np.floating],
    tau_contrail: npt.NDArray[np.floating],
    tau_cirrus: npt.NDArray[np.floating],
    habit_weights_: npt.NDArray[np.floating],
    r_eff_um: npt.NDArray[np.floating] | None = None,
) -> npt.NDArray[np.floating]:
    r"""
    Calculate the local contrail shortwave radiative forcing (:math:`RF_{SW}`).

    All returned values are negative.

    Parameters
    ----------
    r_vol_um : npt.NDArray[np.floating]
        Contrail ice particle volume mean radius, [:math:`\mu m`]
    sdr : npt.NDArray[np.floating]
        Solar direct radiation, [:math:`W m^{-2}`]
    rsr : npt.NDArray[np.floating]
        Reflected solar radiation, [:math:`W m^{-2}`]
    sd0 : npt.NDArray[np.floating]
        Solar constant, [:math:`W m^{-2}`]
    tau_contrail : npt.NDArray[np.floating]
        Contrail optical depth for each waypoint
    tau_cirrus : npt.NDArray[np.floating]
        Optical depth of numerical weather prediction (NWP) cirrus above the
        contrail for each waypoint.
    habit_weights_ : npt.NDArray[np.floating]
        Weights to different ice particle habits for each waypoint,
        ``n_waypoints x 8`` (habit) columns, [:math:`[0 - 1]`]
    r_eff_um : npt.NDArray[np.floating], optional
        Provide effective radius corresponding to elements in ``r_vol_um``, [:math:`\mu m`].
        Defaults to None, which means the effective radius will be calculated using ``r_vol_um``
        and habit types in :func:`effective_radius_by_habit`.

    Returns
    -------
    npt.NDArray[np.floating]
        Local contrail shortwave radiative forcing (negative), [:math:`W m^{-2}`]

    Raises
    ------
    ValueError
        If `r_eff_um` and `sdr` have different shapes.

    References
    ----------
    - :cite:`schumannParametricRadiativeForcing2012`
    """
    # create mask for daytime (sdr > 0)
    day = sdr > 0.0

    # short circuit if no waypoints occur during the day
    if not day.any():
        return np.zeros_like(sdr)

    # get list of habit weight indexs where the weights > 0
    # this is a tuple of (np.array[waypoint index], np.array[habit type index])
    habit_weight_mask = day.reshape(day.size, 1) & (habit_weights_ > 0.0)
    idx0, idx1 = np.nonzero(habit_weight_mask)

    # Convert parametric coefficients for vectorized operations
    t_a = RF_CONST.t_a[idx1]
    A_mu = RF_CONST.A_mu[idx1]
    B_mu = RF_CONST.B_mu[idx1]
    C_mu = RF_CONST.C_mu[idx1]
    delta_sr = RF_CONST.delta_sr[idx1]
    F_r = RF_CONST.F_r[idx1]
    gamma_lower = RF_CONST.gamma_lower[idx1]
    gamma_upper = RF_CONST.gamma_upper[idx1]
    delta_sc = RF_CONST.delta_sc[idx1]
    delta_sc_aps = RF_CONST.delta_sc_aps[idx1]

    sdr_h = sdr[idx0]
    rsr_h = rsr[idx0]
    sd0_h = sd0[idx0]
    tau_contrail_h = tau_contrail[idx0]
    tau_cirrus_h = tau_cirrus[idx0]

    albedo_ = albedo(sdr_h, rsr_h)
    mue = np.minimum(sdr_h / sd0_h, 1.0)

    # effective radius
    if r_eff_um is None:
        r_vol_um_h = r_vol_um[idx0]
        r_eff_um_h = effective_radius_by_habit(r_vol_um_h, idx1)
    else:
        if r_eff_um.shape != sdr.shape:
            raise ValueError(
                "User provided effective radius (`r_eff_um`) must have the same shape as `sdr`"
                f" {sdr.shape}"
            )

        r_eff_um_h = r_eff_um[idx0]

    # Local contrail shortwave radiative forcing calculations
    alpha_c = contrail_albedo(
        tau_contrail_h,
        mue,
        r_eff_um_h,
        A_mu,
        B_mu,
        C_mu,
        delta_sr,
        F_r,
        gamma_lower,
        gamma_upper,
    )

    e_sw = effective_tau_cirrus(tau_cirrus_h, mue, delta_sc, delta_sc_aps)

    # calculate the RF SW per habit type
    # see eqn (5) in :cite:`schumannParametricRadiativeForcing2012`
    rf_sw_per_habit = np.minimum(-sdr_h * ((t_a - albedo_) ** 2) * alpha_c * e_sw, 0.0)

    # Weight and sum the RF contributions of each habit type according the
    # habit weight regime at the waypoint
    # see eqn (12) in :cite:`schumannParametricRadiativeForcing2012`
    # use fancy indexing to re-assign values to 2d array of waypoint x habit type
    rf_sw_weighted = np.zeros_like(habit_weights_)
    rf_sw_weighted[idx0, idx1] = rf_sw_per_habit * habit_weights_[habit_weight_mask]

    return np.sum(rf_sw_weighted, axis=1)


def net_radiative_forcing(
    rf_lw: npt.NDArray[np.floating], rf_sw: npt.NDArray[np.floating]
) -> npt.NDArray[np.floating]:
    """
    Calculate the local contrail net radiative forcing (rf_net).

    RF Net = Longwave RF (positive) + Shortwave RF (negative)

    Parameters
    ----------
    rf_lw : npt.NDArray[np.floating]
        local contrail longwave radiative forcing, [:math:`W m^{-2}`]
    rf_sw : npt.NDArray[np.floating]
        local contrail shortwave radiative forcing, [:math:`W m^{-2}`]

    Returns
    -------
    npt.NDArray[np.floating]
        local contrail net radiative forcing, [:math:`W m^{-2}`]
    """
    return rf_lw + rf_sw


def olr_reduction_natural_cirrus(
    tau_cirrus: npt.NDArray[np.floating], delta_lc: npt.NDArray[np.floating]
) -> npt.NDArray[np.floating]:
    """
    Calculate reduction in outgoing longwave radiation (OLR) due to the presence of natural cirrus.

    Natural cirrus has optical depth ``tau_cirrus`` above the contrail.
    See ``e_lw`` in Eq. (4) of Schumann et al. (2012).

    Parameters
    ----------
    tau_cirrus : npt.NDArray[np.floating]
        Optical depth of numerical weather prediction (NWP) cirrus above the
        contrail for each waypoint.
    delta_lc : npt.NDArray[np.floating]
        Habit specific parameter to approximate the reduction of the outgoing
        longwave radiation at the contrail level due to natural cirrus above the contrail.

    Returns
    -------
    npt.NDArray[np.floating]
        Reduction of outgoing longwave radiation
    """
    # e_lw calculations
    return np.exp(-delta_lc * tau_cirrus)


def contrail_effective_emissivity(
    r_eff_um: npt.NDArray[np.floating], delta_lr: npt.NDArray[np.floating]
) -> npt.NDArray[np.floating]:
    r"""Calculate the effective emissivity of the contrail, ``f_lw``.

    Refer to Eq. (3) of Schumann et al. (2012).

    Parameters
    ----------
    r_eff_um : npt.NDArray[np.floating]
        Effective radius for each waypoint, n_waypoints x 8 (habit) columns, [:math:`\mu m`]
        See :func:`effective_radius_habit`.
    delta_lr : npt.NDArray[np.floating]
        Habit specific parameter to approximate the effective emissivity of the contrail.

    Returns
    -------
    npt.NDArray[np.floating]
        Effective emissivity of the contrail
    """
    # f_lw calculations
    return 1.0 - np.exp(-delta_lr * r_eff_um)


def albedo(
    sdr: npt.NDArray[np.floating], rsr: npt.NDArray[np.floating]
) -> npt.NDArray[np.floating]:
    """
    Calculate albedo along contrail waypoint.

    Albedo, the diffuse reflection of solar radiation out of the total solar radiation,
    is computed based on the solar direct radiation (`sdr`) and reflected solar radiation (`rsr`).

    Output values range between 0 (corresponding to a black body that absorbs
    all incident radiation) and 1 (a body that reflects all incident radiation).

    Parameters
    ----------
    sdr : npt.NDArray[np.floating]
        Solar direct radiation, [:math:`W m^{-2}`]
    rsr : npt.NDArray[np.floating]
        Reflected solar radiation, [:math:`W m^{-2}`]

    Returns
    -------
    npt.NDArray[np.floating]
        Albedo value, [:math:`[0 - 1]`]
    """
    day = sdr > 0.0
    albedo_ = np.zeros(sdr.shape)
    albedo_[day] = rsr[day] / sdr[day]
    albedo_.clip(0.0, 1.0, out=albedo_)
    return albedo_


def contrail_albedo(
    tau_contrail: npt.NDArray[np.floating],
    mue: npt.NDArray[np.floating],
    r_eff_um: npt.NDArray[np.floating],
    A_mu: npt.NDArray[np.floating],
    B_mu: npt.NDArray[np.floating],
    C_mu: npt.NDArray[np.floating],
    delta_sr: npt.NDArray[np.floating],
    F_r: npt.NDArray[np.floating],
    gamma_lower: npt.NDArray[np.floating],
    gamma_upper: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    r"""
    Calculate the contrail albedo, ``alpha_c``.

    Refer to Eq. (6) of Schumann et al. (2012),

    Parameters
    ----------
    tau_contrail : npt.NDArray[np.floating]
        Contrail optical depth for each waypoint
    mue : npt.NDArray[np.floating]
        Cosine of the solar zenith angle (theta), mue = cos(theta) = sdr/sd0
    r_eff_um : npt.NDArray[np.floating]
        Effective radius for each waypoint, n_waypoints x 8 (habit) columns, [:math:`\mu m`]
        See :func:`effective_radius_habit`.
    A_mu : npt.NDArray[np.floating]
        Habit-specific parameter to approximate the albedo of the contrail
    B_mu : npt.NDArray[np.floating]
        Habit-specific parameter to approximate the SZA-dependent contrail sideward scattering
    C_mu : npt.NDArray[np.floating]
        Habit-specific parameter to approximate the albedo of the contrail
    delta_sr : npt.NDArray[np.floating]
        Habit-specific parameter to approximate the effective contrail optical depth
    F_r : npt.NDArray[np.floating]
        Habit-specific parameter to approximate the effective contrail optical depth
    gamma_lower : npt.NDArray[np.floating]
        Habit-specific parameter to approximate the contrail reflectances
    gamma_upper : npt.NDArray[np.floating]
        Habit-specific parameter to approximate the contrail reflectances

    Returns
    -------
    npt.NDArray[np.floating]
        Contrail albedo for each waypoint and ice particle habit
    """
    tau_aps = tau_contrail * (1.0 - F_r * (1 - np.exp(-delta_sr * r_eff_um)))
    tau_eff = tau_aps / (mue + 1e-6)
    r_c = 1.0 - np.exp(-gamma_upper * tau_eff)
    r_c_aps = np.exp(-gamma_lower * tau_eff)

    f_mu = (2.0 * (1.0 - mue)) ** B_mu - 1.0
    return r_c * (C_mu + (A_mu * r_c_aps * f_mu))


def effective_tau_cirrus(
    tau_cirrus: npt.NDArray[np.floating],
    mue: npt.NDArray[np.floating],
    delta_sc: npt.NDArray[np.floating],
    delta_sc_aps: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    r"""
    Calculate the effective optical depth of natural cirrus above the contrail, ``e_sw``.

    Refer to Eq. (11) of :cite:`schumannParametricRadiativeForcing2012`. See Notes for
    a correction to the equation.

    Parameters
    ----------
    tau_cirrus : npt.NDArray[np.floating]
        Optical depth of numerical weather prediction (NWP) cirrus above the
        contrail for each waypoint.
    mue : npt.NDArray[np.floating]
        Cosine of the solar zenith angle (theta), mue = cos(theta) = sdr/sd0
    delta_sc : npt.NDArray[np.floating]
        Habit-specific parameter to account for the optical depth of natural
        cirrus above the contrail
    delta_sc_aps : npt.NDArray[np.floating]
        Habit-specific parameter to account for the optical depth of natural
        cirrus above the contrail

    Returns
    -------
    npt.NDArray[np.floating]
        Effective optical depth of natural cirrus above the contrail,
        ``n_waypoints x 8`` (habit) columns.

    Notes
    -----
    - In a personal correspondence, Dr. Schumann identified a print error in Eq. (11) in
      :cite:`schumannParametricRadiativeForcing2012`, where the positions of ``delta_sc_aps``
      and ``delta_sc`` should be swapped. The correct function is provided below.
    """
    tau_cirrus_eff = tau_cirrus / (mue + 1e-6)
    return np.exp(tau_cirrus * delta_sc_aps - tau_cirrus_eff * delta_sc)


# -----------------------------
# Contrail-contrail overlapping
# -----------------------------


def contrail_contrail_overlap_radiative_effects(
    contrails: GeoVectorDataset,
    habit_distributions: npt.NDArray[np.floating],
    radius_threshold_um: npt.NDArray[np.floating],
    *,
    min_altitude_m: float = 6000.0,
    max_altitude_m: float = 13000.0,
    dz_overlap_m: float = 500.0,
    spatial_grid_res: float = 0.25,
) -> GeoVectorDataset:
    r"""
    Calculate radiative properties after accounting for contrail overlapping.

    This function mutates the ``contrails`` parameter.

    Parameters
    ----------
    contrails : GeoVectorDataset
        Contrail waypoints at a given time. Must include the following variables:
        - segment_length
        - width
        - r_ice_vol
        - tau_contrail
        - tau_cirrus
        - air_temperature
        - sdr
        - rsr
        - olr

    habit_distributions : npt.NDArray[np.floating]
        Habit weight distributions.
        See :attr:`CocipParams.habit_distributions`
    radius_threshold_um : npt.NDArray[np.floating]
        Radius thresholds for habit distributions.
        See :attr:`CocipParams.radius_threshold_um`
    min_altitude_m : float
        Minimum altitude domain in simulation, [:math:`m`]
        See :attr:`CocipParams.min_altitude_m`
    max_altitude_m
        Maximum altitude domain in simulation, [:math:`m`]
        See :attr:`CocipParams.min_altitude_m`
    dz_overlap_m : float
        Altitude interval used to segment contrail waypoints, [:math:`m`]
        See :attr:`CocipParams.dz_overlap_m`
    spatial_grid_res : float
        Spatial grid resolution, [:math:`\deg`]

    Returns
    -------
    GeoVectorDataset
        Contrail waypoints at a given time with additional variables attached, including
        - rsr_overlap
        - olr_overlap
        - tau_cirrus_overlap
        - rf_sw_overlap
        - rf_lw_overlap
        - rf_net_overlap

    References
    ----------
    - Schumann et al. (2021) Air traffic and contrail changes over Europe during COVID-19:
        A model study, Atmos. Chem. Phys., 21, 7429-7450, https://doi.org/10.5194/ACP-21-7429-2021.
    - Teoh et al. (2023) Global aviation contrail climate effects from 2019 to 2021.

    Notes
    -----
    - The radiative effects of contrail-contrail overlapping is approximated by changing the
      background RSR and OLR fields, and the overlying cirrus optical depth above the contrail.
    - All contrail segments within each altitude interval are treated as one contrail layer, where
      they do not overlap. Contrail layers are processed starting from the bottom to the top.
    - Refer to the Supporting Information (S4.3) of Teoh et al. (2023)
    """
    assert "segment_length" in contrails
    assert "width" in contrails
    assert "r_ice_vol" in contrails
    assert "tau_contrail" in contrails
    assert "tau_cirrus" in contrails
    assert "air_temperature" in contrails
    assert "sdr" in contrails
    assert "rsr" in contrails
    assert "olr" in contrails

    if not contrails:
        raise ValueError("Parameter 'contrails' must be non-empty.")

    time = contrails["time"]
    time0 = time[0]
    if not np.all(time == time0):
        raise ValueError("Contrail waypoints must have a constant time.")

    longitude = contrails["longitude"]
    latitude = contrails["latitude"]
    altitude = contrails.altitude

    spatial_bbox = geo.spatial_bounding_box(longitude, latitude)
    west, south, east, north = spatial_bbox

    assert spatial_grid_res > 0.01
    lon_coords = np.arange(west, east + 0.01, spatial_grid_res)
    lat_coords = np.arange(south, north + 0.01, spatial_grid_res)

    dims = ["longitude", "latitude", "level", "time"]
    shape = (len(lon_coords), len(lat_coords), 1, 1)
    delta_rad_t = xr.Dataset(
        data_vars={"rsr": (dims, np.zeros(shape)), "olr": (dims, np.zeros(shape))},
        coords={"longitude": lon_coords, "latitude": lat_coords, "level": [-1.0], "time": [time0]},
    )

    # Initialise radiation fields to store change in background RSR and OLR due to contrails
    rsr_overlap = np.zeros_like(longitude)
    olr_overlap = np.zeros_like(longitude)
    tau_cirrus_overlap = np.zeros_like(longitude)
    rf_sw_overlap = np.zeros_like(longitude)
    rf_lw_overlap = np.zeros_like(longitude)
    rf_net_overlap = np.zeros_like(longitude)

    # Account for contrail overlapping starting from bottom to top layers
    altitude_layers = np.arange(min_altitude_m, max_altitude_m + 1.0, dz_overlap_m)

    for alt_layer0, alt_layer1 in itertools.pairwise(altitude_layers):
        is_in_layer = (altitude >= alt_layer0) & (altitude < alt_layer1)

        # Get contrail waypoints at current altitude layer
        contrails_level = contrails.filter(is_in_layer, copy=True)

        # Skip altitude layer if no contrails are present
        if not contrails_level:
            continue

        # Get contrails above altitude layer
        is_above_layer = (altitude >= alt_layer1) & (altitude <= max_altitude_m)
        contrails_above = contrails.filter(is_above_layer, copy=True)

        contrails_level = _contrail_optical_depth_above_contrail_layer(
            contrails_level,
            contrails_above,
            spatial_bbox=spatial_bbox,
            spatial_grid_res=spatial_grid_res,
        )

        # Calculate updated RSR and OLR with contrail overlapping
        contrails_level = _rsr_and_olr_with_contrail_overlap(contrails_level, delta_rad_t)

        # Calculate local contrail SW and LW RF with contrail overlapping
        contrails_level = _local_sw_and_lw_rf_with_contrail_overlap(
            contrails_level, habit_distributions, radius_threshold_um
        )

        # Cumulative change in background RSR and OLR fields
        delta_rad_t = _change_in_background_rsr_and_olr(
            contrails_level,
            delta_rad_t,
            spatial_bbox=spatial_bbox,
            spatial_grid_res=spatial_grid_res,
        )

        # Save values
        rsr_overlap[is_in_layer] = contrails_level["rsr_overlap"]
        olr_overlap[is_in_layer] = contrails_level["olr_overlap"]
        tau_cirrus_overlap[is_in_layer] = (
            contrails_level["tau_cirrus"] + contrails_level["tau_contrails_above"]
        )
        rf_sw_overlap[is_in_layer] = contrails_level["rf_sw_overlap"]
        rf_lw_overlap[is_in_layer] = contrails_level["rf_lw_overlap"]
        rf_net_overlap[is_in_layer] = contrails_level["rf_net_overlap"]

    # Add new variables to contrails
    contrails["rsr_overlap"] = rsr_overlap
    contrails["olr_overlap"] = olr_overlap
    contrails["tau_cirrus_overlap"] = tau_cirrus_overlap
    contrails["rf_sw_overlap"] = rf_sw_overlap
    contrails["rf_lw_overlap"] = rf_lw_overlap
    contrails["rf_net_overlap"] = rf_net_overlap

    return contrails


def _contrail_optical_depth_above_contrail_layer(
    contrails_level: GeoVectorDataset,
    contrails_above: GeoVectorDataset,
    spatial_bbox: tuple[float, float, float, float],
    spatial_grid_res: float,
) -> GeoVectorDataset:
    r"""
    Calculate the contrail optical depth above the contrail waypoints.

    Parameters
    ----------
    contrails_level : GeoVectorDataset
        Contrail waypoints at the current altitude layer.
    contrails_above : GeoVectorDataset
        Contrail waypoints above the current altitude layer.
    spatial_bbox: tuple[float, float, float, float]
        Spatial bounding box, ``(lon_min, lat_min, lon_max, lat_max)``, [:math:`\deg`]
    spatial_grid_res : float
        Spatial grid resolution, [:math:`\deg`]

    Returns
    -------
    GeoVectorDataset
        Contrail waypoints at the current altitude layer with `tau_contrails_above` attached.
    """
    contrails_above["tau_contrails_above"] = (
        contrails_above["tau_contrail"]
        * contrails_above["segment_length"]
        * contrails_above["width"]
    )

    # Aggregate contrail optical depth to a longitude-latitude grid
    da = contrails_above.to_lon_lat_grid(
        agg={"tau_contrails_above": "sum"},
        spatial_bbox=spatial_bbox,
        spatial_grid_res=spatial_grid_res,
    )["tau_contrails_above"]
    da = da.expand_dims(level=[-1.0], time=[contrails_level["time"][0]])
    da = da.transpose("longitude", "latitude", "level", "time")

    da_surface_area = geo.grid_surface_area(da["longitude"].values, da["latitude"].values)
    da = da / da_surface_area
    mda = MetDataArray(da)

    # Interpolate to contrails_level
    contrails_level["tau_contrails_above"] = contrails_level.intersect_met(mda)
    return contrails_level


def _rsr_and_olr_with_contrail_overlap(
    contrails_level: GeoVectorDataset, delta_rad_t: xr.Dataset
) -> GeoVectorDataset:
    """
    Calculate RSR and OLR at contrail waypoints after accounting for contrail overlapping.

    Parameters
    ----------
    contrails_level : GeoVectorDataset
        Contrail waypoints at the current altitude layer.
    delta_rad_t : xr.Dataset
        Radiation fields with cumulative change in RSR and OLR due to contrail overlapping.

    Returns
    -------
    GeoVectorDataset
        Contrail waypoints at the current altitude layer with `rsr_overlap` and
        `olr_overlap` attached.
    """
    mds = MetDataset(delta_rad_t)

    # Interpolate radiation fields to obtain `rsr_overlap` and `olr_overlap`
    delta_rsr = contrails_level.intersect_met(mds["rsr"])
    delta_olr = contrails_level.intersect_met(mds["olr"])

    # Constrain RSR so it is not larger than the SDR
    contrails_level["rsr_overlap"] = np.minimum(
        contrails_level["sdr"],
        contrails_level["rsr"] + delta_rsr,
    )

    # Constrain OLR so it is not smaller than 80% of the original value
    contrails_level["olr_overlap"] = np.maximum(
        0.8 * contrails_level["olr"],
        contrails_level["olr"] + delta_olr,
    )
    return contrails_level


def _local_sw_and_lw_rf_with_contrail_overlap(
    contrails_level: GeoVectorDataset,
    habit_distributions: npt.NDArray[np.floating],
    radius_threshold_um: npt.NDArray[np.floating],
) -> GeoVectorDataset:
    """
    Calculate local contrail SW and LW RF after accounting for contrail overlapping.

    Parameters
    ----------
    contrails_level : GeoVectorDataset
        Contrail waypoints at the current altitude layer.
    habit_distributions : npt.NDArray[np.floating]
        Habit weight distributions.
        See :attr:`CocipParams().habit_distributions`
    radius_threshold_um : npt.NDArray[np.floating]
        Radius thresholds for habit distributions.
        See :attr:`CocipParams.radius_threshold_um`

    Returns
    -------
    GeoVectorDataset
        Contrail waypoints at the current altitude layer with `rf_sw_overlap`,
        `rf_lw_overlap`, and `rf_net_overlap` attached.
    """
    r_vol_um = contrails_level["r_ice_vol"] * 1e6
    habit_w = habit_weights(r_vol_um, habit_distributions, radius_threshold_um)

    # Calculate solar constant
    theta_rad = geo.orbital_position(contrails_level["time"])
    sd0 = geo.solar_constant(theta_rad)
    tau_contrail = contrails_level["tau_contrail"]
    tau_cirrus = contrails_level["tau_cirrus"] + contrails_level["tau_contrails_above"]

    # Calculate local SW and LW RF
    contrails_level["rf_sw_overlap"] = shortwave_radiative_forcing(
        r_vol_um,
        contrails_level["sdr"],
        contrails_level["rsr_overlap"],
        sd0,
        tau_contrail,
        tau_cirrus,
        habit_w,
    )

    contrails_level["rf_lw_overlap"] = longwave_radiative_forcing(
        r_vol_um,
        contrails_level["olr_overlap"],
        contrails_level["air_temperature"],
        tau_contrail,
        tau_cirrus,
        habit_w,
    )
    contrails_level["rf_net_overlap"] = (
        contrails_level["rf_lw_overlap"] + contrails_level["rf_sw_overlap"]
    )
    return contrails_level


def _change_in_background_rsr_and_olr(
    contrails_level: GeoVectorDataset,
    delta_rad_t: xr.Dataset,
    *,
    spatial_bbox: tuple[float, float, float, float] = (-180.0, -90.0, 180.0, 90.0),
    spatial_grid_res: float = 0.5,
) -> xr.Dataset:
    r"""
    Calculate change in background RSR and OLR fields.

    Parameters
    ----------
    contrails_level : GeoVectorDataset
        Contrail waypoints at the current altitude layer.
    delta_rad_t : xr.Dataset
        Radiation fields with cumulative change in RSR and OLR due to contrail overlapping.
    spatial_bbox: tuple[float, float, float, float]
        Spatial bounding box, ``(lon_min, lat_min, lon_max, lat_max)``, [:math:`\deg`]
    spatial_grid_res : float
        Spatial grid resolution, [:math:`\deg`]

    Returns
    -------
    xr.Dataset
        Radiation fields with cumulative change in RSR and OLR due to contrail overlapping.
    """
    # Calculate SW and LW radiative flux (Units: W)
    segment_length = contrails_level["segment_length"]
    width = contrails_level["width"]

    contrails_level["sw_radiative_flux"] = (
        np.abs(contrails_level["rf_sw_overlap"]) * segment_length * width
    )

    contrails_level["lw_radiative_flux"] = contrails_level["rf_lw_overlap"] * segment_length * width

    # Aggregate SW and LW radiative flux to a longitude-latitude grid
    ds = contrails_level.to_lon_lat_grid(
        agg={"sw_radiative_flux": "sum", "lw_radiative_flux": "sum"},
        spatial_bbox=spatial_bbox,
        spatial_grid_res=spatial_grid_res,
    )
    ds = ds.expand_dims(level=[-1.0], time=[contrails_level["time"][0]])
    da_surface_area = geo.grid_surface_area(ds["longitude"].values, ds["latitude"].values)

    # Cumulative change in RSR and OLR
    delta_rad_t["rsr"] += ds["sw_radiative_flux"] / da_surface_area
    delta_rad_t["olr"] -= ds["lw_radiative_flux"] / da_surface_area
    return delta_rad_t
