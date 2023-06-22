"""
Module for calculating radiative forcing of contrail cirrus.

References
----------
- :cite:`schumannEffectiveRadiusIce2011`
- :cite:`schumannParametricRadiativeForcing2012`
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass
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
rf_const = RFConstants()


# ----------
# Ice Habits
# ----------


def habit_weights(
    r_vol_um: npt.NDArray[np.float_],
    habit_distributions: npt.NDArray[np.float_],
    radius_threshold_um: npt.NDArray[np.float_],
) -> npt.NDArray[np.float_]:
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
    r_vol_um : npt.NDArray[np.float_]
        Contrail ice particle volume mean radius, [:math:`\mu m`]
    habit_distributions : npt.NDArray[np.float_]
        Habit weight distributions.
        See :attr:`CocipParams().habit_distributions`
    radius_threshold_um : npt.NDArray[np.float_]
        Radius thresholds for habit distributions.
        See :attr:`CocipParams.radius_threshold_um`

    Returns
    -------
    npt.NDArray[np.float_]
        Array with shape ``n_waypoints x 8 columns``, where each column is the weights to the ice
        particle habits, [:math:`[0 - 1]`], and the sum of each column should be equal to 1.

    Raises
    ------
    ValueError
        Raises when ``habit_distributions`` do not sum to 1 across columns or
        if there is a size mismatch with ``radius_threshold_um``.
    """
    # all rows of the habit weights should sum to 1
    if not np.all(np.round(np.sum(habit_distributions, axis=1), 3) == 1):
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
    r_vol_um: npt.NDArray[np.float_], radius_threshold_um: npt.NDArray[np.float_]
) -> npt.NDArray[np.intp]:
    r"""
    Determine regime of ice particle habits based on contrail ice particle volume mean radius.

    Parameters
    ----------
    r_vol_um : npt.NDArray[np.float_]
        Contrail ice particle volume mean radius, [:math:`\mu m`]
    radius_threshold_um : npt.NDArray[np.float_]
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
    r_vol_um: npt.NDArray[np.float_], habit_idx: npt.NDArray[np.intp]
) -> np.ndarray:
    r"""Calculate the effective radius ``r_eff_um`` via the mean ice particle radius and habit type.

    The ``habit_idx`` corresponds to the habit types in ``rf_const.habits``.
    Each habit type has a specific parameterization to calculate ``r_eff_um`` based on ``r_vol_um``.
    derived from :cite:`schumannEffectiveRadiusIce2011`.

    Parameters
    ----------
    r_vol_um : npt.NDArray[np.float_]
        Contrail ice particle volume mean radius, [:math:`\mu m`]
    habit_idx : npt.NDArray[np.intp]
        Habit type index for the contrail ice particle, corresponding to the
        habits in ``rf_const.habits``.

    Returns
    -------
    npt.NDArray[np.float_]
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


def effective_radius_sphere(r_vol_um: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
    r"""
    Calculate the effective radius of contrail ice particles assuming a sphere particle habit.

    Parameters
    ----------
    r_vol_um : npt.NDArray[np.float_]
        Contrail ice particle volume mean radius, [:math:`\mu m`]

    Returns
    -------
    npt.NDArray[np.float_]
        Effective radius, [:math:`\mu m`]
    """
    return np.minimum(r_vol_um, 25.0)


def effective_radius_solid_column(r_vol_um: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
    r"""
    Calculate the effective radius of contrail ice particles assuming a solid column particle habit.

    Parameters
    ----------
    r_vol_um : npt.NDArray[np.float_]
        Contrail ice particle volume mean radius, [:math:`\mu m`]

    Returns
    -------
    npt.NDArray[np.float_]
        Effective radius, [:math:`\mu m`]
    """
    r_eff_um = (
        0.2588 * np.exp(-(6.912e-3 * r_vol_um)) + 0.6372 * np.exp(-(3.142e-4 * r_vol_um))
    ) * r_vol_um
    is_small = r_vol_um <= 42.2
    r_eff_um[is_small] = 0.824 * r_vol_um[is_small]
    return np.minimum(r_eff_um, 45.0)


def effective_radius_hollow_column(r_vol_um: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
    r"""Calculate the effective radius of ice particles assuming a hollow column particle habit.

    Parameters
    ----------
    r_vol_um : npt.NDArray[np.float_]
        Contrail ice particle volume mean radius, [:math:`\mu m`]

    Returns
    -------
    npt.NDArray[np.float_]
        Effective radius, [:math:`\mu m`]
    """
    r_eff_um = (
        0.2281 * np.exp(-(7.359e-3 * r_vol_um)) + 0.5651 * np.exp(-(3.350e-4 * r_vol_um))
    ) * r_vol_um
    is_small = r_vol_um <= 39.7
    r_eff_um[is_small] = 0.729 * r_vol_um[is_small]
    return np.minimum(r_eff_um, 45.0)


def effective_radius_rough_aggregate(r_vol_um: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
    r"""Calculate the effective radius of ice particles assuming a rough aggregate particle habit.

    Parameters
    ----------
    r_vol_um : npt.NDArray[np.float_]
        Contrail ice particle volume mean radius, [:math:`\mu m`]

    Returns
    -------
    npt.NDArray[np.float_]
        Effective radius, [:math:`\mu m`]
    """
    r_eff_um = 0.574 * r_vol_um
    return np.minimum(r_eff_um, 45.0)


def effective_radius_rosette(r_vol_um: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
    r"""
    Calculate the effective radius of contrail ice particles assuming a rosette particle habit.

    Parameters
    ----------
    r_vol_um : npt.NDArray[np.float_]
        Contrail ice particle volume mean radius, [:math:`\mu m`]

    Returns
    -------
    npt.NDArray[np.float_]
        Effective radius, [:math:`\mu m`]
    """
    r_eff_um = r_vol_um * (
        0.1770 * np.exp(-(2.144e-2 * r_vol_um)) + 0.4267 * np.exp(-(3.562e-4 * r_vol_um))
    )
    return np.minimum(r_eff_um, 45.0)


def effective_radius_plate(r_vol_um: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
    r"""
    Calculate the effective radius of contrail ice particles assuming a plate particle habit.

    Parameters
    ----------
    r_vol_um : npt.NDArray[np.float_]
        Contrail ice particle volume mean radius, [:math:`\mu m`]

    Returns
    -------
    npt.NDArray[np.float_]
        Effective radius, [:math:`\mu m`]
    """
    r_eff_um = r_vol_um * (
        0.1663 + 0.3713 * np.exp(-(0.0336 * r_vol_um)) + 0.3309 * np.exp(-(0.0035 * r_vol_um))
    )
    return np.minimum(r_eff_um, 45.0)


def effective_radius_droxtal(r_vol_um: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
    r"""
    Calculate the effective radius of contrail ice particles assuming a droxtal particle habit.

    Parameters
    ----------
    r_vol_um : npt.NDArray[np.float_]
        Contrail ice particle volume mean radius, [:math:`\mu m`]

    Returns
    -------
    npt.NDArray[np.float_]
        Effective radius, [:math:`\mu m`]
    """
    r_eff_um = 0.94 * r_vol_um
    return np.minimum(r_eff_um, 45.0)


def effective_radius_myhre(r_vol_um: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
    r"""
    Calculate the effective radius of contrail ice particles assuming a sphere particle habit.

    Parameters
    ----------
    r_vol_um : npt.NDArray[np.float_]
        Contrail ice particle volume mean radius, [:math:`\mu m`]

    Returns
    -------
    npt.NDArray[np.float_]
        Effective radius, [:math:`\mu m`]
    """
    return np.minimum(r_vol_um, 45.0)


# -----------------
# Radiative Forcing
# -----------------


def longwave_radiative_forcing(
    r_vol_um: npt.NDArray[np.float_],
    olr: npt.NDArray[np.float_],
    air_temperature: npt.NDArray[np.float_],
    tau_contrail: npt.NDArray[np.float_],
    tau_cirrus: npt.NDArray[np.float_],
    habit_weights_: npt.NDArray[np.float_],
    r_eff_um: npt.NDArray[np.float_] | None = None,
) -> npt.NDArray[np.float_]:
    r"""
    Calculate the local contrail longwave radiative forcing (:math:`RF_{LW}`).

    All returned values are positive.

    Parameters
    ----------
    r_vol_um : npt.NDArray[np.float_]
        Contrail ice particle volume mean radius, [:math:`\mu m`]
    olr : npt.NDArray[np.float_]
        Outgoing longwave radiation at each waypoint, [:math:`W m^{-2}`]
    air_temperature : npt.NDArray[np.float_]
        Ambient temperature at each waypoint, [:math:`K`]
    tau_contrail : npt.NDArray[np.float_]
        Contrail optical depth at each waypoint
    tau_cirrus : npt.NDArray[np.float_]
        Optical depth of numerical weather prediction (NWP) cirrus above the
        contrail at each waypoint
    habit_weights_ : npt.NDArray[np.float_]
        Weights to different ice particle habits for each waypoint,
        ``n_waypoints x 8`` (habit) columns, [:math:`[0 - 1]`]
    r_eff_um : npt.NDArray[np.float_], optional
        Provide effective radius corresponding to elements in ``r_vol_um``, [:math:`\mu m`].
        Defaults to None, which means the effective radius will be calculated using ``r_vol_um``
        and habit types in :func:`effective_radius_by_habit`.

    Returns
    -------
    npt.NDArray[np.float_]
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
    delta_t = rf_const.delta_t[idx1]
    delta_lc = rf_const.delta_lc[idx1]
    delta_lr = rf_const.delta_lr[idx1]
    k_t = rf_const.k_t[idx1]
    T_0 = rf_const.T_0[idx1]

    olr_h = olr[idx0]
    tau_cirrus_h = tau_cirrus[idx0]
    tau_contrail_h = tau_contrail[idx0]
    air_temperature_h = air_temperature[idx0]

    # effective radius
    if r_eff_um is None:
        r_vol_um_h = r_vol_um[idx0]
        r_eff_um_h = effective_radius_by_habit(r_vol_um_h, idx1)
    else:
        if not isinstance(r_eff_um, np.ndarray) or r_eff_um.shape != olr.shape:
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
    rf_lw_per_habit = np.maximum(rf_lw_per_habit, 0.0)

    # Weight and sum the RF contributions of each habit type according the habit weight
    # regime at the waypoint
    # see eqn (12) in :cite:`schumannParametricRadiativeForcing2012`
    # use fancy indexing to re-assign values to 2d array of waypoint x habit type
    rf_lw_weighted = np.zeros_like(habit_weights_)
    rf_lw_weighted[idx0, idx1] = rf_lw_per_habit * habit_weights_[habit_weight_mask]
    return np.sum(rf_lw_weighted, axis=1)


def shortwave_radiative_forcing(
    r_vol_um: npt.NDArray[np.float_],
    sdr: npt.NDArray[np.float_],
    rsr: npt.NDArray[np.float_],
    sd0: npt.NDArray[np.float_],
    tau_contrail: npt.NDArray[np.float_],
    tau_cirrus: npt.NDArray[np.float_],
    habit_weights_: npt.NDArray[np.float_],
    r_eff_um: npt.NDArray[np.float_] | None = None,
) -> npt.NDArray[np.float_]:
    r"""
    Calculate the local contrail shortwave radiative forcing (:math:`RF_{SW}`).

    All returned values are negative.

    Parameters
    ----------
    r_vol_um : npt.NDArray[np.float_]
        Contrail ice particle volume mean radius, [:math:`\mu m`]
    sdr : npt.NDArray[np.float_]
        Solar direct radiation, [:math:`W m^{-2}`]
    rsr : npt.NDArray[np.float_]
        Reflected solar radiation, [:math:`W m^{-2}`]
    sd0 : npt.NDArray[np.float_]
        Solar constant, [:math:`W m^{-2}`]
    tau_contrail : npt.NDArray[np.float_]
        Contrail optical depth for each waypoint
    tau_cirrus : npt.NDArray[np.float_]
        Optical depth of numerical weather prediction (NWP) cirrus above the
        contrail for each waypoint.
    habit_weights_ : npt.NDArray[np.float_]
        Weights to different ice particle habits for each waypoint,
        ``n_waypoints x 8`` (habit) columns, [:math:`[0 - 1]`]
    r_eff_um : npt.NDArray[np.float_], optional
        Provide effective radius corresponding to elements in ``r_vol_um``, [:math:`\mu m`].
        Defaults to None, which means the effective radius will be calculated using ``r_vol_um``
        and habit types in :func:`effective_radius_by_habit`.

    Returns
    -------
    npt.NDArray[np.float_]
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
    t_a = rf_const.t_a[idx1]
    A_mu = rf_const.A_mu[idx1]
    B_mu = rf_const.B_mu[idx1]
    C_mu = rf_const.C_mu[idx1]
    delta_sr = rf_const.delta_sr[idx1]
    F_r = rf_const.F_r[idx1]
    gamma_lower = rf_const.gamma_lower[idx1]
    gamma_upper = rf_const.gamma_upper[idx1]
    delta_sc = rf_const.delta_sc[idx1]
    delta_sc_aps = rf_const.delta_sc_aps[idx1]

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
        if not isinstance(r_eff_um, np.ndarray) or r_eff_um.shape != sdr.shape:
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
    rf_lw: npt.NDArray[np.float_], rf_sw: npt.NDArray[np.float_]
) -> npt.NDArray[np.float_]:
    """
    Calculate the local contrail net radiative forcing (rf_net).

    RF Net = Longwave RF (positive) + Shortwave RF (negative)

    Parameters
    ----------
    rf_lw : npt.NDArray[np.float_]
        local contrail longwave radiative forcing, [:math:`W m^{-2}`]
    rf_sw : npt.NDArray[np.float_]
        local contrail shortwave radiative forcing, [:math:`W m^{-2}`]

    Returns
    -------
    npt.NDArray[np.float_]
        local contrail net radiative forcing, [:math:`W m^{-2}`]
    """
    return rf_lw + rf_sw


def olr_reduction_natural_cirrus(
    tau_cirrus: npt.NDArray[np.float_], delta_lc: npt.NDArray[np.float_]
) -> npt.NDArray[np.float_]:
    """
    Calculate reduction in outgoing longwave radiation (OLR) due to the presence of natural cirrus.

    Natural cirrus has optical depth ``tau_cirrus`` above the contrail.
    See ``e_lw`` in Eq. (4) of Schumann et al. (2012).

    Parameters
    ----------
    tau_cirrus : npt.NDArray[np.float_]
        Optical depth of numerical weather prediction (NWP) cirrus above the
        contrail for each waypoint.
    delta_lc : npt.NDArray[np.float_]
        Habit specific parameter to approximate the reduction of the outgoing
        longwave radiation at the contrail level due to natural cirrus above the contrail.

    Returns
    -------
    npt.NDArray[np.float_]
        Reduction of outgoing longwave radiation
    """
    # e_lw calculations
    return np.exp(-delta_lc * tau_cirrus)


def contrail_effective_emissivity(
    r_eff_um: npt.NDArray[np.float_], delta_lr: npt.NDArray[np.float_]
) -> npt.NDArray[np.float_]:
    r"""Calculate the effective emissivity of the contrail, ``f_lw``.

    Refer to Eq. (3) of Schumann et al. (2012).

    Parameters
    ----------
    r_eff_um : npt.NDArray[np.float_]
        Effective radius for each waypoint, n_waypoints x 8 (habit) columns, [:math:`\mu m`]
        See :func:`effective_radius_habit`.
    delta_lr : npt.NDArray[np.float_]
        Habit specific parameter to approximate the effective emissivity of the contrail.

    Returns
    -------
    npt.NDArray[np.float_]
        Effective emissivity of the contrail
    """
    # f_lw calculations
    return 1.0 - np.exp(-delta_lr * r_eff_um)


def albedo(sdr: npt.NDArray[np.float_], rsr: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
    """
    Calculate albedo along contrail waypoint.

    Albedo, the diffuse reflection of solar radiation out of the total solar radiation,
    is computed based on the solar direct radiation (`sdr`) and reflected solar radiation (`rsr`).

    Output values range between 0 (corresponding to a black body that absorbs
    all incident radiation) and 1 (a body that reflects all incident radiation).

    Parameters
    ----------
    sdr : npt.NDArray[np.float_]
        Solar direct radiation, [:math:`W m^{-2}`]
    rsr : npt.NDArray[np.float_]
        Reflected solar radiation, [:math:`W m^{-2}`]

    Returns
    -------
    npt.NDArray[np.float_]
        Albedo value, [:math:`[0 - 1]`]
    """
    day = sdr > 0.0
    albedo_ = np.zeros(sdr.shape)
    albedo_[day] = rsr[day] / sdr[day]
    albedo_.clip(0.0, 1.0, out=albedo_)
    return albedo_


def contrail_albedo(
    tau_contrail: npt.NDArray[np.float_],
    mue: npt.NDArray[np.float_],
    r_eff_um: npt.NDArray[np.float_],
    A_mu: npt.NDArray[np.float_],
    B_mu: npt.NDArray[np.float_],
    C_mu: npt.NDArray[np.float_],
    delta_sr: npt.NDArray[np.float_],
    F_r: npt.NDArray[np.float_],
    gamma_lower: npt.NDArray[np.float_],
    gamma_upper: npt.NDArray[np.float_],
) -> npt.NDArray[np.float_]:
    r"""
    Calculate the contrail albedo, ``alpha_c``.

    Refer to Eq. (6) of Schumann et al. (2012),

    Parameters
    ----------
    tau_contrail : npt.NDArray[np.float_]
        Contrail optical depth for each waypoint
    mue : npt.NDArray[np.float_]
        Cosine of the solar zenith angle (theta), mue = cos(theta) = sdr/sd0
    r_eff_um : npt.NDArray[np.float_]
        Effective radius for each waypoint, n_waypoints x 8 (habit) columns, [:math:`\mu m`]
        See :func:`effective_radius_habit`.
    A_mu : npt.NDArray[np.float_]
        Habit-specific parameter to approximate the albedo of the contrail
    B_mu : npt.NDArray[np.float_]
        Habit-specific parameter to approximate the SZA-dependent contrail sideward scattering
    C_mu : npt.NDArray[np.float_]
        Habit-specific parameter to approximate the albedo of the contrail
    delta_sr : npt.NDArray[np.float_]
        Habit-specific parameter to approximate the effective contrail optical depth
    F_r : npt.NDArray[np.float_]
        Habit-specific parameter to approximate the effective contrail optical depth
    gamma_lower : npt.NDArray[np.float_]
        Habit-specific parameter to approximate the contrail reflectances
    gamma_upper : npt.NDArray[np.float_]
        Habit-specific parameter to approximate the contrail reflectances

    Returns
    -------
    npt.NDArray[np.float_]
        Contrail albedo for each waypoint and ice particle habit
    """
    tau_aps = tau_contrail * (1.0 - F_r * (1 - np.exp(-delta_sr * r_eff_um)))
    tau_eff = tau_aps / (mue + 1e-6)
    r_c = 1.0 - np.exp(-gamma_upper * tau_eff)
    r_c_aps = np.exp(-gamma_lower * tau_eff)

    f_mu = (2.0 * (1.0 - mue)) ** B_mu - 1.0
    return r_c * (C_mu + (A_mu * r_c_aps * f_mu))


def effective_tau_cirrus(
    tau_cirrus: npt.NDArray[np.float_],
    mue: npt.NDArray[np.float_],
    delta_sc: npt.NDArray[np.float_],
    delta_sc_aps: npt.NDArray[np.float_],
) -> npt.NDArray[np.float_]:
    r"""
    Calculate the effective optical depth of natural cirrus above the contrail, ``e_sw``.

    Refer to Eq. (11) of Schumann et al. (2012).

    Parameters
    ----------
    tau_cirrus : npt.NDArray[np.float_]
        Optical depth of numerical weather prediction (NWP) cirrus above the
        contrail for each waypoint.
    mue : npt.NDArray[np.float_]
        Cosine of the solar zenith angle (theta), mue = cos(theta) = sdr/sd0
    delta_sc : npt.NDArray[np.float_]
        Habit-specific parameter to account for the optical depth of natural
        cirrus above the contrail
    delta_sc_aps : npt.NDArray[np.float_]
        Habit-specific parameter to account for the optical depth of natural
        cirrus above the contrail

    Returns
    -------
    npt.NDArray[np.float_]
        Effective optical depth of natural cirrus above the contrail,
        ``n_waypoints x 8`` (habit) columns.
    """
    tau_cirrus_eff = tau_cirrus / (mue + 1e-6)
    return np.exp(tau_cirrus * delta_sc) - (tau_cirrus_eff * delta_sc_aps)
