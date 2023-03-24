"""
Radiative heating of contrail cirrus.

Contrails absorb incoming solar radiation and outgoing longwave radiation, causing it to heat up.

1. The additional heating energy drives a local updraft
   (but this is negligible and not included in CoCiP);
2. The  differential heating rate drives local turbulence.
   As radiative emission from the contrail is limited by its low temperature,
   the net result is deposition of radiative energy in the contrail.

These equations are provided by Ulrich Schumann and not included in any scientific publications,
The scientific rationale of including these effects can be found in the references below:

References
----------
- :cite:`jensenSpreadingGrowthContrails1998`
- :cite:`schumannLifeCycleIndividual2017`
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from pycontrails.physics import constants


@dataclass
class RadiativeHeatingConstants:
    """Constants/coefficients used to calculate the radiative heating rate.

    Constants/coefficients used to calculate the:
     - shortwave differential heating rate
     - longwave differential heating rate
     - shortwave heating rate
     - longwave heating rate

    These coefficients were calibrated based on a least-squares fit relative
    to the libRadtran radiative transfer model. (Ulrich Schumann, personal communication).

    References
    ----------
    - :cite:`schumannRadiativeHeatingContrail2010`
    """

    #: Coefficients for shortwave differential heating rate
    dacth: float = 0.205747e01
    dacth3: float = 0.898366e00
    dbcth: float = 0.791045e00
    dccth: float = 0.612725e00
    ddcth: float = 0.517342e-02
    dexalb: float = 0.267568e01
    dfrsw: float = 0.139286e01
    dgalbs: float = 0.178497e01
    d_gamma: float = 0.142104e01
    d_gamma_s: float = 0.882497e00
    dqsw: float = 0.631427e-01
    draddsw: float = 0.261780e00
    dtt: float = 0.339171e-01

    #: Coefficients for longwave differential heating rate
    dak: float = 0.357181e01
    dcrhi: float = 0.623019e-01
    ddelta: float = 0.198000e01
    dfrlw: float = 0.609262e00
    dqlw: float = 0.100000e-05
    dqrlw: float = 0.160286e00
    draddlw: float = 0.898529e-05
    dsigma: float = 0.159884e-06

    #: Coefficients for shortwave heating rate
    acth: float = 0.156899e01
    bcth: float = 0.875130e00
    ccth: float = 0.112445e01
    dcth: float = 0.236688e-01
    exal_b: float = 0.410705e00
    fr_sw: float = 0.537577e01
    gamma_r: float = 0.762254e00
    q_sw: float = 0.454176e-01
    radd_sw: float = 0.991554e00
    ttt: float = 0.985031e-01

    #: Coefficients for longwave heating rate
    ak: float = 0.294930e01
    crhi: float = 0.174422e00
    czlw: float = 0.393884e-01
    delta: float = 0.860746e00
    fr_lw: float = 0.760423e00
    q_lw: float = 0.152075e02
    radd_lw: float = 0.308486e-02
    sigma: float = 0.253499e-04


RAD_HEAT = RadiativeHeatingConstants()


def convective_velocity_scale(
    depth_eff: npt.NDArray[np.float_],
    eff_heat_rate: npt.NDArray[np.float_],
    air_temperature: npt.NDArray[np.float_],
) -> npt.NDArray[np.float_]:
    """
    Calculate the convective velocity scale, i.e., vertical mixing rate.

    Parameters
    ----------
    depth_eff : npt.NDArray[np.float_]
        Effective depth of the contrail plume, [:math:`m`]
    eff_heat_rate: npt.NDArray[np.float_]
        Effective heating rate, i.e., rate of which the contrail plume is heated, [:math:`K s^{-1}`]
    air_temperature : npt.NDArray[np.float_]
        Ambient temperature for each waypoint, [:math:`K`]

    Returns
    -------
    npt.NDArray[np.float_]
        Convective velocity scale, [:math:`m s^{-1}`]
    """
    return ((constants.g * depth_eff**2 * np.maximum(-eff_heat_rate, 0)) / air_temperature) ** (
        1 / 3
    )


def effective_heating_rate(
    d_heat_rate: npt.NDArray[np.float_],
    cumul_rad_heat: npt.NDArray[np.float_],
    dT_dz: npt.NDArray[np.float_],
    depth: npt.NDArray[np.float_],
) -> npt.NDArray[np.float_]:
    """Calculate effective heating rate.

    The effective heating rate accounts for the heat required to overcome the stable stratification.
    Turbulence will occur after the cumulative heating overcomes the stable stratification.

    Parameters
    ----------
    d_heat_rate: npt.NDArray[np.float_]
        Differential heating rate, i.e., rate of which the contrail
        plume is heated, [:math:`K s^{-1}`]
    cumul_rad_heat: npt.NDArray[np.float_]
        Cumulative solar and terrestrial radiative heating energy
        absorbed by the contrail, [:math:`K`]
    dT_dz: npt.NDArray[np.float_]
        Temperature gradient with respect to altitude (dz), [:math:`K m^{-1}`]
    depth : npt.NDArray[np.float_]
        Contrail depth at each waypoint, [:math:`m`]

    Returns
    -------
    npt.NDArray[np.float_]
        Effective heating rate, [:math:`K s^{-1}`]
    """
    filt = cumul_rad_heat > 0.0
    heat_denom = 0.5 * dT_dz[filt] * depth[filt]
    heat_denom.clip(min=0.0, out=heat_denom)

    heat_ratio = np.zeros_like(cumul_rad_heat)
    heat_ratio[filt] = cumul_rad_heat[filt] / (cumul_rad_heat[filt] + heat_denom)

    return d_heat_rate * heat_ratio


def differential_heating_rate(
    air_temperature: npt.NDArray[np.float_],
    rhi: npt.NDArray[np.float_],
    rho_air: npt.NDArray[np.float_],
    r_ice_vol: npt.NDArray[np.float_],
    depth_eff: npt.NDArray[np.float_],
    tau_contrail: npt.NDArray[np.float_],
    tau_cirrus: npt.NDArray[np.float_],
    sd0: npt.NDArray[np.float_],
    sdr: npt.NDArray[np.float_],
    rsr: npt.NDArray[np.float_],
    olr: npt.NDArray[np.float_],
) -> npt.NDArray[np.float_]:
    """
    Calculate the differential heating rate affecting the contrail plume.

    Differential heating rate is the heating rate difference between the upper and
    lower half of the cirrus layer. The radiative heating effect is dominated by the
    longwave component. Therefore, this output will always be a positive
    value (i.e., warmer at the upper contrail edge and cooler at the bottom).

    Parameters
    ----------
    air_temperature : npt.NDArray[np.float_]
        Ambient temperature at each waypoint, [:math:`K`]
    rhi : npt.NDArray[np.float_]
        Relative humidity with respect to ice at each waypoint
    rho_air : npt.NDArray[np.float_]
        Density of air for each waypoint, [:math:`kg m^{-3}`]
    r_ice_vol : npt.NDArray[np.float_]
        Ice particle volume mean radius, [:math:`m`]
    depth_eff : npt.NDArray[np.float_]
        Effective depth of the contrail plume, [:math:`m`]
    tau_contrail : npt.NDArray[np.float_]
        Contrail optical depth for each waypoint
    tau_cirrus : npt.NDArray[np.float_]
        Optical depth of numerical weather prediction (NWP) cirrus above the contrail
    sd0 : npt.NDArray[np.float_]
        Solar constant, [:math:`W m^{-2}`]
    sdr : npt.NDArray[np.float_]
        Solar direct radiation, [:math:`W m^{-2}`]
    rsr : npt.NDArray[np.float_]
        Reflected solar radiation, [:math:`W m^{-2}`]
    olr : npt.NDArray[np.float_]
        Outgoing longwave radiation at each waypoint, [:math:`W m^{-2}`]

    Returns
    -------
    npt.NDArray[np.float_]
        Differential heating rate, [:math:`K s^{-1}`]
    """
    r_ice_vol_um = r_ice_vol * 1e6
    cp_contrail = contrail_heat_capacity(rho_air, depth_eff)
    d_heat_rate_sw = differential_heating_rate_shortwave(
        cp_contrail, r_ice_vol_um, tau_contrail, tau_cirrus, sd0, sdr, rsr
    )
    d_heat_rate_lw = differential_heating_rate_longwave(
        air_temperature, rhi, cp_contrail, r_ice_vol_um, tau_contrail, tau_cirrus, olr
    )
    return np.minimum(d_heat_rate_sw + d_heat_rate_lw, 0.0)


def differential_heating_rate_shortwave(
    cp_contrail: npt.NDArray[np.float_],
    r_ice_vol_um: npt.NDArray[np.float_],
    tau_contrail: npt.NDArray[np.float_],
    tau_cirrus: npt.NDArray[np.float_],
    sd0: npt.NDArray[np.float_],
    sdr: npt.NDArray[np.float_],
    rsr: npt.NDArray[np.float_],
) -> npt.NDArray[np.float_]:
    r"""
    Calculate shortwave differential heating rate.

    Incoming solar radiation heats the contrail top. Therefore, this output will
    always be a positive value (i.e., warmer at the upper contrail edge and cooler
    at the bottom). This effect is approximated using a parametric model
    that is calibrated with the libRadtran radiative transfer model.

    Parameters
    ----------
    cp_contrail : npt.NDArray[np.float_]
        Contrail heat capacity per unit length and width, [:math:`J K^{-1} m^{-2}`]
    r_ice_vol_um : npt.NDArray[np.float_]
        Ice particle volume mean radius, [:math:`\mu m`]
    tau_contrail : npt.NDArray[np.float_]
        Contrail optical depth for each waypoint
    tau_cirrus : npt.NDArray[np.float_]
        Optical depth of numerical weather prediction (NWP) cirrus above the contrail
    sd0 : npt.NDArray[np.float_]
        Solar constant, [:math:`W m^{-2}`]
    sdr : npt.NDArray[np.float_]
        Solar direct radiation, [:math:`W m^{-2}`]
    rsr : npt.NDArray[np.float_]
        Reflected solar radiation, [:math:`W m^{-2}`]

    Returns
    -------
    npt.NDArray[np.float_]
        Shortwave component of the differential heating rate, [:math:`K s^{-1}`]
    """
    # short circuit if no waypoints have sdr > 0
    if not np.any(sdr > 0):
        return np.zeros_like(sdr)

    mue = np.minimum(sdr / sd0, 1.0)
    tau_eff = tau_contrail / (mue + 1.0e-6)

    return (
        (1 - np.exp(-RAD_HEAT.dqsw * r_ice_vol_um))
        * (RAD_HEAT.dtt * sdr - RAD_HEAT.ddcth * rsr)
        * tau_contrail
        * (-RAD_HEAT.d_gamma_s * tau_contrail + tau_eff)
        * (1 / cp_contrail)
        * (
            1
            - RAD_HEAT.dacth * mue
            + RAD_HEAT.d_gamma * mue**2
            + (RAD_HEAT.dacth3 - 1) * mue**3
        )
        * np.exp(RAD_HEAT.dbcth * tau_cirrus - RAD_HEAT.dccth * tau_cirrus / (mue + 1e-6))
        * np.exp(-RAD_HEAT.dgalbs * tau_contrail * (1 - mue) ** RAD_HEAT.dexalb)
        * mue**RAD_HEAT.draddsw
        * (1 + RAD_HEAT.dfrsw * (1 - mue) ** 2)
    )


def differential_heating_rate_longwave(
    air_temperature: npt.NDArray[np.float_],
    rhi: npt.NDArray[np.float_],
    cp_contrail: npt.NDArray[np.float_],
    r_ice_vol_um: npt.NDArray[np.float_],
    tau_contrail: npt.NDArray[np.float_],
    tau_cirrus: npt.NDArray[np.float_],
    olr: npt.NDArray[np.float_],
) -> npt.NDArray[np.float_]:
    r"""
    Calculate longwave differential heating rate.

    Contrails absorb outgoing longwave radiation emitted from the warm surface
    below and heats it up. Therefore, this output will always be a negative value
    (i.e., warmer at the lower contrail edge and cooler at the top). This effect
    is approximated using a parametric model that is calibrated with the
    libRadtran radiative transfer model.

    Parameters
    ----------
    air_temperature : npt.NDArray[np.float_]
        Ambient temperature at each waypoint, [:math:`K`]
    rhi : npt.NDArray[np.float_]
        Relative humidity with respect to ice at each waypoint
    cp_contrail : npt.NDArray[np.float_]
        Contrail heat capacity per unit length and width, [:math:`J K^{-1} m^{-2}`]
    r_ice_vol_um : npt.NDArray[np.float_]
        Ice particle volume mean radius, [:math:`\mu m`]
    tau_contrail : npt.NDArray[np.float_]
        Contrail optical depth for each waypoint
    tau_cirrus : npt.NDArray[np.float_]
        Optical depth of numerical weather prediction (NWP) cirrus above the contrail
    olr : npt.NDArray[np.float_]
        Outgoing longwave radiation at each waypoint, [:math:`W m^{-2}`]

    Returns
    -------
    npt.NDArray[np.float_]
        Longwave component of the differential heating rate, [:math:`K s^{-1}`]
    """
    cool = RAD_HEAT.dsigma * air_temperature**RAD_HEAT.dak
    epsc = 1 - np.exp(-RAD_HEAT.ddelta * (tau_contrail + tau_cirrus))
    return (
        -RAD_HEAT.dfrlw
        * (1 / cp_contrail)
        * (olr - cool)
        * (epsc / RAD_HEAT.ddelta)
        * tau_contrail
        * np.exp(-RAD_HEAT.dqlw * tau_cirrus)
        * np.maximum(1 - RAD_HEAT.draddlw * 10 / (r_ice_vol_um + 30), 0.0)
        * (1 - np.exp(-RAD_HEAT.dqrlw * r_ice_vol_um))
        * np.exp(-(rhi - 0.9) * RAD_HEAT.dcrhi)
    )


def heating_rate(
    air_temperature: npt.NDArray[np.float_],
    rhi: npt.NDArray[np.float_],
    rho_air: npt.NDArray[np.float_],
    r_ice_vol: npt.NDArray[np.float_],
    depth_eff: npt.NDArray[np.float_],
    tau_contrail: npt.NDArray[np.float_],
    tau_cirrus: npt.NDArray[np.float_],
    sd0: npt.NDArray[np.float_],
    sdr: npt.NDArray[np.float_],
    rsr: npt.NDArray[np.float_],
    olr: npt.NDArray[np.float_],
) -> npt.NDArray[np.float_]:
    """
    Calculate the heating rate affecting the contrail plume.

    This is the average heating rate over the contrail cirrus layer.

    Parameters
    ----------
    air_temperature : npt.NDArray[np.float_]
        Ambient temperature at each waypoint, [:math:`K`]
    rhi : npt.NDArray[np.float_]
        Relative humidity with respect to ice at each waypoint
    rho_air : npt.NDArray[np.float_]
        Density of air for each waypoint, [:math:`kg m^{-3}`]
    r_ice_vol : npt.NDArray[np.float_]
        Ice particle volume mean radius, [:math:`m`]
    depth_eff : npt.NDArray[np.float_]
        Effective depth of the contrail plume, [:math:`m`]
    tau_contrail : npt.NDArray[np.float_]
        Contrail optical depth for each waypoint
    tau_cirrus : npt.NDArray[np.float_]
        Optical depth of numerical weather prediction (NWP) cirrus above the contrail
    sd0 : npt.NDArray[np.float_]
        Solar constant, [:math:`W m^{-2}`]
    sdr : npt.NDArray[np.float_]
        Solar direct radiation, [:math:`W m^{-2}`]
    rsr : npt.NDArray[np.float_]
        Reflected solar radiation, [:math:`W m^{-2}`]
    olr : npt.NDArray[np.float_]
        Outgoing longwave radiation at each waypoint, [:math:`W m^{-2}`]

    Returns
    -------
    npt.NDArray[np.float_]
        Heating rate, [:math:`K s^{-1}`]
    """
    r_ice_vol_um = r_ice_vol * 1e6
    cp_contrail = contrail_heat_capacity(rho_air, depth_eff)
    heat_rate_sw = heating_rate_shortwave(
        cp_contrail, r_ice_vol_um, tau_contrail, tau_cirrus, sd0, sdr, rsr
    )
    heat_rate_lw = heating_rate_longwave(
        air_temperature, rhi, cp_contrail, r_ice_vol_um, tau_contrail, tau_cirrus, olr
    )
    return heat_rate_sw + heat_rate_lw


def heating_rate_shortwave(
    cp_contrail: npt.NDArray[np.float_],
    r_ice_vol_um: npt.NDArray[np.float_],
    tau_contrail: npt.NDArray[np.float_],
    tau_cirrus: npt.NDArray[np.float_],
    sd0: npt.NDArray[np.float_],
    sdr: npt.NDArray[np.float_],
    rsr: npt.NDArray[np.float_],
) -> npt.NDArray[np.float_]:
    r"""Calculate shortwave heating rate.

    Parameters
    ----------
    cp_contrail : npt.NDArray[np.float_]
        Contrail heat capacity per unit length and width, [:math:`J K^{-1} m^{-2}`]
    r_ice_vol_um : npt.NDArray[np.float_]
        Ice particle volume mean radius, [:math:`\mu m`]
    tau_contrail : npt.NDArray[np.float_]
        Contrail optical depth for each waypoint
    tau_cirrus : npt.NDArray[np.float_]
        Optical depth of numerical weather prediction (NWP) cirrus above the contrail
    sd0 : npt.NDArray[np.float_]
        Solar constant, [:math:`W m^{-2}`]
    sdr : npt.NDArray[np.float_]
        Solar direct radiation, [:math:`W m^{-2}`]
    rsr : npt.NDArray[np.float_]
        Reflected solar radiation, [:math:`W m^{-2}`]

    Returns
    -------
    npt.NDArray[np.float_]
        Shortwave component of heating rate, [:math:`K s^{-1}`]
    """
    # short circuit if no waypoints have sdr > 0
    if not np.any(sdr > 0):
        return np.zeros_like(sdr)

    mue = np.minimum(sdr / sd0, 1.0)
    tau_eff = tau_contrail / (mue + 1.0e-6)
    heat_rate_sw = (
        (1 - np.exp(-RAD_HEAT.q_sw * r_ice_vol_um))
        * (RAD_HEAT.ttt * sdr + RAD_HEAT.dcth * rsr)
        * tau_eff
        * (1 / cp_contrail)
        * (1 - RAD_HEAT.acth * mue + RAD_HEAT.gamma_r * mue**2)
        * np.exp(RAD_HEAT.bcth * tau_cirrus - RAD_HEAT.ccth * tau_cirrus / (mue + 1e-6))
        * np.exp(-tau_contrail * (1 - mue) ** 2)
        * mue
    )
    return np.maximum(heat_rate_sw, 0)


def heating_rate_longwave(
    air_temperature: npt.NDArray[np.float_],
    rhi: npt.NDArray[np.float_],
    cp_contrail: npt.NDArray[np.float_],
    r_ice_vol_um: npt.NDArray[np.float_],
    tau_contrail: npt.NDArray[np.float_],
    tau_cirrus: npt.NDArray[np.float_],
    olr: npt.NDArray[np.float_],
) -> npt.NDArray[np.float_]:
    r"""Calculate longwave heating rate.

    Parameters
    ----------
    air_temperature : npt.NDArray[np.float_]
        Ambient temperature at each waypoint, [:math:`K`]
    rhi : npt.NDArray[np.float_]
        Relative humidity with respect to ice at each waypoint
    cp_contrail : npt.NDArray[np.float_]
        Contrail heat capacity per unit length and width, [:math:`J K^{-1} m^{-2}`]
    r_ice_vol_um : npt.NDArray[np.float_]
        Ice particle volume mean radius, [:math:`\mu m`]
    tau_contrail : npt.NDArray[np.float_]
        Contrail optical depth for each waypoint
    tau_cirrus : npt.NDArray[np.float_]
        Optical depth of numerical weather prediction (NWP) cirrus above the contrail
    olr : npt.NDArray[np.float_]
        Outgoing longwave radiation at each waypoint, [:math:`W m^{-2}`]

    Returns
    -------
    npt.NDArray[np.float_]
        Longwave component of heating rate, [:math:`K s^{-1}`]
    """
    fzlw = np.exp(-(rhi - 0.9) * RAD_HEAT.czlw)
    cool = RAD_HEAT.sigma * air_temperature**RAD_HEAT.ak
    epsc = 1 - np.exp(-RAD_HEAT.delta * (tau_contrail + tau_cirrus))
    heat_rate_lw = (
        np.exp(-(rhi - 0.9) * RAD_HEAT.crhi)
        * RAD_HEAT.fr_lw
        * (1 / cp_contrail)
        * (olr / fzlw - cool * fzlw)
        * (epsc / RAD_HEAT.delta)
        * np.exp(-RAD_HEAT.q_lw * tau_cirrus)
        * (1 + RAD_HEAT.radd_lw / (r_ice_vol_um + 30))
    )
    return np.maximum(heat_rate_lw, 0)


def contrail_heat_capacity(
    rho_air: npt.NDArray[np.float_], depth_eff: npt.NDArray[np.float_]
) -> npt.NDArray[np.float_]:
    """
    Calculate contrail heat capacity per unit length and width.

    Parameters
    ----------
    rho_air : npt.NDArray[np.float_]
        density of air for each waypoint, [:math:`kg m^{-3}`]
    depth_eff: npt.NDArray[np.float_]
        Effective depth of the contrail plume, [:math:`m`]

    Returns
    -------
    npt.NDArray[np.float_]
        Contrail heat capacity per unit length and width, [:math:`J K^{-1} m^{-2}`]
    """
    return depth_eff * rho_air * constants.c_pd
