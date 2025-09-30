"""Support for volatile particulate matter (vPM) modeling via the extended K15 model.

See the :func:`droplet_apparent_emission_index` function for the main entry point.

A preprint is available :cite:`ponsonbyUpdatedMicrophysicalModel2025`.
"""

import dataclasses
import enum
import warnings
from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import scipy.optimize
import scipy.special

from pycontrails.physics import constants, thermo

# See upcoming Teoh et. al paper "Impact of Volatile Particulate Matter on Global Contrail
# Radiative Forcing and Mitigation Assessment" for details on these default parameters.
DEFAULT_VPM_EI_N = 2.0e17  # vPM number emissions index, [kg^-1]
DEFAULT_EXHAUST_T = 600.0  # Exhaust temperature, [K]
EXPERIMENTAL_WARNING = True


class ParticleType(enum.StrEnum):
    """Enumeration of particle types."""

    NVPM = enum.auto()
    VPM = enum.auto()
    AMBIENT = enum.auto()


@dataclasses.dataclass(frozen=True)
class Particle:
    """Representation of a particle with hygroscopic and size distribution properties.

    Parameters
    ----------
    type : ParticleType
        One of ``ParticleType.NVPM``, ``ParticleType.VPM``, or ``ParticleType.AMBIENT``.
    kappa : float
        Hygroscopicity parameter, dimensionless.
    gmd : float
        Geometric mean diameter of the lognormal size distribution, [:math:`m`].
    gsd : float
        Geometric standard deviation of the lognormal size distribution, dimensionless.
    n_ambient : float
        Ambient particle number concentration, [:math:`m^{-3}`].
        For ambient or background particles, this specifies the number
        concentration entrained in the contrail plume. For emission particles,
        this should be set to ``0.0``.

    Notes
    -----
    The parameters ``gmd`` and ``gsd`` define a lognormal size distribution.
    The hygroscopicity parameter ``kappa`` follows :cite:`pettersSingleParameterRepresentation2007`.
    """

    type: ParticleType
    kappa: float
    gmd: float
    gsd: float
    n_ambient: float

    def __post_init__(self) -> None:
        ptype = self.type
        if ptype != ParticleType.AMBIENT and self.n_ambient:
            raise ValueError(f"n_ambient must be 0 for aircraft-emitted {ptype.value} particles")
        if ptype == ParticleType.AMBIENT and self.n_ambient < 0.0:
            raise ValueError("n_ambient must be non-negative for ambient particles")


def _default_particles() -> list[Particle]:
    """Define particle types representing nvPM, vPM, and ambient particles.

    See upcoming Teoh et. al paper "Impact of Volatile Particulate Matter on Global Contrail
    Radiative Forcing and Mitigation Assessment" for details on these default parameters.
    """
    return [
        Particle(type=ParticleType.NVPM, kappa=0.005, gmd=30.0e-9, gsd=2.0, n_ambient=0.0),
        Particle(type=ParticleType.VPM, kappa=0.2, gmd=1.8e-9, gsd=1.5, n_ambient=0.0),
        Particle(type=ParticleType.AMBIENT, kappa=0.5, gmd=30.0e-9, gsd=2.3, n_ambient=600.0e6),
    ]


@dataclasses.dataclass
class DropletActivation:
    """Store the computed statistics on the water droplet activation for each particle.

    Parameters
    ----------
    particle : Particle | None
        Source particle type, or ``None`` if this is the aggregate result.
    r_act : npt.NDArray[np.floating]
        Activation radius for a given water saturation ratio and temperature, [:math:`m`].
    phi : npt.NDArray[np.floating]
        Fraction of particles that activate to form water droplets, between 0 and 1.
    n_total : npt.NDArray[np.floating]
        Total particle number concentration, [:math:`m^{-3}`].
    n_available : npt.NDArray[np.floating]
        Particle number concentration available for activation, [:math:`m^{-3}`].
    """

    particle: Particle | None
    r_act: npt.NDArray[np.floating]
    phi: npt.NDArray[np.floating]
    n_total: npt.NDArray[np.floating]
    n_available: npt.NDArray[np.floating]


def S(
    D: npt.NDArray[np.floating],
    Dd: npt.NDArray[np.floating],
    kappa: npt.NDArray[np.floating],
    A: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    """Compute the supersaturation ratio at diameter ``D``.

    Implements equation (6) in :cite:`pettersSingleParameterRepresentation2007`.

    Parameters
    ----------
    D : npt.NDArray[np.floating]
        Droplet diameter, [:math:`m`]. Should be greater than ``Dd``.
    Dd : npt.NDArray[np.floating]
        Dry particle diameter, [:math:`m`].
    kappa : npt.NDArray[np.floating]
        Hygroscopicity parameter, dimensionless.
    A : npt.NDArray[np.floating]
        Kelvin term coefficient, [:math:`m`].

    Returns
    -------
    npt.NDArray[np.floating]
        Supersaturation ratio at diameter ``D``, dimensionless.
    """
    D3 = D * D * D  # D**3, avoid power operation
    Dd3 = Dd * Dd * Dd  # Dd**3, avoid power operation
    return (D3 - Dd3) / (D3 - Dd3 * (1.0 - kappa)) * np.exp(A / D)


def _func(
    D: npt.NDArray[np.floating],
    Dd: npt.NDArray[np.floating],
    kappa: npt.NDArray[np.floating],
    A: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    """Compute a term in the derivative of ``log(S)`` with respect to ``D``.

    The full derivative of ``log(S)`` is ``_func / D^2``.
    """
    D2 = D**2
    D3 = D2 * D  # D**3, avoid power operation
    D4 = D2 * D2  # D**4,  avoid power operation
    Dd3 = Dd * Dd * Dd  # Dd**3, avoid power operation

    N = D3 - Dd3
    c = kappa * Dd3

    return (3.0 * D4 * c) / (N * (N + c)) - A


def _func_prime(
    D: npt.NDArray[np.floating],
    Dd: npt.NDArray[np.floating],
    kappa: npt.NDArray[np.floating],
    A: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    """Compute the derivative of ``_func`` with respect to D."""
    D2 = D**2
    D3 = D2 * D  # D**3, avoid power operation
    Dd3 = Dd * Dd * Dd  # Dd**3, avoid power operation
    N = D3 - Dd3
    c = kappa * Dd3

    num = 3.0 * D3 * c * (4.0 * N * (N + c) - 3.0 * D3 * (2.0 * N + c))
    den = (N * (N + c)) ** 2

    return num / den


def _newton_seed(
    Dd: npt.NDArray[np.floating],
    kappa: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    """Estimate a seed value for Newton's method to find the critical diameter.

    This is a crude approach, but it probably works well enough for common values of kappa, Dd,
    and temperature. The coefficients below were derived from fitting a linear model to
    approximate eps (defined by S(D) = (1 + eps) * Dd) as a function of log(kappa) and log(Dd).
    (Dd = 1e-9, 1e-8, 1e-7, 1e-6; kappa = 0.005, 0.05, 0.5; temperature ~= 220 K)
    """
    b0 = 12.21
    b_kappa = 0.5883
    b_Dd = 0.6319

    log_eps = b0 + b_kappa * np.log(kappa) + b_Dd * np.log(Dd)
    eps = np.exp(log_eps)
    return (1.0 + eps) * Dd


def _density_liq_water(T: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    """Calculate the density of liquid water as a function of temperature.

    The estimate below is equation (A1) in Marcolli 2020
    https://doi.org/10.5194/acp-20-3209-2020
    """
    c = [
        1864.3535,  # T^0
        -72.5821489,  # T^1
        2.5194368,  # T^2
        -0.049000203,  # T^3
        5.860253e-4,  # T^4
        -4.5055151e-6,  # T^5
        2.2616353e-8,  # T^6
        -7.3484974e-11,  # T^7
        1.4862784e-13,  # T^8
        -1.6984748e-16,  # T^9
        8.3699379e-20,  # T^10
    ]
    return np.polynomial.polynomial.polyval(T, c)


def critical_supersaturation(
    Dd: npt.NDArray[np.floating],
    kappa: npt.NDArray[np.floating],
    T: npt.NDArray[np.floating],
    tol: float = 1e-12,
    maxiter: int = 25,
) -> npt.NDArray[np.floating]:
    """Compute the critical supersaturation ratio for a given particle size.

    The critical supersaturation ratio is the maximum of the supersaturation ratio ``S(D)``
    as a function of the droplet diameter ``D`` for a given dry diameter ``Dd``.
    This maximum is found by solving for the root of the derivative of ``log(S)`` with
    respect to ``D`` using Newton's method.

    Parameters
    ----------
    Dd : npt.NDArray[np.floating]
        Dry diameter of the particle, [:math:`m`].
    kappa : npt.NDArray[np.floating]
        Hygroscopicity parameter, dimensionless. Expected to satisfy ``0 < kappa < 1``.
    T : npt.NDArray[np.floating]
        The temperature at which to compute the critical supersaturation, [:math:`K`].
    tol : float, optional
        Convergence tolerance for Newton's method, by default 1e-12.
        Should be significantly smaller than the values in ``Dd``.
    maxiter : int, optional
        Maximum number of iterations for Newton's method, by default 25.

    Returns
    -------
    npt.NDArray[np.floating]
        The critical supersaturation ratio, dimensionless.
    """
    sigma = 0.0761 - 1.55e-4 * (T + constants.absolute_zero)
    A = (4.0 * sigma * constants.M_v) / (constants.R * T * _density_liq_water(T))

    x0 = _newton_seed(Dd, kappa)
    D = scipy.optimize.newton(
        func=_func,
        x0=x0,
        fprime=_func_prime,
        args=(Dd, kappa, A),
        maxiter=maxiter,
        tol=tol,
    )
    return S(D, Dd, kappa, A)


def _geometric_bisection(
    func: Callable[[npt.NDArray[np.floating]], npt.NDArray[np.floating]],
    lo: npt.NDArray[np.floating],
    hi: npt.NDArray[np.floating],
    rtol: float,
    maxiter: int,
) -> npt.NDArray[np.floating]:
    """Find root of function func in ``[lo, hi]`` using geometric bisection.

    The arrays ``lo`` and ``hi`` must be such that ``func(lo)`` and ``func(hi)`` have
    opposite signs. These two arrays are freely modified in place during the algorithm.
    """

    f_lo = func(lo)
    f_hi = func(hi)

    out_mask = np.sign(f_lo) == np.sign(f_hi)

    for _ in range(maxiter):
        mid = np.sqrt(lo * hi)
        f_mid = func(mid)

        # Where f_mid has same sign as f_lo, move lo up; else move hi down
        mask_lo = np.sign(f_mid) == np.sign(f_lo)
        lo[mask_lo] = mid[mask_lo]
        f_lo[mask_lo] = f_mid[mask_lo]

        hi[~mask_lo] = mid[~mask_lo]
        f_hi[~mask_lo] = f_mid[~mask_lo]

        if np.all(hi / lo - 1.0 < rtol):
            break

    return np.where(out_mask, np.nan, np.sqrt(lo * hi))


def activation_radius(
    S_w: npt.NDArray[np.floating],
    kappa: npt.NDArray[np.floating] | float,
    temperature: npt.NDArray[np.floating],
    rtol: float = 1e-6,
    maxiter: int = 30,
) -> npt.NDArray[np.floating]:
    """Calculate activation radius for a given supersaturation ratio and temperature.

    The activation radius is defined as the droplet radius at which the
    critical supersaturation equals the ambient water supersaturation ``S_w``.
    Mathematically, it is the root of the equation::

        critical_supersaturation(2 * r) - S_w = 0

    Parameters
    ----------
    S_w : npt.NDArray[np.floating]
        Water saturation ratio in the aircraft plume after droplet condensation, dimensionless.
    kappa : npt.NDArray[np.floating] | float
        Hygroscopicity parameter, dimensionless. Expected to satisfy ``0 < kappa < 1``.
    temperature : npt.NDArray[np.floating]
        Temperature at which to compute the activation radius, [:math:`K`].
    rtol : float, optional
        Relative tolerance for geometric-bisection root-finding algorithm, by default 1e-6.
    maxiter : int, optional
        Maximum number of iterations for geometric-bisection root-finding algorithm, by default 30.

    Returns
    -------
    npt.NDArray[np.floating]
        The activation radius, [:math:`m`]. Entries where ``S_w <= 1.0`` return ``nan``. Only
        supersaturation ratios greater than 1.0 are physically meaningful for activation. The
        returned activation radius is the radius at which the droplet would first activate
        to form a water droplet in the emissions plume.

    """
    cond = S_w > 1.0
    S_w, kappa, temperature = np.broadcast_arrays(S_w, kappa, temperature)

    S_w_cond = S_w[cond]
    kappa_cond = kappa[cond]
    temperature_cond = temperature[cond]

    def func(r: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        # radius -> diameter
        return critical_supersaturation(2.0 * r, kappa_cond, temperature_cond) - S_w_cond

    lo = np.full_like(S_w_cond, 5e-10)
    hi = np.full_like(S_w_cond, 1e-6)

    r_act_cond = _geometric_bisection(func, lo=lo, hi=hi, rtol=rtol, maxiter=maxiter)

    out = np.full_like(S_w, np.nan)
    out[cond] = r_act_cond
    return out


def _t_plume_test_points(
    specific_humidity: npt.NDArray[np.floating],
    T_ambient: npt.NDArray[np.floating],
    air_pressure: npt.NDArray[np.floating],
    G: npt.NDArray[np.floating],
    n_points: int,
) -> npt.NDArray[np.floating]:
    """Determine test points for the plume temperature along the mixing line."""
    target_shape = (1,) * T_ambient.ndim + (-1,)
    step = 0.005

    # Initially we take a shotgun approach
    # We could use some optimization technique here as well, but it's not obviously worth it
    T_plume_test = np.arange(190.0, 300.0, step, dtype=float).reshape(target_shape)
    p_mw = thermo.water_vapor_partial_pressure_along_mixing_line(
        specific_humidity=specific_humidity[..., np.newaxis],
        air_pressure=air_pressure[..., np.newaxis],
        T_plume=T_plume_test,
        T_ambient=T_ambient[..., np.newaxis],
        G=G[..., np.newaxis],
    )
    S_mw = plume_water_saturation_ratio_no_condensation(T_plume_test, p_mw)

    # Each row of S_mw has a single maximum somewhere above 1
    # For the lower bound, take this maximum
    i_T_lb = np.nanargmax(S_mw, axis=-1, keepdims=True)
    T_lb = np.take_along_axis(T_plume_test, i_T_lb, axis=-1) - step

    # For the upper bound, take the maximum T_plume where S_mw > 1
    filt = S_mw > 1.0
    i_T_ub = np.where(filt, np.arange(T_plume_test.shape[-1]), -1).max(axis=-1, keepdims=True)
    T_ub = np.take_along_axis(T_plume_test, i_T_ub, axis=-1) + step

    # Now create n_points linearly-spaced values from T_ub down to T_lb
    # (We assume later that T_plume is sorted in descending order, so we slice [::-1])
    points = np.linspace(0.0, 1.0, n_points, dtype=float)
    return (T_lb + (T_ub - T_lb) * points)[..., ::-1]


def droplet_apparent_emission_index(
    specific_humidity: npt.NDArray[np.floating],
    T_ambient: npt.NDArray[np.floating],
    T_exhaust: npt.NDArray[np.floating],
    air_pressure: npt.NDArray[np.floating],
    nvpm_ei_n: npt.NDArray[np.floating],
    vpm_ei_n: float,
    G: npt.NDArray[np.floating],
    particles: list[Particle] | None = None,
    n_plume_points: int = 50,
) -> npt.NDArray[np.floating]:
    """Calculate the droplet apparent emissions index from nvPM, vPM and ambient particles.

    Parameters
    ----------
    specific_humidity : npt.NDArray[np.floating]
        Specific humidity at each waypoint, [:math:`kg_{H_{2}O} / kg_{air}`]
    T_ambient : npt.NDArray[np.floating]
        Ambient temperature at each waypoint, [:math:`K`]
    T_exhaust : npt.NDArray[np.floating]
        Aircraft exhaust temperature for each waypoint, [:math:`K`]
    air_pressure : npt.NDArray[np.floating]
        Pressure altitude at each waypoint, [:math:`Pa`]
    nvpm_ei_n : npt.NDArray[np.floating]
        nvPM number emissions index, [:math:`kg^{-1}`]
    vpm_ei_n : float
        vPM number emissions index, [:math:`kg^{-1}`]
    G : npt.NDArray[np.floating]
        Slope of the mixing line in a temperature-humidity diagram.
    particles : list[Particle] | None, optional
        List of particle types to consider. If ``None``, defaults to a list of
        ``Particle`` instances representing nvPM, vPM, and ambient particles.
    n_plume_points : int
        Number of points to evaluate the plume temperature along the mixing line.
        Increasing this value can improve accuracy. Values above 40 are typically
        sufficient. See the :func:`droplet_activation` for numerical considerations.

    Returns
    -------
    npt.NDArray[np.floating]
        Activated droplet apparent ice emissions index, [:math:`kg^{-1}`]

    Notes
    -----
    All input arrays must be broadcastable to the same shape. For better performance
    when evaluating multiple points or grids, it is helpful to arrange the arrays so that
    meteorological variables (``specific_humidity``, ``T_ambient``, ``air_pressure``, ``G``)
    correspond to dimension 0, while aircraft emissions (``nvpm_ei_n``, ``vpm_ei_n``) correspond
    to dimension 1. This setup allows the plume temperature calculation to be computed once
    and reused for multiple emissions values.

    """
    if EXPERIMENTAL_WARNING:
        warnings.warn(
            """This model is a minimal framework used to approximate the apparent
            emission index of contrail ice crystals in the jet regime. It does not fully
            represent the complexity of microphysical plume processes, including the
            formation and growth of vPM. Instead, vPM properties are prescribed as model
            inputs, which strongly impact model outputs. Therefore, the model should
            only be used for research purposes, together with thorough sensitivity
            analyses or explicit reference to the limitations outlined above.
            """
        )

    particles = particles or _default_particles()

    # Confirm all parameters are broadcastable
    specific_humidity, T_ambient, T_exhaust, air_pressure, G, nvpm_ei_n = np.atleast_1d(
        specific_humidity, T_ambient, T_exhaust, air_pressure, G, nvpm_ei_n
    )
    try:
        np.broadcast(specific_humidity, T_ambient, T_exhaust, air_pressure, G, nvpm_ei_n, vpm_ei_n)
    except ValueError as e:
        raise ValueError(
            "Input arrays must be broadcastable to the same shape. "
            "Check the dimensions of specific_humidity, T_ambient, T_exhaust, "
            "air_pressure, G, nvpm_ei_n, and vpm_ei_n."
        ) from e

    # Determine plume temperature limits
    T_plume = _t_plume_test_points(specific_humidity, T_ambient, air_pressure, G, n_plume_points)

    # Fixed parameters -- these could be made configurable if needed
    tau_m = 10.0e-3
    beta = 0.9
    nu_0 = 60.0
    vol_molecule_h2o = (18.0e-3 / 6.022e23) / 1000.0  # volume of a supercooled water molecule / m^3

    p_mw = thermo.water_vapor_partial_pressure_along_mixing_line(
        specific_humidity=specific_humidity[..., np.newaxis],
        air_pressure=air_pressure[..., np.newaxis],
        T_plume=T_plume,
        T_ambient=T_ambient[..., np.newaxis],
        G=G[..., np.newaxis],
    )
    S_mw = plume_water_saturation_ratio_no_condensation(T_plume, p_mw)

    dilution = plume_dilution_factor(
        T_plume=T_plume,
        T_exhaust=T_exhaust[..., np.newaxis],
        T_ambient=T_ambient[..., np.newaxis],
        tau_m=tau_m,
        beta=beta,
    )
    rho_air = thermo.rho_d(T_plume, air_pressure[..., np.newaxis])

    particle_droplets = water_droplet_activation(
        particles=particles,
        T_plume=T_plume,
        T_ambient=T_ambient[..., np.newaxis],
        nvpm_ei_n=nvpm_ei_n[..., np.newaxis],
        vpm_ei_n=vpm_ei_n,
        S_mw=S_mw,
        dilution=dilution,
        rho_air=rho_air,
        nu_0=nu_0,
    )
    particle_droplets_all = water_droplet_activation_across_all_particles(particle_droplets)
    n_w_sat = droplet_number_concentration_at_saturation(T_plume)
    b_1, b_2 = particle_growth_coefficients(
        T_plume=T_plume,
        air_pressure=air_pressure[..., np.newaxis],
        S_mw=S_mw,
        n_w_sat=n_w_sat,
        vol_molecule_h2o=vol_molecule_h2o,
    )
    P_w = water_supersaturation_production_rate(
        T_plume=T_plume,
        T_exhaust=T_exhaust[..., np.newaxis],
        T_ambient=T_ambient[..., np.newaxis],
        dilution=dilution,
        S_mw=S_mw,
        tau_m=tau_m,
        beta=beta,
    )
    kappa_w = dynamical_regime_parameter(
        particle_droplets_all.n_available, S_mw, P_w, particle_droplets_all.r_act, b_1, b_2
    )
    R_w = supersaturation_loss_rate_per_droplet(
        kappa_w, particle_droplets_all.r_act, n_w_sat, b_1, b_2, vol_molecule_h2o
    )

    return droplet_activation(
        n_available_all=particle_droplets_all.n_available,
        P_w=P_w,
        R_w=R_w,
        rho_air=rho_air,
        dilution=dilution,
        nu_0=nu_0,
    )


def plume_water_saturation_ratio_no_condensation(
    T_plume: npt.NDArray[np.floating],
    p_mw: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    """Calculate water saturation ratio in the exhaust plume without droplet condensation.

    Parameters
    ----------
    T_plume : npt.NDArray[np.floating]
        Plume temperature evolution along mixing line, [:math:`K`]
    p_mw : npt.NDArray[np.floating]
        PWater vapour partial pressure along mixing line, [:math:`Pa`]

    Returns
    -------
    npt.NDArray[np.floating]
        Water saturation ratio in the aircraft plume without droplet condensation (``S_mw``).

    References
    ----------
    Page 7894 of :cite:`karcherMicrophysicalPathwayContrail2015`.

    Notes
    -----
    - When expressed in percentage terms, ``S_mw`` is identical to relative humidity.
    - Water saturation ratio in the aircraft plume with droplet condensation (``S_w``)
    - In contrail-forming conditions, ``S_w <= S_mw`` because the supersaturation in the contrail
      plume is quenched from droplet formation and growth.
    """
    return p_mw / thermo.e_sat_liquid(T_plume)


def plume_dilution_factor(
    T_plume: npt.NDArray[np.floating],
    T_exhaust: npt.NDArray[np.floating],
    T_ambient: npt.NDArray[np.floating],
    tau_m: float,
    beta: float,
) -> npt.NDArray[np.floating]:
    """Calculate the plume dilution factor.

    Parameters
    ----------
    T_plume : npt.NDArray[np.floating]
        Plume temperature evolution along mixing line, [:math:`K`].
    T_exhaust : npt.NDArray[np.floating]
        Aircraft exhaust temperature for each waypoint, [:math:`K`].
    T_ambient : npt.NDArray[np.floating]
        Ambient temperature for each waypoint, [:math:`K`].
    tau_m : float
        Mixing timescale, i.e., the time for an exhaust volume element at the center of the
        jet plume to remain unaffected by ambient air entrainment, [:math:`s`].
    beta : float
        Plume dilution parameter, set to 0.9.

    Returns
    -------
    npt.NDArray[np.floating]
        Plume dilution factor.

    References
    ----------
    Eq. (12) of :cite:`karcherMicrophysicalPathwayContrail2015`.
    """
    t_plume = _plume_age_timescale(T_plume, T_exhaust, T_ambient, tau_m, beta)
    return np.where(t_plume > tau_m, (tau_m / t_plume) ** beta, 1.0)


def _plume_age_timescale(
    T_plume: npt.NDArray[np.floating],
    T_exhaust: npt.NDArray[np.floating],
    T_ambient: npt.NDArray[np.floating],
    tau_m: float,
    beta: float,
) -> npt.NDArray[np.floating]:
    """Calculate plume age timescale from the change in plume temperature.

    Parameters
    ----------
    T_plume : npt.NDArray[np.floating]
        Plume temperature evolution along mixing line, [:math:`K`].
    T_exhaust : npt.NDArray[np.floating]
        Aircraft exhaust temperature for each waypoint, [:math:`K`].
    T_ambient : npt.NDArray[np.floating]
        Ambient temperature for each waypoint, [:math:`K`].
    tau_m : float
        Mixing timescale, i.e., the time for an exhaust volume element at the center of the
        jet plume to remain unaffected by ambient air entrainment, [:math:`s`].
    beta : float
        Plume dilution parameter, set to 0.9.

    Returns
    -------
    npt.NDArray[np.floating]
        Plume age timescale, [:math:`s`].

    References
    ----------
    Eq. (15) of :cite:`karcherMicrophysicalPathwayContrail2015`.
    """
    ratio = (T_exhaust - T_ambient) / (T_plume - T_ambient)
    return tau_m * np.power(ratio, 1 / beta, where=ratio >= 0.0, out=np.full_like(ratio, np.nan))


def water_droplet_activation(
    particles: list[Particle],
    T_plume: npt.NDArray[np.floating],
    T_ambient: npt.NDArray[np.floating],
    nvpm_ei_n: npt.NDArray[np.floating],
    vpm_ei_n: float,
    S_mw: npt.NDArray[np.floating],
    dilution: npt.NDArray[np.floating],
    rho_air: npt.NDArray[np.floating],
    nu_0: float,
) -> list[DropletActivation]:
    """Calculate statistics on the water droplet activation for different particle types.

    Parameters
    ----------
    particles : list[Particle]
        Properties of different particles in the contrail plume.
    T_plume : npt.NDArray[np.floating]
        Plume temperature evolution along mixing line, [:math:`K`].
    T_ambient : npt.NDArray[np.floating]
        Ambient temperature for each waypoint, [:math:`K`].
    nvpm_ei_n : npt.NDArray[np.floating]
        nvPM number emissions index, [:math:`kg^{-1}`].
    vpm_ei_n : float
        vPM number emissions index, [:math:`kg^{-1}`].
    S_mw : npt.NDArray[np.floating]
        Water saturation ratio in the aircraft plume without droplet condensation.
    dilution : npt.NDArray[np.floating]
        Plume dilution factor, see :func:`plume_dilution_factor`.
    rho_air : npt.NDArray[np.floating]
        Density of air, [:math:`kg / m^{-3}`].
    nu_0 : float
        Initial mass-based plume mixing factor, i.e., air-to-fuel ratio, set to 60.0.

    Returns
    -------
    list[DropletActivation]
        Computed statistics on the water droplet activation for each particle type.
    """
    res = []

    for particle in particles:
        r_act_p = activation_radius(S_mw, particle.kappa, T_plume)
        phi_p = fraction_of_water_activated_particles(particle.gmd, particle.gsd, r_act_p)

        # Calculate total number concentration for a given particle type
        if particle.type == ParticleType.AMBIENT:
            n_total_p = entrained_ambient_droplet_number_concentration(
                particle.n_ambient, T_plume, T_ambient, dilution
            )
        elif particle.type == ParticleType.NVPM:
            n_total_p = emissions_index_to_number_concentration(nvpm_ei_n, rho_air, dilution, nu_0)
        elif particle.type == ParticleType.VPM:
            n_total_p = emissions_index_to_number_concentration(vpm_ei_n, rho_air, dilution, nu_0)
        else:
            raise ValueError("Particle type unknown")

        res_p = DropletActivation(
            particle=particle,
            r_act=r_act_p,
            phi=phi_p,
            n_total=n_total_p,
            n_available=(n_total_p * phi_p),
        )
        res.append(res_p)

    return res


def fraction_of_water_activated_particles(
    gmd: npt.NDArray[np.floating] | float,
    gsd: npt.NDArray[np.floating] | float,
    r_act: npt.NDArray[np.floating] | float,
) -> npt.NDArray[np.floating]:
    """Calculate the fraction of particles that activate to form water droplets.

    Parameters
    ----------
    gmd : npt.NDArray[np.floating] | float
        Geometric mean diameter, [:math:`m`]
    gsd : npt.NDArray[np.floating] | float
        Geometric standard deviation
    r_act : npt.NDArray[np.floating] | float
        Droplet activation threshold radius for a given supersaturation (s_w), [:math:`m`]

    Returns
    -------
    npt.NDArray[np.floating]
        Fraction of particles that activate to form water droplets (phi)

    Notes
    -----
    The cumulative distribution is estimated directly using the SciPy error function.
    """
    z = (np.log(r_act * 2.0) - np.log(gmd)) / (2.0**0.5 * np.log(gsd))
    return 0.5 - 0.5 * scipy.special.erf(z)


def entrained_ambient_droplet_number_concentration(
    n_ambient: npt.NDArray[np.floating] | float,
    T_plume: npt.NDArray[np.floating],
    T_ambient: npt.NDArray[np.floating],
    dilution: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    """Calculate ambient droplet number concentration entrained in the contrail plume.

    Parameters
    ----------
    n_ambient : npt.NDArray[np.floating] | float
        Ambient particle number concentration, [:math:`m^{-3}`].
    T_plume : npt.NDArray[np.floating]
        Plume temperature evolution along mixing line, [:math:`K`].
    T_ambient : npt.NDArray[np.floating]
        Ambient temperature for each waypoint, [:math:`K`].
    dilution : npt.NDArray[np.floating]
        Plume dilution factor.

    Returns
    -------
    npt.NDArray[np.floating]
        Ambient droplet number concentration entrained in the contrail plume, [:math:`m^{-3}`].

    References
    ----------
    Eq. (37) of :cite:`karcherMicrophysicalPathwayContrail2015` without the phi term.
    """
    return n_ambient * (T_ambient / T_plume) * (1.0 - dilution)


def emissions_index_to_number_concentration(
    number_ei: npt.NDArray[np.floating] | float,
    rho_air: npt.NDArray[np.floating],
    dilution: npt.NDArray[np.floating],
    nu_0: float,
) -> npt.NDArray[np.floating]:
    """Convert particle number emissions index to number concentration.

    Parameters
    ----------
    number_ei : npt.NDArray[np.floating] | float
        Particle number emissions index, [:math:`kg^{-1}`].
    rho_air : npt.NDArray[np.floating]
        Air density at each waypoint, [:math:`kg m^{-3}`].
    dilution : npt.NDArray[np.floating]
        Plume dilution factor.
    nu_0 : float
        Initial mass-based plume mixing factor, i.e., air-to-fuel ratio, set to 60.0.

    Returns
    -------
    npt.NDArray[np.floating]
        Particle number concentration entrained in the contrail plume, [:math:`m^{-3}`]

    References
    ----------
    Eq. (37) of :cite:`karcherMicrophysicalPathwayContrail2015` without the phi term.
    """
    return number_ei * rho_air * (dilution / nu_0)


def water_droplet_activation_across_all_particles(
    particle_droplets: list[DropletActivation],
) -> DropletActivation:
    """Calculate the total and weighted water droplet activation outputs across all particle types.

    Parameters
    ----------
    particle_droplets : list[DropletActivation]
        Computed statistics on the water droplet activation for each particle type.
        See :class:`DropletActivation` and :func:`water_droplet_activation`.

    Returns
    -------
    DropletActivation
        Total and weighted water droplet activation outputs across all particle types.

    References
    ----------
    Eq. (37) and Eq. (43) of :cite:`karcherMicrophysicalPathwayContrail2015`.
    """
    # Initialise variables
    target = particle_droplets[0].n_total
    weights_numer = np.zeros_like(target)
    weights_denom = np.zeros_like(target)
    n_total_all = np.zeros_like(target)
    n_available_all = np.zeros_like(target)

    for particle in particle_droplets:
        if not particle.particle:
            raise ValueError("Each DropletActivation must have an associated Particle.")

        weights_numer += np.nan_to_num(particle.r_act) * particle.n_available
        weights_denom += particle.n_available

        # Total particles
        n_total_all += particle.n_total
        n_available_all += particle.n_available

    # Calculate number weighted activation radius
    r_act_nw = np.divide(
        weights_numer,
        weights_denom,
        out=np.full_like(weights_numer, np.nan),
        where=weights_denom != 0.0,
    )

    return DropletActivation(
        particle=None,
        r_act=r_act_nw,
        phi=n_available_all / n_total_all,
        n_total=n_total_all,
        n_available=n_available_all,
    )


def droplet_number_concentration_at_saturation(
    T_plume: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    """Calculate water vapour concentration at saturation.

    Parameters
    ----------
    T_plume : npt.NDArray[np.floating]
        Plume temperature evolution along mixing line, [:math:`K`]

    Returns
    -------
    npt.NDArray[np.floating]
        Water vapour concentration at water saturated conditions, [:math:`m^{-3}`]

    Notes
    -----
    - This is approximated based on the ideal gas law: p V = N k_b T, so (N/v) = p / (k_b * T).
    """
    k_b = 1.381e-23  # Boltzmann constant in m^2 kg s^-2 K^-1
    return thermo.e_sat_liquid(T_plume) / (k_b * T_plume)


def particle_growth_coefficients(
    T_plume: npt.NDArray[np.floating],
    air_pressure: npt.NDArray[np.floating],
    S_mw: npt.NDArray[np.floating],
    n_w_sat: npt.NDArray[np.floating],
    vol_molecule_h2o: float,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Calculate particle growth coefficients, ``b_1`` and ``b_2`` in Karcher et al. (2015).

    Parameters
    ----------
    T_plume : npt.NDArray[np.floating]
        Plume temperature evolution along mixing line, [:math:`K`]
    air_pressure : npt.NDArray[np.floating]
        Pressure altitude at each waypoint, [:math:`Pa`]
    S_mw : npt.NDArray[np.floating]
        Water saturation ratio in the aircraft plume without droplet condensation
    n_w_sat : npt.NDArray[np.floating]
        Droplet number concentration at water saturated conditions, [:math:`m^{-3}`]
    vol_molecule_h2o : float
        Volume of a supercooled water molecule, [:math:`m^{3}`]

    Returns
    -------
    tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]
        Particle growth coefficients ``b_1`` and ``b_2``, [:math:`m s^{-1}`]

    References
    ----------
    - ``b_1`` equation is below Eq. (48) of :cite:`karcherMicrophysicalPathwayContrail2015`.
    - ``b_2`` equation is below Eq. (34) of :cite:`karcherMicrophysicalPathwayContrail2015`.
    """
    r_g = 8.3145  # Global gas constant in m^3 Pa mol^-1 K^-1
    m_w = 18.0e-3  # Molar mass of water in kg mol^-1

    # Calculate `v_thermal_h2o`, mean thermal velocity of water molecule / m/s
    v_thermal_h2o = np.sqrt((8.0 * r_g * T_plume) / (np.pi * m_w))

    # Calculate `b_1`
    b_1 = (vol_molecule_h2o * v_thermal_h2o * (S_mw - 1.0) * n_w_sat) / 4.0

    # Calculate `b_2`
    d_h2o = _water_vapor_molecular_diffusion_coefficient(T_plume, air_pressure)
    b_2 = v_thermal_h2o / (4.0 * d_h2o)

    return b_1, b_2


def _water_vapor_molecular_diffusion_coefficient(
    T_plume: npt.NDArray[np.floating],
    air_pressure: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    """Calculate water vapor molecular diffusion coefficient.

    Parameters
    ----------
    T_plume : npt.NDArray[np.floating]
        Plume temperature evolution along mixing line, [:math:`K`]
    air_pressure : npt.NDArray[np.floating]
        Pressure altitude at each waypoint, [:math:`Pa`]

    Returns
    -------
    npt.NDArray[np.floating]
        Water vapor molecular diffusion coefficient

    References
    ----------
    Rogers & Yau: A Short Course in Cloud Physics
    """
    return (
        0.211
        * (T_plume / (-constants.absolute_zero)) ** 1.94
        * (constants.p_surface / air_pressure)
        * 1e-4
    )


def water_supersaturation_production_rate(
    T_plume: npt.NDArray[np.floating],
    T_exhaust: npt.NDArray[np.floating],
    T_ambient: npt.NDArray[np.floating],
    dilution: npt.NDArray[np.floating],
    S_mw: npt.NDArray[np.floating],
    tau_m: float,
    beta: float,
) -> npt.NDArray[np.floating]:
    """Calculate water supersaturation production rate.

    Parameters
    ----------
    T_plume : npt.NDArray[np.floating]
        Plume temperature evolution along mixing line, [:math:`K`]
    T_exhaust : npt.NDArray[np.floating]
        Aircraft exhaust temperature for each waypoint, [:math:`K`]
    T_ambient : npt.NDArray[np.floating]
        Ambient temperature for each waypoint, [:math:`K`]
    dilution : npt.NDArray[np.floating]
        Plume dilution factor, see `plume_dilution_factor`
    S_mw : npt.NDArray[np.floating]
        Water saturation ratio in the aircraft plume without droplet condensation
    tau_m : float
        Mixing timescale, i.e., the time for an exhaust volume element at the center of the
        jet plume to remain unaffected by ambient air entrainment, [:math:`s`]
    beta : float
        Plume dilution parameter, set to 0.9

    Returns
    -------
    npt.NDArray[np.floating]
        Water supersaturation production rate (P_w = dS_mw/dt), [:math:`s^{-1}`]
    """
    dT_dt = _plume_cooling_rate(T_exhaust, T_ambient, dilution, tau_m, beta)
    dS_mw_dT = np.gradient(S_mw, axis=-1) / np.gradient(T_plume, axis=-1)
    return dS_mw_dT * dT_dt


def _plume_cooling_rate(
    T_exhaust: npt.NDArray[np.floating],
    T_ambient: npt.NDArray[np.floating],
    dilution: npt.NDArray[np.floating],
    tau_m: float,
    beta: float,
) -> npt.NDArray[np.floating]:
    """
    Calculate plume cooling rate.

    Parameters
    ----------
    T_exhaust : npt.NDArray[np.floating]
        Aircraft exhaust temperature for each waypoint, [:math:`K`]
    T_ambient : npt.NDArray[np.floating]
        Ambient temperature for each waypoint, [:math:`K`]
    dilution : npt.NDArray[np.floating]
        Plume dilution factor, see `plume_dilution_factor`
    tau_m : float
        Mixing timescale, i.e., the time for an exhaust volume element at the center of the
        jet plume to remain unaffected by ambient air entrainment, [:math:`s`]
    beta : float
        Plume dilution parameter, set to 0.9

    Returns
    -------
    npt.NDArray[np.floating]
        Plume cooling rate (dT_dt), [:math:`K s^{-1}`]

    References
    ----------
    Eq. (14) of :cite:`karcherMicrophysicalPathwayContrail2015`.
    """
    return -beta * ((T_exhaust - T_ambient) / tau_m) * dilution ** (1.0 + 1.0 / beta)


def dynamical_regime_parameter(
    n_available_all: npt.NDArray[np.floating],
    S_mw: npt.NDArray[np.floating],
    P_w: npt.NDArray[np.floating],
    r_act_nw: npt.NDArray[np.floating],
    b_1: npt.NDArray[np.floating],
    b_2: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    """Calculate dynamical regime parameter.

    Parameters
    ----------
    n_available_all : npt.NDArray[np.floating]
        Particle number concentration that can be activated across all particles, [:math:`m^{-3}`]
    S_mw : npt.NDArray[np.floating]
        Water saturation ratio in the aircraft plume without droplet condensation
    P_w : npt.NDArray[np.floating]
        Water supersaturation production rate (P_w = dS_mw/dt), [:math:`s^{-1}`]
    r_act_nw : npt.NDArray[np.floating]
        Number-weighted droplet activation radius, [:math:`m`]
    b_1 : npt.NDArray[np.floating]
        Particle growth coefficient, [:math:`m s^{-1}`]
    b_2 : npt.NDArray[np.floating]
        Particle growth coefficient, [:math:`m s^{-1}`]

    Returns
    -------
    npt.NDArray[np.floating]
        Dynamical regime parameter (kappa_w)

    References
    ----------
    Eq. (49) of :cite:`karcherMicrophysicalPathwayContrail2015`.
    """
    tau_act = _droplet_activation_timescale(n_available_all, S_mw, P_w)
    tau_gw = _droplet_growth_timescale(r_act_nw, b_1, b_2)
    kappa_w = ((2.0 * b_2 * r_act_nw) / (1.0 + b_2 * r_act_nw)) * (tau_act / tau_gw)
    kappa_w[kappa_w <= 0.0] = np.nan
    return kappa_w


def _droplet_activation_timescale(
    n_available_all: npt.NDArray[np.floating],
    S_mw: npt.NDArray[np.floating],
    P_w: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    """Calculate water droplet activation timescale.

    Parameters
    ----------
    n_available_all : npt.NDArray[np.floating]
        Particle number concentration that can be activated across all particles, [:math:`m^{-3}`]
    S_mw : npt.NDArray[np.floating]
        Water saturation ratio in the aircraft plume without droplet condensation
    P_w : npt.NDArray[np.floating]
        Water supersaturation production rate (P_w = dS_mw/dt), [:math:`s^{-1}`]

    Returns
    -------
    npt.NDArray[np.floating]
        Water droplet activation timescale (tau_act), [:math:`s`]

    References
    ----------
    Eq. (47) of :cite:`karcherMicrophysicalPathwayContrail2015`.
    """
    dln_nw_ds_w = np.gradient(np.log(n_available_all), axis=-1) / np.gradient(S_mw - 1.0, axis=-1)
    return 1.0 / (P_w * dln_nw_ds_w)


def _droplet_growth_timescale(
    r_act_nw: npt.NDArray[np.floating],
    b_1: npt.NDArray[np.floating],
    b_2: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    """
    Calculate water droplet growth timescale.

    Parameters
    ----------
    r_act_nw : npt.NDArray[np.floating]
        Number-weighted droplet activation radius, [:math:`m`]
    b_1 : npt.NDArray[np.floating]
        Particle growth coefficient, [:math:`m s^{-1}`]
    b_2 : npt.NDArray[np.floating]
        Particle growth coefficient, [:math:`m s^{-1}`]

    Returns
    -------
    npt.NDArray[np.floating]
        Water droplet growth timescale (tau_gw), [:math:`s`]

    References
    ----------
    Eq. (48) of :cite:`karcherMicrophysicalPathwayContrail2015`.
    """
    return (1.0 + b_2 * r_act_nw) * (r_act_nw / b_1)


def supersaturation_loss_rate_per_droplet(
    kappa_w: npt.NDArray[np.floating],
    r_act_nw: npt.NDArray[np.floating],
    n_w_sat: npt.NDArray[np.floating],
    b_1: npt.NDArray[np.floating],
    b_2: npt.NDArray[np.floating],
    vol_molecule_h2o: float,
) -> npt.NDArray[np.floating]:
    """
    Calculate supersaturation loss rate per droplet.

    Parameters
    ----------
    kappa_w : npt.NDArray[np.floating]
        Dynamical regime parameter. See `dynamical_regime_parameter`
    r_act_nw : npt.NDArray[np.floating]
        Number-weighted droplet activation radius, [:math:`m`]
    n_w_sat : npt.NDArray[np.floating]
        Droplet number concentration at water saturated conditions, [:math:`m^{-3}`]
    b_1 : npt.NDArray[np.floating]
        Particle growth coefficient, [:math:`m s^{-1}`]
    b_2 : npt.NDArray[np.floating]
        Particle growth coefficient, [:math:`m s^{-1}`]
    vol_molecule_h2o : float
        Volume of a supercooled water molecule, [:math:`m^{3}`]

    Returns
    -------
    npt.NDArray[np.floating]
        Supersaturation loss rate per droplet (R_w), [:math:`m^{3} s^{-1}`]

    Notes
    -----
    Originally calculated using Eq. (50) of :cite:`karcherMicrophysicalPathwayContrail2015`,
    but has been updated in :cite:`ponsonbyUpdatedMicrophysicalModel2025` and is now calculated
    using Eq. (6) and Eq. (7) of :cite:`karcherPhysicallyBasedParameterization2006`.
    """
    delta = b_2 * r_act_nw
    f_kappa = (3.0 * np.sqrt(kappa_w)) / (
        2.0 * np.sqrt(1.0 / kappa_w) + np.sqrt((1.0 / kappa_w) + (9.0 / np.pi))
    )

    c_1 = (4.0 * np.pi * b_1) / (vol_molecule_h2o * n_w_sat * b_2**2)
    c_2 = delta**2 / (1.0 + delta)
    c_3 = (
        1.0
        - (1.0 / (delta**2))
        + (1.0 / (delta**2)) * (((1.0 + delta) ** 2 / 2.0 + (1.0 / kappa_w)) * f_kappa)
    )
    return c_1 * c_2 * c_3


def droplet_activation(
    n_available_all: npt.NDArray[np.floating],
    P_w: npt.NDArray[np.floating],
    R_w: npt.NDArray[np.floating],
    rho_air: npt.NDArray[np.floating],
    dilution: npt.NDArray[np.floating],
    nu_0: float,
) -> npt.NDArray[np.floating]:
    """
    Calculate available particles that activate to form water droplets.

    Parameters
    ----------
    n_available_all : npt.NDArray[np.floating]
        Particle number concentration entrained in the contrail plume, [:math:`m^{-3}`]
    P_w : npt.NDArray[np.floating]
        Water supersaturation production rate (P_w = dS_mw/dt), [:math:`s^{-1}`]
    R_w : npt.NDArray[np.floating]
        Supersaturation loss rate per droplet (R_w), [:math:`m^{3} s^{-1}`]
    rho_air : npt.NDArray[np.floating]
        Air density at each waypoint, [:math:`kg m^{-3}`]
    dilution : npt.NDArray[np.floating]
        Plume dilution factor, see `plume_dilution_factor`
    nu_0 : float
        Initial mass-based plume mixing factor, i.e., air-to-fuel ratio, set to 60.0.

    Returns
    -------
    npt.NDArray[np.floating]
        Activated droplet apparent emissions index, [:math:`kg^{-1}`]

    References
    ----------
    n_2_w -> Eq. (51) of :cite:`karcherMicrophysicalPathwayContrail2015`.
    f -> Eq. (44) of :cite:`karcherMicrophysicalPathwayContrail2015`.
    """
    # Droplet number concentration required to cause supersaturation relaxation at a given `S_w`
    n_2_w = P_w / R_w

    # Calculate the droplet activation that is required to quench the plume supersaturation
    # We will be seeking a root of f := n_available_all - n_2_w, but it's slightly more
    # economic to work in the log-space (ie, we can get by with fewer T_plume points).
    f = np.log(n_available_all) - np.log(n_2_w)

    # For some rows, f never changes sign. This is the case when the T_plume range
    # does not bracket the zero crossing (for example, if T_plume is too coarse).
    # It's also common that f never attains a positive value when the ambient temperature
    # is very close to the SAC T_critical saturation unless T_plume is extremely fine.
    # In this case, n_available_all is essentially constant near the zero crossing,
    # and we can just take the last value of n_available_all. In the code below,
    # we fill any nans in the case that attains_positive is False, but we propagate
    # nans when attains_negative is False.
    attains_positive = np.any(f > 0.0, axis=-1)
    attains_negative = np.any(f < 0.0, axis=-1)
    if not attains_negative.all():
        n_failures = np.sum(~attains_negative)
        warnings.warn(
            f"{n_failures} profiles never attain negative f values, so a zero crossing "
            "cannot be found. Increase the range of T_plume by setting a higher "
            "'n_plume_points' value.",
        )

    # Find the first positive value, then interpolate to estimate the fractional index
    # at which the zero crossing occurs.
    i1 = np.argmax(f > 0.0, axis=-1, keepdims=True)
    i0 = i1 - 1
    val1 = np.take_along_axis(f, i1, axis=-1)
    val0 = np.take_along_axis(f, i0, axis=-1)
    dist = val0 / (val0 - val1)

    # When f never attains a positive value, set i0 and i1 to last negative value
    # If f never attains a negative value, we pass through a nan
    cond = attains_negative & ~attains_positive
    last_negative = np.nanargmax(f[cond], axis=-1, keepdims=True)
    i0[cond] = last_negative
    i1[cond] = last_negative
    dist[cond] = 0.0

    # Extract properties at the point where the supersaturation is quenched by interpolating
    n_activated_w0 = np.take_along_axis(n_available_all, i0, axis=-1)
    n_activated_w1 = np.take_along_axis(n_available_all, i1, axis=-1)
    n_activated_w = (n_activated_w0 + dist * (n_activated_w1 - n_activated_w0))[..., 0]

    rho_air_w0 = np.take_along_axis(rho_air, i0, axis=-1)
    rho_air_w1 = np.take_along_axis(rho_air, i1, axis=-1)
    rho_air_w = (rho_air_w0 + dist * (rho_air_w1 - rho_air_w0))[..., 0]

    dilution_w0 = np.take_along_axis(dilution, i0, axis=-1)
    dilution_w1 = np.take_along_axis(dilution, i1, axis=-1)
    dilution_w = (dilution_w0 + dist * (dilution_w1 - dilution_w0))[..., 0]

    out = number_concentration_to_emissions_index(n_activated_w, rho_air_w, dilution_w, nu_0=nu_0)
    out[~attains_negative] = np.nan

    return out


def number_concentration_to_emissions_index(
    n_conc: npt.NDArray[np.floating],
    rho_air: npt.NDArray[np.floating],
    dilution: npt.NDArray[np.floating],
    nu_0: float,
) -> npt.NDArray[np.floating]:
    """
    Convert particle number concentration to apparent emissions index.

    Parameters
    ----------
    n_conc : npt.NDArray[np.floating]
        Particle number concentration entrained in the contrail plume, [:math:`m^{-3}`]
    rho_air : npt.NDArray[np.floating]
        Air density at each waypoint, [:math:`kg m^{-3}`]
    dilution : npt.NDArray[np.floating]
        Plume dilution factor, see `plume_dilution_factor`
    nu_0 : float
        Initial mass-based plume mixing factor, i.e., air-to-fuel ratio, set to 60.0.

    Returns
    -------
    npt.NDArray[np.floating]
        Particle apparent number emissions index, [:math:`kg^{-1}`]
    """
    return (n_conc * nu_0) / (rho_air * dilution)
