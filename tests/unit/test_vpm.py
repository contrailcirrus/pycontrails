"""Tests for the VPM emissions model."""

import numpy as np
import pytest

from pycontrails import JetA
from pycontrails.models import extended_k15, sac
from pycontrails.models.humidity_scaling import humidity_scaling as hs


def test_critical_supersaturation() -> None:
    """Smoke test the ``critical_supersaturation`` function."""
    rng = np.random.default_rng(12345)
    n = 1000

    # Use physically reasonable values
    Dd = 10 ** rng.uniform(-9, -6, n)
    kappa = rng.uniform(0.0, 1.0, n)
    temperature = rng.uniform(200.0, 300.0, n)
    S_w = extended_k15.critical_supersaturation(Dd, kappa, temperature)

    assert np.all(np.isfinite(S_w))
    assert np.all(S_w > 1.0)


def test_activation_radius() -> None:
    """Smoke test the ``activation_radius`` function."""
    rng = np.random.default_rng(123456)
    n = 1000

    # Use physically reasonable values
    kappa = rng.uniform(0.0, 1.0, n)
    temperature = rng.uniform(200.0, 300.0, n)
    S_w = rng.uniform(0.5, 1.5, n)

    r_act = extended_k15.activation_radius(S_w, kappa, temperature)

    # We get nan values where S_w <= 1.0
    np.testing.assert_array_equal(np.isnan(r_act), S_w <= 1.0)

    r_act_finite = r_act[~np.isnan(r_act)]
    assert np.all(r_act_finite > 0.0)

    # Confirm the min and max are between 1e-9 and 2e-6
    assert np.min(r_act_finite) > 1e-9
    assert np.max(r_act_finite) < 2e-7


def test_droplet_apparent_emission_index() -> None:
    """Pin a value for the ``droplet_apparent_emission_index`` function."""
    aei = extended_k15.droplet_apparent_emission_index(
        specific_humidity=0.000012,
        T_ambient=218.0,
        T_exhaust=600.0,
        air_pressure=18000.0,
        nvpm_ei_n=3.0e14,
        vpm_ei_n=3.0e17,
        G=1.2,
    )
    assert aei == pytest.approx(1.76e14, rel=0.01)


@pytest.mark.parametrize(
    ("nvpm_ei_n", "vpm_ei_n", "expected_aei"),
    [(1e12, 0.0, 2.3e12), (1e15, 0.0, 7.7e14), (1e12, 1e17, 2.2e15)],
)
def test_against_ponsonby_values(nvpm_ei_n: float, vpm_ei_n: float, expected_aei: float) -> None:
    """Compare pycontrails extended_k15 model against Joel Ponsonby's values.

    See Joel's notebook here:
        https://github.com/jponvc/extended-K15-model/blob/main/extended-K15-model-notebook.ipynb

    The differences in output are due to:
    - Different implementations of some of the thermodynamics (pycontrails uses the
      ``thermo`` package, Joel uses his own functions).
    - Joel uses interpolation tables to compute r_act, whereas pycontrails
      computes these directly.
    - Differences in computing the zero of f (pycontrails uses a linear interpolation, Joel
      uses the nearest value).
    """
    T_a = 215.0  # from Joel
    p_a = 22919.20735681785  # from Joel
    q = 1.0 / hs._rhi_over_q(T_a, p_a)  # ie, RHI = 100%
    G = sac.slope_mixing_line(q, p_a, engine_efficiency=0.3, ei_h2o=JetA.ei_h2o, q_fuel=JetA.q_fuel)

    particles = [  # from Joel
        extended_k15.Particle(
            type=extended_k15.ParticleType.NVPM,
            kappa=5.0e-3,
            gmd=3.5e-8,
            gsd=2.0,
            n_ambient=0.0,
        ),
        extended_k15.Particle(
            type=extended_k15.ParticleType.VPM,
            kappa=0.5,
            gmd=2.5e-9,
            gsd=1.3,
            n_ambient=0.0,
        ),
        extended_k15.Particle(
            type=extended_k15.ParticleType.AMBIENT,
            kappa=0.5,
            gmd=3.0e-8,
            gsd=2.2,
            n_ambient=6.0e8,
        ),
    ]

    aei = extended_k15.droplet_apparent_emission_index(
        specific_humidity=q,
        T_ambient=T_a,
        T_exhaust=600.0,
        air_pressure=p_a,
        nvpm_ei_n=nvpm_ei_n,
        vpm_ei_n=vpm_ei_n,
        G=G,
        particles=particles,
    )

    assert aei == pytest.approx(expected_aei, rel=0.15)  # 15% tolerance
