"""Tests for the VPM emissions model."""

import numpy as np
import pytest

from pycontrails.models import extended_k15


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
    assert aei == pytest.approx(1.81e14, rel=0.01)
