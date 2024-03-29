"""Test `unterstrasser_wake_vortex` module."""

import numpy as np

from pycontrails.models.cocip.unterstrasser_wake_vortex import (
    _survival_fraction_from_length_scale,
    initial_contrail_depth,
    plume_area,
    z_atm_length_scale,
    z_desc_length_scale,
    z_emit_length_scale,
    z_total_length_scale,
)


def test_unterstrasser_wake_vortex_length_scales() -> None:
    """Test Unterstrasser (2016) wake vortex length scales using values listed in Table A2."""
    # Input parameters

    # Aircraft types: B777, CRJ, A380, B737, B747, B777, B767
    air_temperature = np.array([209.0, 217.0, 217.0, 217.0, 217.0, 217.0, 218.0])
    rhi_0 = np.array([1.0, 1.0, 1.4, 1.4, 1.2, 1.1, 1.1])
    wingspan = np.array([60.9, 21.2, 79.8, 34.4, 64.4, 60.9, 47.6])
    h2o_per_dist = np.array([15.00, 1.77, 20.03, 3.70, 13.82, 15.0, 7.26]) / 1000  # Units: kg/m**3

    # Derived parameters
    a_p = plume_area(wingspan)
    rho_emit = h2o_per_dist / a_p

    # Test `z_atm`
    z_atm_est = z_atm_length_scale(air_temperature, rhi_0)
    np.testing.assert_array_almost_equal(
        z_atm_est, [0.0, 0.0, 276.9, 276.9, 148.9, 77.6, 78.6], decimal=1
    )

    # Test `z_emit`
    z_emit_est = z_emit_length_scale(rho_emit, air_temperature)
    np.testing.assert_array_almost_equal(
        z_emit_est, [280.8, 90.3, 96.2, 83.5, 99.1, 118.7, 81.5], decimal=1
    )


def test_unterstrasser_wake_vortex_survival_fractions() -> None:
    """Test Unterstrasser (2016) ice crystal survival fraction using values listed in Table A2."""
    # Input parameters

    # Aircraft types: B777, CRJ, A380, B737, B747, B777, B767
    aei_n = np.array([2.8, 2.8, 2.8, 2.8, 2.8, 0.14, 100.0]) * 10**14
    z_atm = np.array([0.0, 0.0, 276.0, 276.0, 148.0, 77.0, 78.0])
    z_emit = np.array([279.0, 90.0, 96.0, 83.0, 98.0, 117.0, 81.0])
    z_desc = np.array([339.0, 169.0, 399.0, 349.0, 548.0, 339.0, 339.0])

    # Derived parameters
    z_total_est = z_total_length_scale(aei_n, z_atm, z_emit, z_desc)
    f_surv_est = _survival_fraction_from_length_scale(z_total_est)
    np.testing.assert_array_almost_equal(
        f_surv_est, [0.38, 0.099, 0.87, 0.88, 0.15, 0.78, 0.096], decimal=2
    )


def test_unterstrasser_initial_contrail_depth() -> None:
    """Test Unterstrasser (2016) initial contrail depth."""
    # Input parameters
    z_desc = np.array([339.0, 169.0, 399.0, 349.0, 548.0, 339.0, 339.0])
    f_surv = np.array([0.384, 0.099, 0.874, 0.884, 0.158, 0.782, 0.017])

    depth_est = initial_contrail_depth(z_desc, f_surv)
    np.testing.assert_array_almost_equal(
        depth_est, [416.2, 100.4, 519.1, 454.6, 519.5, 436.4, 34.6], decimal=1
    )


def test_unterstrasser_final_vertical_displacement() -> None:
    """Test Unterstrasser (2016) wake vortex vertical displacement.

    Unlike other tests, input and output values required for testing
    are not provided by Unterstrasser (2016). The expected values
    used here are the values produced by the function at the time
    of implementation. This test therefore serves primarily to check
    that the implementation runs without crashing, and to detect
    future changes relative to the initial implementation.
    """

    # Input parameters
    # Aircraft types: B777, CRJ, A380, B737, B747, B777, B767
    wingspan = np.array([60.9, 21.2, 79.8, 34.4, 64.4, 60.9, 47.6])  # m
    aircraft_mass = np.array([250e3, 25e3, 420e3, 65e3, 300e3, 250e3, 100e3])  # kg

    air_temperature = np.full_like(wingspan, 215.0)  # K
    air_pressure = np.full_like(wingspan, 300e2)  # Pa
    true_airspeed = np.full_like(wingspan, 235.0)  # m/s, approx mach 0.8
    dT_dz = np.full_like(wingspan, 1e-3)  # K/m, weakly stably stratified

    z_desc = z_desc_length_scale(
        wingspan, air_temperature, air_pressure, true_airspeed, aircraft_mass, dT_dz
    )
    np.testing.assert_array_almost_equal(
        z_desc, [448.7, 240.5, 508.1, 304.4, 478.0, 448.7, 321.0], decimal=1
    )
