"""Test `radiative_forcing.py` module."""

import numpy as np
import pytest

from pycontrails.models.cocip import CocipParams, radiative_forcing
from pycontrails.models.cocip.radiative_forcing import RF_CONST_S2012, RFConstantsS2012
from pycontrails.physics import geo


def test_rf_const() -> None:
    """Test the RFConstant types."""

    assert isinstance(RF_CONST_S2012, RFConstantsS2012)

    # the indexes of the habit_weights should be the (radius_threshold x habits)
    # includes radius < rf_const.radius_threshold_um[0] as the 0th row index
    cp = CocipParams()
    assert cp.habit_distributions.shape == (cp.radius_threshold_um.size + 1, cp.habits.size)

    # make sure habit specific attributes are the same length as the habits
    for attr, val in RFConstantsS2012.__dict__.items():
        if not attr.startswith("_"):
            assert isinstance(val, np.ndarray)
            assert val.size == cp.habits.size


def test_habit_weights() -> None:
    """Test habit weight classification."""

    r_vol_um = np.linspace(1, 500, 100)
    cp = CocipParams()
    habit_distributions = cp.habit_distributions
    radius_threshold_um = cp.radius_threshold_um
    habits = cp.habits

    # calculate habit weight distribution regime based on particle size
    regimes = radiative_forcing.habit_weight_regime_idx(r_vol_um, radius_threshold_um)

    # must correspond to index in `habits`
    assert (regimes < 7).all()

    # should all be increasing as the particle size monotonically increases
    assert (np.diff(regimes) >= 0).all()

    # habit_weights should provide a weight value for each waypoint
    weights = radiative_forcing.habit_weights(r_vol_um, habit_distributions, radius_threshold_um)
    assert weights.shape == (r_vol_um.size, habits.size)


def test_effective_radius() -> None:
    """Test effective radius calculations."""

    r_vol_um = np.linspace(1, 350, 100)
    cp = CocipParams()
    habit_distributions = cp.habit_distributions
    radius_threshold_um = cp.radius_threshold_um

    weights = radiative_forcing.habit_weights(r_vol_um, habit_distributions, radius_threshold_um)
    habit_weight_idxs = np.where(weights > 0)
    r_vol_um_h = r_vol_um[habit_weight_idxs[0]]
    habit_types_idx = habit_weight_idxs[1]

    r_eff_um = radiative_forcing.effective_radius_by_habit(r_vol_um_h, habit_types_idx)

    # all effective radii should be > 0 and with 0.5 and 1.5 of particle size
    # TODO: many values saturate at 45um?
    assert np.all(r_eff_um > 0)
    assert np.all(r_eff_um < 1.5 * r_vol_um_h)
    assert np.all((r_eff_um > 0.4 * r_vol_um_h) | (r_eff_um == 45))


def test_override_effective_radius() -> None:
    """Allow override of r_eff_um in methods."""

    r_vol_um = np.linspace(1, 350, 100)
    r_eff_um = r_vol_um * 0.9

    cp = CocipParams()

    habit_distributions = cp.habit_distributions.astype("float64")
    radius_threshold_um = cp.radius_threshold_um.astype("float64")

    # either floats or the same shape as r_vol_um
    latitude = np.full(r_vol_um.shape, 43)
    longitude = np.full(r_vol_um.shape, -73)
    time = np.full(r_vol_um.shape, np.datetime64("2022-04-01 12:00:00"))

    air_temperature = np.full(r_vol_um.shape, 235)
    tau_contrail = np.full(r_vol_um.shape, 1)  # 0 - 3
    tau_cirrus = np.full(r_vol_um.shape, 0)  # 0 - 10

    top_net_solar_radiation = np.full(r_vol_um.shape, 100)
    top_net_thermal_radiation = np.full(r_vol_um.shape, -200)

    # calculate instantaneous theoretical solar direct radiation based on geo position and time
    theoretical_sdr = geo.solar_direct_radiation(longitude, latitude, time)
    theta_rad = geo.orbital_position(time)
    sd0 = geo.solar_constant(theta_rad)

    # calculate the average reflected shortwave radiation from theoretical sdr and tnsr
    rsr = np.maximum(theoretical_sdr - top_net_solar_radiation, 0)

    # calculate the average outgoing longwave radiation from tnsr
    olr = -1 * top_net_thermal_radiation

    # radiation calculations
    habit_weights = radiative_forcing.habit_weights(
        r_vol_um, habit_distributions, radius_threshold_um
    )

    rf_lw = radiative_forcing.longwave_radiative_forcing(
        r_vol_um, olr, air_temperature, tau_contrail, tau_cirrus, habit_weights
    )
    rf_lw_2 = radiative_forcing.longwave_radiative_forcing(
        r_vol_um, olr, air_temperature, tau_contrail, tau_cirrus, habit_weights, r_eff_um=r_eff_um
    )

    rf_sw = radiative_forcing.shortwave_radiative_forcing(
        r_vol_um, theoretical_sdr, rsr, sd0, tau_contrail, tau_cirrus, habit_weights
    )
    rf_sw_2 = radiative_forcing.shortwave_radiative_forcing(
        r_vol_um,
        theoretical_sdr,
        rsr,
        sd0,
        tau_contrail,
        tau_cirrus,
        habit_weights,
        r_eff_um=r_eff_um,
    )

    # values shouldn't be the same, but should only be off by < 1
    # FIXME: I think some value differences are just floating point errors
    assert (rf_lw != rf_lw_2).all()
    assert (np.abs(rf_lw - rf_lw_2) < 1).all()

    # values shouldn't be the same, but should only be off by < 1
    assert (rf_sw != rf_sw_2).all()
    assert (np.abs(rf_sw - rf_sw_2) < 1).all()


def test_override_habit_distributions() -> None:
    """Test override habit distributions."""
    r_vol_um = np.linspace(1, 500, 100)

    cp = CocipParams()

    habit_distributions = cp.habit_distributions
    radius_threshold_um = cp.radius_threshold_um
    habits = cp.habits

    habit_distributions[1, 1] = 0.0
    habit_distributions[1, 2] = 0.3

    # radiation calculations
    habit_weights = radiative_forcing.habit_weights(
        r_vol_um, habit_distributions, radius_threshold_um
    )
    assert habit_weights.shape == (r_vol_um.size, habits.size)
    assert habit_weights[1, 2] == pytest.approx(0.3)
    assert habit_weights[1, 1] != 0.3

    habit_distributions[1, 2] = 1
    with pytest.raises(ValueError, match="distributions must sum to 1"):
        radiative_forcing.habit_weights(r_vol_um, habit_distributions, radius_threshold_um)

    habit_distributions[1, 2] = 0.3
    with pytest.raises(ValueError, match="number of rows in"):
        radiative_forcing.habit_weights(r_vol_um, habit_distributions, np.array([5, 10]))


def test_parametric_rf_model_v2():
    """Test behaviour of parametric RF model V1 and V2."""
    cp = CocipParams()
    habit_distributions = cp.habit_distributions.astype("float64")
    radius_threshold_um = cp.radius_threshold_um.astype("float64")

    tau_cirrus = np.linspace(0.0, 5.0, 11)
    air_temperature = np.full(tau_cirrus.shape, 210.0)
    r_vol_um = np.full(tau_cirrus.shape, 16.0)
    tau_contrail = np.full(tau_cirrus.shape, 0.75)
    sd0 = np.full(tau_cirrus.shape, 1365.0)
    sdr = np.full(tau_cirrus.shape, 300.0)
    rsr = np.full(tau_cirrus.shape, 180.0)
    olr = np.full(tau_cirrus.shape, 130.0)

    habit_weights = radiative_forcing.habit_weights(
        r_vol_um, habit_distributions, radius_threshold_um
    )

    # Calculate longwave radiative forcing
    rf_lw_v1 = radiative_forcing.longwave_radiative_forcing(
        r_vol_um, olr, air_temperature, tau_contrail, tau_cirrus, habit_weights
    )

    rf_lw_v2 = radiative_forcing.longwave_radiative_forcing_s2025(
        r_vol_um, olr, air_temperature, tau_contrail, tau_cirrus, habit_weights
    )

    # Based on Figure 8 of Schumann et al. (2025), https://doi.org/10.5194/acp-25-18571-2025:
    # (i) `rf_lw_v2` is approximately equal to `rf_lw_v1`, while
    # (i) `rf_lw_v2` should be larger than `rf_lw_v1` when tau_cirrus is small, and
    # (ii) `rf_lw_v2` should be smaller than `rf_lw_v1` when tau_cirrus is large
    rf_lw_v1_actual = np.array(
        [9.657, 9.260, 8.881, 8.519, 8.173, 7.843, 7.527, 7.225, 6.937, 6.661, 6.397]
    )
    rf_lw_v2_actual = np.array(
        [10.555, 9.491, 8.536, 7.679, 6.909, 6.217, 5.595, 5.037, 4.535, 4.084, 3.678]
    )

    np.testing.assert_array_almost_equal(rf_lw_v1, rf_lw_v1_actual, decimal=3)
    np.testing.assert_array_almost_equal(rf_lw_v2, rf_lw_v2_actual, decimal=3)

    # Calculate shortwave radiative forcing
    rf_sw_v1 = radiative_forcing.shortwave_radiative_forcing(
        r_vol_um, sdr, rsr, sd0, tau_contrail, tau_cirrus, habit_weights
    )

    rf_sw_v2 = radiative_forcing.shortwave_radiative_forcing_s2025(
        r_vol_um, sdr, rsr, sd0, tau_contrail, tau_cirrus, habit_weights
    )

    # Based on Figure 8 of Schumann et al. (2025), https://doi.org/10.5194/acp-25-18571-2025:
    # The change in `rf_sw` between V1 and V2 is smaller than `rf_lw`
    rf_sw_v1_actual = np.array(
        [-12.258, -9.496, -7.359, -5.706, -4.425, -3.434, -2.665, -2.070, -1.608, -1.250, -0.972]
    )
    rf_sw_v2_actual = np.array(
        [-12.778, -9.934, -7.726, -6.011, -4.678, -3.642, -2.836, -2.210, -1.722, -1.343, -1.047]
    )

    np.testing.assert_array_almost_equal(rf_sw_v1, rf_sw_v1_actual, decimal=3)
    np.testing.assert_array_almost_equal(rf_sw_v2, rf_sw_v2_actual, decimal=3)
