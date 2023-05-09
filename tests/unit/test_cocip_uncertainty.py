"""Test `cocip_uncertainty` module."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats.distributions import rv_frozen

from pycontrails.models.cocip import CocipParams
from pycontrails.models.cocip.cocip_uncertainty import CocipUncertaintyParams, habit_dirichlet

seeds = [None, 1, 2, 3, 5, 8]


def _get_overriden(c: CocipUncertaintyParams) -> list[str]:
    """Get overriden parameters."""
    # return [split[0] for param in c.as_dict() if len(split := param.split("_uncertainty")) == 2]
    ret = []
    for param in c.as_dict():
        split = param.split("_uncertainty")
        if len(split) == 2:
            ret.append(split[0])
    return ret


@pytest.mark.parametrize("seed", seeds)
def test_cocip_uncertainty_overrides(seed: int | None) -> None:
    """Check that `CocipUncertaintyParams` overrides values from `CocipParams`."""
    c1 = CocipParams()
    c2 = CocipUncertaintyParams(seed=seed)
    for param in _get_overriden(c2):
        assert getattr(c1, param) != getattr(c2, param)


@pytest.mark.parametrize("seed", seeds)
def test_cocip_uncertainty_ranges(seed: int | None) -> None:
    """Check that parameter values are contained in specified ranges."""
    c = CocipUncertaintyParams(seed=seed)
    for param in _get_overriden(c):
        value = getattr(c, param)
        uncertainty_param = getattr(c, f"{param}_uncertainty")
        if uncertainty_param.dist.name in ["uniform", "triang"]:
            assert value > uncertainty_param.kwds["loc"]
            assert value < uncertainty_param.kwds["loc"] + uncertainty_param.kwds["scale"]
        elif uncertainty_param.dist.name == "halfnorm":
            assert value > uncertainty_param.kwds["loc"]
        else:
            assert uncertainty_param.dist.name in ["norm", "lognorm"]


def test_uncertainty_seed() -> None:
    """Check that random value generation is reproducible via the seed parameter."""
    c1 = CocipUncertaintyParams(seed=42)
    c2 = CocipUncertaintyParams(seed=42)
    for param in _get_overriden(c1):
        assert getattr(c1, param) == getattr(c2, param)

    c3 = CocipUncertaintyParams()
    for param in _get_overriden(c1):
        assert getattr(c1, param) != getattr(c3, param)


def test_unspecified_seed() -> None:
    """Check that the default value of seed (None) gives random behavior."""
    c1 = CocipUncertaintyParams()
    c2 = CocipUncertaintyParams()
    for param in _get_overriden(c1):
        assert getattr(c1, param) != getattr(c2, param)


def test_rng_class_variable() -> None:
    """Test class variable `rng`."""
    c1 = CocipUncertaintyParams(seed=5)
    state1 = CocipUncertaintyParams.rng.__getstate__()
    c2 = CocipUncertaintyParams()
    state2 = CocipUncertaintyParams.rng.__getstate__()
    assert state1 != state2

    c3 = CocipUncertaintyParams(seed=5)
    assert state1 == CocipUncertaintyParams.rng.__getstate__()
    c4 = CocipUncertaintyParams()
    assert state2 == CocipUncertaintyParams.rng.__getstate__()

    for param in _get_overriden(c1):
        assert getattr(c1, param) == getattr(c3, param)
        assert getattr(c2, param) == getattr(c4, param)


def test_mean_agrees_with_certain_defaults() -> None:
    """Check that the uncertainty distribution mean agrees with the default `CocipParams` value."""
    c1 = CocipParams()
    c2 = CocipUncertaintyParams()
    for param, dist in c2.uncertainty_params.items():
        if param in [
            "wind_shear_enhancement_exponent",
            "initial_wake_vortex_depth",
            "sedimentation_impact_factor",
            "nvpm_ei_n_enhancement_factor",
            "rf_sw_enhancement_factor",
            "rf_lw_enhancement_factor",
        ]:
            assert dist.mean() == getattr(c1, param)
        elif param == "rhi_adj":
            assert dist.mean() == 0.97
        else:
            assert param in ["rhi_boost_exponent", "habit_distributions"]


def test_uncertainty_default_dist_within_expect_cocip_function_domain() -> None:
    """Check that default distribution fall within the domains of `Cocip` methods."""
    c = CocipUncertaintyParams(seed=1001)
    n = 1_000_000

    # initial wake vortex depth should be nonnegative
    assert np.all(c.initial_wake_vortex_depth_uncertainty.rvs(n) >= 0)

    # nvpm_ei_n_enhancement_factor should be between 0.3 and 3
    assert np.all(c.nvpm_ei_n_enhancement_factor_uncertainty.rvs(n) >= 0.3)
    assert np.all(c.nvpm_ei_n_enhancement_factor_uncertainty.rvs(n) <= 3)

    # habit_distributions should all sum to 1 and have the right shape
    assert np.all(np.round(np.sum(c.habit_distributions_uncertainty.rvs(), axis=1), 3) == 1)

    # rf_lw_enhancement_factor should be bounded near 1
    assert np.all(c.rf_lw_enhancement_factor_uncertainty.rvs(n) > 0.5)
    assert np.all(c.rf_lw_enhancement_factor_uncertainty.rvs(n) < 1.5)

    # rf_sw_enhancement_factor should be bounded near 1
    assert np.all(c.rf_sw_enhancement_factor_uncertainty.rvs(n) > 0.4)
    assert np.all(c.rf_sw_enhancement_factor_uncertainty.rvs(n) < 1.6)


def test_habit_dirichlet() -> None:
    """Test custom habit dirichlet distribution generator."""

    hd = habit_dirichlet()

    # fakes an instance of rv_frozen
    assert isinstance(hd, rv_frozen)

    # defaults to C == 96
    assert hd.C == 96

    # has a rvs() method
    habit_distributions = hd.rvs()
    assert np.all(np.round(np.sum(habit_distributions, axis=1), 4) == 1)

    # doesn't support n
    with pytest.raises(ValueError, match="only supports creating one rv"):
        hd.rvs(4)

    # only support random_state
    with pytest.raises(ValueError, match="only supports creating one rv"):
        habit_distributions = hd.rvs(size=2)

    habit_distributions = hd.rvs(size=1)
    assert habit_distributions.shape == (6, 8)

    # accessible via habit_distribution_uncertainty
    c = CocipUncertaintyParams(seed=1001)
    assert isinstance(c.habit_distributions_uncertainty, rv_frozen)
    habit_distributions = c.habit_distributions_uncertainty.rvs()
    assert np.all(np.round(np.sum(habit_distributions, axis=1), 4) == 1)

    cp = CocipParams()
    radius_threshold_um = cp.radius_threshold_um
    habits = cp.habits
    assert habit_distributions.shape == (radius_threshold_um.size + 1, habits.size)


def test_none_distribution() -> None:
    """Test when distribution is set to None."""

    params = CocipUncertaintyParams().as_dict()
    assert params["rf_lw_enhancement_factor"] != 1
    assert params["rf_sw_enhancement_factor"] != 1

    params = CocipUncertaintyParams(
        rf_lw_enhancement_factor_uncertainty=None, rf_sw_enhancement_factor_uncertainty=None
    ).as_dict()
    assert params["rf_lw_enhancement_factor"] == 1
    assert params["rf_sw_enhancement_factor"] == 1
