"""Test thermo formulas as needed for SAC calculations."""


import numpy as np
import pytest

from pycontrails import JetA, VectorDataset
from pycontrails.models.sac import T_sat_liquid, rh_critical_sac, slope_mixing_line
from pycontrails.physics import thermo


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(4321)
    n = 100000

    v = VectorDataset()
    v["air_pressure"] = rng.uniform(5000, 50000, n)
    v["specific_humidity"] = rng.uniform(0, 0.003, n)
    v["air_temperature"] = rng.uniform(150, 250, n)
    return v


@pytest.mark.parametrize("engine_efficiency", [0.2, 0.3, 0.4, 0.5, 0.6])
def test_slope_mixing_line(data: VectorDataset, engine_efficiency: float):
    """Test the range of the slope_mixing_line formula."""
    fuel = JetA()
    G = slope_mixing_line(
        data["specific_humidity"], data["air_pressure"], engine_efficiency, fuel.ei_h2o, fuel.q_fuel
    )

    # For many calculations, we evaluate log(G - 0.053)
    # Consequently, we want G to be bounded above this constant
    # We use a small buffer to quantify this
    buffer = 0.2
    assert np.all(G > 0.053 + buffer)


@pytest.mark.parametrize("e_sat", [thermo.e_sat_ice, thermo.e_sat_liquid])
def test_e_sat_increasing(e_sat):
    """Check that thermo.e_sat_ice, thermo.e_sat_liquid are increasing and positive."""
    T = np.linspace(150, 350, 10000)
    e_sat_ = e_sat(T)
    assert np.all(e_sat_ > 0)
    assert np.all(np.diff(e_sat_) > 0)


@pytest.mark.parametrize("engine_efficiency", [0.2, 0.4, 0.6])
def test_rh_crit(data: VectorDataset, engine_efficiency: float):
    """Confirm that `rh_critical_sac` takes values in [0, 1]."""
    fuel = JetA()
    G = slope_mixing_line(
        data["specific_humidity"], data["air_pressure"], engine_efficiency, fuel.ei_h2o, fuel.q_fuel
    )
    T_sat_liquid_ = T_sat_liquid(G)
    rh_crit = rh_critical_sac(data["air_temperature"], T_sat_liquid_, G)

    # All values nonnegative
    assert np.all(rh_crit >= 0)

    # Some values are infinity, indicating no SAC possible
    is_finite = np.isfinite(rh_crit)
    assert np.all(rh_crit[~is_finite] == np.inf)

    # Finite values are below 1
    assert np.all(rh_crit[is_finite] <= 1)
