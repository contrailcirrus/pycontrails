"""Test the fuel module."""

import dataclasses

import pytest

from pycontrails import Fuel, HydrogenFuel, JetA, SAFBlend


def test_fuel_and_saf_properties():
    """Check basic properties of JetA and SAFBlend."""
    jet_a = JetA()
    saf = SAFBlend(10)
    assert isinstance(jet_a, Fuel)
    assert isinstance(saf, Fuel)

    # Every attribute on jet_a is also on saf
    assert jet_a.__dict__.keys() - saf.__dict__.keys() == set()

    # And the saf instance has one more attribute
    saf.__dict__.keys() - jet_a.__dict__.keys() == {"pct_blend"}

    assert saf.fuel_name == "Jet A-1 / Sustainable Aviation Fuel Blend"
    assert jet_a.fuel_name == "Jet A-1"


def test_saf_zero_pct_blend():
    """Confirm that SAFBlend(pct_blend=0) is identical to JetA."""
    jet_a = JetA()
    saf_0 = SAFBlend(pct_blend=0)

    for field in dataclasses.fields(jet_a):
        k = field.name
        if k == "fuel_name":
            continue
        assert getattr(jet_a, k) == getattr(saf_0, k)


def test_saf_blend_values():
    """Confirm a few pinned values for `SAFBlend` class."""
    saf_50 = SAFBlend(pct_blend=50)
    saf_100 = SAFBlend(pct_blend=100)

    assert isinstance(saf_50, Fuel)
    assert saf_50.hydrogen_content == 14.55
    assert saf_50.q_fuel == 43.665e6
    assert saf_50.ei_h2o == pytest.approx(1.296848, abs=1e-3)

    assert isinstance(saf_100, Fuel)
    assert saf_100.hydrogen_content == 15.3
    assert saf_100.q_fuel == 44.2e6
    assert saf_100.ei_h2o == pytest.approx(1.363696, abs=1e-3)


def test_hydrogen_fuel():
    """Check some properties of HydrogenFuel."""
    hf = HydrogenFuel()
    assert hf.fuel_name == "Hydrogen"

    jet_a = JetA()
    assert isinstance(hf, Fuel)
    assert hf.q_fuel > jet_a.q_fuel
    assert hf.ei_h2o > jet_a.ei_h2o
