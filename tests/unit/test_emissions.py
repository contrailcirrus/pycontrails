"""Emissions model tests."""

from __future__ import annotations

import numpy as np
import pytest

from pycontrails import Flight
from pycontrails.core.fuel import JetA
from pycontrails.core.models import Model
from pycontrails.models.emissions import Emissions, ffm2
from pycontrails.models.emissions import black_carbon as nvpm
from pycontrails.models.emissions import emissions as emissions_mod
from pycontrails.physics import jet, units
from pycontrails.physics.jet import thrust_setting_nd


def test_emissions_init():
    """Test emissions __init__ method."""
    emissions = Emissions()
    assert "01P11CM114" in emissions.edb_engine_gaseous  # Engine for Boeing 737-700 (CFM56-7B24E)
    assert "01P11CM114" in emissions.edb_engine_nvpm  # Engine for Boeing 737-700 (CFM56-7B24E)

    assert "3GE074" in emissions.edb_engine_gaseous  # Gaseous data for old engine (DC10, CF6-50C2)
    assert "3GE074" not in emissions.edb_engine_nvpm  # No nvPM data for old engine (DC10, CF6-50C2)
    assert not hasattr(emissions, "source")

    assert isinstance(emissions, Model)
    assert isinstance(emissions.params, dict)


def test_emissions_class_variables():
    """Check that `Emissions` class only creates dictionaries once."""
    emissions1 = Emissions()
    emissions2 = Emissions()
    assert emissions1 is not emissions2
    assert emissions1.edb_engine_gaseous is emissions2.edb_engine_gaseous
    assert emissions2.edb_engine_nvpm is emissions2.edb_engine_nvpm

    # Update if emissions data changes
    assert len(emissions1.edb_engine_gaseous) == 557
    assert len(emissions1.edb_engine_nvpm) == 178

    # Check that the gaseous engine UIDs are a superset of the nvPM engine UIDs in the EDB.
    # This logic is used in cocip_grid.calc_emissions
    assert set(emissions1.edb_engine_gaseous).issuperset(emissions1.edb_engine_nvpm)


@pytest.mark.parametrize("aircraft_type", ["B737", "B738", "A320"])
@pytest.mark.parametrize("engine_uid", ["01P11CM114", "01P11CM116", "01P08CM105"])
@pytest.mark.parametrize("bada", ["BADA3", "BADA4"])
def test_emissions_eval(flight_fake: Flight, bada: str, aircraft_type: str, engine_uid: str):
    """Test emissions `eval` method.

    Simple smoke test with several different BADA sources and aircraft types.
    """
    emissions = Emissions()

    match = "Variable `air_temperature` not found. "
    with pytest.raises(KeyError, match=match):
        emissions.eval(source=flight_fake)

    flight_fake["air_temperature"] = 216 * np.ones(flight_fake.size)
    flight_fake["specific_humidity"] = 1e-6 * np.ones(flight_fake.size)
    flight_fake["true_airspeed"] = 247 * np.ones(flight_fake.size)
    flight_fake["fuel_flow"] = 0.3 * np.ones(flight_fake.size)
    flight_fake.attrs["aircraft_type"] = aircraft_type
    flight_fake.attrs["aircraft_type_bada"] = aircraft_type
    flight_fake.attrs["engine_uid"] = engine_uid
    flight_fake.attrs["bada_model"] = bada
    flight_fake.attrs["n_engine"] = 2

    out_fl = emissions.eval(source=flight_fake)
    assert emissions.source is not flight_fake  # since copy is True
    assert isinstance(out_fl, Flight)

    # Constants in => nvpm_ei_n is now a function of altitude
    counts = np.unique(out_fl["nvpm_ei_n"], return_counts=True)[1]
    assert counts.size == 500

    # ensure that "thrust", "nvpm_ei_n" and "nvpm_ei_m" are not overwritten
    flight_fake["thrust_setting"] = 0.3 * np.ones(flight_fake.size)
    out_fl = emissions.eval(source=flight_fake)
    assert np.all(out_fl["thrust_setting"] == 0.3)
    del flight_fake["thrust_setting"]

    # both nvpm_ei_n keys need to be defined here
    flight_fake["nvpm_ei_n"] = 1e15 * np.ones(flight_fake.size)
    out_fl = emissions.eval(source=flight_fake)

    assert np.all(out_fl["nvpm_ei_n"] == 1e15)

    flight_fake["nvpm_ei_m"] = 2e-4 * np.ones(flight_fake.size)
    out_fl = emissions.eval(flight_fake)

    assert np.all(out_fl["nvpm_ei_n"] == 1e15)
    assert np.all(out_fl["nvpm_ei_m"] == 2e-4)

    # TODO: We don't currently guard against overwriting on the pollutants
    flight_fake["co2"] = 15 * np.ones(flight_fake.size)
    with pytest.warns(UserWarning, match="Overwriting data in key `co2`"):
        out_fl = emissions.eval(flight_fake)


def test_emissions_index_ffm2():
    """Test emissions without explicit flight.

    Pin a few values that fit expectations.
    """
    engine_uid = "01P18RR103"  # Engine for Airbus A380 (Trent 970-84)

    # Cruise conditions
    air_pressure = 19650  # Units: Pa
    air_temperature = 216  # Units: K
    true_airspeed = 247.5  # Units: m/s
    fuel_flow_per_engine_cruise = 0.882  # Units: kg/s
    fuel_flow_per_engine_descent = 0.125  # Units: kg/s (Set to < 7% equivalent engine power)

    emissions = Emissions()
    assert not hasattr(emissions, "source")
    edb_gaseous = emissions.edb_engine_gaseous[engine_uid]

    # Nitrogen oxide
    nox_ei = emissions_mod.nitrogen_oxide_emissions_index_ffm2(
        edb_gaseous,
        fuel_flow_per_engine_cruise,
        true_airspeed,
        air_pressure,
        air_temperature,
    )
    np.testing.assert_almost_equal(nox_ei, np.array([0.019298]), decimal=6)

    # Carbon monoxide (descent conditions with very low power)
    co_ei = emissions_mod.carbon_monoxide_emissions_index_ffm2(
        edb_gaseous,
        fuel_flow_per_engine_descent,
        true_airspeed,
        air_pressure,
        air_temperature,
    )
    np.testing.assert_almost_equal(co_ei, np.array([0.028461]), decimal=6)

    # Hydrocarbons (descent conditions with very low power)
    hc_ei = emissions_mod.hydrocarbon_emissions_index_ffm2(
        edb_gaseous,
        fuel_flow_per_engine_descent,
        true_airspeed,
        air_pressure,
        air_temperature,
    )
    np.testing.assert_almost_equal(hc_ei, np.array([0.000077817]), decimal=6)


def test_E45x_hydrocarbon_emissions_index() -> None:
    """Test specific error arising from duplicate indexes in co_hc_emissions_index_profile."""
    engine_uid = "01P06AL032"  # Engine for E45X (AE 3007A1E)

    # Cruise conditions
    air_pressure = np.array([19650])  # Units: Pa
    air_temperature = np.array([216])  # Units: K
    true_airspeed = np.array([247.5])  # Units: m/s
    fuel_flow_per_engine_descent = np.array([0.01])

    emissions = Emissions()
    edb_gaseous = emissions.edb_engine_gaseous[engine_uid]

    # Hydrocarbons (descent conditions with very low power)
    hc_ei = emissions_mod.hydrocarbon_emissions_index_ffm2(
        edb_gaseous,
        fuel_flow_per_engine_descent,
        true_airspeed,
        air_pressure,
        air_temperature,
    )
    np.testing.assert_almost_equal(hc_ei, np.array([0.0097947]), decimal=6)


def test_specific_humidity_correction_factor_display_45():
    """Test assertion following display (45) in Dubois & Paynter 2006."""

    # sea level values
    T = units.m_to_T_isa(0)
    p = units.m_to_pl(0) * 100
    rh = 0.6

    omega = ffm2._estimate_specific_humidity(T, p, rh)

    # value 0.00634 is taken from paragraph following (45)
    assert omega == pytest.approx(0.00634, abs=1e-5)


def test_example_page_11():
    """Confirm example on Dubois & Paynter 2006 page 11.

    The value for omega in the paper looks to be wrong.
    """
    # Given data
    T = 390 / 1.8
    p = units.ft_to_pl(39000) * 100
    rh = 0.6
    delta_amb = 0.1942
    theta_amb = 0.7519
    ei_sl_nox = 19.43
    ei_sl_co = 0.24

    omega = ffm2._estimate_specific_humidity(T, p, rh)
    # The value 0.00053 is taken from page 11 seems to be wrong ...
    # It should actually be 0.000053
    assert omega == pytest.approx(0.000053, abs=1e-5)

    q_factor = ffm2._get_humidity_correction_factor(omega)
    nox = ffm2.ei_at_cruise(ei_sl_nox, theta_amb, delta_amb, "NOX") * q_factor
    co = ffm2.ei_at_cruise(ei_sl_co, theta_amb, delta_amb, "CO")

    assert nox == pytest.approx(15.19, abs=1e-2)
    assert co == pytest.approx(0.50, abs=1e-2)


def test_emission_interp_all_nan():
    """Confirm emission functions gracefully handle missing data.

    Mock flight constructed below contains realworld data raising an exception in pycontrails.

    This test fails on pycontrails 0.20.0.
    """
    with pytest.warns(UserWarning, match="Time data is not np.datetime64"):
        fl = Flight(
            longitude=[-111.99457, -111.988735, -111.980604],
            latitude=[35.53877, 35.537842, 35.536582],
            altitude=[12496.8, 12496.8, 12496.8],
            time=["2022-02-01 03:00:21", "2022-02-01 03:00:23", "2022-02-01 03:00:25"],
            flight_id="killer whale",
        )
    fl["air_temperature"] = [np.nan, np.nan, np.nan]
    fl["specific_humidity"] = [0.002, 0.002, 0.003]
    fl["true_airspeed"] = [250.0, 240.0, 230.0]
    fl["fuel_flow"] = [0.5, 0.5, 0.6]

    fl.attrs["n_engine"] = 2
    fl.attrs["engine_uid"] = "01P06BR014"

    emissions = Emissions()

    out = emissions.eval(source=fl)
    assert np.isnan(out["nvpm_ei_n"]).all()
    assert np.isnan(out["nvpm_ei_m"]).all()

    fl.update(air_temperature=[230.0, 235.0, 240.0])
    out2 = emissions.eval(source=fl)

    assert np.isfinite(out2["nvpm_ei_n"]).all()
    assert np.isfinite(out2["nvpm_ei_m"]).all()


def test_t4_t2_ground_estimates():
    # Ground conditions from ICAO EDB (Airbus A320)
    true_airspeed = np.zeros(4)
    air_temperature = np.ones(4) * 0.5 * (275.8 + 287.2)
    air_pressure = np.ones(4) * 101325
    thrust_setting = np.array([0.134, 0.328, 0.873, 1.049]) / 1.049
    pressure_ratio = 27.1
    fuel = JetA()

    t4_t2_est = thrust_setting_nd(
        true_airspeed,
        thrust_setting,
        air_temperature,
        air_pressure,
        pressure_ratio,
        fuel.q_fuel,
        comp_efficiency=0.9,
        cruise=False,
    )

    t4_t2 = np.array([2.426554, 3.031491, 4.302515, 4.660068])
    assert t4_t2_est == pytest.approx(t4_t2, abs=1e-2)


def test_t4_t2_cruise_estimates():
    # Cruise conditions from ECLIF measurements (Airbus A320)
    fuel_flow = np.array([0.332192, 0.335372, 0.333815, 0.382108])
    air_temperature = np.array([218.116667, 217.845833, 217.720588, 233.272222])
    air_pressure = np.array([357.76667, 357.693750, 358.288235, 359.422222]) * 100
    true_airspeed = np.array([213.32261639, 212.55907557, 212.67744452, 233.40349122])
    mach_num = units.tas_to_mach_number(true_airspeed, air_temperature)

    theta_amb = jet.temperature_ratio(air_temperature)
    delta_amb = jet.pressure_ratio(air_pressure)
    fuel_flow_per_engine_est = jet.equivalent_fuel_flow_rate_at_sea_level(
        fuel_flow, theta_amb, delta_amb, mach_num
    )
    fuel_flow_per_engine_eclif = np.array([0.362298, 0.363894, 0.360877, 0.542128])
    assert fuel_flow_per_engine_est == pytest.approx(fuel_flow_per_engine_eclif, abs=1e-3)

    thrust_setting = fuel_flow_per_engine_est / 1.049
    pressure_ratio = 27.1
    fuel = JetA()

    t4_t2_est = thrust_setting_nd(
        true_airspeed,
        thrust_setting,
        air_temperature,
        air_pressure,
        pressure_ratio,
        fuel.q_fuel,
        comp_efficiency=0.9,
        cruise=True,
    )

    t4_t2_eclif = np.array([3.704481, 3.716919, 3.708952, 3.905757])
    assert t4_t2_est == pytest.approx(t4_t2_eclif, abs=1e-3)


def test_nvpm_ein_reductions_from_saf():
    """Test ``nvpm.nvpm_number_ei_pct_reduction_due_to_saf`` with some pinned values."""
    hydrogen_mass_content = np.array([14.11, 14.15, 14.1, 14.43, 14.47, 14.58, 14.4])
    thrust_setting = np.array([0.30, 0.65, 0.85, 0.3376, 0.4284, 0.3945, np.nan])
    d_nvpm_ein_pct = np.array([-25.55, -15.86, -7.23, -46.30, -42.34, -49.09, -37.96])
    d_nvpm_ein_pct_est = nvpm.nvpm_number_ei_pct_reduction_due_to_saf(
        hydrogen_mass_content, thrust_setting
    )
    np.testing.assert_allclose(d_nvpm_ein_pct_est, d_nvpm_ein_pct, atol=0.01)


@pytest.mark.parametrize("engine_uid", ["01P20CM128", "01P21GE216", "01P17GE210"])
def test_stage_combustors_data_length(engine_uid: str):
    """Confirm a few handpicked engine with multi-stage combustors have the correct data."""
    emissions = Emissions()
    edb_nvpm = emissions.edb_engine_nvpm[engine_uid]

    # Multistage combuster engines have 5 datapoints in the interpolation set
    assert len(edb_nvpm.nvpm_ei_m.fp) == 5
    assert len(edb_nvpm.nvpm_ei_m.xp) == 5
    assert len(edb_nvpm.nvpm_ei_n.fp) == 5
    assert len(edb_nvpm.nvpm_ei_n.xp) == 5

    # But the final three y values are all equal
    assert np.all(edb_nvpm.nvpm_ei_m.fp[2:] == edb_nvpm.nvpm_ei_m.fp[2])
    assert np.all(edb_nvpm.nvpm_ei_n.fp[2:] == edb_nvpm.nvpm_ei_n.fp[2])

    # The final x values are increasing
    assert np.all(np.diff(edb_nvpm.nvpm_ei_m.xp) > 0)
    assert np.all(np.diff(edb_nvpm.nvpm_ei_n.xp) > 0)
