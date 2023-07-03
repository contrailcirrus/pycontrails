"""Test PS model."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import pycontrails.models.ps_model.ps_model as ps
from pycontrails.core import Flight
from pycontrails.physics.jet import minimum_fuel_flow_rate_at_cruise
from pycontrails.physics.units import ft_to_pl, knots_to_m_per_s, m_to_T_isa

from .conftest import get_static_path


def test_aircraft_type_coverage() -> None:
    ps_model = ps.PSModel()

    # There are currently 53 aircraft types supported by the PS model
    assert len(ps_model.aircraft_engine_params) == 53

    # Test PS model coverage: commonly used aircraft types
    aircraft_types = ["A320", "A332", "A359", "A388", "B737", "B763", "B77L", "B789"]

    for atyp in aircraft_types:
        assert atyp in ps_model.aircraft_engine_params

    # Test unsupported aircraft types
    aircraft_types = ["A339", "A35K", "AT76", "B38X", "B78X", "BCS3", "E75L"]

    for atyp in aircraft_types:
        with pytest.raises(KeyError, match=f"Aircraft type {atyp} not covered by the PS model."):
            ps_model.check_aircraft_type_availability(atyp)


def test_ps_model() -> None:
    """
    Test intermediate variables and model outputs to be consistent with outputs from Ian Poll.
    """
    aircraft_type_icao = "A320"
    mach_number = np.array([0.753, 0.753])
    air_temperature = np.array([220.79, 216.65])
    altitude_ft = np.array([34000.0, 41450.0])
    air_pressure = ft_to_pl(altitude_ft) * 100.0
    aircraft_mass = np.array([58800.0, 58800.0])
    climb_angle = np.array([0.0, 0.0])
    dv_dt = np.array([0.0, 0.0])

    # Extract aircraft properties for aircraft type
    ps_model = ps.PSModel()
    atyp_param = ps_model.aircraft_engine_params[aircraft_type_icao]

    # Test Reynolds Number
    rn = ps.reynolds_number(
        atyp_param.wing_surface_area, mach_number, air_temperature, air_pressure
    )
    np.testing.assert_array_almost_equal(rn / 1e7, [6.777, 4.863], decimal=2)

    # Test skin friction coefficient
    c_f = ps.skin_friction_coefficient(rn)
    np.testing.assert_array_almost_equal(c_f, [2.15e-3, 2.26e-3], decimal=2)

    # Test lift coefficient
    c_lift = ps.lift_coefficient(
        atyp_param.wing_surface_area, aircraft_mass, air_pressure, mach_number, climb_angle
    )
    np.testing.assert_array_almost_equal(c_lift, [0.475, 0.679], decimal=2)

    # Test zero-lift drag coefficient
    c_drag_0 = ps.zero_lift_drag_coefficient(c_f, atyp_param.psi_0)
    np.testing.assert_array_almost_equal(c_drag_0, [0.0181, 0.0189], decimal=2)

    # Test Oswald efficiency factor
    e_ls = ps.oswald_efficiency_factor(c_drag_0, atyp_param)
    np.testing.assert_array_almost_equal(e_ls, [0.7805, 0.7740], decimal=2)

    # Test wave drag coefficient
    c_drag_w = ps.wave_drag_coefficient(mach_number, c_lift, atyp_param)
    np.testing.assert_array_almost_equal(c_drag_w, [0.00074, 0.00129], decimal=5)

    # Test airframe drag coefficient
    c_drag = ps.airframe_drag_coefficient(
        c_drag_0, c_drag_w, c_lift, e_ls, atyp_param.wing_aspect_ratio
    )
    np.testing.assert_array_almost_equal(c_drag, [0.0285, 0.0402], decimal=4)

    # Test thrust force
    f_thrust = ps.thrust_force(aircraft_mass, c_lift, c_drag, dv_dt, climb_angle)
    np.testing.assert_array_almost_equal(f_thrust / 1e4, [3.4638, 3.4156], decimal=2)

    # Test thrust force at climb/descent
    theta_climb = np.array([2.5, 2.5])
    theta_descent = np.array([-2.5, -2.5])
    f_thrust_climb = ps.thrust_force(aircraft_mass, c_lift, c_drag, dv_dt, theta_climb)
    f_thrust_descent = ps.thrust_force(aircraft_mass, c_lift, c_drag, dv_dt, theta_descent)
    np.testing.assert_array_less(f_thrust, f_thrust_climb)
    np.testing.assert_array_less(f_thrust_descent, f_thrust)
    np.testing.assert_array_less(f_thrust_descent, f_thrust_climb)

    # Test thrust coefficient
    # This should be the same as the drag coefficient as the aircraft
    # is at level flight with no acceleration
    c_t = ps.engine_thrust_coefficient(
        f_thrust, mach_number, air_pressure, atyp_param.wing_surface_area
    )
    np.testing.assert_array_almost_equal(c_t, [0.0285, 0.0402], decimal=4)

    # Test overall propulsion efficiency
    engine_efficiency = ps.overall_propulsion_efficiency(mach_number, c_t, atyp_param)
    np.testing.assert_array_almost_equal(engine_efficiency, [0.315, 0.316], decimal=3)

    # Test fuel mass flow rate
    fuel_flow = ps.fuel_mass_flow_rate(
        air_pressure,
        air_temperature,
        mach_number,
        c_t,
        engine_efficiency,
        atyp_param.wing_surface_area,
        q_fuel=43e6,
    )

    fuel_flow = ps.correct_fuel_flow(
        fuel_flow,
        altitude_ft,
        air_temperature,
        air_pressure,
        mach_number,
        atyp_param.ff_idle_sls,
        atyp_param.ff_max_sls,
    )
    np.testing.assert_array_almost_equal(fuel_flow, [0.574, 0.559], decimal=3)


def test_normalised_aircraft_performance_curves() -> None:
    """
    For a given Mach number, there is a pair of overall propulsion efficiency (ETA) and
    thrust coefficient (c_t) at which the ETA is at its maximum. A plot of `eta_over_eta_b`
    over `c_t_over_c_t_eta_b` should be an inverse U-shape and the global maximum of
    `eta_over_eta_b` should occur where `c_t_over_c_t_eta_b` is equal to 1.
    """
    aircraft_type_icao = "A320"
    f_thrust = np.arange(10000.0, 60000.0, 500)
    mach_num = np.ones_like(f_thrust) * 0.750
    altitude_ft = np.ones_like(f_thrust) * 40000.0
    air_pressure = ft_to_pl(altitude_ft) * 100.0

    # Extract aircraft properties for aircraft type
    ps_model = ps.PSModel()
    atyp_param = ps_model.aircraft_engine_params[aircraft_type_icao]
    mach_num_design_opt = atyp_param.m_des

    # Derived coefficients
    c_t = ps.engine_thrust_coefficient(
        f_thrust, mach_num, air_pressure, atyp_param.wing_surface_area
    )
    c_t_eta_b = ps.max_thrust_coefficient(mach_num, atyp_param.m_des, atyp_param.c_t_des)
    c_t_over_c_t_eta_b = c_t / c_t_eta_b

    eta = ps.overall_propulsion_efficiency(mach_num, c_t, atyp_param)
    eta_b = ps.max_overall_propulsion_efficiency(
        mach_num_design_opt, mach_num_design_opt, atyp_param.eta_1, atyp_param.eta_2
    )
    eta_over_eta_b = eta / eta_b

    i_max = np.argmax(eta_over_eta_b)

    # Global maximum of `eta_over_eta_b` should occur where `c_t_over_c_t_eta_b` is equal to 1
    assert (c_t_over_c_t_eta_b[i_max] > 0.99) and (c_t_over_c_t_eta_b[i_max] < 1.01)


@pytest.mark.parametrize("load_factor", [0.5, 0.6, 0.7, 0.8])
def test_total_fuel_burn(load_factor: float) -> None:
    """Check pinned total fuel burn values for different load factors."""
    df_flight = pd.read_csv(get_static_path("flight.csv"))

    attrs = {"flight_id": "1", "aircraft_type": "A320", "load_factor": load_factor}
    flight = Flight(df_flight.iloc[:100], attrs=attrs)

    flight["air_temperature"] = m_to_T_isa(flight["altitude"])
    flight["true_airspeed"] = knots_to_m_per_s(flight["speed"])

    # Aircraft performance model
    ps_model = ps.PSModel()
    out = ps_model.eval(flight)

    if load_factor == 0.5:
        assert out.attrs["total_fuel_burn"] == pytest.approx(4558.2, abs=0.1)
    elif load_factor == 0.6:
        assert out.attrs["total_fuel_burn"] == pytest.approx(4931.8, abs=0.1)
    elif load_factor == 0.7:
        assert out.attrs["total_fuel_burn"] == pytest.approx(5267.7, abs=0.1)
    elif load_factor == 0.8:
        assert out.attrs["total_fuel_burn"] == pytest.approx(5414.8, abs=0.1)


def test_fuel_clipping() -> None:
    """Check the ps.correct_fuel_flow function."""
    ps_model = ps.PSModel()
    atyp_param = ps_model.aircraft_engine_params["A320"]

    # Erroneous fuel flow data
    fuel_flow_est = np.array([3.05, 4.13, 5.20, 0.02, 0.03])
    altitude_ft = np.ones_like(fuel_flow_est) * 35000
    air_temperature = np.ones_like(fuel_flow_est) * 220
    air_pressure = ft_to_pl(altitude_ft) * 100
    mach_num = np.ones_like(fuel_flow_est) * 0.75

    fuel_flow_corrected = ps.correct_fuel_flow(
        fuel_flow_est,
        altitude_ft,
        air_temperature,
        air_pressure,
        mach_num,
        atyp_param.ff_idle_sls,
        atyp_param.ff_max_sls,
    )

    min_fuel_flow = minimum_fuel_flow_rate_at_cruise(atyp_param.ff_idle_sls, altitude_ft)
    assert np.all(
        (fuel_flow_corrected < atyp_param.ff_max_sls) & (fuel_flow_corrected >= min_fuel_flow)
    )


def test_zero_tas_waypoints() -> None:
    """Confirm the PSModel gracefully handles waypoints with zero true airspeed."""
    df_flight = pd.read_csv(get_static_path("flight.csv"))
    df_flight["longitude"].iat[48] = df_flight["longitude"].iat[47]
    df_flight["latitude"].iat[48] = df_flight["latitude"].iat[47]

    attrs = {"flight_id": "1", "aircraft_type": "A320"}
    flight = Flight(df_flight.iloc[:100], attrs=attrs)

    flight["air_temperature"] = m_to_T_isa(flight["altitude"])
    flight["true_airspeed"] = flight.segment_groundspeed()
    assert flight["true_airspeed"][47] == 0.0

    # Aircraft performance model
    ps_model = ps.PSModel()
    out = ps_model.eval(flight)

    # Confirm the PSModel sets NaN values for the zero TAS waypoint
    keys = "fuel_flow", "engine_efficiency", "thrust"
    for key in keys:
        assert np.isnan(out[key][47])
        assert np.all(np.isfinite(out[key][:47]))
        assert np.all(np.isfinite(out[key][48:-1]))
