"""Test PS model."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import pycontrails.models.ps_model.ps_aircraft_params as ps_params
import pycontrails.models.ps_model.ps_model as ps
import pycontrails.models.ps_model.ps_operational_limits as ps_lims
from pycontrails import Flight, FlightPhase, GeoVectorDataset, MetDataset
from pycontrails.models.ps_model import PSGrid, ps_nominal_grid
from pycontrails.physics import units

from .conftest import get_static_path


def test_aircraft_type_coverage() -> None:
    ps_model = ps.PSFlight()

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


def test_derived_aircraft_engine_params() -> None:
    """
    Test derived aircraft-engine parameters for PS model.
    """
    # Input parameters are for the following aircraft types: A320, A359

    # Test turbine entry temperature at maximum take-off rating
    first_flight = np.array([1987.0, 2013.0])
    tet_mto = ps_params.turbine_entry_temperature_at_max_take_off(first_flight)
    np.testing.assert_array_almost_equal(tet_mto, [1661.88, 1854.76], decimal=2)

    # Test turbine entry temperature at maximum continuous climb rating
    tet_mcc = ps_params.turbine_entry_temperature_at_max_continuous_climb(tet_mto)
    np.testing.assert_array_almost_equal(tet_mcc, [1528.93, 1706.38], decimal=2)

    # Test maximum permitted operational impact pressure
    max_mach_num = np.array([0.820, 0.890])
    p_i_max = ps_params.impact_pressure_max_operating_limits(max_mach_num)
    np.testing.assert_array_almost_equal(p_i_max, [20882.79, 24441.49], decimal=2)

    # Test max calibrated airspeed over the speed of sound at ISA mean sea level
    v_cas_mo_over_c_msl = ps_params.max_calibrated_airspeed_over_speed_of_sound(max_mach_num)
    np.testing.assert_array_almost_equal(v_cas_mo_over_c_msl, [0.5244, 0.5643], decimal=4)

    # Test crossover pressure altitude
    p_inf_co = ps_params.crossover_pressure_altitude(max_mach_num, p_i_max)
    np.testing.assert_array_almost_equal(p_inf_co, [37618.42, 36319.05], decimal=2)


def test_ps_model() -> None:
    """
    Test intermediate variables and model outputs to be consistent with outputs from Ian Poll.
    """
    aircraft_type_icao = "A320"
    mach_number = np.array([0.753, 0.753])
    air_temperature = np.array([220.79, 216.65])
    altitude_ft = np.array([34000.0, 41450.0])
    air_pressure = units.ft_to_pl(altitude_ft) * 100.0
    aircraft_mass = np.array([58800.0, 58800.0])
    climb_angle = np.array([0.0, 0.0])
    dv_dt = np.array([0.0, 0.0])

    # Extract aircraft properties for aircraft type
    ps_model = ps.PSFlight()
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
    np.testing.assert_array_almost_equal(c_drag_0, [0.0181, 0.0190], decimal=2)

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

    # Test thrust coefficient at maximum overall propulsion efficiency for a given Mach Number
    c_t_eta_b = ps.thrust_coefficient_at_max_efficiency(
        mach_number, atyp_param.m_des, atyp_param.c_t_des
    )
    np.testing.assert_array_almost_equal(c_t_eta_b, [0.0347, 0.0347], decimal=2)

    # Test overall propulsion efficiency
    engine_efficiency = ps.overall_propulsion_efficiency(mach_number, c_t, c_t_eta_b, atyp_param)
    np.testing.assert_array_almost_equal(engine_efficiency, [0.2926, 0.2935], decimal=3)

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
    np.testing.assert_array_almost_equal(fuel_flow, [0.617, 0.601], decimal=3)


def test_mach_number_limits():
    # Extract aircraft properties for aircraft type (A320)
    aircraft_type_icao = "A320"
    ps_model = ps.PSFlight()
    atyp_param = ps_model.aircraft_engine_params[aircraft_type_icao]

    # Test Mach number limits
    altitude_ft = np.arange(10000.0, 41000.0, 5000.0)
    air_pressure = units.ft_to_pl(altitude_ft) * 100.0

    mach_num_lim = ps_lims.max_mach_number_by_altitude(
        altitude_ft,
        air_pressure,
        atyp_param.max_mach_num,
        atyp_param.p_i_max,
        atyp_param.p_inf_co,
        atm_speed_limit=False,
        buffer=0.0,
    )
    np.testing.assert_array_almost_equal(
        mach_num_lim, [0.625, 0.683, 0.750, 0.82, 0.82, 0.82, 0.82], decimal=2
    )


def test_thrust_coefficient_limits():
    # Extract aircraft properties for aircraft type (A320)
    aircraft_type_icao = "A320"
    ps_model = ps.PSFlight()
    atyp_param = ps_model.aircraft_engine_params[aircraft_type_icao]

    mach_number = 0.75
    air_temperature = np.arange(210.0, 251.0, 10.0)
    c_t_eta_b = ps.thrust_coefficient_at_max_efficiency(
        mach_number, atyp_param.m_des, atyp_param.c_t_des
    )

    # Maximum available thrust coefficients
    c_t_max_avail = ps_lims.max_available_thrust_coefficient(
        air_temperature, mach_number, c_t_eta_b, atyp_param
    )
    np.testing.assert_array_almost_equal(
        c_t_max_avail, [0.050, 0.046, 0.041, 0.037, 0.033], decimal=3
    )


def test_aircraft_mass_limits() -> None:
    """Test the max_allowable_aircraft_mass function."""

    # Extract aircraft properties for aircraft type (A388)
    aircraft_type_icao = "A388"
    ps_model = ps.PSFlight()
    atyp_param = ps_model.aircraft_engine_params[aircraft_type_icao]
    mtow = atyp_param.amass_mtow

    mach_number = 0.82
    altitude_ft = np.arange(30000.0, 43000.0, 2500.0)
    air_pressure = units.ft_to_pl(altitude_ft) * 100.0

    amass_lims = ps_lims.max_allowable_aircraft_mass(
        air_pressure,
        mach_number,
        atyp_param.m_des,
        atyp_param.c_l_do,
        atyp_param.wing_surface_area,
        mtow,
    )
    np.testing.assert_array_almost_equal(
        amass_lims, [mtow, mtow, 521823.2, 462861.6, 410455.8, 363983.4], decimal=0
    )


@pytest.mark.filterwarnings("ignore:some failed to converge")
@pytest.mark.parametrize("aircraft_type", ["A320", "A333", "B737", "B753"])
def test_aircraft_mass_limits_ps_grid(aircraft_type: str) -> None:
    """Test the max_allowable_aircraft_mass function with PSGrid."""

    atyp_param = ps.load_aircraft_engine_params()[aircraft_type]
    ps_model = PSGrid()

    altitude_ft = np.arange(25000.0, 45000.0, 1000.0)
    n = len(altitude_ft)
    vector = GeoVectorDataset(
        longitude=np.zeros(n),
        latitude=np.zeros(n),
        altitude_ft=altitude_ft,
        time=np.r_[[np.datetime64("2020-01-01T00:00:00")] * n],
        aircraft_type=aircraft_type,
    )
    vector["air_temperature"] = units.m_to_T_isa(vector.altitude)

    out = ps_model.eval(vector)
    assert out.attrs["mach_number"] == atyp_param.m_des

    pred_amass = out["aircraft_mass"]
    max_amass = ps_lims.max_allowable_aircraft_mass(
        out.air_pressure,
        atyp_param.m_des,
        atyp_param.m_des,
        atyp_param.c_l_do,
        atyp_param.wing_surface_area,
        atyp_param.amass_mtow,
    )
    assert np.all(pred_amass <= max_amass)


def test_fuel_flow_limits() -> None:
    """Check the ps.correct_fuel_flow function."""
    ps_model = ps.PSFlight()
    atyp_param = ps_model.aircraft_engine_params["A320"]

    # Erroneous fuel flow data
    fuel_flow_est = np.array([3.05, 4.13, 5.20, 0.02, 0.03])
    altitude_ft = np.ones_like(fuel_flow_est) * 35000
    air_temperature = np.ones_like(fuel_flow_est) * 220
    air_pressure = units.ft_to_pl(altitude_ft) * 100
    mach_num = np.ones_like(fuel_flow_est) * 0.75

    fuel_flow_corrected = ps.fuel_flow_correction(
        fuel_flow_est,
        altitude_ft,
        air_temperature,
        air_pressure,
        mach_num,
        atyp_param.ff_idle_sls,
        atyp_param.ff_max_sls,
        FlightPhase.CRUISE,
    )

    np.testing.assert_array_almost_equal(
        fuel_flow_corrected, [1.261, 1.261, 1.261, 0.11, 0.11], decimal=2
    )
    assert np.all(fuel_flow_corrected < atyp_param.ff_max_sls)


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
    air_pressure = units.ft_to_pl(altitude_ft) * 100.0

    # Extract aircraft properties for aircraft type
    ps_model = ps.PSFlight()
    atyp_param = ps_model.aircraft_engine_params[aircraft_type_icao]
    mach_num_design_opt = atyp_param.m_des

    # Derived coefficients
    c_t = ps.engine_thrust_coefficient(
        f_thrust, mach_num, air_pressure, atyp_param.wing_surface_area
    )
    c_t_eta_b = ps.thrust_coefficient_at_max_efficiency(
        mach_num, atyp_param.m_des, atyp_param.c_t_des
    )
    c_t_over_c_t_eta_b = c_t / c_t_eta_b

    eta = ps.overall_propulsion_efficiency(mach_num, c_t, c_t_eta_b, atyp_param)
    eta_b = ps.max_overall_propulsion_efficiency(
        mach_num_design_opt, mach_num_design_opt, atyp_param.eta_1, atyp_param.eta_2
    )
    eta_over_eta_b = eta / eta_b

    i_max = np.argmax(eta_over_eta_b)

    # Global maximum of `eta_over_eta_b` should occur where `c_t_over_c_t_eta_b` is equal to 1
    assert c_t_over_c_t_eta_b[i_max] > 0.99
    assert c_t_over_c_t_eta_b[i_max] < 1.01


@pytest.mark.parametrize("load_factor", [0.5, 0.6, 0.7, 0.8])
def test_total_fuel_burn(load_factor: float) -> None:
    """Check pinned total fuel burn values for different load factors."""
    df_flight = pd.read_csv(get_static_path("flight.csv"))

    attrs = {"flight_id": "1", "aircraft_type": "A320", "load_factor": load_factor}
    flight = Flight(df_flight.iloc[:100], attrs=attrs)

    flight["air_temperature"] = units.m_to_T_isa(flight["altitude"])
    flight["true_airspeed"] = units.knots_to_m_per_s(flight["speed"])

    # Aircraft performance model
    ps_model = ps.PSFlight()
    out = ps_model.eval(flight)

    if load_factor == 0.5:
        assert out.attrs["total_fuel_burn"] == pytest.approx(4509, abs=1.0)
    elif load_factor == 0.6:
        assert out.attrs["total_fuel_burn"] == pytest.approx(4573, abs=1.0)
    elif load_factor == 0.7:
        assert out.attrs["total_fuel_burn"] == pytest.approx(4618, abs=1.0)
    elif load_factor == 0.8:
        assert out.attrs["total_fuel_burn"] == pytest.approx(4635, abs=1.0)
    else:
        pytest.fail("Unexpected load factor")


def test_zero_tas_waypoints() -> None:
    """Confirm the PSFlight gracefully handles waypoints with zero true airspeed."""
    df_flight = pd.read_csv(get_static_path("flight.csv"))
    df_flight["longitude"].iat[48] = df_flight["longitude"].iat[47]
    df_flight["latitude"].iat[48] = df_flight["latitude"].iat[47]

    attrs = {"flight_id": "1", "aircraft_type": "A320"}
    flight = Flight(df_flight.iloc[:100], attrs=attrs)

    flight["air_temperature"] = units.m_to_T_isa(flight["altitude"])
    flight["true_airspeed"] = flight.segment_groundspeed()
    assert flight["true_airspeed"][47] == 0.0

    # Aircraft performance model
    ps_model = ps.PSFlight()
    out = ps_model.eval(flight)

    # Confirm the PSFlight sets NaN values for the zero TAS waypoint
    keys = "fuel_flow", "engine_efficiency", "thrust"
    for key in keys:
        assert np.isnan(out[key][47])
        assert np.all(np.isfinite(out[key][:47]))
        assert np.all(np.isfinite(out[key][48:-1]))


@pytest.mark.filterwarnings("ignore:some failed to converge")
@pytest.mark.parametrize("aircraft_type", ["A320", "A333", "B737", "B753"])
def test_ps_nominal_grid(aircraft_type: str) -> None:
    """Test the ps_nominal_grid function assuming the ISA temperature."""

    altitude_ft = np.arange(27000, 44000, 1000, dtype=float)
    level = units.ft_to_pl(altitude_ft)
    ds = ps_nominal_grid(aircraft_type, level=level)
    assert isinstance(ds, xr.Dataset)

    assert list(ds.dims) == ["level"]
    assert list(ds) == ["aircraft_mass", "engine_efficiency", "fuel_flow"]
    assert ds.attrs["aircraft_type"] == aircraft_type
    assert 0.7 < ds.attrs["mach_number"] < 0.8


def test_ps_grid_vector_source(met_era5_fake: MetDataset) -> None:
    """Test the PSGrid model with a vector source."""

    model = PSGrid(met_era5_fake)
    vector = GeoVectorDataset(
        longitude=[3, 5, 8],
        latitude=[13, 21, 34],
        level=[160, 170, 180],
        time=[met_era5_fake.data.time.values[0]] * 3,
    )
    out = model.eval(vector)
    assert isinstance(out, GeoVectorDataset)
    assert out.size == 3
    assert out.dataframe.shape == (3, 9)
    assert list(out) == [
        "longitude",
        "latitude",
        "time",
        "level",
        "air_temperature",
        "aircraft_mass",
        "fuel_flow",
        "engine_efficiency",
        "true_airspeed",
    ]

    assert out.attrs == {
        "crs": "EPSG:4326",
        "aircraft_type": "B737",
        "mach_number": 0.758,
        "wingspan": 34.3,
        "n_engine": 2,
    }


def test_ps_grid_met_source(met_era5_fake: MetDataset) -> None:
    """Test the PSGrid model with source=None."""

    model = PSGrid(met_era5_fake)
    out = model.eval()
    assert isinstance(out, MetDataset)

    ds = out.data
    assert ds.dims == met_era5_fake.data.dims
    assert list(ds) == ["aircraft_mass", "engine_efficiency", "fuel_flow"]

    # Pin some output values
    abs = 1e-2
    assert ds["fuel_flow"].min() == pytest.approx(0.55, abs=abs)
    assert ds["fuel_flow"].max() == pytest.approx(0.73, abs=abs)
    assert ds["fuel_flow"].mean() == pytest.approx(0.65, abs=abs)

    abs = 1e-3
    assert ds["engine_efficiency"].min() == pytest.approx(0.268, abs=abs)
    assert ds["engine_efficiency"].max() == pytest.approx(0.282, abs=abs)
    assert ds["engine_efficiency"].mean() == pytest.approx(0.278, abs=abs)

    abs = 1e3
    assert ds["aircraft_mass"].min() == pytest.approx(53000, abs=abs)
    assert ds["aircraft_mass"].max() == pytest.approx(68000, abs=abs)
    assert ds["aircraft_mass"].mean() == pytest.approx(61000, abs=abs)


def test_ps_grid_raises(met_era5_fake: MetDataset) -> None:
    """Confirm that the PSGrid model raises error if the aircraft_mass model param is not None."""

    model = PSGrid(met_era5_fake)
    with pytest.raises(NotImplementedError, match="The 'aircraft_mass' parameter must be None."):
        model.eval(aircraft_mass=60000)

    model = PSGrid(met_era5_fake, aircraft_mass=60000)
    with pytest.raises(NotImplementedError, match="The 'aircraft_mass' parameter must be None."):
        model.eval()
