"""Support for the Poll-Schumann (PS) aircraft performance over a nominal grid."""

import numpy as np
import scipy.optimize
import xarray as xr

from pycontrails.core.aircraft_performance import (
    AircraftPerformanceGrid,
    AircraftPerformanceGridData,
)
from pycontrails.core.fuel import JetA
from pycontrails.models.ps_model import ps_model
from pycontrails.models.ps_model.ps_aircraft_params import PSAircraftEngineParams
from pycontrails.physics import units
from pycontrails.utils.types import ArrayOrFloat

# mypy: disable-error-code = type-var


class PSGrid(AircraftPerformanceGrid):
    """Compute nominal Poll-Schumann aircraft performance over a grid."""


def _nominal_perf(
    aircraft_mass: ArrayOrFloat,
    atyp_param: PSAircraftEngineParams,
    altitude_ft: ArrayOrFloat,
    mach_num: ArrayOrFloat,
    q_fuel: float,
    air_temperature: ArrayOrFloat | None = None,
) -> AircraftPerformanceGridData:
    """Compute nominal Poll-Schumann aircraft performance."""

    altitude_m = units.ft_to_m(altitude_ft)
    if air_temperature is None:
        air_temperature = units.m_to_T_isa(altitude_m)
    air_pressure = units.ft_to_pl(altitude_ft) * 100.0
    theta = 0.0
    dv_dt = 0.0

    rn = ps_model.reynolds_number(
        atyp_param.wing_surface_area, mach_num, air_temperature, air_pressure
    )

    c_lift = ps_model.lift_coefficient(
        atyp_param.wing_surface_area, aircraft_mass, air_pressure, mach_num, theta
    )
    c_f = ps_model.skin_friction_coefficient(rn)
    c_drag_0 = ps_model.zero_lift_drag_coefficient(c_f, atyp_param.psi_0)
    e_ls = ps_model.oswald_efficiency_factor(c_drag_0, atyp_param)
    c_drag_w = ps_model.wave_drag_coefficient(mach_num, c_lift, atyp_param)
    c_drag = ps_model.airframe_drag_coefficient(
        c_drag_0, c_drag_w, c_lift, e_ls, atyp_param.wing_aspect_ratio
    )

    thrust = ps_model.thrust_force(aircraft_mass, c_lift, c_drag, dv_dt, theta)

    c_t = ps_model.engine_thrust_coefficient(
        thrust, mach_num, air_pressure, atyp_param.wing_surface_area
    )

    engine_efficiency = ps_model.overall_propulsion_efficiency(mach_num, c_t, atyp_param)

    fuel_flow = ps_model.fuel_mass_flow_rate(
        air_pressure,
        air_temperature,
        mach_num,
        c_t,
        engine_efficiency,
        atyp_param.wing_surface_area,
        q_fuel,
    )

    return AircraftPerformanceGridData(
        fuel_flow=fuel_flow,
        engine_efficiency=engine_efficiency,
    )


def _newton_func(
    aircraft_mass: ArrayOrFloat,
    atyp_param: PSAircraftEngineParams,
    altitude_ft: ArrayOrFloat,
    mach_num: ArrayOrFloat,
    q_fuel: float,
) -> ArrayOrFloat:
    """Approximate the derivative of the engine efficiency with respect to mass.

    This is used to find the mass at which the engine efficiency is maximized.
    """
    args = atyp_param, altitude_ft, mach_num, q_fuel
    eta1 = _nominal_perf(aircraft_mass + 0.5, *args).engine_efficiency
    eta2 = _nominal_perf(aircraft_mass - 0.5, *args).engine_efficiency
    return eta1 - eta2


def _estimate_mass_extremes(
    atyp_param: PSAircraftEngineParams,
    q_fuel: float,
    n_iter: int = 3,
) -> tuple[float, float]:
    """Calculate the minimum and maximum mass for a given aircraft type."""

    oem = atyp_param.amass_oew
    lf = 0.7
    mpm = atyp_param.amass_mpl
    mtow = atyp_param.amass_mtow

    min_mass = oem + lf * mpm  # + reserve_fuel

    for _ in range(n_iter):
        # Estimate the fuel required to cruise at 35,000 ft for 90 minutes.
        # This is used to compute the reserve fuel.
        ff = _nominal_perf(min_mass, atyp_param, 35000.0, atyp_param.m_des, q_fuel).fuel_flow
        reserve_fuel = ff * 60.0 * 90.0  # 90 minutes
        min_mass = oem + lf * mpm + reserve_fuel

    # Crude: Assume 2x the fuel flow of cruise for climb
    # Compute the maximum weight at cruise by assuming a 20 minute climb
    ff = _nominal_perf(mtow, atyp_param, 30000.0, atyp_param.m_des, q_fuel).fuel_flow
    max_mass = mtow - 2.0 * ff * 60.0 * 20.0

    return min_mass, max_mass


def ps_nominal_grid(aircraft_type: str, q_fuel: float = JetA.q_fuel) -> xr.Dataset:
    """Calculate the nominal performance grid for a given aircraft type.

    Parameters
    ----------
    aircraft_type : str
        The aircraft type.
    q_fuel : float, optional
        The fuel heating value, by default :attr:`JetA.q_fuel`

    Returns
    -------
    xr.Dataset
        The nominal performance grid. The grid is indexed by altitude and Mach number.
        Contains the following variables:

        - `fuel_flow` : Fuel flow rate, [:math:`kg/s`]
        - `engine_efficiency` : Engine efficiency
        - `aircraft_mass` : Aircraft mass at which the engine efficiency is maximized, [:math:`kg`]
    """

    aircraft_engine_params = ps_model.load_aircraft_engine_params()
    atyp_param = aircraft_engine_params[aircraft_type]

    min_mass, max_mass = _estimate_mass_extremes(atyp_param, q_fuel)

    altitude_ft = np.arange(27000, 43000, 1000, dtype=float)

    opt_mach_num = atyp_param.m_des
    mach_offset = np.arange(-0.05, 0.06, 0.005, dtype=float)
    mach_num = opt_mach_num + mach_offset

    altitude_ft_ext = altitude_ft[:, np.newaxis]
    mach_num_ext = mach_num[np.newaxis, :]

    altitude_ft_ext, mach_num_ext = np.broadcast_arrays(altitude_ft_ext, mach_num_ext)
    altitude_ft_flat = altitude_ft_ext.ravel()
    mach_num_flat = mach_num_ext.ravel()

    aircraft_mass_flat = scipy.optimize.newton(
        _newton_func,
        args=(atyp_param, altitude_ft_flat, mach_num_flat, q_fuel),
        x0=np.full_like(altitude_ft_flat, (max_mass + min_mass) / 2.0),
        tol=1.0,
    )

    aircraft_mass_flat.clip(min=min_mass, max=max_mass, out=aircraft_mass_flat)

    perf = _nominal_perf(aircraft_mass_flat, atyp_param, altitude_ft_flat, mach_num_flat, q_fuel)

    engine_efficiency_flat = perf.engine_efficiency
    fuel_flow_flat = perf.fuel_flow

    aircraft_mass = aircraft_mass_flat.reshape(altitude_ft_ext.shape)
    engine_efficiency = engine_efficiency_flat.reshape(altitude_ft_ext.shape)
    fuel_flow = fuel_flow_flat.reshape(altitude_ft_ext.shape)

    return xr.Dataset(
        {
            "aircraft_mass": (("altitude_ft", "mach_num"), aircraft_mass),
            "engine_efficiency": (("altitude_ft", "mach_num"), engine_efficiency),
            "fuel_flow": (("altitude_ft", "mach_num"), fuel_flow),
        },
        coords={
            "altitude_ft": altitude_ft,
            "mach_num": mach_num,
        },
    )
