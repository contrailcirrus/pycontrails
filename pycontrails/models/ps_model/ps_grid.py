"""Support for the Poll-Schumann (PS) aircraft performance over a nominal grid."""

from typing import Any, overload

import numpy as np
import numpy.typing as npt
import scipy.optimize
import xarray as xr

from pycontrails.core.aircraft_performance import (
    AircraftPerformanceGrid,
    AircraftPerformanceGridData,
    AircraftPerformanceGridParams,
)
from pycontrails.core.fuel import JetA
from pycontrails.core.met import MetDataArray, MetDataset
from pycontrails.core.met_var import AirTemperature
from pycontrails.core.vector import GeoVectorDataset
from pycontrails.models.ps_model import ps_model
from pycontrails.models.ps_model.ps_aircraft_params import PSAircraftEngineParams
from pycontrails.physics import units
from pycontrails.utils.types import ArrayOrFloat

# mypy: disable-error-code = type-var


class PSGrid(AircraftPerformanceGrid):
    """Compute nominal Poll-Schumann aircraft performance over a grid."""

    name = "PSGrid"
    long_name = "Poll-Schumann Aircraft Performance evaluated at arbitrary points"
    met_variables = (AirTemperature,)

    default_params = AircraftPerformanceGridParams

    @overload
    def eval(self, source: GeoVectorDataset, **params: Any) -> GeoVectorDataset:
        ...

    @overload
    def eval(self, source: MetDataset | None = None, **params: Any) -> MetDataset:
        ...

    def eval(
        self, source: GeoVectorDataset | MetDataset | None = None, **params: Any
    ) -> GeoVectorDataset | MetDataset:
        """To do. FIXME."""
        self.update_params(**params)
        self.set_source(source)
        self.source = self.require_source_type((GeoVectorDataset, MetDataset))
        self.set_source_met()

        aircraft_type = self.source.attrs.get("aircraft_type", self.params["aircraft_type"])
        fuel = self.get_source_param("fuel")
        q_fuel = fuel.q_fuel

        if isinstance(self.source, MetDataset):
            ds = ps_nominal_grid(
                aircraft_type,
                air_temperature=self.source["air_temperature"],
                q_fuel=q_fuel,
            )
            return MetDataset(ds)

        ds = ps_nominal_grid(
            aircraft_type,
            level=self.source.level,
            air_temperature=self.source["air_temperature"],
            q_fuel=q_fuel,
        )
        self.source.setdefault("aircraft_mass", ds["aircraft_mass"])
        self.source.setdefault("fuel_flow", ds["fuel_flow"])
        self.source.setdefault("engine_efficiency", ds["engine_efficiency"])
        return self.source


def _nominal_perf(
    aircraft_mass: ArrayOrFloat,
    atyp_param: PSAircraftEngineParams,
    air_pressure: ArrayOrFloat,
    mach_num: ArrayOrFloat,
    q_fuel: float,
    air_temperature: ArrayOrFloat,
) -> AircraftPerformanceGridData:
    """Compute nominal Poll-Schumann aircraft performance."""

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
    perf_kwargs: dict[str, Any],
) -> ArrayOrFloat:
    """Approximate the derivative of the engine efficiency with respect to mass.

    This is used to find the mass at which the engine efficiency is maximized.
    """
    eta1 = _nominal_perf(aircraft_mass + 0.5, **perf_kwargs).engine_efficiency
    eta2 = _nominal_perf(aircraft_mass - 0.5, **perf_kwargs).engine_efficiency
    return eta1 - eta2


def _min_mass(
    oem: float,
    lf: float,
    mpm: float,
    reserve_fuel: float,
) -> float:
    """Calculate the minimum mass given OEM, LF, MPM, and reserve fuel."""
    return oem + lf * mpm + reserve_fuel


def _estimate_mass_extremes(
    atyp_param: PSAircraftEngineParams,
    perf_kwargs: dict[str, Any],
    n_iter: int = 3,
) -> tuple[float, float]:
    """Calculate the minimum and maximum mass for a given aircraft type."""

    oem = atyp_param.amass_oew  # operating empty mass
    lf = 0.7  # load factor
    mpm = atyp_param.amass_mpl  # max payload mass
    mtow = atyp_param.amass_mtow  # max takeoff mass

    min_mass = _min_mass(oem, lf, mpm, 0.0)  # no reserve fuel
    for _ in range(n_iter):
        # Estimate the fuel required to cruise at 35,000 ft for 90 minutes.
        # This is used to compute the reserve fuel.
        ff = _nominal_perf(aircraft_mass=min_mass, **perf_kwargs).fuel_flow
        reserve_fuel = ff * 60.0 * 90.0  # 90 minutes
        min_mass = _min_mass(oem, lf, mpm, reserve_fuel)

    # Crude: Assume 2x the fuel flow of cruise for climb
    # Compute the maximum weight at cruise by assuming a 20 minute climb
    ff = _nominal_perf(aircraft_mass=mtow, **perf_kwargs).fuel_flow
    max_mass = mtow - 2.0 * ff * 60.0 * 20.0

    return min_mass, max_mass


def ps_nominal_grid(
    aircraft_type: str,
    *,
    level: npt.NDArray[np.float_] | None = None,
    air_temperature: MetDataArray | npt.NDArray[np.float_] | None = None,
    q_fuel: float = JetA.q_fuel,
    mach_num: float | None = None,
) -> xr.Dataset:
    """Calculate the nominal performance grid for a given aircraft type.

    Parameters
    ----------
    aircraft_type : str
        The aircraft type.
    level : npt.NDArray[np.float_] | None, optional
        The pressure level, [:math:`hPa`]. If None, the ``air_temperature``
        argument must be a :class:`MetDataArray` with a ``level`` coordinate.
    air_temperature : MetDataArray | npt.NDArray[np.float] | None, optional
        The ambient air temperature, [:math:`K`]. If None (default), the ISA
        temperature is computed from the ``level`` argument. If a :class:`MetDataArray`,
        the ``level`` coordinate must be present and the ``level`` argument must be None
        to avoid ambiguity. If a :class:`np.ndarray` is passed, it is assumed to be 1
        dimensional with the same shape as the ``level`` argument.
    q_fuel : float, optional
        The fuel heating value, by default :attr:`JetA.q_fuel`
    mach_num : float | None, optional
        The Mach number. If None (default), the PS design Mach number is used.

    Returns
    -------
    xr.Dataset
        The nominal performance grid. The grid is indexed by altitude and Mach number.
        Contains the following variables:

        - `fuel_flow` : Fuel flow rate, [:math:`kg/s`]
        - `engine_efficiency` : Engine efficiency
        - `aircraft_mass` : Aircraft mass at which the engine efficiency is maximized, [:math:`kg`]
    """
    if isinstance(air_temperature, MetDataArray) and level is not None:
        raise ValueError("If 'air_temperature' is a MetDataArray, 'level' must be None")

    if air_temperature is None and level is None:
        raise ValueError("One of 'level' or 'air_temperature' must be specified")

    if isinstance(air_temperature, MetDataArray):
        level = air_temperature.data["level"]
        air_temperature = air_temperature.data
        air_temperature, level = xr.broadcast(air_temperature, level)
    elif air_temperature is None:
        altitude_m = units.pl_to_m(level)
        air_temperature = units.m_to_T_isa(altitude_m)

    air_pressure = level * 100.0

    aircraft_engine_params = ps_model.load_aircraft_engine_params()
    atyp_param = aircraft_engine_params[aircraft_type]
    mach_num = mach_num or atyp_param.m_des

    perf_kwargs = {
        "atyp_param": atyp_param,
        "air_pressure": air_pressure,
        "air_temperature": air_temperature,
        "q_fuel": q_fuel,
        "mach_num": mach_num,
    }

    min_mass, max_mass = _estimate_mass_extremes(atyp_param, perf_kwargs)
    x0 = np.full_like(air_temperature, (max_mass + min_mass) / 2.0)

    # Choose aircraft mass to maximize engine efficiency
    # This is the critical step of the calculation
    aircraft_mass = scipy.optimize.newton(
        func=_newton_func,
        args=(perf_kwargs,),
        x0=x0,
        tol=1.0,
        disp=False,
    )

    aircraft_mass.clip(min=min_mass, max=max_mass, out=aircraft_mass)

    perf = _nominal_perf(aircraft_mass=aircraft_mass, **perf_kwargs)

    engine_efficiency = perf.engine_efficiency
    fuel_flow = perf.fuel_flow

    attrs = {
        "aircraft_type": aircraft_type,
        "mach_num": mach_num,
        "q_fuel": q_fuel,
    }

    if isinstance(fuel_flow, xr.DataArray):
        return xr.Dataset(
            {
                "aircraft_mass": (fuel_flow.dims, aircraft_mass),
                "engine_efficiency": (fuel_flow.dims, engine_efficiency),
                "fuel_flow": fuel_flow,
            },
            attrs=attrs,
        )

    return xr.Dataset(
        {
            "aircraft_mass": ("level", aircraft_mass),
            "engine_efficiency": ("level", engine_efficiency),
            "fuel_flow": ("level", fuel_flow),
        },
        coords={"level": level},
        attrs=attrs,
    )
