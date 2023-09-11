"""Support for the Poll-Schumann (PS) theoretical aircraft performance over a grid."""

from __future__ import annotations

import dataclasses
import warnings
from typing import Any, overload

import numpy as np
import numpy.typing as npt
import scipy.optimize
import xarray as xr
import xarray.core.coordinates as xrcc

from pycontrails.core.aircraft_performance import (
    AircraftPerformanceGrid,
    AircraftPerformanceGridData,
    AircraftPerformanceGridParams,
)
from pycontrails.core.flight import Flight
from pycontrails.core.fuel import JetA
from pycontrails.core.met import MetDataset
from pycontrails.core.met_var import AirTemperature
from pycontrails.core.vector import GeoVectorDataset
from pycontrails.models.ps_model import ps_model
from pycontrails.models.ps_model.ps_aircraft_params import PSAircraftEngineParams
from pycontrails.physics import units
from pycontrails.utils.types import ArrayOrFloat

# mypy: disable-error-code = type-var


@dataclasses.dataclass
class PSGridParams(AircraftPerformanceGridParams):
    """Parameters for :class:`PSGrid`."""

    #: Passed into :func:`ps_nominal_grid`
    maxiter: int = 10


class PSGrid(AircraftPerformanceGrid):
    """Compute nominal Poll-Schumann aircraft performance over a grid.

    For a given aircraft type, altitude, aircraft mass, air temperature, and
    mach number, the PS model computes a theoretical engine efficiency and fuel
    flow rate for an aircraft under cruise conditions. Letting the aircraft mass
    vary and fixing the other parameters, the engine efficiency curve attains a
    single maximum at a particular aircraft mass. By solving this implicit
    equation, the PS model can be used to compute the aircraft mass that
    maximizes engine efficiency for a given set of parameters. This is the
    "nominal" aircraft mass computed by this model.

    This nominal aircraft mass is not always realizable. For example, the maximum
    engine efficiency may be attained at an aircraft mass that is less than the
    operating empty mass of the aircraft. This model determines the minimum and
    maximum possible aircraft mass for a given set of parameters using a simple
    heuristic. The nominal aircraft mass is then clipped to this range.
    """

    name = "PSGrid"
    long_name = "Poll-Schumann Aircraft Performance evaluated at arbitrary points"
    met_variables = (AirTemperature,)
    default_params = PSGridParams

    met: MetDataset
    source: GeoVectorDataset

    @overload
    def eval(self, source: GeoVectorDataset, **params: Any) -> GeoVectorDataset: ...

    @overload
    def eval(self, source: MetDataset | None = ..., **params: Any) -> MetDataset: ...

    def eval(
        self, source: GeoVectorDataset | MetDataset | None = None, **params: Any
    ) -> GeoVectorDataset | MetDataset:
        """Evaluate the PS model over a :class:`MetDataset` or :class:`GeoVectorDataset`.

        Parameters
        ----------
        source : MetDataset or GeoVectorDataset, optional
            The source data to use for the evaluation. If None, the source is taken
            from the :attr:`met` attribute of the :class:`PSGrid` instance.
            The aircraft type is taken from ``source.attrs["aircraft_type]``. If this field
            is not present, ``params["aircraft_type"]`` is used instead. See the
            static CSV file :file:`ps-aircraft-params-20230517.csv` for a list of supported
            aircraft types.
        **params : Any
            Override the default parameters of the :class:`PSGrid` instance.

        Returns
        -------
        GeoVectorDataset or MetDataset
            The source data with the following variables added:

                - aircraft_mass
                - fuel_flow
                - engine_efficiency

        See Also
        --------
        :func:`ps_nominal_grid`
        """
        self.update_params(**params)
        self.set_source(source)
        self.require_source_type((GeoVectorDataset, MetDataset))
        self.set_source_met()

        # Check some assumptions
        if "true_airspeed" in self.source or "true_airspeed" in self.source.attrs:
            raise NotImplementedError(
                "PSGrid currently only supports setting a 'mach_number' parameter."
            )
        if self.get_source_param("aircraft_mass", set_attr=False) is not None:
            raise NotImplementedError("The 'aircraft_mass' parameter must be None.")
        if isinstance(self.source, Flight):
            warnings.warn(
                "The 'PSGrid' model is not intended to support 'Flight' objects as 'source' "
                "data. Instead, use the 'PSFlight' model."
            )

        # Extract the relevant source data
        try:
            aircraft_type = self.source.attrs["aircraft_type"]
        except KeyError:
            aircraft_type = self.params["aircraft_type"]
            self.source.attrs["aircraft_type"] = aircraft_type

        fuel = self.source.attrs.get("fuel", self.params["fuel"])
        q_fuel = fuel.q_fuel
        mach_number = self.get_source_param("mach_number", set_attr=False)

        if isinstance(self.source, MetDataset):
            if "fuel_flow" in self.source:
                raise NotImplementedError("PSGrid doesn't support custom 'fuel_flow' values.")
            if "engine_efficiency" in self.source:
                raise NotImplementedError(
                    "PSGrid doesn't support custom 'engine_efficiency' values."
                )

            ds = ps_nominal_grid(
                aircraft_type,
                air_temperature=self.source.data["air_temperature"],
                q_fuel=q_fuel,
                mach_number=mach_number,
                maxiter=self.params["maxiter"],
            )
            return MetDataset(ds)

        air_temperature = self.source["air_temperature"]
        ds = ps_nominal_grid(
            aircraft_type,
            level=self.source.level,
            air_temperature=air_temperature,
            q_fuel=q_fuel,
            mach_number=mach_number,
            maxiter=self.params["maxiter"],
        )

        # Set the source data
        self.source.setdefault("aircraft_mass", ds["aircraft_mass"])
        self.source.setdefault("fuel_flow", ds["fuel_flow"])
        self.source.setdefault("engine_efficiency", ds["engine_efficiency"])
        mach_number = self.source.attrs.setdefault("mach_number", ds.attrs["mach_number"])
        self.source["true_airspeed"] = units.mach_number_to_tas(mach_number, air_temperature)
        self.source.attrs.setdefault("wingspan", ds.attrs["wingspan"])
        self.source.attrs.setdefault("n_engine", ds.attrs["n_engine"])

        return self.source


@dataclasses.dataclass
class _PerfVariables:
    atyp_param: PSAircraftEngineParams
    air_pressure: npt.NDArray[np.float_] | float
    air_temperature: npt.NDArray[np.float_] | float
    mach_number: npt.NDArray[np.float_] | float
    q_fuel: float


def _nominal_perf(aircraft_mass: ArrayOrFloat, perf: _PerfVariables) -> AircraftPerformanceGridData:
    """Compute nominal Poll-Schumann aircraft performance."""

    atyp_param = perf.atyp_param
    air_pressure = perf.air_pressure
    air_temperature = perf.air_temperature
    mach_number = perf.mach_number
    q_fuel = perf.q_fuel

    theta = 0.0
    dv_dt = 0.0

    rn = ps_model.reynolds_number(
        atyp_param.wing_surface_area, mach_number, air_temperature, air_pressure
    )

    c_lift = ps_model.lift_coefficient(
        atyp_param.wing_surface_area, aircraft_mass, air_pressure, mach_number, theta
    )
    c_f = ps_model.skin_friction_coefficient(rn)
    c_drag_0 = ps_model.zero_lift_drag_coefficient(c_f, atyp_param.psi_0)
    e_ls = ps_model.oswald_efficiency_factor(c_drag_0, atyp_param)
    c_drag_w = ps_model.wave_drag_coefficient(mach_number, c_lift, atyp_param)
    c_drag = ps_model.airframe_drag_coefficient(
        c_drag_0, c_drag_w, c_lift, e_ls, atyp_param.wing_aspect_ratio
    )

    thrust = ps_model.thrust_force(aircraft_mass, c_lift, c_drag, dv_dt, theta)

    c_t = ps_model.engine_thrust_coefficient(
        thrust, mach_number, air_pressure, atyp_param.wing_surface_area
    )

    engine_efficiency = ps_model.overall_propulsion_efficiency(mach_number, c_t, atyp_param)

    fuel_flow = ps_model.fuel_mass_flow_rate(
        air_pressure,
        air_temperature,
        mach_number,
        c_t,
        engine_efficiency,
        atyp_param.wing_surface_area,
        q_fuel,
    )

    return AircraftPerformanceGridData(
        fuel_flow=fuel_flow,
        engine_efficiency=engine_efficiency,
    )


def _newton_func(aircraft_mass: ArrayOrFloat, perf: _PerfVariables) -> ArrayOrFloat:
    """Approximate the derivative of the engine efficiency with respect to mass.

    This is used to find the mass at which the engine efficiency is maximized.
    """
    eta1 = _nominal_perf(aircraft_mass + 0.5, perf).engine_efficiency
    eta2 = _nominal_perf(aircraft_mass - 0.5, perf).engine_efficiency
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
    perf: _PerfVariables,
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
        ff = _nominal_perf(min_mass, perf).fuel_flow
        reserve_fuel = ff * 60.0 * 90.0  # 90 minutes
        min_mass = _min_mass(oem, lf, mpm, reserve_fuel)

    # Crude: Assume 2x the fuel flow of cruise for climb
    # Compute the maximum weight at cruise by assuming a 20 minute climb
    ff = _nominal_perf(mtow, perf).fuel_flow
    max_mass = mtow - 2.0 * ff * 60.0 * 20.0

    return min_mass, max_mass


def _parse_variables(
    level: npt.NDArray[np.float_] | None,
    air_temperature: xr.DataArray | npt.NDArray[np.float_] | None,
) -> tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]]:
    """Parse the level and air temperature arguments."""

    if isinstance(air_temperature, xr.DataArray):
        if level is not None:
            raise ValueError("If 'air_temperature' is a DataArray, 'level' must be None")

        level_da = air_temperature["level"]
        air_temperature, level_da = xr.broadcast(air_temperature, level_da)
        return np.asarray(level_da), np.asarray(air_temperature)

    if air_temperature is None:
        if level is None:
            raise ValueError("The 'level' argument must be specified")
        altitude_m = units.pl_to_m(level)
        air_temperature = units.m_to_T_isa(altitude_m)
        return level, air_temperature

    if level is None:
        raise ValueError("The 'level' argument must be specified")

    return level, air_temperature


def ps_nominal_grid(
    aircraft_type: str,
    *,
    level: npt.NDArray[np.float_] | None = None,
    air_temperature: xr.DataArray | npt.NDArray[np.float_] | None = None,
    q_fuel: float = JetA.q_fuel,
    mach_number: float | None = None,
    maxiter: int = PSGridParams.maxiter,
) -> xr.Dataset:
    """Calculate the nominal performance grid for a given aircraft type.

    This function is similar to the :class:`PSGrid` model, but it doesn't require
    meteorological data. Instead, the ambient air temperature can be computed from
    the ISA model or passed as an argument.

    Parameters
    ----------
    aircraft_type : str
        The aircraft type.
    level : npt.NDArray[np.float_] | None, optional
        The pressure level, [:math:`hPa`]. If None, the ``air_temperature``
        argument must be a :class:`xr.DataArray` with a ``level`` coordinate.
    air_temperature : xr.DataArray | npt.NDArray[np.float] | None, optional
        The ambient air temperature, [:math:`K`]. If None (default), the ISA
        temperature is computed from the ``level`` argument. If a :class:`xr.DataArray`,
        the ``level`` coordinate must be present and the ``level`` argument must be None
        to avoid ambiguity. If a :class:`np.ndarray` is passed, it is assumed to be 1
        dimensional with the same shape as the ``level`` argument.
    q_fuel : float, optional
        The fuel heating value, by default :attr:`JetA.q_fuel`
    mach_number : float | None, optional
        The Mach number. If None (default), the PS design Mach number is used.
    maxiter : int, optional
        Passed into :func:`scipy.optimize.newton`.

    Returns
    -------
    xr.Dataset
        The nominal performance grid. The grid is indexed by altitude and Mach number.
        Contains the following variables:

        - `fuel_flow` : Fuel flow rate, [:math:`kg/s`]
        - `engine_efficiency` : Engine efficiency
        - `aircraft_mass` : Aircraft mass at which the engine efficiency is maximized, [:math:`kg`]

    Examples
    --------
    >>> level = np.arange(200, 300, 10, dtype=float)

    >>> # Compute nominal aircraft performance assuming ISA conditions
    >>> # and the design Mach number
    >>> perf = ps_nominal_grid("A320", level=level)
    >>> perf.attrs["mach_number"]
    0.753

    >>> perf.to_dataframe()
           aircraft_mass  engine_efficiency  fuel_flow
    level
    200.0   58564.031595           0.319072   0.544350
    210.0   61772.755626           0.319072   0.571567
    220.0   64992.059200           0.319072   0.598784
    230.0   68196.058808           0.319072   0.626964
    240.0   71364.841888           0.319072   0.656877
    250.0   71872.880280           0.318921   0.664426
    260.0   71863.586336           0.318422   0.669074
    270.0   71850.890419           0.317680   0.674991
    280.0   71835.413315           0.316775   0.681955
    290.0   71817.627308           0.315763   0.689795

    >>> # Now compute it for a higher Mach number
    >>> perf = ps_nominal_grid("A320", level=level, mach_number=0.78)
    >>> perf.to_dataframe()
           aircraft_mass  engine_efficiency  fuel_flow
    level
    200.0   57749.067052           0.324993   0.560391
    210.0   60819.870041           0.324994   0.587377
    220.0   64021.435191           0.324994   0.615348
    230.0   67205.781634           0.324994   0.644307
    240.0   70351.196513           0.324994   0.675047
    250.0   71805.902720           0.324938   0.691873
    260.0   71793.562851           0.324584   0.697880
    270.0   71777.532949           0.323995   0.705239
    280.0   71758.561934           0.323248   0.713689
    290.0   71737.199210           0.322395   0.723033
    """
    coords: dict[str, Any] | xrcc.DataArrayCoordinates
    if isinstance(air_temperature, xr.DataArray):
        dims = air_temperature.dims
        coords = air_temperature.coords
    else:
        dims = ("level",)
        coords = {"level": level}

    level, air_temperature = _parse_variables(level, air_temperature)

    air_pressure = level * 100.0

    aircraft_engine_params = ps_model.load_aircraft_engine_params()

    try:
        atyp_param = aircraft_engine_params[aircraft_type]
    except KeyError as exc:
        raise KeyError(
            f"The aircraft type {aircraft_type} is not currently supported by the PS model. "
            f"Available aircraft types are: {list(aircraft_engine_params)}"
        ) from exc

    mach_number = mach_number or atyp_param.m_des

    perf = _PerfVariables(
        atyp_param=atyp_param,
        air_pressure=air_pressure,
        air_temperature=air_temperature,
        mach_number=mach_number,
        q_fuel=q_fuel,
    )

    min_mass, max_mass = _estimate_mass_extremes(atyp_param, perf)
    x0 = np.full_like(air_temperature, (max_mass + min_mass) / 2.0)

    # Choose aircraft mass to maximize engine efficiency
    # This is the critical step of the calculation
    aircraft_mass = scipy.optimize.newton(
        func=_newton_func,
        args=(perf,),
        x0=x0,
        tol=1.0,
        disp=False,
        maxiter=maxiter,
    )

    # scipy.optimize.newton promotes float32 to float64
    aircraft_mass = aircraft_mass.astype(x0.dtype, copy=False)

    aircraft_mass.clip(min=min_mass, max=max_mass, out=aircraft_mass)

    output = _nominal_perf(aircraft_mass, perf)

    engine_efficiency = output.engine_efficiency
    fuel_flow = output.fuel_flow

    attrs = {
        "aircraft_type": aircraft_type,
        "mach_number": mach_number,
        "q_fuel": q_fuel,
        "wingspan": atyp_param.wing_span,
        "n_engine": atyp_param.n_engine,
    }

    return xr.Dataset(
        {
            "aircraft_mass": (dims, aircraft_mass),
            "engine_efficiency": (dims, engine_efficiency),
            "fuel_flow": (dims, fuel_flow),
        },
        coords=coords,
        attrs=attrs,
    )
