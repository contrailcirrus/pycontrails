"""Support for the Poll-Schumann (PS) aircraft performance model."""

from __future__ import annotations

import dataclasses
import functools
import pathlib
import sys
from collections.abc import Mapping
from typing import Any

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

import numpy as np
import numpy.typing as npt
import pandas as pd

from pycontrails.core import flight
from pycontrails.core.aircraft_performance import (
    DEFAULT_LOAD_FACTOR,
    AircraftPerformance,
    AircraftPerformanceData,
    AircraftPerformanceParams,
)
from pycontrails.core.flight import Flight
from pycontrails.core.met import MetDataset
from pycontrails.models.ps_model import ps_operational_limits as ps_lims
from pycontrails.models.ps_model.ps_aircraft_params import (
    PSAircraftEngineParams,
    load_aircraft_engine_params,
)
from pycontrails.physics import constants, jet, units
from pycontrails.utils.types import ArrayOrFloat

# mypy: disable-error-code = "type-var, arg-type"

#: Path to the Poll-Schumann aircraft parameters CSV file.
PS_SYNONYM_FILE_PATH = pathlib.Path(__file__).parent / "static" / "ps-synonym-list-20250328.csv"


@dataclasses.dataclass
class PSFlightParams(AircraftPerformanceParams):
    """Default parameters for :class:`PSFlight`."""

    #: Clip the ratio of the overall propulsion efficiency to the maximum propulsion
    #: efficiency to always exceed this value.
    eta_over_eta_b_min: float | None = 0.5


class PSFlight(AircraftPerformance):
    """Simulate aircraft performance using Poll-Schumann (PS) model.

    References
    ----------
    :cite:`pollEstimationMethodFuel2021`
    :cite:`pollEstimationMethodFuel2021a`

    Poll & Schumann (2022). An estimation method for the fuel burn and other performance
    characteristics of civil transport aircraft. Part 3 Generalisation to cover climb,
    descent and holding. Aero. J., submitted.

    See Also
    --------
    pycontrails.physics.jet.aircraft_load_factor
    """

    name = "PSFlight"
    long_name = "Poll-Schumann Aircraft Performance Model"
    default_params = PSFlightParams

    aircraft_engine_params: Mapping[str, PSAircraftEngineParams]

    def __init__(
        self,
        met: MetDataset | None = None,
        params: dict[str, Any] | None = None,
        **params_kwargs: Any,
    ) -> None:
        super().__init__(met=met, params=params, **params_kwargs)
        self.aircraft_engine_params = load_aircraft_engine_params(
            self.params["engine_deterioration_factor"]
        )
        self.synonym_dict = get_aircraft_synonym_dict_ps()

    def check_aircraft_type_availability(
        self, aircraft_type: str, raise_error: bool = True
    ) -> bool:
        """Check if aircraft type designator is available in the PS model database.

        Parameters
        ----------
        aircraft_type : str
            ICAO aircraft type designator
        raise_error : bool, optional
            Optional flag for raising an error, by default True.

        Returns
        -------
        bool
            Aircraft found in the PS model database.

        Raises
        ------
        KeyError
            raises KeyError if the aircraft type is not covered by database
        """
        if aircraft_type in self.aircraft_engine_params or aircraft_type in self.synonym_dict:
            return True
        if raise_error:
            msg = f"Aircraft type {aircraft_type} not covered by the PS model."
            raise KeyError(msg)
        return False

    @override
    def eval_flight(self, fl: Flight) -> Flight:
        # Ensure aircraft type is available
        try:
            aircraft_type = fl.get_constant("aircraft_type")
        except KeyError as exc:
            msg = "`aircraft_type` required on flight attrs"
            raise KeyError(msg) from exc

        try:
            atyp_ps = self.synonym_dict.get(aircraft_type) or aircraft_type
            aircraft_params = self.aircraft_engine_params[atyp_ps]
        except KeyError as exc:
            msg = f"Aircraft type {aircraft_type} not covered by the PS model."
            raise KeyError(msg) from exc

        # Set flight attributes based on engine, if they aren't already defined
        fl.attrs.setdefault("aircraft_performance_model", self.name)
        fl.attrs.setdefault("aircraft_type_ps", atyp_ps)
        fl.attrs.setdefault("n_engine", aircraft_params.n_engine)

        fl.attrs.setdefault("wingspan", aircraft_params.wing_span)
        fl.attrs.setdefault("max_mach", aircraft_params.max_mach_num)
        fl.attrs.setdefault("max_altitude", units.ft_to_m(aircraft_params.fl_max * 100.0))
        fl.attrs.setdefault("n_engine", aircraft_params.n_engine)

        amass_oew = fl.attrs.get("amass_oew", aircraft_params.amass_oew)
        amass_mtow = fl.attrs.get("amass_mtow", aircraft_params.amass_mtow)
        amass_mpl = fl.attrs.get("amass_mpl", aircraft_params.amass_mpl)
        load_factor = fl.attrs.get("load_factor", DEFAULT_LOAD_FACTOR)
        takeoff_mass = fl.attrs.get("takeoff_mass")
        q_fuel = fl.fuel.q_fuel

        true_airspeed = fl["true_airspeed"]  # attached in PSFlight.eval
        true_airspeed = np.where(true_airspeed == 0.0, np.nan, true_airspeed)

        # Run the simulation
        aircraft_performance = self.simulate_fuel_and_performance(
            aircraft_type=atyp_ps,
            altitude_ft=fl.altitude_ft,
            time=fl["time"],
            true_airspeed=true_airspeed,
            air_temperature=fl["air_temperature"],
            aircraft_mass=self.get_data_param(fl, "aircraft_mass", None),
            thrust=self.get_data_param(fl, "thrust", None),
            engine_efficiency=self.get_data_param(fl, "engine_efficiency", None),
            fuel_flow=self.get_data_param(fl, "fuel_flow", None),
            q_fuel=q_fuel,
            n_iter=self.params["n_iter"],
            amass_oew=amass_oew,
            amass_mtow=amass_mtow,
            amass_mpl=amass_mpl,
            load_factor=load_factor,
            takeoff_mass=takeoff_mass,
            correct_fuel_flow=self.params["correct_fuel_flow"],
        )

        # Set array aircraft_performance to flight, don't overwrite
        for var in (
            "aircraft_mass",
            "engine_efficiency",
            "fuel_flow",
            "fuel_burn",
            "thrust",
            "rocd",
        ):
            fl.setdefault(var, getattr(aircraft_performance, var))

        self._cleanup_indices()

        fl.attrs["total_fuel_burn"] = np.nansum(aircraft_performance.fuel_burn).item()

        return fl

    @override
    def calculate_aircraft_performance(
        self,
        *,
        aircraft_type: str,
        altitude_ft: npt.NDArray[np.floating],
        air_temperature: npt.NDArray[np.floating],
        time: npt.NDArray[np.datetime64] | None,
        true_airspeed: npt.NDArray[np.floating] | float | None,
        aircraft_mass: npt.NDArray[np.floating] | float,
        engine_efficiency: npt.NDArray[np.floating] | float | None,
        fuel_flow: npt.NDArray[np.floating] | float | None,
        thrust: npt.NDArray[np.floating] | float | None,
        q_fuel: float,
        **kwargs: Any,
    ) -> AircraftPerformanceData:
        try:
            correct_fuel_flow = kwargs["correct_fuel_flow"]
        except KeyError as exc:
            msg = "A 'correct_fuel_flow' kwarg is required for this model"
            raise KeyError(msg) from exc

        if not isinstance(true_airspeed, np.ndarray):
            msg = "Only array inputs are supported"
            raise NotImplementedError(msg)

        atyp_param = self.aircraft_engine_params[aircraft_type]

        # Atmospheric quantities
        air_pressure = units.ft_to_pl(altitude_ft) * 100.0

        # Clip unrealistically high true airspeed
        max_mach = ps_lims.max_mach_number_by_altitude(
            altitude_ft,
            air_pressure,
            atyp_param.max_mach_num,
            atyp_param.p_i_max,
            atyp_param.p_inf_co,
            atm_speed_limit=False,
            buffer=0.02,
        )
        true_airspeed, mach_num = jet.clip_mach_number(true_airspeed, air_temperature, max_mach)

        # Reynolds number
        rn = reynolds_number(atyp_param.wing_surface_area, mach_num, air_temperature, air_pressure)

        # Allow array or None time
        dv_dt: npt.NDArray[np.floating] | float
        theta: npt.NDArray[np.floating] | float
        if time is None:
            # Assume a nominal cruising state
            dt_sec = None
            rocd = np.zeros_like(altitude_ft)
            dv_dt = 0.0
            theta = 0.0

        elif isinstance(time, np.ndarray):
            dt_sec = flight.segment_duration(time, dtype=altitude_ft.dtype)
            rocd = flight.segment_rocd(dt_sec, altitude_ft, air_temperature)
            dv_dt = jet.acceleration(true_airspeed, dt_sec)
            theta = jet.climb_descent_angle(true_airspeed, rocd)

        else:
            msg = "Only array inputs are supported"
            raise NotImplementedError(msg)

        # Aircraft performance parameters
        c_lift = lift_coefficient(
            atyp_param.wing_surface_area, aircraft_mass, air_pressure, mach_num, theta
        )
        c_f = skin_friction_coefficient(rn)
        c_drag_0 = zero_lift_drag_coefficient(c_f, atyp_param.psi_0)
        e_ls = oswald_efficiency_factor(c_drag_0, atyp_param)
        c_drag_w = wave_drag_coefficient(mach_num, c_lift, atyp_param)
        c_drag = airframe_drag_coefficient(
            c_drag_0, c_drag_w, c_lift, e_ls, atyp_param.wing_aspect_ratio
        )

        # Engine parameters and fuel consumption
        if thrust is None:
            thrust = thrust_force(aircraft_mass, c_lift, c_drag, dv_dt, theta)

        c_t = engine_thrust_coefficient(
            thrust, mach_num, air_pressure, atyp_param.wing_surface_area
        )
        c_t_eta_b = thrust_coefficient_at_max_efficiency(
            mach_num, atyp_param.m_des, atyp_param.c_t_des
        )

        if correct_fuel_flow:
            c_t_available = ps_lims.max_available_thrust_coefficient(
                air_temperature, mach_num, c_t_eta_b, atyp_param
            )
            np.clip(c_t, 0.0, c_t_available, out=c_t)

        if engine_efficiency is None:
            engine_efficiency = overall_propulsion_efficiency(
                mach_num, c_t, c_t_eta_b, atyp_param, self.params["eta_over_eta_b_min"]
            )

        if fuel_flow is None:
            fuel_flow = fuel_mass_flow_rate(
                air_pressure,
                air_temperature,
                mach_num,
                c_t,
                engine_efficiency,
                atyp_param.wing_surface_area,
                q_fuel,
            )
        elif isinstance(fuel_flow, int | float):
            fuel_flow = np.full_like(true_airspeed, fuel_flow)

        # Flight phase
        segment_duration = flight.segment_duration(time, dtype=altitude_ft.dtype)
        rocd = flight.segment_rocd(segment_duration, altitude_ft, air_temperature)

        if correct_fuel_flow:
            flight_phase = flight.segment_phase(rocd, altitude_ft)
            fuel_flow = fuel_flow_correction(
                fuel_flow,
                altitude_ft,
                air_temperature,
                air_pressure,
                mach_num,
                atyp_param.ff_idle_sls,
                atyp_param.ff_max_sls,
                flight_phase,
            )

        if dt_sec is not None:
            fuel_burn = jet.fuel_burn(fuel_flow, dt_sec)
        else:
            fuel_burn = np.full_like(fuel_flow, np.nan)

        # XXX: Explicitly broadcast scalar inputs as needed to keep a consistent
        # output spec.
        if isinstance(aircraft_mass, int | float):
            aircraft_mass = np.full_like(true_airspeed, aircraft_mass)
        if isinstance(engine_efficiency, int | float):
            engine_efficiency = np.full_like(true_airspeed, engine_efficiency)
        if isinstance(thrust, int | float):
            thrust = np.full_like(true_airspeed, thrust)

        return AircraftPerformanceData(
            fuel_flow=fuel_flow,
            aircraft_mass=aircraft_mass,
            true_airspeed=true_airspeed,
            thrust=thrust,
            fuel_burn=fuel_burn,
            engine_efficiency=engine_efficiency,
            rocd=rocd,
        )


# ----------------------
# Atmospheric parameters
# ----------------------


def reynolds_number(
    wing_surface_area: float,
    mach_num: ArrayOrFloat,
    air_temperature: ArrayOrFloat,
    air_pressure: ArrayOrFloat,
) -> ArrayOrFloat:
    """
    Calculate Reynolds number.

    Parameters
    ----------
    wing_surface_area : float
        Aircraft wing surface area, [:math:`m^2`]
    mach_num : ArrayOrFloat
        Mach number at each waypoint
    air_temperature : ArrayOrFloat
        Ambient temperature at each waypoint, [:math:`K`]
    air_pressure: ArrayOrFloat
        Ambient pressure, [:math:`Pa`]

    Returns
    -------
    ArrayOrFloat
        Reynolds number at each waypoint

    References
    ----------
    Eq. (3) of :cite:`pollEstimationMethodFuel2021`.
    """
    mu = fluid_dynamic_viscosity(air_temperature)
    return (
        wing_surface_area**0.5
        * mach_num
        * (air_pressure / mu)
        * (constants.kappa / (constants.R_d * air_temperature)) ** 0.5
    )


def fluid_dynamic_viscosity(air_temperature: ArrayOrFloat) -> ArrayOrFloat:
    """
    Calculate fluid dynamic viscosity.

    Parameters
    ----------
    air_temperature : ArrayOrFloat
        Ambient temperature at each waypoint, [:math:`K`]

    Returns
    -------
    ArrayOrFloat
        Fluid dynamic viscosity, [:math:`kg m^{-1} s^{-1}`]

    Notes
    -----
    The dynamic viscosity is a measure of the fluid's resistance to flow and is represented
    by Sutherland's Law. The higher the viscosity, the thicker the fluid.

    References
    ----------
    Eq. (25) of :cite:`pollEstimationMethodFuel2021`.
    """
    return 1.458e-6 * (air_temperature**1.5) / (110.4 + air_temperature)


# -------------------------------
# Lift and drag coefficients
# -------------------------------


def lift_coefficient(
    wing_surface_area: float,
    aircraft_mass: ArrayOrFloat,
    air_pressure: ArrayOrFloat,
    mach_num: ArrayOrFloat,
    climb_angle: ArrayOrFloat,
) -> ArrayOrFloat:
    r"""Calculate the lift coefficient.

    This quantity is a dimensionless coefficient that relates the lift generated
    by a lifting body to the fluid density around the body, the fluid velocity,
    and an associated reference area.

    Parameters
    ----------
    wing_surface_area : float
        Aircraft wing surface area, [:math:`m^2`]
    aircraft_mass : ArrayOrFloat
        Aircraft mass, [:math:`kg`]
    air_pressure: ArrayOrFloat
        Ambient pressure, [:math:`Pa`]
    mach_num : ArrayOrFloat
        Mach number at each waypoint
    climb_angle : ArrayOrFloat
        Angle between the horizontal plane and the actual flight path, [:math:`\deg`]

    Returns
    -------
    ArrayOrFloat
        Lift coefficient

    Notes
    -----
    The lift force is perpendicular to the flight direction, while the
    aircraft weight acts vertically.

    References
    ----------
    Eq. (5) of :cite:`pollEstimationMethodFuel2021`.
    """
    lift_force = aircraft_mass * constants.g * np.cos(units.degrees_to_radians(climb_angle))
    denom = (constants.kappa / 2.0) * air_pressure * mach_num**2 * wing_surface_area
    return lift_force / denom


def skin_friction_coefficient(rn: ArrayOrFloat) -> ArrayOrFloat:
    """Calculate aircraft skin friction coefficient.

    Parameters
    ----------
    rn: ArrayOrFloat
        Reynolds number

    Returns
    -------
    ArrayOrFloat
        Skin friction coefficient.

    Notes
    -----
    The skin friction coefficient, a dimensionless quantity, is used to estimate the
    skin friction drag, which is the resistance force exerted on an object moving in
    a fluid. Given that aircraft at cruise generally experience a narrow range of
    Reynolds number of between 3E7 and 3E8, it is approximated using a simple power law.

    References
    ----------
    Eq. (28) of :cite:`pollEstimationMethodFuel2021`.
    """
    return 0.0269 / (rn**0.14)


def zero_lift_drag_coefficient(c_f: ArrayOrFloat, psi_0: float) -> ArrayOrFloat:
    """Calculate zero-lift drag coefficient.

    Parameters
    ----------
    c_f : ArrayOrFloat
        Skin friction coefficient
    psi_0 : float
        Aircraft geometry drag parameter

    Returns
    -------
    ArrayOrFloat
        Zero-lift drag coefficient (c_d_0)

    References
    ----------
    Eq. (9) of :cite:`pollEstimationMethodFuel2021`.
    """
    return c_f * psi_0


def oswald_efficiency_factor(
    c_drag_0: ArrayOrFloat, atyp_param: PSAircraftEngineParams
) -> ArrayOrFloat:
    """Calculate Oswald efficiency factor.

    The Oswald efficiency factor captures all the lift-dependent drag effects, including
    the vortex drag on the wing (primary source), tailplane and fuselage, and is a function
    of aircraft geometry and the zero-lift drag coefficient.

    Parameters
    ----------
    c_drag_0 : ArrayOrFloat
        Zero-lift drag coefficient.
    atyp_param : PSAircraftEngineParams
        Extracted aircraft and engine parameters.

    Returns
    -------
    ArrayOrFloat
        Oswald efficiency factor (e_ls)

    References
    ----------
    Eq. (12) of :cite:`pollEstimationMethodFuel2021`.
    """
    numer = 1.075 if atyp_param.winglets else 1.0
    k_1 = _non_vortex_lift_dependent_drag_factor(c_drag_0, atyp_param.cos_sweep)
    denom = 1.0 + 0.03 + atyp_param.delta_2 + (k_1 * np.pi * atyp_param.wing_aspect_ratio)
    return numer / denom


def _non_vortex_lift_dependent_drag_factor(
    c_drag_0: ArrayOrFloat, cos_sweep: float
) -> ArrayOrFloat:
    """Calculate non-vortex lift-dependent drag factor.

    Parameters
    ----------
    c_drag_0 : ArrayOrFloat
        Zero-lift drag coefficient
    cos_sweep : float
        Cosine of wing sweep angle measured at the 1/4 chord line

    Returns
    -------
    ArrayOrFloat
        Miscellaneous lift-dependent drag factor (k_1)

    References
    ----------
    Eq. (13) of :cite:`pollEstimationMethodFuel2021a`.
    """
    return 0.8 * (1.0 - 0.53 * cos_sweep) * c_drag_0


def wave_drag_coefficient(
    mach_num: ArrayOrFloat,
    c_lift: ArrayOrFloat,
    atyp_param: PSAircraftEngineParams,
) -> ArrayOrFloat:
    """Calculate wave drag coefficient.

    Parameters
    ----------
    mach_num : ArrayOrFloat
        Mach number at each waypoint
    c_lift : ArrayOrFloat
        Zero-lift drag coefficient
    atyp_param : PSAircraftEngineParams
        Extracted aircraft and engine parameters.

    Returns
    -------
    ArrayOrFloat
        Wave drag coefficient (c_d_w)

    Notes
    -----
    The wave drag coefficient captures all the drag resulting from compressibility,
    the development of significant regions of supersonic flow at the wing surface,
    and the formation of shock waves.
    """
    m_cc = atyp_param.wing_constant - 0.10 * (c_lift / atyp_param.cos_sweep**2)
    x = mach_num * atyp_param.cos_sweep / m_cc

    c_d_w = np.where(
        x < atyp_param.j_2,
        0.0,
        atyp_param.cos_sweep**3 * atyp_param.j_1 * (x - atyp_param.j_2) ** 2,
    )

    return np.where(  # type: ignore[return-value]
        x < atyp_param.x_ref, c_d_w, c_d_w + atyp_param.j_3 * (x - atyp_param.x_ref) ** 4
    )


def airframe_drag_coefficient(
    c_drag_0: ArrayOrFloat,
    c_drag_w: ArrayOrFloat,
    c_lift: ArrayOrFloat,
    e_ls: ArrayOrFloat,
    wing_aspect_ratio: float,
) -> ArrayOrFloat:
    """Calculate total airframe drag coefficient.

    Parameters
    ----------
    c_drag_0 : ArrayOrFloat
        Zero-lift drag coefficient
    c_drag_w : ArrayOrFloat
        Wave drag coefficient
    c_lift : ArrayOrFloat
        Lift coefficient
    e_ls : ArrayOrFloat
        Oswald efficiency factor
    wing_aspect_ratio : float
        Wing aspect ratio

    Returns
    -------
    ArrayOrFloat
        Total airframe drag coefficient

    References
    ----------
    Eq. (8) of :cite:`pollEstimationMethodFuel2021`.
    """
    k = _low_speed_lift_dependent_drag_factor(e_ls, wing_aspect_ratio)
    return c_drag_0 + (k * c_lift**2) + c_drag_w


def _low_speed_lift_dependent_drag_factor(
    e_ls: ArrayOrFloat, wing_aspect_ratio: float
) -> ArrayOrFloat:
    """Calculate low-speed lift-dependent drag factor.

    Parameters
    ----------
    e_ls : ArrayOrFloat
        Oswald efficiency factor
    wing_aspect_ratio : float
        Wing aspect ratio

    Returns
    -------
    ArrayOrFloat
        Low-speed lift-dependent drag factor, K term used to calculate the total
        airframe drag coefficient.
    """
    return 1.0 / (np.pi * wing_aspect_ratio * e_ls)


# -------------------
# Engine parameters
# -------------------


def thrust_force(
    aircraft_mass: ArrayOrFloat,
    c_l: ArrayOrFloat,
    c_d: ArrayOrFloat,
    dv_dt: ArrayOrFloat,
    theta: ArrayOrFloat,
) -> ArrayOrFloat:
    r"""Calculate thrust force summed over all engines.

    Parameters
    ----------
    aircraft_mass : ArrayOrFloat
        Aircraft mass at each waypoint, [:math:`kg`]
    c_l : ArrayOrFloat
        Lift coefficient
    c_d : ArrayOrFloat
        Total airframe drag coefficient
    dv_dt : ArrayOrFloat
        Acceleration/deceleration at each waypoint, [:math:`m \ s^{-2}`]
    theta : ArrayOrFloat
        Climb (positive value) or descent (negative value) angle, [:math:`\deg`]

    Returns
    -------
    ArrayOrFloat
        Thrust force summed over all engines, [:math:`N`]

    Notes
    -----
    - The lift-to-drag ratio is calculated using ``c_l`` and ``c_d``,
    - The first term ``(mg * cos(theta) * (D/L))`` represents the drag force,
    - The second term ``(mg * sin(theta))`` represents the aircraft weight acting on
      the flight direction,
    - The third term ``(m * a)`` represents the force required to accelerate the aircraft.

    References
    ----------
    Eq. (95) of :cite:`pollEstimationMethodFuel2021a`.
    """
    theta = units.degrees_to_radians(theta)
    f_thrust = (
        (aircraft_mass * constants.g * np.cos(theta) * (c_d / c_l))
        + (aircraft_mass * constants.g * np.sin(theta))
        + aircraft_mass * dv_dt
    )
    return f_thrust.clip(min=0.0)


def engine_thrust_coefficient(
    f_thrust: ArrayOrFloat,
    mach_num: ArrayOrFloat,
    air_pressure: ArrayOrFloat,
    wing_surface_area: float,
) -> ArrayOrFloat:
    """Calculate engine thrust coefficient.

    Parameters
    ----------
    f_thrust : ArrayOrFloat
        Thrust force summed over all engines, [:math:`N`]
    mach_num : ArrayOrFloat
        Mach number at each waypoint
    air_pressure : ArrayOrFloat
        Ambient pressure, [:math:`Pa`]
    wing_surface_area : float
        Aircraft wing surface area, [:math:`m^2`]

    Returns
    -------
    ArrayOrFloat
        Engine thrust coefficient (c_t)
    """
    return f_thrust / (0.5 * constants.kappa * air_pressure * mach_num**2 * wing_surface_area)


def overall_propulsion_efficiency(
    mach_num: ArrayOrFloat,
    c_t: ArrayOrFloat,
    c_t_eta_b: ArrayOrFloat,
    atyp_param: PSAircraftEngineParams,
    eta_over_eta_b_min: float | None = None,
) -> npt.NDArray[np.floating]:
    """Calculate overall propulsion efficiency.

    Parameters
    ----------
    mach_num : ArrayOrFloat
        Mach number at each waypoint
    c_t : ArrayOrFloat
        Engine thrust coefficient
    c_t_eta_b : ArrayOrFloat
        Thrust coefficient at maximum overall propulsion efficiency for a given Mach Number.
    atyp_param : PSAircraftEngineParams
        Extracted aircraft and engine parameters.
    eta_over_eta_b_min : float | None, optional
        Clip the ratio of the overall propulsion efficiency to the maximum propulsion
        efficiency to this value. See :func:`propulsion_efficiency_over_max_propulsion_efficiency`.
        If ``None``, no clipping is performed.

    Returns
    -------
    npt.NDArray[np.floating]
        Overall propulsion efficiency
    """
    eta_over_eta_b = propulsion_efficiency_over_max_propulsion_efficiency(mach_num, c_t, c_t_eta_b)
    if eta_over_eta_b_min is not None:
        eta_over_eta_b.clip(min=eta_over_eta_b_min, out=eta_over_eta_b)
    eta_b = max_overall_propulsion_efficiency(
        mach_num, atyp_param.m_des, atyp_param.eta_1, atyp_param.eta_2
    )
    return eta_over_eta_b * eta_b


def propulsion_efficiency_over_max_propulsion_efficiency(
    mach_num: ArrayOrFloat,
    c_t: ArrayOrFloat,
    c_t_eta_b: ArrayOrFloat,
) -> npt.NDArray[np.floating]:
    """Calculate ratio of OPE to maximum OPE that can be attained for a given Mach number.

    Parameters
    ----------
    mach_num : ArrayOrFloat
        Mach number at each waypoint.
    c_t : ArrayOrFloat
        Engine thrust coefficient.
    c_t_eta_b : ArrayOrFloat
        Thrust coefficient at maximum overall propulsion efficiency for a given Mach Number.

    Returns
    -------
    npt.NDArray[np.floating]
        Ratio of OPE to maximum OPE, ``eta / eta_b``

    Notes
    -----
    - ``eta / eta_b`` is approximated using a fourth-order polynomial
    - ``eta_b`` is the maximum overall propulsion efficiency for a given Mach number
    """
    c_t_over_c_t_eta_b = c_t / c_t_eta_b

    sigma = np.where(mach_num < 0.4, 1.3 * (0.4 - mach_num), np.float32(0.0))  # avoid promotion

    eta_over_eta_b_low = (
        10.0 * (1.0 + 0.8 * (sigma - 0.43) - 0.6027 * sigma * 0.43) * c_t_over_c_t_eta_b
        + 33.3333 * (-1.0 - 0.97 * (sigma - 0.43) + 0.8281 * sigma * 0.43) * (c_t_over_c_t_eta_b**2)
        + 37.037 * (1.0 + (sigma - 0.43) - 0.9163 * sigma * 0.43) * (c_t_over_c_t_eta_b**3)
    )
    eta_over_eta_b_hi = (
        (1.0 + (sigma - 0.43) - sigma * 0.43)
        + (4.0 * sigma * 0.43 - 2.0 * (sigma - 0.43)) * c_t_over_c_t_eta_b
        + ((sigma - 0.43) - 6 * sigma * 0.43) * (c_t_over_c_t_eta_b**2)
        + 4.0 * sigma * 0.43 * (c_t_over_c_t_eta_b**3)
        - sigma * 0.43 * (c_t_over_c_t_eta_b**4)
    )
    return np.where(c_t_over_c_t_eta_b < 0.3, eta_over_eta_b_low, eta_over_eta_b_hi)


def thrust_coefficient_at_max_efficiency(
    mach_num: ArrayOrFloat, m_des: float, c_t_des: float
) -> ArrayOrFloat:
    """
    Calculate thrust coefficient at maximum overall propulsion efficiency for a given Mach Number.

    Parameters
    ----------
    mach_num : ArrayOrFloat
        Mach number at each waypoint.
    m_des: float
        Design optimum Mach number where the fuel mass flow rate is at a minimum.
    c_t_des: float
        Design optimum engine thrust coefficient where the fuel mass flow rate is at a minimum.

    Returns
    -------
    ArrayOrFloat
        Thrust coefficient at maximum overall propulsion efficiency for a given
        Mach Number, ``(c_t)_eta_b``
    """
    m_over_m_des = mach_num / m_des
    h_2 = ((1.0 + 0.55 * mach_num) / (1.0 + 0.55 * m_des)) / (m_over_m_des**2)
    return h_2 * c_t_des


def max_overall_propulsion_efficiency(
    mach_num: ArrayOrFloat, mach_num_des: float, eta_1: float, eta_2: float
) -> ArrayOrFloat:
    """
    Calculate maximum overall propulsion efficiency that can be achieved for a given Mach number.

    Parameters
    ----------
    mach_num : ArrayOrFloat
        Mach number at each waypoint
    mach_num_des : float
        Design optimum Mach number where the fuel mass flow rate is at a minimum.
    eta_1 : float
        Multiplier for maximum overall propulsion efficiency model, varies by aircraft type
    eta_2 : float
        Exponent for maximum overall propulsion efficiency model, varies by aircraft type

    Returns
    -------
    ArrayOrFloat
        Maximum overall propulsion efficiency that can be achieved for a given Mach number

    References
    ----------
    Eq. (35) of :cite:`pollEstimationMethodFuel2021a`.
    """
    # XXX: h_1 may be varied in the future
    # The current implementation looks like:
    # h_1 = (mach_num / mach_num_des) ** eta_2
    # return h_1 * eta_1 * mach_num_des**eta_2

    return eta_1 * mach_num**eta_2


# -------------------
# Fuel consumption
# -------------------


def fuel_mass_flow_rate(
    air_pressure: ArrayOrFloat,
    air_temperature: ArrayOrFloat,
    mach_num: ArrayOrFloat,
    c_t: ArrayOrFloat,
    eta: ArrayOrFloat | float,
    wing_surface_area: float,
    q_fuel: float,
) -> ArrayOrFloat:
    r"""Calculate fuel mass flow rate.

    Parameters
    ----------
    air_pressure : ArrayOrFloat
        Ambient pressure, [:math:`Pa`]
    air_temperature : ArrayOrFloat
        Ambient temperature at each waypoint, [:math:`K`]
    mach_num : ArrayOrFloat
        Mach number at each waypoint
    c_t : ArrayOrFloat
        Engine thrust coefficient
    eta : ArrayOrFloat | float
        Overall propulsion efficiency
    wing_surface_area : float
        Aircraft wing surface area, [:math:`m^2`]
    q_fuel : float
        Lower calorific value (LCV) of fuel, [:math:`J \ kg_{fuel}^{-1}`]

    Returns
    -------
    ArrayOrFloat
        Fuel mass flow rate, [:math:`kg s^{-1}`]
    """
    return (
        (constants.kappa / 2)
        * (c_t * mach_num**3 / eta)
        * (constants.kappa * constants.R_d * air_temperature) ** 0.5
        * air_pressure
        * wing_surface_area
        / q_fuel
    )


def fuel_flow_correction(
    fuel_flow: ArrayOrFloat,
    altitude_ft: ArrayOrFloat,
    air_temperature: ArrayOrFloat,
    air_pressure: ArrayOrFloat,
    mach_number: ArrayOrFloat,
    fuel_flow_idle_sls: float,
    fuel_flow_max_sls: float,
    flight_phase: npt.NDArray[np.uint8] | flight.FlightPhase,
) -> ArrayOrFloat:
    r"""Correct fuel mass flow rate to ensure that they are within operational limits.

    Parameters
    ----------
    fuel_flow : ArrayOrFloat
        Fuel mass flow rate, [:math:`kg s^{-1}`]
    altitude_ft : ArrayOrFloat
        Waypoint altitude, [:math: `ft`]
    air_temperature : ArrayOrFloat
        Ambient temperature at each waypoint, [:math:`K`]
    air_pressure : ArrayOrFloat
        Ambient pressure, [:math:`Pa`]
    mach_number : ArrayOrFloat
        Mach number
    fuel_flow_idle_sls : float
        Fuel mass flow rate under engine idle and sea level static conditions, [:math:`kg \ s^{-1}`]
    fuel_flow_max_sls : float
        Fuel mass flow rate at take-off and sea level static conditions, [:math:`kg \ s^{-1}`]
    flight_phase : npt.NDArray[np.uint8] | flight.FlightPhase
        Phase state of each waypoint.

    Returns
    -------
    ArrayOrFloat
        Corrected fuel mass flow rate, [:math:`kg \ s^{-1}`]
    """
    ff_min = ps_lims.fuel_flow_idle(fuel_flow_idle_sls, altitude_ft)
    ff_max = jet.equivalent_fuel_flow_rate_at_cruise(
        fuel_flow_max_sls,
        (air_temperature / constants.T_msl),
        (air_pressure / constants.p_surface),
        mach_number,
    )

    # Account for descent conditions
    # Assume max_fuel_flow at descent is not more than 30% of fuel_flow_max_sls
    # We need this assumption because PTF files are not available in the PS model.
    descent = flight_phase == flight.FlightPhase.DESCENT
    ff_max[descent] = 0.3 * fuel_flow_max_sls
    return np.clip(fuel_flow, ff_min, ff_max)


@functools.cache
def get_aircraft_synonym_dict_ps() -> dict[str, str]:
    """Read `ps-synonym-list-20240524.csv` from the static directory.

    Returns
    -------
    dict[str, str]
        Dictionary of the form ``{"icao_aircraft_type": "ps_aircraft_type"}``.
    """
    # get path to static PS synonym list
    df_atyp_icao_to_ps = pd.read_csv(PS_SYNONYM_FILE_PATH, index_col="ICAO Aircraft Code")
    return df_atyp_icao_to_ps["PS ATYP"].to_dict()
