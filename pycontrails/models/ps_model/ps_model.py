"""Support for the Poll-Schumann (PS) aircraft performance model."""

from __future__ import annotations

import dataclasses
import pathlib
from typing import Any, Mapping

import numpy as np
import numpy.typing as npt
from overrides import overrides

from pycontrails.core import flight
from pycontrails.core.flight import Flight
from pycontrails.core.met import MetDataset
from pycontrails.core.met_var import AirTemperature, EastwardWind, NorthwardWind
from pycontrails.models.aircraft_performance import (
    DEFAULT_LOAD_FACTOR,
    AircraftPerformance,
    AircraftPerformanceData,
    AircraftPerformanceParams,
)
from pycontrails.models.ps_model.ps_aircraft_params import (
    PSAircraftEngineParams,
    get_aircraft_engine_params,
)
from pycontrails.physics import constants, jet, units


@dataclasses.dataclass
class PSModelParams(AircraftPerformanceParams):
    """Default parameters for :class:`PSModel`."""

    #: Default paths
    data_path: str | pathlib.Path = (
        pathlib.Path(__file__).parent / "static" / "ps-aircraft-params-20230517.csv"
    )

    #: Clip the ratio of the overall propulsion efficiency to the maximum propulsion
    #: efficiency to always exceed this value.
    eta_over_eta_b_min: float | None = 0.5


class PSModel(AircraftPerformance):
    """Simulate aircraft performance using Poll-Schumann (PS) model.

    References
    ----------
    :cite:`pollEstimationMethodFuel2021`
    :cite:`pollEstimationMethodFuel2021a`

    Poll & Schumann (2022). An estimation method for the fuel burn and other performance
    characteristics of civil transport aircraft. Part 3 Generalisation to cover climb,
    descent and holding. Aero. J., submitted.
    """

    name = "PSModel"
    long_name = "Poll-Schumann Aircraft Performance Model"
    met_variables = (AirTemperature,)
    optional_met_variables = EastwardWind, NorthwardWind
    default_params = PSModelParams

    aircraft_engine_params: Mapping[str, PSAircraftEngineParams]

    def __init__(
        self,
        met: MetDataset | None = None,
        params: dict[str, Any] | None = None,
        **params_kwargs: Any,
    ) -> None:
        super().__init__(met=met, params=params, **params_kwargs)
        self.aircraft_engine_params = get_aircraft_engine_params(self.params["data_path"])

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
        if aircraft_type in self.aircraft_engine_params:
            return True
        if raise_error:
            raise KeyError(f"Aircraft type {aircraft_type} not covered by the PS model.")
        return False

    def eval(self, source: Flight | list[Flight] | None = None, **params: Any) -> Any:
        """Evaluate the aircraft performance model."""
        self.update_params(params)
        self.set_source(source)
        self.source = self.require_source_type(Flight)
        self.downselect_met()
        self.set_source_met()

        # Calculate true airspeed if not included on source
        self.ensure_true_airspeed_on_source()

        # Ensure aircraft type is available
        try:
            aircraft_type = self.source.attrs["aircraft_type"]
        except KeyError as exc:
            raise KeyError("`aircraft_type` required on flight attrs") from exc

        try:
            aircraft_params = self.aircraft_engine_params[aircraft_type]
        except KeyError:
            raise KeyError(f"Aircraft type {aircraft_type} not covered by the PS model.")

        # Set flight attributes based on engine, if they aren't already defined
        self.source.attrs.setdefault("aircraft_performance_model", self.name)
        self.source.attrs.setdefault("n_engine", aircraft_params.n_engine)

        self.source.attrs.setdefault("wingspan", aircraft_params.wing_span)
        self.source.attrs.setdefault("max_mach", aircraft_params.max_mach_num)
        self.source.attrs.setdefault("max_altitude", units.ft_to_m(aircraft_params.max_altitude_ft))
        self.source.attrs.setdefault("n_engine", aircraft_params.n_engine)

        amass_oew = self.source.attrs.get("amass_oew", aircraft_params.amass_oew)
        amass_mtow = self.source.attrs.get("amass_mtow", aircraft_params.amass_mtow)
        amass_mpl = self.source.attrs.get("amass_mpl", aircraft_params.amass_mpl)
        load_factor = self.source.attrs.get("load_factor", DEFAULT_LOAD_FACTOR)
        takeoff_mass = self.source.attrs.get("takeoff_mass")
        q_fuel = self.source.fuel.q_fuel

        # Run the simulation
        aircraft_performance = self.simulate_fuel_and_performance(
            aircraft_type=aircraft_type,
            altitude_ft=self.source.altitude_ft,
            time=self.source["time"],
            true_airspeed=self.source["true_airspeed"],
            air_temperature=self.source["air_temperature"],
            aircraft_mass=self.get_source_param("aircraft_mass", None),
            thrust=self.get_source_param("thrust", None),
            engine_efficiency=self.get_source_param("engine_efficiency", None),
            fuel_flow=self.get_source_param("fuel_flow", None),
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
            self.source.setdefault(var, getattr(aircraft_performance, var))

        self._cleanup_indices()

        self.source.attrs["total_fuel_burn"] = np.nansum(aircraft_performance.fuel_burn).item()

        return self.source

    @overrides
    def calculate_aircraft_performance(
        self,
        *,
        aircraft_type: str,
        altitude_ft: npt.NDArray[np.float_],
        air_temperature: npt.NDArray[np.float_],
        time: npt.NDArray[np.datetime64] | None,
        true_airspeed: npt.NDArray[np.float_] | float | None,
        aircraft_mass: npt.NDArray[np.float_] | float,
        engine_efficiency: npt.NDArray[np.float_] | float | None,
        fuel_flow: npt.NDArray[np.float_] | float | None,
        thrust: npt.NDArray[np.float_] | float | None,
        q_fuel: float,
        **kwargs: Any,
    ) -> AircraftPerformanceData:
        try:
            correct_fuel_flow_ = kwargs["correct_fuel_flow"]
        except KeyError:
            raise KeyError("A 'correct_fuel_flow' kwarg is required for this model")

        if not isinstance(true_airspeed, np.ndarray):
            raise NotImplementedError("Only array inputs are supported")

        atyp_param = self.aircraft_engine_params[aircraft_type]

        # Atmospheric quantities
        air_pressure = units.ft_to_pl(altitude_ft) * 100.0

        # Clip unrealistically high true airspeed
        max_mach = atyp_param.max_mach_num + 0.02  # allow small buffer
        true_airspeed, mach_num = jet.clip_mach_number(true_airspeed, air_temperature, max_mach)

        # Reynolds number
        rn = reynolds_number(atyp_param.wing_surface_area, mach_num, air_temperature, air_pressure)

        # Allow array or None time
        dv_dt: npt.NDArray[np.float_] | float
        theta: npt.NDArray[np.float_] | float
        if time is None:
            # Assume a nominal cruising state
            dt_sec = None
            rocd = np.zeros_like(altitude_ft)
            dv_dt = 0.0
            theta = 0.0

        elif isinstance(time, np.ndarray):
            dt_sec = flight.segment_duration(time, dtype=altitude_ft.dtype)
            rocd = flight.segment_rocd(dt_sec, altitude_ft)
            dv_dt = jet.acceleration(true_airspeed, dt_sec)
            theta = jet.climb_descent_angle(true_airspeed, rocd)

        else:
            raise NotImplementedError("Only array inputs are supported")

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

        if engine_efficiency is None:
            engine_efficiency = overall_propulsion_efficiency(
                mach_num, c_t, atyp_param, self.params["eta_over_eta_b_min"]
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
        elif isinstance(fuel_flow, (int, float)):
            fuel_flow = np.full_like(true_airspeed, fuel_flow)

        if correct_fuel_flow_:
            fuel_flow = correct_fuel_flow(
                fuel_flow,
                altitude_ft,
                air_temperature,
                air_pressure,
                mach_num,
                atyp_param.ff_idle_sls,
                atyp_param.ff_max_sls,
            )

        if dt_sec is not None:
            fuel_burn = jet.fuel_burn(fuel_flow, dt_sec)
        else:
            fuel_burn = np.full_like(fuel_flow, np.nan)

        # XXX: Explicitly broadcast scalar inputs as needed to keep a consistent
        # output spec.
        if isinstance(aircraft_mass, (int, float)):
            aircraft_mass = np.full_like(true_airspeed, aircraft_mass)
        if isinstance(engine_efficiency, (int, float)):
            engine_efficiency = np.full_like(true_airspeed, engine_efficiency)
        if isinstance(thrust, (int, float)):
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
    mach_num: npt.NDArray[np.float_],
    air_temperature: npt.NDArray[np.float_],
    air_pressure: npt.NDArray[np.float_],
) -> npt.NDArray[np.float_]:
    """
    Calculate Reynolds number.

    Parameters
    ----------
    wing_surface_area : float
        Aircraft wing surface area, [:math:`m^2`]
    mach_num : npt.NDArray[np.float_]
        Mach number at each waypoint
    air_temperature : npt.NDArray[np.float_]
        Ambient temperature at each waypoint, [:math:`K`]
    air_pressure: npt.NDArray[np.float_]
        Ambient pressure, [:math:`Pa`]

    Returns
    -------
    npt.NDArray[np.float_]
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


def fluid_dynamic_viscosity(air_temperature: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
    """
    Calculate fluid dynamic viscosity.

    Parameters
    ----------
    air_temperature : npt.NDArray[np.float_]
        Ambient temperature at each waypoint, [:math:`K`]

    Returns
    -------
    npt.NDArray[np.float_]
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
    aircraft_mass: npt.NDArray[np.float_] | float,
    air_pressure: npt.NDArray[np.float_],
    mach_num: npt.NDArray[np.float_],
    climb_angle: npt.NDArray[np.float_] | float = 0.0,
) -> npt.NDArray[np.float_]:
    r"""Calculate the lift coefficient.

    This quantity is a dimensionless coefficient that relates the lift generated
    by a lifting body to the fluid density around the body, the fluid velocity,
    and an associated reference area.

    Parameters
    ----------
    wing_surface_area : float
        Aircraft wing surface area, [:math:`m^2`]
    aircraft_mass : npt.NDArray[np.float_] | float
        Aircraft mass, [:math:`kg`]
    air_pressure: npt.NDArray[np.float_]
        Ambient pressure, [:math:`Pa`]
    mach_num : npt.NDArray[np.float_]
        Mach number at each waypoint
    climb_angle : npt.NDArray[np.float_] | None
        Angle between the horizontal plane and the actual flight path, [:math:`\deg`]

    Returns
    -------
    npt.NDArray[np.float_]
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


def skin_friction_coefficient(rn: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
    """Calculate aircraft skin friction coefficient.

    Parameters
    ----------
    rn: npt.NDArray[np.float_]
        Reynolds number

    Returns
    -------
    npt.NDArray[np.float_]
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


def zero_lift_drag_coefficient(c_f: npt.NDArray[np.float_], psi_0: float) -> npt.NDArray[np.float_]:
    """Calculate zero-lift drag coefficient.

    Parameters
    ----------
    c_f : npt.NDArray[np.float_]
        Skin friction coefficient
    psi_0 : float
        Aircraft geometry drag parameter

    Returns
    -------
    npt.NDArray[np.float_]
        Zero-lift drag coefficient (c_d_0)

    References
    ----------
    Eq. (9) of :cite:`pollEstimationMethodFuel2021`.
    """
    return c_f * psi_0


def oswald_efficiency_factor(
    c_drag_0: npt.NDArray[np.float_], atyp_param: PSAircraftEngineParams
) -> npt.NDArray[np.float_]:
    """Calculate Oswald efficiency factor.

    The Oswald efficiency factor captures all the lift-dependent drag effects, including
    the vortex drag on the wing (primary source), tailplane and fuselage, and is a function
    of aircraft geometry and the zero-lift drag coefficient.

    Parameters
    ----------
    c_drag_0 : npt.NDArray[np.float_]
        Zero-lift drag coefficient.
    atyp_param : AircraftEngineParams
        Extracted aircraft and engine parameters.

    Returns
    -------
    npt.NDArray[np.float_]
        Oswald efficiency factor (e_ls)

    References
    ----------
    Eq. (12) of :cite:`pollEstimationMethodFuel2021`.
    """
    numer = 1.075 if atyp_param.winglets else 1.0
    k_1 = _non_vortex_lift_dependent_drag_factor(c_drag_0, atyp_param.cos_sweep)
    denom = 1.0 + 0.03 + atyp_param.delta_2 + (k_1 * (np.pi * atyp_param.wing_aspect_ratio))
    return numer / denom


def _non_vortex_lift_dependent_drag_factor(
    c_drag_0: npt.NDArray[np.float_], cos_sweep: float
) -> npt.NDArray[np.float_]:
    """Calculate non-vortex lift-dependent drag factor.

    Parameters
    ----------
    c_drag_0 : npt.NDArray[np.float_]
        Zero-lift drag coefficient
    cos_sweep : float
        Cosine of wing sweep angle measured at the 1/4 chord line

    Returns
    -------
    npt.NDArray[np.float_]
        Miscellaneous lift-dependent drag factor (k_1)

    References
    ----------
    Eq. (13) of :cite:`pollEstimationMethodFuel2021a`.
    """
    return 0.8 * (1.0 - 0.53 * cos_sweep) * c_drag_0


def wave_drag_coefficient(
    mach_num: npt.NDArray[np.float_],
    c_lift: npt.NDArray[np.float_],
    atyp_param: PSAircraftEngineParams,
) -> npt.NDArray[np.float_]:
    """Calculate wave drag coefficient.

    Parameters
    ----------
    mach_num : npt.NDArray[np.float_]
        Mach number at each waypoint
    c_lift : npt.NDArray[np.float_]
        Zero-lift drag coefficient
    atyp_param : AircraftEngineParams
        Extracted aircraft and engine parameters.

    Returns
    -------
    npt.NDArray[np.float_]
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

    return np.where(
        x < atyp_param.x_ref, c_d_w, c_d_w + atyp_param.j_3 * (x - atyp_param.x_ref) ** 4
    )


def airframe_drag_coefficient(
    c_drag_0: npt.NDArray[np.float_],
    c_drag_w: npt.NDArray[np.float_],
    c_lift: npt.NDArray[np.float_],
    e_ls: npt.NDArray[np.float_],
    wing_aspect_ratio: float,
) -> npt.NDArray[np.float_]:
    """Calculate total airframe drag coefficient.

    Parameters
    ----------
    c_drag_0 : npt.NDArray[np.float_]
        Zero-lift drag coefficient
    c_drag_w : npt.NDArray[np.float_]
        Wave drag coefficient
    c_lift : npt.NDArray[np.float_]
        Lift coefficient
    e_ls : npt.NDArray[np.float_]
        Oswald efficiency factor
    wing_aspect_ratio : float
        Wing aspect ratio

    Returns
    -------
    npt.NDArray[np.float_]
        Total airframe drag coefficient

    References
    ----------
    Eq. (8) of :cite:`pollEstimationMethodFuel2021`.
    """
    k = _low_speed_lift_dependent_drag_factor(e_ls, wing_aspect_ratio)
    return c_drag_0 + (k * c_lift**2) + c_drag_w


def _low_speed_lift_dependent_drag_factor(
    e_ls: npt.NDArray[np.float_], wing_aspect_ratio: float
) -> npt.NDArray[np.float_]:
    """Calculate low-speed lift-dependent drag factor.

    Parameters
    ----------
    e_ls : npt.NDArray[np.float_]
        Oswald efficiency factor
    wing_aspect_ratio : float
        Wing aspect ratio

    Returns
    -------
    npt.NDArray[np.float_]
        Low-speed lift-dependent drag factor, K term used to calculate the total
        airframe drag coefficient.
    """
    return 1.0 / (np.pi * wing_aspect_ratio * e_ls)


# -------------------
# Engine parameters
# -------------------


def thrust_force(
    aircraft_mass: npt.NDArray[np.float_] | float,
    c_l: npt.NDArray[np.float_],
    c_d: npt.NDArray[np.float_],
    dv_dt: npt.NDArray[np.float_] | float,
    theta: npt.NDArray[np.float_] | float,
) -> npt.NDArray[np.float_]:
    r"""Calculate thrust force summed over all engines.

    Parameters
    ----------
    aircraft_mass : npt.NDArray[np.float_] | float
        Aircraft mass at each waypoint, [:math:`kg`]
    c_l : npt.NDArray[np.float_]
        Lift coefficient
    c_d : npt.NDArray[np.float_]
        Total airframe drag coefficient
    dv_dt : npt.NDArray[np.float_] | float
        Acceleration/deceleration at each waypoint, [:math:`m \ s^{-2}`]
    theta : npt.NDArray[np.float_] | float
        Climb (positive value) or descent (negative value) angle, [:math:`\deg`]

    Returns
    -------
    npt.NDArray[np.float_]
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
    f_thrust: npt.NDArray[np.float_] | float,
    mach_num: npt.NDArray[np.float_],
    air_pressure: npt.NDArray[np.float_],
    wing_surface_area: float,
) -> npt.NDArray[np.float_]:
    """Calculate engine thrust coefficient.

    Parameters
    ----------
    f_thrust : npt.NDArray[np.float_]
        Thrust force summed over all engines, [:math:`N`]
    mach_num : npt.NDArray[np.float_]
        Mach number at each waypoint
    air_pressure : npt.NDArray[np.float_]
        Ambient pressure, [:math:`Pa`]
    wing_surface_area : npt.NDArray[np.float_]
        Aircraft wing surface area, [:math:`m^2`]

    Returns
    -------
    npt.NDArray[np.float_]
        Engine thrust coefficient (c_t)
    """
    return f_thrust / (0.5 * constants.kappa * air_pressure * mach_num**2 * wing_surface_area)


def overall_propulsion_efficiency(
    mach_num: npt.NDArray[np.float_],
    c_t: npt.NDArray[np.float_],
    atyp_param: PSAircraftEngineParams,
    eta_over_eta_b_min: float | None = None,
) -> npt.NDArray[np.float_]:
    """Calculate overall propulsion efficiency.

    Parameters
    ----------
    mach_num : npt.NDArray[np.float_]
        Mach number at each waypoint
    c_t : npt.NDArray[np.float_]
        Engine thrust coefficient
    atyp_param : AircraftEngineParams
        Extracted aircraft and engine parameters.
    eta_over_eta_b_min : float | None
        Clip the ratio of the overall propulsion efficiency to the maximum propulsion
        efficiency to this value. See :func:`propulsion_efficiency_over_max_propulsion_efficiency`.
        If ``None``, no clipping is performed.

    Returns
    -------
    npt.NDArray[np.float_]
        Overall propulsion efficiency
    """
    eta_over_eta_b = propulsion_efficiency_over_max_propulsion_efficiency(
        mach_num, c_t, atyp_param.m_des, atyp_param.c_t_des
    )
    if eta_over_eta_b_min is not None:
        eta_over_eta_b.clip(min=eta_over_eta_b_min, out=eta_over_eta_b)
    eta_b = max_overall_propulsion_efficiency(
        mach_num, atyp_param.m_des, atyp_param.eta_1, atyp_param.eta_2
    )
    return eta_over_eta_b * eta_b


def propulsion_efficiency_over_max_propulsion_efficiency(
    mach_num: npt.NDArray[np.float_],
    c_t: npt.NDArray[np.float_],
    m_des: float,
    c_t_des: float,
) -> npt.NDArray[np.float_]:
    """Calculate ratio of OPE to maximum OPE that can be attained for a given Mach number.

    Parameters
    ----------
    mach_num : npt.NDArray[np.float_]
        Mach number at each waypoint.
    c_t : npt.NDArray[np.float_]
        Engine thrust coefficient.
    m_des : float
        Design optimum Mach number where the fuel mass flow rate is at a minimum.
    c_t_des : float
        Design optimum engine thrust coefficient where the fuel mass flow rate is at a minimum.

    Returns
    -------
    npt.NDArray[np.float_]
        Ratio of OPE to maximum OPE, ``eta / eta_b``

    Notes
    -----
    - ``eta / eta_b`` is approximated using a fourth-order polynomial
    - ``eta_b`` is the maximum overall propulsion efficiency for a given Mach number
    """
    c_t_eta_b = max_thrust_coefficient(mach_num, m_des, c_t_des)
    c_t_over_c_t_eta_b = c_t / c_t_eta_b

    sigma = np.where(mach_num < 0.4, 1.3 * (0.4 - mach_num), 0.0)

    eta_over_eta_b_low = (
        10.0 * (1.0 + 0.8 * (sigma - 0.43) - 0.6027 * sigma * 0.43) * c_t_over_c_t_eta_b
        + 33.3333
        * (-1.0 - 0.97 * (sigma - 0.43) + 0.8281 * sigma * 0.43)
        * (c_t_over_c_t_eta_b**2)
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


def max_thrust_coefficient(
    mach_num: npt.NDArray[np.float_], m_des: float, c_t_des: float
) -> npt.NDArray[np.float_]:
    """
    Calculate thrust coefficient at maximum overall propulsion efficiency for a given Mach Number.

    Parameters
    ----------
    mach_num : npt.NDArray[np.float_]
        Mach number at each waypoint.
    m_des: float
        Design optimum Mach number where the fuel mass flow rate is at a minimum.
    c_t_des: float
        Design optimum engine thrust coefficient where the fuel mass flow rate is at a minimum.

    Returns
    -------
    npt.NDArray[np.float_]
        Thrust coefficient at maximum overall propulsion efficiency for a given
        Mach Number, ``(c_t)_eta_b``
    """
    m_over_m_des = mach_num / m_des
    h_2 = ((1.0 + 0.55 * mach_num) / (1.0 + 0.55 * m_des)) / (m_over_m_des**2)
    return h_2 * c_t_des


def max_overall_propulsion_efficiency(
    mach_num: npt.NDArray[np.float_] | float, mach_num_des: float, eta_1: float, eta_2: float
) -> npt.NDArray[np.float_]:
    """
    Calculate maximum overall propulsion efficiency that can be achieved for a given Mach number.

    Parameters
    ----------
    mach_num : npt.NDArray[np.float_] | float
        Mach number at each waypoint
    mach_num_des : float
        Design optimum Mach number where the fuel mass flow rate is at a minimum.
    eta_1 : float
        Multiplier for maximum overall propulsion efficiency model, varies by aircraft type
    eta_2 : float
        Exponent for maximum overall propulsion efficiency model, varies by aircraft type

    Returns
    -------
    npt.NDArray[np.float_]
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
    air_pressure: npt.NDArray[np.float_],
    air_temperature: npt.NDArray[np.float_],
    mach_num: npt.NDArray[np.float_],
    c_t: npt.NDArray[np.float_],
    eta: npt.NDArray[np.float_] | float,
    wing_surface_area: float,
    q_fuel: float,
) -> npt.NDArray[np.float_]:
    r"""Calculate fuel mass flow rate.

    Parameters
    ----------
    air_pressure : npt.NDArray[np.float_]
        Ambient pressure, [:math:`Pa`]
    air_temperature : npt.NDArray[np.float_]
        Ambient temperature at each waypoint, [:math:`K`]
    mach_num : npt.NDArray[np.float_]
        Mach number at each waypoint
    c_t : npt.NDArray[np.float_]
        Engine thrust coefficient
    eta : npt.NDArray[np.float_]
        Overall propulsion efficiency
    wing_surface_area : float
        Aircraft wing surface area, [:math:`m^2`]
    q_fuel : float
        Lower calorific value (LCV) of fuel, [:math:`J \ kg_{fuel}^{-1}`]

    Returns
    -------
    npt.NDArray[np.float_]
        Fuel mass flow rate, [:math:`kg s^{-1}`]
    """
    return (
        0.7
        * (c_t * mach_num**3 / eta)
        * (constants.kappa * constants.R_d * air_temperature) ** 0.5
        * air_pressure
        * wing_surface_area
        / q_fuel
    )


def correct_fuel_flow(
    fuel_flow: npt.NDArray[np.float_],
    altitude_ft: npt.NDArray[np.float_],
    air_temperature: npt.NDArray[np.float_],
    air_pressure: npt.NDArray[np.float_],
    mach_number: npt.NDArray[np.float_],
    fuel_flow_idle_sls: float,
    fuel_flow_max_sls: float,
) -> npt.NDArray[np.float_]:
    r"""Correct fuel mass flow rate to ensure that they are within operational limits.

    Parameters
    ----------
    fuel_flow : npt.NDArray[np.float_]
        Fuel mass flow rate, [:math:`kg s^{-1}`]
    altitude_ft : npt.NDArray[np.float_]
        Waypoint altitude, [:math: `ft`]
    air_temperature : npt.NDArray[np.float_]
        Ambient temperature at each waypoint, [:math:`K`]
    air_pressure : npt.NDArray[np.float_]
        Ambient pressure, [:math:`Pa`]
    mach_number : npt.NDArray[np.float_]
        Mach number
    fuel_flow_idle_sls : float
        Fuel mass flow rate under engine idle and sea level static conditions, [:math:`kg \ s^{-1}`]
    fuel_flow_max_sls : float
        Fuel mass flow rate at take-off and sea level static conditions, [:math:`kg \ s^{-1}`]

    Returns
    -------
    npt.NDArray[np.float_]
        Corrected fuel mass flow rate, [:math:`kg \ s^{-1}`]
    """
    min_fuel_flow = jet.minimum_fuel_flow_rate_at_cruise(fuel_flow_idle_sls, altitude_ft)
    max_fuel_flow = jet.equivalent_fuel_flow_rate_at_cruise(
        fuel_flow_max_sls,
        (air_temperature / constants.T_msl),
        (air_pressure / constants.p_surface),
        mach_number,
    )
    return np.clip(fuel_flow, min_fuel_flow, max_fuel_flow)
