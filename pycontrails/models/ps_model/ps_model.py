from __future__ import annotations

import pathlib
import numpy as np
import numpy.typing as npt
from typing import Mapping
from pycontrails.core import flight
from pycontrails.physics import constants, jet, units
from pycontrails.models.ps_model.aircraft_params import AircraftEngineParams, get_aircraft_engine_params

_path_to_static = pathlib.Path(__file__).parent / "static"
default_path: str | pathlib.Path = _path_to_static / "ps-aircraft-params-20230425.csv"


class PollSchumannModel:
    aircraft_engine_params: Mapping[str, AircraftEngineParams]

    def __init__(self):
        # Set class variable with engine parameters if not yet loaded
        if not hasattr(self, "aircraft_engine_params"):
            type(self).aircraft_engine_params = get_aircraft_engine_params(default_path)

    def calculate_aircraft_performance(
            self,
            *,
            aircraft_type_icao: str,
            air_temperature: npt.NDArray[np.float_],
            altitude_ft: npt.NDArray[np.float_],
            time: npt.NDArray[np.datetime64],
            true_airspeed: npt.NDArray[np.float_] | float | None,
            aircraft_mass: npt.NDArray[np.float_] | float | None,
    ):
        atyp_param = self.aircraft_engine_params[aircraft_type_icao]

        # Atmospheric quantities
        altitude_m = units.ft_to_m(altitude_ft)
        pressure_pa = units.ft_to_pl(altitude_ft) * 100
        mach_num = units.tas_to_mach_number(true_airspeed, air_temperature)
        rn = reynolds_number(atyp_param.wing_surface_area, mach_num, air_temperature, pressure_pa)

        # Trajectory parameters
        dt_sec = flight._dt_waypoints(time, dtype=altitude_ft.dtype)
        rocd = jet.rate_of_climb_descent(dt_sec, altitude_ft)
        rocd_ms = units.ft_to_m(rocd) / 60
        dv_dt = jet.acceleration(true_airspeed, dt_sec)
        theta = jet.climb_descent_angle(true_airspeed, rocd_ms)

        # Aircraft performance parameters
        c_lift = lift_coefficient(atyp_param.wing_surface_area, aircraft_mass, pressure_pa, mach_num, theta)
        c_f = skin_friction_coefficient(rn)
        c_drag_0 = zero_lift_drag_coefficient(c_f, atyp_param.psi_0)
        e_ls = oswald_efficiency_factor(c_drag_0, atyp_param)
        c_drag_w = wave_drag_coefficient(mach_num, c_lift, atyp_param)
        c_drag = airframe_drag_coefficient(c_drag_0, c_drag_w, c_lift, e_ls, atyp_param.wing_aspect_ratio)

        # TODO: Calculate engine parameters
        # TODO: Calculate fuel consumption
        return


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
    Calculate Reynolds number

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
    - Refer to Eq. (3) of Poll & Schumann (2021).
    - Poll & Schumann (2021). An estimation method for the fuel burn and other performance characteristics of civil
        transport aircraft in the cruise. Part 1: fundamental quantities and governing relations for a general
        atmosphere. Aero. J., 125(1284), 296-340, doi: 10.1017/aer.2020.62.
    """
    mu = fluid_dynamic_viscosity(air_temperature)
    return (
        wing_surface_area**0.5
        * mach_num
        * (air_pressure / mu)
        * (constants.kappa / (constants.R_d * air_temperature))**0.5
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
    The dynamic viscosity is a measure of the fluid's resistance to flow and is represented by Sutherland's Law.
    The higher the viscosity, the thicker the fluid.

    References
    ----------
    - Refer to Eq. (25) of Poll & Schumann (2021).
    - Poll & Schumann (2021). An estimation method for the fuel burn and other performance characteristics of civil
        transport aircraft in the cruise. Part 1: fundamental quantities and governing relations for a general
        atmosphere. Aero. J., 125(1284), 296-340, doi: 10.1017/aer.2020.62.
    """
    return 1.458E-6 * (air_temperature**1.5) / (110.4 + air_temperature)


# -------------------------------
# Aircraft performance parameters
# -------------------------------


def lift_coefficient(
        wing_surface_area: float,
        aircraft_mass: npt.NDArray[np.float_],
        air_pressure: npt.NDArray[np.float_],
        mach_num: npt.NDArray[np.float_],
        climb_angle: npt.NDArray[np.float_] | None = None,
) -> npt.NDArray[np.float_]:
    """Calculate the lift coefficient.

    This quantity is a dimensionless coefficient that relates the lift generated
    by a lifting body to the fluid density around the body, the fluid velocity,
    and an associated reference area.

    Parameters
    ----------
    wing_surface_area : float
        Aircraft wing surface area, [:math:`m^2`]
    aircraft_mass : npt.NDArray[np.float_]
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
    The lift force is perpendicular to the flight direction, while the aircraft weight acts vertically.
    """
    if climb_angle is None:
        climb_angle = np.zeros_like(aircraft_mass)

    lift_force = aircraft_mass * constants.g * np.cos(units.degrees_to_radians(climb_angle))
    denom = (constants.kappa / 2) * air_pressure * mach_num**2 * wing_surface_area
    return lift_force / denom


def skin_friction_coefficient(rn: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
    """
    Calculate skin friction coefficient.

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
    The skin friction coefficient, a dimensionless quantity, is used to estimate the skin friction drag, which is the
    resistance force exerted on an object moving in a fluid. Given that aircraft at cruise generally experience a
    narrow range of Reynolds number of between 3E7 and 3E8, it is approximated using a simple power law.

    References
    ----------
    - Refer to Eq. (28) of Poll & Schumann (2021).
    - Poll & Schumann (2021). An estimation method for the fuel burn and other performance characteristics of civil
        transport aircraft in the cruise. Part 1: fundamental quantities and governing relations for a general
        atmosphere. Aero. J., 125(1284), 296-340, doi: 10.1017/aer.2020.62.
    """
    return 0.0269 / (rn**0.14)


def zero_lift_drag_coefficient(c_f: npt.NDArray[np.float_], psi_0: float) -> npt.NDArray[np.float_]:
    """
    Calculate zero-lift drag coefficient.

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
    - Refer to Eq. (9) of Poll & Schumann (2021).
    - Poll & Schumann (2021). An estimation method for the fuel burn and other performance characteristics of civil
        transport aircraft in the cruise. Part 1: fundamental quantities and governing relations for a general
        atmosphere. Aero. J., 125(1284), 296-340, doi: 10.1017/aer.2020.62.
    """
    return c_f * psi_0


def oswald_efficiency_factor(
        c_drag_0: npt.NDArray[np.float_], atyp_param: AircraftEngineParams
) -> npt.NDArray[np.float_]:
    """
    Calculate Oswald efficiency factor.

    The Oswald efficiency factor captures all the lift-dependent drag effects, including the vortex drag on the wing
    (primary source), tailplane and fuselage, and is a function of aircraft geometry and the zero-lift drag coefficient.

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
    - Refer to Eq. (12) of Poll & Schumann (2021).
    - Poll & Schumann (2021). An estimation method for the fuel burn and other performance characteristics of civil
        transport aircraft in the cruise. Part 1: fundamental quantities and governing relations for a general
        atmosphere. Aero. J., 125(1284), 296-340, doi: 10.1017/aer.2020.62.
    """
    numer = np.where(atyp_param.winglets, 1.075, 1)
    k_1 = _lift_dependent_drag_factor(c_drag_0, atyp_param.cos_sweep)
    return numer / ((1 + 0.03 + atyp_param.delta_2) + (k_1 * np.pi * atyp_param.wing_aspect_ratio))


def _lift_dependent_drag_factor(c_drag_0: npt.NDArray[np.float_], cos_sweep: float) -> npt.NDArray[np.float_]:
    """
    Calculate miscellaneous lift-dependent drag factor

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
    - Refer to Eq. (26) of Poll & Schumann (2021).
    - Poll, D.I.A. and Schumann, U., 2021. An estimation method for the fuel burn and other performance characteristics
        of civil transport aircraft during cruise: part 2, determining the aircraft’s characteristic parameters. The
        Aeronautical Journal, 125(1284), pp.296-340.
    """
    return 0.8 * (1.0 - 0.53 * cos_sweep) * c_drag_0


def wave_drag_coefficient(
        mach_num: npt.NDArray[np.float_],
        c_lift: npt.NDArray[np.float_],
        atyp_param: AircraftEngineParams
) -> npt.NDArray[np.float_]:
    """
    Calculate wave drag coefficient

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
    The wave drag coefficient captures all the drag resulting from compressibility, the development of significant
    regions of supersonic flow at the wing surface, and the formation of shock waves.

    References
    ----------
    - Refer to Eq. (X) of Poll & Schumann (2021).
    - Poll & Schumann (2022). PART 3
    """
    # TODO: Complete documentation
    m_cc = atyp_param.wing_constant - 0.1 * (c_lift / atyp_param.cos_sweep**2)
    x = mach_num * atyp_param.cos_sweep / m_cc

    if x < atyp_param.j_2:
        return np.zeros_like(mach_num)

    c_d_w = atyp_param.cos_sweep**3 * atyp_param.j_1 * (x - atyp_param.j_2)**2

    if x < atyp_param.x_ref:
        return c_d_w
    else:
        return c_d_w + 70 * (x - atyp_param.x_ref**4)


def airframe_drag_coefficient(
        c_drag_0: npt.NDArray[np.float_],
        c_drag_w: npt.NDArray[np.float_],
        c_lift: npt.NDArray[np.float_],
        e_ls: npt.NDArray[np.float_],
        wing_aspect_ratio: float
) -> npt.NDArray[np.float_]:
    """
    Calculate total airframe drag coefficient

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
    - Refer to Eq. (8) of Poll & Schumann (2021).
    - Poll & Schumann (2021). An estimation method for the fuel burn and other performance characteristics of civil
        transport aircraft in the cruise. Part 1: fundamental quantities and governing relations for a general
        atmosphere. Aero. J., 125(1284), 296-340, doi: 10.1017/aer.2020.62.
    """
    k = 1 / (np.pi * wing_aspect_ratio * e_ls)
    return c_drag_0 + (k * c_lift**2) + c_drag_w
