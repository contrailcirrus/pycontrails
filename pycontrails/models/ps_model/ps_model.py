from __future__ import annotations

import pathlib
import numpy as np
import numpy.typing as npt
from typing import Mapping
from pycontrails.core import flight
from pycontrails.physics import constants, jet, units
from pycontrails.models.ps_model.aircraft_params import AircraftEngineParams, get_aircraft_engine_params

_path_to_static = pathlib.Path(__file__).parent / "static"


class PollSchumannModel:
    aircraft_engine_params: Mapping[str, AircraftEngineParams]
    default_path: str | pathlib.Path = _path_to_static / "ps-aircraft-params-20230424.csv"

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
        # TODO: Extract aircraft parameters
        # Temp
        wing_surface_area = 122.4

        # Trajectory parameters
        dt_sec = flight._dt_waypoints(time, dtype=altitude_ft.dtype)
        rocd = jet.rate_of_climb_descent(dt_sec, altitude_ft)
        rocd_ms = units.ft_to_m(rocd) / 60
        dv_dt = jet.acceleration(true_airspeed, dt_sec)
        mach_num = units.tas_to_mach_number(true_airspeed, air_temperature)
        theta = jet.climb_descent_angle(true_airspeed, rocd_ms)

        # Atmospheric quantities
        altitude_m = units.ft_to_m(altitude_ft)
        pressure_pa = units.ft_to_pl(altitude_ft) * 100
        rn = reynolds_number(wing_surface_area, mach_num, air_temperature, pressure_pa)

        # TODO: Calculate aircraft performance parameters
        # TODO: Lift coefficient

        # TODO: Calculate engine parameters
        # TODO: Calculate fuel consumption
        return


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
        Mach number at each waypoint, [:math: `Ma`]
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
