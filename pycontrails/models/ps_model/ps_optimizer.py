"""A simple two-dimensional optimizer for the PS model."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from pycontrails.core.flight import Flight
from pycontrails.core.met import MetDataArray, MetDataset
from pycontrails.models.ps_model import PSFlight
from pycontrails.models.ps_model import ps_operational_limits as ps_lims
from pycontrails.models.ps_model.ps_aircraft_params import PSAircraftEngineParams
from pycontrails.physics import units


def get_fuel_burn_over_edge(
    atyp_param: PSAircraftEngineParams,
    ps_model: PSFlight,
    atype: str,
    altitude_ft: npt.NDArray[np.float64],
    air_pressure: npt.NDArray[np.float64],
    air_temperature: npt.NDArray[np.float64],
    time: npt.NDArray[np.datetime64],
    mach_num: npt.NDArray[np.float64],
    true_airspeed: npt.NDArray[np.float64],
    mass: float,
) -> float:
    """
    Compute fuel burn over a single segment provided it is within rated limits.

    If the segment requires more than the max rated thrust, or the aircraft will stall,
    return NaN.

    Parameters
    ----------
    atyp_param : PSAircraftEngineParams
        Aircraft and engine parameters for aircraft type.
    ps_model : PSFlight
        PSFlight instance to compute fuel burn.
    atype : str
        ICAO aircraft type string.
    altitude_ft : npt:NDArray[np.float64]
        Waypoint altitude, [:math: `ft`]
    air_pressure : npt:NDArray[np.float64]
        Ambient pressure, [:math:`Pa`]
    air_temperature : npt:NDArray[np.float64]
        Ambient temperature at each waypoint, [:math:`K`]
    time : npt:NDArray[np.datetime64]
        Waypoint time.
    mach_num : npt:NDArray[np.float64]
        Mach number
    true_airspeed : npt:NDArray[np.float64]
        True airspeed for each waypoint, [:math:`m s^{-1}`].
    mass : float
        Aircraft mass at start of edge, [:math: `kg`].

    Returns
    -------
    float
        Fuel consumed by indicated action, or NaN [:math: `kg`].
    """
    in_limits = (
        ps_lims.get_excess_thrust_available(
            mach_num, air_temperature, air_pressure, mass, 0, atyp_param
        )
        > 0
    )
    if isinstance(in_limits, bool):
        if not in_limits:
            return np.nan
    elif not in_limits.all():
        return np.nan

    in_limits = (
        ps_lims.max_allowable_aircraft_mass(
            air_pressure,
            mach_num,
            atyp_param.m_des,
            atyp_param.c_l_do,
            atyp_param.wing_surface_area,
            atyp_param.amass_mtow,
        )
        > mass
    )
    if not in_limits.all():
        return np.nan

    perf = ps_model.calculate_aircraft_performance(
        aircraft_type=atype,
        altitude_ft=altitude_ft,
        air_temperature=air_temperature,
        time=time,
        q_fuel=43.14e6,
        correct_fuel_flow=False,
        model_choice="total_energy_model",
        true_airspeed=true_airspeed,
        aircraft_mass=mass,
        engine_efficiency=None,
        fuel_flow=None,
        thrust=None,
    )

    return perf.fuel_burn[0]


def opt_mach(
    atyp_param: PSAircraftEngineParams,
    ps_model: PSFlight,
    atype: str,
    altitude_ft: float,
    air_pressure: float,
    aircraft_mass: float,
    cost_index: float,
    air_temperature: float,
    headwind: float = 0.0,
) -> float:
    """
    Find the approximate optimal mach number for a given configuration.

    A crude but surprisingly efficient way of approximaing the optimal mach number at a
    given configuration and cost index. The optimal mach number is found by minimizing the
    Economy Cruise Cost Function (ECCF). We are just going to compute the ECCF over a vector
    of mach numbers in the range (min_mach, max_mach) with a resolution of 0.01. The ECCF should
    be convex over this range so this should give us the optimal answer within a tolerance of a
    hundreth which would be the resolution in the flight plan anyways.

    Parameters
    ----------
    atyp_param : PSAircraftEngineParams
        Aircraft and engine parameters for aircraft type.
    ps_model : PSFlight
        PSFlight instance to compute fuel burn.
    atype : str
        ICAO aircraft type string.
    altitude_ft : npt:NDArray[np.float64]
        Waypoint altitude, [:math: `ft`]
    air_pressure : npt:NDArray[np.float64]
        Ambient pressure, [:math:`Pa`]
    aircraft_mass : float
        Aircraft mass at start of edge, [:math: `kg`].
    cost_index : float
        The selected cost index.
    air_temperature : npt:NDArray[np.float64]
        Ambient temperature at each waypoint, [:math:`K`]
    headwind : npt:NDArray[np.float64]
        True airspeed for each waypoint, [:math:`m s^{-1}`].

    Returns
    -------
    float
        Optimal mach number or NaN.
    """
    # We will do a scalar root search to find minimum and maximum speed, but this is faster than
    # checking where the `mach` vector created below is within limits
    mmin = ps_lims.minimum_mach_num(
        air_pressure,
        aircraft_mass,
        atyp_param,
    )

    mmax = ps_lims.maximum_mach_num(
        altitude_ft, air_pressure, aircraft_mass, air_temperature, 0, atyp_param
    )

    # If we didn't find the limits, we can't fly at this altitude
    if np.isnan(mmin) or np.isnan(mmax):
        return np.nan

    # Round to nearest hundreth away from limit
    mmin = int(np.ceil(100 * mmin))
    mmax = int(np.floor(100 * mmax))

    # If min mach number is greater than or equal to max mach number
    # we can't fly at this altitude
    if mmin >= mmax:
        return np.nan

    # We need to list each value twice because calculation_aircraft_performance will
    # look at the delta in TAS when computing fuel burn. We're just going to throw
    # out every other value - this is still faster than doing it as a scalar
    mach = np.array([i * 0.01 for i in range(mmin, mmax) for _ in range(2)])
    tas = units.mach_number_to_tas(mach, air_temperature)
    if isinstance(tas, float):
        tas = np.array([tas])

    time = np.datetime64("2000-01-01T00:00")
    apd = ps_model.calculate_aircraft_performance(
        aircraft_type=atype,
        altitude_ft=np.full_like(tas, altitude_ft),
        air_temperature=np.full_like(tas, air_temperature),
        time=np.array(
            pd.date_range(start=time, end=time + np.timedelta64(1, "m"), periods=tas.size)
        ),
        q_fuel=43.14e6,
        correct_fuel_flow=False,
        model_choice="total_energy_model",
        true_airspeed=tas,
        aircraft_mass=np.full_like(tas, aircraft_mass),
        engine_efficiency=None,
        fuel_flow=None,
        thrust=None,
    )
    eccf = (cost_index + apd.fuel_flow * 60) / (tas - headwind)
    opt_idx = np.argmin(eccf[::2])
    mach = mach[::2]
    return mach[opt_idx]


def _build_grid(
    flight: Flight,
    aircraft_mass: float,
    min_alt_ft: float,
    max_alt_ft: float,
    climb_rate_fpm: float,
    met: MetDataset,
    cocip_grid: MetDataArray,
) -> dict[str, Any]:
    r"""
    Build the initial data structure used for the search.

    Construct the datastructures that we need for the search. These are mostly 2D arrays that hold
    the state of the aircraft and objective function along each possible path. The x-axis of the
    search grid has `nx=flight.size/4` equally spaced points and represents distance along the
    flight path. The z-axis is altitude (in feet), span between `min_alt_ft` and `max_alt_ft`
    inclusive, and divided so that a) the values such that `z % fl_restrict \equiv 0` are included
    and b) climbing or descending one interval over a single step in the x axis will not exceed
    `climb_rate_fpm`.

    Parameters
    ----------
    flight : Flight
        The original flight trajectory to reoptimize.
    aircraft_mass : float
        Starting mass of the aircraft, [:math: `kg`].
    min_alt_ft : float
        The minimum altitude allowed by the optimizer, [:math: `ft`].
    max_alt_ft : float
        The maximum altitude allowed by the optimizer, [:math: `ft`].
    climb_rate_fpm : float
        The nomial ROCD for step climb/descent, [:math: `ft min{^-1}`].
    met : MetDataset
        Meterology data that covers the domain of the flight.
    cocip_grid : MetDataArray
        The contrail forecast over the domain of the flight, [:math: `J km^{-1}`].

    Returns
    -------
    dict[str, Any]
        A dictionary containing the datastructures needed for the search algorithm.
    """
    grid: dict[str, Any] = {}

    # x dimension is distance along flight path in meters,
    grid["nx"] = int(np.floor(flight.size / 4))
    grid["distance"] = np.linspace(0, flight.length, grid["nx"])
    grid["dx"] = grid["distance"][1] - grid["distance"][0]
    grid["latitude"], grid["longitude"], grid["segment_index"] = flight.distance_to_coords(
        grid["distance"]
    )

    sg_sin, sg_cos = flight.segment_angle()
    grid["cos_a"] = sg_cos[grid["segment_index"]]
    grid["sin_a"] = sg_sin[grid["segment_index"]]

    # Vertical grid size is set to that climbing over one step is at most
    # climb_rate_fpm.  Step size evenly divides 1000
    div = np.ceil(1000 / (units.m_to_ft(grid["dx"]) * 0.0118 * (500 / climb_rate_fpm)))
    grid["dz"] = 1000 / div
    grid["nz"] = 1 + round((max_alt_ft - min_alt_ft) / grid["dz"])
    grid["altitude_ft"] = np.linspace(min_alt_ft, max_alt_ft, grid["nz"])
    grid["air_pressure"] = units.ft_to_pl(grid["altitude_ft"]) * 100.0

    grid["time"] = np.zeros((grid["nx"], grid["nz"]), dtype="datetime64[s]")
    grid["segment_time"] = np.zeros((grid["nx"], grid["nz"]), dtype="timedelta64[s]")
    grid["mass"] = np.zeros((grid["nx"], grid["nz"]))
    grid["warming"] = np.zeros((grid["nx"], grid["nz"]))
    grid["previous"] = np.zeros((grid["nx"], grid["nz"]))
    grid["mach_num"] = np.zeros((grid["nx"], grid["nz"]))

    grid["air_temperature"] = (
        met["air_temperature"]
        .interpolate(
            np.repeat(grid["longitude"].reshape(grid["nx"], 1), grid["nz"], axis=1).flatten(),
            np.repeat(grid["latitude"].reshape(grid["nx"], 1), grid["nz"], axis=1).flatten(),
            (
                np.repeat(grid["air_pressure"].reshape(1, grid["nz"]) / 100.0, grid["nx"], axis=0)
            ).flatten(),
            np.repeat(
                flight["time"][grid["segment_index"]].reshape(grid["nx"], 1), grid["nz"], axis=1
            ).flatten(),
        )
        .reshape((grid["nx"], grid["nz"]))
    )

    grid["eastward_wind"] = (
        met["eastward_wind"]
        .interpolate(
            np.repeat(grid["longitude"].reshape(grid["nx"], 1), grid["nz"], axis=1).flatten(),
            np.repeat(grid["latitude"].reshape(grid["nx"], 1), grid["nz"], axis=1).flatten(),
            (
                np.repeat(grid["air_pressure"].reshape(1, grid["nz"]) / 100.0, grid["nx"], axis=0)
            ).flatten(),
            np.repeat(
                flight["time"][grid["segment_index"]].reshape(grid["nx"], 1), grid["nz"], axis=1
            ).flatten(),
        )
        .reshape((grid["nx"], grid["nz"]))
    )

    grid["northward_wind"] = (
        met["northward_wind"]
        .interpolate(
            np.repeat(grid["longitude"].reshape(grid["nx"], 1), grid["nz"], axis=1).flatten(),
            np.repeat(grid["latitude"].reshape(grid["nx"], 1), grid["nz"], axis=1).flatten(),
            (
                np.repeat(grid["air_pressure"].reshape(1, grid["nz"]) / 100.0, grid["nx"], axis=0)
            ).flatten(),
            np.repeat(
                flight["time"][grid["segment_index"]].reshape(grid["nx"], 1), grid["nz"], axis=1
            ).flatten(),
        )
        .reshape((grid["nx"], grid["nz"]))
    )

    grid["ef_per_m"] = cocip_grid.interpolate(
        np.repeat(grid["longitude"].reshape(grid["nx"], 1), grid["nz"], axis=1).flatten(),
        np.repeat(grid["latitude"].reshape(grid["nx"], 1), grid["nz"], axis=1).flatten(),
        (
            np.repeat(grid["air_pressure"].reshape(1, grid["nz"]) / 100.0, grid["nx"], axis=0)
        ).flatten(),
        np.repeat(
            flight["time"][grid["segment_index"]].reshape(grid["nx"], 1), grid["nz"], axis=1
        ).flatten(),
    ).reshape((grid["nx"], grid["nz"]))

    # initialize grid variables
    grid["warming"][:, :] = np.nan
    grid["warming"][0, 0] = 0.0
    grid["time"][0, 0] = flight["time"][0]
    grid["mass"][:, :] = np.nan
    grid["mass"][0, 0] = aircraft_mass
    grid["mach_num"][:, :] = np.nan

    return grid


def _get_grid_point(
    x_idx: int,
    z_idx: int,
    grid: Mapping[str, Any],
    time: np.datetime64 | None = None,
) -> dict[str, Any]:
    """
    Retrieve stored met/performance data corresponding to an x,z coordinate.

    Given an x index and a z index, index the relivant grid variables and interpolate from
    the met data to get all the needed information to compute performance at this point.

    If `time` is `None`, then pull the time from the grid, otherwise, set time to the supplied
    time.

    Parameters
    ----------
    x_idx : int
        The x coordinate to retreive.
    z_idx : int
        The z coordinate to retreive.
    grid : Mapping[str, Any]
        Optimizer data structure containing search results.
    time : np.datetime64 | None
        If not None, this time will be stored in the return value.

    Return
    ------
    dict[str, Any]
        A dictionary containing met and performance data.
    """
    point = {}
    if time is None:
        point["time"] = grid["time"][x_idx, z_idx]
    else:
        point["time"] = time

    point["altitude_ft"] = grid["altitude_ft"][z_idx]
    point["air_pressure"] = grid["air_pressure"][z_idx]
    point["latitude"] = grid["latitude"][x_idx]
    point["longitude"] = grid["longitude"][x_idx]
    point["index"] = grid["segment_index"][x_idx]
    point["air_temperature"] = grid["air_temperature"][x_idx, z_idx]

    point["u"] = grid["eastward_wind"][x_idx, z_idx]
    point["v"] = grid["northward_wind"][x_idx, z_idx]

    point["cos_a"] = grid["cos_a"][x_idx]
    point["sin_a"] = grid["sin_a"][x_idx]

    point["headwind"] = np.sqrt(
        (point["u"] * point["cos_a"]) ** 2 + (point["v"] * point["cos_a"]) ** 2
    )

    point["mass"] = grid["mass"][x_idx, z_idx]

    point["opt_mach"] = grid["mach_num"][x_idx, z_idx]
    point["opt_tas"] = np.nan

    point["ef_per_m"] = grid["ef_per_m"][x_idx, z_idx]

    return point


def _get_allowed_actions(
    grid: Mapping[str, Any],
    x_idx: int,
    z_idx: int,
    fl_restrict: float,
    min_seg_time_mins: float,
) -> Sequence[int]:
    """
    Return list of z indices where the plane can fly in the next leg.

    Somewhat complicated logic to figure out what actions are allowed at the next step
    of the search.  The restrictions are:
    - We can only stay at the same altitude if we are at a multiple of `fl_restrict`
    - We are only allowed to change altitude if we have been at the current altitude for
      at least `min_seg_time_mins`
    - If the last point was part of a climb or descent, then we cannot switch from climb
      to descend or vice versa.

    Parameters
    ----------
    grid : Mapping[str, Any]
        Optimizer data structure containing search results.
    x_idx : int
        The current x index being considered
    z_idx : int
        The current z index being considered
    fl_restrict : float
        Cruise segments are limited to multiples of this value, [:math: ft].
    min_seg_time_mins : float
        Cruise segments must be a minimum of this time duration, [:math: min].

    Returns
    -------
    Sequence[int]
        Array of z indicies conforming to optimizer constraints.
    """

    zi_list = []
    if grid["altitude_ft"][z_idx] % fl_restrict == 0:
        # we can only stay level if we are at an allowed flight level
        zi_list.append(z_idx)

        # Only allow climb or descent if we either have been staying level for at least min time
        # or if we are already climbing or descending or if it is the first point of the search
        if grid["previous"][x_idx, z_idx] == z_idx and x_idx != 0:
            # We are staying level or staring out
            if grid["segment_time"][x_idx, z_idx] > min_seg_time_mins * 60:
                # We have been at this level for at least min_time - allow both climb and descent
                if z_idx != grid["nz"] - 1:
                    zi_list.append(z_idx + 1)
                if z_idx < grid["nz"] - 4:
                    zi_list.append(z_idx + 3)
                if z_idx > 2:
                    zi_list.append(z_idx - 3)
                elif z_idx != 0:
                    zi_list.append(z_idx - 1)
        else:
            # We were already climbing or descending - continuing to do is an option
            if z_idx != grid["nz"] - 1 and grid["previous"][x_idx, z_idx] <= z_idx:
                zi_list.append(z_idx + 1)
            if z_idx < grid["nz"] - 4 and grid["previous"][x_idx, z_idx] <= z_idx:
                zi_list.append(z_idx + 3)
            if grid["previous"][x_idx, z_idx] > z_idx:
                if z_idx > 2:
                    zi_list.append(z_idx - 3)
                elif z_idx != 0:
                    zi_list.append(z_idx - 1)
    elif grid["previous"][x_idx, z_idx] > z_idx:
        # continue previous descent
        if z_idx > 2:
            zi_list.append(z_idx - 3)
        elif z_idx != 0:
            zi_list.append(z_idx - 1)
    else:
        # continue previous climb
        if z_idx != grid["nz"] - 1:
            zi_list.append(z_idx + 1)
        if z_idx < grid["nz"] - 4:
            zi_list.append(z_idx + 3)

    return zi_list


def run_search(
    flight: Flight,
    met: MetDataset,
    cocip_grid: MetDataArray,
    aircraft_mass: float,
    contrail_scale: float = 1.0,
    min_alt_ft: float = 24000.0,
    max_alt_ft: float = 41000.0,
    climb_rate_fpm: float = 500.0,
    fl_restrict: float = 2000.0,
    min_seg_time_mins: int = 10,
    cost_index: float = 60.0,
    mask_threshold: float | None = None,
) -> Mapping[str, Any]:
    """
    Search for an optimal flight trajectory.

    Perform a modified Dykstra's search over a two-dimensional grid to find a pareto-optimal
    flight trajectory that minimizes a linear combination of elapsed time, fuel consumption, and
    contrail impact. The results of this function should be passed to `reconstruct_optimal_flight`.

    Parameters
    ----------
    flight : Flight
        The original flight trajectory to reoptimize.
    met : MetDataset
        Meterology data that covers the domain of the flight.
    cocip_grid : MetDataArray
        The contrail forecast over the domain of the flight, [:math: `J km^{-1}`].
    aircraft_mass : float
        Starting mass of the aircraft, [:math: `kg`].
    contrail_scale : float
        Scaling factor for weighting contrail impacts in objective function.
    min_alt_ft : float
        The minimum altitude allowed by the optimizer, [:math: `ft`].
    max_alt_ft : float
        The maximum altitude allowed by the optimizer, [:math: `ft`].
    climb_rate_fpm : float
        The nomial ROCD for step climb/descent, [:math: `ft min{^-1}`].
    fl_restrict : float
        Cruise segments are limited to multiples of this value, [:math: ft].
    min_seg_time_mins : float
        Cruise segments must be a minimum of this time duration, [:math: min].
    cost_index : float
        The cost index used to select optimal cruise speed.
    mask_threshold : float | None
        If not `None`, then all contrail impacts below this will be ignored, [:math: `J km^{-1}`].

    Returns
    -------
    Mapping[str, Any]
        A dictionary containing the results of the search.
    """
    ps_model = PSFlight(met=met)
    atyp_param = ps_model.aircraft_engine_params[flight.attrs["aircraft_type"]]

    grid = _build_grid(
        flight,
        aircraft_mass,
        min_alt_ft,
        max_alt_ft,
        climb_rate_fpm,
        met,
        cocip_grid,
    )

    # Compute optimal speed at starting point as part of initialization
    p1 = _get_grid_point(0, 0, grid)

    if np.isnan(p1["air_temperature"]):
        raise ValueError("Start of search outside met domain.")

    mach_start = opt_mach(
        atyp_param,
        ps_model,
        flight.attrs["aircraft_type"],
        p1["altitude_ft"],
        p1["air_pressure"],
        p1["mass"],
        cost_index,
        p1["air_temperature"],
        p1["headwind"],
    )

    if np.isnan(mach_start):
        raise ValueError("Aircraft cannot fly at starting altitude/weight.")

    grid["mach_num"][0, 0] = mach_start

    # Modified Dykstra's search across the grid
    # Move across x grid; climb, stay level, or descent at each point.
    # Save 'best' path to each point
    # We will construct path later
    for x in range(grid["nx"] - 1):
        for z in range(grid["nz"]):
            # Skip this grid point if the plane can't fly here
            if np.isnan(grid["warming"][x, z]):
                continue

            p1 = _get_grid_point(x, z, grid)

            if np.isnan(p1["opt_mach"]):
                continue

            # Compute TAS and ground speed
            p1["opt_tas"] = units.mach_number_to_tas(p1["opt_mach"], p1["air_temperature"])
            air_u = p1["opt_tas"] * p1["cos_a"]
            air_v = p1["opt_tas"] * p1["sin_a"]
            gnd_u = air_u + p1["u"]
            gnd_v = air_v + p1["v"]
            gs = np.sqrt(gnd_u**2 + gnd_v**2)  # ground speed meters / sec

            # Time elapsed to next grid point is just distance / ground speed
            time_elapsed = np.timedelta64(round(1000 * grid["dx"] / gs), "ms")

            # list of z indicies to consider for next step in x grid
            zi_list = _get_allowed_actions(grid, x, z, fl_restrict, min_seg_time_mins)

            # now for each allowed action, determine cost and save if it's the best route seen
            for zi in zi_list:
                if zi > grid["nz"] - 1:
                    continue
                p2 = _get_grid_point(x + 1, zi, grid, p1["time"] + time_elapsed)

                if p1["altitude_ft"] != p2["altitude_ft"]:
                    # If we are changing flight levels, compute new optimal mach number
                    p2["opt_mach"] = opt_mach(
                        atyp_param,
                        ps_model,
                        flight.attrs["aircraft_type"],
                        p2["altitude_ft"],
                        p2["air_pressure"],
                        p1["mass"],  # needs to be p1 because we haven't computed fuel burn yet
                        cost_index,
                        p2["air_temperature"],
                        p2["headwind"],
                    )
                    p2["opt_tas"] = units.mach_number_to_tas(p2["opt_mach"], p2["air_temperature"])
                else:
                    # If we are staying level, we are going to assume that the fuel burn
                    # is small enough that the optimal mach number won't change (we will
                    # account for mass change to see if we can climb)
                    p2["opt_mach"] = p1["opt_mach"]
                    p2["opt_tas"] = p1["opt_tas"]

                # Note - the mach optimization function does not account for climb/descent in
                # performance calculation so it's hard to avoid re-runing this here
                burn = get_fuel_burn_over_edge(
                    atyp_param,
                    ps_model,
                    flight.attrs["aircraft_type"],
                    np.array([p1["altitude_ft"], p2["altitude_ft"]]),
                    np.array([p1["air_pressure"], p2["air_pressure"]]),
                    np.array([p1["air_temperature"], p2["air_temperature"]]),
                    np.array([p1["time"], p2["time"]]),
                    np.array([p1["opt_mach"], p2["opt_mach"]]),
                    np.array([p1["opt_tas"], p2["opt_tas"]]),
                    p1["mass"],
                )

                # burn value is nan if this trajectory is not possible (max thrust exceeded)
                # in that case, just move to the next path
                if np.isnan(burn):
                    continue

                # Get contrail impact unless we are ignoring contrails
                if contrail_scale > 0.0:
                    ef_per_m = p2["ef_per_m"]
                    if mask_threshold is not None and ef_per_m < mask_threshold:
                        # If we set a mask_threshold, ignore contrail impact if ef is below
                        # threshold
                        ef_per_m = 0
                else:
                    ef_per_m = 0

                # Objective function is roughly in AGWP100
                seg_warm = burn * 4.7e9
                if not np.isnan(ef_per_m):
                    seg_warm += contrail_scale * ef_per_m * grid["dx"]

                cum_warm = grid["warming"][x, z] + seg_warm

                # If this is the "best" path to the next point, save it
                if np.isnan(grid["warming"][x + 1, zi]) or grid["warming"][x + 1, zi] > cum_warm:
                    grid["mach_num"][x + 1, zi] = p2["opt_mach"]
                    grid["warming"][x + 1, zi] = cum_warm
                    grid["time"][x + 1, zi] = p2["time"]
                    grid["mass"][x + 1, zi] = grid["mass"][x, z] - burn
                    grid["previous"][x + 1, zi] = z
                    # If staying level, save time of segment so we know when we are allowed to move
                    if zi == z:
                        grid["segment_time"][x + 1, zi] = grid["segment_time"][x, zi] + time_elapsed
                    else:
                        grid["segment_time"][x + 1, zi] = 0

    return grid


def reconstruct_optimal_flight(original_flight: Flight, grid: Mapping[str, Any]) -> Flight:
    """
    Reconstruct optimal trajectory given optimizer output.

    Given the output of the search, build the optimal flight. This starts
    at the end of the search and follows the optimal path backward to
    reconstruct the optimal path.

    Parameters
    ----------
    original_flight : Flight
        The unoptimized flight used as input to `run_search`.
    grid : Mapping[str, Any]
        A dictionary containing the output of `run_search`.

    Returns
    -------
    Flight
        Optimized flight trajectory.
    """
    nx = grid["nx"]

    # This will hold z indicies of the optimal flight path
    path = [0]

    # Move backward from the end and get the last visited node until
    # we've reached the start
    for x in range(nx - 1, -1, -1):
        path.append(int(grid["previous"][x, path[-1]]))
    path.reverse()

    # the starting point just has a placeholder value so discard it
    path = path[1:]

    # Reconstruct flight object
    fl_opt = Flight(
        time=[grid["time"][i, p] for i, p in enumerate(path)],
        latitude=grid["latitude"],
        longitude=grid["longitude"],
        altitude=[units.ft_to_m(grid["altitude_ft"][p]) for p in path],
        aircraft_type=original_flight.attrs["aircraft_type"],
        aircraft_mass=grid["mass"][0, 0],
        flight_id=original_flight.attrs["flight_id"],
    )

    return fl_opt
