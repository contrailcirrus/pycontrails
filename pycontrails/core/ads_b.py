from __future__ import annotations

import dataclasses
import pandas as pd
import numpy as np
import numpy.typing as npt
import pycontrails.physics.jet as jet
from pycontrails.physics.units import ft_to_m
from pycontrails.physics.geo import segment_length

# TODO: Interpolate between waypoints with dt < threshold
# TODO: Find nearest airport


def downsample_waypoints(
        df_flight_waypoints: pd.DataFrame, *, time_resolution: int = 10, time_var: str = "timestamp"
) -> pd.DataFrame:
    """
    Downsample flight waypoints to a specified time resolution

    Parameters
    ----------
    df_flight_waypoints: pd.DataFrame
        Raw flight waypoints and metadata
    time_resolution: int
        Downsampled time resolution, [:math:`s`]
    time_var: str
        Time variable in the "df_flight_waypoints" DataFrame

    Returns
    -------
    pd.DataFrame
        Downsampled flight waypoints
    """
    df_flight_waypoints.index = df_flight_waypoints[time_var]
    df_resampled = df_flight_waypoints.resample(f"{time_resolution}s").first()
    df_resampled = df_resampled[df_resampled["longitude"].notna()].copy()
    df_resampled.reset_index(inplace=True, drop=True)
    return df_resampled


def separate_unique_flights_from_waypoints(df_waypoints: pd.DataFrame, *, columns: list) -> list[pd.DataFrame]:
    """
    Separate unique flights from the waypoints, so each subset has the same metadata.

    Parameters
    ----------
    df_waypoints: pd.DataFrame
        Unsorted waypoints of multiple flights
    columns: str
        Metadata variables to check, by order of priority

    Returns
    -------
    list[pd.DataFrame]
        List of DataFrames containing set of flight waypoints with the same metadata.
    """
    if np.any(~pd.Series(columns).isin(df_waypoints.columns)):
        raise KeyError(f"DataFrame does not contain all of the required columns listed in the inputs.")

    flights = list(df_waypoints.groupby(columns))
    return [flight[1] for flight in flights]


def remove_noise_in_cruise_altitude(
        altitude_ft: np.ndarray, *,
        noise_threshold_ft: float = 25,
        threshold_altitude_ft: float = 10000
) -> np.ndarray:
    """
    Remove noise in cruise altitude by rounding up/down to the nearest flight level.

    Parameters
    ----------
    altitude_ft: np.ndarray
        Barometric altitude, [:math:`ft`]
    noise_threshold_ft: float
        Altitude difference threshold to identify noise, [:math:`ft`]
        Barometric altitude from ADS-B telemetry is reported at increments of 25 feet.
    threshold_altitude_ft: float
        Altitude will be checked and corrected above this threshold, [:math:`ft`]
        Currently set to 10,000 feet.

    Returns
    -------
    np.ndarray
        Barometric altitude with noise removed, [:math:`ft`]
    """
    altitude_ft_corrected = np.copy(altitude_ft)
    d_alt_ft = np.diff(altitude_ft, prepend=np.nan)
    is_noise = (np.abs(d_alt_ft) <= noise_threshold_ft) & (altitude_ft > threshold_altitude_ft)

    # Round to the nearest flight level
    altitude_rounded = np.round(altitude_ft / 1000) * 1000
    altitude_ft_corrected[is_noise] = altitude_rounded[is_noise]
    return altitude_ft_corrected


@dataclasses.dataclass
class FlightPhaseDetailed:
    """Container for boolean arrays describing detailed phase of the flight.
    """
    cruise: np.ndarray
    climb: np.ndarray
    descent: np.ndarray
    level_flight: np.ndarray
    nan: np.ndarray


def identify_phase_of_flight_detailed(
        altitude_ft: np.ndarray,
        dt: np.ndarray, *,
        threshold_rocd: float = 250.0,
        min_cruise_alt_ft: float = 20000
) -> FlightPhaseDetailed:
    """
    Identify the phase of flight (climb, cruise, descent, level flight) for each waypoint.

    Parameters
    ----------
    altitude_ft: np.ndarray
        Altitude of each waypoint, [:math:`ft`]
    dt: np.ndarray
        Time difference between waypoints, [:math:`s`].
    threshold_rocd: float
        ROCD threshold to identify climb and descent, [:math:`ft min^{-1}`].
        Currently set to 250 ft/min.
    min_cruise_alt_ft: float
        Minimum threshold altitude for cruise, [:math:`ft`]
        This is specific for each aircraft type, and can be approximated as 50% of the altitude ceiling.

    Returns
    -------
    FlightPhaseDetailed
        Booleans marking if the waypoints are at cruise, climb, descent, or level flight

    Notes
    -----
    This function is more detailed when compared to `jet.identify_phase_of_flight`:
    - There is an additional flight phase "level-flight", where the aircraft is holding at lower altitudes, and
    - The cruise phase of flight only occurs above a certain threshold altitude.
    """
    rocd = jet.rate_of_climb_descent(dt, altitude_ft)

    nan = np.isnan(rocd)
    cruise = (rocd < threshold_rocd) & (rocd > -threshold_rocd) & (altitude_ft > min_cruise_alt_ft)
    climb = ~cruise & (rocd > 0)
    descent = ~cruise & (rocd < 0)
    level_flight = ~(nan | cruise | climb | descent)
    return FlightPhaseDetailed(cruise=cruise, climb=climb, descent=descent, level_flight=level_flight, nan=nan)


@dataclasses.dataclass
class TrajectoryCheck:
    """Container to check if trajectory contains multiple cruise phase and/or diversion, and
    a cut-off point where the second flight begins.
    """
    multiple_cruise_phase: bool
    flight_diversion: bool
    i_cutoff: float


def identify_multiple_cruise_phases_and_diversion(
        flight_phase: FlightPhaseDetailed,
        longitude: npt.NDArray[np.float_],
        latitude: npt.NDArray[np.float_],
        altitude_ft: npt.NDArray[np.float_],
        dt: npt.NDArray[np.float_], *,
        ground_indicator: npt.NDArray[np.bool_] | None = None
) -> TrajectoryCheck:
    """
    Identify trajectory with multiple cruise phases and/or flight diversion, and provide cut-off point.

    Parameters
    ----------
    flight_phase: FlightPhaseDetailed
        Booleans marking if the waypoints are at cruise, climb, descent, or level flight
    longitude: npt.NDArray[np.float_]
        Longitude of flight waypoints, [:math:`\deg`]
    latitude: npt.NDArray[np.float_]
        Latitude of flight waypoints, [:math:`\deg`]
    altitude_ft: npt.NDArray[np.float_]
        Barometric altitude, [:math:`ft`]
    dt: npt.NDArray[np.float_]
        Time difference between waypoints, [:math:`s`].
    ground_indicator: npt.NDArray[np.bool_] | None = None
        Variable that indicates that the aircraft is on the ground, if available

    Returns
    -------
    TrajectoryCheck
        Flight diversion identified in the provided trajectory

    Notes
    -----
    Flights with multiple cruise phases are identified by evaluating the waypoints between the start and end of the
    cruise phase. For these waypoints that should be at cruise, multiple cruise phases are identified when:
    (1) their altitudes fall below 10,000 feet, and
    (2) there is a time difference (dt) > 15 minutes between waypoints.

    If multiple flights are identified, the cut-off point is specified at waypoint with the largest dt.

    Flight "diversion" is defined when the aircraft descends below 10,000 feet and climbs back to cruise altitude
    to travel to the alternative airport. A diversion is identified when all five conditions below are satisfied:
    (1) Altitude in any waypoints between the start and end of cruise is < 10,000 ft,
    (2) Time difference between waypoints that should be at cruise must be < 15 minutes (continuous telemetry),
    (3) Segment length between waypoints that should be at cruise must be > 500 m (no stationary waypoints),
    (4) Time elapsed between waypoint with the lowest altitude (during cruise) and final waypoint should be < 2 h,
    (5) No waypoints should be on the ground between the start and end of cruise.
    """
    # Initialise variables
    multiple_cruise_phase = False
    flight_diversion = False
    i_cutoff = np.nan

    # Return if no cruise phase of flight is identified.
    if np.all(~flight_phase.cruise):
        return TrajectoryCheck(
            multiple_cruise_phase=multiple_cruise_phase, flight_diversion=flight_diversion, i_cutoff=i_cutoff
        )

    # Index of first and final cruise waypoint
    i_cruise_start = np.min(np.argwhere(flight_phase.cruise))
    i_cruise_end = min(np.max(np.argwhere(flight_phase.cruise)), len(flight_phase.cruise))

    # There should not be any waypoints with low altitudes between the start and end of the cruise phase
    should_be_cruise = np.zeros(len(flight_phase.cruise), dtype=bool)
    should_be_cruise[i_cruise_start: i_cruise_end] = True
    is_low_altitude = altitude_ft < 10000
    is_dt_large = dt > (15 * 60)
    multiple_cruise_phase = np.any((should_be_cruise & is_low_altitude & is_dt_large))

    # If there are multiple cruise phases, get cut-off point
    if multiple_cruise_phase:
        dt_max = np.max(dt[should_be_cruise & is_low_altitude & is_dt_large])
        i_cutoff = np.argwhere(should_be_cruise & is_low_altitude & (dt == dt_max))[0][0] + 1

    # Check for presence of a diverted flight
    if np.any((should_be_cruise & is_low_altitude)):
        # Calculate segment length
        seg_length = segment_length(longitude, latitude, ft_to_m(altitude_ft))
        altitude_min = np.min(altitude_ft[should_be_cruise & is_low_altitude])
        i_lowest_altitude = np.argwhere(should_be_cruise & is_low_altitude & (altitude_ft == altitude_min))[0][0] + 1

        # Check for flight diversion
        condition_1 = np.any(altitude_ft[should_be_cruise] < 10000)
        condition_2 = np.all(dt[should_be_cruise] < (15 * 60))
        condition_3 = np.all(seg_length[should_be_cruise] > 500)
        condition_4 = np.nansum(dt[i_lowest_altitude:]) < (2 * 60 * 60)

        if ground_indicator is None:
            condition_5 = True
        else:
            condition_5 = np.all(~ground_indicator[should_be_cruise])

        flight_diversion = condition_1 & condition_2 & condition_3 & condition_4 & condition_5

        if flight_diversion:
            multiple_cruise_phase = False

    return TrajectoryCheck(
            multiple_cruise_phase=multiple_cruise_phase, flight_diversion=flight_diversion, i_cutoff=i_cutoff
        )
