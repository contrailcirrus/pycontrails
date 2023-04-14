"""`Spire Aviation <https://spire.com/aviation/>`_ ADS-B data support."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from pycontrails.core import datalib, flight
from pycontrails.core.fleet import Fleet

MESSAGE_FIELDS = {
    "icao_address": str,
    "timestamp": str,
    "latitude": float,
    "longitude": float,
    "altitude_baro": float,
    "heading": float,
    "speed": float,
    "on_ground": bool,
    "callsign": str,
    "tail_number": str,
    "collection_type": str,
    "aircraft_type_icao": str,
    "aircraft_type_name": str,
    "airline_iata": str,
    "airline_name": str,
    "departure_utc_offset": str,
    "departure_scheduled_time": str,
}


class Spire(datalib.FlightDataSource):
    """Class to process `Spire ADS-B <https://spire.com/aviation/>`_ data.

    Parameters
    ----------
    time : datalib.TimeInput | None
        The time range for data retrieval, either a single datetime or (start, end) datetime range.
        Input must be datetime-like or tuple of datetime-like
        (`datetime`, :class:`pd.Timestamp`, :class:`np.datetime64`)
        specifying the (start, end) of the date range, inclusive.
        If None, all timestamps will be loaded from messages.
    messages: pd.DataFrame
        Raw ADS-B messages from Spire ADS-B stream
        formatted as a :class:`pandas.DataFrame`.

    Notes
    -----
    See `Spire Aviation API docs
    <https://aviation-docs.spire.com/api/tracking-stream/introduction>`_
    for reference documentation.

    Input ADS-B messages are *not* copied, so be sure to
    pass in a copy of messages (e.g. ``messages.copy()``)
    to preserve original dataframe.
    """

    def __init__(
        self,
        time: datalib.TimeInput | None,
        messages: pd.DataFrame,
        **kwargs: Any,
    ) -> None:
        # save a copy of current index
        messages["_index"] = messages.index

        # ensure that timestamp column is a datetime
        messages["timestamp"] = pd.to_datetime(messages["timestamp"])

        if time is not None:
            time_bounds = datalib.parse_timesteps(time, freq=None)
            within_time_bounds = (messages["timestamp"] >= time_bounds[0]) & (
                messages["timestamp"] <= time_bounds[1]
            )
            messages = messages.loc[within_time_bounds].reset_index(drop=True)

        # sort message by time
        messages.sort_values(by=["timestamp"], ascending=True, inplace=True)

        # run cleanup function
        self.messages = cleanup(messages)

    def load_fleet(self) -> Fleet:
        pass


def cleanup(messages: pd.DataFrame) -> pd.DataFrame:
    """
    Remove erroneous messages from raw Spire ADS-B data.

    Parameters
    ----------
    messages: pd.DataFrame
        Raw ADS-B messages

    Returns
    -------
    pd.DataFrame
        ADS-B messages with erroneous data removed.

    Notes
    -----
    This function removes:

    #. Remove messages without data in "icao_address", "aircraft_type_icao",
       "altitude_baro", "on_ground", "tail_number"
    #. Ensure columns have the dtype as specified in :attr:`FIELDS`
    #. Remove messages with tail number set to VARIOUS
       and aircraft type set to "N/A", "GNDT", "GRND", "ZZZZ"
    #. If pycontrails.ext.bada installed,
       remove aircraft types not covered by BADA 3 (mainly helicopters)
    #. Remove terrestrial messages without callsign.
       Most of these messages are below 10,000 feet and from general aviation.
    #. Remove messages when "on_ground" True indicator, but
       speed > 100 knots or altitude > 15,000 ft
    """
    # Remove waypoints without data in "icao_address", "aircraft_type_icao",
    # "altitude_baro", "on_ground", "tail_number"
    non_null_cols = [
        "altitude_baro",
        "on_ground",
        "icao_address",
        "aircraft_type_icao",
        "tail_number",
    ]
    filt = messages[non_null_cols].isna().any(axis=1)
    messages = messages.loc[filt]

    # Ensure columns have the correct dtype
    messages = messages.astype(MESSAGE_FIELDS)

    # Remove messages with tail number set to VARIOUS
    # or aircraft type set to "N/A", "GNDT", "GRND", "ZZZZ"
    filt = (messages["tail_number"] == "VARIOUS") & (
        messages["aircraft_type_icao"].isin(["N/A", "GNDT", "GRND", "ZZZZ"])
    )
    messages = messages.loc[~filt]

    # Try remove aircraft types not covered by BADA 3 (mainly helicopters)
    try:
        from pycontrails.ext.bada import BADA3

        atyps = list(BADA3().synonym_dict.keys())

        messages = messages.loc[messages["aircraft_type_icao"].isin(atyps)]
    except (ImportError, FileNotFoundError):
        pass

    # Remove terrestrial waypoints without callsign
    # Most of these waypoints are below 10,000 feet and from general aviation
    filt = (messages["callsign"].isna()) & (messages["collection_type"] == "terrestrial")
    messages = messages[~filt]

    # Remove waypoints with erroneous "on_ground" indicator
    # Thresholds assessed based on scatter plot (100 knots = 185 km/h)
    filt = messages["on_ground"] & ((messages["speed"] > 100) | (messages["altitude_baro"] > 15000))
    messages = messages[~filt]

    return messages.reset_index(drop=True)


def generate_flight_id(messages: pd.DataFrame) -> pd.DataFrame:
    """Identify unique flights from Spire ADS-B messages.

    Add "flight_id" and "valid" columns to message DataFrame.

    Parameters
    ----------
    messages : pd.DataFrame
        Cleaned ADS-B messages,
        as output from :func:`cleanup`

    Notes
    -----
    The algorithm groups flights initially on "icao_address".

    For each group:

    #. Fill callsign for satellite messages
    #. Group again by "tail_number", "aircraft_type_icao", "callsign"
    #. Remove flights with less than 10 messages

    See Also
    --------
    :func:`cleanup`
    :class:`Spire`
    """
    # Set default column values
    messages["flight_id"] = None
    messages["valid"] = False

    for icao_address, gp in messages.groupby("icao_address", sort=False):
        # fill missing callsigns for satellite records
        gp = _fill_missing_satellite_callsign(gp)

        # group again by "tail_number", "aircraft_type_icao", "callsign"
        for _, fl in gp.groupby(["tail_number", "aircraft_type_icao", "callsign"], sort=False):
            # minimum # of messages > 10
            if len(fl) < 10:
                continue

            # separate flights by "on_ground" column
            _separate_by_on_ground(messages)

            # separate flights by "on_ground" column
            _separate_by_cruise_phase(messages)

    # flights = separate_flights_with_multiple_cruise_phase(flights)

    # flights = clean_flight_altitude(flights)

    # # TODO: Check segment length, dt,
    # flight_trajectories = categorise_flight_trajectories(flights, t_cut_off)
    # return flight_trajectories


def _fill_missing_satellite_callsign(
    messages: pd.DataFrame,
) -> pd.DataFrame:
    """Fill callsign backward and forward for satellite ADS-B messages.

    ADS-B waypoints that are recorded by satellite
    often do not include the callsign metadata.

    Parameters
    ----------
    messages: pd.DataFrame
        Messages with the same ICAO address.
        Must contain the "callsign" and "collection_type" columns

    Returns
    -------
    pd.DataFrame
        Messages with the same ICAO address with
        the "callsign" filled for satellite records
    """
    if np.any(messages["collection_type"] == "satellite"):
        is_missing = (messages["callsign"].isna()) & (messages["collection_type"] == "satellite")

        if np.any(is_missing):
            messages.loc[is_missing, "callsign"] = np.nan
            messages["callsign"] = (
                messages["callsign"].fillna(method="ffill").fillna(method="bfill")
            )

    return messages


def _separate_by_on_ground(messages: pd.DataFrame) -> pd.DataFrame:
    """Separate message by "on_ground" column.

    Parameters
    ----------
    messages : pd.DataFrame
        Messages grouped by "icao_address", "tail_number",
        "aircraft_type_icao", "callsign".
        Must contain the "on_ground" and "altitude_baro" columns.

    Returns
    -------
    pd.Series
        Integer series for each unique flight with same index as "messages"
    """
    # make sure aircraft is actually on ground
    # TODO: use DEM for ground position?
    is_on_ground = messages["on_ground"] & (messages["altitude_baro"] < 15_000)

    # find end of flight indexes using "on_ground"
    end_of_flight = (~is_on_ground).astype(int).diff(periods=-1) == -1

    return end_of_flight.cumsum()


def _separate_by_cruise_phase(messages: pd.DataFrame) -> list[pd.DataFrame]:
    """
    Separate flights by multiple cruise phases.

    Parameters
    ----------
    messages : pd.DataFrame
        Messages grouped by "icao_address", "tail_number",
        "aircraft_type_icao", "callsign".
        Must contain the "on_ground" and "altitude_baro" columns.

    Returns
    -------
    pd.Series
        Integer series for each unique flight with same index as "messages"
    """

    # # Minimum cruise altitude
    # altitude_ceiling_ft = bada_3.ptf_param_dict[
    #     df_flight_waypoints["aircraft_type_icao"].iloc[0]
    # ].max_altitude_ft
    # min_cruise_altitude_ft = 0.5 * altitude_ceiling_ft

    # Calculate flight phase
    altitude_ft = messages["altitude_baro"].to_numpy()
    segment_duration = flight.segment_duration(messages["timestamp"].to_numpy())
    rocd = flight.rate_of_climb_descent(segment_duration, altitude_ft)
    flight_phase = flight.identify_phase(
        rocd,
        altitude_ft,
        threshold_rocd=250,
        # min_cruise_alt_ft=min_cruise_altitude_ft,
    )

    ############# THIS IS WHERE I AM
    # Check trajectory for multiple cruise phases and/or flight diversion
    trajectory_check: TrajectoryCheck = identify_multiple_cruise_phases_and_diversion(
        flight_phase,
        df_flight_waypoints["longitude"].values,
        df_flight_waypoints["latitude"].values,
        df_flight_waypoints["altitude_baro"].values,
        dt_sec,
        ground_indicator=df_flight_waypoints["on_ground"].values,
    )

    # Skip trajectory if there is only one cruise phase.
    if (trajectory_check.multiple_cruise_phase is False) | trajectory_check.flight_diversion:
        flights_checked.append(df_flight_waypoints)
        continue

    # Separate flights with multiple cruise phases
    i_cut = trajectory_check.i_cutoff
    df_flight_1 = df_flight_waypoints.iloc[:i_cut].copy()
    df_flight_2 = df_flight_waypoints.iloc[i_cut:].copy()

    if len(df_flight_1) > min_n_wypt:
        flights_checked.append(df_flight_1)

    if len(df_flight_2) > min_n_wypt:
        flights_checked.append(df_flight_2)


def categorise_flight_trajectories(
    flights: list[pd.DataFrame], t_cut_off: pd.Timestamp
) -> FlightTrajectories:
    """
    Categorise unique flight trajectories (validated, deferred, rejected).

    Parameters
    ----------
    flights: list[pd.DataFrame]
        List of DataFrames containing waypoints from unique flights.
    t_cut_off: pd.Timestamp
        Time of the final recorded waypoint that is provided by the full set of waypoints in the raw ADS-B file

    Returns
    -------
    FlightTrajectories
        Validated, deferred, and rejected flight trajectories.

    Notes
    -----
    Flights are categorised as
        (1) Validated: Identified flight trajectory passes all the validation test,
        (2) Deferred: The remaining trajectory could be in the next file,
        (3) Rejected: Identified flight trajectory contains anomalies and fails the validation test
    """
    validated = list()
    deferred = list()
    rejected = list()

    for df_flight_waypoints in flights:
        status = validate_flight_trajectory(df_flight_waypoints, t_cut_off)

        if status["reject"]:
            rejected.append(df_flight_waypoints.copy())
        elif (
            status["n_waypoints"]
            & status["cruise_phase"]
            & status["no_altitude_anomaly"]
            & status["complete"]
        ):
            validated.append(df_flight_waypoints.copy())
        else:
            deferred.append(df_flight_waypoints.copy())

    return FlightTrajectories(validated=validated, deferred=deferred, rejected=rejected)


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
    dt: npt.NDArray[np.float_],
    *,
    ground_indicator: npt.NDArray[np.bool_] | None = None,
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
            multiple_cruise_phase=multiple_cruise_phase,
            flight_diversion=flight_diversion,
            i_cutoff=i_cutoff,
        )

    # Index of first and final cruise waypoint
    i_cruise_start = np.min(np.argwhere(flight_phase.cruise))
    i_cruise_end = min(np.max(np.argwhere(flight_phase.cruise)), len(flight_phase.cruise))

    # There should not be any waypoints with low altitudes between the start and end of the cruise phase
    should_be_cruise = np.zeros(len(flight_phase.cruise), dtype=bool)
    should_be_cruise[i_cruise_start:i_cruise_end] = True
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
        i_lowest_altitude = (
            np.argwhere(should_be_cruise & is_low_altitude & (altitude_ft == altitude_min))[0][0]
            + 1
        )

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
        multiple_cruise_phase=multiple_cruise_phase,
        flight_diversion=flight_diversion,
        i_cutoff=i_cutoff,
    )


def _clean_flight_altitude(flights: list[pd.DataFrame]) -> list[pd.DataFrame]:
    """
    Clean erroneous and noisy altitude.

    Parameters
    ----------
    flights: list[pd.DataFrame]
        List of DataFrames containing waypoints from unique flights.

    Returns
    -------
    list[pd.DataFrame]
        List of DataFrames containing waypoints with cleaned altitude data.

    Notes
    -----
    (1) Removes waypoints with erroneous altitude, i.e., altitude above service ceiling of aircraft type, and
    (2) Remove noise in cruise altitude, where flights oscillate between 25 ft due to noise in ADS-B telemetry
    """
    flights_checked = list()

    for df_flight_waypoints in flights:
        #: (1) Remove erroneous altitude, i.e., altitude above operating limit of aircraft type
        altitude_ceiling_ft = bada_3.ptf_param_dict[
            df_flight_waypoints["aircraft_type_icao"].iloc[0]
        ].max_altitude_ft
        is_above_ceiling = df_flight_waypoints["altitude_baro"] > altitude_ceiling_ft
        df_flight_waypoints = df_flight_waypoints[~is_above_ceiling].copy()

        #: (2) Remove noise in cruise altitude
        df_flight_waypoints["altitude_baro"] = remove_noise_in_cruise_altitude(
            df_flight_waypoints["altitude_baro"].values
        )

        flights_checked.append(df_flight_waypoints.copy())

    return flights_checked


def unique_flights(messages: pd.DataFrame) -> list[Flight]:
    """Identify unique flights from Spire ADS-B messages.

    Parameters
    ----------
    messages : pd.DataFrame
        Cleaned ADS-B messages,
        as output from :func:`cleanup`
    """
    # Separate flights by ICAO address
    messages.groupby("icao_address").apply(_identify_individual_flights)


def validate_flight_trajectory(
    df_flight_waypoints: pd.DataFrame,
    t_cut_off: pd.Timestamp,
    *,
    min_n_wypt: int = 5,
) -> dict[str, bool]:
    """
    Ensure that the subset of waypoints only contains one unique flight.

    Parameters
    ----------
    df_flight_waypoints: pd.DataFrame
        Subset of flight waypoints
    t_cut_off: pd.Timestamp
        Time of the final recorded waypoint that is provided by the full set of waypoints in the raw ADS-B file
    min_n_wypt: int
        Minimum number of waypoints required for flight to be accepted

    Returns
    -------
    dict[str, bool]
        Boolean indicating if flight trajectory satisfied the three conditions outlined in `notes`

    Notes
    -----
    The subset of waypoints must satisfy all the following conditions to validate the flight trajectory:
    (1) Must contain more than `min_n_wypt` waypoints,
    (2) There must be a cruise phase of flight
    (3) There must not be waypoints below the minimum cruising altitude throughout the cruise phase.
    (4) A flight must be "complete", defined when one of these conditions are satisfied:
        - Final waypoint is on the ground, altitude < 6000 feet and speed < 150 knots (278 km/h),
        - Final waypoint is < 10,000 feet, in descent, and > 2 h have passed since `current_time_slice`, or
        - At least 12 h have passed since `t_cut_off` (remaining trajectory might not be recorded).

    Trajectory is rejected if (1) - (4) are false and > 24 h have passed since `t_cut_off`.
    """
    cols_req = ["altitude_baro", "timestamp", "aircraft_type_icao"]
    validity = np.zeros(4, dtype=bool)

    if np.any(~pd.Series(cols_req).isin(df_flight_waypoints.columns)):
        raise KeyError("DataFrame do not contain longitude and/or latitude column.")

    # Minimum cruise altitude
    altitude_ceiling_ft = bada_3.ptf_param_dict[
        df_flight_waypoints["aircraft_type_icao"].iloc[0]
    ].max_altitude_ft
    min_cruise_altitude_ft = 0.5 * altitude_ceiling_ft

    # Flight duration
    dt_sec = flight.segment_duration(df_flight_waypoints["timestamp"].values)
    flight_duration_s = np.nansum(dt_sec)
    is_short_haul = flight_duration_s < 3600

    # Flight phase
    flight_phase: FlightPhaseDetailed = identify_phase_of_flight_detailed(
        df_flight_waypoints["altitude_baro"].values,
        dt_sec,
        threshold_rocd=250,
        min_cruise_alt_ft=min_cruise_altitude_ft,
    )

    # Validate flight trajectory
    validity[0] = len(df_flight_waypoints) > min_n_wypt
    validity[1] = np.any(flight_phase.cruise)
    validity[2] = no_altitude_anomaly_during_cruise(
        df_flight_waypoints["altitude_baro"].values,
        flight_phase.cruise,
        min_cruise_alt_ft=min_cruise_altitude_ft,
    )

    # Relax constraint for short-haul flights
    if is_short_haul & (validity[1] is False) & (validity[2] is False):
        validity[1] = np.any(flight_phase.level_flight)
        validity[2] = True

    # Check that flight is complete
    wypt_final = df_flight_waypoints.iloc[-1]
    dt = (t_cut_off - wypt_final["timestamp"]) / np.timedelta64(1, "h")

    if np.all(validity[:3]):  # If first three conditions are valid
        is_descent = np.any(flight_phase.descent[-5:]) | np.any(flight_phase.level_flight[-5:])
        complete_1 = (
            wypt_final["on_ground"]
            & (wypt_final["altitude_baro"] < 6000)
            & (wypt_final["speed"] < 150)
        )
        complete_2 = (wypt_final["altitude_baro"] < 10000) & is_descent & (dt > 2)
        complete_3 = dt > 12
        validity[3] = complete_1 | complete_2 | complete_3

    return {
        "n_waypoints": validity[0],
        "cruise_phase": validity[1],
        "no_altitude_anomaly": validity[2],
        "complete": validity[3],
        "reject": np.all(~validity) & (dt > 24),
    }


# TODO: Check this function


def no_altitude_anomaly_during_cruise(
    altitude_ft: npt.NDArray[np.float_],
    flight_phase_cruise: npt.NDArray[np.bool_],
    *,
    min_cruise_alt_ft: float,
) -> bool:
    """
    Check for altitude anomaly during cruise phase of flight.

    Parameters
    ----------
    altitude_ft: npt.NDArray[np.float_]
        Altitude of each waypoint, [:math:`ft`]
    flight_phase_cruise: npt.NDArray[np.bool_]
        Booleans marking if the waypoints are at cruise
    min_cruise_alt_ft: np.ndarray
        Minimum threshold altitude for cruise, [:math:`ft`]
        This is specific for each aircraft type, and can be approximated as 50% of the altitude ceiling.

    Returns
    -------
    bool
        True if no altitude anomaly is detected.

    Notes
    -----
    The presence of unrealistically low altitudes during the cruise phase of flight is an indicator that the flight
    trajectory could contain multiple unique flights.
    """
    if np.all(~flight_phase_cruise):
        return False

    i_cruise_start = np.min(np.argwhere(flight_phase_cruise))
    i_cruise_end = min(np.max(np.argwhere(flight_phase_cruise)), len(flight_phase_cruise))
    altitude_at_cruise = altitude_ft[i_cruise_start:i_cruise_end]
    return np.all(altitude_at_cruise >= min_cruise_alt_ft)


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


def remove_noise_in_cruise_altitude(
    altitude_ft: np.ndarray, *, noise_threshold_ft: float = 25, threshold_altitude_ft: float = 10000
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
    """Container for boolean arrays describing detailed phase of the flight."""

    cruise: np.ndarray
    climb: np.ndarray
    descent: np.ndarray
    level_flight: np.ndarray
    nan: np.ndarray
