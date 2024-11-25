"""`Spire Aviation <https://spire.com/aviation/>`_ ADS-B data support."""

from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd
import pandas.api.types as pdtypes

from pycontrails.core import airports, flight
from pycontrails.datalib.exceptions import (
    BadTrajectoryException,
    DestinationAirportError,
    FlightAltitudeProfileError,
    FlightDuplicateTimestamps,
    FlightInvariantFieldViolation,
    FlightTooFastError,
    FlightTooLongError,
    FlightTooShortError,
    FlightTooSlowError,
    OrderingError,
    OriginAirportError,
    RocdError,
    SchemaError,
)
from pycontrails.physics import geo, units

logger = logging.getLogger(__name__)

#: Minimum messages to identify a single flight trajectory
TRAJECTORY_MINIMUM_MESSAGES = 10

#: Data types of parsed message fields
#: "timestamp" is excluded as its parsed by :func:`pandas.to_datetime`
MESSAGE_DTYPES = {
    "icao_address": str,
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


def clean(messages: pd.DataFrame) -> pd.DataFrame:
    """
    Remove erroneous messages from raw Spire ADS-B data.

    Copies input `messages` before modifying.

    Parameters
    ----------
    messages : pd.DataFrame
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
    #. Ensure columns have the dtype as specified in :attr:`MESSAGE_DTYPES`
    #. Remove messages with tail number "VARIOUS"
       and aircraft types "N/A", "GNDT", "GRND", "ZZZZ"
    #. Remove terrestrial messages without callsign.
       Most of these messages are below 10,000 feet and from general aviation.
    #. Remove messages when "on_ground" indicator is True, but
       speed > :attr:`flight.MAX_ON_GROUND_SPEED` knots
       or altitude > :attr:`flight.MAX_AIRPORT_ELEVATION` ft
    #. Drop duplicates by "icao_address" and "timestamp"
    """
    _n_messages = len(messages)

    # TODO: Enable method to work without copying
    # data multiple times
    mdf = messages.copy()

    # Remove waypoints without data in "icao_address", "aircraft_type_icao",
    # "altitude_baro", "on_ground", "tail_number"
    non_null_cols = [
        "timestamp",
        "longitude",
        "latitude",
        "altitude_baro",
        "on_ground",
        "icao_address",
        "aircraft_type_icao",
        "tail_number",
    ]
    mdf = mdf.dropna(subset=non_null_cols)

    # Ensure columns have the correct dtype
    mdf = mdf.astype(MESSAGE_DTYPES)

    # convert timestamp into a timezone naive pd.Timestamp
    mdf["timestamp"] = pd.to_datetime(mdf["timestamp"]).dt.tz_localize(None)

    # Remove messages with tail number set to VARIOUS
    # or aircraft type set to "N/A", "GNDT", "GRND", "ZZZZ"
    filt = (mdf["tail_number"] == "VARIOUS") | (
        mdf["aircraft_type_icao"].isin(["N/A", "GNDT", "GRND", "ZZZZ"])
    )
    mdf = mdf[~filt]

    # Remove terrestrial waypoints without callsign
    # Most of these waypoints are below 10,000 feet and from general aviation
    filt = (mdf["callsign"].isna()) & (mdf["collection_type"] == "terrestrial")
    mdf = mdf[~filt]

    # Fill missing callsigns for satellite records
    callsigns_missing = (mdf["collection_type"] == "satellite") & (mdf["callsign"].isna())
    mdf.loc[callsigns_missing, "callsign"] = None  # reset values to be None

    callsigns_missing_unique = mdf.loc[callsigns_missing, "icao_address"].unique()

    rows_with_any_missing_callsigns = mdf.loc[
        mdf["icao_address"].isin(callsigns_missing_unique),
        ["icao_address", "callsign", "collection_type"],
    ]

    for _, gp in rows_with_any_missing_callsigns.groupby("icao_address", sort=False):
        mdf.loc[gp.index, "callsign"] = gp["callsign"].ffill().bfill()

    # Remove messages with erroneous "on_ground" indicator
    filt = mdf["on_ground"] & (
        (mdf["speed"] > flight.MAX_ON_GROUND_SPEED)
        | (mdf["altitude_baro"] > flight.MAX_AIRPORT_ELEVATION)
    )
    mdf = mdf[~filt]

    # Drop duplicates by icao_address and timestamp
    mdf = mdf.drop_duplicates(subset=["icao_address", "timestamp"])

    logger.debug(f"{len(mdf) / _n_messages:.2f} messages remain after Spire ADS-B cleanup")

    return mdf.reset_index(drop=True)


def generate_flight_id(time: pd.Timestamp, callsign: str) -> str:
    """Generate a unique flight id for instance of flight.

    Parameters
    ----------
    time : pd.Timestamp
        First waypoint time associated with flight.
    callsign : str
        Callsign of the flight.
        Other flight identifiers could be used if the callsign
        is not defined.

    Returns
    -------
    str
        Flight id in the form "{%Y%m%d-%H%M}-{callsign}"
    """
    t_string = time.strftime("%Y%m%d-%H%M")
    return f"{t_string}-{callsign}"


def identify_flights(messages: pd.DataFrame) -> pd.Series:
    """Identify unique flights from Spire ADS-B messages.

    Parameters
    ----------
    messages : pd.DataFrame
        Cleaned ADS-B messages,
        as output from :func:`clean`

    Returns
    -------
    pd.Series
        Flight ids for the same index as `messages`

    Notes
    -----
    The algorithm groups flights initially on "icao_address".

    For each group:

    #. Fill callsign for satellite messages
    #. Group again by "tail_number", "aircraft_type_icao", "callsign"
    #. Remove flights with less than :attr:`TRAJECTORY_MINIMUM_MESSAGES` messages
    #. Separate flights by "on_ground" indicator. See `_separate_by_on_ground`.
    #. Separate flights by cruise phase. See `_separate_by_cruise_phase`.

    See Also
    --------
    :func:`clean`
    :class:`Spire`
    :func:`_separate_on_ground`
    :func:`_separate_by_cruise_phase`
    """

    # Set default flight id
    flight_id = pd.Series(
        data=None,
        dtype=object,
        index=messages.index,
    )

    for idx, gp in messages[
        [
            "icao_address",
            "tail_number",
            "aircraft_type_icao",
            "callsign",
            "timestamp",
            "longitude",
            "latitude",
            "altitude_baro",
            "on_ground",
        ]
    ].groupby(["icao_address", "tail_number", "aircraft_type_icao", "callsign"], sort=False):
        # minimum # of messages > TRAJECTORY_MINIMUM_MESSAGES
        if len(gp) < TRAJECTORY_MINIMUM_MESSAGES:
            logger.debug(f"Message {idx} group too small to create flight ids")
            continue

        # TODO: this altitude cleanup does not persist back into messages
        # this should get moved into flight module
        gp = _clean_trajectory_altitude(gp)

        # separate flights by "on_ground" column
        gp["flight_id"] = _separate_by_on_ground(gp)

        # further separate flights by cruise phase analysis
        for _, fl in gp.groupby("flight_id"):
            gp.loc[fl.index, "flight_id"] = _separate_by_cruise_phase(fl)

        # save flight ids
        flight_id.loc[gp.index] = gp["flight_id"]

    return flight_id


def _clean_trajectory_altitude(messages: pd.DataFrame) -> pd.DataFrame:
    """
    Clean erroneous and noisy altitude on a single flight.

    TODO: move this to Flight

    Parameters
    ----------
    messages: pd.DataFrame
        ADS-B messages from a single flight trajectory.

    Returns
    -------
    pd.DataFrame
        ADS-B messages with filtered altitude.

    Notes
    -----
    #. If ``pycontrails.ext.bada`` installed,
       remove erroneous altitude, i.e., altitude above operating limit of aircraft type
    #. Filter altitude signal to remove noise and
       snap cruise altitudes to 1000 ft intervals

    See Also
    --------
    :func:`flight.filter_altitude`
    """
    mdf = messages.copy()

    # Use BADA 3 to support filtering
    try:
        from pycontrails.ext.bada import BADA3

        bada3 = BADA3()
        aircraft_type_icao = (
            mdf["aircraft_type_icao"].iloc[0]
            if len(mdf["aircraft_type_icao"]) > 1
            else mdf["aircraft_type_icao"]
        )

        # Try remove aircraft types not covered by BADA 3 (mainly helicopters)
        mdf = mdf.loc[mdf["aircraft_type_icao"].isin(bada3.synonym_dict)]

        # Remove erroneous altitude, i.e., altitude above operating limit of aircraft type
        bada_max_altitude_ft = bada3.ptf_param_dict[aircraft_type_icao].max_altitude_ft
        is_above_ceiling = mdf["altitude_baro"] > bada_max_altitude_ft
        mdf = mdf.loc[~is_above_ceiling]

        # Set the min cruise altitude to 0.5 * "max_altitude_ft"
        min_cruise_altitude_ft = 0.5 * bada_max_altitude_ft

    except (ImportError, FileNotFoundError, KeyError):
        min_cruise_altitude_ft = flight.MIN_CRUISE_ALTITUDE

    # Filter altitude signal
    # See https://traffic-viz.github.io/api_reference/traffic.core.flight.html#traffic.core.Flight.filter
    mdf["altitude_baro"] = flight.filter_altitude(mdf["timestamp"], mdf["altitude_baro"].to_numpy())

    # Snap altitudes in cruise to the nearest flight level.
    # Requires segment phase
    altitude_ft = mdf["altitude_baro"].to_numpy()
    segment_duration = flight.segment_duration(mdf["timestamp"].to_numpy())
    segment_rocd = flight.segment_rocd(segment_duration, altitude_ft)
    segment_phase = flight.segment_phase(
        segment_rocd,
        altitude_ft,
        min_cruise_altitude_ft=min_cruise_altitude_ft,
    )
    is_cruise = segment_phase == flight.FlightPhase.CRUISE
    mdf.loc[is_cruise, "altitude_baro"] = np.round(altitude_ft[is_cruise], -3)

    return mdf


def _separate_by_on_ground(messages: pd.DataFrame) -> pd.Series:
    """Separate individual flights by "on_ground" column.

    The input ``messages`` are expected to grouped by "icao_address", "tail_number",
    "aircraft_type_icao", "callsign".

    Parameters
    ----------
    messages : pd.DataFrame
        Messages grouped by "icao_address", "tail_number",
        "aircraft_type_icao", "callsign".
        Must contain the "on_ground" and "altitude_baro" columns.

    Returns
    -------
    pd.Series
        Flight ids for the same index as `messages`
    """
    # Set default flight id
    try:
        flight_id = messages["flight_id"]
    except KeyError:
        flight_id = pd.Series(
            data=None,
            dtype=object,
            index=messages.index,
        )

    # make sure aircraft is actually on ground
    # TODO: use DEM for ground position?
    is_on_ground = messages["on_ground"] & (
        messages["altitude_baro"] < flight.MAX_AIRPORT_ELEVATION
    )

    # filter this signal so that it removes isolated 1-2 wide variations
    is_on_ground = is_on_ground.rolling(window=5).mean().bfill() > 0.5

    # find end of flight indexes using "on_ground"
    end_of_flight = (~is_on_ground).astype(int).diff(periods=-1) == -1

    # identify each individual flight using cumsum magic
    for _, gp in messages.groupby(end_of_flight.cumsum()):
        flight_id.loc[gp.index] = generate_flight_id(
            gp["timestamp"].iloc[0], gp["callsign"].iloc[0]
        )

    return flight_id


def _separate_by_cruise_phase(messages: pd.DataFrame) -> pd.Series:
    """
    Separate flights by multiple cruise phases.

    The input ``messages`` are expected to grouped by "icao_address", "tail_number",
    "aircraft_type_icao", "callsign".

    Its strongly encouraged to run :func:`flight.filter_altitude(...)` over the messages
    before passing into this function.

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

    Notes
    -----
    Flights with multiple cruise phases are identified by evaluating the
    messages between the start and end of the cruise phase.

    For these messages that should be at cruise, multiple cruise phases are identified when:

    #. Altitudes fall below the minimum cruise altitude.
       If pycontrails.ext.bada is installed, this value is set to half
       the BADA3 altitude ceiling of the aircraft.
       If not, this value is set to :attr:`flight.MIN_CRUISE_ALTITUDE`.
    #. There is a time difference > 15 minutes between messages.

    If multiple flights are identified,
    the cut-off point is specified at messages with the largest time difference.

    Flight "diversion" is defined when the aircraft descends below 10,000 feet
    and climbs back to cruise altitude to travel to the alternative airport.
    A diversion is identified when all five conditions below are satisfied:

    #. Altitude in any messages between
       the start and end of cruise is < :attr:`flight.MAX_AIRPORT_ELEVATION` ft
    #. Time difference between messages that should be
       at cruise must be < 15 minutes (continuous telemetry)
    #. Segment length between messages that should be
       at cruise must be > 500 m (no stationary messages),
    #. Time elapsed between message with the lowest altitude
       (during cruise) and final message should be < 2 h,
    #. No messages should be on the ground between the start and end of cruise.
    """

    # Set default flight id
    try:
        flight_id = messages["flight_id"].copy()
    except KeyError:
        flight_id = pd.Series(
            data=generate_flight_id(messages["timestamp"].iloc[0], messages["callsign"].iloc[0]),
            dtype=object,
            index=messages.index,
        )

    # Use BADA 3 to calculate max altitude
    try:
        from pycontrails.ext.bada import BADA3

        bada3 = BADA3()
        aircraft_type_icao = (
            messages["aircraft_type_icao"].iloc[0]
            if len(messages["aircraft_type_icao"]) > 1
            else messages["aircraft_type_icao"]
        )

        # Set the min cruise altitude to 0.5 * "max_altitude_ft"
        bada_max_altitude_ft = bada3.ptf_param_dict[aircraft_type_icao].max_altitude_ft
        min_cruise_altitude_ft = 0.5 * bada_max_altitude_ft

    except (ImportError, FileNotFoundError, KeyError):
        min_cruise_altitude_ft = flight.MIN_CRUISE_ALTITUDE

    # Calculate flight phase
    altitude_ft = messages["altitude_baro"].to_numpy()
    segment_duration = flight.segment_duration(messages["timestamp"].to_numpy())
    segment_rocd = flight.segment_rocd(segment_duration, altitude_ft)
    segment_phase = flight.segment_phase(
        segment_rocd,
        altitude_ft,
        threshold_rocd=250,
        min_cruise_altitude_ft=min_cruise_altitude_ft,
    )

    # get cruise phase
    cruise = segment_phase == flight.FlightPhase.CRUISE

    # fill between the first and last cruise phase indicator
    # this represents the flight phase after takeoff climb and before descent to landing
    # Index of first and final cruise waypoint
    # i_cruise_start = np.min(np.argwhere(flight_phase.cruise))
    # i_cruise_end = min(np.max(np.argwhere(flight_phase.cruise)), len(flight_phase.cruise))
    within_cruise = np.bitwise_xor.accumulate(cruise) | cruise

    # There should not be any waypoints with low altitudes between
    # the start and end of `within_cruise`
    is_low_altitude = altitude_ft < min_cruise_altitude_ft
    is_long_interval = segment_duration > (15 * 60)  # 15 minutes
    anomalous_phase = within_cruise & is_low_altitude & is_long_interval
    multiple_cruise_phase = np.any(anomalous_phase)

    # if there is only one cruise phase, just return one label
    if not multiple_cruise_phase:
        return flight_id

    # Check for presence of a diverted flight
    potentially_diverted = within_cruise & is_low_altitude
    if np.any(potentially_diverted):
        # Calculate segment length
        segment_length = geo.segment_length(
            messages["longitude"].to_numpy(),
            messages["latitude"].to_numpy(),
            units.ft_to_m(altitude_ft),
        )

        # get the index of the minimum altitude when potentially diverted
        altitude_min_diverted = np.min(altitude_ft[potentially_diverted])
        mask = within_cruise & is_low_altitude & (altitude_ft == altitude_min_diverted)
        i_lowest_altitude = np.flatnonzero(mask)[0] + 1

        # get reference to "on_ground"
        ground_indicator = messages["on_ground"].to_numpy()

        # Check for flight diversion
        condition_1 = np.any(altitude_ft[within_cruise] < flight.MAX_AIRPORT_ELEVATION)
        condition_2 = np.all(segment_duration[within_cruise] < (15.0 * 60.0))
        condition_3 = np.all(segment_length[within_cruise] > 500.0)
        condition_4 = np.nansum(segment_duration[i_lowest_altitude:]) < (2.0 * 60.0 * 60.0)
        condition_5 = np.all(~ground_indicator[within_cruise])

        flight_diversion = condition_1 & condition_2 & condition_3 & condition_4 & condition_5

        # if there is a potential flight diversion, just return a single flight index
        if flight_diversion:
            return flight_id

    # If there are multiple cruise phases, get cut-off point
    i_cutoff = int(np.argmax(segment_duration[anomalous_phase]) + 1)

    # assign a new id after cutoff
    flight_id.iloc[i_cutoff:] = generate_flight_id(
        messages["timestamp"].iloc[i_cutoff], messages["callsign"].iloc[i_cutoff]
    )

    return flight_id


def validate_flights(messages: pd.DataFrame) -> pd.Series:
    """Validate unique flights from Spire ADS-B messages.

    Parameters
    ----------
    messages : pd.DataFrame
        Messages that has been assigned flight ids by :func:`identify_flights`.
        Requires "flight_id" column to be defined in messages

    Returns
    -------
    pd.Series
        Boolean array of `flight_id` validity with the same index as `messages`

    Notes
    -----
    See :func:`is_valid_trajectory` docstring for validation criteria

    See Also
    --------
    :func:`is_valid_trajectory`
    :func:`identify_flights`
    """
    if "flight_id" not in messages:
        raise KeyError("'flight_id' column required in messages")

    # Set default flight id
    valid = pd.Series(
        data=False,
        dtype=bool,
        index=messages.index,
    )

    for _, gp in messages[
        [
            "flight_id",
            "aircraft_type_icao",
            "timestamp",
            "altitude_baro",
            "on_ground",
            "speed",
        ]
    ].groupby("flight_id", sort=False):
        # save flight ids
        valid.loc[gp.index] = is_valid_trajectory(gp)

    return valid


def is_valid_trajectory(
    messages: pd.DataFrame,
    *,
    minimum_messages: int = TRAJECTORY_MINIMUM_MESSAGES,
    final_time_available: pd.Timestamp | None = None,
) -> bool:
    """
    Ensure messages likely contain only one unique flight trajectory.

    Parameters
    ----------
    messages: pd.DataFrame
        ADS-B messages from a single flight trajectory.
    minimum_messages: int, optional
        Minimum number of messages required for trajectory to be accepted.
        Defaults to :attr:`TRAJECTORY_MINIMUM_MESSAGES`
    final_time_available: pd.Timestamp, optional
        Time of the final recorded ADS-B message available.
        Relaxes the criteria for flight completion (see *Notes*).

    Returns
    -------
    bool
        Boolean indicating if messages constitute a single valid trajectory

    Notes
    -----
    The inputs messages must satisfy all
    the following conditions to validate the flight trajectory:

    #. Must contain more than :attr:`TRAJECTORY_MINIMUM_MESSAGES`
    #. Trajectory must contain a `cruise` phase as defined by :func:`flight.segment_phase`
    #. Trajectory must not go below the minimum cruising altitude throughout the cruise phase.
    #. A trajectory must be "complete", defined when one of these conditions are satisfied:
        - Final message is on the ground, altitude < :attr:`flight.MAX_AIRPORT_ELEVATION` feet
          and speed < :attr:`flight.MAX_ON_GROUND_SPEED` knots
        - Final message is < :attr:`flight.MAX_AIRPORT_ELEVATION` feet, in descent,
          and > 2 h have passed since `final_time_available`.
        - At least 12 h have passed since `final_time_available`
          (remaining trajectory might not be recorded).

    Trajectory is not valid if any criteria is False.
    """

    # Use BADA 3 to calculate min cruise altitude
    try:
        from pycontrails.ext.bada import BADA3

        bada3 = BADA3()
        aircraft_type_icao = (
            messages["aircraft_type_icao"].iloc[0]
            if len(messages["aircraft_type_icao"]) > 1
            else messages["aircraft_type_icao"]
        )

        # Remove erroneous altitude, i.e., altitude above operating limit of aircraft type
        altitude_ceiling_ft = bada3.ptf_param_dict[aircraft_type_icao].max_altitude_ft
        min_cruise_altitude_ft = 0.5 * altitude_ceiling_ft

    except (ImportError, FileNotFoundError, KeyError):
        min_cruise_altitude_ft = flight.MIN_CRUISE_ALTITUDE

    # Flight duration
    segment_duration = flight.segment_duration(messages["timestamp"].to_numpy())
    is_short_haul = np.nansum(segment_duration).item() < flight.SHORT_HAUL_DURATION

    # Flight phase
    altitude_ft = messages["altitude_baro"].to_numpy()
    segment_rocd = flight.segment_rocd(segment_duration, altitude_ft)
    segment_phase = flight.segment_phase(
        segment_rocd,
        altitude_ft,
        threshold_rocd=250.0,
        min_cruise_altitude_ft=min_cruise_altitude_ft,
    )

    # Find any anomalous messages with low altitudes between
    # the start and end of the cruise phase
    # See `_separate_by_cruise_phase` for more comments on logic
    cruise = segment_phase == flight.FlightPhase.CRUISE
    within_cruise = np.bitwise_xor.accumulate(cruise) | cruise
    is_low_altitude = altitude_ft < min_cruise_altitude_ft
    anomalous_phase = within_cruise & is_low_altitude

    # Validate flight trajectory
    has_enough_messages = len(messages) > minimum_messages
    has_cruise_phase = np.any(segment_phase == flight.FlightPhase.CRUISE).item()
    has_no_anomalous_phase = np.any(anomalous_phase).item()

    # Relax constraint for short-haul flights
    if is_short_haul and (not has_cruise_phase) and (not has_no_anomalous_phase):
        has_cruise_phase = np.any(segment_phase == flight.FlightPhase.LEVEL_FLIGHT).item()
        has_no_anomalous_phase = True

    if not (has_enough_messages and has_cruise_phase and has_no_anomalous_phase):
        return False

    # Check that flight is complete
    # First option is the flight is on the ground and low
    final_message = messages.iloc[-1]
    complete_1 = (
        final_message["on_ground"]
        & (final_message["altitude_baro"] < flight.MAX_AIRPORT_ELEVATION)
        & (final_message["speed"] < flight.MAX_ON_GROUND_SPEED)
    )

    # Second option is the flight is in descent and 2 hours of data are available after
    if final_time_available:
        is_descent = np.any(segment_phase[-5:] == flight.FlightPhase.DESCENT) | np.any(
            segment_phase[-5:] == flight.FlightPhase.LEVEL_FLIGHT
        )
        elapsed_time_hrs = (final_time_available - final_message["timestamp"]) / np.timedelta64(
            1, "h"
        )
        complete_2 = (
            (final_message["altitude_baro"] < flight.MAX_AIRPORT_ELEVATION)
            & is_descent
            & (elapsed_time_hrs > 2.0)
        )

        # Third option is 12 hours of data are available after
        complete_3 = elapsed_time_hrs > 12.0
    else:
        complete_2 = False
        complete_3 = False

    # Complete is defined as one of these criteria being satisfied
    is_complete = complete_1 | complete_2 | complete_3

    return has_enough_messages and has_cruise_phase and has_no_anomalous_phase and is_complete


def _downsample_flight(
    messages: pd.DataFrame,
    *,
    time_resolution: str | pd.DateOffset | pd.Timedelta = "10s",
) -> pd.DataFrame:
    """
    Downsample ADS-B messages to a specified time resolution.

    .. warning::
        This function is not used.
        Use :meth:`flight.Flight.resample_and_fill` after creating `Flight`
        instead.

    .. note::
        This function does not interpolate when upsampling.
        Nan values will be dropped.

    Parameters
    ----------
    messages: pd.DataFrame
        ADS-B messages from a single flight trajectory.
    time_resolution: str | pd.DateOffset | pd.Timedelta
        Downsampled time resolution.
        Any input compatible with :meth:`pandas.DataFrame.resample`.
        Defaults to "10s" (10 seconds).

    Returns
    -------
    pd.DataFrame
        Downsampled ADS-B messages for a single flight trajectory.

    See Also
    --------
    :meth:`pandas.DataFrame.resample`
    :meth:`flight.Flight.resample_and_fill`
    """
    mdf = messages.copy()
    mdf = mdf.set_index("timestamp", drop=False)
    resampled = mdf.resample(time_resolution).first()

    # remove rows that do not align with a previous time
    resampled = resampled.loc[resampled["longitude"].notna()]

    # reset original index and return
    return resampled.reset_index(drop=True)


class ValidateTrajectoryHandler:
    """
    Evaluates a trajectory and identifies if it violates any verification rules.

    <LINK HERE TO HOSTED REFERENCE EXAMPLE(S)>.
    """

    CRUISE_ROCD_THRESHOLD_FPS = 4.2  # 4.2 ft/sec ~= 250 ft/min
    CRUISE_LOW_ALTITUDE_THRESHOLD_FT = 15000  # lowest expected cruise altitude
    INSTANTANEOUS_HIGH_GROUND_SPEED_THRESHOLD_MPS = 350  # 350m/sec ~= 780mph ~= 1260kph
    INSTANTANEOUS_LOW_GROUND_SPEED_THRESHOLD_MPS = 45  # 45m/sec ~= 100mph ~= 160kph
    AVG_LOW_GROUND_SPEED_THRESHOLD_MPS = 100  # 120m/sec ~= 223mph ~= 360 kph
    AVG_LOW_GROUND_SPEED_ROLLING_WINDOW_PERIOD_MIN = 30  # rolling period for avg speed comparison
    AIRPORT_DISTANCE_THRESHOLD_KM = 200
    MIN_FLIGHT_LENGTH_HR = 0.4
    MAX_FLIGHT_LENGTH_HR = 19

    # expected schema of pandas dataframe passed on initialization
    SCHEMA = {
        "icao_address": pdtypes.is_string_dtype,
        "flight_id": pdtypes.is_string_dtype,
        "callsign": pdtypes.is_string_dtype,
        "tail_number": pdtypes.is_string_dtype,
        "flight_number": pdtypes.is_string_dtype,
        "aircraft_type_icao": pdtypes.is_string_dtype,
        "airline_iata": pdtypes.is_string_dtype,
        "departure_airport_icao": pdtypes.is_string_dtype,
        "departure_scheduled_time": pdtypes.is_datetime64_any_dtype,
        "arrival_airport_icao": pdtypes.is_string_dtype,
        "arrival_scheduled_time": pdtypes.is_datetime64_any_dtype,
        "ingestion_time": pdtypes.is_datetime64_any_dtype,
        "timestamp": pdtypes.is_datetime64_any_dtype,
        "latitude": pdtypes.is_numeric_dtype,
        "longitude": pdtypes.is_numeric_dtype,
        "collection_type": pdtypes.is_string_dtype,
        "altitude_baro": pdtypes.is_numeric_dtype,
    }

    airports_db = airports.global_airport_database()

    def __init__(self):
        self._df: pd.DataFrame | None = None

    def set(self, trajectory: pd.DataFrame):
        """
        Set a single flight trajectory into handler state.

        Parameters
        ----------
        trajectory
            A dataframe representing a single flight trajectory.
            Must include those columns itemized in ValidateTrajectoryHandler.SCHEMA.
        """
        if len(trajectory) == 0:
            raise BadTrajectoryException("flight trajectory is empty.")
        if len(trajectory["flight_id"].unique()) > 1:
            raise Exception(
                "dataset passed to handler must be for a single flight instance (" "flight_id)."
            )

        self._df = trajectory.copy(deep=True)

    def unset(self):
        """Pop _df from handler state."""
        self._df = None

    @classmethod
    def _find_airport_coords(
        cls, airport_icao: str | None
    ) -> tuple[np.floating, np.floating, np.floating]:
        """
        Find the latitude and longitude for a given airport.

        Parameters
        ----------
        airport_icao
            string representation of the airport's icao code

        Returns
        -------
        (latitude, longitude, alt_ft) of the airport.
        Returns (np.nan, np.nan, np.nan) if it cannot be found.
        """

        if not isinstance(airport_icao, str):
            return np.nan, np.nan, np.nan

        matches = cls.airports_db[cls.airports_db["icao_code"] == airport_icao]
        if len(matches) == 0:
            return np.nan, np.nan, np.nan
        if len(matches) > 1:
            raise ValueError(
                f"found multiple matches for aiport icao {airport_icao} " f"in airports database."
            )

        lat = matches.iloc[0]["latitude"]
        lon = matches.iloc[0]["longitude"]
        alt_ft = matches.iloc[0]["elevation_ft"]
        if (
            not isinstance(lat, np.floating)
            or not isinstance(lon, np.floating)
            or not isinstance(alt_ft, np.floating)
        ):
            raise ValueError(
                f"expected (float, float, float) for lat, lon and alt_ft. "
                f"got: ({lat}, {lon}, {alt_ft})"
            )
        return lat, lon, alt_ft

    @staticmethod
    def _calc_distance_m(lat_0, lon_0, alt_ft_0, lat_f, lon_f, alt_ft_f) -> float:
        """Calculate great circle distance between two lat/lon/alt coordinates."""
        dist_m = math.sqrt(
            (0.3048 * (alt_ft_f - alt_ft_0)) ** 2
            + geo.haversine(
                lons1=np.array(lon_f),
                lats1=np.array(lat_f),
                lons0=np.array(lon_0),
                lats0=np.array(lat_0),
            )
            ** 2
        )
        return dist_m

    @staticmethod
    def _rolling_time_delta_seconds(roll_window: pd.DataFrame):
        """
        Calculate the elapsed time in seconds between two consecutive, time-ordered rows.

        Parameters
        ----------
        roll_window
            A pd.Dataframe with ordered timestamps

        Returns
        -------
        np.nan or integer value for seconds
        """
        if len(roll_window) == 1:
            return np.nan
        t_0 = roll_window.iloc[0]["timestamp"]
        t_f = roll_window.iloc[1]["timestamp"]
        dt_sec = (t_f - t_0).total_seconds()
        if dt_sec < 0:
            raise ValueError(
                "found negative elapsed time. " "window must be ordered in ascending timestamp."
            )

        return int(dt_sec)

    @classmethod
    def _rolling_distance_meters(cls, roll_window: pd.DataFrame):
        """
        Impute the distance travelled (given two consecutive, time-ordered rows).

        Parameters
        ----------
        roll_window
            A pd.Dataframe with ordered timestamps

        Returns
        -------
        np.nan or integer value for seconds
        """
        if len(roll_window) == 1:
            return np.nan

        lat_0 = roll_window.iloc[0]["latitude"]
        lat_f = roll_window.iloc[1]["latitude"]

        lon_0 = roll_window.iloc[0]["longitude"]
        lon_f = roll_window.iloc[1]["longitude"]

        alt_0 = roll_window.iloc[0]["altitude_baro"]
        alt_f = roll_window.iloc[1]["altitude_baro"]

        dist_m = cls._calc_distance_m(lat_0, lon_0, alt_0, lat_f, lon_f, alt_f)
        return dist_m

    @classmethod
    def _rolling_rocd_fps(cls, roll_window: pd.DataFrame) -> float:
        """
        Impute the rate of climb/descent (given two consecutive, time-ordered rows).

        Parameters
        ----------
        roll_window
            A pd.Dataframe with ordered timestamps

        Returns
        -------
        np.nan or float value for rocd in feet per second
        """
        if "elapsed_seconds" not in roll_window.columns:
            raise ValueError("field elapsed_seconds must be present.")

        if len(roll_window) == 1:
            return np.nan

        alt_ft_0 = roll_window.iloc[0]["altitude_baro"]
        alt_ft_f = roll_window.iloc[1]["altitude_baro"]
        alt_ft_dt = alt_ft_f - alt_ft_0
        rocd = alt_ft_dt / roll_window.iloc[1]["elapsed_seconds"]
        if np.isinf(rocd):
            rocd = np.nan
        return rocd

    @classmethod
    def _calc_dist_to_departure_airport(cls, row: pd.Series) -> float:
        """
        Calculate the distance from a given waypoint to the departure airport.

        Returns
        -------
        distance in meters to departure airport.
        np.nan if it cannot be calculated.
        """
        if "departure_airport_lat" not in row.index:
            raise ValueError("field departure_airport_lat must be present.")
        if "departure_airport_lon" not in row.index:
            raise ValueError("field departure_airport_lon must be present.")
        if "departure_airport_alt_ft" not in row.index:
            raise ValueError("field departure_airport_alt_ft must be present.")

        departure_lon = row["departure_airport_lon"]
        departure_lat = row["departure_airport_lat"]
        departure_alt_ft = row["departure_airport_alt_ft"]
        if any(
            [
                np.isnan(departure_lon),
                np.isnan(departure_lat),
                np.isnan(departure_alt_ft),
            ]
        ):
            return np.nan

        return cls._calc_distance_m(
            lon_0=row["longitude"],
            lat_0=row["latitude"],
            alt_ft_0=row["altitude_baro"],
            lon_f=departure_lon,
            lat_f=departure_lat,
            alt_ft_f=departure_alt_ft,
        )

    @classmethod
    def _calc_dist_to_arrival_airport(cls, row: pd.Series) -> float:
        """
        Calculate the distance from a given waypoint to the arrival airport.

        Returns
        -------
        distance in meters to arrival airport.
        np.nan if it cannot be calculated.
        """

        if "arrival_airport_lat" not in row.index:
            raise ValueError("field arrival_airport_lat must be present.")
        if "arrival_airport_lon" not in row.index:
            raise ValueError("field arrival_airport_lon must be present.")
        if "arrival_airport_alt_ft" not in row.index:
            raise ValueError("field arrival_airport_alt_ft must be present.")

        arrival_lon = row["arrival_airport_lon"]
        arrival_lat = row["arrival_airport_lat"]
        arrival_alt_ft = row["arrival_airport_alt_ft"]
        if any([np.isnan(arrival_lon), np.isnan(arrival_lat), np.isnan(arrival_alt_ft)]):
            return np.nan

        return cls._calc_distance_m(
            lon_0=row["longitude"],
            lat_0=row["latitude"],
            alt_ft_0=row["altitude_baro"],
            lon_f=arrival_lon,
            lat_f=arrival_lat,
            alt_ft_f=arrival_alt_ft,
        )

    def _calculate_additional_fields(self):
        """
        Add additional columns to the provided dataframe.

        These additional fields are needed to apply the validation ruleset.
        """
        self._df = self._df.assign(
            elapsed_seconds=[
                self._rolling_time_delta_seconds(window) for window in self._df.rolling(window=2)
            ],
        )
        self._df = self._df.assign(
            elapsed_distance_m=[
                self._rolling_distance_meters(window) for window in self._df.rolling(window=2)
            ],
        )
        self._df = self._df.assign(
            ground_speed_m_s=self._df["elapsed_distance_m"]
            .divide(self._df["elapsed_seconds"])
            .replace(np.inf, np.nan)
        )
        self._df = self._df.assign(
            rocd_fps=[self._rolling_rocd_fps(window) for window in self._df.rolling(window=2)]
        )

        if len(self._df["arrival_airport_icao"].value_counts()) > 1:
            raise ValueError("expected only one airport icao for flight arrival airport.")

        if len(self._df["departure_airport_icao"].value_counts()) > 1:
            raise ValueError("expected only one airport icao for flight departure airport.")

        departure_airport_lat_lon_alt = self._df["departure_airport_icao"].apply(
            self._find_airport_coords
        )
        arrival_airport_lat_lon_alt = self._df["arrival_airport_icao"].apply(
            self._find_airport_coords
        )
        self._df = self._df.assign(
            departure_airport_lat=[coord[0] for coord in departure_airport_lat_lon_alt],
            departure_airport_lon=[coord[1] for coord in departure_airport_lat_lon_alt],
            departure_airport_alt_ft=[coord[2] for coord in departure_airport_lat_lon_alt],
            arrival_airport_lat=[coord[0] for coord in arrival_airport_lat_lon_alt],
            arrival_airport_lon=[coord[1] for coord in arrival_airport_lat_lon_alt],
            arrival_airport_alt_ft=[coord[2] for coord in arrival_airport_lat_lon_alt],
        )

        self._df = self._df.assign(
            departure_airport_dist_m=self._df.apply(self._calc_dist_to_departure_airport, axis=1),
            arrival_airport_dist_m=self._df.apply(self._calc_dist_to_arrival_airport, axis=1),
        )

    @classmethod
    def _is_valid_schema(cls, df: pd.DataFrame) -> None | SchemaError:
        """Verify that a pandas dataframe has required cols, and that they are of required type."""
        col_types = df.dtypes
        cols = list(col_types.index)

        missing_cols = [i for i in cls.SCHEMA if i not in cols]
        if len(missing_cols) > 0:
            return SchemaError(f"trajectory dataframe is missing expected fields: {missing_cols}")

        col_w_bad_dtypes = []
        for col, check_fn in cls.SCHEMA.items():
            is_valid = check_fn(col_types[col])
            if not is_valid:
                col_w_bad_dtypes.append(f"{col} failed check {check_fn.__name__}")

        if len(col_w_bad_dtypes) > 0:
            return SchemaError(
                f"trajectory dataframe has columns with invalid data types. "
                f"\n {col_w_bad_dtypes}"
            )

    def _is_timestamp_sorted(self) -> None | OrderingError:
        """Verify that the data is sorted by waypoint timestamp in ascending order."""
        ts_index = pd.Index(self._df["timestamp"])
        if not ts_index.is_monotonic_increasing:
            return OrderingError(
                "trajectory dataframe must be sorted by timestamp in ascending order."
            )

    def _is_valid_invariant_fields(self) -> None | FlightInvariantFieldViolation:
        """
        Verify that fields expected to be invariant are indeed invariant.

        Presence of null values does not constitute an invariance violation.
        """
        invariant_fields = [
            "icao_address",
            "flight_id",
            "callsign",
            "tail_number",
            "aircraft_type_icao",
            "airline_iata",
            "departure_airport_icao",
            "departure_scheduled_time",
            "arrival_airport_icao",
            "arrival_scheduled_time",
        ]

        violations = []
        for k in invariant_fields:
            unique_vals = list(self._df[k].value_counts().index)
            if len(unique_vals) > 1:
                violations.append(k)

        if len(violations) > 0:
            return FlightInvariantFieldViolation(
                f"the following fields have multiple values for this trajectory. " f"{violations}"
            )

    def _is_valid_duplicate_timestamps(self) -> None | FlightDuplicateTimestamps:
        """Verify that we do not have duplicate timestamps in the trajectory."""
        timestamp_dupe_cnt = self._df["timestamp"].duplicated().sum()
        if timestamp_dupe_cnt > 0:
            return FlightDuplicateTimestamps(
                f"duplicate waypoint timestamps found in "
                f"this trajectory. "
                f"found {timestamp_dupe_cnt} duplicates."
            )

    def _is_valid_flight_length(
        self,
    ) -> None | FlightTooShortError | FlightTooLongError:
        """Verify that the flight is of a reasonable length."""
        flight_duration_sec = (self._df["timestamp"].max() - self._df["timestamp"].min()).seconds
        flight_duration_hours = flight_duration_sec / 60.0 / 60.0

        if flight_duration_hours > self.MAX_FLIGHT_LENGTH_HR:
            return FlightTooLongError(
                f"flight exceeds max duration of {self.MAX_FLIGHT_LENGTH_HR} hours."
                f"this trajectory spans {flight_duration_hours:.2f} hours."
            )

        if flight_duration_hours < self.MIN_FLIGHT_LENGTH_HR:
            return FlightTooShortError(
                f"flight less than min duration of {self.MIN_FLIGHT_LENGTH_HR} hours. "
                f"this trajectory spans {flight_duration_hours:.2f} hours."
            )

    def _is_from_origin_airport(self) -> None | OriginAirportError:
        """Verify that the trajectory origin is a reasonable distance from the origin airport."""
        first_waypoint = self._df.iloc[0]
        first_waypoint_dist_km = first_waypoint["departure_airport_dist_m"] / 1000.0
        if first_waypoint_dist_km > self.AIRPORT_DISTANCE_THRESHOLD_KM:
            return OriginAirportError(
                f"first waypoint in trajectory too far from departure airport icao: "
                f"{first_waypoint['departure_airport_icao']}. "
                f"distance {first_waypoint_dist_km}km is greater than "
                f"threshold of {self.AIRPORT_DISTANCE_THRESHOLD_KM}km."
            )

    def _is_to_destination_airport(self) -> None | DestinationAirportError:
        """
        Verify that the trajectory destination is reasonable distance from the destination airport.

        We do not assume that the destination airports are invariant in the dataframe,
        thus we handle the case of multiple airports listed.
        """
        last_waypoint = self._df.iloc[-1]
        last_waypoint_dist_km = last_waypoint["arrival_airport_dist_m"] / 1000.0
        if last_waypoint_dist_km > self.AIRPORT_DISTANCE_THRESHOLD_KM:
            return DestinationAirportError(
                f"last waypoint in trajectory too far from arrival airport icao: "
                f"{last_waypoint['arrival_airport_icao']}."
                f"distance {last_waypoint_dist_km}km is greater than "
                f"threshold of {self.AIRPORT_DISTANCE_THRESHOLD_KM}km."
            )

    def _is_too_slow(self) -> None | list[FlightTooSlowError]:
        """
        Evaluate the flight trajectory for unreasonably slow speed.

        This is evaluated both for instantaneous discrete steps in the trajectory
        (between consecutive waypoints),
        and,
        on a rolling average basis.

        For instantaneous speed, we clip the trajectory by 10 rows on the head and tail.
        (assuming the trajectory is resampled prior to applying the validation handler,
        that is 10min on head or tail).
        """

        violations: list[FlightTooSlowError] = []

        below_inst_thresh = self._df.iloc[10:, :].iloc[:-10, :][
            self._df["ground_speed_m_s"] <= self.INSTANTANEOUS_LOW_GROUND_SPEED_THRESHOLD_MPS
        ]
        if len(below_inst_thresh) > 0:
            violations.append(
                FlightTooSlowError(
                    f"found {len(below_inst_thresh)} instances where speed between waypoints is "
                    f"below threshold of {self.INSTANTANEOUS_LOW_GROUND_SPEED_THRESHOLD_MPS} m/s. "
                    f" max value: {max(below_inst_thresh['ground_speed_m_s'])}, "
                    f"min value: {min(below_inst_thresh['ground_speed_m_s'])},"
                )
            )

        roll_speed = self._df[["timestamp", "ground_speed_m_s"]]
        roll_speed.set_index("timestamp", inplace=True)
        roll_speed = roll_speed.rolling(
            pd.Timedelta(minutes=self.AVG_LOW_GROUND_SPEED_ROLLING_WINDOW_PERIOD_MIN)
        ).mean()
        # only consider averages occurring at least rolling_avg_period_min minutes
        # after the flight origination (rolling window if backward looking)
        roll_speed = roll_speed[
            roll_speed.index
            > roll_speed.index[0]
            + pd.Timedelta(minutes=self.AVG_LOW_GROUND_SPEED_ROLLING_WINDOW_PERIOD_MIN)
        ]

        below_avg_thresh = roll_speed[
            roll_speed["ground_speed_m_s"] <= self.AVG_LOW_GROUND_SPEED_THRESHOLD_MPS
        ]
        if len(below_avg_thresh) > 0:
            violations.append(
                FlightTooSlowError(
                    f"found {len(below_avg_thresh)} instances where rolling average speed is "
                    f"below threshold of {self.AVG_LOW_GROUND_SPEED_THRESHOLD_MPS} m/s "
                    f"(rolling window of "
                    f"{self.AVG_LOW_GROUND_SPEED_ROLLING_WINDOW_PERIOD_MIN} minutes). "
                    f" max value: {max(below_avg_thresh['ground_speed_m_s'])}, "
                    f"min value: {min(below_avg_thresh['ground_speed_m_s'])},"
                )
            )

        if len(violations) > 0:
            return violations

    def _is_too_fast(self) -> None | FlightTooFastError:
        """
        Evaluate the flight trajectory for reasonably high speed.

        This is evaluated on instantaneous discrete steps between consecutive waypoints.
        """
        above_inst_thresh = self._df[
            self._df["ground_speed_m_s"] >= self.INSTANTANEOUS_HIGH_GROUND_SPEED_THRESHOLD_MPS
        ]
        if len(above_inst_thresh) > 0:
            return FlightTooFastError(
                f"found {len(above_inst_thresh)} instances where speed between waypoints is "
                f"above threshold of {self.INSTANTANEOUS_HIGH_GROUND_SPEED_THRESHOLD_MPS} m/s"
                f" max value: {max(above_inst_thresh['ground_speed_m_s'])}, "
                f"min value: {min(above_inst_thresh['ground_speed_m_s'])},"
            )

    def _is_expected_altitude_profile(
        self,
    ) -> None | list[FlightAltitudeProfileError | RocdError]:
        """
        Evaluate flight altitude profile.

        Failure modes include:
        RocdError
        1) flight climbs above alt threshold,
            then descends below that threshold one or more times,
            before making final descent to land.

        FlightAltitudeProfileError
        2) rate of instantaneous (between consecutive waypoint) climb or descent is above threshold,
           while aircraft is above the cruise altitude.
        """

        violations: list[FlightAltitudeProfileError | RocdError] = []

        # only evaluate rocd errors when at cruising altitude
        rocd_above_thres = self._df[
            (self._df["rocd_fps"].abs() >= self.CRUISE_ROCD_THRESHOLD_FPS)
            & (self._df["altitude_baro"] > self.CRUISE_LOW_ALTITUDE_THRESHOLD_FT)
        ]
        if len(rocd_above_thres) > 0:
            violations.append(
                RocdError(
                    f"flight trajectory has rate of climb/descent values "
                    "between consecutive waypoints that exceed threshold "
                    f"of {self.CRUISE_ROCD_THRESHOLD_FPS} ft/sec. "
                    f"Max value found: {np.nanmax(self._df['rocd_fps'].abs())}"
                )
            )

        alt_below_thresh = self._df["altitude_baro"] <= self.CRUISE_LOW_ALTITUDE_THRESHOLD_FT
        alt_thresh_transitions = alt_below_thresh.rolling(window=2).sum()
        transition_pts = alt_thresh_transitions[alt_thresh_transitions == 1]
        if len(transition_pts) > 2:
            violations.append(
                FlightAltitudeProfileError(
                    f"flight trajectory dropped below altitude threshold"
                    f"of {self.CRUISE_LOW_ALTITUDE_THRESHOLD_FT}ft while in-flight."
                )
            )

        if len(violations) > 0:
            return violations

    @property
    def validation_df(self) -> pd.DataFrame:
        """
        Return an augmented trajectory dataframe.

        Returns
        -------
        dataframe mirroring that provided to the handler,
        but including the additional computed columns that are used in verification.
        e.g. elapsed_sec, ground_speed_m_s, etc.
        """
        violations = self.evaluate()
        fatal_violations = [
            SchemaError,
            FlightDuplicateTimestamps,
            FlightInvariantFieldViolation,
        ]
        if any([v in violations for v in fatal_violations]):
            raise Exception(
                f"validation dataframe cannot be returned "
                f"if flight has violations(s): {violations}"
            )
        # safeguard to ensure this call follows the addition of the columns
        # assumes calculate_additional_fields is idempotent
        self._calculate_additional_fields()
        return self._df

    def evaluate(self) -> list[Exception]:
        """Evaluate the flight trajectory for one or more violations."""

        all_violations: list[Exception] = []

        # Checks; Round 1
        schema_check: None | SchemaError
        schema_check = self._is_valid_schema(self._df)
        all_violations.append(schema_check) if schema_check else None
        if len(all_violations) > 0:
            return all_violations

        # Checks; Round 2
        timestamp_ordering_check: None | OrderingError
        timestamp_ordering_check = self._is_timestamp_sorted()
        (all_violations.append(timestamp_ordering_check) if timestamp_ordering_check else None)

        invariant_fields_check: None | FlightInvariantFieldViolation
        invariant_fields_check = self._is_valid_invariant_fields()
        (all_violations.append(invariant_fields_check) if invariant_fields_check else None)

        duplicate_timestamps_check: None | FlightDuplicateTimestamps
        duplicate_timestamps_check = self._is_valid_duplicate_timestamps()
        (all_violations.append(duplicate_timestamps_check) if duplicate_timestamps_check else None)
        # we escape here if there are violations for the above checks.
        # we do this because some of the following checks assume no invariant field violations,
        #   or timestamp dupes
        if len(all_violations) > 0:
            return all_violations

        # Checks; Round 3
        self._calculate_additional_fields()

        flight_length_check: None | FlightTooShortError | FlightTooLongError
        flight_length_check = self._is_valid_flight_length()
        all_violations.append(flight_length_check) if flight_length_check else None

        origin_airport_check: None | OriginAirportError
        origin_airport_check = self._is_from_origin_airport()
        all_violations.append(origin_airport_check) if origin_airport_check else None

        destination_airport_check: None | DestinationAirportError
        destination_airport_check = self._is_to_destination_airport()
        (all_violations.append(destination_airport_check) if destination_airport_check else None)

        slow_speed_check: None | list[FlightTooSlowError]
        slow_speed_check = self._is_too_slow()
        all_violations.extend(slow_speed_check) if slow_speed_check else None

        fast_speed_check: None | FlightTooFastError
        fast_speed_check = self._is_too_fast()
        all_violations.append(fast_speed_check) if fast_speed_check else None

        altitude_profile_check: None | list[FlightAltitudeProfileError | RocdError]
        altitude_profile_check = self._is_expected_altitude_profile()
        (all_violations.extend(altitude_profile_check) if altitude_profile_check else None)

        if len(all_violations) > 0:
            return all_violations
