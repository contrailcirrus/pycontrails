"""`Spire Aviation <https://spire.com/aviation/>`_ ADS-B data support."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from pycontrails.core import flight
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
        mdf.loc[gp.index, "callsign"] = gp["callsign"].fillna(method="ffill").fillna(method="bfill")

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
    mdf["altitude_baro"] = flight.filter_altitude(mdf["altitude_baro"].to_numpy())

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
    mdf.set_index("timestamp", inplace=True, drop=False)
    resampled = mdf.resample(time_resolution).first()

    # remove rows that do not align with a previous time
    resampled = resampled.loc[resampled["longitude"].notna()]

    # reset original index and return
    return resampled.reset_index(drop=True)
