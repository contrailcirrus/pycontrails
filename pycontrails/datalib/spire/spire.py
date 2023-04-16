"""`Spire Aviation <https://spire.com/aviation/>`_ ADS-B data support."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from pycontrails.core import datalib, flight
from pycontrails.core.fleet import Fleet
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
        self.messages = clean(messages)

    def load_fleet(self) -> Fleet:
        pass


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
    #. Ensure columns have the dtype as specified in :attr:`FIELDS`
    #. Remove messages with tail number "VARIOUS"
       and aircraft types "N/A", "GNDT", "GRND", "ZZZZ"
    #. If pycontrails.ext.bada installed:
        a. Remove aircraft types not covered by BADA 3 (primarily helicopters)
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
        "altitude_baro",
        "on_ground",
        "icao_address",
        "aircraft_type_icao",
        "tail_number",
    ]
    filt = mdf[non_null_cols].isna().any(axis=1)
    mdf = mdf.loc[~filt]

    # Ensure columns have the correct dtype
    mdf = mdf.astype(MESSAGE_DTYPES)

    # convert timestamp into a timezone naive pd.Timestamp
    mdf["timestamp"] = pd.to_datetime(mdf["timestamp"]).dt.tz_localize(None)

    # Remove messages with tail number set to VARIOUS
    # or aircraft type set to "N/A", "GNDT", "GRND", "ZZZZ"
    filt = (mdf["tail_number"] == "VARIOUS") & (
        mdf["aircraft_type_icao"].isin(["N/A", "GNDT", "GRND", "ZZZZ"])
    )
    mdf = mdf.loc[~filt]

    # Remove terrestrial waypoints without callsign
    # Most of these waypoints are below 10,000 feet and from general aviation
    filt = (mdf["callsign"].isna()) & (mdf["collection_type"] == "terrestrial")
    mdf = mdf[~filt]

    # Fill missing callsigns for satellite records
    callsigns_missing = (mdf["collection_type"] == "satellite") & (mdf["callsign"].isna())
    mdf.loc[callsigns_missing, "callsign"] = None  # reset values to be None
    for icao_address, gp in mdf.loc[
        mdf["icao_address"].isin(mdf.loc[callsigns_missing, "icao_address"].unique()),
        ["icao_address", "callsign", "collection_type"],
    ].groupby("icao_address", sort=False):
        mdf.loc[gp.index, "callsign"] = gp["callsign"].fillna(method="ffill").fillna(method="bfill")

    # Remove messages with erroneous "on_ground" indicator
    filt = mdf["on_ground"] & (
        (mdf["speed"] > flight.MAX_ON_GROUND_SPEED)
        | (mdf["altitude_baro"] > flight.MAX_AIRPORT_ELEVATION)
    )
    mdf = mdf[~filt]

    # Drop duplicates by icao_address and timestamp
    mdf.drop_duplicates(subset=["icao_address", "timestamp"], inplace=True)

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
        gp = _clean_flight_altitude(gp)

        # separate flights by "on_ground" column
        gp["flight_id"] = _separate_by_on_ground(gp)

        # further separate flights by cruise phase analysis
        for fid, fl in gp.groupby("flight_id"):
            gp.loc[fl.index, "flight_id"] = _separate_by_cruise_phase(fl)

        # save flight ids
        flight_id.loc[gp.index] = gp["flight_id"]

    return flight_id


def _clean_flight_altitude(
    messages: pd.DataFrame,
    *,
    noise_threshold_ft: float = 25,
    threshold_altitude_ft: float | None = None,
) -> pd.DataFrame:
    """
    Clean erroneous and noisy altitude on a single flight.

    Parameters
    ----------
    messages: pd.DataFrame
        ADS-B messages from a single flight trajectory.
    noise_threshold_ft: float
        Altitude difference threshold to identify noise, [:math:`ft`]
        Barometric altitude from ADS-B telemetry is reported at increments of 25 ft.
    threshold_altitude_ft: float
        Altitude will be checked and corrected above this threshold, [:math:`ft`]
        Currently set to :attr:`flight.MAX_AIRPORT_ELEVATION` ft.

    Returns
    -------
    pd.DataFrame
        ADS-B messages with filtered altitude.

    Notes
    -----
    #. If pycontrails.ext.bada installed,
       remove erroneous altitude, i.e., altitude above operating limit of aircraft type
    #. Remove noise in cruise altitude where flights oscillate
       between 25 ft due to noise in ADS-B telemetry

    See Also
    --------
    :func:`flight.filter_altitude`
    """
    threshold_altitude_ft = threshold_altitude_ft or flight.MAX_AIRPORT_ELEVATION

    mdf = messages.copy()

    # Use BADA 3 to support filtering
    try:
        from pycontrails.ext.bada import BADA3

        bada3 = BADA3()

        # Try remove aircraft types not covered by BADA 3 (mainly helicopters)
        atyps = list(bada3.synonym_dict.keys())
        mdf = mdf.loc[mdf["aircraft_type_icao"].isin(atyps)]

        # Remove erroneous altitude, i.e., altitude above operating limit of aircraft type
        altitude_ceiling_ft = bada3.ptf_param_dict[
            mdf["aircraft_type_icao"].iloc[0]
        ].max_altitude_ft
        is_above_ceiling = mdf["altitude_baro"] > altitude_ceiling_ft
        mdf = mdf.loc[~is_above_ceiling]

    except (ImportError, FileNotFoundError):
        pass

    # Remove noise in cruise altitude by rounding up/down to the nearest flight level.
    altitude_ft = mdf["altitude_baro"].to_numpy()
    d_alt_ft = np.diff(altitude_ft, prepend=np.nan)
    is_noise = (np.abs(d_alt_ft) <= noise_threshold_ft) & (altitude_ft > threshold_altitude_ft)
    mdf.loc[is_noise, "altitude_baro"] = np.round(altitude_ft[is_noise] / 1000) * 1000

    return mdf


def _separate_by_on_ground(messages: pd.DataFrame) -> pd.DataFrame:
    """Separate individual flights by "on_ground" column.

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
    if "flight_id" in messages:
        flight_id = messages["flight_id"]
    else:
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

    # find end of flight indexes using "on_ground"
    end_of_flight = (~is_on_ground).astype(int).diff(periods=-1) == -1

    # identify each individual flight using cumsum magic
    for _, gp in messages.groupby(end_of_flight.cumsum()):
        flight_id.loc[gp.index] = generate_flight_id(
            gp["timestamp"].iloc[0], gp["callsign"].iloc[0]
        )

    return flight_id


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
    if "flight_id" in messages:
        flight_id = messages["flight_id"]
    else:
        flight_id = pd.Series(
            data=generate_flight_id(messages["timestamp"].iloc[0], messages["callsign"].iloc[0]),
            dtype=object,
            index=messages.index,
        )

    # Use BADA 3 to calculate max altitude
    try:
        from pycontrails.ext.bada import BADA3

        bada3 = BADA3()

        # Remove erroneous altitude, i.e., altitude above operating limit of aircraft type
        altitude_ceiling_ft = bada3.ptf_param_dict[
            messages["aircraft_type_icao"].iloc[0]
        ].max_altitude_ft
        min_cruise_altitude_ft = 0.5 * altitude_ceiling_ft

    except (ImportError, FileNotFoundError):
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
    cruise = segment_phase == flight.FLIGHT_PHASE["cruise"]

    # fill between the first and last cruise phase indicator
    # this represents the flight phase after takeoff climb and before descent to landing
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
        i_lowest_altitude = (
            np.argwhere(within_cruise & is_low_altitude & (altitude_ft == altitude_min_diverted))[
                0
            ][0]
            + 1
        )

        # get reference to "on_ground"
        ground_indicator = messages["on_ground"].to_numpy()

        # Check for flight diversion
        condition_1 = np.any(altitude_ft[within_cruise] < flight.MAX_AIRPORT_ELEVATION)
        condition_2 = np.all(segment_duration[within_cruise] < (15 * 60))
        condition_3 = np.all(segment_length[within_cruise] > 500)
        condition_4 = np.nansum(segment_duration[i_lowest_altitude:]) < (2 * 60 * 60)
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

    for idx, gp in messages[
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

        # Remove erroneous altitude, i.e., altitude above operating limit of aircraft type
        altitude_ceiling_ft = bada3.ptf_param_dict[
            messages["aircraft_type_icao"].iloc[0]
        ].max_altitude_ft
        min_cruise_altitude_ft = 0.5 * altitude_ceiling_ft

    except (ImportError, FileNotFoundError):
        min_cruise_altitude_ft = flight.MIN_CRUISE_ALTITUDE

    # Flight duration
    segment_duration = flight.segment_duration(messages["timestamp"].values)
    is_short_haul = np.nansum(segment_duration) < flight.SHORT_HAUL_DURATION

    # Flight phase
    altitude_ft = messages["altitude_baro"].to_numpy()
    segment_rocd = flight.segment_rocd(segment_duration, altitude_ft)
    segment_phase = flight.segment_phase(
        segment_rocd,
        altitude_ft,
        threshold_rocd=250,
        min_cruise_altitude_ft=min_cruise_altitude_ft,
    )

    # Find any anomalous messages with low altitudes between
    # the start and end of the cruise phase
    # See `_separate_by_cruise_phase` for more comments on logic
    cruise = segment_phase == flight.FLIGHT_PHASE["cruise"]
    within_cruise = np.bitwise_xor.accumulate(cruise) | cruise
    is_low_altitude = altitude_ft < min_cruise_altitude_ft
    anomalous_phase = within_cruise & is_low_altitude

    # Validate flight trajectory
    has_enough_messages = len(messages) > minimum_messages
    has_cruise_phase = np.any(segment_phase == flight.FLIGHT_PHASE["cruise"]).astype(bool)
    has_no_anomalous_phase = np.any(anomalous_phase).astype(bool)

    # Relax constraint for short-haul flights
    if is_short_haul and (not has_cruise_phase) and (not has_no_anomalous_phase):
        has_cruise_phase = np.any(segment_phase == flight.FLIGHT_PHASE["level_flight"])
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
        is_descent = np.any(segment_phase[-5:] == flight.FLIGHT_PHASE["descent"]) | np.any(
            segment_phase[-5:] == flight.FLIGHT_PHASE["level_flight"]
        )
        elapsed_time_hrs = (final_time_available - final_message["timestamp"]) / np.timedelta64(
            1, "h"
        )
        complete_2 = (
            (final_message["altitude_baro"] < flight.MAX_AIRPORT_ELEVATION)
            & is_descent
            & (elapsed_time_hrs > 2)
        )

        # Third option is 12 hours of data are available after
        complete_3 = elapsed_time_hrs > 12
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
