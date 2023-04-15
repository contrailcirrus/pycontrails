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
        b. Remove erroneous altitude, i.e., altitude above operating limit of aircraft type
    #. Remove terrestrial messages without callsign.
       Most of these messages are below 10,000 feet and from general aviation.
    #. Remove messages when "on_ground" indicator is True, but
       speed > 100 knots or altitude > 15,000 ft
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

    # Remove waypoints with erroneous "on_ground" indicator
    # Thresholds assessed based on scatter plot (100 knots = 185 km/h)
    filt = mdf["on_ground"] & ((mdf["speed"] > 100) | (mdf["altitude_baro"] > 15000))
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
        as output from :func:`cleanup`

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
    #. Remove flights with less than 10 messages
    #. Separate flights by "on_ground" indicator. See `_separate_by_on_ground`.
    #. Separate flights by cruise phase. See `_separate_by_cruise_phase`.

    See Also
    --------
    :func:`cleanup`
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

    pd.Series(
        data=False,
        dtype=bool,
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
        # minimum # of messages > 10
        if len(gp) < 10:
            logger.debug(f"Flight {idx} group too small to create id")
            continue

        # separate flights by "on_ground" column
        gp["flight_id"] = _separate_by_on_ground(gp)
        # flight_id.loc[gp.index] = _separate_by_on_ground(gp)

        # further separate flights by cruise phase analysis
        for fid, fl in gp.groupby("flight_id"):
            gp.loc[fl.index, "flight_id"] = _separate_by_cruise_phase(fl)

        # save flight ids
        flight_id.loc[gp.index] = gp["flight_id"]

    return flight_id

    # # TODO: Check segment length, dt,
    # flight_trajectories = categorise_flight_trajectories(flights, t_cut_off)
    # return flight_trajectories

    # flights = separate_flights_with_multiple_cruise_phase(flights)

    # flights = clean_flight_altitude(flights)

    # # TODO: Check segment length, dt,
    # flight_trajectories = categorise_flight_trajectories(flights, t_cut_off)
    # return flight_trajectories


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
    is_on_ground = messages["on_ground"] & (messages["altitude_baro"] < 15_000)

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
    waypoints between the start and end of the cruise phase.
    For these waypoints that should be at cruise, multiple cruise phases are identified when:

    #. Altitudes fall below 10,000 feet, and
    #. There is a time difference > 15 minutes between messages.

    If multiple flights are identified,
    the cut-off point is specified at waypoint with the largest dt.

    Flight "diversion" is defined when the aircraft descends below 10,000 feet
    and climbs back to cruise altitude to travel to the alternative airport.
    A diversion is identified when all five conditions below are satisfied:

    #. Altitude in any waypoints between the start and end of cruise is < 10,000 ft,
    #. Time difference between waypoints that should be
       at cruise must be < 15 minutes (continuous telemetry)
    #. Segment length between waypoints that should be
       at cruise must be > 500 m (no stationary waypoints),
    #. Time elapsed between waypoint with the lowest altitude
       (during cruise) and final waypoint should be < 2 h,
    #. No waypoints should be on the ground between the start and end of cruise.
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
        min_cruise_altitude_ft = 20_000

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
        condition_1 = np.any(altitude_ft[within_cruise] < 10000)
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


# def categorise_flight_trajectories(
#     flights: list[pd.DataFrame], t_cut_off: pd.Timestamp
# ) -> FlightTrajectories:
#     """
#     Categorise unique flight trajectories (validated, deferred, rejected).

#     Parameters
#     ----------
#     flights: list[pd.DataFrame]
#         List of DataFrames containing waypoints from unique flights.
#     t_cut_off: pd.Timestamp
#         Time of the final recorded waypoint that is provided by the full set of waypoints in the raw ADS-B file

#     Returns
#     -------
#     FlightTrajectories
#         Validated, deferred, and rejected flight trajectories.

#     Notes
#     -----
#     Flights are categorised as
#         (1) Validated: Identified flight trajectory passes all the validation test,
#         (2) Deferred: The remaining trajectory could be in the next file,
#         (3) Rejected: Identified flight trajectory contains anomalies and fails the validation test
#     """
#     validated = list()
#     deferred = list()
#     rejected = list()

#     for df_flight_waypoints in flights:
#         status = validate_flight_trajectory(df_flight_waypoints, t_cut_off)

#         if status["reject"]:
#             rejected.append(df_flight_waypoints.copy())
#         elif (
#             status["n_waypoints"]
#             & status["cruise_phase"]
#             & status["no_altitude_anomaly"]
#             & status["complete"]
#         ):
#             validated.append(df_flight_waypoints.copy())
#         else:
#             deferred.append(df_flight_waypoints.copy())

#     return FlightTrajectories(validated=validated, deferred=deferred, rejected=rejected)


# def validate_flight_trajectory(
#     df_flight_waypoints: pd.DataFrame,
#     t_cut_off: pd.Timestamp,
#     *,
#     min_n_wypt: int = 5,
# ) -> dict[str, bool]:
#     """
#     Ensure that the subset of waypoints only contains one unique flight.

#     Parameters
#     ----------
#     df_flight_waypoints: pd.DataFrame
#         Subset of flight waypoints
#     t_cut_off: pd.Timestamp
#         Time of the final recorded waypoint that is provided by the full set of waypoints in the raw ADS-B file
#     min_n_wypt: int
#         Minimum number of waypoints required for flight to be accepted

#     Returns
#     -------
#     dict[str, bool]
#         Boolean indicating if flight trajectory satisfied the three conditions outlined in `notes`

#     Notes
#     -----
#     The subset of waypoints must satisfy all the following conditions to validate the flight trajectory:
#     (1) Must contain more than `min_n_wypt` waypoints,
#     (2) There must be a cruise phase of flight
#     (3) There must not be waypoints below the minimum cruising altitude throughout the cruise phase.
#     (4) A flight must be "complete", defined when one of these conditions are satisfied:
#         - Final waypoint is on the ground, altitude < 6000 feet and speed < 150 knots (278 km/h),
#         - Final waypoint is < 10,000 feet, in descent, and > 2 h have passed since `current_time_slice`, or
#         - At least 12 h have passed since `t_cut_off` (remaining trajectory might not be recorded).

#     Trajectory is rejected if (1) - (4) are false and > 24 h have passed since `t_cut_off`.
#     """
#     cols_req = ["altitude_baro", "timestamp", "aircraft_type_icao"]
#     validity = np.zeros(4, dtype=bool)

#     if np.any(~pd.Series(cols_req).isin(df_flight_waypoints.columns)):
#         raise KeyError("DataFrame do not contain longitude and/or latitude column.")

#     # Minimum cruise altitude
#     altitude_ceiling_ft = bada_3.ptf_param_dict[
#         df_flight_waypoints["aircraft_type_icao"].iloc[0]
#     ].max_altitude_ft
#     min_cruise_altitude_ft = 0.5 * altitude_ceiling_ft

#     # Flight duration
#     dt_sec = flight.segment_duration(df_flight_waypoints["timestamp"].values)
#     flight_duration_s = np.nansum(dt_sec)
#     is_short_haul = flight_duration_s < 3600

#     # Flight phase
#     flight_phase: FlightPhaseDetailed = identify_phase_of_flight_detailed(
#         df_flight_waypoints["altitude_baro"].values,
#         dt_sec,
#         threshold_rocd=250,
#         min_cruise_alt_ft=min_cruise_altitude_ft,
#     )

#     # Validate flight trajectory
#     validity[0] = len(df_flight_waypoints) > min_n_wypt
#     validity[1] = np.any(flight_phase.cruise)
#     validity[2] = no_altitude_anomaly_during_cruise(
#         df_flight_waypoints["altitude_baro"].values,
#         flight_phase.cruise,
#         min_cruise_alt_ft=min_cruise_altitude_ft,
#     )

#     # Relax constraint for short-haul flights
#     if is_short_haul & (validity[1] is False) & (validity[2] is False):
#         validity[1] = np.any(flight_phase.level_flight)
#         validity[2] = True

#     # Check that flight is complete
#     wypt_final = df_flight_waypoints.iloc[-1]
#     dt = (t_cut_off - wypt_final["timestamp"]) / np.timedelta64(1, "h")

#     if np.all(validity[:3]):  # If first three conditions are valid
#         is_descent = np.any(flight_phase.descent[-5:]) | np.any(flight_phase.level_flight[-5:])
#         complete_1 = (
#             wypt_final["on_ground"]
#             & (wypt_final["altitude_baro"] < 6000)
#             & (wypt_final["speed"] < 150)
#         )
#         complete_2 = (wypt_final["altitude_baro"] < 10000) & is_descent & (dt > 2)
#         complete_3 = dt > 12
#         validity[3] = complete_1 | complete_2 | complete_3

#     return {
#         "n_waypoints": validity[0],
#         "cruise_phase": validity[1],
#         "no_altitude_anomaly": validity[2],
#         "complete": validity[3],
#         "reject": np.all(~validity) & (dt > 24),
#     }


# # TODO: Check this function


# def no_altitude_anomaly_during_cruise(
#     altitude_ft: npt.NDArray[np.float_],
#     flight_phase_cruise: npt.NDArray[np.bool_],
#     *,
#     min_cruise_alt_ft: float,
# ) -> bool:
#     """
#     Check for altitude anomaly during cruise phase of flight.

#     Parameters
#     ----------
#     altitude_ft: npt.NDArray[np.float_]
#         Altitude of each waypoint, [:math:`ft`]
#     flight_phase_cruise: npt.NDArray[np.bool_]
#         Booleans marking if the waypoints are at cruise
#     min_cruise_alt_ft: np.ndarray
#         Minimum threshold altitude for cruise, [:math:`ft`]
#         This is specific for each aircraft type, and can be approximated as 50% of the altitude ceiling.

#     Returns
#     -------
#     bool
#         True if no altitude anomaly is detected.

#     Notes
#     -----
#     The presence of unrealistically low altitudes during the cruise phase of flight is an indicator that the flight
#     trajectory could contain multiple unique flights.
#     """
#     if np.all(~flight_phase_cruise):
#         return False

#     i_cruise_start = np.min(np.argwhere(flight_phase_cruise))
#     i_cruise_end = min(np.max(np.argwhere(flight_phase_cruise)), len(flight_phase_cruise))
#     altitude_at_cruise = altitude_ft[i_cruise_start:i_cruise_end]
#     return np.all(altitude_at_cruise >= min_cruise_alt_ft)


# def downsample_waypoints(
#     df_flight_waypoints: pd.DataFrame, *, time_resolution: int = 10, time_var: str = "timestamp"
# ) -> pd.DataFrame:
#     """
#     Downsample flight waypoints to a specified time resolution

#     Parameters
#     ----------
#     df_flight_waypoints: pd.DataFrame
#         Raw flight waypoints and metadata
#     time_resolution: int
#         Downsampled time resolution, [:math:`s`]
#     time_var: str
#         Time variable in the "df_flight_waypoints" DataFrame

#     Returns
#     -------
#     pd.DataFrame
#         Downsampled flight waypoints
#     """
#     df_flight_waypoints.index = df_flight_waypoints[time_var]
#     df_resampled = df_flight_waypoints.resample(f"{time_resolution}s").first()
#     df_resampled = df_resampled[df_resampled["longitude"].notna()].copy()
#     df_resampled.reset_index(inplace=True, drop=True)
#     return df_resampled
