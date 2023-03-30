import pandas as pd
import numpy as np
import pycontrails.core.ads_b as adsb


def identify_unique_flights(df_flight_waypoints: pd.DataFrame):
    # For each subset of waypoints with the same ICAO address, identify unique flights
    df_flight_waypoints = adsb.downsample_waypoints(df_flight_waypoints, time_resolution=10, time_var="timestamp")
    df_flight_waypoints = _fill_missing_callsign_for_satellite_waypoints(df_flight_waypoints)
    flights = adsb.separate_unique_flights_from_waypoints(
        df_flight_waypoints, columns=["tail_number", "aircraft_type_icao", "callsign"]
    )
    flights = _separate_by_ground_indicator(flights)
    print(" ")
    # TODO: Check segment length, dt, multiple cruise phase
    # TODO: Validate separated flight trajectories
    # TODO: Categorise flights
    return


def _fill_missing_callsign_for_satellite_waypoints(df_flight_waypoints: pd.DataFrame) -> pd.DataFrame:
    """
    Backward and forward filling of missing callsigns.

    ADS-B waypoints that are recorded by satellite often do not include the callsign metadata.

    Parameters
    ----------
    df_flight_waypoints: pd.DataFrame
        Waypoints with the same ICAO address, must contain the "callsign" and "collection_type" columns

    Returns
    -------
    pd.DataFrame
        Waypoints with the same ICAO address, where the "callsign" is filled
    """
    if np.any(df_flight_waypoints["collection_type"] == "satellite"):
        is_missing = df_flight_waypoints["callsign"].isna() & (df_flight_waypoints["collection_type"] == "satellite")

        if np.any(is_missing):
            df_flight_waypoints["callsign"] = df_flight_waypoints["callsign"].fillna(method="ffill")
            df_flight_waypoints["callsign"] = df_flight_waypoints["callsign"].fillna(method="bfill")

    return df_flight_waypoints


def _separate_by_ground_indicator(flights: list[pd.DataFrame]) -> list[pd.DataFrame]:
    """
    Identify and separate unique flights using the ground indicator.

    Parameters
    ----------
    flights: list[pd.DataFrame]
        List of DataFrames containing waypoints from unique flights.

    Returns
    -------
    list[pd.DataFrame]
        List of DataFrames containing waypoints from unique flights that are separated by ground indicator.

    Notes
    -----
    Waypoints in the taxi phase will be omitted.
    """
    flights_checked = list()

    for df_flight_waypoints in flights:
        df_flight_waypoints.reset_index(inplace=True, drop=True)
        is_on_ground = df_flight_waypoints["on_ground"] & (df_flight_waypoints["altitude_baro"] < 6000)

        # Continue to next flight if all recorded waypoints are not on the ground
        if np.all(~is_on_ground):
            flights_checked.append(df_flight_waypoints)
            continue

        # Separate flights: only include flights if there are waypoints that are off the ground
        i_ground = df_flight_waypoints[is_on_ground].index.values
        i_cutoff_1 = i_ground[0] + 1
        i_cutoff_2 = i_ground[-1]
        df_flight_1 = df_flight_waypoints.iloc[:i_cutoff_1].copy()
        df_flight_2 = df_flight_waypoints.iloc[i_cutoff_2:].copy()

        if ~np.all(df_flight_1["on_ground"]):
            flights_checked.append(df_flight_1)

        if ~np.all(df_flight_2["on_ground"]):
            flights_checked.append(df_flight_2)

        #: Check waypoints between i_cutoff_1 and i_cutoff_2,
        #: Include subset if there are waypoints off the ground, as there could be unique flights.
        df_flight_3 = df_flight_waypoints.iloc[i_cutoff_1:i_cutoff_2].copy()

        if ~np.all(df_flight_3["on_ground"]):
            is_off_the_ground = ~df_flight_3["on_ground"].astype(bool)
            flights_checked.append(df_flight_3[is_off_the_ground].copy())

    return flights_checked


def separate_flights_multiple_cruise_phase():
    return


def validate_flight_trajectory():
    return
