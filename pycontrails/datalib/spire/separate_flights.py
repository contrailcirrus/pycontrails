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

    # TODO: Remove ground waypoints?
    # TODO: Check segment length, dt, multiple cruise phase
    # TODO: Validate separated flight trajectories
    # TODO: Categorise flights

    return


def _fill_missing_callsign_for_satellite_waypoints(df_flight_waypoints: pd.DataFrame):
    if np.any(df_flight_waypoints["collection_type"] == "satellite"):
        is_missing = df_flight_waypoints["callsign"].isna() & (df_flight_waypoints["collection_type"] == "satellite")

        if np.any(is_missing):
            df_flight_waypoints["callsign"] = df_flight_waypoints["callsign"].fillna(method="ffill")
            df_flight_waypoints["callsign"] = df_flight_waypoints["callsign"].fillna(method="bfill")

    return df_flight_waypoints
