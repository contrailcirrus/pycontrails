"""Test `Spire` datalib."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .conftest import get_static_path


def test_separate_unique_flights_using_metadata():
    """Test algorithms to identify and separate unique flight trajectories from raw ADS-B data based on the
    difference in flight metadata.
    """
    df_icao_address = pd.read_parquet(get_static_path("flight-spire-data-cleaning.pq"))
    df_icao_address = data_cleaning.downsample_waypoints(
        df_icao_address, time_resolution=10, time_var="timestamp"
    )
    df_icao_address = df_icao_address.astype({"on_ground": bool})
    df_icao_address = data_cleaning._fill_missing_callsign_for_satellite_waypoints(df_icao_address)

    df_test_1 = df_icao_address.copy()
    flights = data_cleaning.separate_unique_flights_from_waypoints(
        df_test_1, columns=["tail_number", "aircraft_type_icao", "callsign"]
    )
    assert len(flights) == 4  # Identified four unique flights


def test_separate_unique_flights_using_ground_indicator():
    """Test algorithms to identify and separate unique flight trajectories from raw ADS-B data based on the
    ground indicator.
    """
    df_icao_address = pd.read_parquet(get_static_path("flight-spire-data-cleaning.pq"))
    df_icao_address = data_cleaning.downsample_waypoints(
        df_icao_address, time_resolution=10, time_var="timestamp"
    )
    df_icao_address = df_icao_address.astype({"on_ground": bool})
    df_icao_address = data_cleaning._fill_missing_callsign_for_satellite_waypoints(df_icao_address)

    # Construct erroneous subset of waypoints consisting of two unique flights with the same callsign
    callsigns = ["SHT88J", "BAW506"]
    is_callsign = df_icao_address["callsign"].isin(callsigns)
    df_test_2 = df_icao_address[is_callsign].copy()
    df_test_2["callsign"] = "killer-whale-1"

    # Unable to identify unique flights because metadata is the same
    flights = data_cleaning.separate_unique_flights_from_waypoints(
        df_test_2, columns=["tail_number", "aircraft_type_icao", "callsign"]
    )
    assert len(flights) == 1

    flights = data_cleaning.separate_flights_with_ground_indicator(flights)
    assert len(flights) == 2  # Identified two unique flights


def test_separate_unique_flights_with_multiple_cruise_phase():
    """Test algorithms to identify and separate unique flight trajectories with multiple cruise phases."""
    df_icao_address = pd.read_parquet(get_static_path("flight-spire-data-cleaning.pq"))
    df_icao_address = data_cleaning.downsample_waypoints(
        df_icao_address, time_resolution=10, time_var="timestamp"
    )
    df_icao_address = df_icao_address.astype({"on_ground": bool})
    df_icao_address = data_cleaning._fill_missing_callsign_for_satellite_waypoints(df_icao_address)

    # Construct erroneous subset of waypoints consisting of two unique flights
    callsigns = ["BAW506", "BAW507"]
    is_callsign = df_icao_address["callsign"].isin(callsigns)
    df_test_3 = df_icao_address[is_callsign].copy()
    df_test_3["callsign"] = "killer-whale-2"
    df_test_3.reset_index(drop=True, inplace=True)
    keep_rows = np.ones(len(df_test_3), dtype=bool)
    keep_rows[635:750] = False
    df_test_3 = df_test_3[keep_rows].copy()

    # Calculate flight phase
    dt_sec = np.diff(df_test_3["timestamp"].values, append=np.datetime64("NaT")) / np.timedelta64(
        1, "s"
    )
    flight_phase: data_cleaning.FlightPhaseDetailed = flight.segment_phase(
        df_test_3["altitude_baro"].values, dt_sec, threshold_rocd=250, min_cruise_altitude_ft=20000
    )

    # Identify multiple cruise phase
    trajectory_check: data_cleaning.TrajectoryCheck = (
        data_cleaning.identify_multiple_cruise_phases_and_diversion(
            flight_phase,
            df_test_3["longitude"].values,
            df_test_3["latitude"].values,
            df_test_3["altitude_baro"].values,
            dt_sec,
            ground_indicator=df_test_3["on_ground"].values,
        )
    )
    assert trajectory_check.multiple_cruise_phase

    # Unable to identify unique flights because metadata is the same
    flights = data_cleaning.separate_unique_flights_from_waypoints(
        df_test_3, columns=["tail_number", "aircraft_type_icao", "callsign"]
    )
    assert len(flights) == 1

    flights = data_cleaning.separate_flights_with_multiple_cruise_phase(flights)
    assert len(flights) == 2


def test_identify_flight_diversion():
    """Test algorithms to identify flight diversion, and no separation is done."""
    df_icao_address = pd.read_parquet(get_static_path("flight-spire-data-cleaning.pq"))
    df_icao_address = data_cleaning.downsample_waypoints(
        df_icao_address, time_resolution=10, time_var="timestamp"
    )
    df_icao_address = df_icao_address.astype({"on_ground": bool})
    df_icao_address = data_cleaning._fill_missing_callsign_for_satellite_waypoints(df_icao_address)

    # Construct flight that is diverted
    df_test_4 = df_icao_address[df_icao_address["callsign"] == "BAW506"].copy()
    df_test_4.reset_index(drop=True, inplace=True)
    altitude_ft_adjusted = np.copy(df_test_4["altitude_baro"].values)
    altitude_ft_adjusted[230:346] = df_test_4["altitude_baro"].values[539:655]
    altitude_ft_adjusted[346:462] = df_test_4["altitude_baro"].values[15:131]
    altitude_ft_adjusted[407:573] = 25000
    df_test_4["altitude_baro"] = altitude_ft_adjusted

    # Calculate flight phase
    dt_sec = np.diff(df_test_4["timestamp"].values, append=np.datetime64("NaT")) / np.timedelta64(
        1, "s"
    )
    flight_phase: data_cleaning.FlightPhaseDetailed = flight.segment_phase(
        df_test_4["altitude_baro"].values, dt_sec, threshold_rocd=250, min_cruise_altitude_ft=20000
    )

    # Identify multiple cruise phase
    trajectory_check: data_cleaning.TrajectoryCheck = (
        data_cleaning.identify_multiple_cruise_phases_and_diversion(
            flight_phase,
            df_test_4["longitude"].values,
            df_test_4["latitude"].values,
            df_test_4["altitude_baro"].values,
            dt_sec,
            ground_indicator=df_test_4["on_ground"].values,
        )
    )
    assert trajectory_check.multiple_cruise_phase is False
    assert trajectory_check.flight_diversion

    # Unable to identify unique flights because metadata is the same
    flights = data_cleaning.separate_unique_flights_from_waypoints(
        df_test_4, columns=["tail_number", "aircraft_type_icao", "callsign"]
    )
    assert len(flights) == 1

    # It is a flight diversion, so no separation
    flights = data_cleaning.separate_flights_with_multiple_cruise_phase(flights)
    assert len(flights) == 1
