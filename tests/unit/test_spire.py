"""Test `Spire` datalib."""

from __future__ import annotations

import pytest
import numpy as np
import pandas as pd
import pycontrails.core.ads_b as adsb
import pycontrails.datalib.spire.data_cleaning as spire
from .conftest import get_static_path


def test_noise_removal_cruise_altitude():
    """Check that noise in cruise altitude is removed."""
    altitude_ft = np.array([40000, 39975, 40000, 40000, 39975, 40000, 40000, 40025, 40025, 40000])
    altitude_ft_cleaned = adsb.remove_noise_in_cruise_altitude(np.copy(altitude_ft))
    np.testing.assert_array_equal(altitude_ft_cleaned, 40000)


def test_separate_unique_flights_using_metadata():
    """Test algorithms to identify and separate unique flight trajectories from raw ADS-B data based on the
    difference in flight metadata.
    """
    df_icao_address = pd.read_parquet(get_static_path("flight-spire-data-cleaning.pq"))
    df_icao_address = adsb.downsample_waypoints(df_icao_address, time_resolution=10, time_var="timestamp")
    df_icao_address = df_icao_address.astype({"on_ground": bool})
    df_icao_address = spire._fill_missing_callsign_for_satellite_waypoints(df_icao_address)

    df_test_1 = df_icao_address.copy()
    flights = adsb.separate_unique_flights_from_waypoints(
        df_test_1, columns=["tail_number", "aircraft_type_icao", "callsign"]
    )
    assert len(flights) == 4        # Identified four unique flights


def test_separate_unique_flights_using_ground_indicator():
    """Test algorithms to identify and separate unique flight trajectories from raw ADS-B data based on the
    ground indicator.
    """
    df_icao_address = pd.read_parquet(get_static_path("flight-spire-data-cleaning.pq"))
    df_icao_address = adsb.downsample_waypoints(df_icao_address, time_resolution=10, time_var="timestamp")
    df_icao_address = df_icao_address.astype({"on_ground": bool})
    df_icao_address = spire._fill_missing_callsign_for_satellite_waypoints(df_icao_address)

    # Construct erroneous subset of waypoints consisting of two unique flights with the same callsign
    callsigns = ["SHT88J", "BAW506"]
    is_callsign = df_icao_address["callsign"].isin(callsigns)
    df_test_2 = df_icao_address[is_callsign].copy()
    df_test_2["callsign"] = "killer-whale-1"

    # Unable to identify unique flights because metadata is the same
    flights = adsb.separate_unique_flights_from_waypoints(
        df_test_2, columns=["tail_number", "aircraft_type_icao", "callsign"]
    )
    assert len(flights) == 1

    flights = spire.separate_flights_with_ground_indicator(flights)
    assert len(flights) == 2        # Identified two unique flights


def test_separate_unique_flights_with_multiple_cruise_phase():
    """Test algorithms to identify and separate unique flight trajectories with multiple cruise phases.
    """
    df_icao_address = pd.read_parquet(get_static_path("flight-spire-data-cleaning.pq"))
    df_icao_address = adsb.downsample_waypoints(df_icao_address, time_resolution=10, time_var="timestamp")
    df_icao_address = df_icao_address.astype({"on_ground": bool})
    df_icao_address = spire._fill_missing_callsign_for_satellite_waypoints(df_icao_address)

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
    dt_sec = np.diff(df_test_3["timestamp"].values, append=np.datetime64("NaT")) / np.timedelta64(1, "s")
    flight_phase: adsb.FlightPhaseDetailed = adsb.identify_phase_of_flight_detailed(
        df_test_3["altitude_baro"].values, dt_sec, threshold_rocd=250, min_cruise_alt_ft=20000
    )

    # Identify multiple cruise phase
    trajectory_check: adsb.TrajectoryCheck = adsb.identify_multiple_cruise_phases_and_diversion(
        flight_phase, df_test_3["longitude"].values, df_test_3["latitude"].values,
        df_test_3["altitude_baro"].values, dt_sec, ground_indicator=df_test_3["on_ground"].values
    )
    assert trajectory_check.multiple_cruise_phase

    # Unable to identify unique flights because metadata is the same
    flights = adsb.separate_unique_flights_from_waypoints(
        df_test_3, columns=["tail_number", "aircraft_type_icao", "callsign"]
    )
    assert len(flights) == 1

    flights = spire.separate_flights_with_multiple_cruise_phase(flights)
    assert len(flights) == 2


def test_identify_flight_diversion():
    """Test algorithms to identify flight diversion, and no separation is done.
    """
    df_icao_address = pd.read_parquet(get_static_path("flight-spire-data-cleaning.pq"))
    df_icao_address = adsb.downsample_waypoints(df_icao_address, time_resolution=10, time_var="timestamp")
    df_icao_address = df_icao_address.astype({"on_ground": bool})
    df_icao_address = spire._fill_missing_callsign_for_satellite_waypoints(df_icao_address)

    # Construct flight that is diverted
    df_test_4 = df_icao_address[df_icao_address["callsign"] == "BAW506"].copy()
    df_test_4.reset_index(drop=True, inplace=True)
    altitude_ft_adjusted = np.copy(df_test_4["altitude_baro"].values)
    altitude_ft_adjusted[230:346] = df_test_4["altitude_baro"].values[539:655]
    altitude_ft_adjusted[346:462] = df_test_4["altitude_baro"].values[15:131]
    altitude_ft_adjusted[407:573] = 25000
    df_test_4["altitude_baro"] = altitude_ft_adjusted

    # Calculate flight phase
    dt_sec = np.diff(df_test_4["timestamp"].values, append=np.datetime64("NaT")) / np.timedelta64(1, "s")
    flight_phase: adsb.FlightPhaseDetailed = adsb.identify_phase_of_flight_detailed(
        df_test_4["altitude_baro"].values, dt_sec, threshold_rocd=250, min_cruise_alt_ft=20000
    )

    # Identify multiple cruise phase
    trajectory_check: adsb.TrajectoryCheck = adsb.identify_multiple_cruise_phases_and_diversion(
        flight_phase, df_test_4["longitude"].values, df_test_4["latitude"].values,
        df_test_4["altitude_baro"].values, dt_sec, ground_indicator=df_test_4["on_ground"].values
    )
    assert trajectory_check.multiple_cruise_phase is False
    assert trajectory_check.flight_diversion

    # Unable to identify unique flights because metadata is the same
    flights = adsb.separate_unique_flights_from_waypoints(
        df_test_4, columns=["tail_number", "aircraft_type_icao", "callsign"]
    )
    assert len(flights) == 1

    # It is a flight diversion, so no separation
    flights = spire.separate_flights_with_multiple_cruise_phase(flights)
    assert len(flights) == 1
