"""Test `Spire` datalib."""

from __future__ import annotations

import pytest
import numpy as np
import pandas as pd
import pycontrails.core.ads_b as adsb
import pycontrails.datalib.spire.data_cleaning as spire
from .conftest import get_static_path
from pycontrails.core.ads_b import remove_noise_in_cruise_altitude


def test_noise_removal_cruise_altitude():
    """Check that noise in cruise altitude is removed."""
    altitude_ft = np.array([40000, 39975, 40000, 40000, 39975, 40000, 40000, 40025, 40025, 40000])
    altitude_ft_cleaned = remove_noise_in_cruise_altitude(np.copy(altitude_ft))
    np.testing.assert_array_equal(altitude_ft_cleaned, 40000)


def test_identify_unique_flight_trajectories():
    """Test Spire data cleaning algorithm to identify unique flight trajectories from raw ADS-B data."""
    df_icao_address = pd.read_parquet(get_static_path("flight-spire-data-cleaning.pq"))
    df_icao_address = adsb.downsample_waypoints(df_icao_address, time_resolution=10, time_var="timestamp")
    df_icao_address = df_icao_address.astype({"on_ground": bool})
    df_icao_address = spire._fill_missing_callsign_for_satellite_waypoints(df_icao_address)

    #: [CASE 1] Identify unique flights based on metadata
    df_test_1 = df_icao_address.copy()
    flights = adsb.separate_unique_flights_from_waypoints(
        df_test_1, columns=["tail_number", "aircraft_type_icao", "callsign"]
    )
    assert len(flights) == 4        # Identified four unique flights

    #: [CASE 2] Identify multiple flights with the same call sign using the ground indicator
    callsigns = ["SHT88J", "BAW506"]
    is_callsign = df_icao_address["callsign"].isin(callsigns)
    df_test_2 = df_icao_address[is_callsign].copy()
    df_test_2["callsign"] = "killer-whale"
    flights = adsb.separate_unique_flights_from_waypoints(
        df_test_2, columns=["tail_number", "aircraft_type_icao", "callsign"]
    )
    assert len(flights) == 1        # Unable to identify unique flights because metadata is the same

    flights = spire._separate_by_ground_indicator(flights)
    assert len(flights) == 2        # Identified two unique flights
    return
