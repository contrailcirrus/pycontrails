"""Test `Airports`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pycontrails.core import airports

try:
    import requests

    r = requests.head("https://github.com", timeout=10)
except Exception:
    offline = True
else:
    offline = False


@pytest.fixture
def airport_database() -> pd.DataFrame:
    """Download copy of airport database."""
    return airports.global_airport_database()


@pytest.mark.skipif(offline, reason="offline")
def test_global_airports_database(airport_database: pd.DataFrame) -> None:
    """Ensure that airport database exists."""
    assert "type" in airport_database
    assert "elevation_m" in airport_database


@pytest.mark.skipif(offline, reason="offline")
def test_nearest_airport_identification(airport_database: pd.DataFrame) -> None:
    """Ensure that the nearest airport for a given coordinates is correctly identified."""
    # Somewhere near Singapore Changi Airport (1.36 N, 103.99 E)
    longitude = 103.5
    latitude = 1.5
    altitude = 2950
    airport_icao = airports.find_nearest_airport(airport_database, longitude, latitude, altitude)
    assert airport_icao == "WSSS"

    # Somewhere near London Heathrow Airport (51.47 N, 0.45 W)
    longitude = -1
    latitude = 51.6
    altitude = 2950
    airport_icao = airports.find_nearest_airport(airport_database, longitude, latitude, altitude)
    assert airport_icao == "EGLL"

    # Somewhere near London Gatwick Airport (51.15 N, 0.18 W)
    longitude = -0.18
    latitude = 50.5
    altitude = 2950
    airport_icao = airports.find_nearest_airport(airport_database, longitude, latitude, altitude)
    assert airport_icao == "EGKK"

    # In the middle of the South Atlantic Ocean
    longitude = -30
    latitude = -60
    altitude = 2950
    airport_icao = airports.find_nearest_airport(airport_database, longitude, latitude, altitude)
    assert airport_icao is None


@pytest.mark.skipif(offline, reason="offline")
def test_invalid_altitude(airport_database: pd.DataFrame) -> None:
    """Ensure that function will not execute if provided altitude is too high."""
    longitude = -1
    latitude = 51.6
    altitude = 5000

    with pytest.raises(ValueError):
        airports.find_nearest_airport(airport_database, longitude, latitude, altitude)


def test_distance_to_airports(airport_database: pd.DataFrame) -> None:
    """Ensure distance to airports are correctly identified."""
    apts = airport_database[airport_database["iata_code"].isin(["BOS", "JFK"])]

    # a point in the massachusetts bay close to BOS
    longitude = -70.842554
    latitude = 42.387466
    altitude = 500
    distance = airports.distance_to_airports(apts, longitude, latitude, altitude)

    assert len(distance) == 2
    assert np.round(distance[0]) == 13616.0  # close to BOS
    assert np.round(distance[1]) == 312342.0  # not terribly far from JFK
