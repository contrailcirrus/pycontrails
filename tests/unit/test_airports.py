"""Test `Airports`."""

from __future__ import annotations

import pytest
from pycontrails.core.airports import find_nearest_airport


def test_nearest_airport_identification():
    """Ensure that the nearest airport for a given coordinates is correctly identified.
    """
    # Somewhere near Singapore Changi Airport (1.36 N, 103.99 E)
    longitude_wypt = 103.5
    latitude_wypt = 1.5
    altitude_wypt = 2950
    airport = find_nearest_airport(longitude_wypt, latitude_wypt, altitude_wypt)
    assert airport == "WSSS"

    # Somewhere near London Heathrow Airport (51.47 N, 0.45 W)
    longitude_wypt = -1
    latitude_wypt = 51.6
    altitude_wypt = 2950
    airport = find_nearest_airport(longitude_wypt, latitude_wypt, altitude_wypt)
    assert airport == "EGLL"

    # Somewhere near London Gatwick Airport (51.15 N, 0.18 W)
    longitude_wypt = -0.18
    latitude_wypt = 50.5
    altitude_wypt = 2950
    airport = find_nearest_airport(longitude_wypt, latitude_wypt, altitude_wypt)
    assert airport == "EGKK"

    # In the middle of the South Atlantic Ocean
    longitude_wypt = -30
    latitude_wypt = -60
    altitude_wypt = 2950
    airport = find_nearest_airport(longitude_wypt, latitude_wypt, altitude_wypt)
    assert airport == "N/A"


def test_invalid_altitude():
    """ Ensure that function will not execute if provided altitude is too high.
    """
    longitude_wypt = -1
    latitude_wypt = 51.6
    altitude_wypt = 5000

    with pytest.raises(ValueError):
        find_nearest_airport(longitude_wypt, latitude_wypt, altitude_wypt)

