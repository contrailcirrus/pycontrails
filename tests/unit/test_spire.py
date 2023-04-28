"""Test `Spire` datalib."""

from __future__ import annotations

import pandas as pd
import pytest

from pycontrails.datalib import spire
from pycontrails.datalib.spire.spire import (
    _clean_trajectory_altitude,
    _separate_by_cruise_phase,
    _separate_by_on_ground,
)

from .conftest import get_static_path


def test_clean() -> None:
    """Test algorithms to identify and separate unique flight trajectories."""

    df = pd.read_parquet(get_static_path("flight-spire-data-cleaning.pq"))
    assert len(df.groupby(["icao_address", "tail_number", "aircraft_type_icao", "callsign"])) == 5

    clean = spire.clean(df)
    assert (
        len(clean.groupby(["icao_address", "tail_number", "aircraft_type_icao", "callsign"])) == 5
    )


def test_separate_using_ground_indicator():
    """Test algorithms to identify unique flight trajectories from on the ground indicator."""
    df = pd.read_parquet(get_static_path("flight-spire-data-cleaning.pq"))
    cdf = spire.clean(df)

    # Construct erroneous messages consisting of two unique flights with the same callsign
    test = cdf.loc[cdf["callsign"].isin(["SHT88J", "BAW506"])].copy()
    test["callsign"] = "killer-whale-1"

    # Unable to identify unique flights because metadata is the same
    assert len(test.groupby(["icao_address", "tail_number", "aircraft_type_icao", "callsign"])) == 1

    flight_ids = _separate_by_on_ground(test)
    assert len(flight_ids.unique()) == 2

    # Should handle slightly noisy "on_ground" signal
    # (there is already a single "on_ground" outlier near the end in the dataset)
    test.loc[2131:2132, "on_ground"] = False
    test.loc[4500:4502, "on_ground"] = True

    flight_ids = _separate_by_on_ground(test)
    assert len(flight_ids.unique()) == 2


def test_separate_with_multiple_cruise_phase():
    """Test algorithms to identify unique flight trajectories with multiple cruise phases."""
    df = pd.read_parquet(get_static_path("flight-spire-data-cleaning.pq"))
    cdf = spire.clean(df)

    # Construct erroneous messages consisting of two unique flights with the same callsign
    test = cdf.loc[cdf["callsign"].isin(["BAW506", "BAW507"])].copy()
    test["callsign"] = "killer-whale-2"

    # In `spire.identify_flights`, we need to clean up individual trajectory
    # altitudes before _separate_by_cruise_phase method works
    test = _clean_trajectory_altitude(test)

    # Unable to identify unique flights because metadata is the same
    assert len(test.groupby(["icao_address", "tail_number", "aircraft_type_icao", "callsign"])) == 1

    flight_ids = _separate_by_cruise_phase(test)
    assert len(flight_ids.unique()) == 1


@pytest.mark.xfail(reason="Test not complete")
def test_identify_flight_diversion() -> None:
    """Test algorithms to identify flight diversion, and no separation is done."""
    df = pd.read_parquet(get_static_path("flight-spire-data-cleaning.pq"))
    cdf = spire.clean(df)

    # Construct flight that is diverted
    test = cdf.loc[cdf["callsign"] == "BAW506"].copy()
    test.reset_index(drop=True, inplace=True)
    altitude_ft_adjusted = test["altitude_baro"].to_numpy()
    altitude_ft_adjusted[230:346] = test["altitude_baro"].values[539:655]
    altitude_ft_adjusted[346:462] = test["altitude_baro"].values[15:131]
    altitude_ft_adjusted[407:573] = 25000
    test["altitude_baro"] = altitude_ft_adjusted

    # Unable to identify unique flights because metadata is the same
    assert len(test.groupby(["icao_address", "tail_number", "aircraft_type_icao", "callsign"])) == 1

    flight_ids = _separate_by_cruise_phase(test)
    assert len(flight_ids.unique()) == 1
