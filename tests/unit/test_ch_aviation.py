"""Test the ch-aviation fleet database."""

import numpy as np
import pandas as pd
import pytest

from pycontrails.core import Flight
from pycontrails.datalib.ch_aviation import ChAviation
from tests import CH_AVIATION_AVAILABLE

pytestmark = pytest.mark.skipif(not CH_AVIATION_AVAILABLE, reason="ch-aviation data not available")


def test_number_of_unique_tail_numbers():
    """Count the number of unique tail numbers in the ch-aviation dataset."""
    ch_a = ChAviation()
    assert len(ch_a.data) == 47878


def test_database_retrieval():
    """Test the ``ch_aviation.registered_aircraft_properties method``."""
    ch_a = ChAviation()

    # Retrieve DLR Advanced Technology Research Aircraft (ATRA)
    aircraft_props = ch_a.registered_aircraft_properties(tail_number="D-ATRA")
    assert aircraft_props is not None
    assert aircraft_props.aircraft_subfamily == "A320-200"
    assert aircraft_props.engine_subtype == "V2527-A5"
    assert aircraft_props.engine_uid == "01P10IA022"

    # Retrieve A320 with different engine type
    aircraft_props = ch_a.registered_aircraft_properties(tail_number="G-EZUT")
    assert aircraft_props is not None
    assert aircraft_props.aircraft_subfamily == "A320-200"
    assert aircraft_props.engine_subtype == "CFM56-5B4/3"
    assert aircraft_props.engine_uid == "01P08CM105"


def test_invalid_tail_number():
    """Confirm that ``registered_aircraft_properties`` returns None for unknown tail number."""
    ch_a = ChAviation()
    aircraft_props = ch_a.registered_aircraft_properties(tail_number="Killer Whale")
    assert aircraft_props is None


def test_icao_address_fallback():
    """Test the fallback using `icao_address` if `tail_number` is not provided."""
    ch_a = ChAviation()
    aircraft_props = ch_a.registered_aircraft_properties(
        tail_number="Killer Whale",  # The tail number is actually "9V-SGA"
        icao_address="76CCE1",
    )
    assert aircraft_props is not None
    assert aircraft_props.aircraft_subfamily == "A350-900(ULR)"
    assert aircraft_props.engine_subtype == "Trent XWB-84"
    assert aircraft_props.engine_uid == "01P18RR124"


def test_aircraft_age_estimates():
    ch_a = ChAviation()
    aircraft_props = ch_a.registered_aircraft_properties(tail_number="9V-SGA")
    assert aircraft_props is not None
    assert aircraft_props.aircraft_age_yrs(pd.Timestamp("2026-01-01")) == pytest.approx(
        7.28, rel=0.1
    )

    aircraft_props = ch_a.registered_aircraft_properties(tail_number="9V-SGA")
    assert aircraft_props is not None
    assert aircraft_props.aircraft_age_yrs(pd.Timestamp("2028-03-09")) == pytest.approx(
        9.46, rel=0.1
    )


def test_eval_function_with_tail_number():
    """Test ChAviation.eval function when the tail number is available within the fleet database."""
    fl = Flight(
        longitude=[10, 50],
        latitude=[30, 40],
        altitude=[10000, 11000],
        time=[np.datetime64("2023-03-14T00"), np.datetime64("2023-03-14T05")],
        flight_id="Killer Whale",
        tail_number="SE-RET",
    )

    ch_a = ChAviation()
    fl = ch_a.eval(fl)

    expected_fl_attrs = [
        "flight_id",
        "tail_number",
        "msn",
        "country_of_registration",
        "atyp_icao_ch_a",
        "atyp_iata_ch_a",
        "atyp_name_ch_a",
        "atyp_manufacturer",
        "engine_name",
        "engine_uid",
        "engine_manufacturer",
        "n_engines_ch_a",
        "amass_mtow",
        "operator_name",
        "operator_icao",
        "operator_iata",
        "operator_type",
        "aircraft_role",
        "aircraft_market_group",
        "n_seats",
        "status",
        "first_flight_date",
        "delivery_date",
        "aircraft_age_yrs",
        "cumulative_reported_hours",
        "cumulative_reported_hours_ttm",
        "cumulative_reported_cycles",
        "cumulative_reported_cycles_ttm",
        "cumulative_stats_as_of_date",
        "average_annual_hours",
        "average_daily_hours",
        "average_daily_hours_ttm",
        "average_annual_cycles",
        "average_stats_as_of_date",
    ]
    for key in expected_fl_attrs:
        assert key in fl.attrs, f"Missing attribute: {key}"


def test_eval_function_with_uncovered_tail_number():
    """Test ``ChAviation.eval`` when the tail number is not available within the fleet database.

    The fleet dataset only covers new aircraft that are introduced to the global fleet up until
    February-2026. For new aircraft introduced after this date, we estimate the engine based on
    the airline-aircraft look-up table.
    """
    fl = Flight(
        longitude=[10, 50],
        latitude=[30, 40],
        altitude=[10000, 11000],
        time=[np.datetime64("2025-03-14T00"), np.datetime64("2025-03-14T05")],
        flight_id="Killer Whale",
        tail_number="9V-SDD",  # Upcoming 787-10 delivery, not in fleet database
        airline_iata="SQ",
        aircraft_type="B78X",
    )

    ch_a = ChAviation()
    fl = ch_a.eval(fl)

    # Ensure attributes are attached

    expected_fl_attrs = [
        "flight_id",
        "tail_number",
        "airline_iata",
        "aircraft_type",
        "engine_name",
        "engine_uid",
        "operator_name",
        "operator_iata",
    ]
    for key in expected_fl_attrs:
        assert key in fl.attrs, f"Missing attribute: {key}"

    # Check outputs
    assert fl.attrs["engine_name"] == "Trent 1000-J3"
    assert fl.attrs["engine_uid"] == "02P23RR131"
    assert fl.attrs["operator_name"] == "Singapore Airlines"


def test_eval_function_unknown_identifying_information():
    """Test ``ChAviation.eval`` when unknown identifying info is available in flight attributes."""
    fl = Flight(
        longitude=[10, 50],
        latitude=[30, 40],
        altitude=[10000, 11000],
        time=[np.datetime64("2025-03-14T00"), np.datetime64("2025-03-14T05")],
        flight_id="ch-aviation-test",
        airline_iata="xyz",  # Unknown airline
        aircraft_type="B78X",
    )

    ch_a = ChAviation()
    fl2 = ch_a.eval(fl)
    assert fl.attrs == fl2.attrs  # No changes made to flight


def test_eval_function_no_identifying_information():
    """Test ``ChAviation.eval`` when no identifying info is available in flight attributes."""
    fl = Flight(
        longitude=[10, 50],
        latitude=[30, 40],
        altitude=[10000, 11000],
        time=[np.datetime64("2025-03-14T00"), np.datetime64("2025-03-14T05")],
        flight_id="ch-aviation-test",
    )

    ch_a = ChAviation()
    fl2 = ch_a.eval(fl)
    assert fl.attrs == fl2.attrs  # No changes made to flight
