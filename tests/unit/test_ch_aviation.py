"""Test the ch-aviation fleet database."""

import pytest
import numpy as np
import pandas as pd
from pycontrails.core import Flight
from pycontrails.datalib.ch_aviation import ChAviation


def test_number_of_unique_tail_numbers():
    """Count the number of unique tail numbers in the ch-aviation dataset."""
    ch_a = ChAviation()
    assert ch_a.data.index.nunique() == 48826


def test_database_retrieval():
    """Test the ``ch_aviation.registered_aircraft_properties method``."""
    ch_a = ChAviation()

    # Retrieve DLR Advanced Technology Research Aircraft (ATRA)
    aircraft_props = ch_a.registered_aircraft_properties(tail_number="D-ATRA")
    assert aircraft_props.aircraft_subfamily == "A320-200"
    assert aircraft_props.engine_subtype == "V2527-A5"
    assert aircraft_props.engine_uid == "01P10IA022"

    # Retrieve A320 with different engine type
    aircraft_props = ch_a.registered_aircraft_properties(tail_number="G-EZUT")
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
        tail_number="Killer Whale", icao_address="76CCE1"
    )  # The tail number is actually "9V-SGA"
    assert aircraft_props.aircraft_subfamily == "A350-900(ULR)"
    assert aircraft_props.engine_subtype == "Trent XWB-84"
    assert aircraft_props.engine_uid == "01P18RR124"


def test_aircraft_age_estimates():
    ch_a = ChAviation()
    aircraft_props = ch_a.registered_aircraft_properties(
        tail_number="9V-SGA", date=pd.to_datetime("2026-01-01")
    )
    assert aircraft_props.aircraft_age_yrs == pytest.approx(7.28, rel=0.1)

    aircraft_props = ch_a.registered_aircraft_properties(
        tail_number="9V-SGA", date=pd.to_datetime("2028-03-09")
    )
    assert aircraft_props.aircraft_age_yrs == pytest.approx(9.46, rel=0.1)

    # If date is not provided, then age should be nan
    aircraft_props = ch_a.registered_aircraft_properties(tail_number="9V-SGA")
    assert np.isnan(aircraft_props.aircraft_age_yrs)


# TODO: Below function no longer needed?

def test_aircraft_status_change():
    """Test the ``Cirium.registered_aircraft_properties`` function for aircraft status changes."""
    cirium = Cirium()

    # G-EUUT should be in storage from 2020-03-28
    aircraft_props = cirium.registered_aircraft_properties(
        tail_number="G-EUUT", date=pd.Timestamp("2020-04-01")
    )
    assert aircraft_props.status_change_date == pd.Timestamp("2020-03-28")
    assert aircraft_props.status == "Storage"

    # G-EUUT should be back to service from 2021-04-23
    aircraft_props = cirium.registered_aircraft_properties(
        tail_number="G-EUUT", date=pd.Timestamp("2021-05-01")
    )
    assert aircraft_props.status_change_date == pd.Timestamp("2021-04-23")
    assert aircraft_props.status == "In Service"

    # If date is not given, function should return the most recent status of the aircraft
    aircraft_props = cirium.registered_aircraft_properties(tail_number="G-EUUT")
    assert aircraft_props.status_change_date == pd.Timestamp("2021-04-23")
    assert aircraft_props.status == "In Service"


def test_eval_function_with_tail_number():
    """Test the Cirium.eval function when the tail number is available within the Cirium dataset."""
    fl = Flight(
        longitude=[10, 50],
        latitude=[30, 40],
        altitude=[10000, 11000],
        time=[np.datetime64("2023-03-14T00"), np.datetime64("2023-03-14T05")],
        attrs={
            "flight_id": "Killer Whale",
            "tail_number": "SE-RET",
        },
    )

    cirium = Cirium()
    fl = cirium.eval(fl)

    fl_attrs = [
        "flight_id",
        "tail_number",
        "crs",
        "country_of_registration",
        "atyp_name_cirium",
        "atyp_icao_cirium",
        "atyp_manufacturer",
        "atyp_modifiers",
        "engine_name",
        "engine_uid",
        "engine_manufacturer",
        "engine_propulsion_type",
        "n_engines_cirium",
        "apu_name",
        "amass_mtow",
        "amass_mzfw",
        "amass_oew",
        "amass_mpl",
        "amass_fuel_capacity",
        "operator_name",
        "operator_icao",
        "operator_iata",
        "operator_type",
        "aircraft_usage",
        "aircraft_market_class",
        "n_seats",
        "status",
        "last_update",
        "aircraft_age",
        "cumulative_reported_hours",
        "cumulative_reported_cycles",
        "average_utilization_hours",
    ]

    assert np.all(pd.Series(fl.attrs.keys()).isin(fl_attrs))


def test_eval_function_with_uncovered_tail_number():
    """Test ``Cirium.eval`` when the tail number is not available within the Cirium dataset.

    The Cirium dataset only covers new aircraft that are introduced to the global fleet up until
    31-December-2023. For new aircraft introduced after this date, we estimate the engine based on
    the airline-aircraft look-up table.
    """
    fl = Flight(
        longitude=[10, 50],
        latitude=[30, 40],
        altitude=[10000, 11000],
        time=[np.datetime64("2025-03-14T00"), np.datetime64("2025-03-14T05")],
        attrs={
            "flight_id": "Killer Whale",
            "tail_number": "9V-SJI",
            "airline_iata": "SQ",
            "aircraft_type": "A359",
        },
    )

    cirium = Cirium()
    fl = cirium.eval(fl)

    # Ensure attributes are attached

    fl_attrs = [
        "flight_id",
        "tail_number",
        "airline_iata",
        "aircraft_type",
        "engine_name",
        "engine_uid",
        "operator_name",
        "operator_iata",
    ]

    assert np.all(pd.Series(fl.attrs.keys()).isin(fl_attrs))

    # Check outputs
    assert fl.attrs["engine_name"] == "Trent XWB-84"
    assert fl.attrs["engine_uid"] == "01P18RR124"
    assert fl.attrs["operator_name"] == "Singapore Airlines"
