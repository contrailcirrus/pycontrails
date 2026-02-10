"""Support for reading and parsing the ch-aviation fleet database."""

from __future__ import annotations

import dataclasses
import functools
import pathlib
from typing import Any

import numpy as np
import pandas as pd
from pycontrails.core.flight import Flight
from pycontrails.core.models import Model
from pycontrails.physics import units


@dataclasses.dataclass(frozen=True)
class AircraftChAviation:
    """Registered aircraft properties from ch-aviation database."""

    #: Aircraft tail number -> Registration
    tail_number: str

    #: ICAO 24-bit address -> Hexcode
    tail_number: str

    #: Country of registration -> Aircraft Register
    country_of_registration: str

    #: ICAO aircraft type designator -> Aircraft ICAO
    aircraft_type_icao: str

    #: IATA aircraft type designator -> Aircraft IATA
    aircraft_type_iata: str

    #: Aircraft family -> Aircraft Family
    aircraft_family: str

    #: Aircraft model -> Aircraft Family
    aircraft_model: str

    #: Aircraft manufacturer -> Manufacturer
    manufacturer: str

    #: Engine model -> Engine Subtype
    engine_subtype: str

    #: Engine unique identification number from the ICAO Aircraft Emissions Databank -> ICAO Engine Emission Databank ID
    engine_uid: str

    #: Engine manufacturer -> Engine Manufacturer
    engine_manufacturer: str

    #: Number of engines -> Number of Engines
    n_engines: int

    #: Maximum take-off weight (MTOW), [:math:`kg`] -> MTOW (kg)
    mtow_kg: float

    #: Operator name -> Operator
    operator_name: str

    #: Operator ICAO code -> Operator ICAO
    operator_icao: str

    #: Operator IATA code -> Operator IATA
    operator_iata: str

    #: Operator type -> Operator Type
    operator_type: str

    #: Registered aircraft usage, i.e., passenger/military -> Aircraft Role
    usage: str

    #: Aircraft market group -> Aircraft Market Group
    market_group: str

    #: Number of seats -> "Seats Y" + "Seats YP" + "Seats W" + "Seats C" + "Seats F"
    n_seats: int

    #: Aircraft status (in-service/storage) -> "Status"
    status: str

    #: First flight date -> "First Flight"
    first_flight_date: pd.Timestamp | str

    #: Delivery date -> "Delivery Date"
    delivery_date: pd.Timestamp | str

    #: Date when the aircraft status is changed -> "As of date"
    status_change_date: pd.Timestamp | str

    #: Aircraft age (years)
    # TODO: This will be a derived quantity
    #age: float

    #: Cumulative reported hours -> "Hours"
    cumulative_reported_hours: int

    #: Cumulative reported cycles -> "Cycles"
    cumulative_reported_cycles: int

    #: Average utilization hours -> "Avg. Annual Hours"
    average_utilization_hours: float

    #: Average utilization cycles -> "Avg. Annual Cycles"
    average_utilization_cycles: float
