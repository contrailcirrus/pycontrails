"""ECMWF Data Access."""

from __future__ import annotations

from pycontrails.datalib.spire.spire import (
    clean,
    generate_flight_id,
    identify_flights,
    is_valid_trajectory,
    validate_flights,
)

__all__ = [
    "clean",
    "generate_flight_id",
    "identify_flights",
    "is_valid_trajectory",
    "validate_flights",
]
