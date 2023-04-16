"""ECMWF Data Access."""

from __future__ import annotations

from pycontrails.datalib.spire.spire import (
    Spire,
    clean,
    generate_flight_id,
    identify_flights,
    is_valid_trajectory,
    validate_flights,
)

__all__ = [
    "Spire",
    "clean",
    "generate_flight_id",
    "identify_flights",
    "is_valid_trajectory",
    "validate_flights",
]
