"""Support for Himawari-8/9 satellite data access."""

from pycontrails.datalib.himawari.header_struct import (
    HEADER_STRUCT_SCHEMA,
    parse_himawari_header,
)
from pycontrails.datalib.himawari.himawari import (
    HIMAWARI_8_9_SWITCH_DATE,
    HIMAWARI_8_BUCKET,
    HIMAWARI_9_BUCKET,
    Himawari,
    HimawariRegion,
    extract_visualization,
    to_true_color,
)

__all__ = [
    "HEADER_STRUCT_SCHEMA",
    "HIMAWARI_8_9_SWITCH_DATE",
    "HIMAWARI_8_BUCKET",
    "HIMAWARI_9_BUCKET",
    "Himawari",
    "HimawariRegion",
    "extract_visualization",
    "parse_himawari_header",
    "to_true_color",
]
