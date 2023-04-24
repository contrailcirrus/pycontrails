"""Simple dataclasses for working with PS aircraft performance data."""

from __future__ import annotations

import pathlib
import pandas as pd
from typing import Mapping


class AircraftEngineParams:
    """Store extracted aircraft and engine parameters for each aircraft type.

    -------------------------------------
    AIRCRAFT INFORMATION
    -------------------------------------
    (1) aircraft_type       Specific aircraft type variant
    (2) manufacturer        Aircraft manufacturer name

    -------------------------------------
    AIRCRAFT PARAMETERS
    -------------------------------------
    (3) winglets            Does the aircraft type contain winglets? (True/False)
    (4) wing_surface_area   Reference wing surface area, [:math:`m^{2}`]
    (5) delta_2             Induced drag wing-fuselage interference factor
    (6) cos_sweep           cosine of wing sweep angle measured at the 1/4 chord line
    (7) wing_aspect_ratio   Wing aspect ratio, wing_span**2 / wing_surface_area
    (8) psi_0               Aircraft geometry drag parameter
    """
    def __init__(self, df_aircraft_engine: pd.Series) -> None:
        self.aircraft_type: str = df_aircraft_engine["Type"]
        self.manufacturer: str = df_aircraft_engine["Manufacturer"]

        # Aircraft parameters
        self.winglets: bool = df_aircraft_engine["winglets"]
        self.wing_surface_area: float = df_aircraft_engine["Sref/m2"]
        self.delta_2: float = df_aircraft_engine["delta_2"]
        self.cos_sweep: float = df_aircraft_engine["cos_sweep"]
        self.wing_aspect_ratio: float = df_aircraft_engine["AR"]
        self.psi_0: float = df_aircraft_engine["psi_0"]
        # TODO: Incomplete


def get_aircraft_engine_params(ps_file_path: pathlib.Path) -> Mapping[str, AircraftEngineParams]:
    """Extract aircraft-engine parameters for each aircraft type supported by the PS model."""
    df = pd.read_csv(ps_file_path, index_col=0)
    return {atyp_icao: AircraftEngineParams(df_aircraft_engine) for atyp_icao, df_aircraft_engine in df.iterrows()}
