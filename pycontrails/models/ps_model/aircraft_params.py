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
    (1) manufacturer        Aircraft manufacturer name
    (2) aircraft_type       Specific aircraft type variant

    -------------------------------------
    AIRCRAFT PARAMETERS
    -------------------------------------
    (3) winglets            Does the aircraft type contain winglets? (True/False)
    (4) wing_surface_area   Reference wing surface area, [:math:`m^{2}`]
    (5) wing_aspect_ratio   Wing aspect ratio, wing_span**2 / wing_surface_area
    (6) wing_span           Wing span, [:math:`m`]
    (7) wing_constant       A constant used in the wave drag model, capturing the aerofoil technology factor and wing
                            geometry
    (8) delta_2             Induced drag wing-fuselage interference factor
    (9) cos_sweep           Cosine of wing sweep angle measured at the 1/4 chord line
    (10) psi_0              Aircraft geometry drag parameter
    (11) x_ref              Threshold condition where the terminating shockwave moves onto the rear part of the wing
    (12) j_1                Wave drag parameter 1
    (13) j_2                Wave drag parameter 2
    (14) j_3                Wave drag parameter 3

    -------------------------------------
    ENGINE PARAMETERS
    -------------------------------------
    (15) ff_idle_sls        Fuel mass flow rate under engine idle and sea level static conditions, summed over
                            all engines, [:math:`kg s^{-1}`]
    (16) m_des              Design optimum Mach number where the fuel mass flow rate is at a minimum.
    (17) c_t_des            Design optimum engine thrust coefficient where the fuel mass flow rate is at a minimum.
    (18) eta_1              Multiplier for maximum overall propulsion efficiency model
    (19) eta_2              Exponent for maximum overall propulsion efficiency model

    -------------------------------------
    OPERATIONAL PARAMETERS
    -------------------------------------
    (20) amass_mtow         Aircraft maximum take-off weight, [:math:`kg`]
    (21) amass_mlw          Aircraft maximum landing weight, [:math:`kg`]
    (22) amass_mzfw         Aircraft maximum zero fuel weight, [:math:`kg`]
    (23) amass_oew          Aircraft operating empty weight, [:math:`kg`]
    (24) amass_mpl          Aircraft maximum payload, [:math:`kg`]
    (25) max_altitude_ft    Maximum altitude, [:math:`ft`]
    (26) max_mach_num       Maximum operational Mach number
    (27) wingspan           Aircraft wingspan, [:math:`m`]
    (28) fuselage_width     Aircraft fuselage width, [:math:`m`]
    """
    def __init__(self, df_aircraft_engine: pd.Series) -> None:
        self.manufacturer: str = df_aircraft_engine["Manufacturer"]
        self.aircraft_type: str = df_aircraft_engine["Type"]

        # Aircraft parameters
        self.winglets: bool = df_aircraft_engine["winglets"] == "yes"
        self.wing_surface_area: float = df_aircraft_engine["Sref/m2"]
        self.wing_aspect_ratio: float = df_aircraft_engine["AR"]
        self.wing_span = (self.wing_aspect_ratio * self.wing_surface_area)**0.5
        self.wing_constant: float = df_aircraft_engine["wing constant"]
        self.delta_2: float = df_aircraft_engine["delta_2"]
        self.cos_sweep: float = df_aircraft_engine["cos_sweep"]
        self.psi_0: float = df_aircraft_engine["psi_0"]
        self.x_ref: float = df_aircraft_engine["X_o"]
        self.j_1: float = df_aircraft_engine["j_1"]
        self.j_2: float = df_aircraft_engine["j_2"]
        self.j_3: float = 70        # Constant for all aircraft-engine types for now, but may vary in the future

        # Engine parameters
        self.ff_idle_sls: float = df_aircraft_engine["nominal((mf)FI)SLS (kg/s)"]
        self.m_des: float = df_aircraft_engine["(M) des"]
        self.c_t_des: float = df_aircraft_engine["(CT)des"]
        self.eta_1: float = df_aircraft_engine["eta_1"]
        self.eta_2: float = df_aircraft_engine["eta_2"]

        # Operational parameters
        self.amass_mtow: float = df_aircraft_engine["MTOM (kg)"]
        self.amass_mlw: float = df_aircraft_engine["MLM (kg)"]
        self.amass_mzfw: float = df_aircraft_engine["MZFM (kg)"]
        self.amass_oew: float = df_aircraft_engine["(OEM)i (kg)"]
        self.amass_mpl: float = df_aircraft_engine["(MPM)i (kg)"]
        self.max_altitude_ft: float = df_aircraft_engine["MaxAlt/ft"]
        self.max_mach_num: float = df_aircraft_engine["MMO"]
        self.wingspan: float = df_aircraft_engine["span/m"]
        self.fuselage_width: float = df_aircraft_engine["bf/m"]


def get_aircraft_engine_params(ps_file_path: pathlib.Path) -> Mapping[str, AircraftEngineParams]:
    """Extract aircraft-engine parameters for each aircraft type supported by the PS model."""
    df = pd.read_csv(ps_file_path, index_col=0)
    return {atyp_icao: AircraftEngineParams(df_aircraft_engine) for atyp_icao, df_aircraft_engine in df.iterrows()}
