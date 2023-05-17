"""Simple dataclasses for working with PS aircraft performance data."""

from __future__ import annotations

import dataclasses
import functools
import pathlib
from typing import Mapping

import pandas as pd


@dataclasses.dataclass(frozen=True)
class AircraftEngineParams:
    """Store extracted aircraft and engine parameters for each aircraft type.

    -------------------------------------
    AIRCRAFT INFORMATION
    -------------------------------------
    - manufacturer        Aircraft manufacturer name
    - aircraft_type       Specific aircraft type variant
    - n_engine            Number of engines

    -------------------------------------
    AIRCRAFT PARAMETERS
    -------------------------------------
    - winglets            Does the aircraft type contain winglets? (True/False)
    - wing_surface_area   Reference wing surface area, [:math:`m^{2}`]
    - wing_aspect_ratio   Wing aspect ratio, wing_span**2 / wing_surface_area
    - wing_span           Wing span, [:math:`m`]
    - wing_constant       A constant used in the wave drag model, capturing the aerofoil technology factor and wing
                            geometry
    - delta_2             Induced drag wing-fuselage interference factor
    - cos_sweep           Cosine of wing sweep angle measured at the 1/4 chord line
    - psi_0              Aircraft geometry drag parameter
    - x_ref              Threshold condition where the terminating shockwave moves onto the rear part of the wing
    - j_1                Wave drag parameter 1
    - j_2                Wave drag parameter 2
    - j_3                Wave drag parameter 3

    -------------------------------------
    ENGINE PARAMETERS
    -------------------------------------
    - ff_idle_sls        Fuel mass flow rate under engine idle and sea level static conditions, summed over
                            all engines, [:math:`kg s^{-1}`]
    - ff_max_sls         Fuel mass flow rate at take-off and sea level static conditions, summed over
                            all engines, [:math:`kg s^{-1}`]
    - f_00               Maximum thrust force that can be supplied by the engine at sea level static conditions,
                            summed over all engines, [:math:`N`].
    - m_des              Design optimum Mach number where the fuel mass flow rate is at a minimum.
    - c_t_des            Design optimum engine thrust coefficient where the fuel mass flow rate is at a minimum.
    - eta_1              Multiplier for maximum overall propulsion efficiency model
    - eta_2              Exponent for maximum overall propulsion efficiency model

    -------------------------------------
    OPERATIONAL PARAMETERS
    -------------------------------------
    - amass_mtow         Aircraft maximum take-off weight, [:math:`kg`]
    - amass_mlw          Aircraft maximum landing weight, [:math:`kg`]
    - amass_mzfw         Aircraft maximum zero fuel weight, [:math:`kg`]
    - amass_oew          Aircraft operating empty weight, [:math:`kg`]
    - amass_mpl          Aircraft maximum payload, [:math:`kg`]
    - max_altitude_ft    Maximum altitude, [:math:`ft`]
    - max_mach_num       Maximum operational Mach number
    - wingspan           Aircraft wingspan, [:math:`m`]
    - fuselage_width     Aircraft fuselage width, [:math:`m`]
    """

    manufacturer: str
    aircraft_type: str
    n_engine: int

    winglets: bool
    wing_surface_area: float
    wing_aspect_ratio: float
    wing_span: float
    wing_constant: float
    delta_2: float
    cos_sweep: float
    psi_0: float
    x_ref: float
    j_1: float
    j_2: float
    j_3: float

    ff_idle_sls: float
    ff_max_sls: float
    f_00: float
    m_des: float
    c_t_des: float
    eta_1: float
    eta_2: float

    amass_mtow: float
    amass_mlw: float
    amass_mzfw: float
    amass_oew: float
    amass_mpl: float

    amass_ref: float
    max_altitude_ft: float
    max_mach_num: float
    wingspan: float
    fuselage_width: float


def _row_to_aircraft_engine_params(row: pd.Series) -> AircraftEngineParams:
    wing_aspect_ratio = row["AR"]
    wing_surface_area = row["Sref/m2"]
    amass_mtow = row["MTOM (kg)"]
    return AircraftEngineParams(
        manufacturer=row["Manufacturer"],
        aircraft_type=row["Type"],
        n_engine=row["n_engine"],
        winglets=row["winglets"] == "yes",
        wing_surface_area=wing_surface_area,
        wing_aspect_ratio=wing_aspect_ratio,
        wing_span=(wing_aspect_ratio * wing_surface_area) ** 0.5,
        wing_constant=row["wing constant"],
        delta_2=row["delta_2"],
        cos_sweep=row["cos_sweep"],
        psi_0=row["psi_0"],
        x_ref=row["Xo"],
        j_1=row["j_1"],
        j_2=row["j_2"],
        # Constant for all aircraft-engine types for now, but may vary in the future
        j_3=70.0,
        # Engine parameters
        ff_idle_sls=row["(mf)idle SLS (kg/s)"],
        ff_max_sls=row["(mf)max T/O SLS (kg/s)"],
        f_00=row["nominal (F00)ISA (kn)"] * 1000.0,
        m_des=row["(M) des"],
        c_t_des=row["(CT)des"],
        eta_1=row["eta_1"],
        eta_2=row["eta_2"],
        # Operational parameters,
        amass_mtow=amass_mtow,
        amass_mlw=row["MLM (kg)"],
        amass_mzfw=row["MZFM (kg)"],
        amass_oew=row["(OEM)i (kg)"],
        amass_mpl=row["(MPM)i (kg)"],
        # Assume reference mass is equal to 70% of the take-off mass (Ian Poll)
        amass_ref=amass_mtow * 0.7,
        max_altitude_ft=row["MaxAlt/ft"],
        max_mach_num=row["MMO"],
        wingspan=row["span/m"],
        fuselage_width=row["bf/m"],
    )


@functools.cache
def get_aircraft_engine_params(ps_file_path: pathlib.Path) -> Mapping[str, AircraftEngineParams]:
    """Extract aircraft-engine parameters for each aircraft type supported by the PS model."""
    df = pd.read_csv(ps_file_path, index_col=0)
    return {
        atyp_icao: _row_to_aircraft_engine_params(df_aircraft_engine)
        for atyp_icao, df_aircraft_engine in df.iterrows()
    }
