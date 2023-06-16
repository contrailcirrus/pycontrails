"""Simple dataclasses for working with PS aircraft performance data."""

from __future__ import annotations

import dataclasses
import functools
import pathlib
from typing import Any, Mapping

import pandas as pd


@dataclasses.dataclass(frozen=True)
class PSAircraftEngineParams:
    """Store extracted aircraft and engine parameters for each aircraft type.

    -------------------------------------
    AIRCRAFT INFORMATION
    -------------------------------------
    - ``manufacturer``        Aircraft manufacturer name
    - ``aircraft_type``       Specific aircraft type variant
    - ``n_engine``            Number of engines

    -------------------------------------
    AIRCRAFT PARAMETERS
    -------------------------------------
    - ``winglets``            Does the aircraft type contain winglets? (True/False)
    - ``wing_surface_area``   Reference wing surface area, [:math:`m^{2}`]
    - ``wing_aspect_ratio``   Wing aspect ratio, ``wing_span**2 / wing_surface_area``
    - ``wing_span``           Wing span, [:math:`m`]
    - ``wing_constant``       A constant used in the wave drag model, capturing the
      aerofoil technology factor and wing geometry
    - ``delta_2``             Induced drag wing-fuselage interference factor
    - ``cos_sweep``           Cosine of wing sweep angle measured at the 1/4 chord line
    - ``psi_0``               Aircraft geometry drag parameter
    - ``x_ref``               Threshold condition where the terminating shockwave moves
      onto the rear part of the wing
    - ``j_1``                 Wave drag parameter 1
    - ``j_2``                 Wave drag parameter 2
    - ``j_3``                 Wave drag parameter 3

    -------------------------------------
    ENGINE PARAMETERS
    -------------------------------------
    - ``ff_idle_sls``        Fuel mass flow rate under engine idle and sea level
      static conditions, summed over all engines, [:math:`kg s^{-1}`]
    - ``ff_max_sls``         Fuel mass flow rate at take-off and sea level static conditions,
      summed over all engines, [:math:`kg s^{-1}`]
    - ``f_00``               Maximum thrust force that can be supplied by the engine at
      sea level static conditions, summed over all engines, [:math:`N`].
    - ``m_des``              Design optimum Mach number where the fuel mass flow rate
      is at a minimum.
    - ``c_t_des``            Design optimum engine thrust coefficient where the fuel mass
      flow rate is at a minimum.
    - ``eta_1``              Multiplier for maximum overall propulsion efficiency model
    - ``eta_2``              Exponent for maximum overall propulsion efficiency model

    -------------------------------------
    OPERATIONAL PARAMETERS
    -------------------------------------
    - ``amass_mtow``         Aircraft maximum take-off weight, [:math:`kg`]
    - ``amass_mlw``          Aircraft maximum landing weight, [:math:`kg`]
    - ``amass_mzfw``         Aircraft maximum zero fuel weight, [:math:`kg`]
    - ``amass_oew``          Aircraft operating empty weight, [:math:`kg`]
    - ``amass_mpl``          Aircraft maximum payload, [:math:`kg`]
    - ``max_altitude_ft``    Maximum altitude, [:math:`ft`]
    - ``max_mach_num``       Maximum operational Mach number
    - ``fuselage_width``     Aircraft fuselage width, [:math:`m`]
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

    max_altitude_ft: float
    max_mach_num: float
    fuselage_width: float


def _row_to_aircraft_engine_params(tup: Any) -> tuple[str, PSAircraftEngineParams]:
    icao = tup.ICAO
    wing_aspect_ratio = tup.AR
    wing_surface_area = tup.Sref_m2
    amass_mtow = tup.MTOM_kg
    params = PSAircraftEngineParams(
        manufacturer=tup.Manufacturer,
        aircraft_type=tup.Type,
        n_engine=tup.n_engine,
        winglets=tup.winglets == "yes",
        wing_surface_area=wing_surface_area,
        wing_aspect_ratio=wing_aspect_ratio,
        wing_span=(wing_aspect_ratio * wing_surface_area) ** 0.5,
        wing_constant=tup.wing_constant,
        delta_2=tup.delta_2,
        cos_sweep=tup.cos_sweep,
        psi_0=tup.psi_0,
        x_ref=tup.Xo,
        j_1=tup.j_1,
        j_2=tup.j_2,
        j_3=70.0,  # use constant value for now, may be updated in the future
        ff_idle_sls=tup.mf_idle_SLS_kg_s,
        ff_max_sls=tup.mf_max_T_O_SLS_kg_s,
        f_00=tup.nominal_F00_ISA_kn * 1000.0,
        m_des=tup.M_des,
        c_t_des=tup.CT_des,
        eta_1=tup.eta_1,
        eta_2=tup.eta_2,
        amass_mtow=amass_mtow,
        amass_mlw=tup.MLM_kg,
        amass_mzfw=tup.MZFM_kg,
        amass_oew=tup.OEM_i_kg,
        amass_mpl=tup.MPM_i_kg,
        max_altitude_ft=tup.MaxAlt_ft,
        max_mach_num=tup.MMO,
        fuselage_width=tup.bf_m,
    )
    return icao, params


@functools.cache
def get_aircraft_engine_params(ps_file_path: pathlib.Path) -> Mapping[str, PSAircraftEngineParams]:
    """Extract aircraft-engine parameters for each aircraft type supported by the PS model."""
    dtypes = {
        "ICAO": object,
        "Manufacturer": object,
        "Type": object,
        "n_engine": int,
        "winglets": object,
        "Sref_m2": float,
        "delta_2": float,
        "cos_sweep": float,
        "AR": float,
        "psi_0": float,
        "Xo": float,
        "wing_constant": float,
        "j_2": float,
        "j_1": float,
        "mf_idle_SLS_kg_s": float,
        "mf_max_T_O_SLS_kg_s": float,
        "nominal_F00_ISA_kn": float,
        "M_des": float,
        "CT_des": float,
        "eta_1": float,
        "eta_2": float,
        "WV": object,
        "MTOM_kg": float,
        "MLM_kg": float,
        "MZFM_kg": float,
        "OEM_i_kg": float,
        "MPM_i_kg": float,
        "MZFM_MTOM": float,
        "OEM_i_MTOM": float,
        "MPM_i_MTOM": float,
        "etaL_D_do": float,
        "MaxAlt_ft": float,
        "MMO": float,
        "span_m": float,
        "bf_m": float,
    }

    df = pd.read_csv(ps_file_path, dtype=dtypes)

    return dict(_row_to_aircraft_engine_params(tup) for tup in df.itertuples(index=False))
