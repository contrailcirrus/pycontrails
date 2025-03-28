"""Simple dataclasses for working with PS aircraft performance data."""

from __future__ import annotations

import dataclasses
import functools
import pathlib
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from pycontrails.core.aircraft_performance import AircraftPerformanceParams
from pycontrails.physics import constants as c
from pycontrails.utils.types import ArrayOrFloat

#: Path to the Poll-Schumann aircraft parameters CSV file.
PS_FILE_PATH = pathlib.Path(__file__).parent / "static" / "ps-aircraft-params-20250328.csv"


@dataclasses.dataclass(frozen=True)
class PSAircraftEngineParams:
    """Store extracted aircraft and engine parameters for each aircraft type.

    -------------------------------------
    AIRCRAFT INFORMATION
    -------------------------------------
    - ``manufacturer``        Aircraft manufacturer name
    - ``aircraft_type``       Specific aircraft type variant
    - ``n_engine``            Number of engines
    - ``winglets``            Does the aircraft type contain winglets? (True/False)

    -------------------------------------
    AIRCRAFT MASS PARAMETERS
    -------------------------------------
    - ``amass_mtow``         Aircraft maximum take-off weight, [:math:`kg`]
    - ``amass_mlw``          Aircraft maximum landing weight, [:math:`kg`]
    - ``amass_mzfw``         Aircraft maximum zero fuel weight, [:math:`kg`]
    - ``amass_oew``          Aircraft operating empty weight, [:math:`kg`]
    - ``amass_mpl``          Aircraft maximum payload, [:math:`kg`]

    -------------------------------------
    AIRCRAFT GEOMETRY
    -------------------------------------
    - ``wing_surface_area``     Reference wing surface area, [:math:`m^{2}`]
    - ``wing_span``             Wing span, [:math:`m`]
    - ``fuselage_width``        Aircraft fuselage width, [:math:`m`]
    - ``delta_2``               Induced drag wing-fuselage interference factor
    - ``cos_sweep``             Cosine of wing sweep angle measured at the 1/4 chord line
    - ``wing_aspect_ratio``     Wing aspect ratio, ``wing_span**2 / wing_surface_area``

    -------------------------------------
    AERODYNAMIC PARAMETERS
    -------------------------------------
    - ``psi_0``             Aircraft geometry drag parameter
    - ``x_ref``             Threshold condition where the terminating shockwave moves onto
                            the rear part of the wing
    - ``wing_constant``     A constant used in the wave drag model, capturing the aerofoil
                            technology factor and wing geometry
    - ``j_1``               Wave drag parameter 1
    - ``j_2``               Wave drag parameter 2
    - ``j_3``               Wave drag parameter 3
    - ``c_l_do``            Design optimum lift coefficient

    -------------------------------------
    ENGINE PARAMETERS
    -------------------------------------
    - ``f_00``          Maximum thrust force that can be supplied by the engine at sea level static
                        conditions, summed over all engines, [:math:`N`].
    - ``ff_max_sls``    Fuel mass flow rate at take-off and sea level static conditions, summed
                        over all engines, [:math:`kg s^{-1}`]
    - ``ff_idle_sls``   Fuel mass flow rate under engine idle and sea level static conditions,
                        summed over all engines, [:math:`kg s^{-1}`]
    - ``m_des``         Design optimum Mach number where the fuel mass flow rate is at a minimum.
    - ``c_t_des``       Design optimum engine thrust coefficient where the fuel mass flow rate is
                        at a minimum.
    - ``eta_1``         Multiplier for maximum overall propulsion efficiency model
    - ``eta_2``         Exponent for maximum overall propulsion efficiency model
    - ``tr_ec``         Engine characteristic ratio of total turbine-entry-temperature to
                        the total freestream temperature for maximum overall efficiency.
    - ``m_ec``          Engine characteristic Mach number associated with `tr_ec`.
    - ``tet_mto``       Turbine entry temperature at maximum take-off rating, [:math:`K`]
    - ``tet_mcc``       Turbine entry temperature at maximum continuous climb rating, [:math:`K`]
    - ``nominal_opr``   Nominal engine operating pressure ratio.
    - ``nominal_bpr``   Nominal engine bypass ratio.
    - ``nominal_fpr``   Nominal engine fan pressure ratio.

    -------------------------------------
    HEIGHT AND SPEED LIMITS
    -------------------------------------
    - ``fl_max``        Maximum flight level
    - ``max_mach_num``  Maximum operational Mach number
    - ``p_i_max``       Maximum operational impact pressure, [:math:`Pa`]
    - ``p_inf_co``      Crossover pressure altitude, [:math:`Pa`]
    """

    manufacturer: str
    aircraft_type: str
    n_engine: int
    winglets: bool

    amass_mtow: float
    amass_mlw: float
    amass_mzfw: float
    amass_oew: float
    amass_mpl: float

    wing_surface_area: float
    wing_span: float
    fuselage_width: float
    delta_2: float
    cos_sweep: float
    wing_aspect_ratio: float

    psi_0: float
    x_ref: float
    wing_constant: float
    j_1: float
    j_2: float
    j_3: float
    c_l_do: float

    f_00: float
    ff_max_sls: float
    ff_idle_sls: float
    m_des: float
    c_t_des: float
    eta_1: float
    eta_2: float
    tr_ec: float
    m_ec: float
    tet_mto: float
    tet_mcc: float
    nominal_opr: float
    nominal_bpr: float
    nominal_fpr: float

    fl_max: float
    max_mach_num: float
    p_i_max: float
    p_inf_co: float


def _row_to_aircraft_engine_params(tup: Any) -> tuple[str, PSAircraftEngineParams]:
    icao = tup.ICAO
    wing_aspect_ratio = tup.AR
    amass_mtow = tup.MTOM_kg
    tet_mto = turbine_entry_temperature_at_max_take_off(tup.Year_of_first_flight)
    p_i_max = impact_pressure_max_operating_limits(tup.MMO)
    params = PSAircraftEngineParams(
        manufacturer=tup.Manufacturer,
        aircraft_type=tup.Type,
        n_engine=tup.n_engine,
        winglets=tup.winglets == "yes",
        amass_mtow=amass_mtow,
        amass_mlw=tup.MLM_kg,
        amass_mzfw=tup.MZFM_kg,
        amass_oew=tup.OEM_i_kg,
        amass_mpl=tup.MPM_i_kg,
        wing_surface_area=tup.Sref_m2,
        wing_span=tup.span_m,
        fuselage_width=tup.bf_m,
        delta_2=tup.delta_2,
        cos_sweep=tup.cos_sweep,
        wing_aspect_ratio=wing_aspect_ratio,
        psi_0=tup.psi_0,
        x_ref=tup.Xo,
        wing_constant=tup.wing_constant,
        j_1=tup.j_1,
        j_2=tup.j_2,
        j_3=70.0,  # use constant value for now, may be updated in the future
        c_l_do=tup.CL_do,
        f_00=tup.nominal_F00_ISA_kn * 1000.0,
        ff_max_sls=tup.mf_max_T_O_SLS_kg_s,
        ff_idle_sls=tup.mf_idle_SLS_kg_s,
        m_des=tup.M_des,
        c_t_des=tup.CT_des,
        eta_1=tup.eta_1,
        eta_2=tup.eta_2,
        tr_ec=tup.Tec,
        m_ec=tup.Mec,
        tet_mto=tet_mto,
        tet_mcc=turbine_entry_temperature_at_max_continuous_climb(tet_mto),
        nominal_opr=tup.nominal_opr,
        nominal_bpr=tup.nominal_bpr,
        nominal_fpr=tup.nominal_fpr,
        fl_max=tup.FL_max,
        max_mach_num=tup.MMO,
        p_i_max=p_i_max,
        p_inf_co=crossover_pressure_altitude(tup.MMO, p_i_max),
    )
    return icao, params


@functools.cache
def load_aircraft_engine_params(
    engine_deterioration_factor: float = AircraftPerformanceParams.engine_deterioration_factor,
) -> Mapping[str, PSAircraftEngineParams]:
    """
    Extract aircraft-engine parameters for each aircraft type supported by the PS model.

    Parameters
    ----------
    engine_deterioration_factor: float
        Account for "in-service" engine deterioration between maintenance cycles.
        Default value reduces `eta_1` by 2.5%, which increases the fuel flow estimates by 2.5%.

    Returns
    -------
    Mapping[str, PSAircraftEngineParams]
        Aircraft-engine parameters for each aircraft type supported by the PS model.
    """
    dtypes = {
        "ICAO": object,
        "Manufacturer": object,
        "Type": object,
        "Year_of_first_flight": float,
        "n_engine": int,
        "winglets": object,
        "WV": object,
        "MTOM_kg": float,
        "MLM_kg": float,
        "MZFM_kg": float,
        "OEM_i_kg": float,
        "MPM_i_kg": float,
        "MZFM_MTOM": float,
        "OEM_i_MTOM": float,
        "MPM_i_MTOM": float,
        "Sref_m2": float,
        "span_m": float,
        "bf_m": float,
        "delta_2": float,
        "cos_sweep": float,
        "AR": float,
        "psi_0": float,
        "Xo": float,
        "wing_constant": float,
        "j_2": float,
        "j_1": float,
        "CL_do": float,
        "nominal_F00_ISA_kn": float,
        "mf_max_T_O_SLS_kg_s": float,
        "mf_idle_SLS_kg_s": float,
        "M_des": float,
        "CT_des": float,
        "eta_1": float,
        "eta_2": float,
        "Mec": float,
        "Tec": float,
        "FL_max": float,
        "MMO": float,
        "nominal_opr": float,
        "nominal_bpr": float,
        "nominal_fpr": float,
    }

    df = pd.read_csv(PS_FILE_PATH, dtype=dtypes)
    df["eta_1"] *= 1.0 - engine_deterioration_factor

    return dict(_row_to_aircraft_engine_params(tup) for tup in df.itertuples(index=False))


def turbine_entry_temperature_at_max_take_off(first_flight: ArrayOrFloat) -> ArrayOrFloat:
    """
    Calculate turbine entry temperature at maximum take-off rating.

    Parameters
    ----------
    first_flight: ArrayOrFloat
        Year of first flight

    Returns
    -------
    ArrayOrFloat
        Turbine entry temperature at maximum take-off rating, ``tet_mto``, [:math:`K`]

    Notes
    -----
    The turbine entry temperature at max take-off is approximated based on the year of first flight
    for the specific aircraft type. This approximation captures the historical trends of
    improvements in turbine cooling technology level. The uncertainty of this estimate is ±75 K.

    References
    ----------
    - :cite:`cumpstyJetPropulsion2015`
    """
    out = 2000.0 * (1.0 - np.exp(62.8 - 0.0325 * first_flight))
    if isinstance(first_flight, np.ndarray):
        return out
    return out.item()


def turbine_entry_temperature_at_max_continuous_climb(tet_mto: float) -> float:
    """
    Calculate turbine entry temperature at maximum continuous climb rating.

    Parameters
    ----------
    tet_mto: float
        Turbine entry temperature at maximum take-off rating, `tet_mto`, [:math:`K`]

    Returns
    -------
    float
        Turbine entry temperature at maximum continuous climb rating, `tet_mcc`, [:math:`K`]
    """
    return 0.92 * tet_mto


def impact_pressure_max_operating_limits(max_mach_num: float) -> float:
    """
    Calculate maximum permitted operational impact pressure.

    Parameters
    ----------
    max_mach_num: float
        Maximum permitted operational Mach number for aircraft type.

    Returns
    -------
    float
        Maximum permitted operational impact pressure for aircraft type, ``p_i_max``, [:math:`Pa`]

    Notes
    -----
    The impact pressure is the difference between the free stream total pressure ``p_0`` and the
    atmospheric static pressure ``p_inf``. By definition, the calibrated airspeed, ``v_cas``, is
    the speed at sea level in the ISA that has the same impact pressure.
    """
    v_cas_mo_over_c_msl = max_calibrated_airspeed_over_speed_of_sound(max_mach_num)
    return c.p_surface * (
        (1.0 + 0.5 * (c.kappa - 1.0) * v_cas_mo_over_c_msl**2) ** (c.kappa / (c.kappa - 1.0)) - 1.0
    )


def max_calibrated_airspeed_over_speed_of_sound(max_mach_num: float) -> float:
    """
    Calculate max calibrated airspeed over the speed of sound at ISA mean sea level.

    Parameters
    ----------
    max_mach_num: float
        Maximum permitted operational Mach number for aircraft type.

    Returns
    -------
    float
        Ratio of maximum operating limits of the calibrated airspeed (CAS) over the speed of sound
        at mean sea level (MSL) in standard atmosphere (ISA), ``v_cas_mo_over_c_msl``
    """
    return 0.57 * (max_mach_num + 0.10)


def crossover_pressure_altitude(max_mach_num: float, p_i_max: float) -> float:
    """
    Calculate crossover pressure altitude.

    Parameters
    ----------
    max_mach_num: float
        Maximum permitted operational Mach number for aircraft type.
    p_i_max : float
        Maximum permitted operational impact pressure for aircraft type, [:math:`Pa`]

    Returns
    -------
    float
        Crossover pressure altitude, ``p_inf_co``, [:math:`Pa`]

    Notes
    -----
    At low altitudes, the calibrated airspeed (CAS) is used to determine the maximum aircraft
    operational speed. However, the ambient temperature drops as altitude increases, which causes
    the Mach number to increase even when the CAS remains constant. This can cause the Mach number
    to exceed the maximum permitted operational Mach number. Therefore, above the crossover
    altitude, the maximum operational speed is determined by the Mach number instead of CAS.
    """
    return p_i_max / (
        0.5 * c.kappa * max_mach_num**2 * (1.0 + (max_mach_num**2 / 4.0) + (max_mach_num**4 / 40.0))
    )
