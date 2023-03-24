"""Cocip flight summary statistics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pycontrails.core.flight import Flight
from pycontrails.physics import units


def _get_total_fuel_consumption(fuel_flow: np.ndarray, time: np.ndarray) -> float:
    """
    Calculate the total fuel consumption of the flight.

    Parameters
    ----------
    fuel_flow : np.ndarray
        fuel mass flow rate [:math:`kg s^{-1}`]
    time : np.ndarray
        timestamp for each waypoint in np.datetime64 for each element.

    Returns
    -------
    float
        Total fuel consumption for the flight [:math:`kg`]
    """
    dt_sec = np.diff(time, append=np.datetime64("NaT")) / np.timedelta64(1, "s")
    return np.nansum(fuel_flow * dt_sec)


def _check_for_initial_contrails(sac: np.ndarray) -> bool:
    """Check if any flight waypoints satisfy the Schmidt-Appleman Criterion.

    Parameters
    ----------
    sac: np.ndarray
        Schmidt-Appleman Criterion (1: satisfies sac, 0: does not satisfy sac,
        nan: outside met domain)

    Returns
    -------
    bool
        Some or all waypoints in this flight forms contrails.
    """
    return (sac == 1).any()


def _get_mean_spatiotemporal_value(contrail: pd.DataFrame, var_name: str) -> np.float64:
    """
    Calculate the mean value of the selected contrail property.

    First, the mean value for each contrail segment is calculated over time, followed by
    taking the mean of means for each waypoint.

    Parameters
    ----------
    contrail: pd.DataFrame
        Contrail properties for each segment and its evolution
    var_name: str
        Variable name of the contrail property in contrail

    Returns
    -------
     np.float64
        Mean value of the selected contrail property
    """
    try:
        df_var = contrail[["waypoint", var_name]]
    except KeyError:
        raise KeyError(f"Variable name {var_name} is not found in the DataFrame.")
    else:
        df_var.set_index("waypoint", inplace=True, drop=True)
        df_wypt_var = df_var.groupby(df_var.index).mean()
        return np.nanmean(df_wypt_var[var_name])


def _get_mean_contrail_age(contrail: pd.DataFrame) -> np.float64:
    """
    Calculate the mean contrail age for the flight.

    This is calculated by taking the mean of the maximum contrail age for each waypoint.

    Parameters
    ----------
    contrail: pd.DataFrame
        Contrail properties for each segment and its evolution

    Returns
    -------
     np.float64
        Mean contrail age for the flight [:math:`hours`]
    """
    df_age = contrail[["waypoint", "age_hours"]]
    df_age.set_index("waypoint", inplace=True, drop=True)
    df_wypt_age_max = df_age.groupby(df_age.index).max()
    return np.nanmean(df_wypt_age_max["age_hours"])


def flight_statistics(flight: Flight, contrail: pd.DataFrame | None) -> pd.Series:
    """
    Calculate/aggregate the flight, emissions and contrail statistics for one flight.

    Parameters
    ----------
    flight : Flight
        Flight object from pycontrails.core.flight
    contrail : pd.DataFrame | None
        Contrail properties for each segment and its evolution

    Returns
    -------
    pd.Series
        Flight statistics as :class:`pandas.Series`
    """

    has_short_lived_contrails = _check_for_initial_contrails(flight["sac"])
    has_persistent_contrails = contrail is not None

    flight_stats = {
        "Flight ID": flight.attrs["flight_id"],
        "ATYP": flight.attrs["aircraft_type"],
        "First wypt": flight["time"].min(),
        "Last wypt": flight["time"].max(),
        # Fuel burn and emissions
        "Total flight dist (km)": np.nansum(flight["segment_length"]) / 1000,
        "Total fuel burn (kg)": _get_total_fuel_consumption(flight["fuel_flow"], flight["time"]),
        "OPE mean": np.nanmean(flight["engine_efficiency"]),
        "Fuel methodology": flight.attrs.get("bada_model", None),
        "ATYP assumed": flight.attrs.get("aircraft_type_bada", None),
        "Engine name": flight.attrs.get("engine_name", None),
        "Aircraft mass mean (kg)": np.nanmean(flight["aircraft_mass"]),
        # 'Total NOx emissions (kg)': self.tot_nox_emissions_kg,
        # 'NOx methodology': self.nox_methodology,
        "BC EI_n mean (kg-1)": np.nanmean(flight.get_data_or_attr("nvpm_ei_n")),
        "BC methodology": (
            flight.attrs["bc_data_source"] if "bc_data_source" in flight.attrs else None
        ),
        # Initial contrail length
        "Short-lived contrails": has_short_lived_contrails,
        "Initial contrail length (km)": (
            np.nansum(flight["segment_length"][flight["sac"] == 1]) / 1000
            if has_short_lived_contrails
            else 0
        ),
        # Persistent contrail properties
        "Persistent contrails": has_persistent_contrails,
        "First contrail wypt": contrail["time"].min() if has_persistent_contrails else "NaT",  # type: ignore  # noqa: E501
        "Last contrail wypt": contrail["time"].max() if has_persistent_contrails else "NaT",  # type: ignore  # noqa: E501
        "RHi initial, Mean": np.nanmean(flight["rhi_1"]) if has_persistent_contrails else 0,
        "RHi initial, Stdev": np.nanstd(flight["rhi_1"]) if has_persistent_contrails else 0,
        "RHi lifetime, Mean": (
            _get_mean_spatiotemporal_value(contrail, "rhi") if has_persistent_contrails else 0
        ),
        "Temp initial, Mean (K)": (
            np.nanmean(flight["air_temperature_1"]) if has_persistent_contrails else 0
        ),
        "Temp initial, Stdev (K)": (
            np.nanstd(flight["air_temperature_1"]) if has_persistent_contrails else 0
        ),
        "Temp SAC, Mean (K)": (
            np.nanmean(flight["T_critical_sac"]) if has_persistent_contrails else 0
        ),  # TODO: bugs to be resolved?
        "Temp SAC, Stdev (K)": (
            np.nanstd(flight["T_critical_sac"]) if has_persistent_contrails else 0
        ),  # TODO: bugs to be resolved?
        "Persistent contrail length (km)": (
            np.nansum(flight["segment_length"][flight["persistent_1"] == 1]) / 1000
            if has_persistent_contrails
            else 0
        ),
        "Contrail altitude initial, Mean (ft)": (
            units.m_to_ft(np.nanmean(flight["altitude_1"])) if has_persistent_contrails else 0
        ),
        "Contrail altitude lifetime, Mean (ft)": (
            units.m_to_ft(_get_mean_spatiotemporal_value(contrail, "altitude"))
            if has_persistent_contrails
            else 0
        ),
        "Contrail age, Mean (h)": (
            _get_mean_contrail_age(contrail) if has_persistent_contrails else 0
        ),
        "Contrail age, Max (h)": (
            np.nanmax(contrail["age_hours"]) if has_persistent_contrails else 0  # type: ignore
        ),
        "Ice number initial, Mean (m-1)": (
            np.nanmean(flight["n_ice_per_m_1"]) if has_persistent_contrails else 0
        ),
        "Ice number initial, Stdev (m-1)": (
            np.nanstd(flight["n_ice_per_m_1"]) if has_persistent_contrails else 0
        ),
        "Ice number lifetime, Mean (m-1)": (
            _get_mean_spatiotemporal_value(contrail, "n_ice_per_m")
            if has_persistent_contrails
            else 0
        ),
        "Ice vol mean radius, Mean (um)": (
            _get_mean_spatiotemporal_value(contrail, "r_ice_vol") * 1e6
            if has_persistent_contrails
            else 0
        ),
        "Tau contrail, Mean": (
            _get_mean_spatiotemporal_value(contrail, "tau_contrail")
            if has_persistent_contrails
            else 0
        ),
        "Tau cirrus, Mean": (
            _get_mean_spatiotemporal_value(contrail, "tau_cirrus")
            if has_persistent_contrails
            else 0
        ),
        # Radiative properties
        "RF SW (W m-2)": (
            _get_mean_spatiotemporal_value(contrail, "rf_sw") if has_persistent_contrails else 0
        ),
        "RF LW (W m-2)": (
            _get_mean_spatiotemporal_value(contrail, "rf_lw") if has_persistent_contrails else 0
        ),
        "RF Net (W m-2)": (
            _get_mean_spatiotemporal_value(contrail, "rf_net") if has_persistent_contrails else 0
        ),
        "Total contrail EF (J)": np.nansum(contrail["ef"]) if has_persistent_contrails else 0,  # type: ignore  # noqa: E501
        "SDR mean (W m-2)": (
            _get_mean_spatiotemporal_value(contrail, "sdr") if has_persistent_contrails else 0
        ),
        "RSR mean (W m-2)": (
            _get_mean_spatiotemporal_value(contrail, "rsr") if has_persistent_contrails else 0
        ),
        "OLR mean (W m-2)": (
            _get_mean_spatiotemporal_value(contrail, "olr") if has_persistent_contrails else 0
        ),
    }

    return pd.Series(flight_stats)
