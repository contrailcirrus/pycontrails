from __future__ import annotations

import dataclasses
import pandas as pd
import numpy as np
import pycontrails.physics.jet as jet

# TODO: Interpolate between waypoints with dt < threshold
# TODO: Find nearest airport


def downsample_waypoints(
        df_flight_waypoints: pd.DataFrame, *, time_resolution: int = 10, time_var: str = "timestamp"
) -> pd.DataFrame:
    """
    Downsample flight waypoints to a specified time resolution

    Parameters
    ----------
    df_flight_waypoints: pd.DataFrame
        Raw flight waypoints and metadata
    time_resolution: int
        Downsampled time resolution, [:math:`s`]
    time_var: str
        Time variable in the "df_flight_waypoints" DataFrame

    Returns
    -------
    pd.DataFrame
        Downsampled flight waypoints
    """
    df_flight_waypoints.index = df_flight_waypoints[time_var]
    df_resampled = df_flight_waypoints.resample(f"{time_resolution}s").first()
    df_resampled = df_resampled[df_resampled["longitude"].notna()].copy()
    df_resampled.reset_index(inplace=True, drop=True)
    return df_resampled


def separate_unique_flights_from_waypoints(df_waypoints: pd.DataFrame, *, columns: list) -> list[pd.DataFrame]:
    """
    Separate unique flights from the waypoints, so each subset has the same metadata.

    Parameters
    ----------
    df_waypoints: pd.DataFrame
        Unsorted waypoints of multiple flights
    columns: str
        Metadata variables to check, by order of priority

    Returns
    -------
    list[pd.DataFrame]
        List of DataFrames containing set of flight waypoints with the same metadata.
    """
    if np.any(~pd.Series(columns).isin(df_waypoints.columns)):
        raise KeyError(f"DataFrame does not contain all of the required columns listed in the inputs.")

    flights = list(df_waypoints.groupby(columns))
    return [flight[1] for flight in flights]


def remove_noise_in_cruise_altitude(
        altitude_ft: np.ndarray, *,
        noise_threshold_ft: float = 25,
        threshold_altitude_ft: float = 10000
) -> np.ndarray:
    """
    Remove noise in cruise altitude by rounding up/down to the nearest flight level.

    Parameters
    ----------
    altitude_ft: np.ndarray
        Barometric altitude, [:math:`ft`]
    noise_threshold_ft: float
        Altitude difference threshold to identify noise, [:math:`ft`]
        Barometric altitude from ADS-B telemetry is reported at increments of 25 feet.
    threshold_altitude_ft: float
        Altitude will be checked and corrected above this threshold, [:math:`ft`]
        Currently set to 10,000 feet.

    Returns
    -------
    np.ndarray
        Barometric altitude with noise removed, [:math:`ft`]
    """
    d_alt_ft = np.diff(altitude_ft, prepend=np.nan)
    is_noise = (np.abs(d_alt_ft) <= noise_threshold_ft) & (altitude_ft > threshold_altitude_ft)

    # Round to the nearest flight level
    altitude_rounded = np.round(altitude_ft / 1000) * 1000
    altitude_ft[is_noise] = altitude_rounded[is_noise]
    return altitude_ft


def identify_phase_of_flight_detailed(
        altitude_ft: np.ndarray,
        dt: np.ndarray, *,
        threshold_rocd: float = 250.0,
        min_cruise_alt_ft: float = 20000
) -> FlightPhaseDetailed:
    """ Identify the phase of flight (climb, cruise, descent, level flight) for each waypoint.

    Parameters
    ----------
    altitude_ft: np.ndarray
        Altitude of each waypoint, [:math:`ft`]
    dt: np.ndarray
        Time difference between waypoints, [:math:`s`].
    threshold_rocd: float
        ROCD threshold to identify climb and descent, [:math:`ft min^{-1}`].
        Currently set to 250 ft/min.
    min_cruise_alt_ft: float
        Minimum threshold altitude for cruise, [:math:`ft`]
        This is specific for each aircraft type, and can be approximated as 50% of the altitude ceiling.

    Returns
    -------
    FlightPhaseDetailed
        Booleans marking if the waypoints are at cruise, climb, descent, or level flight

    Notes
    -----
    This function is more detailed when compared to `jet.identify_phase_of_flight`:
    - There is an additional flight phase "level-flight", where the aircraft is holding at lower altitudes, and
    - The cruise phase of flight only occurs above a certain threshold altitude.
    """
    rocd = jet.rate_of_climb_descent(dt, altitude_ft)

    nan = np.isnan(rocd)
    cruise = (rocd < threshold_rocd) & (rocd > -threshold_rocd) & (altitude_ft > min_cruise_alt_ft)
    climb = ~cruise & (rocd > 0)
    descent = ~cruise & (rocd < 0)
    level_flight = ~(nan | cruise | climb | descent)
    return FlightPhaseDetailed(cruise=cruise, climb=climb, descent=descent, level_flight=level_flight, nan=nan)


@dataclasses.dataclass
class FlightPhaseDetailed:
    """Container for boolean arrays describing detailed phase of the flight.
    """
    cruise: np.ndarray
    climb: np.ndarray
    descent: np.ndarray
    level_flight: np.ndarray
    nan: np.ndarray
