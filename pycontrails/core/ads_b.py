from __future__ import annotations

import pandas as pd
import numpy as np

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
