"""Global airport database.

This module includes the function to read and process the global airport database,
which includes the coordinates and metadata for 74867 unique airports.

Sources:
- https://ourairports.com/data/
- https://github.com/davidmegginson/ourairports-data

As of 30-March-2023, the global airport database contains:
    - small_airport     39327
    - heliport          19039
    - closed            10107
    - medium_airport     4753
    - seaplane_base      1133
    - large_airport       463
    - balloonport          45
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from pycontrails.physics.units import ft_to_m
from pycontrails.physics.geo import haversine

#: Default path
airport_database_path: str = "https://github.com/contrailcirrus/ourairports-data/raw/main/airports.csv"


def global_airport_database(*, return_as_dataframe: bool = True) -> pd.DataFrame() | dict:
    """
    Load and process global airport database

    Parameters
    ----------
    return_as_dataframe: bool
        Return database as DataFrame, or dictionary if set to false

    Returns
    -------
    pd.DataFrame() | dict
        Processed global airport database
    """
    df_airports = pd.read_csv(
        airport_database_path,
        usecols=[
            "type", "name", "latitude_deg", "longitude_deg", "elevation_ft", "iso_country",
            "iso_region", "municipality", "scheduled_service", "gps_code", "iata_code"
        ]
    )

    #: Format dataset by renaming columns & filling nan values
    df_airports.rename(
        columns={"latitude_deg": "latitude", "longitude_deg": "longitude", "gps_code": "icao_code"},
        inplace=True
    )
    df_airports["elevation_ft"].fillna(0, inplace=True)

    # Keep specific airport types used by commercial aviation
    is_subset = df_airports["type"].isin(
        ["large_airport", "medium_airport", "small_airport", "heliport"]
    )
    df_airports = df_airports[is_subset].copy()

    # Keep airports with valid ICAO codes
    is_subset = (df_airports["icao_code"].str.len() == 4) & (df_airports["icao_code"].str.isalpha())
    df_airports = df_airports[is_subset].copy()

    # Format dataset
    df_airports["elevation_m"] = ft_to_m(df_airports["elevation_ft"].values)
    df_airports.sort_values(by=["icao_code"], ascending=True, inplace=True)
    df_airports.reset_index(inplace=True, drop=True)

    if return_as_dataframe:
        return df_airports

    return df_airports.to_dict()


df_airports = global_airport_database(return_as_dataframe=True)


def find_nearest_airport(longitude: float, latitude: float, altitude: float, *, bbox_size: float = 2) -> str:
    """
    Find airport nearest to the provided waypoint.

    Parameters
    ----------
    longitude: float
        Waypoint longitude, [:math:`\deg`]
    latitude: float
        Waypoint latitude, [:math:`\deg`]
    altitude: float
        Waypoint altitude, [:math:`m`]
    bbox_size: float
        Search airports within spatial bounding box of ± `bbox_size` from the waypoint, [:math:`\deg`]

    Returns
    -------
    str
        ICAO code of nearest airport

    Notes
    -----
    Function will first search for large airports around the waypoint vicinity. If none is found, it will
    then search for medium and small airports around the waypoint vicinity. The waypoint must be below
    10,000 feet is constrained to increase the probability of identifying the correct airport.
    """
    if altitude > 3000:
        raise ValueError(
            f"Altitude ({altitude} m) is too high (> 3000 m) to identify nearest airport."
        )

    is_near_waypoint = (
        df_airports["longitude"].between((longitude - bbox_size), (longitude + bbox_size)) &
        df_airports["latitude"].between((latitude - bbox_size), (latitude + bbox_size))
    )

    #: Find the nearest airport
    search_priority = ["large_airport", "medium_airport", "small_airport"]

    for airport_type in search_priority:
        is_airport_type = df_airports["type"] == airport_type
        df_nearest_airports = df_airports[is_near_waypoint & is_airport_type].copy()

        if len(df_nearest_airports) == 1:
            return df_nearest_airports["icao_code"].values[0]

        elif len(df_nearest_airports) > 1:
            dist_wypt_to_airports = distance_to_airports(
                longitude, df_nearest_airports["longitude"].values,
                latitude, df_nearest_airports["latitude"].values,
                altitude, df_nearest_airports["elevation_m"].values
            )
            i_nearest = np.argmin(dist_wypt_to_airports)
            return df_nearest_airports["icao_code"].values[i_nearest]

        else:
            continue

    return "N/A"


def distance_to_airports(
        longitude: float,
        longitude_airports: np.ndarray,
        latitude: float,
        latitude_airports: np.ndarray,
        altitude: float,
        elevation_airports: np.ndarray,
) -> np.ndarray:
    """
    Calculate the 3D distance from the waypoint to the provided airports.

    Parameters
    ----------
    longitude: float
        Waypoint longitude, [:math:`\deg`]
    longitude_airports: np.ndarray
        Longitude of airports, [:math:`\deg`]
    latitude: float
        Waypoint latitude, [:math:`\deg`]
    latitude_airports: np.ndarray
        Latitude of airports, [:math:`\deg`]
    altitude: float
        Waypoint altitude, [:math:`m`]
    elevation_airports: np.ndarray
        Elevation of airports, [:math:`m`]

    Returns
    -------
    np.ndarray
        3D distance from waypoint to airports, [:math:`m`]

    See Also
    --------
    :func:`geo.haversine`
    """
    dist_horizontal = haversine(
        np.array([longitude]), np.array([latitude]), longitude_airports, latitude_airports
    )
    dist_vertical = np.abs(altitude, elevation_airports)
    return (dist_horizontal**2 + dist_vertical**2) ** 0.5
