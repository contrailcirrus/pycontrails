"""Airport data support."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pycontrails.core import cache
from pycontrails.physics import geo, units

#: URL for `Our Airports <https://ourairports.com/>`_ database.
#: Fork of the `ourairports-data repository <https://github.com/davidmegginson/ourairports-data>`_.
OURAIRPORTS_DATABASE_URL: str = (
    "https://github.com/contrailcirrus/ourairports-data/raw/main/airports.csv"
)


def _download_ourairports_csv() -> pd.DataFrame:
    """Download CSV file from fork of ourairports-data github."""
    return pd.read_csv(
        OURAIRPORTS_DATABASE_URL,
        usecols=[
            "type",
            "name",
            "latitude_deg",
            "longitude_deg",
            "elevation_ft",
            "iso_country",
            "iso_region",
            "municipality",
            "scheduled_service",
            "gps_code",
            "iata_code",
        ],
    )


def global_airport_database(
    cachestore: cache.CacheStore | None = None, update_cache: bool = False
) -> pd.DataFrame:
    """
    Load and process global airport database from `Our Airports <https://ourairports.com/>`_.

    The database includes coordinates and metadata for 74867 unique airports.

    Parameters
    ----------
    cachestore : cache.CacheStore | None, optional
        Cache store for airport database.
        Defaults to :class:`cache.DiskCacheStore`.
    update_cache : bool, optional
        Force update to cached airports database.

    Returns
    -------
    pd.DataFrame
        Processed global airport database.

        Global airport database.

    Notes
    -----
    As of 2023 March 30, the global airport database contains:

    .. csv-table::
       :header: "Airport Type", "Number"
       :widths: 70, 30

        "small_airport",    39327
        "heliport",         19039
        "closed",           10107
        "medium_airport",   4753
        "seaplane_base",    1133
        "large_airport",    463
        "balloonport",      45

    References
    ----------
    - :cite:`megginsonOpendataDownloadsOurAirports2023`
    """
    cachestore = cachestore or cache.DiskCacheStore()

    cache_key = "ourairports-data_airports.csv"
    if cachestore.exists(cache_key) and not update_cache:
        airports = pd.read_csv(cachestore.path(cache_key))
    else:
        airports = _download_ourairports_csv()
        airports.to_csv(cachestore.path(cache_key), index=False)

    #: Format dataset by renaming columns & filling nan values
    airports.rename(
        columns={"latitude_deg": "latitude", "longitude_deg": "longitude", "gps_code": "icao_code"},
        inplace=True,
    )
    airports["elevation_ft"].fillna(0, inplace=True)

    # Keep specific airport types used by commercial aviation
    select_airport_types = airports["type"].isin(
        ["large_airport", "medium_airport", "small_airport", "heliport"]
    )

    # Keep airports with valid ICAO codes
    select_icao_codes = (airports["icao_code"].str.len() == 4) & (
        airports["icao_code"].str.isalpha()
    )

    # filter airports
    airports = airports.loc[select_airport_types & select_icao_codes]

    # Format dataset
    airports["elevation_m"] = units.ft_to_m(airports["elevation_ft"].to_numpy())
    airports.sort_values(by=["icao_code"], ascending=True, inplace=True)

    return airports.reset_index(drop=True)


def find_nearest_airport(
    airports: pd.DataFrame,
    longitude: float,
    latitude: float,
    altitude: float,
    *,
    bbox: float = 2.0,
) -> str | None:
    r"""
    Find airport nearest to the waypoints.

    Parameters
    ----------
    airports: pd.DataFrame
        Airport database in the format returned from :func:`global_airport_database`.
    longitude: float
        Waypoint longitude, [:math:`\deg`]
    latitude: float
        Waypoint latitude, [:math:`\deg`]
    altitude: float
        Waypoint altitude, [:math:`m`]
    bbox: float
        Search airports within spatial bounding box of ± `bbox` from the waypoint, [:math:`\deg`]
        Defaults to :math:`2\deg`

    Returns
    -------
    str
        ICAO code of nearest airport.
        Returns None if no airport is found within ``bbox``.

    Notes
    -----
    Function will first search for large airports around the waypoint vicinity.
    If none is found, it will search for medium and small airports
    around the waypoint vicinity.

    The waypoint must be below 10,000 feet to increase the
    probability of identifying the correct airport.
    """
    if altitude > 3000:
        raise ValueError(
            f"Altitude ({altitude} m) is too high (> 3000 m) to identify nearest airport."
        )

    is_near_waypoint = airports["longitude"].between(
        (longitude - bbox), (longitude + bbox)
    ) & airports["latitude"].between((latitude - bbox), (latitude + bbox))

    # Find the nearest airport from largest to smallest airport type
    search_priority = ["large_airport", "medium_airport", "small_airport"]

    for airport_type in search_priority:
        is_airport_type = airports["type"] == airport_type
        nearest_airports = airports.loc[is_near_waypoint & is_airport_type]

        if len(nearest_airports) == 1:
            return nearest_airports["icao_code"].values[0]

        elif len(nearest_airports) > 1:
            distance = distance_to_airports(
                nearest_airports,
                longitude,
                latitude,
                altitude,
            )
            i_nearest = np.argmin(distance)
            return nearest_airports["icao_code"].values[i_nearest]

        else:
            continue

    return None


def distance_to_airports(
    airports: pd.DataFrame,
    longitude: float,
    latitude: float,
    altitude: float,
) -> np.ndarray:
    r"""
    Calculate the 3D distance from the waypoint to the provided airports.

    Parameters
    ----------
    airports : pd.DataFrame
        Airport database in the format returned from :func:`global_airport_database`.
    longitude : float
        Waypoint longitude, [:math:`\deg`]
    latitude : float
        Waypoint latitude, [:math:`\deg`]
    altitude : float
        Waypoint altitude, [:math:`m`]

    Returns
    -------
    np.ndarray
        3D distance from waypoint to airports, [:math:`m`]

    See Also
    --------
    :func:`geo.haversine`
    """
    dist_horizontal = geo.haversine(
        np.full(airports["longitude"].shape, longitude),
        np.full(airports["latitude"].shape, latitude),
        airports["longitude"].to_numpy(),
        airports["latitude"].to_numpy(),
    )
    dist_vertical = altitude - airports["elevation_m"].to_numpy()
    return (dist_horizontal**2 + dist_vertical**2) ** 0.5
