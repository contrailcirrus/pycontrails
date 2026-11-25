"""Tools for searching for low Earth orbit satellite imagery."""

from typing import Any

import numpy as np
import pandas as pd

from pycontrails.core import Flight, GeoVectorDataset
from pycontrails.utils import dependencies

try:
    import requests
except ModuleNotFoundError as exc:
    dependencies.raise_module_not_found_error(
        name="datalib.leo_utils module",
        package_name="requests",
        module_not_found_error=exc,
        pycontrails_optional_package="sat",
    )

try:
    import shapely
except ModuleNotFoundError as exc:
    dependencies.raise_module_not_found_error(
        name="datalib.leo_utils module",
        package_name="shapely",
        module_not_found_error=exc,
        pycontrails_optional_package="sat",
    )


#: Number of items requested per page from a STAC server. The USGS Landsat
#: STAC server returns HTTP 502 for pages of 500 items.
STAC_PAGE_LIMIT = 100


def _stac_time(t: np.datetime64 | pd.Timestamp) -> str:
    """Format a UTC time for the STAC ``datetime`` parameter."""
    return pd.Timestamp(t).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def stac_search(
    url: str,
    collection: str,
    start_time: np.datetime64 | pd.Timestamp,
    end_time: np.datetime64 | pd.Timestamp,
    bbox: tuple[float, float, float, float] | None = None,
) -> list[dict[str, Any]]:
    """Return the items of a STAC collection with a nominal time in a time range.

    Parameters
    ----------
    url : str
        Root URL of the STAC API server.
    collection : str
        STAC collection ID.
    start_time, end_time : np.datetime64 | pd.Timestamp
        Time range, inclusive, in UTC.
    bbox : tuple[float, float, float, float] | None, optional
        Bounding box ``(west, south, east, north)`` in degrees that item footprints
        must intersect. If None, no spatial filter is applied.

    Returns
    -------
    list[dict[str, Any]]
        STAC items as GeoJSON features.
    """
    query: dict[str, Any] = {
        "collections": collection,
        "datetime": f"{_stac_time(start_time)}/{_stac_time(end_time)}",
        "limit": STAC_PAGE_LIMIT,
    }
    if bbox is not None:
        query["bbox"] = ",".join(str(v) for v in bbox)

    items: list[dict[str, Any]] = []
    next_url = f"{url}/search"
    params: dict[str, Any] | None = query
    while True:
        response = requests.get(next_url, params=params, timeout=60)
        response.raise_for_status()
        page = response.json()
        items.extend(page["features"])

        next_link = next((link for link in page["links"] if link["rel"] == "next"), None)
        if next_link is None:
            return items

        next_url = next_link["href"]
        params = None  # the next link includes the query parameters


def _bbox(
    west: float,
    south: float,
    east: float,
    north: float,
) -> tuple[float, float, float, float]:
    """Return a bounding box for a STAC search.

    A bounding box spanning 180 degrees of longitude or more is widened to all
    longitudes, since the minimum and maximum longitude of a trajectory crossing the
    antimeridian would otherwise bound the wrong side of the globe.

    A bounding box with zero width or height, such as that of a point, is slightly widened
    to prevent STAC servers from rejecting it.
    """
    # Possibly an antimeridian crossing
    # This check could be widened if we're not catching enough
    if east - west >= 180.0:
        west = -180.0
        east = 180.0

    pad = 1e-4
    if east - west < pad:
        west -= pad
        east += pad
    if north - south < pad:
        south -= pad
        north += pad

    return max(west, -180.0), max(south, -90.0), min(east, 180.0), min(north, 90.0)


def geometry_bbox(geometry: shapely.Geometry) -> tuple[float, float, float, float]:
    """Return the bounding box of a geometry for a STAC search."""
    return _bbox(*geometry.bounds)


def vector_bbox(vector: GeoVectorDataset) -> tuple[float, float, float, float]:
    """Return the bounding box of the vector for a STAC search."""
    lon = vector["longitude"]
    lat = vector["latitude"]
    return _bbox(lon.min().item(), lat.min().item(), lon.max().item(), lat.max().item())


def intersect_scenes(scenes: pd.DataFrame, flight: Flight) -> pd.DataFrame:
    """Return the scenes whose footprint contains the flight's position at the scene's sensing time.

    The flight is linearly interpolated to each scene's ``sensing_time``, without any
    parallax correction.

    Parameters
    ----------
    scenes : pd.DataFrame
        Scenes with a ``sensing_time`` column and a ``geometry`` column of shapely
        footprints in longitude and latitude.
    flight : Flight
        Flight to intersect.

    Returns
    -------
    pd.DataFrame
        The intersecting scenes with ``longitude`` and ``latitude`` columns holding the
        flight's interpolated position at the scene's ``sensing_time``.
    """
    # Convert datetime to numeric for np.interp
    time = flight["time"].astype("int64")
    sensing_time = scenes["sensing_time"].to_numpy("datetime64[ns]").astype("int64")

    # Unwrap longitude so a flight crossing the antimeridian interpolates correctly
    lon = np.unwrap(flight["longitude"], period=360.0)

    # Scenes sensed before or after the flight get nan; shapely.intersects_xy returns False
    lon = np.interp(sensing_time, time, lon, left=np.nan, right=np.nan)
    lon = (lon + 180.0) % 360.0 - 180.0
    lat = np.interp(sensing_time, time, flight["latitude"], left=np.nan, right=np.nan)

    filt = shapely.intersects_xy(scenes["geometry"].to_numpy(), lon, lat)
    out = scenes[filt].reset_index(drop=True)
    out["longitude"] = lon[filt]
    out["latitude"] = lat[filt]

    return out
