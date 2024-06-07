"""Tools for searching for low Earth orbit satellite imagery."""

import dataclasses
import pathlib

import numpy as np
import pandas as pd

from pycontrails.core import Flight
from pycontrails.utils import dependencies

try:
    import geojson
except ModuleNotFoundError as exc:
    dependencies.raise_module_not_found_error(
        name="datalib._leo_utils module",
        package_name="geojson",
        module_not_found_error=exc,
        pycontrails_optional_package="sat",
    )

_path_to_static = pathlib.Path(__file__).parent / "static"
ROI_QUERY_FILENAME = _path_to_static / "bq_roi_query.sql"


#: GeoJSON polygon that covers the entire globe.
GLOBAL_EXTENT = geojson.dumps(
    geojson.Polygon([[(-180, -90), (180, -90), (180, 90), (-180, 90), (-180, -90)]])
)


@dataclasses.dataclass
class ROI:
    """Spatiotemporal region of interest."""

    #: Start time
    start_time: np.datetime64

    #: End time
    end_time: np.datetime64

    #: GeoJSON representation of spatial ROI.
    extent: str

    def __post_init__(self) -> None:
        """Validate region of interest."""
        if self.start_time > self.end_time:
            msg = "start_time must be before end_time"
            raise ValueError(msg)

        try:
            decoded = geojson.Feature(geometry=geojson.loads(self.extent))
        except Exception as exc:
            msg = "extent cannot be converted to GeoJSON structure"
            raise ValueError(msg) from exc
        if not decoded.is_valid:
            msg = "extent is not valid GeoJSON"
            raise ValueError(msg)


def track_to_geojson(lon: np.ndarray, lat: np.ndarray) -> str:
    """Convert ground track to GeoJSON string, splitting at antimeridian crossings.

    Antimeridian crossings are defined as successive points with one longitude coordinate
    greater than 90 degrees and the other longitude coordinate less than -90 degrees.

    Parameters
    ----------
    lon : np.ndarray
        Longitude coordinates of track [WGS84]

    lat : np.ndarray
        Latitude coordinates of track [WGS84]

    Returns
    -------
    str
        String encoding of GeoJSON LineString (if track does not contain antimeridian crossings)
        or GeoJSON MultiLineString (if track contains one or more antimeridian crossings).
    """
    crossings = np.flatnonzero(
        ((lon[:-1] > 90.0) & (lon[1:] < -90.0))  # eastward
        | ((lon[:-1] < -90.0) & (lon[1:] > 90.0))  # westward
    )

    # can just return a LineString if no antimeridian crossings
    if crossings.sum() == 0:
        feature = geojson.LineString(list(zip(lon.astype(float), lat.astype(float))))
        return geojson.dumps(feature)

    # otherwise, need to split at crossings
    lon_splits = []
    lat_splits = []
    for i in range(crossings.size):
        idx = crossings[i]
        eastward = lon[idx] > 0

        bounding_lon = lon[idx : idx + 2]
        bounding_lat = lat[idx : idx + 2]
        rotated = np.where(bounding_lon < 0, bounding_lon + 360, bounding_lon)
        isort = np.argsort(rotated)  # ensure rotated longitudes are ascending
        crossing_lat = np.interp(180.0, rotated[isort], bounding_lat[isort])

        lon_split = lon[: idx + 1]
        lon = lon[idx + 1 :]
        lat_split = lat[: idx + 1]
        lat = lat[idx + 1 :]
        crossings -= lon_split.size

        # fill gaps around antimeridian if necessary
        if np.abs(lon_split[-1]) < 180.0:
            lon_split = np.append(lon_split, 180.0 if eastward else -180.0)
            lat_split = np.append(lat_split, crossing_lat)
        if np.abs(lon[0]) < 180.0:
            lon = np.insert(lon, 0, -180.0 if eastward else 180.0)
            lat = np.insert(lat, 0, crossing_lat)
            crossings += 1

        lon_splits.append(lon_split)
        lat_splits.append(lat_split)

    # add remaining segment as final split
    lon_splits.append(lon)
    lat_splits.append(lat)

    # return MultiLineString based on splits
    feature = geojson.MultiLineString(
        [
            list(zip(lon.astype(float), lat.astype(float)))
            for lon, lat in zip(lon_splits, lat_splits)
        ]
    )
    return geojson.dumps(feature)


def query(table: str, roi: ROI, columns: list[str], extra_filters: str = "") -> pd.DataFrame:
    """Find satellite imagery within region of interest.

    This function requires access to the
    `Google BigQuery API <https://cloud.google.com/bigquery?hl=en>`__
    and uses the `BigQuery python library <https://cloud.google.com/python/docs/reference/bigquery/latest/index.html>`__.

    Parameters
    ----------
    table : str
        Name of BigQuery table to query

    roi : ROI
        Region of interest

    columns : list[str]
        Columns to return from Google
        `BigQuery table <https://console.cloud.google.com/bigquery?p=bigquery-public-data&d=cloud_storage_geo_index&t=landsat_index&page=table&_ga=2.90807450.1051800793.1716904050-255800408.1705955196>`__.

    extra_filters : str, optional
        Additional selection filters, injected verbatim into constructed query.

    Returns
    -------
    pd.DataFrame
        Query results in pandas DataFrame
    """
    try:
        from google.cloud import bigquery
    except ModuleNotFoundError as exc:
        dependencies.raise_module_not_found_error(
            name="landsat module",
            package_name="google-cloud-bigquery",
            module_not_found_error=exc,
            pycontrails_optional_package="landsat",
        )

    if len(columns) == 0:
        msg = "At least column must be provided."
        raise ValueError(msg)

    start_time = pd.Timestamp(roi.start_time).strftime("%Y-%m-%d %H:%M:%S")
    end_time = pd.Timestamp(roi.end_time).strftime("%Y-%m-%d %H:%M:%S")
    extent = roi.extent.replace('"', "'")

    client = bigquery.Client()
    with open(ROI_QUERY_FILENAME) as f:
        query_str = f.read().format(
            table=table,
            columns=",".join(columns),
            start_time=start_time,
            end_time=end_time,
            geojson_str=extent,
            extra_filters=extra_filters,
        )

    result = client.query(query_str)
    return result.to_dataframe()


def intersect(
    table: str, flight: Flight, columns: list[str], extra_filters: str = ""
) -> pd.DataFrame:
    """Find satellite imagery intersecting with flight track.

    This function will return all scenes with a bounding box that includes flight waypoints
    both before and after the sensing time.

    This function requires access to the
    `Google BigQuery API <https://cloud.google.com/bigquery?hl=en>`__
    and uses the `BigQuery python library <https://cloud.google.com/python/docs/reference/bigquery/latest/index.html>`__.

    Parameters
    ----------
    table : str
        Name of BigQuery table to query

    flight : Flight
        Flight for intersection

    columns : list[str]
        Columns to return from Google
        `BigQuery table <https://console.cloud.google.com/bigquery?p=bigquery-public-data&d=cloud_storage_geo_index&t=landsat_index&page=table&_ga=2.90807450.1051800793.1716904050-255800408.1705955196>`__.

    extra_filters : str, optional
        Additional selection filters, injected verbatim into constructed query.

    Returns
    -------
    pd.DataFrame
        Query results in pandas DataFrame
    """

    # create ROI with time span between flight start and end
    # and spatial extent set to flight track
    extent = track_to_geojson(flight["longitude"], flight["latitude"])
    roi = ROI(start_time=flight["time"].min(), end_time=flight["time"].max(), extent=extent)

    # first pass: query for intersections with ROI
    # requires additional columns for final intersection with flight
    required_columns = set(["sensing_time", "west_lon", "east_lon", "south_lat", "north_lat"])
    queried_columns = list(required_columns.union(set(columns)))
    candidates = query(table, roi, queried_columns, extra_filters)

    if len(candidates) == 0:  # already know there are no intersections
        return candidates[columns]

    # second pass: keep images with where flight waypoints
    # bounding sensing time are both within bounding box
    flight_data = flight.dataframe

    def intersects(scene: pd.Series) -> bool:
        if scene["west_lon"] <= scene["east_lon"]:  # scene does not span antimeridian
            bbox_data = flight_data[
                flight_data["longitude"].between(scene["west_lon"], scene["east_lon"])
                & flight_data["latitude"].between(scene["south_lat"], scene["north_lat"])
            ]
        else:  # scene spans antimeridian
            bbox_data = flight_data[
                (
                    flight_data["longitude"]
                    > scene["west_lon"] | flight.data["longitude"]
                    < scene["east_lon"]
                )
                & flight_data["latitude"].between(scene["south_lat"], scene["north_lat"])
            ]
        sensing_time = pd.Timestamp(scene["sensing_time"]).tz_localize(None)
        return bbox_data["time"].min() <= sensing_time and bbox_data["time"].max() >= sensing_time

    mask = candidates.apply(intersects, axis="columns")
    return candidates[columns][mask]
