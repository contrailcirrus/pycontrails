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


def track_to_geojson(flight: Flight) -> str:
    """Convert ground track to GeoJSON string, splitting at antimeridian crossings.

    Coordinates contain longitude and latitude only (no altitude coordinate)
    and are padded to terminate and restart exactly at the antimeridian when
    antimeridian crossings are encountered.

    Parameters
    ----------
    flight : Flight
        Flight with ground track to convert to GeoJSON string.

    Returns
    -------
    str
        String encoding of a GeoJSON MultiLineString containing ground track split at
        antimeridian crossings.

    See Also
    --------
    :meth:`Flight.to_geojson_multilinestring`
    """

    # Logic assumes longitudes are between -180 and 180.
    # Raise an error if this is not the case.
    if np.abs(flight["longitude"]).max() > 180.0:
        msg = "Flight longitudes must be between -180 and 180."
        raise ValueError(msg)

    # Get feature collection containing a single multilinestring
    # split at antimeridian crossings
    fc = flight.to_geojson_multilinestring(split_antimeridian=True)

    # Extract multilinestring
    mls = fc["features"][0]["geometry"]

    # Strip altitude coordinates
    coords = [[[c[0], c[1]] for c in linestring] for linestring in mls["coordinates"]]

    # No padding required if no antimeridian crossings were encountered
    if len(coords) == 1:
        return geojson.dumps(geojson.MultiLineString(coords))

    # Pad at crossings
    for i in range(len(coords) - 1):
        x0 = coords[i][-1][0]
        x1 = coords[i + 1][0][0]
        if abs(x0) == 180.0 and abs(x1) == 180.0:
            continue
        y0 = coords[i][-1][1]
        y1 = coords[i + 1][0][1]
        xl = 180.0 * np.sign(x0)
        xr = 180.0 * np.sign(x1)
        w0 = np.abs(xr - x1)
        w1 = np.abs(xl - x0)
        yc = (w0 * y0 + w1 * y1) / (w0 + w1)
        if abs(x0) < 180.0:
            coords[i].append([xl, yc])
        if abs(x1) < 180.0:
            coords[i + 1].insert(0, [xr, yc])

    # Encode as string
    return geojson.dumps(geojson.MultiLineString(coords))


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
    extent = track_to_geojson(flight)
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
