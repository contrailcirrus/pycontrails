"""Utilities for low Earth orbit satellite datalibs."""

import dataclasses

import numpy as np

from pycontrails.utils import dependencies

try:
    import geojson
except ModuleNotFoundError as exc:
    dependencies.raise_module_not_found_error(
        name="leo.utils module",
        package_name="geojson",
        module_not_found_error=exc,
        pycontrails_optional_package="leo",
    )

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

    #: GeoJSON representation of spatial ROI, optional.
    #: If not provided, extent will be global.
    extent: str = GLOBAL_EXTENT

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
