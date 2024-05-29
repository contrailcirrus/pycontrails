"""Support for Sentinel-2 imagery retrieval through Google Cloud Platform."""

import pathlib
from collections.abc import Iterable
from xml import etree

import numpy as np
import pandas as pd
import xarray as xr

from pycontrails.core import Flight, cache
from pycontrails.datalib.leo.utils import ROI, track_to_geojson
from pycontrails.datalib.leo.vis import equalize, normalize
from pycontrails.utils import dependencies

try:
    import gcsfs
except ModuleNotFoundError as exc:
    dependencies.raise_module_not_found_error(
        name="sentinel module",
        package_name="gcsfs",
        module_not_found_error=exc,
        pycontrails_optional_package="leo",
    )

try:
    import pyproj
except ModuleNotFoundError as exc:
    dependencies.raise_module_not_found_error(
        name="sentinel module",
        package_name="pyproj",
        module_not_found_error=exc,
        pycontrails_optional_package="leo",
    )

try:
    from PIL import Image
except ModuleNotFoundError as exc:
    dependencies.raise_module_not_found_error(
        name="sentinel module",
        package_name="pillow",
        module_not_found_error=exc,
        pycontrails_optional_package="leo",
    )


_path_to_static = pathlib.Path(__file__).parent / "static"
ROI_QUERY_FILENAME = _path_to_static / "sentinel_roi_query.sql"

#: Default Sentinel channels to use if none are specified.
#: These are visible bands for producing a true color composite.
DEFAULT_BANDS = ["B02", "B03", "B04"]

#: Strip this prefix from GCP URLs when caching Sentinel files locally
GCP_STRIP_PREFIX = "gs://gcp-public-data-sentinel-2/"


def query(roi: ROI, columns: list[str] | None = None) -> pd.DataFrame:
    """Query Sentinel-2 imagery within region of interest.

    This function requires access to the
    `Google BigQuery API <https://cloud.google.com/bigquery?hl=en>`__
    and uses the `BigQuery python library <https://cloud.google.com/python/docs/reference/bigquery/latest/index.html>`__.

    Parameters
    ----------
    roi : ROI
        Region of interest

    columns : list[str], optional.
        Columns to return from Google
        `BigQuery table <https://console.cloud.google.com/bigquery?p=bigquery-public-data&d=cloud_storage_geo_index&t=landsat_index&page=table&_ga=2.90807450.1051800793.1716904050-255800408.1705955196>`__.
        By default, returns imagery base URL and granule ID.

    Returns
    -------
    pd.DataFrame
        Query results in pandas DataFrame
    """
    try:
        from google.cloud import bigquery
    except ModuleNotFoundError as exc:
        dependencies.raise_module_not_found_error(
            name="sentinel module",
            package_name="google-cloud-bigquery",
            module_not_found_error=exc,
            pycontrails_optional_package="landsat",
        )

    columns = columns or ["base_url", "granule_id"]

    start_time = pd.Timestamp(roi.start_time).strftime("%Y-%m-%d %H:%M:%S")
    end_time = pd.Timestamp(roi.end_time).strftime("%Y-%m-%d %H:%M:%S")
    extent = roi.extent.replace('"', "'")

    client = bigquery.Client()
    with open(ROI_QUERY_FILENAME) as f:
        query_str = f.read().format(
            columns=",".join(columns), start_time=start_time, end_time=end_time, geojson_str=extent
        )

    result = client.query(query_str)
    return result.to_dataframe()


def intersect(flight: Flight, columns: list[str] | None = None) -> pd.DataFrame:
    """Find Sentinel-2 imagery intersecting with flight track.

    This function will return all scenes with a bounding box that includes flight waypoints
    both before and after the sensing time.

    This function requires access to the
    `Google BigQuery API <https://cloud.google.com/bigquery?hl=en>`__
    and uses the `BigQuery python library <https://cloud.google.com/python/docs/reference/bigquery/latest/index.html>`__.

    Parameters
    ----------
    flight : Flight
        Flight for intersection

    columns : list[str], optional.
        Columns to return from Google
        `BigQuery table <https://console.cloud.google.com/bigquery?p=bigquery-public-data&d=cloud_storage_geo_index&t=landsat_index&page=table&_ga=2.90807450.1051800793.1716904050-255800408.1705955196>`__.
        By default, returns imagery base URL and granule ID.

    Returns
    -------
    pd.DataFrame
        Query results in pandas DataFrame
    """
    columns = columns or ["base_url", "granule_id"]

    # create ROI with time span between flight start and end
    # and spatial extent set to flight track
    extent = track_to_geojson(flight["longitude"], flight["latitude"])
    roi = ROI(start_time=flight["time"].min(), end_time=flight["time"].max(), extent=extent)

    # first pass: query for intersections with ROI
    # requires additional columns for final intersection with flight
    required_columns = set(["sensing_time", "west_lon", "east_lon", "south_lat", "north_lat"])
    queried_columns = list(required_columns.union(set(columns)))
    candidates = query(roi, columns=queried_columns)

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


class Sentinel:
    """Support for Sentinel-2 data handling.

    Parameters
    ----------
    base_url : str
        Base URL of Sentinel-2 scene. To find URLs for Sentinel-2 scenes at
        specific locations and times, see :func:`query` and :func:`intersect`.

    granule_id : str
        Granule ID of Sentinel-2 scene. To find URLs for Sentinel-2 scenes at
        specific locations and times, see :func:`query` and :func:`intersect`.

    bands : str | set[str] | None
        Set of bands to retrieve. The 13 possible bands are represented by
        the string "B01" to "B12" plus "B8A". For the true color scheme, set
        ``bands=("B02", "B03", "B04")``. By default, bands for the true color scheme
        are used. Bands must share a common resolution. The resolutions of each band are:

        - B02-B04, B08: 10 m
        - B05-B07, B8A, B11, B12: 20 m
        - B01, B09, B10: 60 m

    cachestore : cache.CacheStore, optional
        Cache store for Landsat data. If None, a :class:`DiskCacheStore` is used.

    See Also
    --------
    query
    intersect
    """

    def __init__(
        self,
        base_url: str,
        granule_id: str,
        bands: str | Iterable[str] | None = None,
        cachestore: cache.CacheStore | None = None,
    ) -> None:

        self.base_url = base_url
        self.granule_id = granule_id
        self.bands = _parse_bands(bands)
        _check_band_resolution(self.bands)
        self.fs = gcsfs.GCSFileSystem(token="anon")

        if cachestore is None:
            cache_root = cache._get_user_cache_dir()
            cache_dir = f"{cache_root}/sentinel"
            cachestore = cache.DiskCacheStore(cache_dir=cache_dir)
        self.cachestore = cachestore

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"Sentinel(base_url='{self.base_url}',\n"
            f"\tgranule_id='{self.granule_id}',\n"
            f"\tbands={sorted(self.bands)})"
        )

    def get(self, reflective: str = "reflectance") -> xr.Dataset:
        """Retrieve Sentinel-2 imagery.

        Parameters
        ----------
        reflective : str = {"raw", "reflectance"}, optional
            Whether to return raw values or rescaled reflectances for reflective bands.
            By default, return reflectances.

        Returns
        -------
        xr.DataArray
            DataArray of Sentinel-2 data.
        """
        if reflective not in ["raw", "reflectance"]:
            msg = "reflective band processing must be one of ['raw', 'radiance', 'reflectance']"
            raise ValueError(msg)

        ds = xr.Dataset()
        for band in self.bands:
            ds[band] = self._get(band, reflective)
        return ds

    def _get(self, band: str, processing: str) -> xr.DataArray:
        """Download Sentinel-2 band to the :attr:`cachestore` and return processed data."""
        jp2_path = self._get_jp2(band)
        granule_meta_path, safe_meta_path = self._get_meta()
        return _read(jp2_path, granule_meta_path, safe_meta_path, band, processing)

    def _get_jp2(self, band: str) -> str:
        """Download Sentinel-2 imagery and return path to cached file."""
        fs = self.fs
        base_url = self.base_url
        granule_id = self.granule_id
        prefix = f"{base_url}/GRANULE/{granule_id}/IMG_DATA"
        files = fs.ls(prefix)

        urls = [f"gs://{f}" for f in files if f.endswith(f"{band}.jp2")]
        if len(urls) > 1:
            msg = f"Multiple image files found for band {band}"
            raise ValueError(msg)
        if len(urls) == 0:
            msg = f"No image files found for band {band}"
            raise ValueError(msg)
        url = urls[0]

        sink = self.cachestore.path(url.removeprefix(GCP_STRIP_PREFIX))
        if not self.cachestore.exists(sink):
            fs.get(url, sink)
        return sink

    def _get_meta(self) -> tuple[str, str]:
        """Download Sentinel-2 metadata files and return path to cached files.

        Note that two XML files must be retrieved: one inside the GRANULE
        subdirectory, and one at the top level of the SAFE archive.
        """
        fs = self.fs
        base_url = self.base_url
        granule_id = self.granule_id

        url = f"{base_url}/GRANULE/{granule_id}/MTD_TL.xml"
        granule_sink = self.cachestore.path(url.removeprefix(GCP_STRIP_PREFIX))
        if not self.cachestore.exists(granule_sink):
            fs.get(url, granule_sink)

        url = f"{base_url}/MTD_MSIL1C.xml"
        safe_sink = self.cachestore.path(url.removeprefix(GCP_STRIP_PREFIX))
        if not self.cachestore.exists(safe_sink):
            fs.get(url, safe_sink)

        return granule_sink, safe_sink


def _parse_bands(bands: str | Iterable[str] | None) -> set[str]:
    """Check that the bands are valid and return as a set."""
    if bands is None:
        return set(DEFAULT_BANDS)

    if isinstance(bands, str):
        bands = (bands,)

    available = {f"B{i:2d}" for i in range(1, 12)} | {"B8A"}
    bands = {b.upper() for b in bands}
    if len(bands) == 0:
        msg = "At least one band must be provided"
        raise ValueError(msg)
    if not bands.issubset(available):
        msg = f"Bands must be in {sorted(available)}"
        raise ValueError(msg)
    return bands


def _check_band_resolution(bands: set[str]) -> None:
    """Confirm requested bands have a common horizontal resolution."""
    groups = [
        {"B02", "B03", "B04", "B08"},  # 10 m
        {"B05", "B06", "B07", "B8A", "B11", "B12"},  # 20 m
        {"B01", "B09", "B10"},  # 60 m
    ]
    if not any(bands.issubset(group) for group in groups):
        msg = "Bands must have a common horizontal resolution."
        raise ValueError(msg)


def _read(path: str, granule_meta: str, safe_meta: str, band: str, processing: str) -> xr.DataArray:
    """Read imagery data from Sentinel-2 files."""
    Image.MAX_IMAGE_PIXELS = None  # avoid decompression bomb warning
    src = Image.open(path)
    img = np.asarray(src)
    src.close()

    if processing == "reflectance":
        gain, offset = _read_band_reflectance_rescaling(safe_meta, band)
        img = np.where(img == 0, np.nan, (img + offset) / gain).astype("float32")

    tree = etree.ElementTree.parse(granule_meta)
    elem = tree.find(".//HORIZONTAL_CS_CODE")
    if elem is None or elem.text is None:
        msg = "Could not find imagery projection in metadata."
        raise ValueError(msg)
    epsg = int(elem.text.split(":")[1])
    crs = pyproj.CRS.from_epsg(epsg)

    x, y = _read_image_coordinates(granule_meta, band)

    da = xr.DataArray(
        data=img,
        coords={"y": y, "x": x},
        dims=("y", "x"),
        attrs={
            "long_name": f"{band} {processing}",
            "units": "nondim" if processing == "reflectance" else "none",
            "crs": crs,
        },
    )
    da["x"].attrs = {"long_name": "easting", "units": "m"}
    da["y"].attrs = {"long_name": "northing", "units": "m"}
    return da


def _band_resolution(band: str) -> int:
    """Get band resolution in meters."""
    return (
        60 if band in ("B01", "B09", "B10") else 10 if band in ("B02", "B03", "B04", "B08") else 20
    )


def _band_id(band: str) -> int:
    """Get band ID used in some metadata files."""
    if band in (f"B{i:2d}" for i in range(1, 9)):
        return int(band[1:]) - 1
    elif band == "B8A":
        return 8
    else:
        return int(band[1:])


def _read_band_reflectance_rescaling(meta: str, band: str) -> tuple[float, float]:
    """Read reflectance rescaling factors from metadata file.

    See https://sentiwiki.copernicus.eu/web/s2-processing#S2Processing-TOAReflectanceComputation
    and https://scihub.copernicus.eu/news/News00931.
    """
    # Find quantization gain (present in all files)
    tree = etree.ElementTree.parse(meta)
    elem = tree.find(".//QUANTIFICATION_VALUE")
    if elem is None or elem.text is None:
        msg = "Could not find reflectance quantization gain."
        raise ValueError(msg)
    gain = float(elem.text)

    # See if offset (used in recently processed files) is present
    elems = tree.findall(".//RADIO_ADD_OFFSET")

    # If not, set offset to 0
    if len(elems) == 0:
        return gain, 0.0

    # Otherwise, search for offset with correct band ID
    band_id = _band_id(band)
    for elem in elems:
        if int(elem.attrib["band_id"]) == band_id and elem.text is not None:
            offset = float(elem.text)
            break
    else:
        msg = f"Could not find reflectance offset for band {band} (band ID {band_id})"
        raise ValueError(msg)

    return gain, offset


def _read_image_coordinates(meta: str, band: str) -> tuple[np.ndarray, np.ndarray]:
    """Read image x and y coordinates."""

    # convenience function that satisfies mypy
    def _text_from_tag(parent: etree.ElementTree.Element, tag: str) -> str:
        elem = parent.find(tag)
        if elem is None or elem.text is None:
            msg = f"Could not find text in {tag} element"
            raise ValueError(msg)
        return elem.text

    resolution = _band_resolution(band)

    # find coordinates of upper left corner and pixel size
    tree = etree.ElementTree.parse(meta)
    elems = tree.findall(".//Geoposition")
    for elem in elems:
        if int(elem.attrib["resolution"]) == resolution:
            ulx = float(_text_from_tag(elem, "ULX"))
            uly = float(_text_from_tag(elem, "ULY"))
            dx = float(_text_from_tag(elem, "XDIM"))
            dy = float(_text_from_tag(elem, "YDIM"))
            break
    else:
        msg = f"Could not find image geoposition for resolution of {resolution} m"
        raise ValueError(msg)

    # find image size
    elems = tree.findall(".//Size")
    for elem in elems:
        if int(elem.attrib["resolution"]) == resolution:
            nx = int(_text_from_tag(elem, "NCOLS"))
            ny = int(_text_from_tag(elem, "NROWS"))
            break
    else:
        msg = f"Could not find image size for resolution of {resolution} m"
        raise ValueError(msg)

    # compute pixel coordinates
    xlim = (ulx, ulx + (nx - 1) * dx)
    ylim = (uly, uly - (ny - 1) * dy)
    x = np.linspace(xlim[0], xlim[1], nx)
    y = np.linspace(ylim[0], ylim[1], ny)

    return x, y


def extract_sentinel_visualization(
    ds: xr.Dataset, color_scheme: str = "true"
) -> tuple[np.ndarray, pyproj.CRS, tuple[float, float, float, float]]:
    """Extract artifacts for visualizing Sentinel data with the given color scheme.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset of Sentinel data as returned by :meth:`Sentinel.get`.
    color_scheme : str = {"true"}
        Color scheme to use for visualization. The true color scheme
        (the only option currently implemented) requires bands B02, B03, and B04.

    Returns
    -------
    rgb : npt.NDArray[np.float32]
        3D RGB array of shape ``(height, width, 3)``.
    src_crs : pyproj.CRS
        Imagery projection
    src_extent : tuple[float, float, float, float]
        Imagery extent in projected coordinates

    References
    ----------
    :cite:`mccloskeyHumanLabeledLandsat2021`

    """

    if color_scheme == "true":
        rgb, src_crs = to_true_color(ds)
    else:
        raise ValueError(f"Color scheme must be 'true', not '{color_scheme}'")

    x = ds["x"].values
    y = ds["y"].values
    src_extent = x.min(), x.max(), y.min(), y.max()

    return rgb, src_crs, src_extent


def to_true_color(ds: xr.Dataset) -> tuple[np.ndarray, pyproj.CRS]:
    """Compute 3d RGB array for the true color scheme.

    Parameters
    ----------
    ds : xr.Dataset
        DataArray of Sentinel data with bands B02, B03, and B04.

    Returns
    -------
    np.ndarray
        3d RGB array with true color scheme.

    src_crs : pyproj.CRS
        Imagery projection
    """
    red = ds["B04"]
    green = ds["B03"]
    blue = ds["B02"]

    crs = red.attrs["crs"]
    if not (crs.equals(green.attrs["crs"]) and crs.equals(blue.attrs["crs"])):
        msg = "Bands B02, B03, and B04 do not share a common projection."
        raise ValueError(msg)

    img = np.dstack(
        [equalize(normalize(band.values), clip_limit=0.03) for band in (red, green, blue)]
    )

    return img, crs
