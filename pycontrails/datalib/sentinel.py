"""Support for Sentinel-2 imagery retrieval through Google Cloud Platform."""

from __future__ import annotations

import pathlib
from collections.abc import Iterable
from xml.etree import ElementTree

import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr

from pycontrails.core import Flight, cache
from pycontrails.datalib.leo_utils import correction, search
from pycontrails.datalib.leo_utils.sentinel_metadata import (
    _band_id,
    get_detector_id,
    get_time_delay_detectors,
    parse_ephemeris_sentinel,
    parse_high_res_viewing_incidence_angles,
    parse_sensing_time,
    parse_sentinel_crs,
    read_image_coordinates,
)
from pycontrails.datalib.leo_utils.vis import equalize, normalize
from pycontrails.utils import dependencies

try:
    import gcsfs
except ModuleNotFoundError as exc:
    dependencies.raise_module_not_found_error(
        name="sentinel module",
        package_name="gcsfs",
        module_not_found_error=exc,
        pycontrails_optional_package="sat",
    )

try:
    import pyproj
except ModuleNotFoundError as exc:
    dependencies.raise_module_not_found_error(
        name="sentinel module",
        package_name="pyproj",
        module_not_found_error=exc,
        pycontrails_optional_package="sat",
    )

try:
    from PIL import Image
except ModuleNotFoundError as exc:
    dependencies.raise_module_not_found_error(
        name="sentinel module",
        package_name="pillow",
        module_not_found_error=exc,
        pycontrails_optional_package="sat",
    )


_path_to_static = pathlib.Path(__file__).parent / "static"
ROI_QUERY_FILENAME = _path_to_static / "sentinel_roi_query.sql"

#: BigQuery table with imagery metadata
BQ_TABLE = "bigquery-public-data.cloud_storage_geo_index.sentinel_2_index"

#: Default columns to include in queries
BQ_DEFAULT_COLUMNS = ["base_url", "granule_id", "sensing_time", "source_url"]

#: Default spatial extent for queries
BQ_DEFAULT_EXTENT = search.GLOBAL_EXTENT

#: Default Sentinel channels to use if none are specified.
#: These are visible bands for producing a true color composite.
DEFAULT_BANDS = ["B02", "B03", "B04"]

#: Strip this prefix from GCP URLs when caching Sentinel files locally
GCP_STRIP_PREFIX = "gs://gcp-public-data-sentinel-2/"


def query(
    start_time: np.datetime64,
    end_time: np.datetime64,
    extent: str | None = None,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Find Sentinel-2 imagery within spatiotemporal region of interest.

    This function requires access to the
    `Google BigQuery API <https://cloud.google.com/bigquery?hl=en>`__
    and uses the `BigQuery python library <https://cloud.google.com/python/docs/reference/bigquery/latest/index.html>`__.

    Parameters
    ----------
    start_time : np.datetime64
        Start of time period for search
    end_time : np.datetime64
        End of time period for search
    extent : str | None, optional
        Spatial region of interest as a GeoJSON string. If not provided, defaults
        to a global extent.
    columns : list[str] | None, optional
        Columns to return from Google
        `BigQuery table <https://console.cloud.google.com/bigquery?p=bigquery-public-data&d=cloud_storage_geo_index&t=landsat_index&page=table&_ga=2.90807450.1051800793.1716904050-255800408.1705955196>`__.
        By default, returns imagery base URL, granule ID, and sensing time.

    Returns
    -------
    pd.DataFrame
        Query results in pandas DataFrame

    See Also
    --------
    :func:`search.query`
    """
    extent = extent or BQ_DEFAULT_EXTENT
    roi = search.ROI(start_time, end_time, extent)
    columns = columns or BQ_DEFAULT_COLUMNS
    return search.query(BQ_TABLE, roi, columns)


def intersect(
    flight: Flight,
    columns: list[str] | None = None,
) -> pd.DataFrame:
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
    columns : list[str] | None, optional
        Columns to return from Google
        `BigQuery table <https://console.cloud.google.com/bigquery?p=bigquery-public-data&d=cloud_storage_geo_index&t=landsat_index&page=table&_ga=2.90807450.1051800793.1716904050-255800408.1705955196>`__.
        By default, returns imagery base URL, granule ID, and sensing time.

    Returns
    -------
    pd.DataFrame
        Query results in pandas DataFrame

    See Also
    --------
    :func:`search.intersect`
    """
    columns = columns or BQ_DEFAULT_COLUMNS
    scenes = search.intersect(BQ_TABLE, flight, columns)

    # overwrite the base_url with source_url.
    # After 2024-03-14 there is a mistake in the Google BigQuery table,
    # such that the base_url is written to the source_url column
    scenes["base_url"] = scenes["base_url"].fillna(scenes["source_url"])
    return scenes


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
    bands : str | Iterable[str] | None
        Set of bands to retrieve. The 13 possible bands are represented by
        the string "B01" to "B12" plus "B8A". For the true color scheme, set
        ``bands=("B02", "B03", "B04")``. By default, bands for the true color scheme
        are used. Bands must share a common resolution. The resolutions of each band are:

        - B02-B04, B08: 10 m
        - B05-B07, B8A, B11, B12: 20 m
        - B01, B09, B10: 60 m

    cachestore : cache.CacheStore | None, optional
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
        reflective : str, optional
            Set to "raw" to return raw values or "reflectance" for rescaled reflectances.
            By default, return reflectances.

        Returns
        -------
        xr.Dataset
            Dataset of Sentinel-2 data.
        """
        available = ("raw", "reflectance")
        if reflective not in available:
            msg = f"reflective band processing must be one of {available}"
            raise ValueError(msg)

        data = {band: self._get(band, reflective) for band in self.bands}
        return xr.Dataset(data)

    def get_viewing_angle_metadata(self, scale: int = 10) -> xr.Dataset:
        """Return the dataset with viewing angles.

        See :func:`parse_high_res_viewing_incidence_angles` for details.
        """
        granule_meta_path, _ = self._get_meta()
        _, detector_band_path = self._get_correction_meta()
        return parse_high_res_viewing_incidence_angles(
            granule_meta_path, detector_band_path, scale=scale
        )

    def get_detector_id(
        self, x: npt.NDArray[np.floating], y: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.integer]:
        """Return the detector_id of the Sentinel-2 detector that imaged the given points.

        Parameters
        ----------
        x : npt.NDArray[np.floating]
            x coordinates of points in the Sentinel-2 image CRS
        y : npt.NDArray[np.floating]
            y coordinates of points in the Sentinel-2 image CRS

        Returns
        -------
        npt.NDArray[np.integer]
            Detector IDs for each point. If a point is outside the image, the detector ID is 0.
        """
        granule_sink, _ = self._get_meta()
        _, detector_band_sink = self._get_correction_meta()
        return get_detector_id(detector_band_sink, granule_sink, x, y)

    def get_time_delay_detector(
        self, detector_id: npt.NDArray[np.integer], band: str = "B03"
    ) -> npt.NDArray[np.timedelta64]:
        """Return the time delay for the given detector IDs."""
        datastrip_sink, _ = self._get_correction_meta()
        delays = get_time_delay_detectors(datastrip_sink, band)

        # Map the delays to the input array
        unique_ids, inverse = np.unique(detector_id, return_inverse=True)

        # Get offsets for each unique detector
        offset_list = []
        for did in unique_ids:
            if did == 0:
                offset_list.append(np.timedelta64("NaT"))
            else:
                offset_list.append(delays.get(did, np.timedelta64("NaT")))
        offset_values = np.array(offset_list, dtype="timedelta64[ns]")

        # Map back to the original array
        return offset_values[inverse]

    def get_ephemeris(self) -> pd.DataFrame:
        """Return the satellite ephemeris as a :class:`pandas.DataFrame`."""
        datastrip_sink, _ = self._get_correction_meta()

        return parse_ephemeris_sentinel(datastrip_sink)

    def get_crs(self) -> pyproj.CRS:
        """Return the CRS of the satellite image."""
        granule_meta_path, _ = self._get_meta()
        return parse_sentinel_crs(granule_meta_path)

    def get_sensing_time(self) -> pd.Timestamp:
        """Return the sensing_time of the satellite image."""
        granule_meta_path, _ = self._get_meta()
        return parse_sensing_time(granule_meta_path)

    def colocate_flight(
        self,
        flight: Flight,
        band: str = "B03",
    ) -> tuple[float, float, np.datetime64]:
        """Colocate a flight track with satellite image pixels.

        This function first projects flight waypoints into the UTM coordinate system
        of the satellite image, then applies a viewing angle correction to estimate
        the actual ground position imaged by the satellite. Next, the scan time for
        each point is estimated based on the satellite meta data. Finally, the
        point along the flight track is found where the flight time matches the
        satellite scan time.

        Parameters
        ----------
        flight : Flight
            The flight object.
        band : str, optional
            Spectral band to use for geometry parsing. Default is "B03".

        Returns
        -------
        tuple[float, float, np.datetime64]
            A tuple containing the x and y coordinates of the flight position in the
            satellite image CRS, and the corrected sensing_time of the satellite image.
        """
        utm_crs = self.get_crs()

        # Project from WGS84 to the x and y coordinates in the UTM coordinate system
        transformer = pyproj.Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)
        x, y = transformer.transform(flight["longitude"], flight["latitude"])
        z = flight.altitude

        # Apply sensing angle correction
        ds_angles = self.get_viewing_angle_metadata()
        x_proj, y_proj = correction.scan_angle_correction(ds_angles, x, y, z, maxiter=3)

        # Get the satellite ephemeris data prepared
        ephemeris = self.get_ephemeris()

        # Estimate the scan time for each point
        scan_time_no_offset = correction.estimate_scan_time(ephemeris, utm_crs, x_proj, y_proj)

        detector_id = self.get_detector_id(x_proj, y_proj)
        offset = self.get_time_delay_detector(detector_id, band=band)
        scan_time = scan_time_no_offset + offset

        # To finish up, find the point along the project flight track at which the satellite
        # scan time equals the flight time
        # It's possible that if the flight has a really fine time resolution, that
        # the diff_times array will have multiple zeros. This is okay, as np.interp
        # will just return the first one.
        valid = np.isfinite(x_proj) & np.isfinite(y_proj) & np.isfinite(scan_time)

        diff_times = flight["time"][valid] - scan_time[valid]
        if diff_times[0] > np.timedelta64(0) or diff_times[-1] < np.timedelta64(0):
            msg = "Flight track does not overlap with satellite image in time."
            raise ValueError(msg)

        index = np.arange(len(diff_times))
        frac_idx = np.interp(0, diff_times.astype(int), index)

        x1 = np.interp(frac_idx, index, x_proj[valid]).item()
        y1 = np.interp(frac_idx, index, y_proj[valid]).item()
        t1 = np.interp(frac_idx, index, scan_time[valid].astype(int)).item()

        return x1, y1, np.datetime64(int(t1), "ns")

    # -------------------------------------------------------------------------------------

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

    def _get_correction_meta(self) -> tuple[str, str]:
        """Download Sentinel-2 metadata files and return path to cached files.

        Note that two XML files must be retrieved: one inside the GRANULE
        subdirectory, and one at the top level of the SAFE archive.
        """
        fs = self.fs
        base_url = self.base_url
        granule_id = self.granule_id

        # Resolve the unknown subfolder in DATASTRIP using glob
        # Probably there is a better method, but this worked for now
        pattern = f"{base_url}/DATASTRIP/*"
        matches = fs.glob(pattern)
        if not matches:
            raise FileNotFoundError(f"No DATASTRIP MTD_MDS.xml file found at pattern: {pattern}")
        if len(matches) > 2:
            raise RuntimeError(f"Multiple DATASTRIP MTD_MDS.xml files found: {matches}")

        datastrip_id = matches[0].split("/")[-1].replace("_$folder$", "")
        url = f"{base_url}/DATASTRIP/{datastrip_id}/MTD_DS.xml"
        datastrip_sink = self.cachestore.path(url.removeprefix(GCP_STRIP_PREFIX))
        if not self.cachestore.exists(datastrip_sink):
            fs.get(url, datastrip_sink)

        # Path to the QI_DATA folder
        qi_data_url = f"{base_url}/GRANULE/{granule_id}/QI_DATA/"

        # List files in QI_DATA (assuming fs can list directories)
        files = fs.ls(qi_data_url)

        # Filter for the detector mask files
        mask_files = [f for f in files if "MSK_DETFOO_B03" in f]

        if not mask_files:
            raise FileNotFoundError(f"No detector mask found in {qi_data_url}")

        # Choose the first available file (could prioritize .jp2 over .gml if needed)
        mask_file = None
        for ext in [".jp2", ".gml"]:
            for f in mask_files:
                if f.endswith(ext):
                    mask_file = f
                    break
            if mask_file:
                break

        if mask_file is not None:
            # Path where the file will be cached
            detector_band_sink = self.cachestore.path(mask_file.removeprefix(GCP_STRIP_PREFIX))
            if not self.cachestore.exists(detector_band_sink):
                fs.get(mask_file, detector_band_sink)

        return datastrip_sink, detector_band_sink


def _parse_bands(bands: str | Iterable[str] | None) -> set[str]:
    """Check that the bands are valid and return as a set."""
    if bands is None:
        return set(DEFAULT_BANDS)

    if isinstance(bands, str):
        bands = (bands,)

    available = {f"B{i:02d}" for i in range(1, 13)} | {"B8A"}
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
    with Image.open(path) as src:
        img = np.asarray(src)

    if processing == "reflectance":
        gain, offset = _read_band_reflectance_rescaling(safe_meta, band)
        img = np.where(img == 0, np.nan, (img + offset) / gain).astype("float32")

    tree = ElementTree.parse(granule_meta)
    elem = tree.find(".//HORIZONTAL_CS_CODE")
    if elem is None or elem.text is None:
        msg = "Could not find imagery projection in metadata."
        raise ValueError(msg)
    epsg = int(elem.text.split(":")[1])
    crs = pyproj.CRS.from_epsg(epsg)

    x, y = read_image_coordinates(granule_meta, band)

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


def _read_band_reflectance_rescaling(meta: str, band: str) -> tuple[float, float]:
    """Read reflectance rescaling factors from metadata file.

    See https://sentiwiki.copernicus.eu/web/s2-processing#S2Processing-TOAReflectanceComputation
    and https://scihub.copernicus.eu/news/News00931.
    """
    # Find quantization gain (present in all files)
    tree = ElementTree.parse(meta)
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
            return gain, offset

    msg = f"Could not find reflectance offset for band {band} (band ID {band_id})"
    raise ValueError(msg)


def extract_sentinel_visualization(
    ds: xr.Dataset, color_scheme: str = "true"
) -> tuple[npt.NDArray[np.float32], pyproj.CRS, tuple[float, float, float, float]]:
    """Extract artifacts for visualizing Sentinel data with the given color scheme.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset of Sentinel data as returned by :meth:`Sentinel.get`.
    color_scheme : str, optional
        Color scheme to use for visualization. The true color scheme
        (`"true"`, the only option currently implemented) requires bands B02, B03, and B04.

    Returns
    -------
    rgb : npt.NDArray[np.float32]
        3D RGB array of shape ``(height, width, 3)``.
    src_crs : pyproj.CRS
        Imagery projection
    src_extent : tuple[float, float, float, float]
        Imagery extent in projected coordinates
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
