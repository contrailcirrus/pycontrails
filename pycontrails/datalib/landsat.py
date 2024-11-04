"""Support for LANDSAT 8-9 imagery retrieval through Google Cloud Platform."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
import xarray as xr

from pycontrails.core import Flight, cache
from pycontrails.datalib._leo_utils import search
from pycontrails.datalib._leo_utils.vis import equalize, normalize
from pycontrails.utils import dependencies

try:
    import gcsfs
except ModuleNotFoundError as exc:
    dependencies.raise_module_not_found_error(
        name="landsat module",
        package_name="gcsfs",
        module_not_found_error=exc,
        pycontrails_optional_package="sat",
    )

try:
    import pyproj
except ModuleNotFoundError as exc:
    dependencies.raise_module_not_found_error(
        name="landsat module",
        package_name="pyproj",
        module_not_found_error=exc,
        pycontrails_optional_package="sat",
    )


#: BigQuery table with imagery metadata
BQ_TABLE = "bigquery-public-data.cloud_storage_geo_index.landsat_index"

#: Default columns to include in queries
BQ_DEFAULT_COLUMNS = ["base_url", "sensing_time"]

#: Default spatial extent for queries
BQ_DEFAULT_EXTENT = search.GLOBAL_EXTENT

#: Extra filters for BigQuery queries
BQ_EXTRA_FILTERS = 'AND spacecraft_id in ("LANDSAT_8", "LANDSAT_9")'

#: Default Landsat channels to use if none are specified.
#: These are visible bands for producing a true color composite.
DEFAULT_BANDS = ["B2", "B3", "B4"]

#: Strip this prefix from GCP URLs when caching Landsat files locally
GCP_STRIP_PREFIX = "gs://gcp-public-data-landsat/"


def query(
    start_time: np.datetime64,
    end_time: np.datetime64,
    extent: str | None = None,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Find Landsat 8 and 9 imagery within spatiotemporal region of interest.

    This function requires access to the
    `Google BigQuery API <https://cloud.google.com/bigquery?hl=en>`__
    and uses the `BigQuery python library <https://cloud.google.com/python/docs/reference/bigquery/latest/index.html>`__.

    Parameters
    ----------
    start_time : np.datetime64
        Start of time period for search
    end_time : np.datetime64
        End of time period for search
    extent : str, optional
        Spatial region of interest as a GeoJSON string. If not provided, defaults
        to a global extent.
    columns : list[str], optional.
        Columns to return from Google
        `BigQuery table <https://console.cloud.google.com/bigquery?p=bigquery-public-data&d=cloud_storage_geo_index&t=landsat_index&page=table&_ga=2.90807450.1051800793.1716904050-255800408.1705955196>`__.
        By default, returns imagery base URL and sensing time.

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
    return search.query(BQ_TABLE, roi, columns, BQ_EXTRA_FILTERS)


def intersect(
    flight: Flight,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Find Landsat 8 and 9 imagery intersecting with flight track.

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
        By default, returns imagery base URL and sensing time.

    Returns
    -------
    pd.DataFrame
        Query results in pandas DataFrame

    See Also
    --------
    :func:`search.intersect`
    """
    columns = columns or BQ_DEFAULT_COLUMNS
    return search.intersect(BQ_TABLE, flight, columns, BQ_EXTRA_FILTERS)


class Landsat:
    """Support for Landsat 8 and 9 data handling.

    This class uses the `PROJ <https://proj.org/en/9.4/index.html>`__ coordinate
    transformation software through the
    `pyproj <https://pyproj4.github.io/pyproj/stable/index.html>`__ python interface.
    pyproj is installed as part of the ``sat`` set of optional dependencies
    (``pip install pycontrails[sat]``), but PROJ must be installed manually.

    Parameters
    ----------
    base_url : str
        Base URL of Landsat scene. To find URLs for Landsat scenes at
        specific locations and times, see :func:`query` and :func:`intersect`.
    bands : str | set[str] | None
        Set of bands to retrieve. The 11 possible bands are represented by
        the string "B1" to "B11". For the Google Landsat contrails color scheme,
        set ``bands=("B9", "B10", "B11")``. For the true color scheme, set
        ``bands=("B2", "B3", "B4")``. By default, bands for the true color scheme
        are used. Bands must share a common resolution. The resolutions of each band are:

        - B1-B7, B9: 30 m
        - B8: 15 m
        - B10, B11: 30 m (upsampled from true resolution of 100 m)

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
        bands: str | Iterable[str] | None = None,
        cachestore: cache.CacheStore | None = None,
    ) -> None:
        self.base_url = base_url
        self.bands = _parse_bands(bands)
        _check_band_resolution(self.bands)
        self.fs = gcsfs.GCSFileSystem(token="anon")

        if cachestore is None:
            cache_root = cache._get_user_cache_dir()
            cache_dir = f"{cache_root}/landsat"
            cachestore = cache.DiskCacheStore(cache_dir=cache_dir)
        self.cachestore = cachestore

    def __repr__(self) -> str:
        """Return string representation."""
        return f"Landsat(base_url='{self.base_url}', bands={sorted(self.bands)})"

    @property
    def reflective_bands(self) -> list[str]:
        """List of reflective bands."""
        return [b for b in self.bands if b not in ["B10", "B11"]]

    @property
    def thermal_bands(self) -> list[str]:
        """List of thermal bands."""
        return [b for b in self.bands if b in ["B10", "B11"]]

    def get(
        self, reflective: str = "reflectance", thermal: str = "brightness_temperature"
    ) -> xr.Dataset:
        """Retrieve Landsat imagery.

        Parameters
        ----------
        reflective : str = {"raw", "radiance", "reflectance"}, optional
            Whether to return raw values or rescaled radiances or reflectances for reflective bands.
            By default, return reflectances.
        thermal : str = {"raw", "radiance", "brightness_temperature"}, optional
            Whether to return raw values or rescaled radiances or brightness temperatures
            for thermal bands. By default, return brightness temperatures.

        Returns
        -------
        xr.DataArray
            DataArray of Landsat data.
        """
        if reflective not in ["raw", "radiance", "reflectance"]:
            msg = "reflective band processing must be one of ['raw', 'radiance', 'reflectance']"
            raise ValueError(msg)

        if thermal not in ["raw", "radiance", "brightness_temperature"]:
            msg = (
                "thermal band processing must be one of "
                "['raw', 'radiance', 'brighness_temperature']"
            )
            raise ValueError(msg)

        ds = xr.Dataset()
        for band in self.reflective_bands:
            ds[band] = self._get(band, reflective)
        for band in self.thermal_bands:
            ds[band] = self._get(band, thermal)
        return ds

    def _get(self, band: str, processing: str) -> xr.DataArray:
        """Download Landsat band to the :attr:`cachestore` and return processed data."""
        tiff_path = self._get_tiff(band)
        meta_path = self._get_meta()
        return _read(tiff_path, meta_path, band, processing)

    def _get_tiff(self, band: str) -> str:
        """Download Landsat GeoTIFF imagery and return path to cached file."""
        fs = self.fs
        base_url = self.base_url
        product_id = base_url.split("/")[-1]
        fname = f"{product_id}_{band}.TIF"
        url = f"{base_url}/{fname}"

        sink = self.cachestore.path(url.removeprefix(GCP_STRIP_PREFIX))
        if not self.cachestore.exists(sink):
            fs.get(url, sink)
        return sink

    def _get_meta(self) -> str:
        """Download Landsat metadata file and return path to cached file."""
        fs = self.fs
        base_url = self.base_url
        product_id = base_url.split("/")[-1]
        fname = f"{product_id}_MTL.txt"
        url = f"{base_url}/{fname}"

        sink = self.cachestore.path(url.removeprefix(GCP_STRIP_PREFIX))
        if not self.cachestore.exists(sink):
            fs.get(url, sink)
        return sink


def _parse_bands(bands: str | Iterable[str] | None) -> set[str]:
    """Check that the bands are valid and return as a set."""
    if bands is None:
        return set(DEFAULT_BANDS)

    if isinstance(bands, str):
        bands = (bands,)

    available = {f"B{i}" for i in range(1, 12)}
    bands = {b.upper() for b in bands}
    if len(bands) == 0:
        msg = "At least one band must be provided"
        raise ValueError(msg)
    if not bands.issubset(available):
        msg = f"Bands must be in {sorted(available)}"
        raise ValueError(msg)
    return bands


def _check_band_resolution(bands: set[str]) -> None:
    """Confirm requested bands have a common horizontal resolution.

    All bands have 30 m resolution except the panchromatic band, so
    there are two valid cases: only band 8, or any bands except band 8.
    """
    groups = [
        {"B8"},  # 15 m
        {f"B{i}" for i in range(1, 12) if i != 8},  # 30 m
    ]
    if not any(bands.issubset(group) for group in groups):
        msg = "Bands must have a common horizontal resolution."
        raise ValueError(msg)


def _read(path: str, meta: str, band: str, processing: str) -> xr.DataArray:
    """Read imagery data from Landsat files."""
    try:
        import rasterio
    except ModuleNotFoundError as exc:
        dependencies.raise_module_not_found_error(
            name="landsat module",
            package_name="rasterio",
            module_not_found_error=exc,
            pycontrails_optional_package="sat",
        )

    with rasterio.open(path) as src:
        img = src.read(1)
        crs = pyproj.CRS.from_epsg(src.crs.to_epsg())

    if processing == "reflectance":
        mult, add = _read_band_reflectance_rescaling(meta, band)
        img = np.where(img == 0, np.nan, img * mult + add).astype("float32")
    if processing in ("radiance", "brightness_temperature"):
        mult, add = _read_band_radiance_rescaling(meta, band)
        img = np.where(img == 0, np.nan, img * mult + add).astype("float32")
    if processing == "brightness_temperature":
        k1, k2 = _read_band_thermal_constants(meta, band)
        img = k2 / np.log(k1 / img + 1)

    x, y = _read_image_coordinates(meta, band)

    da = xr.DataArray(
        data=img,
        coords={"y": y, "x": x},
        dims=("y", "x"),
        attrs={
            "long_name": f"{band} {processing.replace('_', ' ')}",
            "units": (
                "W/m^2/sr/um"
                if processing == "radiance"
                else (
                    "nondim"
                    if processing == "reflectance"
                    else "K"
                    if processing == "brightness_temperature"
                    else "none"
                )
            ),
            "crs": crs,
        },
    )
    da["x"].attrs = {"long_name": "easting", "units": "m"}
    da["y"].attrs = {"long_name": "northing", "units": "m"}
    return da


def _read_meta(meta: str, key: str) -> float:
    """Read values from metadata file."""
    with open(meta) as f:
        for line in f:
            if line.strip().startswith(key):
                split = line.split("=")
                return float(split[1].strip())

    msg = f"Could not find {key} in Landsat metadata"
    raise ValueError(msg)


def _read_band_radiance_rescaling(meta: str, band: str) -> tuple[float, float]:
    """Read radiance rescaling factors from metadata file."""
    band = band[1:]  # strip leading B
    mult = _read_meta(meta, f"RADIANCE_MULT_BAND_{band}")
    add = _read_meta(meta, f"RADIANCE_ADD_BAND_{band}")
    return mult, add


def _read_band_reflectance_rescaling(meta: str, band: str) -> tuple[float, float]:
    """Read reflectance rescaling factors from metadata file."""
    band = band[1:]  # strip leading B
    mult = _read_meta(meta, f"REFLECTANCE_MULT_BAND_{band}")
    add = _read_meta(meta, f"REFLECTANCE_ADD_BAND_{band}")
    return mult, add


def _read_band_thermal_constants(meta: str, band: str) -> tuple[float, float]:
    """Read constants for radiance to brightness temperature conversion from metadata file."""
    band = band[1:]  # strip leading B
    k1 = _read_meta(meta, f"K1_CONSTANT_BAND_{band}")
    k2 = _read_meta(meta, f"K2_CONSTANT_BAND_{band}")
    return k1, k2


def _read_image_coordinates(meta: str, band: str) -> tuple[np.ndarray, np.ndarray]:
    """Read image x and y coordinates."""

    # Get coordinates of corners
    ulx = _read_meta(meta, "CORNER_UL_PROJECTION_X_PRODUCT")
    uly = _read_meta(meta, "CORNER_UL_PROJECTION_Y_PRODUCT")
    urx = _read_meta(meta, "CORNER_UR_PROJECTION_X_PRODUCT")
    ury = _read_meta(meta, "CORNER_UR_PROJECTION_Y_PRODUCT")
    llx = _read_meta(meta, "CORNER_LL_PROJECTION_X_PRODUCT")
    lly = _read_meta(meta, "CORNER_LL_PROJECTION_Y_PRODUCT")
    lrx = _read_meta(meta, "CORNER_LR_PROJECTION_X_PRODUCT")
    lry = _read_meta(meta, "CORNER_LR_PROJECTION_Y_PRODUCT")
    if ulx != llx or urx != lrx or uly != ury or lly != lry:
        msg = "Retrieved Landsat image is not aligned with X and Y coordinates"
        raise ValueError(msg)
    xlim = (ulx, urx)
    ylim = (uly, lly)

    # Get size of pixels
    category = (
        "PANCHROMATIC" if band == "B8" else "THERMAL" if band in ("B10", "B11") else "REFLECTIVE"
    )
    pixel_size = _read_meta(meta, f"GRID_CELL_SIZE_{category}")

    # Compute pixel coordinates
    nx = np.round((xlim[1] - xlim[0]) / pixel_size).astype(int) + 1
    ny = np.round((ylim[0] - ylim[1]) / pixel_size).astype(int) + 1
    x = np.linspace(xlim[0], xlim[1], nx)
    y = np.linspace(ylim[0], ylim[1], ny)

    return x, y


def extract_landsat_visualization(
    ds: xr.Dataset, color_scheme: str = "true"
) -> tuple[np.ndarray, pyproj.CRS, tuple[float, float, float, float]]:
    """Extract artifacts for visualizing Landsat data with the given color scheme.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset of Landsat data as returned by :meth:`Landsat.get`.
    color_scheme : str = {"true", "google_contrails"}
        Color scheme to use for visualization. The true color scheme
        requires reflectances for bands B2, B3, and B4; and the
        `Google contrails color scheme <https://research.google/pubs/a-human-labeled-landsat-contrails-dataset>`__
        requires reflectance for band B9 and brightness temperatures for bands B10 and B11.

    Returns
    -------
    rgb : npt.NDArray[np.float32]
        3D RGB array of shape ``(height, width, 3)``.
    src_crs : pyproj.CRS
        Imagery projection
    src_extent : tuple[float,float,float,float]
        Imagery extent in projected coordinates

    References
    ----------
    :cite:`mccloskeyHumanlabeledLandsatContrails2021`
    """

    if color_scheme == "true":
        rgb, src_crs = to_true_color(ds)
    elif color_scheme == "google_contrails":
        rgb, src_crs = to_google_contrails(ds)
    else:
        raise ValueError(f"Color scheme must be 'true' or 'google_contrails', not '{color_scheme}'")

    x = ds["x"].values
    y = ds["y"].values
    src_extent = x.min(), x.max(), y.min(), y.max()

    return rgb, src_crs, src_extent


def to_true_color(ds: xr.Dataset) -> tuple[np.ndarray, pyproj.CRS]:
    """Compute 3d RGB array for the true color scheme.

    Parameters
    ----------
    ds : xr.Dataset
        DataArray of Landsat data with reflectances for bands B2, B3, and B4.

    Returns
    -------
    np.ndarray
        3d RGB array with true color scheme.

    src_crs : pyproj.CRS
        Imagery projection
    """
    red = ds["B4"]
    green = ds["B3"]
    blue = ds["B2"]

    crs = red.attrs["crs"]
    if not (crs.equals(green.attrs["crs"]) and crs.equals(blue.attrs["crs"])):
        msg = "Bands B2, B3, and B4 do not share a common projection."
        raise ValueError(msg)

    if any("reflectance" not in band.attrs["long_name"] for band in (red, green, blue)):
        msg = "Bands B2, B3, and B4 must contain reflectances."
        raise ValueError(msg)

    img = np.dstack(
        [equalize(normalize(band.values), clip_limit=0.03) for band in (red, green, blue)]
    )

    return img, crs


def to_google_contrails(ds: xr.Dataset) -> tuple[np.ndarray, pyproj.CRS]:
    """Compute 3d RGB array for the Google contrails color scheme.

    Parameters
    ----------
    ds : xr.Dataset
        DataArray of Landsat data with reflectance for band B9 and brightness
        temperature for bands B10 and B11.

    Returns
    -------
    np.ndarray
        3d RGB array with Google landsat color scheme.

    src_crs : pyproj.CRS
        Imagery projection

    References
    ----------
    - `Google human-labeled Landsat contrails dataset <https://research.google/pubs/a-human-labeled-landsat-contrails-dataset/>`__
    - :cite:`mccloskeyHumanlabeledLandsatContrails2021`
    """
    rc = ds["B9"]  # cirrus band reflectance
    tb11 = ds["B10"]  # 11 um brightness temperature
    tb12 = ds["B11"]  # 12 um brightness temperature

    crs = rc.attrs["crs"]
    if not (crs.equals(tb11.attrs["crs"]) and crs.equals(tb12.attrs["crs"])):
        msg = "Bands B9, B10, and B11 do not share a common projection."
        raise ValueError(msg)

    if "reflectance" not in rc.attrs["long_name"]:
        msg = "Band B9 must contain reflectance."
        raise ValueError(msg)

    if any("brightness temperature" not in band.attrs["long_name"] for band in (tb11, tb12)):
        msg = "Bands B10 and B11 must contain brightness temperature."
        raise ValueError(msg)

    def adapt(channel: np.ndarray) -> np.ndarray:
        if np.all(np.isclose(channel, 0, atol=1e-3)) or np.all(np.isclose(channel, 1, atol=1e-3)):
            return channel
        return equalize(channel, clip_limit=0.03)

    # red: 12um - 11um brightness temperature difference
    signal = tb12.values - tb11.values
    lower = -5.5
    upper = 1.0
    red = ((signal - lower) / (upper - lower)).clip(0.0, 1.0)

    # green: cirrus band transmittance
    signal = 1 - rc.values
    lower = 0.8
    upper = 1.0
    green = adapt(((signal - lower) / (upper - lower)).clip(0.0, 1.0))

    # blue: 12um brightness temperature
    signal = tb12.values
    lower = 283.0
    upper = 303.0
    blue = adapt(((signal - lower) / (upper - lower)).clip(0.0, 1.0))

    img = np.dstack([red, green, blue])
    return img, crs
