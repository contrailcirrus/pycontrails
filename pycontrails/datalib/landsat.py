"""Support for Landsat 8 and 9 Collection 2 Level-1 imagery.

Scenes are found with the USGS Landsat `STAC server <https://landsatlook.usgs.gov/stac-server>`__.
:class:`Landsat` retrieves imagery from one of two sources:

- ``"usgs"``: the USGS Machine-to-Machine (M2M) API, which is free but requires a USGS
  account with M2M access. See :mod:`pycontrails.datalib.leo_utils.usgs_m2m`.
- ``"gcs"``: a requester-pays Google Cloud Storage bucket ``gs://gee-public-data-landsat``
  listed in the `Earth Engine data catalog
  <https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC08_C02_T1_L2>`__.
  Reading from this bucket is billed to the default project of the Google Cloud credentials,
  which can be set with the ``GOOGLE_CLOUD_PROJECT`` environment variable.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr

from pycontrails.core import Flight, GeoVectorDataset, cache
from pycontrails.datalib.leo_utils import correction, search, usgs_m2m
from pycontrails.datalib.leo_utils import landsat_metadata as lm
from pycontrails.datalib.leo_utils.vis import equalize, normalize
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

try:
    import shapely.geometry
except ModuleNotFoundError as exc:
    dependencies.raise_module_not_found_error(
        name="landsat module",
        package_name="shapely",
        module_not_found_error=exc,
        pycontrails_optional_package="sat",
    )

try:
    import rasterio
except ModuleNotFoundError as exc:
    dependencies.raise_module_not_found_error(
        name="landsat module",
        package_name="rasterio",
        module_not_found_error=exc,
        pycontrails_optional_package="sat",
    )


#: USGS Landsat STAC server
STAC_URL = "https://landsatlook.usgs.gov/stac-server"

#: STAC collection of Landsat Collection 2 Level-1 products
STAC_COLLECTION = "landsat-c2l1"

#: Requester-pays Google Cloud Storage bucket with Landsat Collection 2 files
GCS_BUCKET = "gs://gee-public-data-landsat"

#: Default Landsat channels to use if none are specified.
#: These are visible bands for producing a true color composite.
DEFAULT_BANDS = ["B2", "B3", "B4"]


def query(
    start_time: np.datetime64,
    end_time: np.datetime64,
    extent: str | None = None,
) -> pd.DataFrame:
    """Find Landsat 8 and 9 Collection 2 imagery within spatiotemporal region of interest.

    Parameters
    ----------
    start_time : np.datetime64
        Start of time period for search
    end_time : np.datetime64
        End of time period for search
    extent : str | None, optional
        Spatial region of interest as a GeoJSON string. If not provided, defaults
        to a global extent.

    Returns
    -------
    pd.DataFrame
        Query results in pandas DataFrame
    """
    if extent is None:
        return _search(start_time, end_time, None)

    geometry = shapely.from_geojson(extent)
    scenes = _search(start_time, end_time, search.geometry_bbox(geometry))

    # The STAC search uses the extent's bounding box; keep only scenes that intersect the extent
    filt = shapely.intersects(scenes["geometry"], geometry)
    return scenes[filt].reset_index(drop=True)


def intersect(flight: Flight) -> pd.DataFrame:
    """Find Landsat 8 and 9 Collection 2 imagery intersecting the flight track.

    This function will return all scenes whose footprint includes the flight
    position interpolated to the scene's sensing time.

    Parameters
    ----------
    flight : Flight
        Flight for intersection

    Returns
    -------
    pd.DataFrame
        Query results in pandas DataFrame
    """
    scenes = _search(flight.time_start, flight.time_end, search.vector_bbox(flight))
    return search.intersect_scenes(scenes, flight)


def _search(
    start_time: np.datetime64 | pd.Timestamp,
    end_time: np.datetime64 | pd.Timestamp,
    bbox: tuple[float, float, float, float] | None,
) -> pd.DataFrame:
    """Return Landsat 8 and 9 scenes from the USGS STAC server."""
    items = search.stac_search(STAC_URL, STAC_COLLECTION, start_time, end_time, bbox)
    items = [item for item in items if item["id"].startswith(("LC08", "LC09"))]

    props = [item["properties"] for item in items]

    scenes = pd.DataFrame(
        {
            "product_id": [item["id"] for item in items],
            "scene_id": [p["landsat:scene_id"] for p in props],
            "platform": [p["platform"] for p in props],
            "sensing_time": pd.to_datetime([p["datetime"] for p in props]).tz_localize(None),
            "sun_elevation": [p["view:sun_elevation"] for p in props],
            "geometry": [shapely.geometry.shape(item["geometry"]) for item in items],
        }
    )
    return scenes.sort_values("sensing_time").reset_index(drop=True)


class Landsat:
    """Support for Landsat 8 and 9 Collection 2 Level-1 data handling.

    Parameters
    ----------
    product_id : str
        Landsat product ID, for example ``"LC09_L1TP_095022_20250601_20250602_02_T1"``.
        To find scenes at specific locations and times, see :func:`query` and
        :func:`intersect`.
    bands : str | Iterable[str] | None
        Set of bands to retrieve. The 11 possible bands are represented by
        the string "B1" to "B11". For the Google Landsat contrails color scheme,
        set ``bands=("B9", "B10", "B11")``. For the true color scheme, set
        ``bands=("B2", "B3", "B4")``. By default, bands for the true color scheme
        are used. Bands must share a common resolution. The resolutions of each band are:

        - B1-B7, B9: 30 m
        - B8: 15 m
        - B10, B11: 30 m (upsampled from true resolution of 100 m)

    source : str, optional
        ``"usgs"`` (default) or ``"gcs"``. See the :mod:`pycontrails.datalib.landsat`
        module documentation.
    cachestore : cache.CacheStore | None, optional
        Cache store for Landsat data. If None, a :class:`~pycontrails.DiskCacheStore` is used.

    See Also
    --------
    query
    intersect
    """

    def __init__(
        self,
        product_id: str,
        bands: str | Iterable[str] | None = None,
        source: str = "usgs",
        cachestore: cache.CacheStore | None = None,
    ) -> None:
        if not product_id.startswith(("LC08", "LC09")) or not product_id.endswith(
            ("_02_T1", "_02_T2", "_02_RT")
        ):
            msg = (
                f"'{product_id}' is not a Landsat 8 or 9 Collection 2 Level-1 product ID, "
                "such as 'LC09_L1TP_095022_20250601_20250602_02_T1'."
            )
            raise ValueError(msg)

        if source not in ("usgs", "gcs"):
            msg = f"source must be 'usgs' or 'gcs', not '{source}'"
            raise ValueError(msg)

        self.product_id = product_id
        self.bands = _parse_bands(bands)
        _check_band_resolution(self.bands)
        self.source = source

        if cachestore is None:
            cache_root = cache._get_user_cache_dir()
            cache_dir = f"{cache_root}/landsat"
            cachestore = cache.DiskCacheStore(cache_dir=cache_dir)
        self.cachestore = cachestore

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"Landsat(product_id='{self.product_id}', bands={sorted(self.bands)}, "
            f"source='{self.source}')"
        )

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
        reflective : str, optional
            One of {"raw", "radiance", "reflectance"}.
            Whether to return raw values or rescaled radiances or reflectances for reflective bands.
            By default, return reflectances.
        thermal : str, optional
            One of {"raw", "radiance", "brightness_temperature"}.
            Whether to return raw values or rescaled radiances or brightness temperatures
            for thermal bands. By default, return brightness temperatures.

        Returns
        -------
        xr.Dataset
            Dataset of Landsat data.
        """
        if reflective not in ["raw", "radiance", "reflectance"]:
            msg = "reflective band processing must be one of ['raw', 'radiance', 'reflectance']"
            raise ValueError(msg)

        if thermal not in ["raw", "radiance", "brightness_temperature"]:
            msg = (
                "thermal band processing must be one of "
                "['raw', 'radiance', 'brightness_temperature']"
            )
            raise ValueError(msg)

        paths = self._fetch([f"{band}.TIF" for band in self.bands] + ["MTL.txt"])
        meta = paths["MTL.txt"]

        data = {}
        for band in self.reflective_bands:
            data[band] = _read(paths[f"{band}.TIF"], meta, band, reflective)
        for band in self.thermal_bands:
            data[band] = _read(paths[f"{band}.TIF"], meta, band, thermal)

        return xr.Dataset(data)

    def get_viewing_angle_metadata(self) -> xr.Dataset:
        """Return the viewing zenith (``VZA``) and azimuth (``VAA``) angles in degrees."""
        paths = self._fetch(["VZA.TIF", "VAA.TIF", "MTL.txt"])
        return _read_viewing_angles(paths["VZA.TIF"], paths["VAA.TIF"], paths["MTL.txt"])

    def get_ephemeris(self) -> pd.DataFrame:
        """Return the satellite ephemeris from the angle coefficient (ANG) file."""
        path = self._fetch(["ANG.txt"])["ANG.txt"]
        with open(path) as f:
            return lm.parse_ephemeris_landsat(f.read())

    def get_crs(self) -> pyproj.CRS:
        """Return the CRS of the satellite image."""
        path = self._fetch(["VAA.TIF"])["VAA.TIF"]
        with rasterio.open(path) as src:
            return pyproj.CRS.from_epsg(src.crs.to_epsg())

    def get_sensing_time(self) -> pd.Timestamp:
        """Return the scene center time of the satellite image as a naive UTC timestamp."""
        meta = self._fetch(["MTL.txt"])["MTL.txt"]
        date = _read_meta_str(meta, "DATE_ACQUIRED")
        time = _read_meta_str(meta, "SCENE_CENTER_TIME")
        return pd.Timestamp(f"{date}T{time}").tz_localize(None)

    def colocate(
        self,
        vector: GeoVectorDataset,
        group_by: str | list[str] | None = None,
        variables: list[str] | None = None,
    ) -> pd.DataFrame:
        """Colocate groups of waypoints with satellite image pixels.

        Each group is colocated as in :meth:`colocate_flight`.

        Parameters
        ----------
        vector : GeoVectorDataset
            Waypoints.
        group_by : str | list[str] | None, optional
            Columns of ``vector`` that identify groups, for example ``"flight_id"`` for
            flights or ``["flight_id", "waypoint"]`` for advected waypoints. If None,
            ``vector`` is one group.
        variables : list[str] | None, optional
            Variables of ``vector`` to interpolate to the scan time.

        Returns
        -------
        pd.DataFrame
            One row per group imaged by the satellite, with the ``group_by`` columns, the
            ``x`` and ``y`` coordinates in the satellite image CRS, the scan ``time``, and
            ``longitude``, ``latitude``, ``altitude``, and ``variables`` at that time.
        """
        # Fetch all files in one call so the USGS source logs in to M2M once
        self._fetch(["VZA.TIF", "VAA.TIF", "MTL.txt", "ANG.txt"])

        crs = self.get_crs()
        ds_angles = self.get_viewing_angle_metadata()
        ephemeris = self.get_ephemeris()

        x, y = correction.parallax_correct(vector, crs, ds_angles)
        offset = lm.get_time_delay_detector(ds_angles, ephemeris, crs, x, y)
        scan_time = correction.estimate_scan_time(ephemeris, crs, x, y) + offset
        return correction.colocate(vector, x, y, scan_time, group_by, variables)

    def colocate_flight(self, flight: Flight) -> tuple[float, float, np.datetime64]:
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

        Returns
        -------
        tuple[float, float, np.datetime64]
            A tuple containing the x and y coordinates of the flight position in the
            satellite image CRS, and the corrected sensing_time of the satellite image.
        """
        df = self.colocate(flight)
        if df.empty:
            msg = "Flight track does not overlap with satellite image in time."
            raise ValueError(msg)

        x = df["x"].iloc[0].item()
        y = df["y"].iloc[0].item()
        return x, y, df["time"].iloc[0].to_datetime64()

    # -------------------------------------------------------------------------------------

    def _cache_key(self, suffix: str) -> str:
        return f"{self.product_id}/{self.product_id}_{suffix}"

    def _gcs_url(self, suffix: str) -> str:
        """Return the URL of a product file in the ``gee-public-data-landsat`` bucket."""
        sensor = self.product_id[:4]
        path = self.product_id[10:13]
        row = self.product_id[13:16]
        filename = f"{self.product_id}_{suffix}"
        return f"{GCS_BUCKET}/{sensor}/L1/02/{path}/{row}/{self.product_id}/{filename}"

    def _fetch(self, suffixes: list[str]) -> dict[str, str]:
        """Download files to the cache and return their paths."""
        sinks = {suffix: self.cachestore.path(self._cache_key(suffix)) for suffix in suffixes}
        missing = {s: p for s, p in sinks.items() if not self.cachestore.exists(self._cache_key(s))}
        if not missing:
            return sinks

        if self.source == "usgs":
            usgs_m2m.download(self.product_id, missing)
            return sinks

        fs = gcsfs.GCSFileSystem(requester_pays=True)
        for suffix, sink in missing.items():
            fs.get(self._gcs_url(suffix), sink)
        return sinks


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


def _read_viewing_angles(vza_path: str, vaa_path: str, meta: str) -> xr.Dataset:
    """Read the viewing zenith and azimuth angle bands.

    The bands store hundredths of a degree.
    Pixels outside the scene are 0 in both bands.
    """
    with rasterio.open(vza_path) as src:
        vza = src.read(1)  # int16 dtype
        crs = pyproj.CRS.from_epsg(src.crs.to_epsg())
    with rasterio.open(vaa_path) as src:
        vaa = src.read(1)  # int16 dtype

    fill = (vza == 0) & (vaa == 0)
    x, y = _read_image_coordinates(meta, "B1")

    ds = xr.Dataset()
    for name, img, long_name in (
        ("VZA", vza, "viewing zenith angle"),
        ("VAA", vaa, "viewing azimuth angle"),
    ):
        data = np.where(fill, np.nan, img.astype("float32") / 100.0)
        ds[name] = xr.DataArray(
            data=data,
            coords={"y": y, "x": x},
            dims=("y", "x"),
            attrs={"long_name": long_name, "units": "degrees", "crs": crs},
        )
    ds["x"].attrs = {"long_name": "easting", "units": "m"}
    ds["y"].attrs = {"long_name": "northing", "units": "m"}
    return ds


def _read_meta_str(meta: str, key: str) -> str:
    """Read a value from a Landsat MTL metadata file."""
    with open(meta) as f:
        for line in f:
            if line.strip().startswith(key):
                split = line.split("=")
                return split[1].strip().strip('"')

    msg = f"Could not find {key} in Landsat metadata"
    raise ValueError(msg)


def _read_meta(meta: str, key: str) -> float:
    """Read a numeric value from a Landsat MTL metadata file."""
    return float(_read_meta_str(meta, key))


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
) -> tuple[npt.NDArray[np.float32], pyproj.CRS, tuple[float, float, float, float]]:
    """Extract artifacts for visualizing Landsat data with the given color scheme.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset of Landsat data as returned by :meth:`Landsat.get`.
    color_scheme : str, optional
        One of {"true", "google_contrails"}.
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
    src_extent : tuple[float, float, float, float]
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
    signal = 1.0 - rc.values
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
