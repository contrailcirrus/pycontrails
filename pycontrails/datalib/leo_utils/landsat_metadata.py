"""Download and parse Landsat metadata from USGS.

This modules requires `GeoPandas <https://geopandas.org/>`_.
"""

import re

import numpy as np
import numpy.typing as npt
import pandas as pd
import pyproj
import xarray as xr

from pycontrails.core import cache
from pycontrails.datalib.leo_utils import correction
from pycontrails.utils import dependencies

try:
    import geopandas as gpd
except ModuleNotFoundError as exc:
    dependencies.raise_module_not_found_error(
        name="landsat_metadata module",
        package_name="geopandas",
        module_not_found_error=exc,
        pycontrails_optional_package="sat",
    )

try:
    import shapely
except ModuleNotFoundError as exc:
    dependencies.raise_module_not_found_error(
        name="landsat_metadata module",
        package_name="shapely",
        module_not_found_error=exc,
        pycontrails_optional_package="sat",
    )


def _split_antimeridian(polygon: shapely.Polygon) -> shapely.MultiPolygon:
    """Split a polygon into two polygons at the antimeridian.

    This implementation assumes that the passed polygon is actually situated
    on the antimeridian and does not simultaneously cross the meridian.
    """
    # Shift the x-coordinates of the polygon to the right
    # The `valid_poly` will not be valid if the polygon spans the meridian
    valid_poly = shapely.ops.transform(lambda x, y: (x if x >= 0.0 else x + 360.0, y), polygon)
    if not valid_poly.is_valid:
        raise ValueError("Invalid polygon before splitting at the antimeridian.")

    eastern_hemi = shapely.geometry.box(0.0, -90.0, 180.0, 90.0)
    western_hemi = shapely.geometry.box(180.0, -90.0, 360.0, 90.0)

    western_poly = valid_poly.intersection(western_hemi)
    western_poly = shapely.ops.transform(lambda x, y: (x - 360.0, y), western_poly)  # shift back
    eastern_poly = valid_poly.intersection(eastern_hemi)

    if not western_poly.is_valid or not eastern_poly.is_valid:
        raise ValueError("Invalid polygon after splitting at the antimeridian.")

    return shapely.MultiPolygon([western_poly, eastern_poly])


def _download_landsat_metadata() -> pd.DataFrame:
    """Download and parse the Landsat metadata CSV file from USGS.

    See `the USGS documentation <https://www.usgs.gov/landsat-missions/landsat-collection-2-metadata>`_
    for more details.
    """
    p = "https://landsat.usgs.gov/landsat/metadata_service/bulk_metadata_files/LANDSAT_OT_C2_L1.csv.gz"

    usecols = [
        "Display ID",
        "Ordering ID",
        "Collection Category",
        "Start Time",
        "Stop Time",
        "Day/Night Indicator",
        "Satellite",
        "Corner Upper Left Latitude",
        "Corner Upper Left Longitude",
        "Corner Upper Right Latitude",
        "Corner Upper Right Longitude",
        "Corner Lower Left Latitude",
        "Corner Lower Left Longitude",
        "Corner Lower Right Latitude",
        "Corner Lower Right Longitude",
    ]

    df = pd.read_csv(p, compression="gzip", usecols=usecols)

    # Convert column dtypes
    df["Start Time"] = pd.to_datetime(df["Start Time"], format="ISO8601")
    df["Stop Time"] = pd.to_datetime(df["Stop Time"], format="ISO8601")
    df["Display ID"] = df["Display ID"].astype("string[pyarrow]")
    df["Ordering ID"] = df["Ordering ID"].astype("string[pyarrow]")
    df["Collection Category"] = df["Collection Category"].astype("string[pyarrow]")
    df["Day/Night Indicator"] = df["Day/Night Indicator"].astype("string[pyarrow]")

    return df


def _landsat_metadata_to_geodataframe(df: pd.DataFrame) -> gpd.GeoDataFrame:
    """Convert Landsat metadata DataFrame to GeoDataFrame with polygons."""
    polys = shapely.polygons(
        df[
            [
                "Corner Upper Left Longitude",
                "Corner Upper Left Latitude",
                "Corner Upper Right Longitude",
                "Corner Upper Right Latitude",
                "Corner Lower Right Longitude",
                "Corner Lower Right Latitude",
                "Corner Lower Left Longitude",
                "Corner Lower Left Latitude",
                "Corner Upper Left Longitude",
                "Corner Upper Left Latitude",
            ]
        ]
        .to_numpy()
        .reshape(-1, 5, 2)
    )

    out = gpd.GeoDataFrame(
        df,
        geometry=polys,
        crs="EPSG:4326",
        columns=[
            "Display ID",
            "Ordering ID",
            "Collection Category",
            "Start Time",
            "Stop Time",
            "Day/Night Indicator",
            "Satellite",
            "geometry",
        ],
    )

    # Split polygons that cross the antimeridian
    invalid = ~out.is_valid
    out.loc[invalid, "geometry"] = out.loc[invalid, "geometry"].apply(_split_antimeridian)
    return out


def open_landsat_metadata(
    cachestore: cache.CacheStore | None = None, update_cache: bool = False
) -> gpd.GeoDataFrame:
    """Download and parse the Landsat metadata CSV file from USGS.

    By default, the metadata is cached in a disk cache store.

    Parameters
    ----------
    cachestore : cache.CacheStore | None, optional
        Cache store for Landsat metadata.
        Defaults to :class:`cache.DiskCacheStore`.
    update_cache : bool, optional
        Force update to cached Landsat metadata. The remote file is updated
        daily, so this is useful to ensure you have the latest metadata.

    Returns
    -------
    gpd.GeoDataFrame
        Processed Landsat metadata. The ``geometry`` column contains polygons
        representing the footprints of the Landsat scenes.
    """
    if cachestore is None:
        cache_root = cache._get_user_cache_dir()
        cache_dir = f"{cache_root}/landsat_metadata"
        cachestore = cache.DiskCacheStore(cache_dir=cache_dir)

    cache_key = "LANDSAT_OT_C2_L1.pq"
    if cachestore.exists(cache_key) and not update_cache:
        return gpd.read_parquet(cachestore.path(cache_key))

    df = _download_landsat_metadata()
    gdf = _landsat_metadata_to_geodataframe(df)
    gdf.to_parquet(cachestore.path(cache_key), index=False)
    return gdf


def parse_ephemeris_landsat(ang_content: str) -> pd.DataFrame:
    """Find the EPHEMERIS group in a ANG text file and extract the data arrays.

    Parameters
    ----------
    ang_content : str
        The content of the ANG file as a string.

    Returns
    -------
    pd.DataFrame
        A :class:`pandas.DataFrame` containing the ephemeris track with columns:
        - EPHEMERIS_TIME: Timestamps of the ephemeris data.
        - EPHEMERIS_ECEF_X: ECEF X coordinates.
        - EPHEMERIS_ECEF_Y: ECEF Y coordinates.
        - EPHEMERIS_ECEF_Z: ECEF Z coordinates.
    """

    # Find GROUP = EPHEMERIS, capture everything non-greedily (.*?) until END_GROUP = EPHEMERIS
    pattern = r"GROUP\s*=\s*EPHEMERIS\s*(.*?)\s*END_GROUP\s*=\s*EPHEMERIS"
    match = re.search(pattern, ang_content, flags=re.DOTALL)
    if match is None:
        raise ValueError("No data found for EPHEMERIS group in the ANG content.")
    ephemeris_content = match.group(1)

    pattern = r"EPHEMERIS_EPOCH_YEAR\s*=\s*(\d+)"
    match = re.search(pattern, ephemeris_content)
    if match is None:
        raise ValueError("No data found for EPHEMERIS_EPOCH_YEAR in the ANG content.")
    year = int(match.group(1))

    pattern = r"EPHEMERIS_EPOCH_DAY\s*=\s*(\d+)"
    match = re.search(pattern, ephemeris_content)
    if match is None:
        raise ValueError("No data found for EPHEMERIS_EPOCH_DAY in the ANG content.")
    day = int(match.group(1))

    pattern = r"EPHEMERIS_EPOCH_SECONDS\s*=\s*(\d+\.\d+)"
    match = re.search(pattern, ephemeris_content)
    if match is None:
        raise ValueError("No data found for EPHEMERIS_EPOCH_SECONDS in the ANG content.")
    seconds = float(match.group(1))

    t0 = (
        pd.Timestamp(year=year, month=1, day=1)
        + pd.Timedelta(days=day - 1)
        + pd.Timedelta(seconds=seconds)
    )

    # Find all the EPHEMERIS_* arrays
    array_patterns = {
        "EPHEMERIS_TIME": r"EPHEMERIS_TIME\s*=\s*\((.*?)\)",
        "EPHEMERIS_ECEF_X": r"EPHEMERIS_ECEF_X\s*=\s*\((.*?)\)",
        "EPHEMERIS_ECEF_Y": r"EPHEMERIS_ECEF_Y\s*=\s*\((.*?)\)",
        "EPHEMERIS_ECEF_Z": r"EPHEMERIS_ECEF_Z\s*=\s*\((.*?)\)",
    }

    arrays = {}
    for key, pattern in array_patterns.items():
        match = re.search(pattern, ephemeris_content, flags=re.DOTALL)
        if match is None:
            raise ValueError(f"No data found for {key} in the ANG content.")
        data_str = match.group(1)

        data_list = [float(x.strip()) for x in data_str.split(",")]
        if key == "EPHEMERIS_TIME":
            data_list = [t0 + pd.Timedelta(seconds=t) for t in data_list]
        arrays[key] = data_list

    return pd.DataFrame(arrays)


def get_time_delay_detector(
    ds: xr.Dataset,
    ephemeris: pd.DataFrame,
    utm_crs: pyproj.CRS,
    x: npt.NDArray[np.floating],
    y: npt.NDArray[np.floating],
) -> npt.NDArray[np.timedelta64]:
    """Return the detector time delay at the given (x, y) coordinates.

    Parameters
    ----------
    ds : xr.Dataset
        The Landsat dataset containing the VAA variable.
    ephemeris : pd.DataFrame
        The ephemeris DataFrame containing the EPHEMERIS_TIME and ECEF coordinates.
    utm_crs : pyproj.CRS
        The UTM coordinate reference system for the Landsat scene.
    x : npt.NDArray[np.floating]
        The x-coordinates of the pixels in the dataset's coordinate system.
    y : npt.NDArray[np.floating]
        The y-coordinates of the pixels in the dataset's coordinate system.

    Returns
    -------
    npt.NDArray[np.timedelta64]
        The time delay for each (x, y) coordinate as a timedelta64 array.

    """
    x, y = np.atleast_1d(x, y)

    ephemeris_utm = correction.ephemeris_ecef_to_utm(ephemeris, utm_crs)
    eph_angle_radians = -np.arctan2(ephemeris_utm["y"].diff(), ephemeris_utm["x"].diff())
    avg_eph_angle = (eph_angle_radians * 180.0 / np.pi).mean()

    vaa = ds["VAA"].interp(x=xr.DataArray(x, dims="points"), y=xr.DataArray(y, dims="points"))

    is_odd = np.isfinite(vaa) & ((vaa > avg_eph_angle) | (vaa < avg_eph_angle - 180.0))
    is_even = np.isfinite(vaa) & ~is_odd

    out = np.full(x.shape, fill_value=np.timedelta64("NaT", "ns"), dtype="timedelta64[ns]")
    # We use an offset of +/- 2 seconds as a very rough estimate of the time delay
    # This may only be accurate up to 1 second, but it's better than nothing
    out[is_even] = np.timedelta64(-2000000000, "ns")  # -2 seconds
    out[is_odd] = np.timedelta64(2000000000, "ns")  # 2 seconds

    return out
