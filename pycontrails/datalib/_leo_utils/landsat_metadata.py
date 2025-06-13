"""Download and parse Landsat metadata from USGS.

This modules requires `GeoPandas <https://geopandas.org/>`_.
"""

import geopandas as gpd
import pandas as pd
import shapely

from pycontrails.core import cache


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

    return gpd.GeoDataFrame(
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
    cachestore = cachestore or cache.DiskCacheStore()

    cache_key = "LANDSAT_OT_C2_L1.pq"
    if cachestore.exists(cache_key) and not update_cache:
        return gpd.read_parquet(cachestore.path(cache_key))

    df = _download_landsat_metadata()
    gdf = _landsat_metadata_to_geodataframe(df)
    gdf.to_parquet(cachestore.path(cache_key), index=False)
    return gdf
