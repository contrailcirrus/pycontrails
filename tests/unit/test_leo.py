"""Test low Earth orbit satellite utilites and datalibs."""

import os
from collections.abc import Generator

import geojson
import numpy as np
import pandas as pd
import pyproj
import pytest
import xarray as xr

from pycontrails.core import Flight, cache
from pycontrails.datalib import landsat, sentinel
from pycontrails.datalib._leo_utils import search
from tests import BIGQUERY_ACCESS, OFFLINE

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"
PYCONTRAILS_SKIP_LEO_TESTS = bool(os.getenv("PYCONTRAILS_SKIP_LEO_TESTS"))

# ==========
# Utilities
# ==========


@pytest.mark.parametrize(
    ("start_time", "end_time", "extent", "valid", "match"),
    [
        (
            np.datetime64("2019-01-01 00:00"),
            np.datetime64("2019-01-01 01:00"),
            search.GLOBAL_EXTENT,
            True,
            "",
        ),
        (
            np.datetime64("2019-01-01 02:00"),
            np.datetime64("2019-01-01 01:00"),
            search.GLOBAL_EXTENT,
            False,
            "start_time must be before",
        ),
        (
            np.datetime64("2019-01-01 00:00"),
            np.datetime64("2019-01-01 01:00"),
            "foo",
            False,
            "extent cannot be converted",
        ),
        (
            np.datetime64("2019-01-01 00:00"),
            np.datetime64("2019-01-01 01:00"),
            '{"type": "Point", "coordinates": [0, 0, 0, 0]}',
            False,
            "extent is not valid",
        ),
    ],
)
def test_roi_validation(
    start_time: np.datetime64, end_time: np.datetime64, extent: str, valid: bool, match: str
) -> None:
    """Test region of interest validation."""
    if valid:
        roi = search.ROI(start_time, end_time, extent)
        assert roi.start_time == start_time
        assert roi.end_time == end_time
        assert roi.extent == extent
        return
    with pytest.raises(ValueError, match=match):
        _ = search.ROI(start_time, end_time, extent)


@pytest.mark.parametrize(
    ("flight", "result"),
    [
        (
            # no antimeridian crossing
            Flight(
                longitude=np.array([120, 125, 130, 135, 140]),
                latitude=np.array([-2, -1, 0, 1, 2]),
                altitude=np.full((5,), 10000.0),
                time=pd.date_range("2024-01-01 00:00", "2024-01-01 04:00", freq="1h"),
            ),
            geojson.dumps(
                geojson.MultiLineString(
                    [[(120.0, -2.0), (125.0, -1.0), (130.0, 0.0), (135.0, 1.0), (140.0, 2.0)]]
                )
            ),
        ),
        (
            # single eastward antimeridian crossing
            Flight(
                longitude=np.array([170, 175, 180, -175, -170]),
                latitude=np.array([-2, -1, 0, 1, 2]),
                altitude=np.full((5,), 10000.0),
                time=pd.date_range("2024-01-01 00:00", "2024-01-01 04:00", freq="1h"),
            ),
            geojson.dumps(
                geojson.MultiLineString(
                    [
                        [(170.0, -2.0), (175.0, -1.0), (180.0, 0.0)],
                        [(-180.0, 0.0), (-175.0, 1.0), (-170.0, 2.0)],
                    ]
                )
            ),
        ),
        (
            # single eastward antimeridian crossing with gap filling
            Flight(
                longitude=np.array([170, 175, -175, -170]),
                latitude=np.array([-2, -1, 1, 2]),
                altitude=np.full((4,), 10000.0),
                time=pd.date_range("2024-01-01 00:00", "2024-01-01 03:00", freq="1h"),
            ),
            geojson.dumps(
                geojson.MultiLineString(
                    [
                        [(170.0, -2.0), (175.0, -1.0), (180.0, 0.0)],
                        [(-180.0, 0.0), (-175.0, 1.0), (-170.0, 2.0)],
                    ]
                )
            ),
        ),
        (
            # single westward antimeridian crossing
            Flight(
                longitude=np.array([-170, -175, -180, 175, 170]),
                latitude=np.array([-2, -1, 0, 1, 2]),
                altitude=np.full((5,), 10000.0),
                time=pd.date_range("2024-01-01 00:00", "2024-01-01 04:00", freq="1h"),
            ),
            geojson.dumps(
                geojson.MultiLineString(
                    [
                        [(-170.0, -2.0), (-175.0, -1.0), (-180.0, 0.0)],
                        [(180.0, 0.0), (175.0, 1.0), (170.0, 2.0)],
                    ]
                )
            ),
        ),
        (
            # single westward antimeridian crossing with gap filling
            Flight(
                longitude=np.array([-170, -175, 175, 170]),
                latitude=np.array([-2, -1, 1, 2]),
                altitude=np.full((4,), 10000.0),
                time=pd.date_range("2024-01-01 00:00", "2024-01-01 03:00", freq="1h"),
            ),
            geojson.dumps(
                geojson.MultiLineString(
                    [
                        [(-170.0, -2.0), (-175.0, -1.0), (-180.0, 0.0)],
                        [(180.0, 0.0), (175.0, 1.0), (170.0, 2.0)],
                    ]
                )
            ),
        ),
        (
            # multiple antimeridian crossings
            Flight(
                longitude=np.array(
                    [170, 175, 180, -175, -170, -175, -180, 175, 170, 175, 180, -175, -170]
                ),
                latitude=np.array([-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6]),
                altitude=np.full((13,), 10000.0),
                time=pd.date_range("2024-01-01 00:00", "2024-01-01 12:00", freq="1h"),
            ),
            geojson.dumps(
                geojson.MultiLineString(
                    [
                        [(170.0, -6.0), (175.0, -5.0), (180.0, -4.0)],
                        [
                            (-180.0, -4.0),
                            (-175.0, -3.0),
                            (-170.0, -2.0),
                            (-175.0, -1.0),
                            (-180.0, 0.0),
                        ],
                        [(180.0, 0.0), (175.0, 1.0), (170.0, 2.0), (175.0, 3.0), (180.0, 4.0)],
                        [(-180.0, 4.0), (-175.0, 5.0), (-170.0, 6.0)],
                    ]
                )
            ),
        ),
        (
            # multiple antimeridian crossings with gap filling
            Flight(
                longitude=np.array([170, 175, -175, -170, -175, 175, 170, 175, -175, -170]),
                latitude=np.array([-6, -5, -3, -2, -1, 1, 2, 3, 5, 6]),
                altitude=np.full((10,), 10000.0),
                time=pd.date_range("2024-01-01 00:00", "2024-01-01 09:00", freq="1h"),
            ),
            geojson.dumps(
                geojson.MultiLineString(
                    [
                        [(170.0, -6.0), (175.0, -5.0), (180.0, -4.0)],
                        [
                            (-180.0, -4.0),
                            (-175.0, -3.0),
                            (-170.0, -2.0),
                            (-175.0, -1.0),
                            (-180.0, 0.0),
                        ],
                        [(180.0, 0.0), (175.0, 1.0), (170.0, 2.0), (175.0, 3.0), (180.0, 4.0)],
                        [(-180.0, 4.0), (-175.0, 5.0), (-170.0, 6.0)],
                    ]
                )
            ),
        ),
        (
            # multiple antimeridian crossings with uneven gaps
            Flight(
                longitude=np.array([170, -175, -170, -175, 170, -170]),
                latitude=np.array([-6, -3, -2, -1, 2, 6]),
                altitude=np.full((6,), 10000.0),
                time=pd.date_range("2024-01-01 00:00", "2024-01-01 05:00", freq="1h"),
            ),
            geojson.dumps(
                geojson.MultiLineString(
                    [
                        [(170.0, -6.0), (180.0, -4.0)],
                        [
                            (-180.0, -4.0),
                            (-175.0, -3.0),
                            (-170.0, -2.0),
                            (-175.0, -1.0),
                            (-180.0, 0.0),
                        ],
                        [(180.0, 0.0), (170.0, 2.0), (180.0, 4.0)],
                        [(-180.0, 4.0), (-170.0, 6.0)],
                    ]
                )
            ),
        ),
    ],
)
def test_track_to_geojson(flight: Flight, result: str) -> None:
    """Test GeoJSON string creation with antimeridian splitting."""
    assert search.track_to_geojson(flight) == result


# ========
# Landsat
# ========


@pytest.fixture()
def landsat_base_url() -> str:
    """Landsat base URL for tests."""
    return "gs://gcp-public-data-landsat/LC08/01/095/055/LC08_L1TP_095055_20190101_20190130_01_T1"


@pytest.fixture()
def landsat_cachestore() -> Generator[cache.DiskCacheStore, None, None]:
    """Clearable cache for Landsat data."""
    cache_root = cache._get_user_cache_dir()
    cache_dir = f"{cache_root}/landsat-unit-test"
    cachestore = cache.DiskCacheStore(cache_dir=cache_dir, allow_clear=True)
    yield cachestore
    cachestore.clear()


@pytest.mark.skipif(not BIGQUERY_ACCESS, reason="No BigQuery access")
@pytest.mark.skipif(OFFLINE, reason="offline")
def test_landsat_empty_query() -> None:
    """Test Landsat imagery query that returns an empty result."""
    start_time = np.datetime64("2019-01-01 00:00:00")
    end_time = np.datetime64("2019-01-01 00:15:00")
    extent = geojson.dumps(geojson.Point((0, 0)))

    df = landsat.query(start_time, end_time, extent)
    assert set(df.columns) == {"base_url", "sensing_time"}
    assert len(df) == 0


@pytest.mark.skipif(not BIGQUERY_ACCESS, reason="No BigQuery access")
@pytest.mark.skipif(OFFLINE, reason="offline")
def test_landsat_query(landsat_base_url: str) -> None:
    """Test Landsat imagery query that returns a non-empty result."""
    start_time = np.datetime64("2019-01-01 00:00:00")
    end_time = np.datetime64("2019-01-01 00:15:00")
    extent = geojson.dumps(geojson.Point((151, 7)))

    df = landsat.query(start_time, end_time, extent)
    assert set(df.columns) == {"base_url", "sensing_time"}
    assert len(df) == 1
    assert df["base_url"].item() == landsat_base_url


@pytest.mark.skipif(not BIGQUERY_ACCESS, reason="No BigQuery access")
@pytest.mark.skipif(OFFLINE, reason="offline")
def test_landsat_empty_intersection() -> None:
    """Test Landsat flight intersection that return an empty result."""
    flight = Flight(
        longitude=[0, 1],
        latitude=[0, 1],
        altitude=[10_000, 10_000],
        time=[np.datetime64("2019-01-01 00:00:00"), np.datetime64("2019-01-01 00:15:00")],
    )

    df = landsat.intersect(flight)
    assert set(df.columns) == {"base_url", "sensing_time"}
    assert len(df) == 0


@pytest.mark.skipif(not BIGQUERY_ACCESS, reason="No BigQuery access")
@pytest.mark.skipif(OFFLINE, reason="offline")
def test_landsat_intersection(landsat_base_url: str) -> None:
    """Test Landsat flight intersection that return an non-empty result."""
    flight = Flight(
        longitude=[151, 151.1],
        latitude=[7, 7.1],
        altitude=[10_000, 10_000],
        time=[np.datetime64("2019-01-01 00:00:00"), np.datetime64("2019-01-01 00:15:00")],
    )

    df = landsat.intersect(flight)
    assert set(df.columns) == {"base_url", "sensing_time"}
    assert len(df) == 1
    assert df["base_url"].item() == landsat_base_url


@pytest.mark.parametrize(
    ("bands", "succeed"),
    [
        (
            {
                "B8",
            },
            True,
        ),
        ({"B2", "B3", "B4"}, True),
        ({"B9", "B10", "B11"}, True),
        ({"B1", "B2", "B3", "B9"}, True),
        ({"B2", "B8"}, False),
        ({"B8", "B10", "B11"}, False),
    ],
)
def test_landsat_band_resolution(bands: set[str], succeed: bool) -> None:
    """Test Landsat validation that bands share a common resolution."""
    if succeed:
        assert landsat._check_band_resolution(bands) is None
        return
    with pytest.raises(ValueError, match="Bands must have a common horizontal resolution"):
        landsat._check_band_resolution(bands)


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="data retrieval tests skipped in GitHub actions")
@pytest.mark.skipif(PYCONTRAILS_SKIP_LEO_TESTS, reason="PYCONTRAILS_SKIP_LEO_TESTS set")
@pytest.mark.skipif(OFFLINE, reason="offline")
@pytest.mark.parametrize("band", [f"B{i}" for i in range(1, 9)])
@pytest.mark.parametrize("processing", ["raw", "radiance", "reflectance"])
def test_landsat_get_reflective_bands(
    landsat_cachestore: cache.DiskCacheStore,
    landsat_base_url: str,
    band: str,
    processing: str,
) -> None:
    """Test downloading and processing of reflective bands."""
    downloader = landsat.Landsat(landsat_base_url, cachestore=landsat_cachestore, bands=[band])
    ds = downloader.get(reflective=processing)

    assert isinstance(ds, xr.Dataset)
    assert band in ds
    da = ds[band]
    assert isinstance(da, xr.DataArray)
    assert da.ndim == 2
    assert da.name == band
    assert da.dims == ("y", "x")
    assert da.dtype == "uint16" if processing == "raw" else "float32"
    assert da.attrs["units"] == (
        "none" if processing == "raw" else "W/m^2/sr/um" if processing == "radiance" else "nondim"
    )

    ds.close()


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="data retrieval tests skipped in GitHub actions")
@pytest.mark.skipif(PYCONTRAILS_SKIP_LEO_TESTS, reason="PYCONTRAILS_SKIP_LEO_TESTS set")
@pytest.mark.skipif(OFFLINE, reason="offline")
@pytest.mark.parametrize("band", ["B10", "B11"])
@pytest.mark.parametrize("processing", ["raw", "radiance", "brightness_temperature"])
def test_landsat_get_thermal_bands(
    landsat_cachestore: cache.DiskCacheStore,
    landsat_base_url: str,
    band: str,
    processing: str,
) -> None:
    """Test downloading and processing of thermal bands."""
    downloader = landsat.Landsat(landsat_base_url, cachestore=landsat_cachestore, bands=[band])
    ds = downloader.get(thermal=processing)

    assert isinstance(ds, xr.Dataset)
    assert band in ds
    da = ds[band]
    assert isinstance(da, xr.DataArray)
    assert da.ndim == 2
    assert da.name == band
    assert da.dims == ("y", "x")
    assert da.dtype == "uint16" if processing == "raw" else "float32"
    assert da.attrs["units"] == (
        "none" if processing == "raw" else "W/m^2/sr/um" if processing == "radiance" else "K"
    )

    ds.close()


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="data retrieval tests skipped in GitHub actions")
@pytest.mark.skipif(PYCONTRAILS_SKIP_LEO_TESTS, reason="PYCONTRAILS_SKIP_LEO_TESTS set")
@pytest.mark.skipif(OFFLINE, reason="offline")
def test_landsat_generate_true_color_rgb(
    landsat_cachestore: cache.DiskCacheStore, landsat_base_url: str
) -> None:
    """Test true color RGB generation."""
    downloader = landsat.Landsat(landsat_base_url, cachestore=landsat_cachestore)
    ds = downloader.get()

    rgb, crs, extent = landsat.extract_landsat_visualization(ds, color_scheme="true")
    assert isinstance(rgb, np.ndarray)
    assert rgb.shape == (7761, 7591, 3)
    assert np.isnan(rgb).sum() == 52449498
    assert rgb.dtype == "float32"
    assert np.nanmin(rgb) == 0.0
    assert np.nanmax(rgb) == 1.0
    assert isinstance(crs, pyproj.CRS)
    assert crs.to_epsg() == 32656
    assert extent == (247200.0, 474900.0, 682800.0, 915600.0)

    ds.close()


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="data retrieval tests skipped in GitHub actions")
@pytest.mark.skipif(PYCONTRAILS_SKIP_LEO_TESTS, reason="PYCONTRAILS_SKIP_LEO_TESTS set")
@pytest.mark.skipif(OFFLINE, reason="offline")
def test_landsat_generate_google_contrails_rgb(
    landsat_cachestore: cache.DiskCacheStore, landsat_base_url: str
) -> None:
    """Test Google contrails RGB generation."""
    downloader = landsat.Landsat(
        landsat_base_url, cachestore=landsat_cachestore, bands=["B9", "B10", "B11"]
    )
    ds = downloader.get()

    rgb, crs, extent = landsat.extract_landsat_visualization(ds, color_scheme="google_contrails")
    assert isinstance(rgb, np.ndarray)
    assert rgb.shape == (7761, 7591, 3)
    assert np.isnan(rgb).sum() == 54275245
    assert rgb.dtype == "float32"
    assert np.nanmin(rgb) == 0.0
    assert np.nanmax(rgb) == 1.0
    assert isinstance(crs, pyproj.CRS)
    assert crs.to_epsg() == 32656
    assert extent == (247200.0, 474900.0, 682800.0, 915600.0)

    ds.close()


# ===================
# Sentinel-2 datalib
# ===================


@pytest.fixture()
def sentinel_base_url() -> str:
    """Sentinel base URL for tests."""
    return "gs://gcp-public-data-sentinel-2/tiles/59/V/ND/S2A_MSIL1C_20190101T000001_N0207_R116_T59VND_20190101T013420.SAFE"


@pytest.fixture()
def sentinel_granule_id() -> str:
    """Sentinel granule ID for tests."""
    return "L1C_T59VND_A018417_20181231T235957"


@pytest.fixture()
def sentinel_cachestore() -> Generator[cache.DiskCacheStore, None, None]:
    """Clearable cache for Sentinel data."""
    cache_root = cache._get_user_cache_dir()
    cache_dir = f"{cache_root}/sentinel-unit-test"
    cachestore = cache.DiskCacheStore(cache_dir=cache_dir, allow_clear=True)
    yield cachestore
    cachestore.clear()


@pytest.mark.skipif(not BIGQUERY_ACCESS, reason="No BigQuery access")
@pytest.mark.skipif(OFFLINE, reason="offline")
def test_sentinel_empty_query() -> None:
    """Test Sentinel imagery query that returns an empty result."""
    start_time = np.datetime64("2019-01-01 00:00:00")
    end_time = np.datetime64("2019-01-01 00:01:00")
    extent = geojson.dumps(geojson.Point((0, 0)))

    df = sentinel.query(start_time, end_time, extent)
    assert set(df.columns) == {"base_url", "granule_id", "sensing_time"}
    assert len(df) == 0


@pytest.mark.skipif(not BIGQUERY_ACCESS, reason="No BigQuery access")
@pytest.mark.skipif(OFFLINE, reason="offline")
def test_sentinel_query(sentinel_base_url: str, sentinel_granule_id: str) -> None:
    """Test Sentinel imagery query that returns a non-empty result."""
    start_time = np.datetime64("2019-01-01 00:00:00")
    end_time = np.datetime64("2019-01-01 00:01:00")
    extent = geojson.dumps(geojson.Point((171, 57)))

    df = sentinel.query(start_time, end_time, extent)
    assert set(df.columns) == {"base_url", "granule_id", "sensing_time"}
    assert len(df) == 2
    assert df["base_url"][0] == sentinel_base_url
    assert df["granule_id"][0] == sentinel_granule_id


@pytest.mark.skipif(not BIGQUERY_ACCESS, reason="No BigQuery access")
@pytest.mark.skipif(OFFLINE, reason="offline")
def test_sentinel_empty_intersection() -> None:
    """Test Sentinel flight intersection that return an empty result."""
    flight = Flight(
        longitude=[0, 1],
        latitude=[0, 1],
        altitude=[10_000, 10_000],
        time=[np.datetime64("2019-01-01 00:00:00"), np.datetime64("2019-01-01 00:01:00")],
    )

    df = sentinel.intersect(flight)
    assert set(df.columns) == {"base_url", "granule_id", "sensing_time"}
    assert len(df) == 0


@pytest.mark.skipif(not BIGQUERY_ACCESS, reason="No BigQuery access")
@pytest.mark.skipif(OFFLINE, reason="offline")
def test_sentinel_intersection(sentinel_base_url: str, sentinel_granule_id: str) -> None:
    """Test Sentinel flight intersection that return an non-empty result."""
    flight = Flight(
        longitude=[171, 171.1],
        latitude=[57, 57.1],
        altitude=[10_000, 10_000],
        time=[np.datetime64("2019-01-01 00:00:00"), np.datetime64("2019-01-01 00:01:00")],
    )

    df = sentinel.intersect(flight)
    assert set(df.columns) == {"base_url", "granule_id", "sensing_time"}
    assert len(df) == 2
    assert df["base_url"][0] == sentinel_base_url
    assert df["granule_id"][0] == sentinel_granule_id


@pytest.mark.parametrize(
    ("bands", "succeed"),
    [
        (
            {
                "B02",
                "B03",
                "B04",
                "B08",
            },
            True,
        ),
        ({"B05", "B06", "B07", "B8A", "B11", "B12"}, True),
        ({"B01", "B09", "B10"}, True),
        ({"B02", "B05"}, False),
        ({"B03", "B09"}, False),
        ({"B8A", "B10"}, False),
        ({"B02", "B08", "B10", "B12"}, False),
    ],
)
def test_sentinel_band_resolution(bands: set[str], succeed: bool) -> None:
    """Test Sentinel validation that bands share a common resolution."""
    if succeed:
        assert sentinel._check_band_resolution(bands) is None
        return
    with pytest.raises(ValueError, match="Bands must have a common horizontal resolution"):
        sentinel._check_band_resolution(bands)


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="data retrieval tests skipped in GitHub actions")
@pytest.mark.skipif(PYCONTRAILS_SKIP_LEO_TESTS, reason="PYCONTRAILS_SKIP_LEO_TESTS set")
@pytest.mark.skipif(OFFLINE, reason="offline")
@pytest.mark.parametrize("band", [f"B{i:02d}" for i in range(1, 13)] + ["B8A"])
@pytest.mark.parametrize("processing", ["raw", "reflectance"])
def test_sentinel_get_reflective_bands(
    sentinel_cachestore: cache.DiskCacheStore,
    sentinel_base_url: str,
    sentinel_granule_id: str,
    band: str,
    processing: str,
) -> None:
    """Test downloading and processing of reflective bands."""
    downloader = sentinel.Sentinel(
        sentinel_base_url, sentinel_granule_id, cachestore=sentinel_cachestore, bands=[band]
    )
    ds = downloader.get(reflective=processing)

    assert isinstance(ds, xr.Dataset)
    assert band in ds
    da = ds[band]
    assert isinstance(da, xr.DataArray)
    assert da.ndim == 2
    assert da.name == band
    assert da.dims == ("y", "x")
    assert da.dtype == "uint16" if processing == "raw" else "float32"
    assert da.attrs["units"] == "none" if processing == "raw" else "nondim"

    ds.close()


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="data retrieval tests skipped in GitHub actions")
@pytest.mark.skipif(PYCONTRAILS_SKIP_LEO_TESTS, reason="PYCONTRAILS_SKIP_LEO_TESTS set")
@pytest.mark.skipif(OFFLINE, reason="offline")
def test_sentinel_generate_true_color_rgb(
    sentinel_cachestore: cache.DiskCacheStore, sentinel_base_url: str, sentinel_granule_id: str
) -> None:
    """Test true color RGB generation."""
    downloader = sentinel.Sentinel(
        sentinel_base_url, sentinel_granule_id, cachestore=sentinel_cachestore
    )
    ds = downloader.get()

    rgb, crs, extent = sentinel.extract_sentinel_visualization(ds, color_scheme="true")
    assert isinstance(rgb, np.ndarray)
    assert rgb.shape == (10980, 10980, 3)
    assert np.isnan(rgb).sum() == 0
    assert rgb.dtype == "float32"
    assert rgb.min() == 0.0
    assert rgb.max() == 1.0
    assert isinstance(crs, pyproj.CRS)
    assert crs.to_epsg() == 32659
    assert extent == (499980.0, 609770.0, 6290230.0, 6400020.0)

    ds.close()
