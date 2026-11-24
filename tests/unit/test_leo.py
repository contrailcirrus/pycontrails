"""Test low Earth orbit satellite utilites and datalibs."""

import pathlib
from collections.abc import Generator

import numpy as np
import pandas as pd
import pyproj
import pytest
import shapely
import xarray as xr

from pycontrails.core import Flight, GeoVectorDataset, cache
from pycontrails.datalib import landsat, sentinel
from pycontrails.datalib.leo_utils import correction, search, usgs_m2m
from tests import USGS_M2M_CREDENTIALS
from tests.unit import get_static_path

# ==========
# Utilities
# ==========


def _scenes(footprints: list[shapely.Geometry], times: list[str]) -> pd.DataFrame:
    """Build a synthetic scene table in place of STAC search results."""
    return pd.DataFrame(
        {
            "scene": [f"scene{i}" for i in range(len(footprints))],
            "sensing_time": pd.to_datetime(times).astype("datetime64[ns]"),
            "geometry": footprints,
        }
    )


def test_intersect_scenes_flight() -> None:
    """Test that a flight intersects the scene containing its position at the scene time."""
    scenes = _scenes(
        [shapely.box(0, 0, 1, 1), shapely.box(1, 0, 2, 1), shapely.box(0, 0, 2, 1)],
        ["2024-01-01 00:30", "2024-01-01 00:30", "2024-01-01 02:00"],
    )
    flight = Flight(
        longitude=[0.0, 3.0],
        latitude=[0.5, 0.5],
        altitude=[10_000, 10_000],
        time=pd.to_datetime(["2024-01-01 00:00", "2024-01-01 01:00"]),
    )
    out = search.intersect_scenes(scenes, flight)

    assert out["scene"].tolist() == ["scene1"]
    assert out["longitude"].item() == pytest.approx(1.5)  # interpolated to midpoint
    assert out["latitude"].item() == pytest.approx(0.5)


def test_intersect_scenes_empty() -> None:
    """Test that a flight outside every footprint intersects no scenes."""
    scenes = _scenes([shapely.box(0, 0, 1, 1)], ["2024-01-01 00:30"])
    flight = Flight(
        longitude=[5.0, 6.0],
        latitude=[0.5, 0.5],
        altitude=[10_000, 10_000],
        time=pd.to_datetime(["2024-01-01 00:00", "2024-01-01 01:00"]),
    )
    out = search.intersect_scenes(scenes, flight)

    assert out.empty
    assert list(out.columns) == [*scenes.columns, "longitude", "latitude"]


def test_intersect_scenes_antimeridian() -> None:
    """Test intersection with a footprint split at the antimeridian."""
    footprint = shapely.MultiPolygon([shapely.box(179, 0, 180, 1), shapely.box(-180, 0, -179, 1)])
    scenes = _scenes([footprint], ["2024-01-01 00:30"])
    flight = Flight(
        longitude=[179.0, -179.2],
        latitude=[0.5, 0.5],
        altitude=[10_000, 10_000],
        time=pd.to_datetime(["2024-01-01 00:00", "2024-01-01 01:00"]),
    )
    out = search.intersect_scenes(scenes, flight)

    assert out["longitude"].item() == pytest.approx(179.9)


T0 = np.datetime64("2024-01-01T00:00", "ns")
MINUTE = np.timedelta64(1, "m")


def test_colocate() -> None:
    """Test interpolation of waypoints to the scan time."""
    # Waypoints out of time order, with a NaT scan time on the last one
    vector = GeoVectorDataset(
        {
            "flight_id": ["a", "a", "a"],
            "time": T0 + MINUTE * np.array([20, 10, 0]),
            "longitude": [2.0, 1.0, 0.0],
            "latitude": [0.0, 0.0, 0.0],
            "altitude": [12_000.0, 11_000.0, 10_000.0],
            "rhi": [2.0, 1.0, 0.0],
        }
    )
    x = np.array([20.0, 10.0, 0.0])
    y = np.array([2.0, 1.0, 0.0])
    scan_time = np.array([T0 + 15 * MINUTE, T0 + 15 * MINUTE, np.datetime64("NaT", "ns")])

    out = correction.colocate(vector, x, y, scan_time, "flight_id", ["rhi"])
    expected = {
        "flight_id": "a",
        "x": 15.0,
        "y": 1.5,
        "time": pd.Timestamp("2024-01-01T00:15"),
        "longitude": 1.5,
        "latitude": 0.0,
        "altitude": 11_500.0,
        "rhi": 1.5,
    }
    assert out.to_dict("records") == [expected]
    assert list(out.columns) == list(expected)


def test_colocate_no_crossing() -> None:
    """Test that groups with no waypoints on both sides of the scan time are dropped."""
    # b: first waypoint is already past the scan time; c: last waypoint is before it
    vector = GeoVectorDataset(
        {
            "flight_id": ["b", "b", "c", "c"],
            "time": T0 + MINUTE * np.array([0, 10, 0, 10]),
            "longitude": [0.0, 1.0, 0.0, 1.0],
            "latitude": [0.0, 0.0, 0.0, 0.0],
            "altitude": [10_000.0] * 4,
        }
    )
    scan_time = np.array([T0 - 5 * MINUTE] * 2 + [T0 + 20 * MINUTE] * 2)

    out = correction.colocate(vector, np.zeros(4), np.zeros(4), scan_time, "flight_id")
    assert out.empty


def test_colocate_antimeridian() -> None:
    """Test interpolation of longitude across the antimeridian."""
    vector = GeoVectorDataset(
        {
            "time": T0 + MINUTE * np.array([0, 10]),
            "longitude": [179.5, -179.5],
            "latitude": [0.0, 0.0],
            "altitude": [10_000.0, 10_000.0],
        }
    )
    scan_time = np.array([T0 + 150 * np.timedelta64(1, "s")] * 2)

    out = correction.colocate(vector, np.zeros(2), np.zeros(2), scan_time)
    assert out["longitude"].item() == pytest.approx(179.75)


# ========
# Landsat
# ========

LANDSAT_COLUMNS = {
    "product_id",
    "scene_id",
    "platform",
    "sensing_time",
    "sun_elevation",
    "geometry",
}


@pytest.fixture()
def landsat_product_id() -> str:
    """Landsat product ID for tests."""
    return "LC08_L1TP_095055_20190101_20201016_02_T1"


@pytest.fixture(scope="module")
def landsat_cachestore() -> Generator[cache.DiskCacheStore, None, None]:
    """Yield a Landsat cache shared by the tests in this module and clear it after them."""
    cache_root = cache._get_user_cache_dir()
    cache_dir = f"{cache_root}/landsat-unit-test"
    cachestore = cache.DiskCacheStore(cache_dir=cache_dir, allow_clear=True)
    yield cachestore
    cachestore.clear()


@pytest.fixture()
def landsat_fixture_product_id() -> str:
    """Landsat product ID of the MTL and ANG files in tests/unit/static."""
    return "LC08_L1TP_188234_20260710_20260716_02_T1"


@pytest.fixture()
def landsat_fixture_cachestore(
    tmp_path: pathlib.Path, landsat_fixture_product_id: str
) -> cache.DiskCacheStore:
    """Return a cache store holding the MTL and ANG files in tests/unit/static.

    Landsat 8 metadata courtesy of the U.S. Geological Survey.
    """
    cachestore = cache.DiskCacheStore(cache_dir=str(tmp_path), allow_clear=True)
    pid = landsat_fixture_product_id
    cachestore.put(get_static_path("landsat-mtl.txt"), f"{pid}/{pid}_MTL.txt")
    cachestore.put(get_static_path("landsat-ang.txt"), f"{pid}/{pid}_ANG.txt")
    return cachestore


@pytest.mark.network
def test_landsat_empty_query() -> None:
    """Test Landsat imagery query that returns an empty result."""
    start_time = np.datetime64("2019-01-01 00:00:00")
    end_time = np.datetime64("2019-01-01 00:15:00")
    extent = shapely.to_geojson(shapely.Point(0, 0))

    df = landsat.query(start_time, end_time, extent)
    assert set(df.columns) == LANDSAT_COLUMNS
    assert df.empty


@pytest.mark.network
def test_landsat_query(landsat_product_id: str) -> None:
    """Test Landsat imagery query that returns a non-empty result."""
    start_time = np.datetime64("2019-01-01 00:00:00")
    end_time = np.datetime64("2019-01-01 00:15:00")
    extent = shapely.to_geojson(shapely.Point(151, 7))

    df = landsat.query(start_time, end_time, extent)
    assert set(df.columns) == LANDSAT_COLUMNS
    assert len(df) == 1
    assert df["product_id"].item() == landsat_product_id


@pytest.mark.network
def test_landsat_empty_intersection() -> None:
    """Test Landsat flight intersection that return an empty result."""
    flight = Flight(
        longitude=[0, 1],
        latitude=[0, 1],
        altitude=[10_000, 10_000],
        time=[np.datetime64("2019-01-01 00:00:00"), np.datetime64("2019-01-01 00:15:00")],
    )

    df = landsat.intersect(flight)
    assert set(df.columns) == {*LANDSAT_COLUMNS, "longitude", "latitude"}
    assert df.empty


@pytest.mark.network
def test_landsat_intersection(landsat_product_id: str) -> None:
    """Test Landsat flight intersection that return an non-empty result."""
    flight = Flight(
        longitude=[151, 151.1],
        latitude=[7, 7.1],
        altitude=[10_000, 10_000],
        time=[np.datetime64("2019-01-01 00:00:00"), np.datetime64("2019-01-01 00:15:00")],
    )

    df = landsat.intersect(flight)
    assert set(df.columns) == {*LANDSAT_COLUMNS, "longitude", "latitude"}
    assert len(df) == 1
    assert df["product_id"].item() == landsat_product_id


@pytest.mark.parametrize(
    ("bands", "succeed"),
    [
        ({"B8"}, True),
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


@pytest.mark.network
@pytest.mark.skipif(not USGS_M2M_CREDENTIALS, reason="No USGS M2M credentials")
@pytest.mark.parametrize("band", [f"B{i}" for i in range(1, 9)])
@pytest.mark.parametrize("processing", ["raw", "radiance", "reflectance"])
def test_landsat_get_reflective_bands(
    landsat_cachestore: cache.DiskCacheStore,
    landsat_product_id: str,
    band: str,
    processing: str,
) -> None:
    """Test downloading and processing of reflective bands."""
    downloader = landsat.Landsat(landsat_product_id, cachestore=landsat_cachestore, bands=[band])
    ds = downloader.get(reflective=processing)

    assert isinstance(ds, xr.Dataset)
    assert band in ds
    da = ds[band]
    assert isinstance(da, xr.DataArray)
    assert da.ndim == 2
    assert da.name == band
    assert da.dims == ("y", "x")
    assert da.dtype == ("uint16" if processing == "raw" else "float32")
    assert da.attrs["units"] == (
        "none" if processing == "raw" else "W/m^2/sr/um" if processing == "radiance" else "nondim"
    )

    ds.close()


@pytest.mark.network
@pytest.mark.skipif(not USGS_M2M_CREDENTIALS, reason="No USGS M2M credentials")
@pytest.mark.parametrize("band", ["B10", "B11"])
@pytest.mark.parametrize("processing", ["raw", "radiance", "brightness_temperature"])
def test_landsat_get_thermal_bands(
    landsat_cachestore: cache.DiskCacheStore,
    landsat_product_id: str,
    band: str,
    processing: str,
) -> None:
    """Test downloading and processing of thermal bands."""
    downloader = landsat.Landsat(landsat_product_id, cachestore=landsat_cachestore, bands=[band])
    ds = downloader.get(thermal=processing)

    assert isinstance(ds, xr.Dataset)
    assert band in ds
    da = ds[band]
    assert isinstance(da, xr.DataArray)
    assert da.ndim == 2
    assert da.name == band
    assert da.dims == ("y", "x")
    assert da.dtype == ("uint16" if processing == "raw" else "float32")
    assert da.attrs["units"] == (
        "none" if processing == "raw" else "W/m^2/sr/um" if processing == "radiance" else "K"
    )

    ds.close()


@pytest.mark.network
@pytest.mark.skipif(not USGS_M2M_CREDENTIALS, reason="No USGS M2M credentials")
def test_landsat_generate_true_color_rgb(
    landsat_cachestore: cache.DiskCacheStore, landsat_product_id: str
) -> None:
    """Test true color RGB generation."""
    downloader = landsat.Landsat(landsat_product_id, cachestore=landsat_cachestore)
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


@pytest.mark.network
@pytest.mark.skipif(not USGS_M2M_CREDENTIALS, reason="No USGS M2M credentials")
def test_landsat_generate_google_contrails_rgb(
    landsat_cachestore: cache.DiskCacheStore, landsat_product_id: str
) -> None:
    """Test Google contrails RGB generation."""
    downloader = landsat.Landsat(
        landsat_product_id, cachestore=landsat_cachestore, bands=["B9", "B10", "B11"]
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


@pytest.mark.parametrize(
    "product_id",
    [
        "LC08_L1TP_124048_20190112_20190131_01_T1",  # Collection 1
        "LE07_L1TP_124048_20190112_20200829_02_T1",  # Landsat 7
        "gs://gcp-public-data-landsat/LC08/01/095/055/LC08_L1TP_095055_20190101_20190130_01_T1",
    ],
)
def test_landsat_invalid_product_id(product_id: str) -> None:
    """Test that only Landsat 8 and 9 Collection 2 Level-1 product IDs are accepted."""
    with pytest.raises(ValueError, match="is not a Landsat 8 or 9 Collection 2 Level-1"):
        landsat.Landsat(product_id)


def test_landsat_invalid_source(landsat_product_id: str) -> None:
    """Test Landsat source validation."""
    with pytest.raises(ValueError, match="source must be"):
        landsat.Landsat(landsat_product_id, source="planetary-computer")


def test_landsat_gcs_url(landsat_product_id: str) -> None:
    """Test the URLs of Landsat files in the requester-pays Google Cloud Storage bucket."""
    handler = landsat.Landsat(landsat_product_id, source="gcs")
    assert handler._gcs_url("B9.TIF") == (
        "gs://gee-public-data-landsat/LC08/L1/02/095/055/"
        "LC08_L1TP_095055_20190101_20201016_02_T1/"
        "LC08_L1TP_095055_20190101_20201016_02_T1_B9.TIF"
    )


def test_landsat_metadata(
    landsat_fixture_product_id: str,
    landsat_fixture_cachestore: cache.DiskCacheStore,
) -> None:
    """Test reading Landsat Collection 2 metadata from cached MTL and ANG files."""
    handler = landsat.Landsat(landsat_fixture_product_id, cachestore=landsat_fixture_cachestore)

    assert handler.get_sensing_time() == pd.Timestamp("2026-07-10T10:55:14.0633360")

    ephemeris = handler.get_ephemeris()
    assert list(ephemeris.columns) == [
        "EPHEMERIS_TIME",
        "EPHEMERIS_ECEF_X",
        "EPHEMERIS_ECEF_Y",
        "EPHEMERIS_ECEF_Z",
    ]
    assert len(ephemeris) == 54
    assert ephemeris["EPHEMERIS_TIME"].dtype == "datetime64[ns]"

    meta = landsat_fixture_cachestore.path(
        f"{landsat_fixture_product_id}/{landsat_fixture_product_id}_MTL.txt"
    )
    assert landsat._read_meta(meta, "RADIANCE_MULT_BAND_1") == 1.2149e-02
    assert landsat._read_meta(meta, "RADIANCE_MULT_BAND_10") == 3.3420e-04

    x, y = landsat._read_image_coordinates(meta, "B1")
    assert (len(y), len(x)) == (8611, 8571)
    assert (x[0], x[-1], y[0], y[-1]) == (281400.0, 538500.0, 7999200.0, 7740900.0)

    x, y = landsat._read_image_coordinates(meta, "B8")
    assert (len(y), len(x)) == (17221, 17141)


@pytest.mark.network
@pytest.mark.skipif(not USGS_M2M_CREDENTIALS, reason="No USGS M2M credentials")
def test_landsat_colocate_usgs(landsat_cachestore: cache.DiskCacheStore) -> None:
    """Test colocation of the IAGOS flight in the Landsat notebook."""
    path = pathlib.Path(__file__).parents[2] / "docs/notebooks/data/iagos-flight-landsat.csv"
    df = pd.read_csv(path, parse_dates=["time"])
    df["time"] += pd.Timedelta(seconds=6)  # IAGOS time delay, see the notebook

    flight = Flight(df)

    product_id = "LC08_L1TP_124048_20190112_20200829_02_T1"
    handler = landsat.Landsat(product_id, cachestore=landsat_cachestore)
    x, y, t = handler.colocate_flight(flight)

    # Confirm the colocated position is inside the image
    ds = handler.get_viewing_angle_metadata()
    assert ds["x"].min() <= x <= ds["x"].max()
    assert ds["y"].min() <= y <= ds["y"].max()

    # Confirm the colocated time is during the flight and near the scene center time.
    # Landsat crosses a 185 km scene in about 26 s.
    assert flight.time_start <= t <= flight.time_end
    sensing_time = handler.get_sensing_time().to_datetime64()
    assert abs(t - sensing_time) < np.timedelta64(15, "s")


def test_usgs_m2m_credentials(monkeypatch: pytest.MonkeyPatch, landsat_product_id: str) -> None:
    """Test that M2M downloads require credentials."""
    monkeypatch.delenv("USGS_M2M_USERNAME", raising=False)
    monkeypatch.delenv("USGS_M2M_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="USGS_M2M_USERNAME and USGS_M2M_TOKEN"):
        usgs_m2m.download(landsat_product_id, {"B9.TIF": "unused"})


# ===================
# Sentinel-2 datalib
# ===================

SENTINEL_COLUMNS = {
    "id",
    "base_url",
    "granule_id",
    "platform",
    "sensing_time",
    "processing_baseline",
    "sun_elevation",
    "cloud_cover",
    "geometry",
}


@pytest.fixture()
def sentinel_base_url() -> str:
    """Sentinel base URL for tests."""
    return "gs://gcp-public-data-sentinel-2/tiles/59/V/ND/S2A_MSIL1C_20190101T000001_N0207_R116_T59VND_20190101T013420.SAFE"


@pytest.fixture()
def sentinel_granule_id() -> str:
    """Sentinel granule ID for tests."""
    return "L1C_T59VND_A018417_20181231T235957"


@pytest.fixture(scope="module")
def sentinel_cachestore() -> Generator[cache.DiskCacheStore, None, None]:
    """Yield a Sentinel cache shared by the tests in this module and clear it after them."""
    cache_root = cache._get_user_cache_dir()
    cache_dir = f"{cache_root}/sentinel-unit-test"
    cachestore = cache.DiskCacheStore(cache_dir=cache_dir, allow_clear=True)
    yield cachestore
    cachestore.clear()


@pytest.mark.network
def test_sentinel_colocate_flight(sentinel_cachestore: cache.DiskCacheStore) -> None:
    """Test colocation of the IAGOS flight in the Sentinel-2 notebook."""
    path = pathlib.Path(__file__).parents[2] / "docs/notebooks/data/iagos-flight-sentinel.csv"
    df = pd.read_csv(path, parse_dates=["time"])
    df["time"] += pd.Timedelta(seconds=9)  # IAGOS time delay, see the notebook
    flight = Flight(df)

    base_url = (
        "gs://gcp-public-data-sentinel-2/tiles/30/U/WG/"
        "S2B_MSIL1C_20191226T112359_N0208_R037_T30UWG_20191226T115227.SAFE"
    )
    granule_id = "L1C_T30UWG_A014649_20191226T112436"
    handler = sentinel.Sentinel(base_url, granule_id, cachestore=sentinel_cachestore)

    x, y, t = handler.colocate_flight(flight)

    # Confirm the colocated position is inside the image
    ds = handler.get_viewing_angle_metadata()
    assert ds["x"].min() <= x <= ds["x"].max()
    assert ds["y"].min() <= y <= ds["y"].max()

    # Confirm the colocated time is during the flight and near the granule sensing time.
    assert flight.time_start <= t <= flight.time_end
    sensing_time = handler.get_sensing_time().tz_localize(None).to_datetime64()
    assert abs(t - sensing_time) < np.timedelta64(10, "s")


@pytest.mark.network
def test_sentinel_empty_query() -> None:
    """Test Sentinel imagery query that returns an empty result."""
    start_time = np.datetime64("2019-01-01 00:00:00")
    end_time = np.datetime64("2019-01-01 00:01:00")
    extent = shapely.to_geojson(shapely.Point(0, 0))

    df = sentinel.query(start_time, end_time, extent)
    assert set(df.columns) == SENTINEL_COLUMNS
    assert df.empty


@pytest.mark.network
def test_sentinel_query(sentinel_base_url: str, sentinel_granule_id: str) -> None:
    """Test Sentinel imagery query that returns a non-empty result."""
    start_time = np.datetime64("2019-01-01 00:00:00")
    end_time = np.datetime64("2019-01-01 00:01:00")
    extent = shapely.to_geojson(shapely.Point(171, 57))

    df = sentinel.query(start_time, end_time, extent)
    assert set(df.columns) == SENTINEL_COLUMNS
    assert not df.empty
    assert df["base_url"][0] == sentinel_base_url
    assert df["granule_id"][0] == sentinel_granule_id


@pytest.mark.network
def test_sentinel_empty_intersection() -> None:
    """Test Sentinel flight intersection that return an empty result."""
    flight = Flight(
        longitude=[0, 1],
        latitude=[0, 1],
        altitude=[10_000, 10_000],
        time=[np.datetime64("2019-01-01 00:00:00"), np.datetime64("2019-01-01 00:01:00")],
    )

    df = sentinel.intersect(flight)
    assert set(df.columns) == {*SENTINEL_COLUMNS, "longitude", "latitude"}
    assert df.empty


@pytest.mark.network
def test_sentinel_intersection(sentinel_base_url: str, sentinel_granule_id: str) -> None:
    """Test Sentinel flight intersection that return an non-empty result."""
    flight = Flight(
        longitude=[171, 171.1],
        latitude=[57, 57.1],
        altitude=[10_000, 10_000],
        time=[np.datetime64("2019-01-01 00:00:00"), np.datetime64("2019-01-01 00:01:00")],
    )

    df = sentinel.intersect(flight)
    assert set(df.columns) == {*SENTINEL_COLUMNS, "longitude", "latitude"}
    assert not df.empty
    assert df["base_url"][0] == sentinel_base_url
    assert df["granule_id"][0] == sentinel_granule_id


@pytest.mark.network
def test_sentinel_intersection_antimeridian() -> None:
    """Test Sentinel flight intersection with a scene that spans the antimeridian."""
    flight = Flight(
        longitude=[179.5, -179.9],
        latitude=[-18.5, -18.5],
        altitude=[10_000, 10_000],
        time=[np.datetime64("2019-01-14 22:15:00"), np.datetime64("2019-01-14 22:30:00")],
    )

    # Tiles 01KAV and 60KYE overlap across the antimeridian
    df = sentinel.intersect(flight)
    assert set(df["granule_id"]) == {
        "L1C_T01KAV_A018616_20190114T221935",
        "L1C_T60KYE_A018616_20190114T221935",
    }


@pytest.mark.parametrize(
    ("bands", "succeed"),
    [
        ({"B02", "B03", "B04", "B08"}, True),
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


@pytest.mark.network
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
    assert da.dtype == ("uint16" if processing == "raw" else "float32")
    assert da.attrs["units"] == ("none" if processing == "raw" else "nondim")

    ds.close()


@pytest.mark.network
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
