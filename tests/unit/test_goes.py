"""Test the goes module."""

from collections.abc import Generator

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pycontrails import DiskCacheStore
from pycontrails.datalib import goes
from tests import IS_WINDOWS, OFFLINE


@pytest.mark.parametrize(
    ("region", "succeed"),
    [
        ("C", True),
        ("E", False),
        ("meso scale 1", True),
        ("meso scale 3", False),
        ("full disc", False),
        ("F", True),
        ("m2", True),
    ],
)
def test_parse_region(region: str, succeed: bool) -> None:
    """Test the '_parse_region' function."""
    if succeed:
        parsed = goes._parse_region(region)
        assert isinstance(parsed, goes.GOESRegion)
        return

    with pytest.raises(ValueError, match="Region must be one of"):
        goes._parse_region(region)


@pytest.mark.parametrize(
    ("channels", "succeed"),
    [
        (("C01",), True),
        (("C11",), True),
        (("C01", "C02"), True),
        (("C01", "C02", "C03", "C04"), False),
        (("C01", "C02", "C03", "C05"), True),
        (("C02", "C11"), False),
        (("C11", "C13", "C14", "C15"), True),
    ],
)
def test_channel_resolution(channels: tuple[str, ...], succeed: bool) -> None:
    """Test the '_check_channel_resolution' function."""

    if succeed:
        assert goes._check_channel_resolution(channels) is None
        return

    with pytest.raises(ValueError, match="Channels must have a common horizontal resolution"):
        goes._check_channel_resolution(channels)


@pytest.mark.skipif(IS_WINDOWS, reason="cannot easily install h5py on windows")
@pytest.mark.skipif(OFFLINE, reason="offline")
@pytest.mark.parametrize("t", ["2023-06-15T15:34", "2019-03-14T15:34"])
def test_goes_get_no_cache_default_channels(t: str) -> None:
    downloader = goes.GOES(region="m1", cachestore=None)
    assert downloader.cachestore is None

    da = downloader.get(t)
    assert isinstance(da, xr.DataArray)
    assert da.shape == (3, 500, 500)
    assert da.name == "CMI"
    assert da.dims == ("band_id", "y", "x")
    assert da.dtype == "float32"
    assert da["band_id"].values.tolist() == [11, 14, 15]
    assert da.mean().item() == pytest.approx(275, abs=15)

    # Ensure we can construct the visualization
    rgb, src_crs, src_extent = goes.extract_goes_visualization(da)
    assert isinstance(rgb, np.ndarray)
    assert rgb.shape == (500, 500, 3)
    assert np.all(np.isfinite(rgb))
    assert rgb.min() == 0
    assert rgb.max() == 1

    assert (
        src_crs
        == "+proj=geos +ellps=WGS84 +lon_0=-75.0 +lat_0=0.0 +h=35786023.0 +x_0=0 +y_0=0 +units=m"
        " +sweep=x +no_defs +type=crs"
    )
    assert len(src_extent) == 4


@pytest.mark.skipif(IS_WINDOWS, reason="cannot easily install h5py on windows")
@pytest.mark.skipif(OFFLINE, reason="offline")
def test_goes_get_no_cache_rgb_channels() -> None:
    downloader = goes.GOES(region="m1", cachestore=None, channels=("C01", "C02", "C03"))
    assert downloader.cachestore is None

    da = downloader.get("2023-09-15T15:34")
    assert isinstance(da, xr.DataArray)
    assert da.shape == (3, 1000, 1000)
    assert da.name == "CMI"
    assert da.dims == ("band_id", "y", "x")
    assert da.dtype == "float32"
    assert da["band_id"].values.tolist() == [1, 2, 3]

    rgb, *_ = goes.extract_goes_visualization(da, color_scheme="true")
    assert isinstance(rgb, np.ndarray)
    assert rgb.shape == (1000, 1000, 3)
    assert np.all(np.isfinite(rgb))
    assert rgb.min() == pytest.approx(0.2, abs=0.01)
    assert rgb.max() == 1.0


@pytest.fixture()
def cachestore() -> Generator[DiskCacheStore, None, None]:
    out = DiskCacheStore("test_goes_get", allow_clear=True)
    try:
        yield out
    finally:
        out.clear()


@pytest.mark.skipif(OFFLINE, reason="offline")
@pytest.mark.parametrize("t", ["2023-06-15T15:34", "2021-01-15T00", "2019-03-14T15:34"])
def test_goes_get_with_cache(t: str, cachestore: DiskCacheStore) -> None:
    downloader = goes.GOES(region="m2", cachestore=cachestore, channels="C02")
    assert downloader.cachestore is not None
    assert downloader.cachestore is cachestore
    assert cachestore.listdir() == []

    da = downloader.get(t)
    assert isinstance(da, xr.DataArray)
    assert da.shape == (1, 2000, 2000)
    assert da.name == "CMI"
    assert da.dims == ("band_id", "y", "x")
    assert da.dtype == "float32"
    assert da._in_memory  # hard-coded in the goes module ... we can change this later

    t_str = pd.Timestamp(t).strftime("%Y%m%d%H%M")
    assert cachestore.listdir() == [f"M2_{t_str}_C02.nc"]


@pytest.fixture(scope="module")
def goes_data() -> xr.DataArray:
    """A fixture that returns a sample GOES data array for parallax correction tests."""
    downloader = goes.GOES(region="m1", cachestore=None, channels=("C01", "C02", "C03"))
    return downloader.get("2023-09-15T15:34")


def test_goes_parallax_correct_above_nadir(goes_data: xr.DataArray) -> None:
    """Test the ``parallax_correct`` function."""
    # If we're right above nadir, the parallax correction shouldn't change anything
    lon0 = np.array([-75.0])
    lat0 = np.array([0.0])
    alt = np.array([10000.0])
    lon1, lat1 = goes.parallax_correct(lon0, lat0, alt, goes_data)
    assert lon0 == pytest.approx(lon1)
    assert lat0 == pytest.approx(lat1)


def test_goes_parallax_correct_due_north(goes_data: xr.DataArray) -> None:
    """Test the ``parallax_correct`` function."""
    # If we're due north, the parallax correction should shift the latitude only
    lon0 = np.array([-75.0])
    lat0 = np.array([44.0])
    alt = np.array([10000.0])
    lon1, lat1 = goes.parallax_correct(lon0, lat0, alt, goes_data)
    assert lon0 == pytest.approx(lon1)
    assert lat1.item() == pytest.approx(44.11, abs=0.01)


def test_goes_parallax_correct_due_east(goes_data: xr.DataArray) -> None:
    """Test the ``parallax_correct`` function."""
    # If we're due east, the parallax correction should shift the longitude only
    lon0 = np.array([-33.0])
    lat0 = np.array([0.0])
    alt = np.array([10000.0])
    lon1, lat1 = goes.parallax_correct(lon0, lat0, alt, goes_data)
    assert lat0 == pytest.approx(lat1)
    assert lon1.item() == pytest.approx(-32.90, abs=0.01)


def test_goes_parallax_correct_random(goes_data: xr.DataArray) -> None:
    """Test the ``parallax_correct`` function."""
    # In general, parallax correction doesn't shift more than +/- 0.1 degrees
    rng = np.random.default_rng(444333222111)
    n = 1000
    lon0 = rng.uniform(-100, -50, n)
    lat0 = rng.uniform(-70, 70, n)
    alt = rng.uniform(1000, 20000, n)
    lon1, lat1 = goes.parallax_correct(lon0, lat0, alt, goes_data)
    assert np.all(np.abs(lon0 - lon1) < 1.0)
    assert np.all(np.abs(lat0 - lat1) < 1.0)
    assert np.mean(np.abs(lon0 - lon1)) == pytest.approx(0.07, abs=0.01)
    assert np.mean(np.abs(lat0 - lat1)) == pytest.approx(0.11, abs=0.01)


def test_goes_parallax_correct_opposite_side(goes_data: xr.DataArray) -> None:
    """Test the ``parallax_correct`` function."""

    lon0 = np.arange(-179.5, 180, 1, dtype=float)
    opposite_side = np.abs(((lon0 - -75 + 180) % 360) - 180) > 90
    lat0 = np.zeros_like(lon0)
    alt = np.full_like(lon0, 10000)

    lon1, lat1 = goes.parallax_correct(lon0, lat0, alt, goes_data)

    # If we're on the opposite side of the globe, the parallax correction returns nan
    assert np.all(np.isnan(lon1[opposite_side]))
    assert np.all(np.isnan(lat1[opposite_side]))

    # If we're on the same side of the globe, the parallax correction generally returns non-nan
    # However, when the ray between the satellite and the aircraft is close to the horizon,
    # it may miss the surface. This occurs for the first and last 12 points in this array.
    assert np.all(np.isfinite(lon1[~opposite_side][12:-12]))
    assert np.all(np.isfinite(lat1[~opposite_side][12:-12]))


@pytest.mark.skipif(IS_WINDOWS, reason="cannot easily install h5py on windows")
@pytest.mark.skipif(OFFLINE, reason="offline")
def test_goes_19() -> None:
    """Confirm that we can access GOES-19 data."""
    downloader = goes.GOES(region="m1", cachestore=None, channels="C02")
    assert downloader.cachestore is None

    da = downloader.get("2025-04-15T15:34")
    assert isinstance(da, xr.DataArray)

    assert da.shape == (1, 2000, 2000)  # C02 has 0.5 km resolution
    assert da.name == "CMI"
    assert da.dims == ("band_id", "y", "x")
    assert da.dtype == "float32"
    assert da["band_id"].values.tolist() == [2]
