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


def test_goes_parallax_correct():
    """Test the ``parallax_correct`` function."""
    downloader = goes.GOES(region="m1", cachestore=None, channels=("C01", "C02", "C03"))
    da = downloader.get("2023-09-15T15:34")

    # If we're right above nadir, the parallax correction shouldn't change anything
    lon0 = [-75]
    lat0 = [0]
    alt = [10000]
    lon1, lat1 = goes.parallax_correct(lon0, lat0, alt, da)
    assert lon0 == pytest.approx(lon1)
    assert lat0 == pytest.approx(lat1)

    # If we're due north, the parallax correction should shift the latitude only
    lon0 = [-75]
    lat0 = [44]
    alt = [10000]
    lon1, lat1 = goes.parallax_correct(lon0, lat0, alt, da)
    assert lon0 == pytest.approx(lon1)
    assert lat1.item() == pytest.approx(44.11, abs=0.01)

    # If we're due east, the parallax correction should shift the longitude only
    lon0 = [-33]
    lat0 = [0]
    alt = [10000]
    lon1, lat1 = goes.parallax_correct(lon0, lat0, alt, da)
    assert lat0 == pytest.approx(lat1)
    assert lon1.item() == pytest.approx(-32.90, abs=0.01)

    # In general, parallax correction doesn't shift more than +/- 0.1 degrees
    rng = np.random.default_rng(444333222111)
    n = 1000
    lon0 = rng.uniform(-100, -50, n)
    lat0 = rng.uniform(-70, 70, n)
    alt = rng.uniform(1000, 20000, n)
    lon1, lat1 = goes.parallax_correct(lon0, lat0, alt, da)
    assert np.all(np.abs(lon0 - lon1) < 1.0)
    assert np.all(np.abs(lat0 - lat1) < 1.0)
    assert np.mean(np.abs(lon0 - lon1)) == pytest.approx(0.07, abs=0.01)
    assert np.mean(np.abs(lat0 - lat1)) == pytest.approx(0.11, abs=0.01)
