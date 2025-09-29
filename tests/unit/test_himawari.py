"""Test the Himawari module."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from pycontrails.datalib import himawari
from pycontrails.datalib.himawari import himawari as himawari_module
from tests import OFFLINE


@pytest.mark.parametrize(
    ("region", "succeed"),
    [
        ("FLDK", True),
        ("full disk", True),
        ("Japan", True),
        ("target", True),
        ("mesoscale", True),
        ("invalid", False),
    ],
)
def test_parse_region(region: str, succeed: bool) -> None:
    """Test the '_parse_region' function."""
    if succeed:
        parsed = himawari_module._parse_region(region)
        assert isinstance(parsed, himawari.HimawariRegion)
        return
    with pytest.raises(ValueError, match="Region must be one of"):
        himawari_module._parse_region(region)


@pytest.mark.parametrize(
    ("bands", "succeed"),
    [
        (("B11",), True),
        (("B11", "B14"), True),
        (("B01", "B02", "B03"), True),
        (("B01", "B11"), False),
        (("B01", "B02", "B03", "B04"), True),
        (("B01", "B02", "B03", "B08"), False),
    ],
)
def test_band_resolution(bands: tuple[str, ...], succeed: bool) -> None:
    """Test the '_check_band_resolution' function."""
    if succeed:
        himawari_module._check_band_resolution(bands)
        return

    with pytest.raises(ValueError, match=r"Bands must have a common horizontal resolution"):
        himawari_module._check_band_resolution(bands)


@pytest.mark.skipif(OFFLINE, reason="offline")
@pytest.mark.parametrize("t", ["2024-06-15T12:17:30", "2021-03-14T00:02:30"])
def test_himawari_get_no_cache_default_bands(t: str) -> None:
    """Test the ``Himawari.get`` method with default bands and no cache over the Target region.

    This test mirrors the GOES test_goes_get_no_cache_default_channels test.
    """
    downloader = himawari.Himawari(region="T", cachestore=None)
    assert downloader.cachestore is None

    da = downloader.get(t)
    assert isinstance(da, xr.DataArray)
    assert da.shape == (3, 500, 500)
    assert da.name == "CMI"
    assert da.dims == ("band_id", "y", "x")
    assert da.dtype == "float32"
    assert da["band_id"].values.tolist() == [11, 14, 15]
    assert da.mean().item() == pytest.approx(280.0, abs=5.0)  # this value works for both times
    assert da.notnull().all()  # no null values for Target region
    assert "t" in da.coords
    assert (
        da.attrs["crs"] == "+proj=geos +h=35785863.0 +a=6378137.0 +b=6356752.3 "
        "+lon_0=140.7 +sweep=x +units=m +no_defs"
    )
    assert da.attrs["standard_name"] == "toa_brightness_temperature"
    assert da.attrs["units"] == "K"

    # Ensure we can construct the visualization
    rgb, _, _ = himawari.extract_visualization(da)
    assert isinstance(rgb, np.ndarray)
    assert rgb.shape == (500, 500, 3)
    assert np.all(np.isfinite(rgb))
    assert rgb.min() == 0.0
    assert rgb.max() > 0.9


@pytest.mark.skipif(OFFLINE, reason="offline")
@pytest.mark.parametrize("t", ["2024-06-15T02:17:30"])
def test_himawari_b03(t: str) -> None:
    """Test the ``Himawari.get`` method with default bands and no cache over the Target region.

    This test mirrors the GOES test_goes_get_no_cache_default_channels test.
    """
    downloader = himawari.Himawari(region="J", bands=("B02", "B03"), cachestore=None)
    assert downloader.cachestore is None

    da = downloader.get(t)
    assert isinstance(da, xr.DataArray)
    assert da.shape == (2, 2400, 3000)  # 0.5 km resolution
    assert da.name == "CMI"
    assert da.dims == ("band_id", "y", "x")
    assert da.dtype == "float32"
    assert da["band_id"].values.tolist() == [2, 3]
    assert da.mean().item() == pytest.approx(0.3, abs=0.5)
    assert da.isnull().any()  # some null values for Japan region
    assert "t" in da.coords
    assert (
        da.attrs["crs"] == "+proj=geos +h=35785863.0 +a=6378137.0 +b=6356752.3 "
        "+lon_0=140.7 +sweep=x +units=m +no_defs"
    )
    assert da.attrs["standard_name"] == "toa_reflectance"
    assert da.attrs["units"] == ""
