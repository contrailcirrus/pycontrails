"""Tests for the GRUAN datalib module using real FTP, no mocking."""

from __future__ import annotations

from collections.abc import Generator

import pytest
import xarray as xr

from pycontrails import DiskCacheStore
from pycontrails.datalib.gruan import GRUAN, extract_gruan_time
from tests import OFFLINE


def test_available_sites_live() -> None:
    """Test live retrieval of available sites.

    If this test fails, the cached AVAILABLE dict in gruan.py may need updating.
    """
    products_sites = GRUAN.available_sites()
    assert products_sites == GRUAN.AVAILABLE


@pytest.fixture(scope="module")
def gruan() -> GRUAN:
    """Create a GRUAN instance for testing.

    The module scope aims to avoid reconnecting to FTP server for each test.
    """
    return GRUAN(product="RS92-GDP.2", site="LIN", cachestore=None)


@pytest.fixture()
def cachestore() -> Generator[DiskCacheStore]:
    cache = DiskCacheStore("test_gruan", allow_clear=True)
    assert cache.listdir() == []

    try:
        yield cache
    finally:
        cache.clear()


def test_gruan_repr(gruan: GRUAN) -> None:
    """Test GRUAN __repr__."""
    repr_str = repr(gruan)
    assert repr_str == "GRUAN(product='RS92-GDP.2', site='LIN')"


@pytest.mark.skipif(OFFLINE, reason="offline")
def test_ftp_reuse(gruan: GRUAN) -> None:
    """Test that FTP connection is reused."""
    ftp1 = gruan._connect()
    ftp2 = gruan._connect()
    assert ftp1 is ftp2
    assert ftp1.pwd() == "/"


@pytest.mark.skipif(OFFLINE, reason="offline")
def test_available_products_live(gruan: GRUAN) -> None:
    """Test live retrieval of available products."""
    base_path = "/pub/data/gruan/processing/level2"

    ftp = gruan._connect()
    ftp.cwd(base_path)
    products = ftp.nlst()
    expected = {"RS41-EDT", "RS-11G-GDP", "RS92-GDP", "RS92-PROFILE-BETA"}
    # exclude files with dots (readme.txt etc)
    assert {p for p in products if "." not in p} == expected


@pytest.mark.skipif(OFFLINE, reason="offline")
def test_years(gruan: GRUAN) -> None:
    """Test available years retrieval."""
    years = gruan.years()
    # For this site/product, data is available 2005-2021
    assert years == list(range(2005, 2022))


@pytest.mark.skipif(OFFLINE, reason="offline")
def test_list_files_and_extract_time(gruan: GRUAN) -> None:
    """Test listing files for a given year."""
    files = gruan.list_files(2020)
    assert isinstance(files, list)
    assert all(isinstance(f, str) for f in files)
    assert len(files) == 8  # only 8 files for LIN in 2020

    for file in files:
        t, version = extract_gruan_time(file)
        assert t.year == 2020
        assert version == 1


@pytest.mark.skipif(OFFLINE, reason="offline")
def test_get_with_cache_live(gruan: GRUAN, cachestore: DiskCacheStore) -> None:
    """Test cached retrieval using live FTP."""
    try:
        gruan.cachestore = cachestore  # mutate instance for testing

        file = "LIN-RS-01_2_RS92-GDP_002_20210125T132400_1-000-001.nc"  # known file

        assert cachestore.listdir() == []
        ds1 = gruan.get(file)
        assert isinstance(ds1, xr.Dataset)
        for var in ds1.data_vars:
            assert not ds1[var]._in_memory  # everything lazy

        assert cachestore.listdir() == [file]
        ds2 = gruan.get(file)
        assert ds1.equals(ds2)

    finally:
        gruan.cachestore = None  # restore


@pytest.mark.skipif(OFFLINE, reason="offline")
def test_get_with_no_cache_live(gruan: GRUAN) -> None:
    """Test cached retrieval using live FTP."""
    assert gruan.cachestore is None

    file = "LIN-RS-01_2_RS92-GDP_002_20210125T132400_1-000-001.nc"  # known file

    ds = gruan.get(file)
    for var in ds.data_vars:
        assert ds[var]._in_memory, var  # everything is in memory


def test_paths(gruan: GRUAN) -> None:
    """Test base path properties formatting."""
    assert gruan.base_path_product.endswith("/RS92-GDP/version-002")
    assert gruan.base_path_site.endswith("/RS92-GDP/version-002/LIN")


def test_gruan_unknown_product() -> None:
    """Test GRUAN with unknown product."""
    with pytest.raises(ValueError, match="Unknown GRUAN product"):
        GRUAN(product="UNKNOWN", site="LIN")


def test_gruan_unknown_site() -> None:
    """Test GRUAN with unknown site."""
    with pytest.raises(ValueError, match="Unknown GRUAN site"):
        GRUAN(product="RS92-GDP.2", site="XXX")
