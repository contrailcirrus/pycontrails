"""Test cache module."""

from __future__ import annotations

import os
import pathlib
import random
from datetime import datetime

import pytest
from google.cloud import storage

from pycontrails import DiskCacheStore, GCPCacheStore
from pycontrails.core import cache

# see if GCP credentials exist
try:
    storage.Client()
except Exception:
    gcp_credentials = False
else:
    gcp_credentials = True

try:
    import requests

    r = requests.get("https://github.com", timeout=10)
except Exception:
    offline = True
else:
    offline = False

############
# Disk Cache Store
############

DISK_CACHE_DIR = cache._get_user_cache_dir()


class TestDiskCacheStore:
    def test_cache_dir_exists(self) -> None:
        """Test if default cache exists and is a directory."""
        _cache = DiskCacheStore()
        _dir = pathlib.Path(_cache.cache_dir)
        assert _dir.exists()
        assert _dir.is_dir()

        # init with pathlib
        _cache = DiskCacheStore(cache_dir=pathlib.Path(f"{DISK_CACHE_DIR}/test"))
        assert pathlib.Path(_cache.cache_dir).is_dir()
        assert _cache.cache_dir == str(pathlib.Path(_cache.cache_dir))

    def test_cache_bad_path(self) -> None:
        _cache = DiskCacheStore(cache_dir=f"{DISK_CACHE_DIR}/test")

        # return False for random string
        disk_path = "".join(random.choice(["x", "y"]) for _ in range(20))
        assert not _cache.exists(disk_path)

    @pytest.mark.skipif(not pathlib.Path("README.md").is_file(), reason="No README.md")
    def test_cache_size(self) -> None:
        _cache = DiskCacheStore(cache_dir=f"{DISK_CACHE_DIR}/test", allow_clear=True)
        _cache.clear()
        size = _cache.size
        assert isinstance(size, float) and size == 0

        _cache.put("README.md")
        size = _cache.size
        assert isinstance(size, float) and size > 0

        assert _cache.listdir() == ["README.md"]

        # cleanup
        _cache.clear()

    def test_path(self) -> None:
        _cache = DiskCacheStore(cache_dir=f"{DISK_CACHE_DIR}/test", allow_clear=True)

        disk_path = _cache.path("")
        assert "pycontrails" in disk_path
        assert pathlib.Path(disk_path).is_dir()

    def test_exists(self) -> None:
        _cache = DiskCacheStore(cache_dir=f"{DISK_CACHE_DIR}/test", allow_clear=True)

        assert not _cache.exists("cache/path4.nc")
        disk_path = _cache.path("cache/path4.nc")
        assert _cache.exists("cache")
        open(disk_path, "w").close()
        assert _cache.exists("cache/path4.nc")
        assert _cache.exists(disk_path)

    def test_cache_allow_clear(self) -> None:
        _cache = DiskCacheStore(cache_dir=f"{DISK_CACHE_DIR}/test", allow_clear=False)
        with pytest.raises(RuntimeError):
            _cache.clear()

        _cache = DiskCacheStore(cache_dir=f"{DISK_CACHE_DIR}/test", allow_clear=True)
        disk_path = _cache.path("path.nc")
        open(disk_path, "w").close()
        _cache.clear()
        assert not _cache.exists("path.nc")

    @pytest.mark.skipif(not pathlib.Path("README.md").is_file(), reason="No README.md")
    def test_put_get(self) -> None:
        _cache = DiskCacheStore(cache_dir=f"{DISK_CACHE_DIR}/test", allow_clear=True)
        _cache.clear()

        # put from filename
        assert not _cache.exists("README.md")
        _cache.put("README.md")
        assert _cache.exists("README.md")

        _cache.put("README.md", "anotherfile.nc")
        assert _cache.exists("anotherfile.nc")

        # file doesn't exist
        with pytest.raises(FileNotFoundError, match="No file found"):
            _cache.put("test2.nc")

        # get from filename
        _path = _cache.get("README.md")
        assert _path == _cache.path("README.md")

        # clean up
        _cache.clear()

    @pytest.mark.skipif(not pathlib.Path("README.md").is_file(), reason="No README.md")
    def test_put_multiple(self) -> None:
        _cache = DiskCacheStore(cache_dir=f"{DISK_CACHE_DIR}/test", allow_clear=True)
        _cache.clear()

        # put from filenames
        assert not _cache.exists("README2.md")
        _cache.put_multiple(["README.md", ".gitignore"], ["README2.md", ".gitignore2"])
        assert _cache.exists("README2.md")
        assert _cache.exists(".gitignore2")
        _cache.clear()

        # put from pathlibs
        assert not _cache.exists("README2.md")
        _cache.put_multiple(
            [pathlib.Path("README.md"), pathlib.Path(".gitignore")], ["README2.md", ".gitignore2"]
        )
        assert _cache.exists("README2.md")
        assert _cache.exists(".gitignore2")

        # file doesn't exist
        with pytest.raises(FileNotFoundError, match="No file found"):
            _cache.put_multiple(["README2.md", ".gitignore"], ["README2.md", ".gitignore2"])

        # clean up
        _cache.clear()


############
# GCS Cache Store
############
BUCKET = "contrails-301217-unit-test"
CACHE_DIR = datetime.now().isoformat().replace(":", "-")


@pytest.mark.skipif(offline, reason="offline")
@pytest.mark.skipif(not gcp_credentials, reason="No GCS credentials")
class TestGCPCacheStore:
    def test_cache_init(self) -> None:
        # test if default cache exists and is a directory
        _cache = GCPCacheStore(bucket=BUCKET)
        assert _cache.bucket == BUCKET
        assert _cache.cache_dir == (
            f'{os.getenv("PYCONTRAILS_CACHE_DIR")}/'
            if "PYCONTRAILS_CACHE_DIR" in os.environ
            else ""
        )
        assert _cache.read_only is True
        assert _cache.allow_clear is False
        assert ".gcp" in _cache._disk_cache.cache_dir
        assert BUCKET in _cache._disk_cache.cache_dir

        # test no bucket
        with pytest.raises(ValueError, match="Parameter `bucket` not specified"):
            GCPCacheStore()

        # parse gs:// URI
        _cache = GCPCacheStore(bucket=BUCKET, cache_dir=f"gs://{BUCKET}/{CACHE_DIR}/")
        assert _cache.bucket == BUCKET
        assert _cache.cache_dir == f"{CACHE_DIR}/"

        # parse gs:// URI when bucket not specified
        _cache = GCPCacheStore(cache_dir=f"gs://{BUCKET}/{CACHE_DIR}/")
        assert _cache.bucket == BUCKET
        assert _cache.cache_dir == f"{CACHE_DIR}/"

        # conflicting bucket raises an error
        with pytest.raises(ValueError, match="conflicting bucket names"):
            GCPCacheStore(bucket="xyz", cache_dir=f"gs://{BUCKET}/{CACHE_DIR}/")

        # raise https:// urls
        with pytest.raises(ValueError, match="only specify base object path"):
            GCPCacheStore(bucket=BUCKET, cache_dir="https://mybucket/")

        # append /
        _cache = GCPCacheStore(bucket=BUCKET, cache_dir=f"{CACHE_DIR}")
        assert _cache.cache_dir == f"{CACHE_DIR}/"

        # test custom disk cache
        _disk_cache = DiskCacheStore(cache_dir=f"{DISK_CACHE_DIR}/{CACHE_DIR}")
        _cache = GCPCacheStore(bucket=BUCKET, disk_cache=_disk_cache)
        assert _cache.bucket == BUCKET
        assert (
            ".gcp" not in _cache._disk_cache.cache_dir
            and f"{CACHE_DIR}" in _cache._disk_cache.cache_dir
        )

    def test_cache_size(self) -> None:
        _cache = GCPCacheStore(
            bucket=BUCKET, cache_dir=f"{CACHE_DIR}/", allow_clear=True, read_only=False
        )
        _cache._dangerous_clear(confirm=True)
        size = _cache.size
        assert isinstance(size, float) and size == 0

        _cache.put("README.md")
        size = _cache.size
        assert isinstance(size, float) and size > 0

    def test_cache_allow_clear(self) -> None:
        """Only allowed through _dangerous_clear."""
        _cache = GCPCacheStore(bucket=BUCKET, cache_dir=f"{CACHE_DIR}/", allow_clear=False)
        with pytest.raises(RuntimeError, match="not allowed to be cleared"):
            _cache._dangerous_clear(confirm=True)

        _cache = GCPCacheStore(
            bucket=BUCKET, cache_dir=f"{CACHE_DIR}/", allow_clear=True, read_only=False
        )
        _cache.put("README.md")
        assert _cache.exists("README.md")

        with pytest.raises(RuntimeError):
            _cache._dangerous_clear(confirm=False)

        _cache._dangerous_clear(confirm=True)
        assert not _cache.exists("README.md")
        assert _cache.size == 0

    # Download and upload logic is slightly different with show_progress=True
    # Run a simple smoke test to ensure get and put work as expected
    @pytest.mark.parametrize("show_progress", [False, True])
    def test_put_get(self, show_progress) -> None:
        _cache = GCPCacheStore(
            bucket=BUCKET,
            cache_dir=f"{CACHE_DIR}/",
            allow_clear=True,
            read_only=False,
            show_progress=show_progress,
        )
        _cache._dangerous_clear(confirm=True)

        # put from filename
        assert not _cache.exists("README.md")
        _cache.put("README.md")
        assert _cache.exists("README.md")

        _cache.put("README.md", "anotherfile.nc")
        assert _cache.exists("anotherfile.nc")

        # file doesn't exist
        with pytest.raises(FileNotFoundError, match="No file found"):
            _cache.put("test2.nc")

        # get from filename
        _path = _cache.get("README.md")
        assert _path == _cache._disk_cache.path("README.md")

        # read only
        _cache = GCPCacheStore(
            bucket=BUCKET, cache_dir=f"{CACHE_DIR}/", allow_clear=True, read_only=False
        )
        _cache.put("README.md", "anotherfile.nc")
        _cache = GCPCacheStore(
            bucket=BUCKET, cache_dir=f"{CACHE_DIR}/", allow_clear=True, read_only=True
        )
        with pytest.raises(RuntimeError, match="read only"):
            _cache.put("README.md", "anotherfile2.nc")
        _path = _cache.get("anotherfile.nc")
        assert "anotherfile.nc" in _path

        # clean up
        _cache._dangerous_clear(confirm=True)
