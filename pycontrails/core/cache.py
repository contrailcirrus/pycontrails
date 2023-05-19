"""Pycontrails Caching Support."""

from __future__ import annotations

import logging
import os
import pathlib
import shutil
import warnings
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Sequence

logger = logging.getLogger(__name__)

from overrides import overrides

# optional imports
if TYPE_CHECKING:
    import google


def _get_user_cache_dir() -> str:
    try:
        import platformdirs
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Using the pycontrails CacheStore requires the 'platformdirs' package. "
            "This can be installed with 'pip install pycontrails[ecmwf]' or "
            "'pip install platformdirs'."
        ) from e
    return platformdirs.user_cache_dir("pycontrails")


class CacheStore(ABC):
    """Abstract cache storage class for storing staged and intermediate data."""

    __slots__ = ("cache_dir", "allow_clear")
    cache_dir: str
    allow_clear: bool

    @property
    @abstractmethod
    def size(self) -> float:
        """Return the disk size (in MBytes) of the local cache.

        Returns
        -------
        float
            Size of the disk cache store in MB

        Examples
        --------
        >>> from pycontrails import DiskCacheStore
        >>> disk_cache = DiskCacheStore(cache_dir="cache", allow_clear=True)
        >>> disk_cache.size
        0.0...

        >>> disk_cache.clear()  # cleanup

        >>> from pycontrails import GCPCacheStore
        >>> gcp_cache = GCPCacheStore(bucket="contrails-301217-unit-test", cache_dir="cache")
        >>> gcp_cache.size
        0.0...

        """

    @abstractmethod
    def listdir(self, path: str = "") -> list[str]:
        """List the contents of a directory in the cache.

        Parameters
        ----------
        path : str
            Path to the directory to list

        Returns
        -------
        list[str]
            List of files in the directory
        """

    @abstractmethod
    def path(self, cache_path: str) -> str:
        """Return a full filepath in cache.

        Parameters
        ----------
        cache_path : str
            string path or filepath to create in cache
            If parent directories do not exist, they will be created.

        Returns
        -------
        str
            Full path string to subdirectory directory or object in cache directory

        Examples
        --------
        >>> from pycontrails import DiskCacheStore
        >>> disk_cache = DiskCacheStore(cache_dir="cache", allow_clear=True)
        >>> disk_cache.path("file.nc")
        'cache/file.nc'

        >>> disk_cache.clear()  # cleanup

        >>> from pycontrails import GCPCacheStore
        >>> gcp_cache = GCPCacheStore(bucket="contrails-301217-unit-test", cache_dir="cache")
        >>> gcp_cache.path("file.nc")
        'cache/file.nc'
        """

    @abstractmethod
    def exists(self, cache_path: str) -> bool:
        """Check if a path in cache exists.

        Parameters
        ----------
        cache_path : str
            Path to directory or file in cache

        Returns
        -------
        bool
            True if directory or file exists

        Examples
        --------
        >>> from pycontrails import DiskCacheStore
        >>> disk_cache = DiskCacheStore(cache_dir="cache", allow_clear=True)
        >>> disk_cache.exists("file.nc")
        False

        >>> from pycontrails import GCPCacheStore
        >>> gcp_cache = GCPCacheStore(bucket="contrails-301217-unit-test", cache_dir="cache")
        >>> gcp_cache.exists("file.nc")
        False
        """

    def put_multiple(
        self, data_path: Sequence[str | pathlib.Path], cache_path: list[str]
    ) -> list[str]:
        """Put multiple files into the cache at once.

        Parameters
        ----------
        data_path : Sequence[str | pathlib.Path]
            List of data files to cache.
            Each member is passed directly on to :meth:`put`.
        cache_path : list[str]
            List of cache paths corresponding to each element in the ``data_path`` list.
            Each member is passed directly on to :meth:`put`.

        Returns
        -------
        list[str]
            Returns a list of relative paths to the stored files in the cache
        """

        # TODO: run in parallel?
        return [self.put(d, cp) for d, cp in zip(data_path, cache_path)]

    # In the three methods below, child classes have a complete docstring.

    @abstractmethod
    def put(self, data: str | pathlib.Path, cache_path: str | None = None) -> str:
        """Save data to cache."""

    @abstractmethod
    def get(self, cache_path: str) -> str:
        """Get data from cache."""


class DiskCacheStore(CacheStore):
    """Cache that uses a folder on the local filesystem.

    Parameters
    ----------
    allow_clear : bool, optional
        Allow this cache to be cleared using :meth:`clear()`. Defaults to False.
    cache_dir : str | pathlib.Path, optional
        Root cache directory.
        By default, looks first for ``PYCONTRAILS_CACHE_DIR`` environment variable,
        then uses the OS specific :func:`platformdirs.user_cache_dir` function.

    Examples
    --------
    >>> from pycontrails import DiskCacheStore
    >>> disk_cache = DiskCacheStore(cache_dir="cache", allow_clear=True)
    >>> disk_cache.cache_dir
    'cache'

    >>> disk_cache.clear()  # cleanup
    """

    def __init__(
        self,
        cache_dir: str | pathlib.Path | None = None,
        allow_clear: bool = False,
    ):
        if cache_dir is None:
            # Use a try / except to avoid unnecessary import of platformdirs
            try:
                cache_dir = os.environ["PYCONTRAILS_CACHE_DIR"]
            except KeyError:
                cache_dir = _get_user_cache_dir()

        # make sure local cache directory exists
        pathlib.Path(cache_dir).mkdir(parents=True, exist_ok=True)

        # store root cache dir
        self.cache_dir = str(cache_dir)

        # allow the cache to be clear or not
        self.allow_clear = allow_clear

    def __repr__(self) -> str:
        return f"DiskCacheStore: {self.cache_dir}"

    @property
    @overrides
    def size(self) -> float:
        disk_path = pathlib.Path(self.cache_dir)
        size = sum(f.stat().st_size for f in disk_path.rglob("*") if f.is_file())
        logger.debug("Disk cache size %s bytes", size)
        return size / 1e6

    @overrides
    def listdir(self, path: str = "") -> list[str]:
        path = self.path(path)
        iter_ = pathlib.Path(path).iterdir()
        out = [str(f.relative_to(path)) for f in iter_]
        out.sort()
        return out

    @overrides
    def path(self, cache_path: str) -> str:
        if str(cache_path).startswith(str(self.cache_dir)):
            disk_path = pathlib.Path(cache_path)
        else:
            disk_path = pathlib.Path(self.cache_dir) / pathlib.Path(cache_path)

        # make sure full path to parents exist
        disk_path.parent.mkdir(parents=True, exist_ok=True)

        return str(disk_path)

    @overrides
    def exists(self, cache_path: str) -> bool:
        disk_path = pathlib.Path(self.path(cache_path))
        return disk_path.is_dir() or disk_path.is_file()

    def put(self, data_path: str | pathlib.Path, cache_path: str | None = None) -> str:
        """Save data to the local cache store.

        Parameters
        ----------
        data_path : str | pathlib.Path
            Path to data to cache.
        cache_path : str | None, optional
            Path in cache store to save data
            Defaults to the same filename as ``data_path``

        Returns
        -------
        str
            Returns the relative path in the cache to the stored file

        Raises
        ------
        FileNotFoundError
            Raises if `data` is a string and a file is not found at the string

        Examples
        --------
        >>> from pycontrails import DiskCacheStore
        >>> disk_cache = DiskCacheStore(cache_dir="cache", allow_clear=True)
        >>>
        >>> # put a file directly
        >>> disk_cache.put("README.md", "test/file.md")
        'test/file.md'
        """

        if not pathlib.Path(data_path).is_file():
            raise FileNotFoundError(f"No file found at path {data_path}")

        if cache_path is None:
            cache_path = pathlib.Path(data_path).name

        disk_path = self.path(str(cache_path))

        # copy to disk cache
        logger.debug("Disk cache put %s to %s in disk cache", data_path, cache_path)
        try:
            shutil.copyfile(data_path, disk_path)
        except PermissionError:
            logger.warning(
                "Permission error copying %s to %s. The destination file may already be open.",
                data_path,
                disk_path,
            )

        return cache_path

    def get(self, cache_path: str) -> str:
        """Get data path from the local cache store.

        Alias for :meth:`path()`

        Parameters
        ----------
        cache_path : str
            Cache path to retrieve

        Returns
        -------
        str
            Returns the relative path in the cache to the stored file

        Examples
        --------
        >>> from pycontrails import DiskCacheStore
        >>> disk_cache = DiskCacheStore(cache_dir="cache", allow_clear=True)
        >>>
        >>> # returns a path
        >>> disk_cache.get("test/file.md")
        'cache/test/file.md'
        """
        return self.path(cache_path)

    def clear(self, cache_path: str = "") -> None:
        """Delete all files and folders within ``cache_path``.

        If no ``cache_path`` is provided, this will clear the entire cache.

        If :attr:`allow_clear` is set to ``False``, this method will do nothing.

        Parameters
        ----------
        cache_path : str, optional
            Path to subdirectory or file in cache

        Raises
        ------
        RuntimeError
            Raises a RuntimeError when :attr:`allow_clear` is set to ``False``

        Examples
        --------
        >>> from pycontrails import DiskCacheStore
        >>> disk_cache = DiskCacheStore(cache_dir="cache", allow_clear=True)

        >>> # Write some data to the cache
        >>> disk_cache.put("README.md", "test/example.txt")
        'test/example.txt'

        >>> disk_cache.exists("test/example.txt")
        True

        >>> # clear a specific path
        >>> disk_cache.clear("test/example.txt")

        >>> # clear the whole cache
        >>> disk_cache.clear()
        """
        if not self.allow_clear:
            raise RuntimeError("Cache is not allowed to be cleared")

        disk_path = pathlib.Path(self.path(cache_path))

        if disk_path.is_file():
            logger.debug("Remove file at path %s", disk_path)
            disk_path.unlink()
            return

        # Assume anything else is a directory
        if disk_path.exists():
            # rm directory recursively
            logger.debug("Remove directory at path %s", disk_path)
            shutil.rmtree(disk_path, ignore_errors=True)
            return

        warnings.warn(f"No cache path found at {disk_path}")


class GCPCacheStore(CacheStore):
    """Google Cloud Platform (Storage) Cache.

    This class downloads files from Google Cloud Storage locally to a :class:`DiskCacheStore`
    initialized with ``cache_dir=".gcp"`` to avoid re-downloading files. If the source files
    on GCP changes, the local mirror of the GCP DiskCacheStore must be cleared by initializing
    this class and running :meth:`clear_disk()`.

    Note by default, GCP Cache Store is *read only*.
    When a :meth:`put` is called and :attr:`read_only` is set to *True*,
    the cache will throw an ``RuntimeError`` error.
    Set ``read_only`` to *False* to enable writing to cache store.

    Parameters
    ----------
    cache_dir : str, optional
        Root object prefix within :attr:`bucket`
        Defaults to ``PYCONTRAILS_CACHE_DIR`` environment variable, or the root of the bucket.
        The full GCP URI (ie, `"gs://<MY_BUCKET>/<PREFIX>"`) can be used here.
    project : str , optional
        GCP Project.
        Defaults to the current active project set in the `google-cloud-sdk` environment
    bucket : str, optional
        GCP Bucket to use for cache.
        Defaults to ``PYCONTRAILS_CACHE_BUCKET`` environment variable.
    read_only : bool, optional
        Only enable reading from cache. Defaults to ``True``.
    allow_clear : bool, optional
        Allow this cache to be cleared using :meth:`clear()`. Defaults to ``False``.
    disk_cache : DiskCacheStore, optional
        Specify a custom local disk cache store to mirror files.
        Defaults to :class:`DiskCacheStore(cache_dir="{user_cache_dir}/.gcp/{bucket}")`
    show_progress : bool, optional
        Show progress bar on cache :meth:`put`.
        Defaults to False
    chunk_size : int, optional
        Chunk size for uploads and downloads with progress. Set a larger size to see more granular
        progress, and set a smaller size for more optimal download speed. Chunk size must be a
        multiple of 262144 (ie, 10 * 262144). Default value is 8 * 262144, which will throttle
        fast download speeds.


    Examples
    --------
    >>> from pycontrails import GCPCacheStore
    >>> cache = GCPCacheStore(bucket="contrails-301217-unit-test", cache_dir="cache")
    >>> cache.cache_dir
    'cache/'
    >>> cache.bucket
    'contrails-301217-unit-test'
    """

    __slots__ = (
        "project",
        "bucket",
        "read_only",
        "timeout",
        "show_progress",
        "chunk_size",
        "_disk_cache",
        "_client",
        "_bucket",
    )
    project: str | None
    bucket: str
    read_only: bool
    timeout: int
    show_progress: bool
    chunk_size: int
    _disk_cache: DiskCacheStore
    _client: "google.cloud.storage.Client"
    _bucket: "google.cloud.storage.Bucket"

    def __init__(
        self,
        cache_dir: str = os.getenv("PYCONTRAILS_CACHE_DIR", ""),
        project: str | None = None,
        bucket: str | None = os.getenv("PYCONTRAILS_CACHE_BUCKET"),
        disk_cache: DiskCacheStore | None = None,
        read_only: bool = True,
        allow_clear: bool = False,
        timeout: int = 300,
        show_progress: bool = False,
        chunk_size: int = 64 * 262144,
    ):
        try:
            from google.cloud import storage
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "GCPCache requires the `google-cloud-storage` module, which can be installed "
                "using `pip install pycontrails[gcp]`"
            ) from e

        if "https://" in cache_dir:
            raise ValueError(
                "`cache_dir` should only specify base object path within the GCS bucket. "
                "Expect not to find prefix `https://` in parameter `cache_dir`. "
                f"Found `cache_dir={cache_dir}`."
            )

        # support cache_dir paths that refer to the whole GCP URI path
        if "gs://" in cache_dir:
            bucket_and_cache_dir = cache_dir.split("gs://")[1]
            split_path = bucket_and_cache_dir.split("/", maxsplit=1)
            if len(split_path) == 1:
                uri_bucket = split_path[0]
                cache_dir = ""
            else:
                uri_bucket, cache_dir = split_path

            if bucket is None:
                bucket = uri_bucket
            elif bucket != uri_bucket:
                raise ValueError(
                    f"Found conflicting bucket names: {uri_bucket} in URI path "
                    f"and {bucket} as parameter."
                )

        # TODO: Not sure if we want this ....
        # Do we want to correct for parameters bucket=None and cache_dir=BUCKET/PREFIX?
        # if bucket in cache_dir:
        #     cache_dir = cache_dir.split(f"{bucket}/")[1]

        # raise if bucket is still not defined
        if bucket is None:
            raise ValueError(
                "Parameter `bucket` not specified. Either pass parameter `bucket`, pass a URI "
                "path for `cache_dir`, or set environment variable `PYCONTRAILS_CACHE_BUCKET`"
            )

        # append a "/" for GCP objects
        if cache_dir and not cache_dir.endswith("/"):
            cache_dir = f"{cache_dir}/"

        # set up gcp client
        self._client = storage.Client(project=project)

        # create bucket object and make sure bucket exists
        self._bucket = self._client.bucket(bucket)

        # store root bucket/cache dir
        self.project = project
        self.bucket = bucket
        self.cache_dir = cache_dir

        # read only
        self.read_only = read_only

        # allow the cache to be cleared or not
        self.allow_clear = allow_clear

        # parameters for GCP storage upload
        self.timeout = timeout
        self.show_progress = show_progress
        self.chunk_size = chunk_size

        # set up local DiskCache mirror
        # this keeps a local copy of files so that files are not re-downloaded
        if disk_cache is not None:
            self._disk_cache = disk_cache
        else:
            local_cache_dir = _get_user_cache_dir()
            self._disk_cache = DiskCacheStore(
                cache_dir=f"{local_cache_dir}/.gcp/{bucket}", allow_clear=True
            )

    def __repr__(self) -> str:
        return f"GCPCacheStore: {self.bucket}/{self.cache_dir}"

    @property
    def client(self) -> "google.cloud.storage.Client":
        """Handle to Google Cloud Storage client.

        Returns
        -------
        :class:`google.cloud.storage.Client`
            Handle to Google Cloud Storage client
        """
        return self._client

    @property
    @overrides
    def size(self) -> float:
        # get list of blobs below this path
        blobs = self._bucket.list_blobs(prefix=self.cache_dir)
        size = sum(b.size for b in blobs)
        logger.debug("GCP cache size %s bytes", size)
        return size / 1e6

    @overrides
    def listdir(self, path: str = "") -> list[str]:
        # I don't necessarily think we want to implement this .... it might be
        # very slow if the bucket is large. BUT, it won't be slower than the size
        # method right above this.
        # I typically am more interested in calling self._disk_cache.listdir() to get
        # information about the local cache, which is why I include this
        # particular error message.
        raise NotImplementedError(
            "ls is not implemented for GCPCacheStore. Use ._disk_cache.listdir() to "
            "list files in the local disk cache."
        )

    @overrides
    def path(self, cache_path: str) -> str:
        if cache_path.startswith(self.cache_dir):
            return cache_path
        return f"{self.cache_dir}{cache_path}"

    def gs_path(self, cache_path: str) -> str:
        """Return a full Google Storage (gs://) URI to object.

        Parameters
        ----------
        cache_path : str
            string path to object in cache

        Returns
        -------
        str
            Google Storage URI (gs://) to object in cache

        Examples
        --------
        >>> from pycontrails import GCPCacheStore
        >>> cache = GCPCacheStore(bucket="contrails-301217-unit-test", cache_dir="cache")
        >>> cache.path("file.nc")
        'cache/file.nc'
        """
        bucket_path = self.path(cache_path)
        return f"gs://{self.bucket}/{bucket_path}"

    @overrides
    def exists(self, cache_path: str) -> bool:
        # see if file is in the mirror disk cache
        if self._disk_cache.exists(cache_path):
            return self._disk_cache.exists(cache_path)

        bucket_path = self.path(cache_path)
        blob = self._bucket.blob(bucket_path)

        return blob.exists()

    def put(self, data_path: str | pathlib.Path, cache_path: str | None = None) -> str:
        """Save data to the GCP cache store.

        If :attr:`read_only` is *True*, this method will return the path to the
        local disk cache store path.

        Parameters
        ----------
        data_path : str | pathlib.Path
            Data to save to GCP cache store.
        cache_path : str, optional
            Path in cache store to save data.
            Defaults to the same filename as ``data_path``.

        Returns
        -------
        str
            Returns the path in the cache to the stored file

        Raises
        ------
        RuntimeError
            Raises if :attr:`read_only` is True
        FileNotFoundError
            Raises if ``data`` is a string and a file is not found at the string

        Examples
        --------
        >>> from pycontrails import GCPCacheStore
        >>> cache = GCPCacheStore(
        ...     bucket="contrails-301217-unit-test",
        ...     cache_dir="cache",
        ...     read_only=False,
        ... )

        >>> # put a file directly
        >>> cache.put("README.md", "test/file.md")
        'test/file.md'
        """
        # store on disk path mirror -  will catch errors
        cache_path = self._disk_cache.put(data_path, cache_path)

        # read only
        if self.read_only:
            logger.debug(
                f"GCP Cache Store is read only. File put in local DiskCacheStore path: {cache_path}"
            )
            raise RuntimeError(
                f"GCP Cache Store {self.bucket}/{self.cache_dir} is read only. "
                "File put in local DiskCacheStore path: {cache_path}"
            )

        # get bucket and disk paths and blob
        bucket_path = self.path(cache_path)
        disk_path = self._disk_cache.path(cache_path)
        blob = self._bucket.blob(bucket_path)

        logger.debug("GCP Cache put %s to %s", disk_path, bucket_path)

        if self.show_progress:  # upload with pbar
            _upload_with_progress(blob, disk_path, self.timeout, chunk_size=self.chunk_size)
        else:  # upload from disk path
            blob.upload_from_filename(disk_path, timeout=self.timeout)

        return cache_path

    def get(self, cache_path: str) -> str:
        """Get data from the local cache store.

        Parameters
        ----------
        cache_path : str
            Path in cache store to get data

        Returns
        -------
        str
            Returns path to downloaded local file

        Raises
        ------
        ValueError
            Raises value error is ``cache_path`` refers to a directory

        Examples
        --------
        >>> import pathlib
        >>> from pycontrails import GCPCacheStore
        >>> cache = GCPCacheStore(
        ...     bucket="contrails-301217-unit-test",
        ...     cache_dir="cache",
        ...     read_only=False,
        ... )

        >>> cache.put("README.md", "example/file.md")
        'example/file.md'

        >>> # returns a full path to local copy of the file
        >>> path = cache.get("example/file.md")
        >>> pathlib.Path(path).is_file()
        True

        >>> pathlib.Path(path).read_text()[17:69]
        'Python library for modeling aviation climate impacts'
        """
        if cache_path.endswith("/"):
            raise ValueError("`cache_path` must not end with a /")

        # see if file is in the mirror disk cache
        if self._disk_cache.exists(cache_path):
            return self._disk_cache.get(cache_path)

        # download otherwise
        bucket_path = self.path(cache_path)
        disk_path = self._disk_cache.path(cache_path)

        blob = self._bucket.blob(bucket_path)
        if not blob.exists():
            raise ValueError(f"No object exists in cache at path {bucket_path}")

        logger.debug("GCP Cache GET from %s", bucket_path)

        if self.show_progress:
            _download_with_progress(
                gcp_cache=self,
                gcp_path=bucket_path,
                disk_path=disk_path,
                chunk_size=self.chunk_size,
            )
        else:
            blob.download_to_filename(disk_path)

        return self._disk_cache.get(disk_path)

    def clear_disk(self, cache_path: str = "") -> None:
        """Clear the local disk cache mirror of the GCP Cache Store.

        Parameters
        ----------
        cache_path : str, optional
            Path in mirrored cache store. Passed into :meth:`_disk_clear.clear`. By
            default, this method will clear the entire mirrored cache store.

        Examples
        --------
        >>> from pycontrails import GCPCacheStore
        >>> gcp_cache = GCPCacheStore(bucket="contrails-301217-unit-test", cache_dir="cache")
        >>> gcp_cache.clear_disk()
        """
        self._disk_cache.clear(cache_path)

    def _dangerous_clear(self, confirm: bool = False, cache_path: str = "") -> None:
        """Delete all files and folders within ``cache_path``.

        If no ``cache_path`` is provided, this will clear the entire cache.

        If :attr:`allow_clear` is set to ``False``, this method will do nothing.

        Parameters
        ----------
        confirm : bool, optional
            Must pass True to make this work
        cache_path : str, optional
            Path to sub-directory or file in cache

        Raises
        ------
        RuntimeError
            Raises a RuntimeError when :attr:`allow_clear` is set to ``False``
        """
        if not confirm or not self.allow_clear:
            raise RuntimeError("Cache is not allowed to be cleared")

        # get full path to clear
        bucket_path = self.path(cache_path)
        logger.debug("Clearing GCP cache at path %s", bucket_path)

        # clear disk mirror
        self.clear_disk()

        # get list of blobs below this path
        blobs = self._bucket.list_blobs(prefix=bucket_path)

        # clear blobs one at a time
        for blob in blobs:
            blob.delete()


def _upload_with_progress(blob: Any, disk_path: str, timeout: int, chunk_size: int) -> None:
    """Upload with `tqdm` progress bar.

    Adapted from
    https://github.com/googleapis/python-storage/issues/27#issuecomment-651468428.

    Parameters
    ----------
    blob : Any
        GCP blob to upload
    disk_path : str
        Path to local file.
    timeout : int
        Passed into `blob.upload_from_file`
    chunk_size : int
        Used to set :attr:`chunk_size` on `blob`.
    """
    try:
        from tqdm.auto import tqdm
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Method `put` requires the `tqdm` module, which can be installed using "
            "`pip install pycontrails[gcp]`. "
            "Alternatively, set instance attribute `show_progress=False`."
        ) from e

    # minimal possible chunk_size to allow nice progress bar
    blob.chunk_size = chunk_size

    with open(disk_path, "rb") as local_file:
        total_bytes = os.fstat(local_file.fileno()).st_size
        with tqdm.wrapattr(local_file, "read", total=total_bytes, desc="upload to GCP") as file_obj:
            blob.upload_from_file(file_obj, size=total_bytes, timeout=timeout)


def _download_with_progress(
    gcp_cache: GCPCacheStore, gcp_path: str, disk_path: str, chunk_size: int
) -> None:
    """Download with `tqdm` progress bar."""

    try:
        from google.resumable_media.requests import ChunkedDownload
        from tqdm.auto import tqdm
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Method `get` requires the `tqdm` and `google-cloud-storage` modules, "
            "which can be installed using `pip install pycontrails[gcp]`. "
            "Alternatively, set instance attribute `show_progress=False`."
        ) from e

    blob = gcp_cache._bucket.get_blob(gcp_path)
    url = blob._get_download_url(gcp_cache._client)
    description = f"Download {gcp_path}"

    with open(disk_path, "wb") as local_file:
        with tqdm.wrapattr(local_file, "write", total=blob.size, desc=description) as file_obj:
            download = ChunkedDownload(url, chunk_size, file_obj)
            transport = gcp_cache.client._http
            while not download.finished:
                download.consume_next_chunk(transport, timeout=gcp_cache.timeout)
