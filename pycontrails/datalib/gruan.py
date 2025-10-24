"""Support for accessing `GRUAN <https://www.gruan.org/>`_ data over FTP."""

import datetime
import ftplib
import functools
import os
import tempfile

import xarray as xr

from pycontrails.core import cache

#: GRUAN FTP server address
FTP_SERVER = "ftp.ncdc.noaa.gov"

#: All available GRUAN products and sites on the FTP server as of 2025-10
#: This is simply the hardcoded output of :func:`available_sites` at that time to
#: avoid a lookup that changes infrequently.
AVAILABLE_PRODUCTS_TO_SITES = {
    "RS-11G-GDP.1": ["SYO", "TAT", "NYA", "LIN"],
    "RS41-EDT.1": ["LIN", "POT", "SNG"],
    "RS92-GDP.1": ["BOU", "CAB", "LIN", "PAY", "POT", "SOD", "TAT"],
    "RS92-GDP.2": [
        "BAR",
        "BEL",
        "BOU",
        "CAB",
        "DAR",
        "GRA",
        "LAU",
        "LIN",
        "MAN",
        "NAU",
        "NYA",
        "PAY",
        "POT",
        "REU",
        "SGP",
        "SOD",
        "TAT",
        "TEN",
        "GVN",
    ],
    "RS92-PROFILE-BETA.2": ["BOU", "CAB", "LIN", "POT", "SOD", "TAT"],
    "RS92-PROFILE-BETA.3": ["BOU", "CAB", "LIN", "POT", "SOD", "TAT"],
}


def extract_gruan_time(filename: str) -> tuple[datetime.datetime, int]:
    """Extract launch time and revision number from a GRUAN filename."""
    parts = filename.split("_")
    if len(parts) != 6:
        raise ValueError(f"Unexpected filename format: {filename}")
    time_part = parts[4]
    try:
       time = datetime.datetime.strptime(time_part, "%Y%m%dT%H%M%S")
    except:
       raise ValueError(f"Unexpected time segment: {time_part}")

    revision_part = parts[5].removesuffix(".nc")
    if not revision_part[-3:].isdigit():
        raise ValueError(f"Unexpected revision segment: {revision_part}")
    revision = int(revision_part[-3:])

    return time, revision


@functools.cache
def available_sites() -> dict[str, list[str]]:
    """Get a list of available GRUAN sites for each supported product."""

    base_path = "/pub/data/gruan/processing/level2"

    out = {}

    with ftplib.FTP(FTP_SERVER) as ftp:
        ftp.login()

        ftp.cwd(base_path)
        names = ftp.nlst()
        for name in names:
            try:
                ftp.cwd(f"{base_path}/{name}")
            except ftplib.error_perm:
                continue  # skip non-directories

            versions = ftp.nlst()
            for version in versions:
                ftp.cwd(f"{base_path}/{name}/{version}")
                sites = ftp.nlst()

                key = f"{name}.{int(version.split('-')[-1])}"
                out[key] = sites

    return out


class GRUAN:
    """Access `GRUAN <https://www.gruan.org/>`_ data over FTP.

    GRUAN is the Global Climate Observing System Reference Upper-Air Network. It provides
    high-quality measurements of atmospheric variables from ground to stratosphere
    through a global network of radiosonde stations.

    Parameters
    ----------
    product : str
        GRUAN data product. See :attr:`AVAILABLE` for available products. These currently
        include:
        - ``RS92-GDP.2``
        - ``RS92-GDP.1``
        - ``RS92-PROFILE-BETA.2``
        - ``RS92-PROFILE-BETA.3``
        - ``RS41-EDT.1``
        - ``RS-11G-GDP.1``
    site : str
        GRUAN station identifier. See :attr:`AVAILABLE` for available sites for each product.
    cachestore : CacheStore, optional
        Cache store to use for downloaded files. If not provided, a disk cache store
        will be created in the user cache directory under ``gruan/``. Set to ``None``
        to disable caching.

    Notes
    -----
    The FTP files have the following hierarchy::

        /pub/data/gruan/processing/level2/
            {product-root}/
                version-{NNN}/
                    {SITE}/
                        {YYYY}/
                            <filename>.nc

    - {product-root} is the product name without the trailing version integer (e.g. ``RS92-GDP``)
    - version-{NNN} zero-pads to three digits (suffix ``.2`` -> ``version-002``)
    - {SITE} is the station code (e.g. ``LIN``)
    - {YYYY} is launch year
    - Filenames encode launch time and revision (parsed by :func:`extract_gruan_time`)

    Discovery helpers methods:

        - :attr:`AVAILABLE` or :func:`available_sites` -> products and sites
        - :meth:`years` -> list available years for (product, site)
        - :meth:`list_files` -> list available NetCDF files for the given year
        - :meth:`get` -> download and open a single NetCDF file as an :class:`xarray.Dataset`

    Typical workflow:

        1. Inspect :attr:`AVAILABLE` (fast) or call :func:`available_sites` (live)
        2. Instantiate ``GRUAN(product, site)``
        3. Call ``years()``
        4. Call ``list_files(year)``
        5. Call ``get(filename)`` for an ``xarray.Dataset``

    """

    # Convenience access to available sites
    available_sites = staticmethod(available_sites)
    AVAILABLE = AVAILABLE_PRODUCTS_TO_SITES

    __slots__ = ("_ftp", "cachestore", "product", "site")

    __marker = object()

    def __init__(
        self,
        product: str,
        site: str,
        cachestore: cache.CacheStore | None = __marker,  # type: ignore[assignment]
    ) -> None:
        known = AVAILABLE_PRODUCTS_TO_SITES

        if product not in known:
            known = available_sites()  # perhaps AVAILABLE_PRODUCTS_TO_SITES is outdated
            if product not in known:
                raise ValueError(f"Unknown GRUAN product: {product}. Known products: {list(known)}")
        self.product = product

        if site not in known[product]:
            known = available_sites()  # perhaps AVAILABLE_PRODUCTS_TO_SITES is outdated
            if site not in known[product]:
                raise ValueError(
                    f"Unknown GRUAN site '{site}' for product '{product}'. "
                    f"Known sites: {known[product]}"
                )
        self.site = site

        if cachestore is self.__marker:
            cache_root = cache._get_user_cache_dir()
            cache_dir = f"{cache_root}/gruan"
            cachestore = cache.DiskCacheStore(cache_dir=cache_dir)
        self.cachestore = cachestore

        self._ftp: ftplib.FTP | None = None

    def __repr__(self) -> str:
        return f"GRUAN(product='{self.product}', site='{self.site}')"

    def connect(self) -> ftplib.FTP:
        """Connect to the GRUAN FTP server."""
        if self._ftp is None or self._ftp.sock is None:
            self._ftp = ftplib.FTP(FTP_SERVER)
            self._ftp.login()
            return self._ftp

        try:
            self._ftp.pwd()  # check if connection is still alive
        except (*ftplib.all_errors, ConnectionError):  # type: ignore[misc]
            # If we encounter any error, reset the connection and retry
            self._ftp = None
            return self.connect()
        return self._ftp

    @property
    def base_path_product(self) -> str:
        """Get the base path for GRUAN data product on the FTP server."""
        product, version = self.product.rsplit(".")
        return f"/pub/data/gruan/processing/level2/{product}/version-{version.zfill(3)}"

    @property
    def base_path_site(self) -> str:
        """Get the base path for GRUAN data site on the FTP server."""
        return f"{self.base_path_product}/{self.site}"

    def years(self) -> list[int]:
        """Get a list of available years for the selected product and site."""
        ftp = self.connect()
        ftp.cwd(self.base_path_site)
        years = ftp.nlst()
        return [int(year) for year in years]

    def list_files(self, year: int) -> list[str]:
        """List available files for a given year."""
        path = f"{self.base_path_site}/{year}"

        ftp = self.connect()
        try:
            ftp.cwd(path)
        except ftplib.error_perm as e:
            available = self.years()
            if year not in available:
                msg = f"No data available for year {year}. Available years are: {available}"
                raise ValueError(msg) from e
            raise
        return ftp.nlst()

    def get(self, filename: str) -> xr.Dataset:
        """Download a GRUAN dataset by filename."""
        if self.cachestore is None:
            return self._get_no_cache(filename)
        return self._get_with_cache(filename)

    def _get_no_cache(self, filename: str) -> xr.Dataset:
        t, _ = extract_gruan_time(filename)
        path = f"{self.base_path_site}/{t.year}/{filename}"

        ftp = self.connect()
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            ftp.retrbinary(f"RETR {path}", tmp.write)

        try:
            return xr.load_dataset(tmp.name)
        finally:
            os.remove(tmp.name)

    def _get_with_cache(self, filename: str) -> xr.Dataset:
        if self.cachestore is None:
            raise ValueError("Cachestore is not configured.")

        lpath = self.cachestore.path(filename)
        if self.cachestore.exists(lpath):
            return xr.open_dataset(lpath)

        t, _ = extract_gruan_time(filename)
        path = f"{self.base_path_site}/{t.year}/{filename}"

        ftp = self.connect()
        with open(lpath, "wb") as f:
            ftp.retrbinary(f"RETR {path}", f.write)

        return xr.open_dataset(lpath)
