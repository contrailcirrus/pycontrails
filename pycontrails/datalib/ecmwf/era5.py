"""ECMWF ERA5 data access."""

from __future__ import annotations

import hashlib
import logging
import os
import pathlib
import warnings
from contextlib import ExitStack
from datetime import datetime
from typing import TYPE_CHECKING, Any

LOG = logging.getLogger(__name__)

import pandas as pd
import xarray as xr
from overrides import overrides

import pycontrails
from pycontrails.core import cache, datalib
from pycontrails.core.met import MetDataset, MetVariable
from pycontrails.datalib.ecmwf.common import ECMWFAPI, rad_accumulated_to_average
from pycontrails.datalib.ecmwf.variables import (
    PRESSURE_LEVEL_VARIABLES,
    SURFACE_VARIABLES,
    TOAIncidentSolarRadiation,
    TopNetSolarRadiation,
    TopNetThermalRadiation,
)
from pycontrails.utils.temp import temp_file

if TYPE_CHECKING:
    import cdsapi


class ERA5(ECMWFAPI):
    """Class to support ERA5 data access, download, and organization.

    Requires account with
    `Copernicus Data Portal <https://cds.climate.copernicus.eu/cdsapp#!/home>`_
    and local credentials.

    API credentials can be stored in a ``~/.cdsapirc`` file
    or as ``CDSAPI_URL`` and ``CDSAPI_KEY`` environment variables.

        export CDSAPI_URL=...
        export CDSAPI_KEY=...

    Credentials can also be provided directly ``url`` and ``key`` keyword args.

    See `cdsapi <https://github.com/ecmwf/cdsapi>`_ documentation
    for more information.

    Parameters
    ----------
    time : datalib.TimeInput | None
        The time range for data retrieval, either a single datetime or (start, end) datetime range.
        Input must be datetime-like or tuple of datetime-like
        (`datetime`, :class:`pd.Timestamp`, :class:`np.datetime64`)
        specifying the (start, end) of the date range, inclusive.
        Datafiles will be downloaded from CDS for each day to reduce requests.
        If None, ``paths`` must be defined and all time coordinates will be loaded from files.
    variables : datalib.VariableInput
        Variable name (i.e. "t", "air_temperature", ["air_temperature, relative_humidity"])
    pressure_levels : datalib.PressureLevelInput, optional
        Pressure levels for data, in hPa (mbar)
        Set to -1 for to download surface level parameters.
        Defaults to -1.
    paths : str | list[str] | pathlib.Path | list[pathlib.Path] | None, optional
        Path to CDS NetCDF files to load manually.
        Can include glob patterns to load specific files.
        Defaults to None, which looks for files in the :attr:`cachestore` or CDS.
    timestep_freq : str, optional
        Manually set the timestep interval within the bounds defined by :attr:`time`.
        Supports any string that can be passed to `pd.date_range(freq=...)`.
        By default, this is set to "1H" for reanalysis products and "3H" for ensemble products.
    product_type : str, optional
        Product type, one of "reanalysis", "ensemble_mean", "ensemble_members", "ensemble_spread"
    grid : float, optional
        Specify latitude/longitude grid spacing in data.
        By default, this is set to 0.25 for reanalysis products and 0.5 for ensemble products.
    cachestore : cache.CacheStore | None, optional
        Cache data store for staging ECMWF ERA5 files.
        Defaults to :class:`cache.DiskCacheStore`.
        If None, cache is turned off.
    url : str
        Override `cdsapi <https://github.com/ecmwf/cdsapi>`_ url
    key : str
        Override `cdsapi <https://github.com/ecmwf/cdsapi>`_ key

    Notes
    -----
    ERA5 parameter list:
    https://confluence.ecmwf.int/pages/viewpage.action?pageId=82870405#ERA5:datadocumentation-Parameterlistings

    All accumulated radiative quantities are converted to average instantaneous quantities.
    See https://www.ecmwf.int/sites/default/files/elibrary/2015/18490-radiation-quantities-ecmwf-model-and-mars.pdf
    for more information.

    Local ``paths`` are loaded using :func:`xarray.open_mfdataset`.
    Pass ``xr_kwargs`` inputs to :meth:`open_metdataset` to customize file loading.

    Examples
    --------
    >>> from datetime import datetime
    >>> from pycontrails.datalib.ecmwf import ERA5
    >>> from pycontrails import GCPCacheStore

    >>> # Store data files from CDS to local disk (default behavior)
    >>> era5 = ERA5(
    ...     "2020-06-01 12:00:00",
    ...     variables=["air_temperature", "relative_humidity"],
    ...     pressure_levels=[350, 300]
    ... )

    >>> # cache files to google cloud storage
    >>> gcp_cache = cache.GCPCacheStore(bucket="contrails-301217-unit-test", cache_dir="ecmwf")
    >>> era5 = ERA5(
    ...     "2020-06-01 12:00:00",
    ...     variables=["air_temperature", "relative_humidity"],
    ...     pressure_levels=[350, 300],
    ...     cachestore=gcp_cache
    ... )
    """  # noqa: E501

    __slots__ = (
        "product_type",
        "cds",
        "url",
        "key",
    )

    #: Product type, one of "reanalysis", "ensemble_mean", "ensemble_members", "ensemble_spread"
    product_type: str

    #: Handle to ``cdsapi.Client``
    cds: cdsapi.Client

    #: User provided ``cdsapi.Client`` url
    url: str | None

    #: User provided ``cdsapi.Client`` url
    key: str | None

    __marker = object()

    def __init__(
        self,
        time: datalib.TimeInput | None,
        variables: datalib.VariableInput,
        pressure_levels: datalib.PressureLevelInput = -1,
        paths: str | list[str] | pathlib.Path | list[pathlib.Path] | None = None,
        timestep_freq: str = "1H",
        product_type: str = "reanalysis",
        grid: float = 0.25,
        cachestore: cache.CacheStore | None = __marker,  # type: ignore[assignment]
        url: str | None = None,
        key: str | None = None,
    ):
        # inputs
        self.product_type = product_type
        if cachestore is self.__marker:
            cachestore = cache.DiskCacheStore()
        self.cachestore = cachestore
        self.paths = paths
        self.url = url or os.environ.get("CDSAPI_URL", None)
        self.key = key or os.environ.get("CDSAPI_KEY", None)

        if time is None and paths is None:
            raise ValueError("Time input is required when paths is None")

        product_types = ("reanalysis", "ensemble_mean", "ensemble_members", "ensemble_spread")
        if product_type not in product_types:
            raise ValueError(
                f"Unknown product_type {product_type}. "
                f"Currently support product types: {', '.join(product_types)}"
            )

        if product_type == "reanalysis":
            grid_min = 0.25
        else:
            grid_min = 0.5

        msg = (
            f"The smallest resolution available through the CDS API is {grid_min} degrees. "
            f"Your downloaded data will have resolution {grid}, but it is a reinterpolation "
            f"of the {grid_min} degree data. The same interpolation can be achieved "
            "directly with `xarray`."
        )

        # process inputs
        if "ensemble" in product_type:
            # If using a nonstandard product and default timestep_freq, override default
            if timestep_freq == "1H":
                timestep_freq = "3H"

            # And taking the same procedure for grid
            if grid == 0.25:
                grid = 0.5
            elif grid < 0.5:
                warnings.warn(msg)

        elif grid < 0.25:
            warnings.warn(msg)

        self.grid = grid
        self.timesteps = datalib.parse_timesteps(time, freq=timestep_freq)
        self.pressure_levels = datalib.parse_pressure_levels(
            pressure_levels, self.supported_pressure_levels
        )
        self.variables = datalib.parse_variables(variables, self.supported_variables)

        # ensemble_mean, etc - time is only available on the 0, 3, 6, etc
        if "ensemble" in product_type:
            if {t.hour for t in self.timesteps} - {0, 3, 6, 9, 12, 15, 18, 21}:
                raise NotImplementedError("Ensemble products only support every three hours")

    def __repr__(self) -> str:
        base = super().__repr__()
        return f"{base}\n\tDataset: {self.dataset}\n\tProduct type: {self.product_type}"

    @property
    def hash(self) -> str:
        """Generate a unique hash for this datasource.

        Returns
        -------
        str
            Unique hash for met instance (sha1)
        """
        hashstr = (
            f"{self.__class__.__name__}{self.timesteps}{self.variable_shortnames}"
            f"{self.pressure_levels}{self.grid}{self.product_type}"
        )
        return hashlib.sha1(bytes(hashstr, "utf-8")).hexdigest()

    @property
    def pressure_level_variables(self) -> list[MetVariable]:
        """ECMWF pressure level parameters.

        Returns
        -------
        list[MetVariable] | None
            List of MetVariable available in datasource
        """
        return PRESSURE_LEVEL_VARIABLES

    @property
    def single_level_variables(self) -> list[MetVariable]:
        """ECMWF surface level parameters.

        Returns
        -------
        list[MetVariable] | None
            List of MetVariable available in datasource
        """
        return SURFACE_VARIABLES

    @property
    def supported_pressure_levels(self) -> list[int]:
        """Get pressure levels available from ERA5 pressure level dataset.

        Returns
        -------
        list[int]
            List of integer pressure level values
        """
        return [
            1000,
            975,
            950,
            925,
            900,
            875,
            850,
            825,
            800,
            775,
            750,
            700,
            650,
            600,
            550,
            500,
            450,
            400,
            350,
            300,
            250,
            225,
            200,
            175,
            150,
            125,
            100,
            70,
            50,
            30,
            20,
            10,
            7,
            5,
            3,
            2,
            1,
            -1,
        ]

    @property
    def dataset(self) -> str:
        """Select dataset for download based on :attr:`pressure_levels`.

        One of "reanalysis-era5-pressure-levels" or "reanalysis-era5-single-levels"

        Returns
        -------
        str
            ERA5 dataset name in CDS
        """
        if self.pressure_levels != [-1]:
            return "reanalysis-era5-pressure-levels"
        return "reanalysis-era5-single-levels"

    def create_cachepath(self, t: datetime | pd.Timestamp) -> str:
        """Return cachepath to local ERA5 data file based on datetime.

        This uniquely defines a cached data file ith class parameters.

        Parameters
        ----------
        t : datetime | pd.Timestamp
            Datetime of datafile

        Returns
        -------
        str
            Path to local ERA5 data file
        """
        if self.cachestore is None:
            raise ValueError("self.cachestore attribute must be defined to create cache path")

        datestr = t.strftime("%Y%m%d-%H")

        # set date/time for file
        if self.pressure_levels == [-1]:
            suffix = f"era5sl{self.grid}{self.product_type}"
        else:
            suffix = f"era5pl{self.grid}{self.product_type}"

        # return cache path
        return self.cachestore.path(f"{datestr}-{suffix}.nc")

    @overrides
    def download_dataset(self, times: list[datetime]) -> None:
        download_times: dict[datetime, list[datetime]] = {}
        for t in times:
            unique_day = datetime(t.year, t.month, t.day)
            download_times.setdefault(unique_day, [])

            download_times[unique_day].append(t)

        # download data file for each unique day
        LOG.debug(f"Downloading ERA5 dataset for times {times}")
        for unique_day in download_times:
            self._download_file(download_times[unique_day])

    @overrides
    def cache_dataset(self, dataset: xr.Dataset) -> None:
        if self.cachestore is None:
            LOG.debug("Cache is turned off, skipping")
            return

        with ExitStack() as stack:
            # group by hour and save one dataset for each hour to temp file
            time_group, datasets = zip(*dataset.groupby("time", squeeze=False))

            xarray_temp_filenames = [stack.enter_context(temp_file()) for _ in time_group]
            xr.save_mfdataset(datasets, xarray_temp_filenames)

            # put each hourly file into cache
            cache_path = [
                self.create_cachepath(datetime.utcfromtimestamp(tg.tolist() / 1e9))
                for tg in time_group
            ]
            self.cachestore.put_multiple(xarray_temp_filenames, cache_path)

    @overrides
    def open_metdataset(
        self,
        dataset: xr.Dataset | None = None,
        xr_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> MetDataset:
        xr_kwargs = xr_kwargs or {}

        #  short-circuit dataset or file paths if provided
        if dataset is not None:
            ds = self._preprocess_era5_dataset(dataset)

        # load from local paths
        elif self.paths is not None:
            ds = self._open_and_cache(xr_kwargs)

        # load from cache or download
        else:
            if self.cachestore is None:
                raise ValueError("Cachestore is required to download data")

            # confirm files are downloaded from CDS or MARS
            self.download(**xr_kwargs)

            # ensure all files are guaranteed to be available locally here
            # this would download a file from a remote (e.g. GCP) cache
            disk_cachepaths = [self.cachestore.get(f) for f in self._cachepaths]

            # run MetDataset constructor
            ds = self.open_dataset(disk_cachepaths, **xr_kwargs)

            # TODO: corner case
            # If any files are already cached, they will not have the version attached
            ds.attrs.setdefault("pycontrails_version", pycontrails.__version__)

        # run the same ECMWF-specific processing on the dataset
        mds = self._process_dataset(ds, **kwargs)

        # convert accumulated radiation values to average instantaneous values
        # set minimum for all values to 0

        # accumulations are 3 hours for ensembles, 1 hour for reanalysis
        if "ensemble" in self.product_type:
            dt_accumulation = 3 * 60 * 60
        else:
            dt_accumulation = 60 * 60

        for key in (
            TOAIncidentSolarRadiation.standard_name,
            TopNetSolarRadiation.standard_name,
            TopNetThermalRadiation.standard_name,
        ):
            if key in mds.data:
                rad_accumulated_to_average(mds, key, dt_accumulation)

        return mds

    def _open_and_cache(self, xr_kwargs: dict[str, Any]) -> xr.Dataset:
        """Open and cache :class:`xr.Dataset` from :attr:`self.paths`.

        Parameters
        ----------
        xr_kwargs : dict[str, Any]
            Additional kwargs passed directly to :func:`xarray.open_mfdataset`.
            See :meth:`open_metdataset`.

        Returns
        -------
        xr.Dataset
            Dataset opened from local paths.
        """

        if self.paths is None:
            raise ValueError("Attribute `self.paths` must be defined to open and cache")

        # if timesteps are defined and all timesteps are cached already
        # then we can skip loading
        if self.timesteps and self.cachestore and not self.list_timesteps_not_cached(**xr_kwargs):
            LOG.debug("All timesteps already in cache store")
            disk_cachepaths = [self.cachestore.get(f) for f in self._cachepaths]
            return self.open_dataset(disk_cachepaths, **xr_kwargs)

        ds = self.open_dataset(self.paths, **xr_kwargs)
        ds = self._preprocess_era5_dataset(ds)
        self.cache_dataset(ds)

        return ds

    def _download_file(self, times: list[datetime]) -> None:
        """Download data file for specific sets of times for *unique date* from CDS API.

        Splits datafiles by the hour and saves each hour in the cache datastore.
        Overwrites files if they already exists.

        Parameters
        ----------
        times : list[datetime]
            Times to download from single day
        """

        # set date/time for file
        date_str = times[0].strftime("%Y-%m-%d")

        # check to make sure times are all on the same day
        if any(dt.strftime("%Y-%m-%d") != date_str for dt in times):
            raise ValueError("All times must be on the same date when downloading from CDS")

        time_strs = [t.strftime("%H:%M") for t in times]

        # make request of cdsapi
        request: dict[str, Any] = {
            "product_type": self.product_type,
            "variable": self.variable_shortnames,
            "date": date_str,
            "time": time_strs,
            "grid": [self.grid, self.grid],
            "format": "netcdf",
        }
        if self.dataset == "reanalysis-era5-pressure-levels":
            request["pressure_level"] = self.pressure_levels

        # Open ExitStack to control temp_file context manager
        with ExitStack() as stack:
            # hold downloaded file in named temp file
            cds_temp_filename = stack.enter_context(temp_file())

            LOG.debug(f"Performing CDS request: {request} to dataset {self.dataset}")
            if not hasattr(self, "cds"):
                try:
                    import cdsapi
                except ModuleNotFoundError as e:
                    raise ModuleNotFoundError(
                        "Some `ecmwf` module dependencies are missing. "
                        "Please install all required dependencies using `pip install -e .[ecmwf]`"
                    ) from e

                try:
                    self.cds = cdsapi.Client(url=self.url, key=self.key)
                # cdsapi throws base-level Exception
                except Exception as err:
                    raise CDSCredentialsNotFound from err

            self.cds.retrieve(self.dataset, request, cds_temp_filename)

            # open file, edit, and save for each hourly time step
            ds = stack.enter_context(
                xr.open_dataset(cds_temp_filename, engine=datalib.NETCDF_ENGINE)
            )

            # run preprocessing before cache
            ds = self._preprocess_era5_dataset(ds)

            self.cache_dataset(ds)

    def _preprocess_era5_dataset(self, ds: xr.Dataset) -> xr.Dataset:
        """Process ERA5 data before caching.

        Parameters
        ----------
        ds : xr.Dataset
            Loaded :class:`xr.Dataset`

        Returns
        -------
        xr.Dataset
            Processed :class:`xr.Dataset`
        """

        if "pycontrails_version" in ds.attrs:
            LOG.debug("Input dataset processed with pycontrails > 0.29")
            return ds

        # not pre-processed source file from `download` or `paths`

        # for "reanalysis-era5-single-levels" or if self.pressure_levels length == 1,
        # then the netcdf file does not contain the dimension "level"
        if len(self.pressure_levels) == 1:
            ds = ds.expand_dims({"level": self.pressure_levels})

        ds.attrs["pycontrails_version"] = pycontrails.__version__
        return ds


class CDSCredentialsNotFound(Exception):
    """Raise when CDS credentials are not found by `cdsapi.Client` instance."""
