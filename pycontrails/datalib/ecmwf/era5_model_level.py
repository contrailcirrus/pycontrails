"""Model-level ERA5 data access.

This module supports

- Retrieving model-level ERA5 data by submitting MARS requests through the Copernicus CDS.
- Processing retrieved model-level files to produce netCDF files on target pressure levels.
- Local caching of processed netCDF files.
- Opening processed and cached files as a :class:`pycontrails.MetDataset` object.

Consider using :class:`pycontrails.datalib.ecmwf.ERA5ARCO`
to access model-level data from the nominal ERA5 reanalysis between 1959 and 2022.
:class:`pycontrails.datalib.ecmwf.ERA5ARCO` accesses data through Google's
`Analysis-Ready, Cloud Optimized ERA5 dataset <https://cloud.google.com/storage/docs/public-datasets/era5>`_
and has lower latency than this module, which retrieves data from the
`Copernicus Climate Data Store <https://cds.climate.copernicus.eu/#!/home>`_.
This module must be used to retrieve model-level data from ERA5 ensemble members
or for more recent dates.
"""

from __future__ import annotations

import collections
import concurrent.futures
import contextlib
import hashlib
import logging
import os
import sys
import threading
import warnings
from datetime import datetime
from typing import Any

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

LOG = logging.getLogger(__name__)

import pandas as pd
import xarray as xr

import pycontrails
from pycontrails.core import cache
from pycontrails.core.met import MetDataset, MetVariable
from pycontrails.datalib._met_utils import metsource
from pycontrails.datalib.ecmwf import model_levels as mlmod
from pycontrails.datalib.ecmwf.common import ECMWFAPI, CDSCredentialsNotFound
from pycontrails.datalib.ecmwf.variables import MODEL_LEVEL_VARIABLES
from pycontrails.utils import dependencies, temp

ALL_ENSEMBLE_MEMBERS = list(range(10))


class ERA5ModelLevel(ECMWFAPI):
    """Class to support model-level ERA5 data access, download, and organization.

    The interface is similar to :class:`pycontrails.datalib.ecmwf.ERA5`, which downloads
    pressure-level with much lower vertical resolution.

    Requires account with
    `Copernicus Data Portal <https://cds.climate.copernicus.eu/how-to-api>`_
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
    time : metsource.TimeInput | None
        The time range for data retrieval, either a single datetime or (start, end) datetime range.
        Input must be datetime-like or tuple of datetime-like
        (:py:class:`datetime.datetime`, :class:`pandas.Timestamp`, :class:`numpy.datetime64`)
        specifying the (start, end) of the date range, inclusive.
        NetCDF files will be downloaded from CDS in chunks no larger than 1 month
        for the nominal reanalysis and no larger than 1 day for ensemble members.
        This ensures that exactly one request is submitted per file on tape accessed.
        If None, ``paths`` must be defined and all time coordinates will be loaded from files.
    variables : metsource.VariableInput
        Variable name (i.e. "t", "air_temperature", ["air_temperature, specific_humidity"])
    pressure_levels : metsource.PressureLevelInput, optional
        Pressure levels for data, in hPa (mbar).
        To download surface-level parameters, use :class:`pycontrails.datalib.ecmwf.ERA5`.
        Defaults to pressure levels that match model levels at a nominal surface pressure.
    timestep_freq : str, optional
        Manually set the timestep interval within the bounds defined by :attr:`time`.
        Supports any string that can be passed to ``pd.date_range(freq=...)``.
        By default, this is set to "1h" for reanalysis products and "3h" for ensemble products.
    product_type : str, optional
        Product type, one of "reanalysis" and "ensemble_members". Unlike
        :class:`pycontrails.datalib.ecmwf.ERA5`, this class does not support direct access to the
        ensemble mean and spread, which are not available on model levels.
    grid : float, optional
        Specify latitude/longitude grid spacing in data.
        By default, this is set to 0.25 for reanalysis products and 0.5 for ensemble products.
    model_levels : list[int], optional
        Specify ECMWF model levels to include in MARS requests.
        By default, this is set to include all model levels.
    ensemble_members : list[int], optional
        Specify ensemble members to include.
        Valid only when the product type is "ensemble_members".
        By default, includes every available ensemble member.
    cachestore : cache.CacheStore | None, optional
        Cache data store for staging processed netCDF files.
        Defaults to :class:`pycontrails.core.cache.DiskCacheStore`.
        If None, cache is turned off.
    cache_download: bool, optional
        If True, cache downloaded model-level files rather than storing them in a temporary file.
        By default, False.
    url : str | None
        Override the default `cdsapi <https://github.com/ecmwf/cdsapi>`_ url.
        As of January 2025, the url for the `CDS Server <https://cds.climate.copernicus.eu>`_
        is "https://cds.climate.copernicus.eu/api". If None, the url is set
        by the ``CDSAPI_URL`` environment variable. If this is not defined, the
        ``cdsapi`` package will determine the url.
    key : str | None
        Override default `cdsapi <https://github.com/ecmwf/cdsapi>`_ key. If None,
        the key is set by the ``CDSAPI_KEY`` environment variable. If this is not defined,
        the ``cdsapi`` package will determine the key.
    """

    __marker = object()

    def __init__(
        self,
        time: metsource.TimeInput,
        variables: metsource.VariableInput,
        *,
        pressure_levels: metsource.PressureLevelInput | None = None,
        timestep_freq: str | None = None,
        product_type: str = "reanalysis",
        grid: float | None = None,
        model_levels: list[int] | None = None,
        ensemble_members: list[int] | None = None,
        cachestore: cache.CacheStore = __marker,  # type: ignore[assignment]
        cache_download: bool = False,
        url: str | None = None,
        key: str | None = None,
    ) -> None:
        self.cachestore = cache.DiskCacheStore() if cachestore is self.__marker else cachestore
        self.cache_download = cache_download

        self.paths = None

        self.url = url or os.getenv("CDSAPI_URL")
        self.key = key or os.getenv("CDSAPI_KEY")

        supported = ("reanalysis", "ensemble_members")
        if product_type not in supported:
            msg = (
                f"Unknown product_type {product_type}. "
                f"Currently support product types: {', '.join(supported)}"
            )
            raise ValueError(msg)
        self.product_type = product_type

        if product_type != "ensemble_members" and ensemble_members:
            msg = "No ensemble members available for reanalysis product type."
            raise ValueError(msg)
        if product_type == "ensemble_members" and not ensemble_members:
            ensemble_members = ALL_ENSEMBLE_MEMBERS
        self.ensemble_members = ensemble_members

        if grid is None:
            grid = 0.25 if product_type == "reanalysis" else 0.5
        else:
            grid_min = 0.25 if product_type == "reanalysis" else 0.5
            if grid < grid_min:
                msg = (
                    f"The highest resolution available is {grid_min} degrees. "
                    f"Your downloaded data will have resolution {grid}, but it is a "
                    f"reinterpolation of the {grid_min} degree data. The same interpolation can be "
                    "achieved directly with xarray."
                )
                warnings.warn(msg)
        self.grid = grid

        if model_levels is None:
            model_levels = list(range(1, 138))
        elif min(model_levels) < 1 or max(model_levels) > 137:
            msg = "Retrieval model_levels must be between 1 and 137, inclusive."
            raise ValueError(msg)
        self.model_levels = model_levels

        datasource_timestep_freq = "1h" if product_type == "reanalysis" else "3h"
        if timestep_freq is None:
            timestep_freq = datasource_timestep_freq
        if not metsource.validate_timestep_freq(timestep_freq, datasource_timestep_freq):
            msg = (
                f"Product {self.product_type} has timestep frequency of {datasource_timestep_freq} "
                f"and cannot support requested timestep frequency of {timestep_freq}."
            )
            raise ValueError(msg)

        self.timesteps = metsource.parse_timesteps(time, freq=timestep_freq)
        if pressure_levels is None:
            pressure_levels = mlmod.model_level_reference_pressure(20_000.0, 50_000.0)
        self.pressure_levels = metsource.parse_pressure_levels(pressure_levels)
        self.variables = metsource.parse_variables(variables, self.pressure_level_variables)

    def __repr__(self) -> str:
        base = super().__repr__()
        return f"{base}\n\tDataset: {self.dataset}\n\tProduct type: {self.product_type}"

    @property
    def pressure_level_variables(self) -> list[MetVariable]:
        """ECMWF pressure level parameters available on model levels.

        Returns
        -------
        list[MetVariable]
            List of MetVariable available in datasource
        """
        return MODEL_LEVEL_VARIABLES

    @property
    def single_level_variables(self) -> list[MetVariable]:
        """ECMWF single-level parameters available on model levels.

        Returns
        -------
        list[MetVariable]
            Always returns an empty list.
            To access single-level variables, used :class:`pycontrails.datalib.ecmwf.ERA5`.
        """
        return []

    @property
    def dataset(self) -> str:
        """Select dataset for downloading model-level data.

        Always returns "reanalysis-era5-complete".

        Returns
        -------
        str
            Model-level ERA5 dataset name in CDS
        """
        return "reanalysis-era5-complete"

    @override
    def create_cachepath(self, t: datetime | pd.Timestamp) -> str:
        """Return cachepath to local ERA5 data file based on datetime.

        This uniquely defines a cached data file with class parameters.

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
            msg = "Cachestore is required to create cache path"
            raise ValueError(msg)

        string = (
            f"{t:%Y%m%d%H}-"
            f"{'.'.join(str(p) for p in self.pressure_levels)}-"
            f"{'.'.join(sorted(self.variable_shortnames))}-"
            f"{self.grid}"
        )

        name = hashlib.md5(string.encode()).hexdigest()
        cache_path = f"era5ml-{name}.nc"

        return self.cachestore.path(cache_path)

    @override
    def download_dataset(self, times: list[datetime]) -> None:
        # group data to request by month (nominal) or by day (ensemble)
        requests: dict[datetime, list[datetime]] = collections.defaultdict(list)
        for t in times:
            request = (
                datetime(t.year, t.month, 1)
                if self.product_type == "reanalysis"
                else datetime(t.year, t.month, t.day)
            )
            requests[request].append(t)

        # retrieve and process data for each request
        LOG.debug(f"Retrieving ERA5 ML data for times {times} in {len(requests)} request(s)")
        for times_in_request in requests.values():
            self._download_convert_cache_handler(times_in_request)

    @override
    def open_metdataset(
        self,
        dataset: xr.Dataset | None = None,
        xr_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> MetDataset:
        if dataset:
            msg = "Parameter 'dataset' is not supported for Model-level ERA5 data"
            raise ValueError(msg)

        if self.cachestore is None:
            msg = "Cachestore is required to download data"
            raise ValueError(msg)

        xr_kwargs = xr_kwargs or {}
        self.download(**xr_kwargs)

        disk_cachepaths = [self.cachestore.get(f) for f in self._cachepaths]
        ds = self.open_dataset(disk_cachepaths, **xr_kwargs)

        mds = self._process_dataset(ds, **kwargs)

        self.set_metadata(mds)
        return mds

    @override
    def set_metadata(self, ds: xr.Dataset | MetDataset) -> None:
        if self.product_type == "reanalysis":
            product = "reanalysis"
        elif self.product_type == "ensemble_members":
            product = "ensemble"
        else:
            msg = f"Unknown product type {self.product_type}"
            raise ValueError(msg)

        ds.attrs.update(
            provider="ECMWF",
            dataset="ERA5",
            product=product,
        )

    def _mars_request_base(self, times: list[datetime]) -> dict[str, str]:
        unique_dates = {t.strftime("%Y-%m-%d") for t in times}
        unique_times = {t.strftime("%H:%M:%S") for t in times}

        common = {
            "class": "ea",
            "date": "/".join(sorted(unique_dates)),
            "expver": "1",
            "levtype": "ml",
            "time": "/".join(sorted(unique_times)),
            "type": "an",
            "grid": f"{self.grid}/{self.grid}",
            "format": "netcdf",
        }

        if self.product_type == "reanalysis":
            specific = {"stream": "oper"}
        elif self.product_type == "ensemble_members":
            if self.ensemble_members is None:
                msg = "No ensemble members specified for ensemble product type."
                raise ValueError(msg)
            specific = {"stream": "enda", "number": "/".join(str(n) for n in self.ensemble_members)}

        return common | specific

    def _mars_request_lnsp(self, times: list[datetime]) -> dict[str, str]:
        out = self._mars_request_base(times)
        out["param"] = "152"  # lnsp, needed for model level -> pressure level conversion
        out["levelist"] = "1"
        return out

    def mars_request(self, times: list[datetime]) -> dict[str, str]:
        """Generate MARS request for specific list of times.

        Parameters
        ----------
        times : list[datetime]
            Times included in MARS request.

        Returns
        -------
        dict[str, str]:
            MARS request for submission to Copernicus CDS.
        """

        out = self._mars_request_base(times)
        out["param"] = "/".join(str(p) for p in sorted(set(self.variable_ecmwfids)))
        out["levelist"] = "/".join(str(lev) for lev in sorted(self.model_levels))
        return out

    def _set_cds(self) -> None:
        """Set the cdsapi.Client instance."""
        try:
            import cdsapi
        except ModuleNotFoundError as e:
            dependencies.raise_module_not_found_error(
                name="ERA5ModelLevel._set_cds method",
                package_name="cdsapi",
                module_not_found_error=e,
                pycontrails_optional_package="ecmwf",
            )

        try:
            self.cds = cdsapi.Client(url=self.url, key=self.key)
        # cdsapi throws base-level Exception
        except Exception as err:
            raise CDSCredentialsNotFound from err

    def _download_convert_cache_handler(self, times: list[datetime]) -> None:
        """Download, convert, and cache ERA5 model level data.

        This function builds a MARS request and retrieves a single NetCDF file.
        The calling function should ensure that all times will be contained
        in a single file on tape in the MARS archive.

        Because MARS requests treat dates and times as separate dimensions,
        retrieved data will include the Cartesian product of all unique
        dates and times in the list of specified times.

        After retrieval, this function processes the NetCDF file
        to produce the dataset specified by class attributes.

        Parameters
        ----------
        times : list[datetime]
            Times to download in a single MARS request.
        """
        if self.cachestore is None:
            msg = "Cachestore is required to download and cache data"
            raise ValueError(msg)

        ml_request = self.mars_request(times)
        lnsp_request = self._mars_request_lnsp(times)

        stack = contextlib.ExitStack()
        if not self.cache_download:
            ml_target = stack.enter_context(temp.temp_file())
            lnsp_target = stack.enter_context(temp.temp_file())
        else:
            ml_target = _target_path(ml_request, self.cachestore)
            lnsp_target = _target_path(lnsp_request, self.cachestore)

        with stack:
            threads = []
            for request, target in ((ml_request, ml_target), (lnsp_request, lnsp_target)):
                if not self.cache_download or not self.cachestore.exists(target):
                    if not hasattr(self, "cds"):
                        self._set_cds()
                    threads.append(
                        threading.Thread(
                            target=self.cds.retrieve,
                            args=("reanalysis-era5-complete", request, target),
                        )
                    )

            # Download across two threads
            with concurrent.futures.ThreadPoolExecutor() as executor:
                for thread in threads:
                    executor.submit(thread.run)

            LOG.debug("Opening model level data file")

            ds_ml = xr.open_dataset(ml_target)
            lnsp = xr.open_dataarray(lnsp_target)

            # New CDS (Aug 2024) gives "valid_time" instead of "time"
            if "valid_time" in ds_ml:
                ds_ml = ds_ml.rename(valid_time="time")
            if "valid_time" in lnsp.dims:
                lnsp = lnsp.rename(valid_time="time")

            # Legacy CDS (prior to Aug 2024) gives "level" instead of "model_level"
            if "level" in ds_ml.dims:
                ds_ml = ds_ml.rename(level="model_level")

            # Use a chunking scheme harmonious with self.cache_dataset, which groups by time
            # Because ds_ml is dask-backed, nothing gets computed until cache_dataset is called
            ds_ml = ds_ml.chunk(time=1)
            lnsp = lnsp.chunk(time=1)

            ds = mlmod.ml_to_pl(ds_ml, target_pl=self.pressure_levels, lnsp=lnsp)
            ds.attrs["pycontrails_version"] = pycontrails.__version__
            self.cache_dataset(ds)


def _target_path(request: dict[str, str], cachestore: cache.CacheStore) -> str:
    request_str = ";".join(f"{p}:{request[p]}" for p in sorted(request))
    name = hashlib.md5(request_str.encode()).hexdigest()
    return cachestore.path(f"era5ml-{name}-raw.nc")
