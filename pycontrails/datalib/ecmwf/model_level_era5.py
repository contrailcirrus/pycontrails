"""Model-level ERA5 data access.

This module supports

- Retrieving model-level ERA5 data by submitting MARS requests through the Copernicus CDS.
- Processing retrieved GRIB files to produce netCDF files on target pressure levels.
- Local caching of processed netCDF files.
- Opening processed and cached files as a :class:`pycontrails.MetDataset` object.

Consider using :class:`ARCOERA5` to access model-level data from the nominal ERA5
reanalysis between 1959 and 2022. :class:`ARCOERA5` accesses data through Google's
`Analysis-Ready, Cloud Optimized ERA5 dataset <https://cloud.google.com/storage/docs/public-datasets/era5>`_
and has lower latency than this module, which retrieves data from the
`Copernicus Climate Data Store <https://cds.climate.copernicus.eu/#!/home>`_.
This module must be used to retrieve model-level data from ERA5 ensemble members
or for more recent dates.

This module requires the following additional dependency:

- `metview (binaries and python bindings) <https://metview.readthedocs.io/en/latest/python.html>`_
"""

from __future__ import annotations

import collections
import contextlib
import hashlib
import logging
import os
import warnings
from datetime import datetime
from typing import Any

from overrides import overrides

LOG = logging.getLogger(__name__)

import pandas as pd
import xarray as xr

import pycontrails
from pycontrails.core import cache, datalib, met_var
from pycontrails.core.met import MetDataset, MetVariable
from pycontrails.datalib.ecmwf import variables as ecmwf_var
from pycontrails.datalib.ecmwf.common import ECMWFAPI
from pycontrails.datalib.ecmwf.model_levels import (
    MetviewTempfileHandler,
    pressure_levels_at_model_levels,
)
from pycontrails.utils import dependencies, temp

MODEL_LEVEL_VARIABLES = [
    met_var.AirTemperature,
    met_var.SpecificHumidity,
    met_var.VerticalVelocity,
    met_var.EastwardWind,
    met_var.NorthwardWind,
    ecmwf_var.RelativeVorticity,
    ecmwf_var.Divergence,
    ecmwf_var.CloudAreaFractionInLayer,
    ecmwf_var.SpecificCloudIceWaterContent,
    ecmwf_var.SpecificCloudLiquidWaterContent,
]

ALL_ENSEMBLE_MEMBERS = list(range(10))


def grib_to_dataset(
    source: str,
    time: datetime,
    pressure_levels: list[int],
    variables: list[MetVariable],
    ensemble_members: list[int] | None,
) -> xr.Dataset:
    """Extract pressure-level dataset from retrieved model-level GRIB file.

    Parameters
    ----------
    source : str
        Path to retrieved GRIB file.
    time : datetime
        Time to extract from GRIB file.
    pressure_levels : list[int]:
        List of pressure levels to interpolate data onto.
    variables : list[MetVariable]
        List of variables to extract from GRIB file.
    ensemble_members : list[int], optional
        List of ensemble members to extract from GRIB file.
        If not provided, assumes that the GRIB file contains
        only a single nominal member.

    Notes
    -----
    This function depends on `metview <https://metview.readthedocs.io/en/latest/python.html>`_
    python bindings and binaries.
    """
    try:
        import metview as mv
    except ModuleNotFoundError as exc:
        dependencies.raise_module_not_found_error(
            "model_level.grib_to_dataset function",
            package_name="metview",
            module_not_found_error=exc,
            extra="See https://metview.readthedocs.io/en/latest/install.html for instructions.",
        )
    except ImportError as exc:
        msg = "Failed to import metview"
        raise ImportError(msg) from exc

    # Read contents of GRIB file as metview Fieldset
    LOG.debug("Opening GRIB file")
    fs_ml = mv.read(source)

    # Create new fieldset containing fields interpolated to pressure levels.
    fs_pl = mv.Fieldset()
    dimensions = ensemble_members if ensemble_members else None
    for ens in dimensions:
        date = time.strftime("%Y%m%d")
        t = time.strftime("%H%M")
        selection = dict(date=date, time=t)
        if ens:
            selection |= dict(number=ens)

        lnsp = fs_ml.select(shortName="lnsp", **selection)

        for var in variables:
            LOG.debug(
                f"Converting {var.short_name} at {t}" + (f" (ensemble member {ens})" if ens else "")
            )

            f_ml = fs_ml.select(shortName=var.short_name, **selection)
            f_pl = mv.mvl_ml2hPa(lnsp, f_ml, pressure_levels)
            fs_pl = mv.merge(fs_pl, f_pl)

    # Create dataset
    ds = fs_pl.to_dataset()

    # Confirm dataset passes validation before returning
    ds = ds.rename(isobaricInhPa="level").expand_dims("time")
    return MetDataset(ds).data


class ModelLevelERA5(ECMWFAPI):
    """Class to support model-level ERA5 data access, download, and organization.

    The interface is similar to :class:`ERA5`, which downloads pressure-level
    with much lower vertical resolution.

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
        GRIB files will be downloaded from CDS in chunks no larger than 1 month
        for the nominal reanalysis and no larger than 1 day for ensemble members.
        This ensures that exactly one request is submitted per file on tape accessed.
        If None, ``paths`` must be defined and all time coordinates will be loaded from files.
    variables : datalib.VariableInput
        Variable name (i.e. "t", "air_temperature", ["air_temperature, specific_humidity"])
    pressure_levels : datalib.PressureLevelInput, optional
        Pressure levels for data, in hPa (mbar).
        To download surface-level parameters, use :class:`ERA5`.
        Defaults to pressure levels that match model levels at a nominal surface pressure.
    timestep_freq : str, optional
        Manually set the timestep interval within the bounds defined by :attr:`time`.
        Supports any string that can be passed to `pd.date_range(freq=...)`.
        By default, this is set to "1h" for reanalysis products and "3h" for ensemble products.
    product_type : str, optional
        Product type, one of "reanalysis" and "ensemble_members". Unlike :class:`ERA5`, this
        class does not support direct access to the ensemble mean and spread, which are not
        available on model levels.
    grid : float, optional
        Specify latitude/longitude grid spacing in data.
        By default, this is set to 0.25 for reanalysis products and 0.5 for ensemble products.
    levels : list[int], optional
        Specify ECMWF model levels to include in MARS requests.
        By default, this is set to include all model levels.
    ensemble_members : list[int], optional
        Specify ensemble members to include.
        Valid only when the product type is "ensemble_members".
        By default, includes every available ensemble member.
    cachestore : cache.CacheStore | None, optional
        Cache data store for staging processed netCDF files.
        Defaults to :class:`cache.DiskCacheStore`.
        If None, cache is turned off.
    cleanup_metview_tempfiles : bool, optional
        If True, cleanup all ``TEMP_DIRECTORY/tmp*.grib`` files. Implementation is brittle and may
        not work on all systems. By default, True.
    cache_grib: bool, optional
        If True, cache downloaded GRIB files rather than storing them in a temporary file.
        By default, False.
    url : str
        Override `cdsapi <https://github.com/ecmwf/cdsapi>`_ url
    key : str
        Override `cdsapi <https://github.com/ecmwf/cdsapi>`_ key
    """  # noqa: E501

    __marker = object()

    def __init__(
        self,
        time: datalib.TimeInput,
        variables: datalib.VariableInput,
        pressure_levels: datalib.PressureLevelInput | None = None,
        timestep_freq: str | None = None,
        product_type: str = "reanalysis",
        grid: float | None = None,
        levels: list[int] | None = None,
        ensemble_members: list[int] | None = None,
        cachestore: cache.CacheStore = __marker,  # type: ignore[assignment]
        n_jobs: int = 1,
        cleanup_metview_tempfiles: bool = True,
        cache_grib: bool = False,
        url: str | None = None,
        key: str | None = None,
    ) -> None:

        self.cachestore = cache.DiskCacheStore() if cachestore is self.__marker else cachestore
        self.cleanup_metview_tempfiles = cleanup_metview_tempfiles
        self.cache_grib = cache_grib

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

        if product_type == "reanalysis" and ensemble_members:
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
                    f"The smallest resolution available is {grid_min} degrees. "
                    f"Your downloaded data will have resolution {grid}, but it is a "
                    f"reinterpolation of the {grid_min} degree data. The same interpolation can be "
                    "achieved directly with xarray."
                )
                warnings.warn(msg)
        self.grid = grid

        if levels is None:
            levels = list(range(1, 138))
        if min(levels) < 1 or max(levels) > 137:
            msg = "Retrieval levels must be between 1 and 137, inclusive."
            raise ValueError(msg)
        self.levels = levels

        datasource_timestep_freq = "1h" if product_type == "reanalysis" else "3h"
        if timestep_freq is None:
            timestep_freq = datasource_timestep_freq
        if not datalib.validate_timestep_freq(timestep_freq, datasource_timestep_freq):
            msg = (
                f"Product {self.product_type} has timestep frequency of {datasource_timestep_freq} "
                f"and cannot support requested timestep frequency of {timestep_freq}."
            )
            raise ValueError(msg)

        self.timesteps = datalib.parse_timesteps(time, freq=timestep_freq)
        if pressure_levels is None:
            pressure_levels = pressure_levels_at_model_levels(20_000.0, 50_000.0)
        self.pressure_levels = datalib.parse_pressure_levels(pressure_levels)
        self.variables = datalib.parse_variables(variables, self.model_level_variables)

    def __repr__(self) -> str:
        base = super().__repr__()
        return f"{base}\n\tDataset: {self.dataset}\n\tProduct type: {self.product_type}"

    @property
    def model_level_variables(self) -> list[MetVariable]:
        """ECMWF model level parameters.

        Returns
        -------
        list[MetVariable] | None
            List of MetVariable available in datasource
        """
        return MODEL_LEVEL_VARIABLES

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

    @overrides
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
            msg = "Attribute self.cachestore must be defined to create cache path"
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

    @overrides
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
        LOG.debug(f"Retrieving ERA5 data for times {times} in {len(requests)} request(s)")

        stack = contextlib.ExitStack()

        if self.cleanup_metview_tempfiles:
            stack.enter_context(MetviewTempfileHandler())

        for times_in_request in requests.values():
            with stack:  # clean up after each iteration
                _download_convert_cache_handler(self, times_in_request)

    @overrides
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

    @overrides
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

    def _retrieve_grib(self, times: list[datetime], target: str) -> None:
        """Download data for specified times in a single MARS request.

        This function builds a MARS request and retrieves a single GRIB file.
        Logic in the calling function ensures that all times will be contained
        in a single file on tape in the MARS archive.

        Because MARS requests treat dates and times as separate dimensions,
        retrieved data will include the Cartesian product of all unique
        dates and times in the list of specified times.

        Parameters
        ----------
        times : list[datetime]
            Times to download in a single MARS request.
        target: str
            Location to save downloaded GRIB file.
        """

    def _process_grib(self, times: list[datetime], target: str):
        """Process model-level GRIB file and cache results as netCDF files.

        Parameters
        ----------
        times : list[datetime]
            List of timesteps to process

        target : str
            Path to GRIB file to process

        Notes
        -----
        The dependency on `metview <https://metview.readthedocs.io/en/latest/python.html>`_
        is limited to this method.
        """

    def _mars_request(self, times: list[datetime]) -> dict[str, str]:
        """Generate MARS request for specific list of times.

        Parameters
        ----------
        times : list[datetime]
            Times included in MARS request.
        """
        dates = set(t.strftime("%Y-%m-%d") for t in times)
        times = set(t.strftime("%H:%M:%S") for t in times)
        # param 152 = log surface pressure, needed for metview level conversion
        grib_params = set(self.variable_ecmwfids + [152])
        common = {
            "class": "ea",
            "date": "/".join(sorted(dates)),
            "expver": "1",
            "levelist": "/".join(str(lev) for lev in sorted(self.levels)),
            "levtype": "ml",
            "param": "/".join(str(p) for p in sorted(grib_params)),
            "time": "/".join(sorted(times)),
            "type": "an",
            "grid": f"{self.grid}/{self.grid}",
        }
        if self.product_type == "reanalysis":
            specific = {"stream": "oper"}
        else:
            specific = {"stream": "enda", "number": "/".join(str(n) for n in self.ensemble_members)}
        return common | specific

    def _set_cds(self) -> None:
        """Set the cdsapi.Client instance."""
        try:
            import cdsapi
        except ModuleNotFoundError as e:
            dependencies.raise_module_not_found_error(
                name="ModelLevelERA5._set_cds method",
                package_name="cdsapi",
                module_not_found_error=e,
                pycontrails_optional_package="ecmwf",
            )

        try:
            self.cds = cdsapi.Client(url=self.url, key=self.key)
        # cdsapi throws base-level Exception
        except Exception as err:
            raise CDSCredentialsNotFound from err


def _download_convert_cache_handler(
    era5: ModelLevelERA5,
    times: list[datetime],
) -> None:
    """Download, convert, and cache ERA5 model level data.

    This function builds a MARS request and retrieves a single GRIB file.
    The calling function should ensure that all times will be contained
    in a single file on tape in the MARS archive.

    Because MARS requests treat dates and times as separate dimensions,
    retrieved data will include the Cartesian product of all unique
    dates and times in the list of specified times.

    After retrieval, this function processes the GRIB file
    to produce the dataset specified by :param:`era5`.

    Parameters
    ----------
    era5 : ModelLevelERA5
        :class:`ModelLevelERA5` instance with specifications for processed dataset.
    times : list[datetime]
        Times to download in a single MARS request.
    """
    stack = contextlib.ExitStack()
    request = era5._mars_request(times)

    if not era5.cache_grib:
        target = stack.enter_context(temp.temp_file())
    else:
        request_str = ";".join(f"{p}:{request[p]}" for p in sorted(request.keys()))
        name = hashlib.md5(request_str.encode()).hexdigest()
        target = era5.cachestore.path(f"era5ml-{name}.grib")

    with stack:
        if not era5.cache_grib or not era5.cachestore.exists(target):
            if not hasattr(era5, "cds"):
                era5._set_cds()
            era5.cds.retrieve("reanalysis-era5-complete", request, target)

        # reduce memory overhead by converting one timestep at a time
        for time in times:
            ds = grib_to_dataset(
                target, time, era5.pressure_levels, era5.variables, era5.ensemble_members
            )
            ds.attrs["pycontrails_version"] = pycontrails.__version__
            era5.cache_dataset(ds)


class CDSCredentialsNotFound(Exception):
    """Raise when CDS credentials are not found by :class:`cdsapi.Client` instance."""
