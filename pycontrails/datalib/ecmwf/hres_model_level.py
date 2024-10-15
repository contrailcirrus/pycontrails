"""Model-level HRES data access from the ECMWF operational archive.

This module supports

- Retrieving model-level HRES data by submitting MARS requests through the ECMWF API.
- Processing retrieved model-level files to produce netCDF files on target pressure levels.
- Local caching of processed netCDF files.
- Opening processed and cached files as a :class:`pycontrails.MetDataset` object.
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
import sys
import warnings
from datetime import datetime, timedelta
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
from pycontrails.datalib.ecmwf.common import ECMWFAPI
from pycontrails.datalib.ecmwf.variables import MODEL_LEVEL_VARIABLES
from pycontrails.utils import dependencies, temp
from pycontrails.utils.types import DatetimeLike

LAST_STEP_1H = 96  # latest forecast step with 1 hour frequency
LAST_STEP_3H = 144  # latest forecast step with 3 hour frequency
LAST_STEP_6H = 240  # latest forecast step with 6 hour frequency


class HRESModelLevel(ECMWFAPI):
    """Class to support model-level HRES data access, download, and organization.

    The interface is similar to :class:`pycontrails.datalib.ecmwf.HRES`,
    which downloads pressure-level data with much lower vertical resolution and single-level data.
    Note, however, that only a subset of the pressure-level data available through the operational
    archive is available as model-level data. As a consequence, this interface only
    supports access to nominal HRES forecasts (corresponding to ``stream = "oper"`` and
    ``field_type = "fc"`` in :class:`pycontrails.datalib.ecmwf.HRES`) initialized at 00z and 12z.

    Requires account with ECMWF and API key.

    API credentials can be set in local ``~/.ecmwfapirc`` file:

    .. code:: json

        {
            "url": "https://api.ecmwf.int/v1",
            "email": "<email>",
            "key": "<key>"
        }

    Credentials can also be provided directly in ``url``, ``key``, and ``email`` keyword args.

    See `ecmwf-api-client <https://github.com/ecmwf/ecmwf-api-client>`_ documentation
    for more information.

    Parameters
    ----------
    time : metsource.TimeInput
        The time range for data retrieval, either a single datetime or (start, end) datetime range.
        Input must be datetime-like or tuple of datetime-like
        (:py:class:`datetime.datetime`, :class:`pandas.Timestamp`, :class:`numpy.datetime64`)
        specifying the (start, end) of the date range, inclusive.
        All times will be downloaded in a single NetCDF file, which
        ensures that exactly one request is submitted per file on tape accessed.
        If ``forecast_time`` is unspecified, the forecast time will
        be assumed to be the nearest synoptic hour available in the operational archive (00 or 12).
        All subsequent times will be downloaded for relative to :attr:`forecast_time`.
    variables : metsource.VariableInput
        Variable name (i.e. "t", "air_temperature", ["air_temperature, specific_humidity"])
    pressure_levels : metsource.PressureLevelInput, optional
        Pressure levels for data, in hPa (mbar).
        To download surface-level parameters, use :class:`pycontrails.datalib.ecmwf.HRES`.
        Defaults to pressure levels that match model levels at a nominal surface pressure.
    timestep_freq : str, optional
        Manually set the timestep interval within the bounds defined by :attr:`time`.
        Supports any string that can be passed to ``pandas.date_range(freq=...)``.
        By default, this is set to the highest frequency that can supported the requested
        time range ("1h" out to 96 hours, "3h" out to 144 hours, and "6h" out to 240 hours)
    grid : float, optional
        Specify latitude/longitude grid spacing in data.
        By default, this is set to 0.1.
    forecast_time : DatetimeLike, optional
        Specify forecast by initialization time.
        By default, set to the most recent forecast that includes the requested time range.
    levels : list[int], optional
        Specify ECMWF model levels to include in MARS requests.
        By default, this is set to include all model levels.
    cachestore : CacheStore | None, optional
        Cache data store for staging processed netCDF files.
        Defaults to :class:`pycontrails.core.cache.DiskCacheStore`.
        If None, cache is turned off.
    cache_download: bool, optional
        If True, cache downloaded NetCDF files rather than storing them in a temporary file.
        By default, False.
    url : str
        Override `ecmwf-api-client <https://github.com/ecmwf/ecmwf-api-client>`_ url
    key : str
        Override `ecmwf-api-client <https://github.com/ecmwf/ecmwf-api-client>`_ key
    email : str
        Override `ecmwf-api-client <https://github.com/ecmwf/ecmwf-api-client>`_ email
    """

    __marker = object()

    def __init__(
        self,
        time: metsource.TimeInput,
        variables: metsource.VariableInput,
        pressure_levels: metsource.PressureLevelInput | None = None,
        timestep_freq: str | None = None,
        grid: float | None = None,
        forecast_time: DatetimeLike | None = None,
        model_levels: list[int] | None = None,
        cachestore: cache.CacheStore = __marker,  # type: ignore[assignment]
        cache_download: bool = False,
        url: str | None = None,
        key: str | None = None,
        email: str | None = None,
    ) -> None:
        # Parse and set each parameter to the instance

        self.cachestore = cache.DiskCacheStore() if cachestore is self.__marker else cachestore
        self.cache_download = cache_download

        self.paths = None

        self.url = url
        self.key = key
        self.email = email

        if grid is None:
            grid = 0.1
        else:
            grid_min = 0.1
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

        forecast_hours = metsource.parse_timesteps(time, freq="1h")
        if forecast_time is None:
            self.forecast_time = metsource.round_hour(forecast_hours[0], 12)
        else:
            forecast_time_pd = pd.to_datetime(forecast_time)
            if (hour := forecast_time_pd.hour) % 12:
                msg = f"Forecast hour must be one of 00 or 12 but is {hour:02d}."
                raise ValueError(msg)
            self.forecast_time = metsource.round_hour(forecast_time_pd.to_pydatetime(), 12)

        last_step = (forecast_hours[-1] - self.forecast_time) / timedelta(hours=1)
        if last_step > LAST_STEP_6H:
            msg = (
                f"Requested times requires forecast steps out to {last_step}, "
                f"which is beyond latest available step of {LAST_STEP_6H}"
            )
            raise ValueError(msg)

        datasource_timestep_freq = (
            "1h" if last_step <= LAST_STEP_1H else "3h" if last_step <= LAST_STEP_3H else "6h"
        )
        if timestep_freq is None:
            timestep_freq = datasource_timestep_freq
        if not metsource.validate_timestep_freq(timestep_freq, datasource_timestep_freq):
            msg = (
                f"Forecast out to step {last_step} "
                f"has timestep frequency of {datasource_timestep_freq} "
                f"and cannot support requested timestep frequency of {timestep_freq}."
            )
            raise ValueError(msg)

        self.timesteps = metsource.parse_timesteps(time, freq=timestep_freq)
        if self.step_offset < 0:
            msg = f"Selected forecast time {self.forecast_time} is after first timestep."
            raise ValueError(msg)

        if pressure_levels is None:
            pressure_levels = mlmod.model_level_reference_pressure(20_000.0, 50_000.0)
        self.pressure_levels = metsource.parse_pressure_levels(pressure_levels)
        self.variables = metsource.parse_variables(variables, self.pressure_level_variables)

    def __repr__(self) -> str:
        base = super().__repr__()
        return "\n\t".join(
            [
                base,
                f"Forecast time: {getattr(self, 'forecast_time', '')}",
                f"Steps: {getattr(self, 'steps', '')}",
            ]
        )

    def get_forecast_steps(self, times: list[datetime]) -> list[int]:
        """Convert list of times to list of forecast steps.

        Parameters
        ----------
        times : list[datetime]
            Times to convert to forecast steps

        Returns
        -------
        list[int]
            Forecast step at each time
        """

        def time_to_step(time: datetime) -> int:
            step = (time - self.forecast_time) / timedelta(hours=1)
            if not step.is_integer():
                msg = (
                    f"Time-to-step conversion returned fractional forecast step {step} "
                    f"for timestep {time.strftime('%Y-%m-%d %H:%M:%S')}"
                )
                raise ValueError(msg)
            return int(step)

        return [time_to_step(t) for t in times]

    @property
    def step_offset(self) -> int:
        """Difference between :attr:`forecast_time` and first timestep.

        Returns
        -------
        int
            Number of steps to offset in order to retrieve data starting from input time.
        """
        return self.get_forecast_steps([self.timesteps[0]])[0]

    @property
    def steps(self) -> list[int]:
        """Forecast steps from :attr:`forecast_time` corresponding within input :attr:`time`.

        Returns
        -------
        list[int]
            List of forecast steps relative to :attr:`forecast_time`
        """
        return self.get_forecast_steps(self.timesteps)

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
            To access single-level variables, use :class:`pycontrails.datalib.ecmwf.HRES`.
        """
        return []

    @override
    def create_cachepath(self, t: datetime | pd.Timestamp) -> str:
        """Return cachepath to local HRES data file based on datetime.

        This uniquely defines a cached data file with class parameters.

        Parameters
        ----------
        t : datetime | pd.Timestamp
            Datetime of datafile

        Returns
        -------
        str
            Path to local HRES data file
        """
        if self.cachestore is None:
            msg = "Cachestore is required to create cache path"
            raise ValueError(msg)

        string = (
            f"{t:%Y%m%d%H}-"
            f"{self.forecast_time:%Y%m%d%H}-"
            f"{'.'.join(str(p) for p in self.pressure_levels)}-"
            f"{'.'.join(sorted(self.variable_shortnames))}-"
            f"{self.grid}"
        )

        name = hashlib.md5(string.encode()).hexdigest()
        cache_path = f"hresml-{name}.nc"

        return self.cachestore.path(cache_path)

    @override
    def download_dataset(self, times: list[datetime]) -> None:
        # will always submit a single MARS request since each forecast is a separate file on tape
        LOG.debug(f"Retrieving ERA5 data for times {times} from forecast {self.forecast_time}")
        self._download_convert_cache_handler(times)

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
        ds.attrs.update(
            provider="ECMWF", dataset="HRES", product="forecast", radiation_accumulated=True
        )

    def mars_request(self, times: list[datetime]) -> str:
        """Generate MARS request for specific list of times.

        Parameters
        ----------
        times : list[datetime]
            Times included in MARS request.

        Returns
        -------
        str
            MARS request for submission to ECMWF API.
        """
        date = self.forecast_time.strftime("%Y-%m-%d")
        time = self.forecast_time.strftime("%H:%M:%S")
        steps = self.get_forecast_steps(times)
        # param 152 = log surface pressure, needed for model level conversion
        grib_params = {*self.variable_ecmwfids, 152}
        return (
            f"retrieve,\n"
            f"class=od,\n"
            f"date={date},\n"
            f"expver=1,\n"
            f"levelist={'/'.join(str(lev) for lev in sorted(self.model_levels))},\n"
            f"levtype=ml,\n"
            f"param={'/'.join(str(p) for p in sorted(grib_params))},\n"
            f"step={'/'.join(str(s) for s in sorted(steps))},\n"
            f"stream=oper,\n"
            f"time={time},\n"
            f"type=fc,\n"
            f"grid={self.grid}/{self.grid},\n"
            "format=netcdf"
        )

    def _set_server(self) -> None:
        """Set the ecmwfapi.ECMWFService instance."""
        try:
            from ecmwfapi import ECMWFService
        except ModuleNotFoundError as e:
            dependencies.raise_module_not_found_error(
                name="HRESModelLevel._set_server method",
                package_name="ecmwf-api-client",
                module_not_found_error=e,
                pycontrails_optional_package="ecmwf",
            )

        self.server = ECMWFService("mars", url=self.url, key=self.key, email=self.email)

    def _download_convert_cache_handler(
        self,
        times: list[datetime],
    ) -> None:
        """Download, convert, and cache HRES model level data.

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

        request = self.mars_request(times)

        stack = contextlib.ExitStack()
        if not self.cache_download:
            target = stack.enter_context(temp.temp_file())
        else:
            name = hashlib.md5(request.encode()).hexdigest()
            target = self.cachestore.path(f"hresml-{name}.nc")

        with stack:
            if not self.cache_download or not self.cachestore.exists(target):
                if not hasattr(self, "server"):
                    self._set_server()
                self.server.execute(request, target)

            LOG.debug("Opening model level data file")

            # Use a chunking scheme harmonious with self.cache_dataset, which groups by time
            # Because ds_ml is dask-backed, nothing gets computed until cache_dataset is called
            ds_ml = xr.open_dataset(target).chunk(time=1)

            ds_ml = ds_ml.rename(level="model_level")
            lnsp = ds_ml["lnsp"].sel(model_level=1)
            ds_ml = ds_ml.drop_vars("lnsp")

            ds = mlmod.ml_to_pl(ds_ml, target_pl=self.pressure_levels, lnsp=lnsp)
            ds.attrs["pycontrails_version"] = pycontrails.__version__
            self.cache_dataset(ds)
