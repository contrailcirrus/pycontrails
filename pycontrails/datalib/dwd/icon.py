"""ICON data access.

This module supports

- Retrieving ICON forecasts from the `DWD Open Data Server <https://opendata.dwd.de>`_.
- Interpolating forecasts onto pressure levels and a regular latitude/longitude grid.
- Local caching of processed forecasts as netCDF files.
- Opening processed and cached files as a :class:`pycontrails.MetDataset`.

"""

from __future__ import annotations

import asyncio
import bz2
import contextlib
import hashlib
import itertools
import logging
import math
import sys
import warnings
from collections.abc import Hashable, Iterator
from datetime import datetime, timedelta
from typing import Any

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override
LOG = logging.getLogger(__name__)

import dask.array
import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr
from scipy.spatial import KDTree
from tqdm.auto import tqdm

import pycontrails
from pycontrails.core import MetDataset, MetVariable, cache, met, met_var
from pycontrails.datalib import met_utils
from pycontrails.datalib._met_utils import metsource
from pycontrails.datalib.dwd import ods
from pycontrails.physics import units
from pycontrails.utils import coroutines, dependencies, temp
from pycontrails.utils.types import DatetimeLike

try:
    import aiohttp
except ModuleNotFoundError as exc:
    dependencies.raise_module_not_found_error(
        name="dwd.icon module",
        package_name="aiohttp",
        module_not_found_error=exc,
        pycontrails_optional_package="dwd",
    )

MODEL_LEVEL_VARIABLES = [
    met_var.AirTemperature,
    met_var.SpecificHumidity,
    met_var.NorthwardWind,
    met_var.EastwardWind,
    met_var.GeometricVerticalVelocity,
    met_var.MassFractionOfCloudIceInAir,
]


SINGLE_LEVEL_VARIABLES: list[MetVariable] = [
    met_var.TOAOutgoingLongwaveFlux,
    met_var.TOANetDownwardShortwaveFlux,
]

_met_var_to_ods_mapping = {
    met_var.AirTemperature: "t",
    met_var.SpecificHumidity: "qv",
    met_var.NorthwardWind: "u",
    met_var.EastwardWind: "v",
    met_var.GeometricVerticalVelocity: "w",
    met_var.MassFractionOfCloudIceInAir: "qi",
    met_var.TOAOutgoingLongwaveFlux: "athb_t",
    met_var.TOANetDownwardShortwaveFlux: "asob_t",
}

_icon_to_met_var: dict[Hashable, MetVariable] = {
    "t": met_var.AirTemperature,
    "q": met_var.SpecificHumidity,
    "u": met_var.NorthwardWind,
    "v": met_var.EastwardWind,
    "wz": met_var.GeometricVerticalVelocity,
    "QI": met_var.MassFractionOfCloudIceInAir,
    "avg_tnlwrf": met_var.TOAOutgoingLongwaveFlux,
    "avg_tnswrf": met_var.TOANetDownwardShortwaveFlux,
}


def flight_level_pressure(fl_min: int = 200, fl_max: int = 500) -> list[int]:
    """Get pressure at flight levels.

    Parameters
    ----------
    fl_min : int, optional
        Minimum flight level (default FL200)

    fl_max: int, optional
        Maximum flight level (default FL500)

    Returns
    -------
    list[int]
        Pressure, rounded to the nearest hPa, at each flight level
        between the minimum and maximum.

    """
    start = 1000 * math.ceil(fl_min / 10)
    stop = 1000 * math.floor(fl_max / 10)
    altitude_ft = range(start, stop + 1000, 1000)
    return [round(units.ft_to_pl(ft)) for ft in altitude_ft]


def forecast_frequency(domain: str) -> timedelta:
    """Get forecast cycle frequency.

    Parameters
    ----------
    domain : str
        ICON domain

    Returns
    -------
    timedelta
        Forecast cycle frequency (6h for global forecasts, 3h for others).

    """
    if domain.lower() == "global":
        return timedelta(hours=6)
    if domain.lower() in ("europe", "germany"):
        return timedelta(hours=3)

    msg = f"Unknown domain {domain}"
    raise ValueError(msg)


def valid_forecast_hours(domain: str) -> list[int]:
    """Get valid forecast initialization hours.

    Parameters
    ----------
    domain : str
         ICON domain

    Returns
    -------
    list[int]
        List of valid forecast hours as integers

    """
    if domain.lower() == "global":
        return list(range(0, 24, 6))
    if domain.lower() in ("europe", "germany"):
        return list(range(0, 24, 3))

    msg = f"Unknown domain {domain}"
    raise ValueError(msg)


def latest_forecast(domain: str, time: datetime) -> datetime:
    """Get most recent forecast initialized before the specified time.

    Parameters
    ----------
    domain : str
        ICON domain

    time : datetime
        Specified time

    Returns
    -------
    datetime
        Start time of most recent forecast initialized before `time`.

    """
    freq = pd.Timedelta(forecast_frequency(domain))
    return pd.Timestamp(time).floor(freq).to_pydatetime()


def last_step(domain: str, forecast_time: datetime) -> datetime:
    """Get time of last forecast step.

    Parameters
    ----------
    domain : str
        ICON domain

    forecast_time : datetime
        Forecast initialization time

    Returns
    -------
    datetime
        Time of last forecast step available on the specified domain.

    """
    if forecast_time.hour not in valid_forecast_hours(domain):
        msg = f"Invalid forecast time {forecast_time} for {domain} domain."
        raise ValueError(msg)

    if domain == "global":
        if forecast_time.hour % 12 == 0:
            return forecast_time + timedelta(hours=180)
        return forecast_time + timedelta(hours=120)

    if domain == "europe":
        if forecast_time.hour % 6 == 0:
            return forecast_time + timedelta(hours=120)
        return forecast_time + timedelta(hours=48)

    if domain == "germany":
        return forecast_time + timedelta(hours=48)

    msg = f"Unknown domain {domain}."
    raise ValueError(msg)


def last_hourly_step(domain: str, forecast_time: datetime) -> datetime:
    """Get time of last forecast step with hourly frequency.

    Parameters
    ----------
    domain : str
        ICON domain

    forecast_time : datetime
        Forecast initialization time

    Returns
    -------
    datetime
        Time of last hourly forecast step available on the specified domain.

    """
    if forecast_time.hour not in valid_forecast_hours(domain):
        msg = f"Invalid forecast time {forecast_time} for {domain} domain."
        raise ValueError(msg)

    if domain == "global":
        return forecast_time + timedelta(hours=78)

    if domain == "europe":
        if forecast_time.hour % 6 == 0:
            return forecast_time + timedelta(hours=78)
        return forecast_time + timedelta(hours=30)

    if domain == "germany":
        return forecast_time + timedelta(hours=48)

    msg = f"Unknown domain {domain}."
    raise ValueError(msg)


def extended_forecast_timestep(domain: str, forecast_time: datetime) -> str:
    """Get timestep for portions of forecasts after end of hourly data.

    Parameters
    ----------
    domain : str
        ICON domain

    forecast_time : datetime
        Forecast initialization time

    Returns
    -------
    str
        Timestep for portions of forecast after which hourly data is
        no longer available. Returns ``"1h"`` if hourly
        data is available for the entire forecast duration.

    """
    if forecast_time.hour not in valid_forecast_hours(domain):
        msg = f"Invalid forecast time {forecast_time} for {domain} domain."
        raise ValueError(msg)

    if domain == "global":
        return "3h"

    if domain == "europe":
        if forecast_time.hour % 6 == 0:
            return "3h"
        return "6h"

    if domain == "germany":
        return "1h"

    msg = f"Unknown domain {domain}."
    raise ValueError(msg)


def num_model_levels(domain: str) -> int:
    """Get number of model levels used by ICON.

    Parameters
    ----------
    domain : str
        ICON domain

    Returns
    -------
    int
        Number of model levels used by ICON.

    """
    if domain.lower() == "global":
        return 120
    if domain.lower() == "europe":
        return 74
    if domain.lower() == "germany":
        return 65

    msg = f"Unknown domain {domain}."
    raise ValueError(msg)


class ICON(metsource.MetDataSource):
    """Class to support ICON data access, download, and organization.

    Access is credential-free via the `DWD Open Data Server <https://opendata.dwd.de>`_.

    The current operational version of ICON uses a single-moment microphysics scheme that
    `underestimates humidity in ice-supersaturated regions <https://doi.org/10.5194/egusphere-2025-3312>`_.
    Documentation for this datalib will be updated when the double-moment microphysics scheme
    currently under development becomes operational.

    The DWD provides ICON forecasts on
    `three domains <https://www.dwd.de/EN/ourservices/nwp_forecast_data/nwp_forecast_data.html>`_:
    a global domain (~13 km resolution), a higher-resolution Europe domain (~7 km), and a
    high-resolution Germany domain (~2 km). Global forecasts are initialized every 6 hours and
    Europe and Germany forecasts are initialized every 3 hours. The Open Data Server *does not
    provide a long-term forecast archive*. Data for each forecast cycle is typically available
    for about 24 hours.

    Global ICON forecasts are provided on an icosahedral grid and must be remapped to a
    latitude-longitude grid. *This datalib currently supports nearest-neighbor remapping only.*
    Nearest-neighbor remapping preserves extreme values in forecast fields but can alter the
    locations where they occur. Additional remapping methods may be added in future releases.

    The forecast horizon depends on the domain and forecast initialization time:

    .. list-table:: ICON forecast horizon
        :header-rows: 1

        * - Forecast
          - Global (00, 12)
          - Global (06, 18)
          - Europe (00, 06, 12, 18)
          - Europe (03, 09, 15, 21)
          - Germany (all forecasts)
        * - Horizon (hourly forecast)
          - +78h
          - +78h
          - +78h
          - +30h
          - +48h
        * - Horizon (extended forecast)
          - +180h
          - +120h
          - +120h
          - +48h
          - N/A
        * - Extended forecast timestep
          - 3h
          - 3h
          - 3h
          - 6h
          - N/A

    This datalib currently supports only those variables required to run :class:`Cocip`
    and :class:`CocipGrid`. Please
    `contact the pycontrails developers <https://github.com/contrailcirrus/pycontrails/issues/new?template=feature_request.md>`_
    to request support for additional variables.

    Parameters
    ----------
    time : metsource.TimeInput
        The time range for data retrieval, either a single datetime or (start, end) datetime range.
        Input must be datetime-like or tuple of datetime-like specifying the (start, end)
        of the date range, inclusive.

    variables : metsource.VariableInput
        Variable name (e.g., "t", "air_temperature", ["air_temperature, specific_humidity"])

    pressure_levels : metsource.PressureLevelInput | None, optional
        Pressure levels for processed data, in hPa (mbar).
        To download single-level parameters, set to -1.
        Defaults to pressure levels that match standard flight levels between FL200 and FL500.

    domain : str, optional
        Forecast domain. Must be one of 'global' (global domain with ~13 km resolution, default),
        'europe' (European domain nested inside the global domain with ~7 km resolution),
        or 'germany' (regional domain centered on Germany with ~2.2 km resolution).

    timestep_freq : str | timedelta | None, optional
        Manually set the timestep interval within the bounds defined by :attr:`time`.
        Supports any value that can be passed to ``pandas.date_range(freq=...)``.
        By default, this is set to the highest frequency that can supported the requested
        time range on the requested domain.

    grid : float | None, optional
        Latitude/longitude grid resolution. Used only when `domain` is 'global',
        in which case data must be remapped from ICON's native icosahedral grid. Fields
        from the European nest and the regional Germany forecast are provided on regular
        latitude-longitude grids, so no interpolation is required and this parameter is
        ignored. If no value is provided when `domain` is 'global', data will
        be interpolated to a 0.25 degree grid, which provides spatial resolution comparable
        to ICON's native grid at midlatitude. A warning is issued if a value is provided
        when `domain` is set to 'europe' or 'germany'.

    forecast_time : DatetimeLike | None, optional
        Specify forecast by initialization time.
        By default, set to the most recent forecast that includes the requested time range.
        This is the most recent multiple of 3 hours (00z, 03z, 06z, etc) when `domain`
        is 'europe' or 'germany' and the most recent multiple of 6 hours (00z, 06z, etc)
        when `domain` is 'global'.

    model_levels : list[int] | None, optional
        Specify ICON model levels to include in downloads from the Open Data Server.
        By default, this is set to include all model levels.

    show_progress: bool, optional
        Show progress while downloading and processing ICON data.
        Disabled by default.

    cachestore : cache.CacheStore, optional
        Cache data store for staging processed netCDF files.
        Defaults to :class:`pycontrails.core.cache.DiskCacheStore`.
        If None, cache is turned off.

    cache_download : bool, optional
        If True, cache downloaded GRIB files rather than storing them in a temporary file.
        By default, False.


    See Also
    --------
    :func:`pycontrails.datalib.dwd.ods.list_forecasts`: list available forecast cycles
    :func:`pycontrails.datalib.dwd.ods.list_forecast_steps`: list available forecast steps

    """

    __marker = object()

    __slots__ = (
        "_global_kdtree",
        "cache_download",
        "cachestore",
        "domain",
        "forecast_time",
        "model_levels",
        "show_progress",
    )

    #: ICON forecast domain
    domain: str

    #: Forecast cycle start time
    forecast_time: datetime

    #: Model levels included when downloading raw data files
    model_levels: list[int]

    #: Whether to show progress bar while downloading and processing data
    show_progress: bool

    #: Whether to save raw data files to :attr:`cachestore` for reuse
    cache_download: bool

    def __init__(
        self,
        time: metsource.TimeInput,
        variables: metsource.VariableInput,
        pressure_levels: metsource.PressureLevelInput | None = None,
        domain: str = "global",
        timestep_freq: str | timedelta | None = None,
        grid: float | None = None,
        forecast_time: DatetimeLike | None = None,
        model_levels: list[int] | None = None,
        show_progress: bool = False,
        cachestore: cache.CacheStore = __marker,  # type: ignore[assignment]
        cache_download: bool = False,
    ) -> None:
        # Parse and set instance attributes

        if pressure_levels is None:
            pressure_levels = flight_level_pressure(200, 500)
        self.pressure_levels = metsource.parse_pressure_levels(pressure_levels)

        self.variables = metsource.parse_variables(variables, self.supported_variables)

        self.paths = None

        supported = "global", "europe", "germany"
        if domain not in supported:
            msg = f"Unknown domain {domain}. Supported domains are {', '.join(supported)}."
            raise ValueError(msg)
        self.domain = domain

        if grid is not None and domain != "global":
            msg = (
                "ICON-EU Europe and ICON-D2 Germany forecasts are provided at fixed resolution. "
                f"Ignoring grid={grid}. Set grid=None to silence this warning."
            )
            warnings.warn(msg)
        self.grid = grid or 0.25 if domain == "global" else None

        max_level = num_model_levels(domain)
        if model_levels is None:
            model_levels = list(range(1, max_level + 1))
        elif min(model_levels) < 1 or max(model_levels) > max_level:
            msg = (
                f"Requested model_levels must be between 1 and {max_level}, inclusize, "
                f"when using domain='{domain}'."
            )
            raise ValueError(msg)
        self.model_levels = model_levels

        forecast_hours = metsource.parse_timesteps(time, freq="1h")
        if forecast_time is None:
            self.forecast_time = latest_forecast(self.domain, forecast_hours[0])
        else:
            try:
                self.forecast_time = pd.to_datetime(forecast_time).to_pydatetime()
            except ValueError as e:
                msg = (
                    f"Failed to parse forecast time {forecast_time}. "
                    "Value must be compatible with 'pd.to_datetime'."
                )
                raise ValueError(msg) from e
            valid_hours = valid_forecast_hours(self.domain)
            if (hour := self.forecast_time.hour) not in valid_hours:
                msg = (
                    f"Forecast hour must be one of {[f'{h:02d}' for h in valid_hours]} "
                    f"but is {hour:02d}."
                )
                raise ValueError(msg)

        last_hour = forecast_hours[-1]
        if last_hour > (end := last_step(self.domain, self.forecast_time)):
            msg = f"Requested times extend to {last_hour}, beyond end of forecast at {end}."
            raise ValueError(msg)

        datasource_timestep_freq = (
            "1h"
            if last_hour <= last_hourly_step(self.domain, self.forecast_time)
            else extended_forecast_timestep(self.domain, self.forecast_time)
        )
        if timestep_freq is None:
            timestep_freq = datasource_timestep_freq
        if not metsource.validate_timestep_freq(timestep_freq, datasource_timestep_freq):
            msg = (
                f"Forecast out to time {last_hour} "
                f"has timestep frequency of {datasource_timestep_freq} "
                f"and cannot support requested timestep frequency of {timestep_freq}."
            )
            raise ValueError(msg)

        self.timesteps = metsource.parse_timesteps(
            time, freq=timestep_freq, shift=timedelta(hours=self.forecast_time.hour)
        )
        if (start := self.timesteps[0]) < self.forecast_time:
            msg = f"Selected forecast time {self.forecast_time} is after first timestep at {start}."
            raise ValueError(msg)

        self.show_progress = show_progress
        self.cachestore = cache.DiskCacheStore() if cachestore is self.__marker else cachestore
        self.cache_download = cache_download
        self._global_kdtree: KDTree | None = None

    def __repr__(self) -> str:
        base = super().__repr__()
        return "\n\t".join(
            [
                base,
                f"Domain: {self.domain}",
                f"Forecast time: {self.forecast_time.strftime('%Y-%m-%d %HZ')}",
                f"Steps: {self.steps}",
            ]
        )

    @property
    def pressure_level_variables(self) -> list[MetVariable]:
        """Available pressure-level variables.

        All pressure-level variables are retrieved on model levels
        and interpolated to pressure levels.

        Returns
        -------
        list[MetVariable]
            List of MetVariable available in datasource
        """
        return MODEL_LEVEL_VARIABLES

    @property
    def single_level_variables(self) -> list[MetVariable]:
        """Available single-level variables.

        Returns
        -------
        list[MetVariable]
            List of MetVariable available in datasource
        """
        return SINGLE_LEVEL_VARIABLES

    def get_forecast_step(self, time: datetime) -> int:
        """Convert time to forecast steps.

        Parameters
        ----------
        times : datetime
            Time to convert to forecast steps

        Returns
        -------
        int
            Forecast step at given time
        """
        step = (time - self.forecast_time) / timedelta(hours=1)
        if not step.is_integer():
            msg = (
                f"Time-to-step conversion returned fractional forecast step {step} "
                f"for timestep {time.strftime('%Y-%m-%d %H:%M:%S')}"
            )
            raise ValueError(msg)
        return int(step)

    @property
    def steps(self) -> list[int]:
        """Forecast steps corresponding to input :attr:`time`.

        Returns
        -------
        list[int]
            List of forecast steps relative to :attr:`forecast_time`
        """
        return [self.get_forecast_step(t) for t in self.timesteps]

    @property
    def dataset(self) -> str:
        """MetDataset 'dataset' attribute.

        Returns
        -------
        str
            One of "ICON", "ICON-EU", or "ICON-D2"
        """
        if self.domain.lower() == "global":
            return "ICON"
        if self.domain.lower() == "europe":
            return "ICON-EU"
        if self.domain.lower() == "germany":
            return "ICON-D2"

        msg = f"Unknown domain {self.domain}."
        raise ValueError(msg)

    @override
    def download_dataset(self, times: list[datetime]) -> None:
        if self.show_progress:
            times = tqdm(times)
        for time in times:
            LOG.debug(
                f"Downloading ICON {self.domain} data for time {time} "
                f"from {self.forecast_time} forecast."
            )
            self._download_convert_cache_handler(time)

    @override
    def create_cachepath(self, t: datetime) -> str:
        if self.cachestore is None:
            msg = "Cachestore is required to create cache path"
            raise ValueError(msg)

        string = (
            f"{self.grid or 'default-lat-lon'}-"
            f"{t:%Y%m%d%H}-"
            f"{self.forecast_time:%Y%m%d%H}-"
            f"{'.'.join(str(p) for p in self.pressure_levels)}-"
            f"{'.'.join(sorted(self.variable_shortnames))}-"
        )

        name = hashlib.md5(string.encode()).hexdigest()
        ltype = "sl" if self.is_single_level else "pl"
        cache_path = f"{self.dataset.lower()}-{ltype}-{name}.nc"

        return self.cachestore.path(cache_path)

    @override
    def cache_dataset(self, dataset: xr.Dataset) -> None:
        if self.cachestore is None:
            return

        for t, ds in dataset.groupby("time", squeeze=False):
            cache_path = self.create_cachepath(pd.Timestamp(t).to_pydatetime())
            ds.to_netcdf(cache_path, mode="w")

    @override
    def open_metdataset(
        self,
        dataset: xr.Dataset | None = None,
        xr_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> MetDataset:
        if dataset is not None:
            msg = "Parameter 'dataset' is not supported for ICON data"
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
        ds.attrs.update(provider="DWD", dataset=self.dataset, product="forecast")

    def rpaths(self, time: datetime) -> list[str]:
        """Get list of remote paths for download.

        Note that this function returns remote paths required to
        process a single forecast time step only.

        Parameters
        ----------
        time : datetime
            Forecast timestep

        Returns
        -------
        list[str]
            Open Data Server URLs for all required variables
            at the specified timestep. This includes URLS for
            air pressure when processing model-level variables,
            since the pressure field is required for conversion
            from model to pressure levels.

        """
        step = self.get_forecast_step(time)

        variables = [_met_var_to_ods_mapping[v] for v in self.variables]
        if not self.is_single_level:
            variables += "p"

        levels = [None] if self.is_single_level else sorted(self.model_levels)

        return [
            ods.rpath(self.domain, self.forecast_time, var, step, level)
            for var, level in itertools.product(variables, levels)
        ]

    def _process_dataset(self, ds: xr.Dataset, **kwargs: Any) -> MetDataset:
        """Process the :class:`xr.Dataset` opened from cached files.

        Parameters
        ----------
        ds : xr.Dataset
            Dataset loaded from netcdf cache files.
        **kwargs : Any
            Keyword arguments passed through directly into :class:`MetDataset` constructor.

        Returns
        -------
        MetDataset

        """
        # radiative fluxes are averaged over hour ending at time coordinate
        if self.is_single_level:
            shift_radiation_time = -np.timedelta64(30, "m")
            ds = ds.assign_coords(time=ds["time"] + shift_radiation_time)
            ds = ds.sel(time=slice(self.forecast_time, None))
            ds["time"].attrs["shift_radiation_time"] = str(shift_radiation_time)

        # change sign convention for toa lw flux to positive upward
        if "rlut" in ds:
            ds["rlut"] = -ds["rlut"]

        ds = met.standardize_variables(ds, self.variables)

        kwargs.setdefault("cachestore", self.cachestore)
        return MetDataset(ds, **kwargs)

    def _set_kdtree(self) -> None:
        """Get KDtree for looking up nearest neighbors on global icosahedral grid."""
        if self._global_kdtree:
            return

        downloads = contextlib.ExitStack()
        decompressed = contextlib.ExitStack()
        rpaths = [
            ods.global_longitude_rpath(self.forecast_time),
            ods.global_latitude_rpath(self.forecast_time),
        ]
        lpaths = []
        gribs = []
        for rpath in rpaths:
            if not self.cache_download:
                lpaths.append(downloads.enter_context(temp.temp_file()))
            else:
                lpaths.append(self.cachestore.path(rpath.split("/")[-1]))  # type:ignore
            gribs.append(decompressed.enter_context(temp.temp_file()))

        with decompressed:
            with downloads:
                coroutines.run(self._download_async(rpaths, lpaths))
                for lpath, grib in zip(lpaths, gribs, strict=True):
                    with open(lpath, "rb") as f:
                        data = bz2.decompress(f.read())
                    with open(grib, "wb") as f:
                        f.write(data)

            ds = xr.open_dataset(gribs[0], engine="cfgrib", backend_kwargs={"indexpath": ""})
            lon = ds["tlon"].values

            ds = xr.open_dataset(gribs[1], engine="cfgrib", backend_kwargs={"indexpath": ""})
            lat = ds["tlat"].values

        x, y, z = _ll_to_cartesian(lon, lat)
        self._global_kdtree = KDTree(data=np.stack((x, y, z), axis=-1))

    def _download_convert_cache_handler(self, time: datetime) -> None:
        """Download, convert, and cache ICON model level data."""
        if self.cachestore is None:
            msg = "Cachestore is required to download and cache data"
            raise ValueError(msg)

        downloads = contextlib.ExitStack()
        decompressed = contextlib.ExitStack()
        rpaths = self.rpaths(time)
        lpaths = []
        gribs = []
        for rpath in rpaths:
            if not self.cache_download:
                lpaths.append(downloads.enter_context(temp.temp_file()))
            else:
                lpaths.append(self.cachestore.path(rpath.split("/")[-1]))
            gribs.append(decompressed.enter_context(temp.temp_file()))

        # ecCodes will complain about missing latitude/longitude coordinates
        # when opening grib messages on the unstructured global grid.
        # This complaint comes from `logging.warning`, not `warnings.warn`,
        # so we use a custom content manager to temporarily add a filter
        # to the logger that issues the warning.
        with decompressed, _eccodes_warning_filter():
            with downloads:
                coroutines.run(self._download_async(rpaths, lpaths))
                for lpath, grib in zip(lpaths, gribs, strict=True):
                    with open(lpath, "rb") as f:
                        data = bz2.decompress(f.read())
                    with open(grib, "wb") as f:
                        f.write(data)

            ds = xr.open_mfdataset(
                gribs,
                combine="by_coords",
                compat="equals",
                preprocess=_preprocess_grib,
                engine="cfgrib",
                # Prevent cfgrib from creating index files
                backend_kwargs={"indexpath": ""},
            )

            ds = _rename(ds)

            if self.domain == "global":
                if self.grid is None:
                    msg = "Grid resolution must be set before remapping."
                    raise ValueError(msg)
                self._set_kdtree()
                ds = _global_icosahedral_to_regular_lat_lon(ds, self._global_kdtree, self.grid)

            if self.is_single_level:
                ds = ds.expand_dims(level=self.pressure_levels)
            else:
                ds = _ml_to_pl(ds, target_pl=self.pressure_levels)

            ds.attrs["pycontrails_version"] = pycontrails.__version__
            self.cache_dataset(ds)

    async def _download_async(self, rpaths: list[str], lpaths: list[str]) -> None:
        """Download files asynchronously."""
        if self.cache_download and not self.cachestore:
            msg = "Cachestore is required to cache downloads."
            raise ValueError(msg)

        async with aiohttp.ClientSession(raise_for_status=True) as session:
            tasks = []
            for rpath, lpath in zip(rpaths, lpaths, strict=True):
                if self.cache_download and self.cachestore.exists(lpath):  # type:ignore
                    continue
                tasks.append(ods._get_async(rpath, lpath, session))
            await asyncio.gather(*tasks)


def _preprocess_grib(ds: xr.Dataset) -> xr.Dataset:
    """Ensure consistent coordinates in GRIB messages before merging."""

    # some variables have different names for the model level coordinate
    if "generalVertical" in ds.coords:
        ds = ds.rename(generalVertical="model_level")
    if "generalVerticalLayer" in ds.coords:
        ds = ds.rename(generalVerticalLayer="model_level")
    if "model_level" in ds.coords:
        ds = ds.expand_dims("model_level")

    ds = ds.drop(["step", "time"]).rename(valid_time="time").expand_dims("time")
    return ds.reset_coords(drop=True)


def _rename(ds: xr.Dataset) -> xr.Dataset:
    """Update variable names to standard short names."""

    name_dict = {
        var: _icon_to_met_var[var].short_name for var in ds.data_vars if var in _icon_to_met_var
    }
    return ds.rename(name_dict)


def _ll_to_cartesian(
    longitude: npt.NDArray[np.floating], latitude: npt.NDArray[np.floating]
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Convert from longitude/latitude to cartesian coordinates in unit cube."""

    phi = np.deg2rad(latitude)
    theta = np.deg2rad(longitude)
    x = np.cos(phi) * np.cos(theta)
    y = np.cos(phi) * np.sin(theta)
    z = np.sin(phi)
    return x, y, z


def _remap_on_chunk(
    ds_chunk: xr.Dataset,
    idx: npt.NDArray[np.integer],
    target_lon: npt.NDArray[np.floating],
    target_lat: npt.NDArray[np.floating],
) -> xr.Dataset:
    """Remap using nearest-neighbor interpolation with precomputed indices."""

    if any(da_chunk.dims[-1] != "values" for da_chunk in ds_chunk.values()):
        msg = "The last dimension of all variables in the dataset must be 'values'"
        raise ValueError(msg)

    remapped_dict = {}

    for name, da in ds_chunk.items():
        remapped = da.values[..., idx]

        coords = {k: da.coords[k] for k in da.dims[:-1]}
        coords["latitude"] = target_lat
        coords["longitude"] = target_lon

        remapped_dict[name] = xr.DataArray(
            remapped, dims=tuple(coords), coords=coords, attrs=da.attrs
        )

    return xr.Dataset(remapped_dict)


def _build_remap_template(
    ds: xr.Dataset, target_lon: npt.NDArray[np.floating], target_lat: npt.NDArray[np.floating]
) -> xr.Dataset:
    """Build template dataset for output from horizontal remapping."""

    coords = {k: ds.coords[k] for k in ds.dims if k != "values"} | {
        "latitude": target_lat,
        "longitude": target_lon,
    }

    dims = tuple(coords)
    shape = tuple(len(v) for v in coords.values())

    vars = {k: (dims, dask.array.empty(shape=shape, dtype=da.dtype)) for k, da in ds.items()}

    chunks = {k: v for k, v in ds.chunks.items() if k != "values"}
    chunks["latitude"] = (target_lat.size,)
    chunks["longitude"] = (target_lon.size,)

    return xr.Dataset(data_vars=vars, coords=coords, attrs=ds.attrs).chunk(chunks)


def _global_icosahedral_to_regular_lat_lon(
    ds: xr.Dataset, kdtree: KDTree, resolution: float
) -> xr.Dataset:
    """Regrid from ICON's global icosahedral grid to a regular latitude-longitude grid."""

    # If any variables don't have a "values" dimension,
    # issue a warning and drop them
    for name, da in ds.items():
        if "values" not in da.dims:
            msg = (
                f"Variable '{name}' does not have a 'values' dimension. "
                f"This variable will be dropped before regridding."
            )
            warnings.warn(msg)
            ds = ds.drop_vars([name])

    # Check that "values" dimension is not chunked
    if ds.chunks and len(ds.chunks["values"]) > 1:
        msg = "The 'values' dimension must not be split across chunks."
        raise ValueError(msg)

    # Check that length of "values" dimension is compatible with kdtree
    if ds.sizes["values"] != kdtree.n:
        msg = "Size of 'values' dimension is incompatible with size of kdtree."
        raise ValueError(msg)

    nlat = int(90 / resolution)
    nlon = int(180 / resolution)
    target_lat = resolution * np.arange(-nlat, nlat + 1)
    target_lon = resolution * np.arange(-nlon, nlon + 1)
    template = _build_remap_template(ds, target_lon, target_lat)

    grid_lat, grid_lon = np.meshgrid(target_lat, target_lon, indexing="ij")
    x, y, z = _ll_to_cartesian(grid_lon, grid_lat)
    _, idx = kdtree.query(np.stack((x, y, z), axis=-1))

    return xr.map_blocks(_remap_on_chunk, ds, (idx, target_lon, target_lat), template=template)


def _ml_to_pl(ds: xr.Dataset, target_pl: npt.ArrayLike) -> xr.Dataset:
    """Interpolate from model levels to pressure levels."""

    ds["pressure_level"] = ds["pres"] / 100.0  # Pa -> hPa
    ds = ds.drop_vars("pres").chunk(model_level=-1)
    return met_utils.ml_to_pl(ds, target_pl)


class _ECCodesWarningFilter(logging.Filter):
    """Filter for logging.warnings produced by ecCodes."""

    def filter(self, record: logging.LogRecord) -> bool:
        msg = "ecCodes provides no latitudes/longitudes for gridType='unstructured_grid'"
        return record.levelname != "WARNING" or record.getMessage() != msg


@contextlib.contextmanager
def _eccodes_warning_filter() -> Iterator[None]:
    """Silence ecCodes warnings."""
    logger = logging.getLogger("cfgrib.dataset")
    filter = _ECCodesWarningFilter()
    logger.addFilter(filter)
    try:
        yield
    finally:
        logger.removeFilter(filter)
