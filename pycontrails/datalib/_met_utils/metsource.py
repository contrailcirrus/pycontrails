"""Met datalib definitions and utilities."""

from __future__ import annotations

import abc
import hashlib
import logging
import pathlib
from collections.abc import Sequence
from datetime import datetime
from typing import Any, TypeAlias

import numpy as np
import pandas as pd
import xarray as xr

from pycontrails.core import cache
from pycontrails.core.met import MetDataset, MetVariable
from pycontrails.utils.types import DatetimeLike

logger = logging.getLogger(__name__)

# https://github.com/python/mypy/issues/14824
TimeInput: TypeAlias = str | DatetimeLike | Sequence[str | DatetimeLike]
VariableInput = (
    str | int | MetVariable | np.ndarray | Sequence[str | int | MetVariable | Sequence[MetVariable]]
)

PressureLevelInput = int | float | np.ndarray | Sequence[int | float]

#: NetCDF engine to use for parsing netcdf files
NETCDF_ENGINE: str = "netcdf4"

#: Default chunking strategy when opening datasets with xarray
DEFAULT_CHUNKS: dict[str, int] = {"time": 1}

#: Whether to open multi-file datasets in parallel
OPEN_IN_PARALLEL: bool = False


def parse_timesteps(time: TimeInput | None, freq: str | None = "1h") -> list[datetime]:
    """Parse time input into set of time steps.

    If input time is length 2, this creates a range of equally spaced time
    points between ``[start, end]`` with interval ``freq``.

    Parameters
    ----------
    time : TimeInput | None
        Input datetime(s) specifying the time or time range of the data [start, end].
        Either a single datetime-like or tuple of datetime-like with the first value
        the start of the date range and second value the end of the time range.
        Input values can be any type compatible with :meth:`pandas.to_datetime`.
    freq : str | None, optional
        Timestep interval in range.
        See https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases
        for a list of frequency aliases.
        If None, returns input `time` as a list.
        Defaults to "1h".

    Returns
    -------
    list[datetime]
        List of unique datetimes.
        If input ``time`` is None, returns an empty list

    Raises
    ------
    ValueError
        Raises when the time has len > 2 or when time elements fail to be parsed with pd.to_datetime
    """

    if time is None:
        return []

    # confirm input is tuple or list-like of length 2
    if isinstance(time, str | datetime | pd.Timestamp | np.datetime64):
        time = (time, time)
    elif len(time) == 1:
        time = (time[0], time[0])
    elif len(time) != 2:
        msg = f"Input time bounds must have length 1 or 2, got {len(time)}"
        raise ValueError(msg)

    # convert all to pandas Timestamp
    try:
        t0, t1 = (pd.to_datetime(t) for t in time)
    except ValueError as e:
        msg = (
            f"Failed to parse time input {time}. "
            "Time input must be compatible with 'pd.to_datetime()'"
        )
        raise ValueError(msg) from e

    if freq is None:
        daterange = pd.DatetimeIndex([t0, t1])
    else:
        # get date range that encompasses all whole hours
        daterange = pd.date_range(t0.floor(freq), t1.ceil(freq), freq=freq)
        if len(daterange) == 0:
            msg = f"Time range {t0} to {t1} with freq {freq} has no valid time steps."
            raise ValueError(msg)

    # return list of datetimes
    return daterange.to_pydatetime().tolist()


def validate_timestep_freq(freq: str, datasource_freq: str) -> bool:
    """Check that input timestep frequency is compatible with the data source timestep frequency.

    A data source timestep frequency of 1 hour allows input timestep frequencies of
    1 hour, 2 hours, 3 hours, etc., but not 1.5 hours or 30 minutes.

    Parameters
    ----------
    freq : str
        Input timestep frequency
    datasource_freq : str
        Datasource timestep frequency

    Returns
    -------
    bool
        True if the input timestep frequency is an even multiple
        of the data source timestep frequency.
    """
    return pd.Timedelta(freq) % pd.Timedelta(datasource_freq) == pd.Timedelta(0)


def parse_pressure_levels(
    pressure_levels: PressureLevelInput, supported: list[int] | None = None
) -> list[int]:
    """Check input pressure levels are consistent type and ensure levels exist in ECMWF data source.

    .. versionchanged:: 0.50.0

        The returned pressure levels are now sorted. Pressure levels must be unique.
        Raises ValueError if pressure levels have mixed signs.

    Parameters
    ----------
    pressure_levels : PressureLevelInput
        Input pressure levels for data, in hPa (mbar)
        Set to [-1] to represent surface level.
    supported : list[int], optional
        List of supported pressures levels in data source

    Returns
    -------
    list[int]
        List of integer pressure levels supported by ECMWF data source

    Raises
    ------
    ValueError
        Raises ValueError if pressure level is not supported by ECMWF data source
    """
    # Ensure pressure_levels is array-like
    if isinstance(pressure_levels, int | float):
        pressure_levels = [pressure_levels]

    # Cast array-like to int dtype and sort
    arr = np.asarray(pressure_levels, dtype=int)
    arr.sort()

    # If any values are non-positive, the entire array should be [-1]
    if np.any(arr <= 0) and not np.array_equal(arr, [-1]):
        msg = f"Pressure levels must be all positive or all -1, got {arr}"
        raise ValueError(msg)

    # Ensure pressure levels are unique
    if np.any(np.diff(arr) == 0):
        msg = f"Pressure levels must be unique, got {arr}"
        raise ValueError(msg)

    out = arr.tolist()
    if supported is None:
        return out  # type: ignore[return-value]

    if missing := set(out).difference(supported):  # type: ignore[arg-type]
        msg = (
            f"Pressure levels {sorted(missing)} are not supported. "  # type: ignore[type-var]
            f"Supported levels: {supported}"
        )
        raise ValueError(msg)

    return out  # type: ignore[return-value]


def parse_variables(variables: VariableInput, supported: list[MetVariable]) -> list[MetVariable]:
    """Parse input variables.

    .. versionchanged:: 0.50.0

        The output is no longer copied. Each :class:`MetVariable` is a frozen dataclass,
        so copying is unnecessary.

    Parameters
    ----------
    variables : VariableInput
        Variable name, or sequence of variable names.
        i.e. ``"air_temperature"``, ``["air_temperature, relative_humidity"]``,
        ``[130]``, ``[AirTemperature]``, ``[[EastwardWind, NorthwardWind]]``
        If an element is a list of MetVariable, the first MetVariable that is
        supported will be chosen.
    supported : list[MetVariable]
        Supported MetVariable.

    Returns
    -------
    list[MetVariable]
        List of MetVariable

    Raises
    ------
    ValueError
        Raises ValueError if variable is not supported
    """
    parsed_variables: Sequence[str | int | MetVariable | Sequence[MetVariable]]
    met_var_list: list[MetVariable] = []

    # ensure input variables are a list of str
    if isinstance(variables, str | int | MetVariable):
        parsed_variables = [variables]
    elif isinstance(variables, np.ndarray):
        parsed_variables = variables.tolist()
    else:
        parsed_variables = variables

    short_names = {v.short_name: v for v in supported}
    standard_names = {v.standard_name: v for v in supported}
    long_names = {v.long_name: v for v in supported}
    ecmwf_ids = {v.ecmwf_id: v for v in supported}
    grib1_ids = {v.grib1_id: v for v in supported}
    supported_set = set(supported)

    for var in parsed_variables:
        matched = _find_match(
            var,
            supported_set,
            ecmwf_ids,  # type: ignore[arg-type]
            grib1_ids,  # type: ignore[arg-type]
            short_names,
            standard_names,
            long_names,  # type: ignore[arg-type]
        )
        met_var_list.append(matched)

    return met_var_list


def _find_match(
    var: VariableInput,
    supported: set[MetVariable],
    ecmwf_ids: dict[int, MetVariable],
    grib1_ids: dict[int, MetVariable],
    short_names: dict[str, MetVariable],
    standard_names: dict[str, MetVariable],
    long_names: dict[str, MetVariable],
) -> MetVariable:
    """Find a match for input variable in supported."""

    if isinstance(var, MetVariable) and var in supported:
        return var

    # list of MetVariable options
    # here we extract the first MetVariable in var that is supported
    if isinstance(var, list | tuple):
        for v in var:
            # sanity check since we don't support other types as lists
            if not isinstance(v, MetVariable):
                msg = "Variable options must be of type MetVariable."
                raise TypeError(msg)
            if v in supported:
                return v

    elif isinstance(var, int):
        if ret := ecmwf_ids.get(var):
            return ret
        if ret := grib1_ids.get(var):
            return ret

    elif isinstance(var, str):
        if ret := short_names.get(var):
            return ret
        if ret := standard_names.get(var):
            return ret
        if ret := long_names.get(var):
            return ret

    msg = f"{var} is not in supported parameters. Supported parameters include: {standard_names}"
    raise ValueError(msg)


def parse_grid(grid: float, supported: Sequence[float]) -> float:
    """Parse input grid spacing.

    Parameters
    ----------
    grid : float
        Input grid float
    supported : Sequence[float]
        Sequence of support grid values

    Returns
    -------
    float
        Parsed grid spacing

    Raises
    ------
    ValueError
        Raises ValueError when ``grid`` is not in supported
    """
    if grid not in supported:
        msg = f"Grid input {grid} must be one of {supported}"
        raise ValueError(msg)

    return grid


def round_hour(time: datetime, hour: int) -> datetime:
    """Round time to the nearest whole hour before input time.

    Parameters
    ----------
    time : datetime
        Input time
    hour : int
        Hour to round down time

    Returns
    -------
    datetime
        Rounded time

    Raises
    ------
    ValueError
        Description
    """
    if hour not in range(1, 24):
        msg = f"hour must be between [1, 23], got {hour}"
        raise ValueError(msg)

    hour = (time.hour // hour) * hour
    return datetime(time.year, time.month, time.day, hour, 0, 0)


class MetDataSource(abc.ABC):
    """Abstract class for wrapping meteorology data sources."""

    __slots__ = ("grid", "paths", "pressure_levels", "timesteps", "variables")

    #: List of individual timesteps from data source derived from :attr:`time`
    #: Use :func:`parse_time` to handle :class:`TimeInput`.
    timesteps: list[datetime]

    #: Variables requested from data source
    #: Use :func:`parse_variables` to handle :class:`VariableInput`.
    variables: list[MetVariable]

    #: List of pressure levels. Set to [-1] for data without level coordinate.
    #: Use :func:`parse_pressure_levels` to handle :class:`PressureLevelInput`.
    pressure_levels: list[int]

    #: Lat / Lon grid spacing
    grid: float | None

    #: Path to local source files to load.
    #: Set to the paths of files cached in :attr:`cachestore` if no
    #: ``paths`` input is provided on init.
    paths: str | list[str] | pathlib.Path | list[pathlib.Path] | None

    #: Cache store for intermediates while processing data source
    #: If None, cache is turned off.
    cachestore: cache.CacheStore | None

    def __repr__(self) -> str:
        _repr = (
            f"{self.__class__.__name__}\n\t"
            f"Timesteps: {[t.strftime('%Y-%m-%d %H') for t in self.timesteps]}\n\t"
            f"Variables: {self.variable_shortnames}\n\t"
            f"Pressure levels: {self.pressure_levels}\n\t"
            f"Grid: {self.grid}"
        )

        if self.paths is not None:
            _repr += f"\n\tPaths: {self.paths}"

        return _repr

    @abc.abstractmethod
    def __init__(
        self,
        time: TimeInput | None,
        variables: VariableInput,
        pressure_levels: PressureLevelInput = -1,
        paths: str | list[str] | pathlib.Path | list[pathlib.Path] | None = None,
        grid: float | None = None,
        **kwargs: Any,
    ) -> None: ...

    @property
    def hash(self) -> str:
        """Generate a unique hash for this datasource.

        Returns
        -------
        str
            Unique hash for met instance (sha1)
        """
        hashstr = (
            f"{type(self).__name__}{self.timesteps}{self.variable_shortnames}{self.pressure_levels}"
        )
        return hashlib.sha1(bytes(hashstr, "utf-8")).hexdigest()

    @property
    def variable_shortnames(self) -> list[str]:
        """Return a list of variable short names.

        Returns
        -------
        list[str]
            Lst of variable short names.
        """
        return [v.short_name for v in self.variables]

    @property
    def variable_standardnames(self) -> list[str]:
        """Return a list of variable standard names.

        Returns
        -------
        list[str]
            Lst of variable standard names.
        """
        return [v.standard_name for v in self.variables]

    @property
    def is_single_level(self) -> bool:
        """Return True if the datasource is single level data.

        .. versionadded:: 0.50.0
        """
        return self.pressure_levels == [-1]

    @property
    def pressure_level_variables(self) -> list[MetVariable]:
        """Parameters available from data source.

        Returns
        -------
        list[MetVariable] | None
            List of MetVariable available in datasource
        """
        return []

    @property
    def single_level_variables(self) -> list[MetVariable]:
        """Parameters available from data source.

        Returns
        -------
        list[MetVariable] | None
            List of MetVariable available in datasource
        """
        return []

    @property
    def supported_variables(self) -> list[MetVariable]:
        """Parameters available from data source.

        Returns
        -------
        list[MetVariable] | None
            List of MetVariable available in datasource
        """
        return (
            self.single_level_variables if self.is_single_level else self.pressure_level_variables
        )

    @property
    def supported_pressure_levels(self) -> list[int] | None:
        """Pressure levels available from datasource.

        Returns
        -------
        list[int] | None
            List of integer pressure levels for class.
            If None, no pressure level information available for class.
        """
        return None

    @property
    def _cachepaths(self) -> list[str]:
        """Return cache paths to local data files.

        Returns
        -------
        list[str]
            Path to local data files
        """
        return [self.create_cachepath(t) for t in self.timesteps]

    # -----------------------------
    # Abstract methods to implement
    # -----------------------------
    @abc.abstractmethod
    def download_dataset(self, times: list[datetime]) -> None:
        """Download data from data source for input times.

        Parameters
        ----------
        times : list[datetime]
            List of datetimes to download a store in cache
        """

    @abc.abstractmethod
    def create_cachepath(self, t: datetime) -> str:
        """Return cachepath to local data file based on datetime.

        Parameters
        ----------
        t : datetime
            Datetime of datafile

        Returns
        -------
        str
            Path to cached data file
        """

    @abc.abstractmethod
    def cache_dataset(self, dataset: xr.Dataset) -> None:
        """Cache data from data source.

        Parameters
        ----------
        dataset : xr.Dataset
            Dataset loaded from remote API or local files.
            The dataset must have the same format as the original data source API or files.
        """

    @abc.abstractmethod
    def open_metdataset(
        self,
        dataset: xr.Dataset | None = None,
        xr_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> MetDataset:
        """Open MetDataset from data source.

        This method should download / load any required datafiles and
        returns a MetDataset of the multi-file dataset opened by xarray.

        Parameters
        ----------
        dataset : xr.Dataset | None, optional
            Input :class:`xr.Dataset` loaded manually.
            The dataset must have the same format as the original data source API or files.
        xr_kwargs : dict[str, Any] | None, optional
            Dictionary of keyword arguments passed into :func:`xarray.open_mfdataset`
            when opening files. Examples include "chunks", "engine", "parallel", etc.
            Ignored if ``dataset`` is input.
        **kwargs : Any
            Keyword arguments passed through directly into :class:`MetDataset` constructor.

        Returns
        -------
        MetDataset
            Meteorology dataset

        See Also
        --------
        :func:`xarray.open_mfdataset`
        """

    @abc.abstractmethod
    def set_metadata(self, ds: xr.Dataset | MetDataset) -> None:
        """Set met source metadata on ``ds.attrs``.

        This is called within the :meth:`open_metdataset` method to set metadata
        on the returned :class:`MetDataset` instance.

        Parameters
        ----------
        ds : xr.Dataset | MetDataset
            Dataset to set metadata on. Mutated in place.
        """

    # ----------------------
    # Common utility methods
    # ----------------------
    def download(self, **xr_kwargs: Any) -> None:
        """Confirm all data files are downloaded and available locally in the :attr:`cachestore`.

        Parameters
        ----------
        **xr_kwargs
            Passed into :func:`xarray.open_dataset` via :meth:`is_datafile_cached`.
        """
        if times_to_download := self.list_timesteps_not_cached(**xr_kwargs):
            logger.debug(
                "Not all files found in cachestore. Downloading times %s", times_to_download
            )
            self.download_dataset(times_to_download)
        else:
            logger.debug("All data files already in cache store")

    def list_timesteps_cached(self, **xr_kwargs: Any) -> list[datetime]:
        """Get a list of data files available locally in the :attr:`cachestore`.

        Parameters
        ----------
        **xr_kwargs
            Passed into :func:`xarray.open_dataset` via :meth:`is_datafile_cached`.
        """
        return [t for t in self.timesteps if self.is_datafile_cached(t, **xr_kwargs)]

    def list_timesteps_not_cached(self, **xr_kwargs: Any) -> list[datetime]:
        """Get a list of data files not available locally in the :attr:`cachestore`.

        Parameters
        ----------
        **xr_kwargs
            Passed into :func:`xarray.open_dataset` via :meth:`is_datafile_cached`.
        """
        return [t for t in self.timesteps if not self.is_datafile_cached(t, **xr_kwargs)]

    def is_datafile_cached(self, t: datetime, **xr_kwargs: Any) -> bool:
        """Check datafile defined by datetime for variables and pressure levels in class.

        If using a cloud cache store (i.e. :class:`cache.GCPCacheStore`), this is where the datafile
        will be mirrored to a local file for access.

        Parameters
        ----------
        t : datetime
            Datetime of datafile
        **xr_kwargs : Any
            Additional kwargs passed directly to :func:`xarray.open_mfdataset` when
            opening files. By default, the following values are used if not specified:

                - chunks: {"time": 1}
                - engine: "netcdf4"
                - parallel: False

        Returns
        -------
        bool
            True if data file exists for datetime with all variables and pressure levels,
            False otherwise
        """

        # return false if the cache is turned off
        if self.cachestore is None:
            return False

        # see if cache data file exists, and if so, get the file + path
        cache_path = self.create_cachepath(t)
        if not self.cachestore.exists(cache_path):
            logger.debug("Cachepath %s does not exist in cache", cache_path)
            return False

        logger.debug("Cachepath %s exists, getting from cache.", cache_path)

        # If GCP cache is used, this will download file and return the local mirrored path
        # If the local file already exists, this will return the local path
        disk_path = self.cachestore.get(cache_path)

        # check if all variables and pressure levels are in that path
        try:
            with self.open_dataset(disk_path, **xr_kwargs) as ds:
                return self._check_is_ds_complete(ds, cache_path)

        except OSError as err:
            if isinstance(self.cachestore, cache.GCPCacheStore):
                # If a GCPCacheStore is used, remove the corrupt file and try again.
                # If the file is corrupt in the bucket, we'll get stuck in an infinite loop here.
                logger.warning(
                    "Found corrupt file %s on local disk. Try again to download from %s.",
                    disk_path,
                    self.cachestore,
                    exc_info=err,
                )
                self.cachestore.clear_disk(disk_path)
                return self.is_datafile_cached(t, **xr_kwargs)

            msg = (
                f"Unable to open NETCDF file at '{disk_path}'. "
                "This may be due to a incomplete download. "
                f"Consider manually removing '{disk_path}' and retrying."
            )
            raise OSError(msg) from err

    def _check_is_ds_complete(self, ds: xr.Dataset, cache_path: str) -> bool:
        """Check if ``ds`` has all variables and pressure levels defined by the instance."""
        for var in self.variable_shortnames:
            if var not in ds:
                logger.warning(
                    "Variable %s not in downloaded dataset. Found variables: %s",
                    var,
                    ds.data_vars,
                )
                return False

        pl = np.asarray(self.pressure_levels)
        cond = np.isin(pl, ds["level"].values)
        if not np.all(cond):
            logger.warning(
                "Pressure Levels %s not in downloaded dataset. Found pressure levels: %s",
                pl[~cond].tolist(),
                ds["level"].values.tolist(),
            )
            return False

        logger.debug("All variables and pressure levels found in %s", cache_path)
        return True

    def open_dataset(
        self,
        disk_paths: str | list[str] | pathlib.Path | list[pathlib.Path],
        **xr_kwargs: Any,
    ) -> xr.Dataset:
        """Open multi-file dataset in xarray.

        Parameters
        ----------
        disk_paths : str | list[str] | pathlib.Path | list[pathlib.Path]
            list of string paths to local files to open
        **xr_kwargs : Any
            Additional kwargs passed directly to :func:`xarray.open_mfdataset` when
            opening files. By default, the following values are used if not specified:

                - chunks: {"time": 1}
                - engine: "netcdf4"
                - parallel: False
                - lock: False

        Returns
        -------
        xr.Dataset
            Open xarray dataset
        """
        xr_kwargs.setdefault("engine", NETCDF_ENGINE)
        xr_kwargs.setdefault("chunks", DEFAULT_CHUNKS)
        xr_kwargs.setdefault("parallel", OPEN_IN_PARALLEL)
        return xr.open_mfdataset(disk_paths, **xr_kwargs)
