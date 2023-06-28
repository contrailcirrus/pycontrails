"""Datalib utilities."""

from __future__ import annotations

import abc
import dataclasses
import hashlib
import logging
import pathlib
from datetime import datetime
from typing import Any, Sequence, Union

import numpy as np
import pandas as pd
import xarray as xr

from pycontrails.core import cache
from pycontrails.core.met import MetDataset, MetVariable
from pycontrails.utils.types import DatetimeLike

logger = logging.getLogger(__name__)

TimeInput = Union[str, DatetimeLike, Sequence[Union[str, DatetimeLike]]]
VariableInput = Union[
    str, int, MetVariable, np.ndarray, Sequence[Union[str, int, MetVariable, Sequence[MetVariable]]]
]
PressureLevelInput = Union[int, float, np.ndarray, Sequence[Union[int, float]]]

#: NetCDF engine to use for parsing netcdf files
NETCDF_ENGINE: str = "netcdf4"

#: Default chunking strategy when opening datasets with xarray
DEFAULT_CHUNKS: dict[str, int] = {"time": 1}

#: Whether to open multi-file datasets in parallel
OPEN_IN_PARALLEL: bool = False

#: Whether to use file locking when opening multi-file datasets
OPEN_WITH_LOCK: bool = False


def parse_timesteps(time: TimeInput | None, freq: str | None = "1H") -> list[datetime]:
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
        Defaults to "1H".

    Returns
    -------
    list[datetime]
        List of unique datetimes.
        If input ``time`` is None, returns an empty list

    Raises
    ------
    ValueError
        Raises when the time has len > 2 or when time elements fail to be parsed with pd.to_datetime
    """  # noqa: E501

    if time is None:
        return []

    # confirm input is tuple or list-like of length 2
    if isinstance(time, (str, datetime, pd.Timestamp, np.datetime64)):
        time = (time, time)
    elif len(time) == 1:
        time = (time[0], time[0])
    elif len(time) != 2:
        raise ValueError("Input time bounds must have length < 2 and > 0")

    # convert all to pandas Timestamp
    try:
        timestamps = [pd.to_datetime(t) for t in time]
    except ValueError as e:
        raise ValueError(
            f"Failed to parse all time inputs with error {e}. Time input "
            "must be compatible with 'pd.to_datetime()'"
        )

    if freq is None:
        daterange = pd.DatetimeIndex([timestamps[0], timestamps[1]])
    else:
        # get date range that encompasses all whole hours
        daterange = pd.date_range(timestamps[0].floor(freq), timestamps[1].ceil(freq), freq=freq)

    # return list of datetimes
    return daterange.to_pydatetime().tolist()


def parse_pressure_levels(
    pressure_levels: PressureLevelInput, supported: list[int] | None = None
) -> list[int]:
    """Check input pressure levels are consistent type and ensure levels exist in ECMWF data source.

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
    # ensure pressure_levels is list-like
    if isinstance(pressure_levels, (int, float)):
        pressure_levels = [pressure_levels]

    # Cast array-like to list of ints
    # Use a new variable to satiate mypy
    pressure_levels_ = np.asarray(pressure_levels, dtype=int).tolist()

    # ensure pressure levels are valid
    for pl in pressure_levels_:
        if supported is not None and pl not in supported:
            raise ValueError(f"{pl} is not a valid pressure level {supported}")

    return pressure_levels_


def parse_variables(variables: VariableInput, supported: list[MetVariable]) -> list[MetVariable]:
    """Parse input variables.

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
    if isinstance(variables, (str, int, MetVariable)):
        parsed_variables = [variables]
    elif isinstance(variables, np.ndarray):
        parsed_variables = variables.tolist()
    else:
        parsed_variables = variables

    # unpack list of supported str values from MetVariables
    short_names = [v.short_name for v in supported]
    standard_names = [v.standard_name for v in supported]
    long_names = [v.long_name for v in supported]

    # unpack list of support int values from Met Variables
    ecmwf_ids = [v.ecmwf_id for v in supported]
    grib1_ids = [v.grib1_id for v in supported]

    for var in parsed_variables:
        matched_variable: MetVariable | None = None

        if isinstance(var, MetVariable):
            if var in supported:
                matched_variable = var

        # list of MetVariable options
        # here we extract the first MetVariable in var that is supported
        elif isinstance(var, (list, tuple)):
            for v in var:
                # sanity check since we don't support other types as lists
                if not isinstance(v, MetVariable):
                    raise ValueError("Variable options must be of type MetVariable.")
                if v in supported:
                    matched_variable = v
                    break

        # int code
        elif isinstance(var, int):
            if var in ecmwf_ids:
                matched_variable = supported[ecmwf_ids.index(var)]
            elif var in grib1_ids:
                matched_variable = supported[grib1_ids.index(var)]

        # string reference
        elif isinstance(var, str):
            if var in short_names:
                matched_variable = supported[short_names.index(var)]
            elif var in standard_names:
                matched_variable = supported[standard_names.index(var)]
            elif var in long_names:
                matched_variable = supported[long_names.index(var)]

        if matched_variable is None:
            raise ValueError(
                f"{var} is not in supported parameters. "
                + f"Supported parameters include: {standard_names}"
            )

        # "replace" copies dataclass
        met_var_list.append(dataclasses.replace(matched_variable))

    return met_var_list


def parse_grid(grid: float, supported: list[float]) -> float:
    """Parse input grid spacing.

    Parameters
    ----------
    grid : float
        Input grid float
    supported : list[float]
        List of support grid values

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
        raise ValueError(f"Grid input must be one of {supported}")

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
        raise ValueError("hour must be between [1, 23]")

    hour = np.floor(time.hour / hour) * hour
    return datetime(
        time.year,
        time.month,
        time.day,
        int(hour),
        0,
        0,
    )


class MetDataSource(abc.ABC):
    """Abstract class for wrapping meteorology data sources."""

    __slots__ = ("timesteps", "variables", "pressure_levels", "grid", "paths")

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
        pressure_levels: PressureLevelInput = [-1],
        paths: str | list[str] | pathlib.Path | list[pathlib.Path] | None = None,
        grid: float | None = None,
        **kwargs: Any,
    ) -> None:
        ...

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
        if self.pressure_levels != [-1]:
            return self.pressure_level_variables

        return self.single_level_variables

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
        xr_kwargs: dict[str, int] | None = None,
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
        xr_kwargs : dict[str, int] | None, optional
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
                f"Not all files found in cachestore. Downloading times {times_to_download}"
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

        If using a cloud cache store (i.e. :class:`cache.GCPCacheStore`) this is where the datafile
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
                - parallel: True

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
        if self.cachestore.exists(cache_path):
            logger.debug(f"Cachepath {cache_path} exists, getting from cache.")
            # If GCP cache is used, this will download file and return the local mirrored path
            # If the local file already exists, this will return the local path
            disk_path = self.cachestore.get(cache_path)

            # check if all variables and pressure levels are in that path
            try:
                with self.open_dataset(disk_path, **xr_kwargs) as ds:
                    for var in self.variable_shortnames:
                        if var not in ds:
                            logger.warning(
                                f"Variable {var} not in downloaded dataset. "
                                f"Found variables: {ds.data_vars}"
                            )
                            return False

                    for pl in self.pressure_levels:
                        if pl not in ds["level"].values:
                            logger.warning(
                                f"Pressure Level {pl} not in downloaded dataset. "
                                f"Found pressure levels: {ds['level'].values}"
                            )
                            return False

                    logger.debug(f"All variables and pressure levels found in {cache_path}")
                    return True

            except OSError as err:
                if isinstance(self.cachestore, cache.GCPCacheStore):
                    # If a GCPCacheStore is used, remove the corrupt file and
                    # try again.
                    # If the file is corrupt in the GCP bucket, we'll
                    # get stuck in an infinite loop here
                    logger.warning(
                        "Found corrupt file %s on local disk. Try again to download from %s.",
                        disk_path,
                        self.cachestore,
                    )
                    self.cachestore.clear_disk(disk_path)
                    return self.is_datafile_cached(t, **xr_kwargs)

                raise OSError(
                    f"Unable to open NETCDF file at {disk_path} "
                    "This may be due to a incomplete download. "
                    f"Consider manually removing {disk_path} and retrying."
                ) from err

        logger.debug(f"Cachepath {cache_path} does not exist in cache")
        return False

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
        xr_kwargs.setdefault("lock", OPEN_WITH_LOCK)
        return xr.open_mfdataset(disk_paths, **xr_kwargs)
