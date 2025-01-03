"""GFS Data Access.

References
----------
- `NOAA GFS <https://registry.opendata.aws/noaa-gfs-bdp-pds/>`_
- `Documentation <https://www.ncei.noaa.gov/products/weather-climate-models/global-forecast>`_
- `Parameter sets <https://www.nco.ncep.noaa.gov/pmb/products/gfs/>`_
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
import pathlib
import sys
import warnings
from collections.abc import Callable
from datetime import datetime
from typing import TYPE_CHECKING, Any

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

import numpy as np
import pandas as pd
import xarray as xr

import pycontrails
from pycontrails.core import cache, met
from pycontrails.datalib._met_utils import metsource
from pycontrails.datalib.gfs.variables import (
    PRESSURE_LEVEL_VARIABLES,
    SURFACE_VARIABLES,
    TOAUpwardLongwaveRadiation,
    TOAUpwardShortwaveRadiation,
    Visibility,
)
from pycontrails.utils import dependencies, temp
from pycontrails.utils.types import DatetimeLike

# optional imports
if TYPE_CHECKING:
    import botocore

logger = logging.getLogger(__name__)

#: Default GFS AWS bucket
GFS_FORECAST_BUCKET = "noaa-gfs-bdp-pds"


class GFSForecast(metsource.MetDataSource):
    """GFS Forecast data access.

    Parameters
    ----------
    time : `metsource.TimeInput`
        The time range for data retrieval, either a single datetime or (start, end) datetime range.
        Input must be a single datetime-like or tuple of datetime-like (datetime,
        :class:`pandas.Timestamp`, :class:`numpy.datetime64`)
        specifying the (start, end) of the date range, inclusive.
        All times will be downloaded for a single forecast model run nearest to the start time
        (see :attr:`forecast_time`)
        If None, ``paths`` must be defined and all time coordinates will be loaded from files.
    variables : `metsource.VariableInput`
        Variable name (i.e. "temperature", ["temperature, relative_humidity"])
        See :attr:`pressure_level_variables` for the list of available variables.
    pressure_levels : `metsource.PressureLevelInput`, optional
        Pressure levels for data, in hPa (mbar)
        Set to [-1] for to download surface level parameters.
        Defaults to [-1].
    paths : str | list[str] | pathlib.Path | list[pathlib.Path] | None, optional
        Path to files to load manually.
        Can include glob patterns to load specific files.
        Defaults to None, which looks for files in the :attr:`cachestore` or GFS AWS bucket.
    grid : float, optional
        Specify latitude/longitude grid spacing in data.
        Defaults to 0.25.
    forecast_time : `DatetimeLike`, optional
        Specify forecast run by runtime. If None (default), the forecast time
        is set to the 6 hour floor of the first timestep.
    cachestore : :class:`cache.CacheStore` | None, optional
        Cache data store for staging data files.
        Defaults to :class:`cache.DiskCacheStore`.
        If None, cachestore is turned off.
    show_progress : bool, optional
        Show progress when downloading files from GFS AWS Bucket.
        Defaults to False
    cache_download: bool, optional
        If True, cache downloaded grib files rather than storing them in a temporary file.
        By default, False.

    Examples
    --------
    >>> from datetime import datetime
    >>> from pycontrails.datalib.gfs import GFSForecast

    >>> # Store data files to local disk (default behavior)
    >>> times = ("2022-03-22 00:00:00", "2022-03-22 03:00:00")
    >>> gfs = GFSForecast(times, variables="air_temperature", pressure_levels=[300, 250])
    >>> gfs
    GFSForecast
        Timesteps: ['2022-03-22 00', '2022-03-22 01', '2022-03-22 02', '2022-03-22 03']
        Variables: ['t']
        Pressure levels: [250, 300]
        Grid: 0.25
        Forecast time: 2022-03-22 00:00:00

    >>> gfs = GFSForecast(times, variables="air_temperature", pressure_levels=[300, 250], grid=0.5)
    >>> gfs
    GFSForecast
        Timesteps: ['2022-03-22 00', '2022-03-22 03']
        Variables: ['t']
        Pressure levels: [250, 300]
        Grid: 0.5
        Forecast time: 2022-03-22 00:00:00

    Notes
    -----
    - `NOAA GFS <https://registry.opendata.aws/noaa-gfs-bdp-pds/>`_
    - `Documentation <https://www.ncei.noaa.gov/products/weather-climate-models/global-forecast>`_
    - `Parameter sets <https://www.nco.ncep.noaa.gov/pmb/products/gfs/>`_
    - `GFS Documentation <https://www.emc.ncep.noaa.gov/emc/pages/numerical_forecast_systems/gfs/documentation.php>`_
    """

    __slots__ = ("cache_download", "cachestore", "client", "forecast_time", "grid", "show_progress")

    #: S3 client for accessing GFS bucket
    client: botocore.client.S3

    #: Lat / Lon grid spacing. One of [0.25, 0.5, 1]
    grid: float

    #: Show progress bar when downloading files from AWS
    show_progress: bool

    #: Base time of the previous GFS forecast based on input times
    forecast_time: datetime

    __marker = object()

    def __init__(
        self,
        time: metsource.TimeInput | None,
        variables: metsource.VariableInput,
        pressure_levels: metsource.PressureLevelInput = -1,
        paths: str | list[str] | pathlib.Path | list[pathlib.Path] | None = None,
        grid: float = 0.25,
        forecast_time: DatetimeLike | None = None,
        cachestore: cache.CacheStore | None = __marker,  # type: ignore[assignment]
        show_progress: bool = False,
        cache_download: bool = False,
    ) -> None:
        try:
            import boto3
        except ModuleNotFoundError as e:
            dependencies.raise_module_not_found_error(
                name="GFSForecast class",
                package_name="boto3",
                module_not_found_error=e,
                pycontrails_optional_package="gfs",
            )

        try:
            import botocore
        except ModuleNotFoundError as e:
            dependencies.raise_module_not_found_error(
                name="GFSForecast class",
                package_name="botocore",
                module_not_found_error=e,
                pycontrails_optional_package="gfs",
            )

        # inputs
        self.paths = paths
        if cachestore is self.__marker:
            cachestore = cache.DiskCacheStore()
        self.cachestore = cachestore
        self.show_progress = show_progress
        self.cache_download = cache_download

        if time is None and paths is None:
            raise ValueError("Time input is required when paths is None")

        # Forecast is available hourly for 0.25 degree grid,
        # 3 hourly for 0.5 and 1 degree grid
        # https://www.nco.ncep.noaa.gov/pmb/products/gfs/
        freq = "1h" if grid == 0.25 else "3h"
        self.timesteps = metsource.parse_timesteps(time, freq=freq)

        self.pressure_levels = metsource.parse_pressure_levels(
            pressure_levels, self.supported_pressure_levels
        )
        self.variables = metsource.parse_variables(variables, self.supported_variables)
        self.grid = metsource.parse_grid(grid, (0.25, 0.5, 1))

        # note GFS allows unsigned requests (no credentials)
        # https://stackoverflow.com/questions/34865927/can-i-use-boto3-anonymously/34866092#34866092
        self.client = boto3.client(
            "s3", config=botocore.client.Config(signature_version=botocore.UNSIGNED)
        )

        # set specific forecast time is requested
        if forecast_time is not None:
            forecast_time_pd = pd.to_datetime(forecast_time)
            if forecast_time_pd.hour % 6:
                raise ValueError("Forecast hour must be on one of 00, 06, 12, 18")

            self.forecast_time = metsource.round_hour(forecast_time_pd.to_pydatetime(), 6)

        # if no specific forecast is requested, set the forecast time using timesteps
        else:
            # round first element to the nearest 6 hour time (00, 06, 12, 18 UTC) for forecast_time
            self.forecast_time = metsource.round_hour(self.timesteps[0], 6)

    def __repr__(self) -> str:
        base = super().__repr__()
        return f"{base}\n\tForecast time: {self.forecast_time}"

    @property
    def supported_pressure_levels(self) -> list[int]:
        """Get pressure levels available.

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
            850,
            800,
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
            200,
            150,
            100,
            70,
            50,
            40,
            30,
            20,
            15,
            10,
            7,
            5,
            3,
            2,
            1,
            -1,
        ]

    @property
    def pressure_level_variables(self) -> list[met.MetVariable]:
        """GFS pressure level parameters.

        Returns
        -------
        list[MetVariable] | None
            List of MetVariable available in datasource
        """
        return PRESSURE_LEVEL_VARIABLES

    @property
    def single_level_variables(self) -> list[met.MetVariable]:
        """GFS surface level parameters.

        Returns
        -------
        list[MetVariable] | None
            List of MetVariable available in datasource
        """
        return SURFACE_VARIABLES

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
            f"{self.pressure_levels}{self.grid}{self.forecast_time}"
        )
        return hashlib.sha1(bytes(hashstr, "utf-8")).hexdigest()

    @property
    def _grid_string(self) -> str:
        """Return filename string for grid spacing."""
        if self.grid == 0.25:
            return "0p25"
        if self.grid == 0.5:
            return "0p50"
        if self.grid == 1.0:
            return "1p00"
        raise ValueError(f"Unsupported grid spacing {self.grid}. Must be one of 0.25, 0.5, or 1.0.")

    @property
    def forecast_path(self) -> str:
        """Construct forecast path in bucket for :attr:`forecast_time`.

        String template:

            GFS_FORECAST_BUCKET/gfs.YYYYMMDD/HH/atmos/{filename}",

        Returns
        -------
        str
            Bucket prefix for forecast files.
        """
        datestr = self.forecast_time.strftime("%Y%m%d")
        forecast_hour = str(self.forecast_time.hour).zfill(2)
        return f"gfs.{datestr}/{forecast_hour}/atmos"

    def filename(self, t: datetime) -> str:
        """Construct grib filename to retrieve from GFS bucket.

        String template:

            gfs.tCCz.pgrb2.GGGG.fFFF

        - ``CC`` is the model cycle runtime (i.e. 00, 06, 12, 18)
        - ``GGGG`` is the grid spacing
        - ``FFF`` is the forecast hour of product from 000 - 384

        Parameters
        ----------
        t : datetime
            Timestep to download

        Returns
        -------
        str
            Forecast filenames to retrieve from GFS bucket.

        References
        ----------
        - https://www.nco.ncep.noaa.gov/pmb/products/gfs/
        """
        step = pd.Timedelta(t - self.forecast_time) // pd.Timedelta(1, "h")
        step_hour = str(step).zfill(3)
        forecast_hour = str(self.forecast_time.hour).zfill(2)
        return f"gfs.t{forecast_hour}z.pgrb2.{self._grid_string}.f{step_hour}"

    @override
    def create_cachepath(self, t: datetime) -> str:
        if self.cachestore is None:
            raise ValueError("self.cachestore attribute must be defined to create cache path")

        # get forecast_time and step for specific file
        datestr = self.forecast_time.strftime("%Y%m%d-%H")

        # get step relative to forecast forecast_time
        step = pd.Timedelta(t - self.forecast_time) // pd.Timedelta(1, "h")

        # single level or pressure level
        suffix = f"gfs{'sl' if self.pressure_levels == [-1] else 'pl'}{self.grid}"

        # return cache path
        return self.cachestore.path(f"{datestr}-{step}-{suffix}.nc")

    @override
    def download_dataset(self, times: list[datetime]) -> None:
        # get step relative to forecast forecast_time
        logger.debug(
            f"Downloading GFS forecast for forecast time {self.forecast_time} and timesteps {times}"
        )

        # download grib file for each step file
        for t in times:
            self._download_file(t)

    @override
    def cache_dataset(self, dataset: xr.Dataset) -> None:
        # if self.cachestore is None:
        #     LOG.debug("Cache is turned off, skipping")
        #     return

        raise NotImplementedError("GFS caching only implemented with download")

    @override
    def open_metdataset(
        self,
        dataset: xr.Dataset | None = None,
        xr_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> met.MetDataset:
        xr_kwargs = xr_kwargs or {}

        #  short-circuit file paths if provided
        if dataset is not None:
            raise NotImplementedError("GFS data source does not support passing local dataset")

        if self.paths is not None:
            raise NotImplementedError("GFS data source does not support passing local paths")

            # TODO: This should work but i have type issues

            # if isinstance(self.paths, (str, pathlib.Path)):
            #     self.paths: list[str] | list[pathlib.Path] = [self.paths]

            # for (filepath, t) in zip(self.paths, self.timesteps):
            #     self._open_gfs_dataset(filepath, t)

        # load from cache or download
        if self.cachestore is None:
            raise ValueError("Cachestore is required to download data")

        # confirm files are downloaded any remote (AWS, of Cache)
        self.download(**xr_kwargs)

        # ensure all files are guaranteed to be available locally here
        # this would download a file from a remote (e.g. GCP) cache
        disk_cachepaths = [self.cachestore.get(f) for f in self._cachepaths]

        # run MetDataset constructor
        ds = self.open_dataset(disk_cachepaths, **xr_kwargs)

        # If any files are already cached, they will not have the version attached
        ds.attrs.setdefault("pycontrails_version", pycontrails.__version__)

        # run the same GFS-specific processing on the dataset
        return self._process_dataset(ds, **kwargs)

    @override
    def set_metadata(self, ds: xr.Dataset | met.MetDataset) -> None:
        ds.attrs.update(
            provider="NCEP",
            dataset="GFS",
            product="forecast",
        )

    def _download_file(self, t: datetime) -> None:
        """Download data file for forecast time and step.

        Overwrites files if they already exists.

        Parameters
        ----------
        t : datetime
            Timestep to download

        Notes
        -----
        - ``f000``:
          https://www.nco.ncep.noaa.gov/pmb/products/gfs/gfs.t00z.pgrb2.0p25.f000.shtml
        - ``f000 - f384``:
          https://www.nco.ncep.noaa.gov/pmb/products/gfs/gfs.t00z.pgrb2.0p25.f003.shtml
        """

        if self.cachestore is None:
            raise ValueError("Cachestore is required to download data")

        # construct filenames for each file
        filename = self.filename(t)
        aws_key = f"{self.forecast_path}/{filename}"

        stack = contextlib.ExitStack()
        if self.cache_download:
            target = self.cachestore.path(aws_key.replace("/", "-"))
        else:
            target = stack.enter_context(temp.temp_file())

        # Hold downloaded file in named temp file
        with stack:
            # retrieve data from AWS S3
            logger.debug(f"Downloading GFS file {filename} from AWS bucket to {target}")
            if not self.cache_download or not self.cachestore.exists(target):
                self._make_download(aws_key, target, filename)

            ds = self._open_gfs_dataset(target, t)

            cache_path = self.create_cachepath(t)
            ds.to_netcdf(cache_path)

    def _make_download(self, aws_key: str, target: str, filename: str) -> None:
        if self.show_progress:
            _download_with_progress(self.client, GFS_FORECAST_BUCKET, aws_key, target, filename)
        else:
            self.client.download_file(Bucket=GFS_FORECAST_BUCKET, Key=aws_key, Filename=target)

    def _open_gfs_dataset(self, filepath: str | pathlib.Path, t: datetime) -> xr.Dataset:
        """Open GFS grib file for one forecast timestep.

        Parameters
        ----------
        filepath : str | pathlib.Path
            Path to GFS forecast file
        t : datetime
            Timestep corresponding with GFS forecast

        Returns
        -------
        xr.Dataset
            GFS dataset
        """
        # translate into netcdf from grib
        logger.debug(f"Translating {filepath} for timestep {t!s} into netcdf")

        # get step for timestep
        step = pd.Timedelta(t - self.forecast_time) // pd.Timedelta(1, "h")

        # open file for each variable short name individually
        da_dict = {}
        for variable in self.variables:
            # Radiation data is not available in the 0th step
            is_radiation_step_zero = step == 0 and variable in (
                TOAUpwardShortwaveRadiation,
                TOAUpwardLongwaveRadiation,
            )

            if is_radiation_step_zero:
                warnings.warn(
                    "Radiation data is not provided for the 0th step in GFS. "
                    "Setting to np.nan using Visibility variable"
                )
                v = Visibility
            else:
                v = variable

            try:
                da = xr.open_dataarray(
                    filepath,
                    filter_by_keys={"typeOfLevel": v.level_type, "shortName": v.short_name},
                    engine="cfgrib",
                )
            except ValueError as exc:
                # To debug this situation, you can use:
                # import cfgrib
                # cfgrib.open_datasets(filepath)
                msg = f"Variable {v.short_name} not found in {filepath}"
                raise ValueError(msg) from exc

            if is_radiation_step_zero:
                da = xr.full_like(da, np.nan)  # set all radiation data to np.nan in the 0th step
            da_dict[variable.short_name] = da

        ds = xr.Dataset(da_dict)

        # for pressure levels, need to rename "level" field and downselect
        if self.pressure_levels != [-1]:
            ds = ds.rename({"isobaricInhPa": "level"})
            ds = ds.sel(level=self.pressure_levels)

        # for single level, and singular pressure levels, add the level dimension
        if len(self.pressure_levels) == 1:
            ds = ds.expand_dims({"level": self.pressure_levels})

        # rename fields and swap time dimension for step
        ds = ds.rename({"time": "forecast_time"})
        ds = ds.rename({"valid_time": "time"})
        ds = ds.expand_dims("time")

        # drop step/number
        return ds.drop_vars(["step", "nominalTop", "surface"], errors="ignore")

    def _process_dataset(self, ds: xr.Dataset, **kwargs: Any) -> met.MetDataset:
        """Process the :class:`xr.Dataset` opened from cache or local files.

        Parameters
        ----------
        ds : xr.Dataset
            Dataset loaded from netcdf cache files or input paths.
        **kwargs : Any
            Keyword arguments passed through directly into :class:`MetDataset` constructor.

        Returns
        -------
        MetDataset
        """

        # downselect dataset if only a subset of times, pressure levels, or variables are requested
        ds = ds[self.variable_shortnames]

        if self.timesteps:
            ds = ds.sel(time=self.timesteps)
        else:
            # set timesteps from dataset "time" coordinates
            # np.datetime64 doesn't covert to list[datetime] unless its unit is us
            self.timesteps = ds["time"].values.astype("datetime64[us]").tolist()  # type: ignore[assignment]

        # if "level" is not in dims and
        # length of the requested pressure levels is 1
        # expand the dims with this level
        if "level" not in ds.dims and len(self.pressure_levels) == 1:
            ds = ds.expand_dims({"level": self.pressure_levels})

        else:
            ds = ds.sel(level=self.pressure_levels)

        # harmonize variable names
        ds = met.standardize_variables(ds, self.variables)

        kwargs.setdefault("cachestore", self.cachestore)

        self.set_metadata(ds)
        return met.MetDataset(ds, **kwargs)


def _download_with_progress(
    client: botocore.client.S3, bucket: str, key: str, filename: str, label: str
) -> None:
    """Download with `tqdm` progress bar.

    Parameters
    ----------
    client : botocore.client.S3
        S3 Client
    bucket : str
        AWS Bucket
    key : str
        Key within bucket to download
    filename : str
        Local filename to download to
    label : str
        Progress label

    Raises
    ------
    ModuleNotFoundError
        Raises if tqdm can't be found
    """

    try:
        from tqdm import tqdm
    except ModuleNotFoundError as e:
        dependencies.raise_module_not_found_error(
            name="_download_with_progress function",
            package_name="tqdm",
            module_not_found_error=e,
            pycontrails_optional_package="gfs",
        )

    meta = client.head_object(Bucket=bucket, Key=key)
    filesize = meta["ContentLength"]

    def hook(t: Any) -> Callable:
        def inner(bytes_amount: Any) -> None:
            t.update(bytes_amount)

        return inner

    with tqdm(total=filesize, unit="B", unit_scale=True, desc=label) as t:
        client.download_file(Bucket=bucket, Key=key, Filename=filename, Callback=hook(t))
