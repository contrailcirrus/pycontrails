"""GFS Data Access.

References
----------
- `NOAA GFS <https://registry.opendata.aws/noaa-gfs-bdp-pds/>`_
- `Documentation <https://www.ncei.noaa.gov/products/weather-climate-models/global-forecast>`_
- `Parameter sets <https://www.nco.ncep.noaa.gov/pmb/products/gfs/>`_
"""

from __future__ import annotations

import hashlib
import logging
import pathlib
from contextlib import ExitStack
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import pandas as pd
import xarray as xr
from overrides import overrides

import pycontrails
from pycontrails.core import cache, datalib, met
from pycontrails.datalib.gfs.variables import (
    PRESSURE_LEVEL_VARIABLES,
    SURFACE_VARIABLES,
    TOAUpwardLongwaveRadiation,
    TOAUpwardShortwaveRadiation,
    Visibility,
)
from pycontrails.utils.temp import temp_file
from pycontrails.utils.types import DatetimeLike

# optional imports
if TYPE_CHECKING:
    import botocore

LOG = logging.getLogger(__name__)

#: Default GFS AWS bucket
GFS_FORECAST_BUCKET = "noaa-gfs-bdp-pds"


class GFSForecast(datalib.MetDataSource):
    """GFS Forecast data access.

    Parameters
    ----------
    time : `datalib.TimeInput`
        The time range for data retrieval, either a single datetime or (start, end) datetime range.
        Input must be a single datetime-like or tuple of datetime-like (datetime,
        :class:`pandas.Timestamp`, :class:`numpy.datetime64`)
        specifying the (start, end) of the date range, inclusive.
        All times will be downloaded for a single forecast model run nearest to the start time
        (see :attr:`forecast_time`)
        If None, ``paths`` must be defined and all time coordinates will be loaded from files.
    variables : `datalib.VariableInput`
        Variable name (i.e. "temperature", ["temperature, relative_humidity"])
        See :attr:`pressure_level_variables` for the list of available variables.
    pressure_levels : `datalib.PressureLevelInput`, optional
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
        Specify forecast run by runtime.
        Defaults to None.
    cachestore : :class:`cache.CacheStore` | None, optional
        Cache data store for staging data files.
        Defaults to :class:`cache.DiskCacheStore`.
        If None, cachestore is turned off.
    show_progress : bool, optional
        Show progress when downloading files from GFS AWS Bucket.
        Defaults to False

    Examples
    --------
    >>> from datetime import datetime
    >>> from pycontrails.datalib.gfs import GFSForecast

    >>> # Store data files to local disk (default behavior)
    >>> times = ("2022-03-22 00:00:00", "2022-03-22 03:00:00", )
    >>> gfs = GFSForecast(times, variables="air_temperature", pressure_levels=[300, 250])
    >>> gfs
    GFSForecast
        Timesteps: ['2022-03-22 00', '2022-03-22 01', '2022-03-22 02', '2022-03-22 03']
        Variables: ['t']
        Pressure levels: [300, 250]
        Grid: 0.25
        Forecast time: 2022-03-22 00:00:00
        Steps: [0, 1, 2, 3]

    Notes
    -----
    - `NOAA GFS <https://registry.opendata.aws/noaa-gfs-bdp-pds/>`_
    - `Documentation <https://www.ncei.noaa.gov/products/weather-climate-models/global-forecast>`_
    - `Parameter sets <https://www.nco.ncep.noaa.gov/pmb/products/gfs/>`_
    - `GFS Documentation <https://www.emc.ncep.noaa.gov/emc/pages/numerical_forecast_systems/gfs/documentation.php>`_
    """  # noqa: E501

    __slots__ = ("client", "grid", "cachestore", "show_progress", "forecast_time")

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
        time: datalib.TimeInput | None,
        variables: datalib.VariableInput,
        pressure_levels: datalib.PressureLevelInput = [-1],
        paths: str | list[str] | pathlib.Path | list[pathlib.Path] | None = None,
        grid: float = 0.25,
        forecast_time: DatetimeLike | None = None,
        cachestore: cache.CacheStore | None = __marker,  # type: ignore[assignment]
        show_progress: bool = False,
    ):
        try:
            import boto3
            import botocore
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "`gfs` module dependencies are missing. "
                "Please install all required dependencies using `pip install -e .[gfs]`"
            ) from e

        # inputs
        self.paths = paths
        if cachestore is self.__marker:
            cachestore = cache.DiskCacheStore()
        self.cachestore = cachestore
        self.show_progress = show_progress

        if time is None and paths is None:
            raise ValueError("Time input is required when paths is None")

        self.timesteps = datalib.parse_timesteps(time, freq="1H")
        self.pressure_levels = datalib.parse_pressure_levels(
            pressure_levels, self.supported_pressure_levels
        )
        self.variables = datalib.parse_variables(variables, self.supported_variables)
        self.grid = datalib.parse_grid(grid, [0.25, 0.5, 1])

        # note GFS allows unsigned requests (no credentials)
        # https://stackoverflow.com/questions/34865927/can-i-use-boto3-anonymously/34866092#34866092
        self.client = boto3.client(
            "s3", config=botocore.client.Config(signature_version=botocore.UNSIGNED)
        )

        # set specific forecast time is requested
        if forecast_time is not None:
            forecast_time_pd = pd.to_datetime(forecast_time)
            if forecast_time_pd.hour not in [0, 6, 12, 18]:
                raise ValueError("Forecast hour must be on one of 00, 06, 12, 18")

            self.forecast_time = datalib.round_hour(forecast_time_pd.to_pydatetime(), 6)

        # if no specific forecast is requested, set the forecast time using timesteps
        elif self.timesteps:
            # round first element to the nearest 6 hour time (00, 06, 12, 18 UTC) for forecast_time
            self.forecast_time = datalib.round_hour(self.timesteps[0], 6)

        # when no forecast_time or time input, forecast_time is defined in _open_and_cache

    def __repr__(self) -> str:
        base = super().__repr__()
        return (
            f"{base}\n\tForecast time: {getattr(self, 'forecast_time', '')}\n\tSteps:"
            f" {getattr(self, 'steps', '')}"
        )

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
    def step_offset(self) -> int:
        """Difference between :attr:`forecast_time` and first timestep.

        Returns
        -------
        int
            Number of steps to offset in order to retrieve data starting from input time.
            Returns 0 if :attr:`timesteps` is empty when loading from :attr:`paths`.
        """
        if self.timesteps:
            return int((self.timesteps[0] - self.forecast_time).total_seconds() // 3600)

        return 0

    @property
    def steps(self) -> list[int]:
        """Forecast steps from :attr:`forecast_time` corresponding within input :attr:`time`.

        Returns
        -------
        list[int]
            List of forecast steps relative to :attr:`forecast_time`
        """
        return [self.step_offset + i for i in range(len(self.timesteps))]

    @property
    def _grid_string(self) -> str:
        """Return filename string for grid spacing."""
        if self.grid == 0.25:
            return "0p25"
        if self.grid == 0.5:
            return "0p50"
        if self.grid == 1:
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

    def filename(self, step: int) -> str:
        """Construct grib filename to retrieve from GFS bucket.

        String template:

            gfs.tCCz.pgrb2.GGGG.fFFF

        - ``CC`` is the model cycle runtime (i.e. 00, 06, 12, 18)
        - ``GGGG`` is the grid spacing
        - ``FFF`` is the forecast hour of product from 000 - 384

        Parameters
        ----------
        step : int
            Integer step relative to forecast time

        Returns
        -------
        str
            Forecast filenames to retrieve from GFS bucket.

        References
        ----------
        - https://www.nco.ncep.noaa.gov/pmb/products/gfs/
        """
        forecast_hour = str(self.forecast_time.hour).zfill(2)
        return f"gfs.t{forecast_hour}z.pgrb2.{self._grid_string}.f{str(step).zfill(3)}"

    @overrides
    def create_cachepath(self, t: datetime) -> str:
        if self.cachestore is None:
            raise ValueError("self.cachestore attribute must be defined to create cache path")

        # get forecast_time and step for specific file
        datestr = self.forecast_time.strftime("%Y%m%d-%H")

        # get step relative to forecast forecast_time
        step = self.step_offset + self.timesteps.index(t)

        # single level or pressure level
        suffix = f"gfs{'sl' if self.pressure_levels == [-1] else 'pl'}{self.grid}"

        # return cache path
        return self.cachestore.path(f"{datestr}-{step}-{suffix}.nc")

    @overrides
    def download_dataset(self, times: list[datetime]) -> None:
        # get step relative to forecast forecast_time
        LOG.debug(
            f"Downloading GFS forecast for forecast time {self.forecast_time} and timesteps {times}"
        )

        # download grib file for each step file
        for t in times:
            self._download_file(t)

    @overrides
    def cache_dataset(self, dataset: xr.Dataset) -> None:
        # if self.cachestore is None:
        #     LOG.debug("Cache is turned off, skipping")
        #     return

        raise NotImplementedError("GFS caching only implemented with download")

    @overrides
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

        elif self.paths is not None:
            raise NotImplementedError("GFS data source does not support passing local paths")

            # TODO: This should work but i have type issues

            # if isinstance(self.paths, (str, pathlib.Path)):
            #     self.paths: list[str] | list[pathlib.Path] = [self.paths]

            # for (filepath, t) in zip(self.paths, self.timesteps):
            #     self._open_gfs_dataset(filepath, t)

        # load from cache or download
        else:
            if self.cachestore is None:
                raise ValueError("Cachestore is required to download data")

            # confirm files are downloaded any remote (AWS, of Cache)
            self.download(**xr_kwargs)

            # ensure all files are guaranteed to be available locally here
            # this would download a file from a remote (e.g. GCP) cache
            disk_cachepaths = [self.cachestore.get(f) for f in self._cachepaths]

            # run MetDataset constructor
            ds = self.open_dataset(disk_cachepaths, **xr_kwargs)

            # TODO: corner case
            # If any files are already cached, they will not have the version attached
            if "pycontrails_version" not in ds.attrs:
                ds.attrs["pycontrails_version"] = pycontrails.__version__

        # run the same GFS-specific processing on the dataset
        mds = self._process_dataset(ds, **kwargs)

        # set TOAUpwardShortwaveRadiation, TOAUpwardLongwaveRadiation step 0 == step 1
        for key in [
            TOAUpwardShortwaveRadiation.standard_name,
            TOAUpwardLongwaveRadiation.standard_name,
        ]:
            # if step 0 (forecast time) exists in dimension
            forecast_time = mds.data["forecast_time"].values
            if key in mds.data and forecast_time in mds.data["time"]:
                # make sure this isn't the only time in the dataset
                if np.all(mds.data["time"].values == forecast_time):
                    raise RuntimeError(
                        f"GFS datasets with data variable {key} must have at least one timestep"
                        f" after the forecast time to estimate the value of {key} at step 0"
                    )

                # set the 0th value of the data to the 1st value
                # TODO: this assumption may not be universally applicable!
                mds.data[key][dict(time=0)] = mds.data[key][dict(time=1)]

        return mds

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
        step = self.step_offset + self.timesteps.index(t)
        filename = self.filename(step)
        aws_key = f"{self.forecast_path}/{filename}"

        # Open ExitStack to control temp_file context manager
        with ExitStack() as stack:
            # hold downloaded file in named temp file
            temp_grib_filename = stack.enter_context(temp_file())

            # retrieve data from AWS S3
            LOG.debug(f"Downloading GFS file {filename} from AWS bucket to {temp_grib_filename}")
            if self.show_progress:
                _download_with_progress(
                    self.client, GFS_FORECAST_BUCKET, aws_key, temp_grib_filename, filename
                )
            else:
                self.client.download_file(
                    Bucket=GFS_FORECAST_BUCKET, Key=aws_key, Filename=temp_grib_filename
                )

            ds = self._open_gfs_dataset(temp_grib_filename, t)

            # write out data to temp, close grib file
            temp_nc_filename = stack.enter_context(temp_file())
            ds.to_netcdf(path=temp_nc_filename, mode="w")

            # put each hourly file into cache
            self.cachestore.put(temp_nc_filename, self.create_cachepath(t))

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
        LOG.debug(f"Translating {filepath} for timestep {str(t)} into netcdf")

        # get step for timestep
        step = self.step_offset + self.timesteps.index(t)

        # open file for each variable short name individually
        ds = xr.Dataset()
        for variable in self.variables:
            # radiation data is not available in the 0th step
            if step == 0 and variable in [
                TOAUpwardShortwaveRadiation,
                TOAUpwardLongwaveRadiation,
            ]:
                LOG.debug(
                    "Radiation data is not provided for the 0th step in GFS. Setting to np.nan"
                    " using Visibility variable"
                )
                v = Visibility
            else:
                v = variable

            tmpds = xr.open_dataset(
                filepath,
                filter_by_keys={"typeOfLevel": v.level_type, "shortName": v.short_name},
                engine="cfgrib",
            )

            if not len(ds):
                ds = tmpds
            else:
                ds[v.short_name] = tmpds[v.short_name]

            # set all radiation data to np.nan in the 0th step
            if step == 0 and variable in [
                TOAUpwardShortwaveRadiation,
                TOAUpwardLongwaveRadiation,
            ]:
                ds = ds.rename({Visibility.short_name: variable.short_name})
                ds[variable.short_name] = np.nan

        # for pressure levels, need to rename "level" field and downselect
        if self.pressure_levels != [-1]:
            ds = ds.rename({"isobaricInhPa": "level"})
            ds = ds.sel(dict(level=self.pressure_levels))

        # for single level, and singular pressure levels, add the level dimension
        if len(self.pressure_levels) == 1:
            ds = ds.expand_dims({"level": self.pressure_levels})

        # rename fields and swap time dimension for step
        ds = ds.rename({"time": "forecast_time"})
        ds = ds.rename({"valid_time": "time"})
        ds = ds.expand_dims("time")

        # drop step/number
        ds = ds.drop_vars(["step", "nominalTop", "surface"], errors="ignore")

        return ds

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
            self.timesteps = ds["time"].values.astype("datetime64[us]").tolist()

        # if "level" is not in dims and
        # length of the requested pressure levels is 1
        # expand the dims with this level
        if "level" not in ds.dims and len(self.pressure_levels) == 1:
            ds = ds.expand_dims({"level": self.pressure_levels})

        else:
            ds = ds.sel(dict(level=self.pressure_levels))

        # harmonize variable names
        ds = met.standardize_variables(ds, self.variables)

        if "cachestore" not in kwargs:
            kwargs["cachestore"] = self.cachestore

        ds.attrs["met_source"] = type(self).__name__
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
        raise ModuleNotFoundError(
            "Download with progress requires the `tqdm` module, "
            "which can be installed using `pip install pycontrails[gfs]`. "
            "Alternatively, set instance attribute `show_progress=False`."
        ) from e

    meta = client.head_object(Bucket=bucket, Key=key)
    filesize = meta["ContentLength"]

    def hook(t: Any) -> Callable:
        def inner(bytes_amount: Any) -> None:
            t.update(bytes_amount)

        return inner

    with tqdm(total=filesize, unit="B", unit_scale=True, desc=label) as t:
        client.download_file(Bucket=bucket, Key=key, Filename=filename, Callback=hook(t))
