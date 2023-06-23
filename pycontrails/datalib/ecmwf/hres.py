"""ECWMF HRES forecast data access."""

from __future__ import annotations

import hashlib
import logging
import pathlib
from contextlib import ExitStack
from datetime import datetime
from typing import TYPE_CHECKING, Any

LOG = logging.getLogger(__name__)

import numpy as np
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
from pycontrails.utils.iteration import chunk_list
from pycontrails.utils.temp import temp_file
from pycontrails.utils.types import DatetimeLike

if TYPE_CHECKING:
    from ecmwfapi import ECMWFService


def get_forecast_filename(
    forecast_time: datetime, timestep: datetime, cc: str = "A1", S: str = "D", E: str = "1"
) -> str:
    """Create forecast filename from ECMWF dissemination products.

    The following dissemination filename convention is used for the
    transmission of ECMWF dissemination products:

    ```ccSMMDDHHIImmddhhiiE`` where:

        cc is Dissemination stream name
        S is dissemination data stream indicator
        MMDDHHII is month, day, hour and minute on which the products are based
        mmddhhii is month, day, hour and minute on which the products are valid at
            ddhhii is set to “______” for Seasonal Forecasting System products
            ii is set to 01 for high resolution forecast time step zero, type=fc, step=0
        E is the Experiment Version Number’ (as EXPVER in MARS, normally 1)

    Parameters
    ----------
    forecast_time : datetime
        Forecast time to stage
    timestep : datetime
        Time within forecast
    cc : str, optional
        Dissemination stream name.
        Defaults to "A1"
    S : str, optional
        Dissemination data stream indicator.
        Defaults to "D"
    E : str, optional
        Experiment Version Number.
        Defaults to "1"

    Returns
    -------
    str
        Filename to forecast file

    Raises
    ------
    ValueError
    """

    if forecast_time.hour not in [0, 6, 12, 18]:
        raise ValueError("Forecast time must have hour 0, 6, 12, or 18")

    if timestep < forecast_time:
        raise ValueError("Forecast timestep must be on or after forecast time")

    forecast_time_str = forecast_time.strftime("%m%d%H%M")
    if forecast_time.hour in [6, 18]:
        S = "S"

    timestep_str = timestep.strftime("%m%d%H")
    ii = "00"

    # for some reason "ii" is set to 01 for the first forecast timestep
    if forecast_time == timestep:
        ii = "01"

    return f"{cc}{S}{forecast_time_str}{timestep_str}{ii}1"


class HRES(ECMWFAPI):
    """Class to support HRES data access, download, and organization.

    Requires account with ECMWF and API key.

    API credentials set in local ``~/.ecmwfapirc`` file:

    .. code:: json

        {
            "url": "https://api.ecmwf.int/v1",
            "email": "<email>",
            "key": "<key>"
        }

    Credentials can also be provided directly ``url`` ``key``, and ``email`` keyword args.

    See `ecmwf-api-client <https://github.com/ecmwf/ecmwf-api-client>`_ documentation
    for more information.

    Parameters
    ----------
    time : datalib.TimeInput | None
        The time range for data retrieval, either a single datetime or (start, end) datetime range.
        Input must be a datetime-like or tuple of datetime-like
        (datetime, :class:`pandas.Timestamp`, :class:`numpy.datetime64`)
        specifying the (start, end) of the date range, inclusive.
        If ``forecast_time`` is unspecified, the forecast time will
        be assumed to be the nearest synoptic hour: 00, 06, 12, 18.
        All subsequent times will be downloaded for relative to :attr:`forecast_time`.
        If None, ``paths`` must be defined and all time coordinates will be loaded from files.
    variables : datalib.VariableInput
        Variable name (i.e. "air_temperature", ["air_temperature, relative_humidity"])
        See :attr:`pressure_level_variables` for the list of available variables.
    pressure_levels : datalib.PressureLevelInput, optional
        Pressure levels for data, in hPa (mbar)
        Set to -1 for to download surface level parameters.
        Defaults to -1.
    paths : str | list[str] | pathlib.Path | list[pathlib.Path] | None, optional
        Path to CDS NetCDF files to load manually.
        Can include glob patterns to load specific files.
        Defaults to None, which looks for files in the :attr:`cachestore` or CDS.
    grid : float, optional
        Specify latitude/longitude grid spacing in data.
        Defaults to 0.25.
    stream : str, optional
        "oper" = atmospheric model/HRES, "enfo" = ensemble forecast.
        Defaults to "oper" (HRES),
    field_type : str, optional
        Field type can be e.g. forecast (fc), perturbed forecast (pf),
        control forecast (cf), analysis (an).
        Defaults to "fc".
    forecast_time : DatetimeLike, optional
        Specify forecast run by runtime.
        Defaults to None.
    cachestore : cache.CacheStore | None, optional
        Cache data store for staging data files.
        Defaults to :class:`cache.DiskCacheStore`.
        If None, cache is turned off.
    url : str
        Override `ecmwf-api-client <https://github.com/ecmwf/ecmwf-api-client>`_ url
    key : str
        Override `ecmwf-api-client <https://github.com/ecmwf/ecmwf-api-client>`_ key
    email : str
        Override `ecmwf-api-client <https://github.com/ecmwf/ecmwf-api-client>`_ email

    Notes
    -----
    `MARS key word definitions <https://confluence.ecmwf.int/display/UDOC/Identification+keywords>`_

    - `class <https://apps.ecmwf.int/codes/grib/format/mars/class/>`_:
      in most cases this will be operational data, or "od"
    - `stream <https://apps.ecmwf.int/codes/grib/format/mars/stream/>`_:
      "enfo" = ensemble forecast, "oper" = atmospheric model/HRES
    - `expver <https://confluence.ecmwf.int/pages/viewpage.action?pageId=124752178>`_:
      experimental version, production data is 1 or 2
    - `date <https://confluence.ecmwf.int/pages/viewpage.action?pageId=118817289>`_:
      there are numerous acceptible date formats
    - `time <https://confluence.ecmwf.int/pages/viewpage.action?pageId=118817378>`_:
      forecast base time, always in synoptic time (0,6,12,18 UTC)
    - `type <https://confluence.ecmwf.int/pages/viewpage.action?pageId=127315300>`_:
      forecast (oper), perturbed or control forecast (enfo only), or analysis
    - `levtype <https://confluence.ecmwf.int/pages/viewpage.action?pageId=149335319>`_:
      options include surface, pressure levels, or model levels
    - `levelist <https://confluence.ecmwf.int/pages/viewpage.action?pageId=149335403>`_:
      list of levels in format specified by **levtype** `levelist`_
    - `param <https://confluence.ecmwf.int/pages/viewpage.action?pageId=149335858>`_:
      list of variables in catalog number, long name or short name
    - `step <https://confluence.ecmwf.int/pages/viewpage.action?pageId=118820050>`_:
      hourly time steps from base forecast time
    - `number <https://confluence.ecmwf.int/pages/viewpage.action?pageId=149335478>`_:
      for ensemble forecasts, ensemble numbers
    - `format <https://confluence.ecmwf.int/pages/viewpage.action?pageId=116970058>`_:
      specify netcdf instead of default grib, DEPRECATED `format`_
    - `grid <https://confluence.ecmwf.int/pages/viewpage.action?pageId=123799065>`_:
      specify model return grid spacing

    Local ``paths`` are loaded using :func:`xarray.open_mfdataset`.
    Pass ``xr_kwargs`` inputs to :meth:`open_metdataset` to customize file loading.


    Examples
    --------
    >>> from datetime import datetime
    >>> from pycontrails import GCPCacheStore
    >>> from pycontrails.datalib.ecmwf import HRES

    >>> # Store data files to local disk (default behavior)
    >>> times = (datetime(2021, 5, 1, 2), datetime(2021, 5, 1, 3))
    >>> hres = HRES(times, variables="air_temperature", pressure_levels=[300, 250])

    >>> # Cache files to google cloud storage
    >>> gcp_cache = GCPCacheStore(bucket="contrails-301217-unit-test", cache_dir="ecmwf")
    >>> hres = HRES(
    ...     times,
    ...     variables="air_temperature",
    ...     pressure_levels=[300, 250],
    ...     cachestore=gcp_cache
    ... )
    """

    __slots__ = ("server", "stream", "field_type", "forecast_time", "url", "key", "email")

    #: stream type, "oper" = atmospheric model/HRES, "enfo" = ensemble forecast.
    stream: str

    #: Field type, forecast ("fc"), perturbed forecast ("pf"),
    #: control forecast ("cf"), analysis ("an").
    field_type: str

    #: Handle to ECMWFService client
    server: ECMWFService

    #: Forecast run time, either specified or assigned by the closest previous forecast run
    forecast_time: datetime

    #: User provided ``ECMWFService`` url
    url: str | None

    #: User provided ``ECMWFService`` key
    key: str | None

    #: User provided ``ECMWFService`` email
    email: str | None

    __marker = object()

    def __init__(
        self,
        time: datalib.TimeInput | None,
        variables: datalib.VariableInput,
        pressure_levels: datalib.PressureLevelInput = -1,
        paths: str | list[str] | pathlib.Path | list[pathlib.Path] | None = None,
        cachepath: str | list[str] | pathlib.Path | list[pathlib.Path] | None = None,
        grid: float = 0.25,
        stream: str = "oper",
        field_type: str = "fc",
        forecast_time: DatetimeLike | None = None,
        cachestore: cache.CacheStore | None = __marker,  # type: ignore[assignment]
        url: str | None = None,
        key: str | None = None,
        email: str | None = None,
    ) -> None:
        try:
            from ecmwfapi import ECMWFService
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "Some `ecmwf` module dependencies are missing. "
                "Please install all required dependencies using `pip install -e .[ecmwf]`"
            ) from e

        # constants
        # ERA5 now delays creating the server attribute until it is needed to download
        # from CDS. We could do the same here.
        self.url = url
        self.key = key
        self.email = email
        self.server = ECMWFService("mars", url=self.url, key=self.key, email=self.email)
        self.paths = paths
        if cachestore is self.__marker:
            cachestore = cache.DiskCacheStore()
        self.cachestore = cachestore

        if time is None and paths is None:
            raise ValueError("Time input is required when paths is None")

        self.timesteps = datalib.parse_timesteps(time, freq="1H")
        self.pressure_levels = datalib.parse_pressure_levels(
            pressure_levels, self.supported_pressure_levels
        )
        self.variables = datalib.parse_variables(variables, self.supported_variables)

        self.grid = datalib.parse_grid(grid, [0.1, 0.25, 0.5, 1])  # lat/lon degree resolution
        self.stream = stream  # "enfo" = ensemble forecast, "oper" = atmospheric model/HRES
        self.field_type = (
            field_type  # forecast (oper), perturbed or control forecast (enfo only), or analysis
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

    @classmethod
    def create_synoptic_time_ranges(
        self, timesteps: list[pd.Timestamp]
    ) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
        """Create synoptic time bounds encompassing date range.

        Extracts time bounds for synoptic time range ([00:00, 11:59], [12:00, 23:59])
        for a list of input timesteps.

        Parameters
        ----------
        timesteps : list[pd.Timestamp]
            List of timesteps formatted as :class:`pd.Timestamps`.
            Often this it the output from `pd.date_range()`

        Returns
        -------
        list[tuple[pd.Timestamp, pd.Timestamp]]
            List of tuple time bounds that can be used as inputs to :class:`HRES(time=...)`
        """
        time_ranges = np.unique(
            [pd.Timestamp(t.year, t.month, t.day, 12 * (t.hour // 12)) for t in timesteps]
        )

        if len(time_ranges) == 1:
            time_ranges = [(timesteps[0], timesteps[-1])]
        else:
            time_ranges[0] = (
                timesteps[0],
                time_ranges[1] - pd.Timedelta(1, "H"),
            )
            time_ranges[1:-1] = [(t, t + pd.Timedelta(11, "H")) for t in time_ranges[1:-1]]
            time_ranges[-1] = (time_ranges[-1], timesteps[-1])

        return time_ranges

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
            f"{self.pressure_levels}{self.grid}{self.forecast_time}{self.field_type}{self.stream}"
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
        """Get pressure levels available from MARS.

        Returns
        -------
        list[int]
            List of integer pressure level values
        """
        return [
            1000,
            950,
            925,
            900,
            850,
            800,
            700,
            600,
            500,
            400,
            300,
            250,
            200,
            150,
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

    def list_from_mars(self) -> str:
        """List metadata on query from MARS.

        Returns
        -------
        str
            Metadata for MARS request.
            Note this is queued the same as data requests.
        """
        request = self.generate_mars_request(self.forecast_time, self.steps, request_type="list")

        # hold downloaded file in named temp file
        with temp_file() as mars_temp_filename:
            LOG.debug(f"Performing MARS request: {request}")
            self.server.execute(request, mars_temp_filename)

            with open(mars_temp_filename, "r") as f:
                txt = f.read()

        return txt

    def generate_mars_request(
        self,
        forecast_time: datetime | None = None,
        steps: list[int] | None = None,
        request_type: str = "retrieve",
        request_format: str = "mars",
    ) -> str | dict[str, Any]:
        """Generate MARS request in MARS request syntax.

        Parameters
        ----------
        forecast_time : :class:`datetime`, optional
            Base datetime for the forecast.
            Defaults to :attr:`forecast_time`.
        steps : list[int], optional
            list of steps.
            Defaults to :attr:`steps`.
        request_type : str, optional
            "retrieve" for download request or "list" for metadata request.
            Defaults to "retrieve".
        request_format : str, optional
            "mars" for MARS string format, or "dict" for dict version.
            Defaults to "mars".

        Returns
        -------
        str | dict[str, Any]
            Returns MARS query string if ``request_format`` is "mars".
            Returns dict query if ``request_format`` is "dict"

        Notes
        -----
        Brief overview of `MARS request syntax
        <https://confluence.ecmwf.int/display/WEBAPI/Brief+MARS+request+syntax>`_
        """

        if forecast_time is None:
            forecast_time = self.forecast_time

        if steps is None:
            steps = self.steps

        # set date/time for file
        _date = forecast_time.strftime("%Y%m%d")
        _time = forecast_time.strftime("%H")

        # make request of mars
        request: dict[str, Any] = {
            "class": "od",  # operational data
            "stream": self.stream,
            "expver": "1",  # production data only
            "date": _date,
            "time": _time,
            "type": self.field_type,
            "param": f"{'/'.join(self.variable_shortnames)}",
            "step": f"{'/'.join([str(s) for s in steps])}",
            "grid": f"{self.grid}/{self.grid}",
        }

        if self.pressure_levels != [-1]:
            request["levtype"] = "pl"
            request["levelist"] = f"{'/'.join([str(pl) for pl in self.pressure_levels])}"
        else:
            request["levtype"] = "sfc"

        if request_format == "dict":
            return request

        levelist = f",\n\tlevelist={request['levelist']}" if self.pressure_levels != [-1] else ""
        return (
            f"{request_type},\n\tclass={request['class']},\n\tstream={request['stream']},"
            f"\n\texpver={request['expver']},\n\tdate={request['date']},"
            f"\n\ttime={request['time']},\n\ttype={request['type']},"
            f"\n\tparam={request['param']},\n\tstep={request['step']},"
            f"\n\tgrid={request['grid']},\n\tlevtype={request['levtype']}{levelist}"
        )

    @overrides
    def create_cachepath(self, t: datetime) -> str:
        if self.cachestore is None:
            raise ValueError("self.cachestore attribute must be defined to create cache path")

        # get forecast_time and step for specific file
        datestr = self.forecast_time.strftime("%Y%m%d-%H")

        # get step relative to forecast forecast_time
        step = self.step_offset + self.timesteps.index(t)

        # single level or pressure level
        if self.pressure_levels == [-1]:
            suffix = f"hressl{self.grid}{self.stream}{self.field_type}"
        else:
            suffix = f"hrespl{self.grid}{self.stream}{self.field_type}"

        # return cache path
        return self.cachestore.path(f"{datestr}-{step}-{suffix}.nc")

    @overrides
    def download_dataset(self, times: list[datetime]) -> None:
        """Download data from data source for input times.

        Parameters
        ----------
        times : list[:class:`datetime`]
            List of datetimes to download and store in cache datastore
        """

        # get step relative to forecast forecast_time
        steps = [self.step_offset + self.timesteps.index(t) for t in times]
        LOG.debug(f"Downloading HRES dataset for base time {self.forecast_time} and steps {steps}")

        # download in sets of 24
        if len(steps) > 24:
            for _steps in chunk_list(steps, 24):
                self._download_file(_steps)
        elif len(steps) > 0:
            self._download_file(steps)

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
            self.cachestore.put_multiple(
                xarray_temp_filenames,
                [
                    self.create_cachepath(datetime.utcfromtimestamp(tg.tolist() / 1e9))
                    for tg in time_group
                ],
            )

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
            ds = self._preprocess_hres_dataset(dataset)

        # load from local paths
        elif self.paths is not None:
            ds = self._open_and_cache(xr_kwargs)

        # download from MARS
        else:
            if self.cachestore is None:
                raise ValueError("Cachestore is required to download data")

            # confirm files are downloaded from CDS or MARS
            self.download(**xr_kwargs)

            # ensure all files are guaranteed to be available locally here
            # this would download a file from a remote (e.g. GCP) cache
            disk_cachepaths = [self.cachestore.get(f) for f in self._cachepaths]

            # open cache files as xr.Dataset
            ds = self.open_dataset(disk_cachepaths, **xr_kwargs)

            # TODO: corner case
            # If any files are already cached, they will not have the version attached
            if "pycontrails_version" not in ds.attrs:
                ds.attrs["pycontrails_version"] = pycontrails.__version__

        # run the same ECMWF-specific processing on the dataset
        mds = self._process_dataset(ds, **kwargs)

        # convert accumulated radiation values to average instantaneous values
        # set minimum for all values to 0
        # !! Note that HRES accumulates from the *start of the forecast*,
        # so we need to take the diff of each accumulated value
        # the 0th value is set to the 1st value so each time step has a radiation value !!
        dt_accumulation = 60 * 60

        for key in [
            TOAIncidentSolarRadiation.standard_name,
            TopNetSolarRadiation.standard_name,
            TopNetThermalRadiation.standard_name,
        ]:
            if key in mds.data:
                if len(mds.data["time"]) < 2:
                    raise RuntimeError(
                        f"HRES datasets with data variable {key} must have at least two timesteps"
                        f" to calculate the average instantaneous value of {key}"
                    )

                # take the difference between time slices
                dkey_dt = mds.data[key].diff("time")

                # set difference value back to the data model
                mds.data[key] = dkey_dt

                # set the 0th value of the data to the 1st difference value
                # TODO: this assumption may not be universally applicable!
                mds.data[key][dict(time=0)] = dkey_dt[dict(time=0)]

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

        # set default parameters for loading grib files
        xr_kwargs.setdefault("engine", "cfgrib")
        xr_kwargs.setdefault("combine", "nested")
        xr_kwargs.setdefault("concat_dim", "step")
        xr_kwargs.setdefault("parallel", False)
        ds = self.open_dataset(self.paths, **xr_kwargs)

        # set forecast time if its not already defined
        if not getattr(self, "forecast_time", None):
            self.forecast_time = ds["time"].values.astype("datetime64[s]").tolist()

        # check that forecast_time is correct if defined
        # note the "time" coordinate here is the HRES forecast_time
        elif self.forecast_time != ds["time"].values.astype("datetime64[s]").tolist():
            raise ValueError(
                f"HRES.forecast_time {self.forecast_time} is not the same forecast time listed"
                " in file"
            )

        ds = self._preprocess_hres_dataset(ds)

        # set timesteps if not defined
        # note that "time" is now the actual timestep coordinates
        if not self.timesteps:
            self.timesteps = ds["time"].values.astype("datetime64[s]").tolist()

        self.cache_dataset(ds)

        return ds

    def _download_file(self, steps: list[int]) -> None:
        """Download data file for base datetime and timesteps.

        Overwrites files if they already exists.

        Parameters
        ----------
        steps : list[int]
            Steps to download relative to base date
        """
        request = self.generate_mars_request(self.forecast_time, steps)

        # Open ExitStack to control temp_file context manager
        with ExitStack() as stack:
            # hold downloaded file in named temp file
            mars_temp_grib_filename = stack.enter_context(temp_file())

            # retrieve data from MARS
            LOG.debug(f"Performing MARS request: {request}")
            self.server.execute(request, mars_temp_grib_filename)

            # translate into netcdf from grib
            LOG.debug("Translating file into netcdf")
            mars_temp_nc_filename = stack.enter_context(temp_file())
            ds = stack.enter_context(xr.open_dataset(mars_temp_grib_filename, engine="cfgrib"))

            ##### TODO: do we need to store intermediate netcdf file?
            ds.to_netcdf(path=mars_temp_nc_filename, mode="w")

            # open file, edit, and save for each hourly time step
            ds = stack.enter_context(
                xr.open_dataset(mars_temp_nc_filename, engine=datalib.NETCDF_ENGINE)
            )
            #####

            # run preprocessing before cache
            ds = self._preprocess_hres_dataset(ds)

            self.cache_dataset(ds)

    def _preprocess_hres_dataset(self, ds: xr.Dataset) -> xr.Dataset:
        """Process HRES data before caching.

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

        # for pressure levels, need to rename "level" field
        if self.pressure_levels != [-1]:
            ds = ds.rename({"isobaricInhPa": "level"})

        # for single level, and singular pressure levels, add the level dimension
        if len(self.pressure_levels) == 1:
            ds = ds.expand_dims({"level": self.pressure_levels})

        # for single time, add the step dimension and assign time coords to step
        if ds["step"].size == 1:
            if "step" not in ds.dims:
                ds = ds.expand_dims({"step": [ds["step"].values]})

            ds = ds.assign_coords({"valid_time": ("step", [ds["valid_time"].values])})

        # rename fields and swap time dimension for step
        ds = ds.rename({"time": "forecast_time"})
        ds = ds.rename({"valid_time": "time"})
        ds = ds.swap_dims({"step": "time"})

        # drop step/number
        ds = ds.drop_vars(["step", "number"], errors="ignore")

        ds.attrs["pycontrails_version"] = pycontrails.__version__
        return ds
