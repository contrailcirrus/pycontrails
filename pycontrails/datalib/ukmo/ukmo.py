"""UK Met Office (UKMO) Unified Model (UM) data access.

This module supports

- Retrieving global deterministic 10 km UM forecasts from the public
  ``met-office-atmospheric-model-data`` S3 bucket.
- Selecting pressure levels and deriving specific humidity from relative humidity.
- Local caching of processed forecasts as netCDF files.
- Opening processed and cached files as a :class:`pycontrails.MetDataset`.

.. warning::

    This datalib currently supports only air temperature, relative humidity, and specific
    humidity. Therefore, it cannot be used to run
    :class:`~pycontrails.models.cocip.Cocip` or :class:`~pycontrails.models.cocipgrid.CocipGrid`
    without additional external data.
    It does provide the fields required by :class:`~pycontrails.models.issr.ISSR`,
    :class:`~pycontrails.models.sac.SAC`, and :class:`~pycontrails.models.pcr.PCR`.

    Forecasts are provided on a small set of standard pressure levels rather than on model
    levels. Near cruise altitudes the levels are spaced 25 hPa apart, which is roughly 1,900
    to 3,200 ft. This vertical resolution is coarse compared to model-level data, so thin
    ice-supersaturated layers are not resolved.
"""

from __future__ import annotations

import hashlib
import warnings
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, override

import numpy as np
import pandas as pd
import xarray as xr

import pycontrails
from pycontrails.core import met_var
from pycontrails.core.cache import CacheStore, DiskCacheStore
from pycontrails.core.met import MetDataset, MetVariable, standardize_variables
from pycontrails.datalib._met_utils import metsource
from pycontrails.datalib.ukmo import s3
from pycontrails.physics import thermo
from pycontrails.utils.types import DatetimeLike

if TYPE_CHECKING:
    import s3fs

#: MetDataset.attrs values set by :meth:`UM10.set_metadata`.
PROVIDER = "UKMO"
DATASET = "UM-10"
PRODUCT = "forecast"


#: S3 parameters (the keys of :data:`s3.PARAMETER_VARIABLE`) fetched to provide each variable.
#: Specific humidity is derived from temperature and relative humidity.
_VARIABLE_PARAMETERS = {
    met_var.AirTemperature: ("temperature_on_pressure_levels",),
    met_var.RelativeHumidity: ("relative_humidity_on_pressure_levels",),
    met_var.SpecificHumidity: (
        "temperature_on_pressure_levels",
        "relative_humidity_on_pressure_levels",
    ),
}


class UM10DataNotFoundError(Exception):
    """Raised when requested data could not be fetched from the S3 archive."""


class UM10(metsource.MetDataSource):
    """Class to support UK Met Office UM global deterministic 10 km forecast data access.

    Access is credential-free. Data is read anonymously from the
    ``met-office-atmospheric-model-data`` S3 bucket (``eu-west-2``), which requires the optional
    dependency `s3fs <https://s3fs.readthedocs.io/>`_ (``pip install pycontrails[ukmo]``).

    The Met Office Unified Model (UM) is the Met Office's operational numerical weather
    prediction model. This datalib provides the global deterministic configuration. Fields are
    provided on a regular latitude-longitude grid with 2560 x 1920 points (0.140625 degrees in
    longitude and 0.09375 degrees in latitude, approximately 10 km at mid-latitudes).
    Longitude runs from -179.93 to 179.93 degrees and latitude from -89.95 to 89.95 degrees.

    The bucket holds a rolling two year archive of forecasts, as well as
    the latest forecasts. At the time of writing, the earliest available forecast was
    initialized at 2024-10-14 18Z. Forecasts are initialized every 6 hours (00, 06, 12, and
    18 UTC). The forecast horizon and time step depend on the initialization time:

    .. list-table:: UM forecast horizon
        :header-rows: 1

        * - Forecast
          - Global (00, 12)
          - Global (06, 18)
        * - Horizon (hourly forecast)
          - +54h
          - +54h
        * - Horizon (3-hourly forecast)
          - +144h
          - +69h
        * - Horizon (6-hourly forecast)
          - +168h
          - N/A

    The 06 and 18 UTC forecasts ended at +66h rather than +69h before 2026-01-21 18Z.

    The archive provides fields on 37 pressure levels between 1000 hPa and 0.4 hPa
    (temperature) and on 33 pressure levels between 1000 hPa and 10 hPa (relative humidity).
    This datalib supports the 33 levels between 1000 hPa and 10 hPa that both fields share
    (see :attr:`supported_pressure_levels`). Near cruise altitude the pressure levels are
    300, 275, 250, 225, 200, 175, and 150 hPa (approximately 30,000 to 44,600 ft). The spacing
    between these levels varies from about 1,900 ft to 3,200 ft. This is coarse compared to
    model-level data.

    This datalib currently supports only air temperature, relative humidity, and specific
    humidity (derived from temperature and relative humidity). These fields are sufficient
    to run :class:`~pycontrails.models.issr.ISSR`, :class:`~pycontrails.models.sac.SAC`, and
    :class:`~pycontrails.models.pcr.PCR`, but this datalib does not provide winds and so cannot
    be used to run :class:`~pycontrails.models.cocip.Cocip` or
    :class:`~pycontrails.models.cocipgrid.CocipGrid`. The archive includes other fields
    (for example geopotential height, wind speed and direction, and vertical velocity on pressure
    levels). Please `contact the pycontrails developers
    <https://github.com/contrailcirrus/pycontrails/issues/new?template=feature_request.md>`_ to
    request support for additional variables.

    Fields are served on the native latitude-longitude grid. This datalib does not support
    regridding. Users who want forecasts at a different resolution must regrid the returned
    :class:`~pycontrails.MetDataset` themselves.

    Parameters
    ----------
    time : metsource.TimeInput
        The time range for data retrieval, either a single datetime or (start, end) datetime
        range. Input must be datetime-like or tuple of datetime-like specifying the (start, end)
        of the date range, inclusive.
    variables : metsource.VariableInput, optional
        Requested variables. Defaults to
        ``[met_var.AirTemperature, met_var.SpecificHumidity]``.
        Only the requested variables are cached. ``specific_humidity`` is derived from
        temperature and relative humidity, so requesting it downloads both of those fields.
        Requesting only ``air_temperature`` or ``relative_humidity`` downloads only that field.
    pressure_levels : metsource.PressureLevelInput, optional
        Requested pressure levels in hPa. Must be in :attr:`supported_pressure_levels`.
        Defaults to 300, 275, 250, 225, 200, 175, and 150 hPa (approximately 30,000 to
        44,600 ft). Each requested level and variable is held in memory as a full global
        field while a forecast hour is processed, so requesting many levels uses a lot of
        memory.
    timestep_freq : str | timedelta | None, optional
        Manually set the timestep interval within the bounds defined by ``time``.
        Supports any value that can be passed to ``pandas.date_range(freq=...)``.
        By default, this is set to the highest frequency that the forecast supports over the
        requested time range: hourly if the time range ends within 54 hours of the start of
        the forecast, then 3-hourly, then (for 00 and 12 UTC forecasts) 6-hourly. If
        ``lead_hours`` is set, this defaults to 6 hours (12 hours if ``lead_hours`` is more than
        69) and must be a multiple of that interval.
    grid : float, optional
        Not supported. This datalib does not support regridding. A non-``None`` value
        is ignored with a warning.
    forecast_time : DatetimeLike | None, optional
        Specify the forecast by initialization time. Must be 00, 06, 12, or 18 UTC.
        By default, set to the forecast with the shortest lead times that covers the requested
        time range: the most recent forecast initialized at or before the start of the time
        range, or an older one if it is needed to reach the end of the time range. All data
        come from this single forecast. A ``ValueError`` is raised if no single forecast covers
        the time range.
    lead_hours : int, optional
        Fetch data at this fixed lead time, in hours, instead of from a single forecast.
        Each timestep then comes from the forecast initialized ``lead_hours`` earlier, so the
        time range may be longer than any single forecast. Because forecasts are initialized
        every 6 hours, timesteps are 6-hourly (or a multiple of 6 hours). Only forecasts
        initialized at 00 and 12 UTC reach beyond a lead of 69 hours, so timesteps are
        12-hourly (or a multiple of 12 hours) for longer leads. Cannot be combined with
        ``forecast_time``. By default, ``None``.
    cachestore : cache.CacheStore, optional
        Cache store for fetched, processed hourly data. Defaults to
        :class:`pycontrails.core.cache.DiskCacheStore`. Other cache stores, such as
        ``cache.GCPCacheStore(bucket=...)``, can be used to read from a pre-populated archive,
        but hours missing from it can only be fetched live with a ``DiskCacheStore``.
        A cache store is required: a ``cachestore`` of ``None`` raises a ``ValueError``.

    """

    __marker = object()

    __slots__ = ("cachestore", "forecast_time", "lead_hours")

    #: Forecast initialization time, or None if fetching at a fixed ``lead_hours``
    forecast_time: datetime | None

    def __init__(
        self,
        time: metsource.TimeInput,
        *,
        variables: metsource.VariableInput | None = None,
        pressure_levels: metsource.PressureLevelInput | None = None,
        timestep_freq: str | timedelta | None = None,
        grid: float | None = None,
        forecast_time: DatetimeLike | None = None,
        lead_hours: int | None = None,
        cachestore: CacheStore = __marker,  # type: ignore[assignment]
    ) -> None:
        if grid is not None:
            warnings.warn(
                f"UM10 does not support regridding. Ignoring grid={grid!r}. "
                f"Regrid the returned MetDataset to obtain a different resolution."
            )
        self.grid = None
        self.paths = None

        if pressure_levels is None:
            pressure_levels = [300, 275, 250, 225, 200, 175, 150]
        self.pressure_levels = metsource.parse_pressure_levels(
            pressure_levels, self.supported_pressure_levels
        )

        if variables is None:
            variables = [met_var.AirTemperature, met_var.SpecificHumidity]
        self.variables = metsource.parse_variables(variables, self.supported_variables)

        if lead_hours is not None:
            self._set_timesteps_at_lead(time, timestep_freq, forecast_time, lead_hours)
        else:
            self._set_timesteps_in_forecast(time, timestep_freq, forecast_time)

        if cachestore is None:
            msg = "A cachestore is required: fetched data is only available through the cache"
            raise ValueError(msg)

        self.lead_hours = lead_hours
        self.cachestore = DiskCacheStore() if cachestore is self.__marker else cachestore

    def _set_timesteps_at_lead(
        self,
        time: metsource.TimeInput,
        timestep_freq: str | timedelta | None,
        forecast_time: DatetimeLike | None,
        lead_hours: int,
    ) -> None:
        """Set :attr:`timesteps` for data at a fixed lead, rather than from one forecast."""
        if forecast_time is not None:
            msg = "Only one of forecast_time and lead_hours can be set."
            raise ValueError(msg)
        self.forecast_time = None

        cadence = f"{s3.run_cadence_hours_for_lead(lead_hours)}h"
        if timestep_freq is None:
            timestep_freq = cadence
        if not metsource.validate_timestep_freq(timestep_freq, cadence):
            msg = (
                f"lead_hours={lead_hours} requires a timestep frequency that is a multiple "
                f"of {cadence}, because each timestep comes from a different forecast. "
                f"Got {timestep_freq}."
            )
            raise ValueError(msg)

        self.timesteps = metsource.parse_timesteps(
            time, freq=timestep_freq, shift=timedelta(hours=lead_hours)
        )
        for t in self.timesteps:
            s3.lead_for_valid_time(t - timedelta(hours=lead_hours), t)

    def _set_timesteps_in_forecast(
        self,
        time: metsource.TimeInput,
        timestep_freq: str | timedelta | None,
        forecast_time: DatetimeLike | None,
    ) -> None:
        """Set :attr:`forecast_time` and :attr:`timesteps` for data from a single forecast."""
        forecast_hours = metsource.parse_timesteps(time, freq="1h")
        first_hour, last_hour = forecast_hours[0], forecast_hours[-1]

        if forecast_time is None:
            self.forecast_time = s3.forecast_for_time_range(first_hour, last_hour)
        else:
            try:
                self.forecast_time = pd.to_datetime(forecast_time).to_pydatetime()
            except ValueError as e:
                msg = (
                    f"Failed to parse forecast time {forecast_time}. "
                    "Value must be compatible with 'pd.to_datetime'."
                )
                raise ValueError(msg) from e
            s3.check_forecast_time(self.forecast_time)

        if last_hour > (end := s3.last_step(self.forecast_time)):
            msg = f"Requested times extend to {last_hour}, beyond end of forecast at {end}."
            raise ValueError(msg)

        datasource_timestep_freq = s3.forecast_timestep_freq(self.forecast_time, last_hour)
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
        if (stop := self.timesteps[-1]) > end:
            msg = (
                f"Timesteps extend to {stop} at frequency {timestep_freq}, "
                f"beyond end of forecast at {end}."
            )
            raise ValueError(msg)

    @property
    def pressure_level_variables(self) -> list[MetVariable]:
        """Variables available from the S3 archive.

        Returns
        -------
        list[MetVariable]
            Available pressure-level variables.
        """
        return [met_var.AirTemperature, met_var.SpecificHumidity, met_var.RelativeHumidity]

    @property
    @override
    def supported_pressure_levels(self) -> list[int]:
        """Get pressure levels available.

        These are the pressure levels shared by the temperature and relative humidity fields.
        Temperature is also available at 5, 2, 1, and 0.4 hPa, but these levels are not
        supported.

        Returns
        -------
        list[int]
            List of integer pressure level values, in hPa
        """
        return [
            1000, 975, 950, 925, 900, 850, 800, 750, 700, 650, 600, 550, 500, 450, 400, 375,
            350, 325, 300, 275, 250, 225, 200, 175, 150, 125, 100, 70, 50, 40, 30, 20, 10,
        ]  # fmt: skip

    @property
    def single_level_variables(self) -> list[MetVariable]:
        """Single-level variables available.

        Returns
        -------
        list[MetVariable]
            Always empty; the S3 archive is pressure-level only.
        """
        return []

    def create_cachepath(self, t: datetime) -> str:
        """Return the cache path for the processed hourly data covering ``t``.

        Hashes the forecast initialization time, the valid time, the requested pressure
        levels, and the requested variables, so two differently-configured
        instances sharing a :attr:`cachestore` can't collide on the same path.

        Returns
        -------
        str
            Cache path for ``t``.

        Raises
        ------
        ValueError
            If :attr:`cachestore` is None.
        """
        if self.cachestore is None:
            msg = "Cachestore is required to create cache path"
            raise ValueError(msg)

        run, _ = self._run_and_lead(t)
        string = (
            f"{run:%Y%m%d%H}-{t:%Y%m%d%H}-"
            f"{'.'.join(str(p) for p in self.pressure_levels)}-"
            f"{'.'.join(sorted(self.variable_shortnames))}"
        )
        name = hashlib.md5(string.encode()).hexdigest()
        return self.cachestore.path(f"ukmo-{name}.nc")

    def cache_dataset(self, dataset: xr.Dataset) -> None:
        """Write processed hourly data to :attr:`cachestore`."""
        if self.cachestore is None:
            return

        for t, ds in dataset.groupby("time", squeeze=False):
            cache_path = self.create_cachepath(pd.Timestamp(t).to_pydatetime())
            ds.to_netcdf(cache_path, mode="w")

    def download_dataset(self, times: list[datetime]) -> None:
        """Fetch missing hours directly from the public Met Office S3 archive.

        Only called (via the inherited
        :meth:`~pycontrails.datalib._met_utils.metsource.MetDataSource.download`)
        with hours that
        :meth:`~pycontrails.datalib._met_utils.metsource.MetDataSource.is_datafile_cached`
        has already determined are missing from :attr:`cachestore`.

        Writing fetched hours to the cache is only supported for a
        :class:`~pycontrails.core.cache.DiskCacheStore`. Other cache stores, such as a
        :class:`~pycontrails.core.cache.GCPCacheStore`, can only provide hours already in them.

        Fails fast: before downloading anything, every object needed for ``times`` is
        checked against a listing of the archive, and a single
        :class:`UM10DataNotFoundError` naming every missing hour is raised if any are
        absent. Each hour is roughly 370 MB to download, so a request for data that does
        not exist is reported straight away instead of after the available hours have
        been downloaded. If an object nonetheless disappears between the check and its
        download, :class:`UM10DataNotFoundError` is raised at that hour. Hours already
        downloaded are cached and are not fetched again by a repeated call.

        Raises
        ------
        NotImplementedError
            If :attr:`cachestore` is not a :class:`~pycontrails.core.cache.DiskCacheStore`.
        UM10DataNotFoundError
            If any requested hour is not in the S3 archive.
        ValueError
            If a time in ``times`` is one the forecast doesn't publish (see
            :func:`s3.lead_for_valid_time`). This is a caller error, not a data
            availability issue, and so is not wrapped as :class:`UM10DataNotFoundError`.
        """
        if not isinstance(self.cachestore, DiskCacheStore):
            msg = (
                "Downloading data is only supported with a DiskCacheStore, got "
                f"{type(self.cachestore).__name__}. Other cache stores can only provide "
                "hours that are already in them."
            )
            raise NotImplementedError(msg)

        fs = s3.filesystem()
        self._check_archive_has(fs, times)
        for t in times:
            self._download_convert_cache_handler(fs, t)

    @property
    def _parameters(self) -> list[str]:
        """S3 parameters that must be fetched to provide the requested variables."""
        parameters = (p for v in self.variables for p in _VARIABLE_PARAMETERS[v])
        return list(dict.fromkeys(parameters))

    def _run_and_lead(self, t: datetime) -> tuple[datetime, int]:
        """Get the forecast run and lead time (hours) fetched for valid time ``t``."""
        if self.lead_hours is not None:
            run = t - timedelta(hours=self.lead_hours)
        else:
            assert self.forecast_time is not None  # set whenever lead_hours is None
            run = self.forecast_time
        return run, s3.lead_for_valid_time(run, t)

    def _check_archive_has(self, fs: s3fs.S3FileSystem, times: list[datetime]) -> None:
        """Check that every object needed for ``times`` is in the S3 archive.

        Lists each distinct run once rather than querying each object.

        Raises
        ------
        UM10DataNotFoundError
            If any needed object is absent. The message names every missing hour.
        ValueError
            If a time isn't one the forecast publishes (see :func:`s3.lead_for_valid_time`).
        """
        published: dict[datetime, set[str]] = {}
        missing: dict[datetime, list[str]] = {}
        for t in times:
            run, lead = self._run_and_lead(t)
            if run not in published:
                published[run] = s3.list_run_keys(fs, run)
            for parameter in self._parameters:
                key = s3.object_key(run, t, lead, parameter)
                if key not in published[run]:
                    missing.setdefault(t, []).append(key)

        if missing:
            detail = "; ".join(
                f"{t.isoformat()} (missing {', '.join(k)})" for t, k in missing.items()
            )
            msg = (
                f"{len(missing)} of {len(times)} hour(s) could not be fetched from the "
                f"S3 archive: {detail}"
            )
            raise UM10DataNotFoundError(msg)

    def _download_convert_cache_handler(self, fs: s3fs.S3FileSystem, t: datetime) -> None:
        """Fetch, process, and cache one missing hour.

        Raises
        ------
        UM10DataNotFoundError
            If any requested parameter's object is missing from the S3 archive.
        ValueError
            If ``t`` isn't a time the forecast publishes (see :func:`s3.lead_for_valid_time`)
            -- a caller error, not a data availability issue, so it isn't wrapped as
            :class:`UM10DataNotFoundError`.
        """
        run, lead = self._run_and_lead(t)

        data_vars = {}
        for parameter in self._parameters:
            variable = s3.PARAMETER_VARIABLE[parameter]
            key = s3.object_key(run, t, lead, parameter)
            try:
                data_vars[variable] = s3.fetch_pressure_level_field(
                    fs,
                    key,
                    parameter,
                    levels_hpa=self.pressure_levels,
                    run=run,
                    valid_time=t,
                    lead_hours=lead,
                )
            except FileNotFoundError as exc:
                msg = f"{t.isoformat()} could not be fetched from the S3 archive (key={key}): {exc}"
                raise UM10DataNotFoundError(msg) from exc

        ds = xr.Dataset(data_vars).expand_dims(time=[pd.Timestamp(t)])
        ds = self._process_hour(ds)
        ds.attrs["pycontrails_version"] = pycontrails.__version__
        self.cache_dataset(ds)

    def _process_hour(self, raw_ds: xr.Dataset) -> xr.Dataset:
        """Transform one hour of raw fetched data into the cached schema.

        Selects the requested pressure levels, computes ``specific_humidity`` if it was
        requested, and renames to the short names expected by
        :meth:`~pycontrails.datalib._met_utils.metsource.MetDataSource._check_is_ds_complete`
        (and the rest of pycontrails).

        Returns
        -------
        xr.Dataset
            Processed dataset with the requested pressure levels and variables only,
            named by short name.

        Raises
        ------
        ValueError
            If latitude isn't in ascending order, if a fetched ``relative_humidity`` isn't
            a 0-1 fraction, or if a requested pressure level isn't in the fetched data.
        """
        # MetDataset sorts coordinates on construction (copy=True), which would
        # silently mask a non-ascending latitude axis -- check it here instead,
        # while it's still attributable to this hour's raw fetched data.
        latitude = raw_ds["latitude"].values
        if not np.all(np.diff(latitude) > 0.0):
            msg = f"expected ascending latitude ordering, got {latitude[:3]}...{latitude[-3:]}"
            raise ValueError(msg)

        if "relative_humidity" in raw_ds:
            relative_humidity = raw_ds["relative_humidity"].values
            if np.any(relative_humidity < -1e-3) or np.any(relative_humidity > 1.5):
                msg = (
                    f"expected relative_humidity as a 0-1 fraction, got values in "
                    f"[{relative_humidity.min():.3g}, {relative_humidity.max():.3g}]"
                )
                raise ValueError(msg)

        pressure_pa = raw_ds["pressure"].values.astype(np.float64)
        level_indices = s3.level_indices_for(pressure_pa, self.pressure_levels)

        ds = raw_ds.isel(pressure=level_indices)
        ds = ds.rename(pressure="level")
        ds = ds.assign_coords(level=ds["level"] / 100.0)

        if met_var.SpecificHumidity in self.variables:
            level_pa = ds["level"] * 100.0
            q = ds["relative_humidity"] * thermo.q_sat_liquid(ds["air_temperature"], level_pa)
            ds = ds.assign(specific_humidity=q)

        rename = {v.standard_name: v.short_name for v in self.pressure_level_variables}
        ds = ds.rename({k: v for k, v in rename.items() if k in ds})

        keep = [v.short_name for v in self.pressure_level_variables if v in self.variables]
        return ds[keep]

    @override
    def open_metdataset(
        self,
        dataset: xr.Dataset | None = None,
        xr_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> MetDataset:
        if dataset is not None:
            msg = "Parameter 'dataset' is not supported for UM10 data"
            raise ValueError(msg)
        if self.cachestore is None:
            msg = "Cachestore is required to download data"
            raise ValueError(msg)

        xr_kwargs = dict(xr_kwargs or {})
        self.download(**xr_kwargs)

        disk_paths = [self.cachestore.get(f) for f in self._cachepaths]
        raw_ds = self.open_dataset(disk_paths, **xr_kwargs)
        raw_ds = raw_ds.sel(time=self.timesteps)
        if raw_ds.sizes["time"] != len(self.timesteps):
            msg = (
                f"expected {len(self.timesteps)} timesteps after selection, got "
                f"{raw_ds.sizes['time']} -- an hour was silently dropped"
            )
            raise ValueError(msg)

        raw_ds = standardize_variables(raw_ds, self.variables)

        kwargs.setdefault("cachestore", self.cachestore)

        mds = MetDataset(raw_ds, **kwargs)
        self.set_metadata(mds)
        return mds

    def set_metadata(self, ds: xr.Dataset | MetDataset) -> None:
        """Set ``provider``/``dataset``/``product`` attrs.

        ``PROVIDER``/``DATASET`` are registered in pycontrails core's
        ``provider_attr``/``dataset_attr`` supported lists (see
        :mod:`pycontrails.core.met`), so reading those properties back does not
        trigger a warning.
        """
        ds.attrs.update(provider=PROVIDER, dataset=DATASET, product=PRODUCT)
