"""Whole-file access layer for the UK Met Office ``global-deterministic-10km`` product.

Anonymous access to ``s3://met-office-atmospheric-model-data`` (eu-west-2). Requires
the additional dependency `s3fs <https://s3fs.readthedocs.io/>`_.
"""

from __future__ import annotations

import datetime
from collections.abc import Sequence

import numpy as np
import pandas as pd
import xarray as xr

from pycontrails.utils import dependencies, temp

try:
    import s3fs
except ModuleNotFoundError as exc:
    dependencies.raise_module_not_found_error(
        name="ukmo.s3 module",
        package_name="s3fs",
        module_not_found_error=exc,
        pycontrails_optional_package="ukmo",
    )

#: S3 bucket (anonymous, eu-west-2)
BUCKET = "met-office-atmospheric-model-data"

#: AWS region hosting the bucket
REGION = "eu-west-2"

#: Product prefix within the bucket
PRODUCT_PREFIX = "global-deterministic-10km"

#: Run cadence (hours): Met Office global deterministic runs at 00/06/12/18Z
RUN_CADENCE_HOURS = 6

#: Run hours (UTC)
RUN_HOURS = (0, 6, 12, 18)

#: Mapping from file-name parameter to in-file variable name
PARAMETER_VARIABLE = {
    "temperature_on_pressure_levels": "air_temperature",
    "relative_humidity_on_pressure_levels": "relative_humidity",
}

#: Lead (hours) up to which every run publishes hourly
_LAST_HOURLY_LEAD_HOURS = 54

#: Run hours (UTC) with the longest forecasts. Runs at 06/18Z end sooner.
_LONG_RUN_HOURS = (0, 12)

#: Long runs (00/12Z) publish every 3 hours until this lead (hours), then every 6 hours
#: until :data:`_LONG_RUN_LAST_LEAD_HOURS`
_LONG_RUN_LAST_3HOURLY_LEAD_HOURS = 144
_LONG_RUN_LAST_LEAD_HOURS = 168

#: Short runs (06/18Z) publish every 3 hours after the hourly period until this lead (hours)
_SHORT_RUN_LAST_LEAD_HOURS = 69

#: Short runs initialized before this time end at +66h instead of +69h. The archive holds
#: both kinds of run.
_SHORT_RUN_EXTENDED_FROM = datetime.datetime(2026, 1, 21, 18)
_SHORT_RUN_LAST_LEAD_HOURS_BEFORE_EXTENDED = 66


def latest_forecast(time: datetime.datetime) -> datetime.datetime:
    """Get the most recent forecast cycle initialized at or before a time.

    Parameters
    ----------
    time : datetime.datetime
        Specified time

    Returns
    -------
    datetime.datetime
        Initialization time of the most recent forecast at or before ``time``.

    """
    return pd.Timestamp(time).floor(f"{RUN_CADENCE_HOURS}h").to_pydatetime()


def check_forecast_time(forecast_time: datetime.datetime) -> None:
    """Raise if ``forecast_time`` isn't a time at which forecasts are initialized.

    Raises
    ------
    ValueError
        If ``forecast_time`` isn't exactly 00, 06, 12, or 18 UTC.

    """
    if (
        forecast_time.hour not in RUN_HOURS
        or forecast_time.minute
        or forecast_time.second
        or forecast_time.microsecond
    ):
        msg = (
            f"Forecast time {forecast_time} must be one of "
            f"{[f'{h:02d}:00' for h in RUN_HOURS]} UTC."
        )
        raise ValueError(msg)


def last_step(forecast_time: datetime.datetime) -> datetime.datetime:
    """Get the time of the last step of a forecast.

    Parameters
    ----------
    forecast_time : datetime.datetime
        Forecast initialization time

    Returns
    -------
    datetime.datetime
        Time of the last forecast step available in the archive.

    Raises
    ------
    ValueError
        If ``forecast_time`` isn't a time at which forecasts are initialized.

    """
    check_forecast_time(forecast_time)
    if forecast_time.hour in _LONG_RUN_HOURS:
        lead_hours = _LONG_RUN_LAST_LEAD_HOURS
    elif forecast_time >= _SHORT_RUN_EXTENDED_FROM:
        lead_hours = _SHORT_RUN_LAST_LEAD_HOURS
    else:
        lead_hours = _SHORT_RUN_LAST_LEAD_HOURS_BEFORE_EXTENDED
    return forecast_time + datetime.timedelta(hours=lead_hours)


def last_hourly_step(forecast_time: datetime.datetime) -> datetime.datetime:
    """Get the time of the last forecast step with hourly frequency.

    Parameters
    ----------
    forecast_time : datetime.datetime
        Forecast initialization time

    Returns
    -------
    datetime.datetime
        Time of the last hourly forecast step.

    Raises
    ------
    ValueError
        If ``forecast_time`` isn't a time at which forecasts are initialized.

    """
    check_forecast_time(forecast_time)
    return forecast_time + datetime.timedelta(hours=_LAST_HOURLY_LEAD_HOURS)


def last_3hourly_step(forecast_time: datetime.datetime) -> datetime.datetime:
    """Get the time of the last forecast step with 3-hourly frequency.

    Forecasts that end before 6-hourly steps begin (06/18Z) end with their last 3-hourly step.

    Parameters
    ----------
    forecast_time : datetime.datetime
        Forecast initialization time

    Returns
    -------
    datetime.datetime
        Time of the last 3-hourly forecast step.

    Raises
    ------
    ValueError
        If ``forecast_time`` isn't a time at which forecasts are initialized.

    """
    check_forecast_time(forecast_time)
    if forecast_time.hour in _LONG_RUN_HOURS:
        return forecast_time + datetime.timedelta(hours=_LONG_RUN_LAST_3HOURLY_LEAD_HOURS)
    return last_step(forecast_time)


def forecast_timestep_freq(forecast_time: datetime.datetime, last_time: datetime.datetime) -> str:
    """Get the coarsest timestep needed to reach ``last_time`` in a forecast.

    The archive publishes a forecast hourly, then every 3 hours, then (00/12Z forecasts only)
    every 6 hours. Requesting times up to ``last_time`` limits timesteps to the coarsest
    frequency found up to that time.

    Parameters
    ----------
    forecast_time : datetime.datetime
        Forecast initialization time
    last_time : datetime.datetime
        Last time requested. Must not be after :func:`last_step`.

    Returns
    -------
    str
        ``"1h"``, ``"3h"``, or ``"6h"``.

    """
    if last_time <= last_hourly_step(forecast_time):
        return "1h"
    if last_time <= last_3hourly_step(forecast_time):
        return "3h"
    return "6h"


def forecast_for_time_range(
    first_time: datetime.datetime, last_time: datetime.datetime
) -> datetime.datetime:
    """Get the forecast with the shortest lead times that covers a time range.

    This is the most recent forecast initialized at or before ``first_time`` that reaches
    ``last_time``. It is the most recent forecast for time ranges that are short enough.
    Forecasts initialized at 00/12Z are longer than those initialized at 06/18Z, so a longer
    time range may need an older forecast.

    Parameters
    ----------
    first_time : datetime.datetime
        First time requested
    last_time : datetime.datetime
        Last time requested

    Returns
    -------
    datetime.datetime
        Forecast initialization time.

    Raises
    ------
    ValueError
        If no single forecast covers the time range.

    """
    forecast_time = latest_forecast(first_time)
    oldest = last_time - datetime.timedelta(hours=_LONG_RUN_LAST_LEAD_HOURS)
    while forecast_time >= oldest:
        if last_step(forecast_time) >= last_time:
            return forecast_time
        forecast_time -= datetime.timedelta(hours=RUN_CADENCE_HOURS)

    msg = (
        f"Requested times from {first_time} to {last_time} are not contained in any single "
        f"forecast: forecasts reach at most {_LONG_RUN_LAST_LEAD_HOURS} hours after their "
        "initialization time. Request a shorter time range, or use lead_hours to combine "
        "forecasts."
    )
    raise ValueError(msg)


def run_cadence_hours_for_lead(lead_hours: int) -> int:
    """Get the interval, in hours, between valid times at a fixed lead.

    Each valid time at a fixed lead comes from a different forecast, initialized
    ``lead_hours`` earlier. Forecasts are initialized every :data:`RUN_CADENCE_HOURS`, but
    only the 00 and 12 UTC forecasts reach beyond :data:`_SHORT_RUN_LAST_LEAD_HOURS`, so
    valid times at longer leads are twice as far apart.

    Parameters
    ----------
    lead_hours : int
        Lead time in whole hours.

    Returns
    -------
    int
        6 for leads that every forecast reaches, otherwise 12.

    """
    if lead_hours > _SHORT_RUN_LAST_LEAD_HOURS:
        return 2 * RUN_CADENCE_HOURS
    return RUN_CADENCE_HOURS


def lead_for_valid_time(forecast_time: datetime.datetime, valid_time: datetime.datetime) -> int:
    """Get the lead time, in whole hours, of a valid time in a forecast.

    Parameters
    ----------
    forecast_time : datetime.datetime
        Forecast initialization time
    valid_time : datetime.datetime
        Valid time. Must be a time at which the forecast is published.

    Returns
    -------
    int
        Lead time in whole hours.

    Raises
    ------
    ValueError
        If ``forecast_time`` isn't a time at which forecasts are initialized, or if the
        archive doesn't publish the forecast at ``valid_time``: it falls before the forecast,
        after its end, or between the steps of its 3-hourly or 6-hourly period.

    """
    check_forecast_time(forecast_time)
    if valid_time < forecast_time:
        msg = f"Valid time {valid_time} is before the forecast time {forecast_time}."
        raise ValueError(msg)
    if valid_time > (end := last_step(forecast_time)):
        msg = f"Valid time {valid_time} is after the end of the {forecast_time} forecast at {end}."
        raise ValueError(msg)

    lead = (valid_time - forecast_time) / datetime.timedelta(hours=1)
    step = int(forecast_timestep_freq(forecast_time, valid_time).removesuffix("h"))
    if lead % step:
        msg = (
            f"Valid time {valid_time} is a lead of {lead:g}h in the {forecast_time} forecast, "
            f"which only publishes every {step}h at that lead."
        )
        raise ValueError(msg)
    return int(lead)


def object_key(
    run: datetime.datetime,
    valid_time: datetime.datetime,
    lead_hours: int,
    parameter: str,
) -> str:
    """Build the S3 object key for a given run, valid_time, lead and parameter.

    String template (valid time leads, not run time)::

        global-deterministic-10km/{RUN}Z/{VALID_TIME}Z-PT{LEAD}H{MIN}M-{parameter}.nc

    ``lead_hours`` is always zero-padded to 4 digits and minutes are hardcoded to
    ``00M``, since this product only ever publishes on-the-hour leads.

    Returns
    -------
    str
        Object key relative to :data:`BUCKET`.

    """
    run_str = run.strftime("%Y%m%dT%H%MZ")
    valid_time_str = valid_time.strftime("%Y%m%dT%H%MZ")
    lead_str = f"PT{lead_hours:04d}H00M"
    return f"{PRODUCT_PREFIX}/{run_str}/{valid_time_str}-{lead_str}-{parameter}.nc"


def list_run_keys(fs: s3fs.S3FileSystem, run: datetime.datetime) -> set[str]:
    """List the object keys published for a run.

    Costs a few paginated S3 requests per run, however many objects are then looked up
    in the result. The listing bypasses any directory cache held by ``fs``, so a run still
    being published is not reported from a stale listing.

    Parameters
    ----------
    fs : s3fs.S3FileSystem
        Filesystem to read through, e.g. from :func:`filesystem`.
    run : datetime.datetime
        Run time.

    Returns
    -------
    set[str]
        Object keys relative to :data:`BUCKET`, in the form returned by :func:`object_key`.
        Empty if the run has not been published.

    """
    run_prefix = f"{PRODUCT_PREFIX}/{run.strftime('%Y%m%dT%H%MZ')}"
    try:
        paths = fs.ls(f"{BUCKET}/{run_prefix}", detail=False, refresh=True)
    except FileNotFoundError:
        return set()
    return {path.removeprefix(f"{BUCKET}/") for path in paths}


def filesystem() -> s3fs.S3FileSystem:
    """Get an anonymous-access s3fs filesystem for :data:`BUCKET`.

    ``s3fs`` caches filesystem instances per ``(args, kwargs)`` within a process, so
    repeated calls with the same arguments may return the same shared instance.

    Returns
    -------
    s3fs.S3FileSystem
        Anonymous, region-pinned S3 filesystem.

    """
    return s3fs.S3FileSystem(anon=True, client_kwargs={"region_name": REGION})


def _check_time_coords_match(
    ds: xr.Dataset,
    *,
    run: datetime.datetime | None,
    valid_time: datetime.datetime | None,
    lead_hours: int | None,
) -> None:
    """Check that filename-derived run/valid time/lead match in-file time coordinates.

    Raises
    ------
    ValueError
        If any of ``run``, ``valid_time``, ``lead_hours`` is given and doesn't match
        the corresponding in-file coordinate.

    """
    if run is not None:
        file_run = pd.Timestamp(ds["forecast_reference_time"].item())
        if file_run != pd.Timestamp(run):
            msg = f"filename run {run} does not match in-file forecast_reference_time {file_run}"
            raise ValueError(msg)

    if valid_time is not None:
        file_valid_time = pd.Timestamp(ds["time"].item())
        if file_valid_time != pd.Timestamp(valid_time):
            msg = f"filename valid time {valid_time} does not match in-file time {file_valid_time}"
            raise ValueError(msg)

    if lead_hours is not None:
        file_lead = pd.Timedelta(ds["forecast_period"].item())
        if file_lead != pd.Timedelta(hours=lead_hours):
            msg = f"filename lead {lead_hours}h does not match in-file forecast_period {file_lead}"
            raise ValueError(msg)


def level_indices_for(
    pressure_pa: np.ndarray, levels_hpa: Sequence[float], key: str = "<dataset>"
) -> list[int]:
    """Get the index of each of ``levels_hpa`` within ``pressure_pa``.

    Shared by :func:`select_pressure_levels` and
    :meth:`~pycontrails.datalib.ukmo.ukmo.UM10._process_hour`, which both
    need to locate the same set of pressure levels within a fetched field's
    native ``pressure`` coordinate.

    Parameters
    ----------
    pressure_pa : np.ndarray
        In-file ``pressure`` coordinate values, in Pa.
    levels_hpa : Sequence[float]
        Target pressure levels, in hPa.
    key : str, optional
        Identifier used only in the error message.

    Returns
    -------
    list[int]
        Index into ``pressure_pa`` of each of ``levels_hpa``, in the same order.

    Raises
    ------
    ValueError
        If any level in ``levels_hpa`` isn't matched by exactly one value in
        ``pressure_pa`` (within a 1e-3 Pa tolerance, since the in-file coordinate is
        floating point).

    """
    target_pa = np.asarray(levels_hpa, dtype=np.float64) * 100.0  # convert from hPa -> Pa
    indices = []
    for level_hpa, level_pa in zip(levels_hpa, target_pa, strict=True):
        matches = np.flatnonzero(np.isclose(pressure_pa, level_pa, atol=1e-3))
        if len(matches) != 1:
            msg = f"expected exactly one {level_hpa} hPa level in {key}, found {len(matches)}"
            raise ValueError(msg)
        indices.append(int(matches[0]))
    return indices


def select_pressure_levels(
    ds: xr.Dataset,
    parameter: str,
    levels_hpa: Sequence[float],
    key: str = "<dataset>",
    *,
    extent: tuple[float, float, float, float] | None = None,
) -> xr.DataArray:
    """Select the requested pressure levels, optionally region-cropped, of a parameter.

    Pure selection logic, factored out so the same subsetting can be applied
    regardless of how ``ds`` was opened; notably used to build ground truth in
    tests from a locally-cached reference file.

    Parameters
    ----------
    ds : xr.Dataset
        Opened dataset containing ``parameter``'s variable, plus ``pressure`` and
        ``latitude``/``longitude`` coordinates.
    parameter : str
        Parameter name as it appears in the file name. Must be a key of
        :data:`PARAMETER_VARIABLE`.
    levels_hpa : Sequence[float]
        Pressure levels to select, in hPa.
    key : str, optional
        Identifier used only in error messages.
    extent : tuple[float, float, float, float], optional
        ``(lon_min, lon_max, lat_min, lat_max)`` to crop to. ``None`` (default)
        returns the full longitude/latitude range, cropped only to ``levels_hpa``.

    Returns
    -------
    xr.DataArray
        Lazy array with dims ``(pressure, latitude, longitude)``, restricted to
        ``levels_hpa`` and, if given, ``extent``. Carries ``um_version``
        and ``mosg__grid_version`` in ``.attrs`` when present on ``ds``, copied from
        the file's global attrs so callers can log per-file provenance without a
        second fetch.

    Raises
    ------
    ValueError
        If latitude or longitude isn't in ascending order, if any of
        ``levels_hpa`` isn't matched by exactly one in-file pressure level,
        or if ``extent`` crops to an empty result.

    """
    variable = PARAMETER_VARIABLE[parameter]

    latitude = ds["latitude"].values
    if not np.all(np.diff(latitude) > 0):
        msg = f"expected ascending latitude ordering in {key}"
        raise ValueError(msg)

    longitude = ds["longitude"].values
    if not np.all(np.diff(longitude) > 0):
        msg = f"expected ascending longitude ordering in {key}"
        raise ValueError(msg)

    pressure_pa = ds["pressure"].values.astype(np.float64)
    level_indices = level_indices_for(pressure_pa, levels_hpa, key=key)

    da = ds[variable].isel(pressure=level_indices)
    if extent is not None:
        lon_min, lon_max, lat_min, lat_max = extent
        da = da.sel(longitude=slice(lon_min, lon_max), latitude=slice(lat_min, lat_max))
        if da.sizes["longitude"] == 0 or da.sizes["latitude"] == 0:
            msg = f"extent {extent} crop produced an empty result for {key}"
            raise ValueError(msg)
    for attr in ("um_version", "mosg__grid_version"):
        if attr in ds.attrs:
            da.attrs[attr] = ds.attrs[attr]
    return da


def fetch_pressure_level_field(
    fs: s3fs.S3FileSystem,
    key: str,
    parameter: str,
    *,
    levels_hpa: Sequence[float],
    run: datetime.datetime | None = None,
    valid_time: datetime.datetime | None = None,
    lead_hours: int | None = None,
    extent: tuple[float, float, float, float] | None = None,
) -> xr.DataArray:
    """Download the object in full, then fetch the requested pressure levels.

    Downloads the whole object to a local temp file via ``fs.get``, opens it with
    ``netcdf4``, and loads only the selected subset into memory. The temp file is
    removed once the dataset has been closed, even if the download or selection
    fails.

    Parameters
    ----------
    fs : s3fs.S3FileSystem
        Filesystem to read through, e.g. from :func:`filesystem`.
    key : str
        Object key relative to :data:`BUCKET`, e.g. from :func:`object_key`.
    parameter : str
        Parameter name as it appears in the file name. Must be a key of
        :data:`PARAMETER_VARIABLE`.
    levels_hpa : Sequence[float]
        Pressure levels to load, in hPa.
    run : datetime.datetime, optional
        Filename-derived run time. If given, checked against the in-file
        ``forecast_reference_time``.
    valid_time : datetime.datetime, optional
        Filename-derived valid time. If given, checked against the in-file
        ``time``.
    lead_hours : int, optional
        Filename-derived lead time. If given, checked against the in-file
        ``forecast_period``.
    extent : tuple[float, float, float, float], optional
        ``(lon_min, lon_max, lat_min, lat_max)`` to crop to. ``None`` (default)
        returns the full longitude/latitude range.

    Returns
    -------
    xr.DataArray
        Loaded array with dims ``(pressure, latitude, longitude)``, restricted to
        ``levels_hpa`` and, if given, ``extent``.

    Raises
    ------
    ValueError
        If any of ``run``, ``valid_time``, ``lead_hours`` doesn't match the in-file
        coordinates, or for any of the reasons given in :func:`select_pressure_levels`.

    """
    s3_path = f"s3://{BUCKET}/{key}"
    with temp.temp_file() as target:
        fs.get(s3_path, target)
        with xr.open_dataset(
            target, engine="netcdf4", decode_times=True, decode_timedelta=True
        ) as ds:
            _check_time_coords_match(ds, run=run, valid_time=valid_time, lead_hours=lead_hours)
            return select_pressure_levels(ds, parameter, levels_hpa, key=key, extent=extent).load()
