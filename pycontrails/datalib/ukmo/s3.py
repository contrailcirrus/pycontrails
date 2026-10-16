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

#: Cruise pressure levels to fetch, in hPa
CRUISE_LEVELS_HPA = (300, 275, 250, 225, 200, 175, 150)

#: Run cadence (hours): Met Office global deterministic runs at 00/06/12/18Z
RUN_CADENCE_HOURS = 6

#: Mapping from file-name parameter to in-file variable name
PARAMETER_VARIABLE = {
    "temperature_on_pressure_levels": "air_temperature",
    "relative_humidity_on_pressure_levels": "relative_humidity",
}


def _check_hourly(valid_time: datetime.datetime) -> None:
    """Raise if ``valid_time`` doesn't fall on the hour."""
    if valid_time.minute or valid_time.second or valid_time.microsecond:
        msg = f"valid time {valid_time} must fall on the hour"
        raise ValueError(msg)


def run_for_valid_time(valid_time: datetime.datetime) -> datetime.datetime:
    """Get the run time for the shortest lead covering an hourly valid time.

    Shortest available lead, cycling T+0 -> T+5. Runs occur every
    :data:`RUN_CADENCE_HOURS` hours, so the run is the preceding cadence boundary.

    Parameters
    ----------
    valid_time : datetime.datetime
        Valid time. Must fall on the hour.

    Returns
    -------
    datetime.datetime
        Run time.

    """
    _check_hourly(valid_time)
    run_hour = (valid_time.hour // RUN_CADENCE_HOURS) * RUN_CADENCE_HOURS
    return valid_time.replace(hour=run_hour, minute=0, second=0, microsecond=0)


def lead_for_valid_time(valid_time: datetime.datetime) -> int:
    """Get the shortest lead, in whole hours, covering an hourly valid time.

    See :func:`run_for_valid_time` for the corresponding run time.

    Parameters
    ----------
    valid_time : datetime.datetime
        Valid time. Must fall on the hour.

    Returns
    -------
    int
        Lead time in whole hours.

    """
    _check_hourly(valid_time)
    run_hour = (valid_time.hour // RUN_CADENCE_HOURS) * RUN_CADENCE_HOURS
    return valid_time.hour - run_hour


#: Lead (hours) up to which every run publishes hourly. Beyond this, the archive
#: only publishes every :data:`_EXTENDED_CADENCE_HOURS`-th hour
_HOURLY_CADENCE_MAX_LEAD_HOURS = 54

#: Cadence (hours) beyond :data:`_HOURLY_CADENCE_MAX_LEAD_HOURS`.
_EXTENDED_CADENCE_HOURS = 3

#: Run hours (UTC) every cycle reaches: 00/06/12/18Z all publish out to
#: :data:`_ALL_CYCLE_MAX_LEAD_HOURS`.
_ALL_RUN_HOURS = (0, 6, 12, 18)

#: Run hours (UTC) that continue publishing beyond :data:`_ALL_CYCLE_MAX_LEAD_HOURS`.
#: 06Z/18Z stop there; only 00Z/12Z continue.
_LONG_RUN_HOURS = (0, 12)

#: Lead (hours) at/below which every cycle still reaches it
_ALL_CYCLE_MAX_LEAD_HOURS = 69


def _check_valid_lead(lead_hours: int) -> None:
    """Raise if ``lead_hours`` isn't a lead the archive ever publishes at."""
    if lead_hours > _HOURLY_CADENCE_MAX_LEAD_HOURS and lead_hours % _EXTENDED_CADENCE_HOURS != 0:
        msg = (
            f"lead {lead_hours}h is not a valid archive lead: beyond "
            f"T+{_HOURLY_CADENCE_MAX_LEAD_HOURS}h, only every "
            f"{_EXTENDED_CADENCE_HOURS}h is published"
        )
        raise ValueError(msg)


def run_hours_for_lead(lead_hours: int) -> tuple[int, ...]:
    """Get the run hours (UTC) whose published forecast reaches ``lead_hours``.

    Pure calendar lookup, not a network call.

    Parameters
    ----------
    lead_hours : int
        Lead time in whole hours.

    Returns
    -------
    tuple[int, ...]
        Run hours (UTC) that publish out to at least ``lead_hours``.

    """
    return _ALL_RUN_HOURS if lead_hours <= _ALL_CYCLE_MAX_LEAD_HOURS else _LONG_RUN_HOURS


def run_for_valid_time_at_lead(valid_time: datetime.datetime, lead_hours: int) -> datetime.datetime:
    """Get the run time for a fixed lead, given a valid time.

    Unlike :func:`run_for_valid_time` (shortest-lead), this fixes ``lead_hours``
    and solves for ``run = valid_time - lead_hours``.

    Parameters
    ----------
    valid_time : datetime.datetime
        Valid time. Must fall on the hour.
    lead_hours : int
        Fixed lead time in whole hours.

    Returns
    -------
    datetime.datetime
        Run time.

    Raises
    ------
    ValueError
        If ``valid_time`` isn't hourly, if ``lead_hours`` isn't a lead the archive
        ever publishes at (see :func:`_check_valid_lead`), or if the implied run
        hour is not one that reaches ``lead_hours``. Calling this for a
        valid time/lead combination the archive can't produce is a caller bug.

    """
    _check_hourly(valid_time)
    _check_valid_lead(lead_hours)
    run = valid_time - datetime.timedelta(hours=lead_hours)
    if run.hour not in run_hours_for_lead(lead_hours):
        msg = (
            f"lead {lead_hours}h at valid time {valid_time} implies run {run}, but run "
            f"hour {run.hour} does not reach that lead"
        )
        raise ValueError(msg)
    return run


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

    Shared by :func:`select_cruise_subset` and
    :meth:`~pycontrails.datalib.ukmo.ukmo.UM10._process_hour`, which both
    need to locate the same fixed set of pressure levels within a fetched field's
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


def select_cruise_subset(
    ds: xr.Dataset,
    parameter: str,
    key: str = "<dataset>",
    *,
    extent: tuple[float, float, float, float] | None = None,
) -> xr.DataArray:
    """Select the cruise-level, region-cropped subset of a parameter from a dataset.

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
    key : str, optional
        Identifier used only in error messages.
    extent : tuple[float, float, float, float], optional
        ``(lon_min, lon_max, lat_min, lat_max)`` to crop to. ``None`` (default)
        returns the full longitude/latitude range, cropped only to
        :data:`CRUISE_LEVELS_HPA`.

    Returns
    -------
    xr.DataArray
        Lazy array with dims ``(pressure, latitude, longitude)``, restricted to
        :data:`CRUISE_LEVELS_HPA` and, if given, ``extent``. Carries ``um_version``
        and ``mosg__grid_version`` in ``.attrs`` when present on ``ds``, copied from
        the file's global attrs so callers can log per-file provenance without a
        second fetch.

    Raises
    ------
    ValueError
        If latitude or longitude isn't in ascending order, if any of
        :data:`CRUISE_LEVELS_HPA` isn't matched by exactly one in-file pressure level,
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
    level_indices = level_indices_for(pressure_pa, CRUISE_LEVELS_HPA, key=key)

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
    run: datetime.datetime | None = None,
    valid_time: datetime.datetime | None = None,
    lead_hours: int | None = None,
    extent: tuple[float, float, float, float] | None = None,
) -> xr.DataArray:
    """Download the object in full, then fetch the cruise-level, region-cropped subset.

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
        :data:`CRUISE_LEVELS_HPA` and, if given, ``extent``.

    Raises
    ------
    ValueError
        If any of ``run``, ``valid_time``, ``lead_hours`` doesn't match the in-file
        coordinates, or for any of the reasons given in :func:`select_cruise_subset`.

    """
    s3_path = f"s3://{BUCKET}/{key}"
    with temp.temp_file() as target:
        fs.get(s3_path, target)
        with xr.open_dataset(
            target, engine="netcdf4", decode_times=True, decode_timedelta=True
        ) as ds:
            _check_time_coords_match(ds, run=run, valid_time=valid_time, lead_hours=lead_hours)
            return select_cruise_subset(ds, parameter, key=key, extent=extent).load()
