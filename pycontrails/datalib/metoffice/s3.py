"""Range-read access layer for the Met Office ``global-deterministic-10km`` product.

Anonymous access to ``s3://met-office-atmospheric-model-data`` (eu-west-2). Fetches
only the seven cruise pressure levels, optionally cropped to a caller-supplied
bounding box, via byte-range reads; skips the ``flag`` variable entirely.

Uses a plain ``boto3`` client wrapped in a minimal seekable file-like object.
Real concurrency requires separate processes, not threads, since ``h5py``/HDF5
serializes internally within one process.

This is a standalone module: it knows nothing about pycontrails' met data model. The
pycontrails datalib (``ukmo.py``) wraps it.

Key convention (validity time leads, not run time)::

    global-deterministic-10km/{RUN}Z/{VALIDITY}Z-PT{LEAD}H{MIN}M-{parameter}.nc

Lead selection follows the shortest-available, T+0->T+5 cycling scheme: runs occur
every 6 hours (00/06/12/18Z), so for any hourly validity time the run is the
preceding 6-hour boundary and the lead is the hour offset from it.

Chunk layout varies by archive vintage: older files have a much finer native chunk
layout than the current live product, so per-hour fetch time is dominated by
request count rather than bytes transferred.

This module requires the following additional dependency:

- `boto3 <https://boto3.amazonaws.com/v1/documentation/api/latest/index.html>`_

"""

from __future__ import annotations

import datetime
from collections.abc import Collection

import numpy as np
import pandas as pd
import xarray as xr

from pycontrails.utils import dependencies

try:
    import boto3
    from botocore import UNSIGNED
    from botocore.client import BaseClient
    from botocore.config import Config
except ModuleNotFoundError as exc:
    dependencies.raise_module_not_found_error(
        name="metoffice.s3 module",
        package_name="boto3",
        module_not_found_error=exc,
        pycontrails_optional_package="metoffice",
    )

#: S3 bucket (anonymous, eu-west-2)
BUCKET = "met-office-atmospheric-model-data"

#: AWS region hosting the bucket
REGION = "eu-west-2"

#: Product prefix within the bucket
PRODUCT_PREFIX = "global-deterministic-10km"

#: Cruise pressure levels to fetch, in hPa
CRUISE_LEVELS_HPA = (300, 275, 250, 225, 200, 175, 150)

#: Example extent covering CONUS, (lon_min, lon_max, lat_min, lat_max). Not used as
#: a default anywhere in this module; callers pass it explicitly if they want it.
CONUS_EXTENT = (-134, -63, 20, 50)

#: Run cadence (hours): Met Office global deterministic runs at 00/06/12/18Z
RUN_CADENCE_HOURS = 6

#: Mapping from file-name parameter to in-file variable name
PARAMETER_VARIABLE = {
    "temperature_on_pressure_levels": "air_temperature",
    "relative_humidity_on_pressure_levels": "relative_humidity",
}


def run_and_lead_for_validity(validity: datetime.datetime) -> tuple[datetime.datetime, int]:
    """Get the shortest-lead (run, lead) pair covering an hourly validity time.

    Shortest available lead, cycling T+0 -> T+5. Runs occur every
    :data:`RUN_CADENCE_HOURS` hours, so the run is the preceding cadence boundary and
    the lead is the hour offset from it.

    Parameters
    ----------
    validity : datetime.datetime
        Validity (forecast) time. Must fall on the hour.

    Returns
    -------
    tuple[datetime.datetime, int]
        Run time and lead time in whole hours.

    """
    if validity.minute or validity.second or validity.microsecond:
        msg = f"validity time {validity} must fall on the hour"
        raise ValueError(msg)

    run_hour = (validity.hour // RUN_CADENCE_HOURS) * RUN_CADENCE_HOURS
    run = validity.replace(hour=run_hour, minute=0, second=0, microsecond=0)
    lead_hours = validity.hour - run_hour
    return run, lead_hours


#: Run hours (UTC) every cycle reaches: hourly cadence out to T+54
#: (~4,527 objects/run breakdown: "55 hourly to T+54").
_ALL_RUN_HOURS = (0, 6, 12, 18)

#: Run hours (UTC) that continue publishing beyond T+54. 06Z/18Z stop around
#: T+54-67, so only 00Z/12Z reach T+72 and beyond.
_LONG_RUN_HOURS = (0, 12)

#: Lead (hours) at/below which every cycle still reaches it.
_ALL_CYCLE_MAX_LEAD_HOURS = 54


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


def run_for_validity_at_lead(validity: datetime.datetime, lead_hours: int) -> datetime.datetime:
    """Get the run time for a *fixed* lead, given a validity time.

    Unlike :func:`run_and_lead_for_validity` (shortest-lead), this fixes
    ``lead_hours`` and solves for ``run = validity - lead_hours``.

    Parameters
    ----------
    validity : datetime.datetime
        Validity (forecast) time. Must fall on the hour.
    lead_hours : int
        Fixed lead time in whole hours.

    Returns
    -------
    datetime.datetime
        Run time.

    Raises
    ------
    ValueError
        If ``validity`` isn't hourly, or if the implied run hour is not one that
        reaches ``lead_hours``. Calling this for a validity/lead
        combination the archive can't produce is a caller bug.

    """
    if validity.minute or validity.second or validity.microsecond:
        msg = f"validity time {validity} must fall on the hour"
        raise ValueError(msg)

    run = validity - datetime.timedelta(hours=lead_hours)
    if run.hour not in run_hours_for_lead(lead_hours):
        msg = (
            f"lead {lead_hours}h at validity {validity} implies run {run}, but run "
            f"hour {run.hour} does not reach that lead"
        )
        raise ValueError(msg)
    return run


def available_validity_times_at_lead(
    start: datetime.datetime, end: datetime.datetime, lead_hours: int
) -> list[datetime.datetime]:
    """Get every hourly validity time reachable at a fixed ``lead_hours``.

    Pure calendar arithmetic; no network I/O, no existence check
    against the mirror or the S3 archive.

    Parameters
    ----------
    start, end : datetime.datetime
        Inclusive validity-time bounds.
    lead_hours : int
        Fixed lead time in whole hours.

    Returns
    -------
    list[datetime.datetime]
        Hourly validity times in ``[start, end]`` whose implied run hour reaches
        ``lead_hours``, ascending.

    """
    run_hours = run_hours_for_lead(lead_hours)
    candidates = pd.date_range(start, end, freq="1h").to_pydatetime().tolist()
    return [v for v in candidates if (v - datetime.timedelta(hours=lead_hours)).hour in run_hours]


def matched_validity_times(
    start: datetime.datetime, end: datetime.datetime, leads: Collection[int]
) -> list[datetime.datetime]:
    """Get the intersection of validity times available at every lead in ``leads``.

    This is the set that must be compared across leads: the T+0 baseline must
    also be restricted to this same intersection, not the full per-lead set and
    not the existing shortest-lead mirror's validity set.

    Parameters
    ----------
    start, end : datetime.datetime
        Inclusive validity-time bounds.
    leads : Collection[int]
        Fixed lead times (whole hours) to intersect over.

    Returns
    -------
    list[datetime.datetime]
        Ascending validity times available at every lead in ``leads``. Empty if
        ``leads`` is empty.

    """
    sets = [set(available_validity_times_at_lead(start, end, lead)) for lead in leads]
    return sorted(set.intersection(*sets)) if sets else []


def object_key(
    run: datetime.datetime,
    validity: datetime.datetime,
    lead_hours: int,
    parameter: str,
) -> str:
    """Build the S3 object key for a given run, validity, lead and parameter.

    Parameters
    ----------
    run : datetime.datetime
        Model run time.
    validity : datetime.datetime
        Forecast validity time.
    lead_hours : int
        Lead time in whole hours.
    parameter : str
        Parameter name as it appears in the file name, e.g.
        ``"temperature_on_pressure_levels"``.

    Returns
    -------
    str
        Object key relative to :data:`BUCKET`.

    """
    run_str = run.strftime("%Y%m%dT%H%MZ")
    validity_str = validity.strftime("%Y%m%dT%H%MZ")
    lead_str = f"PT{lead_hours:04d}H00M"
    return f"{PRODUCT_PREFIX}/{run_str}/{validity_str}-{lead_str}-{parameter}.nc"


def filesystem() -> BaseClient:
    """Get a fresh anonymous-access S3 client for :data:`BUCKET`.

    Not cached/shared; each caller (e.g. each mirror worker process) should get its
    own client with its own connection pool.

    Returns
    -------
    boto3.client
        Anonymous, region-pinned S3 client.

    """
    return boto3.client(
        "s3",
        region_name=REGION,
        config=Config(signature_version=UNSIGNED, max_pool_connections=32),
    )


class _S3RangeFile:
    """Minimal seekable, read-only file-like object backed by ``get_object`` range reads.

    Satisfies h5py's generic file-like ("fileobj") driver: ``read``/``seek``/``tell``
    plus the ``seekable``/``readable``/``writable`` capability queries. Issues one
    exact byte-range GET per ``read()`` call, with no speculative over-fetching and
    no shared cache, since ``h5py`` already knows exactly which bytes it needs per
    chunk.
    """

    def __init__(self, client: BaseClient, bucket: str, key: str) -> None:
        self._client = client
        self._bucket = bucket
        self._key = key
        self._pos = 0
        self._size = client.head_object(Bucket=bucket, Key=key)["ContentLength"]

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def writable(self) -> bool:
        return False

    def seek(self, offset: int, whence: int = 0) -> int:
        if whence == 0:
            self._pos = offset
        elif whence == 1:
            self._pos += offset
        elif whence == 2:
            self._pos = self._size + offset
        else:
            msg = f"invalid whence {whence}"
            raise ValueError(msg)
        return self._pos

    def tell(self) -> int:
        return self._pos

    def read(self, size: int | None = -1) -> bytes:
        end = self._size - 1 if size is None or size < 0 else min(self._pos + size, self._size) - 1
        if self._pos > end:
            return b""
        response = self._client.get_object(
            Bucket=self._bucket, Key=self._key, Range=f"bytes={self._pos}-{end}"
        )
        data = response["Body"].read()
        self._pos += len(data)
        return data

    def close(self) -> None:
        """No-op; there is no open resource to release."""


def _assert_time_coords_match(
    ds: xr.Dataset,
    *,
    run: datetime.datetime | None,
    validity: datetime.datetime | None,
    lead_hours: int | None,
) -> None:
    """Assert filename-derived run/validity/lead match in-file time coordinates."""
    if run is not None:
        file_run = pd.Timestamp(ds["forecast_reference_time"].item())
        if file_run != pd.Timestamp(run):
            msg = f"filename run {run} does not match in-file forecast_reference_time {file_run}"
            raise AssertionError(msg)

    if validity is not None:
        file_validity = pd.Timestamp(ds["time"].item())
        if file_validity != pd.Timestamp(validity):
            msg = f"filename validity {validity} does not match in-file time {file_validity}"
            raise AssertionError(msg)

    if lead_hours is not None:
        file_lead = pd.Timedelta(ds["forecast_period"].item())
        if file_lead != pd.Timedelta(hours=lead_hours):
            msg = f"filename lead {lead_hours}h does not match in-file forecast_period {file_lead}"
            raise AssertionError(msg)


def select_cruise_subset(
    ds: xr.Dataset,
    parameter: str,
    key: str = "<dataset>",
    *,
    extent: tuple[float, float, float, float] | None = None,
) -> xr.DataArray:
    """Select the cruise-level, region-cropped subset of a parameter from a dataset.

    Pure selection logic, factored out so the same subsetting can be applied to a
    dataset opened by any means (S3 byte-range read, local file); notably used to
    build ground truth in tests.

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

    """
    variable = PARAMETER_VARIABLE[parameter]

    latitude = ds["latitude"].values
    if not np.all(np.diff(latitude) > 0):
        msg = f"expected ascending latitude ordering in {key}"
        raise AssertionError(msg)

    pressure_pa = ds["pressure"].values.astype(np.float64)
    target_pa = np.asarray(CRUISE_LEVELS_HPA, dtype=np.float64) * 100.0
    level_indices = []
    for level_hpa, level_pa in zip(CRUISE_LEVELS_HPA, target_pa, strict=True):
        matches = np.flatnonzero(np.isclose(pressure_pa, level_pa, atol=1e-3))
        if len(matches) != 1:
            msg = f"expected exactly one {level_hpa} hPa level in {key}, found {len(matches)}"
            raise AssertionError(msg)
        level_indices.append(int(matches[0]))

    da = ds[variable].isel(pressure=level_indices)
    if extent is not None:
        lon_min, lon_max, lat_min, lat_max = extent
        da = da.sel(longitude=slice(lon_min, lon_max), latitude=slice(lat_min, lat_max))
    for attr in ("um_version", "mosg__grid_version"):
        if attr in ds.attrs:
            da.attrs[attr] = ds.attrs[attr]
    return da


def _open_dataset(fs: BaseClient, key: str) -> xr.Dataset:
    """Lazily open the raw dataset for an object key via byte-range-capable I/O."""
    f = _S3RangeFile(fs, BUCKET, key)
    return xr.open_dataset(f, engine="h5netcdf", decode_times=True, decode_timedelta=True)


def open_pressure_level_field(
    fs: BaseClient,
    key: str,
    parameter: str,
    *,
    run: datetime.datetime | None = None,
    validity: datetime.datetime | None = None,
    lead_hours: int | None = None,
    extent: tuple[float, float, float, float] | None = None,
) -> xr.DataArray:
    """Lazily open the cruise-level, region-cropped subset of a pressure-level parameter.

    Does not trigger any byte-range fetch; the returned array is backed by lazy
    ``h5netcdf`` indexing. Never opens the ``flag`` variable.

    Parameters
    ----------
    fs : boto3.client
        S3 client to read through, e.g. from :func:`filesystem`.
    key : str
        Object key relative to :data:`BUCKET`, e.g. from :func:`object_key`.
    parameter : str
        Parameter name as it appears in the file name. Must be a key of
        :data:`PARAMETER_VARIABLE`.
    run : datetime.datetime, optional
        Filename-derived run time. If given, asserted against the in-file
        ``forecast_reference_time``.
    validity : datetime.datetime, optional
        Filename-derived validity time. If given, asserted against the in-file
        ``time``.
    lead_hours : int, optional
        Filename-derived lead time. If given, asserted against the in-file
        ``forecast_period``.
    extent : tuple[float, float, float, float], optional
        ``(lon_min, lon_max, lat_min, lat_max)`` to crop to. ``None`` (default)
        returns the full longitude/latitude range.

    Returns
    -------
    xr.DataArray
        Lazy array with dims ``(pressure, latitude, longitude)``, restricted to
        :data:`CRUISE_LEVELS_HPA` and, if given, ``extent``.

    """
    ds = _open_dataset(fs, key)

    if run is not None or validity is not None or lead_hours is not None:
        _assert_time_coords_match(ds, run=run, validity=validity, lead_hours=lead_hours)

    return select_cruise_subset(ds, parameter, key=key, extent=extent)


def fetch_pressure_level_field(
    fs: BaseClient,
    key: str,
    parameter: str,
    *,
    run: datetime.datetime | None = None,
    validity: datetime.datetime | None = None,
    lead_hours: int | None = None,
    extent: tuple[float, float, float, float] | None = None,
) -> xr.DataArray:
    """Fetch (byte-range read + load) the cruise-level, region-cropped subset of a parameter.

    See :func:`open_pressure_level_field` for parameters. This is the point at which
    the actual byte-range GETs happen.

    Returns
    -------
    xr.DataArray
        Loaded array with dims ``(pressure, latitude, longitude)``.

    """
    da = open_pressure_level_field(
        fs, key, parameter, run=run, validity=validity, lead_hours=lead_hours, extent=extent
    )
    return da.load()
