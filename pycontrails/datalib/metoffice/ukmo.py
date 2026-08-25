"""Met Office data access.

This module supports

- Fetching Met Office UM cruise-level forecast data directly from the public
  ``global-deterministic-10km`` S3 archive, on a cache miss, via
  :mod:`pycontrails.datalib.metoffice.s3`'s byte-range read logic.
- Converting the fetched water-referenced relative humidity to
  ``specific_humidity``, from which pycontrails' ``thermo.rhi`` recovers the
  ice-referenced RHi conversion that ISSR/SAC-style humidity scaling needs.
- Opening the result as a :class:`pycontrails.MetDataset`.

:attr:`cachestore` (default :class:`pycontrails.core.cache.DiskCacheStore`) backs
the inherited
:meth:`~pycontrails.datalib._met_utils.metsource.MetDataSource.download`, which
live-fetches via :meth:`MetOfficeUM.download_dataset` only for genuine cache
misses. Passing a pre-populated archive such as
``cache.GCPCacheStore(bucket=..., read_only=True)`` as ``cachestore`` works,
since its :meth:`~pycontrails.core.cache.CacheStore.exists`/
:meth:`~pycontrails.core.cache.CacheStore.get` check a local mirror before
falling back to the remote archive.
"""

from __future__ import annotations

import hashlib
import warnings
from datetime import datetime
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import xarray as xr

import pycontrails
from pycontrails.core import met_var
from pycontrails.core.cache import CacheStore, DiskCacheStore
from pycontrails.core.met import MetDataset, MetVariable
from pycontrails.datalib._met_utils import metsource
from pycontrails.datalib.metoffice import s3
from pycontrails.physics import thermo
from pycontrails.utils import temp

if TYPE_CHECKING:
    from botocore.client import BaseClient

#: MetDataset.attrs values set by :meth:`MetOfficeUM.set_metadata`.
PROVIDER = "Met Office"
DATASET = "UM-Global-Deterministic-10km"
PRODUCT = "forecast"


class MetOfficeDataNotFoundError(Exception):
    """Raised when requested data could not be fetched from the S3 archive."""


class MetOfficeUM(metsource.MetDataSource):
    """pycontrails datalib for the UK Met Office ``global-deterministic-10km`` product.

    Live-fetches cruise-level pressure data directly from the public S3 archive on
    a cache miss. :attr:`cachestore` also supports reading from a pre-populated
    archive.

    Parameters
    ----------
    time : metsource.TimeInput
        Single datetime or ``(start, end)`` range. Parsed to hourly timesteps.
    variables : metsource.VariableInput, optional
        Requested variables. Defaults to
        ``[met_var.AirTemperature, met_var.SpecificHumidity]``.
        ``specific_humidity`` is always computed internally regardless of
        what's requested; pass ``met_var.RelativeHumidity`` explicitly to also
        keep the raw fetched value.
    pressure_levels : metsource.PressureLevelInput, optional
        Requested pressure levels in hPa. Defaults to
        :data:`pycontrails.datalib.metoffice.s3.CRUISE_LEVELS_HPA` (the 7 levels
        the S3 archive's byte-range crop targets).
    grid : float, optional
        Not supported. The archive serves fixed native-grid data; regridding
        happens downstream of this datalib, not within it. A non-``None`` value
        is ignored with a warning.
    lead_hours : int, optional
        ``None`` (default) fetches the shortest-available lead for each requested
        validity time. A fixed int fetches instead at that lead for every hour.
    cachestore : cache.CacheStore, optional
        Cache store for fetched, processed hourly data. Defaults to
        :class:`pycontrails.core.cache.DiskCacheStore`. Pass
        ``cache.GCPCacheStore(bucket=..., read_only=True)`` to read from a
        pre-populated archive (falling back to a live fetch for any hour it
        doesn't have); pass ``None`` to disable caching (every call re-fetches).

    """

    __marker = object()

    __slots__ = ("cachestore", "lead_hours")

    def __init__(
        self,
        time: metsource.TimeInput,
        *,
        variables: metsource.VariableInput | None = None,
        pressure_levels: metsource.PressureLevelInput = s3.CRUISE_LEVELS_HPA,
        grid: float | None = None,
        lead_hours: int | None = None,
        cachestore: CacheStore | None = __marker,  # type: ignore[assignment]
        **kwargs: Any,
    ) -> None:
        if grid is not None:
            warnings.warn(
                f"MetOfficeUM serves fixed native-grid data; regridding is expected "
                f"to happen downstream of this datalib, not within it. "
                f"Ignoring grid={grid!r}."
            )
        self.grid = None
        self.paths = None

        self.pressure_levels = metsource.parse_pressure_levels(
            pressure_levels, supported=list(s3.CRUISE_LEVELS_HPA)
        )

        if variables is None:
            variables = [met_var.AirTemperature, met_var.SpecificHumidity]
        self.variables = metsource.parse_variables(variables, self.supported_variables)

        self.timesteps = metsource.parse_timesteps(time, freq="1h")

        self.lead_hours = lead_hours
        self.cachestore = DiskCacheStore() if cachestore is self.__marker else cachestore

        del kwargs  # accepted only for MetDataSource ABC compatibility; unused

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

        Hashes the timestamp, the lead selection, the requested pressure
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

        lead = "shortest" if self.lead_hours is None else f"{self.lead_hours:03d}"
        string = (
            f"{lead}-{t:%Y%m%d%H}-"
            f"{'.'.join(str(p) for p in self.pressure_levels)}-"
            f"{'.'.join(sorted(self.variable_shortnames))}"
        )
        name = hashlib.md5(string.encode()).hexdigest()
        return self.cachestore.path(f"ukmo-{name}.nc")

    def cache_dataset(self, dataset: xr.Dataset) -> None:
        """Write processed hourly data to :attr:`cachestore`.

        Stages each hour through a local temp file, then writes via
        :meth:`~pycontrails.core.cache.CacheStore.put`. A read-only
        :attr:`cachestore` raises ``RuntimeError`` from ``put``, which is
        caught and ignored; any other ``RuntimeError`` propagates.
        """
        if self.cachestore is None:
            return
        for t, ds in dataset.groupby("time", squeeze=False):
            cache_path = self.create_cachepath(pd.Timestamp(t).to_pydatetime())
            with temp.temp_file() as tmp_path:
                ds.to_netcdf(tmp_path, mode="w")
                try:
                    self.cachestore.put(tmp_path, cache_path)
                except RuntimeError:
                    if not getattr(self.cachestore, "read_only", False):
                        raise

    def download_dataset(self, times: list[datetime]) -> None:
        """Fetch missing hours directly from the public Met Office S3 archive.

        Only called (via the inherited
        :meth:`~pycontrails.datalib._met_utils.metsource.MetDataSource.download`)
        with hours that
        :meth:`~pycontrails.datalib._met_utils.metsource.MetDataSource.is_datafile_cached`
        has already determined are missing from :attr:`cachestore`.
        """
        fs = s3.filesystem()
        for t in times:
            self._download_convert_cache_handler(fs, t)

    def _download_convert_cache_handler(self, fs: BaseClient, t: datetime) -> None:
        """Fetch, process, and cache one missing hour."""
        if self.lead_hours is not None:
            run, lead = s3.run_for_validity_at_lead(t, self.lead_hours), self.lead_hours
        else:
            run, lead = s3.run_and_lead_for_validity(t)

        data_vars = {}
        for parameter, variable in s3.PARAMETER_VARIABLE.items():
            key = s3.object_key(run, t, lead, parameter)
            try:
                data_vars[variable] = s3.fetch_pressure_level_field(
                    fs, key, parameter, run=run, validity=t, lead_hours=lead
                )
            except Exception as exc:
                msg = (
                    f"{t.isoformat()} could not be fetched from the S3 archive "
                    f"(key={key}): {exc}"
                )
                raise MetOfficeDataNotFoundError(msg) from exc

        ds = xr.Dataset(data_vars).expand_dims(time=[pd.Timestamp(t)])
        ds = self._process_hour(ds)
        ds.attrs["pycontrails_version"] = pycontrails.__version__
        self.cache_dataset(ds)

    def _process_hour(self, raw_ds: xr.Dataset) -> xr.Dataset:
        """Transform one hour of raw fetched data into the cached schema.

        Selects the requested pressure levels, computes ``specific_humidity``,
        and renames to the short names :meth:`~pycontrails.datalib._met_utils.\
metsource.MetDataSource._check_is_ds_complete` (and the rest of pycontrails)
        expect.

        Returns
        -------
        xr.Dataset
            Processed dataset with cruise levels selected, ``specific_humidity``
            computed, and variables named by short name.
        """
        # Must run before `MetDataset` construction: with the default `copy=True`,
        # `MetDataset.__init__` sorts every coordinate, masking a reversed one.
        # `s3.select_cruise_subset` asserts this per-file; this is a cheap check
        # on the assembled hour.
        latitude = raw_ds["latitude"].values
        if not np.all(np.diff(latitude) > 0):
            msg = f"expected ascending latitude ordering, got {latitude[:3]}...{latitude[-3:]}"
            raise AssertionError(msg)

        pressure_pa = raw_ds["pressure"].values.astype(np.float64)
        target_pa = np.asarray(self.pressure_levels, dtype=np.float64) * 100.0
        level_indices = []
        for level_hpa, level_pa in zip(self.pressure_levels, target_pa, strict=True):
            matches = np.flatnonzero(np.isclose(pressure_pa, level_pa, atol=1e-3))
            if len(matches) != 1:
                msg = (
                    f"expected exactly one {level_hpa} hPa level in the fetched "
                    f"data, found {len(matches)}"
                )
                raise ValueError(msg)
            level_indices.append(int(matches[0]))

        ds = raw_ds.isel(pressure=level_indices)
        ds = ds.rename(pressure="level")
        ds = ds.assign_coords(level=ds["level"] / 100.0)

        # q is always computed, regardless of what was requested: downstream
        # ISSR/SAC-style humidity scaling needs it.
        level_pa = ds["level"] * 100.0
        q = ds["relative_humidity"] * thermo.q_sat_liquid(ds["air_temperature"], level_pa)
        ds = ds.assign(specific_humidity=q)

        rename = {"air_temperature": "t", "specific_humidity": "q", "relative_humidity": "r"}
        ds = ds.rename({k: v for k, v in rename.items() if k in ds})

        return ds[list(self.variable_shortnames)]

    def open_metdataset(
        self,
        dataset: xr.Dataset | None = None,
        xr_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> MetDataset:
        """Open the requested window as a ``MetDataset``.

        Returns
        -------
        MetDataset
            The processed dataset for the requested window.
        """
        if dataset is not None:
            msg = "Parameter 'dataset' is not supported for MetOfficeUM data"
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
            raise AssertionError(msg)

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
