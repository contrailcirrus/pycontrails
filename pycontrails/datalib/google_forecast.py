"""Google Contrails Forecast Datalib.

Loads contrail forecasts from https://developers.google.com/contrails.
"""

from __future__ import annotations

import datetime
import hashlib
import logging
import os
import tempfile
from typing import TYPE_CHECKING, Any

import pandas as pd
import requests
import xarray as xr

from pycontrails import MetDataset, MetVariable
from pycontrails.core import cache, met
from pycontrails.datalib._met_utils import metsource
from pycontrails.physics import units

if TYPE_CHECKING:
    import google.auth.credentials

logger = logging.getLogger(__name__)


Severity = MetVariable(
    short_name="contrails",
    standard_name="contrails",
    long_name="Contrail Severity Index",
    description="The severity (0-4) of forecasted contrail warming.",
    units="1",
)


EffectiveEnergyForcing = MetVariable(
    short_name="eeef_per_m",
    standard_name="expected_effective_energy_forcing",
    long_name="Expected Effective Energy Forcing",
    description=(
        "The effective energy forcing of contrail warming. It has the ERF/RF ratio applied."
    ),
    units="J m**-1",
)


class GoogleForecast(metsource.MetDataSource):
    """Google Forecast datalib to download precomputed contrail forecasts from API sources.

    This class provides an interface to the `Google Contrails Forecast API <https://developers.google.com/contrails>`_.
    It returns a :class:`MetDataset` containing the forecasted severity and/orenergy forcing.

    Parameters
    ----------
    time : metsource.TimeInput | None
        The time range for data retrieval, either a single datetime or (start, end) datetime range.
        Input must be datetime-like or tuple of datetime-like
        (`datetime`, :class:`pd.Timestamp`, :class:`np.datetime64`)
        specifying the (start, end) of the date range, inclusive.
    variables : metsource.VariableInput
        Variable name (i.e. "contrails", "eef_per_m", ["contrails", "eef_per_m"])
    key : str | google.auth.credentials.Credentials | None, optional
        Google Cloud Platform credentials or API key.
        If None, looks for ``GOOGLE_API_KEY`` environment variable.
        If that is not found, uses :func:`google.auth.default`.
    url : str, optional
        Google Contrails Forecast API URL.
        Defaults to "https://contrails.googleapis.com/v2/grids".
    cachestore : cache.CacheStore | None, optional
        Cache data store for Google Forecast files. If a cache store is provided, forecasts will be
        first retrieved from the cache store and only from the API if not yet cached. Since the
        Google Forecast API frequently recomputes forecasts, using the cache store may result in
        stale data. It is not recommended to cache future forecasts for longer than one hour.
        Defaults to None (no caching).
    """

    __slots__ = ("_credentials", "cachestore", "url")

    #: Google Contrails Forecast API URL
    url: str

    def __init__(
        self,
        time: metsource.TimeInput | None,
        variables: metsource.VariableInput = (Severity,),
        key: str | google.auth.credentials.Credentials | None = None,
        url: str = "https://contrails.googleapis.com/v2/grids",
        cachestore: cache.CacheStore | None = None,
    ) -> None:
        self.url = url
        self.cachestore = cachestore
        self.timesteps = metsource.parse_timesteps(time)
        self.variables = metsource.parse_variables(variables, self.supported_variables)

        # Metadata
        self.pressure_levels = []
        self.grid = None
        self.paths = None
        self._credentials = key

    def cache_dataset(self, dataset: xr.Dataset) -> None:
        """Cache data from data source.

        If the cache path already exists, this method will merge the new data
        with the existing cached data. This allows different variables to be
        cached in the same file.
        """
        if self.cachestore is None:
            raise ValueError("Cachestore is required to cache data")

        # Ensure dataset is loaded to avoid file locks
        if not dataset.chunks:
            dataset.load()

        # Iterate over time values to handle potential multi-time datasets
        for t_val in dataset["time"].values:
            ts = pd.Timestamp(t_val).to_pydatetime()
            cache_path = self.create_cachepath(ts)
            ds_slice = dataset.sel(time=[t_val])

            if self.cachestore.exists(cache_path):
                try:
                    ds_old = xr.load_dataset(cache_path)
                    # Merge new data into old data
                    # "compat='override'" assumes coordinates are aligned (same grid)
                    ds_slice = ds_slice.merge(ds_old, compat="override")
                except Exception as e:
                    logger.warning(
                        f"Failed to merge with cache at {cache_path}: {e}. Overwriting..."
                    )

            ds_slice.to_netcdf(cache_path)

    @property
    def supported_variables(self) -> list[MetVariable]:
        """Get supported variables."""
        return [Severity, EffectiveEnergyForcing]

    def create_cachepath(self, t: datetime.datetime) -> str:
        """Return cachepath to local data file based on datetime."""
        if self.cachestore is None:
            raise ValueError("self.cachestore attribute must be defined to create cache path")

        datestr = t.strftime("%Y%m%d-%H")
        name = f"google-forecast-{datestr}.nc"
        return self.cachestore.path(name)

    def download_dataset(self, times: list[datetime.datetime]) -> list[xr.Dataset]:
        """Download data from Google API, updating the cache if configured."""
        requested_vars = [v.standard_name for v in self.variables]
        return [self._download_timestep(t, requested_vars) for t in times]

    def _download_timestep(self, t: datetime.datetime, variables: list[str]) -> xr.Dataset:
        """Download and process a batch of variables for a single time step."""
        # The API expects the variables' standard names.
        params = {"time": t.isoformat(), "data": variables}

        logger.info(f"Requesting: {self.url} with params {params}")

        response = requests.get(self.url, params=params, headers=self._request_headers)

        logger.debug(
            f"Received {response.status_code}, response with size {len(response.content)} "
            f"and headers: {response.headers}"
        )
        response.raise_for_status()

        # Workaround: xarray does not support loading NetCDF4 from a bytes buffer.
        # We save directly to cache or temp
        # Write to temp file then load and save to ensure valid NetCDF
        with tempfile.NamedTemporaryFile() as tmp:
            tmp.write(response.content)
            tmp.flush()
            ds = xr.load_dataset(tmp.name, engine="netcdf4")

        # Process: Convert flight_level to level if needed.
        if "level" not in ds.dims and "flight_level" in ds.dims:
            ds["level"] = units.ft_to_pl(ds["flight_level"].astype(float) * 100)
            ds = ds.swap_dims({"flight_level": "level"})

        logger.info("Downloaded dataset: %s", ds)

        if self.cachestore:
            self.cache_dataset(ds)

        return ds

    def open_metdataset(
        self,
        dataset: xr.Dataset | None = None,
        xr_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> MetDataset:
        """Open MetDataset from the Google Contrails Forecast API."""
        xr_kwargs = xr_kwargs or {}

        if dataset is not None:
            dataset = met.standardize_variables(dataset, self.variables)
            self.set_metadata(dataset)
            return MetDataset(dataset, **kwargs)

        datasets = []
        requested_vars = {v.standard_name for v in self.variables}

        for t in self.timesteps:
            # 1. Try to load from cache
            ds_hour = None
            if self.cachestore:
                cache_path = self.create_cachepath(t)
                if self.cachestore.exists(cache_path):
                    try:
                        ds_hour = xr.load_dataset(cache_path, **xr_kwargs)
                    except Exception as e:
                        logger.warning(
                            f"Failed to read cache at {cache_path}: {e}. Redownloading all "
                            "variables."
                        )

            # 2. (Re-)download hour if any variable is missing.
            if ds_hour is None or not all(v in ds_hour.data_vars for v in requested_vars):
                ds_hour = self._download_timestep(t, list(requested_vars))

            datasets.append(ds_hour)

        dataset = xr.concat(datasets, dim="time")
        return self.open_metdataset(dataset=dataset, xr_kwargs=xr_kwargs, **kwargs)

    def set_metadata(self, ds: xr.Dataset | MetDataset) -> None:
        """Set metadata."""
        ds.attrs.update(
            provider="Google",
            dataset="Contrails Forecast",
            product="forecast",
        )

    @property
    def hash(self) -> str:
        """Generate a unique hash for this datasource."""
        hashstr = f"{self.__class__.__name__}{self.timesteps}{self.variable_shortnames}{self.url}"
        return hashlib.sha1(bytes(hashstr, "utf-8")).hexdigest()

    @property
    def _request_headers(self) -> dict[str, str]:
        """Request headers for authentication."""
        key = self._credentials
        headers = {}

        if key is None:
            key = os.getenv("GOOGLE_API_KEY")

        if key is None:
            try:
                import google.auth

                key, _ = google.auth.default()
            except ImportError as e:
                raise ValueError(
                    "No API key or google-auth found. Provide `key` or set "
                    "GOOGLE_API_KEY or install google-auth."
                ) from e

        if isinstance(key, str):
            headers["x-goog-api-key"] = key
        elif hasattr(key, "apply"):
            key.apply(headers)
        else:
            raise ValueError(
                "No credentials found. Provide `key` or set GOOGLE_API_KEY or "
                "GOOGLE_APPLICATION_CREDENTIALS."
            )

        return headers
