"""Google Contrails Forecast Datalib."""

from __future__ import annotations

import datetime
import hashlib
import logging
import os
import tempfile
from typing import TYPE_CHECKING, Any

import requests
import pandas as pd
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
)


EffectiveEnergyForcing = MetVariable(
    short_name="eef_per_m",
    standard_name="expected_effective_energy_forcing",
    long_name="Expected Effective Energy Forcing",
    description="The effective energy forcing of contrail warming. It has the ERF/RF ratio applied.",
)


class GoogleForecast(metsource.MetDataSource):
    """Google Forecast datalib to download precomputed contrail forecasts from API sources.

    This class provides an interface to the Google Contrails Forecast API.
    It returns a :class:`MetDataset` containing the forecasted energy forcing.

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
        Cache data store for Google Forecast files.
        Defaults to :class:`cache.DiskCacheStore`.
    """

    __slots__ = ("_credentials", "url", "cachestore")

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
        self.cachestore = cachestore or cache.DiskCacheStore()
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
                    ds_slice = ds_old.merge(ds_slice, compat="override")
                except Exception as e:
                    logger.warning(f"Failed to merge with cache at {cache_path}: {e}. Overwriting...")
                    
            ds_slice.to_netcdf(cache_path)

    @property
    def supported_variables(self) -> list[MetVariable]:
        """Get supported variables."""
        return [Severity, EffectiveEnergyForcing]

    @property
    def hash(self) -> str:
        """Generate a unique hash for this datasource."""
        hashstr = f"{self.__class__.__name__}{self.timesteps}{self.variable_shortnames}{self.url}"
        return hashlib.sha1(bytes(hashstr, "utf-8")).hexdigest()

    @property
    def request_headers(self) -> dict[str, str]:
        """Request headers for authentication."""
        key = self._credentials
        headers = {}

        if key is None:
            key = os.getenv("GOOGLE_API_KEY")

        if key is None:
            try:
                import google.auth

                key, _ = google.auth.default()
            except ImportError:
                raise ValueError("No API key or google-auth found. Provide `key` or set GOOGLE_API_KEY or install google-auth.")

        if isinstance(key, str):
            headers["x-goog-api-key"] = key
        elif hasattr(key, "apply"):
            key.apply(headers)
        else:
            raise ValueError("No credentials found. Provide `key` or set GOOGLE_API_KEY or GOOGLE_APPLICATION_CREDENTIALS.")

        return headers

    def create_cachepath(self, t: datetime.datetime) -> str:
        """Return cachepath to local data file based on datetime."""
        if self.cachestore is None:
            raise ValueError("self.cachestore attribute must be defined to create cache path")

        datestr = t.strftime("%Y%m%d-%H")
        name = f"google-forecast-{datestr}.nc"
        return self.cachestore.path(name)

    def download_dataset(self, times: list[datetime.datetime]) -> None:
        """Download data from Google API."""
        if self.cachestore is None:
            raise ValueError("Cachestore is required to download data")

        for t in times:
            self._download_file(t)

    def _download_file(self, t: datetime.datetime) -> None:
        """Download single time file."""
        cache_path = self.create_cachepath(t)
        
        # Check if we need to download anything
        requested_vars = {v.standard_name for v in self.variables}
        download_vars = requested_vars.copy()

        if self.cachestore.exists(cache_path):
            try:
                with xr.open_dataset(cache_path) as ds_cache:
                    cached_vars = set(ds_cache.data_vars)
                    download_vars = requested_vars - cached_vars
            except Exception as e:
                logger.warning(f"Failed to read cache at {cache_path}: {e}. Redownloading all variables.")

        if not download_vars:
            return

        # The API expects just the variables' standard names
        params = {"time": t.isoformat(), "data": list(download_vars)}

        logger.info(f"Requesting: {self.url} with params {params}")
        if not self._credentials:
            raise ValueError("No credentials found. Provide `key` or set GOOGLE_API_KEY.")

        response = requests.get(self.url, params=params, headers=self.request_headers)
        
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
        
        # Process: Convert flight_level to level if needed
        if "level" not in ds.dims and "flight_level" in ds.dims:
            ds["level"] = units.ft_to_pl(ds["flight_level"] * 100)
            ds = ds.swap_dims({"flight_level": "level"})
            ds = ds.drop_vars("flight_level")

        self.cache_dataset(ds)


    def open_metdataset(
        self,
        dataset: xr.Dataset | None = None,
        xr_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> MetDataset:
        """Open MetDataset."""
        xr_kwargs = xr_kwargs or {}

        if dataset is not None:
            ds = dataset
        elif self.cachestore is None:
            raise ValueError("Cachestore is required")
        else:     
            self.download_dataset(self.timesteps)
            disk_cachepaths = [self.create_cachepath(t) for t in self.timesteps]
            ds = self.open_dataset(disk_cachepaths, **xr_kwargs)

        # Standardize
        ds = met.standardize_variables(ds, self.variables)
        
        self.set_metadata(ds)
        return MetDataset(ds, **kwargs)

    def set_metadata(self, ds: xr.Dataset | MetDataset) -> None:
        """Set metadata."""
        ds.attrs.update(
            provider="Google",
            dataset="Contrails Forecast",
            product="forecast",
        )
