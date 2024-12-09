"""Common utilities for ECMWF Data Access."""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

LOG = logging.getLogger(__name__)

import numpy as np
import pandas as pd
import xarray as xr

from pycontrails.core import met
from pycontrails.datalib._met_utils import metsource


class ECMWFAPI(metsource.MetDataSource):
    """Abstract class for all ECMWF data accessed remotely through CDS / MARS."""

    @property
    def variable_ecmwfids(self) -> list[int]:
        """Return a list of variable ecmwf_ids.

        Returns
        -------
        list[int]
            List of int ECMWF param ids.
        """
        return [v.ecmwf_id for v in self.variables if v.ecmwf_id is not None]

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

        # downselect variables
        try:
            ds = ds[self.variable_shortnames]
        except KeyError as exc:
            missing = set(self.variable_shortnames).difference(ds.variables)
            msg = f"Input dataset is missing variables {missing}"
            raise KeyError(msg) from exc

        # downselect times
        if not self.timesteps:
            self.timesteps = ds["time"].values.astype("datetime64[ns]").tolist()  # type: ignore[assignment]
        else:
            try:
                ds = ds.sel(time=self.timesteps)
            except KeyError as exc:
                # this snippet shows the missing times for convenience
                np_timesteps = {np.datetime64(t, "ns") for t in self.timesteps}
                missing_times = sorted(np_timesteps.difference(ds["time"].values))  # type: ignore[type-var]
                msg = f"Input dataset is missing time coordinates {[str(t) for t in missing_times]}"
                raise KeyError(msg) from exc

        # downselect pressure level
        # if "level" is not in dims and
        # length of the requested pressure levels is 1
        # expand the dims with this level
        if "level" not in ds.dims and len(self.pressure_levels) == 1:
            ds = ds.expand_dims(level=self.pressure_levels)

        try:
            ds = ds.sel(level=self.pressure_levels)
        except KeyError as exc:
            # this snippet shows the missing levels for convenience
            missing_levels = sorted(set(self.pressure_levels) - set(ds["level"].values))
            msg = f"Input dataset is missing level coordinates {missing_levels}"
            raise KeyError(msg) from exc

        # harmonize variable names
        ds = met.standardize_variables(ds, self.variables)

        kwargs.setdefault("cachestore", self.cachestore)
        return met.MetDataset(ds, **kwargs)

    @override
    def cache_dataset(self, dataset: xr.Dataset) -> None:
        if self.cachestore is None:
            LOG.debug("Cache is turned off, skipping")
            return

        for t, ds_t in dataset.groupby("time", squeeze=False):
            cache_path = self.create_cachepath(pd.Timestamp(t).to_pydatetime())
            if os.path.exists(cache_path):
                LOG.debug(f"Overwriting existing cache file {cache_path}")
                # This may raise a PermissionError if the file is already open
                # If this is the case, the user should explicitly close the file and try again
                os.remove(cache_path)

            ds_t.to_netcdf(cache_path)


class CDSCredentialsNotFound(Exception):
    """Raise when CDS credentials are not found by :class:`cdsapi.Client` instance."""
