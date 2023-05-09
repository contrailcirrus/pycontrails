"""Common utilities for ECMWF Data Access."""

from __future__ import annotations

import logging
from typing import Any

LOG = logging.getLogger(__name__)

import numpy as np
import xarray as xr

from pycontrails.core import datalib, met


def rad_accumulated_to_average(mds: met.MetDataset, key: str, dt_accumulation: int) -> None:
    """Convert accumulated radiation value to instantaneous average.

    Parameters
    ----------
    mds : MetDataset
        MetDataset containing the accumulated value at ``key``
    key : str
        Data variable key
    dt_accumulation : int
        Accumulation time in seconds, [:math:`s`]
    """
    if key in mds.data and not mds.data[key].attrs.get("_pycontrails_modified", False):
        if not np.all(np.diff(mds.data["time"]) == np.timedelta64(dt_accumulation, "s")):
            raise ValueError(
                f"Dataset expected to have time interval of {dt_accumulation} seconds when"
                " converting accumulated parameters"
            )

        mds.data[key] = mds.data[key] / dt_accumulation
        mds.data[key].attrs["units"] = "W m**-2"
        mds.data[key].attrs[
            "_pycontrails_modified"
        ] = "Accumulated value converted to average instantaneous value"


# TODO: Remove this in favor of functional implementation
class ECMWFAPI(datalib.MetDataSource):
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

    # TODO: this could be functional, but there many properties utilized
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
        except KeyError as e:
            raise KeyError(f"Input dataset is missing variables {e}")

        # downselect times
        if not self.timesteps:
            self.timesteps = ds["time"].values.astype("datetime64[ns]").tolist()
        else:
            try:
                ds = ds.sel(time=self.timesteps)
            except KeyError:
                # this snippet shows the missing times for convenience
                np_timesteps = [np.datetime64(t, "ns") for t in self.timesteps]
                missing_times = sorted(set(np_timesteps) - set(ds["time"].values))
                raise KeyError(
                    f"Input dataset is missing time coordinates {[str(t) for t in missing_times]}"
                )

        # downselect pressure level
        # if "level" is not in dims and
        # length of the requested pressure levels is 1
        # expand the dims with this level
        if "level" not in ds.dims and len(self.pressure_levels) == 1:
            ds = ds.expand_dims({"level": self.pressure_levels})

        try:
            ds = ds.sel(level=self.pressure_levels)
        except KeyError:
            # this snippet shows the missing levels for convenience
            missing_levels = sorted(set(self.pressure_levels) - set(ds["level"].values))
            raise KeyError(f"Input dataset is missing level coordinates {missing_levels}")

        # harmonize variable names
        ds = met.standardize_variables(ds, self.variables)

        # modify values

        # rescale relative humidity from % -> dimensionless if its in dataset
        if "relative_humidity" in ds and not ds["relative_humidity"].attrs.get(
            "_pycontrails_modified", False
        ):
            ds["relative_humidity"] = ds["relative_humidity"] / 100
            ds["relative_humidity"].attrs["long_name"] = "Relative humidity"
            ds["relative_humidity"].attrs["standard_name"] = "relative_humidity"
            ds["relative_humidity"].attrs["units"] = "[0 - 1]"
            ds["relative_humidity"].attrs[
                "_pycontrails_modified"
            ] = "Relative humidity rescaled to [0 - 1] instead of %"

        ds.attrs["met_source"] = type(self).__name__

        kwargs.setdefault("cachestore", self.cachestore)
        return met.MetDataset(ds, **kwargs)
